---
title: vLLM 投机解码（Eagle / MTP）实现与运行流程
date: 2026-03-31
tags: [vLLM]
---

# vLLM 投机解码（Eagle / MTP）实现与运行流程

## 原理解释：残差分布与拒绝采样

### 阅读时记录的问题

![](/imgs/sps_formula.png)


1. q(x) * min(1, p(x)/q(x)) = min(q(x), p(x))，其实这个公式就是说：第一个step生成token x的概率等于draft model生成token x 的概率q(x) 乘它被接收的概率，也就得到了min(q(x), p(x))

2. “如果 q(x) <= p(x)，那 x 一定被接受，所以第一步给了它 q(x) 的概率质量”，一定被接受是因为按照P_accept(x)计算，得到1所以一定被接受，我理解的对吗？

3. 为什么还要除以一个总和 这里我没看懂，为什么说这是一个条件分布？分母代表什么？

看了后面的例子我大概理解了，是说需要对所有需要补概率的可选的token进行遍历，求和总的需要补的概率大小，并对当前token需要补的概率进行加权？

4. ”只要发生拒绝，第二步就一定补 B。“我没太理解，以及“第一步接受总概率是 0.5+0.1+0.2=0.8，拒绝概率是 0.2。”，这是说明采样到A,B,C都有可能接受，但概率是0.8，如果拒绝了，那一定会生成B的意思吗？但是拒绝后的B是怎么生成的？由谁来生成？

5. 好的，按你说的 “为什么接受率是 min(1, p/q)，以及它和残差分布一起如何严格保证无偏”，用更容易懂的“面积图/水桶补水”方式再解释一遍。

### 回答

先抓一句话：**残差分布就是“第一步按 `q(x)` 提案并做接受/拒绝后，`p(x)` 还没被覆盖到的那部分概率，在第二步按缺口大小重新分配出来的分布”。**

你这 5 个问题其实都围着这一个核心在转。下面我逐条回答。

## 逐条回答

1. 你对 `q(x) * min(1, p(x)/q(x)) = min(q(x), p(x))` 的理解，**基本是对的**。

更精确一点说，这个式子表示的是：

`x` 在**第一步被 draft model 采到，并且被接受，直接成为输出**的概率质量。

不是“最终输出 `x` 的总概率”，因为最终输出还可能来自第二步的残差补偿。

为什么它等于 `min(q(x), p(x))`？分两种情况看：

- 如果 `q(x) <= p(x)`，那么 `p(x)/q(x) >= 1`，接受率就是 `1`，所以
  `q(x) * 1 = q(x) = min(q(x), p(x))`
- 如果 `q(x) > p(x)`，那么接受率就是 `p(x)/q(x)`，所以
  `q(x) * p(x)/q(x) = p(x) = min(q(x), p(x))`

所以第一步对每个 token 实际“留下来”的概率质量，就是 `q` 和 `p` 的**重叠部分**。

2. 对，你这里理解是对的。

“如果 `q(x) <= p(x)`，那 `x` 一定被接受”就是因为：

`P_accept(x) = min(1, p(x)/q(x))`

此时 `p(x)/q(x) >= 1`，所以 `min(1, ...) = 1`。

直觉上也很好理解：

如果草稿模型给 `x` 的概率还**不超过**目标模型，那说明草稿并没有“高估” `x`，反而可能低估了它。既然没有高估，就没有理由把它拒掉，否则 `x` 的最终概率会更少，更达不到 `p(x)`。

所以这时第一步会把 `q(x)` 这部分概率质量**全部保留下来**。

3. 这里是关键点。你问“为什么还要除以一个总和”，答案是：

因为 `max(0, p(x)-q(x))` 只是每个 token 的“缺口大小”，**还不是一个概率分布**。

一个合法的概率分布必须所有项加起来等于 `1`。  
但这些缺口加起来通常不是 `1`，而是一个更小的数，比如 `0.2`、`0.05` 之类。

所以要除以总和，把它归一化，才能变成“在已经进入第二步这个条件下，该选哪个 token”的概率分布。

也就是说：

`p_res(x) = P(第二步输出 x | 第一步发生拒绝)`

这就是为什么说它是一个**条件分布**。  
因为第二步不是总会发生，只有“第一步拒绝了”才会进入第二步。

分母

`Σ_x' max(0, p(x') - q(x'))`

表示什么？

它表示所有 token 的总缺口，也就是：

- 第一步之后，目标分布还有多少概率质量没被补齐
- 同时它也等于**第一步的总拒绝概率**

这两件事其实是同一个量。

你后面那句理解也是对的：

> 需要对所有需要补概率的 token 求和，得到总共还要补多少；然后当前 token 按“它自己的缺口 / 总缺口”来分配第二步概率。

这正是残差分布的含义。

4. “只要发生拒绝，第二步就一定补 B”这句话**不是普遍规律**，只是那个例子里恰好如此。

我用一个最典型的 3-token 例子讲清楚：

| token | `q(x)` | `p(x)` | 第一步留下 `min(q,p)` | 缺口 `max(0,p-q)` |
|---|---:|---:|---:|---:|
| A | 0.5 | 0.5 | 0.5 | 0.0 |
| B | 0.1 | 0.3 | 0.1 | 0.2 |
| C | 0.4 | 0.2 | 0.2 | 0.0 |

先看第一步：

- A 的接受质量是 `0.5`
- B 的接受质量是 `0.1`
- C 的接受质量是 `0.2`

所以第一步总接受概率是：

`0.5 + 0.1 + 0.2 = 0.8`

那么总拒绝概率就是：

`1 - 0.8 = 0.2`

现在看第二步要补谁：

- A 不缺
- B 缺 `0.2`
- C 不缺

所以残差分布就是：

- `p_res(A) = 0`
- `p_res(B) = 1`
- `p_res(C) = 0`

这就意味着：

**一旦进入第二步，就一定输出 B。**

但注意，这不是说“算法永远偏爱 B”，而是因为在这个例子里，**只有 B 还有没补齐的概率质量**。

你问“拒绝后的 B 是怎么生成的？由谁来生成？”

答案是：

- 不是由 draft model 继续生成
- 而是由 **target model / verifier 对应位置的分布**，按照残差分布重新采样出来

也可以理解成：  
第一步草稿提议失败后，第二步改由目标分布的“缺口补偿机制”来决定输出什么。

在上面这个例子里，B 的最终概率是：

- 第一步直接得到的 `0.1`
- 加上第二步补回来的 `0.2`

所以最终：

`P_final(B) = 0.1 + 0.2 = 0.3 = p(B)`

这就严格对上目标分布了。

5. 用“面积图 / 水桶补水”方式解释，最直观。

把每个 token 看成一个水桶：

- 目标模型 `p(x)` 是这个桶最终应该有的水位
- 草稿模型 `q(x)` 是第一步先往桶里倒进去的水

现在分两种情况：

如果 `q(x) <= p(x)`：

- 这个桶倒得不多，甚至还不够
- 那第一步当然要**全部保留**
- 所以接受率是 `1`

如果 `q(x) > p(x)`：

- 这个桶倒多了
- 最多只能保留到目标水位 `p(x)`
- 所以只能保留 `p(x)/q(x)` 这么一个比例
- 多出来的那部分要倒回公共水箱里

这就是为什么接受率必须是：

`min(1, p(x)/q(x))`

它的作用就是：**保证第一步在每个 token 上，最多只保留到 `p(x)`，绝不超标。**

所以第一步结束后，每个桶里实际留下的水量都是：

`min(q(x), p(x))`

这就是 `q` 和 `p` 的**重叠面积**。

然后会出现两类桶：

- 有些桶第一步倒多了，溢出来了
- 有些桶第一步倒少了，还欠着

神奇的地方在于：

因为 `p` 和 `q` 都是总和为 `1` 的概率分布，  
所以“总溢出量”一定等于“总欠缺量”。

于是第二步就很自然：

- 把所有溢出来的水收集起来
- 只往那些“还欠水”的桶里补
- 哪个桶缺得多，就多补一点
- 哪个桶不缺，就不补

这就是残差分布：

`p_res(x) = 缺口(x) / 总缺口`

其中

`缺口(x) = max(0, p(x)-q(x))`

最后每个桶里的水量就变成：

`第一步保留的重叠部分 + 第二步补上的缺口`
也就是

`min(q(x), p(x)) + max(0, p(x)-q(x)) = p(x)`

这就是“**严格无偏**”的意思：

最终采样结果的分布，**恰好就是目标模型的 `p(x)`**，不多不少。

## 一句最浓缩的总结

- `min(1, p/q)` 负责把第一步保留下来的概率质量，精确裁成 `q` 和 `p` 的重叠部分。
- 残差分布负责把第一步没覆盖到的“缺口部分”，按比例补回去。
- 两步加起来，刚好还原出 `p`。

### 关于 bonus token 与 NO_DRAFT_PROBS 的问答

#### 问题

1. bonus token 是怎么来的，原理是什么？
2. “这意味着当前这条路径不会走“显式 max(p-q, 0)”那个分支，而是走 NO_DRAFT_PROBS 分支。那个分支里 recovered token 的候选概率是：
用 target_probs
但把“刚被拒绝的 draft token”排除掉” 这句话是什么意思，这样采样的概率分布是正确的吗？

#### 回答

### 1. `bonus token` 是怎么来的，原理是什么？

先说结论：

**`bonus token` 是“当本轮所有 draft token 都被 target 接受时，顺手再从 target 模型的下一个位置多采 1 个 token”。**

它的作用不是“补偿拒绝”，而是：

- 如果这一轮 draft 全通过了，说明这几个 speculative token 都已经被 target 验证为有效前缀
- 那么 target 在这次 forward 里，实际上已经有了“再往后一个位置”的 logits
- 这时可以直接再采一个 token，相当于白赚一个 token，进一步提高吞吐

在 `RejectionSampler.forward_impl()` 里，vLLM 先单独取出 `bonus_logits`，然后直接用主 `sampler` 从 target logits 采样出 `bonus_token_ids`：

```python
bonus_logits_indices = metadata.bonus_logits_indices
bonus_logits = logits[bonus_logits_indices]

bonus_sampler_output = self.sampler(
    logits=bonus_logits,
    sampling_metadata=replace(
        sampling_metadata,
        max_num_logprobs=-1,
    ),
    predict_bonus_token=True,
    logprobs_mode_override="processed_logits"
    if self.is_processed_logprobs_mode
    else "raw_logits",
)
bonus_token_ids = bonus_sampler_output.sampled_token_ids
```

然后在 rejection kernel 里，**只有“一个都没拒绝”时**才把这个 `bonus token` 追加到输出末尾：

```python
if not rejected:
    # If all tokens are accepted, append the bonus token.
    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
    tl.store(
        output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
        bonus_token_id,
    )
```

#### 原理上为什么这样是对的？

因为如果前面所有 draft token 都被接受，那么当前前缀已经和“正常 target 逐 token 解码”完全一致了。  
既然前缀一致，那么“下一个 token 应该怎么采样”当然就应该**直接按 target 模型在这个真实前缀上的分布来采样**。

所以 `bonus token` 本质上就是：

- 不是 draft 产生的
- 不是 residual 补出来的
- 而是 **target 模型对“已验证通过的完整前缀”再采的一个真正下一 token**

#### 为什么一旦有拒绝，就不能继续用 bonus？

因为一旦某个 draft token 被拒绝，后面那些位置的 logits 都是基于“错误前缀”算出来的。

举例：

- draft 提了 `A B C`
- target 在 `B` 这里拒绝了，改成了 `X`

那 `C` 对应的位置原来是基于前缀 `... A B` 算的，  
但现在真实前缀已经变成 `... A X` 了，后面的 logits 就全部不可信了。

所以：

- 有拒绝时：只能保留拒绝点之前接受的 token，加上拒绝点上的 recovered token
- 没有拒绝时：才可以额外追加一个 `bonus token`

---

### 2. `NO_DRAFT_PROBS` 分支里“把刚被拒绝的 draft token 排除掉”是什么意思？这样对吗？

这句话的意思是：

在这个分支里，vLLM **没有拿到完整的 draft 分布 `q(x)`**，只知道“draft 在这个位置提议了哪个 token”。  
所以它把 draft proposal 当成一个**退化分布（delta distribution）**来处理：

- 对被提议的那个 token `d`：`q(d)=1`
- 对其他 token `x!=d`：`q(x)=0`

这时理论上的残差分布就变成：

- 对 `x = d`：`max(p(d)-1, 0) = 0`
- 对 `x != d`：`max(p(x)-0, 0) = p(x)`

也就是说：

**recovered token 的候选分布 = target 分布去掉 draft token 本身，再重新归一化。**

这正是你引用那句话的数学含义。

在代码里，`NO_DRAFT_PROBS` 分支确实就是这么干的：

```python
if NO_DRAFT_PROBS:
    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
    prob = tl.load(
        target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
        mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
        other=0,
    )
```

这里的效果就是：

- 读出整条 `target_probs`
- 但把 `draft_token_id` 那个位置 mask 掉，置成 0
- 所以 recovered token 不可能再采到刚刚被拒绝的那个 token

---

## 这样做的概率分布是正确的吗？

**对，在“draft proposal 被视为 delta 分布”这个前提下，它是正确的。**

更准确地说：

### 接受阶段

在 `NO_DRAFT_PROBS` 分支里，accept/reject 判断相当于令 `draft_prob = 1`：

```python
draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
if NO_DRAFT_PROBS:
    draft_prob = 1
else:
    draft_prob = ...
target_prob = ...
uniform_prob = ...
if draft_token_id < vocab_size and (
    draft_prob > 0 and target_prob / draft_prob >= uniform_prob
):
    token_id = draft_token_id
else:
    rejected = True
    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
```

所以接受概率就是：

`min(1, p(d)/1) = p(d)`

这正好对应于 `q(d)=1` 的理论。

### 恢复阶段

理论上残差分布是：

- `x != d` 时，与 `p(x)` 成正比
- `x = d` 时，概率 0

代码虽然**没有显式除以归一化常数**，但它用了一个常见 trick：  
给每个 token 生成一个独立指数随机变量 `q ~ Exp(1)`，然后取

`argmax(prob / q)`

```python
prob = tl.maximum(target_prob - draft_prob, 0)
# NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
# `tl.argmax` will select the maximum value.

q = tl.load(
    q_ptr + req_idx * vocab_size + vocab_offset,
    mask=vocab_offset < vocab_size,
    other=float("-inf"),
)
recovered_id = tl.argmax(prob / q, axis=-1)
```

这个做法等价于“按 `prob` 成比例采样”，所以：

- 不需要先除以总和
- 只要 `prob` 和目标分布成正比即可

因此在 `NO_DRAFT_PROBS` 分支里：

- `prob(x)=p(x)` 对所有 `x!=d`
- `prob(d)=0`

就等价于“从 `p` 去掉 `d` 后重新归一化采样”。

---

## 你最关心的一点：这是不是通用地等价于 `max(p-q,0)`？

**不是通用等价，只是在 `q` 是 delta 分布时等价。**

也就是说：

- 如果真实 draft proposal 确实就是“只提一个确定 token，没有完整概率向量”，那这个分支是对的
- 如果真实 draft 模型其实有完整的 `q(x)`，那更一般、也更严格的理论形式还是：
  `max(p(x)-q(x), 0)`

所以这两种情况要区分：

1. **完整 draft 分布可得**
   用显式 `max(p-q,0)`

2. **只有 draft token id，没有完整 draft 分布**
   把 proposal 视为 delta 分布，残差就退化成“从 `p` 里排除这个 token 再重采样”

---

## 一句话直觉版

- `bonus token`：前面 speculative token 全验过了，target 顺手再往前走一步，多采一个真的下一个 token。
- `NO_DRAFT_PROBS`：不知道完整 `q(x)` 时，就把 draft 看成“100% 提议这个 token”，一旦它被拒绝，恢复采样就只能从“除了它之外的 target 分布”里选。

如果你愿意，我下一条可以继续把这两件事画成一个 **4-token 小例子表格**：

- 一张表讲 `bonus token`
- 一张表讲 `NO_DRAFT_PROBS` 下为什么“排除被拒 token”正好等价于 delta-`q` 的残差分布。

### 关于 4-token 小例子的问答

#### 问题

好的好的，请“把这两件事画成一个 4-token 小例子表格”来说明一下

#### 回答

下面我用两个 **4-token 小例子**来讲：

1. `bonus token` 到底是什么  
2. `NO_DRAFT_PROBS` 时，为什么“拒绝后排除 draft token 再重采样”是对的

---

## 例子一：`bonus token` 是怎么来的

假设某一轮 draft 一次提了 2 个 token：

- 第 1 个 draft token：`B`
- 第 2 个 draft token：`C`

所以这一轮 draft 提案是：

```text
前缀 + B + C
```

target 一次 forward 时，实际上会算出 3 个位置的 logits：

- 位置 1：验证 `B`
- 位置 2：验证 `C`
- 位置 3：如果前两个都对，那下一个真正的新 token 是什么

可以画成这样：

| 位置 | 含义 | 由谁决定 | 用来干什么 |
|---|---|---|---|
| pos1 | 验证第 1 个 draft token `B` | target logits | 看 `B` 接不接受 |
| pos2 | 验证第 2 个 draft token `C` | target logits | 看 `C` 接不接受 |
| pos3 | 已验证前缀之后的下一个 token | target logits | 当作 `bonus token` 候选 |

---

### 情况 A：两个 draft token 都被接受

比如：

| 位置 | draft 提议 | target 验证结果 |
|---|---|---|
| pos1 | `B` | 接受 |
| pos2 | `C` | 接受 |

那这一轮最终输出就会是：

```text
B, C, bonus
```

这里的 `bonus` 是从 **pos3 的 target 分布**直接采样来的。

比如 pos3 的 target 分布是：

| token | target 概率 |
|---|---:|
| A | 0.10 |
| B | 0.20 |
| C | 0.30 |
| D | 0.40 |

那 `bonus token` 就按这个分布采样，可能是 `D`，也可能是 `C/B/A`。

所以这一轮可能直接产出：

```text
B, C, D
```

这就是为什么叫 bonus:  
**本来你只 draft 了 2 个 token，但因为 2 个都通过了，target 又顺手多给了 1 个。**

---

### 情况 B：第 2 个 draft token 被拒绝

比如：

| 位置 | draft 提议 | target 验证结果 |
|---|---|---|
| pos1 | `B` | 接受 |
| pos2 | `C` | 拒绝，改成 recovered token `D` |

那么这一轮最终输出就只能是：

```text
B, D
```

不会再用 pos3 的 `bonus token`。

为什么？

因为 pos3 的 logits 本来是基于前缀：

```text
前缀 + B + C
```

算出来的。

但现在真实前缀已经变成：

```text
前缀 + B + D
```

前缀变了，pos3 的 logits 就不可信了，不能再用。

---

### 一张总表看懂 `bonus token`

| 情况 | draft token 是否全被接受 | 最终输出 |
|---|---|---|
| 全接受 | 是 | `accepted draft tokens + 1 bonus token` |
| 中途有拒绝 | 否 | `接受到拒绝点为止 + 1 recovered token` |

所以你可以把 `bonus token` 理解成：

**“全验过了以后，target 顺手多走一步得到的真正下一 token”。**

---

## 例子二：`NO_DRAFT_PROBS` 时为什么“排除被拒 token”是对的

现在讲你最关心的第二件事。

假设当前位置 draft 提议了 token `B`。  
但 vLLM 在这条路径里没有完整的 draft 分布 `q(x)`，只知道：

- draft 选了 `B`

于是它把 draft 当成一个 **delta 分布**：

| token | draft 分布 `q(x)` |
|---|---:|
| A | 0 |
| B | 1 |
| C | 0 |
| D | 0 |

也就是“draft 100% 提议 B”。

---

### target 分布假设如下

| token | target 概率 `p(x)` |
|---|---:|
| A | 0.10 |
| B | 0.20 |
| C | 0.30 |
| D | 0.40 |

---

## 第一步：接受概率

因为 draft 提的是 `B`，而这里把 draft 看成 `q(B)=1`，所以接受概率就是：

```text
P_accept(B) = min(1, p(B)/q(B)) = min(1, 0.20/1) = 0.20
```

也就是：

- 20% 概率接受 `B`
- 80% 概率拒绝 `B`

---

## 第二步：如果拒绝，recovered token 从哪来？

理论上残差分布是：

```text
max(p(x)-q(x), 0)
```

代进去看：

| token | `p(x)` | `q(x)` | `max(p-q,0)` |
|---|---:|---:|---:|
| A | 0.10 | 0 | 0.10 |
| B | 0.20 | 1 | 0 |
| C | 0.30 | 0 | 0.30 |
| D | 0.40 | 0 | 0.40 |

你会发现：

- `B` 的残差变成了 `0`
- 其他 token 保持它们的 `target` 概率质量

所以 recovered token 的未归一化权重就是：

| token | recovered 权重 |
|---|---:|
| A | 0.10 |
| B | 0 |
| C | 0.30 |
| D | 0.40 |

归一化之后：

总和 = `0.10 + 0.30 + 0.40 = 0.80`

| token | recovered 概率 |
|---|---:|
| A | 0.10 / 0.80 = 0.125 |
| B | 0 |
| C | 0.30 / 0.80 = 0.375 |
| D | 0.40 / 0.80 = 0.500 |

这就正好等价于：

**“从 target 分布里，把刚被拒绝的 `B` 排除掉，再重新归一化采样。”**

---

## 为什么这样最终还是还原成 `p(x)`？

我们把最终概率算一下。

### 对 `B`

`B` 只有一种来源：

- 第一步 draft 提到 `B`
- 并且被接受

所以

```text
P_final(B) = 1 * 0.20 = 0.20
```

正好等于 `p(B)=0.20`。

---

### 对 `A`

`A` 不可能在第一步直接出来，因为 draft 只提了 `B`。  
它只能来自“拒绝后 recovered”。

所以

```text
P_final(A) = P(拒绝 B) * P(recovered=A | 拒绝)
           = 0.80 * 0.125
           = 0.10
```

正好等于 `p(A)=0.10`。

---

### 对 `C`

```text
P_final(C) = 0.80 * 0.375 = 0.30
```

正好等于 `p(C)=0.30`。

---

### 对 `D`

```text
P_final(D) = 0.80 * 0.50 = 0.40
```

正好等于 `p(D)=0.40`。

---

## 一张总表看懂整个过程

### target 分布

| token | `p(x)` |
|---|---:|
| A | 0.10 |
| B | 0.20 |
| C | 0.30 |
| D | 0.40 |

### draft 退化分布

| token | `q(x)` |
|---|---:|
| A | 0 |
| B | 1 |
| C | 0 |
| D | 0 |

### 第一步接受部分

| token | 第一阶段贡献 |
|---|---:|
| A | 0 |
| B | 0.20 |
| C | 0 |
| D | 0 |

### 第二步 recovered 部分

| token | 拒绝后条件概率 | 乘上拒绝概率 0.80 后的贡献 |
|---|---:|---:|
| A | 0.125 | 0.10 |
| B | 0 | 0 |
| C | 0.375 | 0.30 |
| D | 0.500 | 0.40 |

### 最终总概率

| token | 第一阶段 | 第二阶段 | 最终 |
|---|---:|---:|---:|
| A | 0 | 0.10 | 0.10 |
| B | 0.20 | 0 | 0.20 |
| C | 0 | 0.30 | 0.30 |
| D | 0 | 0.40 | 0.40 |

最终恰好恢复成 target 分布 `p(x)`。

---

## 最直觉的一句话

### `bonus token`

- 前面的 speculative token 全通过了
- 那么 target 在这次 forward 里已经站在“真实前缀”的末尾
- 所以可以再直接采一个真正的下一个 token

### `NO_DRAFT_PROBS`

- 不知道完整 draft 分布时，就把 draft 看成“100% 只提这个 token”
- 如果它被拒绝，那补偿分布自然就是：
  **target 分布里除去这个 token 的剩余部分**

---

如果你愿意，我下一条还可以继续给你画一个更贴近代码的流程图，把 `rejection_sampler.py` 里的这几步对应到：

1. `bonus_logits`
2. `target_probs`
3. `recovered_token_ids`
4. `output_token_ids`

这样你再回头看 `rejection_sampler.py` 会更顺。

## 目录

- [一、概述](#一概述)
- [二、关键文件清单](#二关键文件清单)
- [三、配置与初始化流程](#三配置与初始化流程)
- [四、Eagle 模型架构详解](#四eagle-模型架构详解)
- [五、MTP 模型架构详解](#五mtp-模型架构详解)
- [六、Eagle vs MTP 关键设计对比](#六eagle-vs-mtp-关键设计对比)
- [七、EagleProposer 核心逻辑](#七eagleproposer-核心逻辑)
- [八、拒绝采样（Rejection Sampling）](#八拒绝采样rejection-sampling)
- [九、调度器集成](#九调度器集成)
- [十、完整运行流程总结](#十完整运行流程总结)
- [附录：Pre-Norm Transformer 中的残差传递](#附录pre-norm-transformer-中的残差传递)

---

## 一、概述

投机解码（Speculative Decoding）的核心思想：用一个**轻量的 draft 模型**快速生成多个候选 token，再用**完整的 target 模型**一次性验证，接受正确的、拒绝错误的。理想情况下，多个 token 可在一次 target forward 中全部通过验证，从而提升吞吐。

vLLM v1 实现了两种主要的投机解码方案：

- **Eagle**：独立的小型 draft 模型，接收 target 的 hidden states 和 token embeddings 作为输入
- **MTP（Multi-Token Prediction）**：模型自带的多 token 预测头，通常与 target 共享权重

两者在 vLLM 中**共用同一个 Proposer 类** —— `EagleProposer`，通过 `self.method` 区分行为。

---

## 二、关键文件清单

### 配置

| 文件 | 作用 |
|------|------|
| `vllm/config/speculative.py` | `SpeculativeConfig`：用户配置入口，自动检测 method，构建 draft `ModelConfig` |
| `vllm/config/vllm.py` | `VllmConfig`：持有 `speculative_config`，传递给 worker |
| `vllm/transformers_utils/configs/eagle.py` | `EAGLEConfig`：重写 HF config 的 `architectures` 字段 |

### Proposer（草案生成）

| 文件 | 作用 |
|------|------|
| `vllm/v1/spec_decode/eagle.py` | **`EagleProposer`**：Eagle 和 MTP 共用的 draft 提议器 |
| `vllm/v1/spec_decode/utils.py` | Triton 辅助 kernel（padded batch 处理） |
| `vllm/v1/spec_decode/metadata.py` | 投机解码元数据 |

### Eagle 模型实现

| 文件 | 作用 |
|------|------|
| `vllm/model_executor/models/llama_eagle.py` | Llama Eagle draft 模型 |
| `vllm/model_executor/models/qwen3_moe_eagle.py` | Qwen3-MoE Eagle draft 模型 |
| `vllm/model_executor/models/llama_eagle3.py` | Eagle3 变体 |

### MTP 模型实现

| 文件 | 作用 |
|------|------|
| `vllm/model_executor/models/deepseek_mtp.py` | DeepSeek MTP |
| `vllm/model_executor/models/qwen3_next_mtp.py` | Qwen3-Next MTP |
| `vllm/model_executor/models/qwen_mtp.py` | Qwen MTP |
| `vllm/model_executor/models/ernie_mtp.py` | Ernie MTP |

### Worker 与 Runner

| 文件 | 作用 |
|------|------|
| `vllm/v1/worker/gpu_model_runner.py` | 实例化 `EagleProposer`，调度 draft 和 target 的 forward |
| `vllm/v1/worker/gpu_worker.py` | 选择 V1/V2 model runner |

### 验证（拒绝采样）

| 文件 | 作用 |
|------|------|
| `vllm/v1/sample/rejection_sampler.py` | `RejectionSampler`：验证 draft tokens |
| `vllm/v1/sample/tree_rejection_sampler.py` | 树状投机解码的验证器 |

### 调度器

| 文件 | 作用 |
|------|------|
| `vllm/v1/core/sched/scheduler.py` | 统一调度，管理 `spec_token_ids` |
| `vllm/v1/engine/core.py` | Engine 层面的 draft 结果回传 |

---

## 三、配置与初始化流程

### 3.1 配置解析

用户传入 `speculative_config`（指定 method、draft model、`num_speculative_tokens` 等）后：

1. `SpeculativeConfig.__post_init__` 构建 `draft_model_config`
2. 自动检测 method 类型（`eagle` / `eagle3` / `mtp`）
3. **Eagle**：用 `EAGLEConfig` 重写 HF config 的 `architectures`，映射到 vLLM 的 `Eagle*` 模型类
4. **MTP**：重写 `model_type` / `architectures`（如 `qwen3_next` → `qwen3_next_mtp`）

```python
# vllm/config/speculative.py
if self.method in ("eagle", "eagle3"):
    eagle_config = EAGLEConfig(
        self.draft_model_config.hf_config,
        method=self.method,
        model_type="eagle",
    )
    self.draft_model_config.hf_config = eagle_config
```

`use_eagle()` 方法对 `eagle`、`eagle3`、`mtp` 三种 method 都返回 True：

```python
# vllm/config/speculative.py
def use_eagle(self) -> bool:
    return self.method in ("eagle", "eagle3", "mtp")
```

### 3.2 创建 Proposer

在 `GPUModelRunner.__init__` 中，若 `speculative_config.use_eagle()` 为 True，创建 `EagleProposer`：

```python
# vllm/v1/worker/gpu_model_runner.py
if self.speculative_config and get_pp_group().is_last_rank:
    if self.speculative_config.use_eagle():
        self.drafter = EagleProposer(self.vllm_config, self.device, self)
```

同时创建 `RejectionSampler` 用于验证。

### 3.3 加载模型与权重共享

先加载 target 模型，再调用 `EagleProposer.load_model(target_model)`：

```python
# vllm/v1/spec_decode/eagle.py — load_model
self.model = get_model(
    vllm_config=self.vllm_config,
    model_config=draft_model_config,
    enable_spec_decoding=True,
)
```

加载后进行权重共享：

**embed_tokens 共享：**

```python
# Eagle: 可选（检测 has_own_embed_tokens）
if hasattr(self.model, "has_own_embed_tokens"):
    if not self.model.has_own_embed_tokens:
        share_embeddings = True
else:
    # MTP: 总是共享
    share_embeddings = True

if share_embeddings:
    del self.model.model.embed_tokens
    self.model.model.embed_tokens = target_embed_tokens
```

**lm_head 共享：**

```python
# Eagle: 可选
if hasattr(self.model, "has_own_lm_head"):
    if not self.model.has_own_lm_head:
        share_lm_head = True
else:
    # MTP: 总是共享
    share_lm_head = True

if share_lm_head:
    del self.model.lm_head
    self.model.lm_head = target_language_model.lm_head
```

**MTP 额外共享 shared_head.head：**

```python
# 每个 MTP layer 的 shared_head.head 也指向 target 的 lm_head
for layer in items:
    sh = getattr(layer, "shared_head", None)
    if sh is not None and hasattr(sh, "head"):
        del sh.head
        sh.head = target_language_model.lm_head
```

---

## 四、Eagle 模型架构详解

### 4.1 Llama Eagle（经典版）

```python
# vllm/model_executor/models/llama_eagle.py
class LlamaModel(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer_id=0):
        # 词嵌入
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        # decoder layers，层号从 start_layer_id 开始
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(vllm_config, i == 0,
                prefix=f"model.layers.{i + start_layer_id}")
            for i in range(config.num_hidden_layers)
        ])
        # 融合投影层：[hidden_size*2] → [hidden_size]
        self.fc = ReplicatedLinear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, input_ids, positions, hidden_states):
        input_embeds = self.embed_tokens(input_ids)
        # 直接拼接 + 投影
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = hidden_states + residual
        return hidden_states, hidden_states  # 返回 tuple
```

### 4.2 Qwen3-MoE Eagle

```python
# vllm/model_executor/models/qwen3_moe_eagle.py
class Qwen3MoeEagleModel(nn.Module):
    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        # 先 norm 再拼接投影
        self.e_norm = RMSNorm(hidden_size)
        self.h_norm = RMSNorm(hidden_size)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        # 只有 1 层 decoder layer
        self.layers = nn.ModuleList([
            Qwen3MoeEagleDecoderLayer(vllm_config,
                prefix=f"model.layers.{start_layer_id}")
        ])
        self.norm = RMSNorm(hidden_size)

    def forward(self, input_ids, positions, hidden_states):
        input_embeds = self.embed_tokens(input_ids)
        e_feat = self.e_norm(input_embeds)       # norm(embed)
        h_feat = self.h_norm(hidden_states)      # norm(target_hidden)
        hidden_states = self.eh_proj(torch.cat([e_feat, h_feat], dim=-1))
        residual = torch.zeros_like(hidden_states)
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states  # 返回 tuple
```

### 4.3 关键设计

- **fc / eh_proj 层**：将 `[embed, hidden_states]` 拼接后投影回 `hidden_size`，融合词嵌入信息和 target 隐状态
- **Decoder layers 层号从 `target_layer_num` 开始**：续接 target 模型的层号，共享 KV cache 地址空间，避免 attention layer name 冲突
- **返回值**：返回 `(hidden_states, hidden_states)` 的 tuple，两个是同一个 tensor 的引用

---

## 五、MTP 模型架构详解

### 5.1 Qwen3-Next MTP

```python
# vllm/model_executor/models/qwen3_next_mtp.py
class Qwen3NextMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        # 融合投影
        self.fc = ColumnParallelLinear(hidden_size * 2, hidden_size, bias=False)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size)
        self.pre_fc_norm_embedding = RMSNorm(hidden_size)
        # 可能有多层 decoder layer
        self.layers = nn.ModuleList(
            Qwen3NextDecoderLayer(vllm_config,
                prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}")
            for idx in range(self.num_mtp_layers)
        )
        self.norm = RMSNorm(hidden_size)

    def forward(self, input_ids, positions, hidden_states,
                intermediate_tensors=None, inputs_embeds=None, spec_step_idx=0):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # 先 norm 再拼接投影
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None
        # 按 spec_step_idx 选择使用哪一层
        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions, hidden_states=hidden_states, residual=residual,
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states  # 返回单个 tensor（不是 tuple）
```

### 5.2 DeepSeek MTP

```python
# vllm/model_executor/models/deepseek_mtp.py
class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        self.enorm = RMSNorm(hidden_size)
        self.hnorm = RMSNorm(hidden_size)
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.shared_head = SharedHead(config, prefix)  # 每层有独立的 shared_head
        self.mtp_block = DeepseekV2DecoderLayer(vllm_config, prefix)

    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, spec_step_index=0):
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states
        return hidden_states

class DeepSeekMultiTokenPredictor(nn.Module):
    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, spec_step_idx=0):
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids, positions, previous_hidden_states, inputs_embeds,
        )

    def compute_logits(self, hidden_states, spec_step_idx=0):
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        # 每层通过自己的 shared_head 计算 logits
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits
```

---

## 六、Eagle vs MTP 关键设计对比

| 设计点 | Eagle (Llama) | Eagle (Qwen3-MoE) | MTP (Qwen3-Next) | MTP (DeepSeek) |
|--------|---------------|---------------------|-------------------|----------------|
| **融合方式** | `fc(concat(embed, hidden))` | `eh_proj(concat(e_norm(embed), h_norm(hidden)))` | `fc(concat(norm_e(embed), norm_h(hidden)))` | `eh_proj(concat(enorm(embed), hnorm(hidden)))` |
| **Decoder 层数** | `config.num_hidden_layers`（可多层） | **1 层** | `num_nextn_predict_layers`（可多层） | `num_nextn_predict_layers`（可多层） |
| **层号起始** | `target_layer_num` | `target_layer_num` | `num_hidden_layers` | `num_hidden_layers` |
| **多步选层** | 所有层都跑（完整 stack） | 单层 | `spec_step_idx % num_mtp_layers` | `spec_step_idx % num_mtp_layers` |
| **forward 返回值** | `(hidden, hidden)` tuple | `(hidden, hidden)` tuple | 单个 `hidden` tensor | 单个 `hidden` tensor |
| **compute_logits** | 全局 `logits_processor(lm_head)` | 全局 `logits_processor(lm_head)` | 全局 `logits_processor(lm_head)` | 每层 `shared_head.head` |
| **embed_tokens 共享** | 可选（检测 `has_own_embed_tokens`） | 可选 | **总是共享** | **总是共享** |
| **lm_head 共享** | 可选（检测 `has_own_lm_head`） | 可选 | **总是共享** | **总是共享** + 每层 `shared_head.head` 也共享 |

---

## 七、EagleProposer 核心逻辑

`EagleProposer` 是 Eagle 和 MTP 共用的 draft 提议器（`vllm/v1/spec_decode/eagle.py`）。

### 7.1 `propose()` 方法

#### 输入准备

```python
# 将 target_hidden_states 拷贝到缓冲区
self.hidden_states[:num_tokens] = target_hidden_states

# 构造 shifted input_ids：左移一位，末尾填入 next_token_ids
# 原始: [a1, b1, b2, c1, c2, c3]
# 移位: [b1, b2, c1, c2, c3, c3]
self.input_ids[:num_tokens - 1] = target_token_ids[1:]
# 替换每个 request 最后一个位置为采样得到的 next_token
# 结果: [a2, b2, b3, c2, c3, c4]
self.input_ids[last_token_indices] = next_token_ids
```

#### 第 1 步 Draft Forward

```python
ret_hidden_states = self.model(
    input_ids=input_ids,
    positions=self._get_positions(num_input_tokens),
    hidden_states=self.hidden_states[:num_input_tokens],  # target 的 hidden_states
    inputs_embeds=inputs_embeds,
)
# 区分 Eagle 和 MTP 的返回值
if self.method == "mtp":
    last_hidden_states = ret_hidden_states        # 单个 tensor
    hidden_states = last_hidden_states
else:
    last_hidden_states, hidden_states = ret_hidden_states  # tuple 解包

# 取采样位置的 hidden → 算 logits → argmax 得 draft token
sample_hidden_states = last_hidden_states[last_token_indices]
logits = self.model.compute_logits(sample_hidden_states)
draft_token_ids = logits.argmax(dim=-1)
```

#### 第 2 ~ N 步 Draft Forward

```python
for token_index in range(self.num_speculative_tokens - 1):
    input_ids = draft_token_ids_list[-1].int()
    positions += 1
    common_attn_metadata.seq_lens += 1
    # 将 Eagle/MTP 自身上一步的输出作为本步的 hidden_states 输入
    self.hidden_states[:batch_size] = hidden_states

    ret_hidden_states = self.model(
        input_ids=input_ids,
        positions=...,
        hidden_states=self.hidden_states[:input_batch_size],  # Eagle 自身的输出
    )
    # ... 同样的 logits → argmax 逻辑
```

### 7.2 hidden_states 数据流

```
Target Model forward
    → target_hidden_states [num_tokens, hidden_size]  （最后一层 + norm 后，所有 token 位置）
        ↓ (作为输入)
    Eagle/MTP Draft Model forward(input_ids, positions, hidden_states=target_hidden_states)
        → fc(concat(norm(embed), norm(target_hidden)))
        → Eagle decoder layers / MTP decoder layer
        → eagle_hidden_states   ← draft 模型自身的输出
            ↓
        last_hidden_states → [last_token_indices] → compute_logits → draft_token_id_1
        hidden_states → 作为第 2 步的输入
            ↓
        Eagle/MTP Draft Model forward(input_ids=draft_token_1, hidden_states=eagle_hidden)
            → draft_token_id_2
            → ...
```

**第 1 步 draft 用的是 target 的 hidden_states；第 2+ 步 draft 用的是 Eagle/MTP 自身上一步输出的 hidden_states。**

### 7.3 特殊 MTP 变体的 hidden_states 处理

对于 DeepSeek、Ernie 等 MTP 变体，下一步的 hidden_states 取自缓冲区而非模型返回值：

```python
if self.method in ("deepseek_mtp", "ernie_mtp", "longcat_flash_mtp", "pangu_ultra_moe_mtp"):
    hidden_states = self.hidden_states[last_token_indices]  # 从缓冲区取
else:
    hidden_states = hidden_states[last_token_indices]        # 从模型输出取
```

---

## 八、拒绝采样（Rejection Sampling）

实现位于 `vllm/v1/sample/rejection_sampler.py`，严格遵循论文 [https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192) 的算法。

### 8.1 术语

- **accepted tokens**：基于 draft 和 target 概率关系被接受的 token
- **recovered tokens**：被拒绝后，从调整后的概率分布中重新采样的 token
- **bonus tokens**：若所有 draft token 都被接受，额外从 target 概率中采样的 token
- **output tokens** = accepted + recovered + bonus

### 8.2 验证流程

```python
# 1. 取 target logits 在 draft 位置和 bonus 位置
bonus_logits = logits[bonus_logits_indices]
raw_target_logits = logits[target_logits_indices]

# 2. 采样 bonus token（从 target 概率）
bonus_token_ids = sampler(bonus_logits).sampled_token_ids

# 3. 对 target logits 应用 logits processors 和采样约束
target_logits = apply_logits_processors(raw_target_logits, ...)
target_logits = apply_sampling_constraints(target_logits, ...)
target_probs = target_logits.softmax(dim=-1)

# 4. 执行拒绝采样
output_token_ids = rejection_sample(
    draft_token_ids, num_draft_tokens,
    draft_probs, target_probs, bonus_token_ids, ...
)
```

### 8.3 拒绝采样算法

**贪心采样（Greedy）：**

```
对于每个 draft 位置 pos:
    target_argmax = argmax(target_probs[pos])
    output[pos] = target_argmax
    if draft_token_ids[pos] != target_argmax:
        拒绝，后续位置全部丢弃
        break

如果所有 draft 都被接受:
    追加 bonus token
```

**随机采样（Stochastic）：**

```
对于每个 draft 位置 pos:
    draft_prob = draft_probs[pos][draft_token_id]
    target_prob = target_probs[pos][draft_token_id]
    uniform = random()

    if target_prob / draft_prob >= uniform:
        接受 draft token
    else:
        拒绝，使用 recovered token（从 max(target - draft, 0) 分布采样）
        后续位置全部丢弃

如果所有 draft 都被接受:
    追加 bonus token
```

---

## 九、调度器集成

### 9.1 统一调度（无 draft/verify 两阶段）

vLLM v1 的调度器没有单独的"draft 阶段"和"verify 阶段"。每个 request 维护：

- `num_computed_tokens`：已经被 target 模型计算过的 token 数
- `num_tokens_with_spec = len(prompt) + len(output) + len(spec_token_ids)`

调度器每步让 `num_computed_tokens` 追上 `num_tokens_with_spec`。

```python
# vllm/v1/core/sched/scheduler.py
# 调度算法注释：
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

### 9.2 Draft Token 的生命周期

1. **Draft 生成后**：`EagleProposer.propose()` 返回 draft token ids
2. **回传调度器**：Engine `post_step` → `scheduler.update_draft_token_ids()` → 存入 `request.spec_token_ids`
3. **下次调度**：`schedule()` 将 `spec_token_ids` 包含在调度中，复制到 `scheduled_spec_decode_tokens`
4. **验证后修正**：`update_from_output` 根据拒绝数量回退 `num_computed_tokens`

```python
# 修正逻辑
num_accepted = len(generated_token_ids) - 1  # -1 是因为包含了 bonus
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
```

---

## 十、完整运行流程总结

### 循环流程

```
Schedule → Target Forward(验证) → 拒绝采样 → Draft Forward(提议) → 回传 draft → Schedule → ...
```

### 详细步骤

#### Step 1：调度器调度

- 统一处理所有请求，无 draft/verify 两阶段之分
- 若 request 有 `spec_token_ids`（上一步 draft 产生的），包含在本次调度中

#### Step 2：Target 模型 Forward（验证 + 生成 hidden_states）

- Target 模型对整个调度序列（包括 draft token）做一次 forward
- 输出 `hidden_states`：所有 token 位置的最后一层 + final norm 后的表示
- 从中取采样位置 → `compute_logits` → 得到 target logits

#### Step 3：拒绝采样

- `RejectionSampler` 验证上一步的 draft tokens
- 贪心：比较 draft token 与 target argmax
- 随机：比较 `target_prob / draft_prob` 与均匀随机数
- 输出 accepted + recovered + bonus tokens

#### Step 4：Draft 模型 Propose

- **准备输入**：取 target 的 `hidden_states`，构造 shifted `input_ids`
- **第 1 步 draft**：Eagle/MTP forward，输入为 target hidden_states → `compute_logits` → argmax 得第 1 个 draft token
- **第 2 ~ N 步 draft**：Eagle/MTP forward，输入为自身上一步的 hidden_states → 得后续 draft tokens
- 打包为 `[batch_size, num_speculative_tokens]`

#### Step 5：回传 Draft 结果

- Engine `post_step` 取出 draft token ids
- `scheduler.update_draft_token_ids()` 存入 `request.spec_token_ids`

#### Step 6：调度器修正

- 根据拒绝采样结果：`num_computed_tokens -= num_rejected`

---

## 附录：Pre-Norm Transformer 中的残差传递

### 什么是残差

在 Transformer 中，"残差"指残差连接（Residual Connection）中"跳过"子层的那条路径上的值：

```
output = x + sublayer(x)
```

其中 `x` 就是"残差"。

### Pre-Norm 结构的残差传递

vLLM 使用 Pre-Norm 结构，每层的逻辑为：

```
Layer N 输入: (hidden_states, residual)

┌─ input_layernorm(hidden_states, residual):
│    residual = hidden_states + residual   ← 把上层输出加到残差上
│    hidden_states = RMSNorm(residual)     ← 归一化后送给 attention
│
├─ hidden_states = self_attn(hidden_states)
│
├─ post_attention_layernorm(hidden_states, residual):
│    residual = hidden_states + residual   ← 把 attention 输出加到残差上
│    hidden_states = RMSNorm(residual)     ← 归一化后送给 MLP
│
└─ hidden_states = mlp(hidden_states)

Layer N 输出: (hidden_states, residual)
```

到最后一层：

```
final_norm(hidden_states, residual):
    residual_sum = hidden_states + residual   ← 把最后 MLP 输出加回
    normed = RMSNorm(residual_sum)            ← 归一化
    return (normed, residual_sum)

hidden_states, _ = self.norm(hidden_states, residual)
                 ↑ 取归一化后的结果，丢弃 residual_sum
```

### 为什么分开传递 hidden_states 和 residual

性能优化——将加法和归一化融合到一个 kernel 里（fused add + norm），减少 GPU 上的中间 tensor 分配和内存读写。代价是需要同时传递 `hidden_states` 和 `residual` 两个 tensor。



Ref:
https://zhuanlan.zhihu.com/p/2020161669217658288
