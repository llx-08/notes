---
title: "vLLM Decode Context Parallelism (DCP)：attention 计算与集合通信详解"
date: 2026-08-13
tags: [vLLM, DCP, Attention, MLA, KV Cache, 学习笔记]
---

# vLLM Decode Context Parallelism (DCP)：attention 计算与集合通信详解

> 代码基线：社区 vLLM `upstream/main @ b908a21f9a`（2026-08-12），仓库 `~/codes/vllm_comm`，工作树干净。
> 对应博客：<https://vllm.ai/blog/2026-08-07-decode-context-parallelism>
> 关键 PR（均已在 HEAD 祖先链上）：
> - `ac201a0eaf` [Feature] Support Decode Context Parallel (DCP) for MLA (**#23734**) — Moonshot AI 上游化的初版
> - `2396a61108` [Attention][MLA][DCP] Query replication for MLA decode (**#45964**)
> - 之后还有 20+ 个 DCP commit（A2A 打包、FlashInfer MLA、FP8 KV、hybrid attention、Kimi-K3……），所以**当前代码形态比博客描述更完备**。

---

## 0. 一句话抓住本质

> **DCP 把 KV cache 沿序列维切开分给多个 rank；每个 rank 只能算出「局部 softmax」的结果，rank 之间交换的不是 attention score，而是每个 (token, head) 的一个 log-sum-exp 标量**；靠这个标量把各家的局部结果重新加权求和，就得到与不切分完全等价的全局 attention 输出。

如果你熟悉 FlashAttention 的 split-K / flash-decoding，那 DCP 就是**把 split-K 的那些 split 从"一个 GPU 内的不同 CTA"搬到了"不同 GPU"**，合并公式一模一样，只是 reduce 从 shared memory 变成了 NCCL。

<!-- more -->

读这篇可以按图走：为什么切（图 1）→ 怎么切、block 怎么管（图 2）→ 一步里 KV 长什么样、通信搬什么（图 3）→ 因果三角其实只拆最后一行（图 4）→ decode 四步与 LSE（图 5、图 6）→ 通信怎么收（图 7、图 8）→ GQA 与 prefill（图 9、图 10）。ASCII 图保留作精确对照。

---

## 1. 为什么需要它：TP 下 KV 是冗余的

TP 切 KV 的最小单位是 **KV head**。一旦 `tp_size` 超过 `num_key_value_heads`，KV cache 就开始出现完整副本：

```
MLA（DeepSeek-V3 / Kimi-K2）：KV 被压成一个 low-rank latent，等价于「只有 1 个 KV head」
tp=4 时：

        rank0            rank1            rank2            rank3
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Q head 0-31  │ Q head 32-63 │ Q head 64-95 │ Q head 96-127│   ← Q 正常切
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ KV[0 .. L-1] │ KV[0 .. L-1] │ KV[0 .. L-1] │ KV[0 .. L-1] │   ← 4 份一模一样!
  └──────────────┴──────────────┴──────────────┴──────────────┘
                    ↑ 显存被白白浪费了 4 倍

GQA（Qwen3-235B, num_kv_heads=4）：tp=8 时每 2 个 rank 共享同一份 KV head → 冗余度 2
```

DCP 做的事：**把这份冗余换成序列切分**。

![TP 下 KV 冗余 vs DCP 沿序列切开](/imgs/vllm-dcp-tp-vs-dcp.svg)

```
tp=4, dcp=4（MLA）：

        rank0            rank1            rank2            rank3
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Q head 0-31  │ Q head 32-63 │ Q head 64-95 │ Q head 96-127│   ← Q 切法不变
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │  KV 的 1/4   │  KV 的 1/4   │  KV 的 1/4   │  KV 的 1/4   │   ← 序列维切开
  └──────────────┴──────────────┴──────────────┴──────────────┘
        每卡 KV 显存 ÷4  →  同样显存能塞 4 倍并发 / 4 倍上下文
```

### DCP group 的拓扑：TP group 内部的子群

DCP **不是新增一维并行**，它是在 TP group 里划分子群。
`vllm/distributed/parallel_state.py:1852-1866`：

```python
dcp_ranks = all_ranks
if dcp_size > 1:
    dcp_ranks = dcp_ranks.transpose(-1, -2)   # 让 DCP 成员在 TP 维上相邻
group_ranks = dcp_ranks.reshape(-1, dcp_size).unbind(0)
_DCP = init_model_parallel_group(..., group_name="dcp")
```

```
tp=8, dcp=2  →  4 个 DCP group，每组 2 个 rank：

  TP group (8 ranks)
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ r0 │ r1 │ r2 │ r3 │ r4 │ r5 │ r6 │ r7 │
  └────┴────┴────┴────┴────┴────┴────┴────┘
    └─┬──┘    └─┬──┘    └─┬──┘    └─┬──┘
     dcp0      dcp1      dcp2      dcp3     ← 组内原本 KV 完全重复，现在切开

约束的来源就清楚了：
  MLA : tp >= dcp  且  tp % dcp == 0
  GQA : (tp // num_kv_heads) >= dcp  且  (tp // num_kv_heads) % dcp == 0
        （Qwen3-235B num_kv_heads=4，tp=8 → dcp <= 2）
```

---

## 2. KV cache 到底怎么切：**交错**，不是连续大段

⚠️ 博客里画的是 "tokens 0–50K / 50K–100K / …" 的连续分段，**实际代码是按 `cp_kv_cache_interleave_size`（记作 `I`，默认 1）粒度 round-robin 交错**。

![交错切分、负载均衡与 virtual block 的紧凑存放](/imgs/vllm-dcp-interleave.svg)

### 调度侧：两套 `block_size`，乘倍的是记账单位

这里其实有两套「block size」，名字一样，语义不同。

- **物理 page**（GPU 上真分配的那块）：还是原来的 `kv_cache_spec.block_size`，比如 16 个 token slot。每张卡的 KV cache 就是很多个这样的 page，**物理大小没有变成 dcp 倍**。
- **调度器眼里的 logical block**：开了 DCP 之后被乘上 `dcp`。它不负责存数据，只负责「什么时候该再要一个 page ID」。

`vllm/v1/core/single_type_kv_cache_manager.py:59-60`：

```python
if dcp_world_size * pcp_world_size > 1:
    self.block_size *= dcp_world_size * pcp_world_size
```

乘倍的目的就一句话：**让「1 个 block ID」=「DCP 组里每张卡上的 1 个写满的物理 page」。**

以 `block_size=4, dcp=2, I=1` 为例。逻辑序列 8 个 token：

```
逻辑位置:  0  1  2  3  4  5  6  7
归属:     r0 r1 r0 r1 r0 r1 r0 r1
rank0 写入自己 page: t0 t2 t4 t6   → 正好 4 个，填满 1 个物理 page
rank1 写入自己 page: t1 t3 t5 t7   → 正好 4 个，填满 1 个物理 page
```

如果不乘：调度器仍按「每 4 个**逻辑** token 分配 1 个 block」，逻辑走到 4 就再发一个新 ID；但此时每张卡才存了 2 个 token，物理 page 半空。worker 侧 `virtual_block_size = 4 * 2 = 8`，会把 0–7 都映射进**同一个** page。调度器以为用了 2 个 block，worker 只用了 1 个 → block table、prefix cache、引用计数全对不齐。

乘上之后：调度器认为 1 个 block 覆盖 8 个逻辑 token；交错后每张卡刚好 4 个，填满各自那个物理 page。**组内所有 rank 共用同一张 block table、同一个 block ID**。prefix cache 的 hash 粒度也变成「一整页在每张卡上都写满了」，分配/引用计数完全不用改。

落到每张卡上，local 偏移是**紧凑重排**后的：rank0 的 slot `0,1,2,…` 依次是 `t0, t2, t4,…`，中间没有给别的 rank 留空洞。`PAD_SLOT_ID` 只出现在「这个 token 不属于我」的写入 mapping 上，不会把 KV cache 变成稀疏数组。

调度器假装「block 变长了」；worker 负责「按交错规则填满原来那么大的 page」。

### worker 侧：slot mapping 决定 token 归属

`vllm/v1/worker/block_table.py:413-428`

```python
virtual_block_size    = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
virtual_block_indices = pos // virtual_block_size
virtual_block_offsets = pos - virtual_block_indices * virtual_block_size

is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE \
           == TOTAL_CP_RANK

local_block_offsets = (virtual_block_offsets // (WS * I)) * I \
                    + (virtual_block_offsets % I)
```

`is_local == False` 的 token 直接写 `PAD_SLOT_ID` 丢弃。归属规则一句话：**token 位置 `pos` 归 rank `(pos // I) % dcp`**。

```
dcp=4, I=1（默认，逐 token 交错）：

pos:    0    1    2    3    4    5    6    7    8    9   10   11  ...
rank:  [0]  [1]  [2]  [3]  [0]  [1]  [2]  [3]  [0]  [1]  [2]  [3]

 rank0 本地 KV:  t0  t4  t8  ...        ← 在本地 cache 里是紧凑连续存放的
 rank1 本地 KV:  t1  t5  t9  ...
 rank2 本地 KV:  t2  t6  t10 ...
 rank3 本地 KV:  t3  t7  t11 ...


dcp=4, I=2（交错粒度 2）：

pos:    0    1  │  2    3  │  4    5  │  6    7  │  8    9  │ ...
rank:  [ 0    0 ]│[ 1    1 ]│[ 2    2 ]│[ 3    3 ]│[ 0    0 ]│ ...
```

**为什么要交错而不是连续切？—— 负载均衡。**

```
假设 seq_len = 10, dcp = 4

连续切（博客的画法）：            交错切（实际实现，I=1）：
  rank0: t0 t1 t2   (3)            rank0: t0 t4 t8   (3)
  rank1: t3 t4 t5   (3)            rank1: t1 t5 t9   (3)
  rank2: t6 t7 t8   (3)            rank2: t2 t6      (2)
  rank3: t9         (1)  ← 严重倾斜  rank3: t3 t7      (2)
                                    最多差 I 个 token，且 decode 每步增长也是轮转的
```

decode 是**同步的**：所有 rank 都要等最慢的那个算完才能 combine。所以任何长度倾斜都会直接变成 kernel 时间倾斜。交错切让各 rank 的 `seqused_k` 最多差 `I`。

每个 rank 的本地长度由 `get_dcp_local_seq_lens()` 算出（`vllm/v1/attention/backends/utils.py:964-1001`）：

```python
base      = seq_lens // I // dcp_size * I
remainder = seq_lens - base * dcp_size
remainder = clip(remainder - rank * I, 0, I)
local_len = base + remainder
```

这个张量会作为 `seqused_k` 直接喂给 FA / FlashMLA kernel。

### 一步 decode 里：每个 rank 的 KV cache 长什么样

**Decode attention 期间，KV cache 始终保持切开，不会被 gather，也不会在 step 结束时拼回去。** 搬走的是 Q、LSE 和 attention 输出，不是 cache 里的 KV 字节。

![Decode 一步里 KV cache 钉在本地，通信只搬 Q、LSE 和 out](/imgs/vllm-dcp-kv-stays-comm-moves.svg)

以 `tp=2, dcp=2, I=1`，正在 decode 第 8 个 token（下标 7）为例：

```
rank0 物理 page（紧凑）:  [ t0 | t2 | t4 | t6 | 空... ]
rank1 物理 page（紧凑）:  [ t1 | t3 | t5 | t7 | 空... ]
                              本步新写入的是 t7，只追加到 rank1
```

计算过程中这两块 cache **原样拿去点乘**：

- kernel 的 `seqused_k` 是本地长度（这里两边都是 4），看到的就是连续 4 个 KV slot
- kernel **不知道**它们在全局是偶数还是奇数；对它来说就是「1 个 Q × 4 个 K」
- 另一张卡上的 KV 本 rank 根本读不到，也不需要读

跨 step 时 cache 按 round-robin 往前长：下一个 token t8 只追加到 rank0 的下一个 slot；t9 只追加到 rank1。所以「中途」每个 rank 手里永远是自己那条交错子序列，长度约 `seq_len / dcp`。它不是半成品全局 cache。

GQA 有一个局部例外：当前 step 新产生的 K/V 先以 **dense tensor** 进 kernel（本 rank 手里是完整的当前 token），历史才走上面那份切开的 cache；算完后再按 `(pos // I) % dcp` 写入所属 rank。MLA 则是新 token 的 latent 已经写进所属 rank 的 cache，decode 不必拆这两段。

---

## 3. ⭐ Decode 时 attention 怎么算（核心）

代码：`vllm/model_executor/layers/attention/mla_attention.py:932-975`

### 3.1 因果下三角：decode 只拆最后一行，不是两块小三角

Q **按 head 切，不按 token 切**。Decode 每步通常只有最新那一个 query。完整因果矩阵里，DCP 真正在算的是**最后一行**：同一个 `q_t` 去点全部历史 K。奇偶切开的是这一行的**列（K）**。

![因果下三角没有被切成两块：decode 只算最后一行](/imgs/vllm-dcp-causal-last-row.svg)

```
完整因果矩阵（6 token）          decode 真正在算的：

     k0 k1 k2 k3 k4 k5                 q5 · [k0 k1 k2 k3 k4 k5]
  q0  *  .  .  .  .  .                         r0 r1 r0 r1 r0 r1
  q1  *  *  .  .  .  .
  q2  *  *  *  .  .  .           拆成：
  q3  *  *  *  *  .  .             rank0: q5 × [k0,k2,k4] → o0, lse0
  q4  *  *  *  *  *  .             rank1: q5 × [k1,k3,k5] → o1, lse1
  q5  *  *  *  *  *  *  ← 只算这行
     r0 r1 r0 r1 r0 r1             再 LSE 加权加回来，得到完整的这一行
```

常见误解是：rank0 拿偶数 token 做自己的小三角、rank1 拿奇数 token 做自己的小三角，再把两块三角拼回去。**那会把 Q 也按序列切开，DCP 不这么干。** 因果性是靠「query 是最新 token，本地 K 全是历史」保证的；所以 context 段 kernel 用 `causal=False`。

Prefill 不能靠同一套「拆最后一行」：整张下三角里每一行的合法 K 集合都不同（`q2` 不能看 `k4`）。所以 chunked prefill 直接 AllGather 历史 KV，在完整序列上算普通因果注意力。

### 3.2 全景图（dcp=2，每 rank 2 个 Q head，1 个 decode token）

用最小例子：`H_local = 2`，`dcp = 2` → 组内共 4 个 head（h0..h3）；序列 8 个 token，I=1。

![Decode 四步：AllGather Q、本地 attention、AllGather LSE、ReduceScatter](/imgs/vllm-dcp-decode-pipeline.svg)

```
════════════════ Step 0：起点 ════════════════
       rank0                              rank1
  ┌─────────────────────┐           ┌─────────────────────┐
  │ Q: q[h0] q[h1]      │           │ Q: q[h2] q[h3]      │  ← Q 按 TP 切（各 2 head）
  │ KV: t0 t2 t4 t6     │           │ KV: t1 t3 t5 t7     │  ← KV 按序列交错切
  └─────────────────────┘           └─────────────────────┘

  问题：rank0 只有 h0/h1 的 Q，却只有一半的 KV
        rank1 只有 h2/h3 的 Q，也只有一半的 KV
        → 谁都算不出任何一个 head 的完整 attention


════════════════ Step 1：AllGather(Q, dim=1=head) ════════════════
  self.dcp_manager.query_gather(mqa_q)        # mla_attention.py:946

       rank0                              rank1
  ┌─────────────────────────┐       ┌─────────────────────────┐
  │ Q: q[h0] q[h1] q[h2] q[h3]│ ◀──▶ │ Q: q[h0] q[h1] q[h2] q[h3]│
  │ KV: t0 t2 t4 t6         │       │ KV: t1 t3 t5 t7         │
  └─────────────────────────┘       └─────────────────────────┘
       ↑ head 维被撑开 dcp 倍：[B, 2, 576] → [B, 4, 576]

  ★ 为什么便宜：decode 时 B = 1 token/请求，Q 只有 1×H_local×576 那么大
    （MLA: 576 = kv_lora_rank 512 + qk_rope_head_dim 64）


════════════════ Step 2：本地 attention（各算全部 head，但只对本地 KV）════════════════
  attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)
  # kernel 参数：seqused_k = dcp_context_kv_lens（本 rank 的本地长度）

  rank0 算 4 个 head 对 {t0,t2,t4,t6} 的 attention：
       o0[h0..h3]  (partial!)      lse0[h0..h3]
  rank1 算 4 个 head 对 {t1,t3,t5,t7} 的 attention：
       o1[h0..h3]  (partial!)      lse1[h0..h3]

  ★ "partial" 的含义：softmax 的分母只统计了自己那一半 token
        o_n[h] = ( Σ_{j∈S_n} exp(s_j) · v_j ) / exp(lse_n[h])
        lse_n[h] = log Σ_{j∈S_n} exp(s_j)
    所以 o0 和 o1 都不是正确答案，但它们的信息量加起来是完整的


════════════════ Step 3：AllGather(LSE) + 缩放 ════════════════
  lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape((N, B, H))   # common.py:236

  交换的只有 lse：[B, H] 的 fp32 —— B=1,H=4 时才 16 字节!
  attention score / logits 从头到尾没有跨 rank 传输过。

       rank0 现在知道：lse0[h] 和 lse1[h]（对所有 h）
       rank1 现在知道：lse0[h] 和 lse1[h]（对所有 h）  ← 两边算出同一个 lse_g

  triton kernel _correct_attn_cp_out_kernel（common.py:37-121）：
       lse_g[h] = logsumexp over n of lse_n[h]          # 全局归一化因子
       factor   = exp(lse_myrank[h] - lse_g[h])         # 本 rank 的权重
       o *= factor

       rank0: a[h] = o0[h] · exp(lse0[h] - lse_g[h])
       rank1: b[h] = o1[h] · exp(lse1[h] - lse_g[h])

  此时 a[h] + b[h] 就是正确答案（见 §3.4 推导）—— 还差一次跨 rank 求和


════════════════ Step 4：ReduceScatter(out, dim=1=head) ════════════════
  out = cp_group.reduce_scatter(out, dim=1)             # common.py:255

              h0        h1        h2        h3
  rank0 :   a[h0]     a[h1]  │  a[h2]     a[h3]
  rank1 :   b[h0]     b[h1]  │  b[h2]     b[h3]
            └── 段0 ──┘      └── 段1 ──┘
                  │                │
        跨 rank 求和 ↓      跨 rank 求和 ↓
            a+b (h0,h1)        a+b (h2,h3)
                  │                │
                  ▼                ▼
               rank0             rank1

  ★★ 这一步一箭双雕：
     (1) 跨 rank 求和  → 完成 LSE 加权合并，得到精确的全局 attention 输出
     (2) 沿 head 维切回 → 每个 rank 只留自己原本 TP 负责的 H_local 个 head

     head 维被 Step 1 撑开，在这里被收回。出口形状 [B, 2, 512] 正好对上
     后续的 v_up_proj + o_proj，TP 语义完全恢复，下游一行代码都不用改。
     通信量相比 all_reduce 省一半。

     这就是博客里那个名字的含义：AllGather Q → Compute → AllGather + ReduceScatter
```

一步 decode 的通信清单（默认 `cp_lse_ag_out_rs`）：

| 时机 | 操作 | 搬什么 | KV cache 是否参与 |
|---|---|---|---|
| attention 前 | `AllGather(Q, dim=1=head)` | 当前 decode token 的 Q heads | 否。qrep 开启则跳过 |
| 本地计算 | `forward_mqa` / FA | 无通信 | **只读本 rank 切开的 cache** |
| combine | `AllGather(LSE)` | 每 head 一个 fp32 | 否 |
| combine | 本地 `o *= exp(lse_local - lse_g)` | 无通信 | 否 |
| combine 结束 | `ReduceScatter(out, dim=1=head)` | 已缩放的 attention 输出 | 否。同时完成求和 + 按 TP 切回 head |

A2A 后端把表里后两行通信收成一次 `all_to_all_single`（out 和 lse 打包）。PCP 联用时最后一步改成 `AllReduce(out)`。**没有任何一条路径在 decode 里 AllGather KV cache。**

### 3.3 形状变化速查（MLA, DeepSeek-V3, tp=8, dcp=2）

```
num_heads(total)=128, tp=8 → H_local=16;  dcp=2 → H_group=32
kv_lora_rank=512, qk_rope_head_dim=64, B=num_decode_tokens

  q_proj / kv_b_proj  ──▶  mqa_q         [B, 16, 576]
        AllGather(dim=1) ──▶ mqa_q       [B, 32, 576]   ← 撑开
        forward_mqa      ──▶ attn_out    [B, 32, 512]   partial（注意 MLA 出口是 latent 维）
                             lse         [B, 32]        fp32
        AllGather(lse)   ──▶ lses        [ 2, B, 32]    ← 极小
        correct_attn_out ──▶ attn_out    [B, 32, 512]   已缩放（原地写回）
        ReduceScatter(dim=1)─▶ attn_out  [B, 16, 512]   ← 收回，且已求和完成
        _v_up_proj       ──▶ output      [B, 16, 128]   v_head_dim=128
```

**注意 `v_up` 投影在 combine 之后做**：所以跨 rank 搬运的是 512 维的 latent，而不是 128 维的 value。这一步没省通信（512 > 128），但避免了在每个 rank 上重复做 32 个 head 的 `v_up` GEMM。

具体通信量（上表同一配置，1 个 decode token，bf16）：

```
Q AllGather      16 * 576 * 2 B   = 18.4 KB     ← 字节不多，但多一次 collective
LSE AllGather    32 * 4 B         = 128 B       ← 可以忽略，独立小消息才疼
out ReduceScatter ~16 * 512 * 2 B = 16.4 KB     ← 真正的通信主体
```

所以「Q all-gather 是瓶颈」通常不成立：痛的是 **collective 次数** 和 **out 的 512 维 latent**，不是 Q 的字节数。`VLLM_DCP_Q_REPLICATE` 和 A2A 都是在砍次数，不是在砍 Q 的 payload。

### 3.4 为什么 LSE 加权是**精确**的（不是近似）

![LSE 加权合并：局部 softmax 乘上各 rank 抢到的概率质量](/imgs/vllm-dcp-lse-combine.svg)

设 head `h` 的全部 key 被切成 N 个不相交集合 `S_1..S_N`，`s_j` 是 scaled score：

```
真值：           o* = Σ_j exp(s_j)·v_j / Σ_j exp(s_j)

rank n 的局部量： lse_n = log Σ_{j∈S_n} exp(s_j)
                 o_n   = Σ_{j∈S_n} exp(s_j)·v_j / exp(lse_n)
                       ⇒ 分子 Σ_{j∈S_n} exp(s_j)·v_j = o_n · exp(lse_n)

全局归一化：      lse_g = log Σ_n exp(lse_n) = log Σ_j exp(s_j)     ← 因为 S_n 不相交且并集完备

合并：  Σ_n o_n · exp(lse_n - lse_g)
      = Σ_n [ o_n · exp(lse_n) ] / exp(lse_g)
      = Σ_n [ Σ_{j∈S_n} exp(s_j)·v_j ] / Σ_j exp(s_j)
      = Σ_j exp(s_j)·v_j / Σ_j exp(s_j)
      = o*        ∎
```

**数值验算**（head 维简化成标量 v，方便手算复核）：

```
rank0: scores = [1.0, 2.0], v = [1, 2]
rank1: scores = [3.0],      v = [10]

rank0: Σexp = e¹+e² = 2.7183+7.3891 = 10.1073   lse0 = ln(10.1073) = 2.31326
       o0 = (2.7183·1 + 7.3891·2)/10.1073 = 17.4965/10.1073 = 1.73101
rank1: Σexp = e³ = 20.0855                       lse1 = 3.00000
       o1 = 200.855/20.0855 = 10.0

lse_g   = ln(10.1073 + 20.0855) = ln(30.1928) = 3.40767
factor0 = exp(2.31326 - 3.40767) = exp(-1.09441) = 0.33476
factor1 = exp(3.00000 - 3.40767) = exp(-0.40767) = 0.66524

合并  = 1.73101·0.33476 + 10.0·0.66524 = 0.57937 + 6.65240 = 7.23177
真值  = (2.7183 + 14.7781 + 200.855)/30.1928 = 218.351/30.1928 = 7.23189
                                                        ✅ 吻合（残差来自我手算的舍入）
```

注意 `factor0 + factor1 = 1.0`——权重恰好是各 rank "抢到多少 softmax 质量" 的归一化占比。

kernel 里的数值稳定处理（`common.py:80-119`）：
```python
lse = where(isnan(lse) | (lse == +inf), -inf, lse)   # 脏值一律当"没贡献"
lse_max = max_n(lse);  lse_max = where(lse_max == -inf, 0, lse_max)
lse = lse - lse_max                                  # 先减最大值再 exp，防溢出
lse = log(sum(exp(lse))) + lse_max                   # 稳定版 logsumexp
...
output = where(factor == 0.0, 0.0, output)           # 避免 0 * inf = NaN
```

### 3.5 空分片：短序列的坑

`mask_dcp_empty_shards_()`（`common.py:9-35`）在 all-gather 之前把**本地 KV 长度为 0 的行的 LSE 填成 `-inf`**：

```python
empty_rows = (row_indices >= query_start_loc[-1]) | (seq_lens[sequence_indices] == 0)
lse.masked_fill_(empty_rows[:, None], float("-inf"))
```

```
seq_len = 2, dcp = 4, I = 1：
   rank0: t0      (1 token)
   rank1: t1      (1 token)
   rank2: (空!)   lse = 未定义/垃圾值
   rank3: (空!)   lse = 未定义/垃圾值

   不 mask → 垃圾 lse 进 logsumexp → 结果污染或 NaN
   mask 成 -inf → factor = exp(-inf - lse_g) = 0 → 贡献为 0 ✅
```

CUDA graph 场景下 padding 出来的行也走同一条路（`row_indices >= query_start_loc[-1]`）。

---

## 4. combine 的三种实现

`vllm/v1/attention/ops/dcp_utils.py:635-668` 按 `--dcp-comm-backend` 和硬件能力选：

| 实现 | 集合通信 | 说明 |
|---|---|---|
| `cp_lse_ag_out_rs` | AllGather(lse) + ReduceScatter(out) | 默认路径，即 §3 讲的 |
| `dcp_a2a_lse_reduce` | 一次 `all_to_all_single` | `dcp_comm_backend="a2a"`，见下 |
| `DirectDCPA2AWorkspace.lse_reduce` | NVLS symmetric memory 直写 | 绕开 NCCL，靠 `_symm_mem_spans_group()` 探测 multicast 支持 |
| `cp_lse_ag_out_ar` | AllGather(lse) + AllReduce(out) | 与 PCP 联用时（head 维不能切） |

### AG+RS vs A2A

![默认 AG+RS 对比一次打包 A2A](/imgs/vllm-dcp-agrs-vs-a2a.svg)

PCP（Prefill Context Parallel）是另一条切 context 的路，不要和 DCP 混成「都叫 CP」。prefill 的 query 本身很长，不能靠「AllGather 很短的 Q、再按 head ReduceScatter」这套 decode 特化；head 维经常切不动，所以 combine 走 `cp_lse_ag_out_ar`（AllGather LSE + AllReduce out）。DCP 和 PCP 可以叠，但约束更严，本文只把 PCP 当作「为什么还有 AR 这条路径」的背景。

```
【AG + RS】common.py:225-263          两次集合通信

  ①  AllGather(lse)   [B,H] fp32           很小，但是一次独立的 collective（有固定开销）
  ②  各 rank 本地缩放自己的 [B,H,D]
  ③  ReduceScatter(out, dim=1)             搬 (N-1)/N · B·H·D


【A2A】dcp_alltoall.py:399-469        一次集合通信

  ①  把 out 和 lse 打包进同一个 buffer（PR #41160）
      send_buffer: [N, B, H_per_rank, D + lse_pack_dim]
      lse_pack_dim = fp32 的 lse 塞进几个 output-dtype slot（bf16 → 2 个）
                     ↑ PR #47801 修过这里的 bf16 bitcast crash

           发给 rank0        发给 rank1     ...
      ┌──────────────────┬──────────────────┐
      │ o[h0,h1] │ lse   │ o[h2,h3] │ lse   │   ← 只发对方需要的 head 分片
      └──────────────────┴──────────────────┘

  ②  all_to_all_single(recv, send)          一次，搬 (N-1)/N · B·H·(D+pack)

  ③  收到后每个 rank 手里有「自己那些 head 的 N 份 partial + N 份 lse」
      → 纯本地做精确加权（_dcp_a2a_unpack_combine），零额外通信

     rank0 收到：                      → 本地 combine
     ┌────────────────────────────┐
     │ from r0: o0[h0,h1], lse0   │
     │ from r1: o1[h0,h1], lse1   │  ⇒ Σ_n o_n·exp(lse_n - lse_g)  → out[h0,h1]
     └────────────────────────────┘

  收益：collective 从 2 次降到 1 次；lse 搭 output 的车，省掉一次独立小 collective
        （小消息的 NCCL 固定开销在 decode 这种 latency-bound 场景很不划算）
```

数据量对比（N=dcp, D=512, bf16）：

```
                 AG+RS                              A2A
Q gather    (N-1)·B·H_local·576·2B            同（或用 qrep 完全省掉，见 §5）
lse          (N-1)·B·H·4B    ← 独立 collective  0（打包同行）
out          (N-1)/N·B·H·512·2B                (N-1)/N·B·H·(512+2)·2B
collective    3 次                              2 次
```

`_lse_weighted_combine()`（`dcp_alltoall.py:36-102`）是纯 PyTorch 参考实现，测试用；想理解合并数学看它比看 triton 清楚。

---

## 5. 把 Q AllGather 也省掉：`VLLM_DCP_Q_REPLICATE`（PR #45964）

思路：让 `q_proj` / `q_b_proj` 在 DCP group 内**冗余计算**，这样每个 rank 天生就有整组的 Q head，decode 直接跳过 all-gather。

![VLLM_DCP_Q_REPLICATE：组内共享 q_proj 权重](/imgs/vllm-dcp-qrep.svg)

`vllm/model_executor/layers/linear.py:598-636`：

```python
class DCPGroupColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        self.group_size = max(dcp_world_size, 1)
        self.rank_in_group = rank % self.group_size
        super().__init__(*args, **kwargs,
                         tp_rank=rank // self.group_size,      # ← 按 group 而非按 rank 切
                         tp_size=world_size // self.group_size)
```

```
tp=8, dcp=2，总共 128 head：

  普通 ColumnParallelLinear：            DCPGroupColumnParallelLinear：
    r0: head 0-15                          r0: head 0-31  ┐
    r1: head 16-31                         r1: head 0-31  ┘ 组内权重完全相同
    r2: head 32-47                         r2: head 32-63 ┐
    r3: head 48-63                         r3: head 32-63 ┘
    ...                                    ...
    decode 必须 AllGather(Q)               decode 直接跳过 AllGather ✅
                                           prefill 用 _local_view() 切回自己那 16 个
```

启用（`vllm/model_executor/models/deepseek_v2.py:1015-1019`）：
```python
qrep_enabled = (envs.VLLM_DCP_Q_REPLICATE
                and parallel_config.decode_context_parallel_size > 1
                and parallel_config.prefill_context_parallel_size <= 1)
```
调用点 `mla_attention.py:945-947`：`if not qrep_decode: mqa_q = self.dcp_manager.query_gather(mqa_q)`

代价：q_proj 的 GEMM 多算 dcp 倍。decode 时 batch 小、这块 GEMM 便宜，换掉一次通信通常划算——但这是**要实测的 trade-off**，不是无条件的赢。

---

## 6. GQA 路径为什么要算两遍

`vllm/v1/attention/backends/flash_attn.py:1215-1389`

GQA 下当前 step 新产生的 K/V 是以 dense tensor 传进 `forward` 的（本 rank 手里就有完整的），只有 **cache 里的历史 context** 是 DCP 切分的。于是必须拆两段算，再本地 merge：

![GQA 两段 attention：cache 走 DCP，当前 token 本地 causal，再 LSE merge](/imgs/vllm-dcp-gqa-two-pass.svg)

decode 的新 query 一定在全部 cached KV 之后，所以 context 段对本地 KV 做非因果 attention 是对的——那些 key 全部合法。当前 token 段则可能同一 step 里有多个 query（chunked prefill / spec decode），彼此之间仍要 causal。两段的 key 集合不相交，最后一次 `merge_attn_states` 又是 §3.4 那套公式，只是纯本地、零通信。

```
                     ┌── context 段：KV 来自 cache，DCP 切分 ────────────────┐
                     │  query_across_dcp = AllGather(query, dim=1)          │
  query ────────────▶│  flash_attn_varlen_func(                             │
    [T,H_local,D]    │      q=query_across_dcp,  k/v = key_cache/value_cache│
                     │      seqused_k = dcp_context_kv_lens,   ← 本地长度   │
                     │      causal = False )                   ← 关键!      │
                     │  → context_out [T,H_group,D], context_lse [H,T]      │
                     │  → self.dcp_combine(...) 做 §3 那套 LSE 合并          │
                     │  → context_out_cor [T,H_local,D], context_lse_cor    │
                     └──────────────────────────────────────┬───────────────┘
                                                            │
                     ┌── 当前 token 段：KV 是本 step 的 dense tensor，未切分 ─┐
                     │  flash_attn_varlen_func(                             │
  query ────────────▶│      q=query,  k=key, v=value,                       │
                     │      cu_seqlens_k = cu_seqlens_q,                    │
                     │      causal = attn_metadata.causal )   ← 这里要 causal│
                     │  → query_out [T,H_local,D], query_lse                │
                     └──────────────────────────────────────┬───────────────┘
                                                            │
                          merge_attn_states(output,         ▼
                              prefix = context_out_cor, prefix_lse = context_lse_cor,
                              suffix = query_out,       suffix_lse = query_lse)
                          ↑ 又是一次 LSE 加权合并，但是纯本地（prefix/suffix 语义）
```

为什么 context 段 `causal=False`：那段 KV 全都在当前 query 之前，天然满足因果性，不需要 mask。

几个细节：
- **FA 返回的 lse 形状是 `[H, B]`，而 DCP combine 要 `[B, H]`** → `flash_attn.py:1352-1358` 前后各 `.transpose(0,1)` 一次
- `max_dcp_context_kv_len == 0` 时（纯 prefill、无历史 context）走快速路径，完全不做 DCP 通信（`flash_attn.py:1237-1259`）
- `should_split_fa2_dcp_context_attention` / `run_split_fa2_dcp_context_attention`：FA2 在 decode/prefill 混批 + Qwen3.5 那个形状下有问题，需要拆开跑的 workaround
- MLA 不需要这个两段结构：走 `forward_mqa` 时新 token 的 latent 已经写进 cache 了

---

## 7. Prefill / chunked context：这里要**倒付**通信

DCP 下本地 KV 不完整，chunked prefill 需要访问完整历史 KV，只能拉全。

![Decode 保持 KV 切开；chunked prefill 要 all-gather 历史 KV](/imgs/vllm-dcp-decode-vs-prefill.svg)

`vllm/model_executor/layers/attention/mla_attention.py:2106-2128`：

```python
# Note(hc): The local kvcache is incomplete when DCP is triggered,
# an additional kvcache allgather across the DCP group is therefore required,
# so the workspace has to be enlarged by 1/DCP relative to the original TP allocation.
self.chunked_prefill_workspace = torch.empty(
    (workspace_size + workspace_size // dcp_world_size, kv_lora_rank + qk_rope_head_dim), ...)
self.dcp_manager.init_kv_gather(self.chunked_prefill_workspace, workspace_size)
```

```
chunked_prefill_workspace 布局（dcp=4）：

  ┌──────────────────────────────────────────┬──────────┐
  │  gathered KV  (workspace_size)           │ local KV │
  │  ← all_gather_into_tensor 的落点         │ (ws/dcp) │  ← 本 rank 的分片，作为 send buf
  └──────────────────────────────────────────┴──────────┘
              实际 gather 调用在 mla_attention.py:2749
              dcp_manager.kv_gather(cur_allgather_kvcache, local_gathered_kvcache)
```

**所以 DCP 只在 decode 省显存/省带宽，prefill 要额外掏一次 KV all-gather**。这就是它叫 **Decode** CP 的原因，也是它天然适合和 PD 分离配合的原因——把 prefill 摘出去，DCP 的代价就消失了，只剩收益。

和 PD 分离叠在一起时，P 产出的是按序列排好的 KV；D 的每个 rank 只该收下 `(pos // I) % dcp == my_rank` 的 token，写进自己那份紧凑 cache。写错 slot 通常不会立刻 crash，attention 会静默算偏。社区 KV connector 的 global↔block 换算在 DCP 下修过多次（`#46394`, `#41549`, `#45371`）。内部栈上「D 节点开 DCP 时，P→D 的 KV 怎么按交错规则散布」仍待实测，见 §11。

---

## 8. 配置与约束

```bash
# 离线
LLM(..., tensor_parallel_size=8, decode_context_parallel_size=2)

# 在线
vllm serve ... --tensor-parallel-size 8 --decode-context-parallel-size 2

# 相关旋钮
--cp-kv-cache-interleave-size N     # 交错粒度，默认 1
                                    # 要求 block_size >= N 且 block_size % N == 0
--dcp-comm-backend a2a              # 用 all-to-all 版 combine
VLLM_DCP_Q_REPLICATE=1              # 冗余 q_proj，省掉 decode 的 Q all-gather
```

| 后端 | 约束 |
|---|---|
| MLA (DeepSeek-V2/V3/R1, Kimi-K2) | `tp >= dcp` 且 `tp % dcp == 0` |
| GQA (Qwen3-235B, Llama) | `(tp // num_kv_heads) >= dcp` 且 `(tp // num_kv_heads) % dcp == 0` |

`dbd80cc031 [UX] DCP Topology Validation (#49777)` 加了启动期校验。

---

## 9. 容易踩/容易误解的点

**误解**
1. ❌ "DCP 是切 batch" → 切的是 **KV 序列维**；head 维只是被临时撑开又收回
2. ❌ "rank 之间要交换 attention score / logits" → 只交换每个 (token, head) **一个 LSE 标量**
3. ❌ "KV 按连续大段切"（博客画法）→ 实际是按 `I` **交错**，为了负载均衡
4. ❌ "合并是近似的" → 数学上**精确等价**，见 §3.4 推导与验算
5. ❌ "Q all-gather 是瓶颈" → decode 时 B 很小；真正的通信主体是 `[B, H, 512]` 的 out。Q 的痛点是多一次 collective，不是 payload
6. ❌ "DCP 和 PCP 是同一件事" → DCP 特化 decode（短 Q，可 RS 收回 head）；PCP 切 prefill 长 context，head 维切不动时只能 AR
7. ❌ "virtual block 让每张卡存稀疏 KV" → scheduler 放大的是记账单位，worker 把属于自己的 token **紧凑**重排；物理 page 大小不变，cache 里没有空洞
8. ❌ "decode 时要把切开的 KV gather 回来再算 attention" → KV cache 全程钉在本地。通信只搬 Q、LSE、out
9. ❌ "奇偶 token 各自做因果小三角再拼回去" → decode 只拆因果矩阵的**最后一行**（同一个 Q 点被切开的 K）；Q 不按序列切

**实现坑（社区都踩过）**
| 坑 | 修复 |
|---|---|
| LSE 底数不匹配：FA 系是自然底，FlashInfer 是 log2 底 | `is_lse_base_on_e` 参数控制 kernel 走 `exp/log` 还是 `exp2/log2`。`flashinfer.py:285` 传 `False`。**错了会静默出错**。`5b4cb69523 (#47079)` |
| A2A 里 fp32 lse bitcast 成 bf16 崩 | `f05603fa28 (#47801)` cast to fp32 |
| full CUDA graph 下 A2A decode 非法访存 | `9fd737badc (#45487)` |
| DCP 下 build 期主机侧决策（如 `max_dcp_context_kv_len`）会改变 metadata 形状 | `flash_attn.py:463` 直接禁掉 `supports_draft_decode_metadata_update` |
| 某些 kernel 要求 head 数对齐（如 FlashMLA） | `q_pad_num_heads` + `reserve_query_head_storage()` |
| `reduce_scatter(dim=1)` 要求 H 能被 dcp 整除 | `dcp_alltoall.py:436` 显式检查 |
| KV connector / offloading 的 global↔block 换算在 DCP 下要重算 | `#46394`, `#41549`, `#45371`, `#95528527ea` 一堆修复 |

---

## 10. 关键代码索引

```
vllm/distributed/parallel_state.py:1852        DCP group 构建（TP 内 transpose 分组）
vllm/distributed/parallel_state.py:1397        get_dcp_group()

vllm/v1/core/single_type_kv_cache_manager.py:59    block_size *= dcp（logical / virtual block）
vllm/v1/worker/block_table.py:413                  slot mapping 交错归属 kernel
vllm/v1/attention/backends/utils.py:964            get_dcp_local_seq_lens()

vllm/v1/attention/ops/common.py:9                  mask_dcp_empty_shards_()
vllm/v1/attention/ops/common.py:37                 _correct_attn_cp_out_kernel（LSE 校正）
vllm/v1/attention/ops/common.py:225                cp_lse_ag_out_rs ★ 默认 combine
vllm/v1/attention/ops/common.py:266                cp_lse_ag_out_ar（PCP 联用）
vllm/v1/attention/ops/dcp_alltoall.py:36           _lse_weighted_combine（参考实现，最易读）
vllm/v1/attention/ops/dcp_alltoall.py:399          dcp_a2a_lse_reduce ★ A2A combine
vllm/v1/attention/ops/dcp_utils.py:592             MLADCPManager（选实现 + 持有 workspace）
vllm/v1/attention/ops/dcp_utils.py:635             _init_combine（三选一的分发逻辑）

vllm/model_executor/layers/attention/mla_attention.py:932   ★ MLA decode DCP 主流程
vllm/model_executor/layers/attention/mla_attention.py:2106  chunked prefill KV gather workspace
vllm/model_executor/layers/attention/mla_attention.py:2749  实际 kv_gather 调用
vllm/v1/attention/backends/flash_attn.py:1215               ★ GQA decode DCP 主流程

vllm/model_executor/layers/linear.py:598           DCPGroupColumnParallelLinear（qrep）
vllm/model_executor/models/deepseek_v2.py:1015     qrep 启用条件
vllm/envs.py:196                                   VLLM_DCP_Q_REPLICATE
```

---

## 11. TODO / 待实测

- [ ] 跑 `tests/v1/attention/test_dcp*` 验证数值路径（需要多卡）
- [ ] AG+RS vs A2A 在 dcp=2/4/8 下的实测延迟对比（理论上 A2A 赢在 collective 次数）
- [ ] `VLLM_DCP_Q_REPLICATE` 的实际收益：省一次 AllGather vs 多算 dcp 倍 q_proj GEMM，交叉点在哪
- [ ] `cp_kv_cache_interleave_size` 取值对 kernel 效率的影响（I 大 → 本地访存更连续，但负载倾斜变大）
- [ ] DCP 与 PD 分离叠加：D 节点开 DCP 时，KV 从 P 传过来要怎么按交错规则散布
