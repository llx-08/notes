---
title: "QSA Q-block 与 MiniMax MSA：从一个 8×16 的例子理解两种稀疏 Attention"
date: 2026-08-14
tags: []
---
# QSA Q-block 与 MiniMax MSA：从一个 8×16 的例子理解两种稀疏 Attention

> 本文回答一个具体问题：vLLM 分支 `qsa-q-block-pooling-clean` 的做法，是否类似 MiniMax Sparse Attention（MSA）？
>
> 简短答案：**硬件优化目标相同，但“让哪些 Query 共用哪些 KV”的办法不同。**
> QSA Q-block 先强制连续 Query 共用一套 KV token；MSA 保留每个 Query 的独立选择，再把选择同一 KV block 的 Query 动态聚到一起。

## 0. 先记住一句话

稀疏 Attention 已经把“每个 Query 看全部历史 KV”变成“每个 Query 只看 Top-K KV”。但如果 GPU 仍按 Query 逐个执行，就会反复读取 K/V，Tensor Core 的矩阵也很瘦。

- **QSA Q-block：共享路由。** 连续 `Bq` 个 Query 共用同一组 token-level Top-K，因此天然可以一起算。
- **MiniMax MSA：反转路由。** 每个 Query 独立选择 KV blocks，再把 `Query → KV block` 反转成 `KV block → Query 列表`，把命中同一 block 的 Query 一起算。

![三种执行方式总览](/imgs/qsa-msa-01-overview.svg)

---

## 1. 统一的小例子

后文所有图都使用同一个例子：

| 符号 | 含义 | 示例值 |
|---|---|---:|
| `M` | Query token 数 | 8，即 `q0 … q7` |
| `N` | KV token 数 | 16，即 `k0 … k15` |
| `Bq` | QSA 的 Query block 大小 | 4 |
| `Bk` | MSA 的 KV block 大小 | 4；论文实际采用 128 |
| `S` | QSA 每个 Q-block 选择的 KV token 数 | 3 |
| `Kblk` | MSA 每个 Query/GQA group 选择的 KV block 数 | 2；论文实际采用 16 |

将 16 个 KV token 分成 4 个 block：

```text
K0 = [k0,  k1,  k2,  k3 ]
K1 = [k4,  k5,  k6,  k7 ]
K2 = [k8,  k9,  k10, k11]
K3 = [k12, k13, k14, k15]
```

为了先理解数据流，图中暂时省略 batch、head、causal mask 和量化 scale；第 6 节再把它们放回来。

---

## 2. Develop 上原始 token-level QSA 做什么

原始 QSA 已经是稀疏 Attention，不是 Dense Attention。它有两个阶段：

1. **Indexer：**用一个轻量的 Index Query/Key 计算分数矩阵 `score[M,N]`，然后每个 Query 独立取 Top-S KV token。
2. **Main Attention：**真实的 GQA Query 只与选中的 S 个 K/V 做标准 softmax attention。

示例：

```text
q0 → [k1,  k6,  k11]
q1 → [k2,  k7,  k11]
q2 → [k1,  k8,  k12]
q3 → [k3,  k8,  k13]
```

虽然不同 Query 可能大量命中相同 KV，传统 Q-outer kernel 仍然按 Query 启动 program：

```text
处理 q0：读取 k1/k6/k11 和对应 V
处理 q1：读取 k2/k7/k11 和对应 V   # k11 又读一次
处理 q2：读取 k1/k8/k12 和对应 V   # k1 又读一次
处理 q3：读取 k3/k8/k13 和对应 V   # k8 又读一次
```

瓶颈不一定是数学 FLOPs，而是：

- 随机 gather 的 K/V 被重复搬运；
- 一个 Query × 一个 GQA group 的 MMA `M` 维太小；
- program 数量多；
- indexer 要物化 `M×N` 的 FP32 logits，并对 M 行分别做 Top-K。

---

## 3. QSA Q-block：先合并 Query 的路由

### 3.1 Indexer：沿 Query 方向 max-pooling

设置 `Bq=4` 后，QSA 把连续 4 个 Query 组成一组：

```text
Q0 = [q0, q1, q2, q3]
Q1 = [q4, q5, q6, q7]
```

Indexer 仍计算每个 Query 对每个 KV token 的分数，但不再把完整 `8×16` logits 写回。PAI-DeepGEMM kernel 在 epilogue 中直接做：

```text
pooled[Q0, k] = max(score[q0,k], score[q1,k], score[q2,k], score[q3,k])
pooled[Q1, k] = max(score[q4,k], score[q5,k], score[q6,k], score[q7,k])
```

因此输出由 `[8,16]` 变成 `[2,16]`。再对每个 Q-block 取 Top-3 KV token，例如：

```text
Q0 → [k1, k8,  k11]
Q1 → [k2, k10, k14]
```

最后复制回 token 粒度：

```text
q0/q1/q2/q3 → [k1, k8,  k11]
q4/q5/q6/q7 → [k2, k10, k14]
```

![QSA 沿 Query 方向 pooling](/imgs/qsa-msa-02-pooling-axes.svg)

### 3.2 Main Attention：一个 program 同时处理 Bq 个 Query

因为一个 Q-block 中的 indices 完全相同，Triton kernel 只读取第一行 indices：

```text
idx_row_ptr = indices[q_block_start]
```

然后把 `Bq × G` 个 Query heads 放入同一个矩阵。其中 `G=Hq/Hkv` 是每个 KV head 服务的 Query head 数。

```text
Q tile: [Bq × G, D]
K tile: [D, BN]
score : [Bq × G, BN]
```

每一轮 Top-K tile 中，K/V 只 gather 一次，供 Bq 个 Query 复用。kernel grid 第一维也从约 `M` 下降为 `ceil(M/Bq)`。

![QSA Q-block 完整 pipeline](/imgs/qsa-msa-03-qsa-qblock-pipeline.svg)

### 3.3 Causal mask 怎么办

同一 Q-block 内，较晚 Query 能看到更多历史 token。Indexer 用 block 最后一个 Query 的 causal 上界做 Top-K；Attention kernel 再为每个 Query 单独屏蔽未来位置：

```text
q0 可见上界 = offset + 0
q1 可见上界 = offset + 1
q2 可见上界 = offset + 2
q3 可见上界 = offset + 3
```

这保证不会看未来，但可能出现一个副作用：某个 KV token 因为对 q3 很重要而进入共享 Top-K，却对 q0 尚不可见。q0 会把它 mask 掉，于是 q0 的实际有效 KV 数少于 S。

### 3.4 QSA Q-block 到底省了什么

假设 `Bq=4`：

| 项目 | 原始 token QSA | QSA Q-block | 理想变化 |
|---|---:|---:|---:|
| Index GEMM 数学计算 | `M×N` | 仍约 `M×N` | 基本不变 |
| Index logits 写回 | `M×N` | `M/Bq×N` | 约 1/4 |
| Top-K 行数 | `M` | `M/Bq` | 约 1/4 |
| Attention program 数 | `M×Hkv` | `M/Bq×Hkv` | 约 1/4 |
| indices 读取 | 每 Q 一行 | 每 Q-block 一行 | 约 1/4 |
| K/V 加载 | 每 Q 重复 gather | block 内复用 | 理想约 1/4 |
| Main Attention FLOPs | `M×S` | 仍约 `M×S` | 基本不变 |

所以它不是把数学运算减少 4 倍，而是把 **中间 logits、Top-K、kernel launch 和 K/V IO** 大幅压缩。

---

## 4. MiniMax MSA：不共享路由，而是反转路由

### 4.1 Indexer：沿 KV block 方向 max-pooling

MSA 给每个 GQA group 配一个 Index Query head，所有 group 共用一个 Index Key head。对每个 Query `qi` 和 GQA group `r`：

1. 先计算 token-level index score；
2. 在每个 KV block 内取最大值；
3. 对 KV blocks 取 Top-K；
4. 强制包含当前 local block。

在小例子中：

```text
block_score[q, K0] = max(score[q,k0],  ..., score[q,k3])
block_score[q, K1] = max(score[q,k4],  ..., score[q,k7])
block_score[q, K2] = max(score[q,k8],  ..., score[q,k11])
block_score[q, K3] = max(score[q,k12], ..., score[q,k15])
```

输出仍保留每个 Query 的独立路由，例如：

```text
q0 → [K0, K2]
q1 → [K0, K3]
q2 → [K1, K2]
q3 → [K0, K2]
```

注意两种 pooling 的“轴”正好相反：

- QSA：多行 Query 合成一行，K 仍是 token；
- MSA：每行 Query 保留，多列 K token 合成一个 KV block。

这就是上面 `02_pooling_axes.svg` 的核心。

### 4.2 为什么不能直接像 QSA 那样一起算

MSA 中 q0、q1、q2、q3 的 KV block 集合不同，不能简单放进一个 Q tile 并共享全部 K/V。它先生成 Q2K 路由：

```text
q0 → K0, K2
q1 → K0, K3
q2 → K1, K2
q3 → K0, K2
```

再转置为 K2Q：

```text
K0 → q0, q1, q3
K1 → q2
K2 → q0, q2, q3
K3 → q1
```

这样 kernel 处理 K0 时只加载一次 K0/V0，然后连续计算 q0、q1、q3。处理 K2 时同理。

![路由矩阵与转置](/imgs/qsa-msa-04-routing-matrix.svg)

### 4.3 KV-outer kernel 为什么需要两阶段合并

一个 Query 选择了多个 KV blocks。例如 q0 选择 K0 和 K2：

```text
CTA(K0) 只能得到 q0 在 K0 上的 partial output 和 LSE
CTA(K2) 只能得到 q0 在 K2 上的 partial output 和 LSE
```

它们不能各自完成 q0 对全部选中 token 的 softmax。因此 MSA 分两阶段：

1. **Sparse Attention kernel：**每个 `(KV block, 一组 Query)` 产生局部归一化结果 `O_partial` 和 `LSE_partial`；
2. **Combine kernel：**使用 log-sum-exp 权重把一个 Query 的多个 partial 合并成最终输出。

热门 KV block 可能被大量 Query 选择。MSA scheduler 会把它的 Query 列表切成多个 chunk，分发给多个 CTA，并提前给每个 partial 分配输出 slot，从而避免 atomic write 冲突。

![MSA KV-outer 完整 pipeline](/imgs/qsa-msa-05-msa-kv-outer-pipeline.svg)

### 4.4 MSA 如何填满 Tensor Core

若一个 GQA group 有 `G=16` 个 Query heads，单个 Query 只提供 16 行，MMA 的 M 维太小。MSA 会从同一 KV block 的 Q 列表中取 `ceil(128/G)=8` 个 Query positions：

```text
8 个 Query positions × 16 个 Query heads = 128 行
```

它们共享同一个 KV head 和 KV block，于是形成接近 `128×128` 的 score MMA。这与 QSA 的 `Bq×G` 拼接目标相同，只是 MSA 的 Query 可以来自任意位置，而 QSA 必须是连续位置。

---

## 5. 一张表看懂本质区别

| 维度 | QSA Q-block | MiniMax MSA |
|---|---|---|
| 核心目标 | 一次 K/V 读取服务更多 Q | 一次 KV block 读取服务更多 Q |
| Query 是否独立选路 | 否；连续 Bq 个 Q 共用 | 是；每个 Q 独立选择 |
| Head 路由粒度 | 当前实现是一套 indices 供各 KV heads 使用 | 每个 GQA group 独立选择，组内 heads 共享 |
| 稀疏 KV 单位 | KV token | 128-token KV block |
| Pooling 方向 | 沿 Q 维 max-pool | 沿 K block 维 max-pool |
| Kernel 组织 | 静态 Q-block outer | 动态 KV-block outer |
| Query 聚合来源 | 连续、预先固定 | 任意位置、由命中关系决定 |
| 路由元数据 | `[M,S]`，每 Bq 行相同 | Q2K `[Hkv,M,Kblk]` + K2Q CSR + schedule |
| Softmax | 一个 program 内完成 | partial O/LSE + combine |
| 实现代价 | 低 | 高 |
| 质量风险 | 共享 Top-K 改变每个 Q 的路由 | 不强迫 Query 共享路由，但 KV block 粒度更粗 |
| 训练方式 | 对既有 QSA 的推理优化 | 原生可训练 Index Branch；KL、warmup、local block |

![Kernel 与内存路径对比](/imgs/qsa-msa-06-kernel-memory-compare.svg)

---

## 6. 把 batch、GQA、FP8 和 causal 放回来

### 6.1 QSA 分支中的真实张量

Indexer Q-block 路径：

```text
q_fp8              [M, Hidx, Didx]
k_fp8              [N, Didx]
weights            [M, Hidx]
pooled_logits      [ceil(M/Bq), N]        FP32
topk_pooled        [ceil(M/Bq), S]        INT32 token ids
topk_indices       [M, S]                 每 Bq 行相同
```

Main Attention Q-block kernel：

```text
query              [M, Hq, D]
k/value            [N_aligned, Hkv, D]
indices            [M, S]
GROUP_SIZE G       Hq / Hkv
BLOCK_SIZE_M       next_pow2(Bq × G)，至少 16
grid               [ceil(max_q_len/Bq), batch × Hkv]
```

若 core KV cache 是 FP8：

- Q/K/V 保持 E4M3；
- QK 与 PV 使用 FP8 Tensor Core、FP32 accumulator；
- Q/K/V scale 合并到 kernel 内；
- K/V 相对 BF16 的 HBM 流量约减半；
- Max-offset 将 softmax probability 放大 `2^8` 后再 cast FP8，降低下溢，最终归一化会抵消这个常数。

### 6.2 QSA Q-block 的启用条件

该优化不是默认开启。分支中：

```bash
VLLM_QSA_Q_BLOCK_SIZE=4             # 默认 0，即关闭
VLLM_QSA_PREFILL_ATTN_BACKEND=triton
VLLM_QSA_INDEXER_BACKEND=dpsk-fp8
```

同时需要 PAI-DeepGEMM 提供 `fp8_mqa_logits_q_block_pooling`。以下情况会回退到 token-level 路径：

- 一个 prefill batch 含多个 sequence；
- 使用 shared-topk；
- Q-block pooling kernel 不可用；
- `VLLM_QSA_Q_BLOCK_SIZE=0`。

第一 chunk 且 `query.shape == key.shape` 时，Main Attention 仍走 raw sparse kernel；Q-block attention kernel 主要作用于后续 chunk、prefix/cache 命中等路径。Indexer 的 Q-pooling 仍可能生效。

### 6.3 MSA 的真实路由元数据

MSA 公共 kernel 接口使用：

```text
q2k_indices        [Hkv, total_q, Kblk]
k2q_row_ptr        [Hkv, total_k_blocks + 1]
k2q_q_indices      [Hkv, total_q × Kblk]
schedule           work_count / qsplit_indices / split_counts / ...
```

论文采用：

```text
KV block size Bk = 128
Top-K blocks      = 16
最多关注 token   = 16 × 128 = 2048
```

这个 2048-token budget 与常见 QSA token Top-2048 数值相似，但物理布局不同：MSA 会把完整的 128-token block 连续读入，而 QSA 是 2048 个可能完全不连续的 token gather。

---

## 7. 为什么 QSA 的 +40% 和 MSA 的 14.2× 不冲突

两个数字的基线不同：

- QSA 分支的约 40%：相对于**已经是 token-level sparse QSA** 的 develop，进一步优化执行效率；Top-K budget 没有减少。
- MSA 论文的 14.2×：在 H800、1M context 下，相对于**Dense GQA prefill**；主要收益先来自 sparse-vs-dense，再叠加 KV-outer kernel。

QSA Q-block 的收益来源可以写成：

```text
已有 token sparse QSA
  + fused logits/Q-pooling
  + Top-K 行数减少 Bq 倍
  + Q-block 内 K/V 复用
  + program 数减少
  + 可选 FP8 QK/PV
  ≈ 在特定长上下文场景获得约 1.4×
```

MSA 的收益路径则是：

```text
Dense GQA
  → 每 Q/GQA group 只选 16 个 128-token blocks
  → Main Attention 从 O(N²) 降为 O(N × 2048)
  → KV-outer 将同一 block 的 Query 聚合
  → 反向 CSR + load balancing + 两阶段 combine
```

---

## 8. 精度取舍：最需要关注的差别

### 8.1 QSA Q-block 改变了路由语义

原始 QSA：

```text
indices[q] = TopK(score[q, :])
```

QSA Q-block：

```text
indices[Qblock] = TopK(max over q in Qblock(score[q, :]))
```

这两者不数学等价。一个 Query 的重要 KV 可能被同 block 其他 Query 的高分 KV 挤出固定的 S 个位置。因此需要对 `Bq=2/4/8` 分别验证：

- PPL；
- Needle/RULER/长上下文检索；
- Agent、代码库、多轮对话业务集；
- 首 chunk 与后续 chunk 一致性；
- 不同 chunk size、prefix cache 和 batch 组合。

历史测试验证的是“Q-block kernel 对给定共享 indices 的计算正确”和“fused pooling 符合其 PyTorch reference”，并不证明它与逐 Query Top-K 的模型输出等价。

### 8.2 MSA 用训练吸收 block sparsity

MSA 的 Index Branch 不是纯推理 heuristic：

- 用 KL loss 对齐 Main Branch 的 group-average attention distribution；
- warmup 阶段先运行 full attention，避免随机路由控制主分支；
- 强制加入 local block；
- Index Branch 的梯度与 backbone/Main Branch 隔离。

因此 MSA 可以在训练中适应“每次选择完整 KV block”的约束。QSA Q-block 则是在既有 token-level indexer 输出之上做推理期 Q pooling，工程简单，但更需要补充质量回归。

---

## 9. 最终理解模型

可以把两者看成对同一个 GPU 问题的两个答案：

> 问题：每个 Query 只选少量 KV 后，路由很散，怎样让一个大的 Tensor Core tile 复用同一份 K/V？

**QSA Q-block 的答案：**

> 我提前规定连续 Bq 个 Query 必须选同一批 KV token。这样不用整理路由，直接一起算。

**MiniMax MSA 的答案：**

> 我不限制每个 Query 选什么；选择结束后，把命中同一个 KV block 的所有 Query 找出来，再一起算。

因此：

- QSA 是以更强的路由共享约束，换取更简单、低开销的 kernel；
- MSA 是以更复杂的 metadata、调度和归并，换取 Query 路由独立性；
- 两者都不是单纯“少算几个 token”，而是 attention 算法与 GPU 数据流的共同设计。

---

## 10. 对应代码与资料

### vLLM `qsa-q-block-pooling-clean`

- `vllm/model_executor/layers/qwen_sparse_attn_indexer.py`
  - `_prefill_mqa_logits_topk_q_block`
  - fused Q logits + Q max-pooling、block-level Top-K、复制 indices
- `vllm/v1/attention/backends/flash_attn_qsautils.py`
  - `sparse_gqa_fwd_kernel_triton_ck_qblock`
  - `sparse_gqa_fwd_kernel_triton_ck_qblock_fp8`
- `vllm/v1/attention/backends/flash_attn_qsa.py`
  - Q-block/token-level、BF16/FP8 prefill dispatch
- `vllm/v1/attention/backends/qsa_config.py`
  - indexer 与 attention 共享的 Q-block gating
- `vllm/v1/attention/backends/mla/indexer.py`
  - Q-block-aware logits budget 和 chunk splitting

关键提交：

- `401781ab15`：Q-block indexer + attention kernel
- `0e6296c807`：复用 logits buffer、优化 Q-block 内存预算
- `03c26dc49a`：删除每层 D2H `max_q_len` 同步
- `1753d04756`：FP8 Q-block attention
- `42429e4217`：FA3 Max-offset
- `a9ae261b64`：回退 K-block pooling 与 decode Q-block，保留干净的 prefill Q-block 路径

### MiniMax MSA

- 论文：[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)
- 官方实现：[MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)
- CuTe-DSL kernel 文档：[MSA sparse kernel README](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/README.md)
