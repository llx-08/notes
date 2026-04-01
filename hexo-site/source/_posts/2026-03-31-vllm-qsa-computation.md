---
title: QSA (Query-Side Aggregation) 在 vLLM 中的计算流程
date: 2026-03-31
tags: [vLLM]
---

# QSA (Query-Side Aggregation) 在 vLLM 中的计算流程

## 1. QSA 概述

QSA 是 Qwen3-Next 模型在 **full_attention 层**中使用的稀疏注意力机制。其核心思想是：**不对所有历史 token 做 attention，而是先用轻量级的 Indexer 为每个 query token 选出 TopK 个最相关的 KV token，再仅对这 TopK 个 token 做标准 attention。**

启用条件：模型 config 中 `index_topk > 0`。

核心代码文件：
- **Indexer 模块**：`vllm/model_executor/models/qwen3_next.py` → `Indexer` 类
- **Sparse Attention 前端**：`vllm/v1/attention/backends/flash_attn.py` → `FlashAttentionImpl.forward`
- **Sparse Attention 实现**：`vllm/v1/attention/backends/flash_attn_qsautils.py`
- **Indexer Logits 计算**：`vllm/model_executor/models/qsa_indexer_utils.py`

---

## 2. 端到端流程总览

```
┌────────────────────────────────────────────────────────────────────┐
│              Qwen3NextAttention.forward (full_attention 层)        │
│                                                                    │
│  hidden_states                                                     │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐     ┌──────────────┐                                 │
│  │ QKV Proj  │     │ Indexer      │                                 │
│  │ + Q/K Norm│     │ (QSA 专用)   │                                 │
│  │ + RoPE    │     └──────┬───────┘                                 │
│  └────┬─────┘            │                                         │
│       │                  │                                         │
│       │            topk_indices_buffer                              │
│       │              [tokens, topk]                                 │
│       │                  │                                         │
│       ▼                  ▼                                         │
│  ┌─────────────────────────────────┐                               │
│  │   Sparse Flash Attention        │                               │
│  │   (只用 TopK 个 token 的 KV)     │                               │
│  └─────────────┬───────────────────┘                               │
│                │                                                   │
│          [Gate ×] O Proj                                           │
│                │                                                   │
│            output                                                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 第一阶段：Indexer — 选出 TopK 个最相关的 KV Token

### 3.1 Indexer 结构

```python
# vllm/model_executor/models/qwen3_next.py
class Indexer(nn.Module):
    def __init__(self, ...):
        # 独立于主 attention 的小型 QKW 投影
        self.index_qkw_proj = ReplicatedLinear(
            hidden_size, q_dim + k_dim + w_dim, bias=False
        )
        # Q/K 独立的 LayerNorm
        self.index_q_layernorm = RMSNorm(index_head_dim)
        self.index_k_layernorm = RMSNorm(index_head_dim)
        # 独立的 K Cache（不与主 attention 共享）
        self.k_cache = Qwen3NextIndexerCache(...)
        # 评分缩放因子
        self.score_scale = index_head_dim ** 0.5
```

关键参数（来自 model config）：

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `index_topk` | 每个 query token 选多少个 KV | 2048 |
| `index_n_heads` | Indexer 的 Q heads 数量 | 16 |
| `index_kv_heads` | Indexer 的 KV heads 数量 | 1 |
| `index_head_dim` | Indexer 的 head 维度 | 128 |

### 3.2 Indexer Forward 计算步骤

```python
# vllm/model_executor/models/qwen3_next.py → Indexer.forward
def forward(self, hidden_states, positions, rotary_emb):
```

**Step 1: QKW 联合投影**

```python
qkw_mixed = self.index_qkw_proj(hidden_states)
q, k, w = torch.split(qkw_mixed, [q_dim, k_dim, w_dim], dim=-1)
```

从 `hidden_states` 投影出三个张量：
- `q`：indexer query，shape `[tokens, index_q_dim]`
- `k`：indexer key，shape `[tokens, index_k_dim]`
- `w`：per-head weight，shape `[tokens, index_n_heads]`

**Step 2: Q/K LayerNorm**

```python
q = self.index_q_layernorm(q.view(-1, index_n_heads, index_head_dim))
k = self.index_k_layernorm(k.view(-1, index_kv_heads, index_head_dim))
```

对 Q 和 K 分别做 RMSNorm。这一步很关键，因为不同层的 hidden_states norm 差异很大，LayerNorm 能保持 score scale 的稳定性。

**Step 3: 旋转位置编码 (RoPE)**

```python
q = q.view(-1, index_n_heads, index_head_dim)
q, k = rotary_emb(positions, q, k)
```

使用与主 attention 相同的 RoPE（`partial_rotary_factor=0.5`），为 indexer 的 Q/K 注入位置信息。

**Step 4: 写入 K Cache + 计算 Score + TopK 选择**

这一步通过 `torch.ops.vllm.qwen_sparse_attn_indexer` 自定义算子完成（bf16 模式）：

```python
# vllm/model_executor/models/qwen3_next.py → qwen_sparse_attn_indexer
def qwen_sparse_attn_indexer(hidden_states, k_cache_prefix, kv_cache, q, k,
                              weights, topk_tokens, head_dim, max_model_len,
                              total_seq_lens, topk_indices_buffer, score_scale):
```

或通过 `torch.ops.vllm.sparse_attn_indexer`（fp8 模式）完成。

---

### 3.3 Score 计算的数学公式

Indexer 的评分函数与标准 attention 不同，使用的是 **ReLU + 加权求和**，而非 softmax：

$$\text{score}(q_i, k_j) = \frac{1}{\sqrt{d}} \sum_{h=1}^{H_q} w_{i,h} \cdot \text{ReLU}(q_{i,h}^T \cdot k_j)$$

其中：
- $q_{i,h}$：第 $i$ 个 query token 在第 $h$ 个 head 的向量
- $k_j$：第 $j$ 个 key token 的向量（所有 Q heads 共享）
- $w_{i,h}$：第 $i$ 个 token 在第 $h$ 个 head 的可学习权重
- $d$：`index_head_dim`（缩放因子）

对应的 PyTorch 参考实现：

```python
# vllm/model_executor/models/qsa_indexer_utils.py → bf16_mqa_logits_torch
scores = torch.einsum("qhd,khd->qkh", q, k)   # [S_q, S_k, H]
scores = torch.relu(scores)                      # ReLU 激活
logits = (scores * weights.unsqueeze(1)).sum(-1)  # 加权求和 → [S_q, S_k]
logits = logits / score_scale                     # 缩放
```

**为什么用 ReLU 而不是 softmax？**
- 这是 "Lightning Indexer" 的设计思路
- ReLU 使得评分稀疏，大部分为零
- 计算代价远小于 softmax
- W 权重使模型能学习到每个 head 的重要性

### 3.4 Indexer 的 Prefill 和 Decode 路径

**Prefill 时**：
1. 将当前 chunk 的 K 写入 indexer K cache
2. 从 K cache 中按 block table 收集完整的 K 序列
3. 计算 `Q @ K_cache * W` 得到 logits（tilelang 或 PyTorch 实现）
4. 调用 `top_k_per_row_prefill` CUDA kernel 选 TopK

**Decode 时**：
1. 将新的 K 写入 indexer K cache 对应 slot
2. 使用 paged MQA logits 从 K cache 中计算每个 decode query 与所有历史 K 的分数
3. 调用 `top_k_per_row_decode` CUDA kernel 选 TopK

---

## 4. 第二阶段：Sparse Flash Attention — 用 TopK KV 计算 Attention

Indexer 完成后，`topk_indices_buffer` 中存储了每个 query token 应该关注的 TopK 个 KV token 的位置索引。接下来在 `FlashAttentionImpl.forward` 中，prefill 和 decode 分别走不同的 sparse attention 路径。

### 4.1 Sparse Prefill Attention

入口：`flash_attn_sparse_prefill` 方法。

有三种 backend 可选（由 `VLLM_QSA_PREFILL_ATTN_BACKEND` 控制）：

#### Backend 1: triton（默认）

**无 KV Cache 版本（首个 chunk 且 Q 长度 = K 长度）**：直接用当前 chunk 的 Q、K、V 做 sparse attention。

```python
# vllm/v1/attention/backends/flash_attn_qsautils.py → sparse_gqa_fwd_kernel_triton
# 核心循环伪代码：
for each query_token m:
    row_topk = min(topk, m + 1)  # 因果性限制
    for start_n in range(0, row_topk_aligned, BLOCK_SIZE_N):
        # 加载稀疏索引
        rel_idx = load(indices[m, start_n : start_n + BLOCK_SIZE_N])
        # 稀疏加载 K, V
        k = K[rel_idx]   # 只加载 TopK 对应的 K
        v = V[rel_idx]   # 只加载 TopK 对应的 V
        # QK 乘法
        s = dot(q, k.T)
        s = where(is_valid, s, -inf)
        # Online Softmax
        m_next = max(m_i, max(s))
        alpha = exp2(m_i - m_next)
        p = exp2(s - m_next)
        acc = acc * alpha + dot(p, v)
        l_i = l_i * alpha + sum(p)
    output[m] = acc / l_i
```

关键优化：
- 利用因果性，`row_topk = min(topk, token_position + 1)`，短序列位置的循环范围大幅缩小
- 每个 CUDA block 处理一个 kv_head 的所有 q_heads（GQA 分组）

**有 KV Cache 版本（chunked prefill 的后续 chunk 或 Context Parallel）**：

1. 先收集 TopK 对应的 KV（sparse gather）：
   - 从 `topk_indices_buffer` 得到每个 Q 关注的 TopK token 位置
   - AllReduce(MAX) 得到全局 token mask
   - 从 KV cache 中收集这些 token 的 K、V
   - 重新映射 indices 到收集后的紧凑数组中
2. 用重映射后的 indices 做 sparse attention

#### Backend 2: pai-fa3

使用定制版 Flash Attention 3，直接在 paged KV cache 上做 sparse attention：

```python
# 将 topk_indices 转换为 global block 索引
global_topk_indices = triton_convert_req_index_to_global_index(
    req_id_per_token, block_table, topk_indices, BLOCK_SIZE, NUM_TOPK_TOKENS
)
# 调用定制 FA3
output = flash_attn_with_kvcache_qsa(
    q=query, k_cache=k_cache, v_cache=v_cache,
    cache_seqlens=seqused_k,
    page_table=global_topk_indices,  # 用 topk indices 替代 block table
    cu_seqlens_q=cu_seqlens_q,
    softmax_scale=scale,
    causal=True, pack_gqa=True,
)
```

### 4.2 Sparse Decode Attention

Decode 阶段有两种路径：

#### 路径 1: pai-fa3 (推荐)

直接在 paged KV cache 上用 TopK indices 做 sparse attention：

```python
# 将 topk indices 转为全局 block 索引
decode_global_topk_indices = triton_convert_req_index_to_global_index(
    req_id_per_token[:num_decode_tokens],
    block_table,
    topk_indices_buffer[:num_decode_tokens],
)
# FA3 sparse decode
output[:num_decode_tokens] = flash_attn_with_kvcache_qsa(
    q=decode_query,
    k_cache=kv_cache_physical[:, 0],
    v_cache=kv_cache_physical[:, 1],
    cache_seqlens=decode_seqused_k,
    page_table=decode_global_topk_indices,
    cu_seqlens_q=decode_cu_seqlens_q,
    softmax_scale=scale,
    causal=True, pack_gqa=True,
)
```

#### 路径 2: KV 收集 + FA2/FA3

1. **收集 TopK KV**：从 paged KV cache 中提取每个 decode query 的 TopK 个 KV
   - tilelang 版本（`sparse_kv_extraction`）：每个 CUDA block 处理一个 batch × kv_head
   - PyTorch fallback（`collect_topkcache_torch_ref`）

```python
# tilelang sparse KV extraction 伪代码
for each query i, each topk j:
    token_pos = TopkIndices[i, j]
    block_idx = token_pos // block_size
    offset = token_pos % block_size
    physical_block = BlockTables[i, block_idx]
    NewK[i, j] = KV_CACHE[physical_block, 0, offset]  # K
    NewV[i, j] = KV_CACHE[physical_block, 1, offset]  # V
```

2. **标准 FA2/FA3 attention**：对提取出的 TopK KV 做标准 attention

```python
output[:num_decode_tokens] = flash_attn_varlen_func(
    q=decode_query,
    k=decode_key_cache,  # [batch, topk, kv_heads, head_dim]
    v=decode_val_cache,  # [batch, topk, kv_heads, head_dim]
    cu_seqlens_q=decode_cu_seqlens_q,
    cache_seqlens=decode_seqused_k.clamp(max=topk),
    softmax_scale=scale,
    causal=True,
)
```

---

## 5. Sparse Attention 的数学本质

标准 attention：
$$o_i = \text{softmax}\left(\frac{q_i K^T}{\sqrt{d}}\right) V$$

QSA sparse attention：
$$o_i = \text{softmax}\left(\frac{q_i K_{\mathcal{S}_i}^T}{\sqrt{d}}\right) V_{\mathcal{S}_i}$$

其中 $\mathcal{S}_i = \text{TopK}_j\left(\sum_h w_{i,h} \cdot \text{ReLU}(q_{i,h}^{\text{idx}} \cdot k_j^{\text{idx}})\right)$ 是由 Indexer 选出的 TopK 个 KV token 的索引集合。

**关键区别**：
- Indexer 用自己独立的 Q/K/W 投影计算稀疏分数（ReLU + 加权求和）
- 主 attention 仍用标准 softmax，但只在 TopK 子集上计算
- Indexer 和主 attention 的 Q/K 是**完全独立的投影**

---

## 6. Context Parallel (CP) 支持

当 prefill 序列很长时，QSA 支持 Context Parallel 来加速：

```
TP0: q[0:chunk_size], 所有 KV heads → local attention → output[0:chunk_size]
TP1: q[chunk_size:2*chunk_size], 所有 KV heads → local attention → output[chunk_size:2*chunk_size]
...
```

流程：
1. 将 Q 按 TP rank 切分（all-to-all）
2. 每个 rank 用自己的 Q chunk + 全局 TopK indices 做 sparse attention
3. 输出通过 all-to-all 聚合回各 rank

---

## 7. 完整流程图（含 Indexer + Sparse Attention 细节）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    QSA Full Attention Layer                              │
│                                                                          │
│  hidden_states, positions                                                │
│       │                                                                  │
│  ┌────┴──────────────────────┐     ┌─────────────────────────────────┐   │
│  │  主 Attention 投影         │     │  Indexer 投影                    │   │
│  │                            │     │                                 │   │
│  │  qkv = qkv_proj(h)        │     │  qkw = index_qkw_proj(h)       │   │
│  │  q,k,v,[gate] = split     │     │  q_idx, k_idx, w = split       │   │
│  │  q = q_norm(q)             │     │  q_idx = q_layernorm(q_idx)    │   │
│  │  k = k_norm(k)             │     │  k_idx = k_layernorm(k_idx)    │   │
│  │  q,k = rope(pos, q, k)    │     │  q_idx,k_idx = rope(pos,q,k)  │   │
│  │                            │     │                                 │   │
│  │  → q: [T, Hq, D]          │     │  ┌─────────────────────────┐   │   │
│  │  → k: [T, Hkv, D]         │     │  │ Write k_idx to K Cache  │   │   │
│  │  → v: [T, Hkv, D]         │     │  │ cache[slot] = k_idx     │   │   │
│  │  (写入主 KV Cache)         │     │  └──────────┬──────────────┘   │   │
│  └────┬──────────────────────┘     │              │                  │   │
│       │                            │  ┌───────────┴──────────────┐   │   │
│       │                            │  │ 计算 Indexer Score        │   │   │
│       │                            │  │                          │   │   │
│       │                            │  │ Prefill:                 │   │   │
│       │                            │  │  收集 K_cache 到连续内存  │   │   │
│       │                            │  │  scores = Q @ K * W      │   │   │
│       │                            │  │  (ReLU + 加权求和)       │   │   │
│       │                            │  │                          │   │   │
│       │                            │  │ Decode:                  │   │   │
│       │                            │  │  Paged MQA Logits        │   │   │
│       │                            │  │  遍历 KV block pages     │   │   │
│       │                            │  │  scores = Q @ K_page * W │   │   │
│       │                            │  └───────────┬──────────────┘   │   │
│       │                            │              │                  │   │
│       │                            │  ┌───────────┴──────────────┐   │   │
│       │                            │  │ TopK 选择                 │   │   │
│       │                            │  │ topk_indices =            │   │   │
│       │                            │  │   top_k_per_row(scores)   │   │   │
│       │                            │  │ → [tokens, topk] int32    │   │   │
│       │                            │  └───────────┬──────────────┘   │   │
│       │                            └──────────────┤                  │   │
│       │                                           │                      │
│       ▼                                           ▼                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │              Sparse Flash Attention                                │  │
│  │                                                                    │  │
│  │  Prefill 路径:                                                     │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ Backend "triton":                                            │  │  │
│  │  │   首个 chunk: 直接用 Q,K,V + indices 做 sparse GQA          │  │  │
│  │  │   后续 chunk: 收集 TopK KV → remap indices → sparse GQA     │  │  │
│  │  │                                                              │  │  │
│  │  │ Backend "pai-fa3":                                           │  │  │
│  │  │   indices → global block index → FA3 paged sparse attention  │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  Decode 路径:                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ Backend "pai-fa3":                                           │  │  │
│  │  │   indices → global block index → FA3 paged sparse decode     │  │  │
│  │  │                                                              │  │  │
│  │  │ Backend "fa2"/"fa3":                                         │  │  │
│  │  │   Step 1: sparse_kv_extraction (提取 TopK KV)               │  │  │
│  │  │   Step 2: FA2/FA3 varlen_func (标准 attention on TopK KV)   │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  output = softmax(Q @ K_topk^T / √d) × V_topk                    │  │
│  └───────────────────────────┬────────────────────────────────────────┘  │
│                              │                                           │
│                     [gate ×] O Proj                                       │
│                              │                                           │
│                          output                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Indexer Score 计算的多种实现

| 场景 | 函数 | 说明 |
|------|------|------|
| Prefill bf16 (PyTorch) | `bf16_mqa_logits_torch` | `einsum("qhd,khd->qkh")` + ReLU + 加权求和 |
| Prefill bf16 (tilelang) | `mqa_attn_fp16_return_logits_interface` | GEMM + ReLU × W + reduce_sum，高效 kernel |
| Decode bf16 (PyTorch, next_n>1) | `fp16_paged_mqa_logits_torch` | Paged KV cache 遍历 |
| Decode bf16 (PyTorch, next_n=1) | `fp16_paged_mqa_logits_torch_optimize` | 优化版，去掉 causal mask |
| Decode bf16 (tilelang, next_n=1) | `tilelang_paged_mqa_logits_interface` | GEMM on paged KV blocks |
| Decode bf16 (tilelang, MTP) | `tilelang_paged_mqa_logits_mtp_interface` | 大 GEMM, next_n × heads 合并 |

### tilelang Prefill Logits Kernel 核心逻辑

```python
# mqa_attn_fp16_return_logits_kernel 伪代码
for each q_block (block_Q 个 q tokens):
    # 加载 Q: [block_Q × n_heads, head_dim] → shared memory
    load Q_shared from IndexQ[start_m * heads : (start_m + block_Q) * heads]
    # 加载 W: [block_Q, n_heads]
    load weights from Weights[start_m : start_m + block_Q]

    for each k_block in range(cu_k_s_min, cu_k_e_max, block_N):
        # 加载 K: [block_N, head_dim] → shared memory
        load K_shared from IndexK[k_block_start : k_block_start + block_N]
        
        # GEMM: K_shared @ Q_shared^T → s[block_N, block_Q × n_heads]
        s = gemm(K_shared, Q_shared, transpose_B=True)
        
        # ReLU + Weight: s *= max(s, 0) × weights
        for each (kn, qm, h):
            s[kn, qm, h] = max(s[kn, qm, h], 0) × weights[qm, h]
        
        # Reduce Sum over heads: [block_N, block_Q, n_heads] → [block_N, block_Q]
        logits = reduce_sum(s, dim=-1)
        
        # 写回
        Logits[start_m + qm, k_block_start + kn] = logits[kn, qm]
```

### tilelang Decode Logits Kernel 核心逻辑

```python
# fp16_paged_mqa_logits_tilelang 伪代码
for each batch_i, each kv_block_group:
    # 加载 Q: [heads, head_dim] → shared memory
    load Q_shared from Q[batch_i, 0]
    load weights from Weights[batch_i]
    
    for each k_block in block_group:
        if block_start < context_len:
            # 从 paged KV cache 加载物理 block
            physical_block = BlockTables[batch_i, block_idx]
            load K_shared from K_cache[physical_block]
            
            # GEMM: K_shared @ Q_shared^T → s[block_size, heads]
            s = gemm(K_shared, Q_shared, transpose_B=True)
            
            # ReLU + Weight + ReduceSum
            for each (ki, hi):
                s[ki, hi] = max(s[ki, hi], 0) × weights[hi]
            logits_res = reduce_sum(s, dim=1)  # → [block_size]
            
            # 写回
            Logits[batch_i, global_pos] = logits_res[ki] / score_scale
```

---

## 9. Sparse Attention 的多种实现

### Triton Sparse GQA（Prefill，无 paged KV cache）

```python
# sparse_gqa_fwd_kernel_triton 伪代码
# 每个 CUDA block 处理：1 个 query token × 1 个 kv_head_group
for each query_token, each kv_head_group:
    # 预加载 Q [group_size, head_dim]
    q = Q[token, kv_group * group_size : (kv_group + 1) * group_size]
    q *= scale * 1.4426950408  # log2(e) for exp2
    
    # 因果性优化：row_topk = min(topk, position + 1)
    row_topk = min(topk, token_position + 1)
    
    # Online Softmax 遍历 TopK
    for start_n in range(0, row_topk_aligned, BLOCK_SIZE_N):
        # 稀疏加载 K, V（通过 indices 间接寻址）
        rel_idx = indices[token, start_n : start_n + BLOCK_SIZE_N]
        k = K[rel_idx]  # [BLOCK_SIZE_N, head_dim]
        v = V[rel_idx]  # [BLOCK_SIZE_N, head_dim]
        
        # QK^T
        s = dot(q, k.T)  # [group_size, BLOCK_SIZE_N]
        s = where(rel_idx >= 0, s, -inf)
        
        # Online Softmax 累积
        m_next = max(m_i, max(s, dim=1))
        alpha = exp2(m_i - m_next)
        p = exp2(s - m_next[:, None])
        acc = acc * alpha[:, None] + dot(p, v)
        l_i = l_i * alpha + sum(p, dim=1)
        m_i = m_next
    
    output[token] = acc / l_i[:, None]
```

### tilelang Sparse GQA（Prefill，无 paged KV cache）

```python
# sparse_gqa_prefill_nopage 伪代码
# 每个 CUDA block 处理：1 个 q_token × 1 个 kv_head
for each q_token (bx), each kv_head (by):
    # 加载 Q: [padded_H(=16), D] (所有 query heads in this group)
    load Q_shared from Q[bos_q + s_i, H0 : H0 + real_H]
    
    # 遍历 TopK blocks
    for i_i in range(0, topk // block_I):
        # 检查每个 TopK token 的有效性 (因果 + 非-1)
        for each token_j in block_I:
            mask[j] = (indices[token, j] <= max_kv_i) & (indices[token, j] != -1)
        
        # 按 indices 加载 K, V
        for each (j, d):
            K_shared[j, d] = K[bos_k + indices[q_token, j], kv_head, d]
            V_shared[j, d] = V[bos_k + indices[q_token, j], kv_head, d]
        
        # 初始化 score（mask 掉无效位置）
        acc_s[h, j] = 0 if valid else -inf
        
        # GEMM: Q × K^T → acc_s [padded_H, block_I]
        acc_s += gemm(Q_shared, K_shared.T)
        
        # Online Softmax（exp2 版本）
        m_i = max(m_i_prev, row_max(acc_s))
        alpha = exp2((m_i_prev - m_i) * sm_scale)
        acc_s = exp2(acc_s * sm_scale - m_i * sm_scale)
        sumexp = sumexp * alpha + sum(acc_s)
        acc_o = acc_o * alpha + gemm(acc_s, V_shared)
    
    output[q_token] = acc_o / sumexp
```

---

## 10. FP8 Indexer 模式

当 `VLLM_QSA_USE_FP8_INDEXER=True` 时，Indexer 会对 Q 做 per-token-group FP8 量化：

```python
# Indexer.forward (fp8 模式)
q_fp8, q_scale = per_token_group_quant_fp8(q, quant_block_size)
# 将 scale 融入 weight
w = w.unsqueeze(-1) * q_scale / score_scale
# 调用 fp8 版本的 sparse_attn_indexer
torch.ops.vllm.sparse_attn_indexer(
    ..., q_fp8_padded, k, w_padded.to(float32), quant_block_size, scale_fmt, ...
)
```

K cache 也使用 FP8 格式存储，head_dim 扩展为 `head_dim + head_dim // quant_block_size * 4` 以存放 scale。

---

## 11. 关键环境变量

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `VLLM_QSA_USE_FP8_INDEXER` | False | Indexer 是否使用 FP8 量化 |
| `VLLM_QSA_PREFILL_USE_TL_INDEXER` | - | Prefill Indexer score 是否用 tilelang |
| `VLLM_QSA_DECODE_USE_TL_INDEXER` | - | Decode Indexer score 是否用 tilelang |
| `VLLM_QSA_PREFILL_ATTN_BACKEND` | "triton" | Prefill sparse attention backend |
| `VLLM_QSA_DECODE_ATTN_BACKEND` | - | Decode sparse attention backend |
| `VLLM_QSA_PREFILL_USE_CP` | False | Prefill 是否使用 Context Parallel |
| `VLLM_QSA_PREFILL_ONLY_INDEXER_USE_CP` | False | 仅 Indexer 部分使用 CP |
| `VLLM_QSA_PREFILL_CP_USE_SPARSE_GATHER` | - | CP 中是否使用 sparse gather |
| `VLLM_DSA_USE_DENSE_PREFILL_THRESHOLD` | - | 小于此长度用 dense prefill |
| `VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD` | - | 超过此长度启用 CP |

---

## 12. 源码文件索引

| 文件 | 主要内容 |
|------|---------|
| `vllm/model_executor/models/qwen3_next.py` | `Indexer` 类定义、`qwen_sparse_attn_indexer` 算子、`Qwen3NextAttention` 中 Indexer 的调用 |
| `vllm/model_executor/models/qsa_indexer_utils.py` | Indexer score 的各种实现：bf16 torch/tilelang、paged MQA torch/tilelang（含 MTP） |
| `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl.forward` 中 sparse prefill/decode 分支 |
| `vllm/v1/attention/backends/flash_attn_qsautils.py` | sparse GQA triton/tilelang kernel、KV extraction、indices remap、CP 支持 |
| `vllm/v1/attention/backends/mla/indexer.py` | `Qwen3NextIndexerBackend`、`DeepseekV32IndexerMetadata` |
