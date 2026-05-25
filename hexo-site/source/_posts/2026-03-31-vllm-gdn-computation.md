---
title: GDN (GatedDeltaNet) 在 vLLM 中的计算流程
date: 2026-03-31
tags: [vLLM]
---

# GDN (GatedDeltaNet) 在 vLLM 中的计算流程

## 1. 整体架构概览

GDN（Gated Delta Net）是 Qwen3-Next 模型中的一种**线性注意力**机制，与传统的 full attention 交替使用。模型的每一层由 `config.layer_types` 决定是 `"linear_attention"`（GDN）还是 `"full_attention"`（标准 Attention）。

核心代码位于：
- **GDN 模型层**：`vllm/model_executor/models/qwen3_next.py` → `Qwen3NextGatedDeltaNet`
- **GDN Attention Metadata**：`vllm/v1/attention/backends/gdn_attn.py` → `GDNAttentionMetadataBuilder`
- **递归核心 Kernel**：`vllm/model_executor/layers/fla/ops/fused_recurrent.py`
- **分块核心 Kernel**：`vllm/model_executor/layers/fla/ops/chunk.py`

**关键点：QSA（Query-Side Aggregation）只在 full_attention 层中使用，不影响 GDN 层的计算。**

```python
# vllm/model_executor/models/qwen3_next.py : Qwen3NextDecoderLayer.__init__
if self.layer_type == "linear_attention":
    topk_indices_buffer = None  # only use qsa in full-attn
    self.linear_attn = Qwen3NextGatedDeltaNet(...)
elif self.layer_type == "full_attention":
    self.self_attn = Qwen3NextAttention(..., topk_indices_buffer=topk_indices_buffer)
```

---

## 2. GDN 层的三阶段计算流程

GDN 的 `forward` 分为三个阶段。

### 阶段 1：输入投影 (Input Projection)

从 `hidden_states` 通过两个投影矩阵得到 6 个张量：

```python
# vllm/model_executor/models/qwen3_next.py : Qwen3NextGatedDeltaNet.forward
projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)  # → q, k, v, z
projected_states_ba, _ = self.in_proj_ba(hidden_states)        # → b, a
```

有两种 split 方式：
- **Fused Triton kernel**（`fused_qkvzba_split_reshape_cat`）：在满足条件时使用融合 kernel
- **PyTorch 原生**（`fix_query_key_value_ordering`）：fallback 路径

输出的 6 个张量的作用：

| 张量 | 形状 | 用途 |
|------|------|------|
| q | `[tokens, num_k_heads/tp, head_k_dim]` | 查询向量 |
| k | `[tokens, num_k_heads/tp, head_k_dim]` | 键向量 |
| v | `[tokens, num_v_heads/tp, head_v_dim]` | 值向量 |
| z | `[tokens, num_v_heads/tp, head_v_dim]` | 输出门控 |
| b | `[tokens, num_v_heads/tp]` | 计算 beta（更新门控） |
| a | `[tokens, num_v_heads/tp]` | 计算 g（衰减因子） |

最终 q, k, v 被拼接为 `mixed_qkv = cat(q, k, v)`。

### 阶段 2：核心注意力计算 (Core Attention)

核心计算通过 `_forward_core` 完成，分为三个子步骤。

#### 2.1 因果卷积 (Causal Conv1d)

对 `mixed_qkv` 进行 1D 因果卷积，更新 conv_state 缓存。根据是否有 spec decode，分为 spec 部分和 non-spec 部分分别处理：

```python
# vllm/model_executor/models/qwen3_next.py : _forward_core

# Spec decode 部分：单步更新
if spec_sequence_masks is not None:
    mixed_qkv_spec = causal_conv1d_update(
        mixed_qkv_spec, conv_state, conv_weights, ...,
        conv_state_indices=spec_state_indices_tensor[:, 0],
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=spec_query_start_loc,
    )

# Non-spec Prefill 部分：整序列卷积
if attn_metadata.num_prefills > 0:
    mixed_qkv_non_spec = causal_conv1d_fn(
        mixed_qkv_non_spec_T, conv_weights, ...,
        conv_states=conv_state,
        has_initial_state=has_initial_state,
        cache_indices=non_spec_state_indices_tensor,
        query_start_loc=non_spec_query_start_loc,
    ).transpose(0, 1)

# Non-spec Decode 部分：单步更新
elif attn_metadata.num_decodes > 0:
    mixed_qkv_non_spec = causal_conv1d_update(
        mixed_qkv_non_spec, conv_state, conv_weights, ...,
        conv_state_indices=non_spec_state_indices_tensor,
    )
```

#### 2.2 门控计算 (Gating)

通过 fused triton kernel 计算衰减因子 g 和更新门控 beta：

```python
# vllm/model_executor/models/qwen3_next.py
g, beta = fused_gdn_gating(self.A_log, a.contiguous(), b, self.dt_bias)
```

数学公式：

```
g = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
```

其中 `softplus(x) = (1/β) * log(1 + exp(β*x))`，当 `β*x > threshold` 时退化为 `x`。

Triton kernel 实现：

```python
# fused_gdn_gating_kernel
x = a.float() + dt_bias.float()
softplus_x = where(beta * x <= threshold, (1/beta) * log(1 + exp(beta * x)), x)
g = -exp(A_log.float()) * softplus_x
beta_output = sigmoid(b.float())
```

#### 2.3 递归注意力 (Recurrent Attention)

这是 GDN 的核心数学计算，根据 prefill/decode 使用不同的算法。

**Decode 路径**使用 `fused_recurrent_gated_delta_rule`（逐步递推，复杂度 O(1)/step）：

```python
# Triton kernel: fused_recurrent_gated_delta_rule_fwd_kernel
# 文件: vllm/model_executor/layers/fla/ops/fused_recurrent.py

for i_t in range(0, T):
    b_q = load(p_q)    # query
    b_k = load(p_k)    # key
    b_v = load(p_v)    # value

    # L2 normalization
    b_q = b_q / sqrt(sum(b_q * b_q) + 1e-6)
    b_k = b_k / sqrt(sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # State decay
    b_h *= exp(b_g)                       # h = h * exp(g)

    # Delta rule
    b_v -= sum(b_h * b_k[:, None], 0)    # v' = v - h^T @ k
    b_v *= b_beta                         # v' = beta * v'

    # State update
    b_h += b_k[:, None] * b_v[None, :]   # h = h + k ⊗ v'

    # Output
    b_o = sum(b_h * b_q[:, None], 0)     # o = h^T @ q
```

**Prefill 路径**使用 `chunk_gated_delta_rule`（分块并行，更高效地处理长序列）：

```python
# vllm/model_executor/models/qwen3_next.py : _forward_core
if attn_metadata.num_prefills > 0:
    initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
    initial_state[~has_initial_state, ...] = 0
    core_attn_out_non_spec, last_recurrent_state = chunk_gated_delta_rule(
        q=query_non_spec, k=key_non_spec, v=value_non_spec,
        g=g_non_spec, beta=beta_non_spec,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=non_spec_query_start_loc,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    # 将最终状态写回 cache
    ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
```

**数学公式总结（每个 time step t）：**

$$\hat{q}_t = \frac{q_t}{\|q_t\|_2} \cdot \text{scale}, \quad \hat{k}_t = \frac{k_t}{\|k_t\|_2}$$

$$h_t = e^{g_t} \cdot h_{t-1} + \hat{k}_t \otimes \left[\sigma(b_t) \cdot (v_t - h_{t-1}^T \hat{k}_t)\right]$$

$$o_t = h_t^T \hat{q}_t$$

其中：
- $h_t \in \mathbb{R}^{d_k \times d_v}$ 为递归状态矩阵
- $e^{g_t}$ 为指数衰减因子，控制历史信息的遗忘
- $\sigma(b_t)$ 为 sigmoid 门控，控制新信息的写入强度
- $v_t - h_{t-1}^T \hat{k}_t$ 为"delta rule"，只写入与当前状态差异的部分

### 阶段 3：输出投影 (Output Projection)

```python
# vllm/model_executor/models/qwen3_next.py : Qwen3NextGatedDeltaNet.forward
core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
z = z.reshape(-1, z.shape[-1])
core_attn_out = self.norm(core_attn_out, z)     # RMSNormGated: silu(z) * norm(out)
core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
output[:num_tokens], _ = self.out_proj(core_attn_out)  # 线性投影回 hidden_size
```

输出经过 RMSNormGated（使用 z 作为 gate，silu 激活）后通过 `out_proj` 映射回 hidden_size。

---

## 3. Prefill vs Decode 路径对比

| 阶段 | Prefill | Decode |
|------|---------|--------|
| Conv1d | `causal_conv1d_fn`（整序列卷积） | `causal_conv1d_update`（单步更新） |
| 递归注意力 | `chunk_gated_delta_rule`（分块并行） | `fused_recurrent_gated_delta_rule`（逐步递推） |
| State 初始化 | 从 ssm_state 加载，若无历史则置零 | 直接使用 ssm_state 中的已有状态 |
| State 写回 | 计算完成后整体写回 ssm_state | 原地更新 ssm_state（inplace） |
| 复杂度 | O(n) 总计，分块并行加速 | O(d_k × d_v) per head per step |

---

## 4. GDN Attention Metadata 构建

`GDNAttentionMetadataBuilder`（位于 `vllm/v1/attention/backends/gdn_attn.py`）负责为每次 forward 构建 metadata。

### 核心字段

```python
@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None          # prefill 时标记是否有历史状态
    spec_query_start_loc: torch.Tensor | None       # spec decode 的 query 位置
    non_spec_query_start_loc: torch.Tensor | None   # non-spec 的 query 位置
    spec_state_indices_tensor: torch.Tensor | None  # spec decode 的 state 索引
    non_spec_state_indices_tensor: torch.Tensor | None  # non-spec 的 state 索引
    spec_sequence_masks: torch.Tensor | None        # 标记哪些序列是 spec decode
    spec_token_indx: torch.Tensor | None            # spec token 的全局索引
    non_spec_token_indx: torch.Tensor | None        # non-spec token 的全局索引
    num_accepted_tokens: torch.Tensor | None        # 每个序列接受的 token 数
    retrieve_parent_token: torch.Tensor | None      # tree attention 的父节点索引
```

### 构建逻辑（`build` 方法）

1. **判断是否有 spec decode**：通过 `num_decode_draft_tokens_cpu` 判断
2. **无 spec decode 时**：直接按 prefill/decode 切分，state 索引取 `block_table[:, 0]`
3. **有 spec decode 时**：
   - 将 batch 分为 spec 部分（前 `num_spec_decodes` 个）和 non-spec 部分
   - spec 部分的 state 索引取 `block_table[:, :num_spec+1]`（多步 state）
   - 通过 `spec_token_indx` / `non_spec_token_indx` 索引来分离和合并 token
4. **CUDAGraph 支持**：对 decode-only 场景进行 padding 以适配 CUDAGraph 捕获

---

## 5. QSA（Query-Side Aggregation）对 Full Attention 层的影响

QSA **只影响 full_attention 层**，不影响 GDN 层。当 config 中 `index_topk > 0` 时启用。

### 5.1 无 QSA 的 Full Attention 流程

```
hidden_states → QKV Proj → Q/K Norm → RoPE → Standard Flash Attention（全 KV）→ [Gate ×] O Proj
```

### 5.2 有 QSA 的 Full Attention 流程

有 QSA 时，在标准 attention 之前增加了一个 **Indexer** 模块用于稀疏化：

```
hidden_states → QKV Proj → Q/K Norm → RoPE
                                         ↓
                              ┌── Indexer（稀疏索引计算）──┐
                              │                             │
                              │  hidden → QKW Proj          │
                              │  Q/K LayerNorm              │
                              │  RoPE                       │
                              │  Write K to indexer cache   │
                              │  Score = Q @ K_cache * W    │
                              │  TopK indices               │
                              └─────────────────────────────┘
                                         ↓
                              Sparse Flash Attention（仅 TopK 个 token 的 KV）
                                         ↓
                              [Gate ×] O Proj → output
```

### 5.3 Indexer 组件详解

Indexer 类（`vllm/model_executor/models/qwen3_next.py`）核心组件：

```python
class Indexer(nn.Module):
    def __init__(self, ...):
        self.index_topk = config.index_topk           # TopK 数量
        self.index_n_heads = config.index_n_heads     # indexer query heads 数量
        self.index_kv_heads = config.index_kv_heads   # indexer kv heads 数量
        self.index_head_dim = config.index_head_dim   # indexer head 维度

        # QKW 联合投影（replicated，不做 TP）
        self.index_qkw_proj = ReplicatedLinear(hidden_size, q_dim + k_dim + w_dim)

        # Q/K LayerNorm（稳定不同层的 score scale）
        self.index_q_layernorm = RMSNorm(index_head_dim)
        self.index_k_layernorm = RMSNorm(index_head_dim)

        # 独立的 K cache
        self.k_cache = Qwen3NextIndexerCache(...)

        self.score_scale = index_head_dim ** 0.5
```

Indexer 的 forward 计算过程：

1. **QKW 投影**：`qkw = index_qkw_proj(hidden_states)` → 拆分为 q, k, w
2. **LayerNorm**：对 q, k 分别做 RMSNorm
3. **RoPE**：施加旋转位置编码（使用 `partial_rotary_factor=0.5`）
4. **写入 K Cache**：将当前 k 写入 indexer 独立的 block KV cache
5. **计算相关性分数**：
   - bf16 模式：`score = (Q @ K_cache) * W / score_scale`
   - fp8 模式：先对 Q 做 per-token-group FP8 量化，再计算
6. **TopK 选择**：对每个 token 选出 TopK 个最相关的历史 token 索引
7. **输出**：`topk_indices_buffer` 供后续 sparse attention 使用

### 5.4 Sparse Attention 的 Prefill 和 Decode

**Prefill 时**（`flash_attn_sparse_prefill`）：
- 从 K cache 中按 block 收集完整的 K
- 计算 `logits = Q @ K_cache * W`
- 选 TopK 得到稀疏索引
- 用稀疏索引从 KV cache 中提取对应的 K、V，做 sparse attention
- 支持 Context Parallel (CP) 加速长序列

**Decode 时**：
- 使用 paged MQA logits 计算分数
- TopK 选择后，提取稀疏 KV
- 支持 pai-fa3 或 triton 两种 sparse attention backend

---

## 6. 完整流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Qwen3-Next Decoder Layer                             │
│                                                                             │
│  input: hidden_states                                                       │
│         ↓                                                                   │
│  ┌─────────────┐                                                            │
│  │ InputLayerNorm│                                                           │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│   ┌─────┴──────────────────────────────────┐                                │
│   │         layer_type?                      │                               │
│   │                                          │                               │
│   ▼                                          ▼                               │
│ "linear_attention" (GDN)          "full_attention"                          │
│   │                                          │                               │
│   │ ┌───────────────────────┐      ┌────────┴──────────────────────────┐    │
│   │ │ 1. in_proj_qkvz(h)   │      │  qkv_proj(h) → q, k, v [, gate] │    │
│   │ │    in_proj_ba(h)      │      │  q_norm, k_norm                  │    │
│   │ │    → q,k,v,z,b,a     │      │  rotary_emb(pos, q, k)          │    │
│   │ └───────┬───────────────┘      └────────┬──────────────────────────┘    │
│   │         │                                │                               │
│   │ ┌───────┴───────────────┐      ┌────────┴─────────┐                    │
│   │ │ 2. causal_conv1d      │      │  QSA enabled?     │                    │
│   │ │    (prefill: conv1d_fn│      │                   │                    │
│   │ │     decode: conv1d_upd│      │   YES      NO     │                    │
│   │ └───────┬───────────────┘      │    │        │     │                    │
│   │         │                      │    ▼        │     │                    │
│   │ ┌───────┴───────────────┐      │ Indexer:    │     │                    │
│   │ │ 3. fused_gdn_gating   │      │ QKW proj    │     │                    │
│   │ │  g = -exp(A) *        │      │ LayerNorm   │     │                    │
│   │ │     softplus(a+bias)  │      │ RoPE        │     │                    │
│   │ │  β = sigmoid(b)       │      │ Write K$    │     │                    │
│   │ └───────┬───────────────┘      │ Score calc  │     │                    │
│   │         │                      │ TopK select │     │                    │
│   │ ┌───────┴───────────────┐      │    │        │     │                    │
│   │ │ 4. Recurrent Attn     │      └────┤────────┤─────┘                    │
│   │ │ (prefill: chunk_gdr)  │           ▼        ▼                          │
│   │ │ (decode: fused_rec)   │      ┌────────────────────┐                   │
│   │ │                       │      │  Flash Attention    │                   │
│   │ │ For each step t:      │      │  (sparse if QSA)   │                   │
│   │ │  q̂=q/‖q‖₂ * scale   │      │  (dense  if no QSA)│                   │
│   │ │  k̂=k/‖k‖₂           │      └────────┬───────────┘                   │
│   │ │  h *= exp(g)          │               │                               │
│   │ │  v' = β(v - hᵀk̂)    │      ┌────────┴───────────┐                   │
│   │ │  h += k̂ ⊗ v'         │      │ [gate * ] o_proj   │                   │
│   │ │  o = hᵀq̂             │      └────────┬───────────┘                   │
│   │ └───────┬───────────────┘               │                               │
│   │ ┌───────┴───────────────┐               │                               │
│   │ │ 5. norm(out, z) +     │               │                               │
│   │ │    out_proj           │               │                               │
│   │ └───────┬───────────────┘               │                               │
│   │         │                               │                               │
│   └─────────┴───────────────────────────────┘                               │
│             │                                                               │
│         attention_output                                                    │
│             │                                                               │
│    + residual + [layer_scale]                                               │
│             │                                                               │
│    PostAttentionLayerNorm                                                   │
│             │                                                               │
│    MLP (Dense or MoE)                                                       │
│             │                                                               │
│    + residual + [layer_scale]                                               │
│             │                                                               │
│         output                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. KV Cache 结构对比

### GDN 层的 Cache

GDN 使用 Mamba 风格的 state cache，每个序列维护两个状态：

| 状态 | 形状 | 说明 |
|------|------|------|
| conv_state | `[num_slots, conv_dim, conv_kernel_size]` | 因果卷积的滑动窗口状态 |
| ssm_state | `[num_slots, num_v_heads, head_k_dim, head_v_dim]` | 递归注意力的状态矩阵 h |

其中 `conv_dim = key_dim * 2 + value_dim`。

Spec decode 时，每个序列需要 `num_spec + 1` 个 slot 来存储多步状态。

### Full Attention 层的 Cache

标准的 block KV cache：`[2, num_blocks, block_size, num_kv_heads, head_dim]`

QSA 时额外有 Indexer K cache：`[num_blocks, block_size, index_kv_heads, index_head_dim]`

---

## 8. 关键对比总结

| 维度 | GDN (linear_attention) | Full Attention (无 QSA) | Full Attention (有 QSA) |
|------|----------------------|----------------------|----------------------|
| **KV Cache** | conv_state + ssm_state | block KV cache | block KV cache + indexer K cache |
| **Prefill 算法** | chunk_gated_delta_rule | Flash Attention | Indexer + Sparse Flash Attention |
| **Decode 算法** | fused_recurrent | Flash Attention | Indexer + Sparse Flash Attention |
| **Decode 复杂度** | O(d_k × d_v) per head | O(seq_len × d) | O(TopK × d) |
| **位置编码** | 无 RoPE | RoPE | RoPE + Indexer RoPE |
| **QSA 影响** | 无 | - | Indexer 选 TopK，稀疏 attention |
| **状态数学** | h = exp(g)·h + k⊗[β(v-hᵀk)] | softmax(QKᵀ/√d)V | softmax(QK_topk^T/√d)V_topk |

---

## 9. 相关环境变量

| 环境变量 | 作用 |
|---------|------|
| `VLLM_GDN_USE_BLADNN` | 是否使用 bladnn 加速 GDN 递归计算 |
| `VLLM_QSA_USE_FP8_INDEXER` | Indexer 是否使用 FP8 量化 |
| `VLLM_QSA_PREFILL_USE_TL_INDEXER` | Prefill 时 Indexer 是否使用 tilelang kernel |
| `VLLM_QSA_DECODE_USE_TL_INDEXER` | Decode 时 Indexer 是否使用 tilelang kernel |
| `VLLM_QSA_PREFILL_ATTN_BACKEND` | QSA prefill 的 attention backend（triton / pai-fa3） |
| `VLLM_QSA_DECODE_ATTN_BACKEND` | QSA decode 的 attention backend（pai-fa3 等） |
| `VLLM_QSA_PREFILL_USE_CP` | QSA prefill 是否使用 Context Parallel |
| `VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD` | 启用 CP 的 token 数阈值 |
| `VLLM_DSA_USE_DENSE_PREFILL_THRESHOLD` | 使用 dense prefill 的阈值 |

---

## 10. 源码文件索引

| 文件 | 主要内容 |
|------|---------|
| `vllm/model_executor/models/qwen3_next.py` | GDN 层 (`Qwen3NextGatedDeltaNet`)、Attention 层 (`Qwen3NextAttention`)、Indexer、融合 gating kernel |
| `vllm/v1/attention/backends/gdn_attn.py` | GDN Attention Metadata 构建器 |
| `vllm/model_executor/layers/fla/ops/fused_recurrent.py` | 递归 GDN Triton kernel（decode 路径） |
| `vllm/model_executor/layers/fla/ops/chunk.py` | 分块 GDN kernel（prefill 路径） |
| `vllm/v1/attention/backends/flash_attn.py` | Flash Attention + QSA 稀疏 attention 实现 |
| `vllm/v1/attention/backends/flash_attn_qsautils.py` | QSA 辅助工具函数 |
| `vllm/model_executor/models/qsa_indexer_utils.py` | QSA Indexer 的 MQA logits 计算工具 |
| `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` | 因果卷积实现 |
| `vllm/v1/worker/gpu_model_runner.py` | GDN metadata 与 model runner 的集成 |

---

## 11. 门控机制对照图

下面这张图把 Full Attention 中的 `gate`，以及 GDN 中的 `z / g / beta` 放在同一视角下对照：

```text
Full Attention 路径
==================

hidden_states
    |
    v
qkv_proj
    |
    +--> q -----------------> q_norm -> RoPE --+
    |                                          |
    +--> k -----------------> k_norm -> RoPE --+--> Flash / Sparse Attention --> attn_output
    |                                          |
    +--> v ------------------------------------+
    |
    +--> gate -> sigmoid ------------------------------+
                                                       |
attn_output -------------------------------------------* (逐元素相乘)
                                                       |
                                                       v
                                                    o_proj
                                                       |
                                                       v
                                                     output

作用：
- `gate` 是 attention 输出门控
- 控制 attention 结果有多少被送入 `o_proj`


GDN 路径
========

hidden_states
    |
    v
in_proj_qkvz / in_proj_ba
    |
    +--> q, k, v --+--> causal_conv1d --> recurrent core --> core_attn_out
    |              |
    |              +--> a -> softplus + A_log -> g
    |              |
    |              +--> b -> sigmoid ---------> beta
    |
    +--> z ----------------------------------------------+
                                                         |
core_attn_out --> RMSNormGated(norm(out), z) ------------+
                                                         |
                                                         v
                                                      out_proj
                                                         |
                                                         v
                                                       output

作用：
- `g` 是状态衰减门控，控制历史状态遗忘速度
- `beta` 是写入门控，控制新信息写入状态的强度
- `z` 是输出门控，在 `RMSNormGated` 中对核心输出做门控
```

---

## 12. 问答补充

### 问题

`../study_doc/vllm_gdn_computation.md` 这里说明了qwen3.5中GDN和full_attention的计算流程，最后的[gate * ] o_proj 这里是什么作用？

### 回答

这里的 `[gate * ] o_proj`，作用可以直接理解成：

**先对 attention 的输出做一次逐元素门控，再送进 `o_proj`。**

在 `Qwen3NextAttention.forward()` 里，full attention 的实际流程是：

```python
qkv, _ = self.qkv_proj(hidden_states)

if self.attn_output_gate:
    q_gate, k, v = qkv.split(
        [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
    )
    orig_shape = q_gate.shape[:-1]
    q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    q = q.reshape(*orig_shape, -1)
    gate = gate.reshape(*orig_shape, -1)
else:
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

# ... q/k norm, rope, attention ...

attn_output = self.attn(q, k, v)

if self.attn_output_gate:
    gate = torch.sigmoid(gate)
    attn_output = attn_output * gate

output[:], _ = self.o_proj(attn_output)
```

所以它不是“`o_proj` 的一部分”，而是：

\[
\text{output} = o\_proj(\text{attn\_output} \odot \sigma(\text{gate}))
\]

也就是：
- `gate` 是从输入 `hidden_states` 里一起投影出来的
- 经过 `sigmoid` 后变成 0 到 1 之间的门值
- 对 attention 输出按维度逐元素缩放
- 最后再过 `o_proj` 混合回 `hidden_size`

它的主要作用有三点：

1. **控制 attention 输出强度**
   - 哪些通道/哪些 head 的输出该保留更多，哪些该压小，不让所有 attention 结果都等权流向后面。

2. **给 full attention 增加一层动态选择**
   - attention 负责“从哪里取信息”，这个 gate 负责“取回来之后放多少进去”。

3. **提高表达能力和稳定性**
   - 有点像 GLU/门控残差 的思路，让模型可以按 token 自适应地调节 attention 分支输出。

再补一个你这个文档上下文里很容易混淆的点：

- **full attention 里的 `gate`**：是在 `self.attn(...)` 之后、`o_proj` 之前做 `sigmoid(gate) * attn_output`
- **GDN 里的门控**：不是这个东西。GDN 里有 `g`、`beta` 控制递归状态更新，最后还有一个 `z` 进入 `RMSNormGated`：

```python
core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
z = z.reshape(-1, z.shape[-1])
core_attn_out = self.norm(core_attn_out, z)     # RMSNormGated: silu(z) * norm(out)
core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
output[:num_tokens], _ = self.out_proj(core_attn_out)
```

也就是说：
- **full attention**：`gate` 是“attention 输出门控”
- **GDN**：`z` 是“norm 后输出门控”，`g/beta` 是“状态更新门控”

---

## 13. 问答补充

### 问题

“gate 本身不直接决定“这个 token 是否重要”，而是决定attention 已经取回来的这些特征里，哪些维度该更多流向后面的 o_proj。”
这里有什么数学含义？对最终的结果，比如采样到的token概率有什么影响

### 回答

有，而且这个数学含义其实很清楚：

`gate` 不是在做“token 级别的重要性判断”，而是在做**attention 输出特征空间里的逐维重加权**。

在这层里，full attention 分支可以写成：

\[
y = W_o \bigl(o \odot \sigma(g)\bigr)
\]

其中：
- `o` 是 attention 返回的输出 `attn_output`
- `g` 是从输入 `hidden_states` 里一起投影出来的 gate
- `\sigma(g)` 是 sigmoid 后的门值
- `W_o` 就是 `o_proj`

源码对应的是：

```python
qkv, _ = self.qkv_proj(hidden_states)
# ...
attn_output = self.attn(q, k, v)

if self.attn_output_gate:
    gate = torch.sigmoid(gate)
    attn_output = attn_output * gate

output[:], _ = self.o_proj(attn_output)
```

数学上这等价于：

\[
y = W_o \operatorname{Diag}(\sigma(g))\, o
\]

这句话的含义是：

- `o_proj` 本来是一个**固定线性映射**
- `gate` 在它前面插入了一个**输入相关的对角矩阵**
- 所以同样的 `o_proj`，面对不同 token，会先看到不同的“被缩放过的 attention 特征”

这就是我前面说“不是在决定 token 是否重要，而是在决定 attention 取回来的哪些维度更该往后流”的数学版本。

---

### 直观一点

如果没有 gate：

\[
y = W_o o
\]

如果有 gate：

\[
y = W_o (o \odot s), \quad s=\sigma(g)
\]

假设某个 token 的 attention 输出：

\[
o = [2,\ -1,\ 0.5]
\]

如果门值是：

\[
s = [0.1,\ 0.8,\ 0.95]
\]

那么进入 `o_proj` 前先变成：

\[
o' = o \odot s = [0.2,\ -0.8,\ 0.475]
\]

含义就是：
- 第 1 维被强烈抑制
- 第 2 维保留大部分
- 第 3 维几乎原样通过

所以 gate 作用的不是“选哪个历史 token”，而是“attention 聚合完以后，这些输出通道分别该放大到什么程度进入下一步线性混合”。

---

### 为什么这会影响最终 token 概率

因为它会改变每层输出的 hidden state，而最终 logits 就是 hidden state 乘上 `lm_head` 得到的。

在这个模型里，attention 输出先进入 decoder layer，之后还会经过残差/归一化/MLP，最后再经过最终 norm 和 `lm_head`：

```python
self_attention_output = torch.empty_like(hidden_states)
# ...
self.self_attn(
    hidden_states=hidden_states,
    output=self_attention_output,
    positions=positions,
)
hidden_states = self_attention_output

# Fully Connected
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)

return hidden_states, residual
```

```python
hidden_states, _ = self.norm(hidden_states, residual)
return hidden_states
```

```python
def compute_logits(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    return self.logits_processor(self.lm_head, hidden_states)
```

```python
def _get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    # Get the logits for the next tokens.
    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
    # ...
    return logits
```

所以链路是：

\[
\text{gate}
\rightarrow \text{attn output}
\rightarrow \text{layer hidden state}
\rightarrow \text{final hidden state}
\rightarrow \text{logits}
\rightarrow \text{softmax}
\rightarrow \text{token probability}
\]

最终概率是：

\[
p(t \mid x) = \operatorname{softmax}(\text{logits})_t
\]

只要 `gate` 改变了 hidden state，就会改变 logits，进而改变 softmax 后每个 token 的概率。

---

### 它对概率的影响，最准确怎么理解

不是“gate 小了，就某个 token 概率一定小”。

更准确的说法是：

**gate 改变了当前层 attention 分支贡献给表示向量的方式，从而改变了最终 hidden state 在词表方向上的投影。**

也就是说：
- 某些 attention 特征如果和“下一个 token 应该是 `the`”这个方向强相关，gate 把它压低，`the` 的 logit 可能下降
- 某些特征如果更支持另一个 token，gate 保留或增强它们对应的通道，那个 token 的 logit 可能上升
- softmax 是相对竞争，所以一个 token 概率升高，通常意味着别的一些会下降

---

### 线性化视角

如果只看小扰动，可以写成：

\[
\Delta \text{logits} \approx J \cdot \Delta y
\]

其中：
- `y = W_o(o \odot \sigma(g))`
- `J` 是从这一层输出到最终 logits 的整体雅可比

这说明：
- gate 改变的是一个**中间表示**
- 这个中间表示经过后面所有层的非线性变换后，传导到 logits
- 所以影响通常是**间接的、分布式的、非局部的**

也正因为这样，gate 更像是“调整表征几何结构”，而不是直接给某个词加减分的显式规则。

---

### 一个很重要的现实点

它虽然会影响概率，但**不是单独决定概率**，因为还有：
- 残差路径
- 后续层的 attention / GDN / MLP
- 最终 norm
- `lm_head`

所以：
- gate 把某些 attention 特征压低，不等于这些信息彻底没了
- 残差里原始表示还在
- 后续层仍然可能重新恢复或放大相关信息

因此它更像是一个**调节器**，不是一个绝对开关。

---

### 一句话总结

数学上，`gate` 相当于在 `o_proj` 前加了一个**输入相关的对角缩放矩阵** `Diag(sigmoid(g))`；  
它会改变 attention 分支输出的特征组成，进而改变最终 hidden state、logits 和 softmax 概率分布，但这种影响是**连续的、分布式的、通过整条网络链路传播的**，不是“直接决定哪个 token 重要”的硬规则。

如果你想，我下一条可以继续回答这个问题：

**“为什么 gate 放在 `o_proj` 前，而不是放在 attention score 或 softmax 上？”**

这个和“特征选择” vs “token 选择”的区别正好对应。

---

## 14. 问答补充

### 问题

“为什么 gate 放在 o_proj 前，而不是放在 attention score 或 softmax 上？” 这个问题很有意思，请帮我回答

### 回答

核心区别是：

- **放在 attention score / softmax 上**，是在决定“看哪里、看多少历史 token”
- **放在 `o_proj` 前**，是在决定“已经看回来的信息里，哪些特征该继续往后传”

这两者控制的对象不一样。

## 数学上看区别

标准 attention 可以写成：

\[
A = \operatorname{softmax}(QK^\top / \sqrt{d})
\]
\[
o = AV
\]
\[
y = W_o o
\]

现在 Qwen 这个 gate 做的是：

\[
y = W_o (o \odot \sigma(g))
\]

也就是 gate 作用在 **`o` 这个特征向量** 上。

如果把 gate 放到 score 上，那就会变成类似：

\[
A' = \operatorname{softmax}(f(Q,K,g))
\]

它影响的是 **attention 分布 `A`**，也就是每个 query 对各个历史位置的权重。

所以：

- `score/softmax` gate: 控制 **token-to-token 路由**
- `o_proj` 前的 gate: 控制 **feature/channel 选择**

## 为什么很多模型更愿意把这类 gate 放在 `o_proj` 前

### 1. 它更像“特征调制”，比改 attention 分布更稳

attention score 很敏感，因为 softmax 会放大差异。

如果你在 score 上加门控，哪怕是很小的扰动，也可能让注意力分布发生很大变化：

- 原本 0.4 / 0.35 / 0.25
- 稍微改一下 score
- softmax 后可能变成 0.7 / 0.2 / 0.1

这会直接改变“从哪些 token 取信息”，行为更激进。

而在 `o_proj` 前乘 gate，只是在 `AV` 之后做连续缩放：

\[
o' = o \odot \sigma(g)
\]

这是对已经聚合好的表示做后处理，通常更平滑、更容易训练。

### 2. 它不会破坏 attention 的归一化结构

softmax 有明确的概率意义：

\[
\sum_j A_{ij} = 1
\]

如果你把 gate 混进 score/softmax，本质上是在改这个概率分配机制。  
模型不仅要学“内容匹配”，还要学“门控后的匹配”，耦合更重。

放在 `o_proj` 前则不碰 `A` 的定义：

- attention 还是正常算“该看哪些 token”
- gate 只负责“取回来以后怎么用”

这是一种更清晰的职责分离。

### 3. 它能做“通道级选择”，而 score gate 更偏“位置级选择”

attention score 的维度本质上是：

- query 位置
- key 位置
- head

它擅长控制的是：

- 看第几个历史 token
- 哪个 head 更关注哪个位置

但 `o_proj` 前的 gate 作用在 `attn_output` 的 hidden/channel 维上，更适合控制：

- 哪类语义特征保留
- 哪些通道被抑制
- 同一批取回的信息里，哪些成分更该进入后续层

所以它回答的问题不是“该看谁”，而是“看完之后，哪些特征值得继续传下去”。

### 4. 计算上更简单，兼容各种 attention backend

你这个代码里 full attention 既可能是 dense flash attention，也可能是 QSA sparse attention。  
如果 gate 放在 score 上，就必须介入 attention kernel 内部逻辑。

但现在放在 `o_proj` 前：

```python
qkv, _ = self.qkv_proj(hidden_states)
# ...
attn_output = self.attn(q, k, v)

if self.attn_output_gate:
    gate = torch.sigmoid(gate)
    attn_output = attn_output * gate

output[:], _ = self.o_proj(attn_output)
```

这意味着：
- 不管前面是 dense attention 还是 sparse attention
- 只要 `self.attn(q,k,v)` 给出输出
- 后面统一乘 gate 即可

工程上解耦得很好。

## 从功能上说，两种 gate 学到的东西也不同

### 如果 gate 放在 score/softmax 上

模型更容易学到：
- 哪些上下文 token 该被关注
- 哪些位置该被忽略
- 稀疏路由、检索式选择

这更接近“检索机制”或“选择记忆源”。

### 如果 gate 放在 `o_proj` 前

模型更容易学到：
- 当前 token 从上下文取回的信息里，哪些语义分量更有用
- 哪些 head 输出应该弱化
- 哪些 feature 该进入残差流和后续 MLP

这更接近“表示控制”或“信息过滤”。

## 一个直观类比

可以把 attention 想成“查资料”：

- `score/softmax` gate：决定去翻哪几页资料
- `o_proj` 前 gate：资料已经翻回来了，决定摘抄其中哪些内容进入最终总结

前者是“选来源”，后者是“选内容”。

## 对最终 token 概率的影响差别

### 放在 score 上
会更直接改变：
- 哪些历史 token 被聚合
- attention pattern
- 模型的检索行为

所以通常对行为模式影响更结构化、更剧烈。

### 放在 `o_proj` 前
更直接改变：
- attention 输出特征的组成
- 当前层 hidden state 的几何方向
- 进而影响最终 logits

它对概率分布的影响往往更“连续”和“表征级”。

## 一句话总结

`gate` 放在 `o_proj` 前，而不是放在 attention score 或 softmax 上，是因为它想做的不是“改注意力分配”，而是“对 attention 已经聚合出的表示做稳定、细粒度、通道级的动态筛选”。这样既保留了标准 attention 的路由能力，又额外给模型一个更平滑的特征控制旋钮。

如果你愿意，我还可以继续把这两种方案写成一个并排对照表：

- `score gate`
- `value gate`
- `output gate (o_proj 前)`

这样你会更容易看出 Qwen 这里为什么选第三种。
