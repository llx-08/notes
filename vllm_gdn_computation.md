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
