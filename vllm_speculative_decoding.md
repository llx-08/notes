# vLLM 投机解码（Eagle / MTP）实现与运行流程

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
