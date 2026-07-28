# Kimi K3、DSpark 与 PD 分离

> 状态：设计分析与修改计划
> 日期：2026-07-28
> 结论适用范围：本文所列代码版本，不代表后续分支已经自动具备相同能力。

## 1. 结论摘要

基于以下代码版本的静态审查：

- vLLM `origin/hmz/dspark_v3`：`3883053c0ff59fba6b0ad0c0f31ec48cbc834e5b`
- vLLM `alex/kimi-k3-mla-zeroing-20260722`：`d28bf21877006ed90cbd0adb528d25b15ac01ea7`
- blade-kvt `main`：`752697132e8b0409ad134724fec2882c9ca57380`

当前能力可以概括为：

| 组合 | 当前状态 | 说明 |
| --- | --- | --- |
| Kimi K3 + 本地推理 | 支持 | KDA + MLA hybrid cache 已接入 vLLM |
| Kimi K3 + PD 分离 | 支持基础链路 | K3 分支和 blade-kvt main 已实现专用 `cache_shape=8` |
| DFlash/DSpark + 普通 target + PD | `hmz/dspark_v3` 已有适配 | P 节点 prefill 时会保留 drafter context KV |
| Kimi K3 + DFlash/DSpark | 需要兼容的 drafter checkpoint 和集成验证 | 当前 DSpark port 只正式识别 Qwen3 DSpark checkpoint 架构 |
| Kimi K3 + DFlash/DSpark + PD | **当前不支持开箱即用** | 分支未集成，且 K3 KVT block-group/layout contract 无法表示额外 drafter KV groups |

这里的“不支持”不是单纯缺少测试，而是存在确定的代码契约冲突：

1. K3 分支在 `skip_decode_drafting` 时过早返回，P 节点没有写入 DFlash/DSpark context KV。
2. K3 的 Blade-KVT `cache_shape=8` 只接受“若干 KDA groups + 一个 replicated MLA group”。
3. DFlash/DSpark 会增加自己的 FullAttention/SlidingWindow KV groups。
4. vLLM K3 KVT 注册逻辑和 Blade-KVT K3 parser 都对 attention group 数量作了严格断言。

![K3、DFlash/DSpark 与 PD 的数据流](./imgs/kimi-k3-dspark-pd-flow.svg)

## 2. 术语和边界

### 2.1 P、D 与 PD 分离

- **P 节点（Prefill producer）**：执行 prompt 的主要 prefill，产生可复用 cache。
- **D 节点（Decode consumer）**：接收 cache，完成少量本地重算后进入 decode。
- **KVT**：本文特指 vLLM `HybridConnector` 中的 Blade-KVT backend。
- **target cache**：Kimi K3 主模型的 KDA recurrent state 和 MLA latent KV。
- **drafter cache**：DFlash/DSpark draft model 自己的 attention K/V。

target cache 与 drafter cache 服务于不同模型计算，不能因为它们都由同一个 scheduler 管理，就把它们视为相同 layout。

### 2.2 `gamma`

本文使用：

```text
gamma = num_speculative_tokens
```

对于开启 speculative decoding 的 hybrid target，现有 PD 逻辑让 P 节点只计算：

```text
remote_prefill_tokens = prompt_len - (gamma + 1)
```

D 节点加载远端 cache 后，再本地重算最后 `gamma + 1` 个 token。这样做的目的包括：

- 恢复 hybrid recurrent state 到 D 节点将要使用的位置；
- 为 speculative verification 准备连续的 lookahead 空间；
- 避免直接传输仍会被本地重算覆盖的尾部状态；
- 保持 scheduler、block table 和实际 cache 内容的 token 计数一致。

### 2.3 logical block group、physical tensor 与 physical page

这三个概念必须分开：

- **logical block group**：scheduler 为一种 cache spec 维护的 block table。
- **physical tensor**：worker 实际注册给 attention backend/KVT 的 GPU storage。
- **physical page**：一个 block id 在 storage 中对应的定长字节跨度。

vLLM 可以让多个 layer view 共享同一块 physical tensor，但它们仍可能属于不同 logical block group，并持有不同 block table。Blade-KVT 的 parser 接收的是各 logical group 的 block-id 列表，然后把它们映射到已注册 storage 的物理偏移。

## 3. Kimi K3 的 target cache

Kimi K3 是 KDA + MLA hybrid 模型。两类 cache 的语义完全不同。

### 3.1 KDA cache

每个 KDA layer 的 runtime cache 是两个 tensor：

```text
conv_state:
  [num_blocks, state_len, 3 * local_heads * head_dim]

recurrent_state:
  [num_blocks, local_heads, head_dim, head_dim]
```

其中：

```text
state_len = short_conv_kernel_size - 1 + gamma
local_heads = num_heads / tensor_parallel_size
```

当前 PAI-vLLM 的 KDA conv state 使用 SD layout：

```text
[state_len, Q/K/V channels]
```

而不是 DS layout：

```text
[Q/K/V channels, state_len]
```

Blade-KVT 通过 `kda_conv_dim_first=false` 区分这一点。

一个 KDA physical page 的有效字节可表示为：

```text
[conv state][recurrent state][padding]
```

padding 的存在是为了让不同 cache spec 使用统一 page stride。KVT 通过 `kda_page_stride` 获取 block 之间真正的物理间隔，不能简单用有效 state 大小推导下一个 block 地址。

#### TP remap

当 P/D tensor parallel size 不同时：

- `P TP > D TP`：多个 P rank 的 Q/K/V channel shard 和 recurrent head shard合并到一个 D rank。
- `P TP < D TP`：一个 P rank 的 channel/head shard拆分到多个 D rank。
- `P TP == D TP`：同 rank 的整页一对一复制。

KDA state 代表 prefix 末端的递归状态，不是普通的逐 token KV。因此当前 Blade-KVT 在 `reach_last_token` 后发送 KDA page，而不是把它当成 MLA 一样按 token block 连续流式发送。

### 3.2 MLA cache

K3 fused MLA 的逻辑 tensor shape 是：

```text
[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
```

当前 K3 kernel 固定：

```text
kv_lora_rank = 512
qk_rope_head_dim = 64
latent head_size = 576
```

KVT 使用：

```text
hybrid_attn_token_size = 576 * element_size
```

确定每个 token 需要复制的真实 MLA payload。

K3 MLA 是 rank-replicated cache。TP remap 时，Blade-KVT 不应像普通 MHA 那样拼接 KV heads：

- P/D 等 TP：rank 一对一发送。
- P TP 大于 D TP：每个 D rank 只需要一个代表性 P rank 的 MLA 副本。
- P TP 小于 D TP：P 侧 MLA 副本 fan-out 到相应 D ranks。

MLA view 使用 KDA-sized physical page 时，只有页首的真实 MLA prefix 有效：

```text
[token 0 latent][token 1 latent]...[token B-1 latent][padding]
```

Blade-KVT 不复制尾部 padding。

### 3.3 K3 的 KVT `cache_shape=8`

K3 分支强制：

```text
BLLM_KVTRANS_CACHE_SHAPE=8
attn_pack_size=1
```

Blade-KVT 将其解释成：

```text
task block groups:
  0 .. num_gdn_layers-1 : KDA groups
  num_gdn_layers        : exactly one replicated MLA group
```

`parse_block_kimi_k3.cpp` 当前有如下契约：

```cpp
group_idx = num_gdn_layers;
src_groups.size() == group_idx + 1;
dst_groups.size() == group_idx + 1;
src.attn_pack_size == 1;
dst.attn_pack_size == 1;
```

因此 `cache_shape=8` 不是任意 hybrid cache 的通用编码，而是 K3 target cache 的专用协议。

![K3 target 与 drafter KV cache layout](./imgs/kimi-k3-cache-layout.svg)

## 4. DFlash 与 DSpark 的 drafter cache

### 4.1 共同 backbone

当前 Qwen3 DSpark 实现复用 DFlash backbone：

- `DSparkProposer` 继承 `DFlashProposer`。
- `Qwen3DSparkModel` 继承 `DFlashQwen3Model`。
- 两者使用相同的 draft attention layers 和相同的 paged KV cache。

因此 DFlash 与 DSpark 的 cache layout 相同；差别主要在 query 数量和采样过程。

### 4.2 context KV 的产生

DFlash/DSpark 不是直接复用 target 的 K/V。它们执行以下过程：

1. target model forward 产出指定层的 hidden states；
2. drafter 对这些 hidden states 做每层 K/V projection；
3. 对 K 做 RMSNorm 和 RoPE；
4. 按 drafter 自己的 block table/slot mapping 写入 draft KV cache；
5. draft query pass 读取 context KV，并把 query token 的 K/V 写入同一套 paged cache。

因此 PD 场景必须满足下面二选一：

- P 节点产生并传输 drafter context KV；
- D 节点对完整 prefix 重新计算 drafter context KV。

当前设计显然选择第一种，否则 D 节点仍需对整个 prompt 做 drafter prefill，会削弱 PD 分离收益。

### 4.3 DFlash 与 DSpark 的 query 差异

设 `gamma = num_speculative_tokens`：

- DFlash：每个请求运行 `gamma + 1` 个 drafter query。
- DSpark：每个请求运行 `gamma` 个 query，再使用 Markov head 按顺序采样。

这不会改变每层 KV 的基本物理 shape，但会影响：

- lookahead slot 数量；
- 当前 step 写入的 slot mapping；
- CUDA graph capture token 数；
- KDA `state_len` 中为 speculative token 预留的空间；
- P/D 尾部重算长度。

### 4.4 attention backend 对物理 layout 的影响

DFlash/DSpark draft layer 使用普通 FullAttention 或 SlidingWindowAttention cache spec。

FlashAttention 暴露的 tensor view：

```text
[2, num_blocks, block_size, num_kv_heads, head_dim]
```

其中维度 0 为 K/V。底层 storage 可根据 `VLLM_KV_CACHE_LAYOUT` 使用 NHD 或 HND stride。

FlashInfer NHD view：

```text
[num_blocks, 2, block_size, num_kv_heads, head_dim]
```

FlashInfer HND 的物理顺序：

```text
[num_blocks, 2, num_kv_heads, block_size, head_dim]
```

所以“传输 drafter KV”不能只知道它是 DFlash/DSpark，还必须知道：

- attention backend；
- NHD/HND；
- block size；
- local KV heads；
- head dim；
- dtype；
- P/D TP 映射；
- FullAttention 与 SlidingWindow 的 logical group 边界。

## 5. 当前 PD 数据流

目标数据流应是：

```text
1. D scheduler 为 target cache 和 drafter cache 分配 blocks
2. D 将目标 block ids 和 worker/layout 信息发送给 P
3. P 只计算 prompt 前 n-(gamma+1) 个 token
4. P 生成 K3 KDA/MLA cache
5. P 从 target hidden states 生成 DFlash/DSpark context KV
6. KVT 将 target cache 与 drafter cache 传到 D
7. D 等待所有必需 cache domains 完成
8. D 将成功加载的 token 标记为 computed
9. D 本地重算最后 gamma+1 个 token
10. D 进入 speculative decode/verify
```

P 节点在 prompt batch 上不需要生成 draft token，但必须执行步骤 5。

`hmz/dspark_v3` 中的提交：

```text
9cc80052c6 Preserve DFlash context KV when skipping drafting
```

已经把 `skip_decode_drafting` 的返回点移动到：

```python
self.model.precompute_and_store_context_kv(...)
```

之后。这是 DFlash/DSpark 与 PD 组合的必要条件。

K3 分支中，`skip_decode_drafting` 仍在 context KV precompute 之前返回，因此尚未满足该条件。

## 6. 为什么 K3 + DFlash/DSpark + PD 当前冲突

### 6.1 分支能力没有合并

`hmz/dspark_v3` 不包含 K3 model/K3 KVT 提交；K3 分支不包含 `dspark_v3` 最近的 context-KV、dummy run、async rejection 和长度处理修复。

直接选择任一分支都得不到完整能力。

### 6.2 K3 parser 不接受额外 drafter groups

加入 drafter 后，scheduler 看到的 cache groups 类似：

```text
[KDA group 0]
[KDA group 1]
...
[K3 MLA group]
[draft FullAttention group 0]
[draft FullAttention/SWA group 1]
...
```

而 Blade-KVT K3 parser 只接受：

```text
[KDA groups...][one K3 MLA group]
```

额外 group 会触发 `src_groups.size() == num_gdn_layers + 1` 断言。

### 6.3 vLLM 注册阶段也可能先失败

K3 `_build_kvt_args()` 强制 `attn_pack_size == 1`，并要求每个 `KVCacheTensor.shared_by` 中只有一个 non-indexer attention entry。

vLLM 统一 page size 并构造 shared physical tensor 时，K3 MLA layer 和少量 drafter attention layers 可能被放进同一个 `shared_by`。这会使 non-indexer attention 数量大于 1，在进入 Blade-KVT 之前就失败。

即使 cache grouping 恰好把它们分开，Blade-KVT 的 block-group 数量断言仍会失败。

### 6.4 drafter context KV 缺失

K3 分支当前逻辑：

```text
skip_decode_drafting
  -> return
  -> 没有 precompute_and_store_context_kv
```

结果是 P 节点没有可供传输的 drafter prefix cache。D 节点若把对应 token 标记为已计算，将读取未初始化、旧数据或刚被 zero kernel 清空的 KV。

### 6.5 load 与 zeroing 必须覆盖新增 group

D 节点为远端 cache 分配的新 blocks 可能同时进入：

```text
SchedulerOutput.new_block_ids_to_zero
```

外部 load 和 zero kernel 如果作用于同一 block，会存在覆盖竞态。组合实现必须保证：

- 所有计划从 KVT 加载的 target 和 drafter blocks，都从同一个 scheduler output 的 zero list 中排除；
- 只排除 load target blocks，不排除 store blocks；
- block id 如果是 per-group namespace，过滤必须携带 group/domain 信息；
- load 失败时不能把未成功加载的 tokens 标记为 computed。

### 6.6 缺少组合测试

在两个 vLLM 分支和 blade-kvt main 中均未找到 Kimi K3 与 DFlash/DSpark 组合 PD 的测试或显式支持声明。

![当前冲突与建议目标架构](./imgs/kimi-k3-pd-target-architecture.svg)

## 7. 修改计划

### 7.1 总体原则

建议把传输拆成两个逻辑 cache domain：

```text
target-k3 domain:
  KDA groups + one replicated MLA group
  parser = KIMI_K3_MLA_CACHE_SHAPE

drafter-attn domain:
  DFlash/DSpark FullAttention/SWA groups
  parser = standard FlashAttention/FlashInfer parser
```

两个 domain 可以：

1. 复用同一条底层 transport/session，但具有独立 layout descriptor 和完成信号；或
2. 使用两个 KVT client/server 实例，并在 vLLM connector 层聚合完成状态。

不建议简单取消 `num_gdn_layers + 1` 断言后，把 drafter groups 当成 K3 MLA 继续解析。K3 MLA 是 replicated latent cache，而 drafter cache 是普通 K/V cache，TP remap、token size 和布局都不同。

### 7.2 Phase 0：建立可合并基线

目标：在一个集成分支中同时获得 K3 与最新 DSpark/DFlash 能力。

工作项：

1. 以 K3 分支为基线，合入 `hmz/dspark_v3` 的 DFlash/DSpark commits。
2. 至少包含：
   - P 节点跳过 drafting 时仍写 context KV；
   - dummy/profile path 对 `skip_decode_drafting` 的处理；
   - rejection count 的 GPU-side 处理；
   - DFlash max length、DP dummy sequence length 和 CUDA graph 修复。
3. 处理 `gpu_model_runner.py`、`speculative.py` 和 `dflash.py` 的冲突。
4. 保留 K3 KDA/MLA、multimodal boundary 和 KVT zeroing 相关修改。
5. 明确支持的 drafter checkpoint：
   - DFlash checkpoint architecture；
   - DSpark 当前 `Qwen3DSparkModel` 限制是否继续保留；
   - checkpoint hidden size、vocab、embedding/lm_head 是否能与 K3 target 对齐。

交付条件：

- K3 不开 PD 时能分别启动 DFlash 和 DSpark。
- 普通 target 的既有 DFlash/DSpark + PD 测试不回退。
- K3 不开 speculative decoding 时既有 PD 行为不回退。

### 7.3 Phase 1：引入 cache-domain 描述

目标：不要再用一个全局 `cache_shape` 隐式描述所有 cache groups。

建议在 vLLM hybrid connector 内新增类似结构：

```python
@dataclass
class CacheTransferDomain:
    name: str
    group_ids: list[int]
    layer_names: list[str]
    layout_kind: Literal["kimi_k3_target", "flash", "flashinfer"]
    cache_shape: int
    block_size: int
    token_bytes: list[int]
    physical_tensors: dict[str, torch.Tensor]
```

修改位置：

- `vllm/v1/hybrid_connector/__init__.py`
  - group 分类不能只返回 `gdn/attn`；
  - 需要区分 target K3 MLA 与 drafter attention；
  - 使用 model tag、layer-name set 或显式 cache owner，避免依赖脆弱的字符串匹配。
- `vllm/v1/hybrid_connector/kvtbackend.py`
  - `_build_kvt_args()` 拆成按 domain 构建；
  - K3 特有参数只传给 target-k3 domain；
  - drafter domain 使用自身 backend/layout/head metadata。
- `vllm/v1/hybrid_connector/engine_proxy.py`
  - worker info、block ids 和完成状态携带 domain id。

关键不变量：

- 每个 logical cache group 恰好属于一个 transfer domain。
- 一个 domain 的 parser 只能接收该 domain 声明的 group ids。
- domain 之间不能通过列表位置隐式推断类型。

### 7.4 Phase 2：隔离物理注册与 block table

目标：避免 K3 MLA 与 drafter attention 在 `KVCacheTensor.shared_by` 中被误认为一个 attention pack。

需要评估并实现以下方案之一：

#### 方案 A：在 KV cache config 阶段隔离 storage pool

- target K3 groups 使用 KDA-sized physical pages；
- drafter groups 使用普通 attention pages；
- scheduler 维护独立 block pools/block tables；
- KVT 注册天然按 domain 分开。

优点：

- layout 清晰；
- parser 简单；
- 不会把 K3 padding 施加到所有 drafter pages。

代价：

- 需要扩展当前“统一 page size/共享 physical tensor”的分组与分配逻辑；
- scheduler 的多 pool 支持和 prefix cache 行为需要系统验证。

#### 方案 B：允许共享 storage，但注册 domain-specific views

- 保留统一 page stride；
- 为 target 与 drafter 建立互不混淆的 flat view；
- 每个 domain 携带自己的 logical group id 到 physical offset 映射；
- 同一 storage 可被多个 KVT domain 注册，但禁止发送重叠的有效区域。

优点：

- 对 scheduler block allocation 改动较小。

风险：

- 同一物理 storage 多次注册；
- block-id namespace 和 storage offset 更难证明正确；
- zeroing、abort 和复用时更容易出现跨 domain 竞态。

建议优先验证方案 A；若当前 scheduler 架构成本过高，再采用方案 B。

### 7.5 Phase 3：扩展 Blade-KVT 多 domain 传输

目标：让 K3 target parser 与普通 attention parser 在同一请求中协作，而不是互相替代。

blade-kvt 修改点：

- Python API：
  - `blade_kvt/kv_transfer_impl.py`
  - client/server constructor 接受 domain descriptors，或允许创建命名子实例。
- pybind/API：
  - `kvtransfer/kvtransfer_pybind.cpp`
  - 序列化 domain id、layout kind、group range 和完成信号。
- worker metadata：
  - `kvtransfer/include/common.h`
  - `kvtransfer/src/common.cpp`
  - 不再假设一个 worker 只有一个 `cache_shape`。
- parser dispatch：
  - `kvtransfer/src/tx_stub.cpp`
  - 按 domain 选择 `parse_kimi_k3_*`、FlashAttention 或 FlashInfer parser。
- K3 parser：
  - `kvtransfer/src/parse_block_kimi_k3.cpp`
  - 保持其内部 `KDA + one MLA` 契约；
  - 断言应针对 target-k3 domain 的局部 groups，而不是整个请求的全部 groups。

完成语义：

```text
request load complete =
  target-k3 domain complete
  AND drafter-attn domain complete
```

任何必需 domain 失败，都不能把整个 prefix 标记为已加载。

### 7.6 Phase 4：补齐 producer/consumer 生命周期

#### P 节点

修改 `vllm/v1/spec_decode/dflash.py`：

```text
prompt batch + KVT producer:
  build context slot mapping
  precompute_and_store_context_kv
  skip query drafting
  expose drafter domain as ready
```

需要保证：

- context slot mapping 来自 drafter cache group，而不是 K3 MLA group；
- padding request/CUDA graph dummy request 不写真实 cache；
- partial prefix hit 时只写缺失区间；
- multimodal hidden-state selection 与 target layer ids 一致；
- async scheduler 不会在 context KV 尚未 ready 时触发 send。

#### D 节点

需要保证：

- target 与 drafter domain 都完成后才推进 `num_computed_tokens`；
- 本地重算 `gamma + 1` 时使用正确的 target/drafter block tables；
- load 失败、超时和 P abort 时，两类 blocks 都能释放；
- fallback local prefill 会覆盖两类 cache，不保留半完成的远端状态。

### 7.7 Phase 5：zeroing、复用与一致性

目标：外部 load 不能被同 step 的 zero kernel 覆盖。

修改方向：

1. 每个 backend/domain metadata 暴露 load target block ids。
2. `HybridScheduler.build_connector_meta()` 构建 metadata 后，立即从实际发送给 worker 的 `SchedulerOutput.new_block_ids_to_zero` 中过滤。
3. 对多 group namespace 使用：

```text
(cache_group_id, block_id)
```

而不是裸 `block_id`。
4. 保留现有 `sched_discard_zero_block_ids()`，它保护 pending-list 阶段；同时增加 same-output filtering，保护 block ids 已经被 `take_new_block_ids()` drain 的窗口。
5. 只过滤 load blocks，不能过滤 P 侧 store blocks。
6. 只有实际成功加载的 token 才能被 `mark_loaded/_step_loaded` 计入 computed。

还需要覆盖：

- block reuse；
- request abort；
- prefix-cache hit；
- partial block；
- null block；
- CUDA graph padding block；
- P/D 异步完成顺序。

### 7.8 Phase 6：配置检查、可观测性与回退

启动时应明确拒绝不支持的组合：

- K3 + KVT 但 Blade-KVT 缺少 K3 ABI；
- DSpark checkpoint architecture 不支持；
- P/D draft backend layout 不兼容；
- P/D block size 不可映射；
- TP remap 比例不合法；
- K3 target domain 或 drafter domain 缺失；
- multimodal placeholder 被 `n-(gamma+1)` 边界切开。

建议新增日志：

```text
domain name
group ids
layer names
layout kind
registered storage address/size
page stride
token bytes
P/D TP mapping
scheduled/load/ready token count
zero-excluded block ids count
```

建议增加 feature flag：

```text
VLLM_KIMI_K3_DRAFTER_PD=0/1
```

在完成全部测试前默认关闭。关闭时：

- 明确报错，或
- 自动回退到 D 节点本地 prefill；

不能静默进入只加载 target、不加载 drafter 的半支持状态。

## 8. 测试计划

### 8.1 单元测试

vLLM：

- cache group 能稳定分为 target-k3 与 drafter-attn domains；
- domain-specific block ids 不串组；
- `skip_decode_drafting` 仍写 context KV；
- partial prefix hit 只填充缺失 draft slots；
- load blocks 从 same-output zero list 排除；
- load failure 不推进 computed tokens；
- abort 释放所有 domain 的 blocks。

Blade-KVT：

- K3 target domain 仍满足 `KDA + one MLA`；
- drafter FlashAttention NHD/HND block parsing；
- drafter FlashInfer NHD/HND block parsing；
- equal TP、P TP > D TP、P TP < D TP；
- incomplete last block；
- manager block size 与 kernel block size不同；
- 两个 domain 的完成信号乱序到达。

### 8.2 cache correctness 测试

对同一 prompt 比较：

1. 单机本地 prefill 后的 target/drafter cache；
2. P 计算、KVT 传输、D 尾部重算后的 target/drafter cache。

比较范围：

- KDA conv state；
- KDA recurrent state；
- K3 MLA 有效 prefix；
- 每层 drafter K；
- 每层 drafter V；
- 最后 partial block 的有效 slots；
- padding 区域不作为模型正确性比较对象，但要验证不会被读取。

建议保留物理页 dump 工具，同时增加 logical slot dump，避免只比较 flat storage 时被 padding 干扰。

### 8.3 端到端矩阵

至少覆盖：

| 维度 | 取值 |
| --- | --- |
| method | DFlash、DSpark |
| target cache dtype | BF16、K3 支持的 FP8 模式 |
| draft cache dtype | BF16、FP8（若 checkpoint/backend 支持） |
| P/D TP | 相等、P>D、P<D |
| draft backend | FlashAttention、FlashInfer |
| cache layout | NHD、HND |
| gamma | 1、4、checkpoint 最大值 |
| prompt | 短 prompt、跨 block、最后一块不满、长上下文 |
| scheduler | sync、async/substep |
| prefix cache | miss、full hit、partial hit |
| modality | text、K3 image prompt |
| failure | P abort、load timeout、一个 domain 失败 |

正确性判据：

- 与不做 PD 的同配置输出 token 一致；
- verification logits 在容差内一致；
- acceptance length 分布无异常下降；
- 无 NaN、无 stale KV、无非法 block access；
- 不出现同 block 的 load/zero 覆盖；
- request 完成/取消后无 block leak。

### 8.4 性能与资源验收

需要测量：

- P prefill latency；
- target 与 drafter 各自传输字节数；
- 两 domain 传输是否可并行；
- D wait time；
- D 本地 `gamma + 1` 重算耗时；
- TTFT、TPOT；
- GPU cache memory；
- 注册同一 storage 多次时的额外开销；
- P/D TP remap 的 copy kernel 时间。

组合能力只有在“正确传输 drafter KV”后仍显著优于 D 节点完整 drafter prefill，才有上线价值。

## 9. 分阶段提交建议

建议把实现拆成可独立审查和回滚的提交：

1. `[SpecDecode] Preserve DFlash context KV on KVT producers`
2. `[Core] Classify target and drafter KV cache domains`
3. `[KVT] Build domain-specific cache registration metadata`
4. `[Blade-KVT] Support multi-layout domains per PD request`
5. `[PD] Aggregate K3 target and drafter load completion`
6. `[BugFix] Exclude all hybrid load blocks from same-step zeroing`
7. `[Test] Add Kimi K3 DFlash/DSpark PD cache parity coverage`
8. `[Docs] Document Kimi K3 drafter PD layout and constraints`

每个提交都应保持：

- K3 无 speculative decoding 的 PD 可用；
- 非 K3 的 DFlash/DSpark PD 可用；
- feature flag 关闭时不改变现网行为。

## 10. 最终验收标准

只有同时满足以下条件，才能声明支持 Kimi K3 + DFlash/DSpark + PD：

1. P 节点实际生成了 target 与 drafter prefix cache。
2. KVT 使用正确的独立 layout 传输了两类 cache。
3. D 节点在两类 cache 都完成后才推进 computed token。
4. D 本地正确重算最后 `gamma + 1` 个 token。
5. P/D TP 相等和不相等场景均通过 cache parity 测试。
6. FullAttention/SWA、FlashAttention/FlashInfer 的 group 和 layout 均未混淆。
7. load blocks 不会被 zero kernel 覆盖。
8. abort、timeout、fallback 和 prefix-cache reuse 不泄漏 blocks。
9. 端到端输出与本地 prefill 基线一致。
10. 存在持续运行的组合回归测试，而不是只靠一次手工验证。

在这些条件完成前，配置层应明确拒绝该组合或回退到 D 节点本地 prefill。
