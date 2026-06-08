---
title: vLLM EPD 分离与 Vineyard Encoder Cache Transfer 深度分析
date: 2026-06-08
tags: []
---
# vLLM EPD 分离与 Vineyard Encoder Cache Transfer 深度分析

> 基于 develop 分支代码，分析 EPD（Encoder-Prefill-Decode）三级分离架构中 Encoder Cache 的传输机制。

## 一、总体架构

### 1.1 EPD 分离概述

EPD 是 PD（Prefill-Decode）分离的扩展，针对多模态模型（如 Qwen-VL）引入第三级——**Encoder 节点**：

| 节点 | 角色 | 职责 |
|------|------|------|
| E (Encoder) | ec_producer | 只运行 ViT 视觉编码器，生成 embedding，写入 v6d 共享内存 |
| P (Prefill) | ec_consumer | 从 v6d 加载 embedding，执行 LLM prefill，生成 KV cache |
| D (Decode) | — | 从 P 节点接收 KV cache，执行自回归解码 |

E→P 的 Encoder Cache 传输由 **EC Connector** 框架负责，与 KV Connector（P→D）平行但独立。

### 1.2 EC Connector 框架

EC Connector 与 KV Connector 的设计模式一致：

- **配置入口**：`--ec-transfer-config`，解析为 `ECTransferConfig`（`vllm/config/ec_transfer.py`）
- **三个配置字段**：
  - `ec_connector`：连接器类名（如 `ECGlobalVineyardConnector`）
  - `ec_role`：`ec_producer`（E 节点）或 `ec_consumer`（P 节点）
  - `ec_v6d_socket`：Vineyard IPC socket 路径
- **三角色实例化**：`ECConnectorRole.SCHEDULER`、`WORKER`、`APISERVER`，分别在不同进程中运行
- **工厂模式**：`ECConnectorFactory`（`vllm/distributed/ec_transfer/ec_connector/factory.py`）注册了三个实现：
  - `ECExampleConnector`：示例实现
  - `ECVineyardConnector`：单实例 Vineyard
  - `ECGlobalVineyardConnector`：生产环境使用的全局 Vineyard 连接器

### 1.3 核心接口 ECConnectorBase

```python
# vllm/distributed/ec_transfer/ec_connector/base.py

class ECConnectorBase(ABC):
    @abstractmethod
    def has_caches(self, request) -> list[bool]:
        """检查每个 mm_feature 是否在远程缓存中可用"""

    @abstractmethod
    def start_load_caches(self, encoder_cache, **kwargs):
        """从远程加载 encoder cache 到本地 encoder_cache dict"""

    @abstractmethod
    def save_caches(self, encoder_cache, mm_hash, **kwargs):
        """将 encoder cache 保存到远程存储"""

    @abstractmethod
    def build_connector_meta(self, scheduler_output) -> ECConnectorMetadata:
        """构建传输给 Worker 的元数据"""

    @abstractmethod
    def get_prefetch_hints(self, scheduler) -> list[str]:
        """返回下一步可能需要的 mm_hash 列表，供预取使用"""
```

---

## 二、Vineyard (v6d) 存储层

### 2.1 什么是 Vineyard

Vineyard（v6d）是一个分布式共享内存系统，专为大数据和机器学习场景设计。在 EPD 架构中：

- E 节点和 P 节点部署在**同一台物理机**上
- v6d 提供 IPC（进程间通信）级别的共享内存访问
- 数据通过 v6d daemon 管理，不经过网络栈

### 2.2 内容寻址的 Key 设计

```python
# vllm/distributed/ec_transfer/ec_connector/epd_global_v6d_backend.py

@dataclass
class VineyardMultiModalKey:
    mm_hash: str

    @property
    def embd(self) -> str:
        return f"{self.mm_hash}_embd"   # embedding tensor 的 key

    @property
    def meta(self) -> str:
        return f"{self.mm_hash}_meta"   # MultiModalMetadata 的 key
```

**设计核心**：同一张图片 → 同一个 `mm_hash` → 同一个 v6d key → 天然去重。

当多个请求包含同一张图片时，E 节点只需编码一次，后续请求直接复用 v6d 中已有的 embedding（通过 `ObjectAlreadyExistError` 检测）。

### 2.3 GlobalVineyardBackend 实现

`GlobalVineyardBackend`（`epd_global_v6d_backend.py`）是底层的 v6d 操作封装，提供以下核心方法：

#### Save 操作（E 节点使用）

```python
def save_caches(self, mm_hash, embedding_tensor, **kwargs):
    key = VineyardMultiModalKey(mm_hash)
    non_blocking = kwargs.get("non_blocking", False)

    # 获取 v6d 共享内存中的 tensor 引用（可写模式）
    obj = self._base_client.get(key.embd, peer="local", unsafe=True)
    resolver = obj.resolver()

    # 将 GPU tensor 拷贝到 v6d 共享内存
    resolver.copy_(embedding_tensor)
```

- `peer="local"`：只在本机的 v6d 实例内查找
- `unsafe=True`：可写模式（v6d 默认对象不可变，unsafe 允许就地写入）
- `resolver()`：返回一个 numpy/torch 视图，直接指向共享内存区域
- `copy_()`：将 GPU 上的 embedding tensor 拷贝到该共享内存

#### Load 操作（P 节点使用）

```python
def load_caches(self, mm_hash):
    key = VineyardMultiModalKey(mm_hash)
    obj = self._base_client.get(key.embd, peer="local")
    tensor = obj.resolver().detach().clone().cuda()
    return tensor
```

- `resolver()`：获取共享内存视图
- `.detach().clone()`：复制到独立的 CPU tensor（断开共享内存引用，避免后续 v6d GC 影响）
- `.cuda()`：拷贝到 GPU

#### Load to Pinned（异步预取路径使用）

```python
def load_caches_to_pinned(self, mm_hash):
    key = VineyardMultiModalKey(mm_hash)
    obj = self._base_client.get(key.embd, peer="local")
    tensor = obj.resolver().detach().clone().pin_memory()
    return tensor
```

- `.pin_memory()`：返回 pinned memory tensor，后续可用 `non_blocking=True` 做异步 H2D 拷贝

#### 预创建 v6d 对象（API Server 使用）

```python
def prepare_for_save_embd(self, mm_hash, shape, dtype):
    key = VineyardMultiModalKey(mm_hash)
    try:
        obj = self._base_client.create(
            vineyard.Tensor, shape=shape, dtype=dtype)
        self._base_client.put(obj, name=key.embd)
    except ObjectAlreadyExistError:
        pass  # 缓存命中，同 hash 的 embedding 已存在
```

API Server 在请求进入时预创建空的 v6d Tensor 对象，E 节点的 Worker 后续只需 `copy_()` 填充数据。

#### 保存 Metadata

```python
def save_meta(self, mm_hash, mm_meta):
    key = VineyardMultiModalKey(mm_hash)
    pickled = pickle.dumps(mm_meta)
    obj = self._base_client.create(vineyard.Blob, len(pickled))
    obj.copy(0, pickled)
    self._base_client.put(obj, name=key.meta)
```

将 `MultiModalMetadata` pickle 序列化后存为 v6d Blob 对象，key 后缀为 `_meta`。

---

## 三、E 节点（Encoder / Producer）详细流程

### 3.1 E 节点的"伪引擎"特性

E 节点启动完整的 vLLM 引擎（scheduler + worker + model runner），但通过多处判断跳过所有 LLM 相关逻辑：

#### (1) profile_run 提前返回

```python
# gpu_model_runner.py:6972-6978
# EC producer only runs the ViT encoder, skip decoder dummy runs.
if has_ec_transfer() and get_ec_transfer().is_producer:
    self._sync_device()
    self.encoder_cache.clear()
    gc.collect()
    return
```

profile_run 是 vLLM 初始化时计算可用 GPU 内存和编译 CUDA graph 的阶段。E 节点在完成 ViT 编码器的 dummy run 后直接返回，不执行 decoder 的 dummy runs。

#### (2) get_kv_cache_spec 返回空字典

```python
# gpu_model_runner.py:8315-8316
if has_ec_transfer() and get_ec_transfer().is_producer:
    return {}
```

返回空字典意味着 E 节点**不分配任何 KV cache block**，因为完全不需要做 LLM forward。这大幅节省了 GPU 显存，使 E 节点可以用较少的 GPU 资源。

#### (3) execute_model 只跑 ViT + Save

```python
# gpu_model_runner.py:4487-4494
if has_ec_transfer() and get_ec_transfer().is_producer:
    with self.maybe_get_ec_connector_output(
        scheduler_output,
        encoder_cache=self.encoder_cache,
    ):
        self._execute_mm_encoder(scheduler_output)   # 只跑 ViT
        self._save_ec_to_connector(scheduler_output)  # 保存到 v6d
```

完整的 execute_model 方法有数千行代码处理 attention、speculative decoding 等，但 E 节点只执行这几行。

### 3.2 _save_ec_to_connector 实现

```python
# gpu_model_runner.py:2941-2966
def _save_ec_to_connector(self, scheduler_output):
    non_blocking = True
    unique_mm_hashes: set[str] = set()

    # 收集本步调度的所有 unique mm_hashes
    for req_id, input_ids in scheduler_output.scheduled_encoder_inputs.items():
        req_state = self.requests.get(req_id)
        if req_state is None:
            continue
        for mm_input_id in input_ids:
            mm_hash = req_state.mm_features[mm_input_id].identifier
            unique_mm_hashes.add(mm_hash)

    # 逐个保存到 v6d
    for mm_hash in unique_mm_hashes:
        try:
            self.maybe_save_ec_to_connector(
                self.encoder_cache, mm_hash,
                non_blocking=non_blocking)
        except Exception as e:
            logger.error(f"Failed to save encoder cache for mm_hash {mm_hash}: {e}")

    # 确保所有非阻塞拷贝完成
    if non_blocking and unique_mm_hashes:
        torch.cuda.synchronize()
```

调用链：`_save_ec_to_connector` → `maybe_save_ec_to_connector`（mixin 静态方法）→ `ECGlobalVineyardConnector.save_caches()` → `GlobalVineyardBackend.save_caches()`

---

## 四、P 节点（Prefill / Consumer）详细流程

### 4.1 Scheduler 侧：判断是否从远程加载

#### has_caches 判定逻辑

```python
# global_vineyard_connector.py:372-401
def has_caches(self, request):
    if (
        self.is_producer
        or request.sampling_params is None
        or request.sampling_params.extra_args is None
        or "enable_disaggregated_vit" not in request.sampling_params.extra_args
    ):
        return [False for _ in request.mm_features]  # 需要本地计算
    else:
        return [True for _ in request.mm_features]    # 使用远程缓存
```

判定依据是 `sampling_params.extra_args` 中是否包含 `enable_disaggregated_vit`。这是在**请求级别**控制的——同一个 P 节点可以同时处理有 E 节点缓存和没有 E 节点缓存的请求，实现混合部署。

#### Scheduler 调度 encoder inputs

```python
# scheduler.py:1380-1530 — _schedule_encoder_inputs()

# 先检查远程缓存
if self.ec_connector is not None:
    remote_cache_has_item = self.ec_connector.has_caches(request)

# ...遍历每个 mm_feature...

# 如果远程有缓存，标记为外部加载
if self.ec_connector is not None and remote_cache_has_item[i]:
    mm_hashes_to_schedule.add(request.mm_features[i].identifier)
    external_load_encoder_input.append(i)
    num_embeds_to_schedule += num_encoder_embeds
    continue  # 不消耗 encoder_compute_budget
```

关键点：
- 远程加载的 encoder input **不消耗 `encoder_compute_budget`**（不需要跑 ViT）
- 但仍然消耗 `encoder_cache_manager` 的 slot（需要在 GPU 上缓存 embedding）
- 通过 `update_state_after_alloc` 将 mm_hash 加入 `_mm_datas_need_loads` 列表

#### update_state_after_alloc

```python
# global_vineyard_connector.py:402-409
def update_state_after_alloc(self, request, index):
    mm_hash = request.mm_features[index].identifier
    if mm_hash not in self._mm_datas_need_loads:
        self._mm_datas_need_loads.append(mm_hash)
```

每当 scheduler 为某个 encoder input 分配了 cache slot，就将其 mm_hash 记录到待加载列表中。

### 4.2 Scheduler 侧：构建 EC Metadata

```python
# scheduler.py:1162-1190

# Build the connector meta for ECConnector
if self.ec_connector is not None:
    ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
    scheduler_output.ec_connector_metadata = ec_meta

    if not self.is_ec_producer:
        # 生成预取提示
        prefetch_hashes = self.ec_connector.get_prefetch_hints(self)
        if prefetch_hashes:
            scheduler_output.prefetch_ec_mm_hashes = prefetch_hashes

        # 收集过期预取条目释放的 mm_hashes
        expired_freed = self.encoder_cache_manager.get_freed_mm_hashes()
        if expired_freed:
            scheduler_output.free_encoder_mm_hashes.extend(expired_freed)
```

`build_connector_meta` 实现：

```python
# global_vineyard_connector.py:411-421
def build_connector_meta(self, scheduler_output):
    if self.is_producer:
        return ECGlobalVineyardConnectorMetadata(mm_hashes=[])

    mm_hashes = self._mm_datas_need_loads.copy()
    self._mm_datas_need_loads.clear()
    return ECGlobalVineyardConnectorMetadata(mm_hashes=mm_hashes)
```

Consumer 侧将本步需要加载的 mm_hashes 打包到 metadata 中，随 `SchedulerOutput` 传给 Worker。

### 4.3 SchedulerOutput 中的 EC 相关字段

```python
# vllm/v1/core/sched/output.py
class SchedulerOutput:
    free_encoder_mm_hashes: list[str]                        # 需要释放的 mm_hashes
    ec_connector_metadata: ECConnectorMetadata | None = None  # EC 传输元数据
    prefetch_ec_mm_hashes: list[str] | None = None           # 下一步预取提示
```

### 4.4 Worker 侧：加载 Encoder Cache

Worker 通过 `ECConnectorModelRunnerMixin`（`vllm/v1/worker/ec_connector_model_runner_mixin.py`）集成 EC 加载逻辑。

#### Context Manager 模式

```python
# ec_connector_model_runner_mixin.py — _get_ec_connector_output()
@contextmanager
def _get_ec_connector_output(scheduler_output, encoder_cache,
                              prefetch_manager=None, **kwargs):
    output = ECConnectorOutput()
    ec_connector = get_ec_transfer()

    # 绑定 metadata
    ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

    if not ec_connector.is_producer:
        if prefetch_manager is not None:
            # 异步预取路径
            ...
        else:
            # 同步加载路径（默认）
            ec_connector.start_load_caches(encoder_cache, **kwargs)

    try:
        yield output
    finally:
        output.finished_sending, output.finished_recving = (
            ec_connector.get_finished(scheduler_output.finished_req_ids)
        )
        ec_connector.clear_connector_metadata()
```

生命周期：
1. 绑定 metadata
2. 加载 encoder cache（同步或异步）
3. yield → 调用者执行 encoder + gather
4. finally：收集完成信号，清理 metadata

#### Consumer 路径在 execute_model 中的位置

```python
# gpu_model_runner.py:3597-3605
with self.maybe_get_ec_connector_output(
    scheduler_output,
    encoder_cache=self.encoder_cache,
    prefetch_manager=self.ec_prefetch_manager,
) as ec_connector_output:
    self._execute_mm_encoder(scheduler_output)        # 跑本地 encoder（如有）
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

对于 P 节点：
- 远程缓存的 encoder inputs 已经通过 `start_load_caches` 加载到 `encoder_cache`
- `_execute_mm_encoder` 只处理没有远程缓存的 encoder inputs（如果有的话）
- `_gather_mm_embeddings` 统一从 `encoder_cache` 中取出所有 embedding

---

## 五、异步预取管线（Cross-Step Prefetch）

### 5.1 开关与配置

通过环境变量控制，默认**关闭**：

```python
# ec_connector_model_runner_mixin.py
_ENABLE_CROSS_STEP_PREFETCH = os.getenv(
    "VLLM_EPD_ENABLE_CROSS_STEP_PREFETCH", "False"
).lower() in ("true", "1")

_PREFETCH_WORKERS = int(os.getenv("VLLM_EPD_PREFETCH_WORKERS", "16"))
```

### 5.2 ECPrefetchManager 设计

`ECPrefetchManager`（`vllm/v1/worker/ec_prefetch_manager.py`）管理异步 encoder cache 加载，实现计算与传输的重叠。

#### 数据结构

```python
@dataclass
class PendingLoad:
    v6d_future: Future                         # phase 1: v6d → pinned CPU
    pinned_tensor: torch.Tensor | None = None  # phase 1 完成后的 pinned tensor
    gpu_tensor: torch.Tensor | None = None     # phase 2: H2D 后的 GPU tensor
    h2d_event: torch.cuda.Event | None = None  # H2D 完成标记
```

#### 双 CUDA Stream 设计

```python
def __init__(self, max_workers=None):
    self._copy_stream_low: torch.cuda.Stream   # priority=0, 给预取 H2D
    self._copy_stream_high: torch.cuda.Stream  # priority=-1, 给紧急 H2D
```

- **low-priority stream**：用于 `_advance()` 中的后台预取 H2D 拷贝，不阻塞主 forward
- **high-priority stream**：用于 `ensure_loaded()` 中的紧急 H2D 拷贝，优先级高于预取

### 5.3 三阶段流水线

```
Step N:
  scheduler → get_prefetch_hints() → prefetch_ec_mm_hashes

Step N+1 开始:
  ┌──────────────────────────────────────────────────────┐
  │ submit(prefetch_mm_hashes)                           │
  │   └── ThreadPool: v6d → pinned CPU (并行)            │
  │                                                      │
  │ _advance()                                           │
  │   └── low-priority stream: pinned → GPU (non_blocking)│
  │                                                      │
  │ ensure_loaded(mm_hashes_this_step)                    │
  │   └── 等待本步需要的 mm_hashes 就绪                   │
  │       ├── 已在 encoder_cache → 跳过                  │
  │       ├── 有 pending 条目 → 等 future + high-prio H2D │
  │       └── 无 pending → 同步 fallback                  │
  └──────────────────────────────────────────────────────┘

Step N+1 forward:
  _gather_mm_embeddings() → 从 encoder_cache 取出 embedding
```

#### submit() — 提交预取任务

```python
def submit(self, prefetch_mm_hashes, encoder_cache, load_fn):
    for mm_hash in prefetch_mm_hashes:
        if mm_hash in encoder_cache:
            continue  # 已缓存
        if mm_hash in self._pending:
            continue  # 已提交
        future = self._executor.submit(load_fn, mm_hash)
        self._pending[mm_hash] = PendingLoad(v6d_future=future)

    self._advance()  # 尝试推进已完成的条目
    return submitted
```

`load_fn` 绑定为 `GlobalVineyardBackend.load_caches_to_pinned()`，在线程池中执行 v6d 读取。

#### _advance() — 非阻塞推进

```python
def _advance(self):
    for load in self._pending.values():
        if load.gpu_tensor is not None:
            continue  # 已完成 H2D
        if load.pinned_tensor is None and not load.v6d_future.done():
            continue  # v6d 读取尚未完成
        if load.pinned_tensor is None:
            load.pinned_tensor = load.v6d_future.result()
        with torch.cuda.stream(low_stream):
            load.gpu_tensor = load.pinned_tensor.to(
                device=self._device, non_blocking=True)
            load.h2d_event = low_stream.record_event()
```

遍历所有 pending 条目，将已完成 v6d 读取的 tensor 通过 low-priority stream 异步拷贝到 GPU。

#### ensure_loaded() — 阻塞确保就绪

```python
def ensure_loaded(self, mm_hashes, encoder_cache, load_fn):
    for mm_hash in mm_hashes:
        if mm_hash in encoder_cache:
            continue  # 路径 1: 已缓存

        load = self._pending.get(mm_hash)
        if load is None:
            self._sync_load_into_cache(mm_hash, ...)  # 路径 3: 同步 fallback
            continue

        try:
            # 路径 2: 消费 pending 条目
            if load.pinned_tensor is None:
                load.pinned_tensor = load.v6d_future.result()  # 等待 v6d
            if load.gpu_tensor is None:
                with torch.cuda.stream(high_stream):
                    load.gpu_tensor = load.pinned_tensor.to(
                        device=self._device, non_blocking=True)
                    load.h2d_event = high_stream.record_event()
            load.h2d_event.synchronize()  # 同步等待 H2D 完成
            encoder_cache[mm_hash] = load.gpu_tensor
        except Exception:
            self._sync_load_into_cache(mm_hash, ...)  # fallback
        finally:
            self._pending.pop(mm_hash, None)
```

三条路径：
1. **已在 encoder_cache**：直接跳过
2. **有 pending 条目**：等待 future 完成，用 high-priority stream 做 H2D，同步 event
3. **无 pending**：回退到同步 `load_fn` 加载（prefetch miss）

#### cancel_prefetches() — Scheduler 驱动的取消

```python
def cancel_prefetches(self, mm_hashes):
    for mm_hash in mm_hashes:
        load = self._pending.pop(mm_hash, None)
        if load is None:
            continue
        if load.h2d_event is not None:
            load.h2d_event.synchronize()
        load.gpu_tensor = None
        load.pinned_tensor = None
```

当 scheduler 通过 `free_encoder_mm_hashes` 通知某些 mm_hash 不再需要时，worker 取消对应的预取任务，释放 CPU/GPU 内存。

### 5.4 Scheduler 侧预取提示生成

```python
# global_vineyard_connector.py — get_prefetch_hints()
def get_prefetch_hints(self, scheduler) -> list[str]:
```

基于 **proximity scoring**（邻近度评分）：

1. **构建候选集**：扫描 running + waiting 中的请求
2. **计算距离**：每个 mm_feature 到当前 `num_computed_tokens` 的距离
3. **过滤**：排除超过 `proximity_window`（= `max_age × per_req_tokens_per_step`）的候选
4. **排序**：按距离从近到远
5. **预算控制**：在 `encoder_cache_manager` 的空闲 slot 内选取 top-K
6. **过期管理**：`expire_prefetch_reservations(max_age)` 自动释放超龄条目

### 5.5 Worker 侧初始化

```python
# ec_connector_model_runner_mixin.py
def _init_ec_prefetch_manager(self) -> ECPrefetchManager | None:
    if not _ENABLE_CROSS_STEP_PREFETCH:
        return None  # 未启用则不创建
    if has_ec_transfer() and not get_ec_transfer().is_producer:
        return ECPrefetchManager(max_workers=_PREFETCH_WORKERS)
    return None
```

- 仅 Consumer（P 节点）且显式启用时创建
- 返回 None 时 `_get_ec_connector_output` 回退到同步 `start_load_caches` 路径

---

## 六、Worker 侧状态管理

### 6.1 encoder_cache 的释放

```python
# gpu_model_runner.py:1125-1131
# _update_states 中
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)

# 取消对应的预取
if scheduler_output.free_encoder_mm_hashes and self.ec_prefetch_manager:
    self.ec_prefetch_manager.cancel_prefetches(
        scheduler_output.free_encoder_mm_hashes)
```

`free_encoder_mm_hashes` 由 scheduler 的 `EncoderCacheManager` 管理，当 encoder input 对应的所有 decoder token 都已计算完毕时释放。

### 6.2 EC 传输完成通知

```python
# ec_connector_model_runner_mixin.py — _get_ec_connector_output finally 块
output.finished_sending, output.finished_recving = (
    ec_connector.get_finished(scheduler_output.finished_req_ids)
)
```

通过 `ECConnectorOutput` 将完成信号传回上层，最终反映到 `ModelRunnerOutput` 中。

---

## 七、端到端流程总结

### 7.1 完整数据流

```
API Server (收到图文请求)
    │
    ├── 计算 mm_hash (基于图片内容的哈希)
    ├── prepare_for_save_embd()
    │     └── 在 v6d 预创建空 Tensor 对象 (key: {mm_hash}_embd)
    ├── sampling_params.extra_args["enable_disaggregated_vit"] = True
    └── 发送请求到 E 节点 和 P 节点
         │
         ▼
E 节点 (ec_role = ec_producer)
    │
    ├── Scheduler:
    │     ├── has_caches() → [False] (producer 始终返回 False)
    │     ├── 正常调度 encoder inputs
    │     └── build_connector_meta() → 空 metadata
    │
    ├── Worker (execute_model):
    │     ├── 检测 is_producer → 进入 producer 分支
    │     ├── _execute_mm_encoder(scheduler_output)
    │     │     └── 运行 ViT 编码器，输出 embedding tensor
    │     │         结果存入 self.encoder_cache[mm_hash]
    │     └── _save_ec_to_connector(scheduler_output)
    │           ├── 遍历 scheduled_encoder_inputs 的 unique mm_hashes
    │           ├── save_caches(mm_hash, embedding_tensor)
    │           │     ├── v6d.get("{mm_hash}_embd", unsafe=True)
    │           │     │     获取可写共享内存引用
    │           │     └── resolver.copy_(embedding_tensor)
    │           │           GPU tensor → v6d 共享内存
    │           └── torch.cuda.synchronize()
    │                 确保所有拷贝完成
    │
    ├── 初始化特性:
    │     ├── profile_run: 跳过 decoder dummy → 不分配 KV cache
    │     └── get_kv_cache_spec: 返回 {} → 不创建 KV cache
    │
    └── 结果: embedding 已写入 v6d 共享内存
         │
         ▼
  ═══════ v6d 共享内存 (同机 IPC，无网络开销) ═══════
  ║ key: {mm_hash}_embd → embedding tensor          ║
  ║ key: {mm_hash}_meta → MultiModalMetadata (pickle)║
  ═══════════════════════════════════════════════════
         │
         ▼
P 节点 (ec_role = ec_consumer)
    │
    ├── Scheduler:
    │     ├── has_caches(request) → [True]
    │     │     (检测到 enable_disaggregated_vit in extra_args)
    │     ├── 标记为 external_load_encoder_input
    │     │     └── 不消耗 encoder_compute_budget（不需要跑 ViT）
    │     ├── update_state_after_alloc(request, i)
    │     │     └── mm_hash 加入 _mm_datas_need_loads
    │     ├── build_connector_meta()
    │     │     └── ECGlobalVineyardConnectorMetadata(mm_hashes=[...])
    │     └── get_prefetch_hints() → 下一步预取候选
    │
    ├── Worker (execute_model) — 同步路径（默认）:
    │     ├── bind_connector_metadata(ec_meta)
    │     ├── start_load_caches(encoder_cache)
    │     │     └── 遍历 metadata.mm_hashes:
    │     │           load_caches(mm_hash)
    │     │             ├── v6d.get("{mm_hash}_embd", peer="local")
    │     │             └── resolver.detach().clone().cuda()
    │     │                   v6d SHM → CPU copy → GPU
    │     │           encoder_cache[mm_hash] = tensor
    │     ├── _execute_mm_encoder()  → 跳过远程加载的 inputs
    │     └── _gather_mm_embeddings()
    │           └── 从 encoder_cache 取出 embedding → 注入 LLM 输入
    │
    ├── Worker (execute_model) — 异步预取路径（需启用）:
    │     ├── submit(prefetch_mm_hashes)
    │     │     └── 线程池: v6d → pinned CPU (并行于 forward)
    │     ├── _advance()
    │     │     └── low-priority stream: pinned → GPU (non_blocking)
    │     ├── ensure_loaded(mm_hashes_this_step)
    │     │     └── high-priority stream: 等待并同步
    │     └── _gather_mm_embeddings() → 取出 embedding
    │
    └── 继续 LLM prefill forward（带 embedding 的完整输入）
         │
         ▼
  ═══ KV Connector (blade-kvt) ═══
         │
         ▼
D 节点 → 自回归解码
```

### 7.2 关键设计要点

| 设计 | 说明 |
|------|------|
| **内容寻址去重** | 同一图片 mm_hash 相同 → v6d 中只存一份 → `ObjectAlreadyExistError` 跳过 → 多请求共享 |
| **零拷贝写入** | E 节点通过 `resolver().copy_()` 直接写入共享内存视图，避免中间拷贝 |
| **请求级控制** | `extra_args["enable_disaggregated_vit"]` 按请求控制 → 同一 P 节点可混合处理 |
| **不消耗编码预算** | 远程加载的 encoder input 不消耗 `encoder_compute_budget` → P 节点专注 LLM prefill |
| **E 节点极简** | 不分配 KV cache，不做 decoder forward → 资源占用极低 |
| **预取与计算重叠** | cross-step prefetch 在 step N+1 的 forward 期间并行加载 step N+2 的数据 |
| **双优先级 stream** | high-priority 给紧急 H2D → 不被预取 H2D 阻塞 → 确保时效性 |
| **Scheduler 驱动的生命周期** | 预取条目的创建和取消由 scheduler 统一管理 → worker 不自主过期 → 避免异步调度下的不一致 |

### 7.3 与 KV Connector 的对比

| 维度 | EC Connector (E→P) | KV Connector (P→D) |
|------|--------------------|--------------------|
| 传输内容 | Encoder embedding tensor | Attention KV cache blocks |
| 传输介质 | v6d 共享内存 (同机 IPC) | RDMA / blade-kvt (可跨机) |
| 传输粒度 | 按 mm_hash（整张图的 embedding） | 按 KV block（逐层 record_event） |
| 去重机制 | 内容寻址（mm_hash） | 无（每个请求独立传输） |
| 预取机制 | Cross-step prefetch (线程池 + 双 stream) | 无显式预取（layer-by-layer pipeline 自身实现重叠） |
| 配置入口 | --ec-transfer-config | --kv-transfer-config |
| 代码位置 | vllm/distributed/ec_transfer/ | vllm/distributed/kv_transfer/ |

---

## 八、生产配置示例

典型的 EPD 三节点配置中，EC 相关部分：

```json
// E 节点
{
  "ec_transfer_config": {
    "ec_connector": "ECGlobalVineyardConnector",
    "ec_role": "ec_producer",
    "ec_v6d_socket": "/tmp/vineyard.sock"
  }
}

// P 节点
{
  "ec_transfer_config": {
    "ec_connector": "ECGlobalVineyardConnector",
    "ec_role": "ec_consumer",
    "ec_v6d_socket": "/tmp/vineyard.sock"
  },
  "kv_transfer_config": {
    "kv_connector": "HybridConnector",
    "kv_role": "kv_producer"
  }
}

// D 节点
{
  "kv_transfer_config": {
    "kv_connector": "HybridConnector",
    "kv_role": "kv_consumer"
  }
}
```

E 和 P 节点共享同一个 `vineyard.sock`（同机部署），通过 v6d daemon 进行 IPC 通信。P 节点同时配置 EC（作为 consumer）和 KV（作为 producer），是 EPD 链条中的中间节点。D 节点只配置 KV connector。

---

## 九、源码文件索引

| 文件 | 核心内容 |
|------|----------|
| `vllm/config/ec_transfer.py` | ECTransferConfig 定义 |
| `vllm/distributed/ec_transfer/ec_connector/base.py` | ECConnectorBase 抽象基类 |
| `vllm/distributed/ec_transfer/ec_connector/factory.py` | ECConnectorFactory 工厂 |
| `vllm/distributed/ec_transfer/ec_connector/global_vineyard_connector.py` | ECGlobalVineyardConnector（scheduler 侧逻辑） |
| `vllm/distributed/ec_transfer/ec_connector/epd_global_v6d_backend.py` | GlobalVineyardBackend（v6d 操作封装） |
| `vllm/v1/worker/ec_connector_model_runner_mixin.py` | ECConnectorModelRunnerMixin（worker 侧 EC 集成） |
| `vllm/v1/worker/ec_prefetch_manager.py` | ECPrefetchManager（异步预取管线） |
| `vllm/v1/worker/gpu_model_runner.py` | GPUModelRunner 中的 EC producer/consumer 路径 |
| `vllm/v1/core/sched/scheduler.py` | Scheduler 中的 EC 调度集成 |
| `vllm/v1/core/sched/output.py` | SchedulerOutput EC 相关字段 |
