# vLLM PD 分离：社区方案 vs Develop 分支深度对比

> 本文对 vLLM 的 PD（Prefill/Decode）分离方案进行深度代码级分析，对比社区（NIXL）和内部 develop 分支（HybridConnector + blade-kvt）两种实现的架构差异、调度机制和传输粒度。

---

## 一、架构总览

### 1.1 社区方案（NIXL Connector）

- **接口**：`KVConnectorBase_V1` 插件接口，分为 Scheduler 侧和 Worker 侧方法
- **传输库**：NVIDIA NIXL（RDMA 零拷贝传输）
- **调度集成**：通过 `RequestStatus.WAITING_FOR_REMOTE_KVS` 状态，在 scheduler 内部管理 KV 传输生命周期
- **传输粒度**：Block 级别
- **传输方向**：D 端主动 RDMA READ（pull 模式）

### 1.2 Develop 分支（HybridConnector + blade-kvt）

- **接口**：`HybridConnector` 实现 `KVConnectorBase_V1`，但通过 `on_add_req()` 拦截请求，内部使用 `HybridScheduler` + `HybridWorker`
- **传输库**：blade-kvt（内部 C++ KV 传输库，支持 RDMA）
- **调度集成**：请求在进入 scheduler 之前被拦截，KV 传输在独立的 disagg 线程中异步完成
- **传输粒度**：Token 级别
- **传输方向**：P 端主动 PUSH

---

## 二、调度器同步调用分析（Q1）

### 2.1 社区：`_schedule_prefills()` 中的 Connector 调用

社区在 `_schedule_prefills()` 循环中（`vllm/v1/core/sched/scheduler.py` ~line 590-770），对每一个待 prefill 的请求**逐个同步调用**两个 connector 方法：

```python
# scheduler.py - _schedule_prefills() 内部循环
for request in self.waiting:
    # ...省略前置检查...

    # ① 同步调用 connector 获取可复用的 token 数
    computed_tokens, \
    externally_computed_tokens, \
    load_kv_async = connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens)

    # ...分配 KV blocks...

    # ② 同步调用 connector 更新传输状态
    connector.update_state_after_alloc(request, blocks)

    # ③ 如果支持异步 KV 加载，设置请求状态为 WAITING
    if load_kv_async:
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
```

**这里的"同步调用"指的是这两个方法在 scheduler 的主循环（EngineCore 的事件循环）中直接执行，scheduler 在它们返回之前不会继续处理下一个请求。**

### 2.2 这两个方法具体做什么？

#### `get_num_new_matched_tokens()` — NIXL scheduler 端

```python
# vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
def get_num_new_matched_tokens(
    self, request: "Request", num_computed_tokens: int
) -> tuple[int, bool]:
    params = request.kv_transfer_params
    if params is not None and params.get("do_remote_prefill"):
        token_ids = request.prompt_token_ids or []
        actual = self._mamba_prefill_token_count(len(token_ids))
        count = actual - num_computed_tokens
        if count > 0:
            return count, True   # 返回 (匹配 token 数, True=异步加载)
    return 0, False
```

**关键发现：完全不做任何 IO 操作。** 只是检查 `request.kv_transfer_params` 字典中的标记字段，然后做简单的算术运算。没有网络调用，没有 RDMA，没有阻塞。

#### `update_state_after_alloc()` — NIXL scheduler 端

```python
# vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    params = request.kv_transfer_params
    if params.get("do_remote_prefill") or (...):
        if params.get("remote_block_ids"):
            unhashed_local_block_ids = blocks.get_unhashed_block_ids_all_groups()
            local_block_ids = self.get_sw_clipped_blocks(unhashed_local_block_ids)
            # 仅仅是把请求加入 _reqs_need_recv 字典
            self._reqs_need_recv[request.request_id] = (request, local_block_ids)
        params["do_remote_prefill"] = False
```

**关键发现：也完全不做 IO。** 只是把请求及其本地 block_ids 记录到 `self._reqs_need_recv` 字典中，等后续 `build_connector_meta()` 打包成 `NixlConnectorMetadata` 传给 Worker 侧。

### 2.3 `WAITING_FOR_REMOTE_KVS` 状态转换机制

#### 设置状态

```python
# scheduler.py ~line 790
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)
    request.num_computed_tokens = num_computed_tokens
    continue   # 不加入 running，跳过本次调度
```

#### 检查状态（每次调度循环开头）

```python
# scheduler.py ~line 584
if self._is_blocked_waiting_status(
    request.status
) and not self._try_promote_blocked_waiting_request(request):
    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        logger.debug("%s is still in WAITING_FOR_REMOTE_KVS state.", request_id)
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue   # 还没完成传输，跳过
```

#### Promote 逻辑

```python
def _try_promote_blocked_waiting_request(self, request):
    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False      # KV 传输还没完成
        self._update_waiting_for_remote_kv(request)
        request.status = RequestStatus.WAITING  # 重新变为 WAITING
        return True
```

#### `finished_recving_kv_req_ids` 的填充

```python
# 在 update_from_output 中调用
def _update_from_kv_xfer_finished(self, kv_connector_output):
    for req_id in kv_connector_output.finished_recving or ():
        req = self.requests[req_id]
        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
```

### 2.4 请求完整生命周期：需要经过多少个 Scheduler Step

```
Step N: 首次调度
  ├── _schedule_prefills 中请求从 waiting queue 弹出
  ├── get_num_new_matched_tokens() → (ext_tokens, True)
  ├── allocate_slots() 分配 KV blocks
  ├── update_state_after_alloc() 记录到 _reqs_need_recv
  ├── 请求状态 → WAITING_FOR_REMOTE_KVS，放回 step_skipped_waiting
  ├── build_connector_meta() 构建 NixlConnectorMetadata
  └── Worker 在 _get_kv_connector_output 中调用 start_load_kv()，发起 RDMA

Step N+1 ~ N+K-1: KV 传输进行中（空转）
  ├── 调度器遍历 waiting queue，发现 WAITING_FOR_REMOTE_KVS
  ├── _try_promote_blocked_waiting_request → False
  └── 请求被跳过（continue）

Step N+K: KV 传输完成被 Worker 检测到
  ├── Worker 的 get_finished() 发现 RDMA handle 完成
  ├── 返回 finished_recving
  └── update_from_output() 中 _update_from_kv_xfer_finished() 填充 finished_recving_kv_req_ids

Step N+K+1: 请求被 Promote 回 WAITING 并重新调度
  ├── _try_promote_blocked_waiting_request → True
  ├── _update_waiting_for_remote_kv(): kv_cache_manager.cache_blocks()
  ├── request.status → WAITING
  └── 在同一个 _schedule_prefills 循环或下一个 step 中被调度到 RUNNING
```

**总结：最少需要 3 个 scheduler step**（Step N 调度 + Step N+K 完成检测 + Step N+K+1 promote 并重新调度）。由于 `finished_recving` 信号要通过 `model_runner_output.kv_connector_output` 从 Worker 回到 Scheduler（发生在 `update_from_output` 中，即当前 step 的 forward 之后），即使 KV 传输在 Step N 就完成了，最早也要到 Step N+1 的 `update_from_output` 才能知道。

### 2.5 Worker 侧 KV 传输时序

```python
# vllm/v1/worker/kv_connector_model_runner_mixin.py
@contextmanager
def _get_kv_connector_output(scheduler_output, wait_for_save=True):
    output = KVConnectorOutput()
    kv_connector = get_kv_transfer_group()
    kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

    # ① 在 model forward 之前启动异步 KV 加载
    kv_connector.start_load_kv(get_forward_context())
    try:
        yield output   # ② model forward 在这里执行
    finally:
        if wait_for_save:
            kv_connector.wait_for_save()
        # ③ forward 完成后检查哪些传输已完成
        output.finished_sending, output.finished_recving = (
            kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
```

NIXL 的 `start_load_kv()` 是**非阻塞的**，调用 NIXL 的 `make_prepped_xfer` + `transfer` 发起异步 RDMA 读操作后立即返回。传输在后台进行，每个 step 通过 `get_finished()` 检查完成状态。

同时 NIXL 的 `wait_for_layer_load()` 和 `save_kv_layer()` 都是 **no-op**：

```python
# nixl/connector.py
def wait_for_layer_load(self, ...):
    pass   # NO-OP

def save_kv_layer(self, ...):
    pass   # NO-OP
```

### 2.6 Develop 分支：HybridConnector 的异步方案

#### 请求拦截

```python
# vllm/v1/engine/core.py - add_request()
kvconn = self.scheduler.get_kv_connector()
if kvconn:
    eaten = kvconn.on_add_req(request)
    if eaten:
        return   # 请求被 HybridConnector "吃掉"，不进入调度器
```

#### `on_add_req` — 请求拦截入口

```python
# vllm/v1/hybrid_connector/__init__.py
def on_add_req(self, req: "Request") -> bool:
    load_count, save_count = self._backend.get_operations(req)
    if load_count > 0 or save_count > 0:
        self._waiting.append((req, load_count, save_count))
        return True    # 返回 True 表示请求被拦截
    return False       # 不需要 KV 传输，正常走调度器
```

#### `_step_waiting` — Core 线程中分配 Blocks

```python
def _step_waiting(self):
    while self._waiting:
        req, load_count, save_count = self._waiting[0]
        kvblks = sched_allocate_slots(req, load_count > 0, save_count > 0, ...)
        if kvblks is None:
            break
        self._waiting.popleft()

        if load_count > 0:
            self._loading[req.request_id] = _LoadingReq(req)
            # 关键：在 disagg 线程的 event loop 中异步执行
            coro = self._on_add_req(req, kvblks)
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            self._loaded.append(req)
```

#### `_on_add_req` — Disagg 线程中异步执行 KV 传输

```python
async def _on_add_req(self, req: Request, kvblks: KVCacheBlocks):
    local = req.num_computed_tokens
    rmt = await self._backend.async_get_num_new_matched_tokens(req, local)
    ioret = await self._backend.async_update_state_after_alloc(req, kvblks, rmt)
    if rmt <= 0 or ioret is not None:
        await self.mark_loaded(req, ioret)   # KV 加载完成
    else:
        _q_append(self._prepared, req.request_id)
```

#### `_step_loaded` — 请求直接回到调度器

```python
def _step_loaded(self):
    while self._loaded:
        req = self._loaded.popleft()
        ioret = get_param(req, HB_IORET, None)
        req.num_computed_tokens += ioret.n or 0
        sched_add_req(req)   # 直接加入调度器的 waiting queue
```

#### `step()` 在 EngineCore 中的调用位置

```python
# vllm/v1/engine/core.py - step()
def step(self):
    kvconn = self.scheduler.get_kv_connector()
    if kvconn:
        kvconn_outputs = kvconn.step()   # 在 scheduler.schedule() 之前！
    scheduler_output = self.scheduler.schedule()
    # ...
```

### 2.7 开销对比总结

| 维度 | 社区 (NIXL) | Develop (HybridConnector) |
|------|------------|--------------------------|
| **Scheduler 阻塞** | `get_num_new_matched_tokens` + `update_state_after_alloc` 在主循环同步执行（但很轻量，纯内存操作） | `on_add_req` 只做 `_waiting.append()`，O(1) |
| **请求到 Prefill 延迟** | 至少 3 个调度周期（分配 → 传输完成 promote → 调度执行） | KV 就绪后下一个 step 直接 `sched_add_req`，浪费 0 个 step |
| **KV 传输与调度的耦合** | 传输状态通过 `KVConnectorOutput` 在 step 间传递，scheduler 每步检查完成状态 | 独立 asyncio 线程管理传输，传输完成后直接移入 `_loaded` 队列 |
| **状态管理** | 复用 `RequestStatus` 枚举（`WAITING_FOR_REMOTE_KVS`），与正常调度状态混合 | 独立状态队列（`_waiting/_loading/_loaded/_saving/_saved/_aborting`），干净隔离 |
| **主循环空转开销** | 每个 step 都要遍历 waiting queue，对 `WAITING_FOR_REMOTE_KVS` 请求执行 `_try_promote_blocked_waiting_request` 检查 | `step()` 只从队列 pop 已完成请求，O(queue_size) |
| **KV 完成信号路径** | Worker → `model_runner_output` → `update_from_output` → `finished_recving_kv_req_ids` → 下一个 step promote（至少 1 step 延迟） | disagg 线程 → `_loaded` 队列 → `_step_loaded` → `sched_add_req`（几乎 0 延迟） |
| **并行度** | KV 传输只在 Worker model forward 阶段启动 | KV 传输在独立 disagg 线程中随时进行，与 scheduler + model forward 完全并行 |
| **调度器侵入性** | 在调度器核心逻辑中引入状态管理、promote 检查等额外逻辑 | 对调度器几乎零侵入 |

**核心开销差异**：社区方案的主要开销不是"同步调用"本身（那两个函数很快），而是**调度轮次的传递延迟** — 请求需要经过多个 scheduler step 才能从"KV 正在传输"状态被 promote 回正常队列。每个 step 约等于一次 model forward 的时间（几十到几百毫秒），累计可达数百毫秒。Develop 分支通过独立线程 + 队列机制，让请求在 KV 传输完成后立即进入可调度状态。

---

## 三、Block 级别传输与异构 TP 映射（Q2）

### 3.1 KV Cache 注册与 Block 描述符构建

#### GPU 内存注册

在引擎初始化时，`register_kv_caches()` 将所有 KV cache GPU 内存注册到 NIXL：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py - register_kv_caches()
for cache in cache_list:
    base_addr = cache.data_ptr()          # GPU 显存基地址
    caches_data.append(
        (base_addr, curr_tensor_size_bytes, self.device_id, "")
    )

# 注册到 NIXL
descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
```

`block_len_per_layer` 数组记录了每个 region（每层的 K 或 V）中一个 block 的字节数，是后续构建 transfer descriptor 的基础。

#### `_build_fa_local()` — 本地描述符（worker.py ~line 1088）

为本地的每一个 block，生成一个 `(addr, len, device_id)` 三元组：

```python
def _build_fa_local(self, base_addresses, block_size_ratio):
    num_blocks = self.num_blocks * block_size_ratio
    result = []
    for i, base_addr in enumerate(base_addresses):
        kv_block_len = self.get_backend_aware_kv_block_len(i) // block_size_ratio
        page_stride = self.block_len_per_layer[i] // block_size_ratio
        for block_id in range(num_blocks):
            block_offset = block_id * page_stride
            addr = base_addr + block_offset
            result.append((addr, kv_block_len, self.device_id))
        # FlashInfer 的情况下，V 也要单独注册
        if self.transfer_topo.virtually_split_kv_in_blocks:
            for block_id in range(num_blocks):
                v_addr = base_addr + block_id * page_stride + kv_block_len
                result.append((v_addr, second_split, self.device_id))
    return result
```

**每个描述符精确描述了一个 block 在 GPU 显存中的位置和大小。** 对于 FlashAttn（HND layout），K 和 V 在不同 region 中；对于 FlashInfer（blocks-first layout），K/V 在同一 tensor 里交错存放，逻辑上"虚拟拆分"成 K 和 V 两个 region。

#### `_build_fa_remote()` — 远端描述符（worker.py ~line 1124）

为远端 P 节点的每个 block 构建描述符，核心是**处理异构 TP 下的 head 偏移**：

```python
def _build_fa_remote(self, plan, nixl_agent_meta, block_size_ratio):
    for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
        local_block_len = self.get_backend_aware_kv_block_len(i)
        local_block_len = local_block_len // num_attn_reads  # 只读部分 head

        # 偏移到对应的 head 切片
        rank_offset = plan.rank_offset_factor * remote_kv_block_len
        page_size = nixl_agent_meta.block_lens[i]

        for block_id in range(num_blocks):
            block_offset = block_id * page_size
            addr = base_addr + block_offset + rank_offset
            result.append((addr, local_block_len, nixl_agent_meta.device_id))
```

关键参数：
- `rank_offset`：当 D_TP > P_TP 时，不同的 D worker 读取 P 的不同 KV head 切片
- `local_block_len // num_attn_reads`：当 P_TP > D_TP 时，每次读取量为总量除以读取次数
- `page_size`：远端每个 block 的完整大小，用于计算 block 间步进

### 3.2 Block ID → NIXL 描述符索引的映射

`_compute_desc_ids()` 是关键的映射函数（worker.py line 77）：

```python
def _compute_desc_ids(self, block_ids, dst_num_blocks, ...):
    num_blocks = dst_num_blocks
    # 快速路径：numpy 向量化广播计算
    block_arr = np.concatenate(block_ids)[None, :]      # shape: (1, N_blocks)
    region_ids = np.arange(num_fa_regions)[:, None]     # shape: (R_regions, 1)
    return (region_ids * num_blocks + block_arr).flatten()
```

**核心公式：`desc_index = region_id * num_blocks + block_id`**

描述符列表布局：

```
[region0_block0, region0_block1, ..., region0_blockN,
 region1_block0, region1_block1, ..., region1_blockN, ...]
```

### 3.3 RDMA 传输发起流程

#### Scheduler → Worker 的元数据传递

```python
# metadata.py
class NixlConnectorMetadata(KVConnectorMetadata):
    reqs_to_recv: dict[ReqId, ReqMeta]   # 需要接收 KV 的请求
    reqs_to_save: dict[ReqId, ReqMeta]   # 需要保存 KV 的请求
    reqs_to_send: dict[ReqId, float]     # 需要发送的请求（带过期时间）

@dataclass
class ReqMeta:
    local_block_ids: BlockIds       # 本地 block IDs
    local_physical_block_ids: BlockIds  # 物理 block IDs
    tp_size: int
    remote: RemoteMeta | None       # 远端信息

@dataclass
class RemoteMeta:
    block_ids: BlockIds             # 远端 block IDs (list[list[int]])
    host: str
    port: int
    engine_id: str
    request_id: str
```

#### `start_load_kv()` — D 端 Worker 启动传输

```python
# nixl/worker.py ~line 1922
def start_load_kv(self, metadata: NixlConnectorMetadata):
    for req_id, meta in metadata.reqs_to_recv.items():
        meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
            meta.local_block_ids)
        remote_engine_id = meta.remote.engine_id
        if remote_engine_id not in self._remote_agents:
            self._background_nixl_handshake(req_id, remote_engine_id, meta)
            continue
        self._read_blocks_for_req(req_id, meta)
```

#### `_read_blocks_for_req()` — Block ID 重映射

```python
# nixl/worker.py ~line 2010
def _read_blocks_for_req(self, req_id, meta):
    plan = self.tp_mappings[engine_id]
    meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
        meta.remote.block_ids, remote_info.remote_physical_blocks_per_logical)

    # 为每个需要读取的远端 TP rank 构造 ReadSpec
    read_specs = [
        ReadSpec(
            remote_rank=rank,
            local_block_ids=[
                list(local_block_ids[g])
                if rank in plan.source_ranks_per_group[g] else []
                for g in range(num_groups)
            ],
            remote_block_ids=[
                list(remote_block_ids[g])
                if rank in plan.source_ranks_per_group[g] else []
                for g in range(num_groups)
            ],
        )
        for rank in plan.all_source_ranks
    ]
    for spec in read_specs:
        self._read_blocks(spec, ...)
```

#### `_read_blocks()` — 实际 RDMA Transfer

```python
# nixl/worker.py ~line 2124
def _read_blocks(self, read_spec, dst_engine_id, request_id, ...):
    # 1. 计算本地和远端的描述符索引
    remote_block_descs_ids = self._compute_desc_ids(
        block_ids=remote_block_ids,
        dst_num_blocks=self.dst_num_blocks[dst_engine_id], ...)
    local_block_descs_ids = self._compute_desc_ids(
        block_ids=local_block_ids,
        dst_num_blocks=self.dst_num_blocks[self.engine_id], ...)

    # 2. 创建 RDMA 传输对象（本地 desc[i] ← 远端 desc[i] 配对）
    handle = self.nixl_wrapper.make_prepped_xfer(
        "READ",
        local_xfer_side_handle,
        local_block_descs_ids,       # 本地 block 描述符索引列表
        remote_xfer_side_handle,
        remote_block_descs_ids,      # 远端 block 描述符索引列表
        notif_msg=notif_id,
    )

    # 3. 发起异步 RDMA READ（非阻塞）
    self.nixl_wrapper.transfer(handle)

    # 4. 记录 handle 用于后续完成检查
    self._recving_transfers[request_id].append(handle)
```

### 3.4 完整数据流示例

```
Scheduler 侧:
  request 的 kv_transfer_params 包含:
    remote_block_ids = [[1, 5, 7, 12]]
    remote_engine_id = "engine_P_0"
  │
  ▼
  update_state_after_alloc() 分配本地 blocks:
    local_block_ids = [[3, 8, 10, 15]]
  │
  ▼
  打包到 NixlConnectorMetadata.reqs_to_recv

Worker 侧:
  start_load_kv() 遍历 reqs_to_recv
  │
  ▼
  _read_blocks_for_req():
    根据 TPMapping 为每个源 TP rank 构造 ReadSpec
  │
  ▼
  _read_blocks():
    _compute_desc_ids(remote=[1,5,7,12], num_blocks=N):
      对于 region 0: indices = [0*N+1, 0*N+5, 0*N+7, 0*N+12] = [1, 5, 7, 12]
      对于 region 1: indices = [1*N+1, 1*N+5, 1*N+7, 1*N+12] = [N+1, N+5, N+7, N+12]
      合并: [1, 5, 7, 12, N+1, N+5, N+7, N+12]
    同理计算 local desc IDs: [3, 8, 10, 15, N+3, N+8, N+10, N+15]
  │
  ▼
  nixl_wrapper.make_prepped_xfer("READ", local_handle, local_descs, remote_handle, remote_descs)
    NIXL 内部：对 remote_descs[i] 和 local_descs[i] 配对
    每对对应一个 block 的 RDMA READ 操作
  │
  ▼
  nixl_wrapper.transfer(handle) → 发起批量 RDMA READ
```

### 3.5 异构 TP 映射（TPMapping）

`tp_mapping.py` 的 `compute_tp_mapping()` 处理 P 和 D 端 TP size 不同的情况。

#### 情况 1：D_TP ≥ P_TP（例如 P_TP=2, D_TP=4）

每个 D worker 只从一个 P worker 读取**部分 KV head**：

```python
if tp_size >= remote_tp_size:
    attn_ranks = [tp_rank * remote_tp_size // tp_size]
    rank_offset_factor = tp_rank % (tp_size // remote_tp_size)
```

示例（P_TP=2, D_TP=4）：

```
D rank 0: attn_ranks=[0], rank_offset_factor=0 → 读 P rank 0 的前半 head
D rank 1: attn_ranks=[0], rank_offset_factor=1 → 读 P rank 0 的后半 head
D rank 2: attn_ranks=[1], rank_offset_factor=0 → 读 P rank 1 的前半 head
D rank 3: attn_ranks=[1], rank_offset_factor=1 → 读 P rank 1 的后半 head
```

在 `_build_fa_remote()` 中通过 `rank_offset` 实现 head 切片定位：

```python
rank_offset = plan.rank_offset_factor * remote_kv_block_len
addr = base_addr + block_id * page_size + rank_offset
```

#### 情况 2：P_TP > D_TP（例如 P_TP=4, D_TP=2）

每个 D worker 从**多个 P worker** 读取，各自读完整的 head：

```python
else:
    abs_tp = remote_tp_size // tp_size
    start = tp_rank * abs_tp
    # D rank 0 → 从 P rank 0, 1 读取
    # D rank 1 → 从 P rank 2, 3 读取
    attn_ranks = list(range(start, start + abs_tp))
```

此时需要将本地描述符"拆分"，每份对应从一个远端 rank 接收的数据写入位置：

```python
# add_remote_agent 中
if tp_ratio < 0:  # P_TP > D_TP
    for handle_data in self._build_local_splits_from_plan(plan, ...):
        descs = self.nixl_wrapper.get_xfer_descs(handle_data, ...)
        handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
        self.src_xfer_handles_by_tp_ratio[tp_ratio].append(handle)
```

`_build_local_splits_from_plan()` 将本地 block 按 head 维度切分：

```python
def _build_local_splits_from_plan(self, plan, src_blocks_data, ...):
    for p_idx, p_rank in enumerate(plan.all_source_ranks):
        fa_slot = plan.rank_to_attention_slot.get(p_rank, 0)
        handle = []
        for j, (addr, local_len, dev) in enumerate(src_blocks_data):
            chunk = local_len // fa_num_splits
            handle.append((addr + fa_slot * chunk, chunk, dev))
        yield handle
```

#### `_read_blocks_for_req()` 中的 Block ID 按 TP 分配

```python
read_specs = [
    ReadSpec(
        remote_rank=rank,
        local_block_ids=[
            list(local_block_ids[g])
            if rank in plan.source_ranks_per_group[g] else []
            for g in range(num_groups)
        ],
        remote_block_ids=[
            list(remote_block_ids[g])
            if rank in plan.source_ranks_per_group[g] else []
            for g in range(num_groups)
        ],
    )
    for rank in plan.all_source_ranks
]
```

#### `_read_blocks()` 中选择对应的本地 Handle

```python
# P_TP > D_TP: 使用按 tp_ratio 拆分的本地 handle
if tp_ratio < 0 and not self.use_mla:
    local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
else:
    # 标准情况或 D_TP > P_TP: 使用完整的本地 handle
    local_xfer_side_handle = self.src_xfer_handles_by_block_size[remote_block_size]
```

### 3.6 异构 TP 映射示意图

```
Case 1: D_TP=4, P_TP=2
┌─────────────────────────┐   ┌─────────────────────────┐
│  P rank 0               │   │  P rank 1               │
│  [head0, head1, head2,  │   │  [head4, head5, head6,  │
│   head3]                │   │   head7]                │
└───────┬───────┬─────────┘   └───────┬───────┬─────────┘
        │       │                     │       │
  offset=0  offset=1           offset=0  offset=1
        │       │                     │       │
        ▼       ▼                     ▼       ▼
   D rank 0  D rank 1           D rank 2  D rank 3
   [h0,h1]  [h2,h3]            [h4,h5]  [h6,h7]


Case 2: P_TP=4, D_TP=2
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ P rank 0 │ │ P rank 1 │ │ P rank 2 │ │ P rank 3 │
│ [h0,h1]  │ │ [h2,h3]  │ │ [h4,h5]  │ │ [h6,h7]  │
└─────┬────┘ └────┬─────┘ └─────┬────┘ └────┬─────┘
      │           │              │           │
      └─────┬─────┘              └─────┬─────┘
            │                          │
            ▼                          ▼
       D rank 0                   D rank 1
       [h0,h1,h2,h3]             [h4,h5,h6,h7]
       (local split 0 ← P0)      (local split 0 ← P2)
       (local split 1 ← P1)      (local split 1 ← P3)
```

---

## 四、传输粒度对比：NIXL Block-Level vs Blade-KVT Token-Level

### 4.1 Blade-KVT 的 Token-Level Transfer

```python
# vllm/v1/hybrid_connector/kvtbackend.py
# 首次发送：指定 src/dst block IDs + token 范围
self._bladkv_cli.submit_req_send2(
    dst_inst_name, dst_wid, reqm.reqid,
    seen_tokens=reqm.seen_tokens,    # 已传输的 token 数
    new_tokens=reqm.new_tokens,      # 新增需要传输的 token 数
    has_last_token=reqm.has_last_token,
    src_block_ids=reqm.p_block_ids,  # P 端 block IDs
    dst_block_ids=reqm.d_block_ids,  # D 端 block IDs
    dst_worker_info=dst_worker_info,
    stepid=stepid,
)

# 增量发送：只指定新增的 token 范围，不需要再传 block IDs
self._bladkv_cli.submit_delta_send(
    reqm.reqid,
    seen_tokens=reqm.seen_tokens,    # 已传输的 token 位置
    new_tokens=reqm.new_tokens,      # 新增 token 数
    has_last_token=reqm.has_last_token,
    stepid=stepid,
)
```

### 4.2 Blade-KVT 的 Layer-by-Layer Pipeline

```python
# kvtbackend.py - async_save_kv_layer
def async_save_kv_layer(self, layer_name, kv_layer, m):
    layer_idx = extract_layer_index(layer_name)
    # 在每一层 attention 计算完成后记录 CUDA event
    self._bladkv_cli.record_event(layer_idx, torch.cuda.current_stream())
    # blade_kvt 传输线程等待该 event → 立即开始传输该层 KV
    # 同时 GPU 继续计算下一层
```

时序示意：

```
GPU Compute:  [Layer 0] [Layer 1] [Layer 2] [Layer 3] ...
               ↓ event   ↓ event   ↓ event   ↓ event
KV Transfer:  .........[Xfer L0] [Xfer L1] [Xfer L2] [Xfer L3]
                         ↑ overlap with compute ↑
```

NIXL **没有这种 layer-by-layer pipeline 机制**，`save_kv_layer` 和 `wait_for_layer_load` 都是 no-op。D 端发起的 RDMA READ 读取的是 P 端已完成所有层计算后的完整 KV cache。

### 4.3 关键差异总结

| 维度 | NIXL (社区) | Blade-KVT (Develop) |
|------|------------|---------------------|
| **传输粒度** | Block 级别：以完整 block 为单位 | Token 级别：`(seen_tokens, new_tokens)` 精确指定 |
| **传输方向** | D 端主动 RDMA READ（pull） | P 端主动 PUSH（`submit_req_send`） |
| **增量传输** | 不支持，每次传输完整 block 列表 | 支持 `submit_delta_send`，只传新增 token |
| **层级流水线** | 无（`wait_for_layer_load` 是 no-op），等所有层计算完后一次性读取 | `record_event` 实现逐层 CUDA event pipeline，传输与计算重叠 |
| **异构 TP** | 双向支持（D_TP>P_TP 和 P_TP>D_TP），通过 `TPMapping` + 描述符偏移 | 通过 `_get_dist` 计算目标 worker，P_TP>D_TP 暂不支持 |
| **传输发起** | `nixl_wrapper.make_prepped_xfer("READ")` + `transfer()` | `bladekv.KVTransferClient.submit_req_send2()` |
| **完成检测** | 轮询 `check_xfer_state(handle)` | `send_done` RPC 回调 |
| **Prefix Cache** | 裁剪 remote block IDs，跳过已缓存 block | 通过 `seen_tokens` 参数跳过已传输 token |
| **描述符注册** | 预注册所有 block 的地址+长度描述符，零拷贝 | 传入 cache tensor + block_bytes/token_bytes |

---

## 五、总结

### 5.1 社区方案的优势

1. **标准化接口**：`KVConnectorBase_V1` 插件接口清晰，易于替换不同的传输后端
2. **异构 TP 完整支持**：双向（D_TP>P_TP 和 P_TP>D_TP）均支持
3. **与 vLLM Block Manager 天然对齐**：block 级别传输与 vLLM 的 paged attention block 管理一致
4. **RDMA 零拷贝**：通过描述符预注册实现高效传输，无需额外的内存拷贝
5. **透明度高**：传输逻辑在 Python 层可见，便于调试和扩展

### 5.2 Develop 分支的优势

1. **调度零侵入**：请求被拦截后，调度器完全不感知 KV 传输
2. **零浪费 step**：KV 传输完成后请求立即可调度，无 promote 延迟
3. **Token 级别精细控制**：支持增量传输 (`submit_delta_send`)，减少不必要的数据传输
4. **层级流水线**：`record_event` 实现 compute/transfer 重叠，显著降低端到端延迟
5. **完全异步**：独立 disagg 线程 + asyncio event loop，与主调度循环完全解耦
6. **状态管理清晰**：独立的队列系统（waiting/loading/loaded/saving/saved/aborting）

### 5.3 适用场景

- **社区方案**更适合：标准化部署、异构 TP 需求、需要与 vLLM 上游保持兼容的场景
- **Develop 分支**更适合：低延迟生产环境、大规模 PD 分离部署、需要增量传输和层级流水线优化的场景
