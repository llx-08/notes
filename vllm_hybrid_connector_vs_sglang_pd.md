# vLLM HybridConnector + KVTBackend vs SGLang 的 PD 分离对比

> 基于 `/mnt/data/llx/vllm`(分支带 HybridConnector)与 `/mnt/data/llx/sglang` 当前源码整理(2026-06)。

## 0. 设计哲学

LLM 引擎与 PD 分离之间的关系,类似 Linux 内核与驱动:前者提供稳定、通用的核心能力,后者根据具体硬件/生态灵活适配。PD 分离的实现高度依赖公司内部的技术栈,因此**不应将任何 PD 相关逻辑硬编码进 LLM 引擎**。LLM 引擎只需提供有限且克制的接口,遵循"如无必要,勿增接口"的原则。

HybridConnector 正是这一思想的具体实现:它不侵入引擎主链路,仅通过极简接口触发 KV Cache 的 load 和 save,其余所有逻辑均由 Connector 异步完成。

## 1. 整体架构对比

| 维度 | SGLang (`disaggregation/`) | vLLM HybridConnector + KVTBackend (`v1/hybrid_connector/`) |
|---|---|---|
| 抽象基类 | `BaseKVManager` + `BaseKVSender` + `BaseKVReceiver` + `BaseKVBootstrapServer`,共约 12 个抽象方法 (`base/conn.py:79-216`) | `KVConnectorBase_V1`,4 个抽象方法 + 1 个 `step()`;`HybridBackend` 子类只暴露 `get_operations`、`async_load_kv`、`async_save_kv_layer`、`build_backend_meta`、`async_cleanup` (`hybrid_connector/__init__.py:111-353`) |
| 请求状态机 | 显式跨多个队列:`PrefillBootstrapQueue` → `waiting_queue` → `disagg_prefill_inflight_queue`,以及 D 侧 `DecodePreallocQueue` → `DecodeTransferQueue` → `waiting_queue`(`prefill.py:91`,`decode.py:253/1335`) | 单条 `_waiting → _loading → _loaded → _saving → _saved` 全部在 `HybridScheduler` 内部 deque 中流转,不进 engine 主调度结构 (`hybrid_connector/__init__.py:591-610`) |
| KV 状态感知 | scheduler 主循环每轮调用 `pop_bootstrapped()`、`pop_transferred()`、`process_disagg_prefill_inflight_queue()`,内部全是 `polls = poll_and_all_reduce_attn_cp_tp_group(...)` (`prefill.py:282/619`,`decode.py:1477/1799`) — 即"Scheduler 主动 poll"模式 | engine 主循环不做 poll;backend 在自己的 asyncio loop 里完成传输,通过 RPC 回调 `_on_save_done`/`_on_load_done` 把结果推回 (`hybrid_connector/__init__.py:961/993`) |
| 引擎入侵面 | scheduler 里硬编码 `KVPoll` 五状态分支 + `release_kv_cache()` + `prepare_abort()` 内联处理 (`prefill.py:649-700`,`decode.py:1521-1568`) | engine 主链路只见两个 hook(`start_load_kv` / `save_kv_layer`),`wait_for_layer_load`/`wait_for_save` 直接 `return`(`hybrid_connector/__init__.py:1453,1463`) |
| Backend 可插拔性 | 每加一个 backend(mooncake/nixl/ascend/mori/fake)都要重写 ~1.5k 行 conn.py(`mooncake/conn.py` 1787 行,`common/conn.py` 1357 行) | 一个 backend 一套 `HybridBackend` 子类,工厂在 `_get_backend_cls` 列了 9 种 + 组合(kvt/kvs/mooncake/v6d_object/local_file/migration/kvt+migration/kvt+kvs/v6d_object+kvt)(`__init__.py:480-559`) |
| 请求结束语义 | `KVPoll.Success` 之前请求一直停在 `disagg_prefill_inflight_queue`,scheduler 必须把生命周期撑到 KV 传完 | `request_finished_all_groups` 返回 `(False, {kv_transfer_pending: True})` 直接吐 token 给客户端,KV 传输继续在后台跑(`__init__.py:1466-1500`) |

## 2. 接口对比:HybridConnector 的极简 hook

vLLM 上游 `KVConnectorBase_V1` 仍然保留了一组同步语义接口:

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py
class KVConnectorBase_V1(ABC):
    # Worker side
    @abstractmethod start_load_kv(...)
    @abstractmethod wait_for_layer_load(layer_name)   # 同步等待
    @abstractmethod save_kv_layer(...)
    @abstractmethod wait_for_save()                    # 同步等待
    def get_finished(finished_req_ids) -> ...          # 主动轮询
    # Scheduler side
    @abstractmethod get_num_new_matched_tokens(...)
    @abstractmethod update_state_after_alloc(...)
    @abstractmethod build_connector_meta(...)
    def request_finished(...)
```

HybridConnector 把所有同步等待 / 主动轮询接口都退化成空函数:

```python
# vllm/v1/hybrid_connector/__init__.py:1449-1465
def start_load_kv(self, forward_context, **kwargs) -> None:
    self._worker.start_load_kv(**kwargs)              # 触发后立即返回

def wait_for_layer_load(self, layer_name: str) -> None:
    return                                            # no-op

def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    self._worker.save_kv_layer(layer_name, kv_layer, **kwargs)  # 异步派发

def wait_for_save(self):
    return                                            # no-op
```

引擎在 forward 链路上**永远不会等任何 I/O**;save/load 完成由 backend 的 asyncio loop 通过 RPC 调用 `_on_save_done` / `_on_load_done` 反向推进 scheduler 状态。

对比 SGLang 的 `BaseKVSender`/`BaseKVReceiver`:

```python
# sglang/srt/disaggregation/base/conn.py:97-198
class BaseKVSender(ABC):
    @abstractmethod __init__(...)
    @abstractmethod init(num_kv_indices, aux_index)
    @abstractmethod send(kv_indices, state_indices)
    def pop_decode_prefix_len() -> int
    def should_send_kv_chunk(num_pages, last_chunk) -> bool
    @abstractmethod get_transfer_metric() -> KVTransferMetric
    @abstractmethod poll() -> KVPoll                  # 必须实现 poll
    @abstractmethod failure_exception()

class BaseKVReceiver(ABC):
    @abstractmethod __init__(...)
    @abstractmethod init(prefill_dp_rank)
    @abstractmethod send_metadata(...)
    @abstractmethod poll() -> KVPoll                  # 必须实现 poll
    @abstractmethod failure_exception()
    def clear()
    def abort()
```

每个 backend 必须实现一组 `poll() -> KVPoll` 接口,scheduler 则按 `KVPoll.{Bootstrapping, WaitingForInput, Transferring, Success, Failed}` 五状态做 inline 状态机切换。

## 3. SGLang 的 Scheduler 主循环 = 一个大轮询器

### 3.1 P 节点

```python
# sglang/srt/disaggregation/prefill.py:387
def event_loop_normal_disagg_prefill(self):
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        self.waiting_queue.extend(
            self.disagg_prefill_bootstrap_queue.pop_bootstrapped()  # ← poll
        )
        ...
        batch = self.get_next_disagg_prefill_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.on_idle()
        self.process_disagg_prefill_inflight_queue()                # ← poll
```

`pop_bootstrapped` 每次都 `poll_and_all_reduce_attn_cp_tp_group([req.disagg_kv_sender for req in self.queue], ...)`,把 `KVPoll.Bootstrapping/Failed/WaitingForInput` 三态摊到 scheduler 的每一个 step。

请求生命周期被切成多段队列:

```
incoming → PrefillBootstrapQueue → waiting_queue → run_batch
        →  disagg_prefill_inflight_queue → KVPoll.Success → release_kv_cache
```

`KVPoll.Success` 之前请求**必须一直挂在 inflight_queue**,scheduler 才能继续 poll 它的 `kv_sender`。

### 3.2 D 节点

```python
# sglang/srt/disaggregation/decode.py:1592, 1780
def event_loop_normal_disagg_decode(self):
    while True:
        ...
        self.process_decode_queue()           # 内部 pop_preallocated + pop_transferred
        batch = self.get_next_disagg_decode_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            ...

def process_decode_queue(self):
    ...
    self.polling_count = (self.polling_count + 1) % self.polling_interval
    if self.polling_count % self.polling_interval == 0:              # 节流!
        req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
        self.disagg_decode_transfer_queue.extend(req_conns)
        transferred_reqs = self.disagg_decode_transfer_queue.pop_transferred()
        ...
```

D 侧需要 3 段队列:`DecodePreallocQueue` → `DecodeTransferQueue` → `waiting_queue`,而且因为每轮都要做 `poll_and_all_reduce`,SGLang 不得不引入 `polling_interval` 节流。

### 3.3 失败处理也内联在 scheduler

```python
# sglang/srt/disaggregation/prefill.py:659-674
elif poll == KVPoll.Failed:
    error_message = f"Prefill transfer failed for request rank=..."
    try:
        req.disagg_kv_sender.failure_exception()
    except Exception as e:
        error_message += f" with exception {e}"
    logger.warning(error_message)
    req.time_stats.trace_ctx.abort(...)
    release_kv_cache(req, self.tree_cache)
    prepare_abort(req, error_message, ...)
    done_reqs.append(req)
```

Abort、KV 释放、output stream 全部在 scheduler 主线程内联完成。

## 4. HybridConnector:把 KV 传输完全踢出引擎主路径

### 4.1 双 asyncio loop 架构

```python
# vllm/v1/hybrid_connector/__init__.py:1355
class HybridConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        if role == KVConnectorRole.SCHEDULER:
            self._sched = HybridScheduler(vllm_config, kv_cache_config)
        else:
            self._worker = HybridWorker(vllm_config, kv_cache_config)
```

- `HybridScheduler.loop = get_hybrid_sched_loop()`:运行在 engine core 进程,但跑在独立的 asyncio 线程上(`__init__.py:586`)。
- `HybridWorker.loop = get_hybrid_worker_loop()`:运行在 worker 进程,同样独立 asyncio 线程(`__init__.py:1162`)。

引擎主线程从不触碰这两个 loop,只通过 deque 和 RPC 与之交互。

### 4.2 状态推进通过 RPC 反向推送

P 节点完成保存:

```python
# vllm/v1/hybrid_connector/__init__.py:1331-1351
async def _on_req_saved(self, reqid, layer_name):
    state = self._saving.get(reqid, None)
    ...
    if len(state._ready_layers) != self._num_layers:
        return
    savereq = IoDoneReqs(worker_tprank=tprank, reqids=[IoRet(reqid=reqid)])
    await self._io_done_rpc(savereq, _SAVE_DONE_REQ, _SAVE_DONE_RESP)
    # ↑ Worker 主动 RPC 通知 Scheduler,而不是 Scheduler 来 poll Worker
```

Scheduler 端被 RPC 唤醒:

```python
# vllm/v1/hybrid_connector/__init__.py:937-955
async def _do_save_done(self, worker_tprank, ioret):
    state = try_advance(self._saving, ioret, worker_tprank, tpsize)
    if state is None:
        return
    ...
    _q_append(self._saved, ioret.reqid)
    await self._cleanup(state._req)
```

之后 engine core 一次普通 step 调用到 `connector.step()`,`HybridScheduler.step` 把 `self._saved` 里的请求转化成 `EngineCoreOutput` 推回给客户端。

### 4.3 Backend 接口同样精简

每个具体后端(KVT、Mooncake、V6dObject、File、KVS、Migration、组合后端等)只需实现:

```python
# vllm/v1/hybrid_connector/__init__.py:111
class HybridBackend:
    def get_operations(self, req) -> tuple[int, int]                          # 这个请求要 load/save 多少次?
    async def async_get_num_new_matched_tokens(self, req, num_computed_tokens) -> int
    async def async_update_state_after_alloc(self, req, blocks, num_external_tokens)
    async def async_load_kv(self, m: BackendMeta) -> AsyncGenerator[IoRet, None]
    def async_save_kv_layer(self, layer_name, kv_layer, m) -> Optional[AsyncGenerator]
    def build_backend_meta(self, sout) -> BackendMeta
    async def async_cleanup(self, req)
```

工厂注册的 backend 包括:

```python
# vllm/v1/hybrid_connector/__init__.py:480
def _get_backend_cls(cfg):
    backend = cfg.kv_transfer_config.get_from_extra_config("backend", None)
    if backend == "migration":           return MigrationBackend
    if backend in ("kvt+migration", ...): return KVTMigration
    if backend in ("kvt+kvs", ...):      return KVSP
    if backend == "local_file":          return FileBackend
    if backend == "kvt":
        if is_consumer and is_producer:  return FilePBackend
        if is_consumer:                  return DBackend
        if is_producer:                  return PBackend
    if backend == "kvs":                 return VineyardKVSBackend
    if backend == "mooncake":            return MooncakeKVSBackend
    if backend == "v6d_object":          return V6dObjectBackend
    if backend in ("v6d_object+kvt",...):return V6dObjectKVTBackend
```

要做"KVT + Vineyard 持久化"或"KVT + 跨集群迁移"这类多级缓存方案,只需要一个组合 backend 即可,不必动 connector 主体,也不会污染引擎。

## 5. 关键创新:Block 引用计数解耦请求生命周期与 KV 传输

这是 HybridConnector 最重要的一项设计。

### 5.1 用 vLLM 现成的 ref_cnt 把 KV block "钉"住

```python
# vllm/v1/hybrid_connector/engine_proxy.py:139-143
def sched_acquire_blocks(blks: KVCacheBlocks):
    for blk in blks.blocks:
        for block in blk:
            block.ref_cnt += 1                          # 复用 KV cache manager 的引用计数

# engine_proxy.py:130-136
def sched_free_blocks(blks: KVCacheBlocks):
    self = _sched()
    for blk in blks.blocks:
        blk = reversed(blk)
        self.kv_cache_manager.block_pool.free_blocks(blk)  # 计数归零自动回收
```

### 5.2 在请求结束之前先 acquire,传输完成再 free

```python
# vllm/v1/hybrid_connector/__init__.py:728-748
def _setup_save(self, req, kvblks, save_count=1):
    assert not has_setup_save(req)
    _inc_cleanup_rc(req)
    self._saving[req.request_id] = _SavingReq(req, kvblks, save_count)
    sched_acquire_blocks(kvblks)                        # 提前增加引用计数
    set_param(req, _SAVE_PREPARED, True)

def _try_teardown_save(self, reqid):
    state = self._saving.pop(reqid, None)
    if state is None: return
    sched_free_blocks(state.kvblks)                     # 真正传完才释放
```

### 5.3 请求结束钩子可以提前吐 token

```python
# vllm/v1/hybrid_connector/__init__.py:1466-1500
def request_finished_all_groups(self, request, block_ids):
    ...
    if get_param(request, _PD_SAVED):
        # mark_saved 已经触发,直接告诉客户端 kv_transfer 完成
        return (False, {"kv_transfer_done": True})
    # 否则记下 finish 信息,等 _step_saved 之后再发 done 信号
    set_param(request, _PD_FINISH_REASON, finish_reason)
    set_param(request, _PD_STOP_REASON, request.stop_reason)
    set_param(request, _PD_CLIENT_INDEX, request.client_index)
    return (False, {"kv_transfer_pending": True})
```

返回 `delay_free_blocks=False`,scheduler 该释放就释放;但因为 `sched_acquire_blocks` 早已把 ref_cnt 抬起来,block 不会真的被回收,直到 `_try_teardown_save` 调用 `sched_free_blocks` 才进 free list。

**含义**:请求 R 已经吐完所有 token、客户端连接已关闭、Request 对象已 finished — 但其 KV block 仍然在显存里安全等待最后一层传输完成。SGLang 要做到这一点必须把请求一直挂在 `disagg_prefill_inflight_queue`,代价是 scheduler 主循环要持续 poll。

## 6. HybridConnector + KVTBackend 的具体收益

1. **极简且稳定的引擎接口** — 引擎从不需要"等"或"轮询"任何 KV I/O,符合"如无必要,勿增接口"。

2. **请求生命周期与 KV 传输解耦(ref_cnt 复用)** — token 已吐给客户端,KV block 仍能因 ref_cnt > 0 安全留在显存继续传完。SGLang 用 `disagg_prefill_inflight_queue` 撑请求生命周期才能做到同样的事,但代价是 scheduler 要一直 poll。

3. **不在 scheduler 主路径上 poll** — `HybridScheduler.loop` 是独立 asyncio 线程,加上 ZMQ XPUB/SUB bypass(`__init__.py:660-685`,`1240-1253`),backend 在引擎完全空闲时也能继续推进 I/O,无需 dummy step。

4. **Backend 组合非常便宜** — `KVTMigration`(KVT + 迁移)、`KVSP`(KVT + KVS 多级)、`V6dObjectKVTBackend`(KVT + Vineyard 持久化)都是几十到上百行胶水代码组合出的多级缓存方案。SGLang 要走同样的路要在 conn.py 里硬塞 mooncake-staging-buffer 那种侵入式实现(`common/staging_handler.py` 840 行)。

5. **故障隔离更干净** — backend 异常通过 `IoRet.ex` 在 disagg 线程冒泡,经 `mark_loaded → on_abort_req → _aborting` deque 反向通知引擎(`__init__.py:823-829`,`903-909`),不会污染 forward 线程。

6. **bypass substep 支持原生异步路径** — `HCSchedOutput.hc_stepid/hc_substepid` + `send_bypass_task` 让 HybridScheduler 在不需要触发真正 forward 的情况下也能向 worker 推进度元数据(`__init__.py:1040-1086`,`1240-1253`),适合 PD 之间纯 KV 操作。

## 7. 总结一句话

SGLang 的 PD 是把传输状态机(`KVPoll`)折叠进 scheduler 主循环,结果是抽象类大、queue 多、release/abort 内联;HybridConnector 把传输完全推到独立的 asyncio + RPC 子系统,引擎只剩 `start_load_kv` / `save_kv_layer` 两个 hook,再借 vLLM 自身的 block ref_cnt 把"请求结束"和"KV 传完"两个生命周期解耦 — 这正是"内核 vs 驱动"设计哲学的工程化落地,代码量和侵入面都比 SGLang 少了一个数量级。
