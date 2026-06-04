---
title: vLLM HybridConnector / KVT：启动机制与 Bypass-Substep 流水线
date: 2026-06-04
tags: []
---

# vLLM HybridConnector / KVT：启动机制与 Bypass-Substep 流水线

> 基于 `/mnt/data/llx/vllm`(分支带 HybridConnector)当前源码整理(2026-06)。
> 涉及文件：`vllm/v1/engine/core.py`、`vllm/v1/hybrid_connector/__init__.py`、
> `vllm/v1/hybrid_connector/engine_proxy.py`、`vllm/v1/hybrid_connector/kvtbackend.py`、
> `vllm/distributed/kv_transfer/kv_transfer_state.py`。
>
> 关联：跨机 TP、主/副 step 在 Ray 下被误判的线程 bug、以及 KV 发完后 decode 如何得知，
> 见 `vllm_kvt_cross_node_tp_and_completion.md`。其中第 3 节正是本文「主 step `substepid=0`
> vs substep `substepid≥1`」这个判据在 Ray compiled-graph 后台线程下被破坏的实战案例。

---

## 0. TL;DR / 核心结论

- **KVT 不是独立守护进程**，而是作为 vLLM 的 KV connector 插件 `HybridConnector`，跟随引擎进程/worker 进程在固定钩子点自动起线程、建监听、实例化底层 `bladekv` 传输客户端。触发开关是 `--kv-transfer-config`。
- **step 与 substep** 用一对 `(stepid, substepid)` 区分：主 step `substepid=0`（真正跑 GPU、产 token），substep `substepid≥1`（不产 token，只携带 KV 传输任务，通过 zmq PUB 下发给 worker）。
- **substep 在 EngineCore 等主 step GPU 结果的空窗期被合成**，靠给 `SchedulerOutput` 动态挂 `hc_parent` 属性标记归属。循环无固定 tick：有传输工作就连轴转、没活就阻塞，且只有 `reqs` 非空的 substep 才真正发出去。
- **bypass/substep 优化的是"控制面时延"（step 粒度 → sub-step 粒度）**；真正的 **layer 粒度数据传输** 由 worker 的 `save_kv_layer` 逐层回调实现，二者配合形成更细的重叠流水线。
- **`kvconn.step()`** 是 connector 在 EngineCore 线程上的"控制面状态机泵"，推进 save/load/abort 生命周期并产出 `kv_transfer_done` 等输出。
- **`_wait_model_output_future`** 本职是"阻塞等异步 GPU future 出结果"；`EngineCoreProc` 把它 override 成"等的同时跑 bypass 循环"。

---

## 1. KVT 服务如何在 vLLM 启动时自动拉起

KVT 不需要手动 `start` 一个独立进程，而是作为 vLLM 的 **KV connector 插件（`HybridConnector`）**，跟随 vLLM 的引擎进程/worker 进程一起被自动初始化、自动起线程、自动建监听服务。整个触发点是启动 vLLM 时传入的 `--kv-transfer-config`。

### 1.1 触发开关：`kv_transfer_config`

只要 `VllmConfig.kv_transfer_config` 非空且 `kv_connector="HybridConnector"`，整条 KVT 链路就会被激活。connector 名字在工厂里注册：

```python
# vllm/distributed/kv_transfer/kv_connector/factory.py:141
KVConnectorFactory.register_connector(
    "HybridConnector", "vllm.v1.hybrid_connector", "HybridConnector"
)
```

具体走哪个后端（kvt / local_file / v6d_object 等）由 extra_config 里的 `backend` 字段决定，在 `_get_backend_cls` 里分发（`backend="kvt"` 且是 producer→`PBackend`、consumer→`DBackend`）：

```python
# vllm/v1/hybrid_connector/__init__.py:490
elif backend == "kvt":
    if cfg.kv_transfer_config.is_kv_producer and cfg.kv_transfer_config.is_kv_consumer:
        from .filekvtbackend import FilePBackend
        return FilePBackend
    elif cfg.kv_transfer_config.is_kv_consumer:
        from .kvtbackend import DBackend      # decode 侧
        return DBackend
    elif cfg.kv_transfer_config.is_kv_producer:
        from .kvtbackend import PBackend      # prefill 侧
        return PBackend
```

### 1.2 Engine Core 进程侧：起调度线程 + RPC 监听服务

EngineCore 进程构造时（`EngineCoreProc.__init__`）会调用 `_core_init`：

```python
# vllm/v1/engine/core.py:1237
def _core_init(core, vllm_config: VllmConfig):
    if vllm_config.kv_transfer_config is None:
        return
    from vllm.v1.hybrid_connector.engine_proxy import core_init
    core_init(core, vllm_config)
```

`core_init` 做两件关键的"起服务"动作：①起一个名为 `hybridsched` 的 uvloop asyncio 后台线程；②起一个调度侧 RPC server（`schedrpcserver`）监听端口：

```python
# vllm/v1/hybrid_connector/engine_proxy.py:379
def core_init(core: "EngineCoreProc", cfg: VllmConfig):
    global _g_core, _g_sched_loop, _g_sched_rpc_serv
    _g_core = core
    _g_sched_loop = start_asyncio_thread("hybridsched")        # ① asyncio 线程
    port = sched_rpc_server_port(cfg)
    _g_sched_rpc_serv = RpcServer(port, "schedrpcserver")
    _g_sched_rpc_serv.start(_g_sched_loop)                     # ② TCP 监听 (0.0.0.0:port)
```

`RpcServer.start` 内部就是 `asyncio.start_server(..., "0.0.0.0", port)`，这就是 KVT 在调度侧真正对外监听的"服务"（接受 worker 注册、bypass handle、save/load-done 等 RPC）。

随后 Scheduler 通过工厂创建 role=SCHEDULER 的 `HybridConnector` → `HybridScheduler` → 调度侧 backend（`DBackend`/`PBackend`），在这里启动 `PeerManager`、向 naming 服务注册、注册 worker 注册回调等。

### 1.3 Worker 进程侧：起 worker 线程 + 创建 connector

每个 worker 在初始化 KV transfer 时调用 `ensure_kv_transfer_initialized`，它先 `worker_init` 起一个 `hybridworker` asyncio 线程，再用工厂创建 role=WORKER 的 connector：

```python
# vllm/distributed/kv_transfer/kv_transfer_state.py:65
if vllm_config.kv_transfer_config.is_kv_transfer_instance and _KV_CONNECTOR_AGENT is None:
    from vllm.v1.hybrid_connector.engine_proxy import worker_init
    worker_init(vllm_config, local_rank)                       # 起 hybridworker 线程
    _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
        config=vllm_config, role=KVConnectorRole.WORKER, kv_cache_config=kv_cache_config,
    )
```

### 1.4 真正的传输服务端点：`register_kv_caches` 时创建 `KVTransferClient`

KV cache 显存分配完成后，GPU model runner 会调用 `kv_transfer_group.register_kv_caches(kv_caches)`（`vllm/v1/worker/gpu_model_runner.py:8122`）。这一路最终落到 backend 的 `register_kv_caches`，创建底层 KVT 传输引擎 `bladekv.KVTransferClient`——它注册显存、建立 RDMA/TCP 传输端点、并把 worker 信息注册到 naming，这才是真正承载 KV 搬运的"KVT 服务"实例：

```python
# vllm/v1/hybrid_connector/kvtbackend.py:1792
self._bladkv_cli = bladekv.KVTransferClient(
    self._inst_id, self._cfg.parallel_config.tensor_parallel_size,
    worker_id=worker_id, worker_tp_rank=rank,
    block_bytes=block_bytes, token_bytes=token_bytes,
    naming_url=self._naming_url, layers=_flatten_cache(kv_caches),
    protocols=[protocol], num_kv_heads=total_num_kv_heads, **_hybrid_kwargs,
)
```

### 1.5 启动链路总结

```
vllm serve --kv-transfer-config '{"kv_connector":"HybridConnector","kv_role":"kv_producer/consumer",
            "kv_connector_extra_config":{"backend":"kvt","naming_url":...}}'
        │
        ├─ EngineCoreProc.__init__
        │     └─ _core_init → core_init()            ← 起 hybridsched 线程 + schedrpcserver 监听端口
        │     └─ Scheduler 创建 HybridConnector(SCHEDULER)
        │           └─ HybridScheduler → DBackend/PBackend(SCHEDULER)  ← PeerManager / naming / worker 注册回调
        │
        └─ Worker 进程: ensure_kv_transfer_initialized
              └─ worker_init()                        ← 起 hybridworker 线程
              └─ HybridConnector(WORKER) → HybridWorker → PBackend/DBackend(WORKER)
                    └─ register_kv_caches()           ← 创建 bladekv.KVTransferClient（真正的 KVT 传输服务端点）
```

> "自动启动"的本质：配置了 `kv_transfer_config` 后，vLLM 在引擎进程和 worker 进程初始化的固定钩子点（`_core_init` / `ensure_kv_transfer_initialized` / `register_kv_caches`）里，自动起后台 asyncio 线程、建 RPC 监听 server、并在显存就绪后实例化底层 `bladekv` 传输客户端，全程无需单独拉起一个 KVT 守护进程。外部依赖只有 `naming_url` 指向的命名服务（若用 `fake://` 则连命名服务都不需要，走本地直连）。

---

## 2. step 与 substep：如何区分、如何提交

二者用一对 `(stepid, substepid)` 来区分，在调度侧 `HybridScheduler` 里维护：

```python
# vllm/v1/hybrid_connector/__init__.py:579
### R/W: core thread - stepid/substepid 管理
self._stepid: int = 1024     # stepid 起始值为 1024，每次新 step 递增
self._substepid: int = 1     # substepid 在每次新 step 时重置为 1，bypass substep 从 1 开始
```

区分逻辑全在 `build_connector_meta` 里，靠 `SchedulerOutput` 上是否挂了 `hc_parent` 这个动态属性来判断：

```python
# vllm/v1/hybrid_connector/__init__.py:1026
if hc_parent is None:
    # 主 step：分配新的 hc_stepid，hc_substepid 为 0
    stepid = self._stepid
    substepid = 0
    self._stepid += 1
    self._substepid = 1          # 重置 substepid
else:
    # bypass substep：从 hc_parent 的 kv_connector_metadata 中取出 hc_stepid
    ...
    parent_meta = hc_parent.kv_connector_metadata
    stepid = parent_meta.stepid  # 复用父 step 的 stepid
    substepid = self._substepid
    self._substepid += 1
```

- **主 step**：`hc_parent is None` → `substepid=0`，`stepid` 自增。它走的是正常引擎路径，真正跑 GPU、产 token，通过 `scheduler.update_from_output(...)` 提交、把 `EngineCoreOutput` 写回 client。
- **substep**：`hc_parent` 指向它所属的主 step 的 `SchedulerOutput`，`stepid` 复用父 step 的，`substepid` 从 1 递增。substep **不产 token、不更新 scheduler 状态**，它唯一的载荷是 KV 传输任务 `reqs`（`BackendMeta`）。

### 2.1 "提交"的两种含义

| | 主 step | substep |
|---|---|---|
| 是否跑 GPU | 是 | 否 |
| 是否产 token | 是 | 否 |
| `substepid` | 0 | 从 1 递增 |
| "提交" 方式 | `update_from_output` + 输出队列回包给 client | 把 `HybridMetadata` 通过 zmq PUB 发给 worker，worker 收到后 `_do_bypass_meta` 触发 layer 级 load/send |
| token 级 commit | 有 | 无 |

`(stepid, substepid)` 这对值会随 meta 一起带到 worker / 底层 `bladekv`，作为传输任务的排序/归属键——worker 据此知道某个 substep 属于哪个 step、在 step 内的先后顺序。

---

## 3. substep 在 vLLM 中何时生成（bypass 循环）

substep 是 EngineCore 在**等主 step 的 GPU 结果时**合成的。入口在 `_wait_model_output_future`：当 `_enable_bypass` 时，它把"等 GPU future"丢到线程池，自己进入一个循环：

```python
# vllm/v1/engine/core.py:2022 (EngineCoreProc._wait_model_output_future 内)
# process kvconn step
step_outputs = kvconn.step()
...
# process kvconn tasks
substep_sout: SchedulerOutput = SchedulerOutput.make_empty()
# 利用 Python 动态性设置 hc_parent，不修改 SchedulerOutput 的定义
# hc_parent 表明当前 substep_sout 所属的 step_sout
substep_sout.hc_parent = step_sout            # type: ignore[attr-defined]
meta = kvconn.build_connector_meta(substep_sout)
send_bypass_task(meta)
```

每跑一圈循环就：①推进 KV connector 状态机 `kvconn.step()`；②造一个**空的** `SchedulerOutput`（`make_empty()`），挂上 `hc_parent=step_sout` 标记成 substep；③`build_connector_meta` 给它分配 `(stepid, substepid)` 并打包当前 ready 的传输任务；④`send_bypass_task` 发给 worker。

关键：`send_bypass_task` 只有在 `reqs` 非空时才真的通过 zmq 发：

```python
# vllm/v1/hybrid_connector/__init__.py:658
def send_bypass_task(self, kv_connector_metadata: HybridMetadata):
    if kv_connector_metadata.reqs:                       # 只有非空才发
        asyncio.run_coroutine_threadsafe(
            self._send_bypass_task(kv_connector_metadata), self.loop)
```

发送通道是调度侧在 `_start_bypass_rpc_server` 建的 zmq `XPUB` socket，worker 在 `start_bypass_task_loop` 里 `SUB` 订阅，每收到一条就 `pickle.loads` 出 `HybridMetadata` 交给 `_do_bypass_meta`。所以"通过 zmq 传到 hybrid worker"指的就是这些非空 substep。

### 3.1 循环多久一圈？等待期间会合成多少 substep？

**没有固定定时器/间隔**，它是纯事件+工作驱动的：

```python
# vllm/v1/engine/core.py:1992
while True:
    if not kvconn.has_requests():
        item = self.input_queue.get()        # blocking op：没活就阻塞
        ...
    while not self.input_queue.empty():       # 有活则非阻塞地排空输入队列
        item = self.input_queue.get_nowait()
        ...
    # 然后 kvconn.step() + 合成 substep + send_bypass_task
```

- **当 `kvconn.has_requests()` 为 False**（没有待处理传输工作）：循环阻塞在 `input_queue.get()`，**不空转**，直到 GPU 结果 `EXECUTE_MODEL_RESULT` 到达或来了新 client 请求才醒。基本不合成 substep。
- **当 `has_requests()` 为 True**（有传输工作排队）：循环**不阻塞**，以 CPU 能跑多快就多快地一圈圈转，每圈合成一个 substep。循环体里**没有 sleep**，每圈成本是 `kvconn.step()` + `build_connector_meta`，约微秒~亚毫秒级。

`has_requests()` 检查的就是那些会被 `kvconn.step()` 排空的队列：

```python
# vllm/v1/hybrid_connector/__init__.py:1067
def has_requests(self) -> bool:
    if self._waiting: return True
    ...
    return bool(self._prepared)
```

所以"等待期间合成多少 substep"≈ `主step的GPU时长 / 单圈CPU成本`，上界可能很大（decode step GPU 约几~几十 ms，prefill 更长，单圈很便宜，理论上可达数百圈）。但其中**大部分圈 `reqs` 为空、是 no-op，不发 zmq**；**实际发出的 substep 数 = 这段 GPU 窗口里新变 ready 的传输任务批次数**，由传输工作到达节奏决定，而不是固定 tick。队列被排空后 `has_requests()` 转 False，循环又回到阻塞态——不是无脑 busy-spin，而是"有活连轴转、没活就睡"。

---

## 4. 与 layer 粒度传输的关系

要把**两件正交的事**分开看：

### 4.1 substep/bypass：解决"控制面时延"（step 粒度 → sub-step 粒度）

正常情况下，调度器一个 engine step 才 `build_connector_meta` 一次，即每个 step 只能下发一批传输任务。但在 PD 场景里，主 step 的 GPU 正在跑的这几~几十 ms 内，可能不断有新信息到达（D 侧握手/拉取请求、某些 req 的传输条件刚刚满足等）。如果只能等 GPU 跑完 + 下一次 schedule 才处理，就被"卡"在 step 粒度。bypass 循环把 EngineCore 在等 GPU 时本来闲置的 CPU 利用起来，**在一个 step 内多次下发传输任务**，把控制面反应粒度从"每 step 一次"细化到"sub-step 级"。这正是"prefill 在算、decode 的 info 还没来得及汇合"那种时序错位的缓解手段。

### 4.2 layer 粒度数据传输：由 worker 逐层 hook 完成

KV 的逐层发送发生在 `save_kv_layer`——模型每算完一层就回调一次，于是早算完的层可以先发，不必等整个 step 所有层算完：

```python
# vllm/v1/hybrid_connector/__init__.py:1268
def save_kv_layer(self, layer_name, kv_layer, **kwargs) -> None:
    if self._meta is None:
        return
    save_coro = self._backend.async_save_kv_layer(layer_name, kv_layer, self._meta.reqs)
    ...
```

（在 hybrid 模型里还会按 attn pack 聚合，到 pack 的最后一层才触发该 pack 的发送，即 `hybrid_model_send_layer` 的逻辑。）

### 4.3 二者的配合

layer 粒度传输要能"早发"，前提是 worker 已经拿到了"这个 req 要发给谁、发哪些"的 meta；bypass/substep 的作用就是让这份 meta 在 GPU 还在算的过程中就尽早通过 zmq 送到 worker，从而让 `save_kv_layer` 的逐层发送能在本 step 内尽快起跑，而不是等 step 结束后才开始。

- 能否做到 **layer 粒度传输**？能，但那是 `save_kv_layer` 逐层 hook 实现的，不是 substep 本身。
- substep 的价值是把**控制面下发**从 step 粒度细化到 sub-step 粒度，让 layer 粒度的数据搬运"对得上、起得早"。二者配合才把原来的 step 粒度流水线打散成更细的重叠流水线。

---

## 5. `kvconn.step()` 做什么

`kvconn` 是 `HybridConnector`，`step()` 最终走到 `HybridScheduler.step()`。它是 KV connector 在 **EngineCore 线程上的"控制面状态机泵"**——每调一次，就把所有在途请求沿 save/load/abort 生命周期推进一格，并产出要回给 client 的输出：

```python
# vllm/v1/hybrid_connector/__init__.py:669
def step(self) -> Optional[dict[int, list[EngineCoreOutput]]]:
    from vllm.v1.engine.core import combine_outputs
    kvt_done = self._step_saved()
    self._step_waiting()
    self._step_loaded()
    return combine_outputs(kvt_done, self._step_aborting())
```

四个子步骤（PD 语境下：load=decode 侧拉取，save=prefill 侧推送）：

- **`_step_saved()`**：处理 KV **保存/发送完成**的请求。若 finish 已到就发出 `kv_transfer_done` 的 `EngineCoreOutput`，否则只打标记；并 `_try_teardown_save` 释放占用的 block。是唯一会产出 client 输出的子步骤。
- **`_step_waiting()`**：把新加入、还在等显存的 PD 请求取出，调 `sched_allocate_slots` 分配 KV cache block；成功后给 prefill 侧 `_setup_save`、给 decode 侧异步 `_on_add_req` 发起 load（`num_external_tokens==0` 走 fastpath 直接进 `_loaded`）；分配不到就 break 等下次。

```python
# vllm/v1/hybrid_connector/__init__.py:735
def _step_waiting(self):
    while self._waiting:
        # In pd disagg, load means decode, save means prefill
        req, load_count, save_count = self._waiting[0]
        ...
        kvblks = sched_allocate_slots(req, load_count > 0, save_count > 0, prealloc, gamma)
        if kvblks is None:
            break
        ...
```

- **`_step_loaded()`**：处理 KV **load 完成**的请求，使其对 scheduler 可见（这样主 step 才能真正调度它们去算）。
- **`_step_aborting()`**：处理 abort。

> 一句话：`kvconn.step()` 不跑 GPU、不产 token，它只是把传输请求从"等待 → 分配/发起 → 完成"逐格推进，并把诸如 `kv_transfer_done` 这类传输事件包成 `EngineCoreOutput` 返回。它在两处被调用——正常的 `EngineCore.step()` 开头调一次（`core.py:645`），以及 bypass 循环里每圈调一次。

---

## 6. `_wait_model_output_future` 的本职作用

### 6.1 基类版本（没开 PD/bypass）

基类 `EngineCore` 里这个方法是**极简版**，就是它"本来"的样子：

```python
# vllm/v1/engine/core.py:618
def _wait_model_output_future(self, model_output_future, step_sout):
    return model_output_future.result(), {}
```

本职非常单纯：**阻塞等 GPU 算完，拿回 `ModelRunnerOutput`**，第二个返回值（KV connector 的输出）恒为空 `{}`。

### 6.2 为什么是 future？—— v1 的异步 executor

vLLM v1 的 executor 是异步/流水线式的——`execute_model(scheduler_output, non_block=True)` 不原地跑完模型，而是把前向计算提交给 worker（可能是别的进程），立刻返回一个 `Future` 当句柄：

```python
# vllm/v1/engine/core.py:672
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
with self.log_error_detail(scheduler_output):
    model_output, kvconn_outputs2 = self._wait_model_output_future(future, step_sout=scheduler_output)
    if model_output is None:                 # forward/sample 两阶段：forward 先返回 None
        sample_future = self.model_executor.sample_tokens(grammar_output, non_block=True)
        model_output, kvconn_outputs3 = self._wait_model_output_future(sample_future, step_sout=scheduler_output)
```

`_wait_model_output_future` 就是把这个 future `.result()` 取出来。注意 v1 把 "forward" 和 "sample_tokens" 拆成两阶段，forward future 先返回 `None`，再 `sample_tokens(non_block=True)` 拿第二个 future 又 wait 一次，所以普通流程里它也会被调用两次。

### 6.3 为什么抽成单独方法 —— 留扩展点

它被单独抽成方法，正是为了留一个**可重写的扩展点**：

| 场景 | `_wait_model_output_future` 行为 | 第二个返回值 |
|---|---|---|
| 无 PD（基类 `EngineCore`） | 纯 `future.result()`，阻塞等 GPU | `{}` 空 |
| 有 PD/bypass（`EngineCoreProc` override，`core.py:1942`） | 等 GPU 的同时跑 KV connector 的 substep 循环 | 期间 `kvconn.step()` 的传输事件输出 |

`EngineCoreProc` 的 override 把"等 GPU 的这段空窗期"利用起来——CPU 本来闲着等结果，于是在循环里反复跑 `kvconn.step()` + 合成 substep 下发传输任务，直到 `EXECUTE_MODEL_RESULT` 到达才返回。

---

## 7. 关键名词速查

| 名词 | 含义 | 位置 |
|---|---|---|
| `HybridConnector` | KV connector 插件入口，按 role 分 SCHEDULER/WORKER | `__init__.py:1333` |
| `HybridScheduler` | 调度侧状态机，维护 `_waiting/_loading/_loaded/_saving/_saved` 等 deque | `__init__.py:558` |
| `HybridWorker` | worker 侧，持有底层 backend，订阅 bypass、逐层 save | `__init__.py:1135` |
| `PBackend` / `DBackend` | prefill / decode 侧 KVT backend | `kvtbackend.py:679 / 1948` |
| `bladekv.KVTransferClient` | 底层传输引擎（RDMA/TCP），真正的"KVT 服务"实例 | `kvtbackend.py:1792` |
| `stepid` / `substepid` | step（≥1024 自增，substepid=0）与 substep（substepid≥1）的归属/排序键 | `__init__.py:579` |
| `hc_parent` | 动态挂在 `SchedulerOutput` 上，指向 substep 所属的主 step | `core.py:2041` |
| `send_bypass_task` | 调度侧通过 zmq XPUB 把非空 substep meta 发给 worker | `__init__.py:658` |
| `_wait_model_output_future` | 等异步 GPU future；override 后兼跑 bypass 循环 | `core.py:618 / 1942` |

---

## 附：关于注释里的"Python 动态性"

`core.py:2041` 那行注释"利用 Python 动态性"指的是：**利用 Python 可在运行时给任意对象动态新增属性这一特性**，给 `SchedulerOutput` 实例临时挂一个类定义里根本不存在的 `hc_parent` 字段，而不修改 `SchedulerOutput` 这个上游类本身。

```python
substep_sout.hc_parent = step_sout  # type: ignore[attr-defined]
```

- `SchedulerOutput` 是 vLLM 上游定义的类，字段里没有 `hc_parent`。Python 是动态语言，普通对象属性存在实例的 `__dict__` 里，运行时可随时新增，所以这行会在该实例上凭空创建 `hc_parent`。
- 读取方因此用容错方式取（`dict(sout.__dict__).get("hc_parent")`，见 `__init__.py:1024`），避免普通主 step 上没有该属性时 `AttributeError`。
- 末尾 `# type: ignore[attr-defined]` 是为了让 mypy 闭嘴——类型检查器按类定义认为没有该属性。
- **设计意图**：避免侵入式修改上游类（加字段会污染上游、增加 rebase 冲突、影响序列化）。`hc_parent` 只在 EngineCore 进程内、调度线程里使用，不跨进程，所以安全。
