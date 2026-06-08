---
title: vLLM Develop 分支 PD 分离：Scheduler-Worker 消息传递与 Bypass 路径深度分析
date: 2026-06-08
tags: []
---

# vLLM Develop 分支 PD 分离：Scheduler-Worker 消息传递与 Bypass 路径深度分析

> 基于 vLLM `develop` 分支代码阅读整理
> 日期：2026-06-08

---

## 目录

1. [架构总览：两条传输路径](#1-架构总览两条传输路径)
2. [Path A：主路径 — SchedulerOutput + 共享内存](#2-path-a主路径--scheduleroutput--共享内存)
   - 2.1 [Scheduler 侧构建 metadata](#21-scheduler-侧构建-metadata)
   - 2.2 [关键数据结构](#22-关键数据结构)
   - 2.3 [物理传输：pickle + 共享内存](#23-物理传输pickle--共享内存)
   - 2.4 [Worker 侧绑定 metadata](#24-worker-侧绑定-metadata)
3. [Path B：Bypass 路径 — 独立 ZMQ 通道](#3-path-b-bypass-路径--独立-zmq-通道)
   - 3.1 [Engine Core 的 Bypass 循环](#31-engine-core-的-bypass-循环)
   - 3.2 [Bypass 消息分发](#32-bypass-消息分发)
   - 3.3 [Worker Bypass 线程接收](#33-worker-bypass-线程接收)
4. [两个线程，两种角色：bind_backend_metadata 详解](#4-两个线程两种角色bind_backend_metadata-详解)
   - 4.1 [主线程路径](#41-主线程路径)
   - 4.2 [Bypass 线程路径](#42-bypass-线程路径)
   - 4.3 [clear_backend_metadata 的线程区分](#43-clear_backend_metadata-的线程区分)
5. [Step 时间线与 blade-kvt 协调](#5-step-时间线与-blade-kvt-协调)
   - 5.1 [主线程的 Step 生命周期](#51-主线程的-step-生命周期)
   - 5.2 [start_send_step vs start_send_substep](#52-start_send_step-vs-start_send_substep)
   - 5.3 [blade-kvt C++ 的三路时序处理](#53-blade-kvt-c-的三路时序处理)
   - 5.4 [为什么 substep 的 stepid 会大于主线程的 coord_step_id](#54-为什么-substep-的-stepid-会大于主线程的-coord_step_id)
6. [Chunked Prefill 的多步传输与 _infly_kvt 状态管理](#6-chunked-prefill-的多步传输与-_infly_kvt-状态管理)
   - 6.1 [具体示例：8192 token 分两个 chunk](#61-具体示例8192-token-分两个-chunk)
   - 6.2 [Step N：首次调度 (has_last_token=False)](#62-step-n首次调度-has_last_tokenfalse)
   - 6.3 [Step N+1：继续调度 (has_last_token=True)](#63-step-n1继续调度-has_last_tokentrue)
   - 6.4 [中途 Abort 的处理](#64-中途-abort-的处理)
7. [为什么 Bypass 线程不能处理 has_last_token=False](#7-为什么-bypass-线程不能处理-has_last_tokenfalse)
   - 7.1 [状态依赖链](#71-状态依赖链)
   - 7.2 [竞态风险分析](#72-竞态风险分析)
8. [与社区版 (NIXL) 的关键差异](#8-与社区版-nixl-的关键差异)
9. [完成通知：Worker -> Scheduler (TCP RPC)](#9-完成通知worker---scheduler-tcp-rpc)

---

## 1. 架构总览：两条传输路径

Develop 分支的 `HybridConnector` 有**两条独立的 metadata 传输路径**，实现 KV 传输与模型计算的流水线重叠：

```
                    Scheduler 进程
                         |
          +--------------+--------------+
          |                             |
     Path A: 主路径              Path B: Bypass路径
     (SchedulerOutput)           (ZMQ PUB/SUB)
          |                             |
     pickle + SHM                 pickle + ZMQ
          |                             |
          v                             v
                    Worker 进程
          |                             |
  bind_connector_                 _do_bypass_meta()
  metadata()                            |
          |                             |
          +-------------+-------------+
                        |
                bind_backend_metadata()
                        |
                  blade-kvt 发起
                  RDMA KV 传输
```

### 关键文件

| 文件 | 角色 |
|------|------|
| `vllm/v1/hybrid_connector/__init__.py` (1508行) | HybridScheduler, HybridWorker, HybridConnector |
| `vllm/v1/hybrid_connector/kvtbackend.py` (2578行) | PBackend, DBackend, blade-kvt 集成 |
| `blade_kvt/kv_transfer_impl.py` (602行) | blade-kvt C++ 的 Python wrapper |
| `vllm/v1/engine/core.py` | Engine Core bypass 循环 |

---

## 2. Path A：主路径 -- SchedulerOutput + 共享内存

### 2.1 Scheduler 侧构建 metadata

在 `scheduler.py:1157`，调度循环结束后：

```python
# vllm/v1/core/sched/scheduler.py
def schedule(self):
    ...
    if self.connector is not None:
        meta = self.connector.build_connector_meta(scheduler_output)
        scheduler_output.kv_connector_metadata = meta
    return scheduler_output
```

`HybridConnector.build_connector_meta()` 委托给 `HybridScheduler.build_connector_meta()`：

```python
# __init__.py:1024-1070
def build_connector_meta(self, sout: SchedulerOutput) -> HybridMetadata:
    hc_parent = sout.__dict__.get("hc_parent")

    if hc_parent is None:
        # 主 step：分配新的 stepid，substepid 为 0
        stepid = self._stepid
        substepid = 0
        self._stepid += 1
        self._substepid = 1      # 重置 substepid 计数器
    else:
        # bypass substep：从 parent 取 stepid
        if hc_parent.kv_connector_metadata is None:
            # 边界情况：bypass 在主 step 之前触发
            hc_parent.kv_connector_metadata = HybridMetadata(
                reqs=BackendMeta(), stepid=self._stepid, substepid=0)
            self._stepid += 1
            self._substepid = 1
        stepid = hc_parent.kv_connector_metadata.stepid
        substepid = self._substepid
        self._substepid += 1     # 同一 step 内 substep 递增: 1, 2, 3, ...

    # 委托给后端构建具体 metadata
    reqs = self._backend.build_backend_meta(hcsout)
    return HybridMetadata(reqs=reqs, stepid=stepid, substepid=substepid)
```

`PBackend.build_backend_meta()` 构建 `KVTPMeta`，将请求分为三类：

```python
# kvtbackend.py:1670-1705
def build_backend_meta(self, sout):
    kvtpmeta = KVTPMeta(
        stepid=sout.hc_stepid,
        substepid=sout.hc_substepid,
        sched_tokens=sout.total_num_scheduled_tokens,
        freeze_metas=[],      # 已就绪的 KV 传输请求
        abort_metas=[],       # 已终止的请求
        nonfreeze_metas=[],   # 增量更新 / 常规请求
    )

    self._step_stop0(sout)          # 处理 stop0 请求
    self._step_aborting(sout)       # 处理 abort 请求
    substep_news = self._step_dinfoq(kvtpmeta.freeze_metas)  # 消费 D 侧请求 -> freeze_metas
    self._step_finished_reqs(sout, kvtpmeta)                 # 处理已完成请求
    self._step_dash_done()

    if sout.hc_parent is None:
        # 主 step：扫描所有已调度请求
        self._step_sched_req(sout, kvtpmeta.nonfreeze_metas)
    elif substep_news and envs.VLLM_ENABLE_BYPASS_SUBSTEP:
        # bypass substep：只处理新到的请求
        self._step_sched_req(sout.hc_parent, kvtpmeta.nonfreeze_metas,
                            substep_news=substep_news)

    return kvtpmeta
```

### 2.2 关键数据结构

```python
@dataclass
class HybridMetadata(KVConnectorMetadata):
    reqs: BackendMeta      # 后端特定的 metadata
    stepid: int = 0        # 步骤 ID（从 1024 开始）
    substepid: int = 0     # 子步骤 ID（0=主步骤，1+=bypass）

@dataclass
class KVTPMeta(BackendMeta):        # P 侧后端 metadata
    stepid: int
    substepid: int
    sched_tokens: int                # 本 step 总调度 token 数（流控用）
    freeze_metas: list[PReqMeta]     # 已冻结、可立即传输的请求
    abort_metas: list[PReqMeta]      # 已终止的请求
    nonfreeze_metas: list[PReqMeta]  # 未冻结、增量更新的请求

@dataclass
class PReqMeta:                      # 单个请求的传输指令
    # seen_tokens = 0 means submit_req_send       (首次发送，带 block IDs)
    # seen_tokens > 0 means submit_delta_send     (增量发送，不带 block IDs)
    reqid: str
    d_inst_id: str                   # 目标实例: "inst_name|dp_rank|tp_size"
    p_block_ids: list[list[int]]     # 源 (P节点) block IDs
    d_block_ids: list[list[int]]     # 目标 (D节点) block IDs
    new_tokens: int
    has_last_token: bool
    seen_tokens: int
    d_workers_info: list[str]
    has_freeze: bool = False

class KVTState:                      # 附着在 Request 上的 KV 传输状态
    dinfo: KVTDInfo                  # D 侧信息（目标实例、block IDs 等）
    maxtokens: int                   # 需要传输的总 token 数
    untouched: bool = True           # 是否尚未被处理过
```

### 2.3 物理传输：pickle + 共享内存

`SchedulerOutput`（含 `kv_connector_metadata`）通过 `MessageQueue` 传输：

```python
# vllm/distributed/device_communicators/shm_broadcast.py
class MessageQueue:
    def enqueue(self, obj):
        data = pickle.dumps(obj)  # 序列化整个对象
        # 本地 worker: 写入 ShmRingBuffer (POSIX 共享内存)
        # 远程 worker: ZMQ XPUB socket
        ...

    def dequeue(self):
        data = shm_ring_buffer.read()  # 或 zmq_sub_socket.recv()
        return pickle.loads(data)
```

完整流转链路：

```
EngineCore.step()
  -> scheduler.schedule()
    -> build_connector_meta() 设置 scheduler_output.kv_connector_metadata
  -> executor.execute_model(scheduler_output)
    -> collective_rpc("execute_model", args=(scheduler_output,))
      -> MessageQueue.enqueue()     <-- pickle.dumps + SHM/ZMQ
        -------- 进程边界 --------
      -> MessageQueue.dequeue()     <-- pickle.loads
    -> worker.execute_model(scheduler_output)
      -> model_runner.execute_model()
        -> maybe_setup_kv_connector()
          -> bind_connector_metadata(scheduler_output.kv_connector_metadata)
```

### 2.4 Worker 侧绑定 metadata

`KVConnectorModelRunnerMixin.maybe_setup_kv_connector()`（`kv_connector_model_runner_mixin.py:45`）在 `execute_model()` 中被调用：

```python
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
```

`HybridConnector.bind_connector_metadata()` -> `HybridWorker.bind_connector_metadata()` -> `PBackend.bind_backend_metadata()`（详见第4节）。

---

## 3. Path B: Bypass 路径 -- 独立 ZMQ 通道

### 3.1 Engine Core 的 Bypass 循环

在 `core.py:1946-2037`，当 `VLLM_ENABLE_BYPASS_TASK=True` 时：

```python
def _wait_model_output_future(self, model_output_future, step_sout, ...):
    # 1. 把 GPU forward 结果的等待丢给 ThreadPoolExecutor
    #    完成后推 EXECUTE_MODEL_RESULT 到 input_queue
    self.task_executor.submit(wait_model_output)

    # 2. 主线程进入 bypass spin 循环
    while True:
        # 处理新到的 ADD/ABORT 请求
        while not self.input_queue.empty():
            item = self.input_queue.get_nowait()
            if item[0] == EngineCoreRequestType.EXECUTE_MODEL_RESULT:
                return item[1], kvconn_outputs  # GPU forward 完成，退出循环
            if _should_process_now(item):
                self._handle_client_request(*item)

        # 处理 kvconn 事件（save-done / load-done）
        step_outputs = kvconn.step()

        # 构建 bypass substep
        substep_sout = SchedulerOutput.make_empty()
        substep_sout.hc_parent = step_sout       # 挂到主 step 上
        meta = kvconn.build_connector_meta(substep_sout)
        send_bypass_task(meta)                    # ZMQ 发送给 worker bypass 线程
```

关键点：bypass 循环在 **GPU forward 执行期间**持续运转。每次循环都调 `build_connector_meta(substep_sout)` 检查是否有新的 D 侧请求到达 `_dinfoq`。如果有，立即通过 ZMQ 发给 worker 的 bypass 线程，不等 forward 结束。

### 3.2 Bypass 消息分发

```python
# Scheduler 侧 (__init__.py:638-661)
def _start_bypass_rpc_server(self):
    # 创建 ZMQ XPUB socket
    self.bypass_remote_socket = zmq.Context().socket(zmq.XPUB)
    self.bypass_remote_port = self.bypass_remote_socket.bind_to_random_port(...)

def send_bypass_task(meta):
    if meta.reqs:  # 有实际内容才发
        data = pickle.dumps(meta)
        bypass_remote_socket.send(data)
```

### 3.3 Worker Bypass 线程接收

```python
# Worker 侧 (__init__.py:1161-1237)
def start_bypass_task_loop(self):
    """独立线程（disagg 线程），持续监听 bypass ZMQ socket"""
    # 1. 通过 RPC 获取 bypass socket 地址
    addr = self._get_bypass_handle()
    # 2. 创建 ZMQ SUB socket
    bypass_socket = zmq.Context().socket(zmq.SUB)
    bypass_socket.connect(addr)
    bypass_socket.subscribe(b"")
    # 3. 循环接收
    while True:
        recv = bypass_socket.recv()
        meta = pickle.loads(recv)
        self._do_bypass_meta(meta)

def _do_bypass_meta(self, meta: HybridMetadata):
    self._backend.bind_backend_metadata(meta.reqs)   # 绑定 metadata
    self._backend.clear_backend_metadata()            # bypass 线程: no-op
    # PBackend 的 async_load_kv 为 None，直接返回
```

---

## 4. 两个线程，两种角色：bind_backend_metadata 详解

`PBackend.bind_backend_metadata()` 是关键枢纽，通过 `threading.get_native_id()` 区分调用者身份：

```python
# kvtbackend.py:1834-1940
def bind_backend_metadata(self, meta: KVTPMeta):
    mytid = threading.get_native_id()

    stepid = meta.stepid
    substepid = meta.substepid
    sched_tokens = meta.sched_tokens
    failed_reqids: list[str] = []

    # ---- 两个线程都处理 freeze_metas ----
    if meta.freeze_metas:
        failed = self._start_req_send(meta.freeze_metas, stepid, substepid)
        failed_reqids.extend(failed)

    # ---- 分叉点 ----
    if mytid != self._main_tid:
        # BYPASS 线程路径
        assert not meta.abort_metas          # bypass 不处理 abort
        failed = self._start_send_substep(
            meta.nonfreeze_metas, stepid, substepid)
        failed_reqids.extend(failed)
        if failed_reqids:
            self._send_error_done_in_loop(failed_reqids)
        return                               # <-- 到此为止，不碰 start_send_step

    # MAIN 线程路径
    assert substepid == 0

    # 处理 abort_metas
    for reqm in meta.abort_metas:
        assert reqm.new_tokens == 0 and reqm.has_last_token
        self._bladkv_cli.submit_delta_send(
            reqm.reqid, seen_tokens=reqm.seen_tokens,
            new_tokens=0, has_last_token=True, stepid=stepid)

    # 处理 nonfreeze_metas
    for reqm in meta.nonfreeze_metas:
        if reqm.d_block_ids:
            self._bladkv_cli.submit_req_send2(...)   # 首次发送：带 block IDs
        else:
            self._bladkv_cli.submit_delta_send(...)   # 增量发送：靠 seen_tokens 定位

    # 关键：只有主线程调用 start_send_step
    self._bladkv_cli.start_send_step(stepid=stepid, sched_tokens=sched_tokens)
```

### 4.1 主线程路径

主线程处理所有三种 meta，最后调 `start_send_step` 创建真实的 Step/StepGuard：

- **freeze_metas** -> `_start_req_send()` -> `bladkv_cli.start_req_send()`
- **abort_metas** -> `bladkv_cli.submit_delta_send(new_tokens=0, has_last_token=True)`
- **nonfreeze_metas** -> `submit_req_send2()`（有 block IDs）或 `submit_delta_send()`（增量）
- **收口** -> `bladkv_cli.start_send_step(stepid, sched_tokens)` 创建 Step

### 4.2 Bypass 线程路径

Bypass 线程只处理 freeze_metas 和 nonfreeze_metas，不碰 abort，不创建 Step：

- **freeze_metas** -> `_start_req_send()` -> `bladkv_cli.start_req_send()`
- **nonfreeze_metas** -> `_start_send_substep()` -> `bladkv_cli.start_send_substep()` 追加到已有 Step
- **不调** `start_send_step` 或 `flush_send_step`

两个方法的对比：

```python
# _start_req_send: 两个线程都可调用，提交 frozen 请求
def _start_req_send(self, freeze_metas, stepid, substepid):
    kvtmetas, failed = self._build_req_send_batch(freeze_metas, expect_freeze=True)
    if kvtmetas:
        self._bladkv_cli.start_req_send(kvtmetas, stepid, substepid)
    return failed

# _start_send_substep: 仅 bypass 线程调用，追加到已有 step
def _start_send_substep(self, nonfreeze_metas, stepid, substepid):
    kvtmetas, failed = self._build_req_send_batch(nonfreeze_metas, expect_freeze=False)
    if kvtmetas:
        self._bladkv_cli.start_send_substep(stepid, substepid, kvtmetas)
    return failed
```

### 4.3 clear_backend_metadata 的线程区分

```python
# kvtbackend.py:1942-1951
def clear_backend_metadata(self):
    mytid = threading.get_native_id()
    if mytid != self._main_tid:
        return                    # bypass 线程: 什么都不做
    self._bladkv_cli.flush_send_step()  # 主线程: 收口当前 step
```

---

## 5. Step 时间线与 blade-kvt 协调

### 5.1 主线程的 Step 生命周期

```
主线程时间线:
  bind_backend_metadata()
    +-- submit_req_send2() / submit_delta_send()  <-- 把任务放入 targets_tasks_buf_
    +-- start_send_step(stepid, sched_tokens)     <-- 创建 Step/StepGuard，消费 buf

  model forward 执行中...
    +-- record_event(layer_0, stream)             <-- 第0层算完，blade-kvt 可传该层 KV
    +-- record_event(layer_1, stream)             <-- 第1层算完
    +-- ...

  clear_backend_metadata()
    +-- flush_send_step()                         <-- 结束当前 step
```

`record_event(layer_id)` 是 layer-by-layer 流水线的关键--每层 attention 计算完毕后，在当前 CUDA stream 上 record 一个 event，通知 blade-kvt "这一层的 KV 已经可以传了"。blade-kvt 的传输线程在独立 CUDA stream 上等待该 event，然后发起该层的 RDMA 发送，与后续层的计算重叠执行。

### 5.2 start_send_step vs start_send_substep

blade-kvt Python wrapper（`kv_transfer_impl.py`）：

```python
def start_send_step(self, stepid, sched_tokens=1):
    """主线程调用，创建 Step/StepGuard"""
    EMPTY_STEP_ID = 9223372036854775807  # INT64_MAX
    step_id = start_send(stepid, sched_tokens)
    # sched_tokens == 0: C++ 跳过 Step 创建，返回 EMPTY_STEP_ID
    # sched_tokens > 0: 创建 Step/StepGuard（即使 targets_tasks_buf_ 为空）
    if step_id != EMPTY_STEP_ID:
        assert step_id == stepid
        self._cur_step_id = step_id

def start_send_substep(self, stepid, substepid, metas):
    """Bypass 线程调用，追加任务到已有 step"""
    # substepid 必须 > 0 且单调递增
    # metas 必须是 has_freeze=False 的
    start_send_substep(stepid, substepid, metas)

def flush_send_step(self):
    """主线程调用，结束当前 step"""
    flush_send(self._cur_step_id)
    self._cur_step_id = None
```

### 5.3 blade-kvt C++ 的三路时序处理

`start_send_substep(stepid, substepid, metas)` 不是简单的"提交任务"接口，而是一个**带时序协调语义的接口**。

bypass 线程和主线程是并发的，substep 到达时主线程的 step 可能处于三种状态：

```
                    时间线 ------------------------------------------------>

主线程:     ---[start_send_step(N)]================[flush_send_step(N)]---
                        ^                                ^
                     创建 Step                         结束 Step
                     coord_step_id = N

Bypass 线程可能在任何时刻到达:

情况1: 晚到 (stepid < coord_step_id)
  substep -->  X   主线程已经到了 step N+1，这批 metas 属于旧 step
  处理: 直接丢弃

情况2: 早到 (stepid > coord_step_id)
  substep --> [暂存 pending_step_metas_]
  处理: 等之后主线程 start_send_step(stepid) 时统一附加

情况3: 正好 (stepid == coord_step_id 且 step 未 freeze)
  substep --> 直接转成 StepTasks 挂到当前 Step 上
  处理: 立即生效
```

这个设计的核心意义：**bypass 线程不需要和主线程做任何锁同步**。它只管把 metas 往 `start_send_substep` 里丢，C++ 侧根据当前 step 状态自动决定处理方式。

### 5.4 为什么 substep 的 stepid 会大于主线程的 coord_step_id

核心原因：**两条传输路径的延迟不同**。

```
时间 --------------------------------------------------------->

Scheduler 进程:
  step N-1 完成 | step N: schedule() -> build_meta(N,0)
                |   |
                |   +-- execute_model(sout_N) --SHM--> [进 MessageQueue 排队]
                |   |
                |   +-- bypass 循环开始
                |   |   新 D 请求到达
                |   |   build_meta(N,1)
                |   +-- send_bypass(meta) --ZMQ--> [直达 bypass 线程]
                |

Worker 主线程:
  ==== step N-1 的 model forward 还在跑 =====|
                                             | forward 完成
                                             | dequeue sout_N
                                             | bind_meta -> start_send_step(N)
                                             | coord_step_id = N

Worker bypass 线程:
                     ZMQ 收到 substep |
                     start_send_substep(N, 1, metas)
                     此时 coord_step_id 还是 N-1  <-- 主线程还没到 step N！
                     -> stepid(N) > coord_step_id(N-1)
                     -> 早到！暂存 pending_step_metas_
```

两个因素叠加：

| 通道 | 载荷 | 延迟 |
|------|------|------|
| 主路径 (SHM MessageQueue) | 整个 SchedulerOutput（大对象 pickle） | 较高，且要等主线程忙完上一 step |
| Bypass 路径 (ZMQ PUB/SUB) | 仅 HybridMetadata（小对象 pickle） | 较低，bypass 线程始终在 recv 等待 |

所以最常见的场景是：scheduler 进程已经进入 step N 的 bypass 循环并发出了 substep(N,1)，但 worker 主线程还卡在 step N-1 的 forward 里，`coord_step_id` 仍是 N-1。

---

## 6. Chunked Prefill 的多步传输与 _infly_kvt 状态管理

### 6.1 具体示例：8192 token 分两个 chunk

假设一个 prompt 有 8192 个 token，`chunk_size=4096`，需要分两个 step 完成 prefill：

```
Step N:   计算 token [0, 4096)     -> has_last_token=False
Step N+1: 计算 token [4096, 8192)  -> has_last_token=True
```

### 6.2 Step N：首次调度 (has_last_token=False)

在 `_step_sched_req` 中（主线程路径，非 substep_mode）：

```python
# kvtbackend.py:1550-1575
seen_tokens = d_computed_tokens      # D 侧已有的 token 数，通常 0
new_tokens = 4096 - d_computed_tokens  # 本次新算的
has_last_token = (4096 >= 8192)      # False，还没算完

kvtstate.untouched = False           # 标记：这个请求已经被处理过

kvtmeta = PReqMeta(
    reqid=req_id,
    d_inst_id=d_inst_id,
    p_block_ids=p_block_ids,         # 有值：P 侧的 block IDs
    d_block_ids=d_blocks_ids,        # 有值：D 侧的 block IDs
    new_tokens=4096,
    has_last_token=False,             # <-- 关键：没完
    seen_tokens=0,                    # 首次发送
)
ret.append(kvtmeta)                  # 放入 nonfreeze_metas
self._update_infly_kvt(kvtmeta)      # <-- 存入 _infly_kvt
```

`_update_infly_kvt` 的处理：

```python
# kvtbackend.py:1406-1421
def _update_infly_kvt(self, meta: PReqMeta):
    if meta.d_block_ids:                    # 首次：有 block IDs
        assert meta.reqid not in self._infly_kvt
        if not meta.has_last_token:         # 没完 -> 存下来
            self._infly_kvt[meta.reqid] = meta   # <-- 写入状态！
        return
```

**`_infly_kvt[reqid] = meta`** -- 记录这个请求"已经发了哪些 token 的 KV，还差多少"。

在 `bind_backend_metadata` 主线程路径中，这个 meta 会走 `submit_req_send2`（因为 `seen_tokens=0` 且有 `d_block_ids`）：

```python
# 主线程路径
if reqm.d_block_ids:
    self._bladkv_cli.submit_req_send2(
        dst_inst_name, dst_wid, reqm.reqid,
        seen_tokens=0, new_tokens=4096,
        has_last_token=False,
        src_block_ids=reqm.p_block_ids,
        dst_block_ids=reqm.d_block_ids,
        ...)
```

blade-kvt C++ 侧记录下 block 映射关系，开始传输 token [0, 4096) 的 KV cache。

### 6.3 Step N+1：继续调度 (has_last_token=True)

下一个 step，这个请求变成 `scheduled_cached_reqs`（不再是 new_req），走另一段逻辑：

```python
# kvtbackend.py:1637-1655
# 此时 kvtstate.untouched = False（step N 已经设过了）
# seen_tokens = num_computed_tokens = 4096（上次算到的位置）

elif seen_tokens < num_tokens:
    # d_computed_tokens < seen_tokens < num_tokens
    assert not substep_mode           # <-- 断言：不在 substep 模式！
    assert not kvtstate.untouched     # <-- 断言：不是第一次处理

    new_tokens = end_tokens - seen_tokens  # 8192 - 4096 = 4096

    kvtmeta = PReqMeta(
        reqid=req_id,
        d_inst_id="",                 # 空！
        p_block_ids=[],               # 空！
        d_block_ids=[],               # 空！不再需要 block 映射
        new_tokens=4096,
        has_last_token=True,
        seen_tokens=4096,             # <-- seen_tokens > 0 表示 delta_send
    )
    ret.append(kvtmeta)
    self._update_infly_kvt(kvtmeta)
```

注意 `PReqMeta` 的注释约定：

```python
# seen_tokens = 0 means submit_req_send      -> 首次发送，带 block IDs
# seen_tokens > 0 means submit_delta_send     -> 增量发送，不带 block IDs
```

在 `bind_backend_metadata` 主线程路径中，这个 meta 走 `submit_delta_send`：

```python
# 主线程路径
else:  # d_block_ids 为空
    self._bladkv_cli.submit_delta_send(
        reqm.reqid,
        seen_tokens=4096,         # 之前已经发了 4096 个 token
        new_tokens=4096,          # 这次接着发 4096 个
        has_last_token=True,      # 这是最后一批
        stepid=stepid,
    )
```

`delta_send` 的语义：blade-kvt C++ 侧根据 `reqid` 找到之前 `submit_req_send` 时记录的 block 映射信息，从 `seen_tokens` 的偏移继续发 `new_tokens` 个 token 的 KV。

`_update_infly_kvt` 这次的处理：

```python
def _update_infly_kvt(self, meta: PReqMeta):
    if meta.d_block_ids:              # 这次 d_block_ids 为空，跳过
        ...

    assert meta.reqid in self._infly_kvt   # 必须在 _infly_kvt 里！
    if meta.has_last_token:                 # True -> 完成了
        prevmeta = self._infly_kvt.pop(meta.reqid)   # <-- 从状态中移除
        return
```

### 6.4 中途 Abort 的处理

如果请求在 step N 发了一半就被 abort（比如用户取消），`_step_finished_reqs` 处理：

```python
# kvtbackend.py:1434-1460
def _step_finished_reqs(self, sout, kvtpmeta):
    for reqid in sout.finished_req_ids:
        meta = self._infly_kvt.pop(reqid, None)   # 从 _infly_kvt 取出
        if meta is None:
            self._dash_finish_req(reqid, kvtpmeta.freeze_metas)
            continue

        # 请求还在 _infly_kvt 里（还没发完）
        assert not meta.has_last_token and meta.new_tokens > 0
        new_seen_tokens = meta.seen_tokens + meta.new_tokens  # 已发的总量

        new_meta = PReqMeta(
            reqid=meta.reqid,
            d_inst_id="",
            p_block_ids=[],
            d_block_ids=[],
            new_tokens=0,
            has_last_token=True,           # 强制标记为"最后一批"
            seen_tokens=new_seen_tokens,
        )
        # 从 _infly_kvt 处理的 meta 放入 abort_metas -> submit_delta_send 通知收口
        kvtpmeta.abort_metas.append(new_meta)
```

---

## 7. 为什么 Bypass 线程不能处理 has_last_token=False

### 7.1 状态依赖链

Chunked prefill 的多步传输形成一条严格的状态依赖链，全部依赖 `_infly_kvt` 这个 Python dict：

```
Step N (主线程):
  _step_sched_req()
    -> PReqMeta(seen_tokens=0, d_block_ids=[...], has_last_token=False)
    -> _update_infly_kvt()  写入 _infly_kvt[reqid]     <-- 状态写入
    -> bind_backend_metadata()
      -> submit_req_send2(block_ids)                     <-- 首次发送

Step N+1 (主线程):
  _step_sched_req()
    -> PReqMeta(seen_tokens=4096, d_block_ids=[], has_last_token=True)
    -> _update_infly_kvt()  从 _infly_kvt pop            <-- 状态读+删
    -> bind_backend_metadata()
      -> submit_delta_send(seen_tokens=4096)             <-- 增量发送

如果中途 abort (主线程):
  _step_finished_reqs()
    -> _infly_kvt.pop(reqid)                             <-- 状态读+删
    -> abort_metas -> submit_delta_send(has_last_token=True)
```

`_infly_kvt` 是纯 Python dict，没有任何锁保护。整个链条 -- **写入 -> 后续 step 读取/更新 -> 最终移除** -- 都假设在主线程上串行执行。

### 7.2 竞态风险分析

如果 bypass 线程处理 `has_last_token=False` 的请求，会引入以下竞态：

**风险 1：并发写 `_infly_kvt`**

bypass 线程写 `_infly_kvt[reqid] = meta`，同时主线程的 `_step_finished_reqs` 可能正在 `_infly_kvt.pop(reqid)` -- 对同一个 dict 的并发读写是未定义行为。

**风险 2：`seen_tokens` 不一致**

`submit_delta_send` 需要知道 `seen_tokens`（上次发到哪了），这个信息存在 `_infly_kvt` 里。如果 bypass 线程也往里写，两个线程对 `seen_tokens` 的理解可能不一致，导致 KV 数据错位或重复发送。

**风险 3：abort 处理依赖 `_infly_kvt` 完整**

`_step_finished_reqs` 假设 `_infly_kvt` 里的 meta 反映最新的发送进度（`meta.seen_tokens + meta.new_tokens`）。如果 bypass 线程在后台修改了某个 reqid 的 meta，主线程计算出的 `new_seen_tokens` 就是错的，发给 D 侧的 abort 通知会携带错误的偏移量。

### 代码中的防护

`_step_sched_req` 在 substep_mode 下有明确的过滤和断言：

```python
# kvtbackend.py, substep_mode=True
if substep_mode:
    if not has_last_token:
        continue                  # 跳过未完成的请求

    assert kvtstate.untouched     # 必须是第一次处理（没有 _infly_kvt 前序状态）
    ...
    assert kvtmeta.has_last_token and len(kvtmeta.d_block_ids) > 0

# 在增量发送的分支里
elif seen_tokens < num_tokens:
    assert not substep_mode       # 增量发送不允许在 substep 模式
    assert not kvtstate.untouched # 必须有前序处理
```

总结：**bypass 路径选择了一个更收敛的设计 -- 只处理 `has_last_token=True` 且 `kvtstate.untouched=True` 的请求**。这类请求能一次发完所有 KV，不需要写 `_infly_kvt`，不需要后续的 `delta_send`，bypass 线程处理是完全安全的。而需要跨 step 状态跟踪的 chunked prefill 请求，始终走主线程路径。

---

## 8. 与社区版 (NIXL) 的关键差异

| 维度 | 社区版 (NixlConnector) | Develop 版 (HybridConnector) |
|------|----------------------|---------------------------|
| **传输路径** | 仅主路径（SchedulerOutput） | 主路径 + Bypass 路径（双通道） |
| **Bypass 机制** | 无 | 有，通过独立 ZMQ 通道在 forward 期间发送额外 KV 传输指令 |
| **Step/Substep** | 无概念 | stepid + substepid，支持流水线传输 |
| **完成通知** | `get_finished()` 轮询（同步） | TCP RPC 异步推送 (`IoDoneReqs`) |
| **传输后端** | NIXL (UCX/RDMA) | blade-kvt (`bladekv.KVTransferClient/Server`) |
| **Scheduler 线程** | 无额外线程 | 有独立的 asyncio disagg 线程处理异步事件 |
| **Metadata 复杂度** | 简单：`ReqMeta` + block IDs | 复杂：freeze/nonfreeze/abort 三类，流控 (`sched_tokens`) |
| **传输粒度** | Block 级，整块传输 | Token 级，layer-by-layer 流水线，`record_event` 控制同步 |
| **Chunked Prefill 支持** | 不涉及（D 侧等完整 KV） | 支持分 chunk 逐步传输 + delta_send |

---

## 9. 完成通知：Worker -> Scheduler (TCP RPC)

KV 传输完成后，Worker 通过**独立的 TCP RPC 通道**通知 Scheduler：

```python
# Worker 侧
# 传输完成 -> 发送 IoDoneReqs (msgspec 编码) -> TCP socket
io_done = IoDoneReqs(load_done=[req_id1, req_id2, ...])
tcp_socket.send(msgspec.encode(io_done))

# Scheduler 侧
# RpcServer (asyncio.start_server) 接收
async def _on_load_done(self, req_ids):
    for req_id in req_ids:
        self._finished_recving.add(req_id)
        # 请求可以从 WAITING_FOR_REMOTE_KVS 恢复
```

这与社区版的 `get_finished()` 轮询方式不同，develop 版使用异步推送，避免了在每个 step 结束时做同步的完成检查。

---

## 附：整体时间线

```
Engine Core (Scheduler 进程)               Worker 进程

step N:
  scheduler.schedule()
    -> build_connector_meta(stepid=N, substepid=0)
      -> _step_dinfoq() -> freeze_metas
      -> _step_sched_req() -> nonfreeze_metas
    -> HybridMetadata 嵌入 SchedulerOutput

  executor.execute_model(sout)  --pickle+SHM-->  主线程收到 sout
                                                    |
                                               bind_backend_metadata()
                                                 +-- freeze: start_req_send()
                                                 +-- abort: submit_delta_send()
                                                 +-- nonfreeze: submit_req_send2()
                                                 +-- start_send_step(N, tokens)
                                                        <-- 创建 Step/StepGuard
                                                    |
                                               model forward 开始
                                                 +-- layer 0 完成 -> record_event(0)
                                                 +-- layer 1 完成 -> record_event(1)
                                                 |   blade-kvt 传输线程开始发 layer 0
                                                 +-- ...

  同时 bypass 循环:                              |
    新 D 请求到达 _dinfoq                         |
    build_connector_meta(substepid=1)             |
    send_bypass_task() --pickle+ZMQ-->         bypass 线程收到
                                                 +-- freeze: start_req_send()
                                                 +-- nonfreeze: start_send_substep(N, 1)
                                                        <-- C++ 追加到 Step N
                                                    |
                                               forward 完成
                                               clear_backend_metadata()
                                                 +-- flush_send_step()
                                                        <-- 结束 Step N
```
