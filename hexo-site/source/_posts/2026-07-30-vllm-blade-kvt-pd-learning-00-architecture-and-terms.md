---
title: "00 架构与术语：先把每个对象放回正确的层"
date: 2026-07-30
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 00 架构与术语：先把每个对象放回正确的层

## 1. 为什么要做 Prefill/Decode 分离

LLM 推理大致分成两个阶段：

- **Prefill（P）**：一次处理较长 prompt，矩阵乘法规模大，吞吐导向，产生每一层的
  KV Cache。
- **Decode（D）**：通常每轮只生成一个或少量 token，反复读取已有 KV Cache，延迟
  导向。

把 P 和 D 放在独立实例，可以分别选择批大小、并行策略和 GPU 资源，隔离长 prompt
对 decode ITL 的干扰。代价是：P 产生的 KV 不再天然位于 D 的 GPU 内，必须把它们
准确地搬到 D 已分配的 KV Block 中。

“PD 分离”至少包含三件不同的事：

1. **请求路由**：一个请求先去哪一个 P，之后去哪一个 D。
2. **控制面 rendezvous**：P/D 如何用 request ID、实例 ID、TP rank 和 Block ID
   对上同一个请求。
3. **数据面传输**：KV 字节如何从 P GPU 到 D GPU。

本文代码主要覆盖后两项。外部网关如何选 P/D，不属于 Blade-KVT 数据面的职责。

## 2. 分层图

```text
┌─────────────────────────────────────────────────────────────┐
│ 服务/路由层：决定 remote_host、remote_port、P/D 实例       │
└──────────────────────────────┬──────────────────────────────┘
                               │ EngineCoreRequest
┌──────────────────────────────▼──────────────────────────────┐
│ vLLM EngineCore / Scheduler                                  │
│ Request 状态、token budget、KV Block、SchedulerOutput       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HybridMetadata / RPC
┌──────────────────────────────▼──────────────────────────────┐
│ Hybrid Connector + KVT Backend                              │
│ load/save 计划、P/D RPC、TP 完成汇聚、step/substep          │
└──────────────────────────────┬──────────────────────────────┘
                               │ Python extension API
┌──────────────────────────────▼──────────────────────────────┐
│ Blade-KVT                                                    │
│ token/block→IpcBlock、按 target 分组、按 layer 等 Event      │
└──────────────────────────────┬──────────────────────────────┘
                               │ Channel API
┌──────────────────────────────▼──────────────────────────────┐
│ Barex                                                        │
│ 连接、MR、QP/CQ/WQE、TCP/RDMA submit 与 completion           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    PCIe / RNIC / 网络 / GPU HBM
```

## 3. Scheduler Connector 与 Worker Connector

同一个 Hybrid Connector 有两个角色。

### 3.1 Scheduler 侧

它不直接拿 GPU pointer，主要处理：

- 判断请求需要 load、save，还是都不需要；
- 在传输前分配 KV Block；
- 构造 worker 可执行的 metadata；
- 维护 `_waiting/_loading/_saving/_loaded/_saved/_aborting`；
- 聚合各 TP worker 完成信号；
- 在传输结束后让请求回到普通 Scheduler；
- 协调 abort、Block 引用和最终输出。

核心类是：

```text
HybridConnector
  └─ HybridScheduler
       └─ PBackend 或 DBackend（scheduler role）
```

### 3.2 Worker 侧

Worker 进程拥有真实 GPU KV tensor，主要处理：

- 注册 KV cache 的 device pointer；
- 根据 metadata 调用 Blade-KVT client/server；
- 在 layer forward 后记录 CUDA Event；
- 发起或接收数据传输；
- 把 load/save 完成 RPC 返回 Scheduler。

```text
HybridConnector
  └─ HybridWorker
       └─ PBackend 或 DBackend（worker role）
            ├─ P: bladekv.KVTransferClient
            └─ D: bladekv.KVTransferServer
```

这里的 P/D 与 Scheduler/Worker 是两个正交维度：

| 实例 | Scheduler 进程做什么 | Worker 进程做什么 |
|---|---|---|
| P | 接 D 请求、跟踪 P Request、生成发送 metadata、等所有 TP send done | 读 P GPU KV、创建 Client、真正发送 |
| D | 分配目标 Block、向 P 发 RPC、等待 KVTResp、把请求放回调度器 | 注册 D GPU KV、创建 Server、接收远端写入 |

## 4. Request、Block、Step、Layer 的区别

### Request

业务请求生命周期最长，request ID 是跨进程关联主键。它包含 token、采样参数、
状态、`num_computed_tokens` 和 `kv_transfer_params`。

### KV Block

GPU KV cache 的逻辑页。一个请求通常拥有多个 Block；混合模型还可能有多个 KV cache
group。Block ID 本身不是网络地址。

### Step

一次 vLLM 调度/模型执行轮次。当前实现的 Hybrid step ID 从 1024 起递增。一个 Step
里可以包含很多 Request 和很多 layer 的数据。

### Substep

bypass 路径在主 SchedulerOutput 已经形成后，把晚到的 P/D 匹配任务附加到正在运行的
主 Step。主 step 的 `substepid=0`，bypass 从 1 递增。

### Layer

模型的物理 KV 层。P forward 是逐层完成的，Blade-KVT 因而可以做到：

```text
layer 0 ready → 发送 layer 0
layer 1 ready → 发送 layer 1
...
```

这使 GPU 计算和网络发送形成流水线，而不是等所有层都算完后再整批发送。

## 5. 数据对象如何逐层变形

![请求元数据到 RDMA Work Request](/imgs/vllm-pd-object-transform.svg)

以一次发送为例：

```text
vLLM Request
  ↓ PBackend 根据本轮 scheduled token 生成
PReqMeta
  - reqid
  - seen_tokens/new_tokens
  - P/D block IDs
  - D instance / worker info
  ↓ P Worker 做 TP rank 映射
bladekv.ReqMeta
  ↓ Blade-KVT create_step_tasks + add_send_task
ReqSendTask / BatchSendTask
  ↓ parse_block
IpcBlock(src_offset, dst_offset, length)
  ↓ RDMAChannel::send_data
rw_memp_t
  - remote address / rkey
  - local SGE address / length / lkey
  ↓ Barex
WR/WQE → RNIC
```

这解释了为什么不能把 `IpcBlock` 直接等同于 SGE：

- `IpcBlock` 同时描述源、目的偏移，是 Blade-KVT 的布局语义；
- SGE 只描述本地可 DMA 的一段内存；
- RDMA WRITE 的远端地址与 `rkey` 是另一组字段；
- Barex 最终把这些字段编码为 RNIC 能执行的 WQE。

## 6. 三类“异步”

代码里说“异步”可能指三个层次：

1. **CUDA launch 异步**：CPU 发起 kernel 后返回，GPU stream 继续执行。
2. **Python coroutine 异步**：遇到 socket/Future `await` 时让出 uvloop 线程。
3. **Barex submit 异步**：`send_data()` 提交 WR 后返回，CQ callback 以后才完成。

三者不是同一个调度器，也不能用一个 completion 代替另一个。

## 7. 四种 completion

| completion | 生产者 | 消费者 | 能证明什么 |
|---|---|---|---|
| CUDA Event complete | GPU stream | Blade-KVT wait-layer 线程 | 该层 Event 之前的 GPU 工作完成 |
| Barex callback/CQE | RNIC/Barex | Channel future | 本次传输操作达到 transport 的本地完成边界 |
| Blade-KVT `SEND_DONE` | `KvSendStub` | PBackend | 此 P worker 对该请求的发送批次已完成或失败 |
| `KVTResp` / `mark_loaded` | P Scheduler / D backend | D HybridScheduler | D 请求状态机可以进入 loaded/abort 分支 |

因此排障时要问“卡在哪一层 completion”，不能只说“RDMA 没完成”。

## 8. 当前实现的关键文件

### vLLM

```text
vllm/v1/engine/core.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/hybrid_connector/__init__.py
vllm/v1/hybrid_connector/engine_proxy.py
vllm/v1/hybrid_connector/utils.py
vllm/v1/hybrid_connector/kvtbackend.py
vllm/v1/worker/kv_connector_model_runner_mixin.py
```

### Blade-KVT

```text
blade_kvt/kv_transfer_impl.py
kvtransfer/include/client.h
kvtransfer/src/client.cpp
kvtransfer/include/step.h
kvtransfer/src/step.cpp
kvtransfer/include/tx_stub.h
kvtransfer/src/tx_stub.cpp
kvtransfer/include/channel.h
kvtransfer/src/tcp_channel.cpp
kvtransfer/src/rdma_channel.cpp
kvtransfer/src/rdma_staged_channel.cpp
kvtransfer/src/barex_protocol.cpp
```

## 9. 自检

1. 为什么 DBackend scheduler 与 DBackend worker 的职责不同？
2. `block_id=42` 为什么不能直接交给 RNIC？
3. Barex CQE 到达后，为什么 P Scheduler 仍可能不知道整个请求完成？
4. `stepid`、`substepid`、`layer_id`、`request_id` 各自解决什么维度的关联？
5. “异步”一词在 CUDA、Python、Barex 中分别意味着什么？
