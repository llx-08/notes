---
title: blade-kvt 项目架构与 vLLM 对接指南
date: 2026-05-25
tags: []
---
# blade-kvt 项目架构与 vLLM 对接指南

本文档基于仓库源码与 `/mnt/data/llx/vllm` 中 `kvtbackend` 集成代码整理，用于理解 **blade-kvt** 自身结构、组件抽象，以及 **Prefill (P) / Decode (D)** 分离场景下如何与 vLLM 协同工作。

---

## 目录

1. [项目概览](#1-项目概览)
2. [目录结构](#2-目录结构)
3. [核心抽象与组件职责](#3-核心抽象与组件职责)
4. [数据面：KV 传输流程](#4-数据面kv-传输流程)
5. [与 vLLM 的对接](#5-与-vllm-的对接)
6. [P/D 实例发现](#6-pd-实例发现)
7. [请求生命周期与状态维护](#7-请求生命周期与状态维护)
8. [实例与运行时状态](#8-实例与运行时状态)
9. [配置与环境变量速查](#9-配置与环境变量速查)
10. [关键源码索引](#10-关键源码索引)

---

## 1. 项目概览

**blade-kvt** 是面向 **Prefill/Decode 分离（P/D Disaggregation）** 的 KV Cache 传输库：

| 角色 | 含义 | 典型部署 |
|------|------|----------|
| **P (Producer / Prefill)** | 执行 prompt prefill，将 KV cache 块发送给 D | `kv_role=kv_producer` |
| **D (Consumer / Decode)** | 接收 KV，在本地继续 decode | `kv_role=kv_consumer` |

传输协议支持 **RDMA**（`accl-barex`，GDR 直写或 Staged D2H+Send）和 **TCP**（CUDA D2H + 网络发送）。上层通过 Python 扩展 `kvtransfer_ops` 暴露 API，由 vLLM 的 `HybridConnector` + `kvtbackend` 在调度与模型 forward 过程中驱动。

```
┌─────────────────────────────────────────────────────────────────┐
│ vLLM (HybridConnector / kvtbackend)                              │
│  PBackend (Scheduler + Worker)  │  DBackend (Scheduler + Worker) │
└────────────┬────────────────────┴──────────────────┬──────────────┘
             │ RPC (PREFILL_REQ / TRANSFER_KV_REQ)   │
             │                                       │
┌────────────▼──────────────────────────────────────▼──────────────┐
│ blade_kvt (Python)                                                 │
│  KVTransferClient (P)              KVTransferServer (D)            │
│  connect_naming()                  worker_{id} 注册               │
└────────────┬──────────────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────────────┐
│ kvtransfer (C++/CUDA)                                              │
│  KvTransferClient → KvSendStub → IChannel (RDMA/TCP)              │
│  ITransferServer ← 接收并写入 decode 侧 GPU KV                     │
│  parse_block_* / TransferPlan → IpcBlock 描述符                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
blade-kvt/
├── blade_kvt/                    # Python 封装层
│   ├── __init__.py               # 默认 ACCL/RDMA 环境变量
│   ├── kv_transfer.py            # 未编译时的 stub，保证 vLLM 可启动
│   ├── kv_transfer_impl.py       # KVTransferClient / KVTransferServer
│   └── nic_affinity.py           # RDMA NIC 亲和性文件
│
├── kvtransfer/                   # C++/CUDA 核心
│   ├── include/                  # 头文件（接口定义）
│   ├── src/                      # 实现（client、channel、parse_block 等）
│   ├── kvtransfer_pybind.cpp     # pybind11 入口 → kvtransfer_ops
│   ├── docs/cache_transfer_spec.md
│   └── tests/
│
├── setup.py / pyproject.toml
├── CRC_README.md                 # CRC 校验说明
└── study_doc/                    # 本文档所在目录
```

vLLM 侧相关路径（不在本仓库内，但为对接必读）：

```
vllm/v1/hybrid_connector/
├── __init__.py           # HybridConnector, HybridScheduler, HybridWorker
├── kvtbackend.py         # PBackend, DBackend, KVTDInfo, KVTState
├── engine_proxy.py       # 与 EngineCore 的 RPC/调度桥接
└── utils.py              # PeerManager, PeerInfo, ConnManager
```

---

## 3. 核心抽象与组件职责

设计采用 **接口 + 工厂 + 策略** 分层：上层编排（Client/Server），中层发送执行（SendStub），底层传输（Channel），旁路服务发现（Naming）与布局解析（parse_block）。

### 3.1 状态与上下文

| 类型 | 文件 | 职责 |
|------|------|------|
| `Context` | `kvtransfer/include/context.h` | 单 worker 全局状态：TP 信息、层 GPU 地址、block/token 大小、协议上下文 |
| `WorkerInfo` | `kvtransfer/include/common.h` | 可序列化的 worker 元数据（inst_id、IP:port、engine_tp_size、GDN/indexer 字段等） |
| `LayerInfo` | `common.h` | 单层：`token_size`, `block_size`, `layer_addr` |
| `IProtocolContext` | `context.h` | 协议初始化钩子；`BarexProtoContext` 实现 RDMA/TCP |
| `ICUDABarrier` | `context.h` | 每层 CUDA event，与 Python `record_event` 同步 |

### 3.2 发送路径（P 侧）

| 类型 | 文件 | 职责 |
|------|------|------|
| `KvTransferClient` | `include/client.h` | 请求缓冲、Step 生命周期、Target 池、与 forward 的层同步 |
| `ISendStub` / `KvSendStub` | `include/tx_stub.h` | 针对单个目标 worker 的发送执行器 |
| `TargetMgr` | `src/client.cpp` | `(dst_inst, dst_worker)` → SendStub 的 LRU 池 |
| `BatchSendTask` | `tx_stub.h` | 一个 substep 内绑定到 `Step` 的批量任务 |
| `Step` / `StepGuard` | `include/step.h` | step 协调：等待各层 `record_event` 完成后再 send_data |
| `RequestInfo` / `ReqSendTask` | `common.h` | 单请求元数据、块 ID、token 范围 |

**KvTransferClient 关键职责：**

1. `submit_req_send` / `submit_delta_send`：将任务写入 `targets_tasks_buf_`
2. `start_send(stepid)`：创建 `Step` + `StepGuard`，提交到线程池
3. `notify_event_record`：Python 每层 forward 后释放对应层信号量
4. `flush_send_step`：等待所有层就绪并收尾

### 3.3 传输通道

| 类型 | 文件 | 职责 |
|------|------|------|
| `IChannel` | `include/channel.h` | 统一协议：`connect` → `register_data` → `send_data(layer)` → `flush` |
| `IpcBlock` | `channel.h` | `{src_offset, dst_offset, length}` 拷贝描述符 |
| `RDMAChannel` | `protocol/rdma_channel.h` | GDR 注册 + RDMA WriteBatch 直写 D 的 GPU |
| `RDMAStagedChannel` | `protocol/rdma_staged_*.h` | D2H 到 CPU staging，再 RDMA Send |
| `TCPChannel` | `protocol/tcp_channel.h` | CUDA copy kernel D2H + TCP |
| `ITransferServer` | `include/server.h` | D 侧监听、注册 MR、处理 RPC（含 CRC） |
| `BarexCtx` | `protocol/barex_protocol.h` | accl-barex 上下文、MR 池、RPC 回调 |

### 3.4 服务发现（Naming）

| 类型 | 文件 | 职责 |
|------|------|------|
| `INamingClient` | `include/naming.h` | KV 存储：`store` / `get` / `search` / `list` |
| `INamingWorkerClient` | `naming.h` | `register_worker` / `get_worker_info` |
| `NamingManager` | `naming.h` | URL 路由：`file:`、`eas:`、`fake:` |
| `FileSysNaming` | `naming/filesys_naming.h` | 文件系统命名（常用本地/测试） |
| `EASNamingClient` | `naming/eas_naming.h` | 阿里云 EAS 命名服务 |

Naming 中两类 key 的语义：

| Key 模式 | 注册方 | 用途 |
|----------|--------|------|
| `endpoint{dprank}` | P/D 的 **Scheduler**（RPC 控制面） | 对端发现 RPC 地址 `(addr, port)` |
| `worker_{worker_id}` | D 的 **Worker**（数据面） | P 发送 KV 时解析 RDMA/TCP 端点 |

### 3.5 KV 布局解析（parse_block）

不同模型/后端的 KV cache 内存布局不同，通过 **Cache Shape** 选择解析器：

| 常量 | 值 | 源文件 | 说明 |
|------|-----|--------|------|
| `RAGGED_FLASH_CACHE_SHAPE` | 1 | `parse_block_ragged.cpp` | 旧 ragged 布局 |
| `FLASH_CACHE_SHAPE` | 2 | `parse_block_flash.cpp` | vLLM 标准 `(2, blocks, ...)` |
| `QWEN3_NEXT_FLASH_CACHE_SHAPE` | 3 | `parse_block_qwen3_next.cpp` | Hybrid GDN + indexer + attn |
| `DPSK_V32_SPARSE_MLA_SHAPE` | 4 | `parse_block_dpsk.cpp` | 稀疏 MLA |
| `FLASHINFER_CACHE_SHAPE` | 5 | `parse_block_flashinfer.cpp` | HND head-major |
| `TURBOQUANT_CACHE_SHAPE` | 6 | `parse_block_turboquant.cpp` | TurboQuant |
| `QWEN3_NEXT_FLASHINFER_CACHE_SHAPE` | 7 | `parse_block_qwen3_next_flashinfer.cpp` | Qwen3.5 + FlashInfer |

每个 shape 通常有 **P==D** (`PEQD`)、**P>D** (`PGTD`)、**P<D** (`PLTD`) 变体，由 `TaskContext::refresh_dst_info()` 根据源/目的 TP 大小选择。

**新路径**（`BLLM_KVTRANS_TX_PARSE_MODE=cache_spec`）：

- `AttnLayoutDesc` + `build_transfer_plan()` + `generate_all_ipc_blocks()` 替代逐 shape 的函数指针，详见 `kvtransfer/docs/cache_transfer_spec.md`。

### 3.6 Python 层

| 模块 | 职责 |
|------|------|
| `kv_transfer.py` | `kvtransfer_ops` 缺失时提供 stub，避免 vLLM 启动失败 |
| `kv_transfer_impl.py` | 封装 C++ API；管理每层 CUDA Event；`send_done` RPC 通知调度器 |
| `kvtransfer_pybind.cpp` | 导出 `init_kv_transfer_client/server`、`connect_naming`、send API |

---

## 4. 数据面：KV 传输流程

### 4.1 初始化

```mermaid
sequenceDiagram
    participant Py as Python (vLLM Worker)
    participant Ops as kvtransfer_ops
    participant Ctx as Context
    participant Nam as NamingManager

    alt Prefill (P)
        Py->>Ops: init_kv_transfer_client(layers, events, protocols)
        Ops->>Ctx: create_context + CUDA barrier
        Py->>Ops: connect_naming(inst_id, naming_url)
    else Decode (D)
        Py->>Ops: init_kv_transfer_server(layers, protocols)
        Ops->>Ctx: create_context
        Ops->>Nam: register_worker_info → worker_{id}
    end
```

### 4.2 每个 Scheduler Step 的发送（P → D）

```mermaid
sequenceDiagram
    participant Sched as PBackend Scheduler
    participant Worker as PBackend Worker
    participant Cli as KvTransferClient
    participant Stub as KvSendStub
    participant Ch as IChannel
    participant Dec as D KVTransferServer

    Sched->>Sched: build_backend_meta → PReqMeta / KVTPMeta
    Sched->>Worker: bind_backend_metadata
    Worker->>Cli: submit_req_send2 / submit_delta_send
    Worker->>Cli: start_send_step(stepid)
    loop 每层 forward
        Worker->>Cli: record_event(layer_idx)
    end
    Cli->>Stub: send_batch (线程池)
    Stub->>Stub: parse_block / TransferPlan → IpcBlocks
    Stub->>Ch: connect + register_data
    loop 每层
        Stub->>Ch: send_data(layer)
        Ch->>Dec: RDMA Write / TCP
    end
    Stub->>Ch: flush()
    Stub->>Sched: send_done RPC (reach_last_token)
    Worker->>Cli: flush_send_step
```

### 4.3 Block 解析决策树

```
env_cache_shape()  ← BLLM_KVTRANS_CACHE_SHAPE
    │
    ├─ TX_PARSE_MODE == cache_spec ?
    │     └─ build_transfer_plan() → generate_all_ipc_blocks()
    │
    └─ Legacy: 按 P/D TP 比较
          ├─ P == D  → *_p_eq_d  (TPKind::PEQD)
          ├─ P > D   → *_p_gt_d  (TPKind::PGTD)
          └─ P < D   → *_p_lt_d  (TPKind::PLTD)
```

`compute_valid_ranks_pd()` 在 `num_kv_heads < engine_tp_size` 时过滤哪些 P rank 参与发送。

---

## 5. 与 vLLM 的对接

### 5.1 启用方式

vLLM 通过 **KV Connector** 机制接入，需配置：

```json
{
  "kv_connector": "HybridConnector",
  "kv_role": "kv_producer",          // P 节点
  "kv_connector_extra_config": {
    "backend": "kvt",
    "naming_url": "file:/path/to/naming",
    "kvt_inst_id": "prefill-cluster-0"
  }
}
```

D 节点将 `kv_role` 设为 `kv_consumer`，`kvt_inst_id` 设为 decode 集群 ID。

后端类选择逻辑（`vllm/v1/hybrid_connector/__init__.py`）：

```python
# backend == "kvt"
if is_kv_consumer:
    return DBackend
elif is_kv_producer:
    return PBackend
```

### 5.2 分层架构

```
EngineCore
    │ on_add_req / step / schedule
    ▼
HybridConnector (__init__.py)
    ├── HybridScheduler  → PBackend / DBackend (SCHEDULER 角色)
    └── HybridWorker     → PBackend / DBackend (WORKER 角色)
            │
            ├── 控制面：RPC (PREFILL_REQ, TRANSFER_KV_REQ, ABORT_REQS_REQ)
            └── 数据面：blade_kvt.KVTransferClient / KVTransferServer
```

| 组件 | 文件 | 职责 |
|------|------|------|
| `HybridConnector` | `__init__.py` | vLLM `KVConnectorBase_V1` 门面 |
| `HybridScheduler` | `__init__.py` | 请求队列 `_waiting/_loading/_loaded`、step、RPC 处理 |
| `HybridWorker` | `__init__.py` | `bind_backend_metadata`、层 hook `async_save_kv_layer` |
| `PBackend` | `kvtbackend.py` | P：RPC 服务、KV 发送、Dash 远程 decode |
| `DBackend` | `kvtbackend.py` | D：Peer 发现、向 P 发 prefill RPC、KV 接收 |
| `engine_proxy.py` | | `core_add_req`、`sched_add_req`、RPC server 端口 |

### 5.3 Python ↔ C++ 绑定要点

**P Worker** `PBackend.register_kv_caches()`：

- 按模型类型设置 `BLLM_KVTRANS_CACHE_SHAPE`（见 `_set_worker_envs`）
- 创建 `KVTransferClient`，在 `async_save_kv_layer` 中 `record_event`
- `connect_naming(inst_id, naming_url)`；P Scheduler 调用 `_reg_naming` 注册 `endpoint{dprank}`

**D Worker** `DBackend.register_kv_caches()`：

- 创建 `KVTransferServer`
- 将 `worker_{id}` 序列化 `WorkerInfo` 写入 naming；若无 naming，通过 RPC `_REGISTER_WORKER` 上报给 D Scheduler

**每步发送** `PBackend.bind_backend_metadata(KVTPMeta)`：

- `freeze_metas` → `start_req_send`
- `abort_metas` → `submit_delta_send`（收尾）
- `nonfreeze_metas` → `submit_req_send2` / `submit_delta_send`
- `start_send_step` + 各层 `record_event` + `flush_send_step`

**完成通知**：C++ `KvSendStub::send_done()` 向 `BLLM_KVTRANS_SEND_DONE_ADDR`（默认 P Scheduler RPC 端口）发送 `SEND_DONE_REQ`，P 侧 `_mark_send_done` 后向等待中的 D RPC 返回 `KVTResp`。

---

## 6. P/D 实例发现

发现分为 **两层**：控制面（RPC 端点）和数据面（KV worker 端点）。

### 6.1 实例 ID

```python
# kvtbackend._get_inst_id()
kvt_inst_id = kv_connector_extra_config["kvt_inst_id"]
              or POD_NAME
              or uuid

# 完整传输/RPC 身份（含 DP/TP）
full_id = f"{kvt_inst_id}|{data_parallel_rank}|{tensor_parallel_size}"
```

### 6.2 控制面：D 如何找到 P（Prefill RPC）

**P 注册**（Scheduler，`_reg_naming`）：

```python
PeerInfo(role="prefill", tpsize, addr, port=rpc_port, dprank)
naming_cli.store(f"endpoint{dprank}", info.serialize())
```

**D 发现**（`DBackend` + `PeerManager`）：

- 每 **7 秒** 轮询：`naming_cli.list()` → 对每个 instance `search(inst, "endpoint")`
- 过滤 `role == "prefill"`
- 维护 `PeerMap`：`peer_id = f"{kvt_inst_id}|{dprank}|{tpsize}"` → `(addr, port)`
- `get_peer(exclude=self.my_addr_port)` 随机选一个可用 P（避免选自己）

**显式指定 P**（Dash / 固定 P）：

- 请求参数 `remote_host` + `remote_port` → `peer_hint`
- abort 时记录 `D_PID = "{addr}:{port}"` 用于 `_abort_prefill`

### 6.3 数据面：P 如何找到 D Worker（KV RDMA/TCP）

**D Worker 注册**：

```python
winfo = bladekv.current_worker_info('server')
naming_cli.store(f"worker_{worker_id}", winfo)
```

**经 RPC 传递给 P**（`KVTDInfo`）：

```python
KVTDInfo(
    instid=f"{d_inst_id}|{dprank}|{tpsize}",
    blkids=...,           # D 侧 block 映射
    cached_tokens=...,
    max_tokens=...,
    d_workers_info=[...], # 各 TP rank 的 WorkerInfo 序列化串
)
```

**P 侧映射**（`_get_distinfo` / `_get_dist`）：

- 解析 `d_inst_id` 得到 D 的 inst / dprank / tpsize
- 按 P_tp 与 D_tp 计算 `dst_worker_id`（支持 P_tp > D_tp 的多对一）
- `PReqMeta` 携带 `d_workers_info` 时，C++ 可 **跳过 naming 查询** 直接使用

**C++ 回退**：`tx_stub.cpp` 中若未提供 `dst_worker_info`，则 `naming_->get_worker_info(dst_inst, dst_worker_id)`。

### 6.4 Fake Naming 模式（`naming_url: "fake://"`）

| 组件 | 行为 |
|------|------|
| P | 无 naming；`inst_id = POD_NAME-rpc_port-ip` |
| D Scheduler | 无 `PeerManager`；请求必须带 `remote_host`/`remote_port` |
| D Worker | RPC `_REGISTER_WORKER` 注册到 D Scheduler；阻塞直到所有 TP rank 就绪 |

### 6.5 Naming 后端

| Schema | 实现 |
|--------|------|
| `file://` | `filesys_naming.cpp`，目录 per instance，带 keepalive 时间戳 |
| `eas://` | `eas_naming.cpp`，HTTP |
| `fake://` | 本地测试，无真实存储 |

---

## 7. 请求生命周期与状态维护

### 7.1 请求参数（kv_transfer_params）

| Key | 常量 | 设置方 | 含义 |
|-----|------|--------|------|
| `ali_llumnix_disagg` | `D_DISAGG` | 默认 | 启用 PD 分离 |
| `do_remote_prefill` | `D_REMOTE_PREFILL` | Dash | D 向指定 P prefill |
| `do_remote_decode` | `P_REMOTE_DECODE` | Dash | P prefill 后 KV 交给 D decode |
| `remote_host` / `remote_port` | | 调用方 | 固定 P 的 RPC 地址 |
| `hbpkvtstate` | `P_KVT_STATE` | P Scheduler | `KVTState(dinfo, maxtokens, untouched)` |
| `hbpkvtdinfo` | `P_KVTD_INFO` | D→P RPC | 目标 block、cached tokens、workers |
| `_hbkvtpid` | `D_PID` | D | abort 时用的 P 地址 |
| `ignore_output` | `P_IGNORE_OUTPUTS` | P | 远程 decode 时不向客户端输出 |

### 7.2 D（Decode）标准 Disagg 流程

```
Client Request (disagg=true, prompt>1)
    │
    ▼
HybridConnector.on_add_req()
    │ get_operations() → load=1
    ▼
HybridScheduler._waiting
    │ sched_allocate_slots()
    ▼
_on_add_req() [disagg asyncio 线程]
    │ async_get_num_new_matched_tokens()  → 需远程 prefill 的 token 数
    │ async_update_state_after_alloc()
    │   └─ _prefill_rpc(PREFILL_REQ) 或 _dash_prefill_rpc(TRANSFER_KV_REQ)
    ▼
mark_loaded() → _loaded
    │ num_computed_tokens += cached
    ▼
sched_add_req() → 正常 vLLM decode 调度
```

**Hybrid 模型 token 切分**：D 保留最后 `gamma+1` 个 token 本地计算（`gamma = num_speculative_tokens`），远程只 prefill 前面部分。

### 7.3 P（Prefill）两种模式

**模式 A：D 驱动 Prefill**（`PREFILL_REQ`）

1. D RPC 发送 `EngineCoreRequest` + `P_KVTD_INFO`
2. P 设置 `max_tokens=1`、`P_IGNORE_OUTPUTS=True`、`P_KVT_STATE`
3. P 本地 prefill forward
4. `build_backend_meta` → `PReqMeta` → Worker 经 `KVTransferClient` 发送到 D
5. `_wait_kvt_state` 返回 `KVTResp(cached=N)` 给 D

**模式 B：Dash 远程 Decode**（`do_remote_decode=True`）

1. 客户端请求直达 P
2. P `max_tokens=1` prefill
3. D 稍后发 `TRANSFER_KV_REQ`（`RKVTDInfo`）
4. P 经 `_dinfoq` / `_dash_done` 匹配并发 KV

### 7.4 P 每步 KV 发送元数据

```python
@dataclass
class PReqMeta:
    reqid: str
    d_inst_id: str          # "{inst}|{dprank}|{tpsize}"
    p_block_ids / d_block_ids
    new_tokens: int
    has_last_token: bool
    seen_tokens: int        # 0 → submit_req_send; >0 → submit_delta_send
    d_workers_info: list[str]
    has_freeze: bool
```

Scheduler 每步产出 `KVTPMeta`：`stepid`, `substepid`, `freeze_metas`, `abort_metas`, `nonfreeze_metas`。

### 7.5 HybridScheduler 队列（请求维护）

| 结构 | 用途 |
|------|------|
| `_waiting` | 等待 KV 槽位分配与 load/save 设置 |
| `_loading` / `_loaded` | D 侧 KV 加载中 / 完成 |
| `_saving` / `_saved` | P 侧 KV 保存中 / 完成 |
| `_sending` (P) | 等待 `KVTResp` 的 RPC |
| `_infly_kvt` (P) | 多步 delta 发送中的请求 |
| `_dash_done` (P) | Dash：等待 D 的 TRANSFER_KV |
| `_dinfoq` (P) | 来自 D 的 `RKVTDInfo` 队列 |
| `_aborting` | 跨队列的 abort 协调 |

### 7.6 RPC 协议常量（节选）

| 常量 | 值 | 方向 | 说明 |
|------|-----|------|------|
| `PREFILL_REQ` | 0x20181218 | D→P | 携带 `EngineCoreRequest` |
| `PREFILL_RESP` | 0x81218102 | P→D | `KVTResp` |
| `TRANSFER_KV_REQ` | 0x20210912 | D→P | Dash：仅传 KV |
| `SEND_DONE_REQ` | 0x20181219 | C++→P Sched | KV 发送完成 |
| `ABORT_REQS_REQ` | 20250820 | D→P | 加载失败时 abort |

### 7.7 端到端时序（标准 Disagg）

```mermaid
sequenceDiagram
    participant Client
    participant D as D (Decode)
    participant Nam as Naming
    participant P as P (Prefill)

    Client->>D: Request (disagg=true)
    D->>D: Allocate KV blocks
    D->>Nam: PeerManager.get_peer()
    Nam-->>D: P endpoint (addr, port)
    D->>P: PREFILL_REQ + KVTDInfo
    P->>P: core_add_req, prefill forward
    P->>D: KVTransfer RDMA/TCP (multi-layer)
    P->>D: KVTResp(cached=N)
    D->>D: sched_add_req, local decode
    Client->>D: Continue generation
```

---

## 8. 实例与运行时状态

### 8.1 PeerManager 生命周期

```
每 7s:
  list() 所有 naming instance
  对每个 instance: search("endpoint*")
  与 _running 对比:
    - 消失 → 移除 peer + 关闭连接池
    - ctime_us 变化 → 原地重启检测，更新 peer + 清连接
    - 新增 → 加入 PeerMap
```

### 8.2 P 侧单请求状态机

```
请求进入 P
    │
    ├─ do_remote_decode? → 进入 _dash_done，等待 TRANSFER_KV
    │
    ├─ P_KVT_STATE (来自 PREFILL_REQ)
    │     untouched=True  → 等 scheduled tokens 达到 max 再发 KV
    │     untouched=False → 立即 build PReqMeta 发送
    │
    └─ _infly_kvt 跟踪多步 delta
          finished + new_tokens=0 → abort_metas (has_last_token)
```

`KVTState.untouched` 控制首次 KV 传输是否等待 prefill 完全结束。

### 8.3 blade-kvt C++ 侧状态

| 状态 | 所有者 | 生命周期 |
|------|--------|----------|
| `Context` / `WorkerInfo` | 每进程 client 或 server | 进程级 |
| `KV_CLIENT` / `KV_SERVER` | pybind 全局单例 | 进程级 |
| `targets_tasks_buf_` | `KvTransferClient` | 每次 `start_send` 清空 |
| `coord_step_id_`, `pending_step_metas_` | Client | Step/substep 协调 |
| `Target` SendStub LRU | `TargetMgr` | 容量 `env_txstub_cap()` |
| `RequestInfo::state_` | 原子 `ReqState` | INPROCESS → OK/FAILED |
| `TaskContext` | `KvSendStub` | 单次 `send_batch` 后清理 |

### 8.4 失败与 Abort

- D load 异常 → `IoRet.ex` → `AbortReq` → `_abort_prefill()` → `ABORT_REQS_REQ` 到 P
- Dash 模式 `CODE_REQNOTFOUND` → `_delay_s_list` 重试
- `request_finished_all_groups`：远程 decode 时协调 `_PD_FINISH_REASON` 与 `_PD_SAVED` 顺序

---

## 9. 配置与环境变量速查

### 9.1 vLLM kv_transfer_config

| 字段 | 说明 |
|------|------|
| `kv_connector` | `"HybridConnector"` |
| `kv_role` | `"kv_producer"` (P) / `"kv_consumer"` (D) |
| `kv_connector_extra_config.backend` | `"kvt"` |
| `kv_connector_extra_config.naming_url` | `"file:/path"` / `"eas:..."` / `"fake://"` |
| `kv_connector_extra_config.kvt_inst_id` | 逻辑集群名 |

### 9.2 blade-kvt 环境变量（`BLLM_KVTRANS_*`）

| 变量 | 作用 |
|------|------|
| `CACHE_SHAPE` | 选择 parse_block（1–7） |
| `TX_PARSE_MODE` | `cache_spec` 启用 TransferPlan |
| `RDMA_STAGED` | Staged D2H+Send |
| `SEND_DONE_ADDR` | Scheduler RPC `ip:port` |
| `CRC` | KV 完整性校验 |
| `SEND_TPSIZE` / `TXSTUB_CAP` | 发送线程池与 stub 池大小 |
| `BF162FP8_CONV` | 传输时 BF16→FP8 |
| `PORT_BASE` | Server 监听端口基址 |

`kvtbackend._set_worker_envs()` 会根据模型类型（MLA、Hybrid、FlashInfer、TurboQuant）自动设置默认 `CACHE_SHAPE`。

### 9.3 示例启动片段

```bash
# P 节点
export BLLM_KVTRANS_CACHE_SHAPE=7   # Qwen3.5 + FlashInfer
# 通常由 vLLM 根据模型自动设置

# naming 目录需对 P/D 双方可见
# kv_connector_extra_config.naming_url=file:/shared/naming
```

---

## 10. 关键源码索引

### blade-kvt

| 主题 | 路径 |
|------|------|
| 公共类型 | `kvtransfer/include/common.h` |
| Client 编排 | `kvtransfer/include/client.h`, `src/client.cpp` |
| Send 执行 | `kvtransfer/include/tx_stub.h`, `src/tx_stub.cpp` |
| Channel | `kvtransfer/include/channel.h`, `src/rdma_channel.cpp` |
| parse_block | `kvtransfer/include/parse_block_common.h`, `src/parse_block_*.cpp` |
| TransferPlan | `kvtransfer/docs/cache_transfer_spec.md` |
| pybind | `kvtransfer/kvtransfer_pybind.cpp` |
| Python API | `blade_kvt/kv_transfer_impl.py` |

### vLLM

| 主题 | 路径 |
|------|------|
| 后端选择 | `vllm/v1/hybrid_connector/__init__.py` → `_get_backend_cls` |
| P/D 实现 | `vllm/v1/hybrid_connector/kvtbackend.py` |
| Peer 发现 | `vllm/v1/hybrid_connector/utils.py` → `PeerManager` |
| Engine 桥接 | `vllm/v1/hybrid_connector/engine_proxy.py` |
| Dash 示例 | `vllm/examples/dash_proxy.py` |

---

## 附录：当前分支说明

当前 git 分支 `support_3.5_flashinfer` 新增 **`QWEN3_NEXT_FLASHINFER_CACHE_SHAPE` (值 7)**，用于 Qwen3.5 混合模型在 FlashInfer HND 注意力布局下的 KV 解析，对应源文件 `parse_block_qwen3_next_flashinfer.cpp`。

---

*文档生成日期：2026-05-21*
