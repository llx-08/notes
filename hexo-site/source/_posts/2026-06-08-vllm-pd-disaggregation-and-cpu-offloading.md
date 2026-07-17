---
title: vLLM PD 分离与 CPU Offloading 分析
date: 2026-06-08
tags: []
---

# vLLM PD 分离与 CPU Offloading 分析

> 基于 vLLM `llx/github` 分支代码阅读整理
> 日期：2026-06-08

---

## 目录

1. [概述](#1-概述)
2. [PD 分离（Prefill-Decode Disaggregation）](#2-pd-分离prefill-decode-disaggregation)
   - 2.1 [核心思想](#21-核心思想)
   - 2.2 [关键文件](#22-关键文件)
   - 2.3 [KVConnector 插件架构](#23-kvconnector-插件架构)
   - 2.4 [NIXL 连接器实现](#24-nixl-连接器实现)
   - 2.5 [端到端数据流](#25-端到端数据流)
   - 2.6 [调度器集成](#26-调度器集成)
   - 2.7 [Worker 侧 KV 传输](#27-worker-侧-kv-传输)
   - 2.8 [租约与心跳机制](#28-租约与心跳机制)
   - 2.9 [异构 TP 支持](#29-异构-tp-支持)
   - 2.10 [双向 KV 传输](#210-双向-kv-传输)
   - 2.11 [错误处理](#211-错误处理)
   - 2.12 [支持的传输协议](#212-支持的传输协议)
3. [CPU Offloading](#3-cpu-offloading)
   - 3.1 [核心思想](#31-核心思想)
   - 3.2 [关键文件](#32-关键文件)
   - 3.3 [Store 流程（GPU → CPU）](#33-store-流程gpu--cpu)
   - 3.4 [Load 流程（CPU → GPU）](#34-load-流程cpu--gpu)
   - 3.5 [CPU 内存管理](#35-cpu-内存管理)
   - 3.6 [淘汰策略](#36-淘汰策略)
   - 3.7 [异步传输与 CUDA Stream](#37-异步传输与-cuda-stream)
   - 3.8 [与 GPU 前缀缓存的交互](#38-与-gpu-前缀缓存的交互)
   - 3.9 [配置参数](#39-配置参数)
4. [PD 分离与 CPU Offloading 的关系](#4-pd-分离与-cpu-offloading-的关系)
   - 4.1 [架构共性](#41-架构共性)
   - 4.2 [核心差异](#42-核心差异)
   - 4.3 [互补性分析](#43-互补性分析)

---

## 1. 概述

vLLM 的 KV cache 管理系统通过统一的 `KVConnectorBase_V1` 插件接口，支持两种核心的 KV cache 优化策略：

- **PD 分离（Prefill-Decode Disaggregation）**：将推理过程中的 prefill 阶段和 decode 阶段分离到不同的节点执行，P 节点计算 KV cache 后通过 RDMA 传输给 D 节点，避免 D 节点重复计算，降低首 token 延迟（TTFT）。
- **CPU Offloading**：将 GPU 上暂时不活跃的 KV cache blocks 卸载到 CPU pinned memory，需要时再异步加载回 GPU，本质上用 CPU 内存扩展 KV cache 容量，支持更高的并发请求数。

两者都通过 `--kv-connector` 参数启用，共享同一套调度器/Worker 集成接口，但解决的问题维度不同。

---

## 2. PD 分离（Prefill-Decode Disaggregation）

### 2.1 核心思想

传统 LLM 推理中，prefill 和 decode 共享同一个 GPU：

- **Prefill**：一次性处理所有 prompt tokens，计算量大（compute-bound），产出完整的 KV cache
- **Decode**：逐 token 自回归生成，计算量小但对延迟敏感（memory-bound）

当 prefill 请求和 decode 请求混合调度时，长 prompt 的 prefill 会抢占 GPU 算力，导致正在 decode 的请求延迟飙升。PD 分离的核心思路是：

```
P 节点（Prefill-only）          D 节点（Decode-only）
┌─────────────────┐            ┌─────────────────┐
│ 接收用户请求      │            │ 接收 P 的 KV params│
│ 执行 prefill      │            │ 通过 RDMA 拉取 KV  │
│ 计算 KV cache     │──RDMA──→  │ 直接开始 decode    │
│ 返回 kv_params    │            │ 逐 token 输出      │
└─────────────────┘            └─────────────────┘
```

P 节点专注于高吞吐的 prefill 批处理，D 节点专注于低延迟的 decode，互不干扰。

### 2.2 关键文件

```
vllm/
├── config/kv_transfer.py                          # KVTransferConfig 配置
├── distributed/kv_transfer/kv_connector/
│   ├── factory.py                                  # KVConnectorFactory 工厂
│   └── v1/
│       ├── base.py                                 # KVConnectorBase_V1 基类
│       └── nixl/
│           ├── connector.py                        # NixlConnector 门面类
│           ├── scheduler.py                        # NixlConnectorScheduler
│           ├── worker.py                           # NixlConnectorWorker (2483行)
│           ├── metadata.py                         # 数据结构定义
│           └── tp_mapping.py                       # TPMapping 异构TP映射
├── v1/
│   ├── core/sched/scheduler.py                     # 调度器集成
│   ├── worker/kv_connector_model_runner_mixin.py   # ModelRunner 集成
│   └── request.py                                  # WAITING_FOR_REMOTE_KVS 状态
└── entrypoints/serve/disagg/
    ├── api_router.py                               # /inference/v1/generate 端点
    ├── protocol.py                                 # GenerateRequest/Response
    └── serving.py                                  # ServingTokens 服务层
```

### 2.3 KVConnector 插件架构

所有 KV 传输方案都实现统一的 `KVConnectorBase_V1` 接口：

```python
class KVConnectorBase_V1:
    """Base class for V1 KV connectors."""

    # ─── 调度器侧接口 ───
    def get_num_new_matched_tokens(self, request, num_computed_tokens, ...):
        """查询外部是否有该请求的 KV cache 可用"""
        ...

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """block 分配后更新连接器状态"""
        ...

    def build_connector_meta(self, scheduler_output):
        """构建传给 worker 的元数据"""
        ...

    def request_finished(self, request, block_ids):
        """请求完成时的清理工作"""
        ...

    # ─── Worker 侧接口 ───
    def start_load_kv(self, metadata):
        """发起异步 KV 加载（RDMA read / memcpy）"""
        ...

    def wait_for_layer_load(self):
        """等待当前层的 KV 加载完成"""
        ...

    def save_kv_layer(self, layer_name, kv_cache, ...):
        """保存一层的 KV cache"""
        ...

    def wait_for_save(self):
        """等待所有保存操作完成"""
        ...

    def get_finished(self):
        """轮询已完成的传输，返回 KVConnectorOutput"""
        ...
```

`KVConnectorFactory` 通过注册表创建连接器实例，每个连接器会创建两个实例：`role=SCHEDULER` 和 `role=WORKER`。

### 2.4 NIXL 连接器实现

NIXL（NVIDIA Interconnect eXchange Library）是 PD 分离的主力传输层，基于 UCX 实现 RDMA 传输。

#### NixlConnectorScheduler

调度器侧管理请求的 KV 传输生命周期：

```python
class NixlConnectorScheduler:
    _reqs_need_recv: dict   # 需要接收 KV 的请求（D节点）
    _reqs_need_save: dict   # 需要保存 KV 的请求
    _reqs_need_send: dict   # 需要发送 KV 的请求（P节点）

    def get_num_new_matched_tokens(self, request, ...):
        """检测 kv_transfer_params 中的 do_remote_prefill 标记"""
        if request.kv_transfer_params.get("do_remote_prefill"):
            return (num_prompt_tokens - num_computed_tokens, True)  # async
        return (0, False)

    def update_state_after_alloc(self, request, blocks, ...):
        """将请求加入 _reqs_need_recv，记录本地 block IDs 和远程元数据"""
        self._reqs_need_recv[req_id] = ReqMeta(
            local_block_ids=blocks.block_ids,
            remote=RemoteMeta(
                engine_id=params["remote_engine_id"],
                block_ids=params["remote_block_ids"],
                ...
            )
        )

    def request_finished(self, request, block_ids):
        """P节点：delay_free_blocks=True，等待D节点通知后才释放"""
        """D节点：正常释放"""
        ...
```

#### NixlConnectorWorker

Worker 侧执行实际的 RDMA 数据传输（2483 行）：

```python
class NixlConnectorWorker:
    def __init__(self):
        self.nixl_wrapper = NixlWrapper(...)     # NIXL 库封装
        self._handshake_thread = Thread(...)      # 后台 ZMQ 握手线程
        self._bg_transfer_thread = Thread(...)    # 后台传输完成检查

    def start_load_kv(self, metadata):
        """发起异步 RDMA READ"""
        for req in metadata.reqs_to_recv:
            # 1. 确保已与远程引擎完成握手
            self._ensure_handshake(req.remote_engine_id, req.remote_host, req.remote_port)
            # 2. 计算 desc_ids：region_id * num_blocks + block_id
            local_descs = self._compute_desc_ids(req.local_block_ids)
            remote_descs = self._compute_desc_ids(req.remote_block_ids)
            # 3. 创建预备传输
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ", local_descs, remote_descs, remote_agent
            )
            # 4. 发起异步传输
            self.nixl_wrapper.transfer(handle)

    def get_finished(self):
        """轮询传输状态"""
        for handle in self._pending_transfers:
            state = self.nixl_wrapper.check_xfer_state(handle)
            if state == "DONE":
                # 发通知给P节点，让P释放blocks
                self.nixl_wrapper.send_notif(
                    remote_agent, f"{remote_req_id}:{world_size}"
                )
```

#### 握手过程

P 和 D 首次通信前需要交换 NIXL agent 元数据：

```
D节点                          P节点
  │                              │
  ├─ ZMQ REQ ──────────────────→ │  发送 NixlHandshakePayload:
  │  (agent_metadata,            │   - agent_metadata (RDMA 地址)
  │   base_addr, compat_hash)    │   - base_addr (KV cache 基地址)
  │                              │   - compat_hash (版本兼容校验)
  │ ←────────────────── ZMQ REP ─┤  返回对方的 payload
  │                              │
  ├─ nixl_wrapper.import_remote()│  注册远程 agent
  └─ 握手完成，可以发起 RDMA     │
```

### 2.5 端到端数据流

#### Step 1: 请求到达 Proxy

外部 proxy（如 `disagg_proxy_demo.py`）接收用户请求，决定路由策略：

```python
# Proxy 逻辑
async def handle_request(request):
    # 1. 发给 P 节点做 prefill (max_tokens=1)
    prefill_response = await send_to_p_node(request, max_tokens=1)

    # 2. 从 P 的响应中提取 kv_transfer_params
    kv_params = prefill_response.kv_transfer_params

    # 3. 发给 D 节点做 decode，携带 kv_transfer_params
    decode_response = await send_to_d_node(request, kv_transfer_params=kv_params)
    return decode_response
```

#### Step 2: P 节点执行 Prefill

```
P节点收到请求 (max_tokens=1)
  │
  ├─ 正常 prefill，计算所有 prompt token 的 KV cache
  │
  ├─ 请求完成时 request_finished():
  │   ├─ delay_free_blocks = True     # 不释放 blocks！
  │   └─ 构建 kv_transfer_params:
  │       {
  │         "do_remote_prefill": True,
  │         "remote_block_ids": [0, 1, 2, ...],
  │         "remote_engine_id": "engine-abc",
  │         "remote_host": "10.0.0.1",
  │         "remote_port": 5678,
  │         "tp_size": 8,
  │         "remote_num_tokens": 1024
  │       }
  │
  └─ 返回 RequestOutput（包含 kv_transfer_params）给 Proxy
```

#### Step 3: D 节点接收并发起 KV 传输

```
D节点收到请求（携带 kv_transfer_params）
  │
  ├─ Scheduler: get_num_new_matched_tokens()
  │   ├─ 检测 do_remote_prefill = True
  │   └─ 返回 (num_prompt_tokens, async=True)
  │
  ├─ Scheduler: 分配本地 GPU blocks
  │
  ├─ Scheduler: update_state_after_alloc()
  │   ├─ 记录本地 block IDs + 远程元数据
  │   └─ 设置 do_remote_prefill = False（防止重复触发）
  │
  ├─ Request.status → WAITING_FOR_REMOTE_KVS
  │   （请求被放入 skipped_waiting 队列）
  │
  ├─ Worker: start_load_kv()
  │   ├─ 握手（如果首次连接）
  │   ├─ nixl_wrapper.make_prepped_xfer("READ", local_descs, remote_descs)
  │   └─ nixl_wrapper.transfer(handle)  # 异步 RDMA READ
  │
  ├─ 后续 step: get_finished() 轮询
  │   ├─ check_xfer_state(handle) == "DONE"
  │   └─ send_notif() 通知 P 节点释放 blocks
  │
  ├─ Scheduler: _update_from_kv_xfer_finished()
  │   └─ 将 req_id 加入 finished_recving_kv_req_ids
  │
  ├─ Scheduler: _try_promote_blocked_waiting_request()
  │   └─ WAITING_FOR_REMOTE_KVS → WAITING → RUNNING
  │
  └─ 正常 decode 开始，输出 token
```

#### Step 4: P 节点释放 Blocks

```
P节点后台:
  _get_new_notifs()
    ├─ 收到 D 的通知: "{request_id}:{world_size}"
    ├─ 跟踪 consumer 计数（异构 TP 场景可能有多个 consumer）
    ├─ 所有 consumer 都读取完成 → done_sending
    └─ 释放 KV cache blocks
```

### 2.6 调度器集成

调度器（`vllm/v1/core/sched/scheduler.py`）通过以下关键点集成 PD 分离：

#### 初始化

```python
class Scheduler:
    def __init__(self, ...):
        # 创建 KV connector（调度器侧）
        self.connector = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.SCHEDULER,
        )
```

#### 调度循环

```python
def schedule(self):
    # 1. 检查已完成的 KV 传输
    self._update_from_kv_xfer_finished()

    # 2. 对等待中的请求，查询是否有远程 KV 可用
    for req in waiting_requests:
        num_external, is_async = self.connector.get_num_new_matched_tokens(req, ...)
        if num_external > 0:
            # 分配 blocks
            blocks = self.block_manager.allocate(req, ...)
            self.connector.update_state_after_alloc(req, blocks, num_external)
            if is_async:
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                # 不加入当前 batch，等 KV 传输完成

    # 3. 构建 connector 元数据
    connector_meta = self.connector.build_connector_meta(scheduler_output)

    # 4. 提升已完成 KV 传输的请求
    self._try_promote_blocked_waiting_request()
```

#### WAITING_FOR_REMOTE_KVS 状态机

```
WAITING ──(发现远程KV)──→ WAITING_FOR_REMOTE_KVS ──(KV传输完成)──→ WAITING ──→ RUNNING
                              │                                        ↑
                              └────────(传输失败, recompute策略)────────┘
                                        (回退到本地重算)
```

### 2.7 Worker 侧 KV 传输

ModelRunner 通过 mixin 类集成 KV 连接器：

```python
# vllm/v1/worker/kv_connector_model_runner_mixin.py

class KVConnectorModelRunnerMixin:
    @contextmanager
    def _get_kv_connector_output(self):
        """包裹 forward pass 的上下文管理器"""
        # 1. 绑定 connector 元数据
        self.connector.bind_metadata(metadata)

        # 2. 启动异步 KV 加载（RDMA read 开始）
        self.connector.start_load_kv()

        # 3. 执行模型 forward pass
        yield kv_connector_output

        # 4. 等待 KV 保存完成
        self.connector.wait_for_save()

        # 5. 收集已完成的传输
        output = self.connector.get_finished()
```

### 2.8 租约与心跳机制

P 节点的 KV blocks 不能无限持有，需要防止 D 节点崩溃后 blocks 永远不释放：

```
D节点                                    P节点
  │                                        │
  ├─ 首次请求 ────────────────────────────→│  blocks 锁定，租约 30s
  │                                        │
  │  (每 ~5s)                              │
  ├─ 心跳 "HB:req1,req2,..." ────────────→│  续租
  │                                        │
  ├─ KV传输完成                            │
  ├─ 通知 "{req_id}:{world_size}" ───────→│  释放 blocks
  │                                        │
  │  (如果D崩溃，心跳停止)                  │
  │                                        ├─ 租约到期（30s）
  │                                        └─ 自动释放 blocks
```

配置参数：`kv_lease_duration`（默认 30s）

### 2.9 异构 TP 支持

P 和 D 可以使用不同的 tensor parallelism 大小（如 P 用 TP=4，D 用 TP=8）。

`TPMapping`（`nixl/tp_mapping.py`）处理映射逻辑：

```python
class TPMapping:
    def __init__(self, local_tp_size, remote_tp_size, num_kv_heads, ...):
        """计算每个本地 rank 需要从哪些远程 rank 读取"""
        # 示例：P TP=4, D TP=8
        # P rank 0 管 head 0-3, P rank 1 管 head 4-7, ...
        # D rank 0 管 head 0-1, D rank 1 管 head 2-3, ...
        # D rank 0 需要从 P rank 0 读取 head 0-1 的数据

    def get_remote_ranks_for_local(self, local_rank):
        """返回本地 rank 需要读取的远程 rank 列表"""
        ...
```

支持 GQA head 去重和 MLA cache 复制。

### 2.10 双向 KV 传输

启用 `bidirectional_kv_xfer=True` 后，D 节点也能作为 KV 源：

**场景：多轮对话**

```
轮次1: P节点 prefill "你好" → KV传给D → D decode "你好！有什么..."
轮次2: 用户追问 "帮我写代码"
        D节点已有轮次1的 KV cache（decode 过程中积累的）
        新 P 节点可以从 D 节点拉取已有 KV，只需 prefill 增量部分
```

通过 `kv_transfer_params` 中的 `do_remote_decode` 标记触发。

### 2.11 错误处理

```python
# kv_load_failure_policy 配置
"recompute"  # 传输失败 → 回退到本地重新 prefill（默认）
"fail"       # 传输失败 → 请求报错

# 具体错误场景
# - 握手失败 → 请求加入 _failed_recv_reqs
# - 传输超时 → 标记 block 为 invalid
# - 兼容性不匹配 → compat_hash 校验失败，拒绝握手
```

### 2.12 支持的传输协议

`KVConnectorFactory` 注册了 14+ 种连接器：

| 连接器 | 传输方式 | 说明 |
|--------|---------|------|
| `NixlConnector` | RDMA (UCX) | 主力生产连接器 |
| `P2pNcclConnector` | NCCL P2P | 简单场景（1P1D） |
| `LMCacheConnectorV1` | LMCache | 外部 KV cache 存储 |
| `MooncakeConnector` | Mooncake | 分布式存储 |
| `HF3FSKVConnector` | HF 3FS | 文件系统传输 |
| `MoRIIOConnector` | MoRIIO | MoRIIO 引擎 |
| `FlexKVConnectorV1` | 灵活 KV | 可配置传输 |
| `OffloadingConnector` | CUDA memcpy | CPU offloading（见第3节） |

---

## 3. CPU Offloading

### 3.1 核心思想

LLM 推理中，KV cache 是显存消耗的主要来源。当并发请求增多时，GPU 显存不足以容纳所有请求的 KV cache。CPU Offloading 的思路是：

```
GPU 显存（快但小）           CPU 内存（慢但大）
┌──────────────┐            ┌──────────────────┐
│ 活跃请求的     │            │ 不活跃请求的       │
│ KV cache      │  ←──────→  │ KV cache          │
│              │  异步拷贝   │ (pinned memory)   │
└──────────────┘            └──────────────────┘

- 请求暂时不活跃 → KV blocks 异步卸载到 CPU（Store）
- 请求再次被调度 → KV blocks 异步加载回 GPU（Load）
- 效果：用延迟换容量，支持更多并发
```

与传统的 swap-out/swap-in 不同，vLLM 的 CPU Offloading 实现为一个**前缀缓存系统**：它不是简单地挪走再挪回，而是在 CPU 侧维护一个带有淘汰策略的 block 缓存，相同前缀的请求可以直接命中 CPU 缓存。

### 3.2 关键文件

```
vllm/
├── distributed/kv_transfer/kv_connector/v1/
│   ├── offloading_connector.py              # OffloadingConnector 入口
│   └── offloading/
│       ├── scheduler.py                      # OffloadingConnectorScheduler
│       ├── worker.py                         # OffloadingConnectorWorker
│       └── common.py                         # 公共数据结构
├── v1/kv_offload/
│   ├── base.py                               # 抽象基类
│   ├── factory.py                            # Spec 注册工厂
│   ├── file_mapper.py                        # 磁盘分层（SSD tiering）
│   ├── cpu/
│   │   ├── manager.py                        # CPUOffloadingManager（调度器侧）
│   │   ├── gpu_worker.py                     # 实际 CUDA 拷贝
│   │   ├── spec.py                           # CPUOffloadingSpec
│   │   ├── shared_offload_region.py          # 共享 mmap 内存区域
│   │   └── policies/
│   │       ├── base.py                       # CachePolicy 基类
│   │       ├── lru.py                        # LRU 淘汰策略
│   │       └── arc.py                        # ARC 自适应淘汰策略
│   └── worker/
│       └── worker.py                         # OffloadingWorker（调度分发）
└── csrc/libtorch_stable/
    └── cache_kernels.cu                      # swap_blocks_batch CUDA kernel
```

### 3.3 Store 流程（GPU → CPU）

Store 操作将 GPU 上已计算的 KV blocks 卸载到 CPU，为后续相同前缀的请求提供缓存。

#### 触发时机

每个 scheduler step 中，`OffloadingConnectorScheduler._build_store_jobs()` 自动检查：

```python
def _build_store_jobs(self):
    """遍历已调度请求，决定哪些 blocks 需要 offload"""
    for req in scheduled_requests:
        # 计算可卸载的 token 数
        num_offloadable = req.num_computed_tokens + num_scheduled_tokens

        # 默认只卸载 prefill blocks
        if self.offload_prompt_only:
            num_offloadable = min(num_offloadable, req.num_prompt_tokens)

        # 可选：限制每个请求的最大卸载量
        if req.max_offload_tokens:
            num_offloadable = min(num_offloadable, req.max_offload_tokens)

        # 跳过已经在 CPU 缓存中的 blocks（前缀命中）
        # group_state.next_stored_block_idx 跟踪进度
        new_blocks = blocks[group_state.next_stored_block_idx:]

        # 获取 CPU 侧的存储槽位
        cpu_block_ids = self.manager.prepare_store(block_hashes)

        # 生成 store job（延迟提交）
        self._unsubmitted_store_jobs.append(store_job)
```

#### 延迟提交

Store job **不是立即执行**，而是延迟到下一个 engine step 的开头：

```python
def start_kv_transfers(self):
    """在下一个 engine step 开始时提交 store jobs"""
    # 注意：延迟提交是故意的！
    # 原因：offloading 应在 token 采样相关的传输之后启动，
    # 避免增加 token 生成延迟
    for job in self._unsubmitted_store_jobs:
        self.worker.transfer_async(job_id, (gpu_spec, cpu_spec))
    self._unsubmitted_store_jobs.clear()
```

#### 滑动窗口优化

对于混合注意力架构（如 DeepSeek V4 的 MLA + SWA），SWA 层的 KV cache 有固定窗口大小，超出窗口的 blocks 永远不会被复用：

```python
# alignment_block_count 计算
# 对于 100K token 上下文，SWA 窗口 = 4096 tokens
# 可以跳过 ~78% 的 SWA blocks 的 store 操作
swa_skippable = total_blocks - sliding_window_blocks
```

#### Store 阈值

`store_threshold` 参数控制一个 block hash 在 `lookup()` 中被查询多少次后才有资格被 offload。默认为 0（任何 block 都可以 offload），设置为正数可以过滤掉很少被复用的 blocks。

### 3.4 Load 流程（CPU → GPU）

Load 操作将 CPU 缓存中的 KV blocks 加载回 GPU，避免重新计算。

#### 查找匹配

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens, ...):
    """查询 CPU 缓存中是否有该请求的 KV blocks"""
    hit_tokens = self._lookup(request)
    if hit_tokens > 0:
        return (hit_tokens, True)  # async=True，需要异步加载
    return (0, False)

def _lookup(self, request):
    """在 CPUOffloadingManager 中做前缀查找"""
    # 计算请求的 block hashes
    block_hashes = compute_block_hashes(request.prompt_tokens)
    # 最长前缀匹配
    matched_blocks = self.manager.lookup(block_hashes)
    return matched_blocks * block_size  # 转换为 token 数
```

#### 分配与传输

```python
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    """GPU blocks 分配后，准备 CPU→GPU 加载"""
    # 1. 增加 ref_cnt 防止淘汰
    cpu_block_ids = self.manager.prepare_load(block_hashes)
    # ref_cnt > 0 的 block 不会被淘汰策略选中

    # 2. 创建传输规格
    src = CPULoadStoreSpec(cpu_block_ids)   # 源：CPU pinned memory
    dst = GPULoadStoreSpec(gpu_block_ids)   # 目标：GPU KV cache

    # 3. 加入当前 batch 的 load jobs
    self._current_batch_load_jobs.append(TransferJob(src, dst))
```

#### 传输执行

```python
# Worker 侧
def start_kv_transfers(self):
    for job in load_jobs:
        self.worker.transfer_async(job_id, (cpu_spec, gpu_spec))
        # 在专用 CUDA stream 上发起 cudaMemcpyAsync

def get_finished(self):
    for job in pending_jobs:
        if end_event.query():  # 非阻塞检查
            # 传输完成
            self.manager.complete_load(keys)  # 释放 ref_cnt
            finished.append(job_id)
```

### 3.5 CPU 内存管理

#### CPUOffloadingManager（调度器侧）

```python
class CPUOffloadingManager:
    """管理 CPU 侧 KV cache blocks 的分配和追踪"""

    def __init__(self, num_blocks, policy):
        self._free_list: list[int]           # 空闲 block 列表
        self._num_allocated_blocks: int      # 已分配数量
        self._block_status: dict             # hash -> (ref_cnt, block_id)
        self._policy: CachePolicy            # LRU 或 ARC

    # BlockStatus 定义（ctypes 结构体）
    # ref_cnt: int16
    #   -1 = 写入中（store 进行中）
    #    0 = 就绪，可被 load 或淘汰
    #   >0 = 被引用中，不可淘汰
    # block_id: int32
```

#### 物理内存分配（Worker 侧）

**方式一：直接分配**

```python
# gpu_worker.py
cpu_cache = torch.zeros(
    (num_cpu_blocks, cpu_page_size_bytes),
    dtype=torch.int8,
    device="cpu",
    pin_memory=True  # 注册为 CUDA pinned memory
)
```

**方式二：共享 mmap**

```python
# shared_offload_region.py
class SharedOffloadRegion:
    """同节点所有 TP worker 共享的 CPU 内存区域"""

    def __init__(self, instance_id, ...):
        # 共享内存文件: /dev/shm/vllm_offload_{instance_id}.mmap
        # mmap 映射 + 预填充页表 (MADV_POPULATE_WRITE)
        # 注册为 CUDA pinned memory (cudaHostRegister)
```

共享 mmap 的内存布局为 **interleaved**：每个"行"包含所有 worker 对同一个 block 的数据，便于多 worker 并行访问。

#### 内存容量计算

```python
num_cpu_blocks = int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block

# kv_bytes_per_offloaded_block 取决于模型配置
# 例如：每 block 16 tokens, 32 层, 每层 KV head_dim * num_heads * 2(K+V) * dtype_size
```

### 3.6 淘汰策略

当 CPU 缓存已满需要为新 store 腾出空间时，淘汰策略决定移除哪些 blocks。

#### LRU（Least Recently Used）

```python
class LRUCachePolicy:
    """简单的最近最少使用策略"""

    def __init__(self):
        self._cache = OrderedDict()  # hash -> block_id

    def touch(self, key):
        """访问（lookup/store）时移到末尾"""
        self._cache.move_to_end(key)

    def evict(self, num_needed, block_status, protected):
        """从最旧开始淘汰"""
        evicted = []
        for key in self._cache:  # 从头遍历（最旧的）
            if key in protected:
                continue          # 当前操作引用的不淘汰
            if block_status[key].ref_cnt > 0:
                continue          # 正在被 load 的不淘汰
            evicted.append(key)
            if len(evicted) >= num_needed:
                break
        return evicted
```

#### ARC（Adaptive Replacement Cache）

```python
class ARCCachePolicy:
    """自适应替换缓存 - 自动平衡 recency 和 frequency"""

    def __init__(self):
        self.T1 = OrderedDict()    # 近期访问1次
        self.T2 = OrderedDict()    # 频繁访问（>= 2次）
        self.B1 = OrderedDict()    # T1 淘汰后的 ghost list
        self.B2 = OrderedDict()    # T2 淘汰后的 ghost list
        self.target_t1_size = 0    # 自适应调节的 T1 目标大小

    def touch(self, key):
        if key in self.T1:
            # T1 -> T2（从"近期"升级到"频繁"）
            self.T1.pop(key)
            self.T2[key] = True
        elif key in self.T2:
            self.T2.move_to_end(key)
        elif key in self.B1:
            # ghost hit: 说明 T1 太小了，增大 target_t1_size
            self.target_t1_size = min(
                self.target_t1_size + max(1, len(self.B2) // len(self.B1)),
                self.max_size
            )
        elif key in self.B2:
            # ghost hit: 说明 T2 太小了，减小 target_t1_size
            self.target_t1_size = max(
                self.target_t1_size - max(1, len(self.B1) // len(self.B2)),
                0
            )

    def evict(self, ...):
        # T1 过大 -> 从 T1 淘汰，否则从 T2 淘汰
        if len(self.T1) >= self.target_t1_size:
            victim = self.T1.popitem(last=False)  # 最旧的 T1
            self.B1[victim] = True                 # 进入 ghost list
        else:
            victim = self.T2.popitem(last=False)
            self.B2[victim] = True
```

ARC 的优势：自动适应工作负载模式，不需要手动调参。如果最近的 miss 多是"刚被 T1 淘汰的"（B1 ghost hit），就自动增大 T1；反之增大 T2。

### 3.7 异步传输与 CUDA Stream

所有 GPU <-> CPU 数据拷贝都在**专用 CUDA stream** 上异步执行：

```python
# gpu_worker.py
class SingleDirectionOffloadingHandler:
    def transfer_async(self, src_spec, dst_spec):
        # 1. 从 stream 池获取一个 CUDA stream
        stream = self._get_stream()

        # 2. 确保传输顺序（等待上一个传输完成）
        stream.wait_event(last_end_event)

        with torch.cuda.stream(stream):
            # 3. 记录开始事件
            start_event.record(stream)

            # 4. GPU->CPU: 等待当前计算 stream 完成
            if direction == GPU_TO_CPU:
                stream.wait_stream(torch.cuda.current_stream())

            # 5. 执行批量异步拷贝
            ops.swap_blocks_batch(
                batch_src,      # 源地址列表
                batch_dst,      # 目标地址列表
                batch_sizes,    # 每个 block 的字节数
                is_src_access_order_any=is_load,  # CPU->GPU 加载可乱序
            )

            # 6. 记录结束事件
            end_event.record(stream)
```

底层 CUDA kernel（`cache_kernels.cu`）：

```cpp
void swap_blocks_batch(...) {
    // CUDA 12.8+: 单次驱动调用完成批量拷贝
    cuMemcpyBatchAsync(ops, num_ops, flags, stream);

    // 回退方案：逐个 cudaMemcpyAsync
    for (int i = 0; i < num_ops; i++) {
        cudaMemcpyAsync(dst[i], src[i], sizes[i],
                       cudaMemcpyDefault, stream);
    }
}
```

**`is_src_access_order_any`**：CUDA 12.8+ 的优化标志。CPU->GPU 加载时设为 True，允许 DMA 引擎乱序预取源数据，提高传输效率。GPU->CPU 存储时不能用（必须等 GPU 计算完成）。

#### 完成检测

```python
def get_finished(self):
    """非阻塞轮询"""
    for job in pending_jobs:
        if end_event.query():  # 立即返回，不阻塞
            elapsed_ms = start_event.elapsed_time(end_event)
            finished.append(job)
    return finished
```

### 3.8 与 GPU 前缀缓存的交互

CPU Offloading 与 GPU 前缀缓存（prefix cache）之间有重要的交互：

#### Block 重分配保护

当 GPU blocks 被淘汰后重新分配给其他请求时，如果仍有 pending 的 store job 引用这些 blocks，必须先同步完成这些传输：

```python
def build_connector_meta(self, scheduler_output):
    # 检查新分配的 block IDs 是否与 pending store jobs 冲突
    for block_id in newly_allocated_block_ids:
        if block_id in self._block_id_to_pending_jobs:
            # 冲突！必须先刷新这些 store jobs
            jobs_to_flush.add(pending_job)

    # Worker 侧同步等待这些 jobs 完成
    if jobs_to_flush:
        self.worker.wait(jobs_to_flush)  # 阻塞等待
```

#### 避免重复加载

```python
def _lookup(self, request):
    if block_hash in self._blocks_being_loaded:
        # 另一个请求已经在加载这个 block
        return None  # 延迟到下一个 step
```

### 3.9 配置参数

```bash
# 启用 CPU Offloading
vllm serve model_name \
    --kv-connector OffloadingConnector \
    --kv-connector-extra-config '{
        "cpu_bytes_to_use": "10000000000",
        "spec_name": "CPUOffloadingSpec",
        "eviction_policy": "arc",
        "offload_prompt_only": true,
        "store_threshold": 0,
        "max_tracker_size": 64000
    }'
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cpu_bytes_to_use` | （必填） | CPU 内存预算（字节） |
| `spec_name` | `CPUOffloadingSpec` | 也可设为 `TieringOffloadingSpec`（SSD 分层） |
| `eviction_policy` | `lru` | 淘汰策略：`lru` 或 `arc` |
| `offload_prompt_only` | `true` | 只卸载 prefill blocks，不卸载 decode blocks |
| `store_threshold` | `0` | block 被查询多少次后才有资格被 offload |
| `max_tracker_size` | `64000` | block 频率追踪器的最大条目数 |
| `block_size` | `null` | 自定义 offload block 大小（须为 GPU block 大小的倍数） |

---

## 4. PD 分离与 CPU Offloading 的关系

### 4.1 架构共性

两者都实现 `KVConnectorBase_V1` 接口，通过 `--kv-connector` 参数选择：

```
KVConnectorBase_V1
├── NixlConnector          <-- PD 分离
├── P2pNcclConnector       <-- PD 分离（简单场景）
├── OffloadingConnector    <-- CPU Offloading
├── MooncakeConnector      <-- 分布式存储
└── ...
```

调度器和 Worker 通过统一接口与连接器交互，无需感知具体实现：

```python
# 调度器侧 -- 无论哪种连接器，调用方式相同
num_matched, is_async = connector.get_num_new_matched_tokens(req, ...)
connector.update_state_after_alloc(req, blocks, num_external)
meta = connector.build_connector_meta(scheduler_output)

# Worker 侧
connector.start_load_kv(meta)
output = connector.get_finished()
```

### 4.2 核心差异

| 维度 | PD 分离 | CPU Offloading |
|------|---------|---------------|
| **解决的问题** | Prefill/Decode 算力争抢 | KV cache 显存不足 |
| **数据流向** | GPU -> 网络 -> 远程 GPU | GPU <-> 本地 CPU pinned memory |
| **传输协议** | RDMA (NIXL/UCX) | CUDA memcpy (async stream) |
| **触发方式** | 外部 proxy 路由决策 | Scheduler 每步自动检查 |
| **Block 生命周期** | 租约 + 心跳续租 | ref_cnt + 淘汰策略 (LRU/ARC) |
| **多节点** | 是（P 和 D 在不同机器） | 否（同一台机器内） |
| **缓存语义** | 一次性传输，P 完成后释放 | 前缀缓存，可被多个请求复用 |
| **延迟影响** | 网络延迟（~100us RDMA） | PCIe 延迟（~10-50us） |

### 4.3 互补性分析

PD 分离和 CPU Offloading 在概念上是互补的，可以类比为 KV cache 管理的三级存储：

```
+---------------------------------------------------+
|              GPU 显存（最快，最小）                  |
|  +----------+  +----------+  +----------+         |
|  | 活跃请求  |  | 活跃请求  |  | 活跃请求  |         |
|  | KV cache |  | KV cache |  | KV cache |         |
|  +----------+  +----------+  +----------+         |
|         | PCIe (CPU Offloading)                    |
+---------|-----------------------------------------+
|         v                                          |
|           CPU Pinned Memory（中速，中等）            |
|  +----------+  +----------+                        |
|  | 不活跃请求 |  | 前缀缓存  |                       |
|  | KV cache |  | blocks   |                        |
|  +----------+  +----------+                        |
|         | RDMA (PD 分离)                            |
+---------|-----------------------------------------+
|         v                                          |
|           远程 GPU（跨节点，高延迟）                  |
|  +----------------------+                          |
|  | P 节点计算的 KV cache  |                         |
|  +----------------------+                          |
+---------------------------------------------------+
```

**潜在组合场景**：

1. **D 节点 + CPU Offloading**：D 节点接收大量 P 节点传来的 KV cache，显存压力大。如果 D 节点也能把暂时不用的 KV offload 到 CPU，就能服务更多并发请求。

2. **多轮对话优化**：第一轮 KV 从 P 传来，D 做 decode；第二轮新请求来时，第一轮的 KV 如果已被 offload 到 CPU 而不是丢弃，就能直接 load 回来，不用重新找 P 节点计算。

3. **长上下文场景**：超长 prompt 的 KV cache 可能超过单个 D 节点的显存。组合使用可以：P 节点计算 KV -> RDMA 传给 D -> D 把不常访问的层 offload 到 CPU -> 按需加载。

**当前限制**：`--kv-connector` 只能选一个，两者不能同时启用。要实现组合，需要开发 composite connector 或修改连接器框架支持多连接器堆叠。
