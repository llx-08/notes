---
title: "09 调试、测试与代码地图：从现象定位到具体状态"
date: 2026-07-30
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 09 调试、测试与代码地图：从现象定位到具体状态

## 1. 第一原则：先定位卡在哪一个 completion

```text
① request 是否被 Hybrid 接管
② D Block 是否分配成功
③ D→P RPC 是否发出/收到
④ P 是否生成 PReqMeta
⑤ P Worker 是否 bind metadata
⑥ CUDA layer Event 是否完成
⑦ Blade-KVT channel 是否提交
⑧ Barex completion 是否到
⑨ SEND_DONE 是否到 P Scheduler
⑩ 所有 TP/source 是否收齐
⑪ KVTResp 是否回 D
⑫ D mark_loaded 是否唤醒 Core
⑬ request 是否回普通 Scheduler
```

不要一开始就抓所有机器的全量日志。先用 request ID 串出这 13 个里最后成功的一项。

## 2. 按现象排查

### 请求一直没进 running

检查：

```text
Hybrid _waiting/_loading？
allocate_slots 是否返回 None？
free blocks / extra ref usage？
DBackend RPC 是否 pending？
_loaded 是否有内容但 Core 没 wakeup？
```

### P GPU 算完但无网络流量

检查：

```text
PReqMeta freeze/nonfreeze 数量
bind_backend_metadata 是否执行
start_send_step 返回 EMPTY_STEP_ID？
record_event 是否调用
Step WaitUs 是否持续增长
target manager SubQueueUs
```

### 有 RDMA 带宽但 D 输出错误

检查：

```text
P/D cache shape、TP、block/token bytes
parse_block 分支
layer_tensor_masks
IpcBlock bounds
zero race
是否写对 D block IDs
CRC / KV dump compare
```

### D KV 看起来正确但请求不继续

检查：

```text
Blade-KVT send_done RPC
PBackend _sending[reqid]
IoState missing tprank/source/target
KVTResp socket
D Hybrid _loaded
wakeup_core fake abort
```

### GPU Block 使用率只升不降

检查：

```text
_saving 是否长期不清
_try_teardown_save 是否执行
save extra ref by source
abort path 是否走到 save side
deferred_frees fence 是否推进
unknown/duplicate completion 是否导致状态未完成
```

## 3. 推荐日志关联键

每行至少保留：

```text
reqid
P/D instance
worker_tp_rank
dst worker
stepid.substepid
layer
source
code
seen/new/end tokens
P/D block count
channel/protocol
```

时间指标：

```text
Hybrid on_add → load_enqueue → after_alloc → mark_loaded → done
Step SubQueueUs / WaitLayerQueueUs / ForwardExecUs
SendStub QueueUs / WaitUs / WaitAndSendUs / SendNotifyUs
Channel D2H / link / H2D / completion
```

## 4. 实用 grep 顺序

```bash
REQ='<request-id>'

# vLLM scheduler/worker logs
rg "$REQ|mark loaded|mark saved|disagg start|disagg end|abort|gone" vllm.log

# Blade-KVT
rg "$REQ|StepIdx|SendStubMetrics|StepMetrics|tx_stub fail|rpc send done" worker.log

# timeout / connection
rg "timeout|IncompleteRead|connect error|Write ERR|on_send_error" *.log
```

对同一 req 按 `stepid.substepid` 排序，能区分正常 main step 和晚到 freeze/substep。

## 5. 状态快照建议

Hybrid Scheduler 暴露或临时打印：

```text
len(_waiting)
len(_loading), oldest age
len(_saving), oldest age
len(_loaded)
len(_saved)
len(_aborting)
len(PBackend._sending)
len(PBackend._dinfoq)
len(PBackend._infly_kvt)
len(PBackend._dash_done)
extra KV refcounts by load/save source
```

对一个 stuck `_SendingReq` 打印：

```text
expected TP size
signals_per_worker
expected_sources
received worker → [source, n, ex]
missing workers/sources
```

## 6. 网络与 GPU 观察

### RDMA

```bash
ibstat
rdma link
ibv_devinfo
ethtool -S <nic>
```

关注 QP error、retry、timeout、CQ error、PFC/ECN 和端口丢包。详见
[RDMA 操作与完成](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-02b-rdma-operations-completion-and-reliability/)
与 [RoCE 拥塞](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-02c-roce-congestion-and-tuning/)。

### GPU/PCIe

```bash
nvidia-smi topo -m
nvidia-smi dmon
nsys profile ...
```

确认 P GPU 与 RNIC 亲和、D GPU 写入流量、CUDA Event/stream 和 copy kernel 时间线。

## 7. 单元测试阅读地图

### vLLM lifecycle

```text
tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py
tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py
tests/v1/kv_connector/unit/test_kv_connector_lifecyle.py
```

### 错误与资源

```text
test_hybrid_abort_refcount_leak.py
test_error_propagation.py
test_kv_load_failure_recovery.py
test_invalid_blocks_correctness.py
test_cache_pollution_prevention.py
```

### Hybrid 集成

```text
tests/v1/kv_connector/hybrid/test_hybrid_connector.py
tests/v1/kv_connector/hybrid/migration/test_migration.py
```

### Blade-KVT

```text
kvtransfer/tests/client_step_test.cu
kvtransfer/tests/step_sync_test.cu
kvtransfer/tests/client_test.cpp
kvtransfer/tests/rdma_channel_test.cpp
kvtransfer/tests/utils_semaphore_test.cpp
```

## 8. 推荐故障注入实验

### 实验 A：漏一个 TP completion

在一个 P worker 拦截 send_done：

预期：

- 其他 rank 信号已到；
- P `_SendingReq` 不完成；
- watchdog 应报告确切 missing rank；
- request deadline 后失败并释放 extra ref。

当前基线没有完整上层 deadline，此实验会直观暴露风险。

### 实验 B：D 在 RDMA 中途退出

步骤：

1. 大 KV 请求；
2. 确认 WriteBatch 已开始；
3. kill D worker；
4. 观察 Barex callback、P send thread、P `_sending`、Block refcount。

必须设置外部看门狗，避免实验本身永久挂住。

### 实验 C：send_done 端口黑洞

数据面保持正常，只阻断 `BLLM_KVTRANS_SEND_DONE_ADDR`：

验证：

- D GPU KV 已改变；
- P 两次 RPC 失败；
- D 仍等 KVTResp；
- 指标能否区分 data done/control pending。

### 实验 D：zero/load 时序

在 D：

1. 记录分配 Block ID；
2. 延迟外部 load 或 zero；
3. load 后 dump KV；
4. 下一 SchedulerOutput 后再 dump；
5. 验证目标 Block 没被 zero 覆盖。

覆盖 KVT 和组合 backend。

### 实验 E：bypass 乱序

用已有 fault injection sleep 让：

- substep 早于主 step；
- substep 晚于 flush；
- 多 substep 快速到达。

验证 pending、attach、new-freeze 三个分支和 `substepid` 单调性。

## 9. 代码精读顺序

### 第一轮：只读主线

```text
EngineCore.add_request
HybridScheduler.on_add_req / step
DBackend.async_update_state_after_alloc
PBackend.build_backend_meta / bind_backend_metadata
KVTransferClient.submit_req_send2 / start_send_step
KvTransferClient::start_send
KvSendStub::TaskContext::do_send
RDMAChannel::send_data / flush
PBackend._do_send_done
HybridScheduler._step_loaded/_step_saved
```

### 第二轮：所有权与竞态

```text
BlockPool.touch/free_blocks
KVCacheManager.allocate_slots/free
sched_acquire_blocks/sched_free_blocks
request_finished_all_groups
deferred_frees
StepGuard / SyncSemaphore
start_send_substep
try_advance / IoState
```

### 第三轮：异常

```text
kill_me_if_exception
RpcServer._client_main
DBackend._kvt_rpc/_abort_rpc
PBackend._gone_reqs/_step_aborting
KvSendStub catch/send_done retry
CliBarexCtx::on_send_error
TCP/RDMA flush
```

## 10. 重要函数索引

| 问题 | 函数 |
|---|---|
| 请求被谁接管 | `EngineCore.add_request`, `HybridScheduler.on_add_req` |
| D Block 何时分配 | `HybridScheduler._step_waiting`, `sched_allocate_slots` |
| P Block 何时加引用 | `HybridScheduler._setup_save` |
| 请求何时进入普通 Scheduler | `HybridScheduler._step_loaded` |
| D 如何请求 P | `DBackend._prefill_rpc`, `_dash_prefill_rpc`, `_kvt_rpc` |
| P 如何匹配现有请求 | `PBackend._step_dinfoq` |
| P 如何生成发送范围 | `PBackend._step_sched_req` |
| Worker 如何开始发送 | `PBackend.bind_backend_metadata` |
| 按 target 分组 | `add_send_task`, `TargetMgr::do_submit` |
| 等 layer | `StepGuard::wait_layers`, `Step::wait_layer_ready` |
| 生成 byte range | `parse_block_*` |
| 发 RDMA | `RDMAChannel::send_data`, `WriteBatch` |
| 发 TCP | `TCPChannel::send_data`, `TCPServer::handle_kv_cache_data` |
| TP 完成汇聚 | `try_advance`, `IoState.merge` |
| finish/save 竞态 | `request_finished_all_groups`, `_step_saved` |
| Core 被完成事件唤醒 | `_q_append`, `wakeup_core` |

## 11. 生产监控最低集合

### Gauge

```text
hybrid_waiting/loading/saving/sending
oldest_state_age_seconds
missing_completion_count
extra_kv_blocks_load/save
free_kv_blocks
active_targets/channels
```

### Counter

```text
KVTResp code 0/404/410/500
send_done retry/fail
Barex callback error/timeout
unknown/duplicate completion
abort RPC success/fail
channel recreate
zero-excluded blocks
```

### Histogram

```text
rendezvous latency
block allocation wait
event→layer-ready
layer-ready→submit
transport completion
send_done
end-to-end load/save
```

告警不要只看平均值。永久 hang 往往请求数很少，对平均延迟影响不明显，但
`oldest_state_age` 会持续增长。

## 12. 阅读结束后的验收题

给定一个 request ID，你应能回答：

1. 它现在由 Hybrid Connector 还是普通 Scheduler 持有？
2. P/D 各有哪些 Block，refcount 为什么是当前值？
3. 它属于哪个 step/substep，哪些 layer 已 ready？
4. 对每个 P TP rank，目标 D worker 是谁？
5. transport completion 到了哪些，SEND_DONE 到了哪些？
6. 如果现在 abort，哪些本地/远端状态要清理？
7. 如果某一方掉线，最迟由哪个 deadline 结束？

如果第 7 题答不出一个明确 deadline，就应把它记录成系统风险，而不是假设底层网络
最终总会报错。
