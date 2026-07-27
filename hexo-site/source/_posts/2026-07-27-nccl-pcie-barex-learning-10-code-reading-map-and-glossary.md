---
title: "10. 代码阅读地图、API 对照与术语表"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 10. 代码阅读地图、API 对照与术语表

## 1. Barex 阅读顺序

### 第一轮：只看接口

| 顺序 | 文件 | 目标 |
|---:|---|---|
| 1 | `include/accl/barex/barex_types.h` | `memp_t/rw_memp_t/x_wr_id callback` |
| 2 | `include/accl/barex/xchannel.h` | Send/Write/Read 与完成约定 |
| 3 | `include/accl/barex/xcontext.h` | progress 与 channel 生命周期 |
| 4 | `include/accl/barex/xsimple_mempool.h` | MR 注册/释放 |
| 5 | `include/accl/barex/xconnector.h` | client 建联 |
| 6 | `include/accl/barex/xlistener.h` | server 建联 |

### 第二轮：RDMA 主路径

```text
src/barex/impl/xstatic_instance.cc
→ src/barex/impl/rdma/xdevice_manager_impl.cc
→ src/barex/impl/rdma/xcontext_impl.cc
→ src/barex/impl/rdma/xchannel_impl.cc
→ src/barex/impl/rdma/xgpu_mempool_impl.cc
```

建议给以下函数画自己的调用箭头：

- `XContext::NewInstance`
- `XChannelImpl::Incubate/Init`
- `XChannelImpl::Send`
- `XChannelImpl::WriteBatch`
- `XChannelImpl::PostSendOrEnqueue`
- `XContextImpl::ProcessOneIoEvent`
- `XContextImpl::HandleWriteComplete`

### 第三轮：其他 backend 与鲁棒性

- `impl/boost/`：TCP channel/context/device。
- `impl/solar/`：Solar backend。
- `impl/tcp/`：带外 connector/listener protocol。
- heartbeat、close、timeout、statistics。

## 2. blade-kvt 阅读顺序

| 顺序 | 文件 | 关注 |
|---:|---|---|
| 1 | `blade_kvt/kv_transfer_impl.py` | 用户语义 |
| 2 | `kvtransfer/kvtransfer_pybind.cpp` | Python/C++ 边界 |
| 3 | `kvtransfer/src/client.cpp` | request→step→target |
| 4 | `kvtransfer/src/step.cpp` | layer event/barrier |
| 5 | `kvtransfer/src/tx_stub.cpp` | TP mapping、parse、send loop |
| 6 | `kvtransfer/include/channel.h` | channel contract |
| 7 | `kvtransfer/src/rdma_channel.cpp` | direct RDMA |
| 8 | `kvtransfer/src/barex_protocol.cpp` | Barex wrapper/context/MR |
| 9 | `kvtransfer/src/rdma_staged_channel.cpp` | staged |
| 10 | `kvtransfer/src/tcp_channel.cpp` | TCP |
| 11 | `kvtransfer/src/parse_block_*.cpp` | cache shape 与 offset |

## 3. 一次 direct RDMA 的逐函数断点表

| 阶段 | 函数 |
|---|---|
| Python 开始 step | `KvTransfer.start_send_step` |
| C++ 建 step | `KvTransferClient::start_send` |
| 按目标投递 | `TargetMgr::do_submit` |
| 目标 batch | `KvSendStub::send_batch` |
| layout | `TaskContext::refresh_dst_info` + `parse_block_*` |
| 建 channel | `TaskContext::try_create_channel` |
| 取 remote MR | `RDMAChannel::do_init/get_mem_handles` |
| 注册 block | `RDMAChannel::register_data` |
| 发一层 | `RDMAChannel::send_data` |
| Barex batch | wrapper `WriteBatch` |
| WR | `XChannelImpl::WriteBatch/MakeSendBatch` |
| post | `XChannelImpl::PostSend` |
| poll | `XContextImpl::ProcessIoEvents` |
| completion | `HandleWriteComplete` |
| wait | `RDMAChannel::flush` |

## 4. API 语义对照

| API | 远端地址由谁准备 | 远端 callback | 本地完成 |
|---|---|---|---|
| Barex `Send` | Barex 内部协商 | `OnRecvCall` | DoneCallback |
| Barex `WriteSingle` | 应用提供 raddr/rkey | 可选 imm callback | DoneCallback |
| Barex `WriteBatch` | 每项提供 raddr/rkey | 通常无 | 每项/批 callback |
| Barex `WriteBySgList` | 应用提供连续远端范围 | 可选 imm | DoneCallback |
| NCCL collective | NCCL 内部 | 无应用 RPC callback | CUDA stream/event |
| NCCL send/recv | NCCL rank matching | 无应用 RPC callback | CUDA stream/event |

## 5. 关键不变量

### Memory

- MR 生命周期覆盖所有 inflight WR。
- local SGE 必须位于 lkey 对应 MR 范围。
- remote address 必须位于 rkey 对应 MR 范围。
- 目标重启后旧 rkey 无效。
- buffer 不能在 completion 前释放或复用。

### Channel

- post 时 channel 为 active/INIT_SUCCESS。
- `tx_depth` permit 与实际 WR 数一致。
- CQ error 后不能继续信任同一 QP。
- connector/listener 生命周期覆盖其 channel 建立过程。

### blade-kvt

- `send_blocks` 在 `IChannel::flush` 前保持有效。
- layer MR tensor 数与 `data.size()` 一致。
- P/D TP mapping 生成的 offset 不越界。
- direct 模式 server handle 次序与 client tensor 次序一致。
- `send_done` 只能在 request 对应 payload batch 已完成后发送。

## 6. 术语表

| 术语 | 含义 |
|---|---|
| bit / Byte | 1 Byte = 8 bit；`Gb/s` 与 `GB/s` 相差 8 倍 |
| ACS | PCIe Access Control Services，影响 P2P 路由/隔离 |
| BAR | Base Address Register，设备暴露的 MMIO 地址窗口 |
| BDF | PCI Domain:Bus:Device.Function |
| BDP | Bandwidth-delay product，填满链路所需 inflight bytes |
| CQ/CQE | Completion Queue / Entry |
| DMA | 设备无需 CPU copy 直接读写内存 |
| doorbell | 软件写 MMIO/映射寄存器，通知设备有新 WQE |
| full-duplex | 两个方向可同时传输；PCIe lane 天然全双工 |
| GDR | GPUDirect RDMA，NIC 直接 DMA GPU memory |
| GT/s | 每秒十亿次 transfer；还需考虑编码，不能直接等于有效 Gb/s |
| HCA | Host Channel Adapter，RDMA NIC |
| IOMMU | I/O 地址转换与隔离单元 |
| lane / x16 | PCIe 最小双向通道 / 由 16 条 lane 聚合的 link |
| lkey | 本地 MR key，校验 local SGE |
| MR | Memory Region，注册给 RNIC 的内存 |
| MPS | PCIe Max Payload Size |
| MRRS | PCIe Max Read Request Size |
| NUMA | 非一致内存访问架构 |
| PD | Protection Domain，QP/MR 的隔离域 |
| QP | Queue Pair，RDMA send/receive queue |
| RC | Reliable Connected QP；也可指 PCIe Root Complex，需看上下文 |
| rkey | 远端 MR key |
| RNR | Receiver Not Ready |
| RTT | Round-Trip Time，请求到对端再返回的往返时间 |
| SGE | Scatter/Gather Element |
| SQ/RQ | QP 的 Send Queue / Receive Queue |
| TLP | PCIe Transaction Layer Packet |
| WR/WQE | Work Request / Work Queue Element |
| WC | Work Completion |
| one-sided | RDMA read/write，远端 CPU 不必为每次 payload 主动参与 |
| two-sided | send/recv，双方需要匹配接收资源 |

## 7. 环境变量分组

### Barex

```text
Device: ACCL_USE_NICS, ACCL_SET_ERDMA, ACCL_SET_SOLAR
Queue:  ACCL_TX_DEPTH, ACCL_TX_CONN_DEPTH, ACCL_SOFT_TX_DEPTH
Verbs:  ACCL_IBV_MTU, ACCL_MAX_SGE, ACCL_RNR_RETRY, ACCL_RETRY_CNT
MR:     ACCL_MAX_USER_MR_GB
Liveness: ACCL_HEARTBEAT_INTERVAL, ACCL_RETRANSMIT_TIMEOUT
Batch:  ACCL_WRITEBATCH_OPT
Socket compatibility: NCCL_SOCKET_IFNAME, NCCL_SOCKET_FAMILY
```

### blade-kvt

以 `kvtransfer/src/envcfg.cpp` 为真实来源，重点包括：

- protocol；
- send parallel；
- context/connector/send thread pool size；
- RPC timeout；
- port base；
- CRC；
- send-done address；
- cache shape。

### NCCL

```text
Debug: NCCL_DEBUG, NCCL_DEBUG_SUBSYS
Topology: NCCL_TOPO_FILE
Transport: NCCL_P2P_DISABLE, NCCL_SHM_DISABLE, NCCL_IB_DISABLE
Network: NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_NET
Tuning: NCCL_ALGO, NCCL_PROTO, NCCL_MIN_NCHANNELS, NCCL_MAX_NCHANNELS
```

具体默认值随版本变化，以官方文档为准。

## 8. 学习完成标准

能够不看文档回答：

1. 一条 PCIe Memory Write 如何从 NIC 到 GPU。
2. NCCL algorithm/protocol/transport 的区别。
3. Barex `Send` 与 `WriteBatch` 的差别。
4. blade-kvt 如何取得 raddr/rkey。
5. `send_data` 返回、CQ completion、远端 H2D 完成、业务 send-done 四个边界。
6. 为什么 Barex 当前不是 NCCL plugin。
7. 遇到 `REM_ACCESS_ERR` 与 `RNR` 分别从哪里开始查。
