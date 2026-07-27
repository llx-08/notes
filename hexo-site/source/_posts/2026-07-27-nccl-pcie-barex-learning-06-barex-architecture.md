---
title: "06. Barex 1.5.3-1 架构与发送实现"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 06. Barex 1.5.3-1 架构与发送实现

## 1. 定位

Barex 是异步 point-to-point 传输库，向应用提供：

- RDMA、Solar、TCP backend；
- connection/listener/channel；
- CPU、CUDA host、GPU memory pool 与 MR；
- Send/Read/Write/Batch/SG list；
- callback、thread pool、timer、statistics。

它不实现 collective graph，不管理 communicator/rank，也不是 NCCL plugin。

### 1.1 把 Barex 看成“带多种运输方式的异步快递 API”

应用先创建 context 和 channel，再把一次传输描述为：

```text
从哪个 buffer
搬多少字节
到哪个 peer / 远端地址
完成后调用哪个 callback
```

Barex 负责把它落到 RDMA/Solar/TCP backend，并推进队列与 completion。

“异步”的含义是调用线程提交后可以继续做别的事：

```cpp
auto rc = channel->WriteBatch(items, done_callback);
// rc 成功：任务被接受或排队
// 不能在这里立刻释放 items 指向的 payload

// ...
// done_callback 被调用后，再根据 completion 语义释放/复用资源。
```

这也是 blade-kvt 为什么把 callback 包装成 `promise/future`：业务希望在
`flush()` 处把异步完成重新变成一个明确等待点。

## 2. 对象关系

![Barex 对象、线程与数据面](/imgs/barex_architecture.svg)

| 对象 | 所有权/职责 |
|---|---|
| `XDeviceManager` | 枚举并选择 RDMA/Solar/TCP device |
| `XSimpleMempool` | 分配、注册、查询、释放 CPU/GPU MR |
| `XContext` | CQ/epoll/timer、channel 生命周期、callback dispatch |
| `XConnector` | client 侧带外建联 |
| `XListener` | server 侧监听并交换 QP metadata |
| `XChannel` | 一条 peer 连接；发送、读写、流控、心跳 |
| `XThreadpool` | 将回调或应用工作移出 IO thread |

源码入口：

- `include/accl/barex/xcontext.h`
- `include/accl/barex/xchannel.h`
- `include/accl/barex/xsimple_mempool.h`
- `src/barex/impl/xstatic_instance.cc`

## 3. Backend 工厂

`XContext::NewInstance` 根据 `XDevice` 的动态类型选择实现：

```text
IbvDevice   → XContextImpl       (RDMA verbs)
SolarDevice → XContextSolarImpl
TcpDevice   → XContextTcpImpl
```

证据：`src/barex/impl/xstatic_instance.cc:89-131`。

`XDeviceManager::Singleton` 同样按 `XDT_RDMA/XDT_SOLAR/XDT_TCP` 返回 singleton，见 `xstatic_instance.cc:134-186`。

## 4. Context 初始化

应用通常按以下顺序：

```text
XDeviceManager::Singleton
  → device
  → XSimpleMempool::NewInstance
  → RegUserMr / Reserve
  → XThreadpool::NewInstance
  → XContext::NewInstance
  → context->Start()
  → XConnector 或 XListener
```

`XContext` 的 IO 模型：

1. CQ event fd 与 timer fd 加入 epoll；
2. `ProgressEvents()` 非阻塞 `epoll_wait`；
3. CQ 有事件时 `ProcessIoEvents()` 调用 `ibv_poll_cq`；
4. 每个 WC 进入 `ProcessOneIoEvent()`；
5. 根据 `wc.opcode` 分发到 send/write/read/recv handler；
6. callback 可 inline 执行，也可投递到用户 thread pool。

证据：

- `src/barex/impl/rdma/xcontext_impl.cc:780-883`
- `xcontext_impl.cc:1009-1073`
- `xcontext_impl.cc:1319-1354`

## 5. Channel 建立与状态

RDMA `XChannelImpl::Incubate`：

- 校验 config；
- 绑定 context、device、CQ、mempool；
- 准备 heartbeat MR；
- 创建 RC QP；
- 设置 `max_send_wr/max_recv_wr/max_sge`；
- 初始化 `send_semaphore_ = tx_depth`。

见 `src/barex/impl/rdma/xchannel_impl.cc:71-230`。

带外 TCP/connector 交换 `ChannelInitMeta`：

```text
qp_num, psn, lid, gid
heartbeat_addr/rkey
context/channel identity
nic_id
```

随后双方把 QP 推进到可通信状态。Channel 生命周期大致是：

```text
CONSTRUCTED
  → INCUBATE_SUCCESS
  → INIT_SUCCESS
  → CLOSED
  → DESTROYED
  → DELETED
```

`Close`、`Destroy`、`CloseAndDeleteChannel` 是不同阶段；blade-kvt 的 `BarexChannel::destroy()` 依次执行三者。

## 6. Memory 与 MR

`memp_t` 包含：

- `base/buf/buf_len`；
- device type 与 device id；
- 底层 MR，包含 lkey/rkey；
- allocator 与释放状态。

常用 API：

| API | 含义 |
|---|---|
| `AllocBuffer` | 从 mempool 分配并取得可传输 buffer |
| `RegUserMr` | 注册应用已有的 CPU/GPU buffer |
| `FindBufferMr` | 根据地址查 MR，可切换到当前 NIC 对应 MR |
| `DeregUserMr` | 注销用户 MR |
| `ReleaseBuffer` | 释放 pool buffer |
| `ReleaseAndDeregBuffer` | 同时释放和注销 |

多 NIC 场景中同一虚拟地址可能需要不同 PD/MR；`XChannelImpl::TryToChangeMr` 会切换到本 channel 对应 NIC 的 MR。

## 7. `Send`：消息语义

`XChannel::Send` 让远端触发 `XChannelCallback::OnRecvCall`。它不是裸 RDMA Write API。

### 7.1 小消息

当 `sizeof(x_msg_header) + payload <= small_msg_size`：

1. 构造两段 SGE：payload + header；
2. opcode 为 `IBV_WR_SEND_WITH_IMM`；
3. immediate 标记 `SMALL_MSG`；
4. WR 使用 `IBV_SEND_SIGNALED`；
5. 远端 RECV WC 解析并调用 `OnRecvCall`。

证据：`xchannel_impl.cc:925-1005`。

### 7.2 大消息三阶段

![Barex 大消息 Send 三阶段](/imgs/barex_large_message.svg)

```text
A: Send LARGE_MSG_META(len, opaque ptr)
                         → B
B: 分配接收 buffer
B: Send LARGE_MSG_WRITE_BUF_META(addr/key/index)
                         → A
A: RDMA_WRITE_WITH_IMM(payload + header)
                         → B
B: 按 imm index 找到 buffer，调用 OnRecvCall
```

关键代码：

- phase 1：`xchannel_impl.cc:1068-1140`
- phase 2：`xchannel_impl.cc:1142-1203`
- phase 3：`xchannel_impl.cc:1205` 起
- 接收分发：`xcontext_impl.cc:1121-1296`

这是一种 two-sided message abstraction：应用不必预先知道远端地址，Barex 内部协商 buffer。

## 8. `WriteSingle/WriteBatch`：one-sided 数据面

与 `Send` 不同，Write 需要调用者提供：

```text
local addr + lkey + length
remote addr + rkey
```

- `WriteSingle`：一个本地段写一个远端范围。
- `WriteBatch`：多个 `rw_memp_t`，每项可有独立远端地址。
- `WriteBySgList`：多个本地段聚合写入连续远端范围。
- `signal_peer=true` 时使用 WRITE_WITH_IMM，让远端得到通知。

blade-kvt direct RDMA 使用 `WriteBatch` 且不通知对端，因为 KV cache 已直接写入最终 GPU 地址，上层通过独立 send-done RPC 表达 request 完成。

### 8.1 具体例子：一层 KV cache 有三个不连续 block

假设源 GPU 上三个 block 的地址不连续，目标 GPU 已给出三个最终位置：

```text
local block A (64 KiB) → remote slot 9
local block B (64 KiB) → remote slot 2
local block C (64 KiB) → remote slot 7
```

blade-kvt 可以构造三个 `rw_memp_t` 后一次 `WriteBatch`：

```text
item 0: local_addr=A, len=64KiB, raddr=slot9, rkey=...
item 1: local_addr=B, len=64KiB, raddr=slot2, rkey=...
item 2: local_addr=C, len=64KiB, raddr=slot7, rkey=...
```

“Batch”表示减少 API/doorbell/callback 等固定成本，不表示三个远端范围必须连续，
也不表示一次调用返回时三个写已经完成。`datasp` 和其中引用的 MR 必须至少活到
callback。

## 9. TX depth 与软件队列

Barex 用 `send_semaphore_` 限制 inflight WR：

```text
PostSendOrEnqueue(permits)
  ├─ semaphore 足够 → 扣减并 post
  ├─ 软件队列未满 → 入 send_queue_
  └─ soft_tx_depth 也满 → callback QUEUE_FULL
```

CQ completion 后：

```text
IoEventOccur
  → ReleaseAndPostSend(permits)
  → 恢复 semaphore
  → 从队头提交所有当前能容纳的任务
```

证据：`xchannel_impl.cc:871-923`。

重要不变量：

- queue 中的 send task 没有独立取消/超时；
- channel 关闭时必须让 queued/inflight task 进入错误完成或清理；
- `ACCL_TX_DEPTH` 控制硬件 inflight，`ACCL_SOFT_TX_DEPTH` 控制软件排队上限；
- depth 太小不能覆盖 RTT，太大会增加资源和错误清理压力。

## 10. `ibv_post_send` 到 callback

```text
XChannel Write/Send
  → 构造 SGE、WR、x_wr_id
  → PostSendOrEnqueue
  → ibv_post_send
  → NIC 执行
  → CQE
  → XContextImpl::ProcessIoEvents
  → ProcessOneIoEvent
  → HandleSendComplete / HandleWriteComplete
  → DoneCallback
  → ReleaseAndPostSend
```

![Barex WR、CQ 与 callback 完成链](/imgs/barex_completion_path.svg)

`x_wr_id` 保存 channel、buffer、auto-release、callback、opcode 和 message type，用于完成与错误清理，定义在 `src/barex/common.h:652` 起。

## 11. 完成错误

`HandleWcStatusError`：

1. 把 WC status 转成可读错误；
2. 调用失败 callback；
3. 接收 WR 释放内部 recv buffer；
4. 发送 WR 只释放 Barex header，不擅自释放用户 payload；
5. 销毁 channel。

见 `xcontext_impl.cc:913-1006`。

这是为什么应用 callback 必须正确管理 buffer 生命周期：同步提交错误和异步 WC 错误都可能发生。

## 12. 关键环境变量

| 变量 | 影响 |
|---|---|
| `ACCL_USE_NICS` | 可用 NIC |
| `ACCL_TX_DEPTH` | channel 发送 WR depth |
| `ACCL_TX_CONN_DEPTH` | 连接相关 depth |
| `ACCL_SOFT_TX_DEPTH` | 软件发送队列 |
| `ACCL_MAX_SGE` | 单 WR SGE 上限 |
| `ACCL_WRITEBATCH_OPT` | WriteBatch 提交/完成优化 |
| `ACCL_MAX_USER_MR_GB` | 用户 MR 大小限制 |
| `ACCL_IBV_MTU` | RC MTU |
| `ACCL_RETRANSMIT_TIMEOUT` | QP timeout |
| `ACCL_RETRY_CNT` | transport retry |
| `ACCL_RNR_RETRY` | receiver-not-ready retry |
| `ACCL_HEARTBEAT_INTERVAL` | channel 活性检测 |

## 13. 自检

1. `Send` 大消息为什么要先协商远端 buffer，而 `WriteBatch` 不需要？
2. Barex API 返回成功与 CQ completion 有什么区别？
3. 为什么 `tx_depth` completion 后才能归还？
4. callback 为什么既可能在 IO thread，也可能在线程池？
