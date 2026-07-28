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

### 1.2 Barex 到底封装在什么之上

对当前 `accl-barex-v1.5.3-1`，不能只回答“封装了 RDMA”，因为它同时提供多个
backend。以 blade-kvt 最常用的 RDMA 和 TCP 路径为主，软件栈是：

```text
blade-kvt
  │  Send / WriteSingle / WriteBatch / callback
  ▼
Barex public API
  │
  ├─ RDMA backend
  │    ├─ libibverbs / rdma-core
  │    │    ibv_alloc_pd / ibv_reg_mr
  │    │    ibv_create_cq / ibv_create_qp
  │    │    ibv_post_send / ibv_post_recv / ibv_poll_cq
  │    ├─ 可选 mlx5dv/provider-specific 能力
  │    └─ TCP OOB：交换 QPN/GID/LID/PSN、心跳和控制 metadata
  │
  ├─ TCP backend
  │    └─ TCP socket / Boost.Asio / epoll / framing
  │
  └─ Solar backend
       └─ Solar verbs/device API

Linux kernel RDMA uverbs/provider 或 TCP stack
  ▼
RNIC driver / NIC hardware / network
```

源码中的 `src/barex/accl_verbs.h` 可以直接证明这一层关系：例如
`IbvPostSend()` 的实现就是调用 `::ibv_post_send()`，`IbvPostRecv()` 调
`::ibv_post_recv()`，`IbvPollCq()` 调 `::ibv_poll_cq()`。这个 wrapper 也便于
测试、替换 provider 和统一错误处理。

Barex 不是：

- NCCL 上的一层封装；
- Linux 内核 RDMA driver；
- RoCE/InfiniBand wire protocol；
- 一个能被单个 `ibv_*` 函数完整替换的薄 wrapper。

它在原生 transport API 上又实现了：

| Barex 负责 | 如果直接使用原生 API，需要应用自己负责 |
|---|---|
| device/NIC 选择 | `ibv_get_device_list/open_device`、网卡亲和性 |
| PD、MR、mempool | `ibv_alloc_pd/reg_mr/dereg_mr`、注册缓存与生命周期 |
| channel/连接 | `librdmacm`，或 TCP OOB + `ibv_modify_qp` 状态机 |
| `Send` 消息协议 | pre-post receive、framing、大消息 rendezvous |
| `WriteBatch` | SGE/WR 数组、WR 链、signaled 策略、depth/backpressure |
| callback | CQ polling、`wr_id` 映射、线程切换、只回调一次 |
| 接收 buffer | 预分配、post/repost RQ、自动释放 |
| 错误/关闭 | WC status、QP ERR/flush、inflight buffer 清理 |
| 运行时 | completion channel、epoll、timer、heartbeat、metrics |

因此“去掉 Barex”不是把 `WriteBatch` 改成另一个同名标准函数，而是下沉到
`libibverbs/librdmacm` 或 TCP socket，并把 Barex 帮你做的资源管理与协议重新实现。

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

## 9. 如果没有 Barex，这些 API 分别用什么

### 9.1 一张最直接的映射表

| Barex API/回调 | RDMA backend 下的原生基础 | TCP backend 下的原生基础 |
|---|---|---|
| `WriteSingle` | 一个 `IBV_WR_RDMA_WRITE` 或 `WRITE_WITH_IMM` WR | 没有 one-sided remote address；要设计 request/message |
| `WriteBatch` | 多个 `ibv_sge` + 多个链式 `ibv_send_wr`，一次或多次 `ibv_post_send` | `sendmsg/writev` 或自己聚合后 `send` |
| `WriteBySgList` | 一个/多个 WR，每个 WR 的 `sg_list` 含多个 SGE | `writev/sendmsg` 的多个 `iovec` |
| `ReadSingle/Batch` | `IBV_WR_RDMA_READ` + `ibv_post_send` | 发请求，让远端应用主动回传数据 |
| `Send` | `IBV_WR_SEND(_WITH_IMM)` + 对端 `ibv_post_recv`；大消息需 buffer 协商 | `send/sendmsg` + framing |
| `OnRecvCall` | poll 到 `IBV_WC_RECV` 后，应用自己调用 handler | `epoll` 可读后 `recv`、组出完整 message，再调用 handler |
| `OnImmRecvCall` | poll 到 `IBV_WC_RECV_RDMA_WITH_IMM`，读取 `wc.imm_data` | TCP 无原生 immediate data；放进应用 header |
| `DoneCallback` | poll send CQE，用 `wc.wr_id` 找 callback | send queue/协议定义的完成事件 |

如果你希望保留“高性能 one-sided RDMA”的语义，最直接的底层库是
**rdma-core 的 `libibverbs`**；如果不想自己交换 QPN/GID/PSN 和推进 QP，
可以再使用 **`librdmacm`** 做地址解析、建联和 QP 生命周期。

如果你不要求自己控制 verbs 细节，也可以选择另一层通信库，例如 UCX、libfabric、
MPI 或 NVSHMEM。但这些是“用另一套封装替换 Barex”，不是直接调用硬件。

### 9.2 没有 `WriteBatch`：自己链多个 RDMA Write WR

Barex 当前 RDMA 实现本质上会为每项构造：

```text
ibv_sge:
  addr   = local address
  length = bytes
  lkey   = local MR lkey

ibv_send_wr:
  opcode                 = IBV_WR_RDMA_WRITE
  wr.rdma.remote_addr    = remote address
  wr.rdma.rkey           = remote MR rkey
```

然后把多个 WR 用 `next` 链接并 post。简化的原生伪代码：

```cpp
std::vector<ibv_sge> sges(n);
std::vector<ibv_send_wr> wrs(n);

for (size_t i = 0; i < n; ++i) {
    sges[i].addr   = items[i].local_addr;
    sges[i].length = items[i].length;
    sges[i].lkey   = items[i].lkey;

    wrs[i].wr_id      = make_wr_id(batch, i);
    wrs[i].sg_list    = &sges[i];
    wrs[i].num_sge    = 1;
    wrs[i].opcode     = IBV_WR_RDMA_WRITE;
    wrs[i].wr.rdma.remote_addr = items[i].remote_addr;
    wrs[i].wr.rdma.rkey        = items[i].rkey;
    wrs[i].next = i + 1 < n ? &wrs[i + 1] : nullptr;

    // 常见优化：只让最后一个 WR 产生正常 CQE。
    wrs[i].send_flags = i + 1 == n ? IBV_SEND_SIGNALED : 0;
}

ibv_send_wr* bad_wr = nullptr;
int rc = ibv_post_send(qp, &wrs[0], &bad_wr);
```

`ibv_post_send()` 官方接口本身就接受一个 WR linked list，所以原生 verbs 已有
“批量 post”能力；但它没有 Barex 的 `WriteBatch` 业务语义。你还必须自己解决：

- `bad_wr` 之前哪些 WR 已成功 post；
- SQ depth 不足时排队还是返回错误；
- 多少 WR 使用 `IBV_SEND_SIGNALED`，怎样防止 CQ overrun；
- last signaled completion 是否代表整个 batch 可回收；
- 任意 WR 出错时怎样只触发一次 batch callback；
- `sges/wrs/items/MR` 在硬件取走之前的生命周期；
- batch 超过 QP depth 时怎样切片；
- 多线程 post 同一 QP 时的串行化策略。

Barex 当前优化路径同样把多个 `ibv_send_wr` 链起来，普通 Write 使用
`IBV_WR_RDMA_WRITE`，通常只给一批最后一个 WR 设置 `IBV_SEND_SIGNALED`，再用
自己的 `x_wr_id` 和 `call_once` 聚合完成。

现代 rdma-core 还提供 `ibv_wr_start()`、`ibv_wr_rdma_write()`、
`ibv_wr_complete()` builder API，可以在一个 critical region 中构造多个 WR。
它仍然只负责 post，不提供 Barex 的连接、buffer pool、callback 和 heartbeat。

### 9.3 没有 `Send` 和 `OnRecvCall`：使用 SEND/RECV 并自己 dispatch

最小 two-sided message 模型是：

```text
接收方：
  注册 recv buffer
  → ibv_post_recv(qp, recv_wr)

发送方：
  注册 send buffer
  → ibv_post_send(qp, IBV_WR_SEND)

接收方 RNIC：
  把 payload DMA 到预先 post 的 recv buffer
  → CQ 产生 IBV_WC_RECV
  → 应用 ibv_poll_cq()
  → 根据 wc.wr_id 找到 buffer
  → 调用自己的 on_recv(buffer, wc.byte_len)
  → 重新 post 一个 recv WR
```

原生 verbs 中不存在名为 `OnRecvCall()` 的函数。它是 Barex 的 callback interface。
如果不用 Barex，你通常写一个 CQ progress loop：

```cpp
ibv_wc wc[32];
while (running) {
    int n = ibv_poll_cq(recv_cq, 32, wc);
    for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            handle_wc_error(wc[i]);
            continue;
        }
        if (wc[i].opcode == IBV_WC_RECV) {
            RecvSlot* slot =
                reinterpret_cast<RecvSlot*>(wc[i].wr_id);
            on_recv(slot->buffer, wc[i].byte_len);
            repost_recv(slot);
        }
    }
}
```

这里的 `on_recv()` 就是你自己实现的 `OnRecvCall` 等价物。

Barex 小消息 `Send` 使用 `IBV_WR_SEND_WITH_IMM`，对端消耗已 post 的 Receive WR；
Barex 再根据 immediate message type 和内部 header 调用 `OnRecvCall`。

大消息不能简单理解为“verbs SEND 发不了”。RC 会处理 transport packet 分片，
但接收方必须提前准备足够大的 registered receive buffer，而且应用要知道 message
边界与 buffer ownership。Barex 为了提供“调用方不用预先知道 remote buffer”
的消息语义，自己实现了 rendezvous：

```text
SEND(length metadata)
← SEND(remote buffer address + rkey)
RDMA_WRITE_WITH_IMM(payload)
→ OnRecvCall
```

不用 Barex 时，你可以：

1. 始终预 post 足够大的 recv buffers，直接 SEND；
2. 使用 SRQ/receive buffer pool；
3. 自己实现上述 rendezvous；
4. 用 TCP/UCX/libfabric 等承担消息 framing 与 buffer 协商。

### 9.4 没有 `OnImmRecvCall`：检查 WRITE_WITH_IMM 的接收 CQE

普通 RDMA Write：

```text
发送方收到 IBV_WC_RDMA_WRITE
远端内存被写入
远端应用默认没有 receive CQE
```

如果远端应用需要通知，发送方使用：

```cpp
wr.opcode   = IBV_WR_RDMA_WRITE_WITH_IMM;
wr.imm_data = htonl(my_imm);
```

接收方必须提前准备 Receive WR。WRITE_WITH_IMM 会消耗一个 RQ entry，并在接收 CQ
产生 `IBV_WC_RECV_RDMA_WITH_IMM`。应用可以：

```cpp
if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
    (wc.wc_flags & IBV_WC_WITH_IMM)) {
    uint32_t imm = ntohl(wc.imm_data);
    on_imm_recv(imm);
    repost_recv_slot();
}
```

这个 `on_imm_recv()` 就是原生程序里对应 Barex `OnImmRecvCall()` 的 handler。
注意：

- payload 已由 RDMA Write 写到 `remote_addr/rkey` 指定的 MR，不在 receive buffer；
- Receive WR 仍被消耗，所以不及时 repost 会出现 RNR/retry；
- immediate data 在线上传输为 network byte order；
- 原生 immediate 是 32 bit；Barex 用高 8 bit 编码内部 message type，只把低
  24 bit 交给用户。

`OnImmRecvCall` 只告诉你“某个带 immediate 的写到达并满足该 transport 的完成
条件”。`imm` 如何映射成 buffer id、layer id 或 request id，是应用协议的一部分。

### 9.5 从零实现 Barex RDMA 子集的最小清单

即便只实现 RC + WriteBatch + Send，至少需要：

```text
1. ibv_get_device_list / ibv_open_device
2. ibv_alloc_pd
3. ibv_create_comp_channel / ibv_create_cq
4. ibv_reg_mr 或 ibv_reg_dmabuf_mr
5. ibv_create_qp
6. librdmacm 建联，或 TCP 交换 QPN/GID/PSN
7. ibv_modify_qp：RESET → INIT → RTR → RTS
8. 建立 recv buffer pool 并持续 ibv_post_recv
9. 构造 ibv_sge / ibv_send_wr，调用 ibv_post_send
10. ibv_poll_cq，按 wr_id/status/opcode dispatch
11. 实现 depth、backpressure、timeout、error 与 reconnect
12. 等所有 inflight WR 完成后反向销毁 QP/MR/CQ/PD/context
```

所以选择建议是：

| 目标 | 更合适的入口 |
|---|---|
| 学习每个 WQE/CQE、自己控制 GDR 与 QP | 原生 `libibverbs` |
| 想保留 verbs，但不想手写建联 | `libibverbs + librdmacm` |
| 想要 portable transport/active message | UCX 或 libfabric |
| 需要 MPI collective/P2P 语义 | MPI implementation |
| 当前 blade-kvt 的工程能力与监控 | 继续使用 Barex，或实现等价 transport adapter |

## 10. TX depth 与软件队列

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

## 11. `ibv_post_send` 到 callback

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

## 12. 完成错误

`HandleWcStatusError`：

1. 把 WC status 转成可读错误；
2. 调用失败 callback；
3. 接收 WR 释放内部 recv buffer；
4. 发送 WR 只释放 Barex header，不擅自释放用户 payload；
5. 销毁 channel。

见 `xcontext_impl.cc:913-1006`。

这是为什么应用 callback 必须正确管理 buffer 生命周期：同步提交错误和异步 WC 错误都可能发生。

## 13. 关键环境变量

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

## 14. 自检

1. `Send` 大消息为什么要先协商远端 buffer，而 `WriteBatch` 不需要？
2. Barex API 返回成功与 CQ completion 有什么区别？
3. 为什么 `tx_depth` completion 后才能归还？
4. callback 为什么既可能在 IO thread，也可能在线程池？
5. 没有 Barex 时，`WriteBatch` 为什么不是简单循环调用 N 次 `ibv_post_send`？
6. `OnRecvCall` 为什么不是 verbs API？它最底层由哪一种 CQE 触发？
7. 普通 RDMA Write 与 WRITE_WITH_IMM 对远端应用的可见通知有什么区别？
8. WRITE_WITH_IMM 的 payload 在 remote MR 还是 receive buffer？为什么仍要 post
   Receive WR？

## 参考

- [`ibv_post_send(3)`：post 链式 WR、RDMA Write/Send/Immediate](https://man7.org/linux/man-pages/man3/ibv_post_send.3.html)
- [`ibv_post_recv(3)`：Receive Queue 与 buffer 生命周期](https://man7.org/linux/man-pages/man3/ibv_post_recv.3.html)
- [`ibv_poll_cq(3)`：WC opcode、status、wr_id 与 immediate data](https://man7.org/linux/man-pages/man3/ibv_poll_cq.3.html)
- [`ibv_wr_*` builder API：批量构造 RDMA Write/Write With Immediate](https://man7.org/linux/man-pages/man3/ibv_wr_rdma_write_imm.3.html)
- [`rdma_cm(7)`：librdmacm 连接管理](https://man7.org/linux/man-pages/man7/rdma_cm.7.html)
- [Linux Userspace verbs access](https://docs.kernel.org/infiniband/user_verbs.html)
- [rdma-core](https://github.com/linux-rdma/rdma-core)
