---
title: "05 KV 接收路径与完成语义：TCP、RDMA Direct、staged RDMA"
date: 2026-07-30
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 05 KV 接收路径与完成语义：TCP、RDMA Direct、staged RDMA

## 1. D 侧不是“收到包后再找 Block”

D Scheduler 在请求进入普通 Scheduler 前已经分配目标 Block，并把 Block ID 发给 P。
D Worker 启动时又把整块 KV tensor 注册给 Blade-KVT server。

P 计算：

```text
D block ID + layer/tensor layout
  → dst_offset
  → D registered tensor base + dst_offset
```

所以数据一旦到达，就直接落到未来 attention kernel 会读取的位置。

## 2. 三条路径总览

![Blade-KVT 三种接收数据路径](/imgs/vllm-pd-receive-paths.svg)

```text
RDMA Direct:
P GPU MR ──RNIC RDMA WRITE──> D GPU MR

TCP:
P GPU ─D2H gather→ P pinned host ─TCP→ D host
      ─H2D scatter→ D GPU

RDMA staged:
P GPU ─D2H gather→ P pinned host ─RDMA→ D pinned host
      ─H2D scatter→ D GPU
```

“RDMA”只描述 RNIC 跨网络访问注册内存的方式。目标可以是 host MR，也可以是 GPU MR。
因此 staged RDMA 仍是 RDMA，但不是 GPUDirect RDMA data path。

## 3. RDMA Direct 接收

D `RDMAServer::start_server()` 为每层每个 tensor 的 GPU buffer 取得：

```text
base pointer
rkey
```

P 首次连接后用一个 Barex RPC `get_mem_handles()` 获取这些 handle。之后 P 的
`RDMAChannel::send_data()` 直接构造：

```text
local: P GPU addr + src_offset, lkey
remote: D GPU addr + dst_offset, rkey
operation: RDMA WRITE
```

D CPU 不为每个数据块执行 memcpy，也不一定为每个 WRITE 收到应用层消息。RNIC 按
rkey/地址校验后，通过 D 侧 PCIe/互连写 GPU HBM。

### D 什么时候“可见”

当前 Blade-KVT 不在每个 RDMA Write 后让 D Worker Python 收一个回调。可见性依赖
RDMA completion/ordering，以及 P 只有在所有 Write completion 后才发上层
`SEND_DONE/KVTResp`。D Scheduler 收到 KVTResp 后才把请求推进到可运行状态。

因此控制面建立了：

```text
all P local write completions
  happens-before
P SEND_DONE aggregation
  happens-before
KVTResp
  happens-before
D request enters normal scheduler
```

这条链是 D 不会过早运行 attention 的关键，而不是 D Python 为每层检查 CQE。

### CRC 可选校验

打开 CRC 时，P 在写完后向 D 发 RPC，让 D 通过 GDR 映射读取目标范围计算 CRC，再与
P 本地 CRC 比较。它提供更强的数据内容验证，但：

- 有额外读显存和 RPC 开销；
- 没开 GDR/布局不支持时可能降级；
- 默认 write completion 本身不验证每个字节的业务内容。

## 4. TCP 接收

P `TCPChannel::send_data(layer)` 做：

1. 把源 `IpcBlock` metadata 写入 host message；
2. CUDA kernel 将离散 P GPU 区间 gather 到 pinned host blob；
3. 同步 D2H stream，确保 host blob 可发送；
4. Barex `Send()` 发送整条消息。

D `TCPServer::handle_kv_cache_data()`：

1. 校验 magic、layer、tensor 数；
2. 反序列化 `IpcBlock`；
3. 检查每个 `dst_offset + length` 没越过注册容量；
4. H2D scatter 到 D GPU；
5. 同步 H2D stream；
6. 返回带时间戳的响应。

P 把每个 reqid 对应一个 promise。响应进入 `CliBarexCtx::RpcCtxCb` 后：

```text
reqid → pop callback → set promise → TCPChannel::flush future ready
```

所以 TCP flush 包含接收端 H2D 完成。

## 5. 为什么 TCP 要把离散小块聚成一个大 host message

KV Block ID 常常不连续。若每一小块单独发送：

- 每块都有 RPC/header；
- 每块都可能分配 buffer、提交 send、产生 callback；
- syscall/doorbell/WQE/CQE 固定开销占比高；
- 小包更难利用 BDP；
- 接收端 callback queue 更拥挤。

Blade-KVT 先把源 GPU 离散块 gather 到连续 pinned host blob，再一次发送，D 再
scatter。它牺牲两次 GPU↔host copy，换取较高的网络传输粒度。

这不是 TCP 协议强制要求连续“业务对象”；socket 本来就是 byte stream。连续 host
buffer 是为了减少上层提交和 gather I/O 开销。

## 6. staged RDMA 接收

staged 路径与 TCP 相似，但网络段使用 RDMA 写 D 的 host staging MR：

```text
P D2H gather
  → RDMA Write / Write with notification
  → D server 得知 staging data ready
  → D H2D scatter
  → response/completion
```

适用于 GPU Direct 不可用、拓扑/驱动不支持或希望用 host buffer 隔离布局的情况。
性能取决于两次 PCIe copy、host memory bandwidth 和 staging buffer 管理。

Kimi K3 + Eagle 的异构 tensor slot 当前代码明确禁止 staged RDMA，因为 staged path
没有实现相同的 inactive-slot 布局语义。

## 7. 四个“完成点”逐一回答

### A. API submit 成功

Barex API 返回 `BAREX_SUCCESS` 只说明成功接受任务，不代表 NIC 已传输。

### B. transport local completion

Barex callback 成功：

- RDMA WRITE：本地 RNIC 已完成该 WR 的本地完成语义，源 buffer 可按约定复用；
- TCP Send：消息发送操作完成，但 Blade-KVT TCP 还要求对端响应；
- 不等于 D 应用已经执行 decode。

这里的“本地完成语义”不是“数据刚离开 P 的 RNIC”。对 RC RDMA WRITE 来说，正常的
成功 CQE 建立在 RDMA transport 已完成远端操作的基础上；远端 rkey、地址或 QP
状态等可检测错误，会通过 NAK/retry/error CQE 反映到发送端。它仍然不是“D 上的
GPU kernel 已经读取并验证过每个字节”。准确的 ACK/CQE 时序见 §8.1。

### C. Blade-KVT batch completion

`channel->flush()` 返回，`KvSendStub` 将 request 标记 OK/FAILED，并发 `SEND_DONE`。

### D. vLLM request completion

PBackend 等所有 TP worker、所有 fan-out target 的 `SEND_DONE`，再回复 D；D
`mark_loaded` 后才可把 Request 交还 Scheduler。

## 8. RDMA callback 是否等 ACK

RoCE RC/RDMA reliability 的 ACK/retry 在 RNIC/transport 层完成。应用看到成功 CQE 前，
RNIC 已满足该 transport 的完成条件。这个 ACK 不是 D 侧 CQE 返回给 P：

- P 有自己的 CQE；
- D 在 Direct 路径下**没有**任何 CQE。`WriteBatch` 用的是纯
  `IBV_WR_RDMA_WRITE`（`barex/impl/rdma/xchannel_impl.cc:1652`），不是
  `IBV_WR_RDMA_WRITE_WITH_IMM`，因此不消耗 D 的 Receive WQE、不产生 D 的 CQE、
  D CPU 也不被唤醒；
- 网络 ACK 由 RNIC 协议栈处理；
- `SEND_DONE` 是 Blade-KVT 应用层另发的 TCP RPC。

因此本章后文凡是说“成功 CQE”，指的都是 **P 端 send CQ 上的 CQE**。不存在“D 的
CQE”这一环，也就不存在“等 D 的 CQE 再回 ACK”这种时序。

### 8.1 ACK 与 CQE 的先后：ACK 在 CQE 之前

完整时序是：

```text
P post RDMA WRITE
  → 网络可靠传输
  → D RNIC 校验 remote addr / rkey
  → D RNIC 把该 WR 的 Memory Write TLP 按序推给目标 memory domain
  → D RNIC 回 ACK                      ← 责任方是 D 的硬件
  → P RNIC 收到 ACK
  → P 的 send CQ 产生 IBV_WC_SUCCESS    ← 建立在收到 ACK 之上
  → Barex DoneCallback(Status::OK)
```

**D 的 ACK 先，P 的 CQE 后。** P 的 CQE 之所以有意义，正是因为它以收到 D 的
transport ACK 为前提。

补两个容易记错的点：

**（1）ACK 不是“DMA 写完”才发。** IB 规范要求 responder 回 ACK 前请求已
*executed*，对 WRITE 而言即数据已 “placed into memory”。但“placed”的实现含义
是：RNIC 已把全部 Memory Write TLP 按序提交，并依赖 **PCIe posted-write
ordering** 保证后续经同一路径的访问能看到它。PCIe 的 Memory Write 是 posted
的——没有 completion TLP，发送方交给 root complex 就算完事，RNIC 无法知道写是否
真的落到 DRAM，除非额外补一次读。所以：

```text
D RNIC 回 ACK
= 所有 write TLP 已按序推出，同路径后续访问可见
≠ 数据已落到 HBM 存储单元
≠ GPU SM 能读到
```

这个区别对两类目标的后果完全不同：

| 目标 | PCIe ordering 是否够用 |
|---|---|
| host memory（TCP / staged 的落点） | 够。CPU 读也走同一 root complex，必然排在 posted write 之后 |
| GPU HBM（GPUDirect RDMA Direct 路径） | **不够**。GPU SM 直读 HBM，不经过那个 ordering domain |

ACK 发出时，数据可能仍在 root complex 或 GPU 的 PCIe inbound 路径上。这正是
§8.3 末尾“内存可见性”那段的物理根源，也是 NVIDIA GPUDirect RDMA 文档专门写
Synchronization and Memory Ordering 一节的原因。它不是保守措辞，是规范与硬件之间
的真实缺口。

**（2）ACK 可以合并。** RC 下 responder ACK 一个较大的 PSN 就隐式确认了之前所有
PSN，不是每个 WR 一个 ACK 包。

### 8.2 P 端 CQE 的粒度：一个 batch 可能只有一个 CQE

Barex `WriteOrReadBatchOnce()` 支持选择性签名，由 `writebatch_optim_` 控制
（`xchannel_impl.cc:1633-1661`）：

```text
writebatch_optim_ <= 1  → 每个 WR 都有 x_wr_id，都带 IBV_SEND_SIGNALED
writebatch_optim_ == 2  → 只有 batch 内最后一个 WR 带 IBV_SEND_SIGNALED，
                          且只有它持有 done callback（前面的 done 是空 lambda）
```

也就是说 optim=2 时，整个 batch 只产生一个 CQE，靠 RC 的按序完成语义覆盖前面所有
WR：最后一个 WR 成功完成，意味着同 QP 上先于它 post 的 WR 也都已成功。

两个推论：

- 未签名 WR 的 SQ 槽位要等后续签名 CQE 才能回收，`writebatch_optim_` 调大时 SQ
  深度和 batch 大小要匹配，否则 SQ 满会导致 post 失败；
- 调试时不要假设“每个 WRITE 都有一条 completion 日志”，缺日志不等于缺传输。

### 8.3 “数据到达 D RNIC 后，RNIC→GPU 又失败”会怎样

先修正一个容易造成误解的两段式想象：

```text
错误想象：
网络到达 D RNIC → P 立刻收到成功 callback
                 → D RNIC 再慢慢写 GPU，写失败也与 P 无关

更准确的抽象：
P post RDMA WRITE
  → 网络可靠传输
  → D RNIC 校验 remote address/rkey
  → D RNIC 发起对 GPU BAR/HBM 的 peer DMA 写
  → RDMA transport 满足成功条件
  → P 的 send CQ 产生 IBV_WC_SUCCESS
  → Barex DoneCallback(Status::OK)
```

也就是说，对一个正常工作的 RC/GPUDirect RDMA 栈，发送端成功 callback 不应只表示
“包到达了 D 网卡”。如果错误能被 RNIC、PCIe、peer-memory provider 或 QP 检测，并且
发生在 WR 尚未成功完成时，通常会表现成：

```text
D 目标 rkey/地址无效、权限错误、QP/链路故障、可报告的设备错误
  → remote access/operation/retry/fatal 等 WC error
  → Barex HandleWcStatusError
  → DoneCallback(error)
  → Blade-KVT promise.set_exception()
  → RDMAChannel::flush() 的 future.get() 抛异常
  → KvSendStub catch
  → ch.reset() + ReqState::FAILED
  → SEND_DONE code=500
```

对应当前源码：

- Barex `XContextImpl::ProcessOneIoEvent()` 只有在 `wc.status ==
  IBV_WC_SUCCESS` 时才进入 `HandleWriteComplete()` 并调用
  `DoneCallback(Status::OK())`；
- 非成功 WC 进入 `HandleWcStatusError()`，callback 携带
  `BAREX_ERR_WC_STATUS`，并销毁 channel；
- Blade-KVT `rdma_channel.cpp::WriteBatch()` 将 callback error 写入
  promise；
- `RDMAChannel::flush()` 的 `future.get()` 把异常抛给
  `KvSendStub::do_task()`；
- 最终请求状态变成 `FAILED`，request 级 `send-done` 使用 `code=500`。

但是，**P 端的**成功 CQE 有明确边界。以下这些问题它一律覆盖不到——它们要么发生在
CQE 产生之后，要么硬件根本无法检测、无法归因给这个 WR。注意 D 侧没有 CQE，所以这
里说的全部是 P 端观察能力的上限：

1. D GPU 在 CQE 之后 reset、进程退出或目标 Block 被错误地提前释放/复用；
2. GPU kernel 与 GPUDirect RDMA 并发读写同一范围，造成可见性或数据竞争；
3. 合法地址内写入了错误 offset，transport 成功但业务布局错误；
4. 静默数据损坏，没有任何一层产生 error syndrome；
5. 底层已经报告成功，稍后才出现 PCIe AER/GPU Xid；已产生的成功 CQE不会被
   “撤销并重发成失败”。第 5 条与 §8.1 的 posted-write 语义直接相关：ACK/CQE 只
   代表 TLP 已按序推出，写在 PCIe 链路或 GPU 侧出错要晚得多才被观测到。

Direct `WriteBatch` 又明确“不通知 receiver”，因此 D Worker 没有一个逐 WRITE
callback 可以在上述情况发生后主动把失败回传给 P。`SEND_DONE` 也不是 D 发出的
确认；它是 P 在 `flush()` 成功后自己发出的业务通知。因此默认路径的实际语义是：

```text
SEND_DONE success
≈ P 观察到所有 RDMA WR 成功完成，并据此声明该 request 的数据面完成
≠ D GPU kernel 已读取并验证 KV 内容
```

若系统需要更强的端到端保证，可以增加：

- **接收端内容校验**：Blade-KVT 已有 `BLLM_KVTRANS_CRC=1`。P 在 write
  completions 后请求 D 通过 GDR CPU 映射读取目标范围并计算 CRC；CRC 不同会使
  `flush()` 失败，从而发送 `code=500`。该开关默认值是 `0`，而且在 GDR 映射或
  tensor 布局不满足时当前实现会降级，不能把“配置了 CRC”直接等同于“每次都实际
  校验”；
- **D 侧 request-ready ACK**：D 在完成必要的 memory ordering、范围/版本校验后，
  再向 P 或 Scheduler 回 ACK。这样完成语义比发送端 CQE更强，但会多一次控制面
  往返；
- **Block generation/epoch**：把 request、Block ID 与 generation 绑定，拒绝迟到
  DMA 或过期完成对已经复用的 Block 生效；
- **GPU/PCIe health monitoring**：将 Xid、PCIe AER、RNIC async event 与受影响的
  in-flight request 关联；无法精确关联时，至少让相关 worker/channel fail-stop，
  不继续把可疑 KV 标记为可用。

还要单独处理**内存可见性**。NVIDIA GPUDirect RDMA 文档指出，第三方设备写 GPU
内存与 GPU kernel 并发访问时可能观察到旧值、部分值或乱序值；必须让 RDMA 完成先
对将要提交依赖 kernel 的 CPU 控制线程成立，再通过 CUDA work submission/
synchronization 建立正确顺序。这属于同步协议问题，不一定产生 RDMA error CQE。

官方参考：

- NVIDIA GPUDirect RDMA，Synchronization and Memory Ordering：
  <https://docs.nvidia.com/cuda/gpudirect-rdma/#synchronization-and-memory-ordering>
- `ibv_poll_cq(3)` 的 WC status 与错误字段：
  <https://man7.org/linux/man-pages/man3/ibv_poll_cq.3.html>
- responder 何时可以回 ACK（WRITE 的 “placed into memory” 语义）：IBTA
  InfiniBand Architecture Specification Vol.1，Transport Layer 中 Reliable
  Connection 的 ACK 生成与 ordering 章节；
- PCIe Memory Write 是 posted、无 completion TLP，可见性依赖 ordering 规则：PCI
  Express Base Specification，Transaction Ordering 章节。

## 9. 接收端断开时可能看到什么

### 在建连前

- naming 找不到 target；
- Connect future 失败；
- ChannelFactory 尝试其他 protocol，全部失败则抛异常；
- `KvSendStub::do_task()` 捕获并把 batch request 标 FAILED。

### in-flight TCP/staged

- Send callback 返回 error，`on_send_error()` 找到 reqid callback；
- future 以异常完成；
- 或 flush 等到 `env_rpc_timeout_s()` 后主动注入 timeout；
- channel reset，下轮重新创建。

### in-flight RDMA Direct

- provider 通常应以错误 CQE/callback 完成失败 WR；
- `WriteBatch` promise 变异常，flush 抛出；
- 但 `RDMAChannel::flush()` 当前直接 `future.get()`，没有 Blade-KVT 自己的
  `wait_for(env_rpc_timeout_s())`；
- 如果底层既不给 completion 也不给 error，发送线程可能无界等待。

最后一点是代码审计风险，不应因为“RC 理论上可靠”就忽略。

## 10. 数据正确性验证层次

| 层次 | 可发现的问题 | 发现不了的问题 |
|---|---|---|
| 边界检查 | 错 offset、越界、0 length | 合法范围内写错内容 |
| transport completion | WR/连接错误、超时 | 应用布局算错但传输成功；posted write 尚未对 GPU kernel 可见 |
| CRC | 内容不一致 | 双方用同样错误范围且碰巧一致 |
| 请求级数值测试 | 最终 logits/KV 对比 | 极低概率、未覆盖 shape |

生产系统需要至少同时有 transport 指标和端到端正确性抽检。

## 11. 自检

1. RDMA Direct 为什么仍然经过 D 侧 PCIe/互连？
2. D 没有 receive CQE 时，如何避免在数据到齐前运行 request？
3. TCP flush 为什么比 RDMA Direct flush 更接近远端完成？
4. staged RDMA 与 GPUDirect RDMA 的区别是什么？
5. 为什么 transport success 仍可能得到错误 KV？
6. D 的 transport ACK 与 P 的 CQE 谁先谁后？为什么 Direct 路径下不存在“D 的
   CQE”？
7. 为什么“ACK 已返回”对 host memory 目标够用，对 GPU HBM 目标不够用？
8. `writebatch_optim_ == 2` 时一个 batch 只有一个 CQE，为什么这仍然是安全的？
