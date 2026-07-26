# 02b. RDMA 操作、WR/CQ 完成与可靠性

## 1. Two-sided 与 One-sided

![SEND/RECV、WRITE、WRITE_WITH_IMM 数据与完成语义](imgs/rdma_operations.svg)

### SEND/RECV

发送端：

```text
SQ: SEND(local SGE)
```

接收端必须提前：

```text
RQ: RECV(local SGE)
```

RNIC 匹配 RQ WQE，把 payload DMA 到接收 buffer，并在双方 CQ 产生 completion。

### RDMA WRITE

发起端 SQ WQE 同时携带：

```text
local SGE(addr, length, lkey)
remote addr + rkey
```

接收端 CPU 不 post 对应 Recv，也不天然收到“哪个业务消息完成”的通知。

### RDMA READ

发起端从远端 MR 读取到本地 SGE。响应数据沿网络返回，因此对 RTT、outstanding read、`max_rd_atomic` 更敏感。

### Atomic

对远端 8-byte 等受支持位置做 Compare-and-Swap 或 Fetch-and-Add。限制依硬件和 QP type。

## 2. WRITE WITH IMMEDIATE

WRITE_WITH_IMM 同时：

1. 把 payload one-sided 写到远端地址；
2. 在远端 CQ 产生带 `imm_data` 的 receive completion。

但它通常会消耗远端一个 Receive WQE，因此远端仍需维护 RQ；不足会产生 RNR。

blade-kvt staged RDMA 用 immediate 编码 remote staging buffer id：

```text
WriteSingle(signal_peer=true, imm_data=buffer_id)
  → remote OnImmRecvCall
  → 找 staging buffer
  → H2D scatter
```

direct RDMA 不需要远端 per-layer callback，所以 `WriteBatch` 不 signal peer。

## 3. WR、WQE 与 SGE

应用构造 WR：

```cpp
ibv_send_wr wr = {};
wr.wr_id = opaque_cookie;
wr.opcode = IBV_WR_RDMA_WRITE;
wr.sg_list = sges;
wr.num_sge = n;
wr.wr.rdma.remote_addr = remote;
wr.wr.rdma.rkey = rkey;
```

provider 将它转换为硬件 WQE，写入 SQ 并敲 doorbell。

SGE：

```text
addr + length + lkey
```

scatter/gather 能避免应用先把离散 local buffer 合成连续 buffer，但受 `max_send_sge` 限制。

## 4. post 成功不等于完成

三个阶段：

| 阶段 | 能说明什么 |
|---|---|
| `ibv_post_send` 返回 0 | WQE 已被 provider 接受 |
| CQE status success | RNIC 按 transport 语义完成该 WR |
| 应用层 ACK/response | 远端业务处理完成 |

对 RC RDMA Write，本地成功 completion 通常意味着远端 RNIC 已接受并把写操作完成到目标 memory ordering 域，但不意味着远端 CUDA kernel 已消费数据，也不意味着远端应用已处理 request。

blade-kvt 的四层边界：

```text
WriteBatch submit
  → local CQ completion
  → staged/TCP remote H2D response（direct 无此层）
  → send-done 业务通知
```

## 5. Signaled 与 Unsignaled

如果每个 WQE 都 signaled，CQE 与 polling 压力很大。常见优化是：

- 多个 WQE unsignaled；
- 每 N 个或 batch 最后一个 signaled；
- 最后一个 completion 代表此前同 SQ 有序 WQE 已推进。

但必须避免：

- SQ 被 unsignaled WQE 填满；
- 无 completion 可用于回收 WR/buffer；
- 错误时无法准确映射 batch。

Barex `ACCL_WRITEBATCH_OPT` 与 batch completion 聚合相关。阅读 `MakeSendBatch` 时要重点核对：

- 哪个 WR 持有非空 `wr_id`；
- callback 调几次；
- permit 按多少 WR 归还；
- error WC 如何覆盖前序 unsignaled WR。

## 6. Inline

`IBV_SEND_INLINE` 让小 payload 直接复制进 WQE/doorbell record，RNIC 不再 DMA 读取应用 buffer。

结果：

- post 返回后原 buffer 通常可立即复用；
- 减少一次 PCIe DMA read；
- 受 `max_inline_data` 限制；
- post 本身 copy 成本增加。

Barex `InlineSend` 用于大消息 metadata 等控制消息：

```text
IBV_WR_SEND_WITH_IMM | IBV_SEND_INLINE | IBV_SEND_SIGNALED
```

见 `xchannel_impl.cc:1008-1065`。

## 7. 顺序与 Fence

RC 同一 SQ 提供有序语义，但要区分：

- WR 执行顺序；
- PCIe/设备 memory visibility；
- GPU kernel 的缓存与 stream ordering；
- 跨多个 QP 的顺序。

多个 Barex channel 对应多个 QP，不能仅凭单 QP ordering 推导跨 channel 的全局顺序。blade-kvt 用 futures 汇总所有 QP completion。

`IBV_SEND_FENCE` 主要约束某些 read/atomic 与后续操作；不是通用 CPU/GPU memory barrier。

## 8. RNR

Receiver Not Ready 出现在需要 RQ WQE 的操作到达、但接收端没有可用 Recv：

- SEND；
- SEND_WITH_IMM；
- WRITE_WITH_IMM。

RC 可按 `min_rnr_timer/rnr_retry` 重试。重试耗尽出现：

```text
IBV_WC_RNR_RETRY_EXC_ERR
```

Barex 在 channel 初始化时批量 post recv，并在 consume 后补充。若 callback/IO thread 不及时归还，或 `rx_depth` 太小，就可能 RNR。

## 9. Transport retry 与 timeout

RC 发送方维护 PSN、ACK/NAK、retry timer。关键配置：

- timeout；
- retry count；
- RNR retry；
- max outstanding RDMA read/atomic。

Barex 映射：

| Barex 环境变量 | Verbs/QP 语义 |
|---|---|
| `ACCL_RETRANSMIT_TIMEOUT` | local ACK timeout |
| `ACCL_RETRY_CNT` | transport retry count |
| `ACCL_RNR_RETRY` | RNR retry count |
| `ACCL_MIN_RNR_TIMER` | responder RNR timer |
| `ACCL_MAX_RD_ATOMIC` | initiator outstanding read/atomic |
| `ACCL_MAX_DEST_RD_ATOMIC` | responder resources |

timeout 编码通常不是毫秒直填，而是规范定义的指数单位；必须看 Barex 转换逻辑与设备实际值。

## 10. 丢包、乱序与“go-back-N”

教学中常把传统 RC/RoCE 描述成“丢一包后重传后续窗口”。这个直觉能解释丢包放大，但不是所有现代 NIC/模式的精确行为：

- 基础 RC 依靠 PSN、ACK/NAK 与 retry；
- 不同 RNIC 对乱序包的处理能力不同；
- 现代设备可能支持 out-of-order receive 或选择性重传扩展；
- 配置和固件会改变实际表现。

所以生产分析应看：

- packet sequence/retry counters；
- NAK/RNR/timeout；
- out-of-order capability；
- NIC vendor 文档；
- 交换机 drop/ECN/PFC counters。

不要仅凭“RoCE 一定 go-back-N”推导具体重传倍数。

## 11. Completion 错误的传播

Barex `ProcessOneIoEvent`：

```text
wc.status != SUCCESS
  → HandleWcStatusError
  → callback(error)
  → buffer/header cleanup
  → DestroyChannel
  → IoEventOccur(false) 归还/清理 permit
```

blade-kvt wrapper 再把 callback error 写入 promise，最终在 `future.get()` 抛出。

因此一条错误从硬件到 Python 的路径是：

```text
WC status
  → Barex Status
  → C++ exception_ptr
  → KvSendStub catch
  → Request state FAILED
  → send-done code=500
```

## 12. Send 与 Write 如何选择

| 需求 | 更适合 |
|---|---|
| 不想预先交换远端地址 | SEND/RECV |
| 消息到达要触发远端 callback | SEND/RECV |
| 已知目标内存，追求零 CPU payload path | RDMA WRITE |
| 写完还要轻量通知远端 | WRITE_WITH_IMM |
| 拉取远端已发布数据 | RDMA READ |
| 离散 local 段写连续 remote | SG list |
| 多个独立 remote range | WriteBatch |

## 13. 自检

1. 为什么 `ibv_post_send` 成功后还不能释放 non-inline buffer？
2. WRITE_WITH_IMM 为什么既是 one-sided write 又需要 RQ？
3. unsignaled WR 如何回收，错误时有什么复杂性？
4. 多 QP 为什么不能只依赖 RC 的单 SQ 顺序？

## 参考

- [rdma-core libibverbs](https://github.com/linux-rdma/rdma-core)
- [Linux InfiniBand/RDMA interfaces](https://docs.kernel.org/driver-api/infiniband.html)

