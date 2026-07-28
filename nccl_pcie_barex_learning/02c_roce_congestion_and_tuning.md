# 02c. RoCE、拥塞控制、PFC/ECN 与重传

## 0. 先用“排队”理解拥塞

交换机的输出端口像一个收费站。若 8 个输入端口同时各以 100 Gb/s 向一个
100 Gb/s 输出端口发送，短时间到达速率可达 800 Gb/s，而离开速率只有
100 Gb/s：

```text
进入 800 Gb/s → [交换机队列不断增长] → 离开 100 Gb/s
```

队列满后只能：

- 丢包；
- 暂停上游；
- 提前标记拥塞，让发送端降速；
- 或组合这些机制。

因此“每张网卡都是 100G”不代表多对一时不会拥塞。EP 的 All-to-All 和
KV cache 聚合都可能产生瞬时 incast。

三个经常混淆的指标：

- **带宽**：端口每秒最多发送多少 bit；
- **队列深度**：暂时能缓存多少拥塞数据；
- **RTT**：拥塞信号传回发送端并生效要多久。

“队列至少要容纳反馈生效前仍在路上的数据”必须指明是哪一种队列：

- 对发送吞吐而言，发送端 outstanding window 要能覆盖足够的 BDP；
- 对 PFC/ECN 拥塞瞬态而言，拥塞点 buffer/headroom 要吸收反馈生效前继续到达的
  excess traffic；
- verbs Receive Queue 深度解决的是 SEND/WRITE_WITH_IMM 的 receive credit，
  不是用来缓存整条路径 BDP，也不是交换机防丢包 buffer。

三者都受时延影响，但单位、位置和故障表现完全不同。

## 1. InfiniBand、RoCE 与 iWARP

| 技术 | 链路/网络 | 常见 transport |
|---|---|---|
| InfiniBand | 专用 IB fabric | IB RC/UD 等 |
| RoCEv1 | Ethernet L2，不可跨 L3 路由 | IB transport over Ethernet |
| RoCEv2 | UDP/IP，可路由 | IB transport over UDP/IP/Ethernet |
| iWARP | TCP/IP | RDMA over TCP |

Verbs API 可以相似，但 packet、拥塞和运维模型不同。

## 2. RoCEv2 packet 直觉

```text
Ethernet
  → VLAN/PCP（可选）
  → IP/DSCP/ECN
  → UDP dst port 4791
  → IB transport headers
  → payload + invariant CRC
```

RoCE 数据常绕过普通 kernel network stack，因此普通 `netstat`/socket counters 未必能看到完整流量；应看 RDMA/NIC hardware counters。

## 3. 为什么低丢包仍重要

RC 对丢包会 retry，但高带宽长 RTT 网络的 BDP 很大：

```text
BDP = bandwidth × RTT
```

例：

```text
400 Gbit/s × 10 ms
= 4 Gbit
= 500 MB
```

反馈回来前可能已有数百 MB 在途。拥塞点 buffer 远小于此值时，会产生 drop、retry、尾延迟和吞吐塌陷。

### 3.1 先修正一句表述：RTT 是从 us 级“升到”ms 级

跨地域或跨长距离网络中，应该说：

```text
RTT 从微秒级升高到毫秒级
```

而不是“从 us 级降至 ms 级”。数值和等待时间都增大了。RTT 增大时：

```text
带宽保持不变
  → BDP 增大
  → 为保持同样吞吐，需要更多 bytes outstanding
  → 对小 WQE，还需要更多 outstanding WQEs
```

### 3.2 BDP 不是链路能容纳数据的绝对上限

BDP 更准确的含义是：

> 当发送速率等于目标带宽时，一个 RTT 内发送的数据量；也就是让链路持续忙碌所需
> 的典型在途窗口。

#### BDP 是“数据大小”吗？

**从单位和计算结果来看，是。**BDP 是一个数据量（data amount），通常用 bit、byte、
KB、MB 或 GB 表示。因为：

```text
bit/s × s = bit
```

例如：

```text
200 Gbit/s × 10 ms
= 200 × 10^9 bit/s × 0.01 s
= 2 × 10^9 bit
= 250 × 10^6 byte
= 250 MB
```

这里的 250 MB 是一个“窗口大小”的计算结果，不是说网络中存在一个名为 BDP 的
250 MB 文件或数据包。它也不是：

- 单个 application message 的大小；
- 单个 WQE 的 payload 大小；
- 单个以太网包或 RoCE 包的大小；
- 发送队列或交换机 buffer 的实际硬件容量；
- 网络在任何时刻都必定保存的数据量。

更准确地说，BDP 把“速率”和“反馈时延”转换成了“要维持该速率，反馈返回前需要允许
多少数据尚未完成”。可以把链路想象成一根水管：

```text
带宽 bandwidth：水管每秒最多流过多少水
RTT：放出一滴带标记的水，到确认它到达并收到反馈需要多久
BDP：在这段反馈等待时间内，以目标速率总共可以放出多少水
```

因此，应区分三个相近但不完全相同的概念：

| 名称 | 含义 | 是否一定等于 BDP |
| --- | --- | --- |
| 理论 BDP | 目标带宽 × RTT 得到的数据量 | 定义本身 |
| 实际 in-flight bytes | 已发出但尚未被发送端确认完成的数据 | 不一定；取决于发送速率、拥塞、重传和窗口 |
| 配置的发送窗口 | 软件、SQ、RNIC 允许 outstanding 的数据量 | 为跑满链路通常应不小于目标 BDP，但受多层限制 |

还要注意“在物理链路上”和“在发送端看来尚未完成”不是完全相同的集合：

```text
某一个方向的物理路径中正在传播的数据量
  ≈ bandwidth × one-way delay

可靠发送端为了等到返回的 ACK/响应而需保留的 outstanding window
  ≈ bandwidth × RTT
```

网络性能调优通常关心第二个量，所以 TCP/RDMA 的发送窗口讨论常用 RTT 计算 BDP。
例如请求的数据可能已经到达对端、甚至已经写入内存，但 ACK 尚在返回途中；这时数据
不再位于正向物理链路上，发送端却仍把相应请求视为 outstanding。

例如应用只以 20 Gbit/s 在一条 200 Gbit/s、RTT 10 ms 的链路上发送，则实际在途量
大约只有：

```text
20 Gbit/s × 10 ms = 25 MB
```

虽然按链路峰值能力计算的 BDP 是 250 MB。反过来，如果已经发生拥塞和排队，实测
RTT 会增大，实际 outstanding 数据也可能超过使用“空载 RTT”算出的基础 BDP。

如果窗口小于 BDP，链路通常“吃不饱”；如果发送端在拥塞点已出现后仍继续高速发送，
在途数据加上新到达数据可以超过理想 BDP，多出来的部分会在瓶颈队列排队，或者被
PFC/ECN/丢包机制处理。

单位换算：

```text
BDP(bits)  = bandwidth(bit/s) × RTT(s)
BDP(bytes) = bandwidth(bit/s) × RTT(s) / 8
```

例如 200 Gbit/s：

| RTT | BDP |
| --- | --- |
| 100 us | 2.5 MB |
| 1 ms | 25 MB |
| 10 ms | 250 MB |
| 88 ms | 2.2 GB |

RTT 从 100 us 增至 10 ms，是 100 倍；要保持 200 Gbit/s，所需 outstanding bytes
也从约 2.5 MB 增至约 250 MB。

### 3.3 从 BDP bytes 推导所需 WQE 数

如果每个 WQE 平均承载 `S` bytes，第一阶估算是：

```text
required_outstanding_WQEs
  ≈ ceil(BDP_bytes / average_payload_bytes_per_WQE)
```

200 Gbit/s、10 ms RTT 的 BDP 约 250 MB：

| 每个 WQE 的 payload | 填充 250 MB BDP 约需 WQE 数 |
| --- | --- |
| 4 KiB | 61,036 |
| 64 KiB | 3,815 |
| 1 MiB | 239 |

这解释了为什么小包/WQE 场景对 WQE window 特别敏感：BDP 以 bytes 计，queue depth
却以 WQE 个数计。相同 250 MB：

```text
1 MiB/WQE 只需约 239 个 WQE
4 KiB/WQE 却需约 6.1 万个 WQE
```

此外，小 WQE 还有更多 per-message 成本：

- 更多 WQE fetch；
- 更多 doorbell/post；
- 更多 CQE 或 completion bookkeeping；
- 更高 packet/message rate；
- 更多软件 callback 和对象管理。

聚合小包的价值不只是减少 syscall/copy，还包括提高 `bytes per WQE`，让有限 WQE
window 能覆盖更多 BDP。

这个公式是估算，不是硬件能力证明。一个大 WRITE WQE 可以被 RNIC 按 MTU 拆成许多
packet，因此还要同时检查 packet/PSN window、firmware outstanding resource、
QP SQ capacity、CQ progress 和拥塞控制，而不能只计算 WQE 数。

### 3.4 为什么发送窗口不足会限制吞吐

假设发送端最多允许 `N` 个 WQE 未完成，每个 WQE 为 `S` bytes：

```text
outstanding_bytes = N × S

window_limited_throughput
  ≈ outstanding_bytes / RTT

实际吞吐上限
  ≈ min(link_bandwidth, outstanding_bytes / RTT)
```

例：只有 128 个 outstanding WQE、每个 WQE 4 KiB、RTT 10 ms：

```text
outstanding_bytes = 128 × 4 KiB = 512 KiB

throughput ceiling
  ≈ 512 KiB / 10 ms
  ≈ 52.4 MB/s
  ≈ 0.42 Gbit/s
```

这远低于 200 Gbit/s。典型时间线是：

```text
t0       连续发出 128 个小 WQE
t0+很短  RNIC outstanding window 达上限
         后续 WQE 留在 SQ/software queue
         链路可能出现空闲
t0+RTT   ACK/completion 开始返回，旧 WQE 被回收
         RNIC 才继续发后续 WQE
```

于是吞吐呈现“发一小段—等待—再发一小段”的 window-limited 行为。

但要注意：**发送窗口不足本身通常不会直接导致网络丢包**。它首先造成的是：

1. RNIC 不能继续把更多 WQE 变成在途数据；
2. 后续工作停留在 hardware SQ 或应用 software queue；
3. 链路利用率下降；
4. 请求排队延迟和尾延迟上升；
5. 软件队列也满时，提交失败或返回 queue-full；
6. 如果业务有 TTL，排队太久还会表现为应用 timeout。

真正“因为容纳不下而丢包”的队列通常是交换机 egress/ingress buffer 或 RNIC
packet buffer，而不是发送端 verbs SQ。

### 3.5 “发送队列深度”其实还要拆成三层

```text
业务线程
   │
   ▼
[应用/Barex software send queue]
   │ 获得发送 permit 后
   ▼
[QP Send Queue：已 post 的 WQE ring]
   │ RNIC fetch/issue
   ▼
[RNIC active outstanding WQE / packet / PSN window]
   │
   ▼
网络
```

| 层次 | 常见单位 | 满时会怎样 | 与 BDP 的关系 |
| --- | --- | --- | --- |
| Barex software queue | task/request | 等待；再满则 `BAREX_ERR_QUEUE_FULL` | 只存还没 post 的业务，不会增加真实 in-flight bytes |
| QP SQ `max_send_wr` | WQE | 不能继续安全 post；provider/app 必须等待 completion | 决定最多能挂多少未回收 WR |
| RNIC firmware active window | outstanding WQE/packet/PSN | RNIC 延迟发后续 WQE | 直接限制能变成在途数据的窗口 |
| network switch queue | byte/cell | ECN、PFC 或 drop | 吸收拥塞反馈生效前的 excess traffic |

`ibv_create_qp()` 的 `max_send_wr` 是应用请求的“SQ 中最大 outstanding WR 数”；
创建完成后 provider 会写回实际能力。它还受 device `max_qp_wr` 等能力上限约束。

`LOG_MAX_OUTSTANDING_WQE` 则是 ConnectX firmware/NV configuration 层的另一个
限制。该字段的设备说明通常是：

```text
单个 Transmit Work Queue 可持有的最大 outstanding/uncompleted WQE 数的 log2
```

所以值 `L` 表示的量级通常是 `2^L` 个 WQE；超过 active window 的附加 WQE 会被
延后，直到已有 WQE 完成。它不是 Barex 环境变量，也不等于 verbs 的
`max_recv_wr`。具体设备/firmware 对合法值、默认值、是否需要重启生效的定义，必须以
本机 `mlxconfig` 查询结果和对应 firmware 文档为准。

有效窗口是多重上限共同作用的结果，可以用下面的心智模型：

```text
effective outstanding window
  ≤ min(
      应用/Barex tx permit,
      QP max_send_wr,
      device max_qp_wr,
      firmware LOG_MAX_OUTSTANDING_WQE 对应窗口,
      packet/PSN window,
      operation-specific credits
    )
```

只把 `ACCL_TX_DEPTH` 调很大，而 firmware active window 仍很小，WQE 可能只是更多
地堆在 SQ 中；反过来只调大 firmware window，但 QP `max_send_wr` 或 Barex permit
仍为 128，也无法得到更大的业务 outstanding。

### 3.6 在 Barex 中会具体发生什么

当前 Barex RDMA 实现中：

```cpp
send_semaphore_ = config_.tx_depth;

qp_init_attr.cap.max_send_wr = config_.tx_depth;
qp_init_attr.cap.max_recv_wr = config_.rx_depth;
```

对应环境变量：

```text
ACCL_TX_DEPTH
ACCL_SOFT_TX_DEPTH
ACCL_RX_DEPTH
```

每次发送前 `PostSendOrEnqueue(permits, send)`：

```text
send_semaphore_ 足够
  → 扣除 permits
  → 调用 ibv_post_send

send_semaphore_ 不足
  → 放进 Barex send_queue_

send_queue_.size() >= soft_tx_depth
  → callback 返回 BAREX_ERR_QUEUE_FULL
```

completion 被处理后，`ReleaseAndPostSend()` 归还 permit，再从 software queue
继续取任务。因此长 RTT 下，completion/ACK 返回变慢会延长 permit 占用时间：

```text
RTT 增大
  → 同一发送速率下未完成 WQE 增多
  → send_semaphore_ 更容易耗尽
  → Barex software queue depth 增长
  → 再满时 BAREX_ERR_QUEUE_FULL
```

这条路径解释的是“发送端为什么卡住”。它和远端 RQ 是否充足是两个独立问题。

### 3.7 为什么不是优先调接收队列深度

verbs RQ 存放的是 Receive WQE，也就是“接收端为某些 operation 预先准备的接收
描述符”。它不是交换机 packet buffer，也不是通用的 RNIC ingress byte buffer。

不同 operation 是否消费 RQ：

| Operation | 是否消费远端 Receive WQE |
| --- | --- |
| SEND / SEND_WITH_IMM | 是 |
| RDMA WRITE | 否 |
| RDMA WRITE_WITH_IMM | 是，主要用于产生带 immediate 的 receive completion |
| RDMA READ | 否 |
| Atomic | 否 |

所以如果跨域 bulk path 使用普通 RDMA WRITE，远端 `rx_depth` 再大也不会增加 WRITE
的 in-flight window，因为 WRITE 根本不从 RQ 取 Receive WQE。

即使是 SEND，一个 Receive WQE 通常对应一条 message/WR 的接收 buffer，而不是“每个
网络 packet 一个 WQE”。一个大 SEND 被拆成多个 packet 后，仍是同一个 message
匹配一个 Receive WQE。RQ depth 以“可接收 message 数”衡量，不以“可缓存多少 BDP
bytes”衡量。

RQ 太浅的真实故障是：

```text
SEND/WRITE_WITH_IMM 到达
  → 没有 Receive WQE
  → responder 返回 RNR NAK
  → RC sender 按 min_rnr_timer/rnr_retry 重试
  → 重试耗尽后 IBV_WC_RNR_RETRY_EXC_ERR
```

这是 Receiver Not Ready，不是交换机 buffer overflow 导致的 packet drop。

因此：

```text
调大 TX/outstanding window
  → 解决“没有足够未完成工作填满长 RTT 链路”

调大 RX depth、加快 repost Receive
  → 解决“SEND/WRITE_WITH_IMM 到达时没有 receive credit”

调大/正确配置 switch buffer、PFC headroom、ECN/DCQCN
  → 解决“拥塞反馈生效前 packet 仍持续到达”
```

三者不能互相替代。

### 3.8 接收队列更深为什么不能阻止一般意义的丢包

一条包到达接收主机前，可能经过：

```text
sender SQ
  → sender RNIC egress buffer
  → 多级 switch ingress/egress queues
  → receiver RNIC packet/reorder buffer
  → PCIe/DMA
  → application receive buffer
```

verbs RQ 只影响最后阶段中 SEND 类 operation 是否有合法 destination buffer。它不能
扩大前面交换机的 egress queue，也不能改变 ECN threshold、PFC headroom、链路误码
或路由拥塞。

你说的“接收端队列越大越不容易丢包”在传统 socket/NIC RX ring 语境里有一定直觉，
但那个 RX descriptor ring、socket receive buffer、switch buffer 与 RDMA QP RQ
不是同一个对象。即使某些实现中它们最终都消耗内存，也不能把参数直接等同。

### 3.9 “队列要容纳反馈生效前的数据”到底是哪一个 buffer

这句话用在拥塞控制/PFC 时，通常指拥塞点的 byte buffer/headroom。

#### ECN/CNP

```text
switch queue 超过阈值
  → 标记 CE
  → packet 到 receiver
  → receiver 产生 CNP
  → CNP 回到 sender RNIC
  → sender 降速
```

从首次拥塞到 sender 真正减速有反馈延迟。在这段时间，多个 sender 仍可能继续注入
数据。拥塞点需要吸收的不是机械地等于“一整条链路 BDP”，更精确的近似是：

```text
queue growth
  ≈ Σ(max(sender_rate_i - bottleneck_share_i, 0))
    × feedback_delay
```

incast 时多个发送端的 excess rate 会叠加，所以增长可能非常快。

#### PFC

```text
buffer 达到 XOFF
  → 发 pause frame
  → pause 沿当前 hop 传播
  → 上游端口真正停止
```

从 XOFF 到 upstream 停止之间仍有在途 packet 到达，因此需要 lossless headroom。
这个 headroom 与端口速率、MTU、线缆/传播时延等有关。它是 switch/NIC port 的
packet buffer 配置，不是 verbs `rx_depth`。

### 3.10 增大 outstanding window 的收益与风险

收益：

- 长 RTT 下减少 link idle；
- 提高 bytes in flight；
- 小 WQE workload 可以显著提高吞吐；
- 减少任务长期停留在 Barex software queue。

风险：

- 一个 flow 可以更快注入更大的 burst；
- incast 时交换机 queue、PFC 和 drop 压力上升；
- RNIC 要维护更多 retry/completion/WQE state；
- SQ/CQ 和 host memory 占用增加；
- 出错时可能一次 flush 更多 WR；
- 排队更深可能提高 tail latency；
- CQ polling 不够快时，增大 SQ 只是在掩盖 progress bottleneck。

因此 `LOG_MAX_OUTSTANDING_WQE`、`ACCL_TX_DEPTH` 不应该无条件设为最大值。合理方法是：

1. 计算目标带宽与实测 RTT 的 BDP；
2. 统计实际 `bytes/WQE` 分布，而不是只看 packet MTU；
3. 估算所需 WQE window；
4. 核对 QP `max_send_wr`、device `max_qp_wr`、firmware window；
5. 逐级增加 TX depth，观察吞吐是否随之上升并最终平台化；
6. 同时观察 ECN/CNP、PFC、drop、retry、CQ backlog 与尾延迟；
7. SEND/WRITE_WITH_IMM 测试再独立扫描 RX depth；
8. 用业务并发和 incast 复测，不能只看单流带宽。

perftest 可以帮助区分：

```text
ib_write_bw ... -t <tx-depth>
  → 普通 WRITE，不依赖远端 RQ
  → 适合观察 TX/outstanding window

ib_send_bw ... -t <tx-depth> -r <rx-depth>
  → SEND 同时依赖 TX window 和 RX credits
  → 可以分别扫描 -t 与 -r
```

如果提高 `-t` 后吞吐上升，而 RNR/retry/drop counters 基本不变，说明原先更像
TX/window 限制。如果出现 `RNR_RETRY_EXC_ERR`，才优先检查 RX depth、Receive repost
速度和远端 progress。如果 switch queue/drop/PFC/CNP 同时上升，则已经进入网络拥塞
问题，继续增加 outstanding window 可能适得其反。

### 3.11 “调整 BDP”与“调整窗口以覆盖 BDP”不是一回事

BDP 没有一个可以直接写入网卡的 `BDP=500MB` 参数。它是路径属性：

```text
BDP = end-to-end bottleneck bandwidth × RTT
```

所以真正改变 BDP 只有两类办法：

1. 改变带宽：更换链路、聚合更多端口、修改限速，或者因为拥塞导致可用带宽变化；
2. 改变 RTT：改变物理距离、路由、交换层级、主机路径，或者因为排队导致 RTT 变化。

其中“通过拥塞排队把 RTT 和 BDP 做大”不是优化。更合理的目标通常是：

> 测出业务路径的 bandwidth 与 RTT，计算目标 BDP，然后调整发送窗口，使其能够覆盖
> 这个 BDP。

#### 可调的是哪些“窗口旋钮”

```text
应用并发/批大小
  ↓
Barex ACCL_TX_DEPTH 与 ACCL_SOFT_TX_DEPTH
  ↓
QP max_send_wr
  ↓
设备 max_qp_wr
  ↓
固件 active outstanding WQE window
  ↓
operation-specific credits，例如 RDMA READ credits
  ↓
网络 ECN/PFC/queue/retry
```

有效窗口由其中最小的一层决定。主要限制包括：

- QP 创建时请求的 `max_send_wr`；
- `ibv_query_device()` 报告的 `max_qp_wr`；
- ConnectX 固件的 active outstanding WQE 上限；
- 单 WQE payload 大小以及 scatter/gather 能力；
- RDMA READ/Atomic 的 responder resources 和 outstanding read credits；
- CQ 容量和 completion 消费速度；
- RNIC SRAM/状态、host memory、MR 与 pinned memory；
- PCIe、NUMA、GPU/CPU memory controller 的 DMA 路径；
- 单 QP/多 QP 是否能使用 LAG 的全部物理端口；
- 对端能力、交换机 buffer、ECN/PFC 和业务 incast。

#### `target_p` 的实机结果

2026-07-28 在 `target_p` 上做了只读检查，没有修改配置。数据面拓扑如下：

| 逻辑 RDMA 设备 | Linux bond | 物理端口 | PCI domain | NUMA | 端口/聚合速率 |
| --- | --- | --- | --- | --- | --- |
| `mlx5_bond_0` | `bond0` | `reth0,reth1` | `0000` | 0 | 2×200G / 400G |
| `mlx5_bond_1` | `bond1` | `reth2,reth3` | `0002` | 0 | 2×200G / 400G |
| `mlx5_bond_2` | `bond2` | `reth4,reth5` | `0010` | 1 | 2×200G / 400G |
| `mlx5_bond_3` | `bond3` | `reth6,reth7` | `0012` | 1 | 2×200G / 400G |

四组均为：

- ConnectX-7，firmware `28.46.3048`；
- 802.3ad/LACP，`layer3+4` hash；
- 每张卡 PCIe 5.0（32 GT/s）×16，当前和最大宽度均为 ×16；
- verbs `max_qp_wr=32768`；
- `max_qp_rd_atom=16`。

机器没有直接暴露“机头/机尾”这个软件标签。若现场把 NUMA 0 一侧的 `bond0/1`
称为机头、NUMA 1 一侧的 `bond2/3` 称为机尾，则两侧 RNIC 的 verbs 上限、固件和
链路规格是对称的；差别主要是本地性：

```text
NUMA 0 CPU/memory 上的任务 → 优先 bond0/1
NUMA 1 CPU/memory 上的任务 → 优先 bond2/3
```

如果 CPU thread、host buffer 或与 GPU 相关的 DMA 路径位于另一侧 NUMA，访问可能
跨 NUMA interconnect。网络公式算出的 BDP 没有因此改变，但主机侧可达到的 goodput、
提交/完成延迟和抖动可能变差，最终测得的 bandwidth/RTT 也会改变。因此网卡亲和性
是“能否兑现目标 BDP”的条件，不是给机头、机尾分别写入不同 BDP 值。

#### 400G bond 的 BDP 不能无条件按单 QP 400G 计算

每个 bond 由两个 200G 端口组成。400G 是聚合容量，不等于任意一条 QP/flow 都必然
获得 400G：

- legacy RoCE LAG 可能按 QP affinity 选择端口；
- hash LAG 可以按 packet header/steering 分散流量，但依赖驱动、固件与配置；
- 交换机侧 LACP hash、RoCE flow label/UDP source port entropy 也会影响分布；
- 多个 QP 更容易把流量分散到两条成员链路；
- 是否真正聚合应看 `reth*` 的 per-port counters 和实测吞吐，不能只看 `bond` 显示
  的 400000 Mbit/s。

因此至少要分别计算两种预算：

```text
保守的单成员路径：200 Gbit/s × RTT
确认可聚合后的 bond 总路径：400 Gbit/s × RTT
```

以 RTT 10 ms 为例：

| 目标带宽 | BDP |
| --- | --- |
| 单个 200G 成员 | 250 MB |
| 400G bond 聚合 | 500 MB |

若每个 WQE 只有 4 KiB：

```text
200G 需要约 250 MB / 4 KiB ≈ 61,036 WQE
400G 需要约 500 MB / 4 KiB ≈ 122,071 WQE
```

二者都超过 `target_p` 单 QP 报告的 `max_qp_wr=32768`。这时不能通过把单 QP 的
`tx_depth` 无限增大来解决，而应考虑：

- 合并小请求，增大平均 `bytes/WQE`；
- 使用多个 QP，并验证 LAG 端口分布；
- 让多条 QP 合计 outstanding bytes 覆盖 BDP；
- 核对固件 active WQE window，不能只看 verbs `max_qp_wr`；
- 检查 CQ polling、doorbell、PCIe、NUMA 与 memory bandwidth 是否成为新瓶颈。

例如单 QP 允许 32768 个 WQE 时：

```text
4 KiB/WQE  → 约 128 MiB window
64 KiB/WQE → 约 2 GiB window
```

所以“网卡最多 32768 个 WQE”不等于“只能覆盖固定大小的 BDP”；可覆盖的数据量还
取决于每个 WQE 携带多少数据。

#### 推荐的实测流程

1. 分别绑定 `mlx5_bond_0..3` 测空载 RTT 和带宽；
2. 固定 message size，扫描 `tx_depth`，找到吞吐不再上升的平台点；
3. 固定 `tx_depth`，扫描 message size/post list，观察聚合收益；
4. 同时读取 `reth0..7` counters，确认两个 LAG member 是否都承载流量；
5. 把进程、host memory 和 GPU workload 绑定到对应 NUMA 一侧后复测；
6. 分别做单 QP、多 QP，以及 200G 保守预算、400G 聚合预算；
7. 监控 retry、CNP/ECN、PFC、drop、CQ backlog 和 tail latency；
8. 选择刚好能稳定覆盖目标 BDP 的窗口，并预留适度余量，而不是直接使用最大值。

## 4. PFC：逐优先级暂停

Priority Flow Control（802.1Qbb）按 priority 暂停上游发送：

```text
queue 达 XOFF
  → 发 PFC pause
  → 上游停止该 priority
queue 降到 XON
  → 恢复
```

优点：在 buffer 快耗尽时避免 drop。

代价：

- Head-of-Line blocking；
- pause propagation；
- incast 时多端同时被停；
- 错误配置可造成 PFC storm/deadlock；
- 长距离下需要按 cable/RTT 预留更多 headroom。

PFC 是 hop-by-hop，不是端到端 transport ACK。

## 5. ECN 与 DCQCN

ECN 在真正丢包前标记拥塞：

```text
Switch queue 超过 ECN threshold
  → 标记 CE
  → receiver 生成 CNP
  → sender RNIC 降低 rate
  → 无拥塞后逐步恢复
```

DCQCN 将 ECN/CNP 反馈与发送端 rate control 结合。

分工：

- ECN/DCQCN：尽早减速，控制持续拥塞。
- PFC：最后的无损保护，吸收短时 microburst。
- RC retry：链路/网络仍丢包时的可靠性兜底。

理想状态不是“大量 PFC pause 但 0 drop”，而是 ECN 把队列控制住，PFC 只偶尔触发。

## 6. Lossy RoCE 不等于不能用

现代 NIC/switch 可以在：

- ECN only；
- semi-lossless；
- PFC + ECN；
- 支持 OOO/选择性重传的 lossy fabric

等模式运行。可行性取决于硬件、固件和端到端配置。不能把“RoCE 必须绝对无损”写成永久定律；但对不支持现代恢复能力的 RC 路径，drop 代价仍可能很高。

## 7. Incast

多个发送端同时写一个接收端：

```text
P0 ─┐
P1 ─┼──► D0/NIC/PCIe/GPU
P2 ─┤
P3 ─┘
```

瓶颈可能在：

- ToR egress；
- receiver NIC；
- PCIe uplink；
- CQ/poll CPU；
- GPU memory write path。

Barex 提供 `ACCL_INCAST_AVOID/COUNT/THRESHOLD`，大消息 metadata 到达后可限制同时进入 phase 2/3 的数量。见：

```text
xcontext_impl.cc:1165-1168
xcontext_impl.cc:1237-1249
```

它解决的是 endpoint/application admission，不替代 fabric ECN/PFC。

## 8. MTU

常见 RoCE MTU 需同时考虑：

- Ethernet interface MTU；
- RDMA port active MTU；
- 路径中所有 switch；
- VLAN/tunnel overhead。

MTU 不一致可能表现为：

- 小消息可用，大消息失败；
- retry/timeout；
- fragment/路由异常；
- 性能明显低于预期。

Barex `ACCL_IBV_MTU` 最终用于 QP path MTU；不能只改应用变量而忽略 fabric。

## 9. Traffic Class、PCP、DSCP

RoCE QoS 映射链：

```text
application traffic class
  → IP DSCP/ECN
  → switch priority/traffic class
  → ECN threshold + PFC priority + ETS bandwidth
```

每跳 trust mode 必须一致。常见事故：

- host 标 DSCP，但 switch trust PCP；
- RoCE data 与 CNP 进了同一拥塞队列；
- 一侧开启 PFC priority 3，另一侧映射到 priority 4；
- Barex 与 NCCL 使用不同 traffic class，争抢或落入 lossy queue。

## 10. Barex 参数如何对应网络

| 参数 | 作用 | 误配表现 |
|---|---|---|
| `ACCL_IBV_MTU` | QP path MTU | 建联/大包失败、低吞吐 |
| `ACCL_BAREX_TRAFFIC_CLASS` | RoCE traffic class | QoS/PFC/ECN 不匹配 |
| `ACCL_RETRANSMIT_TIMEOUT` | RC ACK timeout | 过小误重传，过大故障恢复慢 |
| `ACCL_RETRY_CNT` | transport retry | 过小易失败，过大卡很久 |
| `ACCL_TX_DEPTH` | Barex permit 与 QP `max_send_wr` | 过小无法覆盖 BDP，过大增加资源/拥塞压力 |
| `ACCL_SOFT_TX_DEPTH` | TX permit 耗尽后的软件等待队列 | 过小 queue-full，过大掩盖拥塞并增加尾延迟 |
| `ACCL_RX_DEPTH` | QP `max_recv_wr` / receive credit | 过小导致 SEND/WRITE_WITH_IMM RNR；对普通 WRITE BDP 无帮助 |
| `ACCL_RNR_RETRY` | RNR retry | recv starvation 时行为 |
| `ACCL_HEARTBEAT_INTERVAL` | channel liveness | 故障发现速度/额外流量 |
| `ACCL_INCAST_*` | endpoint admission | 接收端爆发拥塞 |

`LOG_MAX_OUTSTANDING_WQE` 不属于 Barex 参数；它是特定 ConnectX
device/firmware configuration。修改这类 NV configuration 之前必须先确认设备型号、
firmware 说明、生效方式、回滚方法和端到端压测结果。

## 11. NCCL 与 Barex 共网

若 NCCL collective 与 blade-kvt 同时使用 RoCE：

```text
NCCL NET flows + Barex KV Write flows
  → 同一 NIC/QP set
  → 同一 priority/ECN/PFC pool
  → 同一 fabric bottleneck
```

需要确认：

- HCA 与 port 选择；
- DSCP/traffic class；
- NCCL channel/Barex parallel channel 数；
- 是否同时 incast；
- PFC pause 与 ECN/CNP counters；
- NIC port 与 PCIe utilization。

## 12. 交换机/RNIC 指标

至少采集：

### Switch

- ECN marked packets；
- PFC pause tx/rx 与 duration；
- queue current/max occupancy；
- ingress/egress discard；
- per-priority bytes；
- CNP traffic。

### RNIC

- retry exceeded；
- RNR NAK/retry；
- packet sequence error；
- retransmitted packets；
- CNP sent/received；
- ECN marked receive；
- out-of-order/reorder counters；
- port xmit wait。

不同 vendor counter 名称不同，先保存 `ethtool -S`、`perfquery`、`rdma statistic` 全量快照，再做 delta。

## 13. 对现有 `rdma_learning_2.md` 的校正

现有笔记中的“经典 RoCE go-back-N”适合作为丢包放大的直觉，但需补充：

1. 具体 retry/乱序恢复能力依 RNIC 与模式；
2. 新设备可能支持 OOO 或选择性重传；
3. TCP 也不是永远只重传一个包，行为依 SACK/RACK、拥塞窗口等；
4. PFC 不是跨机房端到端传播的控制协议，而是逐跳配置；
5. 跨机房风险来自 BDP、丢包、路由与能力不一致的组合。

## 14. 自检

1. PFC、ECN/DCQCN、RC retry 各在哪一层解决什么问题？
2. 为什么 400G×10ms 需要关注约 500MB BDP？
3. `RNR_RETRY_EXC_ERR` 为什么通常不是交换机丢包？
4. Barex incast avoidance 与 DCQCN 为什么不能互相替代？

## 参考

- [NVIDIA Cumulus Linux 4.4：RDMA over Converged Ethernet (RoCE)](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-44/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/)
- [NVIDIA Cumulus：PFC buffer、XOFF/XON 与传播期间继续到达的数据](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-43/Layer-1-and-Switch-Ports/Buffer-and-Queue-Management/)
- [NVIDIA Cumulus：Bonding、LAG 与 hash](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-43/Layer-2/Bonding-Link-Aggregation/)
- [NVIDIA MLNX_OFED 24.10 文档（PDF，RoCE 章节）](https://docs.nvidia.com/networking/display/nvidia-mlnx-ofed-documentation-v24-10-0-7-0-0-november-2024-ga-release.0%20%28November%202024%20GA%20Release%29.pdf)
- [`ibv_create_qp(3)`：`max_send_wr` 与 `max_recv_wr`](https://man7.org/linux/man-pages/man3/ibv_create_qp.3.html)
- [`ibv_query_device(3)`：device `max_qp_wr`](https://man7.org/linux/man-pages/man3/ibv_query_device.3.html)
- [linux-rdma perftest：`--tx-depth`、`--rx-depth` 与 post list](https://github.com/linux-rdma/perftest)
- [Linux mlx5 NV parameter source：`log_max_outstanding_wqe`](https://github.com/torvalds/linux/blob/master/drivers/net/ethernet/mellanox/mlx5/core/lib/nv_param.c)
