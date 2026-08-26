---
title: "06a. 从《RDMA 高性能点对点通信库的艺术》看 Barex 的工程设计"
date: 2026-08-26
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 06a. 从《RDMA 高性能点对点通信库的艺术》看 Barex 的工程设计

> 本章整理自 ATA 文章《RDMA高性能点对点通信库的艺术》（2022-01-04 发表，
> 2025-12-29 更新），并与本仓库固定版本的 Barex、blade-kvt 源码笔记互相校验。
> 文章描述的是 ACCL-Barex 的整体设计思想和实践结果；具体类名、环境变量和当前
> 实现仍以 [06 Barex 架构](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-06-barex-architecture/) 与源码为准。

## 1. 文章补上了什么视角

[06 Barex 架构](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-06-barex-architecture/) 从固定版本源码解释对象、线程、WR、
CQ 和 callback。本章换一个角度，回答 Barex 为什么要这样设计：

```text
用户目标：高效、易用、稳定
  │
  ├─ 易用：把 verbs 收敛为 Read / Write / Send 与批量接口
  ├─ 稳定：同步错误立即返回，异步错误经 DoneCallback 归还
  ├─ 稳定：QP 出错后 flush 并排空所有 inflight WR，再释放资源
  ├─ 高效：多 NIC、发送 credit、incast admission、MR pool、SGE/Batch
  └─ 高效：端点软件与 RoCE/PFC/ECN/DCQCN 参数协同
```

文章把集合通信与点对点通信明确分开：

- ACCL 整体可以提供集合通信和点对点通信；
- ACCL-Barex 聚焦 `Send/Recv`、`Read/Write` 等 point-to-point 能力；
- Barex 不是 NCCL collective graph，也不是 NCCL Net Plugin。

![RDMA 绕过内核的数据路径示意](/imgs/ata_barex_01_rdma_kernel_bypass.avif)

## 2. 架构：从 verbs 语义到业务接口

文章中的 Barex 架构以 IB/RoCE 为主，同时支持 Mellanox、EIC、eRDMA 等不同网卡
形态和 TCP 通路。它在 verbs 之上提供：

| 类别 | 接口/语义 | 使用者需要提供 |
|---|---|---|
| 双边消息 | `Send`，远端由库接收并回调 | 本地 buffer、长度、peer、callback |
| 单边访问 | `Read` / `Write` | 本地 MR、远端地址与 rkey |
| 批量/离散内存 | `WriteBatch`、`WriteBySglist`、`ReadBatch`、`ReadBySglist` | 多段描述与完成回调 |
| 连接与运行时 | channel、listener、CQ progress、timer、statistics | 配置、生命周期与业务协议 |

![ACCL-Barex 总体架构](/imgs/ata_barex_02_architecture.avif)

### 2.1 为什么最终暴露 `Read`、`Write`、`Send`

原生 verbs 同时提供 SEND、RECV、READ、WRITE、ATOMIC 等语义。文章把常用业务模式
压缩为三种：

```text
单向读  → Read()
单向写  → Write()
双向消息 → Send()；另一侧的 Recv 由通信库隐式管理
```

![单向读、单向写与双向消息语义](/imgs/ata_barex_03_api_semantics.avif)

这个抽象的价值不是减少几个函数名，而是隐藏：

- PD/MR/QP/CQ 的创建与销毁；
- Receive Queue 的预投递和补充；
- `wr_id` 到请求上下文的映射；
- CQ polling、错误传播和 callback dispatch；
- 大消息的 buffer 协商；
- depth、backpressure 和超时。

因此 Barex API 返回成功只表示“同步校验与提交路径成功”，不能据此释放 payload。
真正的异步完成边界仍是对应的 DoneCallback。

## 3. 稳定性：错误必须回到请求，资源必须晚于 WR 释放

文章把错误分成两类：

| 错误类型 | 发生位置 | 传播方式 |
|---|---|---|
| 同步错误 | API 调用、参数校验、`ibv_post_*` 提交 | API 返回码 |
| 异步错误 | RNIC 执行、网络、远端访问、CQ completion | DoneCallback |

![Barex 的同步与异步错误模型](/imgs/ata_barex_04_error_model.avif)

### 3.1 `wr_id` 是异步世界中的关联键

经典实现为每次 `ibv_post_send()` / `ibv_post_recv()` 创建 `wr_context`：

```text
post_send/post_recv
  → wr_context 保存 channel、buffer、callback、opcode 等
  → wr_context 指针写入 WR.wr_id
  → Channel 同时跟踪所有 inflight wr_context

poll_cq
  → WC.wr_id 找回 wr_context
  → status == SUCCESS：正常完成
  → status != SUCCESS：失败回调并进入连接错误清理
```

这与固定版本源码里的 `x_wr_id` 和
`XContextImpl::ProcessOneIoEvent()` 是同一类设计。

### 3.2 QP 出错后不能立刻 delete

一个 WR 失败时，正确顺序是：

1. 找到对应 `Channel/QP`，把 QP 置为 error state；
2. RNIC 将该 QP 上剩余 inflight WR flush 到 CQ；
3. 继续 poll CQ，对每个 WR 执行失败回调和资源回收；
4. 当 Channel 跟踪的 WR 列表为空后，才删除 QP 和 Channel。

![Server 侧关闭连接的处理顺序](/imgs/ata_barex_05_server_close_sequence.avif)

![连接清理状态机](/imgs/ata_barex_06_channel_state_machine.avif)

诊断时的直接结论是：

> 一串 `WR_FLUSH_ERR` 往往只是后果，应该定位同一 QP 最早出现的非 flush WC error。

如果先销毁 channel/MR，再等待 CQ 排空，就可能产生 use-after-free、重复 callback、
payload 泄露或 callback 永远不返回。

## 4. 多 NIC：吞吐最终受整条 I/O 路径约束

Barex 把 NIC 选择权交给调用方，`Send/Read/Write` 可以指定从哪张网卡收发。多 NIC
能提高吞吐，但提升不会无限线性：

```text
GPU/HBM
  ↕
PCIe switch / CPU root complex
  ↕
多张 RNIC
  ↕
网络
```

![多 NIC 吞吐测试](/imgs/ata_barex_07_multi_nic_throughput.avif)

文章的 8 NIC 测试没有获得 8 倍吞吐，主要受主机 PCIe Gen3 带宽限制；升级
Gen4 才可能继续改善。定位“多卡没有线性加速”时，应同时核对：

- GPU、NIC 是否挂在同一 NUMA/root complex；
- PCIe generation、lane width 与 switch uplink；
- 单 NIC 线速、总 PCIe 可用带宽和内存带宽；
- flow 是否均匀映射到多 NIC；
- CQ progress CPU 是否成为新瓶颈。

## 5. Out of Buffer：用 credit 限制 outstanding 消息

### 5.1 接收侧 OOB 与 RNR

SEND/WRITE_WITH_IMM 等需要接收侧提前 post Receive WR。若发送方 outstanding 消息数
超过接收队列可用 entry，接收 RNIC 会返回 RNR，发送端进入 RNR retry。中短消息中，
重试成本可能高于有效传输本身。

文章给出的核心办法是 semaphore/credit：

```text
初始化 permits ≈ 对端可用 recv depth

提交消息     → permits -= 1
本地完成/协议完成 → permits += 1
permits == 0 → 阻塞或进入有界软件队列
```

同时，发送端自己的 SQ 也有 `max_send_wr`，所以 credit 还要受到本地 TX depth 限制。
固定版本源码中的 `send_semaphore_`、`PostSendOrEnqueue()` 与 `ReleaseAndPostSend()`
正是这类机制。

### 5.2 文章中的 OOB 测试

优化前：`data_size=1280`、`tx_depth=2000`、`iter=2000`、`epoch=1000`。

| `rx_depth` | server OOB | 吞吐 | 总完成时间 |
|---:|---:|---:|---:|
| 8 | 600–1000 | 97 Mbps | 249 s |
| 20 | 800–900 | 600 Mbps | 39.3 s |
| 30 | 400–600 | 1.3 Gbps | 15.9 s |

加入发送 semaphore 后：`rx_depth=30`。

| `post_send_sem` | server OOB | 吞吐 | 总完成时间 |
|---:|---:|---:|---:|
| 30 | 300–400 | 1.4 Gbps | 15.7 s |
| 10 | 0–100 | 1.55 Gbps | 14.1 s |
| 5 | 0–10 | 1.65 Gbps | 13.6 s |

测试说明“最大并发”不等于“最大吞吐”。给 RQ 留出余量，减少 RNR/retry，反而能同时
降低完成时间并提高有效带宽。具体数值依赖报文、QP、网卡和测试环境，不应直接作为
生产默认值。

## 6. Incast：端点 admission 与 fabric 拥塞控制是两层问题

多个发送端同时向一个接收端发送时，会在接收侧接入交换机 egress、NIC、PCIe 或
GPU 写入路径形成拥塞。活跃连接越多，瞬时排队和 ECN/CNP 降速通常越明显。

![多对一 Incast 流量模式](/imgs/ata_barex_08_incast_pattern.avif)

文章给出一种软件侧 admission 协议：

```text
1. Sender → Receiver：meta
2. Receiver：准备 buffer
3. Receiver → Sender：并发配额允许时发送 ready
4. Sender：发送消息前一部分
5. Sender：发送消息后一部分
```

![基于 meta/ready 的 Incast 流控](/imgs/ata_barex_09_incast_flow_control.avif)

本质上，Receiver 用 `ready` 控制同时活跃的数据流数量。这解决端点入口的并发，
但不替代交换机和 RNIC 的 ECN/PFC/DCQCN。

文章在 `2 × 25G` 网络中的结果：

| 报文大小 | 优化前 Incast 吞吐 | 优化后 Incast 吞吐 |
|---:|---:|---:|
| 512 KB | 18.0–21.4 Gbps | 20.1–29.0 Gbps |
| 1 MB | 18.9–26.7 Gbps | 29.6–39.7 Gbps |
| 2 MB | 20.0–32.0 Gbps | 37.1–39.8 Gbps |
| 4 MB | 15.8–33.7 Gbps | 38.0–41.2 Gbps |
| 16 MB | 24.0–40.0 Gbps | 41.7–42.5 Gbps |

这些是文章报告的特定环境结果；可用来说明机制，不应用来承诺其他集群的绝对带宽。

## 7. RDMA 友好内存池：把 MR 注册移出热路径

RDMA 使用的内存需要 pin 并注册成 MR。文章指出，注册耗时随内存大小增长，MB 级
内存注册可能达到毫秒级。如果每次发送都临时注册/注销，MR 会成为主要延迟。

![MR 注册耗时随内存大小增长](/imgs/ata_barex_10_mr_registration_cost.avif)

Barex 内存池采用：

```text
向 OS/CUDA 申请大块内存时注册 MR
  → pool 长期持有已注册内存
  → 后续分配只切分/复用 pool block
  → 低水位或定时调用 Shrink，注销一部分 MR 并归还系统
```

文章报告这种复用使内存申请提升超过百倍，热路径分配维持在微秒级。工程上要一起
权衡：

- pool 过小：频繁慢路径注册；
- pool 过大：长期 pinned，挤压系统/GPU 可用内存；
- 多 NIC/多 PD：同一地址可能需要多份 MR；
- 进程或远端重启：旧 rkey 不能继续使用；
- 销毁：必须晚于所有引用该 MR 的 WR completion。

## 8. 离散小报文与事件线程

大批量离散小报文会放大 per-WR、doorbell、CQE 和 callback 成本。文章提到 Barex
针对 `WriteBatch` 做了 minibatch、`send_flag`、SGE 等优化，以降低 GDR 小报文
时延。

理解时要区分：

- 一个 WR 内多个 SGE：一个操作访问多段本地内存；
- 多个 WR 链式 post：一次 doorbell 提交多个操作；
- selective signaling：减少 CQE 数量，但仍要正确归还每个业务请求；
- batch callback：降低回调开销，但错误必须能映射到受影响的 item。

文章还强调大/小消息分路径以及事件线程非阻塞。原则是 CQ progress 线程只做必要的
completion 推进；重 CPU 回调、拷贝或业务逻辑应转移到 thread pool，避免一个慢
callback 阻塞所有连接。

## 9. 软硬件协同：Barex admission 之外还要调 DCQCN

RoCE 无损网络常组合：

```text
PFC       → 队列接近耗尽时逐优先级暂停，避免丢包
ECN       → 交换机提前标记拥塞
CNP       → Receiver 把拥塞反馈给 Sender RNIC
DCQCN     → Sender RNIC 降速并在无拥塞后恢复
Barex     → 控制 endpoint outstanding、active flows 与消息调度
```

![DCQCN 升速参数调优](/imgs/ata_barex_11_dcqcn_tuning.avif)

文章把 DCQCN 升速分为 Fast Recovery、Active Increase 和 Hyper-active Increase，
并说明当时 Mellanox 实现忽略 HAI，主要调了：

- 无 CNP 后更新 new rate 的周期；
- Fast Recovery 的 target rate；
- Active Increase 的升速步长。

这说明性能优化必须跨层验证：只扩大应用队列可能加剧拥塞，只改 PFC 可能扩大
Head-of-Line blocking，只调 RNIC rate 又可能掩盖 endpoint admission 问题。

## 10. 应用：PD 分离中的 KV Cache 传输

文章把 Barex 的典型应用落到大模型推理的 Prefill/Decode 分离：

```text
Prefill 计算第 0 层 KV ─┐
Prefill 计算第 1 层 KV ─┼─ layer-by-layer Barex transfer → Decode
Prefill 计算第 N 层 KV ─┘
             计算与网络通信 overlap
```

![Barex 用于 Prefill/Decode 分离的 KV Cache 传输](/imgs/ata_barex_12_pd_disaggregation.avif)

文章以 Qwen1.5-72B 为例，报告在其业务和测试环境中：

- 实例数下降 24%；
- 平均推理时延下降 48%；
- P99 时延下降 78%。

这些收益来自完整的 PD 分离架构、调度、异构资源和通信计算重叠，不能全部归因于
Barex。对 blade-kvt，更可验证的对应关系是：

```text
GPU KV MR
  → RPC 取得远端 raddr/rkey
  → XChannel::WriteBatch
  → Barex CQ callback
  → promise/future
  → flush() 建立业务可见完成边界
```

详见 [08 blade-kvt 发送路径](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-08-blade-kvt-barex-send-path/)。

## 11. 文章概念与当前笔记/源码的映射

| 文章中的设计 | 本仓库中的落点 |
|---|---|
| Read/Write/Send 抽象 | [06 第 7–9 节](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-06-barex-architecture/) |
| DoneCallback 传递异步错误 | [06 第 11–12 节](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-06-barex-architecture/) |
| `wr_id` 关联请求上下文 | `x_wr_id`、`ProcessOneIoEvent()` |
| QP error → flush → drain → delete | [09 CQ 异步错误](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-09-debugging-and-performance-playbook/) |
| 发送 semaphore/depth | `send_semaphore_`、`PostSendOrEnqueue()` |
| Incast active-flow admission | `ACCL_INCAST_AVOID/COUNT/THRESHOLD` |
| MR pool 与 Shrink | `XSimpleMempool`、`RegUserMr/Reserve/Release` |
| 多 NIC 选择 | `ACCL_USE_NICS`、channel/device 绑定 |
| 离散 KV block 批量写 | blade-kvt `RDMAChannel::send_data()` → `WriteBatch` |
| PD layer-by-layer overlap | [08 blade-kvt 发送路径](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-08-blade-kvt-barex-send-path/) |

## 12. 调优时的四个边界

1. **API 返回不等于完成。**payload 生命周期必须覆盖 callback。
2. **TX depth 不等于 RQ depth。**前者约束本地 outstanding，后者决定接收 credit；
   两者失配可能触发 queue full 或 RNR。
3. **Endpoint admission 不等于 fabric 拥塞控制。**Barex 限并发和 DCQCN/PFC/ECN
   需要一起观察。
4. **多 NIC 总带宽不等于应用吞吐。**PCIe、NUMA、内存、CQ CPU 和 GPU copy 都可能
   更早饱和。

## 13. 自检

1. 为什么 `Send()` 可以隐藏远端 `Recv()`，而原生 verbs 程序不能省略 Receive WR？
2. 一个 WC 失败后，为什么不能立刻释放 Channel 和 MR？
3. `send_semaphore_` 应只看本地 `max_send_wr`，还是也要考虑远端 receive credit？
4. Incast admission、ECN/DCQCN 和 PFC 分别在什么层工作？
5. 为什么 8 张 NIC 可能被一条 PCIe uplink 限制？
6. MR pool 为什么能降时延，又为什么不能无限增大？
7. PD 分离的性能收益为什么不能全部归因于 Barex？

## 参考

- ATA：[《RDMA高性能点对点通信库的艺术》](https://ata.atatech.org/articles/11000224222)
- [06 Barex 1.5.3-1 架构与发送实现](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-06-barex-architecture/)
- [02c RoCE、拥塞控制、PFC/ECN 与重传](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-02c-roce-congestion-and-tuning/)
- [08 blade-kvt 的 Barex 发送路径](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-08-blade-kvt-barex-send-path/)
- [09 调试与性能分析手册](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-09-debugging-and-performance-playbook/)

> 本章 12 张 `ata_barex_*` 图片均来自上述 ATA 文章，仅用于内部学习笔记。
