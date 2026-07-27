---
title: "05. NCCL 算法、协议与性能模型"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 05. NCCL 算法、协议与性能模型

## 1. Algorithm 与 Protocol 不是一回事

- Algorithm：rank 之间如何组织通信步骤，如 Ring、Tree。
- Protocol：每条连接上如何用 buffer、flag 和 step 传输 chunk，如 Simple、LL、LL128。
- Transport：连接落到 P2P、SHM 或 NET。

一个 AllReduce 可以是：

```text
Ring algorithm + Simple protocol + NET transport
```

也可以是：

```text
Tree algorithm + LL protocol + P2P transport
```

## 2. Ring AllReduce

Ring AllReduce 通常分两段：

1. ReduceScatter：每个 rank 最终持有一段已归约数据；
2. AllGather：交换各段，所有 rank 得到完整结果。

对于 `N` 个 rank，每个 rank 发送/接收的数据量近似：

```text
2 × (N - 1) / N × message_size
```

优点：

- 大消息带宽利用率高；
- 每一步只有固定邻居；
- 容易形成流水线。

缺点：

- 步数约 `2(N-1)`；
- 小消息延迟随 rank 数增长明显；
- 任一弱链路进入 ring 都会限制整体。

### 2.1 四个 rank 的手算例子

每个 rank 有 4 个元素：

```text
rank 0: [ 1,  2,  3,  4]
rank 1: [10, 20, 30, 40]
rank 2: [ 5,  6,  7,  8]
rank 3: [ 2,  4,  6,  8]
```

最终 AllReduce(sum)：

```text
[18, 32, 46, 60]
```

为了理解 ring，把向量切成 4 个 chunk，每个 chunk 先沿 ring 走
`N-1=3` 步完成 reduce-scatter。结束时每个 rank 只拥有一个“已经求和”的 chunk：

```text
rank 0 拥有 [18]
rank 1 拥有 [32]
rank 2 拥有 [46]
rank 3 拥有 [60]
```

再走 3 步 all-gather，大家交换这四个结果，最终都得到完整向量。

如果完整消息是 1 GiB、`N=4`，每 rank 近似通信量：

```text
2 × (4-1)/4 × 1 GiB = 1.5 GiB
```

不是 3 GiB，因为每步只发送约 `1/N` 的 chunk。

## 3. Tree AllReduce

Tree 沿树向上 reduce，再向下 broadcast。深度约 `O(log N)`。

优点：

- 小/中消息延迟较低；
- 步数随 rank 数增长慢。

代价：

- 根和中间节点链路压力不同；
- 大消息带宽未必像 ring 一样均衡；
- 拓扑映射不佳时会重复经过共享路径。

## 4. CollNet、NVLS 与 PAT

- CollNet：利用支持 collective offload 的网络能力。
- NVLS/NVLSTree：利用 NVLink SHARP/NVSwitch 归约能力。
- PAT：Parallel Aggregated Trees，针对特定 collective/scaling 场景。

它们是否可用取决于硬件、plugin、NCCL 版本和 collective 类型，不应通过名称判断一定更快。

## 5. Simple、LL、LL128

### 5.1 Simple

面向吞吐。payload 占比高，适合大消息；通常需要更大的 step/chunk，启动固定开销相对高。

### 5.2 LL

低延迟协议，把 data 与 flag 组织成更容易细粒度推进的形式。小消息延迟好，但有效 payload 比例低，NCCL 源码中的流量模型会对 LL 计入额外 traffic。

### 5.3 LL128

在支持的平台上平衡延迟与带宽，但有更严格的硬件和路径约束。官方文档明确不建议随意强制启用不受支持的 LL128，否则可能出现数据错误。

当前官方允许用 `NCCL_PROTO=LL,LL128,Simple` 约束协议集合，但推荐让 NCCL 自动选择，除非在定位问题。

## 6. Chunk、Slice、Step

消息不是一次性放进连接：

```text
message
  → 分配给多个 channel
  → 每 channel 拆成 chunk
  → chunk 按 algorithm step 前进
  → protocol 再按 slice/step buffer 协调 producer/consumer
```

关键效果：

- pipeline 隐藏网络 RTT；
- 多 channel 并行占满多 rail；
- 固定 buffer 可以循环使用；
- head/tail 防止 producer 覆盖 consumer 尚未处理的数据。

## 7. 性能模型

可用简化的 α-β 模型：

```text
T ≈ steps × α + traffic_bytes × β
```

- `α`：每步固定延迟，包括 kernel、proxy、doorbell、网络 RTT。
- `β`：每字节成本，即有效带宽倒数。

例：假设一次 step 固定开销 `α=5 μs`，有效带宽 `B=25 GB/s`，传
1 MiB 的数据成本约：

```text
1 MiB / 25 GB/s ≈ 41.9 μs
```

若算法需要 6 步，极简估算：

```text
T ≈ 6×5 μs + 41.9 μs = 71.9 μs
```

真实算法会 pipeline，多步数据传输可以重叠，因此这个加法不是性能预测器；
它用于判断固定延迟和字节传输谁更可能占主导。

Ring 与 Tree 的取舍：

| 场景 | 主要矛盾 | 常见更优方向 |
|---|---|---|
| 小消息 | `steps × α` | Tree、LL |
| 大消息 | `traffic × β` | Ring、Simple、多 channel |
| 多节点 | 网络/rail | topology-aware graph、GDR |
| 节点内 NVSwitch | collective offload | NVLS |

## 8. Bus bandwidth 与 algorithm bandwidth

`nccl-tests` 常同时报告：

- algbw：应用 payload / 时间；
- busbw：按 collective 理论链路流量归一化。

AllReduce ring 常用换算：

```text
busbw = algbw × 2 × (N - 1) / N
```

不能直接拿 algbw 与 PCIe 单链路理论值比较，因为：

- collective 同时使用多条链路；
- 同一数据经过多步；
- 可能使用 NVLink/NIC 多 rail；
- 单/双方向统计口径不同。

## 9. 分层定位性能

### 9.1 先确认硬件上限

- PCIe negotiated width/speed；
- NVLink 状态；
- NIC link speed；
- GPU/NIC NUMA 与 topology。

### 9.2 再确认 transport

- P2P 是否启用；
- IB 还是 socket；
- GDR 是否启用；
- multi-rail 是否实际使用。

### 9.3 再确认 tuning

- algorithm/protocol；
- channel 数；
- chunk size；
- message size 分布；
- 是否被计算 stream 或其他 collective 串行化。

### 9.4 最后才调环境变量

强制变量适合 A/B 验证：

```bash
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

每次只改变一个维度，并保存 NCCL INFO 日志。

## 10. `nccl-tests` 基础实验

```bash
./all_reduce_perf -b 8 -e 1G -f 2 -g 8
./all_gather_perf -b 8 -e 1G -f 2 -g 8
./sendrecv_perf -b 8 -e 1G -f 2 -g 2
```

建议记录：

- size；
- out-of-place/in-place；
- time/algbw/busbw；
- error count；
- NCCL topology/NET 日志；
- GPU/NIC 时钟和链路状态。

## 11. 与 blade-kvt 的性能差异

NCCL 针对规则 collective：

- rank graph 稳定；
- payload 易均匀切分；
- GPU kernel 可持续流水。

blade-kvt KV 发送则是：

- request/worker 目标动态；
- block 离散；
- P/D TP 可能不同；
- layer ready 与模型 forward 重叠；
- 需要远端地址协议和 request 完成通知。

所以 blade-kvt 使用 Barex one-sided write 是符合负载形态的选择，并不是“NCCL 性能不够”这一单一原因。

## 12. 自检

1. Ring 为什么大消息强，小消息可能弱？
2. LL 为什么低延迟但有效带宽不一定高？
3. 为什么 busbw 可能高于单个 PCIe link 的理论带宽？
4. 强制 `NCCL_PROTO=LL128` 有什么风险？

## 参考

- [NCCL Environment Variables：`NCCL_ALGO` 与 `NCCL_PROTO`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)
