---
title: "02c. RoCE、拥塞控制、PFC/ECN 与重传"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 02c. RoCE、拥塞控制、PFC/ECN 与重传

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
| `ACCL_RNR_RETRY` | RNR retry | recv starvation 时行为 |
| `ACCL_HEARTBEAT_INTERVAL` | channel liveness | 故障发现速度/额外流量 |
| `ACCL_INCAST_*` | endpoint admission | 接收端爆发拥塞 |

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

- [NVIDIA Cumulus Linux RoCE](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/)
- [NVIDIA RoCE documentation](https://docs.nvidia.com/networking/display/mlnxofedv24070610/rdma%2Bover%2Bconverged%2Bethernet%2B%28roce%29)
