---
title: "03. NCCL 架构：从 API 到 GPU Kernel 与 Proxy"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 03. NCCL 架构：从 API 到 GPU Kernel 与 Proxy

## 1. NCCL 解决什么问题

NCCL 提供 GPU collective 与 point-to-point primitive：

- AllReduce、AllGather、ReduceScatter、Broadcast、Reduce；
- AllToAll；
- `ncclSend` / `ncclRecv`。

它不是通用 RPC 框架，也不负责应用层 request/response、远端 KV block 地址协议或对象存储。NCCL 的核心价值是：**根据通信参与者和硬件拓扑，为 GPU 通信选择算法、传输与协议，并把操作排入 CUDA stream。**

## 2. 五个关键对象

| 对象 | 作用 |
|---|---|
| communicator | 一组 rank、拓扑、channel、连接和调优模型 |
| rank | communicator 内的逻辑参与者，通常一 rank 对应一 GPU |
| channel | 并行通信流水线；不是物理 NIC channel |
| connector/transport | rank 邻接关系上的具体传输资源 |
| kernel plan | 一批将被 launch 的 device work 与 proxy work |

![NCCL 从 API 到数据面的架构](/imgs/nccl_architecture.svg)

## 3. 初始化路径

典型 API：

```cpp
ncclGetUniqueId(&id);
ncclCommInitRank(&comm, nranks, id, rank);
```

当前参考源码主线：

```text
ncclCommInitRank
  → ncclCommInitRankDev
  → ncclCommInitRankFunc
  → initTransportsRank
      ├─ bootstrap 交换 peer 信息
      ├─ 构建 topology graph
      ├─ ncclTopoComputePaths
      ├─ 搜索 Ring/Tree/NVLS/CollNet graph
      ├─ 建立 channel 与 peer connector
      └─ 创建 proxy 资源
```

参考：

- [`src/init.cc:965` `initTransportsRank`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/init.cc#L965)
- [`src/init.cc:1831` `ncclCommInitRankFunc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/init.cc#L1831)

### 3.1 bootstrap 不等于数据面

初始化时 rank 需要交换 host hash、bus ID、网络地址、graph 信息等。bootstrap socket 用来发现和交换元数据，不代表 collective payload 最终一定走 socket。

这与 blade-kvt 很相似：先通过 Barex `Send` 交换 MR handle，再走 RDMA Write；但二者协议和调度完全独立。

## 4. 调用一个 collective 后发生什么

以 `ncclAllReduce` 为例，API 并不是同步执行完数据交换再返回。大致路径：

```text
Host API
  → 参数检查、构造 task
  → communicator planner
  → 选择 algorithm/protocol/channel/chunk
  → 生成 kernel plan
  → 上传 device work
  → 在用户 CUDA stream 上 launch NCCL kernel
  → 如需 NET/host 辅助，同时激活 proxy op
```

完成语义由 CUDA stream 表达：

```cpp
ncclAllReduce(send, recv, count, dtype, ncclSum, comm, stream);
cudaEventRecord(done, stream);
```

只有 event 完成，才能认为该 stream 之前的 NCCL 工作完成。API 返回通常只表示 enqueue 成功。

这与 Barex 的异步 API 很像，但 Barex 用 callback/future 表达完成，NCCL 首先用 CUDA stream/event 表达完成。

## 5. GPU Kernel 为什么参与通信

NCCL 不只是 host 上调用 verbs。GPU kernel 负责：

- 从源 buffer 读数据；
- 执行 reduce/copy；
- 通过连接 buffer、FIFO/head/tail 与 peer 或 proxy 协作；
- 在 ring/tree 的每一步推进 chunk；
- 将结果写到目标 buffer。

同机 P2P/NVLink 路径可能主要由 GPU load/store/copy 完成；跨机时 GPU kernel 与 host proxy 通过共享的 step buffer/FIFO 协调。

## 6. Proxy 的职责

Proxy 是 host 侧进度引擎，常用于 GPU 无法独立推进的传输：

- NET plugin 的 `isend/irecv/test`；
- 某些跨进程 IPC 或 Copy Engine 路径；
- 注册、连接、共享资源管理；
- 将网络进度与 GPU buffer step 对齐。

简化模型：

```text
GPU NCCL kernel              Host proxy                 NIC/plugin
      │                          │                          │
      ├─ 写 ready step ─────────►│                          │
      │                          ├─ isend/irecv ──────────►│
      │                          ├─ test request            │
      │                          │◄──────── completion ─────┤
      │◄─ 更新 tail/head ────────┤                          │
      └─ 消费/产生下一 chunk      │                          │
```

参考：

- [`src/proxy.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/proxy.cc)
- [`src/transport/net.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/transport/net.cc)

## 7. Channel 是并行度，不是连接数的同义词

NCCL 把一个 collective 切到多个 channel：

```text
message
  ├─ channel 0: chunk 0, 4, 8...
  ├─ channel 1: chunk 1, 5, 9...
  ├─ channel 2: chunk 2, 6, 10...
  └─ channel 3: chunk 3, 7, 11...
```

更多 channel 可以：

- 增加链路并行度；
- 同时使用多条 NIC rail；
- 提高大消息吞吐。

但会增加：

- GPU block/SM 占用；
- connector buffer；
- proxy 与队列开销；
- 小消息固定成本。

因此 NCCL 会基于 topology 和 message size 调优，而不是无条件把 channel 数开到最大。

## 8. Group 语义

`ncclGroupStart/End` 有两个常见用途：

1. 把多 GPU、多 communicator 操作成组提交，避免单线程初始化/调用死锁；
2. 让 send/recv pattern 被整体规划。

它不是事务：不能理解为全部成功或全部回滚。

## 9. NCCL 与 CUDA stream

关键规则：

- NCCL 操作与同一 stream 上前序 CUDA work 有序；
- 不同 stream 之间需要 event 依赖；
- host API 返回不代表 GPU 已完成；
- 销毁/复用 buffer 前必须建立正确 stream ordering；
- 多 communicator 共享 GPU 时还要关注 launch 顺序一致性。

官方文档说明，操作状态可通过标准 CUDA stream/event 语义查询。

## 10. 读源码的最短路径

```text
src/init.cc
  → src/graph/topo.cc / paths.cc / search.cc
  → src/transport.cc
  → src/transport/{p2p,shm,net}.cc
  → src/enqueue.cc
  → src/device/
  → src/proxy.cc
```

先掌握 host orchestration，再看 device primitive；直接从 CUDA 模板进入容易迷失在协议细节。

## 11. 自检

1. bootstrap socket 与 NET transport 有什么区别？
2. 为什么 NCCL API 返回不能作为 buffer 可复用的证明？
3. channel 数增大为什么可能降低小消息性能？
4. 跨机 collective 中 GPU kernel 和 proxy 分别推进什么状态？

## 参考

- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NCCL 官方源码](https://github.com/NVIDIA/nccl)
