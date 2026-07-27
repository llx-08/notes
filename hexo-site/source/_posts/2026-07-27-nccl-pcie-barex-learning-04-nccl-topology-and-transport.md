---
title: "04. NCCL 拓扑发现与 Transport 选择"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 04. NCCL 拓扑发现与 Transport 选择

## 1. 三件事必须分开

| 层次 | 回答的问题 |
|---|---|
| topology path | GPU 到 GPU/NIC 的物理路径与代价是什么？ |
| transport | 这对 peer 用 P2P、SHM 还是 NET？ |
| algorithm graph | 所有 rank 如何组成 Ring/Tree/NVLS/CollNet？ |

“机器有 NVLink”不能直接推出所有通信都走 NVLink；“安装 IB”也不能推出所有跨机路径都用 IB。

## 2. NCCL 构建的拓扑图

NCCL 综合：

- `/sys` PCI topology；
- GPU bus ID、NVLink/NVSwitch；
- CPU/NUMA；
- NIC device、port 与带宽；
- P2P/GDR 能力；
- 用户提供的 topology XML；
- 虚拟化和容器中可见设备。

然后计算节点间路径类型与带宽。常见距离概念与 `nvidia-smi topo -m` 相近：

```text
LOC < NVL < PIX < PXB < PHB < SYS < NET
```

它不是简单的“跳数”：路径还包含链路宽度、带宽、CPU 架构和可用能力。

## 3. Transport 选择

简化优先级：

```text
同机且 GPU P2P 可用       → P2P transport
同机但 P2P 不可用         → SHM transport
跨机或需网络               → NET transport
```

实际实现会考虑 read/write mode、PXN、GDR、DMA-BUF、peer access 等条件。

![NCCL topology、transport 与 algorithm 的关系](/imgs/nccl_transport_selection.svg)

### 3.1 P2P transport

常见机制：

- CUDA IPC 映射对端 GPU memory；
- 同进程 direct pointer；
- NVLink/PCIe peer access；
- 某些场景使用 Copy Engine。

P2P transport 是 NCCL 内部相邻 rank 的 transport，不等同于 `ncclSend/ncclRecv` API。

### 3.2 SHM transport

当 GPU P2P 不可用但进程在同一 host，NCCL 可使用 host shared memory 作为中转：

```text
GPU A ↔ host shared buffer ↔ GPU B
```

这通常比跨网络轻，但会消耗 host memory bandwidth，并受 NUMA 影响。

### 3.3 NET transport

NET transport 通过内置 IB/socket 或外部 Net Plugin。核心 plugin ABI 包括：

- `devices/getProperties`；
- `listen/connect/accept`；
- `regMr/deregMr`；
- `isend/irecv/iflush`；
- `test`。

插件必须导出 NCCL 识别的 `ncclNet_v*` 结构。**Barex 1.5.3-1 没有这些符号，因此不是 NCCL Net Plugin。**

## 4. GDR 是否启用

跨机网络常见两条路径：

```text
GDR:
GPU buffer ←PCIe→ NIC ←network→ NIC ←PCIe→ GPU buffer

Non-GDR:
GPU buffer ←PCIe→ host buffer ←→ NIC ... NIC ←→ host buffer ←PCIe→ GPU
```

NCCL 会结合 GPU/NIC 拓扑、驱动能力、NIC plugin 属性等判断 GDR。手动环境变量可以禁用或限制，但不应在不了解拓扑时强制开启。

相关配置包括：

- `NCCL_IB_DISABLE`
- `NCCL_NET`
- `NCCL_NET_PLUGIN`
- `NCCL_NET_GDR_LEVEL`
- `NCCL_DMABUF_ENABLE`
- `NCCL_IB_HCA`
- `NCCL_SOCKET_IFNAME`

以当前官方文档为准，不要把旧博客中的默认阈值长期固化。

## 5. PXN：借邻居 GPU 使用更合适的 NIC

PCIe × NVLink 场景下，本 rank 的 GPU 不一定离目标 NIC 最近。PXN（PCI × NVLink）可先通过 NVLink 把数据送到靠近 NIC 的 GPU，再由该 GPU/NIC rail 出网。

```text
GPU0 ─NVLink→ GPU1 ─PCIe→ NIC1 → Network
```

这解释了为什么 NCCL 的 topology search 不是“每个 GPU 固定绑定最近 NIC”这么简单。

## 6. Ring/Tree graph 与物理路径

拓扑搜索的目标不是找唯一最短路径，而是找一组并行 channel，使：

- ring 邻接边有足够带宽；
- tree 上下行平衡；
- NIC rail 分配合理；
- 尽量减少共享 bottleneck；
- 跨节点与节点内路径可组合。

例如 8 GPU 双 NIC：

```text
channel 0/2/4/... → NIC0 rail
channel 1/3/5/... → NIC1 rail
```

算法 graph 最终决定每个 channel 中 rank 的前驱/后继或 parent/children，transport 再实例化这些边。

## 7. 常见误判

### 误判 1：看到 `NCCL_SOCKET_IFNAME` 就是 payload 走 socket

该变量也影响 bootstrap；而且其他项目可能复用它做网卡选择。必须结合 `NCCL_DEBUG=INFO`/`TRACE` 看最终 NET transport。

### 误判 2：`NV#` 越大一定越快

还要看链路代际、聚合方式、是否共享 NVSwitch、消息方向和 algorithm。

### 误判 3：所有 rank 都应使用同一 NIC

多 rail 系统通常要分散到不同 NIC；强制单 NIC 可能把带宽砍半。

### 误判 4：同一 PCI switch 就一定可 P2P

ACS、IOMMU、驱动、虚拟化和 peer access capability 都可能阻止或改变路径。

## 8. 观察 NCCL 的选择

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,P2P,SHM,TUNING
```

重点找：

- GPU/NIC bus ID 与 topology；
- P2P 是否 disabled；
- NET/IB/socket plugin；
- channel 数；
- Ring/Tree rank 顺序；
- GDR read/write；
- algorithm/protocol tuning 结果。

必要时：

```bash
export NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo.xml
export NCCL_GRAPH_DUMP_FILE=/tmp/nccl-graph.xml
```

变量是否可用及格式以当前 NCCL 文档/版本为准。

## 9. 与 Barex 的对照

| NCCL | Barex |
|---|---|
| topology graph 自动搜索 collective graph | `XDeviceManager` 枚举 device，应用决定 peer/channel |
| transport P2P/SHM/NET | backend RDMA/Solar/TCP |
| Net Plugin ABI | 无 NCCL plugin ABI |
| communicator/rank | context/channel，没有 collective rank graph |
| CUDA stream completion | callback/CQ completion |

## 10. 自检

1. 为什么 transport 是“peer edge”的属性，而 algorithm 是“全局 rank graph”的属性？
2. SHM 为什么仍可能受 PCIe 和 NUMA 双重限制？
3. PXN 为什么可能比 GPU 直接访问远端 NIC 更快？
4. 如何从日志证明一次 collective 真正走了 IB/GDR？

## 参考

- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL `src/transport.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/transport.cc)
- [NCCL topology 源码](https://github.com/NVIDIA/nccl/tree/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph)
