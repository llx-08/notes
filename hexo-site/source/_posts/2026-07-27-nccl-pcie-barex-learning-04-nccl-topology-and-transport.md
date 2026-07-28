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

### 2.1 用一台 4 GPU 服务器走一遍判断

假设拓扑：

```text
CPU0
 ├─ PCIe Switch A ─ GPU0, GPU1, NIC0
 └─ PCIe Switch B ─ GPU2, GPU3, NIC1

GPU0↔GPU1 有 NVLink
GPU2↔GPU3 有 NVLink
Switch A 与 B 之间只能经 CPU/Root Complex
```

NCCL 需要分别回答：

1. GPU0→GPU1：NVLink P2P 可用吗？若可用，代价很低。
2. GPU0→GPU2：是否能 PCIe P2P？若不能，是否用 SHM 中转？
3. GPU0 跨机出网：NIC0 近，是否直接 GDR？
4. GPU2 跨机出网：NIC1 近，是否分到另一条 rail？
5. 4 个 rank 做 AllReduce：ring 顺序如何避免所有流量挤同一上行？

因此“选 transport”是对每条 peer edge 作决定；“选 algorithm”是把这些 edge
组织成整个 collective。二者不是同一个 if/else。

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

## 10. 跟源码走一遍：communicator 初始化时如何建立拓扑

本节固定阅读 **NCCL 2.30.7，commit
`5067397c2676d5aed50042fc39e5c8ee96eb0027`**。不同版本的函数名和策略可能变化，
但“发现硬件 → 计算路径 → 搜索 graph → 为 graph 的边选择 transport”这一分层思想
比较稳定。

入口之一是 `src/init.cc::initTransportsRank()`。这个函数很长，不要试图一次读完，
先抓住下面这条主线：

```text
每个 rank 填写自己的 ncclPeerInfo
  │
  ├─ bootstrapAllGather(peerInfo)
  │    所有 rank 交换 hostHash、pidHash、busId、GPU 能力等
  │
  ├─ ncclTopoGetSystem()
  │    读取/组合 GPU、NIC、CPU、PCI、NVLink/NVSwitch 等信息
  │
  ├─ ncclTopoComputePaths()
  │    计算设备对之间可用路径、类型和带宽
  │
  ├─ ncclTopoTrimSystem()
  │    删除当前 communicator 不可达或不需要的设备
  │
  ├─ ncclTopoComputePaths()
  │    trim 后重新计算路径
  │
  ├─ ncclTopoSearchInit()
  │    初始化 graph 搜索需要的数据
  │
  ├─ ncclTopoCompute(ringGraph)
  ├─ ncclTopoCompute(treeGraph)
  ├─ ncclTopoCompute(collNetGraph / nvlsGraph ...)
  │
  └─ 根据 graph 中每条 rank-to-rank edge 建立 connector
```

### 10.1 第一次 AllGather 交换的不是用户 tensor

`bootstrapAllGather(comm->bootstrap, comm->peerInfo, ...)` 是初始化阶段的控制面通信，
不是 AllReduce 的数据面通信。它让各 rank 获得一份全局身份表。例如：

```text
rank 0:
  hostHash = A
  pidHash  = P0
  busId    = GPU0 的 PCI BDF

rank 1:
  hostHash = A
  pidHash  = P1
  busId    = GPU1 的 PCI BDF

rank 8:
  hostHash = B
  pidHash  = P8
  busId    = 另一台机器上的 GPU0 BDF
```

由此可以先回答：

- 两个 rank 是否在同一 host；
- 是否在同一进程；
- 分别使用哪个 GPU；
- 各 rank 的 NCCL 版本和能力是否兼容。

注意：`busId` 在不同 host 上可能重复，必须和 host 身份一起解释。

### 10.2 topology graph 中有什么节点和边

`ncclTopoGetSystem()` 得到的是硬件拓扑，不是 collective 的 rank graph。可把它理解为：

```text
硬件 topology graph

CPU/NUMA ─ PCI bridge/switch ─ GPU
    │                    └─── NIC
    └─ inter-socket link ─ CPU/NUMA

GPU ─ NVLink/NVSwitch ─ GPU
```

节点表示 GPU、NIC、CPU、PCI 元件等；边记录链路类型和带宽。随后
`ncclTopoComputePaths()` 计算“从任一 GPU/NIC 到另一节点”的候选路径。

这与最终 Ring graph 不同：

```text
硬件 topology graph：设备实际上怎样连接
Ring graph：            rank 0 的 next 是谁、rank 1 的 next 是谁
connector/transport：   这条 next edge 最终怎样传数据
```

### 10.3 为什么 trim 后还要再次 ComputePaths

路径计算可能最初包含当前 communicator 看不见、不可访问或用不到的设备。
`ncclTopoTrimSystem()` 删除这些节点后，原路径表已经不再对应新图，所以源码紧接着再次
调用 `ncclTopoComputePaths()`。

这是一个很实用的读码习惯：看到同一计算调用两次，不要立刻认为重复；检查中间是否改变了
它依赖的数据结构。

### 10.4 graph 搜索不是只找“最短的一条路”

源码分别构造 `ringGraph`、`treeGraph`、CollNet graph、`nvlsGraph`，为它们设置：

- `pattern`：Ring、Balanced Tree、NVLS 等；
- `minChannels` / `maxChannels`；
- 是否需要 CollNet；
- 搜索得到的节点内/节点间带宽与 path type。

然后调用 `ncclTopoCompute()`。目标是找到一组可并行的 channel，并兼顾：

- 慢边是否进入关键路径；
- 多个 channel 会不会争用同一物理上行；
- 多 NIC 是否能形成多 rail；
- 节点内 graph 如何和节点间 graph 拼起来。

因此它更接近“受拓扑和带宽约束的 graph 映射”，而不是普通的单源最短路径。

## 11. 跟源码走一遍：一条 edge 如何选择 Transport

核心代码位于 `src/transport.cc`：

```cpp
struct ncclTransport* ncclTransports[NTRANSPORTS + 1] = {
  &p2pTransport, &shmTransport, &netTransport, &collNetTransport,
  &profilerTransport
};
```

用于正常连接选择的前四个候选按顺序是：

```text
P2P → SHM → NET → CollNet
```

`selectTransport<send_or_recv>()` 的逻辑可简化为：

```cpp
for (每个候选 transport) {
  transport->canConnect(&ret, comm, graph, myInfo, peerInfo);
  if (ret) {
    connector->transportComm = 该 transport 的 send 或 recv 函数表;
    connector->transportComm->setup(...);
    return success;
  }
}
```

这里有三个非常重要的结论。

### 11.1 “顺序靠前”不代表永远使用它

NCCL 不是看见 P2P 在数组第一个就强行用 P2P。每个 transport 都先执行自己的
`canConnect()`：

```text
同机 + CUDA P2P/拓扑/配置允许      → P2P canConnect=true
同机 + P2P 不可用 + SHM 可用       → SHM canConnect=true
需要网络并且 NET 可建立连接         → NET canConnect=true
graph/硬件/plugin 支持 collective  → CollNet 可参与相应连接
```

第一个满足条件的候选才会被写入 connector。

### 11.2 send connector 和 recv connector 分开建立

代码通过模板参数选择：

```text
channel.peers[peer].send[connIndex]
channel.peers[peer].recv[connIndex]
```

同一条逻辑通信 edge 有本地 send 侧和 recv 侧状态。两边的 buffer 指针、head/tail
视角、setup/connect 操作并不相同，不能把 connector 理解成一个无方向的 socket。

### 11.3 Transport 主要在连接建立时决定

普通 collective 调用不是每次传一个 chunk 都重新执行：

```text
if (P2P) ... else if (SHM) ... else if (NET) ...
```

初始化或首次需要某条连接时，NCCL 选好 transport 并保存函数表及资源。运行时 GPU
kernel 使用已经准备好的 `ncclConnInfo`，必要时 proxy 调用保存好的
`proxyProgress`。

有些 P2P 连接采用延迟连接：collective 准备阶段发现本次会使用某 peer/channel，
再执行 preconnect。但这是“按需把连接补齐”，不是每个数据 step 重新做全局拓扑搜索。

## 12. Connector 到底保存了什么

源码中的关系可以画成：

```text
ncclComm
  └─ channels[channelId]
       └─ peers[peer]
            ├─ send[connIndex] : ncclConnector
            └─ recv[connIndex] : ncclConnector
                                  │
                                  ├─ connected
                                  ├─ transportComm ─→ P2P/SHM/NET 的函数表
                                  ├─ transportResources
                                  ├─ proxyConn
                                  └─ conn : ncclConnInfo
```

`ncclConnector` 是 host 侧“这条有向连接”的总记录。`ncclConnInfo` 是 GPU kernel
实际需要看到的紧凑连接描述，其重要字段包括：

```cpp
char*     buffs[NCCL_NUM_PROTOCOLS];
void*     mhandles[NCCL_NUM_PROTOCOLS];
uint64_t* tail;
uint64_t* head;
int       flags;
int       stepSize;
ncclConnFifo* connFifo;
uint64_t  step;
```

可以逐项理解：

| 字段 | 含义 |
|---|---|
| `buffs[protocol]` | Simple、LL、LL128 各协议使用的连接 buffer |
| `mhandles` | NET plugin 注册这些 buffer 后得到的 memory handle；是否使用取决于路径 |
| `head` / `tail` | 有界循环 buffer 的消费/生产进度，即 credit 和完成握手的一部分 |
| `flags` | 是否 direct read/write、是否 `NCCL_DIRECT_NIC` 等 |
| `stepSize` | Simple 协议一个 step 可使用的 buffer 大小 |
| `connFifo` | GPU 与 proxy 交换每个 step 的 size、offset 等元数据 |
| `step` | 这条连接当前推进到的逻辑 step |

send 与 recv 对同一 head/tail 的本地、远端视角不同。源码注释直接说明：

```text
tail：recv 侧是 local，send 侧是 remote
head：send 侧是 local，recv 侧是 remote
```

这也是为什么前面介绍 Proxy 时会看到 GPU 与 CPU proxy 围绕 `head`、`tail`、
`connFifo[step % NCCL_STEPS]` 做生产者—消费者握手。

### 12.1 connIndex 是什么

`connIndex` 不是 rank，也不是 channel。它允许同一 `(channel, peer, direction)` 下存在
不同用途的连接槽，例如普通 collective、P2P 或特定 direct 路径需要不同 connector。

初学时先记住四维定位：

```text
(channelId, peer rank, send/recv, connIndex) → 唯一 connector
```

## 13. NET Setup 与 GDR：源码怎样决定 GPU buffer 还是 Host buffer

`src/transport/net.cc::sendSetup()` 里最关键的几步是：

```text
ncclTopoGetNetDev(...)
  → 为当前 rank、graph、channel、peer 找 NIC 和可能的 proxyRank

ncclTopoCheckGdr(..., isSend=1, &useGdr)
  → 检查发送方向能否 GDR

send->conn.flags |= useGdr ? NCCL_DIRECT_NIC : 0
```

接收侧 `recvSetup()` 类似，但调用时 `isSend=0`，而且还可能执行
`ncclTopoNeedFlush()`，判断 RNIC DMA 写入 GPU memory 后，GPU 消费前是否需要额外
可见性 flush。

### 13.1 为什么 GDR 要分发送方向和接收方向

两种 DMA 方向不同：

```text
发送：RNIC 从 GPU memory 读取数据
接收：RNIC 把网络数据写入 GPU memory
```

平台可能对读和写有不同限制、性能和一致性要求，所以不能用一个布尔值草率代表所有方向。

### 13.2 `NCCL_DIRECT_NIC` 表示什么

它表示这个 connector 的数据 buffer 可由 NIC 直接访问。它不表示：

- 完全没有 CPU；
- 不经过 PCIe/NVLink/C2C 等本机 I/O 路径；
- 网络上没有协议处理；
- 所有 control metadata 也都在 GPU 上完成。

通常仍有 host proxy 负责调用 Net Plugin、提交 `isend/irecv`、轮询 `test`；
payload 是否需要 host staging 是另一个问题。

### 13.3 NET buffer 注册的两条 GPU memory 路径

NET connector 创建 buffer 后，源码根据 buffer 位于 device memory 还是 host memory，
传入：

```text
NCCL_PTR_CUDA 或 NCCL_PTR_HOST
```

若 GPU buffer 且 DMA-BUF 可用：

```text
cuMemGetHandleForAddressRange(...)
  → 获得 dma-buf fd
  → ncclNet->regMrDmaBuf(..., NCCL_PTR_CUDA, fd, ...)
```

否则会退回：

```text
ncclNet->regMr(..., NCCL_PTR_CUDA 或 NCCL_PTR_HOST, ...)
```

`NCCL_PTR_CUDA` 只是在 Net Plugin ABI 中声明“这是 CUDA/device pointer”。
plugin 和驱动仍需真正支持 GPU memory 注册。例如内置 IB backend 会把
`NCCL_PTR_CUDA` 支持与 GDR 能力关联起来。

### 13.4 GDR 与 Non-GDR 的 connector 差异

可粗略理解为：

```text
GDR connector:
  buffs[proto] 位于 GPU/device memory
  regMrDmaBuf/regMr(... NCCL_PTR_CUDA ...)
  conn.flags 带 NCCL_DIRECT_NIC
  RNIC ↔ GPU buffer

Non-GDR connector:
  buffs[proto] 位于 host shared/pinned memory
  regMr(... NCCL_PTR_HOST ...)
  GPU ↔ host network buffer ↔ RNIC
```

这不是说 Non-GDR 会用普通 `memcpy()` 把整个 tensor 一次性搬到 host。NCCL GPU
kernel 和 proxy 通常围绕固定大小的 step buffer 流水推进，host buffer 会循环复用。

## 14. PXN 在源码中是什么样

发送侧 `ncclTopoGetNetDev()` 除了返回 NIC，还返回 `proxyRank`：

```text
当前 rank == proxyRank
  → 当前 GPU 对应的进程直接管理该 NET connector

当前 rank != proxyRank
  → 使用邻近 GPU/rank 的 proxy 和 NIC，即 PXN
```

源码会记录：

```cpp
if (proxyRank != myInfo->rank)
  comm->useNetPXN = true;
```

并通过 `ncclProxyConnect(..., proxyRank, ...)` 连接到代理 rank 的 proxy。日志中也会打印：

```text
via NET/<plugin>/<netDev>(<proxyRank>)/GDRDMA
```

所以“GPU 有 NIC 亲和性”只是起点；NCCL 还可能根据 graph/channel 选择一条 rail，
再让另一个更靠近该 NIC 的 GPU/rank 充当网络代理。

当前这版源码的接收 setup 中明确写有“不支持 receive PXN”的注释。因此读 PXN 时要区分：

- NCCL 版本；
- send/receive 方向；
- 是否使用特定 Net Plugin 或 device offload。

不能只用一张概念图概括所有版本和方向。

## 15. 把 Algorithm、Channel、Connector、Transport 串成一个例子

假设两台机器，每台 2 张 GPU，执行 4-rank Ring AllReduce：

```text
Host A                         Host B
rank0/GPU0 ─ rank1/GPU1 ─NET─ rank2/GPU0 ─ rank3/GPU1
    ▲                                           │
    └──────────────── NET ──────────────────────┘
```

这里只画一个 channel。初始化时大致发生：

```text
1. topology search 选出 Ring rank 顺序：
   0 → 1 → 2 → 3 → 0

2. 针对每条 edge 建 send/recv connector：
   0→1：同机 P2P
   1→2：跨机 NET
   2→3：同机 P2P
   3→0：跨机 NET

3. 对 NET edge 决定：
   使用哪个 NIC
   是否 PXN
   是否 GDR
   buffer 如何注册
   是否需要 host proxy progress

4. 运行 AllReduce 时：
   Ring kernel 只按 prev/next 和 ncclConnInfo 推进
   P2P edge 直接走 peer buffer
   NET edge 与 proxy 通过 head/tail/connFifo 协作
```

这解释了一个容易困惑的现象：同一个 NCCL kernel 中，不同 logical neighbor edge
可以落在不同物理 transport；kernel 的 Ring 算法代码不需要为每种网卡重写一份。

### 15.1 推荐的源码阅读断点

如果可以调试 NCCL，按下面顺序设置断点或日志最容易建立整体认识：

```text
初始化/拓扑：
  initTransportsRank
  ncclTopoGetSystem
  ncclTopoComputePaths
  ncclTopoCompute

连接选择：
  selectTransport
  p2pCanConnect / shmCanConnect / netCanConnect
  sendSetup / recvSetup

运行期：
  ncclEnqueueCheck
  ncclLaunchKernel
  ncclKernelMain
  sendProxyProgress / recvProxyProgress
```

观察 `selectTransport` 时重点记录：

```text
comm->rank
peer
channelId
send/recv
connIndex
每个 canConnect 的返回值
最终 connector->transportComm
connector->conn.flags
```

这比只看一句 `via NET/IB/...` 更能回答“为什么选择了这条路径”。

## 16. 自检

1. 为什么 transport 是“peer edge”的属性，而 algorithm 是“全局 rank graph”的属性？
2. SHM 为什么仍可能受 PCIe 和 NUMA 双重限制？
3. PXN 为什么可能比 GPU 直接访问远端 NIC 更快？
4. 如何从日志证明一次 collective 真正走了 IB/GDR？
5. 为什么 `ncclTransports` 中 P2P 排第一，不代表跨机 edge 会尝试用 P2P 传网络数据？
6. `(channelId, peer, direction, connIndex)` 为什么才能唯一定位 connector？
7. `NCCL_DIRECT_NIC` 为什么不等价于“CPU 完全不参与”？
8. topology graph、Ring graph、connector 三者分别描述什么？

## 参考

- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL `src/init.cc`：`initTransportsRank`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/init.cc)
- [NCCL `src/transport.cc`：候选 Transport 与 `selectTransport`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/transport.cc)
- [NCCL `src/include/device.h`：`ncclConnector` 与 `ncclConnInfo`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/include/device.h)
- [NCCL `src/transport/net.cc`：NET setup、GDR、buffer 注册与 Proxy](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/transport/net.cc)
- [NCCL topology 源码目录](https://github.com/NVIDIA/nccl/tree/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph)
