---
title: "02. GPU、NIC、PCIe 拓扑与 DMA"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 02. GPU、NIC、PCIe 拓扑与 DMA

## 0. 先回答：数据到底存在哪里、谁能访问

CPU、GPU、NIC 都是能发起内存访问的设备，但它们看到的地址空间和访问方式不同：

```text
CPU core  ─load/store→ CPU virtual memory
GPU SM    ─load/store→ GPU virtual memory / HBM
GPU copy engine ─DMA→ host memory 或另一 GPU
NIC DMA engine ─DMA→ 注册过的 host/GPU memory
```

一个 CUDA pointer 只保证当前 CUDA 上下文可以使用，不自动代表 NIC 有权限。
要让 NIC 访问，需要驱动把相应页 pin 住、建立 DMA mapping，并把权限编码进
`lkey/rkey` 等 handle。

### 0.1 DMA 并不是一种“数据格式”

DMA 是“由设备搬数据”的工作方式。它不限定数据是 tensor、图片还是网络包。
例如 D2H copy：

```text
CPU 写 copy descriptor：src=GPU 地址，dst=host 地址，len=64 MiB
GPU copy engine 执行 DMA
CPU/GPU event 告知 copy 完成
```

payload 仍是那 64 MiB 原始字节；DMA 描述符只是告诉硬件去哪里搬、搬多少。

### 0.2 一个 64 MiB tensor 的时间下限

假设 tensor 需要经过 PCIe 4.0 x16，单向编码后理论约 31.5 GB/s：

```text
64 MiB / 31.5 GiB/s ≈ 1.98 ms
```

这是极理想的链路传输下限。CPU bounce 路径至少有 D2H 和后续网络/对端 H2D，
不能只算一次 1.98 ms。Direct GDR 省掉 host 中转，但仍要经历 GPU↔NIC 的 PCIe
访问和网络传输。

## 1. 四种需要分清的搬运路径

### 1.1 通用 Host staging / CPU bounce buffer

```text
Source GPU
  → D2H copy
  → source host buffer
  → I/O device / network
  → destination host buffer
  → H2D copy
  → Destination GPU
```

Host staging 是一个**通用架构模式**，并不是 blade-kvt 定义的协议。只要某个 I/O
设备或通信栈不能直接访问 GPU memory，应用或通信库就可以先把数据复制到它能访问
的 host buffer，再完成 I/O。

中间 host buffer 通常使用 pinned/page-locked memory，因为它：

- 不会在 DMA 期间被 OS 换出或迁移；
- 能建立稳定的 GPU/NIC DMA mapping；
- 支持真正的异步 CUDA D2H/H2D copy；
- 可以复用，避免每次传输临时 pin/unpin。

但“host staged”只规定 payload 经 host memory 中转，**不规定**：

- 一定使用 TCP 还是 RDMA；
- 一定由 GPU copy engine、GPU kernel 还是 CPU `memcpy` 完成 gather/scatter；
- buffer 一定在物理地址上连续；
- 一个应用传输对应一个网络 packet。

例如 MPI、UCX 或自研通信库都可能实现 GPU→pinned host→NIC 的 fallback。NVIDIA
对传统 I/O 路径的概括也是：peer device 与 GPU 之间先经过 system memory，需要
两次 DMA；GPUDirect RDMA 才移除这个 bounce buffer。

### 1.2 普通 Linux TCP socket 的 GPU→GPU 路径

最常见、没有 GPU-direct socket 扩展的 TCP 路径比上一节还要多出 Linux socket
层：

```text
发送端
Source GPU/HBM
  → CUDA D2H
  → application host buffer
  → send()/write()/sendmsg()
  → kernel TCP send buffer / sk_buff
  → TCP 分段、IP/Ethernet header、qdisc
  → NIC TX descriptor
  → NIC DMA-read host pages
  → Ethernet

接收端
Ethernet
  → NIC DMA-write RX buffers
  → NAPI / IP / TCP
  → kernel socket receive buffer
  → recv()/read()
  → application host buffer
  → CUDA H2D
  → Destination GPU/HBM
```

初学时可以先把普通 `send()`/`recv()` 理解为：

```text
GPU → 用户态 host buffer → 内核 socket buffer → NIC
NIC → 内核 socket buffer → 用户态 host buffer → GPU
```

其中 CPU 运行 CUDA API、系统调用和 TCP/IP 协议栈；GPU copy engine 或 kernel
负责 D2H/H2D；NIC DMA engine 负责 host memory 与网线之间的数据移动。这里的
“CPU 参与”不等于 CPU core 亲自驱动网线发送每个 bit。

几个重要例外和优化：

- `sendmsg/writev` 可以让一次系统调用描述多个 `iovec`，即 scatter-gather
  user buffers；这减少 syscall 次数，但不保证完全没有 copy。
- Linux `MSG_ZEROCOPY` 可以对较大的 host buffer 避免发送侧 user→kernel
  payload copy，但要 pin page 并异步回收；Linux 文档指出它通常在约 10 KiB
  以上才可能划算。
- GSO/TSO 可以让协议栈一次交给 NIC 一个较大的逻辑 TCP segment，由软件或 NIC
  再按 MSS 分段。应用的大 `send` 仍不会变成一个超大 Ethernet frame。
- 新的 dma-buf/device-memory TCP 属于专门扩展，不能反推普通 socket 可以直接
  `send(cuda_pointer)`。

因此 TCP 也属于 host-staged 大类，但普通 TCP 的 host path 往往同时包含
application buffer 与 kernel socket buffer；“host staged RDMA”则可以让 RNIC
直接 DMA 已注册的 pinned host MR，二者不能画成完全相同的协议栈。

### 1.3 GPU P2P

同机 GPU 可通过 NVLink 或 PCIe P2P 直接访问对端显存，不经过用户态 host bounce buffer。

```text
GPU0 ── NVLink / PCIe Switch ── GPU1
```

### 1.4 GPUDirect RDMA

NIC 直接 DMA GPU memory：

```text
GPU memory ←→ GPU BAR / peer mapping ←→ PCIe fabric ←→ NIC DMA
```

CPU 仍负责建联、注册内存、post WR 和处理 completion，但不搬运 payload。

![普通 TCP、通用 Host staged 与 GPUDirect RDMA 数据路径](/imgs/gpudirect_paths.svg)

#### 这张图依据什么

这张图首先是**通用机制图，不是从 blade-kvt 代码反推出来的标准定义**：

- Host-staged 与 GPUDirect RDMA 的边界依据 NVIDIA GPUDirect RDMA 文档：前者
  经过 system memory bounce buffer，后者允许第三方 PCIe device 直接访问
  GPU memory。
- TCP 一列依据标准 Linux socket 数据路径、`sendmsg/iovec`、GSO/TSO 和 NIC
  queue/DMA 机制。
- pinned host memory 的作用依据 CUDA Programming Guide。

blade-kvt 是它的一个具体实例：TCP/staged 路径选择用 GPU kernel 把离散 KV
block gather 到连续的 pinned host wire buffer，对端再按 metadata scatter；
别的库完全可以用 `cudaMemcpyAsync`、多个 `iovec`、预注册 buffer pool 或其他
方式实现同一个通用 host-staged 模式。

## 2. 为什么“同一台机器”还不够

Linux P2PDMA 文档强调：PCIe 对同一 hierarchy 内的 TLP 路由定义明确，但事务一旦到达 Host Bridge，跨 hierarchy 的转发由平台决定，内核默认不会假定它安全。

按常见性能顺序：

1. GPU 与 NIC 位于同一 PCIe switch 下：最优。
2. 经同一 CPU/IOH：通常可用，但更慢。
3. 跨 socket，经 UPI/QPI/Infinity Fabric：可能严重降速，甚至不可靠。

因此 `NIC 带宽够`、`GPU 支持 GDR` 仍不能推出端到端性能好。

### 2.1 共享上行的数值例子

```text
GPU0 x16 ┐
GPU1 x16 ├─ PCIe Switch ─ x16 upstream ─ CPU
NIC0 x16 ┘
```

这里每个下行端口都标 x16，但三个设备同时访问 CPU 时共享一个 x16 upstream。
如果是 Gen4，上行总单向编码后理论值约 31.5 GB/s，不是
`3 × 31.5 = 94.5 GB/s`。

若 GPU0 与 NIC0 能在 switch 内直接 P2P，数据可能不占用 upstream；但是否允许
取决于 switch 路由、ACS、IOMMU 与平台支持，不能只根据拓扑图猜测。

## 3. `nvidia-smi topo -m` 的距离

常见标签：

| 标签 | 直觉 |
|---|---|
| `PIX` | 最多经过一个 PCIe bridge，通常同一 switch |
| `PXB` | 经过多个 PCIe bridge |
| `PHB` | 经过 PCIe Host Bridge/CPU |
| `NODE` | 跨同一 NUMA node 内多个 host bridge |
| `SYS` | 跨 NUMA node/socket |
| `NV#` | 通过若干条聚合 NVLink |

命令：

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -p2p n
```

对 blade-kvt，应同时看 GPU↔NIC 距离，而不是只看 GPU↔GPU。

## 4. NUMA：为什么同一地址类型会有不同访问代价

NUMA 是 **Non-Uniform Memory Access，非一致内存访问架构**。“非一致”不是说
数据内容不一致，而是说：CPU 或 I/O device 访问不同位置的 memory 时，延迟、
带宽和经过的互连不同。

### 4.1 从硬件结构理解 NUMA node

一台服务器可以抽象成多个 node：

```text
NUMA node 0
  ├─ CPU cores 0..N
  ├─ LLC / coherence-home agents
  ├─ memory controllers 0..M → local DRAM 0
  └─ PCIe Root Ports → GPU0 / NIC0 / NVMe0

NUMA node 1
  ├─ CPU cores ...
  ├─ other memory controllers → local DRAM 1
  └─ other PCIe Root Ports → GPU1 / NIC1

node 0 ◄──── socket/SoC interconnect ────► node 1
```

对 node 0 的 core 来说：

```text
访问 node 0 DRAM：local memory access
访问 node 1 DRAM：remote memory access
```

remote access 必须多经过 UPI/QPI、Infinity Fabric 或 SoC fabric，通常增加延迟，
占用跨 node 带宽。NUMA 不严格等于“多 CPU socket”：一个 socket/大型 SoC
也可能暴露多个 NUMA node；反过来，平台也可能把多个硬件域交织成较少的 OS node。

Linux 的 NUMA node 同时描述：

- 哪些 logical CPUs 靠近哪些 memory controllers/DRAM；
- 一块物理页属于哪个 memory node；
- PCIe device 靠近哪个 node；
- node-to-node 的相对 distance。

普通匿名内存常遵循 first-touch：哪个 node 上的 CPU thread 第一次实际写入页面，
页面往往就分配到哪个 node；`numactl --membind/--preferred/--interleave` 可以改变
policy。Pinned memory 只是把页锁住，并不会自动把已经分错 node 的页面迁到 NIC
附近。

### 4.2 控制面

创建 QP、post WR、poll CQ 的 CPU thread 如果跑在远端 NUMA node，会增加 MMIO、cache miss 和内存访问延迟。

### 4.3 数据面

- staged/TCP 路径经过 host pinned buffer，直接消耗该 NUMA node 的内存带宽。
- direct GDR payload 不经 host DRAM，但拓扑仍可能经过 CPU I/O fabric。

例如 source GPU、RNIC 和 pinned buffer 分别位于三个不同 node 时，D2H、NIC DMA
和控制线程可能都跨 NUMA。只绑定 CPU、不绑定 memory，或者只选择亲和 NIC、不管
host buffer 的 node，都可能留下瓶颈。

检查：

```bash
numactl --hardware
numactl --show
cat /sys/bus/pci/devices/0000:65:00.0/numa_node
cat /sys/bus/pci/devices/0000:17:00.0/numa_node
```

sysfs 返回 `-1` 时表示内核没有给这个 device 提供明确的 NUMA affinity，不表示
“访问所有 node 一样快”。仍要结合 `lspci -tv`、固件拓扑和实测。

## 5. IOMMU、ACS 与 P2P

### 5.1 IOMMU

传统 GPUDirect RDMA 要求不同 PCIe device 对物理地址有一致视图。NVIDIA 文档指出，执行非 1:1 地址转换的 IOMMU 会破坏这一前提；常见要求是关闭 IOMMU 或使用 pass-through。

现代 dma-buf 路径在驱动和内核支持下可改善注册与地址交换，但不能忽略具体平台约束。

### 5.2 ACS

Access Control Services 可以强制 P2P TLP 上行到 Root Port，从而把原本同 switch 的短路径变成长路径。虚拟化隔离需要 ACS，但性能目标可能希望允许同层 P2P。

不要未经平台安全评估就全局关闭 ACS。正确做法是：

1. 确认当前 ACS capability/control；
2. 确认 GPU/NIC 是否真的需要 P2P；
3. 与虚拟化和安全隔离要求一起评估。

## 6. GPU memory 注册

RDMA 操作不能只拿一个 CUDA pointer 就直接发。通常需要：

1. CUDA 分配的显存保持存活；
2. Barex/verbs 注册或采用该地址；
3. 得到本地 `lkey`，供本地 SGE 使用；
4. 接收端把 `raddr/rkey` 通过控制面交给发送端；
5. NIC 用 `(local addr, lkey, remote addr, rkey)` 执行 RDMA。

在 blade-kvt 中：

- `BarexCtx` 构造时逐 layer、逐 tensor 调用 `RegUserMr(..., GPU, device_id)`；
- server 将 `out.mr->rkey` 与 GPU 地址序列化给 client；
- client 的 `RDMAChannel::send_data` 构造 `rw_memp_t`。

对应源码：

- `blade-kvt/kvtransfer/src/barex_protocol.cpp:444-485`
- `blade-kvt/kvtransfer/src/rdma_channel.cpp:370-385`
- `blade-kvt/kvtransfer/src/rdma_channel.cpp:724-745`

## 7. RDMA Write 的地址四元组

```text
Local SGE:
  addr   = source GPU base + src_offset
  length = bytes
  lkey   = source GPU MR local key

Remote:
  r_addr = destination GPU base + dst_offset
  r_key  = destination GPU MR remote key
```

`lkey` 防止本地 NIC 访问未注册内存；`rkey` 授权远端 QP 对远端 MR 执行指定操作。

错误对照：

| 症状 | 常见原因 |
|---|---|
| `IBV_WC_LOC_PROT_ERR` | 本地 addr/lkey/length 不合法 |
| `IBV_WC_REM_ACCESS_ERR` | raddr/rkey 越界、失效或权限错误 |
| `IBV_WC_RETRY_EXC_ERR` | 对端/QP/网络不可达 |
| `IBV_WC_RNR_RETRY_EXC_ERR` | SEND 或 WRITE_WITH_IMM 消耗 recv WR，但对端未及时 post recv |

## 8. Pinned host memory 的角色

Pinned memory 不会被 OS 换出，GPU copy engine 与 NIC 可稳定 DMA。它不是“零拷贝”的同义词：

- TCP：GPU→pinned host 是一次 D2H；网络栈再发送。
- staged RDMA：GPU→pinned host，然后 NIC RDMA 写远端 pinned host，再 H2D。
- direct RDMA：payload 不经过 pinned host；控制消息仍可能使用 host buffer。

## 9. 实战检查清单

```bash
# 1. 树形拓扑、链路能力和当前状态
lspci -t
lspci -vv -s "$GPU_BDF" | rg 'LnkCap|LnkSta|ACSCtl'
lspci -vv -s "$NIC_BDF" | rg 'LnkCap|LnkSta|ACSCtl'

# 2. GPU/NIC 拓扑
nvidia-smi topo -m

# 3. NUMA
cat "/sys/bus/pci/devices/$GPU_BDF/numa_node"
cat "/sys/bus/pci/devices/$NIC_BDF/numa_node"

# 4. RDMA 设备
ibv_devices
ibv_devinfo
rdma link

# 5. GPU peer-memory/dma-buf 相关
lsmod | rg 'nvidia_peermem|nvidia'
dmesg | rg -i 'iommu|dmabuf|peer.?mem|rdma'
```

## 10. 自检

1. 为什么 direct GDR 不经过 host memory，仍受 PCIe/NUMA topology 影响？
2. 为什么 `rkey` 不能在目标进程重启后继续复用？
3. 为什么 WRITE_WITH_IMM 可能遇到 RNR，而普通 RDMA WRITE 通常不消耗接收 WR？
4. 为什么 staged 模式的网络很快，端到端仍可能慢？
5. 普通 TCP 从 GPU 到 GPU 通常经过哪些 user/kernel/host buffer？TSO 为什么没有
   消除 D2H/H2D？
6. Host staging 为什么是通用模式，而不是 blade-kvt 独有的 copy kernel？
7. NUMA 的“non-uniform”究竟指数据不一致，还是访问代价不一致？
8. Pinned buffer 为什么仍可能分配在错误的 NUMA node？

## 参考

- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/archive/12.6.3/gpudirect-rdma/index.html)
- [NVIDIA CUDA Programming Guide：Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#page-locked-host-memory)
- [Linux：`sendmsg` 与 scatter-gather `iovec`](https://man7.org/linux/man-pages/man2/sendmsg.2.html)
- [Linux：`MSG_ZEROCOPY`](https://docs.kernel.org/networking/msg_zerocopy.html)
- [Linux：TCP Segmentation Offload / GSO / GRO](https://docs.kernel.org/networking/segmentation-offloads.html)
- [Linux：NUMA memory policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)
- [Linux PCI Peer-to-Peer DMA Support](https://docs.kernel.org/driver-api/pci/p2pdma.html)
- [NVIDIA GPUDirect 概览](https://developer.nvidia.com/gpudirect)
