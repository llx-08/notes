# 00. 零基础导读：CPU、内存、PCIe、DMA、网络与 GPU 通信

> 本章写给第一次接触服务器硬件和高性能网络的读者。
> 目标不是记住缩写，而是能回答：**一个 GPU tensor 从机器 A 到机器 B，
> 中间究竟经过哪些硬件与软件？**

---

## 1. 先画出一台服务器

一台多 GPU 服务器可以先粗略画成：

```text
                   ┌──────── CPU socket 0 ────────┐
                   │  CPU cores + memory controller│
                   └───────────┬───────────────────┘
                               │ PCIe Root Complex
                         ┌─────┴─────┐
                         │PCIe Switch│
                         └─┬──┬──┬───┘
                           │  │  │
                         GPU0 GPU1 NIC0

                   ┌──────── CPU socket 1 ────────┐
                   │  CPU cores + memory controller│
                   └───────────┬───────────────────┘
                               │ PCIe Root Complex
                         ┌─────┴─────┐
                         │PCIe Switch│
                         └─┬──┬──┬───┘
                           │  │  │
                         GPU2 GPU3 NIC1
```

先记住四类对象：

- CPU：执行普通程序和控制逻辑。
- DRAM：常说的“主存/内存条”。
- GPU：有自己的 HBM/显存，执行大量并行计算。
- NIC/RNIC：网卡；支持 RDMA 的网卡常叫 RNIC 或 HCA。

这些设备需要一条本机互连。PCIe 就是最常见的设备互连之一。

---

## 2. 带宽、延迟和吞吐不是同一个词

### 2.1 带宽：单位时间最多能搬多少

例如“单方向 32 GB/s”表示理论上 1 秒最多传约 32 GB。

传 1 GiB 的理想时间：

```text
time = data / bandwidth
     = 1 GiB / 32 GiB/s
     ≈ 31.25 ms
```

真实时间会更长，因为还有协议头、排队、软件开销和硬件利用率。

### 2.2 延迟：一次操作从开始到可见要多久

即使消息只有 1 Byte，也要经历：

```text
准备描述符 → doorbell → 设备取任务 → 链路传输 → 对端处理 → completion
```

因此可以用一个简化模型：

```text
总时间 ≈ 固定延迟 + 数据量 / 有效带宽
```

- 小消息：固定延迟占比大。
- 大消息：数据量/带宽占比大。

### 2.3 吞吐：系统实际完成多少工作

吞吐可以是 GB/s，也可以是 requests/s、tokens/s。高链路带宽不保证高业务吞吐：
GPU kernel、锁、队列深度、负载不均衡都可能成为瓶颈。

---

## 3. PCIe 是什么

PCI Express（PCIe）是一种高速、串行、点到点互连。GPU、网卡、NVMe SSD
常通过 PCIe 接入 CPU 与系统。

“点到点”不表示整台机器只能连接两个设备，而是每条 link 的两端是明确的。
更多设备通过 Root Port 和 PCIe Switch 组成树。

### 3.1 lane：一条可同时收发的最小通道

一条 PCIe lane 包含两个方向：

```text
设备 A ──TX────────> RX── 设备 B
设备 A ──RX<────────TX── 设备 B
```

两个方向可同时工作，叫 full-duplex（全双工）。

### 3.2 x1、x4、x8、x16 中的 x 是什么

`xN` 表示一条 link 聚合了 N 条 lanes：

| 宽度 | lane 数 | 常见设备 |
|---|---:|---|
| x1 | 1 | 低速网卡、扩展卡 |
| x2 | 2 | 某些嵌入式设备 |
| x4 | 4 | NVMe SSD、部分网卡 |
| x8 | 8 | 高速网卡、加速卡 |
| x16 | 16 | GPU、高端加速卡 |

在代际相同、其他条件相同时，x16 的原始带宽约是 x8 的 2 倍、x4 的 4 倍。
它类似把一条高速公路从 4 车道扩成 16 车道。

### 3.3 PCIe Gen 表示每条 lane 的速率代际

PCIe 3.0/4.0/5.0 的 `3/4/5` 是协议代际，不是 lane 数。必须把
**代际 × 宽度** 一起说：

```text
PCIe 4.0 x16
```

表示每条 lane 运行 PCIe 4.0 速率，共 16 lanes。

对于 Gen3～Gen5 的 128b/130b 编码，单方向理论有效数据率可近似：

```text
GB/s ≈ GT/s × 128/130 × lane 数 ÷ 8
```

| 代际 | 每 lane 传输率 | x1 单向约值 | x8 单向约值 | x16 单向约值 |
|---|---:|---:|---:|---:|
| Gen3 | 8 GT/s | 0.985 GB/s | 7.88 GB/s | 15.75 GB/s |
| Gen4 | 16 GT/s | 1.969 GB/s | 15.75 GB/s | 31.51 GB/s |
| Gen5 | 32 GT/s | 3.938 GB/s | 31.51 GB/s | 63.02 GB/s |

“双向带宽”会把两个方向相加。例如 PCIe 4.0 x16 常宣传约 64 GB/s 双向，
但一次从 GPU 发往 NIC 的拷贝只使用一个方向，不能把 64 当成单向 64。

### 3.4 GT/s 为什么不是 Gb/s

`GT/s` 是每秒多少次传输（transfer），还要考虑编码方式。Gen1/Gen2 使用
8b/10b：每传 8 bit 有效数据，要在线路上编码成 10 bit，因此有效率 80%。
Gen3～Gen5 使用 128b/130b，有效率约 98.46%。

这就是为什么不能简单地把 `16 GT/s` 写成 `16 Gb/s 有效数据`。

### 3.5 插槽长得像 x16，不代表电气上一定是 x16

主板可能提供一个物理 x16 长度插槽，但只接了 8 或 4 条 lane。Linux 中应查看：

```bash
sudo lspci -vv -s <BDF>
```

关注：

```text
LnkCap: Speed 32GT/s, Width x16   # 最大能力
LnkSta: Speed 16GT/s, Width x8    # 当前协商结果
```

当前只有 Gen4 x8 时，链路上限按 Gen4 x8 估算，而不是按设备铭牌 Gen5 x16。

---

## 4. Root Complex、Switch 与“路径”

### 4.1 Root Complex

Root Complex 是 CPU/内存系统连接 PCIe 树的根。设备发往主存的事务通常要沿
PCIe 树上行到 Root Complex。

### 4.2 PCIe Switch

PCIe Switch 类似网络交换机，有一个或多个上行端口和多个下行端口：

```text
CPU/Root
   │ x16 upstream
PCIe Switch
 ├─ GPU0 x16
 ├─ GPU1 x16
 └─ NIC0 x16
```

下游每个口都是 x16，不代表它们能同时各自向 CPU 跑满 x16。若共享的上行只有
x16，多个设备同时通信会争用这个上行。

### 4.3 Bifurcation（拆分）

CPU 提供的一组 16 lanes 可以被拆成：

```text
x16
或 x8 + x8
或 x4 + x4 + x4 + x4
```

这叫 bifurcation。它改变 lane 的分配方式，不会凭空增加总 lane 数。

### 4.4 为什么 GPU 与 NIC 的相对位置重要

若 GPU0 与 NIC0 在同一 PCIe switch 下，peer-to-peer 事务可能只经过该 switch。
若它们跨 Root Complex 或跨 CPU socket，路径更长，甚至可能不支持直接 P2P：

```text
GPU0 → PCIe switch → Root Complex 0
     → CPU interconnect
     → Root Complex 1 → PCIe switch → NIC1
```

所以“都在同一台机器”不能保证路径相同。

---

## 5. 地址：同一块数据为什么会有多种地址

CPU 程序看到的是虚拟地址。设备发 DMA 时使用 I/O 虚拟地址或总线地址。远端
RDMA peer 还需要 `raddr/rkey`。

可以先建立分层概念：

```text
CPU virtual address
    ↓ 页表
physical memory / GPU memory
    ↓ IOMMU 或 DMA mapping
device-visible I/O address
    ↓ RDMA MR 注册
lkey/rkey + addr
```

不要因为日志里都打印成十六进制，就假设这些地址可以互换。

---

## 6. DMA：谁在搬数据

普通 `memcpy` 由 CPU 执行 load/store。DMA 让设备自己的 DMA engine 搬数据：

```text
CPU:
  ① 准备源地址、目标地址、长度
  ② 通知设备

DMA engine:
  ③ 真正搬运数据

CPU:
  ④ 通过中断或轮询得知完成
```

“不经过 CPU”通常是指 payload 不由 CPU 一字节一字节复制；CPU 仍参与初始化、
提交、连接管理与错误处理。

### 6.1 为什么内存常要 pin

操作系统可以换出或移动普通用户页。DMA 期间若物理页突然变化，设备就会写错
地方。pin memory 会让相关页在操作期间保持稳定，并建立设备可用的映射。

### 6.2 CPU bounce buffer

如果 NIC 不能直接访问 GPU memory：

```text
GPU HBM → pinned host memory → NIC → 网络
```

回程：

```text
网络 → NIC → pinned host memory → GPU HBM
```

中间的 host buffer 就像“中转仓库”，多一次 D2H/H2D 拷贝并占用 PCIe 带宽。

### 6.3 GPUDirect RDMA

若平台、GPU、NIC、驱动和拓扑都支持，RNIC 可以直接 DMA GPU memory：

```text
GPU HBM ↔ RNIC ↔ 网络
```

“direct”不等于数据不走 PCIe；恰恰是 RNIC 通过 PCIe peer-to-peer 能力直接
访问 GPU memory，省掉 host bounce buffer。

---

## 7. 网络最小知识

### 7.1 NIC、switch、packet

- NIC：服务器的网卡。
- switch：连接多台服务器/网卡并转发 packet。
- packet：网络上的数据包。
- queue：临时排队等待发送或处理的 packet。

当很多发送端同时打向一个接收端，接收端口来不及发，队列就会增长，这叫
incast 的典型场景。

### 7.2 400 Gb/s 网卡与 PCIe 的配平

400 Gb/s 的线速约为：

```text
400 ÷ 8 = 50 GB/s
```

如果网卡只有 PCIe 4.0 x8，单方向理论约 15.75 GB/s，PCIe 会先成为明显上限。
PCIe 5.0 x16 理论约 63 GB/s，才有机会承载 400 Gb/s 线速，还要扣除协议和实现
开销。

这只是单端配平。GPU 到 NIC 路径中的共享 switch upstream、NUMA 和内存系统也
可能更早成为瓶颈。

---

## 8. RDMA：把“网络”变成远端内存操作

传统 socket 常经过内核网络栈，并由 CPU 参与拷贝与协议处理。RDMA 允许应用：

- 注册 memory region（MR）；
- 创建 queue pair（QP）；
- 把 work request（WR）放入发送队列；
- 由 RNIC 执行 SEND、WRITE、READ 等操作；
- 从 completion queue（CQ）获得完成结果。

最小心智模型：

```text
应用写任务单 WR
   ↓
发送队列 SQ
   ↓
RNIC 执行网络与 DMA
   ↓
完成通知 CQE
   ↓
应用 poll CQ
```

RDMA fast path 常能绕过每次系统调用，但建立 QP、注册 MR 等 slow path 仍需要
内核和驱动。

---

## 9. NCCL、DeepEP、Barex、blade-kvt 分别解决什么

| 组件 | 主要输入 | 主要抽象 | 典型用途 |
|---|---|---|---|
| NCCL | 多个 GPU rank 的 tensor | collective / P2P | AllReduce、AllGather、AllToAll |
| DeepEP | token、top-k 路由 | dispatch/combine | MoE Expert Parallelism |
| Barex | buffer、地址、channel | Send/WriteBatch | 通用高性能传输 |
| blade-kvt | KV cache blocks、请求元数据 | KV 发送任务 | PD 分离中的 KV cache 传输 |

它们可以使用相似的底层资源，但 API 与语义不同。不能因为都使用 RDMA/NVLink，
就认为 Barex 是 NCCL 的一个函数。

---

## 10. 一次跨机 GPU 数据传输的全路径

以 direct RDMA 为例：

```text
发送端应用
  ① 得到 GPU tensor 的地址和长度
  ② 注册/查询 GPU MR，得到本地设备可用映射
  ③ 从控制面获得远端 raddr/rkey
  ④ 构造 WR/SGE，写入 SQ，敲 doorbell

发送端 RNIC
  ⑤ 通过 PCIe 从 GPU HBM 读取 payload
  ⑥ 切成网络 packet 并发往交换机

网络
  ⑦ 交换机排队、转发；拥塞时 ECN/PFC/重传可能介入

接收端 RNIC
  ⑧ 校验 rkey/地址
  ⑨ 通过 PCIe 把数据 DMA 到接收端 GPU HBM
  ⑩ 生成 completion 或通知

应用
  ⑪ 等待 CQ/future/event
  ⑫ 确认完成后才复用源 buffer 或消费目标 buffer
```

遇到性能问题时，应逐层问：

1. GPU kernel 是否准备好数据？
2. GPU-NIC PCIe path 是否降速或共享上行？
3. MR/地址是否合法？
4. WR 是否真正提交？
5. 网络是否拥塞/重传？
6. CQ 是否被及时 poll？
7. 业务是否等待了比“本地完成”更强的远端完成？

---

## 11. 初学者实验

### 11.1 查看 PCIe 设备树

```bash
lspci -t
lspci -nn
```

### 11.2 查看设备链路能力与当前协商

```bash
sudo lspci -vv -s <BDF> | grep -E 'LnkCap|LnkSta'
```

### 11.3 查看 GPU/NIC 相对拓扑

```bash
nvidia-smi topo -m
```

### 11.4 查看 NUMA

```bash
lscpu
numactl --hardware
cat /sys/bus/pci/devices/0000:<BDF>/numa_node
```

### 11.5 查看 RDMA 对象

```bash
rdma link
ibv_devinfo
```

先记录输出，不要一上来修改 BIOS、IOMMU、ACS、PFC 或网卡参数。

---

## 12. 自检

1. `PCIe 5.0 x16` 中的 `5.0` 与 `x16` 分别表示什么？
2. 为什么 x16 物理插槽可能只运行在 x8？
3. 为什么双向 64 GB/s 不能当作单向 64 GB/s？
4. PCIe switch 下三个 x16 设备为什么可能共享一个 x16 上行？
5. DMA 中 CPU 是否完全没有参与？
6. pinned memory 解决什么问题？
7. GPUDirect RDMA 为什么仍与 PCIe 拓扑有关？
8. `400 Gb/s` 为什么约等于 `50 GB/s`？
9. WR、SQ、CQ 分别像什么？
10. NCCL、Barex 与 blade-kvt 为什么不能画等号？

下一章：[01_pcie_fundamentals.md](01_pcie_fundamentals.md)

## 一手资料

- [PCI-SIG：PCIe Links 基础（x1/x2/x4/x8/x16 与各代编码）](https://pcisig.com/sites/default/files/files/PCI-SIG%20Cabling%20Webinar_FINAL.pdf)
- [PCI-SIG：PCIe 4.0/5.0 每 lane 与 x16 带宽表](https://pcisig.com/blog/pci-express-delivering-needed-bandwidth-open-compute-project)
- [Linux PCI Support Library](https://docs.kernel.org/driver-api/pci/pci.html)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Linux Userspace Verbs Access](https://docs.kernel.org/infiniband/user_verbs.html)
