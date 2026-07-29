---
title: "00. 零基础导读：CPU、内存、PCIe、DMA、网络与 GPU 通信"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

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

### 1.1 CPU DRAM、GPU DRAM 和 CUDA global memory

NVIDIA 文档所说的：

```text
The host CPU has attached DRAM,
and every GPU in a system has its own attached DRAM.
```

是在强调一台异构机器里有**多块物理内存**，而不是所有处理器共享同一组内存芯片。

#### CPU attached DRAM 是不是我们平时说的内存

是。它通常就是操作系统和普通程序所说的：

- 系统内存（system memory）；
- 主存（main memory）；
- host memory；
- 日常口语中的“内存”或“内存条”。

x86 服务器上常见的物理器件是 DDR4/DDR5 DIMM；Grace CPU 使用板载
LPDDR5X。它们的具体封装不同，但都属于 DRAM。CPU core 访问它们的大致路径是：

```text
CPU load/store
  → CPU cache
  → CPU internal interconnect
  → CPU integrated memory controller
  → DDR/LPDDR DRAM
```

这里通常不经过 PCIe。

#### GPU attached DRAM 是什么

它就是 GPU 自己直接连接的大容量片外内存，常被叫作：

- GPU memory；
- device memory；
- 显存或 VRAM；
- 在数据中心 GPU 上通常是 HBM/HBM2e/HBM3/HBM3e；
- 在一些消费级 GPU 上可能是 GDDR6/GDDR6X。

HBM 和 GDDR 都是 DRAM 技术。GPU 里还有 register、shared memory、L1/L2 cache，
但这些主要由片上 SRAM 实现，不是这里所说的 attached DRAM。

GPU SM 访问自己 HBM 的典型路径是：

```text
GPU thread 的 global load/store
  → GPU L1/L2 cache
  → GPU internal interconnect
  → GPU HBM memory controller
  → HBM DRAM
```

这同样不需要先经过 PCIe。只有数据要去 CPU memory、另一块不能通过 NVLink
直达的 GPU、NIC 等外部位置时，才需要 CPU-GPU/设备互连。

#### GPU DRAM 就等于 global memory 吗

对初学阶段，可以先记成：

```text
cudaMalloc() 得到的 device allocation
        ↓ 通常物理驻留
当前 GPU 的 HBM/GDDR
        ↓ 在 CUDA 程序中
以 global memory 的方式被 kernel 访问
```

但严格说，二者属于不同层次：

| 词 | 描述的层次 | 回答的问题 |
|---|---|---|
| DRAM/HBM/GDDR | 物理存储器件 | bit 最终存在哪种芯片里 |
| GPU device memory | 物理归属/分配位置 | allocation 当前属于哪块 GPU |
| CUDA global memory | CUDA 编程模型中的 memory space | 哪些线程可通过 global load/store 访问 |

因此，“GPU attached DRAM 是 global memory 的主要物理承载”比“二者完全等同”
更准确。几个反例能说明为什么不能画绝对等号：

1. mapped pinned host memory 物理上仍在 CPU DRAM，但 GPU kernel 可以通过
   CPU-GPU interconnect 对它发起访问；
2. `cudaMallocManaged()` 的 managed page 可能在 CPU DRAM 和 GPU DRAM 之间迁移；
3. Unified Virtual Addressing 统一的是**虚拟地址空间**，不会把多块物理 DRAM
   自动焊成一块；
4. 多 GPU 机器中，GPU0 和 GPU1 通常各有自己的 HBM。GPU0 上的 `cudaMalloc`
   allocation 不会自动出现在 GPU1 的 HBM 中。

例如：

```cpp
cudaSetDevice(0);
float* p0;
cudaMalloc(&p0, nbytes);  // 通常由 GPU0 的 HBM 物理承载

cudaSetDevice(1);
float* p1;
cudaMalloc(&p1, nbytes);  // 通常由 GPU1 的 HBM 物理承载
```

虽然 `p0`、`p1` 都处在进程的 unified virtual address space 中，但它们属于不同
GPU 的物理内存。GPU1 要访问 `p0`，还需要 peer access、NVLink/PCIe P2P 等条件。

Grace Blackwell 的 ATS、硬件一致性和 NVLink-C2C 让 CPU/GPU 互访更方便，但
“地址可访问/保持一致”仍不等于“物理位置与访问带宽都相同”。分析性能时始终要问：

```text
allocation 现在物理驻留在哪里？
访问它的 processor 是谁？
两者之间经过哪条 interconnect？
```

### 1.2 Memory Controller 到底是什么

Memory Controller（内存控制器）是**处理器与其直接连接的 DRAM 之间的硬件控制器**。
它不像 CPU core 那样执行 C++ 代码，也不像 DMA engine 那样接收一个“把 A 搬到 B”
的高级任务；它处在每一次 DRAM 访问的末端，把来自 CPU core、GPU SM、DMA engine
等发起者的内存请求，转换成 DRAM 芯片能够理解的时序和命令。

它通常负责：

1. **地址解析与映射**：决定某个物理地址落在哪个 memory channel、DIMM、
   rank、bank、row、column。
2. **请求排队与调度**：同时来了很多读写请求时，选择先服务哪一个，尽量提高
   row-buffer 命中率，同时避免某一类请求长期饿死。
3. **DRAM 协议与时序**：发出 activate、read、write、precharge 等命令，并满足
   DRAM 对命令间隔的要求。
4. **刷新（refresh）**：DRAM 单元会漏电，必须周期性刷新。
5. **数据完整性**：在支持的平台上完成 ECC 检查/纠错、错误上报等工作。
6. **并行通道管理**：让多个 memory channel、bank 尽量并行工作。
7. **流量控制和 QoS**：在 CPU core、GPU、NIC 等请求者之间仲裁有限的内存带宽。

一台异构服务器里可能同时存在好几种“内存控制器”：

| 名称 | 控制的物理内存 | 通常位于哪里 | 主要由谁决定 |
|---|---|---|---|
| CPU Integrated Memory Controller（IMC） | DDR/LPDDR 系统内存 | CPU/SoC 内部 | CPU/SoC 厂商与平台设计 |
| GPU HBM Memory Controller | GPU HBM | GPU 芯片内部 | GPU 厂商，例如 NVIDIA |
| 其他设备的内存控制器 | NIC/DPU/SSD 自带的 DRAM 或 Flash | 对应设备内部 | 对应设备厂商 |

所以“memory controller 是不是 NVIDIA 的”没有一个统一答案：

- x86 服务器的 DDR memory controller 一般集成在 Intel/AMD CPU 中；
- Grace CPU 的 LPDDR memory controller 属于 NVIDIA Grace CPU；
- Blackwell GPU 的 HBM controller 属于 NVIDIA GPU；
- 它们是不同控制器，不应只用一个方框概括。

还要区分下面四个容易混淆的部件：

| 部件 | 最简职责 | 会不会执行一次完整拷贝任务 |
|---|---|---|
| Memory Controller | 把读写请求落实到 DRAM/HBM | 不负责理解“整个 buffer 拷贝” |
| PCIe Root Complex | 把 CPU/内存系统接到 PCIe 树 | 负责路由 PCIe transaction，不等于 DRAM 控制器 |
| DMA Engine | 根据地址和长度发起一串读写 | 会，是 payload 搬运的主动执行者之一 |
| CPU/GPU Core | 执行 load/store 指令 | 会，例如 CPU `memcpy` 或 GPU copy kernel |

### 1.3 用 blade-kvt 的 copy kernel 理解“谁在执行”

`blade-kvt/kvtransfer/src/copy_kernels.cu` 中的
`copy_h2d_direct_kernel()` 和 `copy_d2h_direct_kernel()` 是 **GPU SM 执行的
CUDA kernel**。以 H2D 为例，源码的核心行为可以简化成：

```cpp
// src 指向 CPU pinned memory；dst 指向 GPU HBM
dv[lane_id] = sv[lane_id];
```

实际源码会在对齐时用 `int4` 做 16 Byte 向量化访问，不对齐时再退化为更小的
load/store。`copy_kernels.cpp` 中的 `copy_handle_data_kernel_direct()` 则负责构造
offset/length 元数据并 launch kernel。用于 staging 的 host buffer 由
`cudaMallocHost()` 分配，因此它既被 page-lock，又可以映射到 GPU 地址空间。

这条 H2D 路径应拆成：

```text
GPU SM 执行 load(host pinned address)
  → GPU 的地址转换与 CPU-GPU interconnect 接口
  → PCIe（传统独立 GPU）或 NVLink-C2C（Grace Blackwell）
  → CPU/SoC 的 host-memory controller
  → 读取系统 DRAM
  → 数据返回 GPU
  → GPU HBM controller 把 store 写入 HBM
```

D2H 则反过来：

```text
GPU HBM controller 提供源数据
  → GPU SM 发出写 host pinned address 的 store
  → CPU-GPU interconnect
  → CPU/SoC 的 host-memory controller
  → 写入系统 DRAM
```

因此，“copy kernel 通过 memory controller 执行 GPU 访问 CPU pinned memory”
这句话只对了一半：

- **执行 kernel 的是 GPU SM**，不是 memory controller；
- host memory controller 确实会服务最终的系统 DRAM 读写；
- GPU HBM controller 会服务 HBM 一端；
- 中间究竟走 PCIe 还是 NVLink-C2C，由机器的 CPU-GPU 互连拓扑决定；
- pinned 只保证页面稳定并建立设备可访问映射，它不会让远端内存突然拥有 HBM
  一样的延迟和带宽。

NVIDIA CUDA 文档也明确说明：GPU kernel 访问 mapped page-locked host memory 时，
transaction 要跨 CPU-GPU interconnect，在不同平台上可能是 PCIe 或 NVLink-C2C。

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

这里的“固定”是**简化模型中的启动成本**，意思是它不会随着 payload 字节数线性
增加，并不是每次都精确相等。排队、缓存命中、操作系统调度、重传都会让它抖动。

#### 六个阶段各自在做什么

先给出这一段会使用的四个 verbs 缩写；详细结构和代码放在
[02b RDMA 操作与完成](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-02b-rdma-operations-completion-and-reliability/)：

```text
WR  = Work Request：应用提交的一次软件工作请求
WQE = Work Queue Element：provider 编码后放入设备工作队列的元素
SGE = Scatter/Gather Element：描述一段 local memory 的 addr/length/lkey
CQE = Completion Queue Element：RNIC 写回完成队列的结果记录
```

| 阶段 | 典型工作 | 主要执行者 | 常见可变因素 |
|---|---|---|---|
| 准备描述符 | 生成 WR/WQE/SGE，写地址、长度、lkey/rkey、opcode、flags | 应用、通信库、驱动 | 是否分配内存、是否批量、锁竞争 |
| doorbell | 用 MMIO write 或 doorbell record 告诉设备“队列尾部更新了” | CPU，或特定 GPU-initiated 机制 | 写合并、内存屏障、PCIe posted write |
| 设备取任务 | RNIC 读取 WQE，检查 QP 状态，调度队列，DMA 读取源数据 | RNIC | WQE cache、QP 数、队列深度、PCIe credit |
| 链路传输 | packetize、加协议头、序列化比特、交换机转发、可靠性处理 | RNIC、PHY、网络交换机 | MTU、拥塞、PFC/ECN、重传、跳数 |
| 对端处理 | 解析包、校验访问权限、重组、地址转换、DMA 写目标内存 | 对端 RNIC；有时还有对端软件 | rkey、目标内存类型、PCIe 拓扑、队列 |
| completion | 生成 CQE/ACK，更新生产者索引，应用 poll 或处理中断 | RNIC、CPU/应用 | CQ moderation、poll 频率、中断调度 |

“准备描述符”不是准备每一个 payload 字节，而是在填一张任务单。例如一个
RDMA WRITE 的 WQE 可以表达：

```text
从本地 [addr, length, lkey]
写到远端 [remote_addr, rkey]
完成后是否产生 CQE
```

doorbell 也不是门铃式地“唤醒整个操作系统”，而通常是一次很短的寄存器/队列
通知。设备看到新的 producer index 后，才知道有多少个新 WQE 可以执行。

#### “一次发送完成”必须先定义终点

“发送时间”至少有五种不同口径：

1. **submit latency**：应用开始构造任务，到 doorbell 提交完成。
2. **local completion**：本地 buffer 已经可以安全复用。
3. **remote memory visibility**：payload 已经到达远端目标内存。
4. **remote notification**：远端 CQE、立即数或通知已经可见。
5. **application completion**：远端应用/kernel 已经消费或处理完数据。

它们的终点逐渐变强，时间也通常逐渐变长。一个本地 send completion 往往不能
自动证明“远端业务代码已经处理完”；必须阅读具体 transport、opcode 和 API 的
completion 语义。

### 2.3 它是不是三级流水线

可以把发送端、链路、接收端看成三级流水线来入门，但真实系统更像多级流水线：

```text
应用/WQE
   ↓
发送 RNIC：取 WQE → DMA 读 → packetize
   ↓
本机 PCIe → 网卡端口 → 网络交换机
   ↓
接收 RNIC：解析 → 校验 → DMA 写
   ↓
CQE/通知 → 接收应用或 GPU kernel
```

假设每个阶段处理一个 packet 的服务时间分别为 `s1, s2, ... sk`。第一个 packet
要等待流水线填满，近似经历：

```text
first_packet_latency ≈ s1 + s2 + ... + sk
```

但是后续 packet 可以与前一个 packet 重叠。当流水线稳定后，理想吞吐受最慢阶段
限制：

```text
packet_interval ≈ max(s1, s2, ... sk)
N 个 packet 的时间 ≈ 流水线填充时间 + (N - 1) × packet_interval
```

因此，传 100 个 packet 通常不是：

```text
100 × (所有阶段时间之和)
```

这正是网络能做到“高带宽但单包仍有微秒级延迟”的原因。窗口、队列深度和 credit
允许多个 packet/WQE 同时 in-flight。

不过，“每个阶段都不阻塞”并不准确。所有硬件队列都是有限的：

- SQ/CQ 满了会产生 backpressure；
- PCIe flow-control credit 用完要等对端归还；
- TCP congestion window/receive window 会限制在途数据；
- RDMA RC 也有 QP 深度、credit、ACK、重试等约束；
- 接收端写入 HBM/DRAM 太慢，也会反向限制发送速率。

更准确的说法是：**阶段之间通常可以并行重叠，但需要队列与流控，拥塞时会阻塞或
反压。**

### 2.4 延迟和带宽究竟怎么测

如果要测应用看到的有效带宽（goodput），应选定清晰的开始和完成边界，然后计算：

```text
goodput = 成功传到目标位置的有效 payload 字节数 / 总经过时间
```

例如发送 100 次、每次 8 MiB，并以远端写完成为终点：

```text
goodput = 100 × 8 MiB / 从第一笔提交到最后一笔完成的时间
```

不能只测“某个 packet 从发送口出来到接收口进去的时间”再叫它带宽：

- `packet_bits / 物理链路速率` 是 serialization time（序列化时间）；
- 发送 timestamp 到接收 timestamp 是 one-way latency，需要两台机器精确对时；
- 一段时间内收到的有效字节数才是吞吐/goodput；
- 网卡宣称的 400 Gb/s 是 line rate，不等于应用一定能得到 50 GB/s payload。

对于一个大消息，常用近似式仍然是：

```text
T(message) ≈ T_startup + message_size / B_bottleneck
```

其中 `B_bottleneck` 不是只看网络线速，而是 GPU HBM、GPU↔NIC PCIe、RNIC、
交换机端口、接收端写入能力等整条路径的最小有效带宽。

### 2.5 吞吐：系统实际完成多少工作

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

这里：

- **TX = Transmit**，发送；
- **RX = Receive**，接收。

TX/RX 总是相对于当前设备说的。A 的 TX 连接 B 的 RX，B 的 TX 连接 A 的 RX：

```text
A.TX ─────────────> B.RX
A.RX <───────────── B.TX
```

所以“看 TX counter”之前要先问“哪一块网卡、哪一个端口的 TX”。在网卡语境里
`TX queue` 是发送队列，`RX queue` 是接收队列。少数其他语境可能把 `Tx` 当作
transaction 的简写，但在网卡/链路图中通常就是 transmit。

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

### 6.1 “CPU 一字节一字节复制”不是字面上的 `for (i++)`

这句话是在强调 **CPU 位于 payload data path**，并不是说现代 CPU 一定执行下面
这种最低效代码：

```cpp
for (size_t i = 0; i < n; ++i) {
    dst[i] = src[i];
}
```

真实的 `memcpy`/内核 copy routine 通常会使用机器字、SIMD 向量、cache line、
prefetch 等方式一次处理很多字节。但无论一次 load/store 处理 8 Byte、16 Byte
还是 64 Byte，只要这些指令由 CPU core 执行，payload 就会消耗：

- CPU execution cycles，因此会反映为 CPU utilization；
- L1/L2/L3 cache 容量与 cache bandwidth；
- host DRAM read/write bandwidth；
- NUMA interconnect bandwidth（若源、目标或执行 core 跨 NUMA）。

#### 普通 TCP `send()` 大致发生什么

最常见、未使用 zerocopy 的 Linux TCP 发送路径可先理解为：

```text
应用 user buffer
  → send()/sendmsg()
  → CPU 把 payload 复制/引用到 kernel socket/skb 所管理的 buffer
  → CPU 执行 TCP/IP 协议处理
  → NIC DMA 读取 host memory
  → NIC 发到网络
```

接收方向通常是：

```text
网络
  → NIC DMA 写 host receive buffer
  → CPU 执行 TCP/IP 协议处理
  → recv() 把数据复制到应用 user buffer
```

TSO/GSO、checksum offload、GRO/LRO 等机制可以把分段、校验或合并工作部分下放给
NIC/内核批处理；`MSG_ZEROCOPY`、`sendfile`、`splice` 等接口还可以避免某些
user→kernel payload copy。但 zerocopy 会引入 page pinning、引用计数和 completion
通知开销，Linux 文档也指出它通常只在较大的 write 上有收益，而且“zerocopy
completion”并不等于数据已经完成网络发送。

#### CPU copy 会不会占 PCIe

要分动作看，不能把整条 TCP 路径回答成一个“会”或“不会”：

| 动作 | 消耗 CPU | 消耗 host DRAM | 消耗 PCIe/设备互连 |
|---|---:|---:|---:|
| CPU 把 user buffer `memcpy` 到另一个 host buffer | 是 | 是 | 通常否 |
| PCIe NIC DMA 读取 host buffer 并发送 | 少量控制面 | 是 | 是，NIC 的 PCIe link |
| GPU HBM 拷到 host pinned buffer | 可能由 GPU SM/copy engine 执行 | 是 | 传统机器走 PCIe；GB200 CPU-GPU 可走 NVLink-C2C |
| RNIC 直接 DMA 读取 GPU HBM | 少量控制面 | 不经过 bounce buffer | 是，典型 GPUDirect RDMA P2P 路径 |

因此，典型的 GPU→staged TCP 发送可能同时消耗：

```text
GPU HBM
  → CPU-GPU interconnect
  → pinned host buffer
  → CPU/TCP 网络栈与 host memory bandwidth
  → NIC PCIe
  → 网络
```

注意 CPU 执行一次 host-to-host `memcpy` 本身不会神奇地生成 PCIe packet；真正
占用 PCIe 的是 GPU/NIC 等 PCIe device 与 host/device memory 之间的 transaction。

### 6.2 为什么内存常要 pin

操作系统可以换出或移动普通用户页。DMA 期间若物理页突然变化，设备就会写错
地方。pin memory 会让相关页在操作期间保持稳定，并建立设备可用的映射。

pin memory 不等于“数据一定不复制”，也不等于“GPU 访问它与访问 HBM 一样快”。
它主要解决物理页面稳定和设备映射问题；实际数据仍要跨 CPU-GPU interconnect，
并最终由 host memory controller 访问系统 DRAM。

### 6.3 CPU bounce buffer

如果 NIC 不能直接访问 GPU memory：

```text
GPU HBM → pinned host memory → NIC → 网络
```

回程：

```text
网络 → NIC → pinned host memory → GPU HBM
```

中间的 host buffer 就像“中转仓库”，多一次 D2H/H2D 拷贝并占用 PCIe 带宽。

在 `blade-kvt` 的 TCP/RDMA staged 路径中，`copy_h2d_direct_kernel()` /
`copy_d2h_direct_kernel()` 用 GPU SM 直接访问 `cudaMallocHost()` 的 pinned
buffer，并负责 HBM 与 host staging buffer 之间的 gather/scatter。随后 TCP 或
staged RDMA 再处理 host buffer 到网络的一段。这里“kernel direct access host
memory”与“RNIC direct access GPU HBM”是两种不同的 direct，不能混为一谈。

### 6.4 GPUDirect RDMA

若平台、GPU、NIC、驱动和拓扑都支持，RNIC 可以直接 DMA GPU memory：

```text
GPU HBM ↔ RNIC ↔ 网络
```

“direct”不等于数据不走 PCIe；恰恰是 RNIC 通过 PCIe peer-to-peer 能力直接
访问 GPU memory，省掉 host bounce buffer。

一个容易记忆的判断方式是：

```text
绕过 CPU core          ≠ 绕过 CPU memory
绕过 CPU host memory   ≠ 绕过 PCIe
使用 RDMA              ≠ buffer 一定在 GPU
使用 GPU buffer        ≠ 一定启用了 GPUDirect RDMA
```

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

### 8.1 描述符、doorbell 和 completion 的具体对应

在 verbs 风格接口中，应用往往先构造 `ibv_send_wr`，其中的 SGE 描述本地
`addr/length/lkey`；若是 RDMA READ/WRITE，还要带远端 `remote_addr/rkey`。
`ibv_post_send()` 把一个或一串 WR 提交到 QP 的 Send Queue。底层 provider
通常把 WQE 写入用户态映射的队列，并通过 doorbell 通知 RNIC，所以 fast path
不需要每个 WR 都陷入内核。

完成后，RNIC 在 Completion Queue 中写入 CQE。应用可以 busy-poll CQ，也可以让
completion channel/中断唤醒线程。busy-poll 延迟通常更低，但会持续占用 CPU core；
中断更省 CPU，但会增加调度和唤醒延迟。

### 8.2 为什么发送端和接收端都可能有 completion

completion 的含义取决于操作：

- **SEND/RECV**：发送端有 send completion，接收端收到匹配的 RECV 后有 receive
  completion。
- **RDMA WRITE**：数据被写进远端注册内存；是否通知远端取决于额外机制，例如
  WRITE_WITH_IMM、另发 SEND，或者应用层协议。
- **RDMA READ**：本地 RNIC 从远端读数据，本地 completion 表示读结果已经返回到
  本地目标 buffer。
- **unsignaled WR**：为了减少 CQE 压力，并非每个 WR 都要求一个 CQE；程序会每隔
  若干个 WR 设置 signaled。

所以画链路时最好把“发送端本地 CQE”“远端内存可见”“接收端收到通知”画成不同
事件，不要用一个模糊的 `completion` 方框代替全部语义。

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
  ⑨ 通过本机 I/O 路径把数据 DMA 到接收端 GPU HBM
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

### 10.1 “RDMA 为什么还通过 PCIe”——RDMA 与 PCIe 不是竞争关系

RDMA 描述的是**跨机器的远端内存访问语义和网络 transport**；PCIe 描述的是
**一台机器内部 CPU、GPU、RNIC 等设备如何互连**。一次传统 GPU Direct RDMA
跨机写入本来就同时使用二者：

```text
机器 A：
GPU HBM
  → GPU PCIe endpoint/BAR
  → 本机 PCIe P2P path
  → RNIC DMA engine

跨机：
RNIC → RoCE/InfiniBand 网络 → RNIC

机器 B：
RNIC DMA engine
  → 本机 PCIe P2P path
  → GPU PCIe endpoint/BAR
  → GPU HBM
```

可以把它类比成“货物先走园区内部道路到机场，再坐飞机跨城，再走对端园区道路”。
飞机没有让园区道路消失；RDMA 网络也没有让 GPU↔RNIC 的本机互连消失。

NVIDIA 对 GPUDirect RDMA 的官方定义就是：GPU 与第三方 peer device 通过 PCIe
标准能力建立直接数据路径，并特别强调两者的 PCIe Root Complex/拓扑会影响支持
与性能。

在新平台上，第 ⑨ 步不应脱离拓扑武断地永远写成 PCIe。例如：

- Grace CPU ↔ Blackwell GPU 的 mapped system-memory access 可以走 NVLink-C2C；
- 同一 NVLink domain 内 GPU↔GPU 数据可以走 NVLink；
- 但典型 ConnectX RNIC ↔ GPU 的 GPUDirect RDMA 仍可能是 PCIe P2P；
- 未来/特定平台可能存在不同的数据直连设计，因此最终要看该机器的拓扑和厂商
  文档。

### 10.2 `target_p` GB200 实测：RNIC↔GPU 仍显示 PCIe 路径

2026-07-27 在 `target_p` 上做了只读拓扑检查和小规模 Barex 跨机测试。机器信息：

```text
target_p: 4 × NVIDIA GB200
RNIC:     4 × mlx5_bond_*，底层为 ConnectX-7，RoCE/Ethernet
peer:     target_d（同样为 4 × NVIDIA GB200）
Barex:    ACCL 1.5.3, commit 372e9383f12
```

`nvidia-smi topo -m` 的关键部分是：

```text
GPU0/1 ↔ NIC0/1: NODE
GPU2/3 ↔ NIC2/3: NODE
GPU0/1 ↔ NIC2/3: SYS
GPU2/3 ↔ NIC0/1: SYS
GPU0 ↔ GPU1/2/3: NV18
```

工具自己的 legend 写得很明确：

```text
NODE = Connection traversing PCIe and PCIe Host Bridges within a NUMA node
SYS  = Connection traversing PCIe plus the interconnect between NUMA nodes
NV#  = Connection traversing a bonded set of # NVLinks
```

这份输出同时证明了两件事：

1. GB200 GPU 之间存在 NVLink 路径，显示为 `NV18`；
2. GPU 与 ConnectX-7 RNIC 之间没有显示为 `NV#`，而显示为包含 PCIe 的
   `NODE/SYS`。

因此，对这台具体机器，第 ⑨ 步写“RNIC 通过本机 PCIe 路径 DMA 到 GPU HBM”是
合理的。GB200 拥有 NVLink-C2C/NVLink，不代表所有外设访问 GPU 都自动改走 NVLink。

### 10.3 Barex 实测：原命令与 GDR 命令测的不是同一条 buffer 路径

先运行 server（端口用独立测试端口，实际 RDMA IP 不写入公开笔记）：

```bash
# target_p
barex_benchmark -p <test-port>
```

用户给出的 client 形态补上 server 地址和显式 `-t 100`：

```bash
# target_d
barex_benchmark -c -s <target_p-rdma-ip> -p <test-port> \
  -b 128 -r 8388608 -n 10 -t 100
```

参数必须按该版本源码理解：

| 参数 | 实际含义 |
|---|---|
| `-b 128` | request payload 是 128 Byte |
| `-r 8388608` | response payload 是 8 MiB |
| `-n 10` | 10 条并发 client connection，不是 10 次 |
| `-t 100` | 总请求计数；不写时默认也是 100 |
| 没有 `-S/-R/-G/-Q` | 默认 Send/Recv，buffer 在 CPU memory |

原命令的跨机结果为：

```text
conns=10, tx=128 B, rx=8 MiB, iters=100
avg=631.83 us, P50=563 us, P99=833 us
rx_BW=370644 Mb/s, tps=5523.03
```

日志确认 `dtype=XDT_RDMA`、创建了 mlx5 QP，但 CPU mempool 有 MR，GPU mempool 的
MR 计数为 0。因此这个结果验证的是 **Barex RDMA + CPU registered memory**，不能
单独用来证明 RNIC 正在 DMA GPU HBM。

为了验证 GPU Direct RDMA，另外使用：

```bash
# -S: RDMA WRITE single
# -Q: server 源 tensor 在 GPU
# -G: client 目标 tensor 在 GPU
# 不使用 -e/-w，避免显式 GPU↔CPU staging
barex_benchmark -c -s <target_p-rdma-ip> -p <test-port> \
  -b 128 -r 8388608 -n 10 -t 100 -S -Q -G
```

结果为：

```text
conns=10, tx=128 B, rx=8 MiB, iters=100
avg=605.78 us, P50=545 us, P99=810 us
rx_BW=377546 Mb/s, tps=5625.88
```

client 日志同时显示：

```text
is_client_tensor_on_gpu=1
is_server_tensor_on_gpu=1
is_client_copy_data_cpu_to_gpu=0
XGpuMempool ... free_bytes=83886080, mr_count=40
```

`83,886,080 Byte = 10 × 8 MiB`，与 10 个 GPU response buffer 一致；GPU
mempool 中存在注册 MR，而 gpu-host staging mempool 为 0。结合 `-S/-Q/-G` 的
源码语义，这次测试验证了 server GPU buffer 到 client GPU buffer 的 RDMA WRITE
路径。

但要谨慎解释数字：

- `rx_BW` 是 benchmark 按 `response_size × TPS` 计算的应用有效吞吐，不是直接
  读取交换机端口得到的线速；
- 10 条连接在流水线中并发，所以不能用 `1 / avg_latency` 当总 TPS；
- 一次短测只能证明路径可工作并给出当时的样本，不能证明长期稳定极限；
- CPU-buffer 与 GPU-buffer 两组结果接近，不能据此说 PCIe“不存在”；它只说明
  在本次 8 MiB、10 并发配置下，整条路径达到了约 371～378 Gb/s goodput；
- 若要分析单 NIC、单 GPU 的上限，应绑定 GPU/NIC/NUMA，并分别跑 size sweep、
  connection-count sweep、长稳态测试和硬件 counter。

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

### 11.6 做 Barex GPU Direct RDMA 测试前先确认语义

```bash
# 1. 确认编译版本和网卡类型
barex_benchmark -V

# 2. 阅读本机这个版本的参数，不要凭名字猜
barex_benchmark --help

# 3. 两端确认 GPU/NIC 拓扑
nvidia-smi topo -m
ibv_devices

# 4. server
timeout 120s barex_benchmark -p <unused-port>

# 5. 另一台机器上的 client：GPU source → RDMA WRITE → GPU destination
timeout 100s barex_benchmark \
  -c -s <server-rdma-ip> -p <unused-port> \
  -b 128 -r 8388608 -n 10 -t 100 -S -Q -G
```

实验记录至少要包含：

- 日期、两端 hostname、GPU/NIC 型号、Barex commit；
- `nvidia-smi topo -m`；
- RDMA device、GID/link layer、MTU；
- command line 和所有影响 transport 的环境变量；
- request/response size、并发数、迭代数；
- latency percentile、goodput、错误与重传 counter；
- buffer 位于 CPU、GPU，还是显式 staging buffer；
- completion 的测量边界。

不要在同一台机器起 server/client 后就把结果称为“跨机 RDMA 带宽”。本机 loopback
可能采用特殊路径，甚至完全不经过物理网络端口。

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
11. CPU memory controller、GPU HBM controller、DMA engine 各自负责什么？
12. 为什么 copy kernel 的执行者是 GPU SM，而 host DRAM 访问仍会经过 CPU/SoC
    memory controller？
13. TX/RX 分别是什么缩写？A 的 TX 连到 B 的哪一端？
14. 为什么流水线的首包延迟近似是各阶段之和，而稳态吞吐由最慢阶段决定？
15. “local completion”“remote memory visibility”“远端应用消费完成”有什么区别？
16. 为什么 CPU 执行 host-to-host `memcpy` 不直接占 PCIe，而 PCIe NIC DMA 会占？
17. 为什么普通 TCP 仍会消耗 CPU，使用 TSO/GRO 后又减少了哪些 CPU 工作？
18. 在当前 Barex 版本中，`-n 10` 和 `-t 10` 有什么区别？
19. 为什么没有 `-G/-Q` 的 Barex RDMA 测试不能证明 GPU Direct RDMA？
20. 为什么 GB200 的 CPU↔GPU 可走 NVLink-C2C，但 ConnectX RNIC↔GPU 仍可能走
    PCIe？

下一章：[01_pcie_fundamentals.md](/notes/2026/07/27/2026-07-27-nccl-pcie-barex-learning-01-pcie-fundamentals/)

## 一手资料

- [PCI-SIG：PCIe Links 基础（x1/x2/x4/x8/x16 与各代编码）](https://pcisig.com/sites/default/files/files/PCI-SIG%20Cabling%20Webinar_FINAL.pdf)
- [PCI-SIG：PCIe 4.0/5.0 每 lane 与 x16 带宽表](https://pcisig.com/blog/pci-express-delivering-needed-bandwidth-open-compute-project)
- [Linux PCI Support Library](https://docs.kernel.org/driver-api/pci/pci.html)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NVIDIA CUDA Programming Guide：Page-Locked 与 Mapped Host Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#page-locked-host-memory)
- [NVIDIA GB200 NVL72：Grace CPU、Blackwell GPU 与 NVLink-C2C](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA GB200 Multi-Node Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/overview.html)
- [Linux Userspace Verbs Access](https://docs.kernel.org/infiniband/user_verbs.html)
- [Linux MSG_ZEROCOPY](https://docs.kernel.org/networking/msg_zerocopy.html)
