---
title: "02. SM、CUDA Core、Tensor Core、LSU 与 SFU：GPU 到底靠什么执行"
date: 2026-07-29
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# 02. SM、CUDA Core、Tensor Core、LSU 与 SFU：GPU 到底靠什么执行

## 1. 先看一个 SM 的概念图

![NVIDIA SM 概念结构](/imgs/cuda-cute-sm-anatomy.svg)

为了理解 kernel 性能，可以把 SM 分为六类资源：

```text
调度与依赖
  → warp scheduler、dispatch、scoreboard

状态存储
  → register file

普通计算
  → FP32/INT、FP64 等执行流水线（常被概括为 CUDA Core）

矩阵计算
  → Tensor Core

访存与特殊函数
  → LSU、SFU

片上数据与同步
  → shared memory/L1、barrier、async copy/TMA 接口
```

实际芯片比图复杂，并且不同架构会改变分区、数量和吞吐。概念图用于建立职责，
不能当作晶体管级 floorplan。

### 1.1 分区结构：谁独占、谁共享

上面那六类资源不是平铺在 SM 里的。**一个 SM 被切成 4 个 processing block
（也叫 partition / SMSP）**，有些资源每个 block 独占一份，有些是 4 个 block 共享。

![SM 的分区结构与实测吞吐比例](/imgs/cuda-cute-sm-partition-ratios.svg)

**每个 processing block 独占：**

| 资源 | 数量（Hopper/Blackwell 数据中心 SM） |
|---|---|
| warp scheduler + dispatch | 1 套 |
| 寄存器切片 | 16384 个 32-bit 寄存器 = 64 KB |
| FP32 lane | 32 |
| INT32 lane | 16 |
| FP64 lane | 16 |
| **Tensor Core** | **1** |
| LSU / SFU | 各一组 |
| 可驻留 warp | 最多 16 |

**4 个 block 共享：**

- **L1 数据 Cache / Shared Memory**——同一块物理 SRAM，靠配置划分。H20/GB200 实测
  每 SM 最多 228 KB 可给 SMEM；
- L1 指令 Cache；
- **TMA 引擎**（Hopper 起）——每 SM 一个，不属于任何 partition，且**绕过 LSU**；
- **TMEM**（仅 Blackwell）——Tensor Core 累加器空间；
- barrier / 同步资源。

有两个推论值得单独记住：

1. **寄存器堆是静态切分给 partition 的，不是全 SM 的共享池。** 所以 65536 个寄存器
   不能被单个 partition 的线程全部用掉。这也解释了占用率算术：每 partition 16384 个
   寄存器 ÷ (16 warp × 32 lane) = 每线程 32 个寄存器才能满驻留；用到上限 255 时，
   每 partition 只放得下 2 个 warp。
2. **一个 SM 只有 4 个 Tensor Core，但有 128 个 FP32 lane。** 「Tensor Core 数量少」
   不代表它算力小——见下一节的吞吐比。

### 1.2 实测吞吐比例：只有一个单元变快了

单元个数不是好的心智模型，**每 SM 每周期的吞吐**才是。在 `ecs` 的 H20 和
`target_p` 的 GB200 上实测（复现方法见 §13）：

| 每 SM 每周期 | H20 (SM90) | GB200 (SM100) | 代际变化 |
|---|---|---|---|
| FP32 CUDA Core | 232 FLOP | 233 FLOP | **1.00×** |
| Tensor Core (BF16) | 896 FLOP | 7290 FLOP | **8.1×** |
| Shared Memory 读（经 LSU） | 126 B | 127 B | **1.01×** |

静态资源两卡也完全相同：每 SM 65536 个寄存器（256 KB）、228 KB SMEM 上限、
最多驻留 2048 线程 / 64 warp、每线程 255 个寄存器上限。

**整代升级几乎全部发生在 Tensor Core 上，其余单元每 SM 每周期一点没变。** 于是
Tensor Core 与 FP32 CUDA Core 的吞吐比从 3.9× 拉大到 **31×**。两个直接后果：

- 任何**没走 Tensor Core** 的算子（softmax、layernorm、采样、逐字节拷贝、
  RoPE 的标量部分）换到 GB200 都不会变快，而它们在总时间里的占比会显著上升——
  非 GEMM 部分成为新短板；
- 给 Tensor Core 供数据的通道（SMEM 126 B/cycle）也没变宽，这是
  [11 章](/notes/2026/08/26/2026-08-26-cuda-cute-nvidia-learning-11-gemm-pipeline-deep-dive/) 讨论流水线为什么更难填满的另一个侧面。

## 2. Warp Scheduler

SM 可驻留许多 warp。每个周期，scheduler 从 ready warp 中选择可发射者。

warp 不能发射的常见原因：

- 等待 global/shared memory；
- 等待前序指令写回 register；
- barrier 尚未满足；
- 对应 execution pipeline 忙；
- branch/reconvergence；
- 异步 copy/MMA group 尚未完成。

Scoreboard 维护依赖状态。Nsight Compute 中的 warp stall reason 就是在回答：

```text
为什么 scheduler 这一刻没有发出更多有用指令？
```

## 3. Register File

CUDA thread 的局部标量通常放在 register：

```cpp
float acc = 0.0f;
int idx = ...;
```

register：

- 位于 SM；
- 延迟和带宽很高；
- 按 thread 分配逻辑状态；
- 总容量有限；
- 每 thread 用得越多，能同时驻留的 thread/warp 可能越少。

寄存器不足时，编译器可能 spill 到 local memory。Local memory 常落在 device
memory/L2 路径上，代价可能很高。

查看编译结果：

```bash
nvcc -Xptxas=-v kernel.cu
```

会报告 registers、spill stores/loads、shared memory 等。

## 4. CUDA Core：不要按 CPU Core 理解

CUDA Core 通常指普通算术执行 lane。一个 warp 的标量指令会被分若干周期/分区
送入这些执行资源。

它适合：

- FP32/INT 标量/向量化后的 lane 运算；
- 地址和索引计算；
- activation、elementwise；
- reduction 的部分步骤；
- Tensor Core kernel 的控制、epilogue 与非 MMA 运算。

“GPU 有 X 个 CUDA Core”是峰值能力的一个维度，但实际性能还取决于：

```text
SM 数 × 频率 × 每周期吞吐
× 指令组合
× 活跃 warp
× 数据供应
× pipeline 利用率
```

仅用 CUDA Core 数比较 A100/H100/GB200 的 AI 性能通常没有意义，因为大部分大
模型 GEMM 的主要算力来自 Tensor Core。

## 5. LSU：Load/Store Unit

LSU 负责执行内存访问相关指令：

- global load/store；
- shared load/store；
- local memory；
- atomic；
- 地址转换、请求合并等路径的一部分。

访存性能不仅由 LSU 数量决定，还受：

- warp 地址是否 coalesced；
- cache 命中；
- shared bank conflict；
- HBM/L2 带宽；
- outstanding request 数；
- memory-level parallelism。

### 5.1 LSU 不是「负责某一段搬运」的单元

一个常见误解是把 LSU 理解成「在 global 和 shared 之间搬数据的部件」。它不是。
**LSU 的职责是执行「访存指令」这一类指令**，与地址落在哪个存储层次无关：

```text
LDG / STG   global memory      ┐
LDS / STS   shared memory      ├─ 都由 LSU 执行
LDL / STL   local memory       │
LD  / ST    generic（编译期不确定地址空间）┘
ATOM/RED    原子操作
```

LSU 做的具体事情是：算出每个 lane 的地址、把 32 个 lane 的地址**合并**成尽量少的
cache line 事务、发给下游存储系统、数据回来后写进寄存器。**它不搬 DRAM 数据本身**——
那是 L1/L2 和内存控制器的事。

两个重要的例外，都是为了绕开 LSU：

- **`cp.async`（Ampere）** 仍由 LSU 发起，但数据通路不回寄存器；
- **TMA（Hopper 起）** 基本完全绕开 LSU：单线程发一条描述符指令，专用引擎自己做多维
  地址生成。这正是 TMA 的价值之一——把地址计算的指令发射压力从 LSU 卸掉。

### 5.2 GPU kernel 直接读 pinned host memory：走的是同一个 LSU

#### 先定义 zero-copy

先固定两个词的所指，否则极易读反：**内存 = 主机(CPU)侧的 DRAM，显存 = GPU 板上的
HBM**。「主机显存」这个组合不存在。

**zero-copy** 指 GPU kernel 直接读写**主机内存(CPU DRAM)**，而不先把数据拷进
**显存(GPU HBM)**。前提是数据本来就在主机侧：

```text
常规路径：  主机内存(CPU DRAM) --cudaMemcpy--> 显存(GPU HBM) --kernel 读--> SM
            ↑ 数据起点            ↑ 显式拷贝     ↑ 拷过来的副本

zero-copy： 主机内存(CPU DRAM) ---------kernel 直接 LDG---------------> SM
            ↑ 数据起点            ↑ 跳过「拷进显存」，数据从不进 HBM
```

两个短语是同一件事的两面：**因为不拷进显存，所以只能直接读主机内存。**

这件事反直觉的地方在于，平时的心智模型是「数据必须先搬到显存，GPU 才能算」，而
zero-copy 的全部特殊之处就是打破了这一条——LSU 发出的请求可以被路由出片，跨
PCIe/NVLink 落到主机 DRAM 上。反过来说，如果理解成「kernel 直接读显存」，那是所有
正常 kernel 的常态，不需要专门造一个术语。

用法是分配时加 `Mapped` 标志，再取一个设备可用的指针：

```cpp
void *h;  float4 *d;
cudaHostAlloc(&h, size, cudaHostAllocMapped);   // pinned + 可映射
cudaHostGetDevicePointer(&d, h, 0);             // 同一块物理内存的设备指针
my_kernel<<<...>>>(d);                          // kernel 里直接 d[i]
```

「读」和「写」就是 kernel 里那条指令的方向：`x = d[i]` 是 zero-copy 读（主机 DRAM →
互连 → SM 寄存器），`d[i] = x` 是 zero-copy 写。

**名字有误导性：「zero」指零次显式拷贝，不是零数据搬运。** 数据照样每次都过
PCIe/NVLink。所以判断标准是**复用次数**：

- 数据用一次就丢 → zero-copy 划算，省掉一次落地；
- 数据要反复读 → 亏，每次访问都重新过互连，应该先 memcpy 到 HBM。

主机内存的四种形态，只有第三种能被 kernel 直接解引用：

| 分配方式 | kernel 能直接解引用 | 机制 |
|---|---|---|
| `malloc` / `new` | 不能 | 可换出页，`cudaMemcpy` 时驱动内部还要中转一次 |
| `cudaHostAlloc` / `cudaMallocHost` | 不能 | 页锁定，memcpy 快，但只能 memcpy |
| `cudaHostAlloc(..., Mapped)` | **能** | 页锁定 + 映射进 GPU 地址空间 = zero-copy |
| `cudaMallocManaged` | 能 | Unified Memory：按页**迁移**，首次访问触发 page fault 把页搬进 HBM，之后是本地访问 |

最后两行的区别值得记住：**managed 是「搬过来并留下」，zero-copy 是「每次远程访问，
从不留下」。**

#### 那么它经过 LSU 吗

**是。** 从 SM 的角度看那就是一条普通的 `LDG`——LSU 照常算地址、发请求。区别发生在

**是。** 从 SM 的角度看那就是一条普通的 `LDG`——LSU 照常算地址、发请求。区别发生在
**地址翻译之后**：UVA 发现这个地址不属于显存，请求被路由出片，经 PCIe 或 NVLink-C2C
到主机 DRAM。指令一样，下游差了几十倍。

![LSU 发出的请求走向哪里](/imgs/cuda-cute-lsu-memory-paths.svg)

实测（1 GiB 流式访问，`cudaHostAlloc(cudaHostAllocMapped)`）：

| | H20（x86 + PCIe） | GB200（Grace + NVLink-C2C） |
|---|---|---|
| HBM 流式读 | 3594 GB/s | 7011 GB/s |
| pinned host **zero-copy 读** | **39 GB/s**（HBM 的 1/92） | **211 GB/s**（HBM 的 1/33） |
| pinned host zero-copy 写 | 53 GB/s | 194 GB/s |
| `cudaMemcpyAsync` D2H 对照 | 53 GB/s | 194 GB/s |
| HBM 裸延迟 | 676 cycle / 342 ns | 838 cycle / 407 ns |
| pinned host 裸延迟 | 3366 cycle / 1700 ns（5.0×） | 1255 cycle / 609 ns（1.5×） |

三个可以直接用的结论：

**（1）zero-copy 在 x86 上非常贵，在 Grace 上勉强可用。** 39 GB/s vs 211 GB/s，差
5.4 倍；延迟相对 HBM 的倍数从 5.0× 降到 1.5×。这是 NVLink-C2C 相对 PCIe 的差别，
不是 GPU 本身的差别。

**（2）走 SM（zero-copy）和走 copy engine（`cudaMemcpyAsync`）的上限相同。** 两者都
是 53 GB/s（H20）/ 194 GB/s（GB200）——瓶颈在出片互连，不在选哪种搬运方式。区别在
于 zero-copy 可以**顺便做 gather/scatter 和格式转换**，省掉一次落地；而 memcpy 要求
连续。

**（3）blade-kvt 的 staged 路径就跑在这条路上。**
`kvtransfer/src/copy_kernels.cu` 的 `copy_h2d_direct_kernel` 签名里
`const char* __restrict__ host_src` 是**在 kernel 内直接解引用**的
（`dst[i] = src[i]` 或 `fast_copy_int4`），所以它是 zero-copy 而非 memcpy。
同一份代码在 H20 上被 PCIe 卡在 39~53 GB/s，在 Grace 上能拿到 194~211 GB/s。

### 5.3 向量化宽度往往比路径选择更值钱

同样是 HBM→HBM 拷贝，只改每线程一次搬多少字节：

| | H20 | GB200 |
|---|---|---|
| 逐字节 `dst[i] = src[i]` | 753 GB/s | 1274 GB/s |
| `int4` 向量化（16 B/线程） | 3426 GB/s | 6648 GB/s |
| 差距 | **4.5×** | **5.2×** |

原因回到 §5.1：**LSU 的瓶颈是「每周期能发多少条访存指令」，不是字节数。** 一条
warp 级 `LDG.E.128` 一次拿 32×16 = 512 B；换成逐字节就要 16 条指令才搬同样的量，
指令发射和地址合并的开销放大 16 倍。

这解释了 blade-kvt 里那个看似琐碎的对齐判断：

```cpp
if ((sa & 0xF) == 0 && (da & 0xF) == 0) {
    fast_copy_int4(src, dst, length, tid, block_size);   // 16 B/线程
} else {
    for (int64_t i = tid; i < length; i += block_size)
        dst[i] = src[i];                                  // 1 B/线程
}
```

**这个分支值 4~5 倍。** 落进 else 分支（offset 或 length 未 16 B 对齐）的代价远比
「多几条判断指令」大得多，值得在上层保证对齐。

## 6. SFU：Special Function Unit

SFU 处理某些特殊数学操作，例如具体架构支持的近似/快速：

- reciprocal；
- reciprocal square root；
- sin/cos；
- exponent/log 的组成路径。

高层 `exp`、`tanh`、softmax 不一定全部由单一 SFU 指令完成；编译器、精度选项、
数学库和架构会决定指令序列。

## 7. Tensor Core

### 7.0 MMA 是什么的缩写

**MMA = Matrix Multiply-Accumulate（矩阵乘累加）**，即一次完成：

```text
D = A × B + C
```

三个词逐一对上：**Matrix** 指操作数是矩阵 tile 而不是标量，**Multiply** 是 `A × B`，
**Accumulate** 是 `+ C`——把结果累加到已有的部分和上，而不是覆盖它。累加这一步是
GEMM 沿 K 维分块的前提（见 [11 章 §3.2](/notes/2026/08/26/2026-08-26-cuda-cute-nvidia-learning-11-gemm-pipeline-deep-dive/)）。

它和标量世界的 **FMA（Fused Multiply-Add，`a×b+c`）** 是同一个思想在矩阵层面的
放大：FMA 每条指令处理 1 个数，MMA 每条指令处理一个 tile。

指令名里的字母是数据类型和协作范围：

| 名字 | 层次 | 含义 |
|---|---|---|
| `HMMA` | SASS | **H**alf-precision MMA，warp 级 |
| `IMMA` / `DMMA` | SASS | **I**nt / **D**ouble 版本 |
| `mma.sync` | PTX | warp（32 线程）协作的 MMA |
| `wgmma.mma_async` | PTX | **w**arp**g**roup（4 warp = 128 线程）级异步 MMA，Hopper |
| `tcgen05.mma` | PTX | **t**ensor **c**ore **gen 5**，Blackwell |
| `HGMMA` | SASS | Hopper warpgroup 版本（`G` = warp**G**roup） |

命名里的 `m16n8k16` 这类后缀表示这条指令负责的 tile 形状：M=16、N=8，K 方向规约 16。
注意它描述的是**整个协作组**共同更新的 tile，不是单个线程持有 16×8 的矩阵。

### 7.1 Tensor Core 的关键约束

关键点：

1. 计算的是 tile，不是单个标量；
2. 输入与 accumulator 类型可能不同；
3. 多个 thread 协作提供 operand fragment/descriptor；
4. layout、对齐和指令 shape 受到架构约束；
5. MMA 通常异步或带 group completion 语义；
6. Tensor Core 峰值只有在数据持续供应时才有意义。

### 7.2 从 thread 到 collective

不同层次：

```text
WMMA API
  → warp collective

PTX mma.sync
  → warp-level MMA

Hopper wgmma.mma_async
  → warpgroup（4 warps = 128 threads）级异步 MMA

Blackwell tcgen05.mma
  → 第五代 Tensor Core 指令
  → 可单 thread issue，支持 CTA/CTA-pair 协作与 TMEM accumulator
```

“只有一个 thread issue”不等于只有一个 thread 做了全部计算。它描述指令发起
协议，底层 Tensor Core 仍是宽矩阵计算硬件，数据与同步由 collective 约定管理。

## 8. CUDA Core 与 Tensor Core 如何协作

以 GEMM kernel 为例：

```text
普通流水线/CUDA Core
  ├─ 计算 tile 坐标
  ├─ 边界判断
  ├─ 地址与 descriptor
  ├─ pipeline 状态
  └─ epilogue：bias/activation/store

LSU / copy engine / TMA
  └─ GMEM ↔ SMEM 搬数据

Tensor Core
  └─ 大量 MMA
```

因此 Tensor Core kernel 不是“只有 Tensor Core 在工作”。好的 kernel 要让：

```text
搬运、同步、MMA、epilogue
```

形成重叠流水。

## 9. Tensor Core 是否自动启用

高级库可能自动选择 Tensor Core algorithm，例如 cuBLASLt/PyTorch。条件包括：

- GPU 支持；
- dtype 支持；
- shape 与 leading dimension；
- alignment；
- math mode/precision policy；
- 库实现与 workspace；
- deterministic 等约束。

验证方法：

1. 用 Nsight Compute 查看 tensor pipe 指标；
2. 反汇编寻找对应 SASS；
3. 查看 PTX 中 `mma`、`wgmma`、`tcgen05`；
4. 对比禁用/改变精度后的性能；
5. 不要只根据 API 名称推断。

## 10. 指令层次：CUDA C++、PTX、SASS

```text
CUDA C++ / CuTe DSL
  ↓ 编译与 lowering
PTX：虚拟 ISA
  ↓ ptxas / driver JIT
SASS：具体 GPU 机器指令
```

PTX 不是最终硬件指令。相同 PTX 在不同 GPU/工具链上可生成不同 SASS。

常用工具：

```bash
nvcc -ptx kernel.cu
cuobjdump --dump-sass a.out
nvdisasm kernel.cubin
```

## 11. 架构快速对照

| 架构 | 数据中心代表 | Tensor Core 主线 | 搬运主线 |
|---|---|---|---|
| Ampere | A100 | 第三代，`mma.sync`，TF32/BF16/FP64 TC | `cp.async` GMEM→SMEM |
| Hopper | H100/H20 | 第四代，FP8，`wgmma.mma_async` | TMA + mbarrier |
| Blackwell | B100/B200/GB200 | 第五代，FP4/FP6/FP8，`tcgen05`，TMEM | TMA + TMEM/新 pipeline |

不同产品启用的 SM、频率、HBM 与 Tensor Core 吞吐不同。

## 12. 自检

1. MMA 是什么的缩写？「Accumulate」这一步为什么对分块 GEMM 是必需的？
2. 一个 SM 有几个 Tensor Core、几个 FP32 lane？为什么「Tensor Core 数量少」不代表
   它算力小？
3. 寄存器堆是 4 个 partition 共享还是各自独占？这对占用率计算有什么影响？
4. Shared Memory 和 L1 是两块存储还是一块？TMA 引擎属于哪个 partition？
5. H20 → GB200，哪些单元的每 SM 每周期吞吐变了、哪些没变？这对 softmax 这类算子
   意味着什么？
6. LSU 是「global 和 shared 之间的搬运单元」吗？它到底执行哪些指令？
7. kernel 里直接读一个 pinned host 指针，指令上和读显存有区别吗？代价上呢？
8. 为什么 `int4` 向量化拷贝比逐字节快 4~5 倍？瓶颈是字节数还是指令数？

## 13. 复现

本章的实测数字来自两个微基准：

```text
ecs      : ~/nvfix/sm_probe.cu    ~/nvfix/smem_probe.cu
target_p : /tmp/sm_probe.cu       /tmp/smem_probe.cu
```

```bash
# H20 (ecs)：NVML/libcuda 默认版本与内核模块不匹配，需指向 570.133.20
nvcc -O3 -arch=sm_90  sm_probe.cu -o sm_probe
LD_LIBRARY_PATH=$HOME/nvfix CUDA_VISIBLE_DEVICES=0 ./sm_probe

# GB200 (target_p)，CUDA 13.2
nvcc -O3 -arch=sm_100 sm_probe.cu -o sm_probe
CUDA_VISIBLE_DEVICES=0 ./sm_probe
```

`sm_probe` 输出静态资源、FP32/SMEM 吞吐、三种内存的流式带宽、两种拷贝分支和裸延迟；
`smem_probe` 专门扫 SMEM 带宽。

一个踩过的坑：测 Shared Memory 带宽时，如果循环里带取模或复杂地址计算，会测出
**32 B/cycle/SM**（约为真实值的 1/4）——瓶颈落在整数运算而非 SMEM。改成每轮 8 条
独立 `float4` load、地址用异或和掩码计算后，稳定在 126~127 B/cycle/SM。**测吞吐时
要先确认限速的是被测单元。**

## 14. 官方资料

- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
