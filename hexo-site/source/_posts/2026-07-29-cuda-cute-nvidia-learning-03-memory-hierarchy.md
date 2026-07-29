---
title: "03. GPU 内存层次：HBM、L2、Shared Memory、Register 与数据搬运"
date: 2026-07-29
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# 03. GPU 内存层次：HBM、L2、Shared Memory、Register 与数据搬运

## 1. 为什么 GPU 性能经常不是算力问题

假设一次操作：

```text
从 HBM 读取 4 byte
只做一次加法
再写回 4 byte
```

算术很少，数据移动很多，通常 memory-bound。

反之，GEMM 将加载的 A/B tile 反复复用：

```text
加载一次
  → shared/register 中多次参与 MMA
```

可提高 arithmetic intensity：

```text
Arithmetic Intensity = FLOPs / bytes moved from a given memory level
```

GPU 优化很大一部分是在回答：

> 数据应该在哪一级存多久、由谁搬、被复用多少次？

## 2. 整体层次

![GPU 内存层次和数据流](/imgs/cuda-cute-memory-hierarchy.svg)

典型层次：

| 层次 | 作用域 | 特点 |
|---|---|---|
| register | thread | 最接近执行流水线，容量有限 |
| shared memory | block/cluster 扩展 | 软件管理的片上 scratchpad |
| L1/texture | SM | cache；通常与 shared memory 使用统一物理资源/carveout |
| L2 | GPU | 所有 SM 共享，连接 HBM 与 peer/I/O 路径 |
| HBM/global memory | GPU device | 容量大、带宽高、延迟远高于片上存储 |
| host memory | CPU/系统 | 经 PCIe/NVLink-C2C/互连访问 |

“快/慢”必须看访问模式与并行度，不能只记固定延迟数字。

## 3. Register

优点：

- thread 私有；
- 带宽高；
- 编译器直接分配；
- 适合 accumulator、索引、临时值。

代价：

- SM register file 总量有限；
- register pressure 限制 occupancy；
- 动态索引数组可能无法保持在 register；
- spill 进入 local memory。

Tensor Core fragment/accumulator 在不同架构中可能放在 register 或专用空间：

- Ampere/Hopper 许多 accumulator fragment 占用 register；
- Blackwell tcgen05 引入 TMEM，减轻 accumulator 对 register file 的压力。

## 4. Shared Memory

Shared memory 是程序显式管理的片上存储：

```cpp
extern __shared__ float smem[];
```

生命周期：

```text
一个 block 被分配到 SM
  → 为该 block 分配 shared memory
  → block 内 thread 共享
  → block 结束后释放
```

用途：

- 重用 global memory 数据；
- thread 间交换；
- GEMM tile；
- reduction；
- staging/pipeline；
- TMA destination/source。

### 4.1 Shared Memory Bank

Shared memory 被划分为 bank。一个 warp 同时访问时：

- 不同 bank：可并行；
- 同一地址：可利用 broadcast；
- 同一 bank 不同地址：可能 bank conflict，分多次服务。

矩阵转置常见问题：

```text
按列访问一个朴素 row-major shared tile
  → 多个 lane 落入同一 bank
```

解决方法：

- padding，例如 `[32][33]`；
- swizzle；
- 调整 thread/data layout；
- 使用 CuTe 的 layout algebra 显式描述。

## 5. L1、Texture 与 Shared Carveout

现代数据中心架构中，L1/texture/shared 常共享一块统一片上数据资源，通过
carveout 调整偏好。

这不表示：

```text
shared memory 就等于 L1 cache
```

二者编程语义不同：

- shared 由程序明确寻址、同步和管理；
- L1 是 cache，由硬件按访问自动管理；
- 统一的是部分物理容量/数据路径。

静态 shared memory 通常受兼容性限制；要使用更大的 dynamic shared memory，
常需要 `cudaFuncSetAttribute` opt-in。

## 6. L2

L2 是 GPU 全局共享 cache：

- 所有 SM 可访问；
- 缓存 global/local 等流量；
- peer/NVLink/PCIe 数据路径可能经过相关 L2/coherence/translation 逻辑；
- 容量与 residency policy 随架构演进。

命中 L2 不等于免费：仍有互连、queue、partition 和访问并行度成本。

## 7. HBM 与 Global Memory

HBM 是物理堆叠 DRAM。CUDA global memory 是设备全局地址空间。

对 A100/H100/H20/B200 这类数据中心 GPU，可粗略画成：

```text
SM request
  → L1/shared path
  → L2 slice / fabric
  → memory partition/controller
  → HBM stack/channel
```

峰值 HBM bandwidth 要依靠：

- 大量并行 request；
- 合并良好的连续访问；
- 足够 transaction size；
- 均匀访问 memory partition；
- 避免依赖链和过低 occupancy。

## 8. Coalescing

一个 warp 的 global memory 地址会合并为若干 memory transaction。

理想：

```text
lane 0 → a[0]
lane 1 → a[1]
...
lane 31 → a[31]
```

离散：

```text
lane 0 → a[0]
lane 1 → a[1024]
lane 2 → a[2048]
...
```

后者可能产生大量 transaction，浪费带宽。

对齐也重要。向量化 load 如 16-byte 并不自动保证高效，地址必须满足类型与指令
要求。

## 9. Local Memory

Local memory 是每 thread 的逻辑地址空间，但常位于 device memory。

来源：

- register spill；
- 大型局部数组；
- 动态索引导致 compiler 无法寄存器化；
- ABI stack/调用状态。

它可能被 L1/L2 缓存，但延迟与 global path 相近，不能按“名字 local”认为是
片上 scratchpad。

## 10. Constant 与 Texture

Constant memory：

- 只读；
- 有专用 cache；
- warp 访问相同地址时 broadcast 高效；
- 每 lane 访问不同地址时可能序列化。

Texture/read-only 路径：

- 利用空间局部性与专用访问语义；
- 现代编译器/cache 架构下具体收益依访问模式而定。

## 11. 数据搬运演进

### 11.1 传统 thread-driven copy

```cpp
reg = global[idx];
shared[idx] = reg;
```

数据经过 register，thread 负责地址生成与循环。

### 11.2 Ampere `cp.async`

```text
global → shared
```

异步 copy 可避免使用中间 register，并与计算流水重叠。

### 11.3 Hopper TMA

TMA = Tensor Memory Accelerator：

- descriptor 描述 1D～5D tensor；
- 单 thread 可发起大块异步搬运；
- 硬件负责地址生成；
- 支持 GMEM↔SMEM、multicast、部分 reduction；
- 与 mbarrier/pipeline 配合。

### 11.4 Blackwell

Blackwell 延续 TMA，并加入与第五代 Tensor Core、TMEM、block-scaled 数据格式
配套的搬运与同步机制。优化重点从“让每个 thread 搬更多”进一步转向：

```text
让专用数据运动硬件、Tensor Core 与 producer/consumer warp 更独立地并行
```

## 12. Double Buffer / Multi-stage Pipeline

GEMM 主循环：

```text
Stage 0：加载 tile k+1
Stage 1：计算 tile k
Stage 2：写回/推进 barrier
```

双缓冲：

```text
SMEM buffer A：Tensor Core 正在消费
SMEM buffer B：copy/TMA 正在填充
下一轮交换
```

stage 太少无法隐藏延迟；太多消耗 shared/register，降低 occupancy。最佳值与：

- tile shape；
- dtype；
- K 长度；
- TMA/cp.async latency；
- register/shared 容量；
- 架构；
- epilogue

有关。

## 13. `target_p_j` 实机属性如何解读

PyTorch 报告 GB200 GPU 0：

```text
compute capability              10.0
SM count                        152
warp size                       32
max threads/SM                  2048
max threads/block               1024
registers/SM                    65536
shared memory/SM                233472 bytes
default shared memory/block     49152 bytes
memory bus width                7936 bits
```

`shared_memory_per_block=49152` 是默认 block 上限，不代表硬件只能用 48 KiB。
Blackwell tuning guide说明 B200 可支持更大的 dynamic shared memory，但超过兼容
默认值需要显式 opt-in，并且 CUDA 会保留一小部分系统用途。

## 14. 优化检查清单

1. global load/store 是否 coalesced？
2. 实际 DRAM bandwidth 与峰值差多少？
3. L2 hit rate 是否符合数据复用预期？
4. shared 是否 bank conflict？
5. register 是否 spill？
6. shared/register 是否压低 occupancy？
7. 是否有足够 outstanding memory request？
8. copy 与 compute 是否真正 overlap？
9. pipeline stage 是否过多或过少？
10. 是否把一次性数据错误地搬进 shared？

## 15. 官方资料

- [CUDA C++ Programming Guide：Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
- [Ampere Tuning Guide：Memory System](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [Hopper Tuning Guide：Tensor Memory Accelerator](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Blackwell Tuning Guide：Memory System](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
