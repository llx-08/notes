---
title: "CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记"
date: 2026-07-29
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记

这是一套从零开始、面向“能读懂并最终写出高性能 GPU kernel”的系列文档。

它把通常分散学习的四条线放到同一张地图中：

```text
CUDA 编程模型
  ├─ grid / block / thread / warp
  ├─ stream / event / asynchronous launch
  └─ global / shared / register / cache

NVIDIA GPU 硬件
  ├─ GPC / TPC / SM
  ├─ warp scheduler / CUDA Core / LSU / SFU
  ├─ Tensor Core / TMA / TMEM
  └─ HBM / L2 / NVLink / PCIe

矩阵乘与高性能 kernel
  ├─ GEMM / MMA
  ├─ tiling / data reuse / pipeline
  ├─ coalescing / bank conflict / occupancy
  └─ Ampere → Hopper → Blackwell

CUTLASS / CuTe / CuTe DSL
  ├─ Layout / Tensor
  ├─ Copy Atom / MMA Atom
  ├─ TiledCopy / TiledMMA
  └─ Python DSL → MLIR → PTX → CUBIN
```

![CUDA、CuTe DSL 与 NVIDIA GPU 学习栈](/imgs/cuda-cute-learning-stack.svg)

## 1. 为什么要同时学习这些内容

只学 CUDA 语法，容易停留在“能运行”的 kernel；只学硬件参数，容易记住大量
数字却不知道如何使用；直接看 CUTLASS/CuTe，又会被 Layout、Atom、TMA、WGMMA、
TMEM 等概念淹没。

正确的依赖顺序是：

```text
线程怎样映射到硬件
  → 数据存在哪里、如何移动
  → 普通 CUDA Core 与 Tensor Core 分别计算什么
  → GEMM 为什么需要分块和流水
  → 不同架构提供了哪些新搬运/计算指令
  → CuTe 如何用统一的 Layout/Tensor/Atom 描述这些工作
```

## 2. 文档目录

| 顺序 | 文档 | 学习目标 |
|---:|---|---|
| 0 | [零基础导读](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-00-beginner-primer/) | 建立 CPU、GPU、kernel、core、memory、architecture 的词汇表 |
| 1 | [CUDA 执行模型](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-01-cuda-execution-model/) | 理解 host/device、grid/block/warp/thread、SM 调度与异步 launch |
| 2 | [SM、CUDA Core 与 Tensor Core](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-02-sm-cuda-core-tensor-core/) | 看懂一个 SM 内有哪些执行与调度部件，避免把 CUDA Core 当 CPU Core |
| 3 | [GPU 内存层次](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-03-memory-hierarchy/) | 理解 HBM、L2、L1/shared、register、local memory、coalescing 与 bank conflict |
| 4 | [GEMM、MMA 与 Tensor Core](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-04-gemm-mma-tensor-core/) | 从矩阵乘公式走到 tile、warp MMA、数值精度和流水线 |
| 5 | [Ampere、Hopper、Blackwell 架构演进](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-05-ampere-hopper-blackwell/) | 系统比较 cp.async、TMA/WGMMA、TMEM/tcgen05 和 NVLink 演进 |
| 6 | [Hopper/H20 实机导读](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-06-hopper-h20-lab/) | 结合 `ecs` 的 8×H20 理解 compute capability 9.0、NVSwitch 与调试方法 |
| 7 | [Blackwell/GB200 实机导读](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-07-blackwell-gb200-lab/) | 结合 `target_p_j` 的 4×GB200、SM100、CUDA 13.2 和 NV18 拓扑 |
| 8 | [CuTe DSL：Layout 与 Tensor](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-08-cute-dsl-layout-tensor/) | 理解 CuTe 最核心的坐标到地址映射及 Python DSL 编译路径 |
| 9 | [CuTe DSL：Copy、MMA 与 GEMM 流水](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-09-cute-dsl-gemm-pipeline/) | 将 Layout、TiledCopy、TiledMMA 与 Hopper/Blackwell 硬件操作连接起来 |
| 10 | [性能分析、实验与代码阅读路线](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-10-profiling-and-practice/) | 使用 nvidia-smi、Nsight、PTX/SASS、roofline 和微基准定位瓶颈 |

## 3. 两台学习机器

本文档在 2026-07-29 做过只读探测：

| 环境 | 实机结果 | 用途 |
|---|---|---|
| `ecs` | 8×NVIDIA H20，GH100/Hopper，compute capability 9.0，约 96 GiB HBM/GPU，GPU 间显示 `NV18` | Hopper、TMA、WGMMA、NVSwitch、H20 实验 |
| `target_p_j` | 4×NVIDIA GB200，Blackwell，compute capability 10.0，约 185 GiB HBM/GPU，CUDA 13.2，CuTe DSL 4.5.2 | Blackwell、TMEM、tcgen05、CuTe DSL 实验 |

注意：

- `ecs` 当前默认 `nvidia-smi` 使用的 NVML 580.173 与内核驱动 570.133.20
  不匹配；探测时显式加载同版本 NVML 570.133.20。这个问题本身是环境问题，
  不是 GPU 架构特征。
- `target_p` 当前网络不通，Blackwell 实验统一使用 `target_p_j`。
- 产品名、芯片名和架构名不是同一个概念：H20 是产品，GH100 是芯片，
  Hopper 是架构；GB200 是 Grace Blackwell 系统/产品命名，SM100 表示本文实机
  GPU 的 Blackwell compute capability 家族。

## 4. 配套代码

目录 `examples/` 提供最小实验：

```text
examples/
├─ cuda/
│  ├─ 01_vector_add.cu
│  └─ 02_naive_matmul.cu
└─ cutedsl/
   ├─ 01_thread_hierarchy.py
   └─ 02_layout_basics.py
```

原则是：

1. 每个实验先预测输出，再运行；
2. 先验证正确性，再看性能；
3. 先观察 CUDA 执行模型，再引入 CuTe 抽象；
4. 架构专属指令必须核对 compute capability 和工具链版本。

## 5. 资料可信度规则

架构与指令细节优先参考：

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [CUTLASS / CuTe Documentation](https://docs.nvidia.com/cutlass/latest/)
- [CuTe DSL Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)

产品峰值、SM 数量和带宽要注明具体 SKU、dense/sparse、是否启用结构化稀疏，
不能把 A100/H100/B200 的某个产品数字泛化成整个架构。
