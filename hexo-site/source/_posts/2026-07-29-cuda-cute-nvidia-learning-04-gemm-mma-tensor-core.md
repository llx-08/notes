---
title: "04. GEMM、MMA 与 Tensor Core：从三重循环到高性能流水线"
date: 2026-07-29
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# 04. GEMM、MMA 与 Tensor Core：从三重循环到高性能流水线

## 1. GEMM 是什么

GEMM = General Matrix Multiply：

```text
D = α × A × B + β × C
```

最简单情况：

```text
A: M × K
B: K × N
C/D: M × N
```

元素公式：

```text
D[m,n] = Σ(k=0..K-1) A[m,k] × B[k,n] + C[m,n]
```

朴素 CUDA 写法让一个 thread 算一个输出元素：

```cpp
for (int k = 0; k < K; ++k) {
  acc += A[row * K + k] * B[k * N + col];
}
```

它正确，但会重复从 global memory 加载 A/B，数据复用很差。

## 2. 为什么要 Tiling

将输出矩阵分成 block tile：

```text
CTA tile: BM × BN
K 方向每次处理 BK
```

每轮：

```text
加载 A_tile[BM, BK] 到 shared
加载 B_tile[BK, BN] 到 shared
同步
用这些数据做 BM × BN × BK 的乘加
同步
进入下一段 K
```

![GEMM 分块与 Tensor Core 数据流](/imgs/cuda-cute-gemm-tiling.svg)

一个 A 元素可被 BN 个输出复用，一个 B 元素可被 BM 个输出复用。

## 3. 多级 Tiling

高性能 GEMM 不只有 CTA tile：

```text
Problem M×N×K
  → Cluster tile
    → CTA tile
      → Warpgroup/warp tile
        → MMA atom/instruction tile
          → 每个 lane 持有 fragment
```

每一级回答：

- 哪组 thread 负责哪块数据？
- 数据放 register、shared 还是 TMEM？
- 谁发起 copy/MMA？
- 如何同步？
- tile 如何映射到物理地址？

CuTe 的 Layout/TiledCopy/TiledMMA 就是把这些映射变成可组合对象。

## 4. MMA 是什么

MMA = Matrix Multiply-Accumulate：

```text
D_tile = A_tile × B_tile + C_tile
```

PTX 指令名会携带 shape/type/layout 信息，例如概念形式：

```text
mma.sync.aligned.m16n8k16...
wgmma.mma_async.m64nNk16...
tcgen05.mma...
```

`m16n8k16` 表示这条 collective 指令更新一个 M=16、N=8 的输出 tile，K 方向
规约 16。它不是说单个 thread 持有完整的 16×8 矩阵。

## 5. Fragment

矩阵 tile 会按指令协议分布到 thread/register/shared/TMEM：

```text
logical A tile
  → lane 0 持有若干元素
  → lane 1 持有若干元素
  → ...

logical accumulator tile
  → 按 lane/register 或 TMEM layout 分布
```

这种分布叫 fragment layout。程序员直接写 PTX 时必须严格遵守；CuTe 用 MMA
Atom/TiledMMA 封装这些架构相关映射。

## 6. Tensor Core 数据类型

三个不同概念：

```text
input type
multiply precision/format
accumulator/output type
```

常见：

| 输入 | 常见 accumulator | 用途 |
|---|---|---|
| FP16 | FP16/FP32 | 训练、推理 |
| BF16 | FP32 | 训练，指数范围接近 FP32 |
| TF32 | FP32 | Ampere+ 上加速 FP32 风格训练 |
| FP8 E4M3/E5M2 | FP16/FP32 | Hopper+ Transformer |
| FP4/FP6 | 更高精度 accumulator + scale | Blackwell 低精度训练/推理 |
| INT8 | INT32 | 量化推理 |
| FP64 | FP64 | HPC，产品支持与吞吐需核对 |

低位宽提升：

- 每次搬运携带更多元素；
- Tensor Core 每周期处理更多乘加；
- cache/shared/register/TMEM 压力降低。

代价：

- range/precision 损失；
- scaling/quantization overhead；
- saturation/overflow；
- accuracy 与训练稳定性；
- layout/packing 更复杂。

## 7. TF32

TF32 是 Ampere Tensor Core 引入的重要格式：

- FP32 类似的 8-bit exponent 范围；
- 较少 mantissa；
- 常用 FP32 accumulator；
- 让很多 FP32 GEMM 在保持较大范围的同时使用 Tensor Core。

它不是内存中必须存成一种普通 19-bit tensor 的用户数据类型；常见路径是输入
FP32，在 Tensor Core 运算时按 TF32 语义处理。

## 8. FP8 与 Transformer Engine

Hopper Tensor Core 支持：

- E4M3：更多 mantissa、较小 range；
- E5M2：更大 range、较低 precision。

Transformer Engine 在软件与硬件协作下管理：

```text
统计 tensor range
  → 选择 scale/format
  → cast 到 FP8
  → Tensor Core MMA
  → higher precision accumulation
  → 根据下一层需要保存/转换
```

它不是一个“自动让任意 kernel 变快”的单独 core，而是一套格式、Tensor Core、
scaling 和 library/framework 策略。

## 9. Blackwell FP4/FP6 与 Block Scaling

Blackwell 第五代 Tensor Core 支持更低位宽格式与 block-scaled MMA。核心思想：

```text
大 tensor 分成小 block
每个 block 使用自己的 scale
低位宽值 × scale 近似原值
```

相比整个 tensor 一个 scale，block scaling 更能适应局部 range；代价是需要：

- scale tensor；
- scale layout；
- 对齐/pack；
- Tensor Core 指令描述符；
- epilogue/quantization 配合。

## 10. FLOPS 为什么容易算错

一次 FMA：

```text
a × b + c
```

通常计为 2 FLOP：一次乘法 + 一次加法。

理论峰值还要区分：

- dense 还是 2:4 sparse；
- boost/基准频率；
- input dtype；
- accumulator dtype；
- Tensor Core 还是 CUDA Core；
- 产品实际启用 SM 数；
- 是否是单 GPU、Superchip 还是系统总和。

营销表格中 sparse 数值常是 dense 的 2 倍，不能直接用于普通 dense 模型。

## 11. Mainloop 与 Epilogue

GEMM kernel 可分：

```text
Mainloop
  ├─ load A/B tile
  ├─ pipeline/barrier
  └─ MMA accumulate

Epilogue
  ├─ alpha/beta
  ├─ bias
  ├─ activation
  ├─ quantize/cast
  └─ store D
```

小 M/N 或较短 K 时，epilogue、launch 和搬运占比可能很高。只优化 Tensor Core
mainloop 不一定改善端到端。

## 12. Ampere GEMM 流水

> 本节到 §14 只给出三代流水线的骨架。定量分析——要藏的延迟有多大、stage 数怎么
> 推、为什么 roofline 写 `max` 而不是 `sum`、以及 H20/GB200 的实测对比——见
> [11 GEMM 软件流水线深入](/notes/2026/08/26/2026-08-26-cuda-cute-nvidia-learning-11-gemm-pipeline-deep-dive/)。

典型：

```text
producer thread/warp:
  cp.async GMEM → SMEM

consumer warps:
  ldmatrix / register fragment
  mma.sync

cp.async group/barrier 控制 stage
```

## 13. Hopper GEMM 流水

典型 warp-specialized：

```text
producer warp
  → TMA GMEM → SMEM
  → mbarrier

consumer warpgroup
  → WGMMA 从 SMEM/register descriptor 取 operand
  → wgmma.mma_async
  → commit/wait group

epilogue warp
  → accumulator 处理与 store
```

WGMMA 的 warpgroup 是 4 个连续 warp，即 128 thread 的 collective。

## 14. Blackwell GEMM 流水

典型：

```text
TMA producer
  → A/B/scale 进入 SMEM

tcgen05 issue
  → Tensor Core 读取 operand
  → accumulator 写 TMEM

TMEM load / epilogue
  → register
  → cast/activation/store
```

TMEM 让 accumulator 不必长期占用大量普通 register，改变了 Blackwell kernel
的资源平衡与 warp specialization 方式。

## 15. 性能模型

粗略 roofline：

```text
attainable performance
= min(
    peak compute,
    memory bandwidth × arithmetic intensity
  )
```

对具体层级也可做：

```text
HBM roofline
L2 roofline
shared-memory roofline
Tensor Core instruction throughput
```

### 15.1 权重 GEMM 的算术强度 ≈ token 数

对权重矩阵 `[K, N]` 喂进 M 个 token：

```text
FLOPs = 2 · M · K · N
Bytes = 2 · K · N        (BF16 权重，读一次)
强度  = FLOPs / Bytes ≈ M
```

于是「这个 GEMM 算力受限还是带宽受限」化简成一个非常好记的判据：**把 token 数和
机器平衡点（peak FLOPS ÷ HBM 带宽）比大小**。

实测平衡点差异极大，不能跨卡套用：

| | 实测峰值 BF16 | 实测 HBM 读带宽 | 平衡点 |
|---|---|---|---|
| H20 | 138.4 TFLOP/s | 3646 GB/s | 38 FLOP/B |
| GB200 | 2284.6 TFLOP/s | 7138 GB/s | 320 FLOP/B |

对应到实测 GEMM：H20 在 M=64 就到 97% 峰值，GB200 在 M=64 只有 17.6%，要到 M≈768
才接近 91%。完整扫描数据与推导见
[11 GEMM 软件流水线深入](/notes/2026/08/26/2026-08-26-cuda-cute-nvidia-learning-11-gemm-pipeline-deep-dive/)。

高性能 GEMM 的目标是：

1. 增大数据复用；
2. 合并 global access；
3. 避免 shared bank conflict；
4. copy 与 MMA overlap；
5. 保持 Tensor Core pipe 有工作；
6. 控制 register/shared 占用；
7. epilogue 不成为瓶颈。

## 16. 官方资料

- [Ampere Tuning Guide：Improved Tensor Core Operations](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [PTX ISA：Matrix / Warpgroup MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Blackwell tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
