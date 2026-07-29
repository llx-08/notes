---
title: "06. Hopper/H20 实机导读：在 `ecs` 上把架构概念落到设备"
date: 2026-07-29
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# 06. Hopper/H20 实机导读：在 `ecs` 上把架构概念落到设备

## 1. 实机结论

2026-07-29 在 `ecs` 只读采集：

```text
hostname: iZ2zeiytc5rztf9rlvg5zmZ
GPU count: 8
GPU: NVIDIA H20
PCI device: GH100 [H20]
compute capability: 9.0
memory/GPU: 97871 MiB（约 95.6 GiB）
power limit: 500 W
driver kernel module: 570.133.20
NVLink topology: 任意 GPU pair 显示 NV18
CPU NUMA: GPU 0-3 → NUMA 0；GPU 4-7 → NUMA 1
```

H20 是基于 GH100/Hopper 的具体产品，不应把 H100 SXM 的峰值表直接当成 H20
参数。

## 2. 当前 NVML 环境问题

默认：

```bash
nvidia-smi
```

返回：

```text
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.173
```

检查：

```bash
cat /proc/driver/nvidia/version
```

显示内核模块为 570.133.20，而系统同时存在：

```text
/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.133.20
/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
```

只读探测使用：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.133.20 \
  nvidia-smi
```

这说明：

- NVML 是 `nvidia-smi` 使用的用户态管理库；
- NVML 与内核 driver module 版本不匹配可导致管理工具失败；
- `nvcc --version` 只说明编译工具链，不等于当前 kernel driver 版本；
- 诊断 CUDA 环境时要同时检查 driver、runtime、toolkit、framework。

## 3. 不依赖 NVML 识别 GPU

```bash
cat /proc/driver/nvidia/gpus/0000:08:00.0/information
```

得到：

```text
Model: NVIDIA H20
Bus Type: PCIe
DMA Size: 52 bits
Bus Location: 0000:08:00.0
GPU Firmware: 570.133.20
```

```bash
lspci -nn | grep -i NVIDIA
```

得到：

```text
NVIDIA Corporation GH100 [H20] [10de:2329]
NVIDIA Corporation GH100 [H100 NVSwitch] [10de:22a3]
```

这里分别确认：

- GPU PCI function；
- GH100/H20 芯片/产品标识；
- 系统含 NVSwitch bridge devices。

## 4. 拓扑

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.133.20 \
  nvidia-smi topo -m
```

简化结果：

```text
       GPU0 GPU1 ... GPU7  CPU Affinity         NUMA
GPU0    X   NV18 ... NV18  0-47,96-143          0
...
GPU3  NV18   ...           0-47,96-143          0
GPU4  NV18   ...           48-95,144-191        1
...
GPU7  NV18   ...     X     48-95,144-191        1
```

`NV18` 按 `nvidia-smi topo` legend 表示路径经过 bonded set of 18 NVLinks。
系统含 NVSwitch，因此从应用视角看到任意 GPU pair 的高速 NVLink 路径。

不要把 `NV18` 直接当成 NCCL 实测带宽。实际带宽还受：

- link generation/rate；
- NVSwitch routing；
- read/write direction；
- message size；
- GPU clock；
- NCCL protocol；
- concurrent traffic；
- memory bandwidth。

## 5. NVLink link status

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.133.20 \
  nvidia-smi nvlink -s
```

每个 GPU 报告 18 个 link，每 link：

```text
26.562 GB/s
```

简单乘法：

```text
18 × 26.562 ≈ 478.1 GB/s
```

这是工具报告的 link rate 汇总尺度，不等于单个应用一定能得到 478.1 GB/s payload
带宽，也不要在未确认双向计数定义时再乘 2。

## 6. Hopper 对 kernel 作者最重要的硬件能力

### 6.1 第四代 Tensor Core

- FP8 E4M3/E5M2；
- FP16/BF16/TF32/FP64/INT8 等；
- WGMMA warpgroup-level async MMA；
- Transformer Engine 相关低精度路径。

### 6.2 TMA

```text
一个 producer thread
  → 提交 tensor descriptor + coordinate
  → TMA 负责地址生成和 GMEM↔SMEM
  → mbarrier 通知完成
```

对比 Ampere `cp.async`，TMA 减少 thread 地址生成与 copy loop 压力。

### 6.3 Thread Block Cluster

Cluster 允许多个 SM 上的 CTA：

- 保证并发驻留；
- cluster barrier；
- distributed shared memory；
- TMA multicast。

### 6.4 WGMMA

```text
4 warps = 128 threads = warpgroup
```

典型：

```text
TMA producer warp
  → SMEM stage ready

consumer warpgroup
  → wgmma.mma_async
  → commit_group
  → wait_group
```

## 7. 为什么 H20 与 H100 不能只看架构名

即使都基于 GH100/Hopper，产品可在以下方面不同：

- 启用 SM 数量；
- Tensor Core 峰值与限制；
- HBM 容量/带宽；
- NVLink 配置；
- 功耗与频率；
- 合规/市场定位；
- firmware 与系统形态。

文档讲 Hopper 指令和编程能力时可使用架构指南；算 H20 峰值或做容量规划时必须
使用 H20 具体 SKU 或实机探测。

## 8. 建议实验

### 8.1 CUDA 属性

在带正确 framework/toolkit 的容器/环境中：

```python
import torch

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        i,
        p.name,
        (p.major, p.minor),
        p.multi_processor_count,
        p.total_memory,
        p.shared_memory_per_multiprocessor,
    )
```

### 8.2 P2P/NVLink

- `p2pBandwidthLatencyTest`；
- NCCL `all_reduce_perf`；
- 1 GPU pair 与 8 GPU all-to-all；
- message size sweep；
- 对照 `nvidia-smi topo -m`。

### 8.3 TMA/WGMMA

- CUTLASS Hopper GEMM example；
- Nsight Compute 查看 TMA request、tensor pipe、barrier stall；
- 改变 tile/stage；
- 对比普通 CUDA Core GEMM。

## 9. 安全实验原则

- 先只读查询；
- benchmark 前确认机器是否承载在线任务；
- 不修改 persistence/power/clock；
- 使用独立 CUDA_VISIBLE_DEVICES；
- 控制运行时长和显存；
- 保存命令、环境、driver/toolkit、commit。

## 10. 官方资料

- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [PTX WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)
