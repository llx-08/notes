---
title: "01. CUDA 执行模型：从 Kernel Launch 到 SM、Warp 与异步执行"
date: 2026-07-29
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# 01. CUDA 执行模型：从 Kernel Launch 到 SM、Warp 与异步执行

## 1. CPU launch kernel 时发生了什么

CUDA kernel launch 的语法：

```cpp
kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(args...);
```

这四组配置分别表示：

| 参数 | 含义 |
|---|---|
| `grid_dim` | grid 中有多少 block |
| `block_dim` | 每个 block 有多少 thread |
| `shared_bytes` | 每个 block 额外申请多少 dynamic shared memory |
| `stream` | kernel 被排入哪条 CUDA stream |

典型过程：

```text
CPU host thread
  → 检查/打包 kernel 参数
  → 向 CUDA runtime/driver 提交 launch
  → launch 进入指定 stream
  → host 调用通常返回

GPU
  → 按 stream 依赖取得 kernel
  → 将待运行 block 分配到有资源的 SM
  → SM 将 block 内 thread 划分为 warp
  → warp scheduler 持续发射 ready warp 的指令
```

![CUDA kernel 从 host launch 到 GPU 执行](/imgs/cuda-cute-launch-and-scheduling.svg)

“异步”是相对于 host 而言：host launch 返回，不代表 kernel 已执行完。

需要 host 确认完成时，可使用：

```cpp
cudaDeviceSynchronize();
cudaStreamSynchronize(stream);
cudaEventSynchronize(event);
```

三者等待范围不同，应选择最小必要范围。

## 2. Grid、Block、Thread

三维索引都是内建变量：

```cpp
blockIdx.x, blockIdx.y, blockIdx.z
threadIdx.x, threadIdx.y, threadIdx.z
blockDim.x, blockDim.y, blockDim.z
gridDim.x, gridDim.y, gridDim.z
```

一维全局索引：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

二维矩阵：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

线性 thread id：

```cpp
int tid =
    threadIdx.x
  + blockDim.x * (
        threadIdx.y
      + blockDim.y * threadIdx.z);
```

warp/lane：

```cpp
int warp_id_in_block = tid / warpSize;
int lane_id = tid % warpSize;
```

当前 NVIDIA CUDA warp size 为 32，但写通用 CUDA 代码时可使用 `warpSize`。

## 3. Block 为什么重要

Block 不只是 thread 的分组标签，它定义了：

1. 可以使用 `__syncthreads()` 的同步范围；
2. shared memory 的共享范围与生命周期；
3. block 被调度到一个 SM 的资源分配单位；
4. registers/shared memory/block slots 对 occupancy 的约束；
5. Hopper 以前普通编程模型中最主要的协作边界。

一个 block 在整个执行期间驻留在同一个 SM。一个 SM 可同时驻留多个 block，
条件是资源足够：

```text
threads/block
warps/block
registers/thread × threads
shared memory/block
最大 resident blocks/warps/threads
```

任一资源先达到上限，都会限制 resident block 数。

## 4. Warp 是什么

Block 中线性 thread id 相邻的每 32 个 thread 组成一个 warp：

```text
thread 0..31   → warp 0
thread 32..63  → warp 1
thread 64..95  → warp 2
```

warp 是 SM 指令调度和 SIMT 执行的重要粒度。程序写的是每个 thread 的语义，
硬件通常以 warp 为单位取指、发射。

### 4.1 SIMT 不等于简单 SIMD

SIMT = Single Instruction, Multiple Threads。

每个 thread 有自己的：

- logical thread id；
- register state；
- predicate；
- 地址与控制流语义。

但同一个 warp 的 thread 通常一起执行同一条指令。发生分支：

```cpp
if (lane_id < 16) {
  path_a();
} else {
  path_b();
}
```

若两个路径都要执行，warp 会按 mask 分别执行，产生 divergence。最终耗时通常
接近两个路径成本之和，而不是二选一。

### 4.2 Independent Thread Scheduling

Volta 及之后架构支持更灵活的 thread scheduling，但这不代表 warp 内 thread
可以忽略同步和内存可见性。依赖 warp lockstep 的旧代码应使用：

```cpp
__syncwarp(mask);
```

并明确使用 warp-level primitives 的 mask。

## 5. SM 如何隐藏延迟

假设 warp A 发出 global memory load，需要等待数百周期。SM 不一定停下来：

```text
cycle t:
  warp A 发 load，进入等待

cycle t+1:
  scheduler 选择 ready warp B

cycle t+2:
  选择 warp C

...
warp A 数据返回后重新变为 ready
```

这叫 latency hiding。它依赖：

- 有足够多 resident warp；
- warp 之间有可执行的独立工作；
- register/shared memory 不把 occupancy 压得过低；
- 内存系统没有达到吞吐极限。

高 occupancy 不是最终目标。一个 kernel 即使 occupancy 较低，也可能因为数据
复用好、ILP 高、Tensor Core pipeline 饱和而更快。

## 6. Warp Scheduler、Issue 与 Execution Pipeline

概念路径：

```text
resident warps
  → scoreboard 判断依赖是否满足
  → warp scheduler 选择 ready warp
  → dispatch/issue 指令
  → 指令进入对应流水线
     ├─ FP32/INT
     ├─ FP64
     ├─ load/store
     ├─ SFU
     └─ Tensor Core
```

“一个 warp 每周期执行一条指令”不是跨架构永远成立的简单定律。实际受到：

- scheduler 数量；
- 指令类型与吞吐；
- dual issue；
- 数据依赖；
- operand collector/register bank；
- pipeline latency；
- 架构具体实现。

性能分析应查看对应架构的 Nsight Compute metric，而不是凭 CUDA Core 数量推断。

## 7. Stream 的顺序语义

同一 stream 内：

```text
kernel A
memcpy B
event record
kernel C
```

按依赖顺序执行。不同 stream 可能并发，但需要硬件资源与显式依赖允许。

Event 可用于：

```cpp
cudaEventRecord(event, producer_stream);
cudaStreamWaitEvent(consumer_stream, event);
```

这建立 device-side stream 依赖，不需要 host 等待。

对比：

```text
cudaEventSynchronize(event)
  → host thread 等 GPU

cudaStreamWaitEvent(stream, event)
  → GPU stream 等另一个 event
  → host 可继续
```

## 8. CUDA Graph

频繁 launch 很小的 kernel 时，CPU launch overhead 可能显著。CUDA Graph 将一组
操作及依赖捕获/构建成图：

```text
普通：
CPU launch A → launch B → launch C → ...

Graph：
构建一次 A→B→C
每轮 launch graph executable
```

它减少重复的 host launch 开销，但不自动提高单个 kernel 的设备执行效率。

CuTe DSL、PyTorch compile/cudagraph 与自定义 op 结合时，要明确：

- 哪些形状/地址是静态；
- event 是内部依赖还是 external record/wait node；
- graph replay 时 allocation 与 stream 的生命周期。

## 9. Thread Block Cluster

Hopper 增加可选的 cluster 层次：

```text
Grid
└─ Cluster
   ├─ Block 0 → SM A
   ├─ Block 1 → SM B
   └─ ...
```

同一 cluster 的 block 被保证并发调度到一个 GPC 内，可使用：

- cluster synchronization；
- distributed shared memory；
- TMA multicast；
- cluster scope barrier/atomic。

Cluster 不等于把多个 block 放进同一 SM，而是让多个 SM 上的 block 获得明确的
协作与调度保证。

## 10. 结合 `target_p_j` 的真实 CuTe DSL 实验

`target_p_j:/dashscope/caches/workspace/llx/CuteDSL/hello.py` 使用：

```python
@cute.kernel
def kernel(exp_id: cutlass.Constexpr):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, tidy, tidz = cute.arch.thread_idx()
    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()
    cute.printf(...)
```

并运行：

```python
kernel(1).launch(grid=(1, 1, 1), block=(32, 1, 1))
kernel(2).launch(grid=(1, 1, 1), block=(64, 1, 1))
kernel(3).launch(grid=(2, 1, 1), block=(32, 1, 1))
kernel(4).launch(grid=(1, 1, 1), block=(4, 2, 1))
```

实测 `block=(64,1,1)` 中：

```text
thread 0..31  → warp 0, lane 0..31
thread 32..63 → warp 1, lane 0..31
```

输出行顺序不能作为线程执行顺序的保证；device `printf` 只是调试手段。

## 11. 常见误区

### 11.1 Grid 越大一定越快

不一定。Grid 太小无法填满 GPU；足够大后继续增大只增加总工作量。Block size、
资源使用与每线程工作量共同决定性能。

### 11.2 一个 thread 对应一个 CUDA Core

不是永久一一映射。CUDA thread 是软件状态，warp 指令在不同周期使用 SM 流水线。

### 11.3 Kernel launch 返回代表计算完成

通常不代表。launch 相对 host 异步，错误也可能在后续同步 API 才被观察到。

### 11.4 `__syncthreads()` 能同步整个 Grid

不能。它只同步同一 block。跨 block 需要拆 kernel、cooperative groups 或 cluster
等明确机制。

## 12. 官方资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [CUDA C++ Programming Guide：Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
- [CUDA Programming Guide：Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
