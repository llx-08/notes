---
title: "08. CuTe DSL：从 Layout、Tensor 到 Python JIT"
date: 2026-07-29
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# 08. CuTe DSL：从 Layout、Tensor 到 Python JIT

这一章先不写高性能 GEMM，而是解决 CuTe 最容易令人困惑的问题：

> 为什么一个“矩阵布局”会写成 `((2, 4), (1, 2))`？
> `Tensor` 和普通数组有什么区别？
> Python 代码最终如何变成 GPU 指令？

CuTe 的核心思想不是发明另一套 CUDA，而是把“逻辑坐标如何映射到物理地址”变成
可以组合、变换和检查的对象。

![CuTe 坐标、Layout 与 Tensor](/imgs/cuda-cute-cute-layout.svg)

## 1. 三个最重要的名词

### 1.1 Coordinate：逻辑坐标

假设有一个 `M × N` 矩阵：

```text
A[0,0] A[0,1] A[0,2]
A[1,0] A[1,1] A[1,2]
```

`(1, 2)` 是逻辑坐标。它回答“我要第几行、第几列”，没有回答它位于第几个字节。

### 1.2 Layout：坐标到线性 offset 的函数

二维 layout 可先直观地写成：

```text
offset(i, j) = i × stride_i + j × stride_j
```

shape 为 `(2, 3)` 的 row-major 矩阵：

```text
shape  = (2, 3)
stride = (3, 1)

(0,0) → 0
(0,1) → 1
(0,2) → 2
(1,0) → 3
(1,1) → 4
(1,2) → 5
```

column-major：

```text
shape  = (2, 3)
stride = (1, 2)

(0,0) → 0
(1,0) → 1
(0,1) → 2
(1,1) → 3
(0,2) → 4
(1,2) → 5
```

所以 `Layout` 是映射规则，不是存放数据的 buffer。

### 1.3 Tensor：Engine + Layout

CuTe 中可以用下面这个心智模型：

```text
Tensor = Engine + Layout
```

- `Engine`：数据从哪里来，例如 global-memory pointer、shared-memory buffer、
  register fragment；
- `Layout`：给定逻辑坐标后，如何计算 engine 中的 offset。

因此，同一块内存可叠加不同 view：

```text
同一个 pointer
  ├─ 作为 M×N row-major matrix
  ├─ 作为 N×M transposed view
  └─ 分割成 block tile / warp tile / thread fragment
```

只改变 view 通常不搬运 payload；真正 load/store/copy 时才移动数据。

## 2. 为什么 Shape 和 Stride 允许嵌套

普通二维数组只需要 `(M, N)`。高性能 kernel 还要表达：

```text
矩阵
  → CTA tile
    → warp tile
      → MMA atom
        → 每个 lane 持有的 fragment
```

所以 CuTe 的 mode 可以递归嵌套：

```text
shape  = ((M_block, M_in_block), (N_block, N_in_block))
stride = ((...), (...))
```

外层描述有多少 tile，内层描述 tile 内元素。这不是“多了几维物理内存”，而是
把同一个线性地址空间赋予分层坐标。

例如长度 8：

```text
线性坐标：0 1 2 3 4 5 6 7

分为 2 组、每组 4 个：
(group, in_group)
(0,0) (0,1) (0,2) (0,3)
(1,0) (1,1) (1,2) (1,3)
```

映射可写为：

```text
offset(group, in_group) = group × 4 + in_group
```

在 GEMM 中，同样的方法可表示：

- CTA 在 problem 中的位置；
- warp 在 CTA 中的位置；
- lane 在 warp 中的位置；
- lane 负责的多个非连续元素。

## 3. 用 CuTe DSL 创建 Layout

配套示例：

```bash
python3 examples/cutedsl/02_layout_basics.py
```

核心形式：

```python
from cutlass import cute

layout = cute.make_layout(
    (2, 3),
    stride=(3, 1),
)
```

实际版本中动态/静态 shape、整数类型与打印行为可能不同，应以安装版本的 API
文档和示例为准。`target_p_j` 当前 CuTe DSL 版本为 4.5.2。

常用观察量：

```text
shape(layout)   逻辑坐标域的大小
stride(layout)  各 mode 的步长
size(layout)    逻辑元素数
cosize(layout)  映射所覆盖的最小地址空间尺度
rank(layout)    顶层 mode 数
depth(layout)   嵌套深度
```

### 3.1 `size` 与 `cosize` 不总相等

连续 layout：

```text
shape=(2,3), stride=(3,1)
offset={0,1,2,3,4,5}
size=6, cosize=6
```

带 padding：

```text
shape=(2,3), stride=(4,1)
offset={0,1,2,4,5,6}
size=6
覆盖到 offset 6，所以地址空间至少需要 7 个元素
```

padding 可用于对齐、避免某些 shared-memory bank conflict，代价是多占空间。

### 3.2 stride 为 0

如果某一维 stride 为 0：

```text
offset(i,j) = i × 0 + j
```

不同 `i` 可映射到相同物理元素。这可描述 broadcast view。它不是数据真的复制了
多份，而是多个逻辑坐标读取同一地址。

## 4. Layout Algebra：为什么说 Layout 可以“做代数”

CuTe 不仅保存 layout，还能组合 layout：

### 4.1 Composition

```text
result(coord) = A(B(coord))
```

直观上，B 先把新坐标解释为旧坐标，A 再把旧坐标映射到地址。

应用：

- 把线程 layout 与数据 layout 组合，求每个线程访问哪些元素；
- 将 swizzle 与 shared-memory layout 组合；
- 对 transposed/tiled view 做零拷贝变换。

### 4.2 Logical Divide

用 tile 去除一个 layout：

```text
原坐标
  → (tile 内坐标, tile 编号)
```

例如 16 个元素按 4 元素 tile：

```text
0..15
  → (within_tile=0..3, tile_id=0..3)
```

这正是 kernel 常见的“某个 CTA 处理哪个 tile、CTA 内某个线程处理哪个元素”。

### 4.3 Product

把 layout 扩展或拼接成更大的逻辑域。实际使用时要辨别 `logical_product`、
`blocked_product`、`raked_product` 等不同排列语义，不能只看最终 size。

### 4.4 Coalesce

合并可以等价视为连续的一组 mode，减少 layout 表达复杂度。它应保持坐标到 offset
的映射语义，而不是把非连续内存神奇地变连续。

## 5. Thread Layout 与 Value Layout

高性能 copy/MMA 常出现两个 layout：

```text
Thread layout：哪一个逻辑线程参与
Value layout：该线程负责哪些 value
```

例：8 个线程搬 32 个元素，每个线程 4 个：

```text
thread 0 → value 0,  8, 16, 24
thread 1 → value 1,  9, 17, 25
...
thread 7 → value 7, 15, 23, 31
```

这可以实现连续线程访问连续地址，从而更容易合并 global-memory transaction。
另一个合法映射是：

```text
thread 0 → 0,1,2,3
thread 1 → 4,5,6,7
```

两者逻辑工作量相同，但内存 transaction、向量化和后续 MMA fragment 排布可能不同。
CuTe 的价值之一，是让这类映射可以显式推导，而不是散落在大量 `%` 和 `/` 中。

## 6. CuTe DSL 的编译路径

![CuTe DSL 编译路径](/imgs/cuda-cute-dsl-codegen.svg)

典型路径：

```text
Python source
  → @cute.jit 捕获/分析 DSL 函数
  → CUTLASS DSL IR / MLIR dialects
  → NVVM / LLVM / PTX
  → ptxas / driver JIT
  → CUBIN/SASS
  → GPU 执行
```

必须区分两个时间：

### 6.1 DSL 编译期

CuTe 会处理：

- 静态 shape/layout；
- 类型与地址空间；
- thread/value 映射；
- 目标架构；
- 某些 Python 控制流和元编程。

### 6.2 GPU 运行期

GPU 实际处理：

- runtime pointer 和 problem size；
- global/shared/register 数据；
- load/store/TMA/MMA；
- CTA、warp 和线程调度。

普通 Python 对象并不会自动存在于 device。只有 DSL 支持并编译进去的部分成为
kernel 逻辑。

## 7. 为什么 `@cute.jit` 不适合直接粘到 REPL

在 `target_p_j` 直接把装饰函数写进 `python3 - <<'PY'` 时，4.5.2 会报告：

```text
DSL does not support REPL mode, save the function to a file instead
```

原因是 DSL 编译器需要读取函数 source/AST。正确方法是保存为 `.py` 文件再执行。

这不是 GPU 不支持，也不是 CuTe import 失败；属于 source capture 的使用限制。

## 8. 与 NumPy/PyTorch Tensor 的关系

共同点：

- 都有 shape、stride、dtype、storage/pointer；
- view/transpose 可能只改元数据；
- 非连续 tensor 需要正确 stride。

差异：

- PyTorch Tensor 是面向运行时张量计算的用户对象；
- CuTe Tensor 主要服务于 kernel 内部，能够描述 global/shared/register/TMEM
  相关 view；
- CuTe layout 会继续细化到 CTA/warp/lane/value 与硬件 atom；
- CuTe DSL 编译结果是 kernel，而不是在 Python 中逐元素执行。

框架接入通常是：

```text
PyTorch tensor
  → 取 pointer / shape / stride / stream
  → 包装为 CuTe 能理解的 tensor/view
  → launch 已编译或 JIT 的 kernel
  → 结果仍位于 PyTorch tensor storage
```

接入时需要保证：

- dtype 和 alignment 正确；
- shape/stride 满足 kernel 假设；
- device 与 CUDA context 正确；
- 使用正确 stream；
- tensor 生命周期覆盖异步 kernel；
- kernel 出错能传播给框架。

## 9. 初学者常见误区

### 误区一：Layout 就是矩阵

Layout 只是映射，Tensor 才把映射绑定到数据 engine。

### 误区二：改变 Layout 会移动数据

view 类操作通常只改变解释方式；copy 才移动 payload。

### 误区三：shape 相同，性能就相同

stride、alignment、thread/value mapping、向量宽度和 bank mapping 都会改变性能。

### 误区四：CuTe 会自动生成最优 kernel

CuTe 提供正确、可组合的表达能力；tile、stage、copy atom、MMA atom、cluster、
epilogue 等选择仍决定性能。

### 误区五：所有 CuTe API 都跨版本稳定

CuTe DSL 仍快速发展。以本机安装版本、官方 examples 和 release notes 为准。

## 10. 本章练习

1. 手算 `(2,3):(3,1)` 的全部 offset；
2. 改成 `(2,3):(4,1)`，比较 `size` 与地址跨度；
3. 画出 8 threads × 4 values 的两种映射；
4. 判断 transpose 是 view 还是 payload copy；
5. 执行 `02_layout_basics.py`，再把 shape 改成嵌套 shape；
6. 下一章再将 layout 绑定到 Copy Atom 和 MMA Atom。

## 11. 官方资料

- [CuTe DSL Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
- [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)
- [CuTe DSL Code Generation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html)
- [Framework Integration](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html)
