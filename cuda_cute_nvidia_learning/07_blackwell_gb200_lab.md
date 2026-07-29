# 07. Blackwell/GB200 实机导读：在 `target_p_j` 上认识 SM100 与 CuTe DSL

## 1. 实机基线

2026-07-29：

```text
host: target_p_j
hostname: ds-046-016
GPU count: 4
GPU: NVIDIA GB200
compute capability: 10.0
driver: 580.95.05
memory/GPU: 189471 MiB（约 185.0 GiB）
power limit: 1200 W
CUDA toolkit: 13.2 / nvcc 13.2.51
PyTorch: 2.11.0a0 NVIDIA build
PyTorch CUDA: 13.2
CuTe DSL: 4.5.2
Triton: 3.6.0
```

## 2. 查询命令

```bash
ssh target_p_j

nvidia-smi \
  --query-gpu=index,name,compute_cap,driver_version,memory.total,pci.bus_id,pstate,power.limit \
  --format=csv,noheader

nvidia-smi topo -m
nvidia-smi nvlink -s
nvcc --version
```

Python：

```python
import torch

for i in range(torch.cuda.device_count()):
    print(
        i,
        torch.cuda.get_device_name(i),
        torch.cuda.get_device_capability(i),
    )
```

输出均为：

```text
(10, 0)
```

即 SM100/compute capability 10.0 家族。

## 3. GPU 属性

GPU 0 的 PyTorch properties：

```text
SM count                        152
warp size                       32
max threads/SM                  2048
max threads/block               1024
registers/SM                    65536
shared memory/SM                233472 bytes
default shared memory/block     49152 bytes
memory clock                    3996000 kHz（工具字段）
memory bus width                7936 bits
```

解释：

- 152 SM 是当前产品启用数量，不应泛化为所有 Blackwell；
- 48 KiB 是默认 per-block shared limit，使用更大 dynamic shared 通常需 opt-in；
- 约 228 KiB shared/SM 与 Blackwell tuning guide 的片上容量主线一致；
- 总显存来自 `total_memory`/NVML，不能用 bus width × clock 简单推断有效应用带宽。

## 4. 拓扑

`nvidia-smi topo -m`：

```text
GPU0..GPU3 彼此：NV18

GPU0/1 CPU Affinity：0-71，NUMA 0
GPU2/3 CPU Affinity：72-143，NUMA 1

NIC0/1 对 GPU0/1：NODE
NIC2/3 对 GPU2/3：NODE
跨 NUMA 组合：SYS
```

这说明两条拓扑同时存在：

```text
GPU ↔ GPU：第五代 NVLink/NVSwitch 路径

GPU ↔ RNIC/CPU：
  同 NUMA NODE 较近
  跨 NUMA SYS 较远
```

对 NCCL/NVSHMEM/RDMA：

- GPU-GPU collective 关注 NVLink domain；
- GPU-NIC 关注 NUMA/RNIC affinity；
- “GPU 都 NV18 相连”不代表任意 GPU 到任意 NIC 路径同样近。

## 5. NVLink link status

每 GPU 18 link，每 link：

```text
53.125 GB/s
```

相对于 `ecs` H20 报告的 26.562 GB/s/link，工具层面约为 2 倍，符合第五代
NVLink 对第四代链路速率的代际提升方向。

简单聚合：

```text
18 × 53.125 = 956.25 GB/s
```

仍要注意：这是 link status 的聚合尺度，不是某个单向/双向 NCCL payload
benchmark 的承诺值。

## 6. Blackwell Tensor Core 数据流

![GB200/SM100 中 tcgen05 与 TMEM 概念流](imgs/cuda-cute-blackwell-pipeline.svg)

关键部件：

```text
HBM
  ↕ TMA
SMEM
  → tcgen05 operand
5th Gen Tensor Core
  → accumulator
TMEM
  → tcgen05.ld/copy
register / epilogue
  → HBM
```

### 6.1 TMEM 不是 shared memory

区别：

| | Shared Memory | TMEM |
|---|---|---|
| 主要用途 | 通用 CTA staging/共享 | 第五代 Tensor Core accumulator/相关数据 |
| 访问 | 普通 shared load/store、TMA | tcgen05 专用 alloc/load/store/copy |
| 生命周期 | block/cluster 资源 | 显式 TMEM allocation protocol |
| 布局 | bank/swizzle | Tensor Core 定义的 TMEM column/layout |

### 6.2 `tcgen05.mma`

CUTLASS 官方 SM100 文档说明：

- 支持 TF32/FP16/BF16/INT8；
- 支持 FP8/FP6/FP4 和 scaling；
- 可由单 thread issue；
- 支持 CTA group 1/2；
- accumulator 写入 TMEM；
- 指令吞吐相对 Hopper WGMMA 随 dtype 可达 2×～4×。

最后一项是指令/理论层面，端到端 kernel 必须另外测量。

## 7. CuTe DSL 环境

```python
import cutlass
import cutlass.cute as cute

print(cutlass.__version__)
```

实测：

```text
4.5.2
```

安装位置：

```text
/usr/local/lib/python3.12/dist-packages/
  nvidia_cutlass_dsl/python_packages/cutlass/
```

已有工作目录：

```text
/dashscope/caches/workspace/llx/CuteDSL
├─ hello.py
├─ diagrams/
├─ docs/
└─ ir_dump/
```

## 8. 已运行的 CuTe DSL 线程实验

```bash
cd /dashscope/caches/workspace/llx/CuteDSL
python3 hello.py
```

返回码 0，共 176 行。

Exp1：

```text
grid=(1,1,1)
block=(32,1,1)
```

输出确认：

```text
thread 0  → warp 0, lane 0
...
thread 31 → warp 0, lane 31
```

Exp2：

```text
block=(64,1,1)
```

输出确认：

```text
thread 32 → warp 1, lane 0
thread 33 → warp 1, lane 1
...
```

这验证了 CuTe DSL 的：

```python
cute.arch.block_idx()
cute.arch.thread_idx()
cute.arch.warp_idx()
cute.arch.lane_idx()
```

与 CUDA 执行模型一致。

## 9. CuTe DSL 为什么不能直接在 REPL 定义 JIT 函数

在 stdin/REPL 中尝试：

```python
@cute.jit
def show_layout():
    ...
```

当前 4.5.2 返回：

```text
DSLRuntimeError: Failed to parse function show_layout
DSL does not support REPL mode, save the function to a file instead.
```

原因是 DSL 会读取 Python function source/AST 做转换。教学实验应保存成 `.py`
文件运行，而不是直接粘贴到交互式 stdin。

## 10. Blackwell 实验路线

### 10.1 基础

1. thread/warp/lane；
2. vector add；
3. naive GEMM；
4. shared tiled GEMM；
5. CuTe Layout。

### 10.2 Tensor Core

1. CUTLASS/CuTe SM100 GEMM；
2. FP16/BF16；
3. FP8；
4. block-scaled FP4；
5. CTA group 1 vs 2；
6. tile/stage sweep。

### 10.3 Profiling

1. Nsight Systems 看 host launch 与 overlap；
2. Nsight Compute 看 tensor pipe、TMA、TMEM、stall；
3. PTX 查 `tcgen05`；
4. SASS 与 resource usage；
5. correctness/precision。

## 11. 编译注意

```bash
nvcc -arch=sm_100 ...
```

CuTe DSL/CUTLASS example 要与安装版本匹配。官方 quick start 建议使用对应
CUTLASS commit 的 setup script；“pip wheel 最新版 + GitHub main examples”可能
存在 API 不一致。

## 12. 官方资料

- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
- [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
- [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
