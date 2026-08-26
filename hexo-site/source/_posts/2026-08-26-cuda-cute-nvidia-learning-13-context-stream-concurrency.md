---
title: "13. CUDA context 与 stream：执行模型的进程视角"
date: 2026-08-26
categories: [CUDA、CuTe DSL 与 NVIDIA GPU 架构学习笔记]
tags: [CUDA, CuTe DSL, CUTLASS, GPU, Tensor Core, NVIDIA, 学习笔记]
---

# 13. CUDA context 与 stream：执行模型的进程视角

[第 12 章](/notes/2026/08/26/2026-08-26-cuda-cute-nvidia-learning-12-pdl-programmatic-dependent-launch/) 讲的是**同一条 stream 上相邻两个
kernel** 之间的缝隙怎么压。这一章往外扩两层：**不同 stream 之间**怎么并发，以及
**不同进程之间**为什么不能并发。

三层是正交的，机制完全不同：

![三层并发](/imgs/cuda-cute-ctx-three-layers.svg)

本章要回答的具体问题：

```text
CUDA context 到底是什么？和 UVA 是一回事吗？
stream 怎么实现的？两个 kernel 放在不同 stream 上会怎么执行？
blocking 和 non-blocking stream 差在哪？
kernel 优先级怎么设？真的能插队吗？
另一个进程能访问我这个进程的显存地址吗？
```

所有性能数字都在 `target_p`（4×GB200，SM100，CUDA 13.2）和 `ecs`（8×H20，SM90，
CUDA 12.8）上实测，基准源码与复现命令见 §8。文档引文出处见 §10。

---

## 1. CUDA context = GPU 上的「进程」

CUDA context 可以理解成 GPU 侧的进程。它拥有：

- 一套 GPU 虚拟地址空间和页表
- 该 context 内的所有显存分配
- 加载进来的 module / cubin（kernel 代码）
- stream、event 等对象

**粒度是「每进程 × 每设备一个」。** 用 Runtime API 时它叫 *primary context*，懒创建 ——
第一次调用任何 CUDA API 时才建立。

实测建立代价（GB200）：

```text
首次调用建 context (device 0)  : 208.5 ms      # cudaSetDevice(0) + cudaFree(0)
再建 device 1 的 context        : +288.0 ms
```

**每张卡各付一次两三百毫秒。** 这就是多卡程序启动慢的直接原因，也是框架都要做 context
预热的原因 —— 你不希望第一个请求进来时才付这笔钱。

Driver API 还可以用 `cuCtxCreate` 在同一个设备上建**多个** context，但它们之间只能时间片
轮转，切换很贵，一般没有理由这么做。

---

## 2. UVA 不是 context 的一部分

很多入门材料把这两件事混在一起讲，实际上层级不同：

- **context 是 per-(进程, 设备)**：4 张卡就是 4 个 context，各有独立页表
- **UVA（Unified Virtual Addressing）是 per-进程**，横跨主机和所有设备，整个进程只有一个

![context 与 UVA](/imgs/cuda-cute-ctx-context-uva.svg)

实测把这个关系照了出来（同一个进程内的四个指针）：

| 指针来源 | 地址 | `cudaPointerGetAttributes` |
|---|---|---|
| 普通 `malloc` | `0xaaaae69de7b0` | **unregistered**, device = -1 |
| `cudaHostAlloc` | `0xfffce0000000` | host, device = 0 |
| `cudaMalloc` dev0 | `0xfffd1dc00000` | device, device = 0 |
| `cudaMalloc` dev1 | `0xfffcbde00000` | device, device = 1 |

两个观察：

**(a) 后三者落在 `0xfffc~0xfffd` 同一段连续区间。** pinned 主机内存和**两张不同卡**的
显存共享同一个虚拟地址空间 —— 这就是「统一虚拟地址」的直观样子。

**(b) 普通 `malloc` 的内存不在 UVA 里。** 常见的说法「UVA 把所有系统内存都映射进来」是
不准确的：只有**注册过的**主机内存（`cudaHostAlloc` / `cudaHostRegister`）才进 UVA。

这个区别是实际的。能不能给 kernel 直接传一个主机指针、`cudaMemcpy` 能不能用
`cudaMemcpyDefault` 自动推断方向，都取决于地址在不在 UVA 表里。而这张表本身的存在也说明
runtime 确实在维护「地址 → 属于哪个设备」的映射 —— 否则 `cudaPointerGetAttributes` 无从
判断。

---

## 3. stream 的实现

### 3.1 软件层：一条有序队列

Stream 就是一个**命令 FIFO 队列**：

- 同一 stream 内**严格按提交顺序**执行
- 不同 stream 之间**没有任何顺序保证** —— 这就是并发的来源

对应到硬件通讯路径，命令被写进 pinned memory 里的 ring buffer，再通过 MMIO 写 doorbell
寄存器通知 GPU，GPU 的 DMA 引擎取走并由命令处理器解码派发。

### 3.2 硬件层：有限条工作队列

关键在于 **软件 stream 数量不限，硬件工作队列是有限的**。官方文档：

> **`CUDA_DEVICE_MAX_CONNECTIONS`** — 1 to 32 (default is 8)
> Sets the number of compute and copy engine **concurrent connections (work queues)**
> from the host to each device of compute capability 3.5 and above.

![stream 到硬件队列](/imgs/cuda-cute-ctx-stream-hw-queue.svg)

**默认只有 8 条。** 超过就要共享队列，于是可能出现**假依赖**：单队列是 FIFO，`A2` 等
`A1` 是对的（同 stream），但排在 `A2` 后面的 `B1` 会被连坐堵住。

这个效应只在**同一个 stream 内有多个 kernel** 时才会触发 —— 如果每个 stream 只放一个
kernel，提交顺序 `K1 K2 K3 K4` 彼此无依赖，单条队列照样能连续发射。这一点我最初测错了，
见 §8.2。

### 3.3 实测：同 stream vs 不同 stream

GB200，4 个 kernel 各占 1/4 GPU（grid = 38，1 CTA/SM）：

| stream 配置 | 跨度 ms | Σ时长 ms | 并发度 |
|---|---|---|---|
| 全部在 NULL stream (stream 0) | 2.01 | 2.01 | **1.00** |
| 全部在同一个自建 stream | 2.01 | 2.01 | **1.00** |
| 4 个不同 stream (blocking) | 0.52 | 2.01 | **3.90** |
| 4 个不同 stream (nonBlocking) | 0.51 | 2.01 | **3.92** |

**同 stream = 严格串行，不同 stream = 真并发。** 四个 kernel 的活在 1/4 的时间里干完。

并发度略低于 4.00（3.90）的那 2.5% 是四个 kernel 的启动错峰 —— 它们不是同一纳秒开始的。

---

## 4. 并发的三个前提

放在不同 stream 只是**允许**并发，不**保证**并发。实际能不能并发取决于三件事，按重要性
排序。

### 4.1 前提一：有剩余 SM 资源（最主要）

![leftover policy](/imgs/cuda-cute-ctx-leftover.svg)

2 个 kernel 放 2 个 non-blocking stream，扫它们各自占多少 SM（GB200，152 SM）：

| grid / kernel | 占 GPU | 跨度 ms | Σ时长 ms | 并发度 | 判定 |
|---|---|---|---|---|---|
| 19 | 12% | 0.50 | 1.00 | 1.99 | 并发 |
| 38 | 25% | 0.51 | 1.01 | 1.99 | 并发 |
| 76 | 50% | 0.51 | 1.01 | 1.99 | 并发 |
| 114 | 75% | 1.00 | 1.50 | **1.50** | 部分并发 |
| 152 | 100% | 1.01 | 1.01 | **1.00** | 串行 |
| 304 | 200% | 2.01 | 2.01 | **1.00** | 串行 |

硬件按「先来的先铺满，后来的捡剩下的」分配（leftover policy）。75% 那行最能说明问题：
两个各要 114 个 SM 但总共只有 152，于是 K1 全速跑，K2 先拿到 38 个 SM，随着 K1 的 CTA
陆续退出再补上，得到 1.50 的部分并发。

> **推论：如果 kernel 本来就能打满 GPU（大 batch 的 GEMM），多开 stream 一点用都没有。**
> stream 并发的价值只在小 kernel 场景 —— 也就是 bs=1 decode 那种，和第 12 章 PDL 的适用
> 场景重合。

### 4.2 前提二：中间没有 NULL stream 操作

![NULL stream](/imgs/cuda-cute-ctx-null-stream.svg)

官方文档：

> For code that is compiled using the `--default-stream legacy` compilation flag, the
> default stream is a special stream called **the NULL stream** and each device has a
> single NULL stream used for all host threads. **The NULL stream is special as it causes
> implicit synchronization.**

> **Two operations from different streams cannot run concurrently if any CUDA operation
> on the NULL stream is submitted in-between them**, unless the streams are non-blocking
> streams (created with the `cudaStreamNonBlocking` flag).

实测（GB200，在两次 launch 之间插一个提交到 stream 0 的 kernel）：

| stream 配置 | 跨度 ms | 并发度 |
|---|---|---|
| blocking stream，中间不插 | 0.52 | 3.90 |
| blocking stream，**中间插一个 NULL 操作** | 2.04 | **0.99** |
| nonBlocking stream，中间插一个 NULL 操作 | 0.52 | **3.90** |

**一个提交到 stream 0 的操作，就能把整批并发拍平。**

#### blocking vs non-blocking：先破除一个望文生义

这个名字和「主机调用会不会阻塞」**毫无关系**。两种 stream 对主机都是异步的。它指的是：
**这个 stream 会不会和 NULL stream 互相同步。**

```cpp
cudaStreamCreate(&s);                                    // blocking（默认）
cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);    // non-blocking
```

危险在于它**看不见**：你自己某处 `cudaMemcpy` 忘了写 stream 参数、某个第三方库内部用了
默认流、某段调试代码顺手加了同步 —— 都会让精心设计的多流并发悄悄退化成串行，而且从代码
上完全看不出来，只在 profile 里表现为「莫名其妙串行了」。

实测你手上环境的实际情况：

```text
torch 2.11.0a0+...
torch.cuda.Stream()  →  cuStreamGetFlags = 1  →  NON-BLOCKING
                        cuStreamGetPriority = 0（最低优先级）
```

**PyTorch 建的是 non-blocking stream。** 好消息是不会被默认流拖累；代价是**你不能指望
默认流帮你做隐式同步**，该 `record_stream()` / `wait_stream()` 的地方一个都不能省。

另外还有编译期开关 `nvcc --default-stream per-thread`，它把默认流变成每个主机线程一条
独立的普通流，也就没有 NULL stream 的隐式同步了。

### 4.3 前提三：硬件工作队列够（Hopper 上要注意，Blackwell 上似乎已无所谓）

8 streams × 6 kernels，各占 1/8 GPU，对比深度优先（`A1A2A3|B1B2B3`）和广度优先
（`A1B1C1|A2B2C2`）两种提交顺序：

**H20 (SM90, CUDA 12.8) —— 教科书式复现：**

| CONN | 提交顺序 | 跨度 ms | 并发度 |
|---|---|---|---|
| 1 | 深度优先 | 20.58 | **1.17** ← 假串行 |
| 1 | 广度优先 | 3.02 | 7.94 ← 免疫 |
| 2 | 深度优先 | 10.56 | **2.27** ← ≈ 队列数 |
| 2 | 广度优先 | 3.03 | 7.94 |
| 8 | 深度优先 | 3.11 | 7.73 |
| 32 | 深度优先 | 3.11 | 7.72 |

深度优先的并发度**精确等于工作队列数**（1→1.17，2→2.27，8→7.73）。这就是 Hyper-Q 当年
要解决的问题。

**GB200 (SM100, CUDA 13.2) —— 完全没有这个现象：**

| CONN | 深度优先 | 广度优先 |
|---|---|---|
| 1 | 7.81 | 7.94 |
| 2 | 7.79 | 7.96 |
| 8 | 7.79 | 7.94 |
| 32 | 7.79 | 7.94 |

`CONN=1` 照样 7.8 路并发，两种提交顺序没有区别。看起来 Blackwell 的前端不再因为队列头部
阻塞就停止向后派发。

> **但两个数据点没法区分这是架构（SM90 vs SM100）、CUDA 版本（12.8 vs 13.2）还是驱动
> 版本导致的**，这点不装懂。
>
> **实践含义**：「用广度优先提交顺序」「调大 `CUDA_DEVICE_MAX_CONNECTIONS`」这两条流传
> 很广的调优建议，在 Hopper 上仍然有效且效果巨大（1.17 → 7.94，差 6.8 倍），在 Blackwell
> 上似乎已经过时。如果代码里有为此做的 workaround，换到 GB200 上可以考虑去掉。

---

## 5. stream 优先级：实测几乎无效

### 5.1 怎么设

**没有「kernel 优先级」，只有 stream 优先级。**

```cpp
int least, greatest;
cudaDeviceGetStreamPriorityRange(&least, &greatest);
// GB200 与 H20 实测均为: least = 0, greatest = -5

cudaStream_t s;
cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, greatest);  // 最高优先级
int p; cudaStreamGetPriority(s, &p);                                // 可以查回来
```

**数值越小优先级越高**（-5 最高，0 最低），这点反直觉。创建后不能改。

### 5.2 实测：它什么都没做

设计一个能直接看出调度顺序的实验：两个 stream，**每个 kernel 都占满整个 GPU**
（grid = SM 数，1 CTA/SM），于是 8 个 kernel 必须严格串行，「实际执行顺序」就直接暴露了
调度器的选择。

先提交 4 个到 A 流，再提交 4 个到 B 流。若优先级有效、B 为高优先级，应该是
`A0 B0 B1 B2 B3 A1 A2 A3`（A0 已在跑抢不掉，其余 B 插队）。

```text
GB200:
对照组: A=B=最低(0)                  提交 A0..A3 然后 B0..B3   执行: A0 B0 A1 B1 A2 B2 A3 B3
实验组: A=最低(0)  B=最高(-5)         提交 A0..A3 然后 B0..B3   执行: A0 B0 A1 B1 A2 B2 A3 B3
实验组: A=最低(0)  B=最高(-5) 先提交B  提交 B0..B3 然后 A0..A3   执行: B0 A0 B1 A1 B2 A2 B3 A3
反向:   A=最高(-5) B=最低(0)          提交 A0..A3 然后 B0..B3   执行: A0 B0 A1 B1 A2 B2 A3 B3

H20: 四组结果完全一样
```

**四种配置执行顺序完全相同，只取决于哪个 stream 先提交。两个 stream 严格轮转，优先级零
影响。** 而且两代架构一致，不是架构差异。

另一组测的是抢占：

```text
低优先级 kernel 占满 GPU 跑 2.01 ms
高优先级 kernel 在其开始后 2.00 ms 才启动   ← 一直等到低优先级跑完
```

文档其实说清楚了，只是措辞客气：

> these priorities serve as **hints rather than guarantees** ... **Higher-priority tasks
> do not preempt already running lower-priority tasks.** The GPU does not reassess work
> queues during task execution, and increasing a stream's priority will not interrupt
> ongoing work.

> **不要指望用 stream 优先级做 QoS。** 想给低延迟请求插队，靠优先级是不行的 —— 正在跑的
> CTA 不会被踢下来，而且本测试里连待派发队列都没重排。
>
> 真要做资源隔离，现在的手段是 **Green Contexts**（CUDA 12.4+），它是真的把 SM 划分给
> 不同执行上下文，是资源保证而非提示。

---

## 6. 跨进程：地址隔离与 CUDA IPC

### 6.1 结论：不能访问，而且失败方式比报错更危险

![跨进程隔离与 IPC](/imgs/cuda-cute-ctx-ipc.svg)

实验设计：

- **producer 进程**：`cudaMalloc` 一块显存，填入 pattern `0xAA00, 0xAA01, ...`，
  打印裸指针，导出 IPC handle 到文件
- **consumer 进程**（独立进程）：自己也 `cudaMalloc` 一块，填入**不同的** pattern
  `0xBB00`，然后拿 producer 的裸地址值去解引用

```text
[producer] 显存裸指针 = 0xfffcfdc00000   (pattern 0xAA00)

[consumer] 从 producer 拿到的裸指针值 = 0xfffcfdc00000
[consumer] cudaPointerGetAttributes -> type=0 (Unregistered), device=-1
[consumer] 本进程自己的 cudaMalloc  = 0xfffcfdc00000   (填入 pattern 0xBB00)
[consumer] 两者数值完全相同！
[consumer] kernel 解引用那个地址: no error
[consumer] 读到的值 = 0xBB00        ← 不是 0xAA00
```

三个要点：

1. **两个进程的指针数值完全一样** —— CUDA 的虚拟地址分配是确定性的，两个进程都从同一个
   基址开始分配。
2. **解引用不报错** —— 那个地址在 consumer 自己的 context 里确实是有效映射。
3. **但读到的是 `0xBB00`，是 consumer 自己的数据。**

**同一个数值地址，在两个进程里指向完全不同的物理显存。** GPU 的 MMU 和页表提供的隔离，
和 CPU 上的进程隔离是一回事 —— context 就是 GPU 上的进程。

> 危险之处：如果误以为可以跨进程传指针（比如通过共享内存传了个 `cudaMalloc` 的地址），
> **你不会得到崩溃，你会静默地读到错误数据。**

### 6.2 要共享就用 CUDA IPC

同一实验的第二部分：

```text
[consumer] cudaIpcOpenMemHandle 成功，本进程内的指针 = 0xfffcfdc00000
[consumer] 读到 host[0]=0xAA00 host[1]=0xAA01 host[255]=0xAAFF
[consumer] ✓ 正确——确实是同一块物理显存
```

用法：

```cpp
// 进程 A
cudaIpcMemHandle_t h;
cudaIpcGetMemHandle(&h, dev_ptr);
// 把 h 这 64 字节通过任意 IPC 通道(socket / 文件 / 共享内存)传给 B

// 进程 B
void *p;
cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
// p 是 B 自己地址空间里的一个新映射，指向 A 的那块物理显存
cudaIpcCloseMemHandle(p);
```

**传的是那个 64 字节的不透明 handle，不是指针值。** B 拿到的是自己 VA 空间里的新映射
（本次恰好数值相同，那是确定性分配的巧合，不是保证）。

限制：仅 Linux；同一台机器同一张卡（或已建立 peer access 的卡）；不支持 managed memory。
**NCCL、vLLM 的跨进程 KV cache 共享走的都是这条路。**

更现代的通用机制是 VMM API 的 `cuMemExportToShareableHandle`（支持 POSIX fd，配 MNNVL
还能跨节点用 fabric handle）。

### 6.3 即使共享了内存，kernel 也不能跨 context 真并发

> **A kernel from one CUDA context cannot execute concurrently with a kernel from another
> CUDA context.** The GPU may time slice to provide forward progress to each context.
> If a user wants to run kernels from multiple process simultaneously **on the SM**, one
> must enable **MPS**.

两个进程各跑 kernel，GPU 只能**时间片轮转**，不是真并发。要让多进程的 kernel 真正同时
占用 SM，必须开 **MPS**（它让多个进程共享一个 GPU 侧 context）。

这也是为什么推理框架普遍是**一张卡一个进程**：多进程共享一张卡，既有 IPC 的麻烦，又有
时间片轮转的开销。

**顺带订正一个流传很广的说法**：「一个 SM 在同一时刻只能为一个 kernel 服务」——这是错的，
它把 context 级的限制误写成了 kernel 级。准确的版本是上面那句文档：**同 context 内的多个
kernel 可以并发，甚至可以共驻同一个 SM**（第 12 章的 PDL 收益完全建立在这个能力上）；
**不同 context** 之间才不行。

---

## 7. 三层放在一起

| | 作用尺度 | 机制 | 能否共驻同一个 SM | 主要限制 |
|---|---|---|---|---|
| **PDL** | 同一 stream 上相邻 kernel | `griddepcontrol` | ✅ 可以（收益就来自这里） | smem 之和 ≤ 228KB、carveout 要一致 |
| **Stream** | 不同 stream 之间 | 软件 FIFO + 硬件工作队列 | ✅ 可以 | 剩余 SM 资源、NULL stream、队列数 |
| **跨进程** | 不同 CUDA context | MPS（否则时间片轮转） | ❌ 默认不行 | 地址空间隔离，共享要走 IPC |

**先确认瓶颈在哪一层，再选机制。** 三者不叠加也不互斥：

- kernel 太小、一个 step 几百个 kernel 排着队 → 看 PDL（第 12 章）
- 有多路独立的小工作流、GPU 没被打满 → 看 stream
- 多个服务进程要挤一张卡 → 只有 MPS，且要接受它的代价

---

## 8. 微基准设计

源码：`target_p:/tmp/{stream_probe,prio_probe,ipc_probe}.cu`，
`ecs:~/nvfix/{stream_probe,prio_probe}.cu`（同一份）。

```bash
# GB200
ssh target_p && cd /tmp
/usr/local/cuda/bin/nvcc -O3 -arch=sm_100 -o stream_probe stream_probe.cu
./stream_probe                                   # 实验 0/A/B/C/D/E
NSTREAM=8 KPS=6 CUDA_DEVICE_MAX_CONNECTIONS=1 ./stream_probe    # 工作队列扫描
./prio_probe                                     # 优先级四组对照
rm -f ipc_handle.bin; (setsid ./ipc_probe producer &); sleep 5
./ipc_probe consumer_raw; ./ipc_probe consumer_ipc

# H20 —— 注意 PATH 上的 nvcc 是 12.0，且需要驱动 workaround
ssh ecs && cd ~/nvfix
/usr/local/cuda/bin/nvcc -O3 -arch=sm_90 -o stream_probe stream_probe.cu
LD_LIBRARY_PATH=$HOME/nvfix ./stream_probe
```

### 8.1 三个设计点

![实验方法](/imgs/cuda-cute-ctx-method.svg)

**① 用 dynamic smem 钉住 1 CTA/SM。** 要回答「两个 kernel 能不能同时占用 GPU」，需要一个
能精确控制「这个 kernel 占多少个 SM」的旋钮。正常情况下这很难数清：GB200 每 SM 能放 2048
线程，block 是 256 线程，理论上一个 SM 能塞 8 个 CTA，那 `grid=38` 到底占了 5 个 SM 还是
38 个？取决于调度器。

用 shared memory 锁死：

```text
GB200 每 SM 有 228KB smem
给每个 CTA 申请  228/2 + 4 = 118KB
118 × 2 = 236KB > 228KB  →  一个 SM 最多只放得下 1 个 CTA
```

程序里用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 确认输出 `occupancy=1 CTA/SM`。
于是得到干净的等价关系：**`grid = N` ⟺ 占用 N 个 SM**。（256 个线程在这里是无关变量 ——
smem 才是卡住的约束。）

**② 用纯自旋 kernel。**

```cpp
long long t0 = clock64();
while (clock64() - t0 < cyc) { }
```

三个好处：时长完全可预测（不受缓存命中率、访存排队影响）；**不消耗 HBM 带宽**，于是把
「调度问题」和「带宽争用问题」彻底解耦 —— 变慢就只可能是没并发；`cyc` 按时钟频率折算成
0.5 ms，跨机器可比。

对比第 12 章测 PDL 时反而**故意**让 prolog 吃满 HBM —— **测什么就让什么成为唯一变量。**

**③ 并发度指标。**

```text
sum   = Σ(每个 kernel 自己的时长)        ← 总"kernel 工时"
span  = max(所有结束) - min(所有开始)     ← 这批活的墙上跨度
并发度 = sum / span
```

完全串行时 `span = sum` → 1.0；N 个完全重叠时 `span = dur, sum = N·dur` → N。

不用墙上时间是因为它包含 CPU 提交与同步开销；`span` 在**设备上**用 `%globaltimer` 测。
`clock64()` 是每个 SM 各自的计数器，跨 SM / 跨 kernel 不可比，只能用于 CTA 内的时长测量
（自旋循环）。

### 8.2 踩过的四个坑

1. **每个 stream 只放 1 个 kernel，测不出工作队列不足的假依赖。** 第一次做 §4.3 的实验，
   `CONN=1` 下 4 个 stream 仍然 3.94 路并发，我一度以为这个机制不存在。原因是假依赖需要
   **同一 stream 内有多个 kernel**：单队列里 `A2` 等 `A1`，排在后面的 `B1` 才会被连坐。
   4 个 stream 各 1 个 kernel，提交顺序彼此无依赖，单条队列照样连续发射。改成
   8 streams × 6 kernels 并区分深度/广度优先提交，效应立刻出来了。

2. **给 NULL stream 的填充 kernel 传了 0 字节 dynamic smem**，而 kernel 里要写
   `extern __shared__` → 越界，`illegal memory access`。改用一个不带 smem 的独立小 kernel。

3. **跨进程裸地址实验里 kernel 解引用「成功」了，差点误判成可以跨进程访问。** 必须让
   consumer 先用**不同的 pattern** 填自己的显存，再去读那个地址，读到 `0xBB00` 才能证明
   「同一数值地址指向本进程自己的内存」。**一个不报错的结果比报错的结果更需要追问。**

4. **`cudaFuncSetAttribute` 设的 carveout 是持久函数属性，会跨实验泄漏。** 前一组实验的
   MISMATCH 配置泄漏进后一组，导致同样参数测出 12914ns vs 16070ns。每个配置都要显式设。

---

## 9. 实践清单

```text
想让两个 kernel 并发：
  1. 它们加起来占的 SM 有没有超？   → 超了就没戏，多开 stream 无用
  2. 中间有没有 NULL stream 操作？   → 用 cudaStreamNonBlocking 建流
  3. 同一 stream 内有多个 kernel 吗？→ Hopper 上注意提交顺序与 CONN，Blackwell 上可忽略
  4. 需要 QoS 插队？                → 优先级没用，考虑 Green Contexts

跨进程共享显存：
  · 绝不要传裸指针值 —— 不会崩，会静默读错
  · 用 cudaIpcGetMemHandle / cudaIpcOpenMemHandle 传 handle
  · kernel 仍不能跨 context 并发，要真并发得开 MPS

启动开销：
  · 每设备一个 context，各要两三百毫秒 → 服务启动时预热，别让首个请求付
```

---

## 10. 引用与核实状态

| 内容 | 状态 |
|---|---|
| §3.2 / §4.2 / §5.2 / §6.3 的文档引文 | ✅ 逐句取自 CUDA C++ Programming Guide **13.0** 归档单页版 |
| §1–§6 全部性能数字 | ✅ **本文实测**，`stream_probe.cu` / `prio_probe.cu` / `ipc_probe.cu` 可复现 |
| PyTorch stream 为 non-blocking | ✅ 实测 `cuStreamGetFlags` = 1（torch 2.11.0a0，target_p） |
| §4.3 两代架构差异的**归因** | ⚠ 只有两个数据点，无法区分架构 / CUDA 版本 / 驱动版本 |
| §5.2「优先级完全无效」 | ⚠ 仅覆盖「每个 kernel 占满 GPU」这一情形；其他资源配比下未测 |
| Green Contexts 作为替代方案 | ⚠ 未实测，仅据文档 |

参考资料：

- [CUDA C++ Programming Guide 13.0（归档单页版）](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)
  —— 现行版拆成了多页，`02-basics/asynchronous-execution.html` 等路径当前 404，归档单页版
  更适合全文检索
- [CUDA Runtime API — Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
- [CUDA 环境变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)
- [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html)
- 本系列 [第 12 章：PDL](/notes/2026/08/26/2026-08-26-cuda-cute-nvidia-learning-12-pdl-programmatic-dependent-launch/)、
  [第 1 章：CUDA 执行模型](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-01-cuda-execution-model/)、
  [第 2 章：SM / CUDA Core / Tensor Core](/notes/2026/07/29/2026-07-29-cuda-cute-nvidia-learning-02-sm-cuda-core-tensor-core/)
