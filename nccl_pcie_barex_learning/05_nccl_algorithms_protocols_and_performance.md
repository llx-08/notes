# 05. NCCL 算法、协议与性能模型

## 1. Algorithm 与 Protocol 不是一回事

- Algorithm：rank 之间如何组织通信步骤，如 Ring、Tree。
- Protocol：每条连接上如何用 buffer、flag 和 step 传输 chunk，如 Simple、LL、LL128。
- Transport：连接落到 P2P、SHM 或 NET。

一个 AllReduce 可以是：

```text
Ring algorithm + Simple protocol + NET transport
```

也可以是：

```text
Tree algorithm + LL protocol + P2P transport
```

## 2. Ring AllReduce

Ring AllReduce 通常分两段：

1. ReduceScatter：每个 rank 最终持有一段已归约数据；
2. AllGather：交换各段，所有 rank 得到完整结果。

对于 `N` 个 rank，每个 rank 发送/接收的数据量近似：

```text
2 × (N - 1) / N × message_size
```

优点：

- 大消息带宽利用率高；
- 每一步只有固定邻居；
- 容易形成流水线。

缺点：

- 步数约 `2(N-1)`；
- 小消息延迟随 rank 数增长明显；
- 任一弱链路进入 ring 都会限制整体。

### 2.1 四个 rank 的手算例子

每个 rank 有 4 个元素：

```text
rank 0: [ 1,  2,  3,  4]
rank 1: [10, 20, 30, 40]
rank 2: [ 5,  6,  7,  8]
rank 3: [ 2,  4,  6,  8]
```

最终 AllReduce(sum)：

```text
[18, 32, 46, 60]
```

为了理解 ring，把向量切成 4 个 chunk，每个 chunk 先沿 ring 走
`N-1=3` 步完成 reduce-scatter。结束时每个 rank 只拥有一个“已经求和”的 chunk：

```text
rank 0 拥有 [18]
rank 1 拥有 [32]
rank 2 拥有 [46]
rank 3 拥有 [60]
```

再走 3 步 all-gather，大家交换这四个结果，最终都得到完整向量。

如果完整消息是 1 GiB、`N=4`，每 rank 近似通信量：

```text
2 × (4-1)/4 × 1 GiB = 1.5 GiB
```

不是 3 GiB，因为每步只发送约 `1/N` 的 chunk。

## 3. Tree AllReduce

Tree 沿树向上 reduce，再向下 broadcast。深度约 `O(log N)`。

优点：

- 小/中消息延迟较低；
- 步数随 rank 数增长慢。

代价：

- 根和中间节点链路压力不同；
- 大消息带宽未必像 ring 一样均衡；
- 拓扑映射不佳时会重复经过共享路径。

## 4. CollNet、NVLS 与 PAT

- CollNet：利用支持 collective offload 的网络能力。
- NVLS/NVLSTree：利用 NVLink SHARP/NVSwitch 归约能力。
- PAT：Parallel Aggregated Trees，针对特定 collective/scaling 场景。

它们是否可用取决于硬件、plugin、NCCL 版本和 collective 类型，不应通过名称判断一定更快。

## 5. Simple、LL、LL128

### 5.1 Simple

面向吞吐。payload 占比高，适合大消息；通常需要更大的 step/chunk，启动固定开销相对高。

### 5.2 LL

低延迟协议，把 data 与 flag 组织成更容易细粒度推进的形式。小消息延迟好，但有效 payload 比例低，NCCL 源码中的流量模型会对 LL 计入额外 traffic。

### 5.3 LL128

在支持的平台上平衡延迟与带宽，但有更严格的硬件和路径约束。官方文档明确不建议随意强制启用不受支持的 LL128，否则可能出现数据错误。

当前官方允许用 `NCCL_PROTO=LL,LL128,Simple` 约束协议集合，但推荐让 NCCL 自动选择，除非在定位问题。

## 6. Chunk、Slice、Step

消息不是一次性放进连接：

```text
message
  → 分配给多个 channel
  → 每 channel 拆成 chunk
  → chunk 按 algorithm step 前进
  → protocol 再按 slice/step buffer 协调 producer/consumer
```

关键效果：

- pipeline 隐藏网络 RTT；
- 多 channel 并行占满多 rail；
- 固定 buffer 可以循环使用；
- head/tail 防止 producer 覆盖 consumer 尚未处理的数据。

## 7. 性能模型

可用简化的 α-β 模型：

```text
T ≈ steps × α + traffic_bytes × β
```

- `α`：每步固定延迟，包括 kernel、proxy、doorbell、网络 RTT。
- `β`：每字节成本，即有效带宽倒数。

例：假设一次 step 固定开销 `α=5 μs`，有效带宽 `B=25 GB/s`，传
1 MiB 的数据成本约：

```text
1 MiB / 25 GB/s ≈ 41.9 μs
```

若算法需要 6 步，极简估算：

```text
T ≈ 6×5 μs + 41.9 μs = 71.9 μs
```

真实算法会 pipeline，多步数据传输可以重叠，因此这个加法不是性能预测器；
它用于判断固定延迟和字节传输谁更可能占主导。

Ring 与 Tree 的取舍：

| 场景 | 主要矛盾 | 常见更优方向 |
|---|---|---|
| 小消息 | `steps × α` | Tree、LL |
| 大消息 | `traffic × β` | Ring、Simple、多 channel |
| 多节点 | 网络/rail | topology-aware graph、GDR |
| 节点内 NVSwitch | collective offload | NVLS |

## 8. Bus bandwidth 与 algorithm bandwidth

`nccl-tests` 常同时报告：

- algbw：应用 payload / 时间；
- busbw：按 collective 理论链路流量归一化。

AllReduce ring 常用换算：

```text
busbw = algbw × 2 × (N - 1) / N
```

不能直接拿 algbw 与 PCIe 单链路理论值比较，因为：

- collective 同时使用多条链路；
- 同一数据经过多步；
- 可能使用 NVLink/NIC 多 rail；
- 单/双方向统计口径不同。

## 9. 分层定位性能

### 9.1 先确认硬件上限

- PCIe negotiated width/speed；
- NVLink 状态；
- NIC link speed；
- GPU/NIC NUMA 与 topology。

### 9.2 再确认 transport

- P2P 是否启用；
- IB 还是 socket；
- GDR 是否启用；
- multi-rail 是否实际使用。

### 9.3 再确认 tuning

- algorithm/protocol；
- channel 数；
- chunk size；
- message size 分布；
- 是否被计算 stream 或其他 collective 串行化。

### 9.4 最后才调环境变量

强制变量适合 A/B 验证：

```bash
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

每次只改变一个维度，并保存 NCCL INFO 日志。

## 10. `nccl-tests` 基础实验

```bash
./all_reduce_perf -b 8 -e 1G -f 2 -g 8
./all_gather_perf -b 8 -e 1G -f 2 -g 8
./sendrecv_perf -b 8 -e 1G -f 2 -g 2
```

建议记录：

- size；
- out-of-place/in-place；
- time/algbw/busbw；
- error count；
- NCCL topology/NET 日志；
- GPU/NIC 时钟和链路状态。

## 11. 与 blade-kvt 的性能差异

NCCL 针对规则 collective：

- rank graph 稳定；
- payload 易均匀切分；
- GPU kernel 可持续流水。

blade-kvt KV 发送则是：

- request/worker 目标动态；
- block 离散；
- P/D TP 可能不同；
- layer ready 与模型 forward 重叠；
- 需要远端地址协议和 request 完成通知。

所以 blade-kvt 使用 Barex one-sided write 是符合负载形态的选择，并不是“NCCL 性能不够”这一单一原因。

## 12. 跟源码走一遍：NCCL 怎样选择 Algorithm 和 Protocol

本节固定阅读 **NCCL 2.30.7，commit
`5067397c2676d5aed50042fc39e5c8ee96eb0027`**。源码主链是：

```text
ncclPrepareTasks()
  │
  └─ ncclGetAlgoInfo(task)
       │
       ├─ initCollCostTable()
       │    所有 algorithm × protocol 组合先标成不可用
       │
       ├─ updateCollCostTable()
       │    对硬件/collective 支持的组合估算时间
       │      └─ ncclTopoGetAlgoTime()
       │
       ├─ 可选：tuner plugin 修改 cost table / channel 建议
       │
       └─ topoGetAlgoInfo()
            ├─ 选择估算时间最小的 algorithm + protocol
            ├─ 根据消息大小缩减 channel 数
            └─ 决定 nWarps
```

### 12.1 选择对象是二维组合，不是先选算法再选协议

成本表近似是：

| Algorithm \ Protocol | Simple | LL | LL128 |
|---|---:|---:|---:|
| Ring | 时间 | 时间 | 时间 |
| Tree | 时间 | 时间 | 时间 |
| CollNetDirect | 时间或不可用 | 时间或不可用 | 时间或不可用 |
| CollNetChain | 时间或不可用 | 时间或不可用 | 时间或不可用 |
| NVLS | 时间或不可用 | 时间或不可用 | 时间或不可用 |
| NVLSTree | 时间或不可用 | 时间或不可用 | 时间或不可用 |
| PAT | 时间或不可用 | 时间或不可用 | 时间或不可用 |

`topoGetAlgoInfo()` 遍历二维表，选取最小非负值：

```cpp
if (table[a][p] >= 0.0 && table[a][p] < minTime) {
  algorithm = a;
  protocol = p;
  minTime = table[a][p];
}
```

所以真实决策是：

```text
argmin time[algorithm][protocol]
```

而不是：

```text
先选 Ring，再单独判断 Simple/LL
```

这很重要，因为同一 algorithm 换 protocol 会改变带宽、延迟和有效 payload；
同一 protocol 换 algorithm 又会改变逻辑 step 和物理路径。

### 12.2 为什么有些格子根本不参与比较

`updateCollCostTable()` 会先按能力过滤，例如：

- 没有 CollNet 支持，就跳过 CollNetDirect/Chain；
- 不支持 NVLS，或 collective 类型不匹配，就跳过 NVLS/NVLSTree；
- 某些 algorithm 对 local rank 数有上限；
- collective、归约类型或数据类型不适合某组合；
- 环境变量可能把某 algorithm/protocol 排除。

不可用组合保持 `NCCL_ALGO_PROTO_IGNORE`，不会因为理论公式算出“小时间”就被误选。

这是 NCCL 自动 tuning 的两阶段思想：

```text
先判断 correctness/capability：能不能用
再比较 performance model：哪个预计更快
```

### 12.3 源码中的时间模型是什么

`src/graph/tuning.cc::ncclTopoGetAlgoTime()` 先读取 communicator 初始化时得到的：

```cpp
bw  = comm->bandwidths[collective][algorithm][protocol];
lat = comm->latencies [collective][algorithm][protocol];
```

最终核心表达式是：

```cpp
time = lat * latCount + nBytes / (1000 * bw);
```

在这套表的单位约定下，结果是估算时间。概念上仍可写成：

```text
estimated_time ≈ fixed_latency + bytes / estimated_bandwidth
```

但源码并非只有这一条教科书公式，还包含修正：

- Tree 在某些中等消息大小上的静态 correction factor；
- 多节点 Ring + Simple AllReduce 的 plateau latency 修正；
- aggregation 中 Ring 和 Tree 对 `numPipeOps` 的延迟计数不同；
- 初始化时针对平台、graph、节点数和协议填入不同 bandwidth/latency；
- tuner plugin 可以修改 cost table。

因此文档前面的 α-β 模型适合建立直觉，不能替代 NCCL 的实际 cost table。

### 12.4 `numPipeOps` 为什么会影响估算

一次 `ncclGroupStart/End` 中可能聚合多个 collective，或 planner 能把多个操作流水化。
`numPipeOps` 表示这次估算要考虑多少个 pipelined operation。

当前源码中：

```text
Ring：latCount = numPipeOps
Tree：latCount = ceil(numPipeOps / 每个 device batch 可容纳的 collective 数)
```

它反映的是：多个聚合操作并不一定简单地把固定延迟乘同一个系数，算法的 batch/pipeline
能力也会改变模型。

## 13. 选择 Channel 数和 GPU 线程数

选出 algorithm/protocol 后，`topoGetAlgoInfo()` 还没有结束。它继续决定：

```text
task->nMaxChannels
task->nWarps
```

### 13.1 为什么小消息不应该盲目使用全部 Channel

Ring/Tree 的简化逻辑是：

```cpp
while (nBytes < nChannels * nThreads * threadThreshold)
  nChannels--;
```

直觉是：如果消息很小，切给太多 channel 后，每个 CUDA block 只得到很少的数据，
额外 block、同步和连接开销反而占主导。

举例，不代表源码中的真实阈值：

```text
1 MiB / 16 channels = 每 channel 64 KiB
4 KiB / 16 channels = 每 channel 256 B
```

后者启动 16 个 channel 的收益通常很小。NCCL 会随消息大小缩减 channel，必要时继续
缩减线程数。

### 13.2 `nWarps` 最终怎样进入 Kernel

链路如下：

```text
topoGetAlgoInfo()
  → ncclTaskColl.nWarps
  → ncclDevWorkColl.nWarps
  → plan 计算 threadPerBlock
  → ncclLaunchKernel(grid, block)
```

Simple Ring 会额外加入一个 warp 用于同步；Tree 在当前版本中使用最大线程数。
因此 Nsight Systems 中看到的 block size 不是固定的 NCCL 常量，而是算法、协议、
消息大小和版本共同决定的结果。

### 13.3 Channel 不等于 CUDA Stream

一个常见误解是“8 channel 就是 8 CUDA stream”。实际上：

```text
CUDA stream：kernel 的提交和依赖顺序
NCCL channel：collective 数据切分、连接和 graph 并行度
CUDA block：本次 kernel 中执行某个 channel 工作的 CTA
```

普通 launch 路径里，`grid.x` 由本 plan 的 channel mask 中置位数决定；一个参与本次
plan 的 channel 通常由一个 block 推进。多个 block 仍可以属于同一个 NCCL kernel
和同一个 CUDA stream。

## 14. 跟源码读懂 Ring AllReduce 的每一步

设备端入口位于 `src/device/all_reduce.h::runRing()`。先取得：

```cpp
ring->prev
ring->next
```

并构造：

```cpp
Primitives<T, RedOp, FanSymmetric<1>, Direct=1, Proto, ...>
```

`FanSymmetric<1>` 表示一个 recv 邻居和一个 send 邻居，正好对应 Ring 的前驱和后继。

假设 `nranks = k`，一次循环的源码步骤是：

```text
ReduceScatter 部分

step 0:
  directSend
  把自己的某个 chunk 发给 next

中间 k-2 步:
  directRecvReduceDirectSend
  从 prev 收到部分结果
  与本 rank 对应 chunk 做 reduction
  把新结果继续发给 next

第 k-1 步:
  directRecvReduceCopyDirectSend
  做最后一次 reduction
  写入本 rank recvbuff 中的最终 chunk
  同时把最终 chunk 发给 next，开始 AllGather

AllGather 部分

中间 k-2 步:
  directRecvCopyDirectSend
  收到已经归约完成的 chunk
  拷入本 rank recvbuff
  转发给 next

最后一步:
  directRecv
  收到最后缺少的结果 chunk 并写入 recvbuff
```

### 14.1 为什么代码里不是明显的两个独立 Kernel

教科书常画：

```text
ReduceScatter → AllGather
```

源码把两段写在同一个 `runRing()` 循环中，并在
`directRecvReduceCopyDirectSend()` 那一步衔接。好处是：

- 不需要在两段中间退出并重启 kernel；
- 一个 chunk 完成 reduction 后可以立刻进入传播；
- 更容易形成跨 chunk 的流水线。

逻辑上仍是两阶段，执行上不必是两个完全隔离的程序。

### 14.2 Primitive 名字怎样读

不要把 `directRecvReduceCopyDirectSend` 当成一个神秘指令。把词拆开即可：

```text
directRecv   从前驱连接接收
Reduce       与本地数据归约
Copy         把结果写入目标 buffer
directSend   向后继连接发送
```

`Primitives` 模板根据 `Proto` 实例化 Simple、LL 或 LL128 的 buffer/flag/step 操作，
根据 connector 落到 P2P、SHM 或 NET 路径。于是 Ring 算法代码不必分别实现：

```text
Ring-over-NVLink
Ring-over-PCIe
Ring-over-RDMA
Ring-over-Socket
```

算法描述“做什么”，Primitive + connector 描述“怎样沿这条 edge 做”。

### 14.3 最后一个 rank 会不会成为中心节点

不会。每个 rank 同时执行相同结构的代码，只是 `ringIx`、`prev`、`next` 不同。
在稳态流水线中，每条 Ring edge 都在传不同 chunk，因而带宽压力较均匀。

## 15. Tree AllReduce 在代码中是什么样

`runTreeUpDown()` 使用：

```text
tree->up      父节点
tree->down[]  子节点
```

向上 reduce 阶段构造：

```text
FanAsymmetric<最大子节点数, 1>
```

向下 broadcast 阶段构造：

```text
FanAsymmetric<1, 最大子节点数>
```

节点角色不同，调用的 Primitive 也不同：

| 节点角色 | Reduce Up | Broadcast Down |
|---|---|---|
| 叶子 | `directSend` | `directRecv` |
| 中间节点 | `directRecvReduceDirectSend` | `directRecvCopyDirectSend` |
| 根 | `directRecvReduceCopy` | `directSendFromOutput` 一类操作 |

因此 Tree 的“深度约为 `log N`”不是说所有节点串行等待 `N` 次；同一层的不同分支可以并行。
但父节点要汇聚多个 child，拓扑映射不好时会让共享链路或中间节点成为瓶颈。

## 16. Simple、LL、LL128 在源码中的具体差别

协议公共接口在 `src/device/primitives.h`：

```cpp
ProtoSimple
ProtoLL
ProtoLL128
```

三个类型都向算法层提供：

```text
Id
calcBytePerStep()
calcBytePerGrain()
MaxGroupWidth
```

所以 `runRing<T, RedOp, Proto>()` 可以不关心协议的具体 flag layout。

### 16.1 Simple：一个 step 几乎都可放数据

源码计算：

```cpp
buffSizes[SIMPLE] / NCCL_STEPS
```

作为一个 step 的 data bytes。Simple 的同步元数据不以内嵌一半 payload 的方式占用
每条数据 line，因而大消息时有效 payload 比例高。

其实现主要在：

```text
src/device/prims_simple.h
```

它还会根据 `NCCL_DIRECT_NIC`、direct pointer、registered buffer 等 flags 选择具体
load/store 路径。

### 16.2 LL：为什么有效容量只有一半

`ProtoLL::calcBytePerStep()` 明确返回：

```cpp
buffSizes[LL] / NCCL_STEPS / 2
```

源码注释是 `Half is data`。一个 16-byte LL line 里只有 8 bytes 是 data，其余用于
flag。可以把 line 概念化为：

```text
16-byte LL line
┌──────────── 8 B ────────────┬──────────── 8 B ────────────┐
│ payload                     │ flag / 同步信息              │
└─────────────────────────────┴─────────────────────────────┘
```

flag 与数据一起到达，消费者可以细粒度判断某 line 是否准备好，减少对较大独立同步块的
等待；代价是相同物理传输容量只能携带约一半用户 payload。

这正是 LL “低延迟但大消息带宽效率较低”的代码依据，而不只是经验结论。

### 16.3 LL128：15/16 的有效 payload

源码常量：

```cpp
NCCL_LL128_LINESIZE = 128 bytes
NCCL_LL128_LINEELEMS = 16 个 uint64_t
NCCL_LL128_DATAELEMS = 15 个 uint64_t
```

概念布局：

```text
128-byte LL128 line
┌──────────────────── 15 × 8 B data ────────────────────┬─ 8 B flag ─┐
│                     120 B payload                      │    flag     │
└────────────────────────────────────────────────────────┴─────────────┘
```

所以 step 的有效数据量为：

```cpp
(buffer / NCCL_STEPS) * 15 / 16
```

它的 payload 比例明显高于 LL，同时保留 line-level flag 机制，但对架构、内存操作和
传输路径有更严格要求。这也是不能因为看到 `15/16 > 1/2` 就在所有机器上强制 LL128
的原因。

### 16.4 三种协议的源码对照

| Protocol | 每 step 有效数据量（概念） | flag 组织 | 常见倾向 |
|---|---:|---|---|
| Simple | `buffer / steps` | payload line 外协调 | 大消息吞吐 |
| LL | `buffer / steps / 2` | 8 B data 配 8 B flag | 小消息低延迟 |
| LL128 | `buffer / steps × 15/16` | 120 B data 配 8 B flag | 延迟/带宽折中，受平台约束 |

“有效数据量”不是用户 tensor 的总大小，而是固定 connection buffer 的一个循环槽位
能承载多少用户 payload。

## 17. `calcCollChunking()`：消息如何变成 Pattern、Chunk 与 Proxy Step

algorithm/protocol 选好后，`src/enqueue.cc::calcCollChunking()` 继续计算运行布局。

### 17.1 先把 collective + algorithm 映射为 Pattern

以 AllReduce 为例：

```text
NVLS           → ncclPatternNvls
NVLSTree       → ncclPatternNvlsTree
CollNetDirect  → ncclPatternCollnetDirect
CollNetChain   → ncclPatternCollnetChain
Tree           → ncclPatternTreeUpDown
其他（Ring）   → ncclPatternRingTwice
```

`RingTwice` 就对应 ReduceScatter + AllGather 两圈逻辑步骤。

### 17.2 基础 stepSize 和 chunkSize

源码先算：

```cpp
stepSize  = comm->buffSizes[protocol] / NCCL_STEPS;
chunkSize = stepSize * chunkSteps;
```

然后按协议修正 payload：

```text
LL：    chunkSize /= 2
LL128： chunkSize = chunkSize / 16 × 15
```

再按 algorithm、消息大小、channel 数、NVLS/CollNet 深度等继续调小或约束 chunk。
最后还要按 protocol grain size 对齐。

### 17.3 Chunk 为什么不能无限大

大 chunk：

- 每次提交/flag 开销占比低；
- 更利于带宽；
- 但流水粒度粗，首包和尾包延迟大；
- channel/step 数不足时难以并行。

小 chunk：

- pipeline 更快启动；
- 可在多 channel/rail 上交错；
- 但每 chunk 固定开销和同步比例上升。

`calcCollChunking()` 的大量分支，本质是在特定 algorithm 和拓扑下寻找这个折中。

### 17.4 Step buffer 为什么可以循环复用

connection buffer 被切为 `NCCL_STEPS` 个槽：

```text
logical step:  0 1 2 3 4 5 6 7 8 9 ...
physical slot: 0 1 2 3 4 5 6 7 0 1 ...
```

只有消费者推进 `head`、归还 credit 后，生产者才能复用相应槽位。于是 buffer 不需要
和整个 tensor 一样大，也不会覆盖仍在传输或尚未被 GPU 消费的数据。

对于 NET transport，proxy 使用相同逻辑 step 推导：

```text
slot = step % NCCL_STEPS
```

并通过 `connFifo`、head、tail 与 GPU kernel 对齐。这正是算法切块、GPU Primitive
和网络 Proxy 能组成一条流水线的连接点。

## 18. 从一次 1 GiB AllReduce 看所有参数怎样串起来

假设 8 rank、2 个节点、每节点 4 GPU，用户调用：

```cpp
ncclAllReduce(send, recv, 1 GiB / sizeof(float),
              ncclFloat, ncclSum, comm, stream);
```

host 侧的决策链可概括为：

```text
1 GiB float AllReduce
  │
  ├─ ncclInfo：记录 API 参数
  ├─ ncclTaskColl：记录 collective task 和 trafficBytes
  ├─ cost table：
  │    比较 Ring/Simple、Tree/LL128、NVLS/... 的估算时间
  ├─ 假设选择 Ring + Simple
  ├─ 根据消息大小与 topology 决定 nChannels，例如 8
  ├─ calcCollChunking：
  │    pattern=RingTwice
  │    算 chunkSize、proxy nsteps
  ├─ scheduler：
  │    将 task 分配进 8 个 channel 的 work batch
  └─ launch：
       一个 NCCL kernel，grid 中有对应 channel blocks
```

运行时：

```text
每个 block/channel
  ├─ 负责总消息的一部分
  ├─ 再拆成多个 chunk
  ├─ 每个 chunk 沿 Ring 做 2×(8-1) 个逻辑阶段
  ├─ P2P edge 通过 GPU peer connection 推进
  └─ NET edge 与 Proxy/RNIC 通过 step buffer 流水推进
```

总逻辑步骤很多，不表示总时间等于“单 chunk RTT × 所有 chunk × 所有 step”的完全串行
相加。不同 channel、不同 chunk、GPU 计算归约与 NIC 传输会形成流水重叠；最终性能取决于
最慢 stage、可维持的 in-flight 数据量和共享物理瓶颈。

## 19. 如何用日志和强制变量验证源码理解

先自动选择，保存完整日志：

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING,COLL \
./all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

然后只改变一个因素做 A/B：

```bash
# 只允许 Ring
NCCL_ALGO=Ring ./all_reduce_perf ...

# 只允许 Tree
NCCL_ALGO=Tree ./all_reduce_perf ...

# 固定 Ring，分别比较协议
NCCL_ALGO=Ring NCCL_PROTO=Simple ./all_reduce_perf ...
NCCL_ALGO=Ring NCCL_PROTO=LL     ./all_reduce_perf ...
```

推荐表格：

| size | algo | proto | channels | time | algbw | busbw | NET/GDR | error |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 8 KiB | auto | auto | | | | | | |
| 8 KiB | Ring | Simple | | | | | | |
| 8 KiB | Tree | LL | | | | | | |
| 1 GiB | auto | auto | | | | | | |

如果强制组合失败或性能差，不代表源码“没有走你的配置”。也可能是：

- 组合在该 collective/硬件上不可用；
- 强制后只剩一个远差于 cost model 最优值的组合；
- transport、NUMA 或 rail 才是瓶颈，algorithm 变化不解决问题；
- 消息没有大到足以摊薄固定开销；
- LL128 路径不受支持。

## 20. 推荐源码阅读顺序

初学者不建议从 `prims_simple.h` 的模板细节开始。按以下顺序更容易：

```text
第一遍：算法骨架
  src/device/all_reduce.h
  看 runRing / runTreeUpDown 使用哪些 Primitive

第二遍：选择逻辑
  src/enqueue.cc
  看 ncclGetAlgoInfo / topoGetAlgoInfo / calcCollChunking

第三遍：性能模型
  src/graph/tuning.cc
  看 ncclTopoGetAlgoTime 和 bandwidth/latency 表

第四遍：协议接口
  src/device/primitives.h
  比较 ProtoSimple / ProtoLL / ProtoLL128

第五遍：协议实现
  src/device/prims_simple.h
  src/device/prims_ll.h
  src/device/prims_ll128.h

最后：结合 Transport/Proxy
  src/include/device.h
  src/transport/net.cc
  src/proxy.cc
```

每一遍只回答一个问题：

```text
算法骨架：    chunk 要经过哪些 recv/reduce/copy/send 步骤？
选择逻辑：    为什么是这个 algorithm/protocol/channel 数？
性能模型：    NCCL 认为哪个组合耗时更小？
协议实现：    step buffer 和 flag 怎样保证生产消费同步？
Transport：   这个 send/recv 最终落到 NVLink、PCIe、SHM 还是网络？
```

## 21. 自检

1. Ring 为什么大消息强，小消息可能弱？
2. LL 为什么低延迟但有效带宽不一定高？
3. 为什么 busbw 可能高于单个 PCIe link 的理论带宽？
4. 强制 `NCCL_PROTO=LL128` 有什么风险？
5. NCCL 为什么要比较 algorithm × protocol 二维组合，而不是独立选择？
6. `directRecvReduceCopyDirectSend` 的四段分别做什么？
7. 为什么 ReduceScatter 和 AllGather 在逻辑上是两段，在源码中却可位于一个 kernel 循环？
8. LL 的 step buffer 为什么只有约一半可承载 payload？
9. LL128 的 15/16 从哪两个源码常量得到？
10. 为什么小消息可能主动减少 channel 数？
11. `chunkSize`、`NCCL_STEPS` 和 proxy 的 logical step 怎样关联？
12. α-β 模型为什么只能建立直觉，不能精确复现 NCCL 的选择？

## 参考

- [NCCL Environment Variables：`NCCL_ALGO` 与 `NCCL_PROTO`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [NCCL `src/enqueue.cc`：算法/协议选择、channel tuning、chunking](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/enqueue.cc)
- [NCCL `src/graph/tuning.cc`：`ncclTopoGetAlgoTime`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc)
- [NCCL `src/device/all_reduce.h`：Ring/Tree AllReduce kernel](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/all_reduce.h)
- [NCCL `src/device/primitives.h`：Simple/LL/LL128 协议接口](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/primitives.h)
- [NCCL Simple Primitive](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/prims_simple.h)
- [NCCL LL Primitive](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/prims_ll.h)
- [NCCL LL128 Primitive](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/prims_ll128.h)
