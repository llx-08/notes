---
title: "10 ATA 五篇串讲：PD 分离、GDR、传输优化与 asyncio"
date: 2026-08-26
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 10 ATA 五篇串讲：PD 分离、GDR、传输优化与 asyncio

> 本章把盏一的五篇 ATA 文章放进同一条技术主线：PD 分离的数据面如何从 KVT 1.0
> 演进到 Hybrid Connector 与 KVT 2.0，GDR/RDMA 如何保证正确性并优化小包性能，以及
> Python asyncio 的调度与 Task 生命周期为什么会直接影响控制面的可靠性。
>
> 这不是五篇文章的逐段转录，而是结合本系列 00–09 章得到的结构化学习笔记。文中的
> “文章实验”只代表原文给定软硬件、模型和配置下的结果，不能直接外推为所有环境的结论。

## 1. 五篇文章在整套系统中的位置

| 文章 | 主题 | 对本系列的补充 |
|---|---|---|
| 《转折中的 PD 分离》 | KVT 1.0、Hybrid Connector、KVT 2.0 的设计演进 | 解释架构为什么从“传输组件”走向“connector + backend” |
| 《PD 分离中的 kvcache 传输优化》 | RDMA 小包、Unsignaled WR、批量 post、QP 并行 | 补充 [04 KV 发送路径](/notes/2026/07/30/2026-07-30-vllm-blade-kvt-pd-learning-04-kv-send-path/) 的性能动机 |
| 《PD 分离中的 GDR》 | P→D GPU Direct RDMA、内存序、Decode 干扰实验 | 补充 [05 KV 接收与完成](/notes/2026/07/30/2026-07-30-vllm-blade-kvt-pd-learning-05-kv-receive-and-completion/) 的硬件语义 |
| 《一定要保存 asyncio.Task 对象啊!》 | Task 弱引用、GC、后台任务可靠性 | 解释 [07 Python 协程](/notes/2026/07/30/2026-07-30-vllm-blade-kvt-pd-learning-07-hybrid-asyncio-and-no-hang/) 中强引用集合的由来 |
| 《为什么协程》 | 1:1、M:1、M:N，协作式调度和运行时可观测性 | 为 Hybrid Connector 的异步控制面建立基础模型 |

五篇文章不是彼此独立的知识点。它们共同回答一个问题：

```text
怎样把 KV Cache 搬运做成调度主链路之外的异步旁路，
同时保证数据正确、性能足够高、异常不会静默 hang？
```

![五篇文章对应的系统演进](/imgs/vllm-pd-ata-architecture-evolution.svg)

## 2. PD 分离的基本请求链路

文章中的典型单请求模式可以抽象为：

```text
1. 请求 R 先到 D。
2. D 为 R 分配目标 KV Block：R.d_kv_blocks。
3. D 选择一个 P，调用 do_prefill(R, R.d_kv_blocks)。
4. P 执行 Prefill；每算完一层，就把该层 KV 通过 RDMA Write 写入 D。
5. P 产生 first token，并等待所有必要 KV 传输完成。
6. P 的 RPC 返回 first token 与传输结果。
7. D 把 transferred token 数记为已计算 token，开始 Decode。
```

这里最重要的不是“P 把一段连续内存复制到 D”，而是 D **先决定目标位置**，P 再把
每个源 Block 的一部分写到指定的目标 Block/offset。其效果类似于 D 命中了 prefix
cache：对普通 Scheduler 而言，前面的 prompt token 已有 KV，不需要了解 KV 是由远端
P 生成的。

控制面与数据面必须分开：

```text
控制面：request ID、P/D 实例、TP rank、Block ID、token 范围、完成和错误码
数据面：GPU 地址、offset、length、MR、QP、WQE、CQE、实际 KV 字节
```

这也是“零侵入”的核心：Scheduler 主要处理正常的 request/token/block 状态，Connector
负责把远端传输伪装成一次异步的 cache hit。

## 3. KVT 1.0：把传输拆成四层

早期 KVT 的目标是：

- 尽量不改 vLLM step 主流程；
- KV 传输与模型计算按 layer 重叠；
- P、D 都尽可能兼容 full CUDA Graph；
- 传输故障不能导致 Block 被提前复用。

### 3.1 接入层：把模型内存变成可传输元数据

实例启动时注册每层 KV Cache 的：

```text
layer base address
cache shape / stride
block_size_bytes
token_size_bytes
num_blocks
```

每个 step 根据 `SchedulerOutput` 生成 `KVTMeta`，记录：

- request 与源/目标实例；
- 源 Block 和目标 Block；
- 本 step 需要传输的 token 范围；
- TP worker/rank；
- 完成信号应该归属哪个请求。

Attention layer 计算完成后，Python 侧只在 CUDA stream 上 `record` 一个固定 Event。
它表示此前提交到该 stream 的本层计算已经排在 Event 之前，并不要求 Python 同步等待
GPU。

### 3.2 ParseBlock：把逻辑 token 映射为字节区间

传输层最终消费：

```cpp
struct IpcBlock {
    size_t src_off;
    size_t dst_off;
    size_t len;
};
```

语义是：

```text
把源 layer 的 [src_off, src_off + len)
写到目标 layer 的 [dst_off, dst_off + len)
```

ParseBlock 的价值在于隔离“模型 KV layout”与“网络传输”。例如：

```text
早期 layout：
(num_blocks, block_size, 2, num_heads, head_dim)

vLLM FA layout：
(2, num_blocks, block_size, num_kv_heads, head_size)
```

第一种布局中，一个 token 的 K/V 在 Block 内相邻；第二种布局中 K 与 V 可能位于两个
大分区。网络层不应该理解这些差别，ParseBlock 只需把一个逻辑 token 展开成两段或多段
`IpcBlock`。

#### 示例：同一个 token 的两种解析

假设每个 K 或 V token 占 128 B：

```text
布局 A：KV 相邻
token 7 → [src + 7×256, len=256]

布局 B：K/V 分区
token 7 → [K_base + 7×128, len=128]
          [V_base + 7×128, len=128]
```

上层仍然说“传 token 7”，Barex 仍然只看到地址和长度，只有 ParseBlock 需要变化。
这使 Qwen3-Next GDN 等新型 state/cache layout 可以通过新增 parser 接入，而不是重写
控制面与传输层。

### 3.3 控制层：按 layer 等待、提交与容错

控制层负责：

- 维护目标端连接；
- 等待每层 CUDA Event；
- 该层 ready 后调用传输层；
- 聚合每个 worker/rank 的完成；
- 在错误或 abort 时终止剩余任务并上报实际完成 token 数。

逐层发送形成流水线：

```text
GPU:      layer0 compute ─ layer1 compute ─ layer2 compute ─ ...
Network:          layer0 send ─── layer1 send ─── layer2 send
```

这比“整个 Prefill 完成后再一次性传 KV”更早暴露传输并减少 TTFT，但也增加了 completion
和资源生命周期管理的复杂度。

### 3.4 传输层：只负责 `Vec<IpcBlock>`

传输层不关心 request、token 或模型结构，只负责：

- 为目标建立 TCP/RDMA channel；
- 注册内存并管理 local/remote key；
- 把若干 IpcBlock 组织成 WR；
- 报告 transport completion 或错误。

这条边界必须牢记：

```text
CQE / transport completion
    ≠ 一个 TP worker 的 SEND_DONE
    ≠ 所有 TP worker 完成
    ≠ D 请求可以 Decode
```

## 4. Hybrid Connector：把计算与 KV load/save 解耦

文章认为早期社区 Connector 容易把未完成 load/save 的请求仍留在 Scheduler waiting
队列，并依赖“空 step”轮询状态。问题包括：

- KV I/O 与计算请求混在同一状态机；
- 同步查询外部 cache service 可能阻塞 step；
- Scheduler 主链路充斥 PD 特有分支；
- abort、断连和部分完成路径难以闭环。

Hybrid Connector 因此拆成：

```text
Connector
  - 对接 vLLM Scheduler/Worker
  - 请求生命周期和状态机
  - 动态扩缩容、对端发现、RPC、容错
  - Block 引用、完成聚合、mark_loaded/mark_saved

Backend
  - 理解某一种 cache/storage/transfer 机制
  - PD Backend、Migration Backend、KVS Backend 可并存
  - 不要求实现者理解 Scheduler 的全部细节
```

### 4.1 为什么 Block 引用计数是关键

请求可能已经被 abort 或从普通 Scheduler 结束，但 RNIC 仍在往 Block 写。若此时 Block
进入 free list 并被新请求复用，就会出现“晚到的 RDMA Write 覆盖新请求 KV”的灾难。

安全做法是：

```text
传输开始：给相关 Block 增加 transfer ref
请求结束：释放 scheduler ref，但 transfer ref 仍在
传输/失败收尾：释放 transfer ref
ref_cnt == 0：Block 才能真正进入 free list
```

引用计数让“请求生命周期”和“传输生命周期”解耦，而不是要求 Scheduler 为每一种传输
状态新增特殊分支。

### 4.2 单请求模式

请求只先发给 D：

```text
D allocate blocks
  → DBackend 暂时劫持请求
  → 选择 P 并调用 PREFILL_REQ
  → P Prefill + 传输
  → RPC 返回
  → D 设置 num_computed_tokens
  → 请求进入普通 Scheduler
```

优点是 rendezvous 简单；缺点是 P 必须等 D 完成分配、选择和 RPC 后才能开工。

### 4.3 双请求模式

路由层同时把同一请求发给 P、D：

```text
P 先开始 Prefill，但暂时不知道 DInfo
D 分配 Block 后向 P 发 TRANSFER_KV_REQ
P 在后续 step 获得 DInfo，开始传输
```

这样可让 Prefill 与 D 侧准备并行。`TRANSFER_KV_REQ` 后来还能复用于请求迁移：只要告诉
已有 KV 的源端“把这些 Block 传到新目标”即可。

### 4.4 Abort 不是“把请求从队列删掉”

一个稳健的 abort 至少涉及：

1. Connector 通知对应 Backend；
2. PBackend 通知 KVT 停止尚未提交的传输；
3. KVT 仍返回 `SEND_DONE`，但携带实际 `cached_token`；
4. Connector 发现 `cached_token < num_prompt_tokens`，把本次传输判为失败；
5. DBackend 让请求结束并向 P 发 `ABORT_REQS_REQ`；
6. Block 依靠 transfer ref 延迟释放，直到在途 RDMA 不再可能写入。

#### 示例：为什么 abort 后不能立刻 free

```text
t0  D 分配 block 42，P 开始 RDMA Write
t1  用户取消请求，D Scheduler 想释放 block 42
t2  block 42 被新请求复用
t3  旧 WR 晚到，覆盖新请求数据       ← 若没有 transfer ref
```

正确状态应是：

```text
t1 scheduler_ref--
t1 transfer_ref 仍为 1
t3 WR completion / error cleanup
t3 transfer_ref--
t4 ref_cnt==0，block 42 才可复用
```

## 5. GDR：数据怎样直接进入 D 的 GPU

GDR（GPU Direct RDMA）允许 RNIC 通过 PCIe 直接访问 GPU 显存，避免：

```text
P GPU → P host staging → network
       → D host staging → D GPU
```

理想路径变成：

```text
P GPU HBM → P RNIC → network → D RNIC → D GPU HBM
```

在文中的实现里，P 对 D 已注册的 KV Cache memory region 发起单边 RDMA Write。D
不需要为每一层运行一段 Python/CUDA copy 逻辑，因此更容易保持 Decode worker 的
full CUDA Graph 路径。

### 5.1 内存序与同步

“RDMA Write completion 了”只说明 provider 定义的传输完成边界，不能凭直觉推导 GPU
一定可以安全消费数据。正确性需要同时满足：

- P 的 forward 写 KV 必须先于 RNIC 读取源显存；
- D 的 GPU attention 读取 KV 必须晚于远端写入完成；
- CPU、GPU、RNIC 三方使用文档规定的同步原语和 memory-order contract；
- completion 的发布不能早于数据真正达到协议要求的可见性边界。

可以把它看作两条 happens-before：

```text
P GPU 产生 KV
  happens-before
P RNIC 读取并发起 RDMA Write

D RNIC 完成写入
  happens-before
D GPU Decode 读取该 KV
```

CUDA Event、RDMA completion 和应用层 RPC 各自只覆盖其中一部分，不能相互替代。

### 5.2 GDR 会不会抢 Decode 的 HBM 带宽

担忧是合理的：Decode 常常受 HBM 带宽限制，P 的 RDMA Write 同样会写 D HBM。

原文使用 Qwen2 72B、TP=4，在 D forward 前后采集：

- forward latency；
- RDMA RX bytes/packets；
- 当前 batch/request/context 信息。

同时通过限制 Scheduler，使额外压测请求只在 P 做 Prefill、向 D 产生 GDR 流量，却不
进入 D 的 Decode 计算队列。这样尽量隔离“GDR 写流量”与“额外请求调度/计算”的影响。

文章给出的不同 QPS 下 DecodeLatency 相对基线变化约在千分之一量级，并且正负都有，
因此在该实验条件下可视为测量噪声，没有观察到明显退化。

#### 实验怎样只改变 GDR 流量

实验先在 D 上注入固定 batch size 的长 Decode 请求：

```text
input_len = 1
ignore_eos = True
max_tokens = 2000
```

短 prompt 会在 D 本地 Prefill，随后持续 Decode。再把
`scheduler.max_num_scheduled_tokens` 固定为该 batch size，使 D 每个 step 只处理这批
基准请求。作者选择 `context_len > 1500` 后的 500 个 step 作为 DecodeLatency；此时
attention 要扫描较长 KV Cache，对 memory subsystem 的干扰更敏感。

当基准请求运行到 `context_len ≈ 500` 时，另用 `benchmark_serving` 按不同 QPS 提交：

```text
input_len = 2000
output_len = 7
```

这些请求在 P 做 Prefill，并通过 GDR RDMA Write 向 D GPU 写 KV；但 D 的 token budget
已由基准 batch 占满，新请求暂不进入 D forward。于是实验近似得到：

```text
qps0：固定 D Decode
qpsN：相同 D Decode + P→D GDR RDMA Write
```

在每次 D forward 前后，代码同时读取 RNIC 的 RX bytes/packets，forward 后执行
`torch.cuda.synchronize()` 再停止计时。同步是必要的，否则 CPU 只测到异步 kernel
launch，不是 GPU forward 的实际完成时间。

#### `rdma_rx_bps` 不是 HBM 带宽

文章通过 RDMA netlink 读取的 `hw_rx_bytes_cnt/hw_rx_packets` 是 **D RNIC 的网络入口
计数器**。它既不是 GPU PCIe RX counter，也不是 HBM read/write counter。

D 侧 GDR Write 的数据路径是串联的：

```text
network
  → D RNIC
  → PCIe peer-to-peer transaction
  → GPU I/O / L2 / memory partitions
  → D HBM 中的 KV Cache
```

所以问题不是“PCIe 还是 HBM”二选一：

- RNIC 向 GPU 发 peer DMA transaction，会占 PCIe/P2P 入站带宽；
- 数据最终成为显存内容，会消耗 GPU memory controller/HBM 写资源；
- Decode 同时读权重和 KV，可能在 L2、memory partition、HBM 等处与 GDR 写发生竞争。

端到端 GDR 吞吐受最窄环节限制：

```text
min(network, RNIC, PCIe P2P, GPU inbound path, HBM available write bandwidth)
```

这篇文章做的是**应用级干扰测试**：观察 RNIC 流量存在时 Decode latency 是否恶化，
而不是直接量化 HBM 被占用了多少。若要定位具体瓶颈，还应同时采集：

| 层次 | 应观察的量 |
|---|---|
| 网络/RNIC | RX bytes/s、packets/s、拥塞与丢包 |
| PCIe | GPU PCIe RX throughput、拓扑与 root complex |
| GPU memory subsystem | HBM read/write throughput、L2 traffic、memory partition 利用率 |
| 应用 | forward latency、ITL/TPOT、batch/context length |

但结论边界很重要：

- 只说明该模型、GPU、ERDMA、batch token 配置；
- QPS 已足以跑满 Prefill 后，流量主要由每 step 的 chunked-prefill token 数决定；
- 更大 KV、不同 GPU/RNIC/NUMA/PCIe 拓扑可能得到不同结果；
- 线上仍应同时观察 Decode latency、HBM/PCIe/RDMA throughput 和 tail latency。

## 6. 大量 128 B 小包为什么慢

当 P 的 attention TP 大于 D 的 attention TP 时，一个 P worker 每次可能只能为目标
worker 提供某一小部分 KV head。原文场景中，一个 WR 对应的数据可小到 128 B。

小包下瓶颈往往不是链路带宽，而是：

```text
构造 WR
→ ibv_post_send
→ NIC 生成 CQE
→ 软件 poll CQ
→ callback/future
→ 再取下一块并 post
```

若每个 128 B WR 都设置 `IBV_SEND_SIGNALED`，CQE 频率和软件栈开销会极高，NIC 甚至
可能在两次 post 之间空闲。

![RDMA 小包优化实验](/imgs/vllm-pd-rdma-small-message-optimization.svg)

### 6.1 优化一：Unsignaled + 批量提交

把一批 WR 连成链，只给最后一个 WR 设置 `IBV_SEND_SIGNALED`：

```text
WR0 unsignaled ─┐
WR1 unsignaled  ├─ 一次/少次 post
...             │
WRn signaled ───┘ → 只为批次生成必要 CQE
```

收益不只是减少 CQE；它还把发送行为从“一个完成后再提交下一个”改成“预先给 NIC 一批
工作”，降低软件往返并让 NIC 保持繁忙。

文章平均耗时从约 `204508 us` 降到 `56368 us`，约为原来的 27.6%，即约 3.62 倍加速。

### 6.2 优化二：缩短热路径，直接使用 QP/ibverbs

在实验中进一步剥离与传输无关的抽象和调度开销，直接使用 QP 与 ibverbs 接口，平均
耗时从约 `56368 us` 降到 `7843 us`，相对上一阶段约 7.19 倍加速。

这不意味着生产代码一定要丢弃抽象层，而是说明：

- 当前主要瓶颈可能在 host 软件路径，不在 wire bandwidth；
- profiling 应拆分 enqueue、post、poll、callback、batch formation；
- 通用网络库若按大包优化，未必适合 128 B 级密集 KV 分片。

### 6.3 优化三：增加 QP 并行

在未优化的软件栈上，QP 数量翻倍只提升约 2%；在精简后的热路径上，QP 翻倍又让平均
耗时从约 `7843 us` 降到 `4478 us`，耗时降低约 42.9%。

这不是“多 QP 突然产生了神奇效果”，而是主瓶颈发生了迁移。

未精简时，两个 QP 仍共享前面的软件路径：

```text
task/queue/lock/state machine/future/callback
                     ├─ QP0
                     └─ QP1
```

若 CPU 不能及时构造并 post WQE，两个 QP 都处于“吃不饱”的状态。增加 QP 只并行了
总耗时中的很小一部分，共享的软件串行段完全没有缩短，因此只有约 2%。

裸用 QP/ibverbs 后，热路径变成：

```text
prepare WR → post SQ → RNIC execute → necessary CQ poll
```

此时 CPU 能持续给 RNIC 喂 WQE，单 QP 的有序队列、outstanding WR、doorbell/WQE 获取
或 credit 等限制才可能成为显著部分。第二个 QP 提供独立 Send Queue，使 RNIC 可以：

- 保持更多 WR in flight；
- 一个 QP 暂时受顺序或 credit 限制时继续推进另一个；
- 降低单 QP head-of-line blocking；
- 更充分填充 RNIC/PCIe pipeline。

注意文章所谓“提升 42.90%”按耗时降低计算：

```text
1 - 4477.67 / 7843.14 ≈ 42.90%
```

若换成速度/吞吐提升，则：

```text
7843.14 / 4477.67 ≈ 1.75×
```

可用 Amdahl 定律建立直觉。假设增加一倍 QP 最多让 QP 可并行部分加速 2 倍：

```text
speedup = 1 / ((1 - p) + p / 2)
```

- 只加速约 1.02× 时，反推 QP 可并行部分约占 4%；
- 裸接口后加速约 1.75× 时，反推可并行部分约占 86%。

这只是依据总耗时的近似分解，不是文章提供的 profiler 结果，但清晰说明：先去掉共享
串行开销，多 QP 才会成为有效的优化旋钮。

因此不能孤立地问“QP 越多是否越快”。还要检查：

- 是否已有足够 in-flight WR；
- CQ/QP poller 是否成为 CPU 瓶颈；
- 同一 target 的有序性要求；
- RNIC 资源、doorbell、cache locality；
- 多 QP 是否引入完成乱序与更复杂的聚合。

增加 QP 不会增加物理链路的额定带宽；它主要提高并发度，并减少单 QP 串行化造成的
流水线空洞。若瓶颈仍在 CPU 软件栈、PCIe 或 wire bandwidth，继续加 QP 仍不会有效。

## 7. KVT 2.0 为什么还要继续演进

KVT 1.0 隐含了比较规则的 Block/Token/Layers：

```text
每层一块大显存
→ 划分为等大 block
→ block 内划分为等大 token
```

但新模型和新引擎引入：

- Qwen3-Next GDN；
- DeepSeek DSA；
- vLLM Hybrid KVCache Manager 的多种 `KVCacheSpec`；
- 不同 layer 可能有不同 state/cache 形态；
- MoE EP all-to-all 与 KV 传输共享 GPU/RDMA 通路。

所以 KVT 2.0 的方向不是再给旧结构堆条件分支，而是：

1. 把 layout/parser 抽象做得更通用；
2. 允许多种 cache spec 和不规则区间；
3. 让 backend/transport 可替换；
4. 对不同流量提供真正的资源隔离。

文章还提出考虑用 TCP 绕开现有 GPU/GDR 通路，使 KV traffic 与 EP all-to-all 的 RDMA
traffic 隔离。这里的重点不是“TCP 一定更快”，而是**端到端性能可能受共享资源干扰
主导**：一条理论带宽更低但隔离良好的路径，可能比与 EP 抢 RNIC/QP/CQ/PCIe 的路径
提供更稳定的 tail latency。

## 8. 为什么协程适合 Hybrid Connector

### 8.1 线程、协程与执行流

文章把并发抽象为“逻辑执行流”与“运行载体”的映射：

| 模型 | 映射 | 特点 |
|---|---|---|
| 线程 1:1 | 一个逻辑流对应一个 OS 线程 | 内核抢占、公平调度；大量 I/O 等待时切换成本高 |
| 协程 M:1 | 多个逻辑流共享一个线程 | 用户态切换轻；一个协程阻塞可拖住整个 loop |
| 协程 M:N | 多个协程运行在多个工作线程 | 可 work stealing；运行时与线程安全更复杂 |

Python asyncio 通常是 M:1：

```text
一个 event-loop thread
  ├─ Task A：等 P RPC
  ├─ Task B：等 worker SEND_DONE
  ├─ Task C：等 timeout/retry
  └─ Task D：处理 abort
```

当 A `await reader.readexactly()` 时，如果数据尚未就绪，A 挂起，loop 可以执行 B/C/D。
这很适合连接多、单次 CPU 工作少的 Hybrid Connector 控制面。

### 8.2 `await` 不保证让出线程

如果 awaitable 已经 ready，`await` 可以立即返回：

```python
await queue.put(req)  # 无界 queue 未满，可能立即完成
```

若循环中的所有 await 都立即完成，一个 Task 仍可能长期霸占 loop：

```python
async def producer():
    while True:
        req = await get_ready_req()  # 压测时总是 ready
        await unbounded_q.put(req)   # 总是 ready
```

其他 Task 可能长期拿不到调度机会。必要时显式：

```python
await asyncio.sleep(0)
```

但这只是局部补丁。更成熟的运行时会有 cooperative budget，例如资源操作消耗 budget，
耗尽后迫使 Task yield。asyncio 没有同等的统一机制，因此更依赖代码规范、lint、压测
和 event-loop latency 指标。

### 8.3 同步/异步边界

同步函数调用异步库，可在真正的边界使用：

```python
loop.run_until_complete(coro())
```

但不能在已经运行的 event loop 内再次 `run_until_complete`，否则会得到
`RuntimeError: This event loop is already running`。

异步函数若必须调用阻塞 I/O 或 CPU-heavy 逻辑，应放入 executor：

```python
result = await loop.run_in_executor(None, blocking_io)
```

否则一次日志 flush、DNS、文件 I/O、锁等待或长循环，都可能冻结 P/D 控制面上的所有
coroutine。

### 8.4 跨线程提交必须使用线程安全入口

asyncio 的大多数对象不是线程安全的。从 EngineCore/worker thread 向 Hybrid loop 提交
任务，应使用：

```python
future = asyncio.run_coroutine_threadsafe(coro(), loop)
```

或者通过 `loop.call_soon_threadsafe()` 把 callback 放入 loop。不能从任意线程直接修改
loop 内部 Task/Future 状态。

## 9. 一定要保存 `asyncio.Task`

event loop 对 Task 只保留弱引用。典型错误是：

```python
asyncio.create_task(background_rpc())
# 返回值无人保存
```

如果没有其他对象形成强引用，GC 可以在 Task 尚未完成时回收它，出现：

```text
Task was destroyed but it is pending!
```

![asyncio Task 的引用链](/imgs/vllm-pd-asyncio-task-reference.svg)

### 9.1 为什么有些代码“从来没出过问题”

Task 是否被意外保活可能依赖 awaitable 的实现细节。

例如 `asyncio.sleep()` 会建立类似的引用链：

```text
event loop._scheduled
  → TimerHandle
  → Future
  → callback(Task.__wakeup)
  → Task
```

于是一个 `while True: await asyncio.sleep(...)` 的后台 Task 可能一直有强引用，让人误以为
不保存 `create_task()` 返回值也安全。

但等待 `StreamReader.readexactly()` 等路径时，不应假设存在同样的强引用链。原文中的
长稳测试正是在这种路径上暴露了 pending Task 被 GC 的问题。

工程结论不是“判断哪一种 awaitable 会保活 Task”，而是：

```text
所有 fire-and-forget Task 都显式保存；
Task 完成后再从集合删除。
```

### 9.2 推荐写法

```python
running_tasks: set[asyncio.Task] = set()


def spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    running_tasks.add(task)
    task.add_done_callback(running_tasks.discard)
    return task
```

如果后台 Task 对进程存活至关重要，还要处理未观察异常：

```python
def on_done(task: asyncio.Task) -> None:
    running_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("critical background task failed")
        os.abort()  # 是否 fail-fast 取决于系统策略
```

原文采用 `kill_me_if_exception` 装饰器和 loop exception handler 直接 abort，目标是避免
控制面后台 Task 静默死亡后进程表面健康、请求却永久等待。其权衡是：局部可恢复错误也
可能升级为进程退出。因此生产设计还需区分：

```text
request-scoped error → 失败当前请求并清理资源
peer-scoped error    → 断开/摘除对端并重试或重路由
process invariant    → fail-fast，由上层拉起新实例
```

## 10. 把 asyncio 语义映射回 PD 控制面

| asyncio 风险 | PD/KVT 后果 | 防御 |
|---|---|---|
| Task 没有强引用，被 GC | `SEND_DONE`/RPC reader 消失，请求永远等 | running task set + done callback |
| coroutine 含同步阻塞 | 整个 Hybrid loop 无法处理 completion/abort | executor、拆分 CPU 工作、event-loop lag |
| awaitable 总是 ready | 单 Task 饿死其他请求 | 主动 yield、限额队列、budget/公平性测试 |
| `readexactly` 无 deadline | 半开连接导致永久 pending | 单次 I/O timeout + request 总 deadline |
| Task 异常无人观察 | 进程存活但控制面已残缺 | done callback、loop exception handler、健康检查 |
| 跨线程直接操作 Future | 状态竞态或非线程安全异常 | `run_coroutine_threadsafe` / `call_soon_threadsafe` |
| cancellation 只取消 Python Task | 在途 RDMA 仍可能写 D Block | transfer ref、底层 cancel/cleanup、late completion 去重 |

最后一行尤其重要：`asyncio.wait_for()` 超时并不能自动撤销已经 post 给 RNIC 的 WR。
Python 控制面、C++ KVT、Barex/provider 和 Block 生命周期必须共同定义 timeout/abort
协议。

## 11. 运行时可观测性：事前防御与事后追踪

### 11.1 事前防御

- lint 禁止在 async 路径直接调用已知阻塞函数；
- 所有后台 Task 通过统一 `spawn()` 创建；
- RPC read/write/handshake 有单次 timeout；
- request 有跨重试的总 deadline；
- 无界 queue 改为限额、批处理或显式公平让出；
- fault injection 覆盖少一个 TP completion、对端半开、late completion、abort 竞态。

### 11.2 事后追踪

参考内核 Scheduler 指标，为用户态 loop 暴露：

```text
runnable task queue length
task 从 ready 到真正运行的延迟
event-loop lag
当前 Task 连续运行时长
最老的 waiting/loading/saving/sending age
各请求缺失的 TP rank/source
RPC read/write latency 与 timeout count
running_tasks 数量及其 coroutine 名称
```

当某工作线程长期执行一个协程，可采样线程栈；更完善的工具还应展示 coroutine await
链、Future owner 和逻辑异步栈。信号处理函数中直接执行复杂 backtrace 要注意
async-signal-safety，不能为了诊断再引入新的 crash/deadlock。

## 12. 一套端到端排查顺序

### 12.1 TTFT 变差

```text
先看请求时间线
  → P Prefill 是否变慢
  → 每层 Event→post 是否有空洞
  → WR batch size / signaled 比例
  → post→CQE
  → TP SEND_DONE 聚合
  → P RPC 返回
  → D mark_loaded / 重新入队
```

小包场景优先看 WR/CQE/software overhead，而不是只看链路 GB/s。

### 12.2 Decode ITL/TPOT 变差

```text
确认额外请求是否进入 D 计算队列
  → 隔离纯 GDR 流量
  → 对齐 forward latency 与 RDMA RX bytes/pps
  → 检查 HBM、PCIe、RNIC、EP all-to-all 的时间相关性
```

不要仅凭“GDR 会写 HBM”推断一定退化，也不要用一次无退化实验推断所有拓扑都安全。

### 12.3 请求永久 hang

```text
检查 Hybrid event-loop lag / thread stack
  → 后台 Task 是否仍在 running_tasks
  → `_SendingReq`/IoState 缺哪个 rank/source
  → socket 是否卡在无 timeout 的 readexactly
  → transport future 是否无界等待
  → completion 是否到达但 Core 未被唤醒
```

### 12.4 Abort 后错 KV

```text
检查 D Block ref_cnt
  → scheduler ref 与 transfer ref 是否分别释放
  → 在途 WR 是否仍写旧地址
  → late completion 是否按 generation/request 去重
  → partial KV 是否被错误标成 loaded
```

## 13. 最终心智模型

把整套系统压缩成四个互相咬合的原则：

1. **布局隔离**：上层使用 token/block 语义，ParseBlock 负责变成 byte ranges。
2. **计算与 I/O 隔离**：Scheduler 推进计算；Connector/Backend 异步处理 KV load/save。
3. **完成边界分层**：CUDA Event、CQE、worker completion、TP 聚合和请求完成不可混淆。
4. **所有权显式化**：Block、Task、connection、Future 和 timeout 都必须有明确 owner。

性能优化和可靠性并非两条独立路线：

```text
批量 Unsignaled WR 提升性能
  → completion 粒度变化
  → 错误归属与回收逻辑也要变化

双请求模式降低等待
  → P/D rendezvous 更异步
  → abort、超时、Task 生命周期更难

GDR 去掉 host staging
  → 数据路径更短
  → GPU/RNIC memory ordering 与 Block 延迟释放更关键
```

真正成熟的 PD 分离系统，不只是“能把 KV 发过去”，而是同时做到：主调度链路足够轻、
小包传输足够快、每种 completion 有清晰语义，以及任何异常路径最终都能释放正确的
Request、Task、连接和 KV Block。

## 14. 原始文章

- [转折中的 PD 分离](https://ata.atatech.org/articles/11020518835)
- [PD 分离中的 kvcache 传输优化](https://ata.atatech.org/articles/11020460403)
- [PD 分离中的 GDR](https://ata.atatech.org/articles/11020408441)
- [一定要保存 asyncio.Task 对象啊!](https://ata.atatech.org/articles/11020396051)
- [为什么协程](https://ata.atatech.org/articles/11020324515)

## 15. 自检题

1. 为什么 D 必须先分配目标 Block，P 才能做单边 RDMA Write？
2. ParseBlock 抽象如何同时适配 KV 相邻布局与 K/V 分区布局？
3. 为什么 CQE 到达不能直接等价为 D 请求可 Decode？
4. 每个 128 B WR 都 signaled 时，瓶颈为什么可能在 CPU/CQE 而不是网络带宽？
5. 为什么增加 QP 在优化前几乎无效、精简热路径后却明显有效？
6. GDR 对 Decode 无明显影响的实验结论有哪些适用边界？
7. `await` 为什么不保证当前 Task 一定让出 event-loop thread？
8. 为什么 `while True: await asyncio.sleep()` 可能掩盖未保存 Task 的错误？
9. `asyncio.wait_for()` 超时后，为什么仍不能立即复用 D KV Block？
10. 一次 abort 需要哪些组件共同完成资源收尾？
