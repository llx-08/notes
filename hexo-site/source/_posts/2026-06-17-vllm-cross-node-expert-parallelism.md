---
title: vLLM 跨机 Expert Parallelism (EP) 深度解析
date: 2026-06-17
tags: []
---

# vLLM 跨机 Expert Parallelism (EP) 深度解析

> 基于 vLLM v0.11.1 (`/mnt/data/llx/vllm`) + `vllm_ep_deepdive.html` 活体追踪文档
> 撰写日期：2026-06-09

---

## 一、什么是跨机 EP

Expert Parallelism (EP) 是 MoE 模型独有的并行维度：把 N 个 expert 分到各 GPU 上，每个 GPU 只持有 N/EP 个完整 expert。vLLM 中 EP 不是独立的进程轴，而是 **TP 轴和 DP 轴的并集**：

```
EP = TP × DP
```

**跨机 EP** 指的是：当 EP group 的成员分布在**不同物理节点**上时，MoE 层的 dispatch/combine all-to-all 通信就不再只走节点内高速的 NVLink/NVSwitch，而必须跨越**节点间较慢的网络**（InfiniBand RDMA、RoCE 等）。

例如：
- 4 节点 × 8 GPU/节点，TP=1, DP=32 → EP=32，每个 EP group 横跨 4 台机器
- 2 节点 × 8 GPU/节点，TP=8, DP=2 → EP=16，EP group 横跨 2 台机器

跨机检测逻辑（`base_device_communicator.py:55`）：

```python
self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
```

---

## 二、EP 拓扑与进程模型

### 2.1 EP group 的构造

`parallel_state.py:1534-1556` 中，EP group 通过对 `[ExternalDP, DP, PP, PCP, TP]` 五维 rank tensor 做 `transpose(1,2)` 再 `flatten(DP×PCP×TP)` 构造：

```python
global _EP
group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(-1, data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size)
    .unbind(0)
)
_EP = init_model_parallel_group(group_ranks, ..., group_name="ep")
```

不变量断言（`parallel.py:726-734`）：

```python
assert expert_parallel_size == data_parallel_size * tensor_parallel_size
```

### 2.2 进程与 GPU 分配

每个 DP rank 对应一个独立的 `EngineCore` 子进程（`utils.py:117-134`），每个子进程内部再派生 TP 个 `WorkerProc`。GPU 按 `[local_dp_rank * world_size, local_dp_rank * world_size + local_world_size)` 切片分配。

跨机时，不同节点上的 DP rank 通过 `torch.distributed` 的 NCCL/gloo backend 进行通信。

### 2.3 DP Lockstep 保证

由于 MoE 层在 EP group 上做跨所有 DP rank 的 all-to-all，一个 DP rank **绝不能**在其他 rank 还在 forward pass 时跳过——否则 collective 会死锁。

```python
# core.py:2143-2191
# 空闲的 DP rank 仍须运行 execute_dummy_batch()
if self.engines_running and not self.scheduler.has_unfinished_requests():
    dummy_batch_future = self.model_executor.execute_dummy_batch(...)
```

终止同步每 32 步做一次 `all_reduce(MAX)` 来保证所有 rank 一起停止。

---

## 三、跨机 EP 的三种通信后端

### 3.1 后端选择

通过 `VLLM_ALL2ALL_BACKEND` 环境变量（默认 `allgather_reducescatter`）选择，在 `cuda_communicator.py:92-118`：

| 后端 | dispatch | combine | 跨机策略 | 适用场景 |
|---|---|---|---|---|
| `allgather_reducescatter` | `all_gatherv`(NCCL) | `reduce_scatterv`(NCCL) | NCCL 自动处理跨机 ring/tree | 通用默认，简单但冗余 |
| `naive` | broadcast per rank | all_reduce + slice | NCCL broadcast | 调试用，更慢 |
| `pplx` | pplx AllToAll | 同 | 跨机走 NVSHMEM RDMA；机内走 P2P | 需要安装 pplx_kernels |
| `deepep_high_throughput` | `buffer.dispatch` | `buffer.combine` | 跨机分配 RDMA buffer；机内仅 NVLink | 高吞吐场景（prefill） |
| `deepep_low_latency` | `buffer.low_latency_dispatch` | `buffer.low_latency_combine` | RDMA 预注册 buffer + NVLink/MNNVL | 低延迟场景（decode） |

### 3.2 简单路径：allgather + reduce-scatter

`all2all.py:102-149`，`AgRsAll2AllManager`：

```python
def dispatch(self, hidden_states, router_logits, is_sequence_parallel=False):
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
    hidden_states, router_logits = dist_group.all_gatherv(
        [hidden_states, router_logits], dim=0, sizes=sizes)
    return hidden_states, router_logits

def combine(self, hidden_states, is_sequence_parallel=False):
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
    dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
    return dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
```

**数据流**：每个 rank 通过 allgather 收到**完整的 token batch**，对所有 token 运行自己的 local expert，再通过 reduce-scatter 求和并 scatter 回各自的 token 切片。

- 优点：无需预分配 buffer，NCCL 自动选择最优跨机算法
- 缺点：O(world_size) 冗余计算和带宽（每个 rank 都收到全量 token）

### 3.3 PPLX 路径：真正的 all-to-all

`all2all.py:150-225`，`PPLXAll2AllManager`：

```python
if self.internode:
    # 跨机 → 初始化 NVSHMEM (Symmetric Heap + RDMA)
    nvshmem_init(uid, self.rank, self.world_size)
# 根据 internode 选择不同底层实现
pplx.AllToAll.internode if self.internode else pplx.AllToAll.intranode
```

- 跨机通信通过 **NVSHMEM**（NVIDIA Symmetric Memory + RDMA），建立全局可寻址的对称堆
- 机内通信直接走 GPU P2P mapping（NVLink）
- 只发送被路由到对应 rank 的 token，避免冗余

### 3.4 DeepEP 路径：RDMA 预注册 buffer

**High-Throughput**（`all2all.py:230-310`）：

```python
if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
    num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024  # 跨机分配 RDMA buffer
else:
    num_rdma_bytes = 0  # 纯机内 NVLink
```

**Low-Latency**（`all2all.py:315-380`）：

```python
num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
    hidden=token_hidden_size,
    num_ranks=num_ep_ranks,
    num_experts=num_global_experts,
)
# 允许 NVLink 加速 + 可选 MNNVL（Multi-Node NVLink）
allow_nvlink_for_low_latency_mode=True,
allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
```

> **OOM 陷阱**：DeepEP LL 的 RDMA buffer 大小由 `max_num_batched_tokens` 驱动（向上取整到 2 的幂），与 EP_size **基本无关**。实测在 ranks=2 和 ranks=4 下都是约 558.9 GB。增加节点数不会缓解 OOM，应调小 `max_num_batched_tokens` 或设置 `VLLM_NUM_MAX_DISPATCHED_TOKENS_PER_RANK`。

---

## 四、forward 路径中的 dispatch/combine

在 `FusedMoE.forward_impl`（`layer.py:2575-2730`）中，分两条互斥路径：

### 4.1 简单路径（naive dispatch/combine）

触发条件（`layer.py:2615-2630`）：

```python
do_naive_dispatch_combine = (
    (not envs.VLLM_MOE_USE_DEEPEP or self.quant_method.__class__.__name__ not in [...])
    and self.dp_size > 1
    and not isinstance(self.quant_method, FusedMoEModularMethod)
)
```

执行流程：

```
① get_ep_group().dispatch(hidden_states, router_logits)   # allgather
② quant_method.apply(hidden_states_combined, ...)          # expert GEMM on gathered batch
③ get_ep_group().combine(output)                           # reduce-scatter
```

### 4.2 模块化路径（DeepEP/pplx modular kernel）

当 `use_all2all_kernels=True` 时（`config.py:933`），`maybe_init_modular_kernel` 把 `quant_method` 替换为 `FusedMoEModularMethod`，内含 `FusedMoEModularKernel`（`modular_kernel.py:1206-1296`）：

```
_prepare(dispatch+quantize) → _fused_experts(expert GEMM) → _finalize(combine+reduce)
```

此路径中 `do_naive_dispatch_combine` 为 False，所有通信都在 modular kernel 内部完成。

---

## 五、EPLB 分层均衡算法：跨机 EP 的关键优化

EPLB（Expert-Parallel Load Balancing）是跨机 EP 性能优化的核心，位于 `distributed/eplb/policy/default.py`。

### 5.1 问题

MoE 路由是数据相关的：少数热点 expert 吸引远多于平均的 token。跨机 EP 下，如果热点 expert 分散在多台机器上，大量 token 必须跨机传输，成为瓶颈。

### 5.2 分层算法三步走

`rebalance_experts_hierarchical`（`default.py:115-201`）感知节点拓扑，分三步：

**Step 1：将 expert group 打包到节点**（最小化跨机流量）

```python
tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
group_pack_index, group_rank_in_pack = cls.balanced_packing(
    tokens_per_group, num_nodes  # 均衡打包到各节点
)
```

`balanced_packing` 用贪心算法：按负载降序排列 group，每次把 group 放入当前负载最低且有容量的 node。

**Step 2：在每个节点内部复制热点 expert**

```python
tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
phy2mlog, phyrank, mlogcnt = cls.replicate_experts(
    tokens_per_mlog, num_physical_experts // num_nodes
)
```

`replicate_experts` 的贪心策略：每次把冗余物理槽位分配给当前"每副本负载最大"的 logical expert（`weight / logcnt` 的 argmax）。

**Step 3：在每个节点内部将 physical expert 打包到各 GPU**

```python
tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
pack_index, rank_in_pack = cls.balanced_packing(
    tokens_per_phy, num_gpus // num_nodes  # 节点内 GPU 间均衡
)
```

### 5.3 设计思想

```
         节点 A (NVLink 900GB/s)              节点 B (NVLink 900GB/s)
     ┌──────────────────────────┐         ┌──────────────────────────┐
     │  GPU0: expert 0,1,5,7    │         │  GPU4: expert 256,258... │
     │  GPU1: expert 2,3,6,8    │  ← IB → │  GPU5: expert 257,259... │
     │  GPU2: expert 9,10...    │  50GB/s  │  GPU6: expert 300,301... │
     │  GPU3: expert 11,12...   │         │  GPU7: expert 400,401... │
     └──────────────────────────┘         └──────────────────────────┘
```

- 先把同一 expert group 的 expert 尽量放在同一节点，让大部分 token 路由走节点内 NVLink
- 冗余 expert 的复制限制在节点内部
- 最大化利用节点内带宽，最小化跨机流量

### 5.4 回退策略

```python
# default.py:242-249
if num_groups % num_nodes == 0:
    # 分层策略（推荐，利用节点拓扑）
    phy2log, phyrank, logcnt = cls.rebalance_experts_hierarchical(
        weight, num_replicas, num_groups, num_nodes, num_ranks)
else:
    # 回退到全局扁平均衡（num_groups=1, num_nodes=1）
    phy2log, phyrank, logcnt = cls.rebalance_experts_hierarchical(
        weight, num_replicas, 1, 1, num_ranks)
```

### 5.5 权重搬运

重平衡后需要在 GPU 间物理搬运 expert 权重（`rebalance_execute.py:143-208`）：

```python
# 通过 P2P isend/irecv 在 EP group 的 rank 之间交换 expert 权重
for dst in recv_ranks:
    p2p_ops += [P2POp(torch.distributed.isend, weight[src],
                      get_global_rank(ep_group, dst))
                for weight in expert_weights]
reqs = batch_isend_irecv(p2p_ops)
[r.wait() for r in reqs]
```

异步模式下（`async_worker.py`），权重搬运在私有 CUDA stream 上执行，与推理重叠。

---

## 六、跨机通信量统计工具

`layer.py:318-386` 提供了一个 Triton kernel `fused_communication_ops`，用于分析跨机 EP 的实际通信模式：

```python
@triton.jit
def fused_communication_ops_kernel(
    topk_ids_ptr,
    comm_ops_ptr,         # [num_nodes]
    expert_repeats_ptr,   # [num_tokens, num_nodes]
    n_tokens,
    num_nodes: tl.constexpr,
    num_experts_per_node: tl.constexpr,
    enable_dedup_optimization: tl.constexpr,
    ...
):
    expert_nodes = expert_id // num_experts_per_node
    if enable_dedup_optimization:
        # 同一 token 路由到同一 node 的多个 expert → 只计一次跨机传输
        tl.atomic_add(expert_repeats_ptr + pid * num_nodes + expert_nodes, 1, ...)
```

两种统计模式：
- **不去重**（`comm_ops`）：每个 top-k expert 独立计入目标节点
- **去重**（`comm_ops_with_dedup`）：同一 token 发往同一节点只计一次（因为物理上只需传一次 token 数据）

---

## 七、带宽需求分析

### 7.1 各互联类型带宽对比

| 互联类型 | 单向带宽 | 双向带宽 | 典型场景 |
|---|---|---|---|
| NVLink 4.0（机内） | 450 GB/s | 900 GB/s | 同节点 GPU 间 |
| NVSwitch L2（机内全互联） | 900 GB/s（per GPU） | 1.8 TB/s | H100/B200 DGX |
| MNNVL（多节点 NVLink） | 900 GB/s | 1.8 TB/s | GB200 NVL72 |
| InfiniBand HDR（跨机） | 25 GB/s | 50 GB/s | 常见跨机互联 |
| InfiniBand NDR（跨机） | 50 GB/s | 100 GB/s | 高端跨机互联 |
| InfiniBand NDR ×4（跨机） | 200 GB/s | 400 GB/s | 8 卡 4 端口聚合 |
| RoCE v2（跨机） | 12.5-25 GB/s | 25-50 GB/s | 以太网 RDMA |

**关键差距**：跨机 IB 带宽比机内 NVLink 低 **~18-36 倍**（单端口对比）。

### 7.2 每 MoE 层的通信量

以 **Qwen3-Omni-Next**（512 experts, hidden=4096, top-k=10, BF16）为例：

#### 简单路径（allgather/reduce-scatter）

每层每 rank 的通信量：

```
dispatch:  N_tokens × hidden × 2B × (EP-1)/EP    (allgather)
combine:   N_tokens × hidden × 2B × (EP-1)/EP    (reduce-scatter)
```

| EP size | batch=1024 tokens | batch=4096 tokens |
|---|---|---|
| EP=8 | 7.2 MB/层 | 28.7 MB/层 |
| EP=16 | 7.7 MB/层 | 30.7 MB/层 |
| EP=32 | 7.9 MB/层 | 31.7 MB/层 |

> 注：allgather 的通信量随 EP 增大趋于 `N × hidden × 2B`，因为 `(EP-1)/EP → 1`。

#### 真 all-to-all 路径（DeepEP/pplx）

只发送被路由到其他 rank 的 token：

```
每 token 发送次数 ≈ top_k 个 expert 分布在的不同远端 rank 数
单 token 数据量 = hidden × 2B = 4096 × 2 = 8 KB
```

| EP size | batch=1024, top-k=10 | batch=4096, top-k=10 |
|---|---|---|
| EP=8 | ~3.2 MB/层（发送 ~60% token 到远端） | ~12.8 MB/层 |
| EP=16 | ~5.6 MB/层（发送 ~70% token 到远端） | ~22.4 MB/层 |
| EP=32 | ~6.8 MB/层（发送 ~83% token 到远端） | ~27.2 MB/层 |

> dedup 优化后更低：同一 token 路由到同一节点的多个 expert 只传一次。

### 7.3 全模型单次 forward 总通信量

60 MoE 层 × 2 次通信（dispatch + combine）：

| 路径 | EP=16, batch=2048 | EP=32, batch=2048 |
|---|---|---|
| allgather/RS（简单） | ~1.85 GB | ~1.90 GB |
| 真 all-to-all（DeepEP） | ~0.84 GB | ~0.97 GB |

### 7.4 跨机带宽需求估算

假设跨机流量占比 = `(节点数-1) / 节点数`（均匀路由下）：

#### 场景 A：4 节点 × 8 GPU, EP=32, 简单路径

```
跨机流量 = 1.90 GB × 3/4 = 1.43 GB / forward
目标延迟 = 50 ms（decode TPOT 预算中的通信部分）
所需跨机带宽 ≥ 1.43 GB / 0.05 s = 28.6 GB/s ≈ 228 Gb/s
→ 需要 IB NDR（单端口 50 GB/s 足够）或多端口 HDR
```

#### 场景 B：同上, DeepEP all-to-all

```
跨机流量 = 0.97 GB × 3/4 = 0.73 GB / forward
所需跨机带宽 ≥ 0.73 GB / 0.05 s = 14.6 GB/s ≈ 117 Gb/s
→ IB HDR 单端口（25 GB/s）即可满足
```

#### 场景 C：2 节点 × 8 GPU, EP=16

```
跨机流量比例 = 1/2
简单路径: 1.85 GB × 1/2 = 0.93 GB → 需要 18.6 GB/s
DeepEP:   0.84 GB × 1/2 = 0.42 GB → 需要 8.4 GB/s
```

#### 场景 D：有 EPLB 分层优化

EPLB 把 expert group 打包到节点内，实际跨机流量可降低 **40-60%**：

```
场景 A + EPLB: 跨机流量 ≈ 1.43 × 0.5 = 0.72 GB → 14.4 GB/s
```

### 7.5 关键结论

| 因素 | 影响 |
|---|---|
| 节点数增加 | 跨机流量占比上升（`(N-1)/N`），单路 IB 压力增大 |
| batch size 增大 | 通信量线性增长，但 GEMM 计算也同步增长，通信/计算比稳定 |
| top-k 增大 | 真 all-to-all 路径的跨机 token 占比上升 |
| EPLB 分层打包 | 有效降低跨机流量 40-60%，是跨机 EP 的核心优化手段 |
| DeepEP vs allgather | DeepEP 减少约 50% 通信量，但需要预分配 RDMA buffer |

---

## 八、Expert 分片与 expert_map

每个 rank 只持有 `N/EP` 个完整 expert。`determine_expert_map`（`layer.py:166-185`）构建映射：

```python
base_experts = global_num_experts // ep_size
remainder = global_num_experts % ep_size
local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
# linear 模式：连续分配
start_idx = ep_rank * base_experts + min(ep_rank, remainder)
expert_map[start_idx:start_idx + local_num_experts] = torch.arange(0, local_num_experts)
```

开启 EP 时，`FusedMoEParallelConfig.make`（`config.py:1052-1095`）会**关闭权重 TP sharding**（`tp_size=1`），把 `tp_rank` 重新解释为 `ep_rank`。每个 rank 持有完整的 expert 权重，而不是每个 expert 的一个切片。

---

## 九、模型接入层

### Qwen3NextSparseMoeBlock（`qwen3_next.py:160-298`）

模型层不直接操作 EP 通信，只负责：
1. 构造 `SharedFusedMoE`（持有 routed experts + shared expert + gate）
2. 调用 `self.experts(hidden_states, router_logits)`
3. 折叠 shared expert 输出 + 最终 TP/SP reduction

所有 EP dispatch/combine 都封装在 `FusedMoE.forward_impl` 内部。

### SharedFusedMoE（`shared_fused_moe.py`）

支持 overlapped 模式：shared expert 的 GEMM 与 routed expert 的 all-to-all dispatch 交错执行，用计算掩盖跨机通信延迟。

---

## 十、激活格式契约（踩坑经验）

modular kernel 的 PrepareAndFinalize（dispatch/combine）与 Experts kernel 的 `FusedMoEActivationFormat` 必须匹配：

| 激活格式 | dispatch 端 | expert 端 | 场景 |
|---|---|---|---|
| **Standard** | AllGather/ReduceScatter | FlashInfer CUTLASS | naive 路径，无预分配 buffer |
| **BatchedExperts** | DeepEP-LL / pplx | FlashInfer CUTEDSL (masked_gemm) | 需要 DeepEP/pplx buffer |

错误配对（如 CUTEDSL + allgather）会报：
```
RuntimeError: FlashInferAllGatherMoEPrepareAndFinalize.Standard
    == FlashInferCuteDSLExperts.BatchedExperts
```

---

## 十一、总结

| 维度 | 说明 |
|---|---|
| **EP group 构成** | EP = TP × DP，跨机时成员分布在多台物理机 |
| **跨机自动检测** | `self.internode = not all(in_the_same_node_as(...))` |
| **通信后端** | naive（allgather/RS）冗余但简单；DeepEP/pplx 做真 all-to-all 减少流量 |
| **跨机优化核心** | EPLB 分层算法按节点拓扑打包 expert，最小化跨机流量 |
| **机内带宽** | NVLink 900GB/s+，几乎不是瓶颈 |
| **跨机带宽** | IB NDR 50GB/s 是瓶颈；需 DeepEP RDMA + EPLB 分层打包缓解 |
| **RDMA buffer 陷阱** | DeepEP LL buffer 大小随 `max_num_batched_tokens` 增长，与 EP_size **无关** |
| **激活格式契约** | Standard ↔ CUTLASS, BatchedExperts ↔ CUTEDSL，不可混搭 |
