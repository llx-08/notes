# DualPath: Dual-Path KV-Cache Loading for Disaggregated LLM Serving

**Paper**: arXiv 2602.21548v2

## D-side Prefix Cache 的收益

在 PD 分离架构中，D 节点开启 prefix cache 有以下好处：

1. **减少 KV 传输量**：相同前缀的请求（multi-turn 对话、shared system prompt）命中 D 端 cache 后，P 只需传输增量部分，通过 `KVTDInfo.cached_tokens` 告知 P 跳过已缓存 token
2. **降低 P 节点压力**：减少 P 端 prefill 计算量和网络发送量，释放 P 端算力处理更多新请求
3. **改善 TTFT**：传输量减少直接降低首 token 延迟
4. **多轮对话场景高收益**：对话类应用中，前几轮的 KV 可在 D 端复用，避免每轮都从 P 全量传输

## Motivation

PD 分离架构中，D 节点有两类网卡：
- **CNIC (Compute NIC)**: 用于 GPU 间通信（NCCL、KV transfer），带宽已经接近饱和
- **SNIC (Storage NIC)**: 用于访问远程存储（模型加载、checkpoint），推理期间几乎完全空闲

传统方案下，KV cache 全部通过 CNIC 从 P 传到 D。随着请求量增长，CNIC 成为瓶颈，而 SNIC 的带宽被浪费。

在 **agentic workload**（多轮调用、长上下文复用）场景下，prefix cache hit rate 很高。如果能把命中的 KV cache 存到远程存储、通过 SNIC 加载，就能利用空闲带宽减轻 CNIC 压力。

## Insight

D 节点的 SNIC 在推理期间是空闲资源。可以构建一条 **第二通路 (dual path)**：
- **Path 1 (CNIC)**: P→D 传输 cache miss 的 KV（传统路径）
- **Path 2 (SNIC)**: 远程存储→D 加载 cache hit 的 KV（新增路径）

两条路径并行工作，总带宽 = CNIC + SNIC，突破单一网卡瓶颈。

## Solution

### 架构

```
P Node ──CNIC──> D Node (GPU)
                   ↑
Remote Storage ──SNIC──> D Node (Host → GPU)
```

### 核心组件

1. **KV-Cache 存储层**：D 节点 evict 的 KV blocks 不丢弃，而是通过 SNIC 写入远程存储（如 CPFS/NFS），形成二级 cache
2. **Dual-Path Loader**：D 节点同时从两条路径加载 KV：
   - Cache miss tokens → 走 CNIC 从 P 获取（需要 P 做 prefill）
   - Cache hit tokens → 走 SNIC 从远程存储读取（不需要 P 参与）
3. **CNIC-centric Traffic Manager**：以 CNIC 为中心做流量调度，当 CNIC 空闲时优先走 CNIC（延迟更低），CNIC 饱和时将 cache hit 流量卸载到 SNIC
4. **Adaptive Scheduling**：根据 cache hit rate、CNIC/SNIC 实时带宽利用率动态调整两条路径的流量分配，避免任一路径过载

### 关键设计决策

- **写入时机**：block eviction 时异步写入远程存储，不影响 D 端正常 decode
- **一致性**：用 block hash 做 content-addressable 索引，与 vLLM prefix cache 的 hash 机制一致
- **故障处理**：SNIC 路径失败时回退到纯 CNIC 模式，不影响正确性

### 效果

- Agentic workload（cache hit rate 60-80%）下，TTFT 降低 30-50%
- CNIC 利用率从 90%+ 降至 50-60%，headroom 用于处理更多请求
- 对 cache hit rate 低的场景（< 20%）收益有限，因为大部分流量仍走 CNIC
