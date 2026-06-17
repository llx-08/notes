---
title: Step-3 技术报告阅读笔记：AFD 系统设计
date: 2026-06-17
tags: []
---

# Step-3 技术报告阅读笔记：AFD 系统设计

> **论文**: Step-3 is Large yet Affordable: Model-System Co-design for Cost-effective Decoding
> **机构**: StepFun Inc.
> **链接**: arXiv:2507.19427
> **阅读日期**: 2026-06-17

---

## 一、Motivation：为什么要做 AF 分离？

### 1.1 Decoding 是最值得优化的阶段

- Decoding 是推理中**最贵**的阶段（MFU 低），尤其在 test-time scaling 范式下，长思考链使 decoding 开销巨大
- 降低 decoding 成本 = 固定预算下获得更高智能（更长思考 = 更好推理）
- Faster decoding 也加速 RL 训练循环
- 目前优化空间很大，技术上也更有趣

### 1.2 现有模型设计的两个"次优实践"

| 组件 | 问题 | 后果 |
|------|------|------|
| **Attention** | 过度追求压缩 KV cache 大小（如极小 KV head），代价是计算量暴涨 | 在较弱硬件上成本反而更高；挤压了量化、投机解码等加速手段的空间 |
| **FFN** | 过度追求稀疏（MoE 激活参数越少越好），不考虑是否匹配硬件特性 | 实际运行效率低（MFU 差），或牺牲模型性能 |

### 1.3 现有推理系统的问题

- 现有 serving 系统把 Attention + FFN 当作**整体**来处理（monolithic blocks）
- 忽略了两者截然不同的计算特征和硬件亲和性
- 导致 GPU 利用率次优：Attention 受限于 FFN 的 batch 策略，FFN 受限于 Attention 的 KV cache 内存占用

---

## 二、Key Insights：核心洞察

### Insight 1: Attention 和 FFN 的计算特征本质不同

| 维度 | Attention | FFN (MoE) |
|------|-----------|-----------|
| 参数量 | 较少 | 非常多（expert weights） |
| 每 token 状态 | 需要存储 KV cache（随 context 线性增长） | 无需存储中间结果 |
| 计算模式 | memory-bound（访存 KV cache） | compute-bound（矩阵乘） |
| 瓶颈来源 | KV cache 大小 + arithmetic intensity | expert 并行通信 + batch size |
| 对 batch size 的需求 | 受 KV cache 内存限制，batch 不能太大 | 需要大 batch 才能达到高 MFU |

**核心矛盾**：Attention 希望 batch 小（省 KV cache 内存），FFN 希望 batch 大（提高 MFU）。放在一起运行时，两者互相制约。

### Insight 2: 参数量不是 decoding 成本的好指标

- Qwen3 32B 总参数和激活参数都比 DSv3、Step-3 少得多，但 decoding 成本反而**最高**
- 原因：decoding 成本由**模型架构 × 硬件特性**共同决定，不是简单看参数数量

### Insight 3: Attention 设计主导 decoding 成本

- 在 AFD 框架下分别分析 Attention 和 FFN 的成本，发现：
  - 8K context 时 Attention 已经显著贵于 FFN
  - 32K context 时差距进一步拉大
  - FFN 成本和 context 长度基本无关，Attention 成本随 context 线性增长
- **结论**：优化 Attention 的投资回报率远高于优化 FFN

### Insight 4: KV cache 大小不是唯一因素 —— Arithmetic Intensity 才是关键

**Arithmetic Intensity (AI)** = 每访问一字节 KV cache 所做的算术操作数

- 这是 attention 设计的固有属性，与 batch size、context length 无关
- AI 必须匹配硬件的 **compute-bandwidth ratio (roofline)**，否则要么 compute-bound 要么 memory-bound
- 各模型 AI 对比：
  - DSv3 (MLA): AI 很高（~8192），接近 H800 roofline (591)，在较弱硬件上 compute-bound 严重
  - Qwen3 (GQA): AI 较低，memory-bound 但 KV cache 大
  - **Step-3 (MFA): AI = 128**，刻意选在略低于大多数硬件 roofline 的位置，留出量化和 MTP 的优化空间

### Insight 5: MoE 稀疏度必须硬件感知

- 要达到高 MFU，batch size 需要 ≥ FLOPs / (2 × S × Bandwidth)，其中 S 是稀疏度
- H800: 最低稀疏度 S ≥ 0.058（Step-3 满足）；DSv3 要 S ≥ 0.058 则需要激活 14 个 expert（实际只激活 8+1）
- H20: 可以容纳更稀疏的 MoE（S ≥ 0.007）
- **过稀疏的模型（DSv3、Kimi K2、Llama 4 Maverick）在 H800 上实际 MFU 很低**

### Insight 6: 训练成本和推理成本的目标可能冲突

- Pangu Pro MoE：16.5B 激活参数，训练成本比 Step-3 低 50%+
- 但其 decoding 成本在 910B 上反而是 Step-3 的 2 倍以上
- **教训**：模型设计必须明确优化目标是训练还是推理，两者的 co-design 方向不同

---

## 三、Solution 1: MFA (Multi-Matrix Factorization Attention)

### 设计思路

利用 QK 电路的**低秩矩阵分解**，同时控制 attention heads 的数量和维度：

```
原始 Query 维度: 7168
    ↓ 低秩投影 (down-proj)
低秩空间: 2048
    ↓ Normalization
    ↓ 上投影 (up-proj)
最终 Query: 64 heads × 256 dim = 16384
```

### 核心参数

| 参数 | 值 |
|------|-----|
| Hidden dim | 7168 |
| Query heads | 64 |
| K/V heads | 1（shared） |
| Head dim | 256 |
| Low-rank query dim | 2048 |
| Attention effective rank | 16,384（= DSv3 的 MLA） |

### 为什么 MFA 更优？

1. **Arithmetic Intensity = 128**（8-bit KV 量化下），接近 A800 (156) 和 910B (175) 的 roofline，在多种硬件上都能高效运行
2. **KV cache 最小**：只有 1 个 KV head × 256 dim，比所有同级模型都小
3. **计算量适中**：仅 DSv3 MLA 的 1/4，Qwen3 GQA 的 1/3
4. **同时低计算 + 低访存**：Figure 5 中 Step-3 位于左下角，其他模型要么计算高要么访存高
5. **为进一步优化留空间**：AI = 128 略低于 roofline，未来用 4-bit KV 量化可翻倍至 256，或用 MTP 再翻倍

---

## 四、Solution 2: AFD (Attention-FFN Disaggregation)

### 4.1 系统架构

将 Transformer 每层的 Attention 和 FFN 拆分到**不同 GPU 集群**上运行：

```
┌──────────────────────────┐              ┌──────────────────────────┐
│     Attention Instance    │    RoCE     │      FFN Instance        │
│                          │   RDMA      │                          │
│  ┌──────┐  ┌──────────┐ │ ◄═══════► │  ┌────────────────────┐  │
│  │ Norm │→│ Attention │ │   fp8 →    │  │  TP gather/        │  │
│  └──────┘  └──────────┘ │   ← bf16   │  │  EP scatter        │  │
│  ┌──────┐  ┌──────────┐ │             │  ├────────────────────┤  │
│  │ Norm │→│  Router   │ │             │  │  Expert Compute    │  │
│  └──────┘  └──────────┘ │             │  ├────────────────────┤  │
│  ┌──────────────────────┐│             │  │  TP scatter/       │  │
│  │  TopK + Expert       ││             │  │  EP gather         │  │
│  │  Combine + Residual  ││             │  └────────────────────┘  │
│  └──────────────────────┘│             │                          │
│  + Embedding, LM Head    │             │  部署: TP / EP / TP+EP   │
│  并行: DP (每 GPU 独立)   │             │                          │
└──────────────────────────┘              └──────────────────────────┘
```

### 4.2 设计目标

| 目标 | 具体指标 |
|------|---------|
| 性能目标 | TPOT ≤ 50ms（≥ 20 tokens/s），3-stage pipeline 每 stage 16.6ms |
| 流水线优化 | A/F/Communication 三阶段完美流水，通信完全隐藏 |
| 独立设计 | A 和 F 可独立分析和优化，支持灵活架构修改 |
| 硬件选择 | A 和 F 可选择不同硬件，支持异构部署 |

### 4.3 多阶段流水线

以 3 个 micro-batch (D1, D2, D3) 为例：

```
时间轴 →

Attn:  [D1-A] [D2-A] [D3-A]  │  [D1'-A] [D2'-A] [D3'-A]  │ ...
         ↓      ↓      ↓     │     ↓       ↓       ↓      │
A→F:   ═══fp8═══════════════  │  ═══fp8═══════════════════  │
         ↓      ↓      ↓     │     ↓       ↓       ↓      │
FFN:        [D1-F] [D2-F] [D3-F] │  [D1'-F] [D2'-F] [D3'-F]
              ↓      ↓      ↓    │     ↓       ↓       ↓
F→A:        ═══bf16══════════════ │  ═══bf16══════════════════
              ↓      ↓      ↓    │
Attn:            [D1'-A] [D2'-A] [D3'-A]  ...
                  (下一层)
```

关键设计：
- **A→F 传 FP8**（upstream norm 后量化），节省带宽
- **F→A 传 BF16**（保持残差精度）
- **A→F 和 F→A 走独立通信路径**，不竞争带宽，可并发
- 单层时间预算：16.6ms / 61 layers ≈ **272μs**

### 4.4 通信数据内容

| 方向 | 数据 | 精度 | 量级 |
|------|------|------|------|
| A→F | hidden states + expert distribution + FP8 scales | FP8 + 少量 metadata | 较小 |
| F→A | expert combine 后的结果 | BF16 | 较小 |

EP 场景下的优化：FFN 不做跨节点的 EP combine，直接发 partial results 回 Attention，**由 Attention 端做 reduction**（合并了 EP combine 和 F→A 通信）。

### 4.5 与 DeepSeek EP 的对比

| 维度 | DeepSeek EP | Step-3 AFD |
|------|------------|------------|
| 部署规模 | 320 GPUs (DSv3) | 32 GPUs |
| Batch 放大 | 通过分发 expert weights 到多机 | 通过 AFD 自然积累大 batch 给 FFN |
| 灵活性 | 固定 expert-node mapping，负载不均 | A/F 独立扩缩容，灵活负载均衡 |
| 长文本效率 | Attention 层忙、FFN 层闲 | A 和 F 独立扩容（如 4A2F → 16A2F） |
| 异构硬件 | 必须同构 | A/F 可用不同硬件 |
| 性能建模 | 耦合，难以分析 | 解耦，可独立建模 |
| 网络瓶颈 | 大规模时网络拥塞严重 | 小规模，网络可控 |
| 和 TP-EP 的关系 | 替代方案 | **互补**，可结合 TP-EP 使用 |

### 4.6 与 Megascale-Infer 的对比

- Megascale-Infer 是第一个实现 AFD 思路的系统
- 但其 TPOT ~150ms，远高于 Step-3 的 50ms
- Megascale-Infer 主要做系统级优化，Step-3 的核心是**模型-系统 co-design**（MFA 架构本身就是为 AFD 设计的）

---

## 五、Solution 3: StepMesh 通信库

### 为什么需要专用通信库？

- 每层通信预算仅 272μs，现有库（NCCL、DeepEP）无法稳定满足
- NCCL/DeepEP 会占用 **GPU SM 资源**，挤压 A/F 的计算性能
- AFD 的通信模式（点对点、异步、双向独立）与传统 collective 不同

### StepMesh 核心设计

```
┌─────────────────────────────────────────────┐
│              StepMesh 架构                    │
│                                              │
│  Attention 端 API          FFN 端 API         │
│  ┌────────────────┐       ┌────────────────┐ │
│  │ AFTensorWorker │       │ AFTensorServer │ │
│  │  Wait → PushPull│       │ GetBatch → Respond│
│  └───────┬────────┘       └───────┬────────┘ │
│          │                        │           │
│  ┌───────┴────────────────────────┴────────┐ │
│  │         StepMesh Core                    │ │
│  │  NetRecv Thread  │  NetSend Thread       │ │
│  │  (独立收发线程)                            │ │
│  ├──────────────────────────────────────────┤ │
│  │  RDMATransport │ CPUBackend │ GPUBackend │ │
│  │  (RDMA NIC)    │  (CPU)     │  (GPU)     │ │
│  └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

关键特性：
1. **异步 API + 专用线程**：收/发各用独立 NetRecv/NetSend 线程
2. **CPU-based 操作执行**：RDMA PostSend 在 CPU 上完成，**零 GPU SM 占用**
3. **预注册 tensor**：用唯一 key 注册 GPU tensor，FFN 可直接从 Attention 的 GPU 内存 slice，无需拷贝/拼接
4. **NUMA-aware CPU core binding**：减少处理抖动
5. **支持异构加速器**：Backend 接口抽象，可扩展 xPU

### 网络优化

| 优化 | 说明 |
|------|------|
| 同 ToR 部署 | A 和 F 实例在同一 Top-of-Rack 交换机下，延迟均匀 |
| PFC-only 传输 | 禁用拥塞控制，依赖 Priority Flow Control 维持无损网络 |
| 双 NIC 负载均衡 | 每 GPU 两个 NIC 端口，每对 A-F 通信建两组 RDMA QP |

---

## 六、Solution 4: 硬件感知的 MoE 设计

### 最优稀疏度推导

达到高 MFU 的最小 batch size:
$$B_{MoE} \geq \frac{FLOPs}{2 \times S \times Bandwidth}$$

在 AFD + 3-stage pipeline + 50ms TPOT 下，网络通信约束:
$$\frac{H \times FLOPs \times L}{Net \times S \times Bandwidth} \leq 11.1ms$$

最优稀疏度:
$$S \geq \frac{H \times FLOPs \times L}{Net \times Bandwidth \times 11.1ms}$$

### 各硬件最低稀疏度（H=7168, L=61）

| 加速器 | 最小 S |
|--------|--------|
| H800 | 0.058 |
| H20 | 0.007 |
| A800 | 0.031 |
| 910B | 0.034 |

Step-3 选择 S ≈ 0.08（8-in-256 + 1 shared expert），安全覆盖 H800。

### 过稀疏的代价

- DSv3 (S=0.035): 在 H800 上需要 14 个 expert 才能满足 MFU，实际只激活 9 → MFU 低
- 实测 H800 平均带宽只有 40GB/s（标称 50GB/s），最优 S 上升到 0.073
- **Step-3 不需要大规模 EP 或路由限制来缓解 over-sparsity 问题**

---

## 七、非旗舰硬件支持

AFD 的一个独特优势：A 和 F 可以分别选择不同的硬件。

### Attention 部分

- Step-3 MFA 在 H800 上是 memory-bandwidth bound
- **4 张 L20**（内存带宽是 H800 的 25%）可以达到和 H800 一样的 attention 速度（DP 方式）
- L20 单张网络带宽需求也只有 H800 的 25%，容易满足
- 约束：平均 context ≤ 8K tokens 时 batch size ≤ 41 即可

### FFN 部分

- L20 可支撑 FFN 权重 117MB/层，61 层共 7.1GB
- 8 张 L20/server → 56.8GB FFN 权重 → 约 6 台 L20 server（48 张 L20）跑 EP
- 比 DSv3 的 320 GPU 部署小得多
- L4（带宽 300GB/s）太弱，需要 144 张，规模效益差

---

## 八、实际性能

### 端到端性能对比

| 模型 | 平均 Context | GPU 数量 | Peak TGS/GPU |
|------|-------------|---------|-------------|
| DSv3 (blog) | 4989 | 144 H800 | 1,850 |
| DSv3 (profile) | 4096 | 128 H800 | 2,324 |
| **Step-3 (FP8 attn)** | **4096** | **32 Hopper** | **4,039** |
| Step-3 (BF16 attn) | 4096 | 40 Hopper | 3,321 |
| Step-3 (FP8, 8K ctx) | 8192 | 48 Hopper | 2,643 |

Step-3 在 32 张 GPU 上实现了比 DSv3 (128 GPU) 高 74% 的单卡吞吐。

### 部署配置灵活性

| 平均 Context | 配置 | 总 Batch | 总 TGS |
|-------------|------|---------|--------|
| 4K | 2A2F | 6144 | 4039 × 4 ÷ (2+2) = ~4039/GPU |
| 8K | 4A2F | 6144 | ~2693/GPU |
| 32K | 16A2F | 6144 | ~898/GPU |

**Attention 实例数随 context 线性扩展，FFN 不变**，这是 AFD 的核心弹性优势。

---

## 九、总结：Motivation → Insight → Solution 的映射

```
Motivation                    Insight                         Solution
─────────────────────────────────────────────────────────────────────────
Decoding 太贵              → Attn/FFN 特征截然不同          → AFD 分离部署
                           → Attn 成本 >> FFN 成本          → 重点优化 Attn 设计

Attn 设计次优              → AI 匹配 roofline 比 KV 大小更重要 → MFA (AI=128)
(KV 压缩 vs 计算爆炸)      → 同时低计算 + 低访存才是最优     → 低秩 QK 分解

MoE 设计次优               → 过稀疏在强算力卡上 MFU 差      → 硬件感知的 MoE 稀疏度
(越稀疏越好的迷思)          → 网络带宽限制最优 S             → S≈0.08, 不需路由限制

A/F 耦合导致               → A 想小 batch, F 想大 batch     → AFD 让两者独立
GPU 利用率低                → 分离后可独立达到最优 MFU        → 3-stage pipeline

通信是瓶颈                 → 每层仅 272μs 预算              → StepMesh (零SM, RDMA)
                           → NCCL 占用 SM                  → CPU-based 操作
                           → EP combine 可合并              → Attn 端 reduction

硬件成本高                 → 非旗舰卡也能跑 Attn/FFN        → 异构部署 (L20 跑 Attn)
                           → 不同硬件 roofline 不同         → MFA 适配多种 roofline
```

---

## 十、个人思考

1. **AFD vs PD 分离的关系**：AFD 假设已部署 PD 分离，两者是互补关系。PD 解耦 prefill 和 decode 的资源竞争，AFD 进一步优化 decode 内部的资源利用。

2. **通信合并的可能性**：
   - EP combine + F→A 已经合并（Attention 端做 reduction）
   - A→F + EP dispatch 理论可行但拓扑复杂，目前未做
   - TP 内部通信（NVLink）与 A↔F 通信（RoCE）物理独立，用 overlap 更优

3. **对模型设计的启示**：
   - "参数少 = 便宜" 是错误的（Qwen3 32B 反例）
   - Arithmetic Intensity 匹配硬件 roofline 是核心
   - Hybrid linear attention 模型（MM M1, Llama 4 Maverick）的全注意力层会拖累整体 KV cache 优势

4. **局限性**：
   - 论文承认当前互联带宽限制了 MoE 稀疏度的进一步提升
   - 尚未启用 MTP（预估可再提升 50%+ 吞吐）
   - 仍有 jitter 需要优化（peak vs average TGS 有差距）
