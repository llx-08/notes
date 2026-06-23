# GLM5.2与Indexshare

## IndexShare 用于 DSA

为了支持 1M 上下文长度，在 GLM-5.2 中，我们应用 **IndexShare** 来降低 DSA 中 indexer 的计算开销。具体而言，GLM-5.2 中每 4 个 transformer 层共享一个轻量级 indexer。该 indexer 放置在 4 层中的第 1 层，其选出的 top-k 索引被 4 层共同使用。这样可以减少 3/4 层中的 indexer 点积和 top-k 运算的计算量。GLM-5.2 从 mid-training 阶段开始以 128K 序列长度配合 IndexShare 进行训练，在长上下文基准测试上以更少的计算量超越了 GLM-5.1。

## 结合 IndexShare 和 KVShare 的 MTP

我们从两个目标出发改进了 GLM-5.2 的 MTP（Multi-Token Prediction）层用于 speculative decoding（投机解码）：

1. **最小化 MTP 层作为 draft model（草稿模型）的开销**
2. **最大化 speculative decoding 的接受率**

### 第一个目标：降低 MTP 层开销

我们同样在 MTP 层上应用了 IndexShare。在多步 MTP 中，indexer 放置在第一步，其 top-k 索引被后续所有步骤复用。但与 backbone 不同的是，不同 MTP 步骤的输入 token 是不同的。如上图所示，如果我们为 h₅ 复用 h₄ 的 top-k 索引，h₅ 只能注意到 h₁ 到 h₄，而无法注意到 h₅ 自身。我们将证明，这一特性可以帮助我们实现第二个目标——即消除 GLM-5.1 MTP 层中存在的训练-推理不一致问题。

### 第二个目标：消除训练-推理不一致

在上图中，我们展示了两步 MTP 层的推理过程：

- **第一步**：推理与训练保持一致，所有 hidden states 均来自 target model（目标模型）。
- **第二步**：h₁:₄ 来自 target model，而 h₅ 来自 MTP 层。因此，h₅ 的 KV cache 是由 target model 计算的 kv₁:₄ 和 MTP 层计算的 kv₅ 混合而成的。

而使用 IndexShare 时，h₅ 的 KV cache 仅包含 kv₁:₄，全部来自 target model 的 hidden states。

### 训练方式

- 在训练中，我们复用第一个 MTP 步骤的 **KV cache** 和 **top-k 索引**。
- 与 GLM-5.1 相同，不同 MTP 步骤之间也**共享参数**。
- 受 [arxiv:2606.12370](https://arxiv.org/abs/2606.12370) 启发，我们为 speculative decoding 引入了**拒绝采样（rejection sampling）**，并使用**端到端 TV loss** 进行训练。

### 消融实验结果

下表展示了各项技术在编码场景下对接受长度（acceptance length）的消融实验结果：

| 配置 | 实验说明 |
|------|---------|
| Backbone | GLM-5.1 |
| 训练数据 | GLM-5.1 的训练数据 |
| MTP 步数 | 训练和推理中均设为 7 |

与基线相比，最终 MTP 层的**接受长度提升了 20%**。

---

## 相关背景：IndexCache 论文

上述 IndexShare 技术的理论基础来源于 **IndexCache** 论文（arXiv:2603.12201），该论文的核心内容如下：

### Motivation（动机）

DSA（DeepSeek Sparse Attention）的 indexer 自身成为了长上下文推理的新瓶颈：

- DSA 通过轻量级 lightning indexer 为每层选出 top-k 最相关 token，将核心注意力从 O(L²) 降到 O(Lk)。
- **但 indexer 本身仍然是 O(L²) 的复杂度**，且必须在每一层独立运行。跨 N 层的总 indexer 开销为 O(NL²)，随上下文长度增长急剧膨胀。
- Profiling 显示：在 30B DSA 模型中，indexer 占总延迟的比例随上下文长度显著上升——prefill 阶段在 200K token 时 indexer 占比高达 68%，decode 阶段达 38%。

### Insight（洞察）

相邻层的 top-k token 选择高度重叠，绝大多数 indexer 计算是冗余的：

- 相邻层的 top-k 重叠率高达 70%-100%。
- 热力图揭示了明显的层簇结构：层被组织成若干功能块，块内 token 选择高度一致。
- 重叠率在块边界处才快速下降，说明只有少数过渡层会显著改变注意力焦点。

### Solution（方案）

IndexCache：通过跨层 index 复用，将 N 层划分为少量 F（Full）层和多数 S（Shared）层，消除最多 75% 的 indexer 计算。

| 方法 | 适用场景 | 核心思路 |
|------|---------|---------|
| Training-free IndexCache | 已有 DSA 模型，无需权重更新 | 贪心层选择算法：逐步将 F 层翻转为 S 层，每步选择使 LM loss 最小的层翻转 |
| Training-aware IndexCache | 从头训练或继续预训练 | 多层蒸馏损失：训练每个保留的 indexer 同时服务于它覆盖的所有层 |

### 实验结果（30B DSA 模型）

- 保留 1/4 indexer（移除 75%），在 9 个长上下文+推理 benchmark 上质量几乎无损。
- **Prefill 加速最高 1.82×**（200K token），**Decode 加速最高 1.48×**。
- 在 744B 的 GLM-5 上初步验证，保留 1/2 indexer 即可实现 ~1.2× 端到端加速且性能相当。

## Q&A

### Q: DSA 设置的 indexer，是每层 transformer 选择不同的 token 吗？

**A: 是的，但高度相似。**

在原始 DSA 设计中，每一层都有自己独立的 indexer，各层选出的 top-k token 集合是不同的。但关键在于——它们**高度相似但不完全相同**。

根据 IndexCache 论文的实验测量（附录 A 的 pairwise overlap 热力图）：

- **相邻层的 top-k 重叠率高达 70%-100%**——大部分 token 是一样的，但存在少量差异
- 重叠并非均匀分布，而是呈现**块状结构**：某些层簇内部重叠极高（接近 100%），而在**块边界**处的过渡层重叠率会快速下降
- 早期层和晚期层的重叠率很低（≤40%），说明它们关注的 token 子集有本质区别

所以原始 DSA 的做法是：每层独立运行 indexer → 各自得到不同的 top-k 集合 → 各自做 sparse attention。这正是 O(NL²) 开销的来源——N 层各做一次 O(L²) 的 index 计算。

IndexShare/IndexCache 的核心发现就是：既然相邻层选出的 token 绝大部分相同，那 3/4 的 indexer 计算其实是冗余的，完全可以复用最近一个 F 层的 index。少量不同的 token 造成的质量损失可以忽略不计（或通过训练弥补）。
