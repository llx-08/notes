---
title: Attention Drift: 自回归推测解码模型到底学到了什么
date: 2026-06-04
tags: []
---

# Attention Drift: 自回归推测解码模型到底学到了什么

> 论文：*Attention Drift: What Autoregressive Speculative Decoding Models Learn*
> 作者：Doğaç Eldenk, Payal Mohapatra, Yigitcan Comlek, Kaan Oktay, Hongyang Zhang, Stephen Xia
> 机构：Northwestern University / GE Aerospace / fal / University of Waterloo
> arXiv: 2605.09992v1（2026 年 5 月 11 日）
> 代码：https://github.com/Dogacel/Attention-Drift

---

## 一、TL;DR

- 作者首次系统地指出 **EAGLE-3 / MTP 这类自回归 drafter 在投机解码链中存在「注意力漂移（attention drift）」现象**：随着 drafter 不断生成后续 token，其注意力会逐步从 prompt 中的 sink token 漂走，转而集中到自己最近生成的 token 上。
- 根因是 drafter 在 speculation step 之间走的是**未归一化的残差路径**——hidden state 的幅度随链深度单调增大，使 drafter "学得像" 在 verifier 之上再多堆几层 Transformer，而不是一个稳定的自回归预测器。
- 提出两处极简的架构修改：
  1. **Post-norm**：把 drafter 的 RMSNorm 放到输出之前，约束 chain hidden state 的尺度；
  2. **Per-hidden-state RMSNorm**：在 `hlow / hmid / hhigh` 进 FC 融合之前，对每条目标 hidden stream 各自归一化。
- 在 7 个常用 benchmark（MT-Bench / GSM8K / Alpaca / HumanEval / MATH-500 / GPQA / LiveCodeBench）平均接受长度提升 **1.10×**；在长上下文任务（LongBench）提升 **1.18×**；在模板扰动场景下最高提升 **2×**。TTT（训练时投机深度）可以从 8 降到 4，训练耗时减少 1/3。

---

## 二、背景：投机解码与 EAGLE / MTP

**投机解码（Speculative Decoding）** 分两阶段交替：
- **Drafting**：drafter 自回归地写出 k 个候选 token（k 即 *speculation depth*）；
- **Verification**：target 模型一次前向把 k 个候选并行打分，做 rejection sampling，留下最长合法前缀。

衡量效率的核心指标是 **acceptance length τ**：每轮验证平均被接受多少个 draft token，与端到端加速比直接相关。

**EAGLE-3** 是当下 vLLM/SGLang 等推理引擎里最主流的自回归 drafter：把 target 的低/中/高层 hidden state 经 FC 融合后送入**单层 pre-norm Transformer decoder**，自带一个 LM head，loss 用 target 的预测分布做 cross-entropy。

**MTP（Multi-Token Prediction）heads** 则是与 target 一起 pretraining 的辅助预测头，复用 target 的 LM head。

**Attention sink** 指序列开头某个 token 在推理过程中持续吸走异常多的 attention（Llama 在第 0 个 token，Qwen 在第 1 个，GPT-oss 通过 per-head 学习偏置干掉了 sink），被认为是稳定长上下文注意力的锚点。

---

## 三、Attention Drift 现象

作者观察到一个普适规律：

> **target 模型在哪里有 sink，drafter 也会在那里学出一个 sink**——在 EAGLE-3 和 MTP 上同时观察到。

但 drafter 的注意力分布有一个 target 没有的动态：

- **Prefill 阶段**：drafter 在消费 verifier 的 hidden state，注意力分布类似 verifier；
- **Drafting 阶段**：drafter 开始消费自己产生的 hidden state，注意力**逐步从 sink 漂走**，向"最近生成的 token"集中。

这就是 **attention drift**，可视化为一张右下角越来越深的热力图（论文 Figure 4 / Figure 1）。

针对 Llama 3.1 8B / Qwen 3 30B / Qwen 3.5 35B / GPT-oss 120B / Qwen3.5 9B MTP 五个组合都验证了这一现象。即使是 GPT-oss 这类没有 sink 的模型，drafter 仍然在重复的模板 token（template marker）上形成弱 sink 行为，说明 sink-like 行为既来自架构归纳偏置，也来自重复的特殊 token。

作者的核心假说：这种"两段式注意力" 让 drafter 对 **OOD 输入** 异常脆弱，尤其在长上下文和深推测步上。

---

## 四、根因分析：是什么造成了 Drift？

### 4.1 Hidden State 幅度分析

用 RMS = ‖x‖₂/√d 在 80 条 MT-Bench prompt × k=8 推测轮上比较 verifier 的低/中/高层、FC 融合后、drafter 在 k=1/2/3/8 的输出幅度（论文 Table 1）。三个观察：

1. **幅度严重错配**：verifier、FC、drafter 之间的 hidden state 量级差几十倍以上。但 acceptance rate 与幅度并无强相关，说明模型自己已经学会补偿这种偏差。
2. **hFC 不平衡**：target 都是 pre-norm 架构，残差流本身从不被归一化（pre-norm dilution），所以 `‖hhigh‖ ≫ ‖hmid‖ ≫ ‖hlow‖`。EAGLE-3 直接拿 pre-norm 之前的内部状态，使得 `hhigh` 主导融合信号。**修复办法：在每条目标 hidden stream 进 FC 之前各加一个 RMSNorm。**
3. **幅度随推测深度单调生长**：drafter 的 h*ₖ 在所有模型家族上都随 k 单调增大，使得 drafter 在 k=8 时面对的是一个完全脱离训练分布的输入尺度。

> **drift 与 magnitude 的关系**：attention 依赖 query-key 相似度。深处的 drafter hidden state 越来越像最近生成的几个 hidden state（同样幅度大），于是 attention mass 自然就被吸过去了。

### 4.2 Norm 位置：层堆叠 vs 自回归

这是论文最有洞察力的论断之一：

> 标准 pre-norm 的 drafter，由于残差路径未归一化、hidden state 单调增长，drafter **在结构上更像在 target 上又堆了 N+1, N+2, …, N+k 层 Transformer**，而不是一个深度无关的自回归预测器。它学到的是"沿深度方向继续 refine"，而非"自回归地预测下一个 token"。

实验设计：用不同 train-time-test depth（TTT）训练 pre-norm 和 post-norm drafter：
- pre-norm + TTT=2：在 k≤2 内表现不错，但 k 超过 2 后**直接崩盘**（hidden state RMS 飙升、条件接受概率几乎归零）；
- pre-norm + TTT=8：在 k≤8 表现最好，超过 8 后开始衰减；
- **post-norm + TTT=2：在 k=2 之外仍然稳定**，magnitude 始终被 norm 钉住。

说明：post-norm 阻断了"用幅度编码深度"的捷径，把 drafter 正则成一个真正的、深度不变的自回归预测函数。

### 4.3 Magnitude 与 Sink 是相互独立的失败模式

对比 4 种架构（Llama 3.1 8B，MT-Bench，论文 Table 2）：

| Drafter | h*₁→₈ | 熵 H₁→₈ | sink₁→₈ | recent₁→₈ |
|---|---|---|---|---|
| Pre-norm | 3.92 → 14.02 | 2.04 → 2.73 | 0.46 → 0.08 | 0.14 → 0.31 |
| Gated-Attn | 8.41 → 39.07 | 2.39 → 2.13 | 0.03 → 0.02 | 0.29 → 0.32 |
| Post-norm | 1.21 → 1.20 | 2.24 → 2.29 | 0.11 → 0.10 | 0.50 → 0.53 |
| Gated + Post-norm | 0.87 → 0.88 | 0.63 → 0.64 | 0.00 → 0.00 | 0.50 → 0.51 |

结论：
- **Gated Attention** 能把 sink 杀掉，让 drift 现象消失，但 hidden state 幅度仍会 5× 放大；
- **Post-norm** 能把幅度死死按住，attention 分布也稳定；
- **二者叠加**反而出现**熵塌缩**（H ≈ 0.62 ≈ 仅 ~1.85 个有效 attended 位置），模型 attention 几乎只聚在两个 token 上，等于过度正则化；
- **熵** 与 acceptance length 之间没有稳定的对应关系，不能作为质量诊断指标。

### 4.4 Noise / 误差累积

实验：对 drafter 的 hidden 路径或 embedding 路径分别加幅度 α·rms(x) 的高斯噪声，看接受长度相对无噪声的下降比例（论文 Table 3）：

| 架构 | 通路 | α=0.1 | α=0.5 | α=1.0 |
|---|---|---|---|---|
| Pre-norm (3.06) | Hidden | 82% | 5% | 0% |
| Pre-norm | Embedding | 99% | 86% | 64% |
| Post-norm (3.16) | Hidden | 99% | 58% | 22% |
| Post-norm | Embedding | 98% | 93% | 75% |

post-norm 对 hidden-path 噪声的容忍度比 pre-norm 高一个数量级，可能也意味着对 verifier 量化、轻度分布漂移更鲁棒。

**反向实验**：推理时人为把 pre-norm 的 hidden RMS 缩到 FC 的尺度，attention drift 明显减弱但仍存在，且 accuracy 反而下降 56%（post-norm 下降 34%）。说明 **magnitude 累积是 drift 的一个原因，但不是唯一原因**，光靠推理时规约不行——必须训练时就用 post-norm 这种归一化结构。

### 4.5 训练窗口与 MTP 行为

- **训练窗口效应**：EAGLE 类训练用固定 context window，drafting 时旧 token 会被挤出窗口，drafter 自然学会减少对早期 prompt 位置的依赖——这可能进一步放大 sink 弱化。
- **MTP 的 drift 形态不同**：MTP 是 post-norm，但 drift 仍存在，呈"前几步陡降→稳定到新基线"的模式（论文 Figure 7）。MTP 与 target 联合训练、复用 LM head，训练损失对 drift 的贡献尚不清楚，留作 future work。

---

## 五、性能影响（Performance Impact）

### 5.1 提议的新架构

post-norm 架构（论文 Figure 13 右图）做了两件事：
1. 在 hlow / hmid / hhigh 各加一个 RMSNorm，再喂给 FC；
2. drafter 的 RMSNorm 移到累加之后、LM Head 之前（即 post-norm）。

使用 post-norm 后，TTT 从 8 → 4，训练耗时减少约 1/3，且性能不降反升。

**SGLang 7 个 benchmark（GPT-OSS 20B）平均接受长度**全部提升，最大 +12%；
4 个 target 模型（Llama 3.1 8B, Qwen 3 8B, Qwen 3.5 9B, GPT-OSS 20B）上 post-norm 普遍 ≥ pre-norm，唯一回退是 Llama 3.1 8B 在 MATH-500 上 5.87 → 5.81（在评测噪声范围内）。

### 5.2 模板敏感性（Template Sensitivity）

drafter 通常在 SFT 数据上训练，会过拟合 chat template。作者制造四种扰动：`template+BoS / template no BoS / no template (BoS auto) / no template no BoS`。Llama 3.1 8B：

| 扰动 | Pre-norm | Post-norm |
|---|---|---|
| template + BoS | 2.74 | 2.73 |
| template, no BoS | 2.28 | 2.65 |
| no template (BoS auto) | 1.69 | 2.61 |
| no template, no BoS | 1.34 | 2.59 |

pre-norm 最差掉 52%，post-norm 最差只掉 5%；Qwen3-8B 和 GPT-oss 20B 上 post-norm 同样在 5–35% 的范围内领先，扰动越大领先越多。

作者还验证了一个关键问题：**模板敏感性是不是 sink 的副作用？** 不是——gated attention（杀掉 sink）仍然高度敏感、对 system prompt 长度甚至更敏感。**只有修复 hidden-residual dynamics 才能彻底缓解 template fragility。**

另一个可能因素：EAGLE 的训练 loss 只在 assistant token 上算，user 位置不直接受约束，当模板变了，监督/非监督边界一变，模型就懵了。

### 5.3 长上下文（Long Context）

drafter 通常用 4K 上下文训练，但 target 经常面对几十 K 上下文。作者构造了多轮对话超长上下文 benchmark + LongBench：

- **Full attention** 下 pre-norm 直接挂掉（acceptance 0.05），post-norm 0.83（仍不可用，但是 15× 好）；
- 引入 **SWA（sliding window attention）** 后两者都能 rescue：
  - 单纯 SWA：pre 2.43 / post 2.97；
  - SWA + 携带 BoS：pre 2.53 / post 2.99（pre-norm 受益更多，因为它依赖 sink）；
  - SWA + 携带 system prompt：pre 0.06（崩溃，无法适应更宽的 positional embedding 范围）/ post 3.01（再涨 1%）；
- SWA 各种模式下，post-norm 比 pre-norm 一致领先约 20%；
- **LongBench**（Window=1024）：post-norm 在 GovReport（summarization）/ Samsum（few-shot）/ Repobench（coding）三类任务上比 pre-norm 高 20–25%。

**Window 大小** 实验：哪怕只有 256 token 的窗口也能回收 80% 的 full-context baseline，1024 token 之后收益就饱和——这意味着 drafter 主要依赖近期 token，SWA 几乎可以无损削减计算。

---

## 六、Related Work / Limitations / 结论

- **Related work**：投机解码族（Medusa / Hydra / D-Flash / EAGLE 系列）、attention sink（Xiao 等）、gated attention（Qiu 等）、Transformer 中 pre-norm vs post-norm（Xiong / Wu 等）。作者首次在 drafter 上系统刻画 attention 行为，识别出 chain-residual 特有的 drift 失败模式。
- **Limitations**：(1) 主要聚焦 EAGLE-3；MTP 上观察到 drift 但没深入；其他 drafter（Medusa/Hydra）未验证；(2) 受算力限制只在 ≤120B 规模上做了实验。
- **结论**：attention drift 本身只是表象，**真正的病根是 unnormalized residual 路径导致的 hidden state 单调增长**。post-norm + per-stream RMSNorm 直接对症下药——既稳定了幅度，又阻止 drafter 学到"深度依赖的层堆叠"行为，从而带来性能与鲁棒性的双重收益。建议生产系统默认改用 post-norm drafter。

---

## 七、附录要点

- **A. Gated Attention**：在 EAGLE3 attention 输出乘上一个 per-element sigmoid gate `g = σ(xWg)`，再走 output 投影。
- **B. Benchmark 详表**：覆盖 GPT-oss 20B / Qwen3.5 9B / Qwen3 8B / Llama 3.1 8B 四个模型 × 多种 reasoning mode；LongBench-E 按 prompt 长度（0–4k 一直到 32–36k）分桶，post-norm 全胜。
- **C. Training**：基于改造的 *SpecForge* 仓库，Llama/Qwen3 用 Open-PerfectBlend（target 重新生成的回答），Qwen3.5/GPT-oss 用 Nemotron post-training 数据。LR=1.5e-4，bsz=4，单模型训练约 36–48 H200·hour。
- **D. 模板扰动样本**：`regular` / `no_bos` / `no_template` / `no_bos_no_template` 四种 Llama 模板的具体长相。

---

## 八、对工程的启示

1. **EAGLE-3 类 drafter 的默认架构在生产中潜在脆弱**：模板扰动、长上下文、深推测都会显著掉点。
2. **改 post-norm 是几乎无副作用的优化**：只需移动 RMSNorm 位置 + 在 FC 前各加一个 RMSNorm，外加重训。
3. **TTT 可以从 8 降到 4**：训练吞吐 +33%，性能反而更好。
4. **SWA 是长上下文的强工具**：哪怕 256 token 窗口都能 rescue 80% 性能，1024 token 即饱和。在 SWA 中携带 BoS 对 pre-norm 收益大，但若想再携带 system prompt，post-norm 才扛得住——pre-norm 会因 positional embedding 范围扩大而崩溃。
5. **不要盲目相信"杀掉 sink 就解决问题"**：gated attention 并不能根治模板敏感性，反而可能引入新病灶（熵塌缩、对 system prompt 长度过敏）。Drift 是症状，幅度累积才是病灶。



---

## 九、补充 Q&A

### Q1. "Sink token" 指什么？

**Attention Sink（注意力汇聚点）** 这个概念最早由 Xiao et al. 2024（*Efficient Streaming Language Models with Attention Sinks*）提出，论文里直接引用了这篇工作 [12]。

**定义**：序列里某个特定位置（通常在序列开头）的 token，会**吸收远超平均水平的 attention 权重**——很多 head 在很多 query 位置上都会把大量 attention "倒进"这个 token，几乎不管它的语义内容是什么。

**为什么会出现 sink？** 直观解释是 softmax 强制要求 attention 权重总和为 1，但很多时候模型其实"不太需要关注任何位置"，于是就把多余的 attention "倒"到一个固定的早期 token 上当垃圾桶用。这个 token 也起到了**稳定锚点**的作用，尤其在长上下文 / 滑动窗口下，sink 一旦被丢出窗口，模型质量就会显著退化。

**论文里观察到的三种 sink 行为**（Section 3）：

| 模型族 | Sink 位置 | 备注 |
|---|---|---|
| Llama | 第 0 个 token（即 BoS `<\|begin_of_text\|>`） | 经典的 sink，attention 占比可达 40%+ |
| Qwen | 第 1 个 token（紧跟 BoS 之后） | sink 不在 BoS 而在第二个位置 |
| GPT-oss | 无明显 sink | 架构上引入 per-head 可学习偏置 logit，让 softmax 显式有"什么都不关注"的选项 |
| Qwen3-Next | 无明显 sink | 用 gated attention 在每个 head 输出乘 sigmoid 门，可乘法抑制贡献 |

**关键发现**：**target 在哪里有 sink，drafter 就在哪里学出 sink**——drafter 完全继承了 target 的 sink 位置。但 drafter 有个 target 没有的额外动态——**随着 speculation step 加深，注意力会从 sink 漂走，向最近生成的 token 集中**，这就是论文命名的 attention drift。

**和我们工程上的关系**：
- 模板扰动（去掉 BoS / 改 chat template）相当于把 sink 位置挪走或抹掉，pre-norm drafter 因为强依赖 sink 直接掉点；
- 长上下文 + SWA 场景里，"携带 BoS 通过窗口"（论文叫 SWA+BoS）对 pre-norm 收益巨大、对 post-norm 几乎无影响，也是因为 post-norm 不再死磕 sink；
- 即使是 GPT-oss 这种"无 sink"的模型，drafter 仍会在重复出现的特殊 template token（如 `<|start_header_id|>`）上形成弱 sink，说明 **sink-like 行为既来自架构归纳偏置，也来自重复的模板 marker**。

---

### Q2. "drafter 自回归地写出 k 个候选 token（k 即 speculation depth）"——这就是 gamma 吗？对应 `speculative_config` 里的字段？文章是不是希望解决"gamma 很大时仍然保持高接受率"？

**前半部分：是的，对应关系如下。**

| 名词 | 出处 | 含义 |
|---|---|---|
| **γ (gamma)** | Leviathan et al. 2023（投机解码原始论文） | 每轮 drafter 生成的候选 token 数 |
| **k / speculation depth** | 本文 Attention Drift 论文 | 同上 |
| **`num_speculative_tokens`** | vLLM `SpeculativeConfig` | 同上，工程实现里的名字 |
| **TTT (train-time-test depth)** | EAGLE-3 论文 / 本文 | **训练时**最大推测深度（与推理时的 k 可以不同） |

在 vLLM 仓库里可以直接看到（`/mnt/data/llx/vllm/vllm/config/speculative.py:68`）：

```python
num_speculative_tokens: int = Field(default=None, gt=0)
"""The number of speculative tokens, if provided. It will default to the
number in the draft model config if present, otherwise, it is required."""
```

也就是说，**论文里的 k 就是你启动 vLLM 时设置的 `--num-speculative-tokens` 或 SGLang 里 `--speculative-num-draft-tokens` 那个值**。EAGLE 训练侧（如 SpecForge）会另有一个 TTT 配置，决定训练时 drafter 一次性 unroll 多少步算 loss。

**后半部分：基本对，但不完整。** 论文要解决的问题比"调大 gamma"更宽，可以拆成三件事：

1. **k 很大时不要崩**（你说的这一条）
   - pre-norm + TTT=2 的 drafter 在 k>2 后**直接塌掉**（条件接受概率掉到 0、hidden state RMS 飙升）；
   - pre-norm + TTT=8 的 drafter 在 k≤8 表现最好，但超过 8 仍会衰减；
   - post-norm + TTT=2 的 drafter **在 k=2 之外仍然稳定**，可以推广到 k=16、k=20。
   - 也就是说：post-norm 让 drafter **能在推理时把 k 开得比训练时更大**，而不需要为了"想跑更大的 gamma"就把训练 TTT 也调大。

2. **训练时 TTT 可以变小，省训练成本**
   - 后文实际上把 TTT 从 8 降到 4，**训练耗时减少约 1/3**，性能不降反升。
   - 这点是单纯"调大 gamma 解决接受率"框架里没有的好处。

3. **OOD 场景下的鲁棒性**（这才是论文真正的卖点）
   - **模板扰动**：去掉 BoS / 换 chat template / 改 system prompt 长度，pre-norm 最差掉 52%，post-norm 最差只掉 5%；
   - **长上下文**：full attention 下 pre-norm 几乎全军覆没（0.05），post-norm 也只剩 0.83；SWA 救场后，post-norm 在所有窗口大小、所有 SWA 变体上一致领先 pre-norm 约 20%；
   - **噪声容忍**：post-norm 对 hidden-path 噪声的容忍度比 pre-norm 高一个数量级。

**所以更准确的概括是**：
> 论文不是只为了"把 gamma 调大还能高接受率"——而是想让 drafter **既能在更深的 speculation chain 上稳定（k 大时不崩）**，**又能在 OOD 部署场景（模板/长上下文/system prompt）下保持鲁棒**，**还能让训练 TTT 变小省钱**。三件事的共同病根都是 unnormalized residual 导致的 hidden state 幅度累积，所以 post-norm 一招解决。

**实操建议（落到 vLLM）**：
- 如果你只是把 `num_speculative_tokens` 从 4 调到 8，用现成的 pre-norm EAGLE-3 drafter，acceptance 很可能反而**下降**（因为 drafter 训练时 TTT 是固定的，推理超过 TTT 就开始漂）；
- 如果你想要可用的大 k，要么训练时把 TTT 也调大（成本高），要么换成 post-norm drafter（论文方案，几乎免费）；
- 如果你的 prompt 模板和训练数据不一致（比如自定义 system prompt、关掉 thinking 标签），pre-norm drafter 的接受率会被严重侵蚀，这种场景 post-norm 收益最大（最高 2×）。


---

## 十、补充 Q&A（第二轮）

### Q3. "target 在哪里有 sink，drafter 也会在那里学出 sink" 这句话到底是什么意思？

先把 "sink" 这个词彻底讲清楚。

#### 3.1 Sink 在 attention 热力图上到底长什么样

attention 热力图的画法是：
- **行 = query 位置**（"现在我在算谁的输出"）
- **列 = key 位置**（"现在我在看谁"）
- **颜色越深 = 该 query 给该 key 的 attention 权重越大**

正常情况下，对角线（自己看自己附近）会比较深，其他地方比较浅。

**Sink 的视觉特征：某一列从头到尾几乎全部很深**——也就是说，几乎每一个 query（不管它在哪里）都会给这一列分配一大块 attention。这个 "永远被看着的列" 对应的 key 位置，就是 sink token。

论文 Figure 4 的 Llama3 8B 热力图里，最左边 "sink" 那一列整列都是深红色（0.4~0.5 的权重），而其他列都很浅（<0.1）——这就是典型的 sink。

#### 3.2 "Sink" 是哪个 token？

| 模型 | sink 位置 | 具体是哪个 token |
|---|---|---|
| Llama-3 系列 | 第 0 个 | `<\|begin_of_text\|>` (BoS) |
| Qwen-3 系列 | 第 1 个 | BoS 后的第一个 token（往往是 `<\|im_start\|>`） |
| GPT-oss | 无明显 sink | 架构里有显式 "attend-to-nothing" 偏置 |

注意：**sink 不一定是 BoS**。Qwen 的 sink 就在第二个位置而不是第一个。研究者也不完全清楚为什么，但实证上每个 LLM 家族都有一个相对固定的 sink 位置。

#### 3.3 为什么 sink 会自发出现？

直观解释：softmax 强制 `Σ attention = 1`，但很多 query 其实**没什么特别想看的**（比如生成一个空格、标点）。这时它必须把 attention 倒到某个地方——干脆挑一个固定的早期位置当**垃圾桶 / 备用栈**，把"用不上的 attention 配额"都倒进去。

这个 sink 位置一旦被模型选定，就成了**稳定锚点**：所有 head、所有 query 都默认它在那里，相当于一个全局共享的"零参考点"。

#### 3.4 "target 在哪里有 sink，drafter 也在那里有 sink" 的真正含义

这句话不是说"所有模型都有 sink"——而是说**drafter 的 sink 位置完全继承自它的 target**：

- 你用 **Llama-3 8B** 当 target 训出来的 EAGLE-3 drafter → drafter 的 sink 也在**第 0 个 token**
- 你用 **Qwen3 30B** 当 target 训出来的 EAGLE-3 drafter → drafter 的 sink 也在**第 1 个 token**
- 你用 **GPT-oss 120B**（没有 sink）当 target 训出来的 drafter → drafter 也**没有明显 sink**（只有弱的模板 token 上的聚集）

这种"一一对应"在 5 个模型组合上都得到验证。

为什么会这样？因为 EAGLE-3 drafter 的输入是 target 的 `hlow / hmid / hhigh` 融合后的 hidden state，而这些 hidden state 本身就在 target 的 sink 位置上累积了大量信号；drafter 想要复现 target 的预测分布，自然就学会了在同一个位置上也分配大量 attention。

> **一句话总结**：sink 是 attention 热力图上"永远被看着的那一列对应的 token"，drafter 会**完美继承 target 的 sink 位置**，这是 drafter 学到的第一个 attention 模式。

---

### Q4. Figure 7 那张图到底在画什么？

Figure 7 是两张并排的折线图，对象是 Qwen3.5 9B 的 MTP head，数据来自 MT-Bench 的 80 条 prompt。我们一项一项拆。

#### 4.1 横轴：相对于 "drafting 开始"的位置

x 轴范围是 **-6 到 +10**，中间有一条**虚线 x=0**，标注 `prefill → drafted`。这条虚线代表 **drafter 从 "消费 verifier 的 hidden state（prefill）" 切换到 "消费自己生成的 hidden state（drafting）"**的那一瞬间。

- **x = -6 到 -1**：drafter 当前正在处理的 query 位置还在 prompt 区域（prefill 阶段的最后 6 个位置）；
- **x = 0**：drafter 正好处理 prompt 的最后一个位置，下一步就要开始自己 draft 了；
- **x = 1 到 10**：drafter 正在 draft，处理的是它自己刚刚生成的第 1、2、3 ... 10 个 token。

> 你也可以把 x 轴理解成 "speculation step k 的连续版本"——负数是 prefill 还没开始 draft，正数是已经 draft 到第几个 token。

#### 4.2 左图：Sink Attention

- y 轴范围：4% ~ 12%
- 含义：**当 query 位置在 x 时，它分配给 sink token 的 attention 权重百分比**
- 走势：
  - prefill 阶段（x<0）：稳定在 **10~12%** 左右
  - 一过 x=0（drafting 开始），**陡降到 4~5%** 并保持

**这意味着**：drafter 一旦切换到"消费自己生成的 hidden state"，它就立刻开始"忘记" sink token——sink 的 attention 权重瞬间被砍到原来的 1/3。

#### 4.3 右图：Token Self Attention

- y 轴范围：30% ~ 45%
- 含义：**当 query 位置在 x 时，它分配给"自己（或紧挨着的最近几个 token）"的 attention 权重百分比**
- 走势：
  - prefill 阶段（x<0）：稳定在 **30~35%**
  - 一过 x=0（drafting 开始），**跳升到 40~45%** 并保持

**这意味着**：被从 sink 那里"省下"的 attention 权重，**全部跑到了 drafter 最近自己生成的 token 上**——attention 的"质量重心"从 prompt 头部漂到了链尾。

#### 4.4 整张 Figure 7 的核心信息

把两张图放在一起，左图掉多少（~7%），右图就涨多少（~10%），这就是**attention 漂移的定量证据**：

> drafter 从 prefill 切换到 drafting 的那一瞬间，attention mass 从 "看 prompt 开头的 sink" **整体迁移**到 "看自己刚生成的几个 token"，而且这种迁移**在 drafting 一开始就发生、之后稳定保持**，不是慢慢累积的。

这一点和 EAGLE-3 的 drift 形态略有不同（EAGLE-3 是**逐步**漂移）：MTP 是**陡降然后稳定**。论文用这一点说明：post-norm（MTP 本身就是 post-norm 结构）能压住"渐进漂移"，但因为还有其他机制（联合预训练、共享 LM head 等），仍然存在一次性的"模式切换"。

---

### Q5. "超过 TTT 推理 acceptance 反而下降" 这个说法对吗？我之前的解释有错吗？

你的质疑是对的，**我那句话表述不准确，需要修正。** 让我重新说明清楚。

#### 5.1 你的直觉是对的：早期 token 不会被晚期 token "污染"

drafter 确实是自回归生成的。在第 k 步生成 token 时，drafter 看到的输入是：
- prompt 的 hidden state（来自 target）
- 第 1 步到第 k-1 步自己生成的 hidden state

**对于 k=1 来说，输入完全和"训练时 k=1"是一样的**——根本看不到任何超出 TTT 的东西。所以**第 1 个 draft token 的质量不会因为你把 num_speculative_tokens 调大就变差**。

这一点论文 Figure 10 的数据也直接证实了：

| Drafter / TTT | k=1 acceptance | k=16 acceptance | 平均 acceptance length |
|---|---|---|---|
| Pre-norm / TTT=8 | 0.91 | 0.50 | **7.12** |
| Pre-norm / TTT=2 | 0.93 | 0.00 | **2.64** |
| Post-norm / TTT=2 | 0.93 | 0.53 | 5.05 |

无论 TTT 设几，**k=1 的接受概率都是 0.91~0.93**——证明早期 token 的质量与 TTT 无关。

#### 5.2 那为什么平均 acceptance length 差别这么大？

因为**投机解码的接受规则是"一刀切"**：

> 一旦某一步的 token 被 reject，**它之后的所有 token 全部被丢弃**，不管那些 token 本身质量如何。

所以平均 acceptance length 的实际含义是：**"chain 在第几步被首次 reject"** 的期望值。

- Pre-norm/2 在 k=3 之后就 reject 率飙高 → chain 经常在 k=2、k=3 就被截断 → 平均 acceptance length 只有 2.64
- Pre-norm/8 在 k=8 之后才开始崩 → chain 平均能延续到 k=7 → 平均 acceptance length 7.12

**这并不是"前面的 token 被污染了"，而是"后面的 token 容易把整条 chain 提前砍掉"。**

#### 5.3 那么"把 num_speculative_tokens 从 4 调到 8" 到底会怎样？

假设你用的是**现成 production 的 EAGLE-3 drafter（TTT=8）**：

| `num_speculative_tokens` | 实际发生的事 |
|---|---|
| 4 | 4 步都在 TTT 内，每一步质量都不错；平均 accept ≈ 3.5（举例） |
| 8 | 8 步都在 TTT 内，每一步质量都不错；平均 accept ≈ 6.5 |
| 16 | 前 8 步质量好，后 8 步漂移严重；但因为前面被接受的概率已经接近上限，chain 多半还是在 k=7~8 附近被截断；平均 accept ≈ 6.8（**几乎不再增长，进入 plateau**） |

**所以更准确的说法应该是**：
- ✗ ~~"调大 num_speculative_tokens 会让 acceptance length 下降"~~ —— 这是错的
- ✓ "调大 num_speculative_tokens 超过 TTT 后，**acceptance length 会饱和**（plateau），多算的那些步基本是浪费"
- ✓ "更严重的是，**端到端吞吐反而可能下降**"——因为 drafter 每多 forward 一步都要花时间，但这些步基本都会被 reject，性价比为负

#### 5.4 那论文真正想解决的问题是什么？

> **想让"训练时 TTT 小、推理时 k 大"也能 work**——这样既能省训练成本（TTT 8→4 训练时间省 1/3），又能在推理时自由调大 k 拿到更深的 chain 收益。

pre-norm 的 drafter 死死被 TTT 锁住：训练 TTT=2 就只能在 k=2 范围内用，调大没意义。
post-norm 的 drafter 能"超规格使用"：训练 TTT=2 也能在 k=16 上保持 0.53 的接受率，相当于把 chain 长度从 ~2 提到 ~5。

#### 5.5 对前一轮回答的修正

我之前写的"如果你只是把 num_speculative_tokens 从 4 调到 8 ... acceptance 很可能反而下降"这句话**表述不严谨**。

**修正版**：
> 如果你的 EAGLE-3 drafter 是 TTT=8 训练的，你把 num_speculative_tokens 从 4 调到 8，acceptance length **会涨**（毕竟还在 TTT 内）。但如果你把它从 8 调到 16，acceptance length **不会继续涨**，只会饱和在 8 附近，多花的 drafter forward 算力基本是浪费、端到端吞吐反而可能掉。如果你用的是 TTT=4 的 drafter，那么 num_speculative_tokens 一旦超过 4，就立刻进入 plateau，多调毫无意义。
>
> 论文真正的卖点是：用 post-norm 训出来的 drafter，**即使训练 TTT 很小，推理时也能把 k 调大并真正吃到收益**——这才是"调大 gamma 仍然有效"的正确语境。
