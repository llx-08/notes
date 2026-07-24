---
title: Expert Parallelism (EP) 系统学习笔记
date: 2026-07-24
tags: [EP, MoE, 学习笔记]
---

# Expert Parallelism (EP) 系统学习笔记

> 目标：从 MoE 原理 → EP 通信 → 相关 Kernel → DeepEP → MegaMoE → EPLB，形成可落地的代码阅读与调试能力。  
> 本地主代码：`~/codes/vllm_comm`（优先）、`~/codes/vllm`、`~/codes/sglang`  
> 已有专题笔记：[vllm_cross_node_expert_parallelism.md](/notes/2026/06/17/2026-06-17-vllm-cross-node-expert-parallelism/)、[rdma_learning_1.md](/notes/2026/05/25/2026-05-25-rdma-learning-1/)

---

## 学习路线（建议顺序）

| 阶段 | 文档 | 核心问题 | 预计 |
|------|------|----------|------|
| 0 | 本文 | 全景图、术语、仓库地图 | 0.5h |
| 1 | [01_ep_fundamentals.md](/notes/2026/07/24/2026-07-24-ep-learning-01-ep-fundamentals/) | 什么是 MoE/EP？`EP=TP×DP` 怎么来的？ | 2–3h |
| 1a | [01a_moe_all_to_all.md](/notes/2026/07/24/2026-07-24-ep-learning-01a-moe-all-to-all/) | All-to-All 原因/载荷/流程；与 All-Gather、All-Reduce 对比 | 2–3h |
| 2 | [02_modular_kernel_and_moe_kernels.md](/notes/2026/07/24/2026-07-24-ep-learning-02-modular-kernel-and-moe-kernels/) | Router → Dispatch → Experts → Combine；align/permute/GEMM | 4–6h |
| 2a | [02a_deepgemm.md](/notes/2026/07/24/2026-07-24-ep-learning-02a-deepgemm/) | DeepGEMM：grouped contiguous、FP8、vLLM Experts 路径（含图） | 3–4h |
| 3 | [03_deepep.md](/notes/2026/07/24/2026-07-24-ep-learning-03-deepep/) | DeepEP HT/LL、RDMA buffer、与 NCCL/PPLX 对比 | 4–6h |
| 4 | [04_megamoe.md](/notes/2026/07/24/2026-07-24-ep-learning-04-megamoe/) | DeepSeek V4 MegaMoE、DeepGEMM、`prepare_megamoe` | 3–4h |
| 5 | [05_eplb_and_load_balance.md](/notes/2026/07/24/2026-07-24-ep-learning-05-eplb-and-load-balance/) | 热点 expert、分层 rebalance、物理/逻辑 expert | 2–3h |
| 6 | [06_code_reading_map.md](/notes/2026/07/24/2026-07-24-ep-learning-06-code-reading-map/) | 按文件/函数的精读顺序 | 随读 |
| 7 | [07_practice_checklist.md](/notes/2026/07/24/2026-07-24-ep-learning-07-practice-checklist/) | 动手实验、调试清单、常见坑 | 随练 |

---

## 一页全景图

![EP 学习全景：Router → Dispatch → Experts → Combine，及各章分工](/imgs/ep_overview.png)

*图：本系列主线。通信见 01a/03，计算见 02/02a/04，负载均衡见 05。*

```text
Token batch
    │
    ▼
┌─────────┐     topk_ids / topk_weights
│ Router  │ ─────────────────────────────┐
└─────────┘                              │
    │ hidden                             │
    ▼                                    ▼
┌──────────────────────────────────────────────┐
│ Prepare / Dispatch (All2All / DeepEP / …)    │
│  - 可选量化 (fp8 / nvfp4 …)                  │
│  - 按 expert 归属把 token 发到对应 EP rank   │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Local Experts (Fused MoE / DeepGEMM / MegaMoE)│
│  - moe_align_block_size / permute            │
│  - GEMM1 → Act×Mul → GEMM2                   │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Finalize / Combine                           │
│  - 按 topk 加权 / reduce                     │
│  - 把结果送回 token 的 home（DP 数据归属）   │
└──────────────────────────────────────────────┘
```

> 「home / 原 rank」≠「唯一做 Attn 的 TP」：Attn 同 DP 内各 TP 都算同一批 token；详见 [01 §3.3](/notes/2026/07/24/2026-07-24-ep-learning-01-ep-fundamentals/#33-token-原-rank--home指什么易混点)。

vLLM Modular MoE 把这条链路拆成可插拔组件：

```text
[Router] → [PrepareAndFinalize.dispatch] → [Experts] → [PrepareAndFinalize.combine]
```

通信后端（DeepEP HT/LL、naive AG+RS、PPLX、FlashInfer A2A…）与计算后端（Triton / CUTLASS / DeepGEMM / MegaMoE…）可以组合，不必为每种组合写一套完整实现。

---

## 关键术语速查

| 术语 | 含义 |
|------|------|
| MoE | Mixture-of-Experts：每层有多个 expert FFN，token 只激活 top-k 个 |
| EP | Expert Parallelism：把 expert 分到不同 GPU；vLLM 中通常 `EP = TP × DP` |
| TP / DP / PP | Tensor / Data / Pipeline Parallelism |
| Dispatch | 把 token 按路由结果发到持有对应 expert 的 rank |
| Combine | 把各 expert 输出按路由权重归约，并送回 token 的 home（DP 归属；见 01 §3.3） |
| All-to-All | MoE 下 token↔expert rank 的多对多交换；细讲见 [01a](/notes/2026/07/24/2026-07-24-ep-learning-01a-moe-all-to-all/) |
| Logical / Physical Expert | 逻辑 expert（模型定义）vs 物理副本（EPLB 可复制热点） |
| DeepEP | DeepSeek 开源的 EP all-to-all 库（`deep_ep`），含 HT / LL 两套 kernel |
| DeepGEMM | DeepSeek FP8/FP4 GEMM 库；MoE 用 grouped contiguous；见 [02a](/notes/2026/07/24/2026-07-24-ep-learning-02a-deepgemm/) |
| EPLB | Expert Parallelism Load Balancing：按负载重排/复制 expert |
| Modular Kernel | vLLM 将 prepare/finalize 与 experts 解耦的 MoE 框架 |

---

## 本地仓库对照

| 主题 | 优先阅读路径 |
|------|----------------|
| EP group / parallel state | `vllm_comm/vllm/distributed/parallel_state.py` |
| All2All 后端选择 | `vllm_comm/vllm/distributed/device_communicators/all2all.py` |
| MoE layer + modular | `vllm_comm/vllm/model_executor/layers/fused_moe/` |
| DeepEP HT/LL 适配 | `.../prepare_finalize/deepep_ht.py`, `deepep_ll.py`, `deepep_v2.py` |
| MegaMoE | `vllm_comm/vllm/models/deepseek_v4/nvidia/` |
| EPLB | `vllm_comm/vllm/distributed/eplb/` |
| SGLang DeepEP dispatcher | `sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |

---

## 前置知识

1. **CUDA / 通信基础**：stream、event、P2P、NCCL collective；跨机部分建议先过一遍 [rdma_learning_1.md](/notes/2026/05/25/2026-05-25-rdma-learning-1/)。
2. **PyTorch Distributed**：process group、all_gather / reduce_scatter / all_to_all。
3. **GEMM 直觉**：token×hidden 与 expert 权重的分块矩阵乘；block size / padding 为什么存在。

不必先啃完 DeepEP C++/CUDA 源码；先走通 vLLM 适配层与数据布局，再下沉到 kernel。

---

## 与已有笔记的关系

- [vllm_cross_node_expert_parallelism.md](/notes/2026/06/17/2026-06-17-vllm-cross-node-expert-parallelism/)：跨机 EP 拓扑、后端表、OOM 陷阱——本系列的「实战专题」，学完 01–03 后精读。
- 本目录：系统教材 + 代码导读；会引用上述专题，不重复粘贴全部细节。

---

## 维护约定

- 代码路径以 `~/codes/vllm_comm` 为准；若与 upstream `vllm` 不一致，在对应小节标注差异。
- 学完一章可在文末「自检」打勾；动手实验记录写在 `07_practice_checklist.md`。
- **配图**：矢量稿在 `ep_learning/imgs/*.svg`，预览 PNG 同步到 `notes/imgs/` 与 `ep_learning/imgs/`；正文用 `../imgs/xxx.png` 引用，并在相关小节旁加中文图注。
- **Hexo 博客**：本目录不会被 Hexo 直接扫描；需经 `scripts/sync_root_to_hexo_posts.py` 同步到 `hexo-site/source/_posts/`（CI / git hook 会跑）。系列入口文章 slug：`ep-learning`。

### 配图一览

| 图 | 文档 |
|----|------|
| `ep_overview.png` | README |
| `ep_tp_dp_home.png` / `ep_dp_lockstep.png` | 01 |
| `ep_collectives_compare.png` / `ep_dp0_vs_dp_a2a.png` | 01a |
| `ep_modular_kernel.png` | 02 |
| `deepgemm_*.png` | 02a |
| `ep_deepep_ht_ll.png` | 03 |
| `ep_megamoe_path.png` | 04 |
| `ep_eplb_hierarchical.png` | 05 |
