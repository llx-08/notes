---
title: 06 · 本地代码精读地图
date: 2026-07-24
tags: [EP, MoE, 学习笔记]
---

# 06 · 本地代码精读地图

> 上一章：[05_eplb_and_load_balance.md](/2026/07/24/ep-learning-05-eplb-and-load-balance/)  
> 实践：[07_practice_checklist.md](/2026/07/24/ep-learning-07-practice-checklist/)

默认根目录：`~/codes/vllm_comm`。括号内为次要对照仓库。

---

## Day A · 并行与入口（半日）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `vllm/distributed/parallel_state.py` | `_EP` group 如何从 DP×TP 构造；`get_ep_group()` |
| 2 | `vllm/config/` 里 parallel / kernel | `enable_expert_parallel`、`enable_eplb`、`moe_backend` |
| 3 | `vllm/envs.py` | `VLLM_ALL2ALL_BACKEND`、DeepEP 相关 env |
| 4 | `vllm/v1/worker/gpu_model_runner.py` | MoE/EPLB 在 step 中的挂钩点（搜 `eplb`） |

产出：自己画一张「进程 × GPU × EP rank」图。

---

## Day B · Modular MoE 骨架（1 日）

先读完 [01a_moe_all_to_all.md](/2026/07/24/ep-learning-01a-moe-all-to-all/)，再进代码。

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `.../fused_moe/modular_kernel.py` | 文件头设计注释；`FusedMoEModularKernel` |
| 2 | `.../fused_moe/layer.py` | `forward` / modular 初始化 / naive vs modular 分支 |
| 3 | `.../prepare_finalize/no_dp_ep.py` | 无通信基线 |
| 4 | `.../prepare_finalize/naive_dp_ep.py` | AG/RS 如何实现 prepare/finalize |
| 5 | `.../moe_align_block_size.py` | 文档例子手推 |
| 6 | `.../experts/deep_gemm_moe.py` 或 `triton_moe.py` | align 输出如何喂给 GEMM；DeepGEMM 细读见 [02a](/2026/07/24/ep-learning-02a-deepgemm/) |

产出：用文字写出一次 forward 的张量名字与 shape 变化。

---

## Day C · DeepEP 适配（1–2 日）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `vllm/distributed/device_communicators/all2all.py` | Buffer 创建、HT/LL 分支、RDMA 大小 |
| 2 | `.../prepare_finalize/deepep_ht.py` | dispatch/combine、hidden roundup、DBO handle |
| 3 | `.../prepare_finalize/deepep_ll.py` | 固定 hidden、fp8 dispatch、capacity |
| 4 | `.../prepare_finalize/deepep_v2.py` | 与 HT/LL 的 API 差异 |
| 5 | `tests/kernels/moe/test_deepep_*.py` | 最小正确性用例 |
| 6 | （对照）`sglang/.../token_dispatcher/deepep.py` | 同一库不同胶水 |

产出：HT vs LL 对比表（自己填，不要抄笔记）。

---

## Day D · MegaMoE（半日–1 日）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `vllm/config/kernel.py` | `deep_gemm_mega_moe` |
| 2 | `vllm/models/deepseek_v4/nvidia/model.py` | 约束、`finalize_weights`、`_run_mega_moe` |
| 3 | `.../nvidia/ops/prepare_megamoe.py` | staging kernel |
| 4 | `tests/models/test_deepseek_v4_mega_moe.py` | 预期行为 |
| 5 | `distributed/eplb/eplb_utils.py` | mega_moe 的 env override |

产出：画「MegaMoE 是否还走 DeepEP PrepareAndFinalize」的结论图（以你读到的代码为准）。

---

## Day E · EPLB（1 日）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `distributed/eplb/eplb_state.py` | step / prepare_forward |
| 2 | `distributed/eplb/policy/` | hierarchical packing |
| 3 | `distributed/eplb/rebalance_execute.py` | P2P 搬权重 |
| 4 | `distributed/eplb/async_worker.py` + `CpuGpuEvent` | 异步安全 |
| 5 | 跨机笔记 §五、§六 | 拓扑优化与通信量统计 |

产出：一次 rebalance 的时序图（主线程 vs async 线程）。

---

## 可选加深

| 主题 | 去哪 |
|------|------|
| RDMA / 跨机 | `notes/rdma_learning_*.md`；DeepEP 上游 internode 代码 |
| nsys 抓 PD/MoE | 技能 `dashllm-vllm-nsys`；对 MoE 层打 NVTX |
| 量化 MoE | `fused_moe/oracle/*`、`experts/*fp8*`、`*nvfp4*` |
| 弹性 EP | `distributed/elastic_ep/`（若存在实验代码） |
| 旧版 vllm（非 comm） | `~/codes/vllm/.../deepep_*_prepare_finalize.py` 扁平布局对照 |

---

## 阅读纪律（建议）

1. **先跑通调用栈，再进 CUDA**：适配层 → Python 测试 → 再开 `.cu`。  
2. **每个函数记下「同步还是异步」**：否则 DBO/EPLB 必懵。  
3. **改代码前先写复现命令**到 `07_practice_checklist.md`。  
4. 与 `vllm`（非 `_comm`）差异用一行备注记下，避免混仓库结论。
