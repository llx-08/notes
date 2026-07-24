---
title: 02 · Modular Kernel 与 MoE 相关 Kernel
date: 2026-07-24
tags: [EP, MoE, 学习笔记]
categories: [EP 学习笔记]
---

# 02 · Modular Kernel 与 MoE 相关 Kernel

> 上一章：[01a_moe_all_to_all.md](/notes/2026/07/24/2026-07-24-ep-learning-01a-moe-all-to-all/) ← [01_ep_fundamentals.md](/notes/2026/07/24/2026-07-24-ep-learning-01-ep-fundamentals/)  
> 下一章：[02a_deepgemm.md](/notes/2026/07/24/2026-07-24-ep-learning-02a-deepgemm/)（DeepGEMM 细讲）→ [03_deepep.md](/notes/2026/07/24/2026-07-24-ep-learning-03-deepep/)

---

## 1. 为什么要 Modular

历史问题：每种「通信 × 量化 × GEMM 实现」组合都会爆炸成一份 fused 代码。

vLLM 的解法（`modular_kernel.py` 文件头注释）：

```text
[Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]
```

三大类接口：

| 类 | 职责 |
|----|------|
| `FusedMoEPrepareAndFinalizeModular` | 量化 + dispatch；combine +（可选）topk 加权归约 |
| `FusedMoEExpertsModular` | 本地 expert：permute / GEMM / act / unpermute |
| `FusedMoEModularKernel` | 把上面两者粘成统一 `forward` |

**Prepare 与 Finalize 绑在同一对象**：因为集体通信的 handle、buffer、布局必须配对（DeepEP 的 `dispatch` handle 要交给 `combine`）。

![Modular MoE：Router + PrepareAndFinalize + Experts 可插拔组合](/imgs/ep_modular_kernel.png)

*图：通信后端与计算后端解耦，由 Oracle 选合法组合。*

路径（`vllm_comm`）：

```text
model_executor/layers/fused_moe/
├── modular_kernel.py          # 抽象与编排
├── layer.py                   # FusedMoE 层、选后端
├── prepare_finalize/          # 通信+量化适配
│   ├── deepep_ht.py
│   ├── deepep_ll.py
│   ├── deepep_v2.py
│   ├── naive_dp_ep.py
│   ├── no_dp_ep.py
│   ├── batched.py
│   ├── flashinfer_nvlink_*.py
│   └── ...
├── experts/                   # 各种 GEMM 后端
├── router/                    # topk / grouped topk
├── moe_align_block_size.py
├── moe_permute_unpermute.py
└── runner/                    # 高层 runner / shared experts
```

---

## 2. 端到端一次 forward（概念 trace）

```text
1. Router
   hidden [T, H] → topk_ids [T, K], topk_weights [T, K]

2. Prepare (dispatch)
   - 可选：对 hidden 做 fp8/nvfp4 量化，得到 (x_q, scales)
   - all-to-all：按 expert 归属交换 token
   - 产出：local_x、expert_num_tokens、（可能已交换的）topk 元数据
   - format：Standard 或 BatchedExperts

3. Experts
   - align / permute：按 expert 聚集 token，pad 到 block
   - w1 GEMM → SiLU×Mul（或其它 activation）→ w2 GEMM
   - 部分实现内部已做 weight×reduce；通过 TopKWeightAndReduce 告知 Finalize

4. Finalize (combine)
   - 按需要应用 topk_weights 并 reduce
   - all-to-all 回传 / 或本地 reduce
   - 输出 [T_local, H]
```

DBO（dual batch overlap / microbatching）打开时：Prepare 可异步，Experts 与下一 ubatch 的通信重叠；DeepEP HT 会按 ubatch 存多份 handle。

---

## 3. 核心 Kernel 清单

### 3.1 `moe_align_block_size`

文件：`fused_moe/moe_align_block_size.py`（底层常进 `_custom_ops` / Triton）

**问题**：Grouped GEMM 要求每个 expert 的 token 数能被 `block_size` 整除。

**做法**：

1. 展平 `topk_ids` → 得到「token×expert 槽位」列表  
2. 按 expert 排序 → `sorted_token_ids`  
3. 每个 expert 不足 block 的部分 **pad**（padding 的 token id 指向哨兵，后续 GEMM mask 掉）  
4. 产出 `expert_ids`（每个 block 属于哪个 expert）和 `num_tokens_post_padded`

EP 注意：

- 计数时常按 **global experts** 对齐；  
- 再通过 `expert_map` 把非本卡 expert 标成 `-1`，让 matmul 跳过。

这是读懂几乎所有 fused MoE Triton/CUTLASS 路径的第一块砖。

### 3.2 Permute / Unpermute

文件：`moe_permute_unpermute.py` 及 experts 内联实现

```text
Permute:   [T, H] + routing  → 按 expert 聚集的连续块（+ pad）
Unpermute: expert 输出块    → 按原 token 顺序写回，并对 K 路加权求和
```

BatchedExperts 格式下，permute 的结果更接近 `[E_local, capacity, H]`。

### 3.3 Fused Expert GEMM 族

| 后端 | 典型入口 | 备注 |
|------|----------|------|
| Triton fused MoE | `experts/triton_moe.py` 等 | 灵活，易读 |
| DeepGEMM | `experts/deep_gemm_moe.py`, `batched_deep_gemm_moe.py` | DeepSeek 系高性能；细讲见 [02a_deepgemm.md](/notes/2026/07/24/2026-07-24-ep-learning-02a-deepgemm/) |
| CUTLASS / FlashInfer | `cutlass_moe`, `flashinfer_*` | 量化路径多 |
| Marlin / MXFP4 / NVFP4 | 各 `*_moe.py` | 权重量化 |
| MegaMoE | DeepSeek V4 专用 | 见第 04 章 |

共同模式：**一次 kernel 或紧耦合 kernel 链**完成 `gate_up` + `act_mul` + `down`，减少全局显存往返。

### 3.4 TopKWeightAndReduce

抽象：专家输出何时乘 `topk_weights`、何时做 K 路求和。

- 有的 Experts 内核内部已做完 → Finalize 用 `TopKWeightAndReduceDelegate`（空操作）  
- 有的只出「未加权」结果 → Finalize 用 `TopKWeightAndReduceContiguous` 等

读某条路径时，**先查谁负责 apply weights**，避免重复乘或漏乘。

### 3.5 Router kernels

`fused_moe/router/`：

- `fused_topk_router` / `grouped_topk_router`  
- DeepSeek V4：`router/dsv4_topk.py`  
- 可能含 bias、sigmoid、softmax、sqrtsoftplus 等 scoring

Router 与 EP 解耦，但 **topk_ids 的 dtype/坐标系** 会被 DeepEP / MegaMoE 约束（如 DeepEP 常要 `int64`）。

---

## 4. Prepare/Finalize 后端地图（通信侧）

| 模块 | 场景 |
|------|------|
| `no_dp_ep` | 无跨 DP EP，基本本地 |
| `naive_dp_ep` | AG + RS 或朴素广播 |
| `deepep_ht` | DeepEP 高吞吐（偏 prefill） |
| `deepep_ll` | DeepEP 低延迟（偏 decode）；Batched + 固定 hidden |
| `deepep_v2` | 新一代 DeepEP 适配 |
| `flashinfer_nvlink_*` | 机内 NVLink 优化 A2A |
| `batched` | 与 batched experts 布局配合 |
| `nixl_ep` / `mori` | 其它传输栈实验/集成 |

选择入口通常在：

- 环境变量 `VLLM_ALL2ALL_BACKEND`  
- `FusedMoE` / oracle 根据量化与并行配置挑 Experts + PrepareFinalize 组合  

详见跨机 EP 笔记中的后端表。

---

## 5. 读 Kernel 时的统一检查表

对任意 fused MoE 路径，按顺序回答：

1. **Token 布局**：Standard 还是 Batched？谁负责转换？  
2. **Expert 坐标系**：global / local / physical？`expert_map` 在哪用？  
3. **量化**：activation 在 dispatch 前还是后量化？scale 如何随 token 走？  
4. **Padding / capacity**：`max_tokens_per_rank`、block align、无效 topk（-1）如何处理？  
5. **权重归约**：在 Experts 内还是 Finalize？  
6. **同步点**：CUDA event / DeepEP handle / DBO yield 在哪？

---

## 6. 建议的「第一周」精读顺序

1. `modular_kernel.py`：类注释 + `FusedMoEModularKernel.forward`  
2. `moe_align_block_size.py`：文档字符串里的例子手推一遍  
3. `prepare_finalize/no_dp_ep.py`：无通信基线  
4. `prepare_finalize/naive_dp_ep.py`：看 AG/RS 如何塞进 Prepare/Finalize  
5. 任选一个 `experts/triton_moe.py` 或 `deep_gemm_moe.py`：对照 align 输出  
6. 再进 `deepep_ht.py`（下一章）

---

## 自检

- [ ] 能说明为何 Prepare 与 Finalize 必须成对  
- [ ] 能手推 `moe_align_block_size` 小例子（文档里的 4 token / 4 expert）  
- [ ] 知道 Standard vs BatchedExperts 各自适配哪类通信  
- [ ] 能指出本仓库里 DeepEP / naive / no_dp 适配文件位置

---

## 附录：和 SGLang 的对照

SGLang 把通信叫 **token_dispatcher**（`sglang/.../token_dispatcher/deepep.py`），计算在 `ep_moe` / `moe_runner`。  
概念一一对应：dispatcher ≈ PrepareAndFinalize，runner ≈ Experts。对比阅读有助于分清「框架粘合」与「DeepEP 本体」。
