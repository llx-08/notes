---
title: 04 · MegaMoE（DeepSeek V4 + DeepGEMM）
date: 2026-07-24
tags: [EP, MoE, 学习笔记]
---

# 04 · MegaMoE（DeepSeek V4 + DeepGEMM）

> 上一章：[03_deepep.md](/2026/07/24/ep-learning-03-deepep/)  
> 下一章：[05_eplb_and_load_balance.md](/2026/07/24/ep-learning-05-eplb-and-load-balance/)

---

## 1. MegaMoE 在栈里的位置

```text
Router (dsv4 / sqrtsoftplus …)
        │
        ▼
prepare_megamoe (Triton)
  - hidden → fp8 + E8M0 group scales
  - topk_ids / weights 打成 DeepGEMM 约定布局
        │
        ▼
deep_gemm.fp8_fp4_mega_moe(...)
  - 融合 EP 相关对称缓冲 + FP8×FP4 expert 计算
        │
        ▼
输出 hidden（常已 reduce）
```

**和 DeepEP 的关系**：

- DeepEP：通用 MoE **通信**库（dispatch/combine）。  
- MegaMoE：DeepSeek V4 的 **计算（+ 与 EP 紧耦合的缓冲）** 路径，后端名 `deep_gemm_mega_moe`。  
- 二者可同属 DeepSeek 生态，但解决的问题不同；读代码时不要混成「一个东西」。

![MegaMoE 路径：Router → prepare_megamoe → fp8_fp4_mega_moe](/imgs/ep_megamoe_path.png)

*图：V4 特化一体路径；相对 Modular「DeepEP + DeepGEMM Experts」更融合。*

本地入口（`vllm_comm`）：

```text
models/deepseek_v4/nvidia/model.py          # DeepseekV4MegaMoEExperts 等
models/deepseek_v4/nvidia/ops/prepare_megamoe.py
tests/models/test_deepseek_v4_mega_moe.py
tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml
config/kernel.py  →  moe_backend == "deep_gemm_mega_moe"
```

---

## 2. 开启条件与硬约束

从 `nvidia/model.py` / XPU 对照实现可归纳：

| 约束 | 说明 |
|------|------|
| `moe_backend == "deep_gemm_mega_moe"` | kernel config 显式选择 |
| **必须 EP** | `enable_expert_parallel`；否则直接报错 |
| Scoring | 当前要求 `sqrtsoftplus` |
| Expert dtype | **fp4** experts |
| GPU | DeepGEMM MegaMoE 要求 **SM100**（`device_capability[0] == 10`） |
| Shape | `hidden_size`、`intermediate_size` 均为 **128 的倍数** |

这些约束说明 MegaMoE 是 **为特定 checkpoint + 硬件特化的高速路径**，不是通用 fallback。

---

## 3. 权重变换：`finalize_weights`

加载后并不直接拿 HF 布局去算，而是：

1. Scale 从 UE8M0/uint8 语义转成 float，再 `transform_sf_into_required_layout`  
2. `deep_gemm.transform_weights_for_mega_moe(w13, w2)` → `_transformed_l1/l2_weights`  
3. 丢掉原始 `Parameter`（注释说明 L1 会新分配；L2 可能 alias 原 storage）

学习点：

- **布局变换是一等公民**：和通信库一样，计算库也有「原子友好」的 weight/scale packing。  
- EPLB 搬权重时必须拿 **变换后** 的 contiguous view（`get_expert_weights`），否则映射会错。

---

## 4. `prepare_megamoe` Triton kernel

文件：`ops/prepare_megamoe.py`

核心工作（读 kernel 时按这个清单对）：

1. **按 group（如 GROUP_K）** 对 hidden 求 amax → 得到近似 E8M0 的 scale  
2. `hidden / scale` → `float8e4nv` 存入 `x_fp8`  
3. 把 scale exponent **打包**进 `x_sf`（按 block 压缩）  
4. 把 `topk_ids` 写成 **int64**，padding token 置 `-1`  
5. `topk_weights` 写成 float；padding 置 0  

这是「框架侧 staging」：把 PyTorch 友好张量变成 **MegaMoE / DeepGEMM 契约布局**，类似 DeepEP 对 hidden 对齐的要求。

---

## 5. 运行时：`fp8_fp4_mega_moe` + 对称缓冲

`DeepseekV4MegaMoEExperts` 路径概念上：

```text
_stage_inputs (prepare_megamoe)
get_symm_buffer_for_mega_moe(ep_group, num_experts, max_tokens, top_k, H, I)
fp8_fp4_mega_moe(staged_inputs, transformed_weights, symm_buffer, ...)
```

`symm_buffer`：

- 按 `(process_group, device, E, max_tokens, top_k, H, I)` 缓存  
- 与 EP group 绑定 → **计算 kernel 内可能直接使用跨 rank 对称内存**完成部分通信或归约  

这与「DeepEP Buffer + 独立 Experts」的拆分模型不同：**MegaMoE 把更多东西收进 DeepGEMM 一体路径**。读性能分析时，NVTX/nsys 里通信与计算边界可能不如 modular DeepEP 清晰。

---

## 6. 与 Modular FusedMoE 的对比

| | 通用 Modular（DeepEP + DeepGEMM MoE） | MegaMoE |
|--|--------------------------------------|---------|
| 通信 | PrepareAndFinalize 显式 dispatch/combine | 更多在 DeepGEMM / symm buffer 内 |
| 计算 | `FusedMoEExpertsModular` 可插拔 | 固定 `fp8_fp4_mega_moe` |
| 模型 | 多模型共用 | DeepSeek V4 特化 |
| 硬件 | 视具体 experts | SM100 + 尺寸约束 |
| EPLB | 通用 eplb_state | 专用 `set_eplb_state` / weight view；且 `eplb_utils.override_envs_for_eplb` 对 mega_moe 有特殊 env 覆盖 |

学 EP「通识」时以 Modular + DeepEP 为主；学 **V4 极致路径** 再专攻本章。

---

## 7. 推荐阅读顺序

1. `config/kernel.py`：`deep_gemm_mega_moe` 枚举与注释  
2. `nvidia/model.py`：`use_mega_moe` 分支、`_init_mega_moe_experts`、约束 assert  
3. `ops/prepare_megamoe.py`：手推一个 token 的 quant + topk pack  
4. `finalize_weights` / `get_symm_buffer` / `_run_mega_moe`  
5. `distributed/eplb/eplb_utils.py`：`override_envs_for_eplb` 里对 mega_moe 的分支  
6. 单测 `test_deepseek_v4_mega_moe.py` + eval yaml  

若本机有 DeepGEMM 源码，再搜 `fp8_fp4_mega_moe`、`transform_weights_for_mega_moe`、`get_symm_buffer_for_mega_moe`。

---

## 自检

- [ ] 能区分 DeepEP、DeepGEMM 普通 MoE、MegaMoE 三者职责  
- [ ] 能说出开启 MegaMoE 的硬件与配置硬约束  
- [ ] 知道 `prepare_megamoe` 输出哪些张量、padding 如何标记  
- [ ] 理解为何 EPLB 必须基于 transformed weights
