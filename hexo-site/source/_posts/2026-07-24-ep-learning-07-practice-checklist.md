---
title: 07 · 实践清单与常见坑
date: 2026-07-24
tags: [EP, MoE, 学习笔记]
---

# 07 · 实践清单与常见坑

> 精读地图：[06_code_reading_map.md](/2026/07/24/ep-learning-06-code-reading-map/)  
> 目录：[README.md](/2026/07/24/ep-learning/)

把实验命令、现象、结论直接记在本文「实验日志」区。

---

## 1. 环境准备清单

- [ ] 确认当前用的是 `vllm_comm` 还是 `vllm`，Python 里 `import vllm; print(vllm.__file__)`  
- [ ] `python -c "import deep_ep; print(deep_ep.__file__)"`（DeepEP 路径）  
- [ ] DeepGEMM / MegaMoE：`import deep_gemm` 是否可用；GPU 是否 SM100  
- [ ] NCCL / RDMA：多机前先 `ibstat` / 集群文档确认  
- [ ] 相关 env 列表备份一份（避免和别的实验互相污染）

---

## 2. 最小实验阶梯

### L0 · 单测（正确性）

```bash
# 在 vllm_comm 可运行环境中，按你们 CI/README 调整路径
pytest tests/kernels/moe/test_deepep_moe.py -q
pytest tests/kernels/moe/test_deepep_deepgemm_moe.py -q
# 有 V4 / SM100 时再开：
# pytest tests/models/test_deepseek_v4_mega_moe.py -q
```

记录：通过/失败、GPU 型号、deep_ep 版本。

### L1 · 单机多卡 EP（功能）

- 小 MoE（如 Qwen3-MoE 类）`TP×DP` 组合使 `EP>1`  
- 对比 backend：`allgather_reducescatter` vs `deepep_high_throughput`  
- 看日志是否创建 DeepEP Buffer、有无 hidden roundup 警告  

### L2 · Prefill vs Decode 后端

- Prefill 偏 HT，Decode 偏 LL（按框架实际配置）  
- 记录：吞吐、TPOT、DeepEP 显存占用  

### L3 · 跨机 EP

- 对照 `vllm_cross_node_expert_parallelism.md`  
- 重点观察 LL OOM 与 `max_num_batched_tokens` 关系  
- 打开通信量统计（若层内有 `fused_communication_ops`）  

### L4 · EPLB

- `enable_eplb`，看 balancedness 日志  
- 对比开关前后热点 GPU 利用率  
- 若异步：确认无偶发错结果（map 切换竞态）  

### L5 · MegaMoE（条件具备时）

- `moe_backend=deep_gemm_mega_moe` + EP  
- 对照 yaml：`tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`  
- nsys：通信与计算是否糊成大块 kernel  

---

## 3. 调试检查表（挂了先过一遍）

| 症状 | 优先怀疑 |
|------|----------|
| 集体通信挂死 / NCCL timeout | 某 DP rank 没进同一 MoE（缺 dummy batch）；EP group 不一致 |
| DeepEP LL OOM | `max_num_batched_tokens` / per-rank dispatch 上限过大 |
| 数值对不齐 | topk 权重乘了两次或漏乘；expert_map -1 未跳过；dtype int32/int64 |
| Hidden 相关断言 | HT 对齐 / LL 不在 `SUPPORTED_HIDDEN_SIZES` |
| MegaMoE 直接 NotImplemented | 非 SM100；H/I 非 128 对齐；未开 EP；scoring/dtype 不符 |
| EPLB 开了更慢 | 跨机搬权重过频；map 抖动；异步同步错误 |
| DBO 下偶发错 | DeepEP handle 槽位串 ubatch；ROCm HT workspace 未 sync |

---

## 4. 建议自己实现的「纸上练习」

不写代码也能练：

1. 手推 `moe_align_block_size` 文档例子，写出 `sorted_token_ids`。  
2. 给定 `EP=4, E=64, K=6, T=128`，估算真 all-to-all 相对 AG 的流量上界（量级即可）。  
3. 画 HT dispatch→expert→combine 的 stream 时间线（含可选 overlap）。  
4. 说明为何 `EP=TP×DP` 时「只加节点不加 DP」不一定减小 LL buffer。  

---

## 5. 实验日志（模板）

### 实验 #____ · 日期 ____

- 仓库 / commit：  
- 模型 / 并行配置：  
- Backend / env：  
- 命令：  

```bash
# paste command
```

- 现象：  
- 结论：  
- 链到笔记章节：  

---

## 6. 与其它笔记的交叉引用

| 需求 | 文档 |
|------|------|
| 跨机拓扑与 OOM | `../vllm_cross_node_expert_parallelism.md` |
| RDMA 对象（QP/MR） | `../rdma_learning_1.md` |
| PD 分离调度（若和 EP 同开） | `../vllm_pd_disaggregation_*.md` |
| nsys 工作流 | Cursor skill `dashllm-vllm-nsys` |

---

## 完成定义（学完本系列）

当你能够：

1. 不看笔记画出 Modular MoE 四段流水线，并指出 DeepEP / MegaMoE 各接在哪；  
2. 在 `vllm_comm` 里 5 分钟内定位 HT/LL 适配与 Buffer 创建代码；  
3. 独立解释一次 EPLB rebalance 的 map 与权重搬运；  
4. 对真实挂死/OOM 按第 3 节表给出可验证假设；  

即可认为 EP 主线达标；之后按需下沉 DeepEP/DeepGEMM 上游 CUDA。
