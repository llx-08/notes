---
title: 05 · EPLB 与 Expert 负载均衡
date: 2026-07-24
tags: [EP, MoE, 学习笔记]
---

# 05 · EPLB 与 Expert 负载均衡

> 上一章：[04_megamoe.md](/2026/07/24/ep-learning-04-megamoe/)  
> 下一章：[06_code_reading_map.md](/2026/07/24/ep-learning-06-code-reading-map/)  
> 算法细节展开：[../vllm_cross_node_expert_parallelism.md](/2026/06/17/vllm-cross-node-expert-parallelism/) §五

---

## 1. 为什么 EP 需要 Load Balance

MoE 路由是 **数据依赖** 的：真实流量下少数 expert 会成为热点。

在 EP 下：

- 热点 expert 所在 GPU：**算力打满、排队**  
- 其它 GPU：**空闲**  
- 跨机时：热点若分散在多节点，**跨机 token 流量** 暴涨  

EPLB（Expert Parallelism Load Balancing）要做的事：

1. **统计** 各 logical expert 的负载（token 计数等）  
2. **决策** 逻辑→物理映射：复制热点、打包到节点/GPU  
3. **执行** 在 EP ranks 间搬运 expert 权重，更新 map  
4. （可选）与推理 **异步重叠**

---

## 2. Logical vs Physical Expert

```text
Logical experts  (模型定义的 E 个)
        │  EPLB 映射
        ▼
Physical slots   (≥ E，可含副本)
        │  再均分到 EP ranks
        ▼
Local weights on each GPU
```

路由仍产生 **logical id**；运行时查表得到 physical id / 本卡 local id。  
DeepEP LL 适配层可接收 `global_to_physical` / `physical_to_global` 一类张量——与 EPLB 输出对齐。

---

## 3. 分层 rebalance（思想）

跨机笔记中的三步（感知节点拓扑）：

```text
Step 1  Expert group → 打包到 Node
        目标：同 group 尽量同机，减少跨机 dispatch

Step 2  Node 内：给热点 logical expert 分配冗余 physical 副本

Step 3  Node 内：physical expert → 打包到各 GPU
        目标：节点内 NVLink 上再均衡
```

若 `num_groups` 不能整除 `num_nodes`，常回退为「当成单节点」的扁平均衡。

分层三步示意：

![EPLB 分层：Node 打包 → 复制热点 → GPU 打包](/imgs/ep_eplb_hierarchical.png)

*图：先减跨机流量，再在节点内复制热点并均分到 GPU，最后 P2P 搬权重。*

贪心工具：

- `balanced_packing`：按负载降序，每次放进当前最轻且有空位的桶  
- `replicate_experts`：冗余槽位反复给「每副本负载最大」的 logical expert  

---

## 4. vLLM 代码地图

```text
vllm_comm/vllm/distributed/eplb/
├── eplb_state.py          # 每层状态、prepare_forward / step
├── eplb_utils.py          # CpuGpuEvent、env override（含 mega_moe）
├── eplb_communicator.py   # 均衡相关通信
├── policy/                # rebalance 算法（含 hierarchical）
├── rebalance_execute.py   # P2P isend/irecv 搬权重
└── async_worker.py        # 异步搬权重线程 + CUDA stream
```

与 runner 的衔接：

- `gpu_model_runner.py`：`setup_eplb` / `eplb_step` / dummy run 时是否 skip  
- MoE 层：`set_eplb_state`、`get_expert_weights`  
- MegaMoE：transformed weight 的 EPLB view（见第 04 章）

### 异步同步要点

`CpuGpuEvent`：CUDA Event  alone 在「尚未 record」时 wait 可能变成空操作；因此用 **threading.Event + CUDA Event** 保证「主线程 record 之后，异步线程再 wait」。  
这是读 `async_worker` 时最容易踩的概念坑。

---

## 5. 与通信后端的耦合

| 后端 | EPLB 关注点 |
|------|-------------|
| NCCL AG+RS | 算力不均仍痛；跨机流量模式不同 |
| DeepEP HT/LL | map 更新后 dispatch 目标 rank 变；LL 还受 capacity 影响 |
| MegaMoE | symm buffer / cooperative 路径；`override_envs_for_eplb` 可能强制通信相关 env |

开启 EPLB 后应用 **先看 balancedness 日志 / metrics**，再看端到端吞吐，避免「map 在动但瓶颈在别处」。

---

## 6. 学习时建议做的两张表

**表 A：一次 rebalance 的输入输出**

| 字段 | 来源 | 去向 |
|------|------|------|
| `expert_load_view` | forward 统计 | policy |
| `logical_to_physical_map` | policy | router 后处理 / dispatcher |
| `logical_replica_count` | policy | 调试与加权 |
| `expert_weights` 列表 | MoE.get_expert_weights | rebalance_execute |

**表 B：一次权重搬运**

| 步骤 | 同步点 |
|------|--------|
| 算新 map | CPU / 控制面 |
| isend/irecv 权重 | EP group，可能私有 stream |
| 切换 map 对外可见 | 与 forward 的 barrier / event |
| 旧 map 退役 | 确保无 in-flight 层仍用旧映射 |

---

## 自检

- [ ] 能解释 logical/physical 与「复制热点」如何降低尾部延迟  
- [ ] 能说出分层三步各自优化的带宽层级（跨机 / 机内）  
- [ ] 知道权重搬运与推理 overlap 时为何需要 CpuGpuEvent  
- [ ] 能指出 MegaMoE 路径上 EPLB 的特殊点
