---
title: 01 · MoE 与 Expert Parallelism 基础
date: 2026-07-24
tags: [EP, MoE, 学习笔记]
---

# 01 · MoE 与 Expert Parallelism 基础

> 回到目录：[README.md](/notes/2026/07/24/2026-07-24-ep-learning/)  
> 下一章：[01a_moe_all_to_all.md](/notes/2026/07/24/2026-07-24-ep-learning-01a-moe-all-to-all/)（All-to-All 细讲 + 图） → [02_modular_kernel_and_moe_kernels.md](/notes/2026/07/24/2026-07-24-ep-learning-02-modular-kernel-and-moe-kernels/)

---

## 1. MoE 层在算什么

稠密 Transformer 的 FFN 对每个 token 做同一套线性变换。MoE 把它换成「多专家 + 门控」：

```text
h = LayerNorm(x)
gate_logits = Gate(h)                    # [T, E]
topk_ids, topk_weights = TopK(gate)      # [T, K], [T, K]
y_e = Expert_e(h)                        # 仅对路由到 e 的 token
y = Σ_k topk_weights[...,k] * y_{topk_ids[...,k]}
```

要点：

- **稀疏激活**：每 token 只算 K 个 expert（K≪E），算力随 K 增长而非 E。
- **参数随 E 增长**：容量大，但单 token 算力可控。
- **通信新瓶颈**：一旦 expert 分到多卡，就必须 **dispatch / combine**。

常见变体：


| 变体                       | 说明                                  |
| ------------------------ | ----------------------------------- |
| Shared Expert            | 所有 token 额外过一个共享 FFN，再与 routed 输出相加 |
| Grouped TopK             | DeepSeek 风格：先选 group 再组内选 expert    |
| Hash / simulated routing | 调试或负载研究用                            |


---

## 2. 并行轴：TP / DP / EP 的关系

### 2.1 直觉


| 轴   | 切什么             | MoE 相关影响                                      |
| --- | --------------- | --------------------------------------------- |
| TP  | 单层权重矩阵按列/行切     | 同一 expert 可被 TP 切开；通信多为 all-reduce            |
| DP  | 数据切分，权重复制       | 各 DP 持有完整模型副本；需梯度/激活同步                        |
| EP  | **按 expert 切分** | 每卡只持有 `E / EP` 个完整 expert；需要 token all-to-all |


### 2.2 vLLM 中的关键事实：`EP = TP × DP`

在 vLLM 里，EP **不是**独立于 TP/DP 的第三条进程轴，而是：

```text
EP world size = tensor_parallel_size × data_parallel_size
```

含义：

- 开启 DP 后，不同 DP rank 上的「同位置」TP rank 组成更大的 EP group。
- MoE 的 dispatch/combine 在 **EP group** 上做集体通信。
- 因此 **DP>1 的 MoE serving 几乎必然涉及 EP 语义**（哪怕你口头只说「开了 DP」）。

跨机时：EP group 成员分布在多节点 → all-to-all 走 IB/RoCE，见已有专题笔记。

下图用「同色 = 同一 DP home」说明：Attn 后同色卡持有同一份完整 hidden；expert 则可能摊在所有颜色的 GPU 上。

![EP=TP×DP 与 token home（同色为同一 DP 数据归属）](/imgs/ep_tp_dp_home.png)

*图：蓝色 / 黄色各是一个 DP home；TP 是同色内切权重，EP 可跨色分 expert。*

### 2.3 进程与 DP Lockstep

- 每个 DP rank 通常对应独立 EngineCore；内部再有 TP workers。
- EP 上的 collective **要求所有参与 rank 同步进入同一 MoE 层**。
- 空闲 DP rank 也必须跑 `dummy_batch`，否则其他 rank 在 all-to-all 上死锁。

**Lockstep（齐步行进）是什么意思？**

不是「所有 DP 的 scheduler 队列内容必须一模一样」，而是：

> 只要 **全局还有活**（某个 DP 上还有未完成请求、wave 还在跑），**每一个 DP rank 每一拍都要进入同一次 model step**（有真 batch 就 `execute_model`，没可跑的就 `dummy_batch`），从而一起穿过 MoE 的 all-to-all / all-reduce。

![DP Lockstep：有活的跑真 batch，空闲的跑 dummy，一起进 MoE](/imgs/ep_dp_lockstep.png)

*图：DP1 本步无可调度仍必须 `dummy_batch`，否则 A2A 死锁。同步标准是全局 OR(unfinished)。*

```text
DP0: 有真请求 → forward（含 MoE A2A）──┐
DP1: 本步无可跑 → dummy_batch（也进 A2A）─┼─ 同一拍 collective，不挂死
DP2: 有真请求 → forward（含 MoE A2A）──┘
```

若允许「我没请求就整步跳过、不进 MoE」→ 别的 DP 在 A2A 上等你 → **死锁 / NCCL timeout**。

引擎侧还会用 DP 间 all-reduce 同步「全局是否还有 unfinished」，维护 `engines_running`；全员都空了才停 wave，避免有人提前躺平。

### 2.3.1 `has_unfinished_requests` 指什么？

**指调度器里是否还有「生命周期未结束」的请求，不是「此刻 GPU 正在 forward 的请求」。**

`Scheduler.get_num_unfinished_requests()` 大致是：

```text
len(waiting) + len(skipped_waiting) + len(running)
  − streaming 等待输入的特殊计数
```

| 队列 | 含义 |
|------|------|
| `waiting` | 已进引擎、尚未（或暂时不能）进入 running 的请求 |
| `running` | 正在多步生成中的请求（prefill/decode 跨很多 step） |
| 从上述队列移除 | 通常已 EOS / 达 max tokens / abort 等，生命周期结束 |

因此：

- **≈「未完成请求（尚未结束生成）」**，包含还在排队、本步没被 schedule 到的。  
- **≠「本步正在执行 forward 的请求」**。本步有没有真的跑 GPU，看的是 `_process_engine_step()` 是否 `executed`，不是这个 flag。

和 dummy 的关系（`DPEngineCore.run_busy_loop` 思路）：

```text
executed = 本步是否跑了真实 batch
local_unfinished = has_unfinished_requests()

if not executed:
  if 本地也没 unfinished 且 engines_running==False:
      # 全局都闲，可以空转等新请求
  else:
      # 本步没真活，但 wave 还在 / 本地或别人还有未完成
      → execute_dummy_batch()   # 对齐 lockstep
```

口播：

> `has_unfinished` = 调度账本里还有活单；`executed` = 这一拍有没有派出真车。  
> Lockstep = 只要全局账本或 wave 说「还在干活」，没真车的 rank 也要派 dummy 车去 MoE 路口会合。

### 2.3.2 为什么「本地 unfinished」能当跨 DP 同步标准？

关键点：同步的不是「大家队列里是同一批请求」，而是 **「全集群还有没有活」**。

- 每个 DP rank 有**自己的** scheduler、**自己的**请求子集（数据并行）。  
- `has_unfinished_requests()` 只回答：**我这边账本空了没有**。  
- 跨 DP 时做集体运算（如 all-reduce OR/MAX）：

```text
global_unfinished = OR( DP0.unfinished, DP1.unfinished, … )
engines_running   ≈ global_unfinished   # 思想：任一 rank 还有活 → 全体继续齐步
```

因此：

| 问题 | 答案 |
|------|------|
| 各 rank 的 unfinished 指同一批请求吗？ | **否**，各管各的请求 |
| 那同步的是什么？ | **「是否还有任一 DP 仍有未完成请求」** |
| 为何够用？ | 决定 **wave 是否继续 / 空闲 rank 是否还要 dummy**：有人有活 → 全体不能停；全员空 → 一起停 |

和 lockstep 的两层关系：

```text
① 步内齐步（防 A2A 死锁）
   engines_running==True 且本步没真 batch → dummy_batch
   （别人可能还在真 forward，你必须到 MoE 路口）

② 波次停表（防有人先躺平）
   周期性同步 global_unfinished
   全 False → engines_running=False，一起结束 wave
```

所以：拿 `has_unfinished_requests` 做标准，不是因为「它等于正在 forward」，而是因为它是各 rank 上 **「我还有没有要继续多步生成的活」** 的本地布尔，OR 起来就是全局是否该继续齐步行进。

### 2.4 `dummy_batch`：有没有 token？怎么「分辨」？

**会有 token，而且会进 all-to-all。** 空闲 rank 不是发一个「空 collective」，而是构造一小批**人造 batch**，跑完整（或等价）的 `model.forward`，从而和有活的 rank 一样穿过每一层 MoE 的 dispatch/combine。

vLLM 路径（`vllm_comm`）：

```text
EngineCore（本 DP 本步没跑成真实请求）
  → execute_dummy_batch()
  → Worker.execute_dummy_batch()
  → model_runner._dummy_run(num_tokens≈uniform_decode_query_len, uniform_decode=True)
  → 伪造 input_ids / positions / attn meta …
  → self.model(...)   # 含 MoE → 真的进 A2A
```

| 问题 | 答案 |
|------|------|
| dummy 有没有 token？ | **有**。idle lockstep 常见是 **至少 1 个**（decode 长度），不是 `T=0`。 |
| 这些 token 参不参与 A2A？ | **参与**。MoE 通信层一般**没有**「这是 dummy 就跳过 dispatch」的分支；人造 hidden 一样走 router → dispatch → experts → combine。 |
| 和真实请求怎么分辨？ | **主要在引擎侧分辨，不在 NCCL/DeepEP 里贴标签。** |

分辨落在哪一层：

```text
┌─────────────────────────────────────────────────────────┐
│ Engine / Scheduler（能分辨）                              │
│  - 本 DP 有无 unfinished / 本步是否 execute 成功           │
│  - 无活 → execute_dummy_batch；有活 → 正常 execute_model   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Model / MoE / All-to-All（基本不分辨）                     │
│  - 只看见 [T, H] + topk；dummy 与真实在通信语义上同类      │
│  - 结果不写回用户输出；KV slot_mapping 常填 -1 跳过写 cache │
│  - EPLB 统计常 skip / is_dummy，避免人造负载污染均衡       │
└─────────────────────────────────────────────────────────┘
```

要点：

1. **为何不能 0 token？** `T=0` 容易走不同代码路径或根本不进 MoE，其它 rank 仍在 A2A 上等待 → 死锁。人造少量 token 是为了 **对齐 collective 参与**，不是为了产出答案。  
2. **其它 rank 会「吃到」dummy token 吗？** 会。真 A2A / AG 都会把空闲 rank 造出来的 token（或其 gather 视图）算进本步通信；代价通常很小（T≈1），换来不挂死。  
3. **「分辨」不等于「通信可跳过」**：能跳过的是采样、写 KV、给用户返回、EPLB 计数；**不能**跳过的是与 EP group 绑定的 MoE collective。

代码锚点：`v1/engine/core.py`（空闲时调 `execute_dummy_batch`）、`gpu_worker.py:execute_dummy_batch`、`gpu_model_runner.py:_dummy_run`（`slot_mappings.fill_(-1)`、EPLB `is_dummy` 注释）。

---

## 3. Dispatch / Combine：两种抽象

### 3.1 「假 EP」：AllGather + ReduceScatter

```text
Dispatch ≈ all_gather(tokens)     # 每卡拿到全局 token
Local experts 对「自己负责的 expert」计算（或对全量 token 算本地 expert）
Combine ≈ reduce_scatter(outputs) # 按原 DP 切片归约回去
```

- 实现简单，NCCL 成熟。
- 带宽与算力都有 **O(EP)** 冗余：不该来的 token 也到了本卡。

### 3.2 「真 EP」：Token All-to-All

```text
Dispatch: 只把 token 发给持有其 top-k expert 的 ranks
Local experts: 只算本地 expert
Combine: 把结果送回 token 的 home，并按 topk weight 归约
```

- 通信量与路由稀疏度相关，跨机时更划算。
- 需要更复杂的元数据：每 expert token 数、排序、padding、handle。
- DeepEP / PPLX / FlashInfer A2A 都属于这一类。

> **细讲（原因 / 传什么 / 流程图）** → [01a_moe_all_to_all.md](/notes/2026/07/24/2026-07-24-ep-learning-01a-moe-all-to-all/)

### 3.3 「token 原 rank / home」指什么（易混点）

**不要和 Attention 的 TP 搞混。** Attn 里每个 TP rank 都在算——但那是「同一批 token、切权重」；MoE dispatch 是「token 按 expert 搬家」。


| 阶段        | 切什么              | 各 rank 上的 token                                           |
| --------- | ---------------- | --------------------------------------------------------- |
| Attn / 稠密 | **TP 切权重**       | 同一 **DP** 内，各 TP rank 处理**同一批** token（算不同分片，再 all-reduce） |
| MoE EP    | **按 expert 切权重** | token 要按 top-k **寄到**持有对应 expert 的 EP rank                |


口播版：

> **home（原 rank）= MoE dispatch 之前持有该 token hidden 的数据归属（主要是 DP home）；不是「Attn 只跑在某一个 TP 上」。**

再拆开看：

1. **DP>1**：调度时序列/batch 落在哪个 DP rank，MoE 前 hidden 就在那一套进程里。
  Dispatch 可能发到别的 DP/TP（只要那边有对应 expert）；  
   Combine 必须**送回这个 DP home**，下一层 Attn 才能继续。
2. **同一 DP、TP>1**：Attn 后 all-reduce，**该 DP 下每个 TP rank 上都有这份完整 hidden**。
  此时 home 是「这一整组 TP ranks（同一数据分片）」，不是「只有 TP0 有 token」。  
   Dispatch 在 EP group（`TP×DP`）里按 expert 重排；Combine 后再恢复成进入 MoE 前、本 DP 各 TP 上应有的布局。

和两种后端的对应关系：

```text
真 all-to-all:  token 离开 home → 专家 rank 计算 → 送回 home
AG + RS:       先 gather 到大家都能算 → reduce-scatter 回各自原来的 DP token 切片
```

两种写法里的「回原处」，指的都是上面的 **home / DP 数据归属**，不是某个唯一做 Attn 的 TP rank。

---

## 4. 数据布局：Standard vs BatchedExperts

Modular kernel 里两种 activation format：


| Format             | Shape 直觉                                  | 谁喜欢                           |
| ------------------ | ----------------------------------------- | ----------------------------- |
| **Standard**       | `[num_tokens, hidden]`，token 连续           | DeepEP HT、多数 Triton fused MoE |
| **BatchedExperts** | `[num_local_experts, max_tokens, hidden]` | DeepEP LL、部分 batched DeepGEMM |


从 Standard → Batched 通常伴随 **permute / pack**；反过来是 **unpermute / scatter**。  
`moe_align_block_size` 解决的是：按 expert 排序后，token 数要对齐到 GEMM block（如 64/128），不足则 pad。

---

## 5. Expert Map：全局 ID ↔ 本地 ID

开启 EP 后，模型配置里的 expert id 是 **全局** 的，本卡只存一部分权重：

```text
expert_map[global_id] = local_id   # 本卡持有
expert_map[global_id] = -1         # 本卡不持有 → GEMM 应跳过
```

EPLB 打开后还有：

- **logical expert**：训练/配置定义的 E 个
- **physical expert**：可复制热点后的物理槽位（≥E）
- `logical ↔ physical` 映射表随 rebalance 更新

读代码时先分清「路由输出的 id 是哪一套坐标系」。

---

## 6. Prefill vs Decode 对 EP 的不同压力


| 阶段      | 特征              | 通信偏好                               |
| ------- | --------------- | ---------------------------------- |
| Prefill | token 多、延迟可摊销   | **高吞吐** all-to-all（DeepEP HT）      |
| Decode  | 每步 token 少、延迟敏感 | **低延迟** 路径（DeepEP LL、小 buffer、少同步） |


同一模型常在不同阶段切不同 `VLLM_ALL2ALL_BACKEND`，或依赖框架自动选择。

---

## 7. 和「普通 TP MoE」的边界

- **纯 TP、DP=1**：expert 权重按 TP 切分，token 不跨 DP；通信主要是 TP all-reduce，不一定走 DeepEP。
- **DP>1 或显式 enable_expert_parallel**：必须处理跨 rank 的 token 路由 → 进入本系列后半部分。

MegaMoE（DeepSeek V4）当前在 vLLM 路径上 **要求开启 EP**，并与特定 routing / dtype 绑定——见第 04 章。

---

## 自检

- [ ] 能画出 Router → Dispatch → Experts → Combine 四步，并说明每步输入输出 shape 量级
- [ ] 能解释为何 vLLM 里 `EP = TP × DP`，以及 dummy batch 的必要性
- [ ] 知道跨 DP 同步的是 `OR(各 rank 本地 unfinished)`，不是同一批请求
- [ ] 能对比 AG+RS 与真 all-to-all 的带宽冗余
- [ ] 知道 Standard / BatchedExperts、global/local expert id 的区别
- [ ] 能说明「token home」是 DP 数据归属，且与「Attn 每个 TP 都参与」不矛盾

---

## 建议精读（本章）

1. `vllm_comm/vllm/distributed/parallel_state.py` — EP group 构造
2. `vllm_comm/vllm/model_executor/layers/fused_moe/layer.py` — `FusedMoE.forward` 总控
3. `../vllm_cross_node_expert_parallelism.md` — §一、§二（拓扑与进程模型）
