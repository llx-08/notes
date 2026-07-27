# 00 · 零基础导读：从 token、矩阵到 MoE 与 Expert Parallelism

> 如果你刚学完 C/Python，还没有系统学过深度学习、CUDA 或分布式系统，请先读本章。
> 本章不追求“背 API”，而是先建立一条完整因果链：
>
> **文字 → token → 向量 → FFN/MoE → expert 分布到多张 GPU → token 必须跨卡搬运 → EP。**

---

## 1. 先认识最小单位

### 1.1 bit、Byte 与常见容量单位

- `bit` 是一个二进制位，只能是 0 或 1。
- `Byte`（字节，缩写 `B`）通常由 8 bit 组成。
- `KB/MB/GB` 表示容量；`GB/s` 表示每秒能搬多少 GB。
- `Gb/s` 中的小写 `b` 表示 bit，因此 `400 Gb/s` 理论上等于 `50 GB/s`，还没有扣除协议开销。

最容易犯的错：

```text
400 Gb/s ÷ 8 = 50 GB/s
```

不能把 `400 Gb/s` 直接当作 `400 GB/s`。

### 1.2 标量、向量、矩阵与 tensor

- 标量：一个数，例如温度 `25`。
- 向量：一排数，例如 `[0.2, -1.1, 0.7, 2.0]`。
- 矩阵：很多行、很多列的数。
- tensor（张量）：对标量、向量、矩阵以及更高维数组的统称。

在大模型里，一个 token 常用一个长度为 `H` 的向量表示。`H` 叫
`hidden_size`。假设 `H=4`：

```text
token "cat" → [0.2, -1.1, 0.7, 2.0]
```

真实模型中的 `H` 往往是几千，而不是 4。我们使用小数字只是为了能手算。

如果一批里有 `T=3` 个 token，hidden states 的形状通常写成：

```text
[T, H] = [3, 4]

[
  [ 0.2, -1.1,  0.7, 2.0],  # token 0
  [ 1.0,  0.5, -0.3, 0.8],  # token 1
  [-0.4,  1.2,  0.9, 0.1],  # token 2
]
```

“shape 是 `[T, H]`”只描述有多少行和列，不描述数值内容。

### 1.3 dtype 决定一个元素占多少字节

常见 dtype：

| dtype | 每元素字节数 | 粗略理解 |
|---|---:|---|
| FP32 | 4 B | 精度高、占用大 |
| BF16 / FP16 | 2 B | 大模型常见计算格式 |
| FP8 | 1 B | 更省带宽和显存，但需要 scale |
| INT8 | 1 B | 8 位整数 |
| FP4 / INT4 | 0.5 B | 两个 4-bit 元素装进 1 Byte |

一个 `[2048, 7168]` 的 BF16 tensor 大小约为：

```text
2048 × 7168 × 2 Byte
= 29,360,128 Byte
≈ 28 MiB
```

这条公式非常重要：

```text
tensor 字节数 = 所有维度相乘 × 每元素字节数
```

EP 的通信量最终都可以还原成这条公式。

---

## 2. 大模型的一层在做什么

一个简化的 Transformer 层可以看成两大部分：

```text
hidden states
  ├─ Attention：token 之间交换信息
  └─ FFN：每个 token 独立做两到三个矩阵乘法
```

本系列聚焦 FFN 被 MoE 替换后的情况。

### 2.1 Dense FFN：所有 token 使用同一套参数

先忽略 bias 和激活函数，一个 FFN 可以粗略写成：

```text
y = x · W
```

如果：

```text
x.shape = [T, H]
W.shape = [H, N]
```

那么：

```text
y.shape = [T, N]
```

矩阵乘法可以理解为：输入的每一行与权重的每一列做“对应元素相乘再求和”。

例如：

```text
x = [2, 3]
W = [[1, 4],
     [5, 2]]

y[0] = 2×1 + 3×5 = 17
y[1] = 2×4 + 3×2 = 14
所以 y = [17, 14]
```

Dense FFN 的关键是：所有 token 都使用同一个 `W`。

### 2.2 MoE：准备很多套 FFN，但每个 token 只选少数几套

Mixture of Experts（MoE）把一套 FFN 变成多套：

```text
expert 0: W0
expert 1: W1
expert 2: W2
expert 3: W3
```

每个 expert 本质上仍是 FFN，不是一个独立完整的大模型。

router（路由器）为每个 token 计算每个 expert 的分数，再选 Top-K。假设
`TopK=2`：

```text
token A 的 router 分数：
expert 0: 0.10
expert 1: 0.60
expert 2: 0.25
expert 3: 0.05

Top-2 = expert 1 和 expert 2
```

token A 会分别经过 expert 1、expert 2，最后按 router weight 加权：

```text
output_A = 0.70 × expert1(A) + 0.30 × expert2(A)
```

这里的 `0.70/0.30` 通常由选中后的分数归一化得到。不同模型的 router
实现可能用 softmax、sigmoid、分组 Top-K 或额外 bias，但“选 expert”
这一核心思想不变。

### 2.3 为什么 MoE 能扩大参数量，却不等比例扩大计算量

假设有 64 个 experts、TopK=2：

- 模型保存了 64 套 expert 权重；
- 每个 token 只激活其中 2 套；
- 单 token 的 expert 计算量更接近 2 套 FFN，而不是 64 套。

这叫“稀疏激活”。但权重仍然要放进显存。单卡放不下时，就需要把 experts
分散到多张 GPU，这正是 EP 的起点。

---

## 3. GPU、进程、rank 到底是什么

### 3.1 一张 GPU 不等于一个程序

操作系统运行的是进程。分布式推理通常启动多个 worker 进程，每个进程控制
一张 GPU。为了在通信组中标识成员，每个进程获得一个整数编号：

```text
rank 0 → GPU 0
rank 1 → GPU 1
rank 2 → GPU 2
rank 3 → GPU 3
```

- `world_size=4`：组里共有 4 个 rank。
- `rank=2`：当前进程是第 2 号成员。
- rank 是逻辑编号，不必永远等于物理 GPU 编号。

### 3.2 local rank、global rank 与 group rank

跨两台机器、每台 4 卡时：

```text
机器 A: global rank 0,1,2,3；local rank 0,1,2,3
机器 B: global rank 4,5,6,7；local rank 0,1,2,3
```

同一个进程还可能同时属于 DP group、TP group、EP group。它在不同 group
里的 `rank_in_group` 可能不同。读代码时一定先问：

> 这个 rank 是哪个通信组里的编号？

---

## 4. DP、TP、PP、EP：用“做小组作业”理解

| 并行方式 | 小组作业类比 | 模型中的含义 |
|---|---|---|
| DP | 每组拿同一本书，处理不同题目 | 复制模型，处理不同请求/token |
| TP | 几个人共同切分同一道大题 | 切分一次矩阵乘法的权重与计算 |
| PP | 每个人负责流水线的一道工序 | 不同层放到不同设备 |
| EP | 每个人保管不同专业的参考书 | 不同 experts 放到不同设备 |

### 4.1 TP 与 EP 最容易混淆

假设有 4 张 GPU。

TP=4 时，一个 expert 的权重可能被切成 4 份：

```text
expert 0 的 1/4 → GPU 0
expert 0 的 1/4 → GPU 1
expert 0 的 1/4 → GPU 2
expert 0 的 1/4 → GPU 3
```

EP=4 时，不同 expert 整体分给不同 GPU：

```text
expert 0 → GPU 0
expert 1 → GPU 1
expert 2 → GPU 2
expert 3 → GPU 3
```

现实系统可以组合 TP、DP、EP，所以不要只凭“用了 4 卡”判断并行方式。

---

## 5. 为什么 EP 必然引出跨卡通信

现在用一个最小例子。

```text
2 个 rank，4 个 experts，TopK=1

rank 0 保存 expert 0、1
rank 1 保存 expert 2、3
```

输入 token 最初按请求/批次分布：

```text
rank 0 拥有 token A、B
rank 1 拥有 token C、D
```

router 的选择：

```text
A → expert 3（在 rank 1）
B → expert 0（在 rank 0）
C → expert 1（在 rank 0）
D → expert 2（在 rank 1）
```

于是：

```text
rank 0 必须把 A 发给 rank 1
rank 1 必须把 C 发给 rank 0
```

expert 算完后，结果通常还要回到 token 原来的 rank，以便继续后续层：

```text
expert 3(A) 从 rank 1 返回 rank 0
expert 1(C) 从 rank 0 返回 rank 1
```

第一次搬运叫 dispatch，返回叫 combine。

### 5.1 “home rank”不是 expert 所在 rank

- home rank：token 原本属于哪个 rank。
- expert rank：被选中的 expert 存在哪个 rank。

两者相同就可以本地处理；不同才产生远端流量。combine 必须记住 home rank，
否则不知道结果该送回哪里。

### 5.2 TopK=2 时，一个 token 会产生两份 expert 工作

如果 A 同时选择 expert 1 和 expert 3：

```text
A → rank 0 上的 expert 1
A → rank 1 上的 expert 3
```

A 的 hidden vector 在逻辑上有两条 expert 路径。combine 时不仅要送回，还要
按 router weight 求和：

```text
output_A = w1 × expert1(A) + w3 × expert3(A)
```

所以“token 数量”与“expert assignment 数量”不是一回事：

```text
assignment 数量 ≈ token 数 × TopK
```

---

## 6. 一次 MoE 层的完整数据流

```text
① 输入 hidden states [T, H]
       │
② Router 计算 topk_ids、topk_weights [T, K]
       │
③ 根据 expert 所在 rank 统计 send counts
       │
④ Dispatch：把 token hidden 和路由元数据发到 expert rank
       │
⑤ Permute/Sort：同一 expert 的 token 排到一起
       │
⑥ Grouped GEMM：各 expert 执行 FFN
       │
⑦ Combine：结果送回 home rank，并按 topk weight 归并
       │
⑧ 输出 hidden states [T, H]
```

注意输入和最终输出通常仍是 `[T, H]`，但中间为了 Top-K 会临时扩张为约
`T×K` 个 assignments，还会因为 kernel 对齐而 padding。

---

## 7. 通信量怎样手算

假设：

```text
本 rank 有 T = 1024 个 token
hidden_size H = 4096
dtype = BF16 = 2 Byte
TopK = 2
其中 75% assignment 需要发到远端
```

仅 hidden payload 的 dispatch 量近似为：

```text
1024 × 2 × 75% × 4096 × 2 Byte
= 12,582,912 Byte
≈ 12 MiB
```

combine 的 expert output 通常也是 hidden vector，因此量级相近。实际还要加：

- expert id；
- top-k weight；
- source/home rank；
- token offset；
- 对齐 padding；
- 通信协议头和控制消息。

这个估算能回答一个重要问题：优化 4 Byte 的 metadata，通常没有优化
几千维 hidden vector 的收益大；但在 decode 小 batch 下，固定 metadata 和
同步延迟又可能成为主导。

---

## 8. 一个可以在 CPU 上运行的路由模拟器

下面代码不需要 GPU，也不实现真实通信；它只展示“谁应该发给谁”。

```python
from collections import defaultdict

# 4 个 expert 均匀放到 2 个 rank。
expert_to_rank = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
}

# (home_rank, token_name, topk_experts)
tokens = [
    (0, "A", [3, 1]),
    (0, "B", [0, 2]),
    (1, "C", [1, 3]),
    (1, "D", [2, 0]),
]

send_buckets = defaultdict(list)
for home_rank, token, experts in tokens:
    for expert_id in experts:
        dst_rank = expert_to_rank[expert_id]
        send_buckets[(home_rank, dst_rank)].append((token, expert_id))

for (src, dst), assignments in sorted(send_buckets.items()):
    print(f"rank {src} -> rank {dst}: {assignments}")
```

预期输出：

```text
rank 0 -> rank 0: [('A', 1), ('B', 0)]
rank 0 -> rank 1: [('A', 3), ('B', 2)]
rank 1 -> rank 0: [('C', 1), ('D', 0)]
rank 1 -> rank 1: [('C', 3), ('D', 2)]
```

真实 DeepEP/vLLM 做的事情更复杂、更并行，但路由表的核心含义就是这四个桶。

---

## 9. 对照本地 vLLM 代码

以当前 `~/codes/vllm_comm` 为例：

| 概念 | 代码入口 | 初读时看什么 |
|---|---|---|
| AG+RS dispatch/combine | `vllm/distributed/device_communicators/all2all.py` | `all_gatherv` 与 `reduce_scatterv` |
| DeepEP HT | `.../prepare_finalize/deepep_ht.py` | `get_dispatch_layout`、`buffer.dispatch`、handle |
| expert 映射 | `.../expert_map_manager.py` | `determine_expert_map` 里 `-1` 的含义 |
| expert 计算 | `.../experts/deep_gemm_moe.py` | workspace、对齐、两次 GEMM |
| EPLB | `vllm/distributed/eplb/policy/default.py` | 复制热点 expert、分组装箱 |

不要一开始逐行读 Triton/CUDA kernel。先在 Python 层回答：

1. 输入 tensor 的 shape/dtype 是什么？
2. 哪个函数改变了布局？
3. 哪个函数发生跨 rank 通信？
4. 哪个 handle/offset 用于恢复原顺序？
5. 输出何时才能安全被下一条 CUDA stream 使用？

---

## 10. 初学者常见误区

### 误区 1：expert 是一张 GPU

不是。一张 GPU 可以保存多个 experts，一个 expert 也可能因 TP 被切分。

### 误区 2：All-to-All 表示每个 rank 每次都给所有 rank 发一样多

不是。MoE 路由是数据相关的，真实 send count 往往不均匀。All-to-All 描述的是
“每个 rank 都可能向不同 rank 发送不同数据”的通信模式。

### 误区 3：TopK=2 就一定产生两倍网络流量

assignment 数约变成两倍，但其中一部分可能是本地 expert；量化、去重、布局和
实现也会影响真实字节数。它是很好的量级估算，不是无条件精确公式。

### 误区 4：通信 API 返回就代表 GPU 可以读结果

异步系统里“提交成功”和“完成”不同。通常要用 CUDA event、stream wait、
future 或 CQ completion 建立完成边界。

### 误区 5：带宽高就一定延迟低

带宽像高速公路每秒能通过多少车，延迟像一辆车从入口到出口要多久。
prefill 大消息更看带宽，decode 小消息更容易受固定延迟影响。

---

## 11. 读完本章应该能回答

1. `[T, H]` 中的 `T`、`H` 各是什么？
2. BF16 的 `[1024, 4096]` tensor 约占多少字节？
3. Dense FFN 与 MoE 的根本差异是什么？
4. 为什么 64 experts、TopK=2 不等于每个 token 计算 64 个 experts？
5. rank、GPU、expert 三者为什么不能画等号？
6. dispatch 与 combine 分别搬什么、为什么要搬回来？
7. TopK=2 时 token 数和 assignment 数有什么区别？
8. DP、TP、EP 分别切分什么？

下一章：[01_ep_fundamentals.md](01_ep_fundamentals.md)
All-to-All 细讲：[01a_moe_all_to_all.md](01a_moe_all_to_all.md)

## 参考

- [vLLM Parallelism and Scaling](https://github.com/vllm-project/vllm/blob/main/docs/serving/parallelism_scaling.md)
- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [DeepEP README](https://github.com/deepseek-ai/DeepEP)
- [CUDA Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
