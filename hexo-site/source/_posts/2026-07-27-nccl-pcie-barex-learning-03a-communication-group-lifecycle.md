---
title: "03a. 通信组是怎样构建和变化的：PyTorch、NCCL 与 DeepEP"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 03a. 通信组是怎样构建和变化的：PyTorch、NCCL 与 DeepEP

> 本章回答一个看似简单、实际上跨越很多层的问题：  
> “一组 GPU 是怎样成为一个通信 group 的？运行中能不能加入或删除节点？”

先给结论：

1. **group 的直接成员通常是进程/rank，不是物理节点。**一个节点可以运行 1 个、
   8 个甚至更多 worker；“加入一个 8-GPU 节点”通常意味着启动 8 个新进程并把
   8 个新 rank 纳入下一代通信组。
2. `torch.distributed.ProcessGroup`、NCCL `ncclComm_t` 和 DeepEP
   `Buffer/ElasticBuffer` 是三层不同对象。它们常一层包一层，但不能混为一谈。
3. 常规 `dist.new_group()` 是**从现有 world 中选子集，新建另一个组**，不是修改
   原组，也不能把尚未启动的外部进程加进来。
4. TorchElastic 的扩缩容通常是**停止整组 worker → 重新 rendezvous → 以新的
   `RANK/WORLD_SIZE` 重启**，不是让一个新 rank 插入正在执行的 AllReduce。
5. 新版 NCCL 已有 `ncclCommSplit`、`ncclCommShrink` 和 `ncclCommGrow`，但它们
   仍然返回一个**新的 communicator**；应用必须先让旧通信静止，再统一切换句柄。
6. DeepEP V1 `Buffer` 和 V2 `ElasticBuffer` 都在构造时读取
   `group.rank()/group.size()`，其成员集合仍是固定的。V1 的 `enable_shrink`
   只是让 LL kernel 动态 mask 某些 rank，不等于重建一个更小的 ProcessGroup。
7. DeepEP V2 的 **Elastic** 目前指底层通信内存的灵活性，不代表成员热插拔。
   真正的在线 EP 扩缩容还要由 vLLM 一类上层框架创建新组、搬权重、重排专家并
   在安全点切换。

![从 rendezvous 到 DeepEP Buffer 的通信组构建流程](/imgs/communication_group_construction.svg)

![子组、弹性重启、NCCL grow/shrink 与 DeepEP mask 的区别](/imgs/communication_group_reconfiguration.svg)

---

## 1. 先统一五个最容易混淆的词

### 1.1 Node、Process、Worker、GPU 和 Rank

假设有两台机器，每台 4 张 GPU，使用：

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=node-a:29400 \
  train.py
```

一个常见映射是：

```text
物理节点 node-a
  ├─ process A0 → LOCAL_RANK=0 → GPU 0 → global RANK=0
  ├─ process A1 → LOCAL_RANK=1 → GPU 1 → global RANK=1
  ├─ process A2 → LOCAL_RANK=2 → GPU 2 → global RANK=2
  └─ process A3 → LOCAL_RANK=3 → GPU 3 → global RANK=3

物理节点 node-b
  ├─ process B0 → LOCAL_RANK=0 → GPU 0 → global RANK=4
  ├─ process B1 → LOCAL_RANK=1 → GPU 1 → global RANK=5
  ├─ process B2 → LOCAL_RANK=2 → GPU 2 → global RANK=6
  └─ process B3 → LOCAL_RANK=3 → GPU 3 → global RANK=7
```

这里：

- **Node**：物理机、虚拟机或容器，是调度系统管理的机器单位；
- **Process/Worker**：实际执行 Python/C++ 程序的操作系统进程；
- **GPU**：计算设备；NCCL 常见模式是一进程绑定一张 GPU；
- **global rank**：默认 world 中的编号；
- **local rank**：本节点内的编号，用来选择本地 GPU；
- **group rank**：某个子组内部重新从 0 开始的编号；
- **world size/group size**：对应 group 中的成员数。

例如：

```python
ep_group = dist.new_group(ranks=[0, 2, 4, 6])
```

全局 rank 与这个子组的 group rank 对应为：

| global rank | 是否在 `ep_group` | `ep_group` 内的 rank |
|---:|---|---:|
| 0 | 是 | 0 |
| 1 | 否 | -1 / NON_GROUP_MEMBER |
| 2 | 是 | 1 |
| 4 | 是 | 2 |
| 6 | 是 | 3 |

所以看到代码中的 `rank=1` 时，一定先问：

> 它是默认 world 的 global rank，节点内的 local rank，还是某个 EP/TP/DP
> group 里的 group rank？

### 1.2 “一组通信 group”究竟保存什么

一个可工作的通信组不仅是一张成员名单。它通常至少包含：

| 信息 | 作用 |
|---|---|
| 有序成员列表 | 决定谁参加，以及 group rank 怎样编号 |
| `rank` 与 `world_size` | 每个进程知道自己是谁、总共有多少成员 |
| generation/epoch | 区分扩缩容前后的两代 group，避免新旧消息串线 |
| rendezvous/Store namespace | 交换地址、unique ID、状态和 barrier key |
| backend | Gloo、NCCL、UCC 等实际执行者 |
| rank → device 映射 | 当前 rank 使用哪张 GPU |
| transport/topology | NVLink、PCIe、SHM、Socket、IB/RoCE 等路径 |
| communicator/连接资源 | NCCL channel、网络 connector、QP、注册内存等 |
| collective sequence | 保证各 rank 以相同顺序执行相同 collective |

可以把它理解成一支球队：

- 成员列表只是球员名单；
- rank 是球衣号码；
- Store/rendezvous 是赛前签到和交换战术的会议室；
- NCCL communicator 是已经建好的传球线路和跑位图；
- collective sequence 是全队共同执行的战术编号；
- generation 是“第几场比赛”，防止拿上一场的号码和战术加入本场。

---

## 2. 三层对象：ProcessGroup、NCCL communicator、DeepEP Buffer

### 2.1 它们不是三个名字指同一个东西

```text
应用 / 并行策略
  └─ “我要一个 DP/TP/EP 组，成员是哪些 global ranks？”

PyTorch c10d
  └─ ProcessGroup
       - 保存逻辑成员、group rank 映射、Store、backend
       - 提供 all_reduce / all_to_all / barrier 等统一 API

ProcessGroupNCCL backend
  └─ ncclComm_t
       - 绑定 nranks、ncclRank、CUDA device
       - 建 topology、channel、transport connector
       - 真正提交 NCCL collective

DeepEP
  └─ Buffer / ElasticBuffer
       - 接受现有 ProcessGroup
       - 读取 group.rank() / group.size()
       - 建 EP 专用显存、IPC/RDMA/NCCL GIN 资源
       - 提供 dispatch / combine
```

因此：

- 创建 ProcessGroup 不一定立刻创建 NCCL communicator；
- 创建 NCCL communicator 不等于创建 DeepEP Buffer；
- ProcessGroup 发生变化后，旧 DeepEP Buffer 不会自动理解新的成员；
- DeepEP mask 一个 rank，也不会自动修改 PyTorch 的 `group.size()`。

### 2.2 控制面和数据面

通信初始化常被误解为“一上来就在 GPU 间传大 tensor”。实际先发生的是小规模的
**控制面通信**：

```text
控制面：
  rendezvous、Store set/get、交换 unique ID、IP、IPC handle、QP 信息、barrier

数据面：
  NCCL collective、DeepEP token dispatch/combine、NVLink/PCIe/RDMA payload
```

Store 中传输的通常是几十到几千 Byte 的元数据。真正的模型 tensor 不通过
TCPStore 绕一圈；初始化完成后，它们走 NCCL/DeepEP 的数据面。

---

## 3. PyTorch 怎样构建默认 ProcessGroup

### 3.1 第一步：launcher 先让进程“相遇”

`torchrun` 在每个节点启动 worker，并给它们设置：

```text
RANK
WORLD_SIZE
LOCAL_RANK
LOCAL_WORLD_SIZE
MASTER_ADDR
MASTER_PORT
TORCHELASTIC_RUN_ID
```

随后程序通常执行：

```python
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

dist.init_process_group(
    backend="nccl",
    device_id=torch.device("cuda", local_rank),
)
```

`env://` 初始化方法根据环境变量连接 rendezvous/Store。每个进程必须拿到一致的
world 信息，默认 ProcessGroup 才能建立。

### 3.2 rendezvous 不是 AllReduce

rendezvous 解决的是：

1. 哪些 worker 属于这一代 WorkerGroup；
2. 这一代的 `RANK/WORLD_SIZE` 是什么；
3. 大家使用哪个 Store；
4. 谁已经到齐，何时可以开始初始化数据面。

它不负责执行大 tensor 的 AllReduce。可以把 rendezvous 看成“建群和发群成员
编号”，把 NCCL 看成“群建立后实际传数据”。

### 3.3 `device_id` 会影响 NCCL 初始化时机

PyTorch 的 `init_process_group(..., device_id=cuda_device)` 可以让 NCCL backend
在初始化时就创建 communicator，较早暴露网络或拓扑错误。未指定时，
`ProcessGroupNCCL` 常在第一次遇到该 CUDA device 上的 collective 时懒创建
communicator。

这解释了一个常见现象：

```text
init_process_group 返回成功
  ≠ 所有 NCCL 网络连接都已经验证成功

第一次 all_reduce 才报错
  可能只是 communicator/connector 在此时才真正初始化
```

---

## 4. `dist.new_group()` 怎样创建静态子组

### 4.1 最小例子

```python
# 所有 world rank 都必须按相同顺序执行这些 new_group
tp_group_0 = dist.new_group(
    ranks=[0, 1, 2, 3],
    backend="nccl",
)
tp_group_1 = dist.new_group(
    ranks=[4, 5, 6, 7],
    backend="nccl",
)

if dist.get_rank() < 4:
    dist.all_reduce(tensor, group=tp_group_0)
else:
    dist.all_reduce(tensor, group=tp_group_1)
```

默认情况下，即使 rank 7 不属于 `tp_group_0`，它也要进入第一次
`new_group([0,1,2,3])`。所有进程还必须以相同顺序创建多个 group，否则不同
进程可能把相同的 Store key 理解成不同 group，最终超时或死锁。

### 4.2 源码做了什么

当前 PyTorch `torch/distributed/distributed_c10d.py` 的主要路径是：

```text
new_group(ranks)
  → _new_group_with_tag(...)
      ① 读取 default group 的 backend、Store、global rank/world size
      ② 排序并校验 ranks，不允许重复或超出 default world
      ③ 计算 global rank → group rank
      ④ 生成 group_name
      ⑤ 为该 group 建 PrefixStore("<group_name>/", default_store)
      ⑥ _new_process_group_helper(...) 创建 backend
      ⑦ 保存 pg_group_ranks 映射
      ⑧ 可选执行初始化 barrier
```

`PrefixStore` 很重要。它相当于给同一个底层 KV Store 划 namespace：

```text
default_pg/...
group_1/...
group_2/...
```

否则两个同时存在的 ProcessGroup 都使用 key `"0"` 交换 NCCL unique ID，就会
互相覆盖。

### 4.3 `new_group` 不能做什么

`new_group(ranks=[...])` 要求每个 rank 都位于默认 world 的 `[0, WORLD_SIZE)`
范围内。因此：

```text
当前 WORLD_SIZE = 8

允许：new_group([0, 2, 4, 6])
不允许：new_group([0, 1, ..., 8])
                         ↑ rank 8 根本不属于当前 world
```

所以它是**静态划分/新增子组 API**，不是集群扩容 API。父 group 也不会因为创建
子组而少掉成员。

---

## 5. ProcessGroupNCCL 怎样建立底层 communicator

### 5.1 unique ID 为什么必须经过带外交换

NCCL 的经典多进程初始化接口是：

```c
ncclUniqueId id;
ncclComm_t comm;

if (rank == 0) {
  ncclGetUniqueId(&id);
}
oob_broadcast(&id, sizeof(id));  // MPI、TCPStore、文件或其他控制面

cudaSetDevice(local_device);
ncclCommInitRank(&comm, nranks, id, rank);
```

`ncclUniqueId` 是让所有 rank 知道“我们要加入的是同一个 communicator”的
bootstrap 标识。NCCL communicator 还没有建立时，显然不能用这个 communicator
自己广播自己的 ID，所以必须使用 **out-of-band，带外控制通道**。

PyTorch 的做法是：

```text
ProcessGroupNCCL::initNCCLComm
  ├─ rank 0: ncclGetUniqueId(&ncclID)
  ├─ broadcastUniqueNCCLID
  │    ├─ rank 0: store_->set(sequence_key, id_bytes)
  │    └─ other ranks: store_->get(sequence_key)
  └─ NCCLComm::create
       └─ ncclCommInitRank(nranks, rank, ncclID)
```

一个 ProcessGroup 可能按 device 或 P2P pair 创建多个 NCCL communicator，因此
源码用 `ncclCommCounter_` 产生递增 Store key，不能只用固定 key。

### 5.2 `ncclCommInitRank` 内部还会做什么

从应用视角可把过程概括为：

```text
拿到相同 unique ID
  → bootstrap 交换每个 rank 的地址与基本信息
  → 识别 GPU、NVLink、PCIe、NIC 拓扑
  → 搜索 ring/tree 等通信图
  → 选择 channel、算法、协议和 transport
  → 建立 P2P/SHM/NET connector
  → 注册/分配内部 buffer
  → 所有 rank 初始化完成
```

因此 communicator 不是一个只有 `rank` 和 `size` 的小整数；它持有大量与
拓扑、连接、stream 和资源生命周期相关的状态。

### 5.3 group rank 与 GPU 的绑定

`ncclCommInitRank(&comm, nranks, id, rank)` 中的 `rank` 是 **NCCL communicator
内部 rank**。调用前当前线程/进程应选择正确 CUDA device。典型一进程一 GPU
模式下是：

```text
ProcessGroup group rank i
  → ProcessGroupNCCL rank i
  → ncclComm rank i
  → 当前进程绑定的 CUDA device
```

这只是常见映射，不是“rank 天生就是 GPU 编号”。global rank 4 在第二台机器上
完全可能绑定它的本地 `cuda:0`。

---

## 6. “动态改变成员”到底有哪几种

“动态”一词至少包含四种完全不同的需求。

### 6.1 从固定 world 里建一个子组：`new_group` / `ncclCommSplit`

```text
父组 P = [0,1,2,3,4,5,6,7]

子组 A = [0,1,2,3]
子组 B = [4,5,6,7]

父组 P 仍然存在，成员没有改变。
```

NCCL 对应的底层接口是：

```c
// 父 communicator 的所有 rank 都要参与
int color = parent_rank < 4 ? 0 : 1;
int key = parent_rank;
ncclComm_t child;
ncclCommSplit(parent, color, key, &child, NULL);
```

相同 `color` 的 rank 进入同一 child communicator，`key` 决定 child rank
顺序。它适合 TP/DP/EP 维度划分，不是扩容。

### 6.2 TorchElastic 扩缩节点：整组重启并产生新 generation

TorchElastic 的 membership change 流程是：

```text
新节点到达或旧节点离开
  → agent 检测 membership change
  → 停止所有现有 workers，包括没有故障的 survivors
  → 新一轮 rendezvous
  → 形成新的 WorkerGroup
  → 重新分配 RANK/WORLD_SIZE
  → 所有 worker 从程序入口重启
  → 加载 checkpoint，重新 init_process_group
```

所以 TorchElastic 的“elastic”是**作业级弹性重启**：

- rank 在重启后可能改变，不能硬编码稳定 rank；
- world size 可以改变；
- 需要 checkpoint，否则会丢失重启前尚未保存的训练进度；
- 模型切分、optimizer state、sampler 都可能要按新 world size 恢复或重分片；
- 不会把新 worker 热插入一条正在飞行的 AllReduce。

### 6.3 在同一个进程生命周期里删除 rank：PyTorch `shrink_group`

当前 PyTorch 源码提供：

```python
# 只允许未被排除的 rank 调用
new_pg = dist.shrink_group(
    ranks_to_exclude=[failed_rank],
    group=old_pg,
    shrink_flags=0,  # SHRINK_DEFAULT
)
```

主要路径是：

```text
dist.shrink_group
  → 校验排除列表与 backend.supports_shrinking
  → ProcessGroupNCCL::shrink
  → NCCLComm::shrink
  → ncclCommShrink
  → 得到新的、更小的 ProcessGroup/backend/communicator
  → 重新建立连续 group rank 映射
  → 清理旧 ProcessGroup
```

关键限制：

1. 被排除的 rank **不能**调用 `shrink_group`；
2. 剩余 rank 必须传入一致的排除列表；
3. 默认模式下旧 communicator 不应有 outstanding NCCL work；
4. `SHRINK_ABORT` 可先终止父 communicator 上的操作，但出错 collective 的
   结果不能再假设正确；
5. 如果 shrink 默认 group，PyTorch 会销毁其他 ProcessGroup，因为旧的 global
   rank 映射已经不再一致；
6. 上层 DDP/FSDP/optimizer/EP mapping 是否支持继续运行，是另一个问题。

也就是说，通信 backend 能缩小，不代表训练状态能自动缩小。

### 6.4 直接用 NCCL Grow/Shrink：新 communicator 替换旧 communicator

现代 NCCL 提供：

```c
ncclCommShrink(parent, exclude, exclude_count,
               &smaller, &config, NCCL_SHRINK_DEFAULT);

ncclCommGrow(parent_or_null, new_total_nranks,
             unique_id_or_null, assigned_rank_or_minus_one,
             &larger, &config);
```

#### Shrink

```text
parent: [0,1,2,3]
exclude old rank 1

remaining old rank 0 → new rank 0
remaining old rank 2 → new rank 1
remaining old rank 3 → new rank 2

返回 smaller；不是把 parent 指针原地改小。
```

被排除 rank 不调用 `ncclCommShrink`；所有保留 rank 都要调用。默认模式要求父
communicator 没有 outstanding work。故障恢复场景可以使用
`NCCL_SHRINK_ABORT`。

#### Grow

Grow 同样会创建新 communicator：

```text
parent: [0,1,2,3]
加入两个新进程，目标 size = 6

① coordinator 在 parent 上调用 ncclCommGetUniqueId
② 控制面把 grow unique ID 发给两个新进程
③ 旧 rank 共同调用 ncclCommGrow(parent, 6, ..., -1, &larger, ...)
④ 新进程分别以 comm=NULL、rank=4/5 调用 ncclCommGrow
⑤ 所有人拿到 larger
⑥ 安全销毁 parent，统一把当前句柄切到 larger
```

伪代码：

```c
ncclUniqueId grow_id;

if (is_coordinator) {
  ncclCommGetUniqueId(parent, &grow_id);
  send_id_to_new_ranks(grow_id);  // 应用负责控制面分发
}

if (is_existing_rank) {
  ncclCommGrow(
      parent,
      6,
      is_coordinator ? &grow_id : NULL,
      -1,
      &larger,
      NULL);
} else {
  ncclCommGrow(
      NULL,
      6,
      &grow_id,
      assigned_new_rank,  // 4 或 5
      &larger,
      NULL);
}
```

Grow 的重要边界：

- 旧 communicator 上不能还有未完成操作；
- 所有旧 rank 和新 rank 都必须进入同一次 grow；
- 新 rank 编号位于原 size 之后，旧 rank 保持编号；
- grow ID 只能用于一次 grow；
- Grow 只解决 NCCL communicator，不负责启动新进程、搬模型权重、重建
  ProcessGroup、修改 optimizer 或 DeepEP expert mapping；
- PyTorch 当前有 `shrink_group()`，但没有对等的公共 `grow_group()` 封装。

### 6.5 为什么不能让新节点直接加入“正在执行的 collective”

假设 rank 0～3 已经开始：

```text
AllReduce #100，算法是 4-rank ring：
0 → 1 → 2 → 3 → 0
```

此时 rank 4 加入后，如果它认为这是 5-rank ring：

```text
0 → 1 → 2 → 3 → 4 → 0
```

同一个操作已经出现两个不同通信图。旧 rank 3 把 chunk 发给 0，而新 rank 4
等待 rank 3；所有人对数据分块数量、归约次数和完成条件的理解都不同，结果只能是
hang、越界或错误数据。

因此可靠扩缩容必须有明确 cutover：

```text
停止接收旧 epoch 的新任务
  → 等旧 collective/stream 到安全点
  → 构建下一代 group/communicator
  → 同步权重与路由状态
  → barrier / commit epoch
  → 原子切换当前句柄
  → 新 epoch 才开始发任务
  → 最后释放旧资源
```

这像数据库 schema migration：不能让一半线程使用旧 schema，另一半线程使用
新 schema。

---

## 7. DeepEP V1 怎样从 ProcessGroup 构建 Buffer

当前 DeepEP V1 `deep_ep/buffers/legacy.py` 的构造主线是：

```python
buffer = deep_ep.Buffer(
    group=ep_group,
    num_nvl_bytes=...,
    num_rdma_bytes=...,
    low_latency_mode=True,
    num_qps_per_rank=...,
)
```

内部可概括为：

```text
Buffer.__init__(group)
  ① self.rank = group.rank()
  ② self.group_size = group.size()
  ③ 创建 C++ _C.Buffer(rank, group_size, ...)
  ④ all_gather_object(local_device_id)
  ⑤ all_gather_object(local_cuda_ipc_handle)
  ⑥ 需要 RDMA/LL 时：
       - 选 root NVSHMEM unique ID
       - all_gather_object(unique IDs)
       - 配置 IBGDA/QP
  ⑦ runtime.sync(device_ids, ipc_handles, root_unique_id)
  ⑧ runtime.is_available() 后才能 dispatch/combine
```

这里的 `all_gather_object` 使用传入的 PyTorch ProcessGroup 做控制面元数据交换。
真正 token payload 走 DeepEP CUDA/NVLink/RDMA 数据面。

### 7.1 为什么旧 Buffer 不能直接接纳新 rank

Buffer 初始化时已经固定了：

- `rank` 和 `group_size`；
- 每个 peer 的 device ID；
- CUDA IPC handle；
- NVSHMEM/RDMA bootstrap 信息；
- QP 数量和 peer 映射；
- 通知区、mask 区、dispatch/combine buffer layout；
- 诸如 `[num_ranks]` 或 `[num_ranks, ...]` 的元数据尺寸。

如果从 8 rank 变成 10 rank，仅修改一个 Python 整数远远不够。旧 Buffer 根本
没有为 rank 8、9 分配连接、槽位和路由状态。

### 7.2 `enable_shrink` 和 mask 的真实语义

V1 LL 模式可以：

```python
buffer = deep_ep.Buffer(
    group=ep_group,
    ...,
    enable_shrink=True,
)

buffer.low_latency_update_mask_buffer(rank_to_mask=3, mask=True)
```

源码说明它为通信分配 mask buffer，kernel 在 dispatch/combine/clean 中不再与
被 mask rank 收发。

但此时：

```text
group.size()           仍是原值
buffer.group_size      仍是原值
其他 rank 的编号       不重排
PyTorch ProcessGroup   没变
NCCL communicator      没变
专家/权重映射           不会自动搬迁
```

所以它更接近**数据面隔离/跳过故障 peer**，不是完整的成员删除协议。应用仍要决定：

- 被屏蔽 rank 上的专家由谁接管；
- 发给那些专家的 token 如何重路由；
- 已在途请求如何失败或重试；
- 何时创建真正更小的新 group；
- 何时回收旧 QP、显存和 communicator。

`vllm_comm` 当前 `DeepEPLLAll2AllManager` 甚至明确写着：

```python
self.support_fault_tolerance = False
# TODO: set to True when FT is supported.
```

所以不能因为 DeepEP 暴露 mask API，就推导 vLLM 当前 DeepEP LL 后端已经完成
端到端容错。

---

## 8. DeepEP V2 `ElasticBuffer` 的 “Elastic” 是什么

DeepEP 2.1.0 的 V2 接口：

```python
buffer = deep_ep.ElasticBuffer(
    group=ep_group,
    num_max_tokens_per_rank=max_tokens,
    hidden=hidden,
    num_topk=topk,
    use_fp8_dispatch=True,
    explicitly_destroy=True,
)
```

类注释明确说明：

```text
"Elastic" refers to the flexibility of underlying memory:
currently GPU-only, with CPU and mixed backends on the roadmap.
```

也就是说，命名首先描述的是 Buffer 的内存与统一通信能力，而不是 worker
membership。

构造路径仍然固定成员：

```text
ElasticBuffer.__init__(group)
  ① rank_idx = group.rank()
  ② num_ranks = group.size()
  ③ get_nccl_comm_handle(group)
       ├─ 能复用时：取 ProcessGroupNCCL backend._comm_ptr()
       └─ 否则：
            all_gather_object(各 rank 本地 NCCL unique ID)
            _C.create_nccl_comm(root_id, group.size(), group.rank())
  ④ 根据 nranks/token/hidden/topk 计算并分配 buffer
  ⑤ _C.ElasticBuffer(group.rank(), group.size(), nccl_comm, ...)
  ⑥ CUDA synchronize + group.barrier()
```

当前 `ElasticBuffer` 没有 `connect_rank`、`disconnect_rank`、`grow_group` 或
`shrink_group` 成员 API。改变 ProcessGroup 后应该销毁/不再使用旧 Buffer，再用
新 group 构建新 Buffer。

---

## 9. vLLM 的 Elastic EP 为什么仍然需要很多上层步骤

`~/codes/vllm_comm` 当前基线 `f12b80c6e` 中，
`vllm/distributed/elastic_ep/elastic_state.py` 展示了真正的 EP 扩缩容控制面。

### 9.1 Scale-up

代码中的状态机大致是：

```text
旧 engines 等待新 engines 初始化
  → CREATE_STANDBY_GROUPS
  → TRANSFER_EXPERT_MAPPING
  → 等新 engines 完成权重初始化
  → TRANSFER_WEIGHTS
  → SYNC_KV_CACHE_MEMORY_SIZE
  → SWITCH_AND_PREPARE
  → EPLB_RESHUFFLE
  → COMPLETE
```

特别是：

```python
self.new_dp_group, self.new_dp_store = (
    self.new_parallel_config.stateless_init_dp_group(return_store=True)
)
```

它创建的是一套 **standby 新组**，不是把旧 `old_dp_group` 的 size 原地加一。

### 9.2 Scale-down

剩余 engine 的主线是：

```text
旧组 barrier
  → 在旧组仍完整时先做 EPLB reshuffle
  → 把待删除 rank 上的专家迁到保留 rank
  → 创建 standby groups
  → switch_and_prepare
  → 更新 parallel config
  → 待删除 engine shutdown
```

顺序不能反过来。如果先杀掉 rank，再想从它搬专家权重，数据已经不可达。

### 9.3 为什么 DeepEP 只是其中一个数据面组件

扩缩容需要同时改变：

```text
进程集合
  + PyTorch ProcessGroup
  + NCCL communicator
  + DeepEP Buffer
  + global expert → physical rank 映射
  + expert 权重实际放置
  + EPLB 统计与路由
  + KV cache/显存预算
  + 请求准入与 barrier epoch
```

DeepEP 负责快速搬 token，但不会替上层决定“专家 137 在扩容后应该属于哪个
rank”。因此 Elastic EP 是**控制面状态机 + 多种数据面工具**，不是调用一个
`ElasticBuffer` 构造函数就完成。

---

## 10. 三套实现能力对照

| 需求 | PyTorch | NCCL | DeepEP |
|---|---|---|---|
| 建默认 world | `init_process_group` | 通常由 backend 调 `ncclCommInitRank` | 接受已有 group |
| 固定 world 内建子组 | `new_group(ranks)` | `ncclCommSplit` | 用子 ProcessGroup 新建 Buffer |
| 删除 rank | 当前有 `shrink_group`，backend 必须支持 | `ncclCommShrink` 返回新 comm | V1 mask 可跳过 peer；完整缩容仍需新 group/Buffer |
| 新增 rank | TorchElastic 常用整组重启；当前无公共 `grow_group` | `ncclCommGrow` 返回新 comm | 无成员 grow API，需上层重建 |
| 故障中止 | PG watchdog/abort，具体依 backend | `ncclCommRevoke/Abort`、`SHRINK_ABORT` | V1 LL mask 是局部机制，框架仍要处理状态 |
| rank 重编号 | 新 ProcessGroup/弹性重启负责 | shrink 后新 comm 连续编号；grow 保留旧编号 | 构造时读取新 group |
| 权重/专家迁移 | 框架负责 | 不负责 | 不负责全局 placement 决策 |
| 安全切换 epoch | launcher/框架负责 | 只提供 communicator 操作 | 不提供完整控制面 |

### 10.1 在两台 GB200 机器上的只读核对

2026-07-27 在 `target_p` 与 `target_d` 上读取运行时：

```text
PyTorch: 2.11.0a0+a6c236b9fd.nv26.03.46836102
CUDA:    13.2
NCCL:    2.29.7

torch.distributed:
  new_group      present
  shrink_group   present
  grow_group     absent

libnccl.so.2:
  ncclCommSplit        present
  ncclCommShrink       present
  ncclCommGetUniqueId  present
  ncclCommGrow         present
  ncclCommRevoke       present

deep_ep:
  Buffer         present
  ElasticBuffer  absent
  enable_shrink  present
```

这正好说明“分层能力”：

```text
NCCL runtime 有 Grow
  ≠ PyTorch Python API 有 grow_group
  ≠ 当前安装的 DeepEP Buffer 会自动扩容
  ≠ vLLM 的模型/专家状态已经完成热迁移
```

---

## 11. 一个安全的在线扩缩容协议应包含什么

下面不是某个库的单一 API，而是一套应用级协议。

### 11.1 Scale-up：4 rank → 6 rank

```text
1. Admission
   控制器确认新节点健康，启动新 worker，分配下一代 epoch。

2. Quiesce old epoch
   旧 rank 停止接收需要旧拓扑的新请求，等待关键 collective/stream 到安全点。

3. Build standby control group
   新旧 worker 通过 Store/rendezvous 建下一代 ProcessGroup。

4. Build standby data plane
   创建新 NCCL communicator、DeepEP Buffer、QP/IPC/registered memory。

5. Transfer state
   复制模型/专家权重，恢复 optimizer/KV 元数据，计算新 expert placement。

6. Validate
   barrier、checksum、一个小 collective/dispatch smoke test。

7. Commit epoch
   所有 rank 原子地把 current_group/current_buffer 切到新对象。

8. Resume traffic
   新请求只使用新 epoch；旧 epoch 的剩余请求排空。

9. Retire old resources
   在确认无人引用后 destroy old Buffer/ProcessGroup/communicator。
```

### 11.2 Scale-down：6 rank → 4 rank

Scale-down 多一步“先救数据”：

```text
标记待删除 rank
  → 停止把新 token/请求路由过去
  → 在它仍活着时搬走专家权重和必要状态
  → 等旧通信到安全点
  → 建 4-rank 新组与新 Buffer
  → barrier + commit
  → 最后关闭待删除进程
```

如果是突然故障，无法搬数据，就必须从副本或 checkpoint 恢复；`shrink` 只能让
剩余通信者重新组成组，不能凭空恢复丢失的权重或请求状态。

---

## 12. 最常见的死锁与错误

### 12.1 成员列表或创建顺序不一致

```text
rank 0: new_group([0,1]), 然后 new_group([2,3])
rank 2: new_group([2,3]), 然后 new_group([0,1])
```

两个进程可能在相同 Store namespace 等待不同 peer。

### 12.2 新旧 group 混用

```text
rank 0～2 已切到 generation 11
rank 3 仍用 generation 10
```

即使两代 group 都叫 `ep_group`，内部 communicator、rank size 与 sequence
已经不同。

### 12.3 旧 CUDA work 还没完成就销毁 communicator

NCCL host API 返回常只表示 work 已入 CUDA stream。切换前要建立正确的 stream
同步；只做 Python barrier 不一定证明另一条 CUDA stream 上的数据面 work 已完成。

### 12.4 只改通信组，不改 tensor/layout

很多 buffer 的形状包含 `world_size`：

```text
send_counts[num_ranks]
recv_counts[num_ranks]
expert_offsets[num_ranks + 1]
mask[num_ranks]
registered_slots[num_ranks][...]
```

group 从 8 变 10 后继续复用旧数组，轻则路由错误，重则 GPU 越界访问。

### 12.5 把“mask”误认为“删除”

mask 只能使某条数据路径跳过 peer。成员元数据、rank 重排、Store、其他
ProcessGroup、DDP reducer 和模型 placement 仍可能保留旧世界。

---

## 13. 读代码的最短路径

### 13.1 PyTorch

```text
torch/distributed/run.py
  → torch/distributed/elastic/rendezvous/
  → torch/distributed/distributed_c10d.py
       init_process_group
       new_group / _new_group_with_tag
       shrink_group
  → torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
       broadcastUniqueNCCLID
       initNCCLComm
       ProcessGroupNCCL::shrink
  → torch/csrc/distributed/c10d/NCCLUtils.cpp
       NCCLComm::create / split / shrink
```

### 13.2 NCCL

```text
src/nccl.h.in
  → src/init.cc
       ncclCommInitRank
       ncclCommSplit
       ncclCommShrink
       ncclCommGetUniqueId
       ncclCommGrow
  → src/bootstrap.cc
  → src/graph/
  → src/transport.cc + src/transport/
```

### 13.3 DeepEP 与 vLLM

```text
DeepEP 2.1.0
  deep_ep/buffers/legacy.py
       Buffer.__init__
       low_latency_update_mask_buffer
  deep_ep/buffers/elastic.py
       ElasticBuffer.__init__
  deep_ep/utils/comm.py
       get_nccl_comm_handle

vllm_comm @ f12b80c6e
  vllm/distributed/device_communicators/all2all.py
       DeepEPHTAll2AllManager
       DeepEPLLAll2AllManager
       DeepEPV2All2AllManager
  vllm/distributed/elastic_ep/elastic_state.py
       ElasticEPScalingState
       _create_standby_groups
       _transfer_weights
       _switch_and_prepare
       _eplb_reshuffle_before_scale_down
```

---

## 14. 自检题

1. 为什么“添加节点”并不等于“给 group 添加一个 rank”？
2. global rank 4 为什么可能使用本机 `cuda:0`？
3. `dist.new_group([0,2,4,6])` 会修改默认 world 吗？
4. ProcessGroup 已创建成功，为什么第一次 AllReduce 仍可能在 NCCL 初始化时报错？
5. NCCL unique ID 为什么不能通过尚未创建的 NCCL communicator 自己广播？
6. `ncclCommGrow` 为什么返回新 communicator，而不能安全地原地修改旧 ring？
7. TorchElastic 的 surviving worker 为什么也要重启？
8. PyTorch `shrink_group` 能否自动把失败 rank 上的 optimizer state 恢复出来？
9. DeepEP `enable_shrink` 为什么不是完整 group shrink？
10. DeepEP V2 的 `ElasticBuffer` 中 “Elastic” 当前主要指什么？
11. EP scale-down 时为什么要在关闭待删除 rank 之前搬专家？
12. 一次安全 cutover 为什么既需要 CPU/Store barrier，也需要考虑 CUDA stream 上
    尚未完成的数据面 work？

---

## 一手资料与代码

- [PyTorch distributed：init、new_group、destroy 与 reinitialization](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch torchrun：rendezvous、membership change 与整组重启](https://docs.pytorch.org/docs/main/elastic/run.html)
- [PyTorch `distributed_c10d.py`：`new_group` 与 `shrink_group`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py)
- [PyTorch `ProcessGroupNCCL.cpp`：unique ID、lazy communicator 与 shrink](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)
- [NVIDIA NCCL communicator API：Init、Split、Shrink、Grow、Revoke](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)
- [NVIDIA NCCL `nccl.h.in`：公开 API 与 flags](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)
- [DeepEP 2.1.0 `legacy.py`](https://github.com/deepseek-ai/DeepEP/blob/dd758caf451848bd150e1046af3d0a73e5fff38d/deep_ep/buffers/legacy.py)
- [DeepEP 2.1.0 `elastic.py`](https://github.com/deepseek-ai/DeepEP/blob/dd758caf451848bd150e1046af3d0a73e5fff38d/deep_ep/buffers/elastic.py)
- [DeepEP 2.1.0 `comm.py`](https://github.com/deepseek-ai/DeepEP/blob/dd758caf451848bd150e1046af3d0a73e5fff38d/deep_ep/utils/comm.py)
