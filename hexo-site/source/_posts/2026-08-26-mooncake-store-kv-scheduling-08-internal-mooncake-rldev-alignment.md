---
title: "从统一内存池到多 Master：内部 Mooncake rldev 对我们的启示"
date: 2026-08-26
categories: [Mooncake Store 与 cache-aware 调度]
tags: [Mooncake, KV Cache, PD 分离, cache-aware 调度, vLLM, 驱逐策略, 学习笔记]
---

# 从统一内存池到多 Master：内部 Mooncake rldev 对我们的启示

> 本文整理自两篇内部文章的本地 HTML 原文、内部 Mooncake 仓库分支实查，以及我们在 PD 分离 +
> Mooncake Store 实验中得到的测量。文中会区分“文章设计”“仓库快照”和“我们的实验/推断”，
> 重点回答一个工程问题：**面对已经量化的 Master 瓶颈，哪些能力内部已经实现，哪些仍需自己补，
> 哪些 TODO 应当撤销或降级。**

## 1. 为什么这两篇文章正好命中当前问题

我们从 PD 分离推理出发，在内部 vLLM 上组合了两条 KV 通路：blade-kvt 负责当前请求的
Prefill → Decode 直传，Mooncake Store 负责跨请求、跨实例的共享 KV。随着 store 跑通，问题很快从
「能否命中」变成了「元数据控制面能否撑住」。

实测给出的压力不是抽象担忧：

```
单 Mooncake master 实测上限       约 4.5M keys/s
百实例冷缓存场景外推             约 14M keys/s
差距                             约 3.1 倍
```

王伟（盏一）的《Mooncake 统一内存池：AI Vibe Coding 与 Rust》开头也从同一个现象出发：
client 数量和并发 get/put 上升后，Master 同时承担心跳、状态同步、元数据查询和空间分配，逐渐成为瓶颈。
但他的场景是 **RL Pod 内的多 client**，而不是我们的 **PD 分离推理跨实例复用**。场景差异决定了
方案可以互相借鉴，却不能直接照搬。

第二篇《Mooncake 统一内存池：从默认 Evict 到 Linux Reclaim》继续回答统一内存池建成之后的
回收问题。它对应我们原 TODO 中最重的两项：B1「驱逐改为迁移」和 B3「稳定的近似 LRU」。随后对
内部 Mooncake 仓库的分支实查又发现，多 Master 分区也已经在 `rldev` 上实现。

因此，这次阅读真正改变的是路线图：**B1、B3、C2 不应再从零实现；C1 可以独立推进；采用
`rldev` 前则必须处理 group 原子性、部署介质和版本基线三类兼容问题。**

## 2. 第一篇：统一的不是一块内存，而是资源视图

### 2.1 原问题：一个 Pod 内存在四套彼此割裂的池

文章面对的典型形态是：一个 Pod 里有一个 RealClient 和多个 DummyClient。优化前，数据可能同时存在于：

- global memory；
- RealClient 的 local memory；
- local hot cache；
- 每个 DummyClient 独占的 shared-memory 区域。

这不只是「多了几次 memcpy」。真正的问题是 allocator、对象副本、元数据和生命周期分别属于不同的池，
系统没有一份统一的资源视图。同一个 key 随 DummyClient 数量增长而复制，内存占用与 client 数量近似线性增长；
put 还可能先写私有中转区，再搬到最终位置。

![统一内存池改造前后](/imgs/mcstore_internal_unified_pool.svg)

### 2.2 两条腿：LocalMaster + 共享池 G

改造由控制面和数据面两部分组成：

```
LocalMaster          控制面本地化：Pod 内可回答的问题不再全部访问全局 Master
Unified Pool G       数据面统一化：RealClient 创建 G，DummyClient 只 attach
```

共享一块 shm 还不够。要让它成为真正的统一池，至少要同时统一三件事：

1. **统一 allocator**：所有进程从同一份空间状态分配，避免每个 client 维护一套互相不知道的 free list。
2. **统一 MetaMap / HandleTable**：key 到 replica、handle 到调用实例的映射必须全局一致。
3. **统一生命周期**：`AcquireReplica` / `ReleaseReplica` 共同维护引用计数，任何进程都不能在别的进程仍持有
   replica 时提前释放。

改造后的直接效果是：一个 Pod 内同一个 key 只保留一份 memory replica；DummyClient 的增加不再带来
同等数量的局部副本。put 也可以直接序列化到最终位置，省掉私有 staging buffer。

### 2.3 两个容易被低估的实现细节

#### 虚拟地址不能跨进程共享

同一个 shm fd 在不同进程中 `mmap` 后，映射到的虚拟地址通常不同。共享元数据若保存裸指针，另一个进程
解引用时会指向错误位置。正确做法是保存相对偏移：

```text
offset = ptr_in_creator - creator_g_base
ptr_in_process_i = process_i_g_base + offset
```

因此，真正可跨进程传递的是 `(shm identity, offset, length)`，不是某个进程看到的 `void *`。

#### replica 地址不能代表一次 get 的生命周期

同一个 key 多次 `get_buffer()`，底层 `replica.ptr` 可能完全相同，但这些调用需要分别 acquire/release。
若用地址当 handle，第二次 get 会与第一次混在一起，引用计数无法判断哪次调用已经释放。文章采用单调递增的
`handle_id` 区分每次调用：

```text
handle_id -> {replica, caller, acquire state}
```

这里的 `handle_id` 不是对象身份，而是**一次借用关系的身份**。这是统一生命周期能成立的必要条件。

## 3. LocalMaster 能否直接解决我们的 Master 瓶颈

LocalMaster 的本质不是让全局 Master 更快，而是让一部分请求不再到达全局 Master：

```text
全局 Master 扩容：  增加控制面的处理能力
LocalMaster：       减少进入全局控制面的流量
```

这更接近我们 C1 `kv_events` 负向过滤的目标，而不是 C2 多 Master 分区。但 RL 与 PD 的局部性来源不同：

| | RL Pod 内多 client | 我们的 PD + 共享 KV Store |
|---|---|---|
| 数据共享关系 | 多个 client 常访问 Pod 内同一份对象 | store 的价值主要来自跨 Prefill 实例复用 |
| TP rank 数据 | 可能是同一对象的重复视图 | 各 rank 是不同 KV head shard，不能互相去重 |
| LocalMaster 可回答比例 | 预期较高 | 取决于请求命中的对象是否就在本地 |
| 主要收益 | 本地去重 + 少访问全局 Master | 只能减少已知本地状态的查询 |

所以 LocalMaster 对我们不是无效，而是必须先测一个比例：**全部 `batch_is_exist` key 中，有多少能由节点本地
的完整状态直接判定？** 如果大部分命中来自其他节点，LocalMaster 只优化少量本地查询，不能替代全局分区。

## 4. 第二篇：从默认 Evict 到 Linux Reclaim

### 4.1 先区分两类完全不同的对象语义

统一池 G 建成后，第二篇首先把对象按语义分成两类：

| 对象 | 典型场景 | put / get 语义 | 回收约束 |
|---|---|---|---|
| `softpin=true` | RL / Omni / EPD 中转数据 | put 要尽量成功；lease 内必须可恢复 | 释放本地前必须 swapout/offload 成功 |
| `softpin=false` | KV cache / 普通缓存 | put 可以失败；get 允许 miss | 可保留 cache 语义，允许删除可重建副本 |

文章的主要痛点来自第一类：生产节点的 G 被中转数据塞满，而默认 evict 对受保护对象既不能直接删，也没有
完整的迁移/降级流程，最终 `PutStart` 分配失败。

我们的 Mooncake KV cache 属于第二类。它可以 miss，miss 后可以重算。这一差异非常重要：它意味着文章的
reclaim 设计很有工程价值，但**不自动等于我们的最高优先级**。

### 4.2 新语义：evict 是移出本地 G，不等于从系统删除

改造把回收定义成「从当前节点的统一池 G 中移走」：

- `softpin=true`：必须先 remote swapout 到其他节点内存，或 offload 到更低层存储；成功后才能释放本地副本。
- `softpin=false`：非主副本可直接删；主副本仍优先 swapout/offload，避免系统最后一份数据意外消失。

这里的「迁移」只有在目标是另一个容量层、另一台尚有空闲的节点、segment draining 或 locality rebalance 时
才有意义。对于已经全局池化、且总 DRAM 达到高水位的 store，简单的 DRAM → DRAM 搬家不会减少总占用，
不能凭空扩容。

### 4.3 借的是 Linux reclaim 的职责划分

这套设计没有逐行照搬 Linux，而是借了几个成熟的职责边界：

- active / inactive 冷热链表；
- 后台 reclaim（类似 kswapd）与前台 direct reclaim 分开触发，但共享候选判断；
- 一次访问只设 referenced，第二次访问才晋升 active，避免扫描型流量污染热点；
- 先 isolate 候选，再在锁外检查 metadata、lease 和迁移条件。

![Mooncake Linux-reclaim 式状态机](/imgs/mcstore_internal_reclaim_state.svg)

在线路径被刻意压薄：

```text
PutEnd                   创建 KeyEvictInfo，插入 inactive 头部
AcquireReplica local hit 标记 referenced；若再次命中且仍在 inactive，则晋升 active
LocalRemove              幂等地从 evict list 摘除
```

新对象不因一次偶然访问就变成热点。第一次访问只设置 `referenced`，第二次访问才 promote；active 经过 aging
后再回到 inactive，形成一个近似的冷热循环。

### 4.4 为什么按 MetadataShard 维护，而不是全局 LRU

Mooncake metadata 原本就是 1024 个 shard。若再加一条全局 LRU 链表，所有 get/put/remove 都会竞争同一把锁，
热点很容易从 Master RPC 转移成 LRU 锁。

内部实现让每个 `MetadataShard` 维护自己的 `ShardEvictInfo`，再按
`shard.used_bytes / total_used_bytes` 分配回收 quota。这保留了原有分片并发性，也让热度结构与 metadata 的
锁域一致。

这条设计与我们的源码阅读完全吻合：默认 `BatchEvict` 会遍历 1024 个 shard；我们原本若只把排序键换成
`last_access`，很可能仍保留全局扫描和锁竞争，没有触及真正的扩展性问题。

### 4.5 isolate + epoch：链表不是难点，并发状态机才是

慢操作不能在 shard 锁内完成。原文描述的实现会在 `seinfo.mutex` 下把主链表交换到线程私有临时 list，
增加 epoch，随后在锁外检查对象是否仍存在、lease 是否有效、能否删除或迁移；其中节点跨 list 的移动使用
`splice` 来保持 iterator 有效。

几个并发边界不能混淆：

- `referenced_` 只用于近似热度，锁外 relaxed 读写可以接受；
- `lru_`、`epoch_`、`iter_` 决定链表结构，锁外修改会直接破坏容器；
- 节点跨 list 移动必须用 `std::list::splice` 保持 iterator 有效，不能 `erase + insert`；
- `LocalRemove` 必须幂等，因为对象可能同时处在主链表、isolate 临时链表或已被回收。

这解释了为什么文章强调「难点不在写两条链表，而在把所有并发状态列完整」。后续《Mooncake Evict：一次
`std::make_pair` 让 `iter_` 悄悄失效》也说明 iterator 生命周期是实际发生过的故障，不是理论警告。

### 4.6 direct reclaim 为什么必须按调用路径开关

direct reclaim 由前台分配失败触发，目标是尽快腾出 inactive 空间，不主动 shrink active。更关键的是，
它不能在所有路径上无条件开启：

```text
普通单次读 miss           可以允许 direct reclaim
batch read                不允许：容易把批处理路径变成回收风暴
remote swapout 写远端      不允许：远端分配再触发 reclaim 会形成回收环
```

如果 A 为腾空间 swapout 到 B，而 B 的接收分配又同步触发 direct reclaim 并尝试 swapout 回 A，就破坏了原来
IO 线程的单向等待假设。相关阅读《一行代码引起的分布式死锁》正是这类风险的前车之鉴。

## 5. 默认 Evict 到底是不是“随机”

第一篇把默认驱逐概括成「某种随机」，看起来与我们的源码结论冲突：我们确认 `lease_timeout` 是 last-touch
的单调函数——写入相当于 `now + 0`，读取授予约 10 秒 lease。

第二篇补全语境后，两者并不矛盾：

- 单个对象的 `lease_timeout` 确实与最近触碰时间单调相关；
- 但 10 秒 lease 把时间严重离散化；
- 系统没有一份持续维护的冷热顺序；
- 每轮跨 shard 扫描得到的是非原子快照；
- `nth_element` 只按分位点切候选，而不是维护严格次序。

所以「不是字面随机」和「不构成稳定的近似 LRU」可以同时成立。softpin 也不是单纯为了补偿随机驱逐，
而是在表达一种更强的可恢复性语义。

## 6. 内部仓库实查：文章能力最终落在 rldev

内部仓库为 `PAI-LLM/mooncake`。截至 2026-08-26 的分支快照显示：

| 分支 | 状态 | 关键能力 |
|---|---|---|
| `main` | 上游镜像，版本线较新 | 不包含文章中的 LocalMaster / Linux reclaim |
| `zy-unify-mp` | 盏一在 3 月的原始统一池分支 | `Add unify memory pool`，对应第一篇文章 |
| `rldev` | 主推、仍活跃，tip 为 2026-08-21 | 已吸收统一池，并包含 LocalMaster、AcquireReplica、ShardEvictInfo、direct reclaim、partitioned cluster mode |
| `rldev-egm` | EGM / NVLink host fabric 方向 | 面向扩展内存介质 |
| `vllm-kvs` | 较早的 EAS tiered storage 方向 | 与 vLLM KVS 接入相关，但不是当前主线 |

`rldev` 上还能找到完整的多 Master 部署文档：
`docs/source/deployment/vcns-multi-master-mooncake-store-deployment-guide.md`，通过
`--cluster_mode=partitioned` 启用分区模式。

![社区、rldev 与我们分支的能力关系](/imgs/mcstore_internal_branch_map.svg)

这次实查把三条重量级 TODO 的性质改了：

| 我们原来的 TODO | `rldev` 状态 | 新判断 |
|---|---|---|
| B1 驱逐 = 迁移而非删除 | `try_swapout`、primary、副本回收语义已存在 | 不从零实现；先确认 KV cache 是否需要覆盖 |
| B3 真 LRU / `last_access` | `ShardEvictInfo` + active/inactive + direct reclaim | 内部实现远超“换排序键”的 scope |
| C2 多 Master 分区 | `cluster_mode=partitioned` + 部署文档 | 暂停自研，先验证 DRAM KV 场景兼容性 |

## 7. rldev 的多 Master：与我们 M1–M4 模型的差异

### 7.1 我们的简化模型

合成压测中使用了一个静态 owner 函数：

```text
owner = (hash(key) % 1024) / (1024 / N)
```

它把 Mooncake 内部 1024 个 metadata shard 划成连续区间，用来验证「纯函数 owner + 多个独立 master」能否扩展。
这个模型回答了容量和 fan-out 代价，但不是 `rldev` 的真实成员管理方式。

### 7.2 rldev：一致性哈希环 + Redis 注册表

`rldev` 使用一致性哈希环定位 master，并用 Redis 保存集群成员/注册信息。master 之间不需要为每次查询互相通信。
扩缩容时只需迁移环上受影响的一部分，而不是让简单 `% N` 导致近乎全量重映射。

因此，我们文档里「N 必须整除 1024」不是实现约束，只是简化模型的产物，应当撤销。1024 仍是单个
Mooncake master 的 metadata shard 数，但不是 partitioned cluster 的 master 数量限制。

### 7.3 `max_registrable_capacity` 是均衡机制

分区文档里一个容易误读的参数是 `--max_registrable_capacity`。它不只是“留多少安全余量”：当某个 segment
哈希到的 master 已达可注册容量上限时，client 会沿哈希环继续探测下一个 master。把上限设为每个 master
应承载的目标份额，可以通过“拒绝 + 环上重探”强制 segment 容量更均匀。

如果为了留裕量盲目把上限调大，segment 会停留在原始哈希分布，反而保留不均衡。这个机制平衡的是容量，
并不能消除热 key 的元数据热点。

### 7.4 我们的实测仍然有效

owner 机制不同，不影响以下结论：

- 单 master 真实上限约 4.5M keys/s；
- 8 master 合成压测达到 11.74M keys/s，已突破单点天花板；
- 部署形态下 fan-out 的主要代价约为 2ms 查询延迟，而不是压测器在 GIL 平台上表现出的 50% 吞吐损失；
- 一个热 key 的 metadata 仍然只属于一个 master，一致性哈希不能把同一个 key 的元数据查询拆给多台机器；
- 热共享前缀只有约 180 个 block，N=8 时小样本不均衡可达 29%–47%。

换句话说，`rldev` 已经提供了 C2 的实现，而 M1–M4 提供了这套实现缺少的**容量数字、fan-out 定价和热点上界**。

## 8. kv_events：不需要升级版本，但必须重新编译

两篇文章和 `rldev` 都没有给出我们 C1 的直接实现。反而是社区 Mooncake 0.3.12.post1 对应源码
`6041a609` 已经包含 RFC #1527 KV events publisher，关键 CMake 开关是：

```cmake
option(ENABLE_KV_EVENTS
       "Build master KV events ZMQ publisher (requires libzmq when ON)"
       OFF)
```

默认是 `OFF`。容易误判的地方在于：Master 的命令行 flag 和配置字段无条件存在；关闭编译开关时，
`KvEventPublisher` 会被编译成 stub。因此会出现一组非常像“功能已开启”的现象：

```text
--enable_kv_events 被接受
启动配置正常回显
进程不链接 libzmq
ZMQ 端口不 bind
没有任何 event 发出
```

所以 C1 的前置不是升级 268 个 commit，而是在同一源码基线上安装 `libzmq3-dev`，用
`-DENABLE_KV_EVENTS=ON` 重新构建，并运行已有的 `kv_event_publisher_test.cpp`。

但“能发事件”还不等于“可以安全过滤”。publisher 使用有界异步队列和 ZMQ PUB/SUB，事件可能因队列满、
slow joiner 或连接中断而丢失。负向过滤必须有保守状态机：

```text
UNKNOWN  不相信索引 absence；全部回退到 batch_is_exist
TRUSTED  从已知空 store/可靠快照开始，且 sequence 连续，才允许 absence 判 miss
GAP      发现 sequence gap 或 publisher epoch 变化，立即清空判断并退回 UNKNOWN
```

event 只能用于**否定**一部分查询，不能用“曾收到 stored event”替代最终的 `batch_is_exist/get`。第一阶段应做
observe-only：只统计理论可省 key 数、sequence gap、publisher dropped events 和索引内存，不改变请求行为。

![Master 瓶颈的三条缓解路线](/imgs/mcstore_internal_master_paths.svg)

## 9. 采用 rldev 的三个真实阻碍

### 9.1 没有 `group_ids`：TP shard 原子性会退化

我们的 `mooncake_store` backend 用 `ReplicateConfig.group_ids` 把一个逻辑 block 的所有 TP rank shard 绑定为
同一驱逐组。原因是 lookup 要求所有 rank 同时存在；如果驱逐只删 rank 0、留下 rank 1，剩余 shard 永远不可用，
还会因后续 `BatchExistKey` 被续租，成为长期死重。

`rldev` 的 `ReplicateConfig` 有 `replica_num`、`with_soft_pin`、`primary`、
`allow_direct_reclaim`、preferred segment 等字段，但该分支快照中没有社区版的 `group_ids`。

`primary` 解决“最后一个系统副本能不能删”，不等价于“一个逻辑 block 的多 rank shard 必须一起驱逐”。采用
`rldev` 前必须回答：

1. 内部版本是否通过 LocalMaster 或其他对象模型把 TP shard 合并了；
2. 若没有，能否把社区 `group_ids` 向前移植到 `rldev`；
3. active/inactive 和 swapout 是否对 group 原子执行；
4. 多 Master 分区后，group 的所有成员能否稳定路由到同一 master。

最后一点尤其重要：group ID 不仅要去掉 TP rank，还要保留 model、cache prefix 和 kv cache group 命名空间，
否则不同模型的相同 block hash 会被意外绑在一起。

### 9.2 主推部署是 vCNS / SSD 池化，不等于我们的 DRAM KV Store

`rldev` 的 partitioned 文档面向 vCNS、NVMe-oF、RAID0 文件段和 SPDK。provider 示例把
`global_segment_size=0`，只贡献 SSD segment，不贡献本地 DRAM；而我们的部署恰好相反：Prefill worker
贡献 host DRAM segment，并通过 RDMA 直接读写 GPU HBM。

因此要验证的不是“partitioned 能否启动”，而是：

- DRAM segment 能否在多个 master 间正确注册、发现和卸载；
- RDMA zero-copy get/put 是否仍保持原路径；
- client/master 故障和扩缩容时，DRAM 对象如何迁移或失效；
- `max_registrable_capacity` 对内存 segment 是否同样适用；
- LocalMaster 与 vLLM scheduler/worker 多进程模型如何对应。

### 9.3 版本线分叉：能力不是简单超集

内部 `main` 是较新的上游镜像，`rldev` 从更早基线长期演进。`rldev` 有统一池、reclaim 和 partitioned，
社区较新版本则有 `group_ids` 和 kv_events publisher。两条线不是谁完全包含谁，而是各自增加了不同能力。

这意味着“切到 rldev”不是一次普通升级，而是一次 feature merge。需要用能力矩阵逐项移植，不能只比较版本号。

另外，partitioned 部署文档里存在一个典型静默错配：master 的默认 `cluster_id` 是
`mooncake_cluster`，client 的 `redis://` URL 省略后缀时回退到 `mooncake`。两端都能正常启动，但加入的是
不同逻辑集群。生产配置必须显式写同一个 cluster ID，不能依赖默认值。

## 10. 对我们的实验结论该如何重新定位

内部 reclaim 已经实现，不代表我们的 KV cache 也应立刻切换。文章的核心动机来自 `softpin=true` 中转数据，
而我们的负载是 `softpin=false`、可重算的 cache。

我们已经拿到一组有价值的反面证据：

```
驱逐数据量       442 GB
驱逐 key 数       约 24 万
重算率变化       基本不变，仍贴 2/(N+1) 下界
端到端 p50 变化  约 +0.6%
```

这不能证明默认 evict 普遍足够好，因为 salt-per-arm 让旧数据严格比工作集更冷，恰好是 LRU 的理想情况。
但它足以说明：**在当前实验负载上，reclaim 的主要价值更可能是把 O(全部 key) 的扫描和锁竞争改掉，而不是
提升驱逐质量。**

如果要评估驱逐质量，仍需先构造复用距离重叠：反复重放同一批 trial、不换 salt、store 装不下全部，
让旧数据仍会再次被访问。没有这个基线，就无法区分“reclaim 更聪明”和“负载本来就不在乎砍谁”。

## 11. 修订后的 TODO

### 立即做：C1 的最小闭环

1. 在社区 `6041a609` 基线上安装 libzmq，并用 `-DENABLE_KV_EVENTS=ON` 重编。
2. 先跑 publisher 自带单测和真实 master 端口/消息 smoke。
3. 写 observe-only subscriber，验证 sequence、丢包、重连和 master restart epoch。
4. 记录理论过滤比例；只有收益足够且状态机保守时才真正截断 `batch_is_exist`。

### 同时做：与 Token Foundry 对齐，而不是重写 B1/B3/C2

我们能提供：

- 单 master 4.5M keys/s 的实测上限；
- 8 master 11.74M keys/s 的聚合结果；
- fan-out 在部署形态下约 2ms 的真实定价；
- 热 key metadata 无法分摊、N=8 偏差 29%–47% 的上界；
- `softpin=false` KV cache 下 442GB 驱逐几乎无损的反面证据；
- TP rank shard 被驱逐切开的复现、`group_ids` 修复和偶数性验证。

需要向盏一 / `rldev` 维护者确认：

1. `rldev` 如何保证 TP shard 原子性，是否有 `group_ids` 的内部替代；
2. partitioned mode 是否在 DRAM + RDMA zero-copy KV 场景跑过；
3. LocalMaster 对跨实例 KV lookup 的可回答比例与一致性边界；
4. reclaim 是否计划覆盖 `softpin=false` KV cache，还是只服务中转数据；
5. 社区 kv_events publisher 能否与 `rldev` 合并；
6. 多 master 下 group、租约、snapshot 和扩缩容的语义。

### 暂缓或撤销

- 暂缓自研 C2：内部已有 partitioned cluster mode。
- 暂缓自研 B3：内部已有 shard-local active/inactive reclaim。
- 撤销“DRAM→DRAM 搬家等于扩容”的表述；只有跨 tier 或局部容量失衡时才成立。
- 不因内部实现存在就直接迁移我们的 KV cache；先做兼容性与动机验证。

## 12. 最终图景

Master 瓶颈并不存在唯一解，而是三种不同杠杆：

1. **LocalMaster**：利用 Pod 内局部性，让请求不到达全局 Master；
2. **kv_events 负向过滤**：保留全局 Master，但减少不可能命中的查询；
3. **partitioned cluster**：把仍然必须到达的全局元数据请求分摊到多个 Master。

它们可以叠加：先在本地回答，再用可信事件索引截断 miss，剩余查询进入一致性哈希分区。真正需要避免的，
是把三个问题混成一个：LocalMaster 不等于跨实例 store，kv_events 不等于权威 metadata，容量均衡也不等于
热 key QPS 均衡。

两篇文章和 `rldev` 调查最重要的结论不是“内部已经全做完”，而是：**内部与我们拥有的是互补证据。**
他们有成熟的统一池、并发 reclaim 和 partitioned 实现；我们有真实 KV cache 负载下的容量、延迟、热点和
驱逐损伤测量。下一步最有价值的工作不是各自重写对方已有的功能，而是把这两组能力合起来，先补 C1，
再决定 `rldev` 的哪些部分值得进入 DRAM + PD 的主路径。

## 材料与版本范围

- 王伟（盏一），《Mooncake 统一内存池：AI Vibe Coding 与 Rust》，3 月 18 日；内容按本地 HTML 原文复核。
- 《Mooncake 统一内存池：从默认 Evict 到 Linux Reclaim》，4 月 22 日；内容按本地 HTML 原文复核。
- 内部仓库 `PAI-LLM/mooncake`，分支状态观察时间：2026-08-26。
- 社区 Mooncake Store：`0.3.12.post1` 对应提交 `6041a609`。
- 我们的测量与实现：`llx/kvs-mooncake-store` 及本系列 00–07 章。

文中对内部分支的判断是上述时间点的快照；后续合并、改名或回迁都应重新核对，不能把分支搜索结果当成永久 API 契约。
