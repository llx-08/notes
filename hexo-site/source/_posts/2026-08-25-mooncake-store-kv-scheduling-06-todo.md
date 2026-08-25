---
title: "06 · TODO"
date: 2026-08-25
categories: [Mooncake Store 与 cache-aware 调度]
tags: [Mooncake, KV Cache, PD 分离, cache-aware 调度, vLLM, 驱逐策略, 学习笔记]
---

# 06 · TODO

按优先级分档，每条给出**依据**（为什么值得做）和**代价**。
没有依据的条目不放进来——这个项目已经因为「凭直觉排优先级」走过弯路。

## A. 高优先级：有实测支撑，代价小

| # | 事项 | 依据 | 代价 |
|---|---|---|---|
| A1 | **`local_first` vs `random` 对照** | 这是迁移设计的唯一定量前提。已知不利事实：RDMA 模式下本机命中也走 loopback、不退化成 memcpy，所以差距可能很小。**差距小 → `local_first` 不值得 → 迁移失去目的**，必须先测 | 一个 flag + 一次重启（3 分钟）+ 两轮 campaign |
| A2 | **Config C：收窄 store 带宽** | 唯一被实测过能产生两位数策略差异的条件（跨机单网卡曾 +27%）。`run_node.sh` 已支持 `MC_DEVICE`，设成单个 `mlx5_bond_0` 即可；或 protocol 换 tcp | 一个变量 + 一次重启 |
| A3 | **metadata key 跨 tp rank 去重** | 零精度代价，砍 `tp_size` 倍 master 压力。语义上本来冗余：lookup 的逻辑就是「全 rank 都为 1 才算命中」，一个 block 的各 rank 分片要么都在要么都不在 | vLLM 侧 Python，中等 |
| A4 | **`VLLM_KVS_ON_MIN_LENGTH` 提到 `2 × block_size`** | 命中 1 个 block 只省 `block_size` 个 token 的重算，却要付 `tp_size` 个 key 的 RPC + master 侧租约写锁 | 一行 |

## B. 驱逐改动（需从源码重编 mooncake）

**共同前置，比原先设想的严格得多**：[05 章](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-05-eviction-and-migration/) 已证明
「让驱逐发生」不够——还必须**让驱逐有机会砍错**。
salt-per-arm 的负载里垃圾严格比工作集更旧，LRU 怎么砍都对，所以任何驱逐改动都测不出差异。

> **正确的前置实验**：`replay_trace.py` 加 `--repeat`，反复重放同一批 trial（不换 salt）
> + store 装不下全部 → 制造复用间隔重叠。有了这个基线才能评估任何驱逐改动。

| # | 事项 | 依据 |
|---|---|---|
| B1 | **驱逐 = 迁移而非删除**：`BatchEvict` 出口从「删副本」改成「发 MoveTask」 | 原语已存在（`CreateMoveTask` + `FetchTasks` + `MoveStart/End` + `MarkTaskToComplete`），`offload_on_evict` 已是同一模式的实例。把分层从两级扩成三级，多一级比 SSD 快两个数量级的容量 |
| B2 | **按前缀深度加权** | 纯 recency 会「砍掉最值钱的、留下最不值钱的」。信号已存在：`group` 机制 + block hash 链编码位置。与「探测即续命」互补 |
| B3 | 内存驱逐排序键 `lease_timeout` → 真实 `last_access` | 变成真正的批量 LRU；`storage_backend.h` 的 SSD 路径有现成范式可抄（`{last_access_ns_, bucket_id}` 有序集合） |
| B4 | 复用频率信号 | HEAD 的 `dynamic_replication` 有 per-key heat；我们这版可自己在 `ObjectMetadata` 加字段（已确认现在没有 per-key 命中计数） |

优先级上 **B1 > B2 > B3**：B3 只是换个受害者、总容量不变；B1 真正扩展了有效容量。

## C. 独立线：master 扩展性

| # | 事项 | 依据 |
|---|---|---|
| C1 | **接 `kv_events`（mooncake 侧 + vLLM 侧两条流）** | 消除 `batch_is_exist` RPC——冷缓存外推 14M keys/s 必然过载，这是单 master 会先撞的墙。顺带消除索引假阳性（但那个只值 2%，不是主要理由）。**零 C++ 改动**，master 加两个 flag + 写 ZMQ 订阅端 |
| C2 | 按 key hash 把 metadata 分片到多 master | 当前完全不支持（grep 过 `master_shard`/`multi_master`/`consistent_hash` 全无）。是比改驱逐**更有上游价值**的贡献方向 |

## D. 已知但暂缓

- **hybrid 模型支持 M2~M6**（见 `mooncake_store_progress.md` §6.1）。
  前置：先估命中率——hybrid 下 attention `block_size` 被强制成 784，
  `lcm_block_size` 对齐后 mamba 的候选边界很稀疏，收益可能不如预期。
- **旧 KVS backend 的 group 维切片 + 缺 `is_hybrid` 保护**，hybrid 下会静默出错（既存 bug）。
- **PD 只有冷启动第一个请求输出正确**，之后退化成 `. . . .`。
  已确认与本分支改动无关（纯 kvt 的 P 也一样）。待查 `--async-scheduling`。
- **vLLM safetensors 加载器单流读**：280 GB 模型 90 分钟，其中绝大部分是白等
  （共享盘单流 71 MB/s，12 并行 490 MB/s）。本身是个可优化点。

## 附：复现步骤

```bash
# 0. 前提：4 台机器已预热模型到 /dev/shm
#    笔记本上有 bringup_150b.sh；test1 的 runs/mcstore/ 有其余脚本

# 1. 起集群（含 master 重启 = 清空 store）。第二个参数是 GPU_BLOCKS
bash bringup_150b.sh n1 8000       # Config A: 512k token/P
bash bringup_150b.sh n2 3000       # Config B: 192k token/P

# 2. 等 8/8 就绪（用端口连通性，不要用 pgrep）

# 3. 交替重复对照（4 臂，每轮查驱逐非零即中止）
ssh test1 'cd .../runs/mcstore && setsid nohup ./campaign.sh c8k 2 >/dev/null 2>&1 &'

# 4. 饱和实验（8 臂，不中止，靠累积溢出）
ssh test1 'cd .../runs/mcstore && setsid nohup ./saturate.sh 8 >/dev/null 2>&1 &'

# 5. 分析
python3 compare_campaigns.py campaign_c8k.log campaign_c3k.log
python3 analyze_saturate.py
python3 master_load.py
python3 recover_mix.py n2          # NODE_TAG 配错时事后补算 mix
```

**注意**：`campaign.sh` / `saturate.sh` 里的 `NODE_TAG` 必须与 bringup 的 tag 一致
（`n1` / `n2`），否则 `run_policy.sh` 会扫错节点日志，mix 三列静默变 0%。

### 脚本清单（都在 `runs/mcstore/`，旧版备份 `*.bak_32b`）

| 脚本 | 作用 |
|---|---|
| `run_node.sh` | 起单个 P/D；支持 `MODEL_ROOT`/`QUANT`/`KV_DTYPE`/`BLOCK_SIZE`/`HF_OVERRIDES`/`MAX_BATCHED`/`P_GPU_UTIL`/`DISABLE_HYBRID_KVCM`/`EAGER`；就绪循环「超 deadline 后只要 shard 计数还在动就继续等」 |
| `launch_local.sh` | 起本机那一份（slot 1-4），透传上述全部 + `GPU_BLOCKS` |
| `run_policy.sh` | 跑一个策略一轮；`MODEL_PATH` 全路径覆盖 + `NODE_TAG` |
| `bringup_150b.sh` | 从笔记本驱动的全集群重启，含 master 重启（清空 store）+ 4 slot 并行派发 |
| `preload_shm2.sh` | 模型预热到 `/dev/shm`，12 并行 + 逐文件大小校验 |
| `campaign.sh` | A B A B 交替对照，每轮查驱逐非零即中止 |
| `saturate.sh` | 让 store 自然溢出，不中止，逐臂记录驱逐 delta |
| `master_load.py` | 从 master 日志抽 batch op 的峰值 req/s 和 item/s |
| `compare_campaigns.py` | 合并对照表 + 噪声判据（自动拒绝小于噪声的「结论」） |
| `analyze_saturate.py` | 饱和 breakdown，按驱逐状态自动分组 |
| `recover_mix.py` | 按时间戳聚类事后补算 mix |
