# 06 · TODO

按优先级分档，每条给出**依据**（为什么值得做）和**代价**。
没有依据的条目不放进来——这个项目已经因为「凭直觉排优先级」走过弯路。

## A. 高优先级：有实测支撑，代价小

| # | 事项 | 依据 | 代价 |
|---|---|---|---|
| A0 | ~~**`block_size` 64 → 512**~~ | **撤销。** `block_size` 由实际业务需求决定，不是我们能动的旋钮。原依据（把 14M 压到 1.75M）仍然成立，但杠杆不在我们手上 | — |
| A1 | ~~`local_first` vs `random` 对照~~ | 原为迁移设计的前提，但迁移已判定不需要（store 是全局池），此条随之降级。若将来做 `local_first` 才需要 | — |
| A2 | ~~**Config C：收窄 store 带宽**~~ | **已做，INDISTINGUISHABLE。** 见 [03 章阶段八](03_experiments.md)。只补了「store 读贵」而没补「store 流量大」，而 store 份额只有 2%，给一条只承载 2% 流量的通道降速没用 | 已花费 25 分钟 |
| A2b | ~~**Config D：3P+1D 让 store 承担流量**~~ | **已做，INDISTINGUISHABLE。** store 份额 2%→44%、单网卡、仅 3 节点，两个策略差 +0.39%（噪声 0.2%）。见 [03 章阶段九](03_experiments.md)。**cache-aware 这条线到此收结论** | 已花费 45 分钟 |
| A3 | ~~**metadata key 跨 tp rank 去重**~~ | **撤销**（早前已判定）。三项收益里两项已被 grouping 零协调拿到（租约合并刷新、元数据路由局部性），剩下「key 数量」那一项要付跨 rank 协调的代价 | — |
| A4 | **`VLLM_KVS_ON_MIN_LENGTH` 提到 `2 × block_size`** | 命中 1 个 block 只省 `block_size` 个 token 的重算，却要付 `tp_size` 个 key 的 RPC + master 侧租约写锁 | 一行 |

## B. 驱逐改动（需从源码重编 mooncake）

**共同前置，比原先设想的严格得多**：[05 章](05_eviction_and_migration.md) 已证明
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

M1~M4 已经把这条线的前提测清楚了，见 [07 章](07_partitioning_measured.md)。优先级随之重排。

| # | 事项 | 依据 | 代价 |
|---|---|---|---|
| C0 | ~~**把 store 调用做成真异步**~~ | **撤销，两个理由都是硬的。**（1）**做不到**：这版 87 个方法全同步，无 submit/poll 接口；包里 `MooncakeDistributedStoreAsync` 的实现就是 `run_in_executor(None, ...)`，与我们已用的 `asyncio.to_thread` 等价。要真异步只能改 C++ 重编。（2）**不需要**：它修的是客户端吞吐，而部署里每 rank 独立进程独立 GIL，即使百实例冷缓存外推也只用到每进程上限（2.1M keys/s）的 3.3%。原依据「fan-out 付 50%」是压测器伪影，见 [07 章 §3](07_partitioning_measured.md) | — |
| C1 | **接 `kv_events` 做负向过滤** | **现在排第一。** C0 撤销后，方向从「加快发送」改成「减少发出去的 key 数」——瓶颈在 master 侧而不在客户端。消除大部分 `batch_is_exist` RPC。冷缓存正是超载场景，而负向过滤恰好在冷缓存时最有效、在不需要时零成本（索引为空 → 全部「可能有」→ 完全等于现状）。**零 C++ 改动**，master 加两个 flag + 写 ZMQ 订阅端。注意只能作负向过滤，不能反过来用 | 中等 |
| C2 | 按 key hash 分片到多 master | **前提已验证**：M1 实测 8 个 master 聚合 11.74M（单 master 4.5M 的 2.6 倍），14M 外推需 ≥4 个。而且 owner 是纯函数 → **分区逻辑可以完全做在客户端，不需要 mooncake 支持「多 master」**。代价比原先估的低：fan-out 只值约 2ms 查询延迟（原写的 50% 吞吐是压测器伪影），真正的代价是热点探测流量 N=8 最坏偏差 29-47% 且无法用副本缓解 | 客户端侧可先做 |
| C3 | 多机 reader 的热点读实验 | M4 的唯一缺口：现在 reader 全在一台机上，所以永远测不到源端热点。按实测外推 1 台读端拉 28 GB/s、3 台就要 84 GB/s（超过单 bond），所以热点在更大规模下是真实的 | 需要多机 harness |

## D. 已知但暂缓

- **hybrid 模型支持 M2~M6**（见 `mooncake_store_progress.md` §6.1）。
  前置：先估命中率——hybrid 下 attention `block_size` 被强制成 784，
  `lcm_block_size` 对齐后 mamba 的候选边界很稀疏，收益可能不如预期。
- **旧 KVS backend 的 group 维切片 + 缺 `is_hybrid` 保护**，hybrid 下会静默出错（既存 bug）。
- **PD 只有冷启动第一个请求输出正确**，之后退化成 `. . . .`。
  已确认与本分支改动无关（纯 kvt 的 P 也一样）。待查 `--async-scheduling`。
- **`store.remove()` 对有租约的对象返回 -706（OBJECT_HAS_LEASE）**，而「探测即续命」意味着刚被
  `batch_is_exist` 碰过的 key 在 10 秒内删不掉。写清理工具要用
  `remove_by_regex(pattern, force=True)` 或等 10 秒。四步实验确证见记忆库。
- **vLLM safetensors 加载器单流读**：280 GB 模型 90 分钟，其中绝大部分是白等
  （共享盘单流 71 MB/s，12 并行 490 MB/s）。本身是个可优化点。
- **`preferred_segment` / `preferred_segments` 在 0.3.12.post1 被忽略**：实测把落位钉到具名
  远端 segment，对象照样落在本机。所以客户端无法请求落位，只能读回落位后拒绝采样。
  要做任何依赖落位的实验（本机 vs 跨机、热点复制）都会撞上这条。
- **`dynamic_replication` 热点自动扇出**是 HEAD 才有的（`#3389`），我们这版没有。
  手动版 `replica_num=2` 在 M4 里测不出读带宽收益，但那是因为瓶颈在读端，
  **所以「要不要 backport」这个问题还没被回答**——需要先做 C3。

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

# 5. master 上限（合成负载，不占 GPU、不经过 vLLM）
export LD_LIBRARY_PATH=/dashscope/caches/workspace/llx/mcstore_deps
python3 master_stress.py --seed-keys 20000 --batches 1,32,256,1024 --threads 1,4,16,32,64
#    多进程聚合以区分 client 上限与 master 上限：
#    N 个进程各 --single --batches 256 --threads 4，用不同 --prefix 和 --local-hostname

# 6. 分析
python3 compare_campaigns.py campaign_c8k.log campaign_c3k.log
python3 analyze_saturate.py
python3 master_load.py
python3 recover_mix.py n2          # NODE_TAG 配错时事后补算 mix
```

**`NODE_TAG` 已不需要手工对齐**：`run_policy.sh` 现在从最新的 `p0_*.log` 自动推导，
并在任一节点日志缺失时大声警告。这条曾经踩过——campaign 里写死 `n1` 而节点日志是 `_n2`，
扫到旧日志、时间窗对不上，mix 三列静默变 0%。同一天在 `saturate.sh` 的 `metrics()` 里
又踩了一次（写死 `master_n2.log`，读到上一轮的 store/驱逐数字），也已改成 `ls -t master_*.log | head -1`。

**中止后重启必须换 label**：salt 取自 label，同名重跑会命中被中止那次的残留，
第一个臂变成热启动（实测重算 4% < 8% 下界，被判据当场抓出）。用 `--exclude <tag>` 剔除。

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
| `master_stress.py` | 合成负载直打 `batch_is_exist`，不经过 vLLM；`--single` 供多进程聚合。用于测 master 的真实上限 |
| `analyze_saturate.py` | 支持 `--exclude <tag>`，并自动检查驱逐 key 数的偶性（tp2 组原子驱逐必为偶数） |
