# 01 · Store 架构与源码

本章是读源码的产物。版本信息先说清楚，否则读错代码：

```
我们跑的      0.3.12.post1 (git: 6041a60, 2026-07-25)
              查法：strings /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master
                    | grep -E "^v?0\.3\."
源码 clone     ecs ~/codes/Mooncake，HEAD 是 0.3.13 (26dd3ed9)
差距          268 commits, 334 files, +61,244 / −15,622 行
```

**读我们那一版必须用 `git show 6041a609:mooncake-store/src/master_service.cpp`**
（8,956 行），别直接读 HEAD（13,262 行）。我第一次就读错了，把 HEAD 里 `#3118` 之后才有的
`compact_frontier_prebypass` 当成了我们的实现。

## 1. KVS 与 KVT 不在同一层

| 维度 | Mooncake Store（KVS） | blade-kvt（KVT） |
|---|---|---|
| 职责 | 分布式 KV Cache 存储 | 实例间实时 KV 传输 |
| 数据寻址 | block hash（内容寻址） | request + 源/目标实例 + 显式 src/dst block ID |
| 生命周期 | 跨请求、跨轮次、跨实例保留，直到淘汰 | 一次 P→D 传输完成即结束 |
| 容量扩展 | 是，增加集群级 DRAM 容量 | 否，只搬运已有 KV |
| 前缀查询 | 能按 hash 查最长命中前缀 | 不做内容寻址式 lookup |
| 是否要求 P/D 同时在线 | 不要求，可时间上解耦 | 要求双方实时协调 |

所以两者是**叠加而非替代**。想要跨轮次高命中率，不能用 Mooncake 替换 blade-kvt，
而是「保留 blade-kvt 做低延迟 P→D 直传，再加 Mooncake KVS 做跨请求共享」。
内部 vLLM 的 `backend="kvs+kvt"` 就是这个组合，与社区
`MultiConnector(MooncakeConnector + MooncakeStoreConnector)` 是同一种分层设计。

## 2. 查询路径的确切输入输出

代码：`vllm/v1/hybrid_connector/mooncake_kvsbackend.py`（父类，调度器侧逻辑）
+ `mooncake_store_kvsbackend.py`（我们基于 PyPI 标准 API 的实现）。

![KVS 查询与加载路径](../imgs/mcstore_lookup_path.svg)

### 第 1 步 `async_get_num_new_matched_tokens(req, num_computed_tokens) -> int`（`:448`）

```
输入  req.block_hashes[]    vLLM 算好的链式 block hash，每 block_size token 一个
      num_computed_tokens   本地 HBM 已命中数（必须是 block_size 整数倍）

过程  1. 短路：should_load 为假，或剩余未算 < block_size → 直接返回 0
      2. 只取 [num_computed_blocks, +num_uncomputed_blocks) 这一段 hash
      3. 每个 hash 展开成 tp_size 个 key：f"{hash_hex}_{tp_rank}"
      4. store.batch_is_exist(keys) → List[int]
      5. 按 tp_size 分组，组内全部 rank 为 1 才算命中，遇第一个不全 1 立即 break

输出  new_matched_tokens = 连续命中 block 数 × block_size
```

三条决定开销的性质：

- **只查本地未命中的那一段** → 本地命中率高会自动降低 master 压力。
  这是后面 master 扩展性分析的关键。
- **key 数 = 未命中 block 数 × tp_size**。256k 上下文 / block_size 64 / tp2 = **8,192 key/请求**。
- **前缀语义 + 全 rank 要求**：遇断即停；任一 rank 缺失整块作废（因为 KV 按 rank 切分）。

### 第 2 步 `async_update_state_after_alloc` → `async_load_kv`

顺序是**先定命中长度，再由 vLLM 分配目标 GPU block**，然后：

```python
_blocks_to_kv():  block_id   → prepare_value_for_block() → (addrs[], sizes[])
                              # 各层 view 的 base_addr + block_id × block_len，是 GPU 地址
                  block_hash → key_for_hex() → key
store.batch_get_into_multi_buffers(keys, addrs, sizes)
# 遇第一个返回码 <0 立即 break —— 前缀留洞会被当有效 KV 读出去污染输出
```

### 数据路径：一跳直达 GPU HBM

```
远端 host DRAM（别的实例的 mooncake segment）
      │  一次 RDMA READ（GPUDirect）
      ▼
本机 GPU HBM 的目标 block
```

没有中间落地。`local_buffer_size=1GB` 那个参数不在这条路上——它是给非零拷贝 API
（普通 `put`/`get`）用的暂存区，`batch_get_into_multi_buffers` /
`batch_put_from_multi_buffers` 这对零拷贝接口绕过它。

**RDMA 模式下即使数据就在本机也走 RDMA loopback，不退化成 memcpy**：

```cpp
// transfer_task.cpp
memcpy_enabled_ = engine_.isTcpOnly();     // 只有 TCP-only 环境才启用

bool canUseLocalMemcpy(endpoint) {
    return memcpy_enabled_ &&
           (isSameProcessEndpoint(endpoint, local_hostname_) || ...);
}
// 注释：Same host is not enough: two processes on the same host share an IP but
//       have distinct virtual address spaces, so a memcpy on a peer process's
//       address would segfault.
```

两个条件都很严：必须是 TCP-only 环境，且必须是**同一个进程**（同主机不够）。
这是一个已知的可优化点，也是后面判断 `local_first` 是否值得的关键输入。

### `VLLM_KVS_ON_MIN_LENGTH` 的正确取值

```python
def should_load(req):
    return not (req.num_prompt_tokens <= VLLM_KVS_ON_MIN_LENGTH + 1)
```

默认 2048。**理论下界是 `block_size + 1`**：vLLM 故意让最后一个 token 不参与 hash
（`(num_tokens - 1) // block_size`），所以 `num_prompt_tokens ≤ block_size` 时得到 0 个 hash block、
必然查不中——代码里那个 `+1` 就是这个偏移。

**但按收益应该设 `≥ 2 × block_size`**：命中 1 个 block 只省 `block_size` 个 token 的重算，
却要付 `tp_size` 个 key 的 RPC + master 侧的租约写锁，不划算。
生产的 2048 相当于「至少能省 32 个 block 才去查」。

## 3. 驱逐机制：为什么是「近似 LRU」

**仓库里有一个精确的 O(1) LRU，但它是死代码。** `mooncake-store/include/eviction_strategy.h`
里的 `LRUEvictionStrategy`（list + hashmap，`UpdateKey` 移到队头，`EvictKey` 弹队尾）
只在 `master_service.h:60` 被前向声明，**全仓库没有任何实例化，HEAD 上也一样**。

![Mooncake 内存驱逐流程](../imgs/mcstore_eviction.svg)

真实路径是 `EvictionThreadFunc` → `BatchEvict`，「近似」精确来自四处，**第一处是根本**：

1. **排序键是 `lease_timeout`，而不是一个独立的 last_access 字段。**
   TTL 窗口内被访问过的对象整体豁免（连候选都进不去）；其余对象的时间戳被 TTL 量化——
   10 秒内的重复读不会把时间戳推得更远，所以读 100 次和读 1 次排序相同。
   **注意**：`lease_timeout` 实际上仍是 last-touch 时间的单调函数（写入触摸 `+0`，
   读取触摸 `+10s`），并不是我一度以为的「与访问时间脱钩」。详见 §5 末尾那条订正。
2. 批量按分位点砍，周期之间不维护任何顺序。
3. 每 shard 独立加锁 + `RandomIndex` 随机起点 → 普查是**非原子快照**。
4. 软钉是第二层池，形成两级优先而非单一序。

### `ExistKey` / `BatchExistKey` 会授予租约 —— 「探测即续命」

`master_service.cpp:2772`（`ExistKey`）和 `:2797`（`BatchExistKey`）都调
`GrantLeaseForGroup` / `GrantReadLease(default_kv_lease_ttl_)`，默认 TTL **10000 ms**
（`master.cpp:33` 有 `static_assert(DEFAULT_DEFAULT_KV_LEASE_TTL == 10000)` 钉死）。

- **好消息**：lookup → load 的窗口有 10 秒保护，不需要额外防护。
- **坏消息**：每次前缀探测都给对象延寿、豁免驱逐。高并发下把可驱逐池压小，
  反过来让驱逐去砍「刚写进去还没被探过」的键。

这个机制在饱和实验里的表现出乎意料——它**起了保护作用**，见 [05 章](05_eviction_and_migration.md)。

### 可用的调优 flag（不改代码）

```
-eviction_high_watermark_ratio    默认 0.90
-eviction_ratio                   一次驱逐的对象比例
-allow_evict_soft_pinned_objects  默认 true
-memory_allocator                 cachelib | offset
-allocation_strategy              random(默认) | free_ratio_first | cxl
                                  | ssd_free_ratio_first | local_first
```

**最后一条特别重要**：默认 `random` 意味着一次 put 落在本机 segment 的概率只有 1/7，
**6/7 的读是跨机 RDMA**。我们整个实验都跑在这个默认值上。

## 4. master 架构与扩展性

- **一个集群一个 master，每个 TP rank 一个 client**（7 节点 × 2 rank = 14 clients）
- 职责：**只管元数据**——key→replica→(segment, offset, size, status)、PutStart 分配、租约、
  驱逐决策、segment mount/unmount、client 健康（每 client 1 Ping/s）、etcd leader 选举、
  snapshot / SSD offload / 多租户配额。**不在数据面**，client↔client 直连 RDMA。
- 并发：`kNumShards = 1024` 个 metadata shard 各自独立读写锁，驱逐普查 16 线程。
- **不支持横向扩展**：grep 过 `master_shard` / `multi_master` / `consistent_hash` 全无；
  HA 是 etcd leader/follower，一个时刻只有一个 leader 在服务——解决可用性不解决吞吐；
  客户端配置也只接受单个 `master_server_addr`。

### 实测压力与外推

```
峰值 ExistKey：23.7 batch-req/s → 5,758 keys/s，约 228 key/batch
```

关键性质是 lookup 只查未命中段，所以本地命中 90% 时 master 只看到约 10% 的 key
（228 个，而非 46k 上下文对应的 1,437 个）。**高本地命中率会自动保护 master。**

| 场景 | key/请求 | 100 实例外推 |
|---|---|---|
| agentic（1.5 req/s，本地命中 90%） | 228 | 65k keys/s — 毫无压力 |
| 同命中率 + 吞吐型（50 req/s/实例） | 228 | 2.2M keys/s — 吃紧 |
| **冷缓存（本地命中 ≈ 0）** | 1,437 | **14M keys/s — 必然过载** |

所以单 master 的风险不在实例数，而在**「低本地命中 × 高吞吐」的组合**——
也就是冷启动或缓存被打爆的时刻，而那恰好是最需要 store 的时刻。这个耦合值得记住。

### 降压的三个旋钮（按杠杆排序）

1. **`block_size` 64 → 512**：metadata key 直接砍 8 倍（2,656 → 332）。代价是最小命中粒度变粗。
2. **跨 tp rank 去重（最值得做，零代价）**：当前 key 格式
   `{prefix}@{model}@tp_rank:{r}@...@{hash}` 把 rank 编进 key，tp8 的压力是 tp2 的 4 倍。
   但这在语义上冗余——`async_get_num_new_matched_tokens` 的逻辑就是「全 rank 都为 1 才算命中」，
   即一个 block 的各 rank 分片要么都在要么都不在。改成 key 不带 rank、value 里存各 rank 地址，
   直接砍 `tp_size` 倍，**无精度也无命中率代价**。
3. `VLLM_KVS_ON_MIN_LENGTH`：只影响短 prompt，长上下文场景没用。

### 别混淆两种 key

```
调度器侧 prefix hash key（几何网格）   85k 上下文 → 29 个     纯本地字典查
mooncake 侧 metadata key（block hash） 85k 上下文 → 2,656 个   每个走 master RPC + 授租约
```

差 92 倍，而压力大的是后者。几何网格解决的是**调度器自己**的 CPU 和内存，
**完全不影响 master 压力**——那由 `block_size` 和 `tp_size` 决定。

## 5. tp>1 的分片粒度与 group 绑定

### 每个 rank 各存一份，不按 head 维合并

key 格式（`mooncake_store_data.py:47`）把 rank 编在里面：

```
{cache_prefix}@{model}@tp_rank:{r}@pcp0@dcp0@pp_rank:0@group:{kv_group}@{block_hash}
                       ↑ 每个 rank 是一个完全独立的 object
```

每个 TP rank 是独立进程、有自己的 store client，各自 save 自己那半个 head 维分片。
（注意 key 里那个 `@group:{kv_group}@` 是 **vLLM 的 kv_cache_group**，区分 attn / mamba，
只是字符串的一部分，**与 mooncake 的 grouping 机制无关**。）

而 lookup 要求**全 rank 命中**：

```python
if not all(x == 1 for x in all_tp_ret):   # 任一 rank 缺失
    break                                  # 整块作废，后续全部放弃
```

### 两个窗口的性质完全不同

| | put 窗口 | 驱逐切开 |
|---|---|---|
| 成因 | rank0 已 PutEnd、rank1 未完成 | `nth_element` 的切点落在同一 block 的各 rank 之间 |
| 持续 | 毫秒级 | **永久** |
| 自愈 | 会（rank1 到达即完整） | **不会** |
| 后果 | lookup 按 miss 处理 → 重算 → **正确**，只是错过一次命中 | 孤儿占空间且永不可用 |
| 会自然老化掉吗 | — | **不会**：`BatchExistKey` 给「找到的那些 key」授租约，所以后续每次 lookup 都在给孤儿续 10s 命 |

**所以 put 不需要任何同步**（曾考虑「初始化定好 tp_size，put 时等所有 rank 完成」——
不必要，因为 `RegisterGroupMember` 是逐 key 增量的，组会随 rank 到达自然长大；
而且 barrier 要在 save 的关键路径上加跨进程 collective，有 hybridsched loop 争用风险，
更重要的是它治不了驱逐切开这个真问题）。

**需要修的是驱逐切开，而 mooncake 有现成机制**：

```cpp
// replica.h:95  ReplicateConfig
// Optional per-key routing group IDs. Empty string keeps that key
// ungrouped. Grouped keys share metadata routing, coalesced lease refresh,
// and memory eviction behavior.
std::optional<std::vector<std::string>> group_ids{};
```

`BatchEvict` 对 grouped 对象的行为：**组内有一个成员租约未过期就整组不动，要走一起走。**

### 已落地的改动

```
mooncake_store_kvsbackend.py   +30 −6
  _blocks_to_kv()  多返回 group_ids = block_hashes（不含 rank）
  _put_config()    新增，按 batch 构造带 group_ids 的 ReplicateConfig
  put 调用         传 self._put_config([group_ids[i] for i in todo])
  删除             不再使用的 self._replicate_config

test_mooncake_store_kvs.py     +42 −1
  FakeStore 记录 put_configs
  test_blocks_to_kv_returns_rank_free_group_ids
  test_put_config_group_ids_line_up_with_keys
```

**单测：24 passed**（原有 22 + 新增 2，无回归），在 test1 上跑
（`pytest 8.1.1` + `torch 2.11.0a0`，需 `--noconftest`；ecs 无 torch 跑不了）。

三个收益，都是免费的：

| 收益 | 说明 |
|---|---|
| **驱逐原子性** | 不再产生自我续命的孤儿分片 |
| **租约合并刷新** | `GrantLeaseForGroup` 一次刷整组而非 tp_size 次，**直接降低 master 的租约写锁压力**——那正是我们实测的主要负载来源 |
| **元数据路由共享** | 同组落同一个 shard，局部性更好 |

它和 §4 提的「跨 tp rank 去重」是同一问题的两种解法，但 grouping **不改 key 格式、不改 lookup 逻辑**，
只加一个 config 字段，几乎零风险，应该先做。去重能砍 key 数量（tp 倍），grouping 只合并租约刷新。

### 验证结果（2026-08-25，tag `g1`，9 臂饱和跑）

集群重启（master 重启清空 store，满足上面那个部署前提），4 台 dist-packages 同步到带 grouping
的版本，然后跑 9 臂累积饱和。**三个判据全部通过**：

| arm | 策略 | storeGB | 驱逐 | p50 | HBM | store | 重算 | damage |
|---|---|---|---|---|---|---|---|---|
| sat2_sco | score | 89→177 | 0 | 2435 | 91% | 1% | 8% | +0.0% |
| sat3_aff | affinity | 177→266 | 0 | 2444 | 91% | 2% | 7% | −1.0% |
| sat4_sco | score | 266→354 | 0 | 2449 | 92% | 1% | 7% | −1.0% |
| sat5_aff | affinity | 354→443 | 0 | 2455 | 91% | 2% | 7% | −1.0% |
| sat6_sco | score | 443→532 | 0 | 2442 | 91% | 1% | 7% | −1.0% |
| sat7_aff | affinity | 532→620 | 0 | 2397 | 90% | 2% | 7% | −1.0% |
| sat8_sco | score | 620→**614** | **3** | 2418 | 92% | 0% | 7% | −1.0% |
| sat9_aff | affinity | 614→608 | 3 | 2423 | 90% | 2% | 7% | −1.0% |

```
① group 错误          0                                   put 全部被接受
② 驱逐 key 数偶性      [51842, 51644] 全偶
   逐周期增量 6 次采样  17396 / 17240 / 17206 / 17216 / 17210 / 17218  全偶 → 偶然概率 1/64
③ 混合指标            HBM 91% / store 1.5% / 重算 7.2%     与 grouping 前的 Config A(90/2/7) 一致
   跨越水位代价        HBM ±0  store −0.5  重算 −0.2  p50 −17ms
```

**判据 ② 的逻辑**：tp2 下组原子驱逐必然成对摘除，所以驱逐 key 数**只能是偶数**。
对比证据很硬——ungrouped 那轮的累计驱逐终值是 **241,619（奇数）**，
而奇数在 grouped 语义下**不可能出现**，所以那个奇数本身就确证了当时驱逐在切开分片。
这个检查已经固化进 `analyze_saturate.py`，会自己报 `all even? YES/NO`。

**arm 1 被排除**（`--exclude sat1_aff`）：我为修一处 bug 中止过第一次启动，重启时沿用了同一个
label，于是 arm 1 命中自己上次的残留，重算只有 **4%**——低于 8% 下界，被判据当场抓出来。
这条已写成硬规则第 7 条。

### 三点如实说明

**1. 偶性是必要条件，不是充分条件。** 六个周期全偶 + ungrouped 是奇数，是很强的间接证据，
但严格说不排除巧合。而且它**只覆盖驱逐路径**。

**2. `batch_put` 的部分失败 grouping 管不了。**
```python
results = await asyncio.to_thread(self.store.batch_put_from_multi_buffers, ...)
failed = sum(1 for code in results if code < 0)
if failed:
    logger.warning(...)      # 只是 warning
```
put 是逐 key 返回码的，rank0 成功而 rank1 失败同样会留下半残 block。
grouping 保证的是「不切开已完整的组」，不是「写入时一定写全」。
所以还需要一个直接探针：在 lookup 的存在性向量里检测组内 0/1 混合，
它覆盖**任何**导致半残的原因，而且是永久 instrumentation（已写好，待部署）。

**3. 饱和态的策略对比在本轮无法判定**：每策略只有 1 个饱和臂，没有重复间噪声可比
（脚本自己报 `worst repeat spread nan → no spread to judge against`）。
但这个问题上一轮 ungrouped 饱和实验已用 n=5 答过：INDISTINGUISHABLE。
**没有机制让 grouping 改变这个结论**——它改驱逐原子性，不碰路由，两轮 store 命中率都是 1~2%。

**4. p50 比 grouping 前快 4.6%（2554.6 → 2437.1 ms），不可归因。**
超出 2% 噪声下限，但 grouping 唯一可能的加速机制是租约刷新从 tp_size 次合并成 1 次，
而 lookup RPC 只有几毫秒量级，解释不了 117ms。更可能是集群刚重启、状态更干净。
要证实得在同一次 bring-up 里做 grouped/ungrouped A/B，而其期望结果是 null——
**记成「不可归因」比硬造一个解释更诚实**。

### ⚠️ 部署顺序的坑

master 侧有这个检查（`master_service.cpp:3183`）：

```cpp
// Group membership is immutable while an object exists.
if (config.group_ids.has_value() && metadata.group_id != group_id) {
    LOG(ERROR) << "error=group_membership_is_immutable";
    return INVALID_PARAMS;
}
```

**如果 store 里已存在同名的 ungrouped 对象，带 `group_ids` 的 put 会被拒。**
我们的代码在 put 前会 `batch_is_exist` 跳过已存在的 key，所以正常路径不会撞上；
但只要发生一次 re-put 就会看到 `INVALID_PARAMS`。

**所以这个改动必须配合清空 store 部署。** `bringup_150b.sh` 每次都重启 master（= 清空），
天然满足；但**只重启 P 节点而不重启 master 的话，会出现零星 put 失败**。

### ⚠️ 一处订正：新写入的对象不是最先被驱逐的

我一度根据 `ObjectMetadata` 的 `lease_timeout()` 默认构造成 epoch，判断「刚写入的对象
排在驱逐序最前、最先被砍」。**这是错的**——漏了 `PutEnd` 会盖时间戳：

```cpp
// master_service.cpp:2905  PutEnd
metadata.GrantLease(0, default_kv_soft_pin_ttl_);
// GrantLease(ttl=0): lease_timeout = max(lease_timeout, now + 0) = now
```

所以：

```
刚写入的 key      lease_timeout = now       ← 排序上是「最新」，最后才被砍
60 秒前被探测的   lease_timeout = now − 50s  ← 更早，先被砍
```

订正之后，`lease_timeout` **其实是 last-touch 时间的单调函数**，只差一个偏移：
写入触摸给 `+0`，读取触摸给 `+10s`（读比写多 10 秒保护）。
这比 §3 原先的描述合理得多，**真正的近似只剩**：10 秒内的重复读不再推进时间戳（TTL 量化）、
批量分位点切割、非原子普查、软钉两级。

这也解释了饱和实验为什么零损伤——旧臂的键 last-touch 更早，排在前面先死。

## 6. 版本差异：ours 0.3.12.post1 vs HEAD 0.3.13

**驱逐机制一点没变**：`LRUEvictionStrategy` 仍是死代码，`BatchEvict` 仍按
`lease_timeout` + `nth_element` + soft-pin 两级。**所以我们想做的驱逐改动上游没做，仍是真缺口。**

已在我们版本里、不需要升级：

| 特性 | ours | HEAD |
|---|---|---|
| `enable_kv_events` | ✓ | ✓ |
| `promotion_on_hit` + `promotion_admission_threshold=2` | ✓ | ✓ |
| `offload_on_evict` | ✓ | ✓ |
| `allow_evict_soft_pinned_objects` | ✓ | ✓ |
| `enable_cxl` | ✓ | ✓ |
| `last_access_ns_`（真 LRU 有序索引） | ✓ | ✓ |
| `CreateMoveTask` / `CreateCopyTask` | ✓ | ✓ |

最后两条要注意。`last_access_ns_` 在 `storage_backend.h`，服务的是 **SSD/bucket 路径**：

```cpp
// LRU eviction index: ordered set of {last_access_ns_, bucket_id}.
// Maintained lazily — reads update last_access_ns_ atomically without ...
```

也就是**「真 LRU + 有序索引」在磁盘路径上已经实现并验证过了，内存路径仍是 `lease_timeout` 近似**。
这对我们有利：要改内存路径，有同仓库的现成范式可抄。

HEAD 新增、我们没有的：

- **`dynamic_replication_mode`（off / observe / enforce）** + `heat_window_seconds=10`
  + `admission_qps_threshold=0.8` + `max_memory_replicas=2`（`#3389 Add dynamic hot replica fanout`）
  —— 引入了**内存路径的 per-key 访问频率计数**，正是价值感知驱逐需要的信号，
  顺带解决了「mooncake 无热点复制」。
- 大量 snapshot / HA / oplog / 多租户配额加固。
- 三个驱逐 bugfix：`#3421` 保护未就绪的恢复副本、`#3419` 按 segment 数 gate PutStart 驱逐、
  `#3118` BatchEvict 候选选择性物化（性能）。

**是否升级：倾向不升。** 268 个 commit / 6 万行、涉及 HA/snapshot/oplog 大改，
而唯一真正想要的频率计数器可以自己在 `ObjectMetadata` 上加个字段实现
（已确认我们版本的 `ObjectMetadata` 没有 per-key 命中计数，`promotion_admission_threshold_`
只是个 MasterService 阈值成员）。且 0.3.13 还没有 PyPI wheel，得从源码编。

## 7. `kv_events`：现成能用，零 C++ 改动

我们的 binary 已经有：

```
--enable_kv_events                            RFC #1527 KV cache event publisher over ZMQ
--kv_events_bind_endpoint tcp://0.0.0.0:5557  ZMQ PUB
--kv_events_emit_object_key                   带上 mooncake object_key
--kv_events_emit_legacy_compat                兼容 vLLM/SGLang 字段名
```

master 侧发三类事件，**驱逐也发**：`PublishKvStored`(put) / `PublishKvRemoved`(delete) /
`PublishKvRemovedAfterEvict`（BatchEvict / NoFBatchEvict / DfsEviction 各路径都调）。

**但要分清两条事件流**（我一开始混为一谈过）：

```
mooncake master 的 kv_events  →  共享 store 里有什么   ← store 是共享的，不区分节点
vLLM 每个节点的 kv_events     →  该节点 HBM 里有什么    ← 这才是亲和索引要的
```

我们 90% 的命中来自各 P 节点本地 HBM 的 prefix cache，那是 vLLM 内部状态，mooncake 不知道。
好消息是两边都现成有，内部 vLLM：

```
vllm/config/kv_events.py        enable_kv_cache_events + zmq publisher
vllm/v1/core/block_pool.py:272  BlockStored   (block_hashes, parent_block_hash)
                          :445  BlockRemoved  ← 就在 block pool 的驱逐路径上
                          :552  AllBlocksCleared
```

**但当前做它的理由不是「修索引」**。用实测核一下：索引陈旧现在最多值 2 个百分点，
而噪声下限就是 2%。真正被量化过的收益是另一条——
**用事件流替掉 `batch_is_exist` 那轮 RPC**，直接命中「冷缓存下 14M keys/s 必然过载」那面墙。

## 8. master 吞吐实测与分片设计（2026-08-26）

### 8.1 为什么必须用合成负载

集群压不动 master，而且不是「压力不够大」那么简单——是结构上不可能：

```
master 容量   ~50,000 req/s（估算）
GPU 容量      7 节点 × (1 / 2.4s 每个 46k token 的 prefill) ≈ 3 req/s
                                                             ↑ 差 4 个数量级
```

**缩短 prompt 也没用**，因为 key 数与 prefill token 数同比例：

```
keys/s ≈ (GPU 实际 prefill 的 token/s) / block_size × tp_size × 未命中比例
```

prompt 变短 → req/s 上去 → 每请求 key 数同比例下降 → **keys/s 基本不变**。
所以「一个 master 能否撑 100 实例」这个问题，用 7 个实例的集群**原理上答不了**，
只能用不经过 GPU 的合成负载。脚本：`runs/mcstore/master_stress.py`。

### 8.2 单 client 矩阵

```
batch    1  单线程    15,241 keys/s   p50 0.06ms
batch  256  4 线程 1,427,653          p50 0.61   p99 0.92
batch  256 64 线程 1,472,575          p50 5.44   p99 55.88   ← 吞吐持平，p50×9，p99×60
batch 1024 64 线程 1,980,123          p50 16.08  p99 172.94
```

**每次调用的固定成本 ≈ 60 个 key**：batch1 是 0.06 ms/call，batch1024 是 1.11 ms/call
→ 每 key ≈ 1µs，固定开销 ≈ 60µs（含 RPC 往返 + `BatchExistKey` 里 `for scanned in 0..1023` 的空转）。
所以 batch 从 32 加到 256，单线程吞吐 272k → 505k。

**hit 与 miss 无可测差异。** 这订正了一个我从 2 秒冒烟测得出的错误结论（当时看到「hit 慢 40%」）：
全矩阵里 hit 有时反而更快，授租约的 SpinLock 写被 RPC 与 shard 锁的成本盖住了。

### 8.3 多进程聚合：分离 client 与 master 的上限

单 client 在 **4 个线程**就饱和，而 master 有 16 个 RPC 线程——说明瓶颈在客户端。
用多进程验证：

| 进程 | 聚合 keys/s | 最差 p99 | 相对 |
|---|---|---|---|
| 1 | 1,534,495 | 0.80 ms | 1.00× |
| 2 | 2,664,142 | 1.02 ms | 1.74× |
| 4 | 3,447,576 | 2.12 ms | 2.25× |
| 8 | 4,479,558 | 4.77 ms | **2.92×** |

```
单 client 上限  ≈ 1.5M keys/s     客户端 RPC 流水线深度
master 上限     ≈ 4.5M keys/s     8 进程已明显亚线性，p99 从 0.8ms 涨到 4.8ms（排队）
集群实测         5,758 keys/s     → master 利用率 0.13%
冷缓存 100 实例外推 14M keys/s     → 超载约 3 倍
```

**这把「百实例会不会过载」从算术外推变成了实测结论：会，约 3 倍，而且是乐观估计**——
4.5M 是 batch=256 的最优形态（真实 batch 由未命中 block 数决定，小 batch 掉一个数量级）；
真实负载还有 PutStart/PutEnd/Ping/FetchTasks 抢同一批 16 个线程；未计驱逐普查的写锁干扰。

### 8.4 四层瓶颈

| 层 | 上限 | 证据 | 位置 |
|---|---|---|---|
| ① 客户端 RPC 流水线 | 1.5M keys/s / client | 4 线程饱和，64 线程吞吐持平 | `MasterClient` 的连接池 |
| ② master 进程 | 4.5M keys/s | 8 进程只到 2.92× | 16 RPC 线程 + 1024 片锁 |
| ③ 每调用固定成本 | ≈60 key 等价 | batch1 vs batch1024 | RPC 往返 + 1024 次空转 |
| ④ 驱逐普查写锁 | ∝ key 总数 | 代码：普查用 RW 锁，查询用 RO，互斥 | 338k→2ms/周期(0.2%)；10M→60ms(6%) |

### 8.5 最便宜的解：`block_size` 64 → 512

key 数直接砍 8 倍 → 14M **降到 1.75M，低于 4.5M 上限**。零代码改动，
代价是最小命中粒度从 64 变 512 token，而这个代价可以实测。
**这条应排在 kv_events 负向过滤和多 master 分片之前。**

### 8.6 多 master 分片的设计

要分的只有元数据，**数据池完全不变**——两者本来就解耦（master 从不碰数据面）。

```
逻辑上仍是一个全局 KV 池（内容寻址，跨实例复用与去重不变）
  master A: shard 0..255    B: 256..511    C: 512..767    D: 768..1023
  client 发请求前算 owner = (hash(key) % 1024) / (1024 / num_masters)
  一次 batch 按 owner 拆成 N 个并发子请求
  数据仍在各 client 的 DRAM segment 里，与分片无关
```

**天然成立的原因**：`shard_idx = hash(key) % 1024` 已经是现成的分片函数，
分片只是加上「片 → 进程」这一层映射，纯计算、无状态、不需要一致性哈希环或目录——
因为 key 是内容寻址的，任何 client 独立算出的 owner 都相同。

要解决的四件事：

| 问题 | 难度 | 说明 |
|---|---|---|
| client 侧拆分批次 | 低 | 按 owner 分组并发发出、按原下标合并。**顺带解决瓶颈 ①**：N 个 master = N 条独立流水线 |
| segment 注册广播 | 中 | 所有 master 都要知道全部 segment，因为任何 master 都可能分配到任何 segment |
| 分配的空间竞争 | 中高 | 现在单 master 独占空间账本；分片后需要空间分区或一个轻量协调者 |
| 驱逐全局协调 | 高 | 每个 master 只看到自己 1/N 的 key，各自按水位驱逐 = N 个独立 LRU，全局最优性丢失 |

**最后一条有个优雅解**：空间也按 master 分区，则每个 master 独立管「自己的空间 + 自己的 key」，
驱逐和水位都是分区内的，不需要跨 master 协调。代价是**池化被削弱成「N 个子池」**，
空间互换性变差（某个分区先满就先驱逐，即使别的分区还空着）。

### 8.7 为什么按 key hash 分片优于按实例分片

这是同一个「拆分以降低成本」的直觉，但拆分轴不同，后果差很远：

| | 按实例分片 | 按 key hash 分片 |
|---|---|---|
| 全局命名空间 | **破坏**：跨实例复用要 N 路查询或目录 | 保留 |
| 跨实例内容去重 | 失去（共享前缀存 N 份） | 保留 |
| 容量互换 | 失去，且需要迁移策略 | 削弱成 N 个子池，但仍在池内互换 |
| 负载均衡 | **系统性倾斜**（热实例写得多） | **统计性均匀**（内容哈希天然均匀） |

最后一行是 key-hash 分片的第二个优势：不均衡只是统计波动，不会因为某个实例热而系统性倾斜。

**现实评估**：上游完全没实现（grep 过 `master_shard`/`multi_master`/`consistent_hash` 全无，
HA 只是 etcd leader/follower、不分担吞吐）。但按实测，`block_size 64→512` 就能把 14M 降到上限之下，
**所以分片是「量级再上一个台阶」时才需要的，不是当下的解**。
