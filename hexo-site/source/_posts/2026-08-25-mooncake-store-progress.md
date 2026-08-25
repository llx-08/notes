---
title: "Mooncake Store + blade-kvt 接入进展"
date: 2026-08-25
tags: []
---

# Mooncake Store + blade-kvt 接入进展

> 最后更新：2026-08-19
> 分支：`llx/kvs-mooncake-store`（ecs `~/codes/vllm`，基于 `origin/develop` bc355b97db）
> 提交：`429c6e6c69` 首版接入 · `d56f40bcd4` 地址按 (group, layer) 计算（M1）

标注约定：**[已验证]** = 读过代码或跑过实验；**[未验证]** = 推断或文档说法，需要时要复核。

---

## 1. 目标

在内部 vLLM 上**同时**开启两件事：

- **blade-kvt 的 P→D 实时传输**：当前请求的 KV 从 prefill 节点直传 decode 节点
- **Mooncake 共享 KV Store**：跨请求、跨实例复用 prefix KV

这两者不是替代关系，是分层关系。

---

## 2. 机制理解

### 2.1 分层：Mooncake Store 和 blade-kvt 不在同一层 [已验证]

最容易混淆的点：

| 组件 | 职责 | 内部对应 |
|---|---|---|
| Mooncake Store / 社区 `MooncakeStoreConnector` | KVS，跨请求跨实例的共享 KV 存储（GPU HBM ⇄ Mooncake Client 的 CPU DRAM） | `kvs` |
| 社区 `MooncakeConnector`（无 Store 后缀） | 点对点 P→D 传输 | 与 blade-kvt 同层 |
| blade-kvt | 点对点 P→D 传输 | `kvt` |

社区要同时要这两样，用 `MultiConnector` 挂两个 connector。内部的等价物是 `backend="kvs+kvt"`。

**Mooncake 不接管 vLLM 的 GPU block allocator**：vLLM 分配 GPU block，Mooncake 只把这些 buffer 注册成 RDMA MR，再 batch get/put。它自己管理的是外部 Store 的 CPU DRAM segment。

### 2.2 为什么必须做成 HybridConnector 的 KVS backend [已验证]

**MultiConnector 路线是死的**：

- 内部 `vllm/v1/core/sched/scheduler.py`（:980 / :1930 / :4493 / :4527）和 `vllm/v1/engine/core.py`（:617 / :1324）多处 `isinstance(self.connector, HybridConnector)`
- 内部 `multi_connector.py` 没有 `step` / `on_add_req` / `on_abort_req` / `has_requests` / `pending_requests`

包一层 MultiConnector，`self.connector` 就不再是 `HybridConnector`，HybridScheduler 的 step 循环没人驱动，blade-kvt 的 PD 链路直接失效。

另外：内部 vLLM 的 connector factory **没有**注册 `MooncakeStoreConnector`。`~/codes/dashllm/examples/pd_disaggregated/vllm_mooncake_store/` 那套 `MultiConnector(MooncakeConnector + MooncakeStoreConnector)` 脚本在当前内部 vLLM 上跑不起来，它对应的是另一个 engine 构建。

### 2.3 接线位置：`kvsp.py` 是唯一的路由点 [已验证]

`kvsbackend.py` 里装的是 `VineyardKVSBackend`，它和 `MooncakeKVSBackend`（`mooncake_kvsbackend.py:172`）一样直接实现 `HybridBackend`，两者**平级**，没有"从 kvsbackend 再路由到 mooncake"这一层。

```
HybridConnector  (KVConnectorBase_V1, 被 scheduler isinstance 检查)
  └─ HybridScheduler / HybridWorker
       └─ _get_backend_cls()          __init__.py:590   backend="kvs+kvt" → KVSP
            └─ KVSP (HybridBackend)   kvsp.py:31
                 ├─ _s ← 按 kvs_backend 字符串分派   ← 唯一路由点
                 │    ├─ "vineyard"       → VineyardKVSBackend       kvsbackend.py
                 │    ├─ "mooncake"       → MooncakeKVSBackend       mooncake_kvsbackend.py  (私有融合 API)
                 │    └─ "mooncake_store" → MooncakeStoreKVSBackend  mooncake_store_kvsbackend.py  ← 本次新增
                 └─ _p → PBackend (blade-kvt)  kvtbackend.py
```

`backend="kvs+kvt"` 无条件返回 `KVSP`，`_p` 恒为 `PBackend`，所以 **kvs+kvt 只用在 P 节点**。D 节点是 `backend="kvt"` + `kv_role="kv_consumer"`。

### 2.4 为什么要新写一个 backend：私有 API vs 标准 API [已验证]

现有 `kvs_backend="mooncake"` 依赖 3 个**内部私有融合调用**：

| 调用 | 位置 |
|---|---|
| `store.register_kv_caches(tensors, tp_rank)` | `mooncake_kvsbackend.py:295` |
| `store.batch_load_kv_async(block_ids, hashes, loop, futures)` | `:331` |
| `store.batch_save_kv_async(block_ids, hashes, loop, futures)` | `:430` |

在 target_d 上实测 PyPI 官方 wheel（`mooncake-transfer-engine==0.3.12.post1`）的 API 表面：

```
setup                          True
register_buffer                True
batch_is_exist                 True
batch_get_into_multi_buffers   True
batch_put_from_multi_buffers   True
put / get / is_exist           True
register_kv_caches             False   ←
batch_load_kv_async            False   ←
batch_save_kv_async            False   ←
```

这直接证明了改动的必要性：那 3 个融合 API 只存在于 PAI 内部的 mooncake 定制构建里（`mooncake_kvsbackend.py` 最近提交在 2026-08，说明内部确实有那个 wheel，但我们拿不到）。新 backend 用标准 API 重新实现同一套语义，地址计算移到 Python 侧。

### 2.5 请求处理流程 [已验证]

P 节点每个请求和 store 交互两次，一次在调度期，一次在计算后：

```
_step_waiting()                                  HybridScheduler
 ├─ sched_allocate_slots(req, load, save, ...)    engine_proxy.py:163
 │    └─ _get_computed_blocks_for_kvt()           engine_proxy.py:105
 │         └─ kv_cache_manager.get_computed_blocks(req)   ← ① 本地 HBM prefix cache 命中
 │    allocate_slots() 为剩余 token 分配 GPU block
 │    req.num_computed_tokens = num_new_local_computed_tokens
 │
 └─ _on_add_req(req, kvblks)                      __init__.py:1346
      ├─ local = req.num_computed_tokens
      ├─ rmt = await backend.async_get_num_new_matched_tokens(req, local)  ← ② store 查询
      │        （只查 HBM 未命中的那段：从 num_computed_tokens // block_size 起）
      ├─ req.num_pending_external_computed_tokens = rmt      ← 预告值
      └─ backend.async_update_state_after_alloc(req, kvblks, rmt)
           → _prepared 队列 → async_load_kv()
             → batch_get_into_multi_buffers → 直写 GPU HBM
             → yield IoRet(reqid, n=实际加载量)               ← 提交值
```

然后计算后：

```
async_save_kv_layer()   最后一层才真正发
  CUDA event 确认 GPU 写完
  → batch_is_exist 去重 → batch_put_from_multi_buffers
  同时 KVSP 并行驱动 blade-kvt PBackend 把 KV 发给 D
```

**关键点：store 不参与 P→D 这一跳。** `KVSP.async_save_kv_layer` 里 `_p`（blade-kvt，送当前请求给 D）和 `_s`（store，给未来请求留副本）是并行的两件事。D 节点完全不接 store。

### 2.6 两级查找：HBM → store，中间没有 CPU 层 [已验证]

顺序是严格串行的两级（见 2.5 的 ①②）。没有独立的 CPU 层——Mooncake store 本身就是 CPU DRAM（可能在别的机器上），所以实际是「本机 HBM → 远端 CPU DRAM」。Mooncake 自己能挂 SSD 层，但要 `enable_offload`，本次没接。

两个容易忽略的细节：

- **GPU block 是在 store 查询之前就分配好的**。步骤 ① 只按本地命中量算需要多少新 block 然后 `allocate_slots()`。store 加载是往已分配的 block 里写，这也是 `async_update_state_after_alloc` 能直接拿 `blocks.get_block_ids()` 的原因。
- **为什么按所有 tp_rank 生成 key**：每个 tp rank 存自己那份 KV 分片，key 里带 `tp_rank`。但 lookup 在 scheduler 单进程里做、load 是各 worker 各自做，所以必须确认**所有** rank 的分片都在才算命中，任一 rank 缺就 break。这也是 `external_computed` 总是 block_size 整数倍的原因。

### 2.7 `VLLM_KVS_ON_MIN_LENGTH` [已验证]

短 prompt 的开关闸门，只在 `should_save` / `should_load` 用：

```python
# mooncake_kvsbackend.py:641,649   kvsbackend.py:540,545   v6d_object_backend.py:536
if req.num_prompt_tokens <= envs.VLLM_KVS_ON_MIN_LENGTH + 1:
    return False
```

短 prompt 走一趟 store 往返（查询 + 传输 + 元数据 RPC）比直接重算贵，所以直接跳过。**不影响 blade-kvt 的 P→D 传输**，只 gate KVS。

- ecs 分支默认 **2048**（`envs.py:318`）
- target_d 那份旧 dist-packages 默认 **4096**
- 测试时设成 64，让几百 token 的 prompt 也能走 store

### 2.8 embedded vs standalone-store [已验证]

目前用的是 **embedded**。`init_mooncake()` 里每个 worker rank：

```python
per_tp_global_segment_size = global_segment_size // tp_size
store.setup(hostname, metadata_server, per_tp_global_segment_size, ...)
```

`global_segment_size > 0` 意味着**这个 vLLM 进程自己**分配那块 DRAM 并注册进池子，数据物理上住在 vLLM 进程的地址空间里。

**后果：P1 重启，它那段 segment 和上面所有对象一起消失。** 这就是验证 load 路径时我们用「再起一个 P2」而不是「重启 P1」的原因。

**standalone-store** 把容量和推理进程解耦：

- vLLM rank 设 `global_segment_size: 0` → 纯 client，只保留 `local_buffer` 做读写暂存
- 容量由独立进程持有：`dist-packages/mooncake/mooncake_client` 或 `mooncake_store_service.py`

好处：跨 vLLM 重启存活、容量独立扩、store 节点单独挂 SSD、推理进程不占那份 DRAM。

对当前实现来说这是**纯配置切换，不用改代码**——内部 `MooncakeStoreConfig` 没有 `mode` 字段，直接把 `global_segment_size` 传给 `setup()`，设 0 再另起 store 进程即可。**[未验证]** 还没实际跑过。

### 2.9 写入是 write-through，不存在「HBM 满了驱逐到 mooncake」 [已验证]

写入时机是 **prefill 期间的 `async_save_kv_layer`**，不是等 HBM 紧张才驱逐。

HBM 紧张时是 vLLM 自己的 block pool 按 LRU 淘汰（`block_pool.py:_maybe_evict_cached_block`），**connector 拿不到任何回调**——`KVConnectorBase_V1` 里没有 on_evict / on_block_freed 之类的钩子。淘汰的效果只是「本地副本没了，store 里那份还在」，下次请求从 store 拉回来。

connector 参与 block 生命周期的地方全是**反方向**的：

- `block_pool.py:505` 那个方法是 connector 报告 load 失败后，把坏 block 从 prefix cache 哈希表里踢掉（connector → pool）
- HybridConnector 的 `_track_extra_kv_blocks` / delay free、v6d 的 ref_cnt bump —— 都是**阻止** block 在异步存盘完成前被复用

`BlockRemoved` 事件（`block_pool.py:445`）只是对外发布的 KV event，给外部观测/router 用，不触发 spill。

要「HBM 满了主动往 mooncake 溢出」是另一类设计，社区 `OffloadingConnector` / `simple_cpu_offload_connector` 是那个模式。

### 2.10 Mooncake 自己的驱逐策略 [算法已读源码定论，见 `mooncake_store_scheduling_experiments.md` §7.2]

**结论先说**：仓库里的精确 LRU(`eviction_strategy.h` 的 `LRUEvictionStrategy`) 是死代码，从未实例化(HEAD 亦然)。真实路径 `EvictionThreadFunc`→`BatchEvict` 是**按 `lease_timeout` 排序的批量近似 LRU**：16 线程扫 1024 个 shard 收集候选，`nth_element` 取分位点作切点，soft-pin 为第二层池。近似来自四处，根本一条是**排序键是租约到期时刻而非最后访问时间**。详见实验文档 §7.2。

这是 **Master 侧策略**，vLLM 完全不参与。从 target_d 上装的 `mooncake_master`（0.3.12.post1）`--help` 读到的实际默认值：

| flag | 默认值 | 作用 |
|---|---|---|
| `eviction_high_watermark_ratio` | **0.90** | 内存占用超过该比例触发驱逐 |
| `eviction_ratio` | **0.05** | 每个驱逐周期干掉 5% 的对象 |
| `default_kv_lease_ttl` | **10000 ms** | 对象租约期，租约内不被驱逐 |
| `default_kv_soft_pin_ttl` | **1800000 ms**（30 min） | soft pin 保护期 |
| `allow_evict_soft_pinned_objects` | **true** | 空间紧张时允许打破 soft pin |
| `memory_allocator` | `offset`（可选 `cachelib`） | 分配器 |
| `allocation_strategy` | `random`（可选 `free_ratio_first` / `local_first` / `ssd_free_ratio_first` / `cxl`） | 新对象放哪个 segment |

即**两级保护 + 高水位批量驱逐**：`lease`（硬保护）→ `soft pin`（软保护，可破）→ 超 90% 按比例清 5%。淘汰顺序是近似 LRU **[未验证]**（Mooncake 设计文档说法，flag 层面只能确认水位和比例）。

SSD 相关（默认开着）：`enable_disk_eviction=true`、`nof_eviction_high_watermark_ratio=0.90`、`nof_eviction_ratio=0.05`；`offload_on_evict` 把 LOCAL_DISK 落盘从 `PutEnd` 推迟到驱逐时刻；`offload_force_evict` / `offload_cap_ratio=0.5` 在落盘队列打满时强制丢弃。

从 `store.so` 的字符串能确认租约确实拦驱逐：`has active lease, skipping`、`matched by regex, but has lease. Skipping`、`OBJECT_HAS_LEASE`，以及 `remove` 的 docstring「Requires lease to be expired」。API 名是 `GrantReadLease` / `GrantLeaseForGroup`。

### 2.11 支持 p2p 的 prefix cache 传递吗

要分开两个「p2p」：

**store 内部的数据传输是 p2p 的** [已验证]。Master 只管元数据（key → segment/offset/replica）和分配，不走数据面。真正的读写是 client 之间直连——本次实验里 P1 的 8GB segment embedded 在 P1 进程内，P2 读的时候是 P2 ⇄ P1 直连，master 只告诉 P2 去哪儿找。protocol 可以是 tcp（本次用的）或 rdma。

**但「P 主动把 prefix cache 推给另一个 P」不支持** [已验证]。没有任何主动推送：

- 写入方只管 put，不知道谁会读
- 读取方只在自己收到请求、调度期 `batch_is_exist` 命中时才拉
- 没有热点检测、没有按访问频率加副本、没有预取。副本策略只有静态的 `ReplicateConfig`（当前用默认值，没暴露配置）

所以是 **pull-based、内容寻址（block hash）的共享池**，不是 push-based 的 p2p 缓存同步。「避免请求都打到持有缓存的机器」靠的是「任何实例都能拉到同一份 KV」这个性质，不是主动分发。代码里的 TP replication factor 是 GQA/MLA 场景下同一 KV tensor 在不同 rank 上的布局重复，与负载均衡无关。

---

## 3. 当前进度

### 3.1 提交内容

分支 `llx/kvs-mooncake-store`，提交 `429c6e6c69`，4 files / 937 insertions。pre-commit hooks（ruff / ruff-format / mypy / typos / signoff）全过。

| 文件 | 行数 | 说明 |
|---|---|---|
| `vllm/v1/hybrid_connector/mooncake_store_data.py` | 98（新增） | 从社区 `mooncake/store/data.py` 搬 `KeyMetadata` / `PoolKey` / `ChunkedTokenDatabase` |
| `vllm/v1/hybrid_connector/mooncake_store_kvsbackend.py` | 321（新增） | `MooncakeStoreKVSBackend(MooncakeKVSBackend)`，覆写 4 个方法 |
| `vllm/v1/hybrid_connector/kvsp.py` | +5 −2 | 加 `elif kvs_backend == "mooncake_store"` 分派 |
| `tests/v1/kv_connector/unit/test_mooncake_store_kvs.py` | 513（新增） | 21 个单测 |

### 3.2 `mooncake_store_data.py` 做了什么

从 `~/codes/vllm_comm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` 搬三个类，保留 Apache-2.0 头 + adapted-from 注明：

- `KeyMetadata` — `model_name / tp_rank / pcp_rank / dcp_rank / pp_rank / group_id / cache_prefix`
- `PoolKey` — `build_prefix()` / `build_key_string()`，key 格式与社区**逐字节兼容**：
  `{prefix@}{model}@tp_rank:{r}@pcp{p}@dcp{d}@pp_rank:{pp}@group:{g}@{hash_hex}`
- `ChunkedTokenDatabase` — 只留 `key_for` / `key_for_hex` / `set_kv_caches_base_addr` / `set_block_len` / `prepare_value_for_block`

**没搬**：`prepare_values`、`process_tokens`、`_CompactChunkHashList`、`chunk_hashes_for_block_size`、`MooncakeStoreConnectorMetadata`（用不上，搬进来是死代码）。因此连 numpy 依赖都不需要，只依赖 `BlockHash`（`vllm/v1/core/kv_cache_utils.py:36`，`NewType("BlockHash", bytes)`）。

**为什么用社区 PoolKey 而不是内部的 `hash_hex + "_" + tp_rank`**：拿到 model_name 命名空间（多模型共用一个 master 不撞 key）、`cache_prefix`、以及为 hybrid 预留的 `group_id` 维度。代价是 lookup 侧 key 构造要跟着改。**新旧 key 格式不兼容，两种 kvs_backend 不要共用同一个 mooncake master。**

### 3.3 `MooncakeStoreKVSBackend` 覆写了哪 4 个方法

继承 `MooncakeKVSBackend`，其余（配置解析、scheduler 侧 meta 构建、EventPool、`get_operations`、`build_backend_meta`、`SOURCE_LABEL="kvs"`、`should_save`/`should_load`）全部沿用，所以 hybridsched 的统计 / `save_done_source` / abort 链路零改动。

| 方法 | 替换掉的父类逻辑 | 新逻辑 |
|---|---|---|
| `register_kv_caches` | `store.register_kv_caches(...)` | 每个唯一 storage 一次 `store.register_buffer(base, len)`；stride 探测识别 K/V-first（FlashAttn `(2,nblk,...)`，切 2 段）vs blocks-first（FlashInfer/MLA，1 段），产出 `kv_caches_base_addr` + `block_len` |
| `async_get_num_new_matched_tokens` | key 构造 | 用预生成的 `self._lookup_prefixes`（长度 = tp_size）+ `PoolKey.build_key_string`；`batch_is_exist` 调用形式不变 |
| `async_load_kv` | `batch_load_kv_async` + futures + gather | `batch_get_into_multi_buffers(keys, addrs, sizes)`，**遇到第一个负码即停** |
| `async_save_kv_layer` | `batch_save_kv_async` | `batch_is_exist` 去重 → `batch_put_from_multi_buffers(k, addrs, sizes, replicate_config)` |

**所有 store 调用都走 `asyncio.to_thread`**，绝不在 hybridsched loop 上同步调用（已知的 loop 争用卡点模式）。顺带把父类 lookup 里那个同步 `batch_is_exist`（直接阻塞 hybridsched loop）也改成 to_thread 了。

`async_load_kv` / `async_save_kv_layer` 因为 store 调用在函数中段，是**整体复制父类函数体再替换中间几行**，没有试图 `super()` 复用。

### 3.4 单元测试：21 个，全过

`tests/v1/kv_connector/unit/test_mooncake_store_kvs.py`。设计要点：

- `sys.modules` 里 stub 掉 `mooncake` / `mooncake.store`，测试不需要装 mooncake
- `object.__new__(MooncakeStoreKVSBackend)` 绕过 `__init__`，不需要 GPU 和 mooncake setup
- 用 CPU tensor 测布局探测
- 自己写 `_run` 装饰器包 `asyncio.run`，**不用 `@pytest.mark.asyncio`**（target_p/target_d 没装 pytest-asyncio）

覆盖：key 格式与 cache_prefix 命名空间、三种布局（FlashAttn 2 段 / FlashInfer 1 段 / MLA 1 段）的 segment 数与 region 精确铺满、地址不重叠不越界、`register_buffer` 失败即抛、lookup 要求全 tp rank 命中、lookup 跳过不足一 block 的请求、load 遇错即停、load 留一个 token 给本地算、save 去重后下标不错位、save 只在最后一层落盘、chunked prefill 不重发、**2D block_ids 归一化（save/load 各一个）+ 多 group 拒绝**。

跑法（target_d 或 target_p，ecs 没 torch）：

```bash
cd /dashscope/caches/workspace/llx/vllm
python3 -m pytest tests/v1/kv_connector/unit/test_mooncake_store_kvs.py --noconftest -q
```

### 3.5 target_d 端到端验证：通过

**拓扑**（4× GB200 aarch64，ds-049-013）：

| 端口 | GPU | 角色 | 配置 |
|---|---|---|---|
| 8000 | 0 | P1 | `backend=kvs+kvt`、`kvs_backend=mooncake_store`、`kv_role=kv_both` |
| 8001 | 1 | P2 | 同上，共享同一 master + 同一 `cache_prefix`，本地 prefix cache 冷 |
| 8100 | 2 | D | `backend=kvt`、`kv_role=kv_consumer` |
| — | — | mooncake_master | `127.0.0.1:50051` |

**结果**：

- Qwen3-0.6B：`external_computed=400`（405-token prompt）、`=768`（771-token）
- Qwen3-8B：`external_computed=736 / 896 / 912`
- KV 注册（Qwen3-8B）：`num_layers=36, num_segments=36, num_blocks=67480, block_lens=[65536]`
  校验：16 tokens × 2(K/V) × 8 kv_heads × 128 head_dim × 2 bytes = 65536 ✓
- 日志里零 store 错误（无 `mooncake batch_put failed` / `register_buffer failed`）

**正确性判据（重要）**：见 §5.4。基准必须是**同实例的本地 prefix cache 命中**，不是全量重算。

---

## 4. 环境配方

### 4.1 机器与代码位置

| 位置 | 内容 |
|---|---|
| ecs `~/codes/vllm` | 内部 vLLM，分支 `llx/kvs-mooncake-store`。**没有 torch，只能读代码 / py_compile** |
| ecs `~/codes/vllm_comm` | 社区 vLLM，`b908a21f9a`，参考实现来源 |
| target_d (`ds-049-013`) | 4× GB200 aarch64。**服务用的是 dist-packages 里的 vllm（0.11.1），不是 checkout** |
| target_p / target_d 共享 | `/dashscope/caches/workspace/llx/`（同一 NAS，两台机看到同一份文件） |
| `/dashscope/caches/workspace/llx/vllm` | 共享 checkout，`develop` 2188a0308（比 ecs 分支旧） |

**target_d 进不了 gitlab**（`gitlab.alibaba-inc.com:80` 连不上），同步代码要 git bundle 中转。但 **pypi.org 和 huggingface.co 都可达**。

### 4.2 mooncake 安装（aarch64 + cu13 的两个坑）

```bash
# 坑 1：aliyun 镜像源没有这个包，必须指定官方源
python3 -m pip download mooncake-transfer-engine --no-deps -d /tmp/mcwheel -i https://pypi.org/simple
python3 -m pip install --no-deps /tmp/mcwheel/mooncake_transfer_engine-0.3.12.post1-cp312-cp312-manylinux_2_28_aarch64.whl

# 坑 2：wheel 链 libcudart.so.12，但机器是 cu13
#   ImportError: libcudart.so.12: cannot open shared object file
# 从 nvidia-cuda-runtime-cu12 里抽出那个 .so，靠 LD_LIBRARY_PATH 挂上
# soname 不同（.12 vs .13），与 torch 的 cu13 不冲突
python3 -m pip download nvidia-cuda-runtime-cu12 --no-deps -d /tmp/cu12 -i https://pypi.org/simple
# 解出 nvidia/cuda_runtime/lib/libcudart.so.12 → /dashscope/caches/workspace/llx/mcstore_deps/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/dashscope/caches/workspace/llx/mcstore_deps
```

master 二进制在 `/usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master`，起法 `mooncake_master --port 50051`。

### 4.3 部署方式（target_d）

服务走 `vllm serve` 命令 → **dist-packages 里的 vllm**。所以：

```bash
# 2 个新文件 copy 进去
cp mooncake_store_data.py mooncake_store_kvsbackend.py \
   /usr/local/lib/python3.12/dist-packages/vllm/v1/hybrid_connector/
# kvsp.py 打 3 行补丁（原文件备份在 target_d /tmp/kvsp.py.bak）
```

dist-packages 的 vllm 比 ecs 分支旧（没有 `SOURCE_LABEL` / `OperationPlan` / `sched_discard_zero_block_ids`，`IoRet` 签名也不同），但覆写的 4 个方法不依赖这些，兼容。

### 4.4 脚本与日志

全在 `/dashscope/caches/workspace/llx/runs/mcstore/`：

| 脚本 | 用途 |
|---|---|
| `run_pd_mcstore.sh [tag]` | 起 mooncake_master + D(8100) + P1(8000) |
| `run_p2.sh [tag]` | 起 P2(8001)，共享同一 master + cache_prefix |
| `try_store.sh [n_words] [max_tokens]` | 一键演示跨实例命中（每次生成新随机 prompt） |
| `ab_test.sh [n_words] [max_tokens]` | A/B/C 三路正确性对照（见 §5.4） |

日志：`p_t5.log`（P1）、`p2_t5.log`（P2）、`d_t5.log`（D）、`master.log`。`t1`~`t4` 是历史（t1/t2 失败、t3 是 0.6B 且漏了 PYTHONHASHSEED、t4 是修好的 0.6B），**t5 是 Qwen3-8B 的当前配置**。

脚本支持 `MODEL=Qwen3-0.6B` 和 `TAG=xxx` 环境变量。

有用的 grep：

```bash
L=/dashscope/caches/workspace/llx/runs/mcstore
grep -E "Request added" $L/p2_t5.log | tail -5        # computed= 本地命中, external_computed= store 加载
grep "Registered KV caches" $L/p_t5.log               # 布局探测结果
grep -E "Mooncake store|VLLM_KVS" $L/p_t5.log         # store 连接 + KVS env
grep "HybridScheduler status" $L/p_t5.log | tail -5   # 队列深度, 每 10s 一条
```

日志里 4 条 `ERROR ... Failed to import Triton kernels`（`triton_kernels.matmul_ogs` 缺失）是这台机器预存在的噪音，与 store 无关，每次启动都有。

### 4.5 模型

target_d 上原有模型除 `qwen3-235b-a22b-256k-0717`（438G，`is_hybrid=False`）**全是 hybrid**（qwen3_next / qwen3_5 / qwen3_5_moe），会被 v1 的 assert 挡住。从 HF 下了两个非 hybrid 的小模型：

- `models/Qwen3-0.6B`（`is_hybrid=False`）
- `models/Qwen3-8B`（`is_hybrid=False`，16G）

判断是否 hybrid 不能看 config.json 的字段，要问 vLLM：

```python
from vllm.config import ModelConfig
ModelConfig(model=path).is_hybrid
```

**注意**：hybrid 模型下 attention block_size 会被强制改掉。qwen3.8-27b 实测：
```
Setting attention block size to 784 tokens to ensure that attention page size is >= mamba page size
Padding mamba page size by 0.13% to ensure that mamba page size and attention page size are exactly equal
```

### 4.6 配置样例

P 节点（`kv_role` 必须 `kv_both`，父类 `:185` 有 assert；`is_kv_producer` 仍为 True，blade-kvt P 侧逻辑不受影响）：

```json
{
  "kv_connector": "HybridConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "kvs+kvt",
    "kvs_backend": "mooncake_store",
    "cache_prefix": "mcstore-t5",
    "naming_url": "file:/dashscope/caches/workspace/llx/vllm.naming.mcstore",
    "kvt_inst_id": "prefill",
    "master_server_address": "127.0.0.1:50051",
    "metadata_server": "P2PHANDSHAKE",
    "global_segment_size": "8GB",
    "local_buffer_size": 1073741824,
    "protocol": "tcp",
    "device_name": ""
  }
}
```

D 节点不挂 KVS：

```json
{
  "kv_connector": "HybridConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "backend": "kvt",
    "naming_url": "file:/dashscope/caches/workspace/llx/vllm.naming.mcstore",
    "kvt_inst_id": "decode"
  }
}
```

必需的环境变量：`PYTHONHASHSEED=0`（见 §5.1）、`VLLM_KVS_ON_MIN_LENGTH=64`（测试用）、`LD_LIBRARY_PATH` 含 `mcstore_deps`。

调试开关（父类已有）：`VLLM_MCKVS_DISABLE_SAVE=1` / `VLLM_MCKVS_DISABLE_LOAD=1` / `VLLM_KVS_IO_TIMEOUT_SECONDS`。

---

## 5. 已定位的坑

### 5.1 `PYTHONHASHSEED` 必须固定且各实例一致 —— 最阴的一个

`kv_cache_utils.init_none_hash()` 在 `PYTHONHASHSEED` 未设时用 `NONE_HASH = BlockHash(os.urandom(32))` 作为 block-hash 链的**根**。跨进程根不同 → 所有 block hash 不同 → key 永远对不上。

**表现是 `external_computed=0` 但一条错误都没有。** 排查成本极高。

`prefix_caching_hash_algo` 默认是 `sha256`（确定性的），所以问题不在算法，只在种子。dashllm 的 mooncake 示例里那句 `PYTHONHASHSEED=0` 就是干这个的。

### 5.2 `ReqMeta.block_ids` 两条路径维度不一致

类型标注写的是 `list[int]`（`mooncake_kvsbackend.py:55`），但：

| 路径 | 来源 | 实际维度 |
|---|---|---|
| save | `sched_get_kvblk_ids()`（`engine_proxy.py:170`，返回 `list[list[int]]`） | **二维**，per kv_cache_group |
| load | `blocks.get_block_ids()[0]`（`:489`） | **一维**，已取 group 0 |

运行时报 `TypeError: unsupported operand type(s) for +: 'int' and 'list'`。加了 `_group0_block_ids()` 归一化 + 3 个回归测试。

顺带发现一个**既存 bug**：父类 `mooncake_kvsbackend.py:580` 的 `sched_get_kvblk_ids(request_id)[:num_populated_blocks]` 切的是 **group 维**。单 group 模型下无害（长度 1 的 list 切了还是它自己），hybrid 下会取前 N 个 GDN group。而且两个旧 KVS backend（vineyard / mooncake）**都没有任何 `is_hybrid` 保护**——新 backend 的 assert 目前是唯一护栏。

### 5.3 `cache_config.num_gpu_blocks` 在 worker 的 config 副本里是 None

要用 `kv_cache_config.num_blocks`（KV tensor 本来就是按它分配的，更权威）。社区连接器 assert 前者非 None 能过，是因为社区的调用时机不同。

### 5.4 正确性判据：基准必须是本地 prefix cache，不是全量重算

Qwen3-8B 上偶发（5 轮里 1 轮）出现 store 命中的输出与全量重算不一致。用 `ab_test.sh` 三路对照定位：

- **A** = P1 首次（全量计算）
- **B** = P1 二次（本地 prefix cache 命中）
- **C** = P2（store 命中）

分歧那轮的结果是 **A ≠ B 且 A ≠ C 但 B == C**。即本地 prefix cache 自己也偏离全量重算，而 store 加载的 KV 与本地缓存的 KV **逐字节等价**。

根因是复用 KV 时 prefill chunk 边界变化导致浮点累加顺序不同，在接近平票的 token 上翻转，与 store 无关。

**结论：验证 KV 通道正确性时，基准是同实例的本地 prefix cache 命中结果。拿全量重算做基准会误判。**

### 5.5 `pkill -f` 会杀掉自己的 ssh shell

`pkill -9 -f "vllm serve"` / `pkill -9 -f mooncake_master` 的模式串出现在 ssh 自己的命令行里，`-f` 匹配全命令行 → 杀掉自己（exit 255）。用 `"vllm [s]erve"` / `"mooncake_[m]aster"`。

### 5.6 `scp` 多个源文件到单个目标文件名会建目录

`scp a.py b.py host:/path/x.py` 会创建 `x.py/` 目录并把两个文件放进去。

---

## 6. 未完成的工作

### 6.1 hybrid 模型支持（多 kv_cache_group）—— 最大的一块

v1 用 `assert not self.is_hybrid` 挡住。这块的估算被修正过**三次**，记录下来避免重蹈：

| 版本 | 说法 | 为什么错 |
|---|---|---|
| 初版 | 「把 `ReqMeta.block_ids` 改二维就行」 | 维度不是问题，语义才是。GDN group 存的是 1 份 recurrent state 快照、attn group 存的是 N 个 block，两者有效性判定完全不同 |
| 二版 | 「key 里编码确切 token 数 `@tok:T`」 | 多余。v6d 用「只在 lcm 对齐边界存快照」这个约束替代了显式位置编码，block hash 本身就唯一确定位置 |
| 三版（当前） | 「按 v6d 范式移植」+ 地址计算按 view | 地址计算又被修正一次，见 §6.2 |

#### hybrid 的真实结构 [已验证]

`HybridBackend._parse_kv_cache_config()`（`__init__.py:196`）+ `handle_hybrid_blocks()`（`kvtbackend.py:190`）：

```
group 布局: [GDN group × num_gdn_layers] [indexer?] [attn group × 1] [PLE?]
```

**每个 GDN 层是一个独立 group，且每个 group 只有 1 个 block**：

```python
# kvtbackend.py handle_hybrid_blocks
if group_idx < num_gdn_layers or group_idx == ple_group_index:
    handled_blks.append(blocks[runtime_idx:runtime_idx + 1])   # GDN/PLE: 只取 1 块
else:
    handled_blks.append(grouped_blks[group_idx])                # attn: 全部 block
```

那 1 块不是「第 N 个 token 的 KV」，是**整段 prefix 跑完后的那一份 recurrent state 快照**。light 模式下常驻 `2 + num_speculative_blocks` 块，只保留最近一个快照。`runtime_idx = 1 if has_null_prefix_block else 0`（light/light_flex 模式下 `blocks[0]` 是 null block）。

Qwen3-Next 例：27 个 GDN group（各 1 层）+ 1 个 attn group（12 层）。

#### 要移植的 6 个机制（都在 `vllm/distributed/kv_transfer/kv_connector/v1/v6d_object_connector.py`）

**M1 · group 分类 + 地址计算**（:1279-1305）

```
full_attention_group_ids / mamba_group_ids   ← 按 isinstance(spec, MambaSpec) 分
group_block_sizes[i]                          ← 每 group 自己的 block_size
hash_block_size = gcd(所有 block_size)         ← vLLM 的 block_hashes 是这个粒度
lcm_block_size  = lcm(所有 block_size)         ← 唯一合法的「交集边界」粒度
```

换算用 `BlockHashListWithBlockSize(block_hashes, hash_block_size, group_block_size)`。

地址计算部分见 §6.2（已修正）。

**M2 · hybrid 下故意少算一个 token**（:1587-1592）

```python
num_hash_blocks = (request.num_tokens - 1) // self.hash_block_size
```
因为最后一个 token 的 GDN state 必须本地算出来。

**M3 · 两阶段交集 lookup**（:1566-1770）

```
阶段A  attn group: 从 num_computed // group_block_size 起连续命中
                   hit_length = min(各 attn group)

阶段B  mamba group: max = fa_hit_length // lcm * lcm
                    候选边界 = range(max, 0, -lcm)              # 从右往左
                    每 group 对所有候选边界算 key:
                        idx = (num_computed + L) // group_block_size - 1
                        key = hash_of_block[idx]                # 就是边界那块的 hash
                    一次 batch 探测全部候选

取第一个(=最大的)所有 mamba group 都命中的边界
若有 mamba group 但无一命中 → hit_length = 0
收尾 assert hit_length % lcm_block_size == 0                     # :1764
```

**注意阶段 B 的 key 就是边界那一块的 block hash，不带位置字段**——对齐约束替代了显式编码。

**M4 · mamba 存哪一块**（`_get_mamba_store_block_id`, :1536）

```
num_fixed_blocks = 2 + num_speculative_blocks        # light 常驻布局
若 len(step_block_ids) == num_fixed_blocks  → None    # 本步无新快照, 不存
否则 → step_block_ids[-1]
```
所以**不是每 step 都存**，是「这一步产生了新的对齐快照才存」。

**M5 · mamba 载到哪一块**（`_get_mamba_load_target_block_ids`, :1483）

必须写进 **runtime block** = `block_ids[0]`（且非 0，0 是 null block）。`_is_hybrid_backend` 为真时只写 runtime block，不额外写 cache block。

**M6 · ref_cnt 保护**（:1167-1215）

异步存盘期间对 mamba 的 `gpu_block_ids` 做 `block.ref_cnt += 1`，登记到 `_swap_protected_blocks`；完成后 `free_blocks()` 减回，跨线程走 `_defer_release_protected_blocks`。目的是防 light 模式的 `remove_skipped_blocks` 在 DMA 还在飞时把快照块回收掉。**这是正确性必需，不是优化。**

#### 明确不搬的部分

v6d 的 object/DMA 模型：`create` / `get` / `async_get` / `describe_kv_cache`、`_cached_objs` + `unfetched_objs` 两级暂存、`V6dSwapHandler`、tier 统计。mooncake 是 key→buffer 的 batch get/put，这些都不需要。

**也不建议把 mooncake 接成 v6d 的 manager**：`V6dObjectManager` 的 client 接口是对象/DMA 形状的（`create`/`get`/`async_get`/`exists`/`delete`/`describe_kv_cache` + `v6d.client.peers.vineyard.mmap_manager`），在 mooncake 上模拟这套语义大概比移植调度逻辑更麻烦、更脆。**[未验证]** 只做了浅层判断。

#### 还要额外解决的 4 件事

1. **`_block_pool` 引用**：v6d 通过 `set_block_pool()`（:1163）拿到。新 backend 要走 `engine_proxy._sched().kv_cache_manager.block_pool`。
2. **ReqMeta 要重构成 per-group**（每 group 一组 keys + block_ids），不是把 `block_ids` 二维化就完事——mamba group 存 1 块快照、attn group 存 N 块，语义不同。
3. **key 的 group 维度要真正用起来**：现在 `KeyMetadata.group_id` 恒为 0，要变成真实 group id，且 mamba group 与 attn group 的 key 命名空间要分开。
4. **收益要重新算**：hybrid 下 `block_size` 是 **784**（不是 16），`VLLM_KVS_ON_MIN_LENGTH=2048` 只够 2~3 个 block，`lcm_block_size` 对齐后候选边界会很稀疏。**动手前建议先估一遍命中率，可能收益不如预期。**

### 6.2 地址计算（M1）—— 已落地，提交 `d56f40bcd4`

这是 §6.1 M1 的一部分，也**同时是对当前已跑通实现的简化**。

#### 要算的是什么

mooncake 的 API 契约：

```
batch_put_from_multi_buffers(keys, addrs, sizes)
    addrs[i] = [addr0, addr1, ...]    # 第 i 个 key 对应的 GPU 地址列表
    sizes[i] = [len0, len1, ...]      # 对应字节长度
```

mooncake 就是把这些 (地址, 长度) 字节区间从 GPU 显存拷进 store，get 时拷回同样的区间。它对 KV cache 的语义一无所知。

所以要算的是：**给定一个逻辑单元（某 kv_cache_group 的第 b 个 block），它在显存里对应哪几段 (地址, 字节数)。**

#### 三条已验证的布局事实

1. **page size 严格统一**：`get_uniform_page_size()`（`kv_cache_utils.py:904`）里 `assert len(page_sizes) == 1`。
2. **`shared_by` 装的是「各 group 的第 i 层」**（`kv_cache_utils.py:1237-1244`）：
   ```python
   for i in range(group_size):                    # group_size = max(各 group 层数)
       shared_by = []
       for j in range(len(kv_cache_groups)):
           if i < len(kv_cache_groups[j].layer_names):
               shared_by.append(kv_cache_groups[j].layer_names[i])
       kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
   ```
   这些层共享同一块物理 tensor 且**故意重叠**——所有 group 共用**同一个全局 block-id 空间**，block manager 保证同一时刻一个 block id 只归一个 group 用。Qwen3-Next：27 GDN group（各 1 层）+ 1 attn group（12 层）→ `group_size = 12` → 12 块 tensor，tensor 0 被 28 层共享。
3. **mamba 的 block 步长就是一整个 page**（`gpu_model_runner.py` `_reshape_kv_cache_tensors` 的 MambaSpec 分支）：
   ```python
   storage_offset_bytes = 0
   for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
       num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
       stride = torch.empty((num_blocks, *shape)).stride()
       target_stride = (num_element_per_page, *stride[1:])     # ← block 步长 = 整个 page
       tensor = torch.as_strided(raw_tensor.view(dtype),
                                size=(num_blocks, *shape),
                                stride=target_stride,
                                storage_offset=storage_offset_bytes // dtype_size)
       storage_offset_bytes += stride[0] * dtype_size           # ← ssm 的 offset 来自 conv 的大小
   kv_caches[layer_name] = state_tensors                        # ← list, 不是 Tensor
   ```
   即 conv 和 ssm 是同一块 raw tensor 上的两个 strided view，ssm 带 storage_offset，一个 page 内布局是 `[conv | ssm | padding]`（padding 约 0.13%）。

#### 结论

```
第 b 个 block 在第 i 号 tensor slot 里的位置
    = tensors[i].base + b * page_size_bytes,  长度 page_size_bytes
```

- **mamba group**（1 层）→ 只占 slot 0 → **1 段**，一段 `page_size_bytes` 就把 conv+ssm+padding 全覆盖
- **attn group**（12 层）→ slot 0..11 → **每层 1~2 段**，取决于该后端是 blocks-first（1 段）还是 K/V-first（2 段）

**是按 tensor slot 算，不是按每一层算。** 一个 slot 上叠着 28 层（27 个 GDN group 各 1 层 + attn group 的第 0 层），但它们用不同的 block id，所以从寻址角度 slot 只是「一块 `num_blocks × page_size` 的数组」。

这也解释了 kvtbackend 为什么把整块 storage 拍平成 1-D view 注册（`kvtbackend.py:900-906`）：

```python
flat_view = torch.empty(0, dtype=..., device=...)
flat_view.set_(storage, 0, (total_elems,), (1,))
physical_tensors[last_attn] = flat_view
```
它**故意放弃 per-layer 形状**，把 layout 元信息（`conv_state_shape` / `ssm_state_shape` / `gdn_conv_elem_size` / `gdn_ssm_elem_size` / `num_gdn_layers` / `num_ple_layers` / `attn_pack_size`）作为 `_hybrid_kwargs` 单独传给 C++，让 blade-kvt 自己算——因为寻址纯粹是 `base + block_id * page_size`，跟层的 shape 无关。

#### 已落地的改动（`d56f40bcd4`）

旧实现是「遍历 `kv_caches` dict、按 `untyped_storage().data_ptr()` 去重、`region_len // num_blocks` 算 page size、stride 探测外层维」。它的问题：

- **非 hybrid 下是对的**（每层独占一块 tensor，恰好等价）
- **hybrid 下坏两处**：① 去重会把共享同一 slot 的 27 个 GDN 层全部跳过，只留第一个；② mamba 层是 `list`，`cache.element_size()` / `cache.ndim` 直接 AttributeError

改成：

```python
for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
    page_size_bytes = group.kv_cache_spec.page_size_bytes      # 来自 spec, 权威
    for layer_name in group.layer_names:                        # 按 group 的层走
        cache = kv_caches[layer_name]
        repr_tensor = cache[0] if isinstance(cache, (list, tuple)) else cache
        # register_buffer 仍按 storage 去重
        for seg_addr, seg_len in _layer_segments(cache, page_size_bytes, num_blocks):
            ...
    self._dbs[group_id].set_kv_caches_base_addr(addrs)
```

`_layer_segments()` 统一三种情况：

| 输入 | 段数 | 依据 |
|---|---|---|
| mamba `[conv, ssm]` | 1 段，`page_size_bytes` | `stride(0)` 跨整个 page，一段覆盖 conv+ssm+padding |
| attn blocks-outermost（FlashInfer/MLA） | 1 段，`page_size_bytes` | 无维度的 byte-stride 超过一个 page |
| attn K/V-first（FlashAttn `(2, nblk, ...)`） | 2 段 | K/V 维 stride 跨整个 K 区，切成两段 |

同时 `KeyMetadata.group_id` 从恒 0 变成真实 group id，同一 block hash 在不同 group 不再撞 key。

**验证**：22 个单测（新增 1 个用合成 hybrid config：2 个 GDN group + 1 个 2 层 attn group、slot 0 被 3 层共享、mamba 用真实的 `as_strided` 视图），非 hybrid 端到端回归通过——Qwen3-8B 仍是 `group=0 num_segments=36 page_size=65536 block_lens=[65536]`，`external_computed=896/912/928`，A/B/C 三轮里 `B == C` 恒成立。

**[未验证]** attn 分支是否存在「K 和 V 不在同一 page 内」的布局。`_layer_segments` 的 stride 探测保留了这个分支，单测覆盖了 FlashAttn 的 2 段情况，但没在真实 FlashAttn 后端上跑过（GB200 上跑出来是 1 段的 blocks-outermost 布局）。

### 6.3 `standalone-store` 模式

纯配置切换（`global_segment_size: 0` + 独立起 `mooncake_client` 或 `mooncake_store_service.py`），不用改代码。**没跑过。** 做这个才能验证跨 vLLM 重启的持久性。

### 6.4 `batch_is_exist` 是否授予 lease —— **已查清：会，TTL 10s**

`master_service.cpp:2772`(`ExistKey`) 和 `:2797`(`BatchExistKey`) 都调 `GrantLeaseForGroup`/`GrantReadLease(default_kv_lease_ttl_)`，默认 **10000ms**(`master.cpp:33` 有 `static_assert(DEFAULT_DEFAULT_KV_LEASE_TTL == 10000)`)。

- **好消息**：lookup→load 窗口有 10s 保护，不需要额外防护，本节原先担心的竞态不存在。
- **坏消息**：**探测即续命**。每次前缀探测都给对象延寿、豁免驱逐，高并发下把可驱逐池压小，反过来让驱逐去砍「刚写进去还没被探过」的键。

（下面是当初的待查记录，保留作背景）

**问题**：lookup（scheduler 线程，`batch_is_exist`）到 load（worker 线程，`batch_get_into_multi_buffers`）之间有真实时间差（实测约 1.1 秒：`add req` 12:57:05.832 → `Request added` 12:57:06.928）。这期间 Master 可以自主驱逐对象（超 0.90 水位触发、清 5%；或 soft pin 30 分钟到期），vLLM 拿不到通知。

**已确认**：租约确实拦驱逐（`store.so` 里有 `has active lease, skipping`、`OBJECT_HAS_LEASE`；`remove` 的 docstring 写「Requires lease to be expired」）。API 名是 `GrantReadLease` / `GrantLeaseForGroup`。

**未确认**：`batch_is_exist` 会不会授予/刷新租约。pybind docstring 只写返回值语义；抓了 `master_service.cpp` 但源码在 `ExistKey` handler 处被截断，只看到 `GrantReadLease` 在 `ReMountSegment` 和 `GrantLeaseForGroup` 里被调。要定论得完整拉一遍那个文件，grep `ExistKey` / `BatchExistKey` 有没有调 `GrantLeaseForGroup`。

**为什么优先级不高**：这个竞态**不影响正确性**，架构本身已容忍——

```
_on_add_req():
    rmt = await async_get_num_new_matched_tokens(...)     ← 只是「预告」
    req.num_pending_external_computed_tokens = rmt
    → _prepared → async_load_kv()
    → yield IoRet(reqid, n=num_loaded_token)              ← 这个 n 才是「实际」
    → mark_loaded(req, ioret)
```

lookup 返回待定值，真正提交给 scheduler 的是 load 之后 `IoRet.n` 里的实际加载量。load 遇错即停并据此算 `num_loaded_token`，所以中途被驱逐的结果是「声明 25 块、实际报 7 块」，scheduler 按 7 块推进、剩下正常重算，再加 `kv_load_failure_policy='recompute'` 兜底。

所以 lease TTL 影响的是**命中率和白做的功**，不是正确性。等真跑出命中率不达预期时再回来查。

### 6.5 其他没做的

- `kvsp.py` 的 dispatch 分支**没有运行时验证过**（target_d 的 dist-packages 里那份是手工打的补丁，逻辑上走到了；ecs 分支上的那份只做了 py_compile）。
- 社区的 disk/SSD offload、partial-tail offload、replica tier 日志、独立 metrics/prom 指标。
- 多副本 placement、cache-aware routing。
- 真实模型上量 store 命中带来的 TTFT 收益。
- GSM8K 精度对齐（当前只做了逐字节输出对比）。
- `kvs_backend="vineyard"` 与纯 `backend="kvt"` 两条老路径的回归（改动是新增分支 + 新文件，理论上零影响，但没实测）。

---

## 7. 风险

| 风险 | 说明 | 状态 |
|---|---|---|
| **hybridsched loop 争用** | store 的 C++ batch 调用是阻塞的，必须 `asyncio.to_thread`。已知的 loop 争用会导致请求卡住 | 已按 to_thread 实现；上线前要看 `kvt_dur` / lifecycle 统计有没有队列等待放大 |
| **同一块 GPU KV 内存被两库注册** | `KVSP.register_kv_caches` 先 `_s` 后 `_p`，mooncake 的 `register_buffer` 与 blade-kvt/barex 的 MR 注册并存 | NVIDIA + tcp 实测无问题 |
| **PPU 上的失效地址** | `kvtbackend.py:374` 的 `alloc_kv_cache_ppu()` 会 `cache.set_(mem_storage, ...)` **换掉底层 storage**。若它在 `_s` 注册之后执行，mooncake 会持有失效地址 | NVIDIA + tcp 路径该函数提前 return；**PPU 未验证，v1 不支持** |
| **key 格式与老 `kvs_backend="mooncake"` 不兼容** | PoolKey 格式 vs `hash_hex + "_" + tp_rank` | 别共用同一个 mooncake master |
| **失败即停的 prefix 语义** | `batch_get_into_multi_buffers` 返回逐 key 码，任一 block 失败后面的必须全丢，否则 KV 出现空洞导致输出跑题 | 已实现；有单测覆盖 |
| **embedded 模式数据不持久** | store 内存住在 vLLM 进程里，重启即丢 | 已知；要持久得上 standalone-store |
| **hybrid 静默出错** | 两个旧 KVS backend（vineyard / mooncake）都没有 `is_hybrid` 保护，且父类 `:580` 切 group 维。hybrid 下会算错但不报错 | 新 backend 有 assert 挡住；**旧 backend 的隐患仍在，值得单独修** |
| **target_d 运行时版本太旧** | dist-packages vllm（0.11.1）没有 `_ple_group_index`，ecs 分支的 hybrid 代码在那儿跑不起来（探针实测 `AttributeError`） | **阻塞 hybrid 的端到端验证**，需先升级 target_d 或换机器 |
| **hybrid 命中率可能不达预期** | hybrid 下 block_size=784，`lcm_block_size` 对齐后候选边界稀疏 | 动手前先估算 |

---

## 8. 下一步

M1（地址层）已完成。剩下的：

1. **hybrid 的 M2~M6**（§6.1）+ 4 件额外事项（`_block_pool` 引用、ReqMeta 改 per-group、mamba key 命名空间、命中率估算）。**前置**：target_d 运行时缺 `_ple_group_index`，端到端验证需先升级或换机器。建议**动手前先估命中率**——hybrid 下 block_size=784，`lcm_block_size` 对齐后候选边界很稀疏，收益可能不如预期。
2. **standalone-store**（§6.3）—— 纯配置，能验证跨重启持久性，顺带解锁「重启 P」这个更自然的验证方式。

另外两件小事：修掉 §5.2 里旧 KVS backend 的 group 维切片 + 加 `is_hybrid` 保护（hybrid 下会静默出错）；补 §6.5 的老路径回归。

**一条判断修正**：本文档早先把「地址计算修正」列为可独立先做的第一步，这个判断偏了——非 hybrid 下 `kv_cache_tensors` 是每层一个（`kv_cache_utils.py:1215` 的 `shared_by=[layer_name]`），没有共享 slot，去重永远不会丢层，所以单独做是行为上的 no-op。它只有作为 hybrid 的前置才有意义，因此最终是连着合成 hybrid config 的单测一起落的。
