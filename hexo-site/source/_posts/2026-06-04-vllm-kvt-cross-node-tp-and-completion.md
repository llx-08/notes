---
title: vLLM KVT：跨机 TP、Ray 线程 bug 与 KV 传输完成通知机制
date: 2026-06-04
tags: []
---

# vLLM KVT：跨机 TP、Ray 线程 bug 与 KV 传输完成通知机制

> 基于 `/mnt/data/llx/vllm`（带 HybridConnector 分支，2026-06）与 `/mnt/data/llx/blade-kvt`
> 源码整理。是 `vllm_hybrid_connector_bypass_substep.md`（substep 机制）和
> `project_architecture_and_vllm_integration.md`（整体架构）的补充，聚焦三件事：
> ① P/D 如何注册/获取 worker_info；② 跨机（多机）TP 的支持点与坑；③ Ray
> compiled-graph 导致的「主 step 被误判成 substep」线程 bug；④ KV 数据发完后
> decode 怎么知道。

---

## 0. TL;DR

- **P/D 实例 = 一个 TP 组**，用 `inst_id = name|dprank|tpsize` 标识，`worker_info`
  是**按 tp_rank 索引的 list**。`_get_dist` 把 P 的 tp_rank 映射到 D 的 tp_rank，
  只支持 `P.tp >= D.tp` 且整除；`P.tp < D.tp` 返回 None（不支持）。
- **数据面与「几台机器」无关**（按 inst_id + tp_rank 寻址）。卡住多机 TP 的是
  **控制面**：worker→scheduler 的 RPC 原本硬编码 `127.0.0.1`。修法是让这 4 处客户端
  用 engine-core 节点真实 IP（`parallel_config.master_addr`，即 `--master-addr`），
  server 端早已 bind `0.0.0.0`。**Ray 下 `nnodes` 可能仍为 1，必须显式配
  `scheduler_ip`/`core_ip`。**
- **Ray compiled-graph 把 `execute_model` 放到独立后台线程执行**，使
  `bind_backend_metadata` 里「`mytid != _main_tid` 判主/副 step」失准 → 主 step 被
  误当 bypass substep → 触发 `_build_req_send_batch` 断言。**正确判据是
  `substepid`（主=0，substep≥1），不是线程 ID。**
- **KV 数据是 P 单边 RDMA WRITE 直写进 D 的显存块**。默认 RDMA_DIRECT 下这个 write
  对 D **静默**（不触发 OnRecvCall）；decode「知道发完了」走**控制面 RPC 应答**：
  P worker `send_done` → P scheduler → 回 `PREFILL_RESP` 给 D scheduler。

---

## 1. worker_info 的注册与获取

### 1.1 两个 rank 与 inst_id

```python
# kvtbackend.py（register_kv_caches 内，P/D 一致）
rank = get_tensor_model_parallel_rank()   # TP 组内 local rank → worker_info 的下标
worker_id = get_tp_group().rank           # 全局 rank → naming 里 worker_{id} 的 key
```

`worker_info` 是按 **tp_rank** 索引的 list；`d_inst_id = dst_inst_name|dst_dprank|dst_tpsize`。

### 1.2 D 侧注册（接收方）

每个 D worker 建好 `bladekv.KVTransferServer` 后取 `winfo = current_worker_info('server')`，
按是否有命名服务分两条路：

- **有 naming**：`naming_cli.store(f"worker_{worker_id}", winfo)`；P 的 bladekv client
  自带 `naming_url`，按 `dst_inst_name + worker_id` 自行从 naming 解析 D worker 地址。
  请求里的 `d_workers_info` 为空（`_check_kvtdinfo` 里 `if self.naming_cli(): return True`）。
- **fake naming（`fake://`，无命名服务）**：D worker 通过 RPC 把 `"{tp_rank}|{winfo}"`
  发给**本机 D scheduler**，scheduler 在 `_on_register_worker` 里填 `_workers_info[tp_rank]`，
  集齐后置位 `_workers_info_event`；之后每个请求把整份 `_workers_info` 放进
  `KVTDInfo.d_workers_info` 发给 P（`_prefill_rpc`）。

### 1.3 P 侧（发送方）

P worker 建 `KVTransferClient`，但**不上报自己的 winfo**（P 是主动发送方，注册那行是注释掉的）。
每个 P worker 用自己的 `src_tprank` 经 `_get_dist`/`_get_distinfo` 算出目标 D worker，
再 `submit_req_send2(dst_inst_name, dst_wid, ..., dst_worker_info=...)`。

### 1.4 `_get_dist` 的 P→D tp_rank 映射

```python
# kvtbackend.py _get_dist
if idst_tpsize == src_tpsize:
    return dst_inst_name, idst_dprank * idst_tpsize + src_tprank, src_tprank
if src_tpsize <= idst_tpsize:          # P.tp < D.tp：不支持
    return None
group_n = src_tpsize // idst_tpsize    # P.tp > D.tp 且整除
dst_tprank = src_tprank // group_n
return dst_inst_name, idst_dprank * idst_tpsize + dst_tprank, dst_tprank
```

C++ 侧对应 `compute_valid_ranks_pd`（`tx_stub.cpp`）：在 `num_kv_heads` 约束下挑出每个 D
subgroup 的代表 P rank（如 num_kv_heads=2, d_tp=4, p_tp=8 → valid_ranks=10101010）。

---

## 2. 跨机（多机）TP

### 2.1 结论

数据面（KV 传输本身）按 inst_id + tp_rank 寻址，**与机器数无关**，跨机 TP 不需要改传输协议。
真正卡住的是**控制面**：worker→scheduler 的 RPC 通道原本写死 `127.0.0.1`，默认 worker 与
engine-core/scheduler 同机。

### 2.2 Server 端 OK，Client 端是问题

- Scheduler 的 RPC server 早已 `asyncio.start_server(..., "0.0.0.0", port)`
  （`hybrid_connector/utils.py` `RpcServer._main`）——server 端跨机没问题。
- 需要改成「engine-core 节点真实 IP」的 4 处 client：
  1. `HybridWorker.__init__` 的 `ConnPool`（worker 回 io_done）；
  2. `HybridWorker._get_bypass_handle`（取 bypass handle）；
  3. `_set_worker_envs` 的 `BLLM_KVTRANS_SEND_DONE_ADDR`（bladekv 发送完成回调地址）；
  4. `DBackend._register_worker_rpc`（D worker 注册 winfo 到 scheduler）。

### 2.3 正确的 IP 来源：`master_addr`，不是 `data_parallel_master_ip`

- **TP 跨机**（一个实例 TP 跨多机，单 engine-core）：engine-core/driver 在
  `parallel_config.master_addr`（`--master-addr`，mp 多机分布式 master）节点上，
  这才是 KVT scheduler 所在节点。
- `data_parallel_master_ip`（= `VLLM_DP_MASTER_IP`）是 **DP master（DP rank 0）** 节点，
  只对 dp_rank 0 正确；dp_rank>0 的 engine-core 在别的节点用它就错。
- `scheduler_rpc_host()` helper 解析优先级：extra_config `core_ip`/`core_rpc_host`/
  `scheduler_ip` → `master_addr`（`nnodes>1` 时）→ `127.0.0.1`。

### 2.4 Ray / external_launcher 的坑

`parallel_config.nnodes` 是 mp 多机（`--nnodes/--node-rank/--master-addr`）那条路设置的。
**用 Ray 或 external_launcher 起多机时 `nnodes` 可能仍是 1** → `nnodes>1` 闸门不成立 →
回落 `127.0.0.1`，跨机又断。**这种情况必须在 `kv_connector_extra_config` 里显式配
`scheduler_ip`/`core_ip`。**

### 2.5 环境变量辨析

- `MASTER_ADDR`（+ `MASTER_PORT`）是 torch.distributed / vLLM 的标准约定；
  `VLLM_DP_MASTER_IP` 解析链为 `VLLM_DP_MASTER_IP` → `MASTER_ADDR` → `127.0.0.1`。
- 在线多机 DP 的 `data_parallel_master_ip` 来自 `--data-parallel-address`（mp 回落到
  `--master-addr`）；`VLLM_DP_MASTER_IP` 这个 env 只在离线/SPMD（`dp_size<=1`）分支才被读。
- `MASTER_ADDRESS` **不是任何标准变量**，全仓库无人设置，`os.getenv("MASTER_ADDRESS")`
  恒为 None——是拼写错误，应为 `MASTER_ADDR`。

---

## 3. Ray compiled-graph 线程 bug：主 step 被误判成 substep

### 3.1 现象

P 侧 `AssertionError`（`kvtbackend.py` `_build_req_send_batch`）：

```python
assert reqm.has_last_token and len(reqm.d_block_ids) > 0 and reqm.d_inst_id
```

traceback：`execute_model → bind_connector_metadata → bind_backend_metadata →
_start_send_substep → _build_req_send_batch`。**只在 Ray（跨机 TP 用 Ray）下出现，
单机 MultiprocExecutor 没事。**

### 3.2 根因

`bind_backend_metadata` / `clear_backend_metadata` 用 `mytid != self._main_tid` 来区分
主 step / bypass substep。`_main_tid` 是在 `PBackend.__init__`（WORKER）里抓的，跑在
**actor 默认线程**（worker 初始化走普通 Ray actor 调用）。

但 **Ray Compiled Graph(aDAG) 把 `execute_model` 放到一个独立后台线程执行**——
`ray_utils.py` 注释明说（还因此要重设 CUDA device）：

```cpp
// on a background thread, so we need to reset torch's current device.
```

于是 `execute_model → bind_connector_metadata → bind_backend_metadata` 跑在 compiled-graph
后台线程上，`mytid`（后台线程）**永远 ≠** `_main_tid`（actor 默认线程），分流恒为真 →
**主 step 的元数据被误当 bypass substep**，走进 `_start_send_substep`。

主 step 的 `nonfreeze_metas` 合法地含「增量发送 / 非 last-token」项（`d_block_ids` 空、
`has_last_token=False`，本应由主分支 `submit_delta_send` 处理），但 substep 路径断言每项
都必须是完整发送 → 炸。**副作用更严重**：主分支被跳过，`start_send_step`/`flush_send_step`
（C++ `start_send`/`flush_send`）从不被调用 → 即使没断言，KV 也根本发不出去。

### 3.3 修复

判据从「线程 ID」改成 meta 里的 `substepid`（主 step=0，bypass substep≥1）：

```python
# bind_backend_metadata
# - 删除 mytid = threading.get_native_id()
# - 记录 self._tls.substepid = substepid（threading.local，给无参的 clear 用）
# - 分流：if mytid != self._main_tid:  →  if substepid != 0:
# - 删除内部两处 assert mytid == self._main_tid

# clear_backend_metadata
# - if mytid != self._main_tid:  →  if getattr(self._tls, "substepid", 0) != 0:
```

用 `threading.local()` 而非普通字段：主 step（compiled-graph 线程）与 bypass substep
（worker loop 线程）可能并发，`bind`→`clear` 在同一阶段同一线程上顺序执行，thread-local
各读各的，正确且无竞争。`_main_tid` 改后成死字段。

### 3.4 为什么改后仍满足 C++ 的线程要求

`client.cpp`：主 step 接口（`submit_req_send`/`submit_delta_send`/`start_send`/`flush_send`）
**直接读写 `targets_tasks_buf_`/`reqs_`，不加锁** → 要求所有主 step 调用来自**同一个线程
（串行）**，但**不要求是初始化那个线程**（无 thread-id 断言、无 thread_local）。
`start_send_substep` 注释 `// thread safe!`，用 `coord_lock_` 保护，给 worker loop 线程并发用。

Ray 下主 step 的所有 bladekv 调用都来自 `execute_model` 流程 = 同一个 compiled-graph 后台
线程，**「同一线程串行」的无锁前提仍成立**（只是这个「主线程」从 MP 的初始化线程换成了
Ray 后台线程，而 C++ 不在乎是哪个具体线程，只在乎一致）。所以 substepid 分流既修了误判，
又天然满足 C++ 线程模型。

> 注意：bladekv C++ 内部另有自己的工作线程池（`mgr_thd_{1}`、`single_thd_`、
> `target_thdpool_`/XThreadpool "clisend"）做真正的 RDMA 发送；上面讨论的「调用线程」
> 指的是 Python 层调用 pybind 接口的线程，与 C++ 内部工作线程是两回事。

---

## 4. KV 传输完成通知：decode 怎么知道 prefill 发完了

KV 数据由 P **单边 RDMA WRITE 直写进 D 预先注册好的 KV cache 显存块**（D 的块地址/rkey 是 P
握手阶段通过 `MEM_HANDLES_REQ` 找 D 要来的）。「知道发完了」按协议分三种：

### 4.1 RDMA_DIRECT（默认）：write 静默，走控制面 RPC

`WriteBySgList`/`WriteBatch` 用 `signal_peer=false`、`imm_data=0`（`rdma_channel.cpp`）——
纯 RDMA WRITE，**D 侧 CPU 收不到任何完成事件，不进 OnRecvCall**。WriteBatch 的 future 在
**P 自己这侧**（发送方 CQE）触发，只用于 P 知道自己 write 完了。

D 的 `OnRecvCall` 只处理控制 RPC（`MEM_HANDLES_REQ`、`REMOTE_CRC_REQ`），数据到达通知
已不走这里（注释：*"No longer need to handle notification callbacks - using event-based
mechanism now"*）。完成通知走控制面：

```
P worker（do_send: register_data → 逐层 send_data → flush）
   └─ 对 reach_last_token 的请求 send_done → rpc_send_done（TCP, SEND_DONE_REQ）
        └─ 目标 = env_send_done_addr() = BLLM_KVTRANS_SEND_DONE_ADDR = P 自己的 scheduler
              └─ P scheduler _on_send_done → try_advance → 标记 _SendingReq future 完成
                    └─ 回 PREFILL_RESP 给 D scheduler（D 当初 _prefill_rpc await 的应答）
                          └─ D scheduler 得知 prefill 完成 → 放行 decode
```

关键：`send_done` 是 P worker 通知 **P 自己的 scheduler**，不是直接通知 D。D 的 GPU worker
**不需要被网络层信号唤醒**（数据早已被单边写进它的 HBM 块），由 scheduler 编排何时解码——
这就是注释说的 "event-based mechanism"。

### 4.2 RDMA_STAGED：WRITE_WITH_IMM → OnImmRecvCall

`WriteSingleWithImm` 用 `signal_peer=true` + `imm_data`（`rdma_staged_channel.cpp`），
**会在 D 侧产生 recv 完成** → `RDMAStagedServer::...::OnImmRecvCall(imm_data)`，`imm_data`
里编了 staging buffer id，D 据此把数据从中转 buffer 拷进 KV cache。

### 4.3 TCP：SEND → OnRecvCall

数据本身以 SEND 消息发出，`TCPServer::CtxCallback::OnRecvCall` 被触发并携带 KV 数据——
这里 OnRecvCall 就是数据通路。

### 4.4 一句话

> `write_batch` 在 D 侧触发 `OnRecvCall` 只在 **STAGED/TCP** 成立；默认 **RDMA_DIRECT**
> 下 write 静默，D 通过 **P-scheduler 的 RPC 应答** 得知完成。

---

## 5. 关键源码索引

| 主题 | 位置 |
|---|---|
| P→D tp_rank 映射 | `vllm .../kvtbackend.py` `_get_dist` / `_get_distinfo` |
| D worker 注册 | `kvtbackend.py` `DBackend._register_worker_rpc` / `_on_register_worker` |
| 主/副 step 分流（含 bug） | `kvtbackend.py` `bind_backend_metadata` / `clear_backend_metadata` |
| substep 发送断言 | `kvtbackend.py` `_build_req_send_batch` |
| 跨机 IP helper | `hybrid_connector/__init__.py` `scheduler_rpc_host` / `engine_proxy.py` |
| scheduler RPC server bind | `hybrid_connector/utils.py` `RpcServer._main`（0.0.0.0） |
| Ray 后台线程 | `vllm/v1/executor/ray_utils.py` `execute_model_ray` |
| P 发送 + send_done | `kvtransfer/src/tx_stub.cpp` `do_task`/`do_send`/`rpc_send_done` |
| RDMA 静默 write | `kvtransfer/src/rdma_channel.cpp` `WriteBySgList`/`WriteBatch`/`OnRecvCall` |
| STAGED WRITE_WITH_IMM | `kvtransfer/src/rdma_staged_channel.cpp` `WriteSingleWithImm` |
| C++ 主/副 step 线程模型 | `kvtransfer/src/client.cpp`（`targets_tasks_buf_` 无锁 / `start_send_substep` thread safe） |
