# 03 PD 控制面：rendezvous、RPC、step/substep 与 TP 映射

## 1. 这里的 rendezvous 到底是什么

Rendezvous 原意是“会合”。在 PD 系统里不是指“所有 rank 一起做 collective”的唯一
固定 API，而是泛指双方交换足够信息，最终对上同一笔传输：

```text
同一个 request_id
同一个 P/D 实例
P 的源 Block 表
D 的目标 Block 表
P rank → D rank 映射
D worker 的 endpoint / rkey 可发现信息
token 边界
```

当前 KVT 有两层 rendezvous：

1. **Scheduler rendezvous**：D Scheduler 向 P Scheduler 发 `PREFILL_REQ` 或
   `TRANSFER_KV_REQ`。
2. **Worker rendezvous**：P Worker 通过 naming 或 D 直接提供的 `WorkerInfo` 找到
   D Worker 的 Blade-KVT server，并取得传输所需信息。

## 2. 两种 D→P 请求模式

### `_prefill_rpc`

D 构造一个新的 `EngineCoreRequest` 发给 P。P 收到后把它加入自己的 EngineCore，执行
prefill，并在 KV 完成后发送。

适合“D 决定找一个 P 从头执行该请求”的模式。

### `_dash_prefill_rpc`

D 已知 `remote_host/remote_port`，向那个 P 发送 `RKVTDInfo`。P 可能已在执行相同
request，只需把晚到的 D 传输意图与现有 P Request 匹配。

适合外部系统已经把请求同时送到 P/D，D 只需与 P 现存状态会合的模式。

控制面 wire 数据核心是：

```python
KVTDInfo(
    instid="D实例|dp_rank|tp_size",
    blkids=D目标Block IDs,
    cached_tokens=D已有token数,
    max_tokens=希望覆盖的边界,
    d_workers_info=D各worker信息,
)
```

P 的响应：

```python
KVTResp(
    code=0/404/410/500,
    cached=最终可用token数,
    computed=P当前计算进度,
)
```

## 3. 为什么 D 把 WorkerInfo 一起发给 P

真实部署不一定有一个可用的 naming service。`naming_url == "fake://"` 时：

1. 每个 D Worker 创建 `KVTransferServer`；
2. 得到 `bladekv.current_worker_info("server")`；
3. 通过 `_REGISTER_WORKER` RPC 注册给 D Scheduler；
4. D Scheduler 等齐 TP workers；
5. D 在 `KVTDInfo.d_workers_info` 中把这些信息交给 P；
6. P Worker 创建 `FakeNamingWorkerClient`，不再依赖外部 naming 查询。

`WorkerInfo` 含：

- instance/worker ID；
- engine TP size/rank；
- server address；
- Block/token byte sizes；
- layer 数和每层 Block 数；
- 支持的 protocol；
- 混合模型 layout 参数。

因此它既是 endpoint 描述，也是双方布局兼容性契约。

## 4. 当前 fake-naming 启动屏障

D Scheduler 构造函数在 fake naming 模式创建：

```python
self._workers_info_event = threading.Event()
...
self._workers_info_event.wait()
```

最后一个 worker 注册时 `set()`。这是一个真正的同步启动 barrier：所有 D worker 信息
就绪前，D Scheduler backend 初始化不会继续。

优点：

- 后续每个请求不必处理“某个 TP worker endpoint 还不知道”；
- P 不会拿到半截 `d_workers_info`。

风险：

- `Event.wait()` 当前没有 timeout；
- 任意 worker 初始化失败或注册协程永远失败，Scheduler 初始化会表现为 hang；
- worker 注册协程自身会无限重试，虽然每次读响应有 3 秒 timeout。

这属于“启动一致性强，但缺少整体 deadline”的典型权衡。

## 5. P 如何把 D 意图与 P Request 对上

P 收到 `TRANSFER_KV_REQ` 后：

```text
submit_transfer_kv
  → 创建 _SendingReq Future
  → 写入 _sending[reqid]
  → RKVTDInfo 放入 _dinfoq
  → 唤醒 EngineCore
  → await _SendingReq._fut
```

Core 线程 `_step_dinfoq()` 查现有请求：

- 找不到、且从未见过：返回 404，表示 D 可以稍后重试；
- 找不到、但 P 已 abort/timeout：返回 410，D 应停止重试；
- 请求还在计算：设置 `KVTState`，可能进入 bypass substep；
- 请求已完成：直接生成 freeze `PReqMeta`；
- D 已有 token 已覆盖边界：不用发数据，直接完成。

这里的 `_dinfoq` 是跨线程移交点：RPC coroutine 不直接读取/修改普通 Scheduler 内部
队列，避免 asyncio 线程与 Core 线程同时破坏 Scheduler 状态。

## 6. main step 与 bypass substep

### main step

普通 EngineCore 调度形成 `SchedulerOutput` 后，Hybrid Scheduler 分配：

```text
stepid = 1024, 1025, ...
substepid = 0
```

metadata 随正常 model execution 送到 P Worker。

### bypass substep

如果 D 的 rendezvous 在 P 主 step 已经构造后才到，但 P Request 本轮恰好已经调度，
等下一完整 step 会损失延迟。bypass 做：

```text
复用 parent stepid
substepid = 1, 2, ...
通过 XPUB/SUB 发给 worker
Blade-KVT start_send_substep 把任务附加到当前 Step
```

它不拥有整个主 Step，因此：

- `bypass_bind()` 只附加 nonfreeze/freeze metadata；
- `bypass_clear()` 不调用 flush；
- 主 step 的 `clear_backend_metadata()` 最终 flush。

## 7. substep 到得太早或太晚怎么办

Blade-KVT `start_send_substep()` 在 `coord_lock_` 下比较：

```text
substep.stepid < coord_step_id
  → 当前 worker 已进入后续 step
  → 不能再附加，转成独立 freeze Step

substep.stepid > coord_step_id
  → 主 step 尚未来
  → 放 pending_step_metas_

substep.stepid == coord_step_id 且 last_step_guard 存在
  → 附加到当前 Step

stepid 相同但主 Step 已 flush/guard 消失
  → 创建独立 freeze Step
```

这处理了两条独立消息通道的乱序：正常 SchedulerOutput 与 bypass PUB/SUB 并不保证谁先
到 worker。

## 8. TP rank 如何映射

P/D TP size 可能不同，`_get_dist()` 给当前 P rank 生成目标 D rank。

### P TP = D TP

```text
P0→D0, P1→D1, ...
```

### P TP > D TP

多个 P rank 汇入一个 D rank：

```text
P TP=8, D TP=4
P0,P1→D0
P2,P3→D1
P4,P5→D2
P6,P7→D3
```

具体 KV head/字节切片由 cache-shape-specific `parse_block_p_gt_d` 处理。

### P TP < D TP

一个 P rank fan-out 给多个 D rank：

```text
P TP=4, D TP=8
P0→D0,D1
P1→D2,D3
P2→D4,D5
P3→D6,D7
```

这时一个 P worker 对同一 req 会产生多个 `SEND_DONE`。`_SendingReq` 的
`signals_per_worker = D_tp / P_tp`，不能收到第一个目标完成就错误宣布整个 P rank 完成。

## 9. 为什么还要 `valid_ranks`

当 KV heads 小于 TP size 时，一些 rank 的 attention KV 可能是复制或无独立分片。
Blade-KVT `compute_valid_ranks_pd()` 选择真正需要发送的 P rank，同时兼顾：

- `num_kv_heads`；
- P/D engine TP；
- GDN/混合模型的分组规则；
- P<D 时每个 D target 必须有人服务。

它与 `_get_dist()` 分工：

- `_get_dist()` 决定当前 P rank 对应哪些 D worker；
- `valid_ranks` 决定哪些 P rank 的数据切片是有效发送源；
- `parse_block` 决定最终 byte offsets。

## 10. 连接池与 PeerManager

D 的 `PeerManager` 定期从 naming 获取 P：

- 新 P 加入：可被后续请求选中；
- P ctime 改变：视为重启，丢弃旧连接；
- P 删除：从可选集合和连接池移除。

`ConnManager` 复用 asyncio TCP 连接。当前容量主要限制池中保留的空闲连接数，并不是
严格限制所有并发 `open_connection` 的 semaphore。

## 11. 自检

1. Scheduler rendezvous 与 Blade-KVT worker connection 有什么区别？
2. 404 和 410 为什么不能合并？
3. fake naming 为什么可能卡在启动，而不是卡在第一个请求？
4. bypass substep 为什么必须复用 parent stepid？
5. P<D 时，为什么每个 P worker 需要多个完成信号？
