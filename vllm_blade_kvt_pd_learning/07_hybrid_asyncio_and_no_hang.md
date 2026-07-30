# 07 Hybrid Connector 的 Python 协程：如何工作，如何避免静默 hang

## 1. 为什么要单独的 asyncio 线程

EngineCore 主循环包含调度和模型执行协调。如果直接在它上面：

```python
await reader.readexactly(...)
```

一个慢 P、断开的 socket 或延迟 completion 会阻塞所有请求调度。

当前实现创建两个独立 uvloop：

```text
hybridsched thread
  - Scheduler side RPC server
  - P/D control RPC
  - mark_loaded / mark_saved
  - abort RPC / PeerManager

hybridworker thread
  - Worker completion RPC
  - worker registration
  - bypass subscriber
  - backend async load/save
```

每个 loop 都在 daemon `threading.Thread` 中 `run_forever()`。EngineCore/ModelRunner 通过
`asyncio.run_coroutine_threadsafe()` 投递 coroutine。

## 2. 协程不是新的 OS 线程

同一个 uvloop 线程上多个 coroutine 的并发来自 `await`：

```python
async def transfer(req):
    writer.write(msg)
    await writer.drain()          # socket 暂不可写时让出线程
    resp = await reader.readexactly(8)  # 等响应时让出线程
    return resp
```

当 `req1` 等网络时，loop 可运行 `req2` 的 coroutine。若 coroutine 中执行长时间同步
CPU 代码或 `threading.Event.wait()`，则会阻塞整个 loop。

## 3. 跨线程提交与结果

### `run_coroutine_threadsafe`

从 Core/model thread 向 uvloop 提交：

```python
asyncio.run_coroutine_threadsafe(coro, loop)
```

它返回 `concurrent.futures.Future`。当前很多调用没有保存返回值，意味着调用方不等待
结果，错误处理必须由 coroutine 自身完成。

### `asyncio.create_task`

在 loop 内启动子任务：

```python
asyncio.create_task(coro)
```

Python 官方文档指出 event loop 只保存 Task 的弱引用；fire-and-forget task 应保留强
引用，避免执行中被 GC。

## 4. `kill_me_if_exception` 的两个设计目的

装饰器：

```python
task = asyncio.current_task()
running_task.add(task)
task.add_done_callback(running_task.discard)
```

目的 A：loop 上的 set 保留强引用，任务完成后自动删除。

目的 B：任何未处理异常：

```python
logger.exception(...)
os.abort()
```

这是一种 fail-fast：

- 优点：不会让关键 RPC loop 悄悄死掉，进程表面存活但所有请求永远等 completion；
- 缺点：一个可局部恢复的 coroutine bug 也可能杀死整个进程。

loop 的 exception handler `_loop_on_ex()` 同样 `os.abort()`，覆盖未被 await 的 task
异常。

所以这里的“不 hang”策略之一不是“永远恢复”，而是“关键后台任务异常时不要静默带病
存活”。

## 5. Core 睡眠时完成事件怎样唤醒它

Hybrid uvloop 把请求放入 `_loaded/_saved/_aborting` 后，Core 可能正在阻塞等新的输入
请求。仅修改 deque 不会自动唤醒它。

`_q_append()`：

```python
q.append(item)
if len(q) == 1:
    wakeup_core()
```

`wakeup_core()` 向 EngineCore input queue 注入：

```text
ABORT "__FAKE_REQID_FOR_HYBRID_CONNECTOR__"
```

abort 不存在 request 是 no-op，但能唤醒 Core，使下一轮调用 `kvconn.step()`。

为什么只在 0→1 时唤醒：

- 队列已非空说明之前已有一次 wakeup；
- 每追加一个 item 都注入假消息会制造 wakeup storm；
- Core 醒来后用 while 一次 drain 整个 deque。

## 6. RPC Server 如何处理连接

`RpcServer._client_main()` 为每条连接循环：

```text
read 4-byte head
  → 查 callback
  → await callback(reader, writer)
  → 继续读下一条
```

`IncompleteReadError`：

- partial 非空：记录异常；
- 正常 EOF/partial 空：关闭 writer；
- 单条 client connection 退出，不直接杀整个 server。

`_client_main` 本身受 `kill_me_if_exception` 包装，但它内部已经捕获多数 client 级异常，
避免普通对端断连升级成进程 abort。

## 7. Connection Pool 的作用与边界

worker/P/D 频繁完成 RPC 若每次 TCP handshake，延迟较大。`ConnPool/ConnManager`：

- `_acquire` 优先复用 reader/writer；
- 完成后 `_release`；
- 连接坏时重建；
- 容量限制主要控制池内保留数量。

注意：复用连接意味着半开连接可能直到下一次读写才被发现；所有 RPC 都必须定义
deadline，否则 `readexactly` 理论上可无限等。

## 8. 当前已有 timeout/retry 的位置

| 路径 | 机制 |
|---|---|
| worker 获取 bypass handle | 每次响应 3 秒 timeout，0.1 秒重试 |
| D worker 注册 | 每次响应 3 秒 timeout，0.1 秒重试 |
| Blade-KVT TCP flush | `env_rpc_timeout_s()` |
| staged RDMA flush | `env_rpc_timeout_s()` |
| RDMA mem-handle/CRC RPC | `env_rpc_timeout_s()` |
| dash P request 404 | 总延迟由 `VLLM_KVT_MAX_DELAY_MS` 生成的 backoff 限制 |
| P `_dash_done` | `VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS` |
| send_done socket | 失败后重连重试一次 |

## 9. 当前可能无界等待的位置

### 9.1 P `_wait_kvt_state`

```python
ioret = await state._fut
```

没有 `asyncio.wait_for`。任意一个必要 P TP worker 没发 `SEND_DONE`，Future 不完成。

### 9.2 `IoState.try_advance`

必须收齐：

```text
所有 tprank
× 每 worker 的 signals_per_worker
× 所有 expected source
```

没有 age/deadline。丢一个信号就永远不 ready。

### 9.3 D `_kvt_rpc`

它捕获 `IncompleteReadError` 并重试两次，但 `open_connection/drain/readexactly` 没统一
`wait_for`。明确 RST 容易返回；黑洞/半开连接可能依赖 OS TCP timeout。

### 9.4 RDMA Direct flush

```cpp
for (future : write_futs_)
    future.get();
```

没有显式 timeout。依赖 Barex/provider 最终 callback error。

### 9.5 fake naming Scheduler barrier

`threading.Event.wait()` 无 timeout，缺一个 D worker 注册就不继续。

## 10. 为什么“有 retry”仍可能 hang

retry 只有在一次尝试已经返回错误后才能发生。若第一次尝试永不返回：

```text
await readexactly() 永久 pending
→ 进入不了 except
→ retry loop 形同虚设
```

所以健壮结构应是：

```python
for attempt:
    try:
        return await asyncio.wait_for(one_attempt(), timeout=attempt_timeout)
    except ...
raise RequestScopedError
```

再在更外层设置 request 总 deadline，避免每层 retry 叠加后超出 SLO。

## 11. `asyncio.wait_for` 不是无代价

官方语义：

- timeout 时取消被等待 task；
- 等待它实际处理完 cancellation；
- 因此墙钟时间可能略超过 timeout；
- 被调 coroutine 若吞掉 `CancelledError` 或底层 C 扩展不可取消，仍可能拖延。

对 KVT 必须同时定义：

- Python task cancellation 如何关闭 writer；
- 已提交 Barex WR 是否可取消；
- D Block 中可能只有部分 KV 时如何标记无效；
- P Block extra ref 由谁释放；
- late completion 到达后如何去重。

加一行 `wait_for` 不等于完成完整超时设计。

## 12. `IoRet.ex` 的序列化陷阱

`IoRet` 是 msgspec Struct，但字段 `Exception` 不能正常 msgspec 序列化。代码注释明确
警告：若把非空 `ex` 通过 `io_done_rpc` 发出去，会导致 worker 崩溃。

当前 KVT worker 的 `send_error_done_req()` 在线上协议中编码的是整数 code=500，而不是
Python Exception；D 本地 `_prefill_rpc` 异常则只存在当前 scheduler 进程内。

设计原则：

```text
跨进程：传稳定的 error code + message + retryable
进程内：可以保留 Exception 对象和 traceback
```

## 13. 建议的“防 hang”分层

```text
每次 socket I/O timeout
  ↓
单次 transfer attempt timeout
  ↓
request KV transfer deadline
  ↓
TP/source barrier watchdog
  ↓
process health/liveness watchdog
  ↓
router 摘除与请求重试/失败
```

同时暴露 state age：

```text
oldest _waiting age
oldest _loading age
oldest _saving age
oldest _sending Future age
missing TP ranks/sources
```

只有队列长度，没有 age，很难区分健康吞吐和永久挂起。

## 14. 当前代码中的巧思

1. Scheduler 与 worker 各有独立 loop，隔离控制面 I/O。
2. 通过 deque + Core wakeup 把 Scheduler 写操作收敛到 Core 线程。
3. fire-and-forget task 强引用集合避免 task 消失。
4. 后台主任务 fail-fast，避免 silent coroutine death。
5. `IoState` 拒绝未知 request、重复 worker/source，避免重复 completion 过早推进。
6. 404/410 区分“尚未出现”与“明确消失”，终止无意义重试。
7. completion 队列仅在 0→1 时唤醒，减少假 ABORT 风暴。
8. cleanup refcount 让多个异步 backend source 可以独立结束。

## 15. 自检

1. coroutine 为什么能并发，却不能执行长时间同步阻塞？
2. `kill_me_if_exception` 为什么既保留 Task 引用又 `os.abort()`？
3. 完成消息到了，为什么还需要 `wakeup_core()`？
4. retry loop 在什么情况下完全不起作用？
5. 给 `_SendingReq` 加 timeout 时还必须解决哪些资源清理问题？
