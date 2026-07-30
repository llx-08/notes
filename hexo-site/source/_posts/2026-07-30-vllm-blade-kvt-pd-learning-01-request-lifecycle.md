---
title: "01 请求生命周期：Hybrid Connector 如何接管、等待并交还请求"
date: 2026-07-30
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 01 请求生命周期：Hybrid Connector 如何接管、等待并交还请求

## 1. 入口：请求为什么会被“吃掉”

普通请求大致是：

```python
EngineCore.add_request(request)
    -> scheduler.add_request(request)
    -> waiting
    -> running
```

当前代码在中间插入了 connector：

```python
kvconn = self.scheduler.get_kv_connector()
if kvconn:
    eaten = kvconn.on_add_req(request)
    if eaten:
        return
self.scheduler.add_request(request)
```

`eaten=True` 不是丢弃请求，而是“Hybrid Connector 暂时取得请求的调度所有权”。
此时它还没有进入普通 Scheduler 的 `waiting/running`。这样可防止 GPU 开始使用尚未
从 P 写入的 D-side KV Block。

`HybridScheduler.on_add_req()` 调用 backend 的 `get_operations(req)`，得到：

```text
load_count, save_count, load_sources, save_sources
```

在 PD 场景：

- D 请求通常是 `load_count=1`：需要从 P 得到 KV；
- P 远端 decode 请求通常是 `save_count=1`：计算后要把 KV 发给 D；
- 不需要传输时二者都是 0，请求直接进入普通 Scheduler。

Hybrid Connector 把需要接管的请求放入：

```python
self._waiting.append((req, load_count, save_count))
```

注意 `_waiting` 与 Scheduler 自己的 `waiting` 不是同一队列。

## 2. 完整状态机

![Hybrid Connector 请求状态机](/imgs/vllm-pd-request-state-machine.svg)

用一个 D 侧远端 prefill 请求说明：

```text
EngineCore.add_request
  → HybridScheduler._waiting
  → 分配 D-side KV blocks
  → HybridScheduler._loading
  → DBackend 发 RPC 并等待 P
  → mark_loaded
  → HybridScheduler._loaded
  → _step_loaded 更新 token 计数
  → sched_add_req
  → 普通 Scheduler waiting/running
  → decode
  → finish/free
```

P 侧则多一个 `_saving`：

```text
请求被 Hybrid 接管
  → _setup_save
  → 普通 Scheduler 执行 prefill
  → PReqMeta → Blade-KVT send
  → 每个 TP worker SEND_DONE
  → _saved
  → _step_saved
  → _try_teardown_save
```

同一请求可以同时存在 load/save 两类操作，因此 cleanup 使用引用计数，而不是一个
布尔值。

## 3. `_step_waiting()`：先拿到目标 Block，再启动 I/O

`HybridScheduler.step()` 每次由 EngineCore 调用：

```python
def step(self):
    kvt_done = self._step_saved()
    self._step_waiting()
    self._step_loaded()
    return combine_outputs(kvt_done, self._step_aborting())
```

`_step_waiting()` 查看队首请求并调用：

```python
kvblks = sched_allocate_slots(
    req, load_count > 0, save_count > 0, prealloc, gamma)
```

如果 Block 不足，返回 `None`，函数直接 `break`，保留队首等待下一轮。这里有两点：

1. 它不是 `await` GPU 内存，而是让本轮 Core step 结束；以后 Block 释放时再试。
2. 使用 FIFO 队首阻塞，避免后面的请求不断越过大请求，但也可能产生 head-of-line
   blocking。

分配成功后：

- save 路径调用 `_setup_save()`；
- load 路径创建 `_LoadingReq`，并把 `_on_add_req()` 投递到 scheduler uvloop；
- 无 load 的 fast path 直接放入 `_loaded`。

## 4. D load：两阶段推进

### 阶段 A：异步查找和传输

```python
async def _on_add_req(req, kvblks):
    local = req.num_computed_tokens
    rmt = await backend.async_get_num_new_matched_tokens(req, local)
    ioret = await backend.async_update_state_after_alloc(req, kvblks, rmt)
    ...
    await mark_loaded(req, ioret)
```

KVT DBackend 中，`async_update_state_after_alloc()` 会走 `_prefill_rpc()` 或
`_dash_prefill_rpc()`。数据本身不是这条 Python socket 携带的；RPC 把 D Block ID、
worker info 和 token 边界发给 P，P 随后通过 Blade-KVT 写 D GPU。

异常会被捕获进本进程内的 `IoRet.ex`：

```python
try:
    cached = await ...
except Exception as e:
    ioret.ex = e
```

这里的 `Exception` 没有通过 msgspec 跨进程序列化，因此不会触发 `IoRet` 注释里所说的
不可序列化问题。

### 阶段 B：Core 线程提交状态

`mark_loaded()` 不直接操作 Scheduler，因为它运行在 uvloop 线程：

```python
set_param(req, HB_IORET, ioret)
_q_append(self._loaded, req)
await self._cleanup(req)
```

下一次 Core step 中，`_step_loaded()` 才：

1. 从 `_loading` 删除状态；
2. 把 `ioret.n` 累加到 `num_computed_tokens`；
3. 累加 `num_external_computed_tokens`；
4. 条件满足时把已收到的完整 Block 加入 prefix cache；
5. `sched_add_req(req)`，正式进入普通 Scheduler；
6. 如果 `ioret.ex` 非空，则再排入 abort。

为什么失败请求也先 `sched_add_req()`？因为普通 Scheduler 拥有通用 finish/free 路径。
把请求重新注册后再 abort，资源清理可以沿同一条代码走，不需要 Hybrid Connector
复制一套 Scheduler 内部清理逻辑。

## 5. P save：finish 和 save 谁先到都要正确

P 请求可能先结束推理，也可能先完成 KV 发送，这是经典竞态。

### 情况一：save 先完成

```text
Blade-KVT SEND_DONE
  → PBackend _do_send_done
  → HybridScheduler _do_save_done
  → _saved
  → _step_saved 设置 _PD_SAVED=True
  → 以后 request_finished_all_groups 直接返回 kv_transfer_done
```

### 情况二：request finish 先到

`request_finished_all_groups()` 保存：

```text
_PD_FINISH_REASON
_PD_STOP_REASON
_PD_CLIENT_INDEX
```

并返回：

```python
{"kv_transfer_pending": True}
```

等 `_step_saved()` 看到发送完成后，再构造空 token 的最终
`EngineCoreOutput(kv_transfer_done=True)`。

这是一种“两事件 join”：

```text
finish_seen AND save_seen → 对外完成
```

### 为什么不 `delay_free_blocks=True`

当前实现明确不让 Scheduler 无限延迟释放。原因是如果 `mark_saved` 永远不来，永久
delay 会泄漏 Block。取而代之的是：

- Scheduler finish 时释放自己的引用；
- `_setup_save()` 额外引用仍保护 Block；
- save completion 时 `_try_teardown_save()` 释放额外引用。

因此生命周期由 refcount 精确表达，而不是靠“请求对象还在不在”猜测。

## 6. abort 在不同阶段怎么走

`_step_aborting()` 依次查：

1. `_waiting`：还未分配/启动 I/O；
2. `_loading`：正在 load；
3. 普通 Scheduler 中的 request；
4. `_saving`：正在 save。

关键设计是 load/save abort 分开：

- load 中 abort：设置 `_ABORTED`，放入 `_abortmeta_load`；
- save 中 abort：设置 `_SAVE_ABORTED`，放入 `_abortmeta_save`；
- 同一请求同时 load/save 时，可能需要两轮 abort 才分别清理；
- 标志用于去重，避免重复 abort 产生重复 metadata。

对应回归测试：

```text
tests/v1/kv_connector/unit/test_hybrid_abort_refcount_leak.py
```

它覆盖：

- stale `_ABORTED` 不能阻止后续 save abort；
- 重复 save abort 只追加一次；
- load 与 save 跨两轮 abort；
- abort + teardown 后额外 Block 引用归零。

## 7. `_cleanup_rc` 为什么必要

backend 可能同时为一个请求执行多个 source，例如 KVT + 对象存储。一个分支完成时不能
立刻 `async_cleanup(req)`，否则另一个分支还在使用的状态会被删除。

逻辑是：

```text
每启动一个 load/save branch → cleanup_rc + 1
每个 branch 完成/失败        → cleanup_rc - 1
cleanup_rc <= 0              → backend.async_cleanup(req)
```

这与 Block 的 `ref_cnt` 不同：

- cleanup rc 保护 backend 控制面状态；
- Block refcount 保护 GPU 内存页。

## 8. stop0 与最后一个 token 重算

D 得到的远端 KV 并不一定覆盖完整 prompt。为了生成第一个 logits，通常要让 D 重算
边界 token；混合模型/spec decode 还保留 `gamma+1` 个 token。

因此：

```text
P 发送到 n - (gamma + 1)
D 本地计算最后 gamma + 1
```

`_step_loaded()` 中 `will_stop0` 表示远端 load 已到计算边界，不需要再安排普通的长
prefill。它会把 `max_computed_tokens` 固定在收到的边界并加入 `_stop0`，让 backend
在下一份 metadata 中完成相应处理。

## 9. 一个具体例子

假设：

- prompt 1024 token；
- block size 16；
- D 本地 prefix 命中 160 token；
- speculative `gamma=4`；
- P/D 约定最后 `gamma+1=5` token 由 D 重算。

则大致为：

```text
D 已有：160
远端可覆盖上限：1024 - 5 = 1019
远端新增：1019 - 160 = 859 token
D load 后 num_computed_tokens：1019
D 再本地计算 5 token，得到首个 decode logits
```

实际发送会按 block/token layout 对齐；不是简单发 859 个连续 token 字节。

## 10. 自检

1. “connector 吃掉请求”与 abort request 有什么区别？
2. 为什么 `mark_loaded()` 不直接调用 Scheduler？
3. load 失败时，为什么仍要把请求放回普通 Scheduler 再 abort？
4. finish/save 竞态为什么不能只用一个 `finished` 布尔量？
5. cleanup rc 和 KV Block refcount 各保护什么？
