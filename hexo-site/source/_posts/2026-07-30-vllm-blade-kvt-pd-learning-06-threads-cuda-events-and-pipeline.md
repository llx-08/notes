---
title: "06 线程、CUDA Event 与流水线：谁被阻塞，谁继续前进"
date: 2026-07-30
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 06 线程、CUDA Event 与流水线：谁被阻塞，谁继续前进

## 1. 线程全景

![Hybrid Connector 与 Blade-KVT 线程模型](/imgs/vllm-pd-thread-model.svg)

一个 P worker 里至少涉及：

```text
model execution thread
  - launch layer kernels
  - record torch.cuda.Event
  - bind/clear connector metadata

hybridworker uvloop thread
  - RPC、bypass SUB、error done

Blade-KVT wait-layer ThreadPool
  - 等 record_signal
  - 等 CUDA Event
  - 推进 layer-ready semaphore

TargetMgr manager thread（1）
  - 查/建 target、分配 thread_hint、投递 batch

Barex clisend XThreadpool
  - KvSendStub::send_batch
  - 按 layer 等 ready、提交 channel、flush

Barex context/connection/CQ threads
  - 网络 progress、callback
```

“发送线程被阻塞”通常只指 clisend 中处理这个 target batch 的线程，不是 vLLM 主线程，
也不是所有 Barex 线程。

## 2. Event 在哪里创建和记录

Python `KVTransferClient` 初始化：

```python
self._events = [torch.cuda.Event() for _ in range(num_layers)]
for event in self._events:
    event.record()
self._event_addrs = [event.cuda_event for event in self._events]
```

初始 record 让事件对象处于已知可完成状态，地址传入 C++ Context/CUDA barrier。

每层 forward 后，PBackend：

```python
client.record_event(layer_idx, torch.cuda.current_stream())
```

内部：

```python
self._events[layer_id].record(stream)
notify_event_record(self._cur_step_id)
```

顺序非常重要：

1. `event.record(stream)` 把 Event marker 排进 CUDA stream；
2. `notify_event_record(stepid)` 只通知 C++ “第 N 个 event 已经 record”；
3. C++ wait-layer 线程才可以安全调用 barrier 等 Event。

如果先 notify 再 record，等待线程可能观察到上一次 Event 的完成状态并过早放行。

## 3. Event 标记的到底是什么

CUDA Event 捕获 `record()` 调用时该 stream 前面已经排入的工作：

```text
stream:
... attention layer L kernels
... KV write kernels
event[L]
... later kernels
```

Event complete 说明 event 前面的工作完成，不代表后面的工作完成。重复 record 同一个
Event 会更新它捕获的 stream 位置，所以 Step 协调必须保证 record/wait 次序不跨 step
混淆。

## 4. `notify_event_record(step_id)` 为什么不传 layer_id

C++：

```cpp
void KvTransferClient::notify_event_record(size_t step_id) {
  assert(targets_tasks_buf_.empty());
  if (!last_step_guard_ || last_step_guard_->step_id() != step_id)
    throw ...
  last_step_guard_->after_record_one();
}
```

`after_record_one()` 只是：

```cpp
record_signal_.release();  // ready_++
```

它靠以下不变量隐式确定 layer：

- Python 按 layer 0,1,2... 的顺序调用 `record_event`；
- 每调用一次就 release 一次；
- wait-layer 线程也按 layer 0,1,2... 消费计数；
- `step_id` 防止事件被记到错误 Step。

它不是“任意层乱序完成”的位图。如果未来某模型按非单调 layer 顺序 record，当前协议
需要扩展为显式 layer ID/bitmap。

## 5. 两级 `SyncSemaphore`

### `record_signal_`

语义：有多少个 layer Event 已经被 CPU record 到 CUDA stream。

```text
Python record layer 0 → ready=1
Python record layer 1 → ready=2
```

wait-layer 线程：

```cpp
record_signal_.wait(layer_i);
```

`wait(cond)` 条件是 `ready_ > cond`，所以：

```text
wait(0) 需要 ready>=1：第 0 层已 record
wait(1) 需要 ready>=2：第 1 层已 record
```

### `Step.data_signal_`

语义：多少层的数据已经真正可由发送线程读取。

wait-layer：

```cpp
cu_barrier_->wait(layer_i);       // 等 GPU Event complete
step_->notify_layer_ready(layer_i + 1);
```

发送线程：

```cpp
step->wait_layer_ready(layer_i);  // 也要求 ready > layer_i
```

所以 layer 0 要等 `data_signal.ready >= 1`。

## 6. CPU 是 busy-wait 还是睡眠

两段等待要区分：

1. `SyncSemaphore::wait()` 使用 `std::condition_variable`，未满足时 CPU 线程睡眠。
2. `cu_barrier_->wait(layer)` 最终等待 CUDA Event。当前 `torch.cuda.Event()` 默认
   `blocking=False`，CUDA 对默认 Event 的 host synchronize 可以采用 active wait，
   而不是 `cudaEventBlockingSync` 的睡眠等待。

因此准确表述是：

> wait-layer CPU 线程不能越过 CUDA Event 等待；它可能在 CUDA runtime 内 active
> wait。vLLM model execution 主线程不在这里等待，仍可继续 launch 后续 layer。

不能简单说“GPU 中断唤醒整个主进程”，也不能说“Event 完全不阻塞 CPU”。它阻塞的是
调用 synchronize 的那个 wait-layer 线程。

## 7. `start_send_step()` 的时序

主 step：

```text
P Worker bind metadata
  → submit request/delta tasks
  → start_send_step
      ├─ 创建 Step + StepGuard
      ├─ wait-layer pool 开始 StepGuard::wait_layers
      ├─ 当前 buffered StepTasks 交给 TargetMgr
      └─ 接纳可能早到的 pending substep
```

`StepGuard::wait_layers()`：

```cpp
notify_layer_ready(0);  // 初始化，不会放行 layer 0
for layer_i:
    record_signal.wait(layer_i)
    cuda_barrier.wait(layer_i)
    notify_layer_ready(layer_i + 1)
```

`notify_layer_ready(0)` 是幂等初始化/检查，不代表第 0 层已 ready。因为发送线程
`wait_layer_ready(0)` 要求 `ready_ > 0`。

## 8. `flush_send_step()` 为什么会提前放行所有 layer

它调用：

```cpp
last_step_guard_->layer_ready_all();
```

即 `record_signal_.release(num_layers)`，让 wait-layer 线程不再等待“record 通知数量”。
但它仍会对每一层执行 CUDA barrier，因此不会仅凭 release 就越过未完成的 GPU Event。

Python 在 flush 前用最后一个 Event 的 `query()` 做告警：

```text
last event 未 ready
→ 可能启用了 async scheduling，但集成不完整
```

告警不是强制失败，而且它只能发现“最近一次 record 的 Event 还没完成”。Event 对象
会跨 step 重复使用；如果本 step 漏掉了某层 `record_event()`，barrier 可能观察到上个
step 已经完成的旧 Event，`query()` 也可能返回 ready。因此真正的不变量是：

```text
本 step 每个待发送 layer 都必须先完成一次新的 event.record(stream)
  → 再 notify_event_record
  → wait-layer 才能等待这一代 Event
```

C++ CUDA barrier 能防止“本 step 已正确 record、但 GPU 尚未完成”的过早发送，不能
单独发现“本 step 根本没有 record”的集成错误。生产监控最好额外校验每 step 的 record
计数恰好等于 `num_layers`。

## 9. 流水线时间线

```text
时间 →

model thread:
launch L0 ─ record E0 ─ launch L1 ─ record E1 ─ launch L2 ─ record E2

GPU stream:
compute L0 ─ E0 ─ compute L1 ─ E1 ─ compute L2 ─ E2

wait-layer:
wait rec0 ─ wait E0 ─ release data0
                   └ wait rec1 ─ wait E1 ─ release data1

send thread:
wait data0 ─ submit/send L0 ─ wait data1 ─ submit/send L1 ─ flush
```

流水线不是三个阶段都“永不阻塞”：

- wait-layer 会阻塞在 record_signal/CUDA Event；
- send thread 会阻塞在 data_signal；
- channel flush 会阻塞到 completion；
- model thread 通常不等网络，因此可以继续。

## 10. 多 target 并发时的 backpressure

同一 Step shared_ptr 被多个 target batch 引用。某个慢 target：

- 不阻止其他 target 的 send thread 消费 ready layer；
- 会延长自己的 `flush()`；
- 延长 Step 的最后引用寿命和源 Block 的 save 引用；
- PBackend 请求级完成要等所有预期 target/TP 信号，因此最终 KVTResp 仍被最慢 target
  限制。

这是请求级 barrier，而不是每层全局 barrier。

## 11. 线程池大小的影响

- `env_waitlayer_tpsize()` 太小：多个 step 的 Event wait 排队，网络重叠变差；
- send target pool 太小：不同 target 的 batch 排队；
- 每 target 批次长时间 `future.get()`：占住 clisend worker；
- 连接/CQ progress 线程太少：callback 延迟；
- 太多线程：CPU active wait、上下文切换和 NUMA 访问增加。

调优时应分别看：

```text
SubQueueUs
WaitLayerQueueUs
ForwardExecUs
WaitUs
WaitAndSendUs
SendNonoverlapUs
```

## 12. 自检

1. `notify_event_record()` 为什么必须在 `event.record()` 之后？
2. `record_signal_` 与 `data_signal_` 分别表示什么？
3. `wait(layer_i)` 为什么判断 `ready_ > layer_i`？
4. 默认 Event 不使用 blocking-sync，究竟阻塞了谁？
5. `layer_ready_all()` 在什么前提下不会导致未完成 GPU 数据被发送？如果本 step 漏
   record，为什么旧 Event 可能掩盖错误？
