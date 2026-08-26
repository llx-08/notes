---
title: "11 无中生有的 NaN：vLLM 全局 CUDA Stream 污染"
date: 2026-08-26
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 11 无中生有的 NaN：vLLM 全局 CUDA Stream 污染

> 本章整理 ATA 文章《无中生有的 Nan》的排查过程。问题表面上发生在新增 KVStore
> Backend 后的 model forward，最终却定位到 vLLM 对 `torch.cuda.set_stream` 的全局
> monkey patch：Connector 线程改变 stream 时污染了主线程，导致 NCCL 使用错误的
> CUDA stream。

## 1. 问题背景

Hybrid Connector 把职责分成：

```text
Connector
  - vLLM Scheduler/Worker 接入
  - 请求生命周期、线程/loop、动态扩缩、容错

Backend
  - KV Cache load/save/transfer/storage
  - 只需要理解 cache layout 与后端机制
```

同事新增了一个 Global KVStore Backend，在独立 Connector 线程中保存和加载 KV Cache。
开启该 Backend 后，主线程的 model forward 偶发产生 NaN。

这类问题容易先怀疑：

- KVStore 写坏了 KV；
- CUDA Event record 时机错误；
- Connector stream 与 model stream 缺同步；
- NCCL 或某个 kernel 自身异常。

最小复现把问题缩小成了与 KV 数据内容无关的两条线程：

```text
主线程                         Connector 线程
model forward
model forward                 with torch.cuda.stream(CONNECTOR_STREAM):
model forward                     NEW_EVENT.record()
model forward
```

只要 Connector 线程进入自己的 stream context，主线程 forward 就可能异常。

## 2. 两种表面现象

`CONNECTOR_STREAM` 的创建位置会改变症状：

| Stream 创建位置 | 现象 |
|---|---|
| 主线程创建，再交给 Connector 线程使用 | forward 偶发 NaN |
| Connector 线程自己创建 | NCCL 报 `CUDA_ERROR_INVALID_HANDLE / invalid resource handle` |

症状不同，但根因相同：主线程 NCCL 最终拿到了 Connector 的 stream。

## 3. 最关键的 GDB 证据

在 NCCL kernel launch 处检查：

```text
ncclLaunchKernel(...)
launchStream = 0x7fa1187104e0
```

再用日志查该 handle，发现它正是：

```text
CONNECTOR_STREAM = 0x7fa1187104e0
```

也就是说，Connector 线程中的：

```python
with torch.cuda.stream(CONNECTOR_STREAM):
```

竟然改变了主线程 pynccl AllReduce 使用的 stream。

![vLLM 全局 CUDA Stream 污染链路](/imgs/vllm-pd-cuda-stream-thread-pollution.svg)

## 4. 为什么原生 PyTorch Demo 复现不了

原生 PyTorch 中，CUDA current stream 按线程维护。如下两个线程应彼此独立：

```python
def thread_a():
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        ...


def thread_b():
    print(torch.cuda.current_stream())
```

当 A 进入 `s` 时，B 的：

```python
torch.cuda.current_stream()
```

仍返回 B 自己的 current stream，通常是该设备的 default stream。文章最初看到一段
`global _current_stream` 代码，但后来发现它来自 `torch/cpu/__init__.py`，不能解释
CUDA 路径。

这个失败的假设很重要：不要因为变量名相似就认定调用链；必须确认运行时实际 import
的是哪个模块、哪个 symbol。

## 5. 真正的根因：vLLM 把线程局部状态变成了进程全局状态

vLLM 中存在类似逻辑：

```python
_current_stream = None


def current_stream() -> torch.cuda.Stream:
    global _current_stream
    return _current_stream


def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    global _current_stream
    _current_stream = stream
    prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream
```

问题在于：

```text
PyTorch/CUDA current stream：线程相关状态
vLLM _current_stream：进程级 Python global
```

Connector 线程进入 `torch.cuda.stream(s)` 时，context manager 最终调用被 patch 的
`torch.cuda.set_stream(s)`：

```text
Connector thread
  → torch.cuda.set_stream(CONNECTOR_STREAM)
  → vLLM._current_stream = CONNECTOR_STREAM
```

主线程实际的 PyTorch current stream 没有改变：

```python
torch.cuda.current_stream()      # 仍是 main/default stream
```

但 vLLM 封装返回了被 Connector 覆盖的全局值：

```python
vllm.utils.current_stream()      # 错误地返回 CONNECTOR_STREAM
```

pynccl 在没有显式传入 stream 时使用：

```python
if stream is None:
    stream = current_stream()

ncclAllReduce(..., cudaStream_t(stream.cuda_stream))
```

所以 NCCL AllReduce 被提交到了 Connector stream。

## 6. 为什么一种情况是 NaN，另一种是 invalid handle

### 6.1 在主线程创建 Stream：句柄有效，但依赖关系错误

若 `CONNECTOR_STREAM` 在主线程、正确 device/context 下创建，再交给 Connector 使用，
其 handle 对主线程的 NCCL communicator 仍可能是有效的。因此不会在 launch 时立即
报错。

但 model kernels 和 NCCL collective 原本应在正确 stream 或显式同步关系下执行：

```text
model kernel produces tensor
  happens-before
NCCL AllReduce reads tensor
  happens-before
consumer kernel reads reduced tensor
```

NCCL 被偷偷换到 Connector stream 后，这些 stream-order 保证被破坏：

```text
main/model stream:       produce ───────────── consumer
connector stream:                 all-reduce
                         缺少 event/wait 依赖
```

AllReduce 可能读到尚未完成的输入，或下游 kernel 可能在 collective 完成前消费结果。
错误未必触发 CUDA API 异常，却会表现为错误数值，最终扩散成 NaN。

### 6.2 在 Connector 线程创建 Stream：device/context 也可能不匹配

CUDA current device 也是线程相关状态。vLLM worker 主线程通常已经：

```python
torch.cuda.set_device(local_rank)
```

新 Connector 线程不会自动继承主线程的 current device。若它直接：

```python
s = torch.cuda.Stream()
```

可能在默认 `cuda:0` 上创建 Stream。文章日志中可看到 rank worker 与
`<torch.cuda.Stream device=cuda:0 ...>` 的组合。

当主线程对应另一个 GPU/context，却把这个 stream handle 传给其 NCCL communicator，
CUDA driver 无法把该 handle 解释为当前 context 的合法资源，于是报：

```text
CUDA_ERROR_INVALID_HANDLE
invalid resource handle
```

需要精确区分：

- CUDA stream 不是简单的“只能由创建它的 Python 线程使用”；
- 真正决定 handle 合法性的核心是 device/context；
- Python 线程决定 current device/current stream，因此漏掉每线程初始化会间接制造
  device/context mismatch。

## 7. 完整故障链

```text
KVStore Backend 启动 Connector thread
  ↓
Connector 进入 torch.cuda.stream(CONNECTOR_STREAM)
  ↓
vLLM patched set_stream 写进进程全局 _current_stream
  ↓
Main thread 的 vllm.current_stream() 被污染
  ↓
pynccl AllReduce 使用 CONNECTOR_STREAM
  ├─ device/context 匹配，但缺同步 → 数据竞态 → NaN
  └─ device/context 不匹配        → invalid resource handle
```

因此，根因不是 KVStore 中存在 NaN，也不是 GDR/RDMA 写坏数据，而是：

> vLLM 的全局 `_current_stream` 破坏了 PyTorch/CUDA current stream 的线程局部语义。

## 8. 修复原则

### 8.1 最优：不要用单个全局变量缓存 current stream

需要当前 stream 时直接读取正确的线程/设备状态：

```python
stream = torch.cuda.current_stream(device)
```

如果某条热路径确实需要缓存，缓存至少必须：

- 按 Python thread 隔离；
- 按 CUDA device 隔离；
- 在 device/context 切换时更新；
- 不能让 Connector 的 set 操作影响主线程。

但自建缓存很容易再次偏离 PyTorch 语义，优先显式传递或调用官方接口。

### 8.2 对 NCCL 显式传递 stream

关键 collective 不应依赖一个来源含糊的全局 getter：

```python
pynccl_comm.all_reduce(tensor, stream=model_stream)
```

显式 stream 让调用者同时承担同步责任，调用链也更容易审计。

### 8.3 每个 CUDA 线程显式设置 device

任何会创建 Stream/Event、launch kernel 或调用 CUDA runtime 的后台线程都应先：

```python
torch.cuda.set_device(worker_device)
```

再创建：

```python
connector_stream = torch.cuda.Stream(device=worker_device)
connector_event = torch.cuda.Event()
```

这只能解决 device/context mismatch，不能解决全局 `_current_stream` 污染；两者必须
分别修复。

### 8.4 跨 stream 必须建立 happens-before

正确的异步流水线需要显式依赖，例如：

```python
ready = torch.cuda.Event()
ready.record(model_stream)
connector_stream.wait_event(ready)
```

反方向在消费 Connector 结果前也要 wait。不能因为两个 stream 位于同一 GPU 就假定
它们天然按程序文本顺序执行。

## 9. 建议的回归测试

### 9.1 Thread-local stream 隔离

```text
Thread A 反复进入 connector stream
Thread B 同时检查 torch.cuda.current_stream 与 vLLM current_stream
期望：B 始终得到自己的 stream
```

### 9.2 多 GPU/rank

至少覆盖：

```text
worker device = cuda:1/2/...
Connector thread 未预设 current device
Connector thread 显式 set_device
Stream 在主线程/后台线程分别创建
```

单 GPU `cuda:0` 测试可能让错误 handle 恰好合法，只暴露数值竞态而不暴露 device mismatch。

### 9.3 并发 NCCL + Backend

持续执行：

```text
model forward + NCCL AllReduce
并发 KVStore save/load Event record
数值校验、CUDA error 检查、stream handle 日志
```

不要只检查是否 crash，还要校验输出和中间 tensor 是否出现 NaN/Inf。

### 9.4 诊断日志

在问题复现时同时打印：

```text
pid / Python thread id
local rank / torch.cuda.current_device()
torch.cuda.current_stream()
vLLM current_stream()
传给 ncclAllReduce 的 cuda_stream handle
Connector stream 的 device 与 handle
```

只打印对象 repr 不够；必须把 thread、device、context/handle 的对应关系放在同一条
时间线上。

## 10. 对 Hybrid Connector 的启示

Connector/Backend 的“异步旁路、零主链路开销”依赖线程隔离，但隔离不只意味着使用
独立 thread/uvloop：

```text
Python global
CUDA current device
CUDA current stream
allocator / memory pool
NCCL communicator
logging / tracing context
```

任何隐式全局或线程局部状态都可能穿透模块边界。新增 Backend 即使只 record Event，
也可能经 monkey patch 影响主模型执行。

工程审查时应问：

1. 这个 CUDA API 依赖 current device 还是显式 device？
2. current stream 来自 PyTorch，还是项目自己缓存的全局值？
3. 后台线程是否完成了 CUDA 初始化？
4. Stream/Event 属于哪个 device/context？
5. 跨 stream 的生产者—消费者依赖由哪个 Event 建立？
6. 单 GPU 测试是否掩盖了跨 device handle 错误？

## 11. 原始文章

- [无中生有的 Nan](https://ata.atatech.org/articles/11020437654)

## 12. 自检题

1. 为什么原生 PyTorch 两线程 Demo 不会让 Thread A 改变 Thread B 的 current stream？
2. vLLM 的 `_current_stream` 为什么破坏了这一保证？
3. 为什么错误 stream handle 有时产生 NaN，有时直接 `invalid resource handle`？
4. `torch.cuda.set_device()` 能否单独修复全局 stream 污染？为什么？
5. 为什么“没有 CUDA error”不能证明跨 stream 执行顺序正确？
6. 后台 CUDA 线程至少需要显式初始化哪些状态？
