---
title: "08. blade-kvt 调用 Barex 的完整发送逻辑"
date: 2026-07-27
categories: [NCCL、PCIe 与 Barex 学习笔记]
tags: [NCCL, PCIe, RDMA, Barex, blade-kvt, 学习笔记]
---

# 08. blade-kvt 调用 Barex 的完整发送逻辑

## 1. 总体路径

![blade-kvt 从 Python 到 Barex 的发送路径](/imgs/blade_kvt_send_path.svg)

```text
Python
  KvTransfer.submit_req_send2 / start_send_step
    ↓ pybind
  KvTransferClient
    ↓ StepTasks 按目标分组
  TargetMgr::do_submit
    ↓ target thread pool
  KvSendStub::send_batch
    ↓ parse_block
  IChannel::register_data
    ↓ layer ready
  IChannel::send_data(layer)
    ↓
  Barex Send / WriteBatch / WriteSingle
    ↓
  WR → NIC → CQ → callback/future
```

### 1.1 用一个 request 贯穿后文

假设 prefill worker 已为 request 42 计算出 KV cache：

```text
模型层数：3（为了示例；真实模型更多）
需要发送的 source blocks：[5, 8]
decode worker 分配的 target blocks：[12, 2]
每层每 block：64 KiB
```

逻辑映射：

```text
layer 0: src block 5 → dst block 12
         src block 8 → dst block 2
layer 1: src block 5 → dst block 12
         src block 8 → dst block 2
layer 2: src block 5 → dst block 12
         src block 8 → dst block 2
```

总 payload：

```text
3 layers × 2 blocks × 64 KiB = 384 KiB
```

但 blade-kvt 不一定等所有层算完再一次发 384 KiB。forward 完成 layer 0 后，
对应 CUDA event/barrier 就绪，发送线程可以先调用 `send_data(0)`；计算 layer 1
时，layer 0 的网络传输可能同时进行，这就是 compute/communication overlap。

在 direct RDMA 路径中，每个 block 最终会变成类似：

```text
local_addr = layer_base + 5 × block_stride
remote_addr = remote_layer_base + 12 × block_stride
length = 64 KiB
lkey/rkey = 该 GPU MR 的本地/远端权限
```

因此 `parse_block` 不只是“解析一个整数”，而是在把模型的 block id 转成传输层
可执行的地址、长度和权限。

## 2. Python 与 pybind

Python `blade_kvt/kv_transfer_impl.py` 导入：

- `submit_req_send2`
- `submit_delta_send`
- `start_send`
- `start_send_substep`
- `flush_send`

pybind 对应：

- `kvtransfer_pybind.cpp:218-254`：提交 request；
- `:259-270`：开始 step/substep；
- `:281-284`：flush；
- `:487-493`：模块导出。

`submit` 阶段只积累任务，不发送数据。

### 2.1 当前 blade-kvt 与 vLLM 是否在同一进程、共享 CUDA context

对本文固定的版本：

```text
vLLM       058f6b130e789096ab685c58fe90fdb97c149774
blade-kvt  752697132e8b0409ad134724fec2882c9ca57380
```

答案是：

> blade-kvt 作为 Python/C++ extension 加载在 vLLM 的 **GPU model worker
> 进程**中；它和这个 worker 内的 PyTorch/vLLM 对同一 device 使用同一个 CUDA
> primary context。它不等于运行在 vLLM scheduler/front-end 主进程，也不表示
> 所有 TP worker 共用一个进程。

代码证据是一条完整的指针传递链：

```text
vllm/v1/worker/gpu_model_runner.py
  → kv_transfer_group.register_kv_caches(kv_caches)

vllm/v1/hybrid_connector/kvtbackend.py
  → bladekv.KVTransferClient/Server(
        layers=_flatten_cache(kv_caches))

blade_kvt/kv_transfer_impl.py
  → cache.data_ptr()
  → init_kv_transfer_client/server(layer_addrs=...)

kvtransfer_pybind.cpp
  → C++ Context / Barex MR registration
```

vLLM 刚分配好的 `torch.Tensor` 被直接交给 blade-kvt，后者把 `data_ptr()` 的数值
传入 C++，没有经过 CUDA IPC handle、Unix socket 或共享内存。这只有在当前
blade-kvt 对象与 tensor 地址位于同一个 process address space 时才成立。

blade-kvt 当前 CUDA 路径使用 `cudaSetDevice`、CUDA Runtime、自己创建的
thread-local streams 和 kernels；源码中没有显式 `cuCtxCreate` 创建私有
Driver API context。CUDA 官方语义是：同一进程里，Runtime API 的使用者默认共享
“每 device、每 process 一个”的 primary context。因此可以概括为：

```text
同一个 OS worker process
  └─ 同一个 GPU primary CUDA context
       ├─ PyTorch/vLLM allocations、kernels、streams
       └─ blade-kvt streams、events、copy kernels、GPU MR address
```

但是，“共享 context”不等于“共享默认 stream”或“自动有执行顺序”。blade-kvt
创建自己的 CUDA stream，并通过 vLLM/KVT 记录的 CUDA event、`wait_layer_ready`
以及必要的 stream synchronization 建立依赖。若缺少这些同步，网络线程可能在
forward 尚未写完 KV block 时就开始读取。

### 2.2 如果把 blade-kvt 拆成独立进程

拆分后，两个进程各有自己的 CUDA context 和 virtual address space：

```text
vLLM worker process                 KVT service process
PyTorch allocation ptr=0xABC        不能直接使用数值 0xABC
        │
        ├─ export CUDA IPC/VMM handle ─► import → local device pointer
        ├─ export IPC event/同步协议 ──► wait before NIC/copy
        └─ lifetime/control RPC       ──► register MR、transfer、completion
```

CUDA 官方文档明确规定：device pointer 和 CUDA event handle 只能被创建它的
进程直接使用；跨进程必须交换 CUDA IPC 或 VMM 的 process-portable handle，
再在目标进程获得一个 process-local pointer。

因此当前代码若只把 blade-kvt 对象搬到另一个进程，会立即遇到：

1. **裸 `data_ptr` 失效**：KVT 不能直接注册或 launch kernel 访问 vLLM 的数值地址；
2. **同步句柄失效**：当前 event/stream 依赖要改为 CUDA IPC event 或显式 RPC；
3. **生命周期变复杂**：vLLM 不能在 KVT 尚未 completion 时 free/reuse allocation；
4. **MR 要重建**：KVT 要对自己 import 后的 mapping 做 RNIC registration，并在
   unmap 前等待 WR 完成、deregister；
5. **资源增加**：独立 CUDA context 会占用 device/host memory、driver threads，
   GPU 还可能产生 context scheduling 开销；
6. **错误边界更清楚**：KVT 崩溃不一定直接杀死 vLLM worker，升级和限流也更独立，
   这是拆进程的主要工程收益。

另一种更简单但性能更低的拆分方法是：

```text
vLLM GPU
  → vLLM 进程 D2H 到共享 host buffer
  → KVT 进程从 host buffer 发 TCP/RDMA
```

这样不必把 GPU pointer 跨进程交给 KVT，但变成 host-staged 路径，增加 D2H/H2D、
host memory bandwidth 和 NUMA 成本。

所以“拆进程”不是不可行，而是需要把当前的函数调用集成改造成一套明确的：

```text
GPU allocation export/import
+ event synchronization
+ MR registration cache
+ buffer lifetime protocol
+ control IPC / error recovery
```

它是架构重构，不是简单把现有线程换成一个 subprocess。

## 3. StepTasks：按目标聚合

`KvTransferClient::submit_req_send`：

1. 创建 `RequestInfo`；
2. `add_send_task`；
3. 放入 `targets_tasks_buf_`；
4. 未到 last token 的 request 还保存在 `reqs_`，供 delta send 复用。

`start_send`：

1. 创建 `Step` 与 `StepGuard`；
2. 交换出 `targets_tasks_buf_`；
3. 交给 `TargetMgr::submit`；
4. 单独线程等待每层 CUDA event/barrier，然后 `notify_layer_ready`。

源码：`kvtransfer/src/client.cpp:256-410`、`:463-513`。

## 4. TargetMgr：并发与同目标串行

`TargetMgr::do_submit` 按：

```text
InstanceId → WorkerId → BatchSendTask
```

遍历任务，为每个目标取得或创建一个 `KvSendStub`，再投递到 Barex `XThreadpool`。

同一个 target 使用稳定 `thread_hint`，目的是让发送批次落到一致 worker，降低同目标状态并发风险。见 `client.cpp:72-108`。

## 5. KvSendStub：数据布局决策

`KvSendStub::TaskContext::do_task`：

1. 过滤失败/空 delta request；
2. 获取目标 `WorkerInfo`；
3. 计算有效发送 rank；
4. 按 P/D TP 大小与 cache shape 选择 `parse_block_*`；
5. 生成 `vector<vector<IpcBlock>> send_blocks`；
6. 创建/复用 channel；
7. 调用 `do_send`。

`IpcBlock` 的关键字段：

```text
src_offset, dst_offset, length
```

它是后续 direct/staged/TCP 三条路径共同的逻辑数据描述。

源码：`kvtransfer/src/tx_stub.cpp:128-263`、`:416-604`。

## 6. 与模型 forward 的逐层 overlap

`TaskContext::do_send`：

```cpp
ch->register_data(send_blocks, tpkind);
for (layer = start_layer; layer < num_layers; ++layer) {
  step->wait_layer_ready(layer);
  ch->send_data(layer);   // 异步提交
}
ch->flush(metrics);       // channel 级完成等待
```

见 `tx_stub.cpp:395-413`。

因此 overlap 关系是：

```text
GPU forward layer i 完成并 record event
  → StepGuard 等 CUDA barrier
  → release layer i
  → send stub 提交 layer i KV
  → GPU 继续算后续 layer
```

## 7. Barex context 与 GPU MR 初始化

`BarexCtx` 构造时：

1. 按 protocol 取得 NIC device 与共享 mempool；
2. 遍历每层、每个 cache tensor；
3. `RDMA_DIRECT` 调用 `RegUserMr(... GPU ...)`；
4. 创建 Barex thread pool；
5. `XContext::NewInstance`；
6. `context->Start()`。

见 `barex_protocol.cpp:405-502`。

服务器随后把每层 tensor 的 GPU base 与 rkey 放入 `RDMAMemHandle`：

```text
handle.ptrs[tensor]  = GPU cache base
handle.rkeys[tensor] = GPU MR rkey
```

见 `rdma_channel.cpp:345-385`。

## 8. 连接与 MR handle 控制面

client `RDMAChannel::do_init`：

1. 建立 `BLLM_KVTRANS_SEND_PARALLEL` 个 Barex channel；
2. 通过第一个 channel 调用 `get_mem_handles`；
3. client 发送 `MEM_HANDLES_REQ_MAGIC + reqid`；
4. server `OnRecvCall` 返回全部 `RDMAMemHandle`；
5. client `CliBarexCtx` 用 reqid 匹配 promise。

![blade-kvt direct RDMA 控制面与数据面时序](/imgs/blade_kvt_direct_rdma_sequence.svg)

控制面调用的是 `Send`，因为它需要远端 callback/响应；payload 数据面不需要每层回调，直接 Write。

## 9. Direct RDMA：真正的 KV 数据发送

### 9.1 `register_data`

`RDMAChannel::register_data`：

- 保存 `IpcBlock` 引用；
- 建立/验证 channel；
- 对 `PEQD` 合并连续 interval；
- 统计 block 数、大小；
- 根据 channel 数计算每 channel 分配量 `dataperch_`。

见 `rdma_channel.cpp:612-678`。

### 9.2 `send_data(layer)`

对每个 tensor/block 构造：

```cpp
rwmemp.r_addr   = remote_gpu_base + dst_offset;
rwmemp.r_key    = remote_rkey;
rwmemp.sg.addr  = local_gpu_base + src_offset;
rwmemp.sg.length = len;
rwmemp.sg.lkey  = local_gpu_mr->lkey;
```

积累到 `dataperch_` 后：

```cpp
ch->WriteBatch(datas, callback)
```

每批通过 `RDMAChannel::ch()` round-robin 选择多个 Barex channel。future 保存到 `write_futs_`。

见 `rdma_channel.cpp:696-760`。

### 9.3 Barex 内部

`XChannelImpl::WriteBatch`：

1. 校验/切换本地 MR；
2. 构造一个或多个 `ibv_send_wr`，opcode 为 RDMA WRITE；
3. 受 `tx_depth` semaphore 控制；
4. `ibv_post_send`；
5. CQ 返回 `IBV_WC_RDMA_WRITE`；
6. `HandleWriteComplete` 调 callback；
7. promise 完成。

### 9.4 `RDMAChannel::flush`

遍历 `write_futs_` 调 `future.get()`：

- 任一 CQ 异步错误会以 exception 抛出；
- 全部完成后清空 futures；
- 可选地请求接收端计算 CRC；
- 输出 send latency/block metrics。

见 `rdma_channel.cpp:809-854`。

## 10. 两层不同的 flush

必须区分：

### 10.1 `IChannel::flush`

由 `KvSendStub::do_send` 调用。Direct RDMA 实现会等待本 batch 的所有 Barex Write completion。这是真正的数据面完成等待。

### 10.2 `KvTransferClient::flush_send`

当前版本 `client.cpp:524-541` 只做：

1. `layer_ready_all()`，解除所有 layer wait；
2. 记录时间；
3. reset `last_step_guard_`。

它没有显式 join `TargetMgr` thread pool，也没有等待所有 `KvSendStub` 结束。实际每个 target task 会在自己的 `IChannel::flush` 中等待完成，但公共 `flush_send()` 本身从代码上看不保证在这些 task 结束后才返回。

因此当前版本应谨慎理解 Python 注释中的“check whether progress is finished”。若上层依赖严格同步完成，应进一步验证调用方时序或增加 step completion barrier。

## 11. Staged RDMA

适用于需要 host staging 的路径：

```text
source GPU blocks
  → CUDA gather kernel
  → local CUDA_HOST buffer
  → Barex WriteSingle(signal_peer=true)
  → remote CUDA_HOST preallocated buffer
  → remote OnImmRecvCall
  → H2D scatter
  → Barex response
```

特点：

- 先 RPC 请求每层 remote staging buffer；
- payload 用 `RDMA_WRITE_WITH_IMM`；
- immediate data 编码 buffer id；
- local CQ completion 只说明 RDMA write 完成；
- `staged_write_futs_` 等待远端 H2D 后的 response，才是端到端完成。

见 `rdma_staged_channel.cpp:81-130`、`:144-300`、`:302-409`。

## 12. TCP 路径

```text
GPU blocks
  → CUDA D2H gather（可选 BF16→FP8）
  → CUDA_HOST wire buffer
  → Barex XChannel::Send
  → TCP backend
  → server parse
  → H2D scatter
  → response
```

与 staged RDMA 相同点：

- payload 都先 gather 到 host buffer；
- 都用 reqid/future 等远端完成；
- `flush` 统计 D2H、network、H2D、recv queue。

不同点：

- staged RDMA 数据使用 one-sided WriteWithImm；
- TCP 使用 Barex message `Send`；
- staged 远端 buffer 需提前注册和交换 rkey。

见 `tcp_channel.cpp:348-587`、`:674` 起。

![blade-kvt 三种发送模式](/imgs/blade_kvt_send_modes.svg)

### 12.1 为什么先把离散小块 gather 成连续大 buffer

假设一层需要发送 64 个不连续 KV block，每个 4 KiB：

```text
GPU source:
[block 7] ... [block 91] ... [block 13] ...  共 64 段

blade-kvt gather 后的 pinned host wire buffer:
[RPC header][IpcBlock metadata][block 7 bytes][block 91 bytes]...[block 13 bytes]
                                      连续 256 KiB payload
```

这里“连续地址”主要指一个连续的**进程虚拟地址/wire byte range**。`cudaMallocHost`
保证页被 pinned，但底层 DRAM page 不必物理连续；GPU/NIC 的 DMA mapping 与
scatter-gather table 可以处理多个物理页。它也不表示 256 KiB 会成为一个
Ethernet packet。

当前源码分两步减少碎片：

1. `TCPChannel::register_data()` 对 `PEQD` 的相邻 interval 调
   `merge_interval()`，能相邻的 GPU range 先合并；
2. `TCPChannel::send_data()` 为一层构造一个预分配 `CUDA_HOST` message，
   `copy_handle_data_with_kernel()` 按 `IpcBlock.src_offset` 从离散 GPU 地址
   gather 到每个 tensor 的连续 blob，最后只调用一次 Barex `Send()`。

接收端解析同一个 message，利用其中的 `IpcBlock.dst_offset` 把连续 blob scatter
回离散 GPU KV block。

#### 好处 1：摊薄每次操作的固定延迟

沿用前文性能模型：

```text
T(one operation) ≈ Tfixed + bytes / bandwidth
```

若 64 个小 block 各自发送：

```text
Tsmall ≈ 64 × Tfixed + total_bytes / bandwidth
```

先聚合再发送：

```text
Tlarge ≈ Tgather + 1 × Tfixed + total_bytes / bandwidth
```

`Tfixed` 可能包含函数调用、锁、队列分配、`send`/doorbell、调度、callback、
promise/future 和 completion 处理。只要省下的 `63 × Tfixed` 大于额外 gather
成本，聚合就更快。小 block 越小、数量越多，通常越容易满足这个条件。

#### 好处 2：减少 CPU、内核网络栈与 Barex 对象开销

如果每个 block 分开发送，可能产生更多：

- C++/Barex `Send` 调用；
- socket send 操作、`sk_buff` 组织和 queue 操作；
- application message header；
- callback、future 和 reqid 状态；
- buffer ownership/refcount 与错误处理分支。

聚合后 metadata 仍需记录每段的 source/destination offset，但控制对象从“每
block 一套”变成“一层/一批一套”。

#### 好处 3：让 TCP/GSO/TSO 和 NIC 更容易进入吞吐区间

TCP 是 byte stream。应用提交一个 256 KiB buffer 后，Linux TCP 和 NIC 仍会按
MSS/MTU 分成许多 TCP segment/Ethernet frame；如果启用 GSO/TSO，协议栈可以先
处理一个较大的逻辑 buffer，再由软件/NIC 完成 segmentation 和 checksum。

```text
1 个 256 KiB application message
  → 若干 large skb / TSO work
  → 很多个 MSS-sized TCP segment
  → NIC TX DMA
  → PCIe 上又是许多 Memory Read TLP
```

因此聚合优化的是**应用、通信库、内核和 NIC 的任务粒度**，不是突破 MTU，也不是
把整批数据变成一个 TLP。它与 01 章“一个 DMA descriptor 会分成许多 TLP”是同一
种分层思想。

#### 好处 4：连续 destination 简化 framing 与接收端解析

当前 wire layout 是：

```text
固定 header
+ tensor 数量
+ 每个 tensor 的 block metadata
+ 每个 tensor 的连续 data blob
```

接收端收到一个完整 Barex message 后就能检查总长度、逐 tensor 解析并启动 H2D
scatter，不需要维护 64 个小 message 的乱序业务状态和“这一层是否收齐”计数。
TCP 自己保证 byte stream 有序，但应用仍需要 framing；合并后 framing 数量更少。

#### 好处 5：更容易复用 pinned buffer 和做流水线

blade-kvt 按 layer 预分配 host buffer，并在 layer ready 后执行：

```text
layer i:   D2H gather → TCP send → remote H2D
layer i+1:      forward / gather ...
```

buffer size 和生命周期可预测，更适合池化、NUMA 绑定、容量检查和端到端计时。

#### 代价与边界

聚合不是免费优化：

- 增加一次 source D2H gather 和 destination H2D scatter；
- 消耗两端 host memory bandwidth 与 pinned memory；
- 必须等到一批中足够多的 block ready，过大的 batch 会增加首字节延迟；
- gather kernel 访问过于离散时，GPU memory coalescing 可能较差；
- 大 buffer 占用 socket send buffer/队列更久，backpressure 与超时策略要匹配；
- pinned buffer 如果分配在错误 NUMA node，会让 GPU copy 或 NIC DMA 跨 NUMA。

因此应通过 block count/size sweep 找 break-even，而不是无条件“越大越好”。
Linux `MSG_ZEROCOPY` 文档同样展示了这个规律：避免 copy 会引入 page pin 和
completion bookkeeping，小消息反而可能不划算。

#### 为什么不用 `writev/sendmsg` 直接发送离散 GPU block

普通 `writev/sendmsg` 的 `iovec` 可以描述多个**CPU 可访问的 host buffer**，能减少
system call 数，但不能让标准 Linux TCP 自动读取普通 CUDA device pointer。当前
blade-kvt 还需要：

- 把 GPU KV block 搬到 TCP 可消费的 host memory；
- 可选完成 BF16→FP8；
- 构造统一 metadata；
- 在对端按不同 `dst_offset` scatter。

因此连续 pinned wire buffer 是合理实现。若未来使用专门的 GPU-aware TCP、
dma-buf device-memory TCP 或不同的通信接口，可以重新评估 copy 与 scatter-gather
策略。

## 13. send-done 是第四条独立路径

KV payload 完成后，`KvSendStub::send_done` 经普通 TCP 长连接发送业务完成通知：

```text
SEND_DONE_REQ / SAVE_DONE2_REQ
  + worker_tp_rank
  + req count
  + plen/code/reqid
```

它不是 Barex CQ completion，也不是 NCCL。失败时关闭 socket 并重建一次。见 `tx_stub.cpp:274-393`。

所以系统中同时存在：

1. Barex 建联/control RPC；
2. Barex KV payload；
3. staged/TCP 远端完成响应；
4. 业务 send-done TCP RPC。

## 14. 错误传播

| 层 | 机制 |
|---|---|
| Barex 同步提交 | `BarexResult != BAREX_SUCCESS` |
| Barex 异步 WR | `DoneCallback(Status)` |
| blade wrapper | callback 设置 promise value/exception |
| channel flush | `future.get()` 抛异常 |
| send stub | catch，reset channel，标记 request FAILED |
| 业务完成 | send-done code 500 |

channel reset 后下次 batch 会通过 naming 刷新地址并重建连接。

## 15. 最短代码阅读顺序

```text
blade_kvt/kv_transfer_impl.py
→ kvtransfer/kvtransfer_pybind.cpp
→ kvtransfer/src/client.cpp
→ kvtransfer/src/tx_stub.cpp
→ kvtransfer/include/channel.h
→ kvtransfer/src/rdma_channel.cpp
→ kvtransfer/src/barex_protocol.cpp
→ Barex include/accl/barex/xchannel.h
→ Barex src/barex/impl/rdma/xchannel_impl.cc
→ Barex src/barex/impl/rdma/xcontext_impl.cc
```

## 16. 自检

1. 为什么取 MR handle 用 `Send`，KV payload 用 `WriteBatch`？
2. `send_data()` 返回后为什么不能立即复用源 KV block？
3. direct RDMA 与 staged RDMA 的完成边界分别是什么？
4. `flush_send` 与 `RDMAChannel::flush` 为什么不能混为一谈？
5. send-done 为什么不等于网络 CQ completion？
6. 为什么 blade-kvt 与 vLLM worker 当前能直接交换 `torch.Tensor.data_ptr()`？
7. 拆成独立进程后，为什么不能只把这个数值地址通过 socket 发给 KVT service？
8. 把 64 个小 block 聚合后，为什么网络上仍会有多个 TCP segment 和 PCIe TLP？
9. 聚合在什么情况下可能因 gather、等待或 NUMA 成本而得不偿失？

## 参考

- [CUDA Driver/Runtime interoperability：每 device、每 process 的 primary context](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html)
- [CUDA Programming Guide：跨进程 device pointer/event 必须使用 IPC/VMM handle](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)
- [Linux `sendmsg(2)`：`iovec` scatter-gather](https://man7.org/linux/man-pages/man2/sendmsg.2.html)
- [Linux `MSG_ZEROCOPY`](https://docs.kernel.org/networking/msg_zerocopy.html)
- [Linux TCP Segmentation Offload / GSO](https://docs.kernel.org/networking/segmentation-offloads.html)
