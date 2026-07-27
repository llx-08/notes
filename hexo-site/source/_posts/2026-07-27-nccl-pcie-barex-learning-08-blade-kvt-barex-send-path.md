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
