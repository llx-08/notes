---
title: "08 错误传播与 corner cases：断连、超时、取消和迟到完成"
date: 2026-07-30
categories: [vllm、Blade-kvt与PD分离]
tags: [vLLM, Blade-KVT, PD 分离, KV Cache, Hybrid Connector, Barex, 学习笔记]
---

# 08 错误传播与 corner cases：断连、超时、取消和迟到完成

## 1. 先建立错误传播分层

```text
CUDA / layout error
  ↓ C++ exception
Blade-KVT ReqState::FAILED
  ↓ SEND_DONE code=500
PBackend IoRet / KVTResp
  ↓ DBackend IoRet.ex
HybridScheduler _loaded
  ↓ sched_add_req + abort
Scheduler finish/free
  ↓ EngineCoreOutput ABORT
用户/路由层
```

不是所有错误都能完整走完这条链。某些通知本身发送失败时，系统可能停在中间，
因此要同时讨论“理想路径”和“通知丢失路径”。

## 2. P Worker 本地构造 metadata 失败

例如 P/D TP size 不能整除，`_get_distinfo()` 返回空。

PBackend worker 不会只 log 后继续，它把 reqid 收集到 `failed_reqids`，在独立 worker
uvloop 调：

```python
KVTransferClient.send_error_done_req(reqids)
```

该函数直接构造与 Blade-KVT `SEND_DONE_REQ` 兼容的 TCP 消息：

```text
plen=0
code=500
reqid
worker_tp_rank
```

目的：即使真正的 C++ send task 根本没创建，P Scheduler 的 `_SendingReq` 也能收到
失败信号，不会永远等“本不存在的 WQE”。

## 3. Blade-KVT 发送过程中异常

`KvSendStub::TaskContext::do_task()` 用 try/catch 包住：

```text
try_create_channel
parse_block
register_data
wait layer
send_data
flush
```

异常后：

1. `ch.reset()`，不复用可疑 channel；
2. batch 中每个 task 设 `ReqState::FAILED`；
3. 对 `reach_last_token` 的请求仍构造 send_done，code=500；
4. PBackend 汇聚失败。

这体现一个重要原则：错误 completion 也是 completion。失败分支必须尽量产生终结信号，
否则上层只看到“还没完成”。

## 4. P 或 D 突然断开：分阶段分析

### 4.1 D 在 P 建连前消失

```text
naming/WorkerInfo 有旧 endpoint
→ Connect 失败
→ ChannelFactory 尝试候选 protocol
→ 全部失败
→ task FAILED
→ SEND_DONE 500
```

下轮 `try_create_channel()` 会重新 refresh/建连。

### 4.2 D 在 RDMA Write 中消失

预期 provider 通过错误 CQE/callback 结束 WR，future 抛异常。然后 task failed。

风险：如果底层连接故障没有及时生成 callback，RDMA Direct `future.get()` 没有本层
timeout，clisend 线程会一直占用。

### 4.3 D 数据收完但 P→P-Scheduler send_done 断开

Blade-KVT send_done 长连接失败会重连再试一次。两次都失败时只 log：

```text
D KV 正确
P 数据面已完成
P Scheduler 不知道
_SendingReq Future pending
D Scheduler 仍等 KVTResp
```

这是典型“数据成功、控制面丢 completion”。要靠上层 deadline/watchdog 打破。

### 4.4 P 在 D 等 KVTResp 时消失

若 socket 收到 EOF，D `_kvt_rpc` 捕获 `IncompleteReadError`，重试一次；再失败则
`RuntimeError("kvt rpc failed")`，包装进 D 本地 `IoRet.ex`，请求 abort。

若连接半开且无读 timeout，可能依赖 OS TCP keepalive/timeout，响应时间不可控。

### 4.5 D 收到部分 KV 后 P 消失

D Block 可能被部分写入。正确处理不是“用已到部分继续 decode”，而是：

- request load 失败；
- 不把 Block 加入 prefix cache；
- abort 后释放/以后零化或覆盖；
- late WR/completion 不得把已复用 Block 污染。

最后一点需要 transport teardown 与 Block release fence 协调。当前高层通过等待 P
完成后才让 D request 可运行，但失败后的 late DMA 安全仍依赖 Barex/provider 的连接
错误语义和释放时序，需要专项故障注入。

## 5. 404 与 410：避免“永远等一个不会出现的 P Request”

dash 模式 D 可能比 P 早到。

### 404 Not Found

含义：“P 当前找不到，但可能只是请求还没到。”D 按 delay list 重试。

### 410 Gone

P `_gone_reqs` 记录曾存在但已经 abort/timeout 的 reqid。含义：“以后也不会出现，
不要再等。”D 立即失败。

LRU：

- TTL 由 `VLLM_KVT_GONE_REQ_TTL_S`；
- 最大 8192，防止 abort 风暴导致无界内存；
- TTL 过期后又退化为 404；
- TTL=0 关闭该保护，保留旧行为。

这是用有限 tombstone 解决分布式“未到达”与“已删除”歧义。

## 6. request abort 与正在进行的 I/O

D 的 `_abort_prefill()`：

1. 从 Request 参数读取之前选中的 P 地址；
2. 按 peer 分组 reqid；
3. 并发发 `ABORT_REQS_REQ`；
4. 每个 peer 尝试两次；
5. 失败只 log，不阻塞本地 abort 完成。

因此 abort RPC 是 best effort。设计理由：

- 用户取消不能永远等待故障 P；
- P 仍有自己的 timeout/gone 清理；
- 本地必须先释放服务能力。

代价是远端可能短时间继续计算/发送，要求 Block/连接层能容忍 late activity。

## 7. finish/save 竞态

四种事件排序：

| 顺序 | 处理 |
|---|---|
| save → finish | `_PD_SAVED=True`，finish 立即回 done |
| finish → save | 先回 pending，保存 finish 信息；save 后发空输出 done |
| abort → save | abort 输出立即；extra ref 保护 Block，save 后 teardown |
| save error → finish | save failure 记入 IoRet/统计，按失败/abort 路径收尾 |

关键不变量：

```text
每个 _setup_save 必须恰好有一次 _try_teardown_save
每次 teardown 必须释放一次额外 Block ref
```

重复 teardown 会 log，不能重复减 refcount。

## 8. duplicate/unknown completion

`try_advance()` 检查：

- reqid 不在 state dict：unknown，log 并忽略；
- 同 worker 已达到期望 signal 数：duplicate，忽略；
- source 缺失或不在 expected set：忽略；
- 同 worker 同 source 重复：忽略。

这样 late/retry completion 不会：

- 把计数加两次而过早完成；
- 对已释放请求再次 teardown；
- 用错误 backend source 冒充另一个 source。

但“忽略”也意味着如果发送方标错 source，真正所需 source 永远缺失，仍需 watchdog 报出
missing source，而不能只靠 warning。

## 9. 多 TP worker 中一个失败

`IoState.merge()` 汇聚所有 ready worker signal：

- 任一 `ex` 优先成为总错误；
- `n` 取最小值，避免部分 rank 声称更多 token；
- P<D 时也检查同一 P worker 对所有 fan-out target 的 signal。

如果一个 worker 明确返回失败，整个 request 失败；如果一个 worker完全不返回，当前
没有上层 barrier timeout，request pending。

## 10. Block 不足与长时间传输形成反压

不是所有“卡住”都是网络错误：

```text
慢网络
→ P save extra refs 长时间不释放
→ free blocks 下降
→ Hybrid _waiting allocate 返回 None
→ 新请求不能启动 load/save
```

这是一条合法 backpressure。判断它与死锁的区别要看：

- in-flight completion 是否仍增长；
- oldest saving age；
- free block 是否随 completion 回升；
- clisend/CQ thread 是否有进展。

## 11. async scheduling 和 substep corner cases

### bypass 比主 step 先到

放入 `pending_step_metas_`，主 step 创建时按递增 substepid 附加。

### bypass 比主 step 晚太多

当前 step 已结束，转成独立 freeze Step，所有层直接 ready。

### flush 时 pending substep 仍存在

代码 `assert(pending_step_metas_.empty())`。违反说明 step 协调协议出错，debug build
会立即暴露；release 下 assert 可能消失，因此生产还应有指标/运行时错误。

### sched_tokens=0 且没有 task

返回 `EMPTY_STEP_ID`，不创建无意义 Step/线程等待。若 sched_tokens>0 即使当前无发送
任务也创建 Step，因为 bypass 可能稍后附加。

## 12. 多模态 placeholder 跨 PD 边界

混合模型保留 `gamma+1` token 在 D 重算。如果切分点落在一个 multimodal placeholder
内部，P/D 看到的特征边界不一致。

当前：

- 显式 remote-prefill：raise，失败；
- 可本地 fallback 场景：warning 后不做远端 prefill。

这是语义 corner case，不是网络能修复的问题。

## 13. 短 prompt

若 prompt 不超过 `gamma+1`：

- 普通 disagg 可 fallback 到 D 本地 prefill；
- 显式 remote-prefill 配置则报错；
- 不应发 0/负 token 的 KVT task。

## 14. cache shape / TP 不兼容

Blade-KVT 对不同 cache shape 选择不同 `parse_block`：

- Flash/Ragged/FlashInfer；
- Qwen3-Next hybrid GDN；
- DPSK sparse MLA 多 tensor；
- Kimi K3 MLA；
- Kimi K3 + Eagle；
- TurboQuant。

例如 Kimi K3 + Eagle 当前要求 P/D TP 相等，staged RDMA 也不支持 inactive tensor
slots。遇到不支持的组合应 fail fast，不能退化成“按普通 MLA 发”，否则传输层成功但
KV layout 错误。

## 15. 已有测试覆盖与缺口

已有上游/本地测试覆盖：

- remote prefill/decode 基本生命周期；
- prefix cache 与 partial Block；
- abort 期间 Block 引用；
- load 失败恢复；
- invalid block 与 cache pollution；
- Hybrid abort/save 引用泄漏；
- migration 与 Hybrid connector 功能。

建议补充的故障注入矩阵：

| 故障点 | 断开时机 | 必查结果 |
|---|---|---|
| D server | 建连前/写中/写后 | P future、send_done、Block ref |
| P scheduler RPC | 数据前/数据后 | D deadline、P orphan state |
| send_done RPC | 首次/重连也失败 | `_SendingReq` watchdog |
| 单 TP worker | 不发/重复发/错 source | barrier timeout 与诊断 |
| CUDA Event | 漏 record/错 stream/乱序 | wait-layer 能否定位 layer |
| zero kernel | load 前/中/后 | KV 对比不被清零覆盖 |
| abort | load 前/写中/KVTResp 后 | late completion 去重与 Block 安全 |

## 16. 自检

1. 为什么“数据传完但 send_done 丢失”比普通传输失败更难发现？
2. 404/410 的 tombstone 为什么要 TTL 和容量上限？
3. abort RPC 为什么是 best effort？
4. 某 TP rank 明确失败与完全沉默，当前行为有何不同？
5. 如何区分合法显存反压与永久 hang？
