# 09. 调试与性能分析手册

## 1. 原则：先分层，再猜原因

建议固定按以下顺序：

```text
硬件链路
  → OS/driver/RDMA 能力
  → Barex device/context/channel
  → MR 与远端 handle
  → WR 提交
  → CQ completion
  → blade-kvt future/flush
  → 业务 send-done
```

否则容易把 `send-done 超时` 错判成 RDMA 未发送，或把 `RNR` 错判成 PCIe 带宽问题。

## 2. 建立一次诊断快照

### 2.1 PCIe/GPU

```bash
date
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi nvlink --status
lspci -t
lspci -Dnn | rg -i 'nvidia|ethernet|infiniband'
```

对目标 GPU/NIC：

```bash
lspci -vv -s "$GPU_BDF" | rg 'LnkCap|LnkSta|ACSCtl'
lspci -vv -s "$NIC_BDF" | rg 'LnkCap|LnkSta|ACSCtl'
cat "/sys/bus/pci/devices/$GPU_BDF/numa_node"
cat "/sys/bus/pci/devices/$NIC_BDF/numa_node"
```

### 2.2 RDMA

```bash
ibv_devices
ibv_devinfo
rdma link
rdma resource show
ip -br addr
```

### 2.3 软件与模块

```bash
nvidia-smi
modinfo nvidia | head
lsmod | rg 'nvidia_peermem|nvidia|ib_core|mlx5'
ldd "$(python - <<'PY'
import kvtransfer_ops
print(kvtransfer_ops.__file__)
PY
)" | rg 'barex|ibverbs|cuda|nccl'
```

## 3. 证明选择了哪条 blade-kvt 路径

不要只看配置值，检查运行日志和 `WorkerInfo.transfer_protocols`。

| 证据 | Direct RDMA | Staged | TCP |
|---|---:|---:|---:|
| `RegUserMr ... GPU` | 是 | context 可能仍注册，但 payload staging | TCP 不注册 RDMA MR |
| `RDMAChannel connect` | 是 | 基类连接后还有 prealloc | 否 |
| `Staged=1` metric | 否 | 是 | 否 |
| D2H/H2D metric | 通常无 | 有 | 有 |
| `TCPChannel connect` | 否 | 否 | 是 |
| `WriteBatch` | 是 | 否 | 否 |
| `WriteSingle` + imm | 否 | 是 | 否 |
| Barex `Send` payload | 仅控制面 | 控制面/响应 | 是 |

## 4. 建联失败

### 症状

- `Connect` future 超时；
- `invalid rdma address`；
- `can't start RDMA transfer server`；
- channel 很快进入 dead/destroyed。

### 检查

1. naming 返回的 `ip:port` 是否仍属于当前 worker。
2. server listener 是否真的监听 `env_port_base + worker_id`。
3. client/server 的 `NCCL_SOCKET_IFNAME`/`NCCL_SOCKET_FAMILY` 是否把 Barex 带外流量选到不同网络。
4. `ACCL_USE_NICS` 是否过滤掉预期 HCA。
5. GID、RoCE VLAN、MTU、traffic class 是否一致。
6. 防火墙/安全组是否允许带外 TCP 端口。
7. QP init meta 交换后是否能进入 RTS。

## 5. MR 注册失败

### 常见原因

- GPU pointer 不是预期 device 上的有效 CUDA allocation；
- 注册长度越过 allocation；
- `ACCL_MAX_USER_MR_GB` 太小；
- MR 数量达到限制；
- GPU/NIC 不支持 peer memory/dma-buf；
- IOMMU/driver 配置不兼容；
- 多 NIC 下使用了错误 PD 对应的 lkey。

blade-kvt 每层每 tensor 注册一次：

```text
MR count ≈ num_layers × tensors_per_layer
MR bytes ≈ layer_num_blocks × block_size
```

先打印并核对 cache 的真实 allocation 边界，不要只提高限制掩盖越界。

## 6. WR 同步提交失败

Barex API 直接返回错误时，说明尚未等到 CQ：

| 返回 | 方向 |
|---|---|
| `BAREX_ERR_STAT` | channel 非 `INIT_SUCCESS` |
| `BAREX_ERR_ARG` | MR、地址、长度、callback 参数 |
| `BAREX_ERR_QUEUE_FULL` | `soft_tx_depth` 满 |
| `BAREX_ERR_RDMA_SEND` | `ibv_post_send` 同步失败 |

`QUEUE_FULL` 不等于 NIC 丢包，而是应用提交速度超过 Barex 硬件 depth + 软件 queue 承载。

## 7. CQ 异步错误

| WC status | 首查 |
|---|---|
| `LOC_PROT_ERR` | local addr/lkey/length/MR lifetime |
| `REM_ACCESS_ERR` | remote addr/rkey、目标重启、越界 |
| `RETRY_EXC_ERR` | 网络、对端 QP、路由、链路 |
| `RNR_RETRY_EXC_ERR` | 对端 recv WR 不足；常见于 SEND/WRITE_WITH_IMM |
| `WR_FLUSH_ERR` | QP 已转 ERROR；寻找最早的非 flush 错误 |

一个 QP 出错后，后续 WR 常批量出现 `WR_FLUSH_ERR`。根因通常是日志中该 QP 的第一条异常。

## 8. `rkey` 失效与目标重启

blade-kvt 在 channel 初始化时缓存全部 `dst_handles_`。接收端进程重启后：

- GPU virtual address 可能变化；
- MR 重新注册，rkey 一定应视为变化；
- 旧 channel/QP 与 handle 都不可复用。

遇到 `REM_ACCESS_ERR`：

1. 对比 naming 中 worker identity；
2. 检查是否仅刷新了 IP，未重建 channel；
3. 清理 channel 后重新 `get_mem_handles`；
4. 检查 `same_dst` 是否把重启后的 worker 误判为同一目标。

## 9. flush/超时定位

先确认卡在哪种 future：

### Direct RDMA

`RDMAChannel::write_futs_` 等本地 Barex Write callback。卡住说明：

- CQ 未产生；
- CQ progress thread 未运行；
- callback 未调用；
- promise 生命周期或 batch completion 有问题。

### Staged/TCP

除了本地提交，还等 `CliBarexCtx::rpc_` 中的远端响应。卡住可能是：

- payload 已到但 server H2D 慢；
- `OnImmRecvCall/OnRecvCall` 未执行；
- reqid response 丢失或未匹配；
- server callback thread pool 堵塞；
- client timeout 后 pop，迟到 response 成为 `UnknownReqId`。

### 公共 `flush_send`

当前实现没有 join target task。若观察到 Python flush 已返回但后台仍有 `SendStubMetrics`：

- 这是当前代码结构允许的现象；
- 需要用 step/task 完成信号验证严格边界；
- 不应仅以 `_cur_step_id=None` 认定全部 CQ 已完成。

## 10. 性能拆解

### 10.1 Direct RDMA

```text
WaitUs       = 等 layer ready
SendUs       = WriteBatch submit 到 CQ callback
SbSizeTotal  = 实际 KV bytes
InflyWrite   = batch/future 数
```

估算吞吐：

```text
effective_GBps = total_bytes / wall_clock_seconds / 1e9
```

不要用所有 future 的 `SendUs` 相加作为 wall time，因为它们并发。

### 10.2 Staged/TCP

```text
SendUs ≈ D2H + transport + recv queue + H2D + response
```

代码已有：

- `D2HUs`
- `TransUs`
- `LinkTxUs`
- `RecvUs`
- `OnRecvQueueUs`
- `H2DUs`

先看最大项，再优化。

## 11. 常见性能形态

### 11.1 小 block 太多

症状：

- `OriginSbNum` 很大；
- `MergedSbNum` 接近 Origin；
- 单 WR 很小；
- CPU/post WR 占比高。

方向：

- 检查 block 是否能按 destination interval 合并；
- 检查 P/D TP mapping 是否导致碎片化；
- 调整 batch/SG list，但不越过 max SGE/depth。

### 11.2 tx depth 不足

症状：

- queue depth 增长；
- 链路未跑满；
- completion 一到就继续提交。

方向：

- 估算 bandwidth-delay product；
- 小步增大 `ACCL_TX_DEPTH`；
- 同时检查 CQ progress 与 `soft_tx_depth`。

### 11.3 多 channel 没收益

原因可能是：

- 所有 channel 仍在同一 QP/CQ bottleneck；
- `dataperch_` 分配不均；
- block 数少于 channel 数；
- 单 NIC/PCIe link 已饱和；
- CPU poll thread 成为瓶颈。

### 11.4 staged D2H/H2D 占主导

方向：

- CUDA copy stream 与 forward 的依赖；
- pinned buffer NUMA；
- gather/scatter kernel block 合并；
- direct GDR 是否可用；
- FP8 转换收益是否覆盖 kernel 成本。

## 12. NCCL 与 Barex 竞争

同一进程或节点同时运行两套通信时检查：

- NCCL 和 Barex 是否绑定同一 NIC；
- 是否共享同一 PCIe uplink；
- NCCL proxy 与 Barex CQ thread 是否抢同一 CPU；
- traffic class/QoS；
- NCCL collective 与 KV transfer 是否在同一时间窗爆发。

A/B：

1. 仅 NCCL；
2. 仅 blade-kvt；
3. 同时运行；
4. 分 NIC；
5. 同 NIC 不同 QoS。

## 13. 日志检索模板

```bash
rg -n 'RegUserMr|RDMAChannel connect|get_mem_handles' app.log
rg -n 'Write Submit Err|wc status error|QUEUE_FULL|RNR|REM_ACCESS' app.log
rg -n 'SendStubMetrics|OriginSbNum|MergedSbNum|InflyWrite' app.log
rg -n 'UnknownReqId|flush timeout|send done' app.log
```

按 channel/QP/reqid/stepid 关联，而不是只按时间相邻判断。

## 14. 修改前的最小实验

任何调参前记录：

```text
代码 commit
机器/GPU/NIC 型号
PCIe topology
消息大小分布
协议路径
环境变量
吞吐/延迟/错误
日志片段
```

一次只改一个变量，至少重复多轮，并区分冷启动建联与稳态传输。

