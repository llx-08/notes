---
title: Blade-KVT / Barex TCP 重传与传输长尾排查记录
date: 2026-07-17
tags: []
---

# Blade-KVT / Barex TCP 重传与传输长尾排查记录

> 日期：2026-07-16  
> 场景：vLLM + Blade-KVT 的 PD 分离，Blade-KVT 通过 Barex TCP 传输 KV Cache  
> 发送节点：`10.44.132.30`  
> 主要接收节点：`10.44.172.215`、`10.44.173.154`、`10.44.173.157`、`10.44.172.222`

## 1. 问题背景

服务在长上下文 PD 分离负载下出现明显的传输长尾。Blade-KVT 的 `SendStubMetrics` 中主要表现为：

- `LinkTxUs`、`TransUs`、`SendUs` 波动很大，部分请求达到数百毫秒至 1 秒以上；
- `RecvUs` 偶尔出现明显长尾；
- D 节点的 `OnRecvQueueUs` 也可能达到数十至数百毫秒；
- 将 `ACCL_TCP_WORK_THREAD_CNT` 从默认的 16 增加到 32 后，没有看到明显改善；
- TCP socket 的历史统计中曾出现约 2%～3% 的 `bytes_retrans / bytes_sent`。

本次排查的目标是回答以下问题：

1. `LinkTxUs` 长尾是否由 Barex 工作线程不足引起；
2. 是否真的存在 TCP 重传，重传集中在哪些 P→D 连接；
3. 重传是实际丢包，还是乱序导致的伪重传；
4. MTU、Barex 配置和 D 侧 callback 排队分别扮演什么角色；
5. 下一步应该从代码、容器网络、宿主机还是交换机侧优化。

## 2. Blade-KVT 与 Barex TCP 链路映射

当前 Blade-KVT 使用 Barex `XDT_TCP` 路径。简化后的链路如下：

```text
P GPU KV Cache
  │
  ├─ D2H：Blade-KVT 将待发送数据拷入 pinned host buffer
  │
  ├─ Barex XChannelTcpImpl::Send
  │    └─ 将发送任务投递到 TCP worker pool
  │         └─ channel id % ACCL_TCP_WORK_THREAD_CNT
  │              └─ boost::asio::write(header)
  │              └─ boost::asio::write(body)
  │
  ├─ TCP/IP 网络
  │
  ├─ D 端 async_read(header/body)
  │
  ├─ Barex callback pool
  │    └─ BLLM_KVTRANS_CTX_TPSIZE
  │
  └─ H2D：写入 D GPU KV Cache
```

几个重要性质：

- 同一 TCP channel 内消息有序，单个 channel 不能通过增加 worker 数并行发送；
- Barex 当前使用同步阻塞的 `boost::asio::write`，慢连接会占用对应 worker；
- TCP 是有序字节流，任何一个数据段丢失都会阻塞后续已到达数据的交付，形成 Head-of-Line blocking；
- `ACCL_TCP_WORK_THREAD_CNT` 只影响发送任务被映射到多少个 Barex TCP worker；
- `BLLM_KVTRANS_CTX_TPSIZE` 影响 D 侧接收完成后的 callback/H2D 并行度；
- 网络丢包不会通过增加上述线程数得到解决。

### 2.1 Blade-KVT 指标的含义

当前指标大致可以拆分为：

```text
D2HUs
  + LinkTxUs
  + RecvUs
  + OnRecvQueueUs
  + H2DUs
  ≈ SendUs / TransUs 的主要组成部分
```

- `D2HUs`：P GPU 到 P pinned host buffer；
- `LinkTxUs`：包含 Barex sender worker 排队、同 channel 前序消息阻塞、header 发送和网络等待等，不只是纯粹的线上传输时间；
- `RecvUs`：D 端收到 header 后，读取完整 body 的时间；TCP 中间有缺口时，该值会被放大；
- `OnRecvQueueUs`：body 读取结束到 Barex/KVT callback 真正开始执行之间的等待；
- `H2DUs`：D callback/H2D 阶段。

需要注意：旧版 `LinkTxUs` 用 P、D 两台机器上的 `system_clock` 时间戳直接相减。如果 D 的时钟比 P 慢几百微秒，负数会被转换成 `uint64_t`，产生类似 `18446744073709551xxx` 的下溢值。因此，出现这种极大值时不能用其 min/max/avg 判断网络；本次 TCP 重传结论全部来自 Linux kernel socket 计数，不依赖这个跨机时间戳。

## 3. 排查方案

本次采用“单 socket → 主机全局 → 全连接归因 → TCP 扩展计数 → PMTU/接口 → 双端抓包”的顺序。

### 3.1 找到实际活跃的 TCP 对端

先通过以下命令确认当前节点是 P 还是 D，以及真正的对端地址：

```bash
ss -Htnp | grep ':31225'
```

在 `10.44.132.30` 上观察到：

```text
10.44.132.30:<随机端口> -> <D IP>:312xx
```

因此 `10.44.132.30` 是客户端/发送端 P。曾误将本机 IP 作为诊断脚本的 peer，`ip route get` 显示：

```text
local 10.44.132.30 dev lo src 10.44.132.30
```

这只会诊断 loopback，不能反映 P/D 网络。最终脚本增加了本机 IP 防误用检查。

### 3.2 定时窗口采集

脚本在采样开始和结束时分别保存：

- `ss -tinmp`：每 socket 的 bytes、重传段、RTT、cwnd、PMTU、MSS；
- `nstat -az`：主机 TCP/IP MIB 绝对计数；
- `ip -j -s link`：路由接口流量和 drop/error；
- `tc -s qdisc`：qdisc 队列统计；
- `ethtool -S`：网卡 drop/error/PFC/ECN 等；
- `/proc/net/softnet_stat`：内核收包 backlog drop 和 time_squeeze；
- PMTU ping；
- 可选的 tcpdump。

所有结论使用采样窗口内的增量，避免把服务启动以来的累计值当成当前重传率。

## 4. 一键诊断脚本

将以下脚本保存为 `/tmp/tcp_retrans_diag.sh`：

```bash
#!/usr/bin/env bash

set -u

PEER_IP="${1:-}"
PORT="${2:-}"
DURATION="${3:-60}"
CAPTURE="${CAPTURE:-0}"

if [[ -z "${PEER_IP}" ]]; then
    echo "Usage: $0 <peer_ip> [tcp_port] [duration_seconds]"
    echo "Example: $0 10.44.172.215 31218 60"
    echo "Capture: CAPTURE=1 $0 10.44.172.215 31218 15"
    exit 1
fi

ROUTE=$(ip route get "${PEER_IP}" 2>&1 | head -n 1)

if [[ "${ROUTE}" == local\ * ]]; then
    echo "ERROR: ${PEER_IP} is a local IP."
    echo "Please specify the remote P/D peer IP."
    exit 2
fi

IFACE=$(awk '
{
    for (i = 1; i <= NF; i++) {
        if ($i == "dev") {
            print $(i + 1)
            exit
        }
    }
}' <<<"${ROUTE}")

SRC_IP=$(awk '
{
    for (i = 1; i <= NF; i++) {
        if ($i == "src") {
            print $(i + 1)
            exit
        }
    }
}' <<<"${ROUTE}")

TS=$(date '+%Y%m%d-%H%M%S')
SAFE_PEER=${PEER_IP//:/_}
OUT="/tmp/tcp-retrans-${SAFE_PEER}-${PORT:-all}-${TS}"
mkdir -p "${OUT}"

export DIAG_PEER="${PEER_IP}"
export DIAG_PORT="${PORT}"
export DIAG_OUT="${OUT}"

echo "Output directory : ${OUT}"
echo "Peer             : ${PEER_IP}"
echo "Port             : ${PORT:-all}"
echo "Duration         : ${DURATION}s"
echo "Route            : ${ROUTE}"
echo "Source IP        : ${SRC_IP:-unknown}"
echo "Interface        : ${IFACE:-unknown}"
echo "Packet capture   : ${CAPTURE}"
echo

printf '%s\n' "${ROUTE}" >"${OUT}/route.txt"

snapshot()
{
    local tag="$1"

    date '+%F %T.%N %z' >"${OUT}/time.${tag}"

    if command -v ss >/dev/null 2>&1; then
        ss -tinmp >"${OUT}/ss.${tag}" 2>&1
    fi

    if command -v nstat >/dev/null 2>&1; then
        nstat -az >"${OUT}/nstat.${tag}" 2>&1
    fi

    if [[ -n "${IFACE:-}" ]]; then
        ip -j -s link show dev "${IFACE}" \
            >"${OUT}/link.${tag}.json" 2>&1 || true

        if command -v tc >/dev/null 2>&1; then
            tc -s qdisc show dev "${IFACE}" \
                >"${OUT}/qdisc.${tag}" 2>&1 || true
        fi

        if command -v ethtool >/dev/null 2>&1; then
            ethtool -S "${IFACE}" \
                >"${OUT}/ethtool.${tag}" 2>&1 || true
        fi
    fi

    cp /proc/net/softnet_stat \
        "${OUT}/softnet.${tag}" 2>/dev/null || true
}

echo "Collecting baseline..."
snapshot before

TCPDUMP_PID=""

if [[ "${CAPTURE}" == "1" ]] &&
   command -v tcpdump >/dev/null 2>&1 &&
   command -v timeout >/dev/null 2>&1 &&
   [[ -n "${IFACE:-}" ]]; then

    PCAP_FILTER="host ${PEER_IP} and tcp"

    if [[ -n "${PORT}" ]]; then
        PCAP_FILTER="${PCAP_FILTER} and port ${PORT}"
    fi

    timeout "${DURATION}s" tcpdump \
        -i "${IFACE}" -nn -s 128 -B 16384 \
        -w "${OUT}/tcp.pcap" "${PCAP_FILTER}" \
        >"${OUT}/tcpdump.log" 2>&1 &

    TCPDUMP_PID=$!
fi

echo "Monitoring for ${DURATION}s..."
sleep "${DURATION}"
snapshot after

if [[ -n "${TCPDUMP_PID}" ]]; then
    wait "${TCPDUMP_PID}" 2>/dev/null || true
fi

{
    echo "Route:"
    ip route get "${PEER_IP}" 2>&1 || true

    echo
    echo "Normal ping:"
    ping -c 3 -W 1 "${PEER_IP}" 2>&1 || true

    echo
    echo "MTU 1500 test, payload 1472:"
    ping -M do -c 3 -W 1 -s 1472 "${PEER_IP}" 2>&1 || true

    echo
    echo "MTU 9000 test, payload 8972:"
    ping -M do -c 3 -W 1 -s 8972 "${PEER_IP}" 2>&1 || true

    if command -v tracepath >/dev/null 2>&1; then
        echo
        tracepath -n "${PEER_IP}" 2>&1 || true
    fi
} >"${OUT}/pmtu.txt"

python3 <<'PY' | tee "${OUT}/report.txt"
import json
import os
import re
from pathlib import Path

peer = os.environ["DIAG_PEER"]
port = os.environ.get("DIAG_PORT", "")
out = Path(os.environ["DIAG_OUT"])


def read_text(name):
    try:
        return (out / name).read_text(errors="replace")
    except Exception:
        return ""


def split_endpoint(endpoint):
    endpoint = endpoint.strip()
    if endpoint.startswith("["):
        pos = endpoint.rfind("]:")
        host = endpoint[1:pos]
        endpoint_port = endpoint[pos + 2:]
    else:
        host, _, endpoint_port = endpoint.rpartition(":")
    if host.startswith("::ffff:"):
        host = host[len("::ffff:"):]
    return host, endpoint_port


def endpoint_matches(local, remote):
    local_host, local_port = split_endpoint(local)
    remote_host, remote_port = split_endpoint(remote)
    if peer not in (local_host, remote_host):
        return False
    return not port or port in (local_port, remote_port)


def parse_ss(name):
    records = {}
    current = None

    for raw in read_text(name).splitlines():
        line = raw.strip()
        if re.match(r"^(ESTAB|CLOSE-WAIT|SYN-SENT|SYN-RECV|FIN-WAIT|LAST-ACK)\s", line):
            parts = line.split()
            if len(parts) < 5:
                current = None
                continue
            state, local, remote = parts[0], parts[3], parts[4]
            if state == "ESTAB" and endpoint_matches(local, remote):
                current = f"{local} -> {remote}"
                records[current] = {
                    "bytes_sent": 0,
                    "bytes_retrans": 0,
                    "data_segs_out": 0,
                    "retrans_segs": 0,
                    "rtt": "",
                    "cwnd": 0,
                    "pmtu": 0,
                    "mss": 0,
                }
            else:
                current = None
            continue

        if current is None:
            continue

        patterns = {
            "bytes_sent": r"\bbytes_sent:(\d+)",
            "bytes_retrans": r"\bbytes_retrans:(\d+)",
            "data_segs_out": r"\bdata_segs_out:(\d+)",
            "pmtu": r"\bpmtu:(\d+)",
            "mss": r"\bmss:(\d+)",
            "cwnd": r"\bcwnd:(\d+)",
            "rtt": r"\brtt:([0-9.]+/[0-9.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                records[current][key] = value if key == "rtt" else int(value)

        match = re.search(r"\bretrans:(\d+)/(\d+)", line)
        if match:
            records[current]["retrans_segs"] = int(match.group(2))

    return records


def parse_nstat(name):
    result = {}
    for line in read_text(name).splitlines():
        fields = line.split()
        if len(fields) >= 2 and re.fullmatch(r"-?\d+", fields[1]):
            result[fields[0]] = int(fields[1])
    return result


def delta(after, before, key):
    return max(after.get(key, 0) - before.get(key, 0), 0)


before = parse_ss("ss.before")
after = parse_ss("ss.after")
common = sorted(set(before) & set(after))

print("=" * 78)
print("PER-SOCKET TCP DELTAS")
print("=" * 78)

total_sent = 0
total_retrans = 0
total_retrans_segs = 0

if not common:
    print("No persistent ESTAB socket matched the requested peer and port.")

for connection in common:
    sent = delta(after[connection], before[connection], "bytes_sent")
    retrans = delta(after[connection], before[connection], "bytes_retrans")
    segments = delta(after[connection], before[connection], "retrans_segs")
    rate = retrans * 100.0 / sent if sent else 0.0
    total_sent += sent
    total_retrans += retrans
    total_retrans_segs += segments

    print(f"\n{connection}")
    print(f"  bytes_sent_delta       = {sent:,}")
    print(f"  bytes_retrans_delta    = {retrans:,}")
    print(f"  retransmitted_segments = {segments:,}")
    print(f"  retrans_byte_rate      = {rate:.4f}%")
    print(f"  rtt                    = {after[connection].get('rtt') or 'unknown'} ms")
    print(f"  cwnd                   = {after[connection].get('cwnd') or 'unknown'}")
    print(f"  pmtu / mss             = {after[connection].get('pmtu') or 'unknown'} / {after[connection].get('mss') or 'unknown'}")
    if sent == 0:
        assessment = "IDLE"
    elif rate >= 1.0:
        assessment = "SEVERE"
    elif rate >= 0.1:
        assessment = "SUSPICIOUS"
    elif retrans:
        assessment = "RETRANSMISSION OBSERVED"
    else:
        assessment = "OK"
    print(f"  assessment             = {assessment}")

total_rate = total_retrans * 100.0 / total_sent if total_sent else 0.0
print("\n" + "-" * 78)
print(f"TOTAL bytes sent          = {total_sent:,}")
print(f"TOTAL bytes retransmitted = {total_retrans:,}")
print(f"TOTAL retrans segments    = {total_retrans_segs:,}")
print(f"TOTAL retrans byte rate   = {total_rate:.4f}%")

nb = parse_nstat("nstat.before")
na = parse_nstat("nstat.after")
print("\n" + "=" * 78)
print("HOST TCP STACK DELTAS")
print("=" * 78)
pattern = re.compile(r"Retrans|Timeout|Lost|Sack|DSACK|OFO|Reorder|RcvQDrop|ListenDrop|ECN|InCE|Congestion|Abort", re.I)
for key in sorted(set(nb) | set(na)):
    value = max(na.get(key, 0) - nb.get(key, 0), 0)
    if value and pattern.search(key):
        print(f"{key:40s} {value:,}")

print(f"\nRaw data directory: {out}")
PY

echo
echo "Completed."
echo "Summary : ${OUT}/report.txt"
echo "Raw data: ${OUT}"
```

执行示例：

```bash
chmod +x /tmp/tcp_retrans_diag.sh
bash -n /tmp/tcp_retrans_diag.sh
/tmp/tcp_retrans_diag.sh 10.44.172.215 31218 60
```

双端短时抓包：

```bash
CAPTURE=1 /tmp/tcp_retrans_diag.sh 10.44.172.215 31218 15
```

## 5. 现场数据

### 5.1 初次选中的 peer 在采样窗口内空闲

最初针对 `10.44.172.222:31225` 采集 60 秒。8 条 socket 均显示：

```text
bytes_sent_delta       = 0
bytes_retrans_delta    = 0
retransmitted_segments = 0
rtt                    ≈ 0.19～0.26 ms
cwnd                   ≈ 391～609
pmtu / mss             = 1500 / 1448
```

这不能说明该链路“正常”，只能说明这些连接在该窗口内处于 `IDLE`。

与此同时，主机全局出现：

```text
TcpRetransSegs = 1,503,935
tx_bytes       = 82,973,428,847
tx_packets     = 1,873,715
softnet dropped/time_squeeze = 0/0
```

因此需要用同一批 `ss.before/after` 对所有连接归因。

### 5.2 全连接归因结果

窗口内主要 KVT/Barex TCP 流量如下：

| 对端 | 连接数 | 发送量 | 重传量 | 重传率 | 重传段 |
|---|---:|---:|---:|---:|---:|
| `10.44.172.215:31218` | 8 | 47.890 GB | 1.347 GB | 2.812% | 930,261 |
| `10.44.173.154:31222` | 8 | 11.864 GB | 0.319 GB | 2.691% | 220,535 |
| `10.44.173.154:31225` | 8 | 6.759 GB | 0.163 GB | 2.408% | 112,378 |
| `10.44.172.215:31219` | 8 | 6.283 GB | 0.197 GB | 3.137% | 136,135 |
| `10.44.173.154:31224` | 8 | 5.801 GB | 0.100 GB | 1.731% | 69,340 |
| `10.44.173.157:31221` | 8 | 3.959 GB | 0.046 GB | 1.163% | 31,792 |
| `10.44.172.222:31223` | 8 | 0.065 GB | 0.003 GB | 4.083% | 1,836 |
| `10.44.173.157:31219` | 8 | 0.049 GB | 0.001 GB | 2.043% | 685 |
| `10.44.173.157:31218` | 8 | 0.036 GB | 0.001 GB | 1.997% | 503 |

汇总：

```text
KVT 发送量       ≈ 82.706 GB
KVT 重传量       ≈ 2.177 GB
KVT 重传率       ≈ 2.63%
KVT 重传段       = 1,503,465
主机 TcpRetrans  = 1,503,935
```

KVT 连接贡献了约 99.97% 的主机 TCP 重传段，并贡献了约 99.7% 的接口发送字节。因此主机全局重传几乎全部来自 KVT/Barex TCP 大流量，而不是其他控制连接。

最大流量链路 `10.44.172.215:31218` 的 8 条连接均发生重传，单连接重传率约为：

```text
1.761%, 2.319%, 2.573%, 2.639%,
2.904%, 3.123%, 3.487%, 3.720%
```

`10.44.172.215:31219` 的多条连接甚至达到约 3.1%～4.1%。这不是单个坏 socket，而是跨 socket、跨端口、跨 D 节点普遍存在的问题。

### 5.3 TCP 扩展计数

同一窗口内的 TCP MIB 增量：

```text
TcpRetransSegs                           1,503,935
TcpExtTCPFastRetrans                    1,498,368
TcpExtTCPLostRetransmit                   172,717
TcpExtTCPTimeouts                              533
TcpExtTCPSackRecovery                         9,680
TcpExtTCPSackRecoveryFail                       68
TcpExtTCPSlowStartRetrans                     4,991
TcpExtTCPDSACKRecv                            5,389
TcpExtTCPDSACKIgnoredDubious                  5,381
TcpExtTCPSACKReorder                            192
TcpExtTCPTSReorder                                3
TcpExtTCPOFOQueue                                 7
TcpExtTCPSACKDiscard                          4,614
TcpExtTCPSackMerged                          73,276
TcpExtTCPSackShiftFallback                  28,721
TcpExtTCPSackShifted                        18,100
TcpExtTCPSynRetrans                            462
```

关键比例：

```text
FastRetrans / RetransSegs     ≈ 99.63%
LostRetransmit / RetransSegs  ≈ 11.48%
DSACKRecv / RetransSegs       ≈ 0.36%
```

解释：

- 几乎全部重传通过 Fast Retransmit 触发，接收端通过重复 ACK/SACK 明确报告中间存在缺口；
- `TCPLostRetransmit` 很高，说明相当数量的重传段也再次被判断丢失，符合突发丢包或拥塞队列持续丢包；
- `TCPTimeouts=533` 数量相对小，但每次 RTO 都可能带来约 200ms 级停顿，是秒级长尾的重要放大因素；
- DSACK、SACK reorder、timestamp reorder、OFO queue 相比 150 万重传极少，因此“严重乱序导致的伪重传”不是主因；
- `TCPDSACKIgnoredDubious` 接近 `TCPDSACKRecv`，这些 DSACK 大部分没有被内核接受为可以撤销拥塞恢复的可靠证据。

### 5.4 PMTU 与接口状态

路由：

```text
10.44.172.222 via 10.44.159.253 dev eth0 src 10.44.132.30
```

PMTU 测试：

```text
1472-byte ICMP payload：成功
8972-byte ICMP payload：message too long, mtu=1500
ss：pmtu=1500, mss=1448
```

结论：TCP 路径使用 MTU 1500，MSS 1448。`ACCL_IBV_MTU=9000` 不会改变该 TCP 路径，它只影响 IBV/RDMA 配置；要使用 TCP Jumbo Frame，必须保证容器 veth、宿主机 bond/物理网卡、交换机和端到端路径全部支持并配置相同 MTU。

容器内观察到：

```text
softnet dropped       = 0
softnet time_squeeze  = 0
eth0 drop/error delta = 0
```

这说明容器网络栈没有明显 backlog drop，但 `eth0` 很可能只是容器虚拟网卡；它不能排除宿主机 vSwitch、物理 NIC、QoS policer 或交换机队列丢包。

### 5.5 15 秒定向抓包窗口没有命中活跃流

随后在 P 节点执行：

```bash
CAPTURE=1 /tmp/tcp_retrans_diag.sh 10.44.172.215 31218 15
```

采样结果中，`.172.215:31218` 的 8 条 TCP 连接均为：

```text
bytes_sent_delta       = 0
bytes_retrans_delta    = 0
retransmitted_segments = 0
```

但同一 15 秒窗口的主机全局数据为：

```text
tx_bytes        = 26,937,035,371
TcpRetransSegs  = 368,269
softnet dropped = 0
time_squeeze    = 0
```

换算：

```text
平均发送速率 ≈ 26.937GB × 8 / 15s ≈ 14.37Gbps
按 MSS 1448 粗估重传字节 ≈ 368,269 × 1448 ≈ 0.533GB
粗估主机重传比例 ≈ 0.533 / 26.937 ≈ 1.98%
```

因此，这个新窗口说明：

1. 主机级高流量和约 2% 的重传仍在持续；
2. KVT 调度/路由已经将实际发送切换到其他 D 端口或节点；
3. 针对 `.172.215:31218` 的 pcap 没有覆盖活跃 data flow，不能用于定位该链路的丢包点；
4. 连接上显示的 RTT/cwnd 是 socket 最近一次状态，空闲窗口中的 `cwnd=75～740` 不能替代发送期样本；
5. 再次观察到容器 `eth0` 和 softnet 无 drop，仍需检查容器外的宿主机/交换机路径。

后续抓包不能只依赖上一个窗口的热点端口。应先使用 `ss.before/after` 全连接归因找到当前活跃 peer，或者扩大抓包过滤范围。

针对某个 D IP 的所有 KVT TCP 端口：

```bash
CAPTURE=1 /tmp/tcp_retrans_diag.sh 10.44.172.215 "" 15
```

直接抓取 P 节点发往所有 D 的 KVT 端口范围：

```bash
timeout 15 tcpdump \
  -i eth0 -nn -s 128 -B 16384 \
  -w /tmp/kvt-all-31218-31225.pcap \
  'tcp and dst portrange 31218-31225'
```

定向抓包后必须先确认 pcap 确实包含 payload 流量：

```bash
ls -lh /tmp/tcp-retrans-*/tcp.pcap
tcpdump -nn -r /path/to/tcp.pcap 2>/dev/null | head
```

## 6. 最终结论

### 6.1 已确认的结论

1. **KVT/Barex TCP 链路存在真实且严重的重传。** 采样窗口内整体约 2.63%，单连接约 1%～4%。
2. **重传不是单 socket 或单 D worker 问题。** 多个 D IP、端口和 8 条并行连接均出现类似重传。
3. **重传几乎全部由 KVT 大流量贡献。** KVT 重传段占主机 `TcpRetransSegs` 约 99.97%。
4. **主要不是乱序/伪重传。** 99.63% 是 Fast Retransmit，而 DSACK/Reorder/OFO 证据远小于重传数量，指向实际数据缺口。
5. **存在重复丢包和少量 RTO。** `TCPLostRetransmit=172,717`、`TCPTimeouts=533` 足以造成 Barex TCP body 读取和同 channel 后续消息的长尾。
6. **`ACCL_TCP_WORK_THREAD_CNT=32` 不解决该问题。** 它只能缓解不同 channel 共享阻塞发送 worker，不能减少 kernel TCP 重传。
7. **TCP 实际 PMTU 是 1500。** `ACCL_IBV_MTU=9000` 与当前 Barex TCP 路径无关。
8. **D 侧 callback/H2D 排队是另一个独立问题。** 之前观察到较高 `OnRecvQueueUs`，它由 `BLLM_KVTRANS_CTX_TPSIZE`、callback 数量和 GPU copy stream 竞争决定，不能与 TCP 重传混为一谈。

### 6.2 最可能的根因方向

P 节点 60 秒发送约 82.97GB，平均约：

```text
82.97 GB × 8 / 60s ≈ 11.06 Gbps
```

多个目的端同时重传，说明应优先怀疑发送端的公共出口链路，包括：

- 实例/容器存在约 10Gbps 带宽上限，当前流量触发宿主机 vSwitch/QoS policer；
- P 宿主机物理 NIC 或上联交换机 egress queue 拥塞；
- 多个 P 同时向 D 发送导致 D 侧或 ToR 发生 incast；
- 网络路径上的 buffer/drop/ECN/PFC 配置问题。

“平均 11Gbps 接近或超过潜在 10Gbps 上限”目前是高优先级假设，但仍需查询实例带宽规格、宿主机和交换机计数后才能最终确认。

## 7. 后续验证与优化优先级

### 7.1 降低 offered load 做 A/B

保持软件版本和线程配置不变，将请求并发或 KVT 发送流量降低约 30%，目标降至 7～8Gbps，再采样 60 秒。

如果重传率从 2%～3% 降至接近 0，基本可以确认是出口限速或网络队列拥塞。该实验比继续调整 Barex worker 数更有判别力。

### 7.2 检查宿主机和交换机

容器侧：

```bash
ethtool eth0 2>/dev/null | grep -Ei 'Speed|Duplex|Link detected'
tc -s qdisc show dev eth0
ip -s link show dev eth0
```

宿主机物理接口/bond 重点查看：

```text
tx/rx drop
discard/error/missed/no-buffer
qdisc backlog/drop/overlimit
pause/PFC
ECN/congestion mark
policer/shaper drop
NIC ring miss
```

交换机需要检查 P 宿主机出口和各 D 宿主机入口端口的：

```text
ingress/egress discard
queue/buffer drop
ECN mark
PFC/pause
端口利用率和突发流量
```

### 7.3 P/D 双端抓包

选择最大流量链路 `10.44.132.30 → 10.44.172.215:31218`，在 P、D 同时抓 15 秒：

P：

```bash
CAPTURE=1 /tmp/tcp_retrans_diag.sh 10.44.172.215 31218 15
```

D (`10.44.172.215`)：

```bash
CAPTURE=1 /tmp/tcp_retrans_diag.sh 10.44.132.30 31218 15
```

判断：

| P 抓包 | D 抓包 | 结论 |
|---|---|---|
| 有原始包 | 无原始包、后续只有重传包 | 中间网络/交换机真实丢包 |
| 有原始包 | 有原始包但严重晚到/乱序 | 路径乱序导致伪重传 |
| 有原始包 | D NIC/host drop 同时增长 | D 宿主机/NIC 丢包 |
| P 侧 qdisc/policer drop 增长 | D 无原始包 | P 宿主机出口限速/丢包 |

抓包时需要注意 TSO/GSO 和 checksum offload 可能使单端 Wireshark 产生假象，最终应通过 P/D 两份 pcap 的 TCP sequence number 对齐判断。

### 7.4 软件侧优化顺序

1. 先解决网络重传或带宽限速；
2. 再观察 D 侧 `OnRecvQueueUs`，对 `BLLM_KVTRANS_CTX_TPSIZE=12/24` 做 A/B；
3. 暂时不要因为发送长尾继续增加 `BLLM_KVTRANS_RDMA_SP`，更多并行 TCP flow 可能加剧 incast；
4. `ACCL_TCP_WORK_THREAD_CNT=32` 可以保留做隔离实验，但它不是当前 2%～3% 重传的修复；
5. 中长期可为 Barex 增加 sender-local 的 queue/start/header/body 时间点，避免依赖跨机时钟计算 `LinkTxUs`；
6. 适度合并 layer 消息（例如 2～4 层、16～32MB）可以减少 callback 和消息数量，但不能替代网络丢包修复；一次合并成约 300MB 会降低流水、放大 HOL、增加 pinned memory 和失败重试粒度，不建议作为首选。

## 8. 复用：对已有 ss 快照进行全连接归因

主诊断脚本如果选中了空闲 peer，可以直接使用已有 `ss.before/after` 找出真正活跃的远端，无需重新等待采样窗口。核心方法是按 `(local, remote)` 匹配两份快照，计算：

```text
delta_bytes_sent
delta_bytes_retrans
delta_retrans_segments
```

然后按 remote IP:port 聚合。此次正是通过该方法发现：

- `.172.222:31225` 在窗口内空闲；
- 真正的主流量是 `.172.215:31218`；
- 82.7GB KVT 流量与 150 万重传段几乎完全解释了主机全局计数。

完整离线归因脚本如下。将 `DIR` 指向任意一次诊断输出目录即可，无需重新采样：

```bash
DIR=/tmp/tcp-retrans-10.44.172.222-31225-20260716-222536

python3 - "$DIR" <<'PY'
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])


def parse(path):
    result = {}
    current = None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()

        if re.match(
            r"^(ESTAB|CLOSE-WAIT|SYN-SENT|SYN-RECV|FIN-WAIT|LAST-ACK)\s",
            line,
        ):
            fields = line.split()
            if len(fields) >= 5 and fields[0] == "ESTAB":
                current = (fields[3], fields[4])
                result[current] = {
                    "bytes_sent": 0,
                    "bytes_retrans": 0,
                    "retrans_segs": 0,
                }
            else:
                current = None
            continue

        if current is None:
            continue

        for key, pattern in {
            "bytes_sent": r"\bbytes_sent:(\d+)",
            "bytes_retrans": r"\bbytes_retrans:(\d+)",
        }.items():
            match = re.search(pattern, line)
            if match:
                result[current][key] = int(match.group(1))

        match = re.search(r"\bretrans:(\d+)/(\d+)", line)
        if match:
            result[current]["retrans_segs"] = int(match.group(2))

    return result


def split_endpoint(endpoint):
    if endpoint.startswith("["):
        pos = endpoint.rfind("]:")
        host, port = endpoint[1:pos], endpoint[pos + 2:]
    else:
        host, _, port = endpoint.rpartition(":")
    if host.startswith("::ffff:"):
        host = host[len("::ffff:"):]
    return host, port


before = parse(root / "ss.before")
after = parse(root / "ss.after")
peers = defaultdict(lambda: {
    "connections": 0,
    "sent": 0,
    "retrans": 0,
    "retrans_segs": 0,
})

for socket in set(before) & set(after):
    local, remote = socket
    remote_host, remote_port = split_endpoint(remote)
    sent = max(after[socket]["bytes_sent"] - before[socket]["bytes_sent"], 0)
    retrans = max(
        after[socket]["bytes_retrans"] - before[socket]["bytes_retrans"],
        0,
    )
    retrans_segs = max(
        after[socket]["retrans_segs"] - before[socket]["retrans_segs"],
        0,
    )

    if sent == 0 and retrans == 0:
        continue

    peer = f"{remote_host}:{remote_port}"
    peers[peer]["connections"] += 1
    peers[peer]["sent"] += sent
    peers[peer]["retrans"] += retrans
    peers[peer]["retrans_segs"] += retrans_segs

print(
    f"{'peer':30s} {'conn':>5s} {'sent_GB':>12s} "
    f"{'retrans_GB':>14s} {'rate':>10s} {'retrans_seg':>14s}"
)

for peer, stats in sorted(
    peers.items(),
    key=lambda item: item[1]["sent"],
    reverse=True,
):
    rate = (
        stats["retrans"] * 100.0 / stats["sent"]
        if stats["sent"]
        else 0.0
    )
    print(
        f"{peer:30s} "
        f"{stats['connections']:5d} "
        f"{stats['sent'] / 1e9:12.3f} "
        f"{stats['retrans'] / 1e9:14.3f} "
        f"{rate:9.3f}% "
        f"{stats['retrans_segs']:14,d}"
    )
PY
```

## 9. 结论摘要

```text
现象：Blade-KVT/Barex LinkTxUs、RecvUs、TransUs 出现百毫秒至秒级长尾

直接证据：
  KVT 发送量约 82.706GB / 60s
  KVT 重传量约 2.177GB
  KVT 重传率约 2.63%
  KVT 重传段 1,503,465，占主机重传约 99.97%
  Fast Retransmit 占总重传约 99.63%
  Lost Retransmit 172,717，Timeout 533
  DSACK/Reorder/OFO 远小于重传量

最终判断：
  当前存在跨多个 D 节点和 TCP flow 的真实网络丢包/拥塞；
  不是单个 Barex worker、单 socket 或主要由乱序导致；
  TCP PMTU 为 1500，ACCL_IBV_MTU=9000 不影响该链路；
  ACCL_TCP_WORK_THREAD_CNT 从 16 增至 32 无法修复网络重传；
  应优先检查 P 公共出口带宽上限、宿主机 vSwitch/QoS、物理 NIC、ToR queue/drop，
  并通过降载 A/B 与 P/D 双端抓包确认具体丢包点。
```
