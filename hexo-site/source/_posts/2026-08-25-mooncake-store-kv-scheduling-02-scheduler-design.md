---
title: "02 · 调度器设计"
date: 2026-08-25
categories: [Mooncake Store 与 cache-aware 调度]
tags: [Mooncake, KV Cache, PD 分离, cache-aware 调度, vLLM, 驱逐策略, 学习笔记]
---

# 02 · 调度器设计

调度器是一个独立进程（`runs/mcstore/scheduler.py`），对客户端讲 OpenAI 协议，
内部挑一对 (Prefill, Decode) 然后只把请求转发给 D，靠 `kv_transfer_params` 里的
`remote_host` + `remote_port` 把 P 钉死。策略藏在 `SchedulingPolicy` 抽象后面，
`--policy` 切换：`round_robin` / `random` / `least_inflight` / `prefix_affinity` / `cache_aware_score`。

## 1. 亲和索引：调度器怎么知道谁持有哪段前缀

这个信息没有现成来源，所以我们自己造了一个索引：

```python
# on_result()：请求成功后，把它的全部前缀 hash 指向被选中的节点
for _end, hsh in hashes:
    self._owners.setdefault(hsh, set()).add(p.name)
```

也就是**「谁被选中，谁就声称持有」**——乐观假设，不校验。dashscope 网关同构，只是多了 900s TTL。

这个索引有三个缺陷，按发现顺序：

1. **驱逐一旦真实发生就有静默假阳性**：索引说 X 持有 P，X 其实早淘汰了。
2. **错误会自我维持**：写回是无条件的，「被误导而路由到 X」这件事本身又把声称重写一遍、
   刷新 TTL。**一条过期记录能靠它自己引起的误判无限期存活。**
3. **探测即续命**：`batch_is_exist` 会授予 10s 租约（见 [01 章](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-01-store-architecture/)），
   所以每次探测都在给对象延寿、豁免驱逐。

实测这三个缺陷现在最多值 2 个百分点，而噪声下限就是 2%，所以修它不是当前优先级——
但它是理解后面所有实验的必要背景。

## 2. 几何前缀网格

### 改动前：线性 512 字符网格

prompt 切 512 字符定长 chunk 做累积哈希，`hash[i]` 覆盖 chunk 0..i，匹配的 chunk 个数即亲和分。
26k token 需要 152 个 key，**1M token 需要 7,558 个**——索引表和 CPU 都会爆。

### 改动后：几何网格

![线性网格 vs 几何网格](/imgs/mcstore_prefix_grid.svg)

```python
def _boundaries(self, n):            # min_chars=512 起翻倍
    end = self._min_chars            # dense_from=4096 之后每八度切 steps_per_octave=4 份
    while end <= n:
        out.append(end)
        if end >= self._dense_from:
            for i in range(1, self._steps_per_octave):
                extra = end + end * i // self._steps_per_octave
                if extra <= n:
                    out.append(extra)
        end *= 2

def _prefix_hashes(self, text):      # 一次线性扫描 + 每边界克隆 digest
    h = hashlib.blake2b(digest_size=16)
    pos = 0
    for end in self._boundaries(len(text)):
        h.update(text[pos:end].encode())
        pos = end
        out.append((end, h.copy().hexdigest()))
```

三个连带的**必要**改动，少一个就错：

1. **打分单位从「chunk 个数」变成「匹配字符数」**（`_match_chars`）。
   几何网格下 key 数与长度是 log 关系，个数不再正比于长度。
2. **扫描方向翻转**：从最长边界往下扫，且 miss 用 `continue` 而不是 `break`。
   嵌套前缀下「长的没命中」不代表「短的没命中」，而最长命中天然包含所有更短的
   （这就是 dashscope 的 max-over-keys 语义）。
3. **增量哈希**：`h.copy()` 让总哈希量保持 O(n)，而非每边界从头重算的 O(n log n)。
   dashscope 用 guava `hashBytes(b, 0, len)` 付的就是 O(n log n)，**这一点我们比它好**。
   实测 3 MB prompt：增量 2.3 ms vs 每次重算 7.9 ms。

### 为什么默认 `steps_per_octave = 4` 而不是 dashscope 的 2

这是实测逼出来的，不是拍的。**agentic 负载把判别信息全放在共享系统前缀之后——
也就是大位置，而几何网格恰在大位置最粗。**

用真实 trace 的形状（35k 字符共享前缀，每轮 +14k）测「落在 35k 之后的可区分边界数」：

| | 60k prompt | 150k prompt |
|---|---|---|
| k=1 | **0 个（亲和全瞎）** | 2 个 |
| k=2 | 1 个 | 4 个 |
| k=4 | 3 个 | 8 个 |

k=1 时 60k 的 prompt 上亲和是完全瞎的。24 trial × 8 轮的离线重放也印证：
平均命中比例 k=1 是 0.61，k=4 是 0.77，而路由分散度 k=4 反而更好。

### 收益

| 上下文 | 字符 | 几何 key | 线性 key | 倍数 | 哈希耗时 |
|---|---|---|---|---|---|
| 26k | 100,620 | 22 | 196 | 9× | 0.1 ms |
| 85k | 328,950 | 29 | 642 | 22× | 0.4 ms |
| 256k | 1,014,497 | 35 | 1,981 | 57× | 0.9 ms |
| 1M | 3,870,000 | 43 | 7,558 | **176×** | 3.7 ms |

附带好处：索引表从每请求 642 个条目降到 29 个。实测 433 个请求写入约 12,500 次前缀，
去重后 `tracked_prefixes` 只有 572 个。

### 一个残余偏差

`match_chars / len(prompt)` 这个比例被网格系统性低估，幅度随 n 在 `[1/(1+1/k), 1]`
即 k=4 时 `[0.8, 1.0]` 之间振荡。**同一请求内所有候选同网格，所以排序不受影响**；
但一旦进入跨请求比较（指数打分的 ratio），必须改用 `boundaries[-1]` 作分母消除它。

## 3. 从硬阈值到指数打分

### 硬阈值版本（`prefix_affinity`）

最初的护栏是「亲和节点在飞的请求比最闲的多 `load_slack` 个就让位」。**它从不触发**：
轮次间有 5~7 秒 think time，瞬时 inflight 几乎恒为 0 或 1，条件永不成立 →
亲和永远赢 → 流量雪球到「首次落地」的节点（7 节点里 43 vs 7）。

改成相对阈值也不行：`inflight[idlest] = 0` 时均值 ≈ 0.3，1.5 倍 ≈ 0.5，
任何 `inflight ≥ 1` 都被拒 → **从「永不触发」跳到「永远触发」，中间没有可用区间**。
根因是**阈值是一个只取几个小整数的变量上的阶跃函数**。

最终形式改成约束「衰减的累积分配量」：

```python
load[n] = 衰减累积分配数 + 当前 inflight      # 每次决策 *= (1 - 1/decay_window), window=64
跳过亲和节点 if load[best] > load_ratio * max(mean(load), 1.0)
```

`max(mean, 1.0)` 的下限防止冷启动时均值 ≈ 0 导致过度约束。`load_ratio ≈ 1.5` 最优。

### 指数打分版本（`cache_aware_score`）

借鉴 dashscope 网关的 `CacheAwareBalancedRouteSelector`：

```python
weight[n] = 2 ** (hit_weight * hit_ratio[n] + load_weight * (mean_load - load[n]))
# 从 topk = max(sqrt(N), 5) 个最重的节点里按 weight 加权采样
```

指数形式的意义是 `weight[a] / weight[b] = 2^(score[a] - score[b])`——**只依赖分数之差**。
给所有节点的 load 加同一个常数会在差里抵消，所以**对负载的整体平移免疫**。
这正是硬阈值缺的性质：阈值比较的是绝对值，而绝对值随负载水平变化。

采样而非 argmax 也有理由：argmax 下所有会话都挑同一个节点，而负载信号更新太慢来不及打破平局；
采样让节点概率随负载平滑下降，是个自限反馈环而不是悬崖。

### 两个权重不是一个自由度

这是实测发现的，很关键：

```
20/1 和 10/0.5 的 K 都是 20，但 routes 42/3 vs 36/9、argmax 一致率 0.89 vs 0.68
→ 比值 K = hit_weight/load_weight   决定权衡点（一次满命中值几个请求的排队余量）
→ 绝对幅度                          决定采样锐度（相当于温度）
```

`_load` 的量纲：均值 ≈ `decay_window / N` ≈ 8.7，节点间极差 ≈ 6。
所以 `load_weight=1` 时负载项幅度是 `2^±3`，与 `hit_weight=10 × ratio≈0.8 = 8` 同量级
——这解释了 dashscope 为什么标定成 `hit=10`。

离线扫描（stub 掉 aiohttp 直接 exec scheduler.py，几秒一个点）：

| hit/load | K | routes 最多/最少 | hit | argmax 一致率 |
|---|---|---|---|---|
| 20 / 1 | 20 | 42/3 | 185 | 0.89 |
| 10 / 0.5 | 20 | 36/9 | 183 | 0.68 |
| 10 / 1 | 10 | 32/17 | 180 | 0.81 |
| 10 / 2 | 5 | 31/24 | 171 | 0.83 |
| 10 / 4 | 2.5 | 30/26 | 165 | 0.84 |
| 1 / 1 | 1 | 28/26 | 97 | 0.34 |

整条前沿**单调平滑**——这正是硬阈值缺的东西，它只有「永不触发」和「永远触发」两个点。

## 4. 参考对象：dashscope 网关的 cache-aware 实现

代码在 `dashscope-platform/dashscope-api/dashscope-api-component/src/main/java/com/alibaba/dashscope/api/`，
三个变体：`CacheAwareRouteSelector` / `CacheAwarePureRouteSelector` / `CacheAwareBalancedRouteSelector`。

**bucket = turbo 实例 id**（不是哈希桶）。turbo 启动时用 `compare_and_set.lua` 抢一个 instanceId 锁，
抢到就是自己的 bucket。所以「选 bucket」= 「选实例」。

**前缀索引**：`PrefixUtils.hashTokens` 不做真 tokenize，用 `bytesPerToken=3` 估算，
对前缀长度取 2 的幂逐个哈希（`minPrefixTokenLength=64` 停），
`hashTokensWithDenseSteps` 从 512 起插中点。key 存成 **Redis ZSET**，成员是 bucket id
（**score 恒为 1，这个字段被浪费了**——它本该放时间戳），TTL 900 秒。
`PrefixCacheManager` **完全没有删除接口**，只有 `update`(ZADD) / `query`(ZRANGE) + TTL。

**打分**：默认 `prefixMatchWeight=10`、`loadWeight=40`，其余 5 项为 0。
`loadRatio = curLoad / maxLoad(40)`，所以指数化简成：

```
指数 = 10 × hitRatio + (比平均少几个在跑的请求)
```

**两个权重是标定过的：一次 100% 命中正好值 10 个请求的排队余量。**
临界点是亲和节点比其他节点多跑 9 个请求时优势抵平。
`topK = max(sqrt(实例数), 5)`，`samplingWithWeight` 按归一化权重随机采样。

**投递**：`push_request_cache_aware.lua` 用 `ZADD queue <bucket_id> <request>`（score 就是实例 id）；
`fetch_request_cache_aware.lua` 出队时 `ZRANGEBYSCORE queue min max LIMIT 0 1` 只拿自己的，
拿不到且 `ZCARD > requestSeizeThreshold(2)` 才 `ZRANGE 0 0` 抢别人的并标记 `hit=0`。
**饥饿保护做在消费端而非生产端**，这是整套设计里最值得借鉴的一点。

**bucket 区间 = 一致性哈希环**：每个存活实例负责 `[前一个存活实例 id + 1, 自己的 id]`，
环形回绕时拆成 primary + secondary，实例挂掉后 bucket 自动被下一个存活实例吸收。

**硬过滤在打分之前**：`filterPrefillBatchSize` / `filterPrefillTokenNum` /
`filterCacheFirstQueueLimit`，剔空了就退化成 least-loaded。

最后这一点纠正了我们的一个设计错误：**指数打分不是它的过载保护**——
渐变交给打分，饱和交给前置硬过滤 + 消费端抢占。我们最初想让一个阈值同时干这两件事，
那是设计上的混淆。
