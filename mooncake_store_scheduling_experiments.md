# Mooncake Store：cache-aware 调度与驱逐实验记录

配套文档：`mooncake_store_progress.md`（backend 实现、接线、环境配方、hybrid M1~M6）。
本文只记**调度策略与驱逐**这条线：做了什么实验、结论如何被推翻、最后站得住的是什么。

顺带回填了 progress 文档里两个「待查」项：
- §2.10「Mooncake 自己的驱逐策略 [未验证算法]」→ 本文 §7.2 已读源码定论
- §6.4「`batch_is_exist` 是否授予 lease —— 待查」→ 本文 §7.1 **答案是会，TTL 10s**

---

## 1. 问题的演化

起点是一个具体问题：既然有了共享 store，**PD 分离下要不要做 cache-aware 的 P 节点选择**？

朴素直觉是要：把请求发给已经持有这段前缀的 P 节点，能省一次 store 读。
但四轮实验下来，问题被逐步改写成了三个更精确的问题：

1. cache-aware 调度的收益取决于什么？（答：**选错节点的代价**，而不是命中率高低）
2. 什么条件下这个代价才够大？（答：store 读昂贵时——窄带宽，或 store 饱和到不保证命中）
3. 我们之前测到的收益是真的吗？（答：**四次结论里三次是基础设施缺陷伪装的**）

---

## 2. 环境与拓扑

```
test1 10.56.46.228   p0:8000(gpu0,1)  p1:8001(gpu2,3)  + mooncake master:50051
test2 10.56.47.155   p2:8000          p3:8001
test3 10.56.44.17    p4:8000          p5:8001
test4 10.56.46.239   p6:8000          d0:8100
```
4 × GB200(189GB) 每机 4 卡，共 16 卡 = 8 个 tp2 服务 = 7P + 1D。
共享盘 `/dashscope/caches/workspace/llx/`，实验目录 `runs/mcstore/`。

**跳板机约 1/3 概率失败**（`channel 0: open failed: Bad file descriptor`）。所有编排必须带重试，且
**绝不能 `2>/dev/null` 屏蔽 ssh 错误**——曾因此静默派发失败却显示成功。
test1 无法免密 ssh 到 test2/3/4，所以跨机编排只能从笔记本驱动。

### 模型（两个阶段）

| | 阶段一 | 阶段二 |
|---|---|---|
| 模型 | Qwen3-32B | qwen3-150b-a14b-256k-1106 |
| KV/token | 256 KiB (8 KV head, bf16) | **60 KiB** (4 KV head, fp8) |
| 权重 | bf16 原生 | **BF16 checkpoint + 在线 fp8 量化** |
| 上下文 | 26k (8 轮) | 46k p50 / 87k max (24 轮) |

qwen3-150b 的关键配置事实（详见 §9.1）：非 hybrid（`Qwen3MoeForCausalLM`）所以 store 可用；
`max_position_embeddings` 只有 35840 但 `rope_theta=1e7`，靠
`--hf-overrides '{"max_position_embeddings":262144}'` 开到 256k；fp8 **不需要** BLADNN。

---

## 3. 代码改动

### 3.1 `scheduler.py`：几何前缀网格

**改动前**：prompt 切 512 字符定长 chunk 做累积哈希，`hash[i]` 覆盖 chunk 0..i，
匹配的 chunk 个数即亲和分。26k token 需要 152 个 key，1M token 需要 **7,558 个**。

**改动后**：嵌套前缀的几何网格。

```python
def _boundaries(self, n):            # min_chars=512 起翻倍
    end = self._min_chars            # dense_from=4096 之后每八度切 steps_per_octave=4 份
    while end <= n:
        out.append(end)
        if end >= self._dense_from:
            for i in range(1, self._steps_per_octave):
                extra = end + end * i // self._steps_per_octave
                if extra <= n: out.append(extra)
        end *= 2

def _prefix_hashes(self, text):      # 一次线性扫描 + 每边界克隆 digest
    h = hashlib.blake2b(digest_size=16)
    pos = 0
    for end in self._boundaries(len(text)):
        h.update(text[pos:end].encode()); pos = end
        out.append((end, h.copy().hexdigest()))
```

三个连带的必要改动：

1. **打分单位从「chunk 个数」变成「匹配字符数」**（`_match_chars`）。几何网格下 key 数与长度是
   log 关系，个数不再正比于长度。
2. **扫描方向翻转**：从最长边界往下扫，且 miss 用 `continue` 而不是 `break`。嵌套前缀下
   「长的没命中」不代表「短的没命中」，最长命中天然包含所有更短的（= dashscope 的 max-over-keys 语义）。
3. **增量哈希**：`h.copy()` 让总哈希量保持 O(n)，而非每边界从头重算的 O(n log n)。
   dashscope 用 guava `hashBytes(b,0,len)` 付的就是 O(n log n)，这点我们更好。

**`steps_per_octave` 默认 4 而不是 dashscope 的 2**，这是实测逼出来的：
判别信息全在共享系统前缀**之后**（大位置），而几何网格恰在大位置最粗。
35k 字符共享前缀 + 每轮 14k 的形状下：

| | 60k prompt 的可区分边界 | 150k prompt |
|---|---|---|
| k=1 | **0 个（亲和全瞎）** | 2 个 |
| k=2 | 1 个 | 4 个 |
| k=4 | 3 个 | 8 个 |

平均命中比例 k=1 是 0.61，k=4 是 0.77。

**收益（vs 线性 512）**

| 上下文 | 字符 | 几何 key | 线性 key | 倍数 | 哈希耗时 |
|---|---|---|---|---|---|
| 26k | 100,620 | 22 | 196 | 9× | 0.1 ms |
| 85k | 328,950 | 29 | 642 | 22× | 0.4 ms |
| 256k | 1,014,497 | 35 | 1,981 | 57× | 0.9 ms |
| 1M | 3,870,000 | 43 | 7,558 | **176×** | 3.7 ms |

**残余偏差**：`match_chars / len(prompt)` 这个比例被网格系统性低估，幅度随 n 在
`[1/(1+1/k), 1]` 即 k=4 时 `[0.8, 1.0]` 间振荡。同一请求内所有候选同网格，**排序不受影响**；
但一旦进入跨请求比较（指数打分的 ratio），必须改用 `boundaries[-1]` 作分母消除它。

### 3.2 `scheduler.py`：`cache_aware_score` 策略

借鉴 dashscope 网关 `CacheAwareBalancedRouteSelector`：

```python
weight[n] = 2 ** (hit_weight * hit_ratio[n] + load_weight * (mean_load - load[n]))
从 topk = max(sqrt(N), 5) 个最重的节点里按 weight 加权采样
```

指数形式的意义是 `weight[a]/weight[b] = 2^(score[a]-score[b])`——**只依赖分数之差**，
给所有节点的 load 加同一常数会抵消，因此对负载整体平移免疫。这正是硬阈值缺的性质。

**两个权重不是一个自由度**（实测）：

```
20/1 和 10/0.5 的 K 都是 20，但 routes 42/3 vs 36/9、argmax 一致率 0.89 vs 0.68
→ 比值 K = w_hit/w_load  决定权衡点（一次满命中值几个请求的排队余量）
→ 绝对幅度               决定采样锐度（温度）
```

`_load` 的量纲：均值 ≈ `decay_window/N` ≈ 8.7，节点间极差 ≈ 6。所以 `w_load=1` 时负载项
幅度是 `2^±3`，与 `w_hit=10 × ratio≈0.8 = 8` 同量级——这解释了 dashscope 为何标定成 hit=10。

### 3.3 编排脚本（都在 `runs/mcstore/`，旧版备份 `*.bak_32b`）

| 脚本 | 作用 | 本轮改动 |
|---|---|---|
| `run_node.sh` | 起单个 P/D | 加 `MODEL_ROOT`/`QUANT`/`KV_DTYPE`/`BLOCK_SIZE`/`HF_OVERRIDES`/`MAX_BATCHED`/`P_GPU_UTIL`/`DISABLE_HYBRID_KVCM`/`EAGER`；就绪循环改为「超 deadline 后只要 shard 计数还在动就继续等」 |
| `launch_local.sh` | 起本机那一份（slot 1-4） | 透传上述全部 + `GPU_BLOCKS` |
| `run_policy.sh` | 跑一个策略一轮 | 加 `MODEL_PATH` 全路径覆盖 + `NODE_TAG` |
| `bringup_150b.sh` | 从笔记本驱动的全集群重启 | **新增**，含 master 重启（= 清空 store）+ 4 slot 并行派发 |
| `preload_shm2.sh` | 模型预热到 `/dev/shm` | **新增**，12 并行 + 逐文件大小校验 |
| `campaign.sh` | A B A B 交替对照 | **新增**，每轮查驱逐非零即中止 |
| `saturate.sh` | 让 store 自然溢出 | **新增**，不中止，逐臂记录驱逐 delta |
| `master_load.py` | 从 master 日志抽 batch op 峰值 req/s 和 item/s | **新增** |
| `compare_campaigns.py` | 合并对照表 + 噪声判据 | **新增** |
| `analyze_saturate.py` | 饱和 breakdown | **新增** |
| `recover_mix.py` | 按时间戳聚类事后补算 mix | **新增**（修 NODE_TAG bug 的产物） |

---

## 4. 实验编年史

### 4.1 阶段一：Qwen3-32B，结论被推翻两次

同一个 `prefix_affinity` 策略，三轮实验分别是**落后 37% → 领先 27% → 领先 7%**，
而这三个数字的差异**全部来自基础设施缺陷**：

| 轮次 | 结果 | 真实原因 |
|---|---|---|
| 跨机单网卡 | 领先 27% | 7 个 P 挤一张 `mlx5_bond_0`，store 读被单口带宽卡住，虚高了「避免走 store」的价值 |
| 跨机多网卡 | 落后 37% | **绝对护栏在低并发下从不触发**（轮次间 5-7s think time，瞬时 inflight 恒为 0/1），流量雪球到单节点（7 节点里 43 vs 7），排队惩罚超过省下的传输时间 |
| 修好护栏 | 领先 7% | 护栏改成「衰减累积分配量的相对阈值」`load_ratio≈1.5` |

护栏的最终形式：

```python
load[n] = 衰减累积分配数 + 当前 inflight      # 每次决策 *= (1 - 1/decay_window), window=64
跳过亲和节点 if load[best] > load_ratio * max(mean(load), 1.0)
```
`max(mean, 1.0)` 下限防止冷启动时均值≈0 导致过度约束。
**直接把 inflight 的不等式改成相对也不行**：`inflight[idlest]=0` 时均值≈0.3，1.5 倍≈0.5，
任何 `inflight≥1` 都被拒 → 亲和被整体关掉，从「永不触发」跳到「永远触发」。

### 4.2 阶段二：读 dashscope 网关

网关早有成熟实现（`dashscope-platform/dashscope-api/.../virtual/route/`）：
`CacheAwareRouteSelector` / `CacheAwarePureRouteSelector` / `CacheAwareBalancedRouteSelector`。
详细机制见 §9.3。搬了两个：几何 key 生成、指数打分。

### 4.3 阶段三：几何网格 A/B 无害性验证

同一批 trial、同 `load_ratio=1.5`，旧线性网格 vs 新几何网格：

| | 线性 512 | 几何 k=4 |
|---|---|---|
| p50 | 1706.4 ms | 1693.8 ms |
| p90 | 2383.4 ms | 2404.2 ms |
| 本地HBM / store / 重算 | 75% / 2% / 23% | 75% / 2% / 23% |
| **决策** | miss 1, hit 118, 让位 22 | **miss 1, hit 118, 让位 22** |
| 每请求 key（26k） | 152 | **20** |

**141 个决策一字不差地相同。** 换网格没有损失任何排序能力。

### 4.4 阶段四：指数打分 K 扫描

| | 基线 硬阈值 | K=10 | K=5 | K=2.5 |
|---|---|---|---|---|
| p50 | **1694** | 1708 | 1724 | 1749 |
| 本地HBM | **75%** | 69% | 66% | 60% |
| store | 2% | 6% | 10% | 13% |
| routes 最多/最少 | 26/15 | 25/15 | **22/18** | **21/18** |
| 采样命中 argmax | — | 0.745 | 0.702 | 0.766 |

指数打分更平衡但**丢本地命中**：采样有 25~30% 概率不选 argmax，每次偏离就把一个
本地命中换成 store 读。于是加了一档「固定 K、放大绝对幅度 4 倍」（40/4）让采样趋近确定性——
首次测出 p50 1635ms（−3.5%）、**重算 3%**，看起来赢了。

### 4.5 阶段五：关键订正——store 常驻驱逐污染了整个阶段一

`重算 3%` 触发警报，因为它**低于理论下界**：

```
N 轮会话：所有 prompt 之和 = N(N+1)/2 单位，真正新增 = N 单位
       → 重算下界 = 2/(N+1)
8 轮 → 22%（其余各轮实测 23% ✓）    3% 物理上不可能
```

重跑验证：基线复现良好（1705ms / 74% / 2% / 23%），**sharp 那档不复现**
（1700ms / 76% / **1%** / **23%**）。查 store 状态发现根因：

```
占用      88.5% 89.8% 87.3% 85.7% 87.7% 87.0% 87.8% 88.6% 90.0% 86.7%
累计驱逐   3     3     7     11    14    18    28    28    31    35   次
累计驱逐量 172GB ..................................→ 1.96 TB
```

**占用被死死钉在 86~90%，从第一轮起就在持续驱逐。** mooncake 是近似 LRU，会误伤新键，
所以「某一轮自己写进去的 KV 能不能活到被读回」基本是运气——这就是 store 命中率在
1%~24% 之间随机漂移的原因，而那恰好是我们要测的量。

### 4.6 阶段六：换 qwen3-150b + 干净重测

三个前提全部堵死：多网卡按 NUMA 绑 rank、护栏改成相对累积量、**master 重启清空 store**
（700GB 对 122GB/臂 工作集，7 倍余量）。加上 A B A B 交替 + 每轮查驱逐非零即中止。

**Config A（`GPU_BLOCKS=8000` = 512k token/P）**

| arm | p50 | p90 | 本地HBM | store | 重算 |
|---|---|---|---|---|---|
| aff_r1 | 2528.5 | 2970.9 | 90% | 2% | 7% |
| sco_r1 | 2538.5 | 2956.9 | 91% | 2% | 7% |
| aff_r2 | 2580.7 | 3002.2 | 91% | 2% | 8% |
| sco_r2 | 2569.7 | 2970.3 | 91% | 2% | 7% |

```
prefix_affinity   均值 2554.6ms   重复间差 52.2ms (2.0%)
cache_aware_score 均值 2554.1ms   重复间差 31.2ms (1.2%)
                  策略间差 0.5ms (0.02%)   ← 比噪声小 40~100 倍
```
**无法分辨。** 而且有机制解释，不是巧合：

```
prefix_affinity  : miss=1,  hit=405, 让位=27.5  → 偏离纯亲和 6.4%
cache_aware_score: miss=28, hit=406             → 偏离纯亲和 6.5%
```
两种完全不同的机制（硬阈值否决 vs 加权采样，`picked_argmax=0.95`）达到几乎相同的偏离率。

**Config B（`GPU_BLOCKS=3000` = 192k token/P）**：本地 KV 砍掉 62%，
结果 HBM 90%→88%、store 2%→4%、延迟一动不动（2556 vs 2554ms）。

为什么压小 HBM 没用——**两个约束的窗口不存在**：

```
每个 P 服务 24/7 = 3.4 个 trial，需常驻 3.4 个上下文
Config B 给了 192k / p50 46k = 4.2 个   ← 仍够装，命中率没掉
要迫使一半走 store，只能常驻 1.7 个 = 78k token = 1,220 block
但单个最长请求 87k = 1,360 block，两个并发要 2,720 block
                    ↑ 1,220 < 2,720，窗口是空的
```
根本原因是**单请求上下文与「节点需常驻的全部上下文」是同一量级（1 : 3.4）**。

更根本的：**每次 store 读本来就很便宜**。

```
Config A: store 2% × 46k = 920 token/请求  = 55 MB  → RDMA 约 3 ms
Config B: store 4% × 46k = 1,840 token     = 110 MB → 约 5 ms
                                             对比 p50 = 2,556 ms → 占 0.2%
```
提高 store 读的**频率**不可能产生可测效应，因为单次成本接近零。

### 4.7 阶段七：饱和实验（不重启、不缩容，靠累积让 store 自然溢出）

设计要点：每臂一个独立 salt（≈88 GB 新增），从 354 GB 起累积，第 4 臂前后跨过
`0.90 × 700 = 630 GB` 水位，之后 4 臂是稳态。**损伤指标 = 重算率 − 8% 下界**，
它有绝对基准、不会随运行顺序漂移。

**结果：驱逐持续激活 14 次、扔掉 442 GB / 24 万 key，而重算率一动不动。**

| arm | 策略 | storeGB | 驱逐 | p50 | HBM | store | 重算 | damage |
|---|---|---|---|---|---|---|---|---|
| sat1_aff | affinity | 354→443 | 0 | 2562 | 86% | 7% | 8% | +0.0% |
| sat2_sco | score | 443→531 | 0 | 2576 | 87% | 5% | 7% | −1.0% |
| sat3_aff | affinity | 531→620 | 0 | 2564 | 88% | 4% | 8% | +0.0% |
| sat4_sco | score | 620→**614** | **3** | 2587 | 87% | 5% | 7% | −1.0% |
| sat5_aff | affinity | 614→608 | 3 | 2586 | 88% | 5% | 8% | +0.0% |
| sat6_sco | score | 608→601 | 3 | 2575 | 89% | 3% | 7% | −1.0% |
| sat7_aff | affinity | 601→627 | 2 | 2582 | 86% | 6% | 7% | −1.0% |
| sat8_sco | score | 627→620 | 3 | 2585 | 84% | 8% | 8% | +0.0% |

```
跨越水位的代价   HBM 87.0→86.8%   store 5.3→5.4%   重算 7.7→7.4%   p50 +15.3ms (+0.6%)
累计驱逐        14 次, 241,619 key, 442.39 GB
稳态            store 在 600~627 GB 震荡 —— 驱逐量精确抵消新写入量
策略对比(饱和 n=5) 差 1.8ms，重复间噪声 12.2ms → INDISTINGUISHABLE
master 压力      峰值 ExistKey 5,758 keys/s，与 Config A 的 5,165 基本一致
                （饱和不增加元数据压力，因为 lookup 只查本地未命中段，而本地命中率没变）
```

### 为什么零损伤 —— 机制

```
1. 每臂一个独立 salt      → 前面各臂的数据是 100% 垃圾，永远不会再被命中
2. 驱逐按 lease_timeout 最早排序 → 恰好命中那批垃圾
3. 当前臂数据被持续探测(batch_is_exist) → 10s 租约反复续命 → 豁免驱逐
```
第 3 点就是 §7.2 的「探测即续命」——**它在这里起保护作用而不是危害**。近似 LRU 的不精确
（租约量化、批量分位点切割）在这个负载下无关紧要，**因为工作集和垃圾在时间上干净分离**。

### 关键自我批评：salt 隔离与驱逐评估互相冲突

```
salt 隔离的作用   两个策略跑完全相同的 trial、互不预热 → 对比干净
salt 隔离的副作用  旧数据 100% 是垃圾、严格比工作集更旧 → LRU 的理想情况
```

真实负载里「冷数据」不是 100% 垃圾，而是**复用间隔较长的数据**。那时近似 LRU 的不精确才会
真正伤人——它可能砍掉「剩余复用距离很短」的块。而这个设计从结构上排除了那种情况。

所以准确的结论是：

> **没能测出驱逐损伤，原因是负载的垃圾与工作集在时间上干净分离。这不是「近似 LRU 足够好」
> 的证明，而是「这个负载对驱逐策略不敏感」。**

**要真正评估驱逐质量，需要复用间隔重叠的负载**：反复重放**同一批 trial（不换 salt）**，
让旧数据仍然被需要，同时 store 装不下全部。那时 LRU 砍错才有代价，也才能对比不同驱逐策略。
这是所有 §8.2 驱逐改动的真正前置条件——比「让驱逐发生」更严格：**要让驱逐有机会砍错**。

---

## 5. 站得住的结论

1. **几何前缀网格无害且必要**。141 个决策与线性网格逐一相同；1M 上下文下 key 数省 176 倍、
   索引表项从每请求 7,558 降到 43。调度器 CPU 在 26k~85k 尺度量不出差异（都 <1s），
   收益要到长 prompt 才显现。
2. **指数打分 ≈ 修好的硬阈值**，在 store 充裕时无法分辨（0.02% vs 2% 噪声）。
   指数打分的结构性优势（尺度无关、无「永不触发/永远触发」失效模式）仍然成立，只是无处兑现。
3. **`K = w_hit/w_load` 太小确实更差**（K=5 → 本地 66%，K=2.5 → 60%），机制是采样偏离 argmax。
   方向可信，幅度不可信。
4. **cache-aware 调度的收益取决于「选错节点的代价」，不取决于命中率高低。**
   store 带宽充裕时这个代价是 0.2%，所以任何策略差异都被噪声吞掉。
   之前测到的 +27% 真正来源是单网卡把 store 读卡住了。
5. **store 和本地 HBM 是完全的替代关系，不是叠加**。多轮实测：HBM 命中下降的百分点被
   store 命中几乎等量接住，而**重算量不变**（一直贴着 `2/(N+1)` 下界）。
6. **这个模型上「长上下文 → store 成本更高」不成立**：qwen3-150b 的 KV 密度是 Qwen3-32B 的 1/4
   （60 vs 256 KiB/token），85k 上下文的每请求字节数（5.1 GB）反而比 26k 的 Qwen3-32B（6.4 GB）少。
7. **饱和本身几乎无代价，但这个结论有强前提。** 驱逐扔掉 442 GB / 24 万 key，重算率
   仍贴着下界，端到端只贵 0.6%。机制是「垃圾严格比工作集更旧」+「探测即续命保护活跃前缀」。
   **salt 隔离（为了对比干净而引入）恰恰制造了 LRU 的理想情况**，所以这不能推广成
   「近似 LRU 足够好」。要评估驱逐质量必须换成复用间隔重叠的负载（重放同一批 trial 不换 salt）。
8. **master 元数据压力的关键性质：lookup 只查本地未命中的那一段**，所以高本地命中率会自动保护
   master。实测本地命中 90% 时只有 5,165 keys/s（每请求 228 个 key，而非上下文对应的 1,437 个）。
   风险不在实例数，而在**「低本地命中 × 高吞吐」的组合**——也就是冷启动/缓存被打爆的时刻，
   恰好是最需要 store 的时刻。

---

## 6. 方法论

### 6.1 三个绝对基准（不会漂移，用来否定假结论）

| 基准 | 公式/来源 | 用它抓到了什么 |
|---|---|---|
| **重算下界** | `2/(N+1)`：8 轮 22%、24 轮 8% | 「重算 3%」在 8 轮下物理不可能 → 定性否定 |
| **噪声下限** | 同配置重复间差，实测 1.2~2.0% | 「领先 3.5%」本来就在噪声内 |
| **恒等式** | 重算 token 数 = store 净增长 = Σ(各 trial 最终上下文) | `1,547,716 token × 60 KiB = 88.6 GiB`，与实测 store 增长 88.5 GB 吻合 → 三重交叉验证 |

第三条尤其好用：一个臂真正新增的内容既是「重算量」也是「store 新增量」，两个独立测量必须相等。

### 6.2 「沉默不等于成功，回显也不等于成功」——三个自制的假成功

| # | bug | 表现 | 修法 |
|---|---|---|---|
| 1 | 预热脚本没查 `cp` 退出码 | `cp` 被 kill 后照样打印 `PRELOAD_SHM_DONE`，监控信了，实际只拷 8.3GB/280GB | 逐文件比对源/目标 size，全部一致才打印哨兵 |
| 2 | `ssh h 'cat > f && ... & disown'` | `&` 把整条 `&&` 链塞进后台，`cat` 拿不到 stdin，脚本根本没写成，而 `echo launched` 照样打印 | 推送和启动拆两步，各自 md5 校验 |
| 3 | `HF_OVERRIDES={"..."}` 在双引号 ssh 串里 | 远端 shell 剥掉内层双引号 → JSON 静默变非法 → 表现是「起来了但 max_model_len 还是 35840」 | 内层用单引号，并实测 `json.loads` 验一遍 |
| 4 | `campaign.sh` 里 `NODE_TAG=n1` 写死 | Config B 节点日志是 `_n2`，扫了旧日志时间窗对不上 → mix 三列全 0% | 参数化；已用 `recover_mix.py` 按时间戳聚类事后补算 |

第 3 个如果没提前测，8 节点铺开后极难归因。

### 6.3 实验编排的硬规则

1. **交替重复 A B A B，不是 A A B B**。运行顺序漂移必须被平均掉。
2. **每轮查驱逐计数，非零立即中止整个 campaign**——不是记录下来继续跑。一旦驱逐开始，
   后面所有数字都不可信，继续跑只是生产更多垃圾。
3. **store 预算是 arm 数的硬上限**：每臂 88 GB 独立 salt 命名空间，
   4 臂 = 490 GB / 700 GB = 70%（安全），8 臂 = 980 GB（必然驱逐）。
4. **preemption 双侧都查**。曾只 grep 7 个 P、漏了 D，而 D 侧有 111 条 abort
   （1 个 D 承接 7 个 P 的流量，配额是 P 的 7 倍需求）。
5. **验证服务存活用端口连通性，不用 `pgrep -f`**。`pgrep -cf "mooncake_master"` 会把自己
   那条 ssh 命令串也算进去，返回 1 → 误判「已在运行」→ 跳过启动 → 8 个 P 全在
   `assert store.put()` 上失败。
6. **salt 隔离而不是重启集群**：salt 插在 prompt 最开头，block hash 从开头链式算，
   换 salt 等于把整轮搬进独立命名空间。两个策略可跑完全相同的 trial，样本量和 prompt 总量一致。

### 6.4 一个省时间的做法

**先离线模拟再跑真集群**：stub 掉 `aiohttp` 直接 `exec` scheduler.py，用假 `ClusterState`
模拟 24 trial × N 轮 × 7 节点（含共享系统提示），几秒就能看出分配是否拉平。
真集群一轮 5 分钟，模拟能挡掉无效方案。
注意 stub 时要先 `sys.modules["<name>"] = module` 再 exec，否则 `@dataclass`
会因找不到模块而报 `AttributeError: 'NoneType' object has no attribute '__dict__'`。

---

## 7. mooncake 源码分析

版本：我们跑的是 `0.3.12.post1 (git: 6041a60, 2026-07-25)`，
用 `strings /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master | grep -E "^v?0\.3\."` 查出。
源码 clone 在 ecs `~/codes/Mooncake`，HEAD 是 `0.3.13 (26dd3ed9, 2026-08-25)`。
**读我们那一版必须用 `git show 6041a609:mooncake-store/src/master_service.cpp`**，
别直接读 HEAD（8956 行 vs 13262 行，差 268 个 commit）。

### 7.1 查询路径的确切输入输出

代码：`vllm/v1/hybrid_connector/mooncake_kvsbackend.py`（父类，调度器侧逻辑）
+ `mooncake_store_kvsbackend.py`（我们的 PyPI API 实现）。

**第 1 步 `async_get_num_new_matched_tokens(req, num_computed_tokens) -> int`**（`:448`）

```
输入  req.block_hashes[]    vLLM 算好的链式 block hash，每 block_size token 一个
      num_computed_tokens   本地 HBM 已命中数（必须是 block_size 整数倍）
过程  1. 短路：should_load 为假 或 剩余未算 < block_size → 返回 0
      2. 只取 [num_computed_blocks, +num_uncomputed_blocks) 这一段 hash
      3. 每个 hash 展开成 tp_size 个 key: f"{hash_hex}_{tp_rank}"
      4. store.batch_is_exist(keys) → List[int]
      5. 按 tp_size 分组，组内**全部** rank 为 1 才算命中，遇第一个不全 1 立即 break
输出  new_matched_tokens = 连续命中 block 数 × block_size
```

三条决定开销的性质：
- **只查本地未命中的那一段** → 本地命中率高会自动降低 master 压力
- **key 数 = 未命中 block 数 × tp_size**（256k/block64/tp2 = 8192 key/请求）
- **前缀语义 + 全 rank 要求**：遇断即停；任一 rank 缺失整块作废

**第 2 步 `async_update_state_after_alloc` → `async_load_kv`**

顺序是**先定命中长度，再由 vLLM 分配目标 block**，然后：
```
_blocks_to_kv():  block_id   → prepare_value_for_block() → (addrs[], sizes[])
                              （各层 view 的 base_addr + block_id × block_len）
                  block_hash → key_for_hex() → key
store.batch_get_into_multi_buffers(keys, addrs, sizes)   GPUDirect RDMA 直写 GPU
遇第一个返回码 <0 立即 break —— 前缀留洞会被当有效 KV 读出去污染输出
```

**`VLLM_KVS_ON_MIN_LENGTH` 的正确取值**（回填 progress §2.7）

```python
def should_load(req):  return not (req.num_prompt_tokens <= VLLM_KVS_ON_MIN_LENGTH + 1)
```
默认 2048。**理论下界是 `block_size + 1`**：vLLM 故意让最后一个 token 不参与 hash
（`(num_tokens-1)//block_size`），所以 `num_prompt_tokens ≤ block_size` 时 0 个 hash block、
必然查不中（代码里 `+1` 就是这个偏移）。
**但按收益应设 `≥ 2 × block_size`**：命中 1 个 block 只省 block_size 个 token 的重算，
却要付 tp_size 个 key 的 RPC + master 侧租约写锁，不划算。

### 7.2 驱逐机制：为什么是「近似 LRU」（回填 progress §2.10）

**仓库里有一个精确 O(1) LRU，但它是死代码。** `mooncake-store/include/eviction_strategy.h`
的 `LRUEvictionStrategy`（list + hashmap）只在 `master_service.h:60` 前向声明，
**全仓库无实例化，HEAD 上也一样**。

真实路径 `EvictionThreadFunc` → `BatchEvict`：

```
触发   used_ratio > eviction_high_watermark_ratio (默认 0.90)
       evict_ratio_target = max(eviction_ratio, used_ratio - watermark + eviction_ratio)

Phase 1 普查（16 线程并行扫全部 1024 个 metadata shard，起点 RandomIndex(1024)）
       IsHardPinned()        → 跳过（永不驱逐）
       !IsLeaseExpired(now)  → 跳过（租约内，10s 保护）
       !can_evict_replicas() → 跳过（无可驱逐内存副本）
       IsSoftPinActive()     → 进「软钉池」（次优先）
       其余                   → 进「候选池」，收集的排序键是 ★lease_timeout★
Phase 2 定切点
       evict_num = ceil(可驱逐总数 × target)
       std::nth_element(candidates, begin+evict_num)   按 lease_timeout 找第 k 早
Phase 3 执行
       候选池不够 → 才动软钉池（allow_evict_soft_pinned_objects 默认 true）
       grouped 对象：组内★所有★成员租约都过期才动，且整组一起走
       有 LOCAL_DISK 副本 → 直接删内存副本
       否则若开 offload_on_evict → 先推 offload 队列再删
```

**「近似」精确来自四处，第一处是根本**：

1. **排序键是 `lease_timeout = 授予时刻 + TTL`，不是 last_access**。TTL 窗口内被访问过的对象
   整体豁免（连候选都进不去）；其余对象的时间戳被 TTL 量化，实际访问差 9 秒可能拿到几乎相同的值。
   它不是「最近最少使用」，而是「租约最早过期」。
2. 批量按分位点砍，周期之间不维护任何顺序。
3. 每 shard 独立加锁 + 随机起点 → 普查是**非原子快照**。
4. 软钉是第二层池，形成两级优先而非单一序。

**`ExistKey`/`BatchExistKey` 会授予租约**（回填 progress §6.4 的待查项）：
`master_service.cpp:2772` 和 `:2797` 都调 `GrantLeaseForGroup` / `GrantReadLease(default_kv_lease_ttl_)`，
默认 TTL **10000ms**（`master.cpp:33` 有 `static_assert(DEFAULT_DEFAULT_KV_LEASE_TTL == 10000)` 钉死）。

- 好消息：lookup→load 窗口有 10s 保护，不需要额外防护。
- 坏消息：**探测即续命**。每次前缀探测都给对象延寿、豁免驱逐，高并发下把可驱逐池压小，
  反过来让驱逐去砍「刚写进去还没被探过」的键。这是 store 命中率反复无常的一个可能机制。

**可用的调优 flag（不改代码）**
```
-eviction_high_watermark_ratio   默认 0.90
-eviction_ratio                  一次驱逐的对象比例
-allow_evict_soft_pinned_objects 默认 true
-memory_allocator                cachelib | offset
```

### 7.3 master 架构与扩展性

- **一个集群一个 master，每个 TP rank 一个 client**（7 节点 × 2 rank = 14 clients）
- 职责：**只管元数据**——key→replica→(segment, offset, size, status)、PutStart 分配、租约、
  驱逐决策、segment mount/unmount、client 健康（每 client 1 Ping/s）、etcd leader 选举、
  snapshot / SSD offload / 多租户配额。**不在数据面**，client↔client 直连 RDMA
- 并发：`kNumShards = 1024` 个 metadata shard 各自独立读写锁，驱逐普查 16 线程
- **不支持横向扩展**：grep 过 `master_shard`/`multi_master`/`consistent_hash` 全无；
  HA 是 etcd leader/follower，一个时刻只有一个 leader 在服务——解决可用性不解决吞吐；
  客户端配置也只接受单个 `master_server_addr`

**实测压力与外推**（Config A，433 请求/臂）
```
峰值 ExistKey: 22.7 batch-req/s → 5,165 keys/s，约 228 key/batch
```
| 场景 | key/请求 | 100 实例外推 |
|---|---|---|
| agentic（1.5 req/s，本地命中 90%） | 228 | 65k keys/s — 毫无压力 |
| 同命中率 + 吞吐型（50 req/s/实例） | 228 | 2.2M keys/s — 吃紧 |
| **冷缓存（本地命中≈0）** | 1437 | **14M keys/s — 必然过载** |

**降压的三个旋钮（按杠杆排序）**
1. `block_size` 64→512：metadata key 直接砍 8 倍（2656→332）。代价是最小命中粒度变粗
2. **跨 tp rank 去重（最值得做，零代价）**：当前 key 格式
   `{prefix}@{model}@tp_rank:{r}@...@{hash}` 把 rank 编进 key，tp8 的压力是 tp2 的 4 倍。
   但这在语义上冗余——`async_get_num_new_matched_tokens` 的逻辑就是「全 rank 都为 1 才算命中」，
   即一个 block 的各 rank 分片要么都在要么都不在。改成 key 不带 rank、value 存各 rank 地址，
   直接砍 tp_size 倍，无精度/命中率代价
3. `VLLM_KVS_ON_MIN_LENGTH`：只影响短 prompt，长上下文场景无用

**注意别混淆两种 key**
```
调度器侧 prefix hash key（几何网格）  85k 上下文 → 29 个    纯本地字典查
mooncake 侧 metadata key（block hash） 85k 上下文 → 2,656 个  每个走 master RPC + 授租约
```
差 92 倍，而压力大的是后者。几何网格解决的是调度器自己的 CPU 和内存，**完全不影响 master 压力**。

### 7.4 版本差异（ours 0.3.12.post1 vs HEAD 0.3.13）

```
268 commits, 334 files, +61,244 / −15,622 行
mooncake-store 102 个 commit，transfer-engine 101 个   ← 整一个月的量
```

**驱逐机制一点没变**：`LRUEvictionStrategy` 仍是死代码，`BatchEvict` 仍按
`lease_timeout` + `nth_element` + soft-pin 两级。**所以我们想做的改动上游没做，仍是真缺口。**

已在我们版本里、不需要升级：

| 特性 | ours | HEAD |
|---|---|---|
| `enable_kv_events` | ✓ | ✓ |
| `promotion_on_hit` + `promotion_admission_threshold=2` | ✓ | ✓ |
| `offload_on_evict` | ✓ | ✓ |
| `allow_evict_soft_pinned_objects` | ✓ | ✓ |
| `enable_cxl` | ✓ | ✓ |
| `last_access_ns_`（真 LRU 有序索引） | ✓ | ✓ |

最后一条要注意：`last_access_ns_` 在 `storage_backend.h`，服务的是 **SSD/bucket 路径**：
```cpp
// LRU eviction index: ordered set of {last_access_ns_, bucket_id}.
// Maintained lazily — reads update last_access_ns_ atomically without ...
```
也就是**「真 LRU + 有序索引」在磁盘路径上已实现并验证过，内存路径仍是 `lease_timeout` 近似**。
对我们有利：要改内存路径有同仓库的现成范式可抄。

HEAD 新增、我们没有的：
- **`dynamic_replication_mode`（off/observe/enforce）**+ `heat_window_seconds=10`
  + `admission_qps_threshold=0.8` + `max_memory_replicas=2`（`#3389 Add dynamic hot replica fanout`）
  —— 引入了**内存路径的 per-key 访问频率计数**，正是价值感知驱逐需要的信号，
  顺带解决了「mooncake 无热点复制」
- 大量 snapshot / HA / oplog / 多租户配额加固
- 三个驱逐 bugfix：`#3421` 保护未就绪恢复副本、`#3419` 按 segment 数 gate PutStart 驱逐、
  `#3118` BatchEvict 候选选择性物化（性能）

**是否升级：倾向不升。** 268 个 commit / 6 万行、涉及 HA/snapshot/oplog 大改，
而唯一真正想要的频率计数器可以自己在 `ObjectMetadata` 上加个字段实现
（已确认我们版本的 `ObjectMetadata` 没有 per-key 命中计数，`promotion_admission_threshold_`
只是个 MasterService 阈值成员）。且 0.3.13 还没 PyPI wheel，得从源码编。

### 7.5 `kv_events`：现成能用，零 C++ 改动

我们的 binary 已经有：
```
--enable_kv_events                            (RFC #1527 KV cache event publisher over ZMQ)
--kv_events_bind_endpoint tcp://0.0.0.0:5557  (ZMQ PUB)
--kv_events_emit_object_key                   (带上 mooncake object_key)
--kv_events_emit_legacy_compat                (兼容 vLLM/SGLang 字段名)
```
master 侧发三类事件，**驱逐也发**：`PublishKvStored`(put) / `PublishKvRemoved`(delete) /
`PublishKvRemovedAfterEvict`（BatchEvict / NoFBatchEvict / DfsEviction 各路径都调）。

**但要分清两条事件流**（我一开始混为一谈过）：
```
mooncake master 的 kv_events  →  共享 store 里有什么   ← store 是共享的，不区分节点
vLLM 每个节点的 kv_events     →  该节点 HBM 里有什么    ← 这才是亲和索引要的
```
我们 90% 的命中来自各 P 节点本地 HBM 的 prefix cache，那是 vLLM 内部状态，mooncake 不知道。

好消息是两边都现成有。内部 vLLM：
```
vllm/config/kv_events.py        enable_kv_cache_events + zmq publisher
vllm/v1/core/block_pool.py:272  BlockStored   (block_hashes, parent_block_hash)
                          :445  BlockRemoved  ← 就在 block pool 的驱逐路径上
                          :552  AllBlocksCleared
```

**但当前做它的理由不是「修索引」**。用实测核一下三个缺陷现在值多少钱：

| 缺陷 | 当前实际影响 |
|---|---|
| store 侧假阳性 | Config A/B 下 Eviction 0/0，不存在 |
| HBM 侧假阳性 | 本地命中 90%（HBM 也没驱逐）；压到 192k 后 88% —— 上限约 2 个百分点 |
| 探测即续命 | 只在驱逐激活时有意义 |

**索引陈旧现在最多值 2%，而噪声下限就是 2%。** 真正被量化过的收益是另一条：
**用事件流替掉 `batch_is_exist` 那轮 RPC**，直接命中「冷缓存下 14M keys/s 必然过载」那个墙。

---

## 8. 未来 TODO

### 8.1 高优先级

| # | 事项 | 依据 | 代价 |
|---|---|---|---|
| 1 | **收窄 store 带宽的 Config C** | 唯一被实测过能产生两位数差异的条件（跨机单网卡曾 +27%）。`run_node.sh` 已支持 `MC_DEVICE`，设成单个 `mlx5_bond_0` 即可；或 protocol 换 tcp | 一个变量 + 一次重启 |
| 2 | **metadata key 跨 tp rank 去重** | 零精度代价，砍 tp_size 倍 master 压力；语义上本来冗余 | vLLM 侧 Python，中等 |
| 3 | **`VLLM_KVS_ON_MIN_LENGTH` 提到 `2 × block_size`** | 命中 1 个 block 省 block_size token，却付 tp_size 个 key 的 RPC + 租约写锁 | 一行 |

### 8.2 mooncake C++ 改动（需从源码重编）

**前置（比原先设想的更严格）**：§4.7 已证明「让驱逐发生」不够——还必须**让驱逐有机会砍错**。
salt-per-arm 的负载里垃圾严格比工作集更旧，LRU 怎么砍都对，所以任何驱逐改动都测不出差异。
正确的前置实验是**反复重放同一批 trial（不换 salt）+ store 装不下全部**，制造复用间隔重叠。

| # | 事项 | 依据 |
|---|---|---|
| 4 | 内存驱逐排序键 `lease_timeout` → 真实 `last_access` | 变成真正的批量 LRU；`storage_backend.h` 的 SSD 路径有现成范式可抄 |
| 5 | 驱逐按**前缀深度**加权 | 纯 recency 在 agentic 负载下会反向：共享系统提示（24 会话依赖、深度浅）一旦某轮没被探到就租约过期被砍，而刚访问过的 24 条叶子尾巴全部豁免——**砍掉最值钱的、留下最不值钱的**。信号已存在：`group` 机制（key 带 `@group:0@`）+ block hash 链本身编码位置（第 i 个 block 深度就是 i） |
| 6 | 复用频率信号 | HEAD 的 `dynamic_replication` 有 per-key heat；我们版本可自己在 `ObjectMetadata` 加字段 |

第 5 条与「探测即续命」的副作用**互补**：现在浅层前缀被频繁探测也只拿 10s 租约，
加权后其驱逐代价被显式抬高，不再依赖「恰好被探到」的运气。

### 8.3 独立线：master 扩展性

| # | 事项 | 依据 |
|---|---|---|
| 7 | 接 `kv_events`（mooncake 侧 + vLLM 侧两条流） | 消除 `batch_is_exist` RPC（冷缓存外推 14M keys/s 必然过载）+ 消除索引假阳性 |
| 8 | 按 key hash 把 metadata 分片到多 master | 当前完全不支持，是比改驱逐更有上游价值的贡献方向 |

### 8.4 已知但暂缓

- **hybrid 模型支持 M2~M6**（见 progress 文档 §6.1）。前置：先估命中率——hybrid 下
  attention `block_size` 被强制成 784，`lcm_block_size` 对齐后 mamba 的候选边界很稀疏
- **旧 KVS backend 的 group 维切片 + 缺 `is_hybrid` 保护**，hybrid 下会静默出错（既存 bug）
- **PD 只有冷启动第一个请求输出正确**，之后退化成 `. . . .`。已确认与本分支改动无关
  （纯 kvt 的 P 也一样）。待查 `--async-scheduling`
- **vLLM safetensors 加载器单流读**：280GB 模型 90 分钟，其中绝大部分是白等
  （共享盘单流 71 MB/s，12 并行 490 MB/s）。本身是个可优化点

---

## 9. 附录

### 9.1 qwen3-150b-a14b-256k-1106 实测参数

```
路径 /dashscope/caches/workspace/llx/models/qwen3-150b-a14b-256k-1106  281GB, 76 shards
Qwen3MoeForCausalLM  60 层  64 attn head  4 KV head  head_dim 128
128 experts top-8  attn_output_gate=True  use_gemma_rms_norm=True   → 非 hybrid，store 可用
torch_dtype=bfloat16  quantization_config={}  index 无 scale 键     → BF16 checkpoint
max_position_embeddings=35840  rope_scaling=null  rope_theta=1e7    → 靠 hf-overrides 开 256k
```
- fp8 必须在线量化：`--quantization fp8` + `VLLM_QUANTIZATION_LAYER_WISE=1`
- **fp8 不需要 BLADNN**（`VLLM_FP8_USE_BLADNN=0`）；BLADNN 只对 fp4 是硬依赖
- KV/token = **60 KiB**。已被精确证实：40,090 token 请求 → store 恰好 2.29 GB，
  Keys 恰好 1266 = 626 block × 2 rank + 14 warmup
- **代码类 trace 是 3.87 字符/token**，不是 3.0
- tp2 + `P_GPU_UTIL=0.75`：`Available KV cache memory: 65.17 GiB` → 未受限约 2.28M token；
  D 用 util 0.9 → 实测 `GPU KV cache size: 3,244,224 tokens`
- 无害告警：`No module named 'triton_kernels.matmul_ogs'`（gpt-oss 专用 triton MoE 路径）

### 9.2 `/dev/shm` 预热：共享盘是单流限速

| 并发流 | 单机 | 4 机聚合 |
|---|---|---|
| 1 (`cp`) | 71 MB/s | — |
| 8 | 452 MB/s | — |
| 12 | ~490 MB/s | **~2.3 GB/s** |

test4 完整数据：`280GiB / 587s = 489 MB/s`。`/dev/shm` 默认就有 792G，不用 remount。
配套：`run_node.sh` 的 `MODEL_ROOT=/dev/shm/models`；`SEGMENT` 从 180GB 降到 100GB
（fp8 KV 后工作集小了）腾 RAM 给 tmpfs。

**真正的红利**：让「重启整个集群」从一小时变成 3 分钟，于是「每换一组参数就把环境重置一次」
从不可行变成默认做法——而交替重复的对照实验本来就需要反复重启。

### 9.3 dashscope 网关的 cache-aware 实现（参考对象）

代码 `dashscope-platform/dashscope-api/dashscope-api-component/src/main/java/com/alibaba/dashscope/api/`

**bucket = turbo 实例 id**（不是哈希桶）。turbo 启动时用 `compare_and_set.lua` 抢一个
instanceId 锁，抢到就是自己的 bucket。所以「选 bucket」= 「选实例」。

**前缀索引**：`PrefixUtils.hashTokens` 不做真 tokenize，用 `bytesPerToken=3` 估算，
对前缀长度取 2 的幂逐个哈希（`minPrefixTokenLength=64` 停），
`hashTokensWithDenseSteps` 从 512 起插中点。key 存成 **Redis ZSET**，
成员是 bucket id（score 恒为 1，**这个字段被浪费了**），TTL `expireSeconds=900`。
`PrefixCacheManager` **完全没有删除接口**——只有 `update`(ZADD) / `query`(ZRANGE) + TTL。
写回**无条件**：选定后把请求全部 hash key 指向该 bucket，不校验实际命中。

**打分**（`cacheAwareInferenceSelect`）：默认权重 `prefixMatchWeight=10`、`loadWeight=40`，
其余 5 项为 0。`loadRatio = curLoad/maxLoad(40)`，所以
```
指数 = 10 × hitRatio + (比平均少几个在跑的请求)
```
**两个权重是标定过的：一次 100% 命中正好值 10 个请求的排队余量。**
临界点：亲和节点比其他多跑 9 个请求时优势抵平。
`topK = max(sqrt(实例数), 5)`，`samplingWithWeight` 按归一化权重**随机采样**（不是 argmax）。

**投递**：`push_request_cache_aware.lua` 用 `ZADD queue <bucket_id> <request>`（score 就是实例 id）；
`fetch_request_cache_aware.lua` 出队时 `ZRANGEBYSCORE queue min max LIMIT 0 1` 只拿自己的，
拿不到且 `ZCARD > requestSeizeThreshold(2)` 才 `ZRANGE 0 0` 抢别人的并标记 `hit=0`。
**饥饿保护做在消费端而非生产端**，这是整套设计里最值得借鉴的一点。

**bucket 区间 = 一致性哈希环**：每个存活实例负责 `[前一个存活实例 id + 1, 自己的 id]`，
环形回绕时拆成 primary + secondary，实例挂掉后 bucket 自动被下一个存活实例吸收。

**硬过滤在打分之前**：`filterPrefillBatchSize` / `filterPrefillTokenNum` /
`filterCacheFirstQueueLimit`，剔空了就退化成 least-loaded。
**所以指数打分不是它的过载保护**——渐变交给打分，饱和交给前置硬过滤 + 消费端抢占。
我们最初的设计问题是想让一个阈值同时干这两件事。

### 9.4 复现步骤

```bash
# 0. 前提：4 台机器已预热模型到 /dev/shm（见 9.2）
#    笔记本上有 bringup_150b.sh；test1 的 runs/mcstore/ 有其余脚本

# 1. 起集群（含 master 重启 = 清空 store）。第二个参数是 GPU_BLOCKS
bash bringup_150b.sh n1 8000       # Config A: 512k token/P
bash bringup_150b.sh n2 3000       # Config B: 192k token/P

# 2. 等 8/8 就绪
ssh test1 'for hp in <8 个 host:port>; do curl -s -o /dev/null -w "%{http_code}\n" http://$hp/health; done'

# 3. 交替重复对照（4 臂，每轮查驱逐非零即中止）
ssh test1 'cd .../runs/mcstore && setsid nohup ./campaign.sh c8k 2 >/dev/null 2>&1 &'

# 4. 饱和实验（8 臂，不中止，靠累积溢出）
ssh test1 'cd .../runs/mcstore && setsid nohup ./saturate.sh 8 >/dev/null 2>&1 &'

# 5. 分析
ssh test1 'cd .../runs/mcstore && python3 compare_campaigns.py campaign_c8k.log campaign_c3k.log'
ssh test1 'cd .../runs/mcstore && python3 analyze_saturate.py'
ssh test1 'cd .../runs/mcstore && python3 master_load.py'
```

**注意**：`campaign.sh` / `saturate.sh` 里的 `NODE_TAG` 必须与 bringup 的 tag 一致
（`n1` / `n2`），否则 `run_policy.sh` 会扫错节点日志，mix 三列静默变 0%。
