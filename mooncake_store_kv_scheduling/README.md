# Mooncake Store + PD 分离：cache-aware 调度与 KV 驱逐

这一系列记录的是在 **7 Prefill + 1 Decode 的 PD 分离集群上，给 Mooncake Store 做 cache-aware 调度**
这条线的完整过程：为什么做、设计了什么、实验怎么一次次推翻自己的结论、最后什么站得住、接下来做什么。

配套的实现细节（backend 接线、环境配方、hybrid 模型的 M1~M6）在仓库根目录的
`mooncake_store_progress.md`；本系列只讲调度与驱逐。

## 一句话结论

> **cache-aware 调度的收益取决于「选错节点的代价」，而不是「亲和识别得多准」。**
> 而这个代价有个上界：**store 命中和本地 HBM 命中省掉的是同一段 prefill**，两者只差一次远端读。
> 我们把它推到了这套负载的极限——store 份额从 2% 推到 44%、store 读收窄到单张网卡、
> 节点数砍到 3 个——**两个策略仍然无法分辨**（+0.39%，而噪声下限只有 0.2%）。
>
> 前后测出过 +27% / −37% / +7% / +3.5% 四个「收益」，**其中三个是基础设施缺陷伪装的**；
> 另有九个「假成功」是自己制造的，其中三个是**诊断** bug——它们让跑对了的实验被读错。

## 目录

| 章节 | 内容 |
|---|---|
| [00 · 背景与动机](00_background_and_motivation.md) | 为什么需要 cache-aware 调度、问题如何从一个变成三个、环境与拓扑 |
| [01 · Store 架构与源码](01_store_architecture.md) | KVS/KVT 分层、查询路径的确切输入输出、驱逐机制、master 扩展性、版本差异 |
| [02 · 调度器设计](02_scheduler_design.md) | 几何前缀网格、指数打分、dashscope 网关的参考实现 |
| [03 · 实验编年史](03_experiments.md) | 七个阶段，含被推翻的结论与推翻它们的证据 |
| [04 · 结论与方法论](04_conclusions_and_methodology.md) | 站得住的 8 条结论、三个绝对基准、四个「假成功」陷阱 |
| [05 · 驱逐与迁移设计](05_eviction_and_migration.md) | 饱和实验、迁移原语、前提辨析 |
| [06 · TODO](06_todo.md) | 按优先级分档，每条带依据与代价 |
| [07 · 多 master 分区实测](07_partitioning_measured.md) | M1 聚合吞吐、M2 fan-out 代价、M3 均衡度、M4 热点读与副本 |
| [08 · 内部统一池与 rldev 对照](08_internal_mooncake_rldev_alignment.md) | 两篇内部文章详解、LocalMaster/Linux reclaim、内部分支实查，以及对 B1/B3/C1/C2 TODO 的修订 |

## 关键数字速查

```
集群       4 × GB200(189GB) × 4 卡 = 16 卡 = 8 个 tp2 服务 = 7P + 1D
模型       qwen3-150b-a14b-256k（MoE，非 hybrid，BF16 checkpoint + 在线 fp8）
KV/token   60 KiB（fp8, 4 KV head, 60 层）—— 已被 store 字节数精确证实
store 池   7 × 100 GB = 700 GB，0.90 水位 = 630 GB
一轮实验    24 trial × 24 turn = 433 请求，20.7M prompt token，工作集 88 GB
延迟基线    p50 ≈ 2,555 ms，本地 HBM 命中 90%，store 2%，重算 7%
噪声下限    同配置重复间差 1.2~2.0% —— 小于这个的差异不构成结论
重算下界    2/(N+1)：8 轮 22%、24 轮 8% —— 低于此值的重算率必是测量假象
```
