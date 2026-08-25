---
title: "Mooncake Store + PD 分离：cache-aware 调度与 KV 驱逐"
date: 2026-08-25
categories: [Mooncake Store 与 cache-aware 调度]
tags: [Mooncake, KV Cache, PD 分离, cache-aware 调度, vLLM, 驱逐策略, 学习笔记]
---

# Mooncake Store + PD 分离：cache-aware 调度与 KV 驱逐

这一系列记录的是在 **7 Prefill + 1 Decode 的 PD 分离集群上，给 Mooncake Store 做 cache-aware 调度**
这条线的完整过程：为什么做、设计了什么、实验怎么一次次推翻自己的结论、最后什么站得住、接下来做什么。

配套的实现细节（backend 接线、环境配方、hybrid 模型的 M1~M6）在仓库根目录的
`mooncake_store_progress.md`；本系列只讲调度与驱逐。

## 一句话结论

> **cache-aware 调度的收益取决于「选错节点的代价」，而不是「亲和识别得多准」。**
> 在带宽充裕的共享 store 下，一次 store 读只值端到端延迟的 0.2%，所以任何调度策略差异都被噪声吞掉。
> 我们前后测出过 +27% / −37% / +7% / +3.5% 四个结论，**其中三个是基础设施缺陷伪装的**。

## 目录

| 章节 | 内容 |
|---|---|
| [00 · 背景与动机](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-00-background-and-motivation/) | 为什么需要 cache-aware 调度、问题如何从一个变成三个、环境与拓扑 |
| [01 · Store 架构与源码](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-01-store-architecture/) | KVS/KVT 分层、查询路径的确切输入输出、驱逐机制、master 扩展性、版本差异 |
| [02 · 调度器设计](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-02-scheduler-design/) | 几何前缀网格、指数打分、dashscope 网关的参考实现 |
| [03 · 实验编年史](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-03-experiments/) | 七个阶段，含被推翻的结论与推翻它们的证据 |
| [04 · 结论与方法论](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-04-conclusions-and-methodology/) | 站得住的 8 条结论、三个绝对基准、四个「假成功」陷阱 |
| [05 · 驱逐与迁移设计](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-05-eviction-and-migration/) | 饱和实验、迁移原语、前提辨析 |
| [06 · TODO](/notes/2026/08/25/2026-08-25-mooncake-store-kv-scheduling-06-todo/) | 按优先级分档，每条带依据与代价 |

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
