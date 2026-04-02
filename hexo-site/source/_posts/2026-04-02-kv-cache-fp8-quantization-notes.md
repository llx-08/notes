---
title: 和GPT-5.4 的KV Cache FP8 量化讨论记录
date: 2026-04-02
tags: []
---
# 和GPT-5.4 的KV Cache FP8 量化讨论记录

## 背景

本文档整理了围绕 `kvtransfer` 链路中 KV cache 量化方案的技术讨论，重点包括：

- 当前已有的 direct cast FP8 路径
- 曾经评估过的 per-block 量化思路
- 最终决定继续采用 direct cast 的原因

## 当前实现概述

### 1. 现有的 BF16 -> FP8 转换方式

当前代码已经支持在 TCP 链路上对 KV cache 做可选的 BF16 -> FP8 转换。

核心特点如下：

- 转换发生在 sender 侧的 D2H 过程中
- 实现位于 `copy_d2h_bf16_to_fp8_kernel`
- `bf16_to_fp8_e4m3()` 对每个 BF16 元素直接执行到 FP8 E4M3 的转换
- 没有显式 scale
- 没有 amax 统计
- 没有 per-block、per-token、per-channel 等额外量化元数据

也就是说，当前实现本质上是：

- 按元素转换
- 直接 cast
- 无 scale
- receiver 侧无需反量化

### 2. 当前量化粒度

函数：

- `bf16_to_fp8_e4m3(__nv_bfloat16 val)`

输入一个 BF16 元素，输出一个 FP8 字节。因此当前量化的语义是：

- 量化单位：单个元素
- scale 粒度：不存在
- kernel 中一次打包 8 个值只是向量化读写优化，不代表量化粒度是 8

### 3. 为什么 decode 侧今天可以直接使用收到的 KV cache

当前 direct cast 路径之所以简单有效，是因为：

- sender 直接把 BF16 转成目标 FP8 格式
- receiver 只需要把字节搬运到目标 KV cache 区域
- decode 侧 GPU 上常驻的 KV cache 已经是下游可以直接使用的最终格式

因此 receiver 不需要做：

- scale 查找
- 反量化
- 再量化

这也是当前方案最大的工程优势之一。

## 关于不同量化粒度的讨论

讨论中涉及了几种常见量化粒度：

- `per-tensor`
- `per-channel`
- `per-token`
- `per-block / per-group`

### Per-tensor

一个大 tensor 或一大块 tensor 共享一个 scale。

优点：

- 实现最简单
- 元数据开销最低

缺点：

- 对 KV cache 来说通常过于粗糙
- 容易被少量 outlier 影响
- 对这种动态变化的 activation/cache 类数据，精度一般较差

### Per-channel

每个 channel 使用一个 scale。对于 KV cache，可以理解成按 head 或某种逻辑切片分配 scale。

优点：

- 比 per-tensor 好很多

缺点：

- 在 KV cache 场景下不一定天然贴合，除非 channel 语义定义得很明确
- 仍然可能无法捕捉 token 维度上的变化

### Per-token

每个 token 一个 scale，或者更细地按 token 和 head/group 组合分配 scale。

优点：

- 表达能力强
- 对 activation 类数据通常精度较好

缺点：

- scale 开销明显更高
- layout 和计算链路会更复杂

### Per-block / per-group

这里我们重点澄清了 “block” 的含义。

在 KV cache 语义下，“block” 一般指缓存系统里的物理 block；但为了更方便落地实现，我们还评估过把当前传输中的 `IpcBlock` 作为量化单元。

在讨论的方案里，最终想采用的定义是：

- `per-block` = 每个传输的 `IpcBlock` 对应一个 scale

这样设计的优势是：

- 能复用当前已有的 `IpcBlock` 元数据链路
- 比 direct cast 有更好的局部精度
- 比 per-token 的改动和开销更小

对应的风险是：

- 精度提升依赖于单个 `IpcBlock` 内部的数据分布是否足够集中
- receiver 必须理解并处理 scale 元数据

## 曾经评估过的 per-block 方案

在正式实现之前，我们评估过给系统加一个选项，用环境变量控制两种模式：

- 当前 direct cast 模式
- per-block 量化模式

大致思路是：

- 在 `envcfg` 中新增量化模式配置
- sender 侧：
  - direct cast 模式：保持现有 BF16 -> FP8 direct cast
  - per-block 模式：为每个 `IpcBlock` 计算一个 scale，对 payload 做量化，并传输 `fp8_data + scales`
- receiver 侧：
  - 解析 payload 和 scales
  - 为了性能，将 scale 处理融合进 H2D kernel

### 额外涉及的链路修改

评估后发现，per-block 方案并不是只改一个 kernel 就够了，还会牵涉到整条链路。

需要改动的部分包括：

- env 配置接口
- TCP channel 的 send buffer 大小计算
- TCP payload 格式
- copy kernel 的 metadata staging
- receiver 侧 payload 解析
- H2D 的 scale-aware kernel
- 当前很多地方默认 “FP8 大小就是 BF16 的一半” 的字节规划逻辑

## 关键技术问题：per-block 能否提升精度

答案是：

- 相比当前 direct cast，per-block 通常可以提升量化精度

原因是：

- 每个 block 拥有自己的局部 scale
- 更少的值会因为全局范围不匹配而饱和
- 更少的小值会在固定格式下直接损失掉

**但是，这个精度收益只有在 scale 一直保留到最终消费者时，才是真正有意义的。**

## 讨论中最关键的架构结论

本次讨论里最重要的一个结论是：

如果 sender 使用了 per-block 量化，但 receiver 在 H2D 过程中只是把 scale 用掉，然后最终仍然写回一个不携带 scale 的 plain FP8 cache，那么 per-block 的大部分精度收益都会丢失。

原因如下：

- 带 scale 的量化本质上表达的是近似 `q * scale`
- 如果 scale 在最终常驻表示中被消掉，而数据又重新落回普通 FP8 表示，那么最终常驻格式依然没有局部 scale
- 从 decode 的实际使用角度看，这样的 resident cache 会重新接近当前 direct cast FP8 的语义

因此：

- “在 H2D 里乘一下 scale，然后存成普通 FP8” 并不足以完整保留 per-block 的价值

## 关于 decode 侧使用方式的讨论

我们也专门讨论了一个问题：decode 在收到量化后的 KV cache 之后，是不是只需要乘 scale 就可以直接用了？

结论是：

- 如果最终常驻 cache 仍然希望保持和今天一样的 plain FP8 格式，那么只乘 scale 是不够的

为了真正保留 per-block 量化的语义，至少需要满足以下三种之一：

1. receiver 侧反量化后存 BF16
2. receiver 侧存 `FP8 + scale`
3. decode kernel 原生理解并消费 block-scaled FP8

如果 receiver 的行为是：

- 乘 scale
- 然后再写回普通 FP8 cache

那么最终表示中 scale 还是被丢掉了。

## 最终建议

经过讨论，最终结论是：

- 当前需求下，direct cast 已经足够

### 为什么最后选择 direct cast

1. 当前 direct cast 链路明显更简单。
2. sender 侧已经能够直接产出 decode 期望的最终 resident FP8 格式。
3. receiver 不需要解析 scale，也不需要做反量化和再量化。
4. per-block 方案会引入较大的协议、kernel 和 layout 级改造。
5. 如果不连同 decode 侧 resident 格式和消费路径一起改，per-block 最核心的精度收益无法被完整保留。
6. 因此，在当前阶段，per-block 的复杂度和收益不匹配。

## 实际结论

当前推荐并接受的方向是：

- 保持现有 direct-cast 的 BF16 -> FP8 路径
- 暂时不引入 per-block 量化方案

这样做的结果是，系统继续保留以下优点：

- sender 侧量化逻辑简单
- TCP payload 结构简单
- receiver 侧 H2D 链路简单
- decode 侧拿到的 KV cache 可以直接使用，不需要额外 scale 逻辑

## 如果未来重新考虑 per-block

如果后续真的要重新启用 per-block 方案，更合理的方向应当是：

- sender：发送 `fp8_data + scale`
- receiver 常驻格式：`FP8 + scale`，或者完整反量化后的 BF16
- decode 消费路径：显式理解 scale

否则系统虽然承担了额外复杂度，但精度收益却无法真正保留下来。

## 简短问答总结

### 问：当前量化粒度是什么？

答：按元素直接从 BF16 cast 到 FP8 E4M3，没有 scale。

### 问：per-block 量化会提升精度吗？

答：通常会，比 direct cast 更好。

### 问：decode 侧是不是只要乘一下 scale 就可以用了？

答：如果最终常驻 cache 仍然是今天这种 plain FP8，并且期望和当前格式行为一致，那么只乘 scale 不够。

### 问：为什么现在没有采用 per-block？

答：因为要真正保留它的精度收益，需要连同 resident format 和 decode 消费路径一起调整；而当前 direct cast 方案已经足够简单且满足需求。

补充一下：vllm侧在使用e4m3 fp8的存储格式时，如果模型checkpoint中没有设置，默认的scale是1，这说明实际在bf16 -> fp8的转换中，与直接cast无异。这也说明，尽管当时开发的时候没有详细的考虑这部分，但是可以直接使用并通过精度测试的原因。
实际上，pd两侧只需要对kv cache的格式有共识即可，显然目前的设计有着约定俗成：直接cast。后续如果要设计更复杂的量化策略，只要d能够知道，且在启动服务时的转换对齐了这种量化策略，也是可行的。
