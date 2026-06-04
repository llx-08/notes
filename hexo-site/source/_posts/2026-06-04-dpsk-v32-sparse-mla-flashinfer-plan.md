---
title: Blackwell + FlashInfer + SparseMLA 在 KVT 上的支持计划
date: 2026-06-04
tags: []
---
# Blackwell + FlashInfer + SparseMLA 在 KVT 上的支持计划

> 基于 `/mnt/data/llx/vllm` `llx/github` 分支(含 FlashInfer sparse MLA 后端)与 `/mnt/data/llx/vllm` `develop` 分支(含 HybridConnector + KVTBackend)+ `/mnt/data/llx/blade-kvt` 当前源码,2026-06-02 整理。

## 0. 结论先行(TL;DR)

**与最初设想不同**:重新阅读 vllm 主仓与 blade-kvt 源码后发现 — **现有 `DPSK_V32_SPARSE_MLA_SHAPE = 4` 路径在物理布局上与 Blackwell + FlashInfer + SparseMLA 完全兼容**,在双方使用相同 `(block_size, kv_cache_dtype)` 配置时,**几乎不需要新增 cache shape**。

真正需要做的事是:

1. (vllm 侧)放宽几处隐含假设(register_kv_caches 注释、报错文案等);
2. (vllm 侧)在 `_set_worker_envs` 里加显式的 sparse MLA + FlashInfer 路径覆盖测试与日志;
3. (blade-kvt 侧)如果未来需要支持"P 用 Hopper FA + fp8_ds_mla / D 用 Blackwell FlashInfer + fp8"这种异构混部,再单独处理 token_bytes 转换;
4. 加端到端测试覆盖。

## 1. 关键事实

### 1.1 FlashAttention sparse MLA(Hopper)的 KV cache 形状

`vllm/v1/attention/backends/mla/flashmla_sparse.py:139-150`:

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
    if cache_dtype_str == "fp8_ds_mla":
        # V3.2 main MLA: 656-byte custom storage format
        return (num_blocks, block_size, 656)
    else:
        return (num_blocks, block_size, head_size)  # 576
```

- `get_supported_kernel_block_sizes()` → `[64]`(只允许 64)
- `get_supported_head_sizes()` → `[576]`(= 512 kv_lora_rank + 64 qk_rope_head_dim)
- 支持 cache dtype: `auto / bfloat16 / fp8_ds_mla / fp8`

### 1.2 FlashInfer sparse MLA(Blackwell)的 KV cache 形状

`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:133-145`:

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
    return (num_blocks, block_size, head_size)  # 576

@classmethod
def get_required_kv_cache_layout(cls):
    return "HND"
```

- `get_supported_kernel_block_sizes()` → `[32, 64]`(Blackwell 多支持 32)
- `get_supported_head_sizes()` → `[576]`
- 支持 cache dtype: `auto / float16 / bfloat16 / fp8 / fp8_e4m3` — **不支持 `fp8_ds_mla` 656 字节自定义格式**
- **`supports_compute_capability` 限定 capability.major == 10**(SM10x,Blackwell)

### 1.3 Indexer cache 在两种后端下完全一致

`vllm/v1/attention/backends/mla/indexer.py:118-155`(`DeepseekV32IndexerBackend`):

```python
return (num_blocks, block_size, head_size)   # head_size ∈ {32, 64, 128}
```

block_size 固定 64(非 ROCm)。FlashInfer 与 FlashAttention 共用同一 indexer backend。

### 1.4 现有 `DPSK_V32_SPARSE_MLA_SHAPE = 4` 实际行为

`blade-kvt/kvtransfer/src/cache_transfer_spec.cpp:257-266`:

```cpp
{DPSK_V32_SPARSE_MLA_SHAPE, {
  /*dim_order=*/ {D::BLOCK, D::TOKEN, D::KV, D::HEAD, D::HEAD_DIM},
  /*num_tensors=*/ 0,                        // 运行时从 token_sizes.size() 推
  /*rank0_only_when_same_sizes=*/ true,
  /*force_peqd_tpkind=*/ true,               // MLA: 每 rank 持有全量 → 强制走 P==D
}},
```

**关键洞察**:`force_peqd_tpkind=true` 使所有 sparse MLA 流量走 P==D 分支,而 P==D 分支在 `parse_block_dpsk.cpp:88-127` (`vllm_parse_block_send_multi_tensor_p_eq_d`) 实际只做"按张量整段拷贝":

```cpp
for (size_t i = 0; i < token_sizes.size(); ++i) {
    do_parse_block_send_p_eq_d(block_sizes[i], token_sizes[i], task, send_blocks[i]);
}
```

即每张量 `block_size * token_size` 字节整段 memcpy,**不解释内部 KV/HEAD 布局**,所以 `dim_order` 在 P==D 路径下基本是装饰性的。这意味着 FA 还是 FlashInfer,只要每张量的 `(num_blocks, block_size, dim)` 一致,字节布局就一致,传输就正确。

### 1.5 vllm 侧 cache_shape 选择已经覆盖 FlashInfer sparse MLA

`vllm/v1/hybrid_connector/kvtbackend.py:516-521`(develop 分支):

```python
if use_mla():
    if use_sparse_mla():
        os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "4")
```

- `use_mla()` 检查 `"mla" in attn_backend.lower()` → `FLASHINFER_MLA_SPARSE` 命中 ✓
- `use_sparse_mla()` 在此基础上检查 `"sparse" in attn_backend.lower()` → 命中 ✓

所以 **FlashInfer sparse MLA 已经被路由到 shape 4**,且 `tensor_shape_dict` 多张量分支 (`kvtbackend.py:1735-1741`) 的 `assert len(cache_shape) == 3` 对 MLA(576)+ Indexer(32/64/128)双张量场景也已成立。

## 2. FA sparse MLA vs FlashInfer sparse MLA 对照表

| 维度 | FA sparse(Hopper) | FlashInfer sparse(Blackwell) | KVT 影响 |
|---|---|---|---|
| MLA 张量物理形状 | `(num_blocks, block_size, 576)` 默认 / `(_, _, 656)` 当 `fp8_ds_mla` | `(num_blocks, block_size, 576)` | **同构场景一致**;异构 fp8_ds_mla 混部时 token_bytes 不同 |
| Indexer 张量形状 | `(num_blocks, 64, head_size)` head_size∈{32,64,128} | 同 | 一致 |
| 张量数 / layer | 2(MLA + Indexer) | 2(同) | 一致,走 multi-tensor 路径 |
| `block_size` 允许值 | `[64]` | `[32, 64]` | 同构都用 64 时无影响;若 D 用 32、P 用 64 → block_bytes 不同,需混部支持 |
| 支持的 cache dtype | `auto/bf16/fp8/fp8_ds_mla` | `auto/fp16/bf16/fp8/fp8_e4m3` | **`fp8_ds_mla` 是 FA 专属**;FlashInfer 用标准 fp8 |
| `get_required_kv_cache_layout` | None | `"HND"` | 对单 head MLA 而言等价(num_kv_heads=1 时 HND 与 NHD 物理布局相同) |
| `get_kv_cache_stride_order` | 无(默认 identity) | 默认 identity | 一致 |
| `compute_capability` | major∈{9,10} | major == 10 | FlashInfer 强制 Blackwell |
| KV/V 分离 | 无(MLA 单张量 latent) | 无 | `force_peqd_tpkind=true` 路径不解释 KV/HEAD |

### 三类部署场景

| 场景 | P 节点 | D 节点 | 是否兼容现状 |
|---|---|---|---|
| **A 同构 Blackwell** | Blackwell + FlashInfer + bf16/fp8 | 同 | ✅ **现有代码已支持**,无需改动 |
| **B 同构 Hopper** | Hopper + FA + bf16/fp8 | 同 | ✅ 一直支持 |
| **C 异构 Hopper→Blackwell(同 dtype)** | Hopper + FA + bf16 | Blackwell + FlashInfer + bf16 | ✅ 双方都是 `(_, 64, 576)` |
| **C' 异构混 dtype** | Hopper + FA + fp8_ds_mla | Blackwell + FlashInfer + fp8 | ❌ token_bytes 不同(656 vs 576),需新增异构通路 |
| **D 异构 block_size** | Hopper + FA + bf16 + bs=64 | Blackwell + FlashInfer + bf16 + bs=32 | ❌ block_bytes 不同 |

## 3. 修改清单

### 3.1 P0 — 必做,代价小,价值高

| # | 文件 | 改动 |
|---|---|---|
| P0-1 | `vllm/v1/hybrid_connector/kvtbackend.py:516-521` | 在 `if use_sparse_mla(): "4"` 处加注释 + INFO 日志,注明此 shape 同时覆盖 FA / FlashInfer 两种 sparse MLA backend |
| P0-2 | `vllm/v1/hybrid_connector/kvtbackend.py:1731-1741` 与 `2712-2717` | 把 `assert len(cache_shape) == 3, "Currently only support Dpsk-V32"` 文案改成 `"Currently only support Dpsk-V32 sparse MLA (FA / FlashInfer)"`,并补一句注释说明为什么 FlashInfer 也走这里 |
| P0-3 | `vllm/v1/hybrid_connector/kvtbackend.py` register_kv_caches 入口处 | 加日志 `logger.info("KVT sparse MLA detected: backend=%s, dtype=%s, block_size=%s", VLLM_ATTENTION_BACKEND, cache_dtype, block_size)`,便于线上排查 |
| P0-4 | `vllm/v1/hybrid_connector/engine_proxy.py:337-349` `use_flashinfer()` | `if capability.major != 10: raise ValueError("Currently KVT only support HND layout of flashinfer")` 当前文案误导 — 把消息改成更准确的 `"FlashInfer KVT path requires Blackwell (SM 10.x), got SM%d.%d"` |

### 3.2 P1 — 防御性硬化

| # | 文件 | 改动 |
|---|---|---|
| P1-1 | `vllm/v1/hybrid_connector/kvtbackend.py` `_set_worker_envs` 末尾 | 加显式校验:当 `use_sparse_mla() and use_flashinfer()` 时,`assert cfg.cache_config.block_size in (32, 64)`(对齐 `flashinfer_mla_sparse.py:73`);当 `use_sparse_mla() and not use_flashinfer()` 时,`assert cfg.cache_config.block_size == 64` |
| P1-2 | `vllm/v1/hybrid_connector/kvtbackend.py` `_set_worker_envs` 末尾 | 加 dtype 校验:当 `use_sparse_mla() and use_flashinfer()` 时,断言 `cache_dtype` 不在 `{"fp8_ds_mla"}` — FlashInfer 不支持 656 字节自定义格式,提前拒绝 |
| P1-3 | `blade-kvt/kvtransfer/src/cache_transfer_spec.cpp` `build_transfer_plan` 入口 | P==D 分支里加一行 `assert(src.token_sizes == dst.token_sizes)` 已有;在断言失败时打印更详细的 message,把 token_sizes 双方都列出来,便于排查 fp8_ds_mla / fp8 混部 |
| P1-4 | `blade-kvt/kvtransfer/include/envcfg.h:35` 注释 | 把 `// Each layer contains two tensors: k tensor for select and mla tensor for attention.` 扩展为 `// Each layer contains two tensors: indexer cache (for sparse selection) and main MLA cache (kv_lora_rank + qk_rope_head_dim). Compatible with both FlashAttention and FlashInfer attn backends as long as P/D use the same (block_size, kv_cache_dtype).` |
| P1-5 | `blade-kvt/kvtransfer/docs/cache_transfer_spec.md:458` `DPSK_V32_SPARSE_MLA_SHAPE` 章节 | 补一段 "支持的 attn 后端" 表格,列明 FlashAttention(Hopper) / FlashInfer(Blackwell) 都走该 shape,以及 block_size / dtype 兼容性约束 |

### 3.3 P2 — 测试覆盖

| # | 文件 | 改动 |
|---|---|---|
| P2-1 | `vllm/tests/v1/kv_connector/hybrid/test_hybrid_connector.py` | 加单测 `test_dpsk_v32_flashinfer_cache_shape`:mock `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE`,mock device capability=(10, 0),断言 `_set_worker_envs` 后 `BLLM_KVTRANS_CACHE_SHAPE == "4"`,且 `register_kv_caches` 不抛错 |
| P2-2 | `blade-kvt/kvtransfer/benchmarks/` | 加一个 standalone 测试程序构造两组 `(num_blocks=N, block_size=32, dim=576) + (num_blocks=N, block_size=32, dim=64)` 的 fake KV cache,通过 P==D 路径在两个进程间走一遍 KVT,对比 src/dst bytes 一致 |
| P2-3 | (可选)`vllm/test_pd_kvt_run.sh` | 加一个 `--attn-backend FLASHINFER_MLA_SPARSE` 的运行模式,用于 e2e 联调 |

### 3.4 P3 — 异构混部(场景 C' / D),非阻塞

| # | 内容 |
|---|---|
| P3-1 | 评估是否需要支持 Hopper FA(fp8_ds_mla, 656B)→ Blackwell FlashInfer(fp8, 576B)的字节级转换。当前 `parse_block_dpsk.cpp` 假设 src/dst token_size 相同,需要在 D 侧引入按 dtype 转换的 staging buffer,工作量较大且依赖业务上是否真要混部 |
| P3-2 | 评估 P 用 block_size=64 / D 用 block_size=32 的混部需求。这等价于 ntpb 不同的"逻辑 P==D",可在 `parse_block_dpsk.cpp` 加一个新的 helper 处理 |
| P3-3 | 如果 P3-1 或 P3-2 真要做,**才**新增 `DPSK_V32_SPARSE_MLA_FLASHINFER_SHAPE = 8`,在 `cache_transfer_spec.cpp` registry 中显式带上 dtype/block_size 元数据,并在 `tx_stub.cpp` 增加分支 |

## 4. 验证步骤

1. **同构 Blackwell A 场景**:
   - P 与 D 都 `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE`,`block_size=64`,`kv_cache_dtype=auto`
   - 启动 PD 分离 → 发请求 → 检查 KV 命中率与生成正确性
   - 应该**直接可用**,无需任何修改

2. **同构 Blackwell + block_size=32**:
   - 同上但 `block_size=32`
   - 期望:P0/P1 修改完成后,正常工作

3. **异构(场景 C)**:
   - P:Hopper + `FLASH_MLA_SPARSE` + bf16 + bs=64
   - D:Blackwell + `FLASHINFER_MLA_SPARSE` + bf16 + bs=64
   - 期望:正常工作

4. **应该被拒绝的场景**:
   - P:Hopper + `FLASH_MLA_SPARSE` + `fp8_ds_mla` + bs=64
   - D:Blackwell + `FLASHINFER_MLA_SPARSE` + `fp8` + bs=64
   - 期望:P1-3 的 `token_sizes != dst.token_sizes` 断言能在 PD 握手阶段拦下来,而不是在传输中悄悄出错

## 5. 落地优先级

```
P0 (~半天) → 同构 Blackwell 场景验证(1 天) → P1 (~1 天) → P2 (~2 天) → 上线
                                                           │
                                                           └→ P3(按需,规划单独迭代)
```
