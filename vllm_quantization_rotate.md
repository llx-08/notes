# 量化中的 Rotate（旋转变换）技术及 vLLM 实现

## 1. 背景：为什么需要旋转变换

神经网络的权重和激活值中经常存在**离群值（outliers）**——少数维度上的数值远大于其他维度。当我们做低比特量化（如 INT4/FP4）时：

- 量化范围由最大值决定：`scale = max(|x|) / (2^{bits} - 1)`
- 离群值把 scale 拉得很大，导致大量正常值只能映射到很少几个量化级别
- 精度损失严重，模型质量退化

```
例：原始向量 x = [0.1, 0.2, 0.15, 0.12, 100.0]

用 INT4 (16 个级别) 量化：
  scale = 100.0 / 15 ≈ 6.67
  量化后 → [0, 0, 0, 0, 15]
  反量化 → [0.0, 0.0, 0.0, 0.0, 100.0]

前 4 个值完全丢失了！
```

### 加入 Rotate 后的效果

下面用一个简化的 4 维例子，完整展示旋转变换如何改善量化精度。

**原始向量**：`x = [0.3, 0.5, 0.2, 80.0]` （第 4 维是离群值）

#### 步骤 0：不旋转，直接量化（对照组）

```
直接用 INT4 量化（有符号 4-bit, 范围 -8 ~ +7）：
  scale = 80.0 / 7 ≈ 11.43
  量化：  round([0.3, 0.5, 0.2, 80.0] / 11.43) = round([0.026, 0.044, 0.018, 7.0])
                                                 = [0, 0, 0, 7]
  反量化：[0, 0, 0, 7] × 11.43 = [0.0, 0.0, 0.0, 80.0]

  量化误差：[0.3, 0.5, 0.2, 0.0]
  均方误差 MSE = (0.3² + 0.5² + 0.2² + 0²) / 4 = 0.095
  → 前 3 个值完全丢失！
```

#### 步骤 1：旋转

使用 4×4 归一化 Hadamard 矩阵（每个元素为 ±0.5）：

```
         [ 0.5   0.5   0.5   0.5]
H_4 =    [ 0.5  -0.5   0.5  -0.5]
         [ 0.5   0.5  -0.5  -0.5]
         [ 0.5  -0.5  -0.5   0.5]

x' = x @ H_4^T = [0.3, 0.5, 0.2, 80.0] @ H_4^T

x'[0] = 0.5×0.3 + 0.5×0.5 + 0.5×0.2 + 0.5×80.0 = 40.50
x'[1] = 0.5×0.3 - 0.5×0.5 + 0.5×0.2 - 0.5×80.0 = -39.90
x'[2] = 0.5×0.3 + 0.5×0.5 - 0.5×0.2 - 0.5×80.0 = -39.70
x'[3] = 0.5×0.3 - 0.5×0.5 - 0.5×0.2 + 0.5×80.0 = 39.90

x' = [40.50, -39.90, -39.70, 39.90]
```

离群值 80.0 被均匀分散到了 4 个维度，每个维度的值都在 ±40 左右！

#### 步骤 2：对旋转后的向量做量化

```
对 x' = [40.50, -39.90, -39.70, 39.90] 做 INT4 量化：
  scale = 40.50 / 7 ≈ 5.786
  量化：  round([40.50, -39.90, -39.70, 39.90] / 5.786) = round([7.0, -6.9, -6.86, 6.9])
                                                          = [7, -7, -7, 7]
  反量化：[7, -7, -7, 7] × 5.786 = [40.50, -40.50, -40.50, 40.50]

  旋转域的量化误差很小：[0.0, -0.60, -0.80, 0.60]
```

#### 步骤 3：逆旋转，恢复原始空间

```
x_recover = x'_dequant @ H_4    （Sylvester Hadamard: H^T = H, 所以逆变换也是乘 H）

x_recover = [40.50, -40.50, -40.50, 40.50] @ H_4

x_recover[0] = 0.5×40.50 + 0.5×(-40.50) + 0.5×(-40.50) + 0.5×40.50 = 0.0
x_recover[1] = 0.5×40.50 - 0.5×(-40.50) + 0.5×(-40.50) - 0.5×40.50 = 0.0
x_recover[2] = 0.5×40.50 + 0.5×(-40.50) - 0.5×(-40.50) - 0.5×40.50 = 0.0
x_recover[3] = 0.5×40.50 - 0.5×(-40.50) - 0.5×(-40.50) + 0.5×40.50 = 81.0

x_recover = [0.0, 0.0, 0.0, 81.0]
```

等一下——看起来和不旋转差不多？这是因为 4-bit 只有 16 级，这个例子的离群值太极端了（80 vs 0.3）。
让我们看一个**更现实的例子**，离群值没那么极端：

---

#### 更现实的例子

**原始向量**：`x = [1.2, 0.8, 1.5, 12.0]` （离群值比例 10:1，更贴近真实模型）

```
═══════════════════════════════════════════════════════════════
 【不旋转】直接 INT4 量化
═══════════════════════════════════════════════════════════════

  scale = 12.0 / 7 ≈ 1.714
  量化：  round([1.2, 0.8, 1.5, 12.0] / 1.714) = round([0.70, 0.47, 0.88, 7.0])
                                                  = [1, 0, 1, 7]
  反量化：[1, 0, 1, 7] × 1.714 = [1.714, 0.0, 1.714, 12.0]

  误差：  [0.514, -0.8, 0.214, 0.0]
  MSE  = (0.514² + 0.8² + 0.214² + 0²) / 4 = 0.237

═══════════════════════════════════════════════════════════════
 【加 Rotate】Hadamard 旋转 → INT4 量化 → 逆旋转
═══════════════════════════════════════════════════════════════

  旋转：  x' = x @ H_4^T
         x'[0] = 0.5×(1.2+0.8+1.5+12.0) = 7.75
         x'[1] = 0.5×(1.2-0.8+1.5-12.0) = -5.05
         x'[2] = 0.5×(1.2+0.8-1.5-12.0) = -5.75
         x'[3] = 0.5×(1.2-0.8-1.5+12.0) = 5.45
         x' = [7.75, -5.05, -5.75, 5.45]    ← 分布均匀多了！

  量化：  scale = 7.75 / 7 ≈ 1.107
         round([7.75, -5.05, -5.75, 5.45] / 1.107) = round([7.0, -4.56, -5.19, 4.92])
                                                     = [7, -5, -5, 5]
         反量化 = [7.75, -5.536, -5.536, 5.536]

  逆旋转：x_recover = x'_dequant @ H_4
         x_recover[0] = 0.5×(7.75 + 5.536 + 5.536 + 5.536)  ≈ 1.107 × 0.5 × ... 
                       = 0.5 × (7.75 - 5.536 - 5.536 + 5.536) ... 

  （直接给出计算结果）
         x_recover[0] = 0.5×(7.75+(-5.536)+(-5.536)+5.536)   = 0.5×2.214  = 1.107
         x_recover[1] = 0.5×(7.75-(-5.536)+(-5.536)-5.536)   = 0.5×2.214  = 1.107
         x_recover[2] = 0.5×(7.75+(-5.536)-(-5.536)-5.536)   = 0.5×2.214  = 1.107
         x_recover[3] = 0.5×(7.75-(-5.536)-(-5.536)+5.536)   = 0.5×24.358 = 12.179

         x_recover = [1.107, 1.107, 1.107, 12.179]

  误差：  [1.107-1.2, 1.107-0.8, 1.107-1.5, 12.179-12.0]
        = [-0.093, 0.307, -0.393, 0.179]
  MSE  = (0.093² + 0.307² + 0.393² + 0.179²) / 4 = 0.073

═══════════════════════════════════════════════════════════════
 对比总结
═══════════════════════════════════════════════════════════════

                    不旋转              加 Rotate
  原始值        [1.2, 0.8, 1.5, 12.0]   [1.2, 0.8, 1.5, 12.0]
  恢复值        [1.71, 0.0, 1.71, 12.0] [1.11, 1.11, 1.11, 12.18]
  MSE           0.237                    0.073
  最大绝对误差  0.8 (第2维完全丢失)       0.393
  精度提升      —                        MSE 降低 69%

关键观察：
  - 不旋转时：第 2 维 (0.8) 被量化为 0，完全丢失
  - 加旋转后：所有维度都保留了合理的近似值，没有任何维度被"牺牲"
  - 旋转把离群值的能量分散到所有维度，让 scale 更合理
```

## 2. Rotate 的核心思想

**Rotate（旋转）** 通过在量化前施加一个**正交变换矩阵 R**（如 Hadamard 矩阵），将离群值的能量"分散"到所有维度上，使分布更均匀、更适合量化。

### 2.1 数学原理

对于线性层 `y = x @ W`，旋转变换插入一对互逆的正交矩阵：

```
y = x @ W
  = x @ (R^T @ R) @ W        （插入 R^T @ R = I）
  = (x @ R^T) @ (R @ W)      （重新结合）
  = x' @ W'
```

其中：
- `W' = R @ W`：离线旋转权重（量化前做一次）
- `x' = x @ R^T`：在线旋转激活值（推理时每次做）

因为 R 是正交矩阵（`R^T @ R = I`），整个变换数学上**严格等价**，不引入任何近似误差。旋转后的 W' 和 x' 的分布更"圆润"，离群值被分散，量化精度大幅提升。

### 2.2 为什么选 Hadamard 矩阵

实践中最常用的旋转矩阵是**Hadamard 矩阵**，原因：

| 性质 | 说明 |
|------|------|
| 正交性 | `H @ H^T = n * I`，归一化后即正交矩阵 |
| 对称性 | Sylvester Hadamard 满足 `H = H^T`，因此 `H = H^{-1}`（归一化后） |
| 递归结构 | 可以像 FFT 一样递归计算，复杂度 O(n log n) 而非 O(n^2) |
| 元素简单 | 所有元素为 ±1，不需要浮点乘法 |
| 均匀分散 | 能最大程度地将能量均匀分配到所有维度 |

Sylvester Hadamard 矩阵的递归构造：

```
H_1 = [1]

H_2 = [ H_1   H_1 ]   = [ 1   1 ]
       [ H_1  -H_1 ]     [ 1  -1 ]

H_4 = [ H_2   H_2 ]   = [ 1  1  1  1 ]
       [ H_2  -H_2 ]     [ 1 -1  1 -1 ]
                          [ 1  1 -1 -1 ]
                          [ 1 -1 -1  1 ]
```

### 2.3 相关论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| **QuIP** (Quantization with Incoherence Processing) | 2023 | 提出用随机正交矩阵做旋转，降低量化误差 |
| **QuIP#** | 2024 | 改用 Hadamard 矩阵，支持更高效的在线变换 |
| **SpinQuant** | 2024 | 用 Cayley 参数化优化旋转矩阵 |
| **FP-Quant** | 2024 | 将 Hadamard 旋转融合到 FP4 量化 kernel 中 |
| **Hadacore** | 2024 | 高效 Hadamard 变换的 CUDA kernel 实现 |

## 3. vLLM 中的实现

vLLM 中有**两套**旋转量化实现，以及一个底层的高效 Hadamard kernel。

### 3.1 Hadacore CUDA Kernel

**文件**：`vllm/_custom_ops.py`

```python
def hadacore_transform(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    使用 Hadacore (https://arxiv.org/abs/2412.08832) kernel 执行 Hadamard 变换。
    利用 Sylvester Hadamard 的递归性质，不需要显式存储变换矩阵。
    Sylvester Hadamard 变换也是对称的 (H = H^T)，
    因此同一个函数既是正变换也是逆变换。
    """
    return torch.ops._C.hadacore_transform(x, inplace)
```

关键特性：
- 利用递归结构，复杂度 O(n log n)
- 不需要显式存储矩阵，节省内存
- 由于对称性，正变换和逆变换是同一个操作（做两次 = 恒等变换）
- 支持 inplace 操作

测试验证（`tests/kernels/quantization/test_hadacore.py`）：

```python
# 正确性：与 dense Hadamard 矩阵乘结果一致
y = ops.hadacore_transform(x.clone())
y_true = (x.to(hadamard.dtype) @ hadamard.T).to(y.dtype)
assert torch.allclose(y, y_true)

# 对称性：做两次变换 = 恒等
y = ops.hadacore_transform(y)
assert torch.allclose(y, x)
```

### 3.2 Compressed Tensors Transform 框架

**目录**：`vllm/model_executor/layers/quantization/compressed_tensors/transform/`

这是一个**通用的旋转变换框架**，通过配置文件驱动，可以灵活地在不同层、不同位置（输入/输出）应用变换。

#### 核心类：`HadamardTransform`（module.py）

```python
class HadamardTransform(torch.nn.Module):
    """
    处理变换矩阵的权重加载、后处理和应用。
    配合 CompressedTensorsLinearTransformMethod 使用。
    """

    def forward(self, value: Tensor, part_id: int = 0) -> Tensor:
        # 路径 1：Hadamard 类型 → 使用高效的 Hadacore kernel
        if self.transforms[part_id].scheme.type == "hadamard":
            if self.transforms[part_id].scheme.head_dim is not None:
                # 按 head_dim 分块做变换
                value = value.unflatten(-1, (-1, weight_size))
                value = ops.hadacore_transform(value)
                value = value.flatten(-2, -1)
                return value
            return ops.hadacore_transform(value)

        # 路径 2：通用 dense 矩阵 → 矩阵乘法
        else:
            weight = self.weight.partitions[part_id]
            scale = self.scales[part_id]
            return dispatch_unquantized_gemm()(self, value, weight, None) * scale
```

#### 线性层包装：`CompressedTensorsLinearTransformMethod`（linear.py）

将旋转变换包裹在量化线性层的前后：

```python
def apply(self, layer, x, bias=None):
    # 1. 输入旋转变换（在线）
    if self.input_transform is not None:
        x = self.input_transform(x)

    # 2. 量化矩阵乘法
    x = self.quant_method.apply(layer, x, bias)

    # 3. 输出旋转变换（如需要）
    if self.output_transform is not None:
        for part_id, (start, length) in enumerate(self.partition_ranges):
            x[:, start : start + length] = self.output_transform(
                x[:, start : start + length].clone(), part_id=part_id
            )

    return x
```

数据流示意：

```
输入 x
  │
  ▼
[Input Rotate]  x' = x @ R^T  （Hadacore 或 dense matmul）
  │
  ▼
[Quantized MatMul]  y = x' @ W_quantized  （INT4/FP4 量化推理）
  │
  ▼
[Output Rotate]  y' = y @ R_out  （可选，视配置而定）
  │
  ▼
输出 y'
```

#### 配置驱动

变换规则通过 `TransformConfig` 配置，指定：
- `targets`：匹配哪些层（如 `qkv_proj`, `gate_up_proj` 等）
- `location`：变换位置（INPUT / OUTPUT）
- `type`：变换类型（`hadamard` 或自定义 dense 矩阵）
- `head_dim`：按 head 维度分块变换（用于 attention 相关层）

### 3.3 FP-Quant 融合实现

**文件**：`vllm/model_executor/layers/quantization/fp_quant.py`

FP-Quant 将 Hadamard 旋转**融合到量化 kernel 中**，避免额外的 kernel launch 开销。

```python
class FPQuantConfig(QuantizationConfig):
    def __init__(self,
        hadamard_group_size: int = 32,    # Hadamard 变换的分组大小
        forward_dtype: str = "mxfp4",     # 量化类型：mxfp4 或 nvfp4
        forward_method: str = "abs_max",  # 量化方法
        ...
    ):
```

推理流程（以 mxfp4 为例）：

```python
def quantized_forward(x, qweight, weight_scales, ..., forward_hadamard_matrix, ...):
    x_flat = x.contiguous().flatten(end_dim=-2)

    # 融合操作：Hadamard 旋转 + FP4 量化，在一个 CUDA kernel 中完成
    x_flat_q, x_flat_scales = torch.ops.vllm.fused_quantize_mx(
        x_flat,
        forward_hadamard_matrix,  # 32x32 的 Hadamard 矩阵
        forward_method
    )

    # 量化矩阵乘法
    y = torch.ops.vllm.matmul_mxf4_bf16(
        x_flat_q, qweight, x_flat_scales, weight_scales, alpha
    )
    return y
```

FP-Quant 的特点：
- 使用**分组 Hadamard**（默认 group_size=32），而非全维度旋转
- **正向和反向**各有一个 Hadamard 矩阵（`forward_hadamard_matrix` / `backward_hadamard_matrix`）
- 旋转和量化**融合在一个 kernel** 中（`fusedQuantizeMx` / `fusedQuantizeNv`），减少显存访问
- 支持 mxfp4 和 nvfp4 两种 FP4 格式

## 4. 两种实现的对比

| 特性 | Compressed Tensors Transform | FP-Quant |
|------|------------------------------|----------|
| **变换粒度** | 全维度或按 head_dim 分块 | 固定分组（默认 32） |
| **变换方式** | Hadacore kernel 或 dense matmul | 融合 kernel（旋转 + 量化一体） |
| **灵活性** | 高，配置驱动，支持任意层/位置 | 低，专为 FP4 量化设计 |
| **性能** | 旋转和量化分开执行 | 融合执行，减少 kernel launch |
| **适用场景** | 通用量化变换 | mxfp4 / nvfp4 专用 |
| **TP 支持** | 暂不支持 | 需对齐 group_size |

## 5. 关键代码文件索引

| 文件路径 | 说明 |
|----------|------|
| `vllm/_custom_ops.py` (L3490-3503) | `hadacore_transform` 接口 |
| `vllm/model_executor/layers/quantization/compressed_tensors/transform/module.py` | `HadamardTransform` 变换模块 |
| `vllm/model_executor/layers/quantization/compressed_tensors/transform/linear.py` | `CompressedTensorsLinearTransformMethod` 线性层包装 |
| `vllm/model_executor/layers/quantization/compressed_tensors/transform/utils.py` | `TransformTuple` 工具类 |
| `vllm/model_executor/layers/quantization/compressed_tensors/transform/schemes/linear_qutlass_nvfp4.py` | NvFP4 专用实现 |
| `vllm/model_executor/layers/quantization/fp_quant.py` | FP-Quant 融合实现 |
| `tests/kernels/quantization/test_hadacore.py` | Hadacore kernel 测试 |
