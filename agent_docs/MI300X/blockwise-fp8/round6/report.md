# Round 6: Backward waves_per_eu=2 占用率优化

## 优化概述

对 NN 和 TN backward 路径设置 `waves_per_eu=2`，引导 AMD GPU 编译器为精确匹配的占用率做寄存器分配，大幅提升 backward 性能。

## 技术分析

### waves_per_eu 参数原理

MI300X 每 CU 有 4 个 SIMD 单元，每个 SIMD 有 512 个 VGPR（向量通用寄存器）。

```
┌────────────── CU ──────────────┐
│ SIMD0   SIMD1   SIMD2   SIMD3 │
│ 512 VGPRs per SIMD             │
│                                │
│ num_warps=8 → 8 wavefronts     │
│ 8 warps / 4 SIMDs = 2 waves/SIMD │
└────────────────────────────────┘
```

#### `waves_per_eu=0`（Round 5 默认）

编译器自动选择目标占用率，可能为更高占用率（如 4 waves/SIMD）预留寄存器：

$$\text{VGPRs/wave} = \frac{512}{4} = 128 \quad \text{（过度约束）}$$

即使实际只有 2 waves/SIMD，编译器仍可能按 128 VGPR/wave 分配，导致寄存器溢出（spill）。

#### `waves_per_eu=2`（Round 6）

明确告知编译器目标占用率为 2 waves/SIMD：

$$\text{VGPRs/wave} = \frac{512}{2} = 256 \quad \text{（充分利用）}$$

编译器可以：
1. 使用更多寄存器存放 accumulator 和中间结果
2. 减少寄存器溢出到 LDS/HBM 的开销
3. 更好地调度指令以隐藏 MFMA 延迟

### 为什么只对 Backward 有效

| 路径 | async_copy | SCALE_2D_B | 寄存器压力 | waves_per_eu=2 效果 |
|------|-----------|------------|-----------|-------------------|
| NT Forward | 是 | True（标量 b_s） | 低 | 无显著提升（-1.06%） |
| NN Backward | 是 | True（标量 b_s） | 中 | **+9-12%** |
| TN Backward | 否 | False（外积 a_s⊗b_s） | 高 | **+7-13%** |

- **Forward（NT）**：SCALE_2D_B=True 路径简单（b_s 是标量），寄存器使用本就充裕，额外寄存器空间无法利用。async_copy 的 DMA 管线已经高效隐藏延迟。
- **Backward（NN/TN）**：寄存器压力较大。TN 的外积 scale 尤其消耗寄存器。额外 128 VGPR/wave 空间让编译器减少溢出并改善调度。

### 失败的尝试：TN pre-transpose + async_copy

在确定 `waves_per_eu=2` 之前，尝试了 TN 数据预转置（使 K 维度连续）+ 启用 async_copy：
- 大 K 获得 +2-6% 提升
- 小 K 严重回退（最大 -28%）
- 原因：SCALE_2D_B=False + async_copy 的 LDS pipeline 导致寄存器/LDS 竞争

## 代码修改

仅修改 `gemm_fp8_kernel.py` 中 `_blockwise_nn` 和 `_blockwise_tn` 的 `waves_per_eu` 参数：

```python
# _blockwise_nn (backward grad_X)
waves_per_eu=2,  # was 0

# _blockwise_tn (backward grad_W)
waves_per_eu=2,  # was 0

# _blockwise_nt (forward) — unchanged
waves_per_eu=0,
```

## 性能结果

### Forward

| 指标 | 值 |
|------|-----|
| Round 5 平均 | 493.34 TFLOPS |
| Round 6 平均 | 493.77 TFLOPS |
| Geomean vs R5 | **+0.10%** |
| Geomean vs Baseline | **+16.26%** |

Forward 无实质性变化（仅修改了 backward 路径）。

### Backward

| 指标 | 值 |
|------|-----|
| Round 5 平均 | 357.92 TFLOPS |
| Round 6 平均 | 392.64 TFLOPS |
| Geomean vs R5 | **+9.84%** |
| Geomean vs Baseline | **+68.01%** |
| 提升 shape 数 | **69 / 69 (100%)** |
| 回退 shape 数 | **0 / 69 (0%)** |

**Top 5 提升（Backward）**：

| Shape | R5 TFLOPS | R6 TFLOPS | 提升 |
|-------|-----------|-----------|------|
| 16384x16384x53248 | 351.3 | 397.8 | +13.2% |
| 32768x16384x53248 | 348.4 | 393.7 | +13.0% |
| 16384x18432x16384 | 376.5 | 424.4 | +12.7% |
| 16384x106496x16384 | 359.0 | 404.6 | +12.7% |
| 8192x106496x16384 | 355.6 | 399.8 | +12.4% |

### 精度验证

192/192 blockwise FP8 accuracy tests 全部通过。

### 累计优化效果（Round 0 → Round 6）

| 指标 | Baseline | Round 6 | 累计提升 |
|------|----------|---------|----------|
| Forward 平均 | 429.07 TFLOPS | 493.77 TFLOPS | **+15.1%** |
| Backward 平均 | 234.56 TFLOPS | 392.64 TFLOPS | **+67.4%** |
| Benchmark Status | 84 PASS | 84 PASS | 零错误 |
| Accuracy Tests | 192 passed | 192 passed | 零回退 |
