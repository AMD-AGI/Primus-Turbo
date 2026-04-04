# Round 10: 256×256 Tile + Multi-Block B-Scale Kernel

## 目标

通过支持 256×256 tile 提升大矩阵 FP8 blockwise GEMM 性能。

## 分析

### 背景

Round 2 的 256×128 tile 已经在 M≥2048 时带来 +49.7% forward TFLOPS 提升。
理论上 256×256 tile 可以进一步减少 tile 数量、提升计算密度。

### 关键发现：B-Scale 索引问题

Weight 量化 (`quant_fp8_blockwise_for_weight_impl`) 产生真正的 2D block scale：

$$\text{b\_scale\_inv} \in \mathbb{R}^{\lceil N/128 \rceil \times \lceil K/128 \rceil}$$

当 `BLOCK_N=256` 时，每个 tile 覆盖 2 个 N-scale block。
原始 kernel 只加载 1 个 scalar scale per tile (`pn * stride_bs_0`)，
导致第二个 N-block 使用错误的 scale → 精度退化。

### Kernel 修改

引入 `N_SCALE_BLOCKS = BLOCK_N // 128` (constexpr)：

```python
# Base pointer 调整
bs_ptr_base = B_scales_ptr + (pn * N_SCALE_BLOCKS) * stride_bs_0

# 多 scale block 加载 (仅 BLOCK_N > 128 时编译)
if N_SCALE_BLOCKS > 1:
    bs0 = tl.load(bs_ptr_base + ki * stride_bs_1)
    bs1 = tl.load(bs_ptr_base + stride_bs_0 + ki * stride_bs_1)
    bn_idx = tl.arange(0, BLOCK_N)
    b_s_vec = tl.where(bn_idx < 128, bs0, bs1)
    acc += partial * a_s[:, None] * b_s_vec[None, :]
```

当 `BLOCK_N=128` 时 `N_SCALE_BLOCKS=1`，多 scale 分支被编译器消除 → 零开销。

## 精度验证

| 配置 | 测试数 | 结果 |
|------|--------|------|
| 256×256 tile, NT forward (14 shapes) | 14 | 14 PASS, SNR ≥ 28.7 dB |
| 256×256 tile, NN backward (4 shapes) | 4 | 4 PASS, SNR ≥ 28.7 dB |
| Regular blockwise pytest | 192 | 192 PASS, 0 FAIL |

## 性能对比

### 256×256 vs 256×128 vs 128×128 (kernel-level, M=N=K=4096)

| Tile | TFLOPS | 相对 256×128 |
|------|--------|------------|
| 256×128 | 1377 | baseline |
| 256×256 | 284 | -79.4% |
| 128×128 | 968 | -29.7% |

### 256×256 寄存器压力分析

```
256×256 accumulator: 256×256 / (8 warps × 64 threads) = 128 VGPRs/thread
256×128 accumulator: 256×128 / (8 warps × 64 threads) =  64 VGPRs/thread
```

128 VGPRs 导致 occupancy 大幅下降，register spilling 造成 4-5× 性能回退。
测试 `num_warps=4/8/16` 均无改善。

### Full benchmark (reverted to 256×128)

| 指标 | Round 2 (baseline) | Round 10 | 变化 |
|------|-------------------|----------|------|
| Avg Forward TFLOPS | 1297.4 | 1293.1 | -0.3% |
| Avg Backward TFLOPS | 976.8 | 974.1 | -0.3% |

**结论**: kernel 修改对 256×128 路径零性能影响。

## 决策

**不合入 256×256 tile 选择**：register pressure 导致 4-5× 性能退化。

**保留 multi-block scale kernel 代码**：
- 对 `BLOCK_N=128` 路径零开销 (constexpr 分支被消除)
- 为未来硬件 (更大 VGPR file) 预留能力
- 已验证精度正确

**tile 选择回退到 Round 2 配置**：`_select_blockwise_tile_gfx950` 继续使用 256×128。

## 修改的文件

| 文件 | 变更 |
|------|------|
| `primus_turbo/triton/gemm/gemm_fp8_kernel.py` | 添加 N_SCALE_BLOCKS multi-block scale loading (zero-cost for BN=128) |
| `primus_turbo/triton/gemm/gemm_fp8_kernel.py` | `_select_blockwise_tile_gfx950` 保持 256×128 |
