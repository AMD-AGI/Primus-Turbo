# MI355X Blockwise FP8 GEMM — Round 0 Baseline

## 硬件环境

| 参数 | 值 |
|------|-----|
| GPU | AMD Instinct MI355X |
| 架构 | gfx950 (CDNA4) |
| CU 数量 | 256 |
| XCD 数量 | 8 |
| HBM3E | 288 GB @ 8.0 TB/s |
| LDS / CU | 160 KB |
| FP8 峰值 | 5000 TFLOPS |
| BF16 峰值 | 2500 TFLOPS |

## Baseline 性能 (main 分支, commit 333b68d)

| 指标 | 值 |
|------|-----|
| **Forward Avg TFLOPS** | 488.08 |
| **Backward Avg TFLOPS** | 228.99 |
| **Forward FP8 利用率** | 9.76% |
| **Backward FP8 利用率** | 4.58% |
| **精度测试** | 84/84 PASS |

## 瓶颈分析

### Roofline 分析

$$\text{AI}_{blockwise} = \frac{2MNK}{MK + KN + MN + \lceil M/128 \rceil \cdot \lceil K/128 \rceil \cdot 4 \cdot 2}$$

对于典型 shape M=4096, N=4096, K=4096:
- 数据量: $4096^2 \times 3 = 50.3 \text{ MB}$ (FP8) + scale ~0.5 MB
- FLOP: $2 \times 4096^3 = 137.4 \text{ GFLOP}$
- AI ≈ 2700 FLOP/Byte

MI355X FP8 平衡点: $5000 / 8.0 = 625 \text{ FLOP/Byte}$

大多数 shape 的 AI >> 625，应为**计算受限**，但 FP8 利用率仅 ~10%，说明：
1. Kernel 内部效率极低（autotune kernel + 非 persistent grid）
2. Scale tensor 额外内存访问未优化
3. MFMA 调度不充分

### 与 MI300X Baseline 对比

| 指标 | MI300X Baseline | MI355X Baseline | 实际倍率 | 理论倍率 |
|------|:-:|:-:|:-:|:-:|
| Fwd TFLOPS | 429 | 488 | **1.14x** | 1.91x |
| Bwd TFLOPS | 234 | 229 | **0.98x** | 1.91x |

> MI355X 在 main 分支上仅达到 MI300X 的 **1.14x (Fwd)** 和 **0.98x (Bwd)**，远低于 1.91x 理论倍率。
> 这说明 main 分支的 blockwise 代码未针对 CDNA4 优化，可能使用了非最优 kernel 路径。
