# MI355X Blockwise FP8 GEMM 优化总结报告

## 概述

对 Primus-Turbo 代码库中 blockwise FP8 GEMM Triton kernel 在 AMD MI355X (gfx950/CDNA4) 上进行了 10 轮系统化优化。

## 硬件平台

| 参数 | 值 |
|------|---|
| GPU | AMD Instinct MI355X |
| 架构 | CDNA4 (gfx950) |
| CU 数量 | 256 |
| XCD 数量 | 8 |
| LDS/CU | 160 KB |
| HBM 带宽 | 8 TB/s |
| FP8 峰值 | 5000 TFLOPS |
| BF16 峰值 | 2500 TFLOPS |

## 性能演进

```
阶段                  Fwd TFLOPS    Bwd TFLOPS    vs Baseline (Fwd)
────────────────────────────────────────────────────────────────────
Round 0: main 基线       488.1         229.0        baseline
Round 1: MI300X 适配     866.4         813.7        +77.5%
Round 2: BM=256 tile    1297.4         976.8        +165.8%  ★
Round 10: 最终确认      1293.1         974.1        +164.9%
```

### 核心成果

| 指标 | 基线 (main) | 最终 | 提升 |
|------|-------------|------|------|
| Forward TFLOPS | 488.1 | 1293.1 | **+165%** (2.65×) |
| Backward TFLOPS | 229.0 | 974.1 | **+325%** (4.25×) |
| FP8 利用率 (Fwd) | 9.8% | 25.9% | +16.1pp |
| FP8 利用率 (Bwd) | 4.6% | 19.5% | +14.9pp |

## 各轮详情

### Round 1: MI300X 优化移植 ✓

MI300X agent-optimize 分支的优化直接适用于 MI355X：
- persistent kernel + split-K
- gfx950 knobs (`async_copy`, `block_pingpong`)
- `waves_per_eu=2` for backward

**结果**: Fwd +77.5%, Bwd +255%

### Round 2: BLOCK_M=256 Tile ✓ (唯一有效的 MI355X 特定优化)

利用 MI355X 160 KB LDS (vs MI300X 64 KB) 启用 256×128 tile：

$$\text{LDS} = 2 \times (256 \times 128 + 128 \times 128) \times 1\text{B} = 96\text{KB} < 160\text{KB} \checkmark$$

关键约束：`BLOCK_N` 必须 ≤ 128，因为 `SCALE_2D_B=True` 路径中 B-scale 按 tile 索引加载，
假设 `BLOCK_N == scale_block_size (128)`。

**结果**: Fwd +49.7%, Bwd +20.0%

### Round 3: BLOCK_K 调优 ✗

- `BLOCK_K=64`: Triton gfx950 后端编译失败 (`PassManager::run failed`)
- `BLOCK_K=256`: 超出 LDS 预算 (192 KB > 160 KB)

**结论**: `BLOCK_K=128` 是唯一可行值

### Round 4: num_warps / waves_per_eu ✗

系统 sweep 8 种 `(num_warps, waves_per_eu)` 组合：
- Forward (NT): `num_warps=8, waves_per_eu=0` 已是最优
- `waves_per_eu=2` 在 M≥8192 时有 +3~10% 提升，但平均仅 +0.15% → 低于 2% 阈值

**结论**: 回滚

### Round 5: Pipeline + Cache ✗

- Triple buffer (`num_stages=3`): LDS 不足 (256×128×3 = 144KB for A alone)
- `.cs` cache modifier: gfx950 编译错误

**结论**: 无改进

### Round 6-8: 已有优化确认 ✗

- `async_copy` / `block_pingpong`: 已通过 `_set_knobs_gfx950()` 全局启用
- Scale 融合: scale 加载开销相对于 compute 可忽略

**结论**: 无额外优化空间

### Round 9: CK 后端对比 ✗

CK (Composable Kernel) 后端在 blockwise FP8 GEMM 上比 Triton 慢 1.5-3×。
不做 shape-based backend dispatch。

**关键发现**: `quant_fp8_blockwise_for_weight_impl` 产生真 2D block scale
`[N//128, K//128]`，区别于 activation 的 per-row scale `[M, K//128]`。

### Round 10: 256×256 Tile ✗

**Kernel 修改 (已保留)**:
添加 `N_SCALE_BLOCKS = BLOCK_N // 128` 支持多 scale block 加载：

```
BLOCK_N=128 → N_SCALE_BLOCKS=1 → scalar scale (原始路径，零开销)
BLOCK_N=256 → N_SCALE_BLOCKS=2 → 加载 2 个 scale + tl.where 构建向量
```

**精度**: 18/18 shapes PASS (SNR ≥ 28.7 dB)

**性能**: 256×256 比 256×128 慢 **4-5×**

| Tile | 累加器 VGPRs/thread | TFLOPS (4096³) |
|------|-------------------|---------------|
| 256×128 | 64 | 1377 |
| 256×256 | 128 | 284 |
| 128×128 | 32 | 968 |

根因：128 VGPRs 超出最优占用窗口，导致 register spilling。

**决策**: 保留 kernel 代码 (零开销)，tile 选择回退 256×128

## 最终代码变更

### 生效的优化

| 文件 | 变更 | 来源 |
|------|------|------|
| `gemm_fp8_kernel.py` | `_select_blockwise_tile_gfx950()`: M≥2048 时使用 256×128 tile | Round 2 |
| `gemm_fp8_kernel.py` | `_select_blockwise_config()`: gfx950 分支 | Round 2 |
| `gemm_fp8_kernel.py` | `N_SCALE_BLOCKS` multi-block scale (dead code for BN=128) | Round 10 |

### 探索但未合入的优化

| 优化 | 原因 |
|------|------|
| BLOCK_K=64/256 | 编译失败 / LDS 超限 |
| waves_per_eu=2 (forward) | 平均增益 < 2% |
| num_stages=3 | LDS 超限 |
| .cs cache modifier | gfx950 不支持 |
| 256×256 tile | 4-5× 性能回退 (register pressure) |
| CK 后端 dispatch | CK 1.5-3× 慢于 Triton |

## 进一步优化方向

1. **OCP FP8 native MFMA**: 当 Triton 完全支持 gfx950 的 `V_MFMA_F32_32x32x64_F8F6F4` 指令时，可能带来显著提升
2. **Triton 编译器改进**: 更好的 gfx950 代码生成 (BLOCK_K=64 支持, .cs 支持)
3. **MXFP8 支持**: gfx950 原生 MXFP8 可能比 blockwise FP8 更高效
4. **Per-shape offline tuning**: 用 autotune 为每个 (M, N, K) shape 选择最优 (tile, warps, stages) 组合
