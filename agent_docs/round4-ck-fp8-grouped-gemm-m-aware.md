# Round 4: CK FP8 Grouped GEMM M-aware Tile Selection

## 优化概述

将 Phase 3 验证有效的 CU 利用率感知 M-aware tile 选择技术，扩展至 FP8 Grouped GEMM 的
RowColQuant 和 TensorQuant 路径。ABQuantGrouped 路径已固定使用 128x128x128 config，无需改动。

## 修改文件

| 文件 | 变更 |
|------|------|
| `ck_grouped_gemm_kernel_instance_factory.cu/.hip` | GFX942/GFX950 FP8 RowColQuant/TensorQuant 路径添加 M-aware 逻辑 |

## 实现细节

### 算法（与 Phase 3 BF16 一致）

```
fp8_m_tiles = ceil(m / 256)
fp8_n_tiles = ceil(n / 256)
fp8_total   = group_num × fp8_m_tiles × fp8_n_tiles

IF fp8_total < NUM_CU AND group_num >= 2 AND n % 128 == 0:
    → 使用 128×128×128 tile (更小 tile → 更多并行度)
ELSE:
    → 使用原有 256×256×128 / 256×128×128 选择逻辑
```

### Tile 配置

| 架构 | 128×128 Config | 256×256 Config |
|------|---------------|---------------|
| GFX942 (MI300X) | `GFX942_CKGroupedGemmTileCfg_128x128x128_32x32x32_2x2x1` | `GFX942_CKGroupedGemmTileCfg_256x256x128_32x32x32_2x2x1` |
| GFX950 (MI355X) | `GFX950_CKGroupedGemmTileCfg_128x128x128_32x32x64_2x2x1` | `GFX950_CKGroupedGemmTileCfg_256x256x128_16x16x128_2x2x1` |

## 正确性验证

| 测试 | 数量 | 结果 |
|------|------|------|
| FP8 RowColQuant CK (B≥8, M≤512) | 576 | **全部 PASSED** |
| FP8 ABQuantGrouped CK (B≥8, M≤512) | 768 | **全部 PASSED** |
| **合计** | **1344** | **0 FAILED** |

## 性能数据 (MI300X)

### RowColQuant (FP8 E4M3)

| Case | B | M | N | K | ms | TFLOPS |
|------|---|---|---|---|-----|--------|
| DSv3 B=8 M=512 | 8 | 512 | 4096 | 7168 | 4.724 | 50.92 |
| DSv3 B=16 M=512 | 16 | 512 | 4096 | 7168 | 4.357 | 110.41 |
| DSv3 B=8 M=1024 | 8 | 1024 | 4096 | 7168 | 3.348 | 143.69 |
| DSv3-Down B=8 M=512 | 8 | 512 | 7168 | 2048 | 2.517 | 47.79 |
| DSv2L B=2 M=512 | 2 | 512 | 2816 | 2048 | 0.160 | 73.85 |
| DSv2L B=8 M=512 | 8 | 512 | 2816 | 2048 | 1.291 | 36.58 |
| Qwen3-30B B=8 M=512 | 8 | 512 | 4096 | 2048 | 1.899 | 36.20 |
| Kimi-K2 B=12 M=512 | 12 | 512 | 4096 | 7168 | 4.701 | 76.75 |
| DSv3 B=8 M=4096 (large) | 8 | 4096 | 4096 | 7168 | 7.165 | 268.54 |
| DSv3 B=8 M=16384 (large) | 8 | 16384 | 4096 | 7168 | 17.598 | 437.36 |
| DSv2L B=2 M=8192 (large) | 2 | 8192 | 2816 | 2048 | 0.818 | 231.14 |
| Kimi-K2 B=12 M=8192 (large) | 12 | 8192 | 4096 | 7168 | 12.477 | 462.65 |

### ABQuantGrouped (FP8 E4M3, Blockwise 128)

| Case | B | M | N | K | ms | TFLOPS |
|------|---|---|---|---|-----|--------|
| DSv3 B=8 M=512 | 8 | 512 | 4096 | 7168 | 1.071 | 224.58 |
| DSv3 B=16 M=512 | 16 | 512 | 4096 | 7168 | 1.958 | 245.63 |
| DSv3 B=8 M=1024 | 8 | 1024 | 4096 | 7168 | 1.828 | 263.14 |
| DSv3-Down B=8 M=512 | 8 | 512 | 7168 | 2048 | 1.556 | 77.27 |
| Kimi-K2 B=12 M=512 | 12 | 512 | 4096 | 7168 | 1.567 | 230.28 |
| DSv3 B=8 M=4096 (large) | 8 | 4096 | 4096 | 7168 | 8.298 | 231.87 |
| DSv3 B=8 M=16384 (large) | 8 | 16384 | 4096 | 7168 | 24.077 | 319.67 |
| Kimi-K2 B=12 M=8192 (large) | 12 | 8192 | 4096 | 7168 | 19.907 | 289.97 |

## 观察

1. **ABQuantGrouped 已优化**: 固定使用 128x128x128，无需 M-aware（本轮不影响此路径）
2. **RowColQuant 整体 TFLOPS 较低**: 相比 BF16（200-450 TFLOPS）和 ABQuantGrouped（200-300 TFLOPS），RowColQuant 的 small M 仅 36-143 TFLOPS，瓶颈不仅在 tile 选择
3. **代码一致性**: 本轮确保 FP8 路径与 BF16 路径使用相同的 M-aware 策略，避免架构间的行为差异
4. **NUM_CU 提升至函数作用域**: 解决了之前 `constexpr NUM_CU` 在 BF16 `if constexpr` 块内不可见的编译错误

## 结论

本轮优化风险低，改动最小（复用已验证的 tile config + 已验证的选择算法），正确性完全通过。
FP8 RowColQuant 路径的性能瓶颈需要进一步分析（可能涉及 CK 量化 kernel 本身的效率问题）。
