# Round 1: Triton Blockwise FP8 GEMM 持久化 Kernel + 离线配置选择

## 目标
将 Triton blockwise FP8 GEMM 从 autotune 数据并行模式改为持久化 kernel + 离线配置选择，消除 autotune 编译开销并优化 grid 调度。

## 分析

### 问题
原始 Triton blockwise kernel 有两个关键问题：

1. **autotune 编译开销**: 32 个 config × 6 个 key 组合，首次调用需要编译大量 kernel variant
2. **数据并行 grid**: `NUM_SMS = num_m × num_n`（总 tile 数），且 `NUM_SMS` 不是 `constexpr`

对比 tensorwise kernel：
- 使用 `_compute_sk_grid` 计算最优 grid 大小
- `NUM_SMS` 是 `constexpr`，Triton 编译器可以更好地优化持久化循环
- 使用 `_chiplet_transform_chunked` 进行 XCD-aware 调度

### 首次尝试（已放弃）
尝试为 CK ABQuantGrouped 添加 256x256 tile。结果：**全面退化**（80/84 配置变慢，最差 0.47x）。

根因：ABQuantGrouped pipeline 的 2x2x1 warp 布局在 256x256 tile 下需要 16 个 warp-tile 迭代（vs 128x128 的 4 个），导致严重的寄存器溢出。

$$\text{Warp tiles per warp} = \frac{M_{tile}/M_{warp}}{M_{warp\_tile}} \times \frac{N_{tile}/N_{warp}}{N_{warp\_tile}} = \frac{256/2}{32} \times \frac{256/2}{32} = 4 \times 4 = 16$$

而 128x128 tile 仅需 $2 \times 2 = 4$ 个 warp-tile。

## 方案

### 新增持久化 kernel: `_blockwise_fp8_persistent_kernel`

关键改进：
1. **`NUM_SMS` 作为 `tl.constexpr`** — 编译器可优化持久化循环调度
2. **`EVEN_K` 优化** — K 对齐 128 时跳过 mask 检查
3. **`_chiplet_transform_chunked`** — XCD-aware PID 变换（8-XCD MI300X）
4. **int64 指针算术** — 防止大矩阵 int32 溢出
5. **`tl.multiple_of` 提示** — 帮助编译器生成 vectorized load
6. **`cache_modifier`** — 控制 L2 cache 行为

### 离线配置选择: `_select_blockwise_config`

- `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128`（匹配量化 block size）
- `NUM_SMS` 由 `_compute_sk_grid` 计算
- `GROUP_M` 基于 tile 数量和布局选择
- `CHUNK` 基于持久化/数据并行模式选择

## 实现

### 修改文件
- `primus_turbo/triton/gemm/gemm_fp8_kernel.py`
  - 新增 `_blockwise_fp8_persistent_kernel` — 持久化 kernel
  - 新增 `_select_blockwise_config` — 离线配置选择
  - 修改 `_blockwise_nt` — NT forward 使用新 kernel
  - 修改 `_blockwise_nn` — NN grad_X 使用新 kernel
  - 修改 `_blockwise_tn` — TN grad_W 使用新 kernel

## 精度验证

```
pytest tests/pytorch/ops/test_gemm_fp8.py -k "blockwise and not mx_blockwise" -x
192 passed, 496 skipped (41.82s vs 原来 280.84s autotune)
```

测试时间从 **280s → 42s**（6.7x 加速），因为不再需要编译 32 个 autotune config。

## 性能对比

### Forward (NT layout)

| 指标 | CK Baseline | Triton Persistent | 变化 |
|------|-------------|-------------------|------|
| 平均 TFLOPS | 429.07 | 438.72 | **+2.25%** |
| Geomean speedup | — | **1.025x** | |
| 改善配置数 | — | 40/84 | |
| 退化配置数 | — | 30/84 | |

Forward 按 K 维度分析：
- K ≥ 8192: 一致改善（+6% ~ +46%）
- K = 3584-4096: 部分退化（-5% ~ -15%）

### Backward (NN + TN layouts)

| 指标 | CK Baseline | Triton Persistent | 变化 |
|------|-------------|-------------------|------|
| 平均 TFLOPS | 233.66 | 329.12 | **+40.8%** |
| Geomean speedup | — | **1.408x** | |
| 改善配置数 | — | **84/84** | |
| 退化配置数 | — | **0/84** | |

### Top 5 Forward 改善

| Config | CK (ms) | Triton (ms) | Speedup |
|--------|---------|-------------|---------|
| 405B M=32768 N=106496 K=16384 | 343.22 | 234.30 | **1.465x** |
| 405B M=16384 N=106496 K=16384 | 172.56 | 136.23 | **1.267x** |
| 405B M=8192 N=106496 K=16384 | 87.19 | 70.34 | **1.240x** |
| 405B M=16384 N=16384 K=53248 | 81.86 | 66.98 | **1.222x** |
| 70B M=16384 N=57344 K=8192 | 40.50 | 33.71 | **1.201x** |

### Top 5 Backward 改善

| Config | CK (ms) | Triton (ms) | Speedup |
|--------|---------|-------------|---------|
| 405B M=8192 N=16384 K=53248 | 132.41 | 83.80 | **1.580x** |
| 72B M=16384 N=8192 K=29568 | 68.19 | 44.59 | **1.529x** |
| 405B M=8192 N=16384 K=16384 | 36.94 | 24.23 | **1.525x** |
| 72B M=8192 N=8192 K=29568 | 32.45 | 21.37 | **1.518x** |
| 7B M=32768 N=3584 K=18944 | 39.21 | 26.08 | **1.503x** |

## 结论

**合入**。理由：
1. Backward 全面改善 40.8%，无任何退化
2. Forward 大矩阵场景（405B 级别）改善 20-46%
3. 测试编译时间加速 6.7x（消除 autotune 开销）
4. 精度验证全部通过（192/192 PASS）

Forward 小 K 退化问题将在后续 Round（后端选择优化）中解决。
