# Round 9: NN Backward 禁用 async_copy — 巨大突破

## 概要

| 指标 | Round 7 基线 | Round 9 结果 | 变化 (vs R7) | 累计 (vs 原始基线) |
|------|-------------|-------------|-------------|-------------------|
| Forward TFLOPS | 515.01 | 514.24 | **-0.18%** (噪声) | **+21.06%** |
| Backward TFLOPS | 409.10 | 466.04 | **+13.65%** | **+99.04%** |

## 发现过程

### 1. NN vs TN 独立 Profiling

首先分别测量了 NN (grad_X) 和 TN (grad_W) 的独立性能：

| Shape | NN (async=True) | TN (async=False) | NN 落后幅度 |
|-------|-----------------|-------------------|------------|
| 4096×4096×4096 | 405 TFLOPS | 473 TFLOPS | **-14.4%** |
| 4096×4096×8192 | 421 TFLOPS | 503 TFLOPS | **-16.3%** |
| 8192×28672×4096 | 430 TFLOPS | 490 TFLOPS | **-12.2%** |
| 4096×14336×4096 | 453 TFLOPS | 519 TFLOPS | **-12.7%** |

**NN 是 backward 的瓶颈，比 TN 慢 12-16%！**

### 2. 根因分析

NN 路径的配置：
- `_set_amd_knobs(enable=True)` → `use_async_copy=True, scalarize_packed_fops=True`
- A: [M, K], **A_K_CONTIGUOUS=True** → K 维连续，可用 DMA
- B: [K, N], **B_K_CONTIGUOUS=False** → K 维非连续（stride=N）

问题：`async_copy` 和 `scalarize_packed_fops` 是**全局 knob**，同时影响 A 和 B 的加载：

```
async_copy 对 A 加载: ✅ 有益 — K 维连续，DMA 高效
async_copy 对 B 加载: ❌ 有害 — K 维非连续，DMA 无法高效处理 strided 访问
scalarize_packed_fops: ❌ 有害 — 阻止了 FP8 packed 操作优化
```

**净效果：B 端损失 > A 端收益。** 且 `waves_per_eu=2`（Round 6）已提供充分的延迟隐藏。

### 3. 修复

```python
# _blockwise_nn() 中:
# 之前:
_set_amd_knobs(enable=True)   # async_copy + scalarize for A AND B

# 之后:
_set_amd_knobs(enable=False)  # 禁用，让 wave switching 隐藏延迟
```

### 4. NN 性能对比（禁用 async_copy 后）

| Shape | NN (async=True) | NN (async=False) | 提升 |
|-------|-----------------|-------------------|------|
| 4096×4096×4096 | 405 TFLOPS | **549 TFLOPS** | **+35.5%** |
| 4096×4096×8192 | 421 TFLOPS | **601 TFLOPS** | **+42.9%** |
| 8192×28672×4096 | 430 TFLOPS | **601 TFLOPS** | **+39.9%** |
| 4096×14336×4096 | 453 TFLOPS | **625 TFLOPS** | **+38.0%** |
| 1024×4096×8192 | 380 TFLOPS | **515 TFLOPS** | **+35.5%** |

**NN 单路径提升 35-43%！**

## 完整基准结果

- **69/69 backward shapes 全部提升，0 回退**
- **Backward geomean: +13.65%**（受 TN 不变拖累，NN 实际提升更大）
- 最大提升: 32768×106496×16384 → **+20.9%**
- 最小提升: 32768×8192×29568 → **+8.1%**
- 84/84 全部 PASS，0 FAIL，0 ERROR

## 深层原因分析

```
NN Backward 数据流:

A[M,K] ──DMA可用──→ [LDS] ─→ MFMA
                              ↑
B[K,N] ──DMA失败──→ [注册] ─→ MFMA   ← async_copy 强制 DMA
                                        但 K-stride=N 无法高效 DMA
                                        
禁用后:

A[M,K] ──寄存器加载──→ MFMA    ← waves_per_eu=2 提供延迟隐藏
                       ↑
B[K,N] ──寄存器加载──→ MFMA    ← 无 DMA 开销，直接通过 L1/L2 缓存
```

## 架构启示

| 路径 | async_copy | 原因 |
|------|-----------|------|
| NT (forward) | ✅ True | A, B 都 K-连续 → DMA 两端均受益 |
| NN (backward dX) | ❌ **False** | B 是 K-strided → DMA 有害 |
| TN (backward dW) | ❌ False | A, B 都 K-strided → DMA 有害 |

**规则: 仅当所有输入矩阵的 K 维都连续时才启用 async_copy。**

## 累计优化成果 (Round 0 → Round 9)

| 指标 | 原始基线 | Round 9 | 累计提升 |
|------|---------|---------|---------|
| Forward TFLOPS | 424.92 | 514.24 | **+21.06%** |
| Backward TFLOPS | 234.16 | 466.04 | **+99.04%** |
