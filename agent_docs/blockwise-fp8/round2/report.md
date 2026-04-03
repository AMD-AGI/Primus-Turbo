# Round 2: Blockwise FP8 GEMM Shape-Based Backend Dispatch

## 目标
基于 Round 1 的 profiling 数据，实现 blockwise FP8 GEMM 的 shape-based 后端选择：
- Forward (NT): K >= 8192 用 Triton（持久化 kernel），K < 8192 保持 CK
- Backward (NN/TN): 全部用 Triton（Round 1 验证 Triton bwd 全面优于 CK）

## 分析

Round 1 数据按 K 维度分析：

| K 范围 | 最优后端 (Forward) | Triton/CK Speedup |
|--------|:-----------------:|:-----------------:|
| K = 3584 | CK | 0.957x |
| K = 4096 | CK | 0.934x |
| **K = 8192** | **Triton** | **1.104x** |
| K = 11008 | Triton | 1.009x |
| K = 14336 | borderline | 0.998x |
| **K = 16384** | **Triton** | **1.191x** |
| K = 28672 | Triton | 1.090x |
| K = 53248 | Triton | 1.209x |

$$\text{分界点: } K = 8192 \quad (\text{NT forward 仅此布局, bwd 全部用 Triton})$$

## 方案

在 `gemm_fp8_impl.py` 中添加 `_blockwise_preferred_backend()`:

```python
def _blockwise_preferred_backend(a, b, trans_a, trans_b) -> BackendType:
    _, _, K = get_gemm_logical_shape(a, b, trans_a, trans_b)
    is_nt = not trans_a and trans_b
    if is_nt and K < 8192:
        return BackendType.CK
    return BackendType.TRITON
```

仅在用户未指定后端且 auto-tune 未启用时生效。

## 实现

### 修改文件
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`
  - 新增 `_blockwise_preferred_backend()` — shape-based 后端选择
  - 修改 `gemm_fp8_impl()` — blockwise 时使用动态后端选择替代硬编码 CK

## 精度验证

```
192 passed, 496 skipped (12.25s)
```

## 性能对比（vs CK-only baseline）

### Forward

| 指标 | CK Baseline | Round 2 | 变化 |
|------|-------------|---------|------|
| 平均 TFLOPS | 429.07 | **450.16** | **+4.92%** |
| Geomean speedup | — | **1.0516x** | |
| 改善 | — | 39/84 | |
| 退化 | — | **6/84** (max -6%) | |
| 持平 | — | 39/84 | |

### Backward

| 指标 | CK Baseline | Round 2 | 变化 |
|------|-------------|---------|------|
| 平均 TFLOPS | 233.66 | **328.81** | **+40.7%** |
| Geomean speedup | — | **1.4065x** | |
| 改善 | — | **84/84** | |
| 退化 | — | **0/84** | |

### vs Round 1 (Triton-only)

| 指标 | Round 1 Triton | Round 2 Dispatch | 改进 |
|------|---------------|-----------------|------|
| Forward geomean | 1.0249x | **1.0516x** | +2.6% |
| Fwd 退化数 | 30 | **6** | -24 configs |
| Backward geomean | 1.4078x | 1.4065x | ≈持平 |

## 结论

**合入**。理由：
1. Forward 从 +2.5% 提升到 **+5.2%**，退化配置从 30 降到 **6**
2. Backward 保持 **+40.7%** 全面改善
3. 不影响用户手动设置后端或 auto-tune 流程
4. 实现简洁，仅添加 1 个函数 + 4 行 dispatch 逻辑
