# Round 6: GEMM Shape-based Backend Dispatch

## 优化概述

为 BF16 GEMM 添加 shape-based 静态后端选择。当 user 未指定后端且 auto-tune 未启用时，
根据 (M, N, K) 维度自动选择 hipBLASLt 或 Triton，避免 runtime auto-tune 的 profiling 开销。

## 修改文件

| 文件 | 变更 |
|------|------|
| `primus_turbo/pytorch/kernels/gemm/gemm_impl.py` | 添加 `_shape_preferred_backend()` 函数，`gemm_impl` 中调用 |

## 实现规则

```
IF a.dtype ∈ {BF16, FP16}:
    IF K >= 40000:        → Triton (persistent kernel 更好地隐藏延迟)
    IF N >= 65536 AND M >= 8192: → Triton (tile scheduling 优势)
    ELSE:                 → 保持默认 hipBLASLt
```

### 规则推导依据（MI300X BF16 benchmark）

| Case | hipBLASLt | Triton | Winner | 规则 |
|------|----------|--------|--------|------|
| M=8192 N=106496 K=16384 | 410-476 TFLOPS | 445-567 TFLOPS | **Triton +8-19%** | N≥65536 |
| M=8192 N=16384 K=53248 | 456-515 TFLOPS | 458-577 TFLOPS | **Triton +5-12%** | K≥40000 |
| M=32768 N=10240 K=8192 | 395-454 TFLOPS | 236-537 TFLOPS | **Mixed** | 不触发 |
| M=4096 N=12288 K=4096 | 419-598 TFLOPS | 204-426 TFLOPS | **hipBLASLt** | 不触发 |
| M=8192 N=28672 K=4096 | 301-626 TFLOPS | 238-244 TFLOPS | **hipBLASLt** | 不触发 |

## 设计原则

1. **保守策略**: 仅在 Triton 胜出 >5% 且跨多次测量一致的 shape 空间触发
2. **零侵入**: 用户显式设置 `set_gemm_backend()` 或启用 auto-tune 时不干预
3. **无运行时开销**: 纯 shape 判断，O(1) 决策
4. **不影响 FP32/FP8**: 仅对 BF16/FP16 触发

## 正确性验证

| 验证项 | 结果 |
|--------|------|
| Shape dispatch 逻辑测试 | **5/5 PASS** |
| GEMM pytest (38 cases) | **38 passed** |
| 数值一致性（BF16 K=53248, N=106496）| HBLt/Triton 误差一致（~4.0 vs FP32，BF16 累积舍入） |

## 优先级层次

```
用户指定后端 > auto-tune > shape-based dispatch > 默认 hipBLASLt
```
