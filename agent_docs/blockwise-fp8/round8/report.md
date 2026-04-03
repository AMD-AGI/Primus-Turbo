# Round 8: 微优化探索（tl.assume + 多种尝试）

## 概要

| 指标 | Round 7 基线 | Round 8 结果 | 变化 |
|------|-------------|-------------|------|
| Forward TFLOPS (geomean) | 515.01 | 515.51 | **+0.09%** |
| Backward TFLOPS (geomean) | 409.10 | 409.66 | **+0.10%** |
| 累计 vs 原始基线 Forward | +21.28% | +21.39% | |
| 累计 vs 原始基线 Backward | +75.13% | +75.37% | |

## 尝试的优化方向

### 1. Forward `waves_per_eu=2` ❌
- **假设**: Scale 重排序（R7）消除了 forward 的 scale 延迟，forward 可能从更好的寄存器分配中受益
- **结果**: Forward -0.65%，**回退**
- **原因**: Forward NT 路径使用 async_copy，占用率 1 (waves_per_eu=0) 允许更多 VGPR 用于 DMA pipeline

### 2. 指针 advance 重排序到 data load 之后 ❌
- **假设**: 将 `a_ptrs += BLOCK_K` 移到 `tl.load(a_ptrs)` 之后，让指针算术在 MFMA 延迟窗口中完成
- **结果**: Forward +0.01%, Backward +0.17% — **可忽略**
- **原因**: Triton 编译器的指令调度器已经很好地处理了指针算术的位置

### 3. Scale 增量偏移代替乘法 ❌
- **假设**: 用 `as_k_off += stride_as_k` 代替 `ki * stride_as_k`，消除乘法
- **结果**: Forward **-7.0%**, Backward **-1.6%** — **严重回退**
- **原因**: 引入了跨迭代的串行数据依赖链。原始 `ki * stride_as_k` 中 `ki` 是循环计数器，编译器可以独立计算每次迭代的偏移量；而增量方式要求每次迭代等待前一次完成

### 4. `tl.assume` Scale 步幅提示 ✅（微弱）
- **假设**: 为 scale 步幅添加 `tl.assume(stride > 0)` 提示，帮助编译器优化索引计算
- **结果**: Forward +0.09%, Backward +0.10%
- **决定**: 保留——无性能损失，符合最佳实践

## 保留的代码变更

```python
# _blockwise_fp8_persistent_kernel 中添加:
tl.assume(stride_as_k > 0)
tl.assume(stride_as_m > 0)
tl.assume(stride_bs_0 > 0)
tl.assume(stride_bs_1 > 0)
```

## 分析: 为什么微优化已到极限

### 内核已充分优化的证据
1. **Scale 加载开销**: 仅占数据加载的 1.6-3.1%，已通过 R7 重排序最大化重叠
2. **MFMA 利用率**: `num_warps=8` (R3/R4) 和 `waves_per_eu=2` (R6) 已优化延迟隐藏
3. **内存访问模式**: `async_copy` (NT/NN), `tl.multiple_of` 对齐提示, `.ca` cache modifier 已优化
4. **编译器指令调度**: Triton + LLVM 后端已充分重排内循环指令

### 当前性能 vs 理论峰值
- MI300X FP8 MFMA 峰值: ~1.3 PFLOPS (理论)
- Forward 515 TFLOPS ≈ 39.6% 峰值效率
- Backward 410 TFLOPS ≈ 31.5% 峰值效率
- Blockwise 的额外 scale 开销（加载+乘法）限制了达到 tensorwise 水平

## 结论

Round 8 探索了 4 种微优化方向，确认 kernel 内循环已充分优化。后续优化应转向更高层面。
