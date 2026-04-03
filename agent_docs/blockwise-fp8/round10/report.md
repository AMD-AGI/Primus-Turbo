# Round 10: NN backward 消除 B scale 转置 + 最终总结

## 概要

| 指标 | Round 9 基线 | Round 10 结果 | 变化 (vs R9) | **累计 (vs 原始基线)** |
|------|-------------|-------------|-------------|----------------------|
| Forward TFLOPS | 514.24 | 514.50 | +0.08% | **+21.15%** |
| Backward TFLOPS | 466.04 | 467.27 | **+0.32%** | **+99.68%** |

## 优化内容

### NN backward 消除 B scale 转置

**之前:**
```python
b_scale_inv_t = b_scale_inv.T.contiguous()  # GPU memcpy
_blockwise_fp8_persistent_kernel[(num_sms,)](
    ...
    b_scale_inv_t,
    ...
    b_scale_inv_t.stride(0),  # stride_bs_0
    b_scale_inv_t.stride(1),  # stride_bs_1
)
```

**之后:**
```python
# 无需转置，直接使用原始 tensor，交换步幅顺序
_blockwise_fp8_persistent_kernel[(num_sms,)](
    ...
    b_scale_inv,  # 原始 tensor，无 GPU 拷贝
    ...
    b_scale_inv.stride(1),  # stride_bs_0 (K_orig block stride for pn)
    b_scale_inv.stride(0),  # stride_bs_1 (N_orig block stride for ki)
)
```

**分析:**
- NN backward 中 B 的维度映射：`pn → K_orig blocks, ki → N_orig blocks`
- 原始 b_scale_inv `[N_orig//128, K_orig//128]` 的步幅 (K_orig//128, 1)
- 交换步幅顺序使 `pn * 1 + ki * K_orig//128` 正确访问 `b_scale[ki, pn]`
- 数学上与转置方案等价，但省去了 `.T.contiguous()` 的 GPU memcpy

### 其他尝试（Round 10 中未采用）

1. **TN A 预转置**: 对小 shape 有害（-19%），大 shape 微弱改善（+6%），不采用
2. **NT 禁用 async_copy**: 效果混合（小 shape +2.5%，大 shape -0.7%），保持不变
3. **NN waves_per_eu=1**: 仅 +0.21%（噪声级别），保持 wpe=2

## 10 轮优化完整总结

| Round | 优化内容 | Forward 变化 | Backward 变化 | 累计 Fwd | 累计 Bwd |
|-------|---------|-------------|--------------|---------|---------|
| **1** | Triton persistent kernel | +2.5% | **+40.8%** | +2.5% | +40.8% |
| **2** | Shape-based CK/Triton dispatch | +5.2% | — | +7.8% | +40.7% |
| **3** | Forward NT num_warps=8 | **+7.0%** | — | +15.3% | +40.7% |
| **4** | Backward NN/TN num_warps=8 | — | **+9.1%** | +15.3% | +53.0% |
| **5** | 移除 CK dispatch + GROUP_M | +2.5% | — | +16.1% | +53.0% |
| **6** | Backward waves_per_eu=2 | — | **+9.8%** | +16.1% | +68.0% |
| **7** | Scale 加载重排序 | **+4.3%** | +4.2% | +21.3% | +75.1% |
| **8** | tl.assume hints | +0.09% | +0.10% | +21.4% | +75.4% |
| **9** | NN 禁用 async_copy | — | **+13.7%** | +21.1% | **+99.0%** |
| **10** | NN 消除 B scale 转置 | — | +0.3% | **+21.2%** | **+99.7%** |

## 最终性能数据

| 指标 | 原始基线 | 最终结果 | 提升 |
|------|---------|---------|------|
| Forward TFLOPS | 424.92 | **514.50** | **+21.15%** |
| Backward TFLOPS | 234.16 | **467.27** | **+99.68%** |
| 84/84 shapes | — | 全部 PASS | 0 FAIL, 0 ERROR |

## 关键优化原则总结

1. **Persistent kernel + NUM_SMS constexpr** (R1): 将 NUM_SMS 作为编译期常量，使编译器生成更优代码
2. **num_warps=8** (R3/R4): 比默认的 4 warps 提供更好的内存延迟隐藏
3. **waves_per_eu=2** (R6): 允许更多 VGPR 用于寄存器分配，同时保持占用率
4. **Scale 加载先于数据加载** (R7): 利用 MFMA pipeline 延迟窗口重叠 scale 内存访问
5. **async_copy 按路径启用** (R9): 仅当所有输入的 K 维都连续时启用 DMA，否则禁用

$$\text{总体提升} = \begin{cases} \text{Forward}: +21.15\% \\ \text{Backward}: +99.68\% \approx 2\times \end{cases}$$
