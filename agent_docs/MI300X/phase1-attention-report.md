# Phase 1: Triton Attention 优化总结

> **日期**: 2026-04-02 | **硬件**: AMD Instinct MI300X (gfx942) | **软件**: Triton 3.6.0, PyTorch 2.10+rocm7.1

---

## 1. 关键发现

### 后端分层

| 路径 | 后端 | 说明 |
|------|------|------|
| `flash_attn_func()` (BF16) | **AITER** (ASM kernel) | 已高度优化，~450 TFLOPS Fwd |
| `flash_attn_fp8_func()` (FP8) | **Triton** | 本次优化目标 |
| `TritonFlashAttnFunc` (BF16) | **Triton** | 作为备选/可扩展路径 |

**结论**: BF16 生产路径使用 AITER，Triton kernel 优化影响 FP8 路径和 Triton BF16 备选路径。

---

## 2. 五轮优化详情

### Round 1: 扩展 Autotune 配置空间
- **改动**: Forward autotune 从 1 配置扩展到 4 配置
  - `num_warps`: {4} → {4, 8}
  - `PRE_LOAD_V`: {False} → {False, True}
- **原理**: MI300X wavefront=64, num_warps=8 提供 512 线程/WG，改善内存延迟隐藏

### Round 2: BLOCK_M 动态选择 64→128
- **改动**: 新增 `select_block_sizes()` 函数，当 padded head_dim ≤ 128 且 seqlen ≥ 256 时选择 BLOCK_M=128
- **原理**: BLOCK_M=128 使 Q tile 翻倍，提高对 K/V 数据的复用率，减少总迭代次数
- **影响**: 需同步更新 Forward + Backward wrapper 中所有 `FIXED_BLOCK_M/N` 引用

### Round 3: 极端 num_warps 探索
- **改动**: 添加 `num_warps=16` (1024 线程/WG) 到 Forward 和 Backward autotune
- **结果**: Autotune 未选择 num_warps=16，说明寄存器压力已是瓶颈

### Round 4: Software Pipelining (num_stages=2)
- **改动**: 添加 `num_stages=2` 到 Forward autotune
- **原理**: 2-stage pipeline 允许下一迭代的内存加载与当前迭代的计算重叠
- **效果**: 额外 +9-11% 提升

### Round 5: Backward 优化 + 最终验证
- **改动**: Backward autotune 添加 `num_stages=2` + `num_warps=8` 组合
- **验证**: 所有配置正确性通过 (SNR > 52dB)

---

## 3. 性能对比 (Triton BF16 Forward)

| 配置 | Baseline TFLOPS | Optimized TFLOPS | 提升 | vs AITER |
|------|:-:|:-:|:-:|:-:|
| MHA B=2 S=2048 D=128 causal | ~65 (est.) | 67.9 | +~5% | 18.3% |
| MHA B=2 S=4096 D=128 causal | 99.7 | 143.9 | **+44%** | 32.6% |
| MHA B=2 S=4096 D=128 full | ~120 (est.) | 229.0 | **+91%** | 47.1% |
| MHA B=2 S=8192 D=128 causal | 119.4 | 208.5 | **+75%** | 44.2% |
| GQA B=2 S=4096 H=64/8 D=128 | 106.6 | 164.6 | **+54%** | 36.9% |
| GQA B=4 S=4096 H=48/8 D=128 | ~110 (est.) | 171.8 | **+56%** | 38.6% |
| MLA B=1 S=4096 D=192/128 | 92.9 | 100.5 | **+8%** | 22.4% |

**Geometric Mean 提升 (head_dim=128 configs): ~+50%**

### FP8 Forward (完整 Benchmark, 78 configs)
- **Avg Forward TFLOPS**: 182.83
- **Avg Backward TFLOPS**: 140.74

---

## 4. 修改的文件

| 文件 | 改动类型 |
|------|----------|
| `primus_turbo/triton/attention/attention_kernel.py` | Forward/Backward autotune 扩展, `select_block_sizes()` 新增 |
| `primus_turbo/pytorch/kernels/attention/attention_triton_impl.py` | Forward/Backward 使用动态 block_m/block_n |
| `benchmark/ops/bench_attention_triton_vs_aiter.py` | 新增 Triton vs AITER 对比脚本 |

---

## 5. 仍存在的差距与分析

Triton 仍与 AITER 有 2.2-5.5x 差距，根本原因：

1. **寄存器压力**: BLOCK_M=128, D=128 时 acc 需 256 VGPRs（MI300X 上限），导致 spill 到 LDS
2. **MFMA 调度**: Triton 通过 LLVM→AMDGPU 生成代码，无法达到 ASM 级别的指令流水线效率
3. **内存控制**: 缺乏 ASM 级别的 prefetch/DMA 控制
4. **MLA (D=192)**: padded 到 256 导致 ~25% 计算浪费

---

## 6. 建议的后续方向

| 方向 | 优先级 | 预期收益 |
|------|--------|----------|
| 优化 GEMM/Grouped GEMM（Triton 是主力后端） | **高** | 直接影响生产性能 |
| AITER 集成更多 Attention 变体 (MLA/Paged) | 高 | 生产路径使用 AITER |
| Triton Attention 进一步优化 (tl.dot 调度提示) | 中 | Triton 编译器改进的杠杆 |
| FP8 Attention 专项优化 | 中 | 训练精度需求增长 |

---

## 7. 确认项

- [x] 正确性: 所有配置 SNR > 50dB
- [x] 确定性: Forward/Backward 结果确定
- [x] 回归: BF16 生产路径 (AITER) 未受影响
- [x] 兼容性: MLA (D=192) 正确 fallback 到 BLOCK_M=64
