# Blockwise FP8 优化技能

## 触发条件
当用户讨论 blockwise FP8 性能、分析 blockwise 量化瓶颈、需要优化 GEMM/Grouped GEMM/Attention 的 blockwise FP8 路径时使用。

## Blockwise FP8 量化原理

### 量化公式

```
block_size = 128
对于每个 block (连续128个元素):
  scale = max(|block|) / FP8_E4M3_MAX   (FP8_E4M3_MAX = 448.0)
  quantized = clamp(block / scale, -448, 448).to(fp8_e4m3)
```

### 与其他量化粒度对比

| 粒度 | Scale 数量 | 精度 | 额外内存带宽 |
|------|-----------|------|------------|
| Tensorwise | 1 per tensor | 低 | 极小 |
| Rowwise | M (per row) | 中 | 小 |
| Blockwise-128 | M×ceil(K/128) | 高 | 较大 |

## 代码路径

### GEMM FP8 Blockwise
1. **PyTorch 入口**: `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`
   - `GEMMFp8KernelDispatcher` → dispatch 到 CK/Triton
2. **CK 后端**: `csrc/kernels/gemm/ck_gemm_kernel_instance_factory.{hip,cu}`
   - 使用 `ABQuantGrouped` quantization mode
   - Tile config: `256x256x128` 或 `128x128x128`
3. **Triton 后端**: `primus_turbo/triton/gemm/gemm_kernel.py`
   - `_gemm_fp8_blockwise_kernel` 或 gemm kernel 内部 dequant

### Grouped GEMM FP8 Blockwise
1. **PyTorch 入口**: `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`
2. **CK 后端**: `csrc/kernels/grouped_gemm/ck_grouped_gemm_kernel_instance_factory.{hip,cu}`
   - `ABQuantGrouped` mode
3. **Triton 后端**: `primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py`

### Attention FP8 Blockwise
1. **Attention 入口**: `primus_turbo/pytorch/ops/attention/`
2. **量化**: `primus_turbo/pytorch/ops/attention/attention_utils.py` → `block_scaling_node()`
3. **Kernel**: `primus_turbo/triton/attention/attention_kernel.py`

## 优化方向

### P0 (高优先级)
1. **CK ABQuantGrouped tile 选择优化** — M-aware 选择已对 RowCol/Tensor 生效，但 ABQuantGrouped 可能需要不同策略
2. **Scale tensor 内存访问优化** — blockwise 的 scale tensor 较大，优化其 layout/预取
3. **Triton blockwise kernel fusion** — 将 scale 计算融合进 GEMM kernel

### P1 (中优先级)
4. **Block size 自适应** — 不同 shape 可能适合不同 block_size (64/128/256)
5. **CK kernel 实例扩展** — 添加针对 blockwise 优化的新 tile 配置
6. **后端选择优化** — blockwise 场景下 CK vs Triton vs HIPBLASLt 的最优选择

### P2 (探索性)
7. **混合精度 blockwise** — 关键路径用 FP8，非关键用 BF16
8. **Online quantization** — 将量化操作融合到上游 kernel
9. **Scale compression** — 利用 scale 的空间局部性压缩存储

## 性能分析方法

### Roofline 分析
MI300X 参数:
- HBM 带宽: 5.3 TB/s
- FP8 峰值: 2600+ TFLOPS (per-chip)
- BF16 峰值: 1300+ TFLOPS (per-chip)

Blockwise FP8 额外带宽:
```
scale_bytes = ceil(M/block_size) × ceil(K/block_size) × sizeof(float32) × 2  (A+B的scale)
total_bytes = M×K + K×N + M×N + scale_bytes
arithmetic_intensity = 2×M×N×K / total_bytes
```

### Profiling
```bash
# 快速 kernel 级 profiling
rocprof --stats python3 -c "
import torch
from primus_turbo.pytorch.ops import gemm
# ... blockwise FP8 benchmark
"
```

## 精度验证标准

| 场景 | SNR 阈值 | 绝对误差阈值 |
|------|---------|-------------|
| GEMM FP8 blockwise fwd | ≥ 20 dB | atol=0.1 |
| GEMM FP8 blockwise bwd | ≥ 15 dB | atol=0.5 |
| Grouped GEMM FP8 blockwise | ≥ 20 dB | atol=0.1 |
| Attention FP8 blockwise | ≥ 20 dB | — |
