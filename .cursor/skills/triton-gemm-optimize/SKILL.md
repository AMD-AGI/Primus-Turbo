---
name: triton-gemm-optimize
description: >-
  优化 Primus-Turbo 中的 Triton GEMM kernel 性能。
  当用户要求优化 GEMM kernel、修改 gemm_kernel.py / gemm_fp8_kernel.py、
  讨论 GEMM 性能、或分析 GEMM benchmark 结果时使用。
---

# Triton GEMM 优化指南

## 核心文件

| 文件 | 作用 |
|------|------|
| `primus_turbo/triton/gemm/gemm_kernel.py` | BF16/FP16 persistent GEMM (~700行) |
| `primus_turbo/triton/gemm/gemm_fp8_kernel.py` | FP8 tensorwise + blockwise GEMM |
| `primus_turbo/pytorch/kernels/gemm/gemm_impl.py` | PyTorch 封装 (Triton/CK/hipBLASLt dispatch) |
| `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py` | FP8 GEMM PyTorch 封装 |
| `primus_turbo/pytorch/ops/gemm.py` | 高级 API |
| `benchmark/ops/bench_gemm_turbo.py` | 性能 benchmark |
| `benchmark/accuracy/eval_gemm_accuracy.py` | 精度验证 |
| `benchmark/ops/config.py` | 配置 (模型 shapes, batch sizes) |

## 当前实现概况

- **BF16 GEMM**: Persistent kernel，使用 origami 选择 tile 参数
- **FP8 GEMM**: Tensorwise + blockwise 量化支持
- **多后端**: Triton / CK (Composable Kernel) / hipBLASLt，通过环境变量选择
- **硬件常量**: NUM_XCDS = 8 (MI300X), XCD swizzle 已集成
- **gfx950 特化**: MI350X 专用参数路径

## 优化方向（按优先级）

### P0: M-Bucketed 配置选择

**来源**: AITER `aiter/ops/triton/README.md`

当前 origami 选择参数，但可进一步按 M 值分段优化:
- **M ≤ 16** (decode): 小 BLOCK_M (32/64)，大 BLOCK_N，考虑 Split-K
- **16 < M ≤ 256** (small batch): 中等 tile，平衡计算和访存
- **M > 256** (prefill): 大 tile (128/256)，最大化 MFMA 利用率

关键: 不同 M 区间性能差异可达 2-5x，静态配置无法兼顾。

### P0: Split-K (小 M 场景)

**来源**: AITER `compute_splitk_params`, Triton tutorials

对于 decode 场景 (M=1~16):
- 标准 GEMM 的 grid 维度不足以占满所有 CU
- Split-K 将 K 维切分到多个 CU 并行计算，最后归约
- 需要额外的中间缓冲和原子操作或二次 kernel 归约
- AITER 使用 `tl.atomic_add` 进行 FP32 归约

```python
# Split-K 参数选择启发式
def compute_splitk(M, N, K, num_sms=304):
    tiles = (M // BLOCK_M) * (N // BLOCK_N)
    if tiles >= num_sms:
        return 1  # 不需要 split-K
    split_k = min(K // BLOCK_K, num_sms // max(tiles, 1))
    return max(split_k, 1)
```

### P1: XCD Swizzle 优化

**来源**: AITER, 当前已在 gemm_kernel.py 中实现

MI300X 有 8 个 XCD (Accelerated Compute Die)，每个 XCD 38 CU。
Swizzle 确保相邻 tile 分布到不同 XCD，最大化 L2 局部性:
```python
pid_m = (pid // width) * GROUP_M + (pid % GROUP_M)
pid_n = (pid % width) // GROUP_M
# XCD 映射: pid → (pid % 8) 对应不同 XCD
```

优化点: 验证 GROUP_M 值是否在各 shape 下最优。

### P1: Persistent Kernel 调优

当前已使用 persistent kernel，优化点:
- 验证 `num_programs` 是否匹配 CU 数 (304 for MI300X)
- Super-M 行调度是否有效利用 L2 cache
- epilogue 是否有不必要的同步

### P2: Block-Scale FP8 优化

**来源**: AITER `triton_gemm_a8w8_blockscale`

- Per-block 量化比 per-tensor 更精确，但需要额外的 scale 加载
- Scale 布局需对齐 MFMA tile 边界
- AITER 的 preshuffle weight layout 可减少 dequant 开销

## 后端选择决策树

```
输入: M, N, K, dtype
  │
  ├── dtype == BF16/FP16?
  │     ├── M > 256 → hipBLASLt (厂商深度优化，大矩阵最优)
  │     ├── M ≤ 256 → Triton (persistent + split-K 灵活性)
  │     └── 特殊 shape (非 16 对齐) → 需要实测对比
  │
  ├── dtype == FP8 tensorwise?
  │     ├── M > 256 → hipBLASLt 或 CK (视 shape)
  │     └── M ≤ 256 → Triton (split-K)
  │
  └── dtype == FP8 blockwise?
        └── Triton 或 CK (hipBLASLt 对 blockwise 支持有限)
```

## 标准验证流程

```bash
# 1. 编译
pip3 install --no-build-isolation -e . -v

# 2. 精度验证
python3 benchmark/accuracy/eval_gemm_accuracy.py
pytest tests/pytorch/ops/test_gemm.py -x
pytest tests/pytorch/ops/test_gemm_fp8.py -x

# 3. 性能 benchmark
PRIMUS_TURBO_GEMM_BACKEND=TRITON python3 benchmark/ops/bench_gemm_turbo.py

# 4. 多后端对比
python3 benchmark/ops/bench_gemm_turbo.py   # default (auto)
python3 benchmark/ops/bench_gemm_torch.py   # PyTorch baseline
python3 benchmark/ops/bench_gemm_te.py      # TransformerEngine baseline
```

## 关键性能指标

MI300X BF16 理论峰值: 1307.4 TFLOPS
MI300X FP8 理论峰值: 2614.9 TFLOPS
GEMM 效率 = 实际 TFLOPS / 理论峰值

目标效率:
- 大矩阵 (M>1024): > 70%
- 中矩阵 (M=128~1024): > 50%
- 小矩阵 (M<128): > 30% (主要受 launch overhead 和并行度限制)

## 参考
- AITER Triton GEMM: `ref/OP_optimize/aiter/aiter/ops/triton/gemm/`
- Triton persistent matmul tutorial: `ref/OP_optimize/triton/python/tutorials/09-persistent-matmul.py`
- origami 参数选择: `primus_turbo/triton/gemm/gemm_kernel.py::_select_params_origami`
