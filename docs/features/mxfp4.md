# MXFP4 training

> **Maturity:** `gemm_fp4` Beta; `grouped_gemm_fp4`, `mlp_fp4` and `grouped_mlp_fp4` Experimental,
> until they have a benchmark ([what this means](./maturity.md))
>
> **Since:** `gemm_fp4` v0.2.0; `grouped_gemm_fp4` v0.4.0

## Summary

MXFP4 runs the GEMMs of dense and MoE layers in 4-bit floating point: values in FP4 (E2M1, two per
byte) with one E8M0 scale per 32-element block, forward and backward. To keep training accurate at
4 bits it supports 2D block quantization and the random Hadamard transform (RHT), see
[arXiv:2509.25149](https://arxiv.org/pdf/2509.25149). Besides the
GEMMs it provides the quantization kernels that feed them, fused where a training step needs it.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| `gemm_fp4` | — | ✅ | — |
| `mlp_fp4` (dense GLU MLP with fused activation) | — | ✅ | — |
| `grouped_gemm_fp4` | — | ✅ | ✅ |
| `grouped_mlp_fp4` (expert MLP with fused activation) | — | ✅ | — |
| `quantize_fp4` / `_with_trans`, `dequantize_fp4`, grouped variants | — | ✅ | ✅ |
| `rmsnorm_residual_fp4` (residual add + RMSNorm + MXFP4 quantization) | — | ✅ | — |

Frameworks: PyTorch. See [Supported Operators § GEMM](../supported_ops.md#21-gemm) and
[§ Grouped GEMM](../supported_ops.md#22-grouped-gemm-moe-experts).

## Usage

```python
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import Float4QuantConfig, Format, ScaleDtype, ScalingGranularity

config = Float4QuantConfig(
    format=Format.E2M1_X2,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    block_size=32,
    scale_dtype=ScaleDtype.E8M0,
)
c = turbo.ops.gemm_fp4(a, b, trans_a=False, trans_b=True, out_dtype=a.dtype, config=config)
```

Full example: [Examples § MXFP4 GEMM](../examples.md#35-mxfp4-gemm).

## Configuration

- Backends:
  - `gemm_fp4`: AITER (NT layout only), HIPBLASLT, FLYDSL.
  - `grouped_gemm_fp4`: FLYDSL (gfx950), TRITON.
  - `mlp_fp4` / `grouped_mlp_fp4`: FLYDSL.

  Choose with `PRIMUS_TURBO_GEMM_BACKEND` / `PRIMUS_TURBO_GROUPED_GEMM_BACKEND` (for example
  `fp4:aiter`), or let `PRIMUS_TURBO_AUTO_TUNE=1` pick per shape. See
  [Environment variables § Backend selection](../env.md#backend-selection).
- Accuracy options: `use_2d_block` and `use_rht` on `ScalingRecipe`; `use_gradient_sr`
  (stochastic rounding of gradients) and `scale_rounding_mode` on `Float4QuantConfig`.
- `Float4QuantConfig.use_preshuffle` passes operands to AITER already in its shuffled layout.
- `PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT=1` lets the grouped wgrad overwrite its output instead of
  accumulating into it, when the caller allows it.

## Performance

No published numbers yet. Benchmark for `gemm_fp4`:

```bash
python benchmark/ops/training/bench_gemm_turbo.py --dtype fp4 --granularity mxfp4
```

`grouped_gemm_fp4`, `mlp_fp4` and `grouped_mlp_fp4` have no benchmark yet.

## End-to-end impact

Used by Primus-LM's MXFP4 recipes (for example Llama 3.1 8B and Qwen3 30B-A3B). Measured
training-step gain not recorded yet. Required for Stable.

## Known limitations

- Requires gfx950 or later.
- Only `MX_BLOCKWISE` with `E2M1_X2` values, `E8M0` scales and block size 32.
- The AITER backend takes only the NT layout (`trans_b=True`).
- `grouped_gemm_fp4` on gfx950: the FLYDSL default does not cover every shape. Shapes with N or K
  such as 96, 160, 1568 or 2080 fall back to TRITON, which is slower; `dispatch_report()` lists
  them, see [Environment variables § Dispatch diagnostics](../env.md#dispatch-diagnostics).

## Further reading

- MoE gather fused into MXFP4 quantization: [MoE gather quantization](../design/moe_gather_quantization.md)
- Tests: `tests/pytorch/ops/test_gemm_fp4.py`, `test_grouped_gemm_fp4.py`, `test_grouped_mlp_fp4.py`
- Benchmark: `benchmark/ops/training/bench_gemm_turbo.py`
