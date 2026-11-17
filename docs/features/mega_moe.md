# MegaMoE

> **Maturity:** BF16 — Beta; MXFP8 — Experimental ([what this means](./maturity.md))
>
> **Since:** BF16 v0.4.0, MXFP8 v0.5.0

## Summary

MegaMoE runs the expert part of an MoE layer under intra-node expert parallelism (EP) as two fused
kernels: `dispatch_grouped_gemm` (token dispatch + FC1 grouped GEMM) and `grouped_gemm_combine`
(FC2 grouped GEMM + combine). Each kernel overlaps the cross-rank traffic with the GEMM, instead of
running a dispatch, a grouped GEMM, and a combine one after another. Backward recomputes the
dispatched activations instead of saving them, so it also uses less activation memory. Routing is
computed outside: the op takes `topk_idx` and `topk_weights`.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| BF16: `fused_mega_moe`, `fused_mega_moe_stage1` / `_stage2` | — | ✅ | ✅ |
| MXFP8: `fused_mega_moe_fp8_stage1` / `_stage2` | — | ✅ | — |

Intra-node EP only. Frameworks: PyTorch. See [Supported Operators § MoE](../supported_ops.md#24-moe).

## Usage

```python
from primus_turbo.pytorch.ops.moe import fused_mega_moe
```

A complete 8-rank example is in [Examples § MegaMoE](../examples.md#14-megamoe).

## Configuration

- The gated activation defaults to SiLU-SwiGLU; pass `activation=GLUActivation(...)` for others,
  such as `GLUActivation.swigluoai()`.
- The staged ops take per-expert `fc1_bias` / `fc2_bias`, and the BF16 staged ops take
  `fuse_wgrad_accum_pattern="megatron"` to add the expert wgrads into `main_grad` directly.
- MXFP8 combine tuning: `PT_MEGA_FP8_COMBINE_*` and `PT_MEGA_FP8_L1/L2_COMBINE_CU`, see
  [Environment variables § Mega MoE](../env.md#mega-moe-flydsl-fp8-combine).

## Performance

MI355X, 8 ranks (EP8), DeepSeek-V3 expert shape (hidden 7168, intermediate 2048, 256 experts,
top-8, 8192 tokens per rank), BF16. Overlap efficiency is `max(T_comm, T_gemm) / T_measured`.

| Kernel, stage | T_comm (ms) | T_gemm (ms) | T_measured (ms) | Overlap efficiency |
|---|---|---|---|---|
| dispatch_grouped_gemm, forward | 1.53 | 2.90 | 3.20 | 90.5% |
| dispatch_grouped_gemm, dgrad | 1.53 | 1.35 | 1.82 | 84.3% |
| dispatch_grouped_gemm, wgrad dW1 | 1.53 | 2.82 | 3.08 | 91.5% |
| grouped_gemm_combine, forward | 1.48 | 1.68 | 2.19 | 76.7% |
| grouped_gemm_combine, dgrad | 1.48 | 3.49 | 3.62 | 96.5% |

On a DeepSeek-V3 4-layer proxy at EP8, sizing the buffers to the dispatched rows lowers peak memory
from 152.9 to 122.3 GB.

Reproduce:

```bash
python benchmark/ops/training/bench_mega_moe.py --mode dispatch_grouped_gemm --models DeepSeek-V3 --num-processes 8
python benchmark/ops/training/bench_mega_moe.py --mode grouped_gemm_combine --models DeepSeek-V3 --num-processes 8
```

## End-to-end impact

Not recorded yet. Required for Stable.

## Known limitations

- Intra-node EP only. For inter-node EP use [DeepEP](./deep_ep.md).
- Not CUDA Graph capturable: every layer reads the number of dispatched rows back to the host once.
- MXFP8 is available only through the staged ops, and needs hidden % 1024 == 0 and
  intermediate % 256 == 0. Other shapes run zero-padded up to that grid.
- `fuse_wgrad_accum_pattern="megatron"` follows Megatron DDP's `main_grad` contract; do not use it
  under Megatron-FSDP.

## Further reading

- Design: [MegaMoE](../design/mega_moe.md)
- Tests: `tests/pytorch/ops/test_fused_mega_moe.py`, `tests/pytorch/ops/test_fused_mega_moe_fp8.py`
- Benchmark: `benchmark/ops/training/bench_mega_moe.py`
