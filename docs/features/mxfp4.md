# MXFP4 training

## Summary

MXFP4 runs the GEMMs of dense and MoE layers in 4-bit floating point, forward and backward: values
in FP4 (E2M1, two per byte) with one E8M0 scale per block of 32 elements. To keep training accurate
at 4 bits, every GEMM op quantizes its operands with a fixed recipe, after
[arXiv:2509.25149](https://arxiv.org/pdf/2509.25149):

- weights use 32 × 32 blocks (2D block quantization), so a weight and its transpose share scales;
- the column-wise copy of the activations, which the weight gradient reads, goes through a random
  Hadamard transform (RHT);
- gradients can be quantized with stochastic rounding.

Besides the GEMMs, Primus-Turbo provides the quantization kernels that feed them and fused ops for
the parts of a training step that run around the GEMMs: the dense and expert MLPs with their
activation, residual add + RMSNorm, the MoE gather, and weight de-oscillation.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| `gemm_fp4` | — | ✅ | — |
| `mlp_fp4` (dense GLU MLP, activation fused into the GEMMs) | — | ✅ | — |
| `grouped_gemm_fp4` | — | ✅ | ✅ |
| `grouped_mlp_fp4` (expert GLU MLP, activation fused into the GEMMs) | — | ✅ | — |
| `quantize_fp4` / `quantize_fp4_with_trans`, `dequantize_fp4` | — | ✅ | ✅ |
| `grouped_quantize_fp4` / `_with_trans`, `grouped_dequantize_fp4` (fuse the MoE gather) | — | ✅ | ✅ |
| `rmsnorm_residual_fp4` (residual add + RMSNorm + MXFP4 quantization) | — | ✅ | — |
| `weight_deosc_update` / `weight_deosc_close` | ✅ | ✅ | — |
| `weight_deosc_qdq` (fused MXFP4 QDQ + oscillation tracking) | — | ✅ | — |

The GEMM, MLP and quantization ops need gfx950 or later and raise on gfx942. `weight_deosc_update`
and `weight_deosc_close` also run on gfx942.

Frameworks: PyTorch. See [Supported Operators § GEMM](../supported_ops.md#21-gemm),
[§ Grouped GEMM](../supported_ops.md#22-grouped-gemm-moe-experts) and
[§ Quantization](../supported_ops.md#26-quantization-and-low-precision-training-support).

## Usage

Install Primus-Turbo as in the [Quick Start](../../README.md#-quick-start).

Dense GEMM. Pass bf16 / fp16 operands; the op quantizes them in the forward and backward:

```python
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import Float4QuantConfig

config = Float4QuantConfig()  # E2M1 values, E8M0 scales, MX blocks of 32
c = turbo.ops.gemm_fp4(a, b, trans_b=True, config=config)  # a [M, K], b [N, K]
```

MoE experts. `a` is the tokens sorted by expert, `group_lens` the number of tokens per expert:

```python
c = turbo.ops.grouped_gemm_fp4(a, b, group_lens, trans_b=True, config=config)  # b [E, N, K]
```

Whole MLPs, with the GLU activation fused into the GEMM epilogues:

```python
# dense: w1 [2I, K], w2 [K, I]
y = turbo.ops.mlp_fp4(x, w1, w2, config=config)
# experts: w1 [E, 2I, K], w2 [E, K, I]; probs are the router weights of the sorted tokens
y = turbo.ops.grouped_mlp_fp4(
    x, w1, w2, group_lens, probs, trans_w1=True, trans_w2=True, config=config
)
```

`w1` holds the gate rows, then the up rows. The default activation is SiLU (SwiGLU);
`clamp_limit` clamps the GLU inputs before the activation.

Already-quantized operands: every GEMM op also takes a `QuantizedTensor` (or a
`QuantizedTensorPair` with its column-wise copy), and then skips that quantization. This is how a
framework caches the quantized weights across micro-batches.

Full example: [Examples § MXFP4 GEMM](../examples.md#35-mxfp4-gemm).

### Weight de-oscillation

At 4 bits, a weight that sits near a rounding boundary can flip between two FP4 values every step
while its FP32 master copy barely moves. The `weight_deosc_*` ops find and stop these weights. They
run on the optimizer's FP32 shard, between steps:

- `weight_deosc_update` adds up, over a period, how far each weight moved in BF16 and how far its
  MXFP4 quantize-dequantize (QDQ) moved;
- `weight_deosc_close` ends the period: where the QDQ moved at least `ratio_threshold` times more
  than the weight itself, it snaps the master weight to its QDQ value, then clears the
  accumulators;
- `weight_deosc_qdq` (gfx950) does the QDQ, the tracking and, with `close=True`, the closure in one
  kernel, without materializing the QDQ copy. Pass the same `scale_rounding_mode` as the forward
  GEMMs so both quantize on the same scale grid.

## Configuration

### `Float4QuantConfig`

| Field | Default | Meaning |
|---|---|---|
| `use_gradient_sr` | `False` | Quantize gradients with stochastic rounding. |
| `scale_rounding_mode` | `0` | Bias of the E8M0 scale exponent: 0 = half ULP, 1 = one ULP, 2 = three-eighths ULP ([arXiv:2607.24377](https://arxiv.org/abs/2607.24377)). |
| `use_preshuffle` | `False` | Keep operands and scales in AITER's shuffled layout. Not accepted by `mlp_fp4` / `grouped_mlp_fp4`. |

The format (`E2M1_X2`), granularity (`MX_BLOCKWISE`), scale type (`E8M0`) and block size (32) are
the only supported values; other values raise.

### Gradient accumulation into `main_grad`

`fuse_bgrad_accum_pattern="megatron"` (`gemm_fp4`, `grouped_gemm_fp4`) and
`fuse_wgrad_accum_pattern="megatron"` (`mlp_fp4`, `grouped_mlp_fp4`) add the weight gradient
straight into the weight's `main_grad` in the GEMM epilogue, following Megatron DDP's contract,
instead of returning it for the framework to add. `PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT=1` lets
the expert wgrad overwrite `main_grad` on its first write of a step instead of reading it back.

### Backends

| Op | Backends |
|---|---|
| `gemm_fp4` | AITER (gfx950, NT only), HIPBLASLT, FLYDSL (gfx950) |
| `grouped_gemm_fp4` | FLYDSL (gfx950), TRITON |
| `mlp_fp4` | FLYDSL for the fused GLU GEMMs (required), HIPBLASLT for the others |
| `grouped_mlp_fp4` | FLYDSL (gfx950) |

Pick one with `PRIMUS_TURBO_GEMM_BACKEND` / `PRIMUS_TURBO_GROUPED_GEMM_BACKEND` (for example
`fp4:aiter`), or let `PRIMUS_TURBO_AUTO_TUNE=1` pick per shape. `PRIMUS_TURBO_DISPATCH_REPORT`
shows which backend ran for each shape. See [Environment variables](../env.md#backend-selection).

## Performance

No MXFP4-versus-BF16 comparison is published yet. Kernel results on MI355X, from the pull requests
that made them:

| Change | Result | Source |
|---|---|---|
| Dense GEMM: GPT-OSS shapes with a 2880 dimension run without a padded operand | 2.0–4.9% faster than AITER's best path on those shapes | [#487](https://github.com/AMD-AGI/Primus-Turbo/pull/487) |
| Dense GEMM and its fused quantization, timed together as one step | GPT-OSS shapes +22.2%, Llama shapes +8.5% | [#514](https://github.com/AMD-AGI/Primus-Turbo/pull/514) |
| Grouped GEMM: FC2 dgrad with the GLU backward fused | +8–15% on GPT-OSS, Qwen3 and DeepSeek-V3 expert shapes | [#523](https://github.com/AMD-AGI/Primus-Turbo/pull/523) |

Benchmark for `gemm_fp4`:

```bash
python benchmark/ops/training/bench_gemm_turbo.py --dtype fp4 --granularity mxfp4
```

The grouped and MLP ops have no benchmark in `benchmark/` yet, and the nightly benchmark machine
(MI325X) cannot run MXFP4.

## End-to-end impact

Primus-LM runs MXFP4 in these MI355X recipes:

- Megatron pretraining: Llama 3.1 8B, Qwen2.5 72B, Qwen3 30B-A3B;
- MLPerf: Llama 3.1 8B pretraining, GPT-OSS 20B pretraining (with weight de-oscillation),
  Llama 2 70B LoRA fine-tuning (the LoRA update is folded into the frozen MXFP4 base GEMM,
  [#609](https://github.com/AMD-AGI/Primus-Turbo/pull/609));
- diffusion: Flux 12B, Wan2.1.

One measured training-step result: on GPT-OSS 20B MXFP4 (8 × MI355X, DP8, MBS 4, GBS 32, sequence
8192), letting the expert wgrad overwrite `main_grad` on its first write of a step cut the step from
572.1 to 554.3 ms (−3.1%), with the same peak memory
([#534](https://github.com/AMD-AGI/Primus-Turbo/pull/534)). The step-time gain of MXFP4 over BF16
is not recorded yet.

## Known limitations

- The GEMM, MLP and quantization ops need gfx950 or later.
- One format: E2M1 values, E8M0 scales, MX blocks of 32.
- Weights are NT: `trans_b=True` for the GEMM ops, and `trans_w1=True, trans_w2=True` for the MLP
  ops. `grouped_mlp_fp4` defaults both to False, so pass them.
- `grouped_gemm_fp4` on gfx950: FLYDSL covers the NT layout with N a multiple of 64. Other shapes
  fall back to TRITON; `PRIMUS_TURBO_DISPATCH_REPORT` lists them, and
  `PRIMUS_TURBO_STRICT_DISPATCH=1` turns the fallback into an error
  ([Dispatch diagnostics](../env.md#dispatch-diagnostics)).
- On gfx1250, the grouped GEMM and the quantization ops run; the dense GEMM, the MLP ops and
  `rmsnorm_residual_fp4` do not.

## Further reading

- MoE gather fused into MXFP4 quantization: [MoE gather quantization](../moe_gather_quantization.md)
- Examples: [§ 3.5 MXFP4 GEMM](../examples.md#35-mxfp4-gemm)
- Tests: `tests/pytorch/ops/test_gemm_fp4.py`, `test_grouped_gemm_fp4.py`,
  `test_grouped_mlp_fp4.py`, `test_rmsnorm_mxfp4_fusion.py`, `test_deoscillation.py`
- Benchmark: `benchmark/ops/training/bench_gemm_turbo.py`
