# MXFP6 (E2M3)

## Overview

MXFP6 is a 6-bit microscaled float format: an E2M3 element (1 sign, 2 exponent, 3 mantissa
bits) with one shared E8M0 power-of-two scale per 32 values along the contraction axis.
Primus-Turbo exposes it as a single training-ready GEMM, `gemm_fp6`, backed by AITER's
gfx950 A6W6 assembly kernels and by a fused packer implemented here.

It sits between MXFP4 and MXFP8: three mantissa bits rather than one, at 1.5x the operand
bytes of FP4. On Flux 12B the measured forward and gradient SNR is around 30 dB, against
the roughly 10 dB an equivalent MXFP4 configuration reaches.

### Key points

- gfx950 only (MI350X / MI355X). The A6W6 GEMM is prebuilt assembly and the packer uses
  the hardware FP6 conversion instruction, so neither exists on other architectures.
- The output is always bf16. The A6W6 assembly writes nothing else.
- The 32-point Hadamard rotation is mandatory, not an option. It is fused into the packer,
  and the GEMM depends on it cancelling between the two operands.
- Operands are opaque packed blobs, not strided tensors with a separate scale. They carry
  no shape, so the caller keeps M, N and K.

## Requirements

### Hardware

gfx950. `check_mxfp6_support()` returns `(False, reason)` on anything else rather than
raising, so a caller can branch on it.

### Build

The MXFP6 packer is guarded by `BUILD_MXFP6_BACKEND`, which `setup.py` defines only when
gfx950 is among the offload architectures. A build configured for other architectures
cannot pack MXFP6 even when it later runs on a gfx950 machine:

```bash
GPU_ARCHS="gfx950" pip3 install --no-build-isolation -e ".[pytorch]" -v
```

### AITER

Two tiers, which are deliberately not the same pin.

| Tier | Revision | Enforced? |
| ---- | -------- | --------- |
| Functional minimum | [ROCm/aiter#4859](https://github.com/ROCm/aiter/pull/4859), merge commit `0c2b0f77b2ff6d13c677d12466abf87299f8b260` | Yes, by `check_mxfp6_support()` |
| Flux performance | a revision containing [ROCm/aiter#5117](https://github.com/ROCm/aiter/pull/5117) | No, recommendation only |

The minimum is what supplies `gemm_a6w6`, `quant_mxfp6_gemm` and `mxfp6_gemm_pack_size`.
Nothing below it runs, and nothing above it is required in order to run. The repository's
declared pin `v0.1.14.post1` predates that merge, so MXFP6 reports itself unsupported
until AITER is upgraded.

The performance tier adds non-temporal-store kernel variants and a tuned table for Flux
projection shapes. Primus-Turbo picks that up with no code change -- it names no kernel and
no config table, and AITER merges the tuned rows internally -- so it is a recommendation
rather than a dependency. Its kernels are bitwise-equal to the defaults, so moving between
the two tiers does not change results.

## Usage

```python
import torch
from primus_turbo.pytorch.ops import gemm_fp6
from primus_turbo.pytorch.ops.quantization import check_mxfp6_support

supported, reason = check_mxfp6_support()
if not supported:
    raise SystemExit(reason)

a = torch.randn(4096, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
b = torch.randn(3072, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = gemm_fp6(a, b)          # a @ b.T, bf16
out.backward(torch.ones_like(out))
```

### Public surface

`gemm_fp6`, `check_mxfp6_support` and `mxfp6_pack_sizes` are public. The packers
(`quantize_mxfp6_row` / `_col` / `_dual` / `_fused_dual`) and their helpers are internal:
they traffic in AITER's packed blob layout, which is not ours to keep stable, and their
blobs carry no shape for a caller to validate. Reach MXFP6 through `gemm_fp6`.

### Alignment

Two different contracts, on purpose:

| Entry point | M | N | K |
| ----------- | - | - | - |
| `gemm_fp6` (training) | % 256 | % 256 | % 256 |
| `gemm_fp6_impl` / `GEMMFP6AITERBackend` (one GEMM) | % 256 | % 256 | % 128 |

A single forward GEMM is correct with K a multiple of 128. `gemm_fp6` is stricter because
its backward GEMMs permute the roles: dgrad produces an M x K result and wgrad an N x K
one, so K becomes an output dimension and inherits the 256 rule. Checking it at the
autograd entry point turns what would be a mid-backward failure into an error at the call
site.

`gemm_a6w6` itself pads internally and is correct on unaligned shapes, so the backend-level
check is a padding-waste guard rather than a correctness one.

### Input dtype

bf16 and fp16 only. fp32 is rejected with a `TypeError`: only those two templates are
instantiated in the packer.

### torch.compile

Both the packers and the GEMM are opaque `custom_op`s with hand-written fakes, so a
compiled forward and backward trace without a graph break. They are opaque deliberately:
AITER's kernel selection does `lru_cache`d pandas lookups on M/N/K that SymInts cannot
satisfy.

## Numerical behavior

### The fused epilogue rounds tanh differently

`quantize_mxfp6_fused_dual` can fold a bias-add plus tanh-GELU into the packer's staging
read, which removes an entire HBM round-trip of the activation. Every part of that is
bit-identical to running the epilogue separately except the tanh: the kernel evaluates it
in closed form from one hardware `exp2` rather than calling a libm tanh. That is a
54-instruction-per-element difference, and it is what decides whether fusing is faster than
not fusing at all.

The consequence is a different rounding of the activation, not a less accurate one. About
0.0003% of packed codes differ by one level, and the E8M0 scales are untouched.
`MXFP6_PROLOGUE_IDENTITY` has no tanh and is exactly equal.

Around the GELU cutoff at roughly `-9.02`, where tanh saturates to -1, the fused and eager
paths can disagree on which side of the boundary an input falls.
`test_fused_prologue_cutoff_matches_eager_for_adjacent_bf16_inputs` pins the behavior at
every adjacent bf16 input.

### The bias-gradient partial is not bit-exact

`want_col_sum=True` returns per-column sums of the staged values, because the tensor a bias
gradient would otherwise be reduced from no longer exists in HBM. This is a different
(tree-ordered, fp32-accumulated) summation of the same values, so it is not bitwise equal
to the eager reduction.

### Guard tiles

Packed blobs carry two trailing K-tiles whose contents are never read but whose space is
mandatory: the assembly derives its row-tile stride from `k/128 + 2`. The packer leaves
them uninitialised, so two packs of identical input differ there. Anything comparing blobs
must go through `mxfp6_data_region`.

## Benchmark

```bash
python benchmark/ops/training/bench_mxfp6_fused_pack.py
```

Measures the fused epilogue against the path it replaces (an eager epilogue writing bf16 to
HBM, then a pack that reads it back) at Flux 12B MLP shapes. Record the AITER revision
alongside any numbers: GEMM throughput at those shapes is a property of AITER's tuned table
rather than of Primus-Turbo, so an unstamped result goes stale when the pin advances.

## Not implemented yet

These are tracked follow-ups, not oversights. MXFP6 does not have MXFP4 parity.

- **`PackedQuantizedTensor` weight reuse.** A pre-quantized weight is not reused across
  microbatches the way FP4 allows. Because MXFP6 blobs are opaque and carry no shape, this
  needs a wrapper type first, which would also reopen the public-surface decision above.
- **`beta=1` wgrad accumulation.** The A6W6 entry point has no accumulate epilogue, so
  wgrad cannot write `main_grad` in place. `fuse_bgrad_accum_pattern` raises
  `NotImplementedError`.
- **`grouped_mlp_fp6`.** There is no grouped MXFP6 MoE path.
- **Stochastic rounding.** With three mantissa bits the MXFP4 motivation largely does not
  apply. `Float6QuantConfig(use_gradient_sr=True)` raises `NotImplementedError`.

## Implementation map

| Path | Role |
| ---- | ---- |
| [`primus_turbo/pytorch/ops/gemm_fp6.py`](../primus_turbo/pytorch/ops/gemm_fp6.py) | Autograd entry point, training-level contracts |
| [`primus_turbo/pytorch/kernels/gemm/gemm_fp6_impl.py`](../primus_turbo/pytorch/kernels/gemm/gemm_fp6_impl.py) | Custom op, backend dispatch, blob validation |
| [`primus_turbo/pytorch/kernels/quantization/mxfp6_pack.py`](../primus_turbo/pytorch/kernels/quantization/mxfp6_pack.py) | Pack API, capability detection, blob sizing |
| [`csrc/kernels/quantization/quantization_mxfp6_gfx950.cu`](../csrc/kernels/quantization/quantization_mxfp6_gfx950.cu) | Fused dual packer with Hadamard, E2M3 conversion, optional epilogue |
| [`csrc/pytorch/quantization/quantization_mxfp6.cpp`](../csrc/pytorch/quantization/quantization_mxfp6.cpp) | Torch bindings and their shape/dtype/device checks |
| [`tests/pytorch/ops/test_gemm_fp6.py`](../tests/pytorch/ops/test_gemm_fp6.py) | Correctness, bit-exactness against AITER, contracts |
