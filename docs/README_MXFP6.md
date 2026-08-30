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

Enforcement is by probing for those three symbols, not by comparing versions. The minimum
is a commit on `main` rather than a release, a source build reports whatever `git describe`
produced, and a fork can carry any version string it likes, so no version test can decide
the question; the symbols can. `MXFP6_MIN_AITER_COMMIT` in `mxfp6_pack.py` holds the commit
so that the error message, this document and any future CI lane read it from one place.

A consequence worth knowing: MXFP6 necessarily runs ahead of the repository pin, so the
generic `check_aiter_version_once` reports a newer AITER without offering the install
command for the pin, which for an MXFP6 user would be advice to break their build.

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

## Differences from MXFP4

This is the canonical list. The `Float6QuantConfig` and `gemm_fp6` docstrings point here
rather than restating it, so new entries belong here.

MXFP4 steers its quantizers with a `ScalingRecipe` because it has real choices to make: two
backends, two layouts, stochastic rounding, 2D blocking, and RHT on one of the three GEMM
directions. MXFP6 has one of everything, so each FP4 knob has exactly one value here.

| MXFP4 knob | MXFP6 | Why |
| ---------- | ----- | --- |
| `use_rht` | always on | The 32-point Hadamard is fused into the packer and AITER ships no un-rotated entry point. MXFP4 enables its 16-point RHT for wgrad only, where the transform is a pass it would otherwise skip; MXFP6's rides a gather the packer runs regardless, on a bandwidth-bound kernel. |
| `use_2d_block` | never | Scaling is strictly per-1x32 along the contraction axis, so a 2D block has no meaning. |
| `use_sr` | never | Not implemented; see below. |
| `shuffle_scale` / `shuffle_out` | not applicable | `mxfp6_c0c1_256_padk2` is the only layout A6W6 reads. |
| `use_preshuffle` | not applicable | Same: the layout is part of the format, not an option applied on top of it. |

Structural differences that are not knobs:

- **No `is_fp6_dtype`.** There is no torch FP6 dtype. Operands are `uint8` blobs, so
  `is_fp8_dtype` and `is_fp4_dtype` have no counterpart here.
- **No `ScalingRecipe`, no `QuantizedTensor`.** A blob carries neither shape nor recipe, so
  there is nowhere to record one and nothing downstream that could check it.
- **Quantization is dual-direction and fused.** MXFP4 quantizes one tensor at a time with a
  recipe per direction; the MXFP6 packer reads the input once and emits both directions,
  plus an optional bias-GELU epilogue and bias-gradient partials. On Flux 12B the
  materialised transpose this replaces was 82% of the MXFP6-vs-MXFP4 step-time gap, so the
  divergence is the point rather than an accident.
- **No autotuning.** `gemm_fp6_impl` does not go through `AutoKernelDispatcher`. There is
  one backend to choose between, and the dispatcher keys on operand shapes that opaque
  blobs do not carry.

## Not implemented yet

These are tracked follow-ups, not oversights. MXFP6 does not have MXFP4 parity.

- **`PackedQuantizedTensor` weight reuse.** A pre-quantized weight is not reused across
  microbatches the way FP4 allows. Because MXFP6 blobs are opaque and carry no shape, this
  needs a wrapper type first, which would also reopen the public-surface decision above.
- **`beta=1` wgrad accumulation.** The A6W6 entry point has no accumulate epilogue, so
  wgrad cannot write `main_grad` in place. `fuse_bgrad_accum_pattern` raises
  `NotImplementedError`.
- **`grouped_mlp_fp6`.** There is no grouped MXFP6 MoE path.
- **`dequantize_fp6`.** Not simply unwritten: it is not the mirror of `dequantize_fp4`.
  Decoding a packed blob yields `x H` rather than `x`, because the rotation is applied at
  pack time. Recovering `x` means applying the normalised transform a second time, which
  works since it is self-inverse, but no kernel does that today.
- **Stochastic rounding.** With three mantissa bits the MXFP4 motivation largely does not
  apply. `Float6QuantConfig(use_gradient_sr=True)` raises `NotImplementedError`.

## Testing

`tests/pytorch/ops/test_gemm_fp6.py` covers numerics, bit-exactness against AITER and the
API contracts, and needs a gfx950 machine for most of it:

```bash
pytest tests/pytorch/ops/test_gemm_fp6.py
```

CI does not run that part. Its PyTorch lane is gfx942, which is also where the MXFP4 and
MXFP8 suites stand -- they gate on `check_mxfp4_support` / `check_mxfp8_support` and skip
there too -- so a gfx950 lane is what would close the gap, for all of them at once rather
than for MXFP6 alone.

What does run anywhere is grouped at the end of that file: the `gemm_fp6` argument
contracts, `_validate_blobs`, the pack-size and guard-tile arithmetic, the eager prologue
reference, and the AITER half of the capability check, none of which reach a kernel.
`tests/pytorch/core/test_aiter_utils.py` is hardware-independent in the same way. Keep them
so; the docstring on `_skip_if_unsupported` explains what belongs on each side of the line.

## Implementation map

| Path | Role |
| ---- | ---- |
| [`primus_turbo/pytorch/ops/gemm_fp6.py`](../primus_turbo/pytorch/ops/gemm_fp6.py) | Autograd entry point, training-level contracts |
| [`primus_turbo/pytorch/kernels/gemm/gemm_fp6_impl.py`](../primus_turbo/pytorch/kernels/gemm/gemm_fp6_impl.py) | Custom op, backend dispatch, blob validation |
| [`primus_turbo/pytorch/kernels/quantization/mxfp6_pack.py`](../primus_turbo/pytorch/kernels/quantization/mxfp6_pack.py) | Pack API, capability detection, blob sizing |
| [`csrc/kernels/quantization/quantization_mxfp6_gfx950.cu`](../csrc/kernels/quantization/quantization_mxfp6_gfx950.cu) | Fused dual packer with Hadamard, E2M3 conversion, optional epilogue |
| [`csrc/pytorch/quantization/quantization_mxfp6.cpp`](../csrc/pytorch/quantization/quantization_mxfp6.cpp) | Torch bindings and their shape/dtype/device checks |
| [`tests/pytorch/ops/test_gemm_fp6.py`](../tests/pytorch/ops/test_gemm_fp6.py) | Correctness, bit-exactness against AITER, contracts |
| [`tests/pytorch/core/test_aiter_utils.py`](../tests/pytorch/core/test_aiter_utils.py) | The AITER version check's advice, in both directions |
