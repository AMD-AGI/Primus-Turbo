# HIP MX-FP8 Grouped GEMM

Pure-HIP MX-FP8 (OCP Microscaling, e4m3 data + e8m0 per-32-K scales) grouped
GEMM for MoE gate_up on AMD gfx950 (MI355X). Drop-in acceleration for the
Primus-Turbo Triton baseline.

## Status (2026-04-24)

Shipped on `feat/mxfp8-grouped-gemm-e8m0`. Head-to-head vs Triton on the
gpt_oss_20B gate_up shape (M=65536, K=2880, N=5760, G=32, balanced):

| kernel | Triton | HIP | ratio |
|---|---:|---:|---:|
| fwd (256×256×128 tile) | 1.56 ms | 1.38 ms | **1.14×** |
| dgrad (fwd reuse with dgrad-layout B) | 1.42 ms | 1.10 ms | **1.29×** |
| wgrad (v1.3 fused via fwd reuse) | 1.68 ms | 1.69 ms | 0.99× |

| autograd step | ms | vs pure Triton |
|---|---:|---:|
| pure Triton (`FP8GroupedGemmMXFunc`) | 7.28 | 1.00× |
| hybrid HIP (no prequant) | 9.20 | 0.79× |
| **hybrid + MXFP8WeightPrequantHip** | **6.24** | **1.17×** |

Correctness: 28.44 / 28.46 / 28.46 dB (out / grad_a / grad_b vs bf16
reference) = fp8 e4m3 noise floor, bit-consistent with Triton.

## Public API

```python
from primus_turbo.hip.grouped_gemm_mxfp8 import (
    # Low-level kernel entry points (take pre-quanted fp8 + scales)
    grouped_gemm_mxfp8_hip_fwd,
    grouped_gemm_mxfp8_hip_dgrad,
    grouped_gemm_mxfp8_hip_variable_k,       # balanced-MoE wgrad
    grouped_gemm_mxfp8_hip_variable_k_padded,  # unbalanced-MoE wgrad (opt-in)
)

from primus_turbo.hip.grouped_gemm_mxfp8.autograd import (
    # High-level autograd Function (quant-then-matmul, saves ctx for bwd)
    FP8GroupedGemmMXHipFunc,
    grouped_gemm_mxfp8_hip,          # wrapper around apply()
    grouped_gemm_mxfp8,              # env-selected (TURBO_MXFP8_GG_BACKEND=hip|triton)

    # Prequant container for k>=2 gradient-accumulation training
    MXFP8WeightPrequantHip,
    prequant_mxfp8_weights_hip,
)
```

### Usage

```python
import torch
from primus_turbo.hip.grouped_gemm_mxfp8.autograd import (
    grouped_gemm_mxfp8_hip, prequant_mxfp8_weights_hip,
)

# MoE gate_up: A [M_total, K] bf16, B [G, N, K] bf16 (NT layout)
a = torch.randn(65536, 2880, device="cuda", dtype=torch.bfloat16, requires_grad=True)
b = torch.randn(32, 5760, 2880, device="cuda", dtype=torch.bfloat16, requires_grad=True)
group_lens = torch.full((32,), 2048, dtype=torch.int64, device="cuda")
group_offs = torch.arange(0, 65537, 2048, dtype=torch.int64, device="cuda")

# One-off fwd (quantises B every call)
out = grouped_gemm_mxfp8_hip(a, b, group_lens, group_offs)

# Training loop with gradient accumulation: prequant B once per optimizer step.
# 1.17× vs pure-Triton step, mostly from hoisted B-quant + HIP fwd+dgrad kernels.
for _ in range(num_optim_steps):
    pq = prequant_mxfp8_weights_hip(b.detach())   # once
    for _ in range(k_accum):
        out = grouped_gemm_mxfp8_hip(a, pq, group_lens, group_offs)
        out.backward(grad_out)
```

## Constraints

| dim | constraint | reason |
|---|---|---|
| B layout | `[G, N, K]` (NT, Triton's `trans_b=True` convention) | HIP kernel NT-only |
| K | `% 32 == 0` and `>= 384` | MX scale group + kernel pipeline prologue |
| N | `% 16 == 0` | fp8 MFMA + preshuffle alignment |
| per-expert M_g (HIP path) | `% 16 == 0` | fp8 preshuffle alignment per-expert |
| HIP wgrad v1 (balanced) | all M_g equal AND `>= 384` AND `% 128 == 0` | stacked-M kernel trick |

**Unbalanced MoE** (any M_g misaligned): autograd auto-falls-back to Triton
for fwd/dgrad/wgrad. No user action needed — correctness always preserved.
`grouped_gemm_mxfp8_hip_variable_k_padded` is available as opt-in for
all-HIP builds; slower than Triton for general unbalanced (4-5×) but wins
on catastrophic few-non-zero-expert cases (0.67× Triton on 4-expert warmup).

## Files

### Kernels
- `turbo_grouped_gemm_mxfp8.hip` — HIP kernel source + pybind (JIT-built via torch.utils.cpp_extension). Reuses the production `GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950` tile struct from `csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h`.
- `_permute_fp8.py` — Triton LDS-tile transpose for fp8 (9× faster than torch's `.permute().contiguous()` at fp8 dtype). Key component of the v1.3 wgrad win.

### Python wrappers
- `__init__.py` — JIT build glue + kernel entry points (fwd / dgrad / wgrad / variable_k / variable_k_padded).
- `autograd.py` — `FP8GroupedGemmMXHipFunc` autograd Function, `MXFP8WeightPrequantHip` prequant container, env selector.

### Tests / benches
- `test_phase_a.py` — fwd kernel correctness + kernel-only bench.
- `test_phase_b.py` — dgrad correctness + bench.
- `test_variable_k.py` — wgrad kernel-only bench (HIP vs Triton).
- `test_wgrad_padded.py` — padded-HIP wgrad correctness + bench on balanced/unbalanced/catastrophic.
- `test_autograd.py` — full step correctness + bench.
- `test_real_shapes.py` — sweep across 8 real gpt_oss_20B production shapes from `gem_shape_summary.txt`.
- `bench_fwd_bwd.py` — end-to-end perf table (kernel-only + step).

### Design
- `WGRAD_DESIGN.md` — history of wgrad iterations (v0 → v1.5).
- `WGRAD_V2_PLAN.md` — 3-day plan for a true LDS-transpose variable-K kernel (closes remaining balanced + all unbalanced gaps). Currently deferred; the v2 engineering gain is 5% e2e vs 2-3 days of deep kernel work — deferred until other training-stack wins are exhausted.

## Key design decisions

1. **Reuse, not rewrite**. The HIP kernel is a `flat_grid + per-expert base-ptr` wrapper around the production `GEMM_Tile_MXFP8_NT_256x256x128_...` tile struct. ~500 lines of delta, inherits all compute-side tuning. Never tried to hand-code MFMA layout from scratch (see skill [`mfma_16x16x128_f8f6f4_operand_layout`](../../../../../kernel-agents/knowledge_base/skills/mfma_16x16x128_f8f6f4_operand_layout.json) — a prior 128×128×128 attempt failed at 8.6 dB SNR due to ISA lane-layout pitfalls).

2. **Profile first**. The initial HIP wgrad was 0.44× Triton. Profiling found 60% of time in `torch.permute(...).contiguous()` on fp8 (running at 12% of HBM peak), not the kernel. A Triton LDS-tile transpose (85% HBM peak) moved it to 0.99× Triton — no kernel changes. Lesson: [`feedback_profile_before_kernel_rewrite`](../../../../../../../home/yanyuqin/.claude/projects/-mnt-vast-john-rocm-dynamo/memory/feedback_profile_before_kernel_rewrite.md).

3. **Don't fuse permute into upstream**. Tried emitting pre-permuted col output from the quant kernel — bit-exact but 20% SLOWER due to Triton's `BlockedEncoding` fighting the permuted-store pattern. Post-permute via dedicated LDS-tile kernel wins. Lesson: [`triton_permuted_store_blocking_layout_pitfall`](../../../../../kernel-agents/knowledge_base/skills/triton_permuted_store_blocking_layout_pitfall.json).

4. **Triton wgrad fallback is a feature, not a bug**. For unbalanced MoE (arbitrary M_g), Triton's `variable-K` kernel is the right tool — purpose-built, single-launch, memory-bound-optimized. HIP wgrad wins on balanced via its faster fwd/dgrad kernels; unbalanced correctly routes to Triton. Both paths share the same `quant_mxfp8_dual_jagged` upstream, so no quant overhead difference.

5. **hipBLASLt MX-FP8 wgrad is architecturally blocked**. `VEC32_UE8M0` scale mode groups along the innermost dim of the operand; combined with the `opA=T / opB=N` requirement this puts scales along the output axis — wrong for wgrad (which needs scales along reduction). See skill [`pmc_before_kernel_rewrite`](../../../../../kernel-agents/knowledge_base/skills/pmc_before_kernel_rewrite.json) and the C++ patches at [`hipblaslt_grouped_gemm.cpp:131`](../../../../csrc/pytorch/grouped_gemm/hipblaslt_grouped_gemm.cpp) (kept harmless in the binding, unblocks future use when hipBLASLt's `hipblaslt_ext::GroupedGemm` API is wired up).

## Running the tests

Dev container: `rocm/primus-training-private:20260414_nightly_ainic77`.

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --ipc=host --network=host \
  -v $(pwd):/work/Primus-Turbo -w /work/Primus-Turbo \
  rocm/primus-training-private:20260414_nightly_ainic77 \
  bash -c "PYTHONPATH=. python primus_turbo/hip/grouped_gemm_mxfp8/bench_fwd_bwd.py --iters 30"
```

Rebuild after C++ changes (the `.hip` source is JIT-built; the `csrc/`
hipblaslt patch needs a full reinstall):

```bash
# In container
git config --global --add safe.directory /work/Primus-Turbo
pip install -e . --no-build-isolation
```
