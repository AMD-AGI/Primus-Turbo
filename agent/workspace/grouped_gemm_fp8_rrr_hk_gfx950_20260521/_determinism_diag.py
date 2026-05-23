"""Determinism diagnostic for the 4 SNR-borderline cases.
If kernel is non-deterministic across repeated calls with identical input,
that's a race-condition bug (atomicAdd order, uninitialized accumulator, etc.)
and it explains SNR landing in 25-28 dB band non-reproducibly.

Each case: 10 repeated calls; pairwise bitwise equality.
"""
import os, sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
import primus_turbo
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.core.backend import GlobalBackendManager, BackendType
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)
DEV = "cuda"
N_REPEATS = 10

# All 4 SNR-FAIL or borderline cases + 1 control
CASES = [
    ("dsv3-down-B16-M2048",  16, 2048, 7168, 2048),
    ("dsv3-down-B16-M4096",  16, 4096, 7168, 2048),
    ("qwen-down-B16-M4096",  16, 4096, 4096, 1536),
    ("qwen-down-B16-M8192",  16, 8192, 4096, 1536),
    # control: a case that PASSed cleanly (28.5 dB)
    ("dsv3-up-B4-M4096-CTRL", 4, 4096, 4096, 7168),
]

GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
GlobalBackendManager.set_auto_tune(False)
cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)

for label, B, M_g, N, K in CASES:
    print(f"\n=== {label}: B={B} M_g={M_g} N={N} K={K} ===")
    # Build identical inputs each run (don't reuse autograd graph state)
    a0 = (torch.randn((B * M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    b0 = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    group_lens = torch.full((B,), M_g, dtype=torch.int64, device=DEV)

    # Reference (bf16 path) once
    a_ref = a0.detach().clone().requires_grad_(True)
    b_ref = b0.detach().clone().requires_grad_(True)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)

    outs = []
    snrs = []
    for r in range(N_REPEATS):
        a = a0.detach().clone().requires_grad_(True)
        b = b0.detach().clone().requires_grad_(True)
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
        torch.cuda.synchronize()
        outs.append(out.detach().clone())
        snrs.append(compute_snr(out_ref, out))

    # Pairwise bitwise check
    out0 = outs[0]
    n_diff = 0
    max_abs_diff = 0.0
    for i in range(1, N_REPEATS):
        if not torch.equal(out0, outs[i]):
            n_diff += 1
            d = (out0.float() - outs[i].float()).abs().max().item()
            max_abs_diff = max(max_abs_diff, d)

    print(f"  SNR samples: {' '.join(f'{s:.2f}' for s in snrs)}")
    print(f"  SNR range:   min={min(snrs):.2f}  max={max(snrs):.2f}  spread={max(snrs)-min(snrs):.3f}")
    print(f"  Bitwise non-deterministic? {n_diff}/{N_REPEATS-1} runs differ from run-0")
    if n_diff > 0:
        print(f"  max |delta| across runs: {max_abs_diff:.6e}")
        print(f"  *** NON-DETERMINISTIC KERNEL BUG ***")
    else:
        print(f"  Output is bitwise deterministic. SNR variation comes from input differences (none here) or fp8 quant boundary instability.")
