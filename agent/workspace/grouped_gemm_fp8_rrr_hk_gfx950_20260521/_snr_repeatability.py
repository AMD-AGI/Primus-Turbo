"""Test SNR repeatability on the 4 FAIL cases.
If SNR varies across runs (>0.5 dB): non-deterministic — real bug (race or undef).
If SNR is stable: deterministic numerics ceiling (FP8 accum ordering in bn128 down path).

For each FAIL case:
- run 5 times with fresh tensors (seed=42 reset per iter for input determinism, but kernel internal nondeterminism if any will surface as drift)
- run 5 times with SAME tensors (seed=42 set once, run kernel 5x): kernel-side determinism check
"""
import os, sys
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
import primus_turbo  # registers ops
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.core.backend import GlobalBackendManager, BackendType
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref
from tests.pytorch.test_utils import compute_snr

DEV = "cuda"
cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
GlobalBackendManager.set_auto_tune(False)

# 4 FAIL cases from round-1 baseline.
CASES = [
    ("dsv3-down-B16-M2048", 16, 2048, 7168, 2048),
    ("dsv3-down-B16-M4096", 16, 4096, 7168, 2048),
    ("qwen-down-B16-M4096", 16, 4096, 4096, 1536),
    ("qwen-down-B16-M8192", 16, 8192, 4096, 1536),
]

print("Mode A: fresh tensors each run (seed=42 reset every iter)")
print(f"{'label':<22} {'snr_1':>8} {'snr_2':>8} {'snr_3':>8} {'snr_4':>8} {'snr_5':>8} {'delta':>8}")
for label, B, M_g, N, K in CASES:
    snrs = []
    for i in range(5):
        torch.manual_seed(42)
        a = (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        b = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
        a.requires_grad_(True); b.requires_grad_(True)
        group_lens = torch.full((B,), M_g, dtype=torch.int64, device=DEV)
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)
        out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
        snrs.append(compute_snr(out_ref, out))
    print(f"{label:<22} " + "  ".join(f"{s:>7.2f}" for s in snrs) + f"   {max(snrs)-min(snrs):>6.3f}")

print("\nMode B: same tensors, kernel run 5x (kernel-side determinism)")
print(f"{'label':<22} {'snr_1':>8} {'snr_2':>8} {'snr_3':>8} {'snr_4':>8} {'snr_5':>8} {'delta':>8}")
for label, B, M_g, N, K in CASES:
    torch.manual_seed(42)
    a = (torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    b = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    a.requires_grad_(True); b.requires_grad_(True)
    group_lens = torch.full((B,), M_g, dtype=torch.int64, device=DEV)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    snrs = []
    for i in range(5):
        out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
        snrs.append(compute_snr(out_ref, out))
    print(f"{label:<22} " + "  ".join(f"{s:>7.2f}" for s in snrs) + f"   {max(snrs)-min(snrs):>6.3f}")
