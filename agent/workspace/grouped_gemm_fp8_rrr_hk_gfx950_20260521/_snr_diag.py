"""Diagnose SNR FAIL on dsv3-up-B16-M8192 using PT high-level API + PT ref.
If SNR < 25 dB here too: real numerics issue (kernel or quant pipeline).
If SNR >= 25 dB here: my quick_test_bench correctness path has a bug.
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

torch.manual_seed(42)
DEV = "cuda"

# dsv3-up-B16-M8192 — the FAIL case
CASES = [
    ("dsv3-up-B16-M8192", 16,  8192, 4096, 7168),  # the FAIL one
    ("dsv3-up-B4-M2048",   4,  2048, 4096, 7168),  # control: bn128 path that PASSed
    ("qwen-down-B16-M2048", 16, 2048, 4096, 1536), # control: smaller K
]

GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
GlobalBackendManager.set_auto_tune(False)

cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)

for label, B, M_g, N, K in CASES:
    print(f"\n=== {label}: B={B} M_g={M_g} N={N} K={K} ===")
    a = torch.randn((B*M_g, K), dtype=torch.bfloat16, device=DEV, requires_grad=True) * 0.05
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV, requires_grad=True) * 0.05
    a = a.contiguous().detach().requires_grad_(True)
    b = b.contiguous().detach().requires_grad_(True)
    group_lens = torch.full((B,), M_g, dtype=torch.int64, device=DEV)

    out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    snr = compute_snr(out_ref, out)
    print(f"  PT-API + PT-ref Out-SNR: {snr:.2f} dB  ({'PASS' if snr >= 25 else 'FAIL'})")

    # Compare against bf16 unscaled reference (no fp8 in the loop) as sanity
    snr_ref_only = compute_snr(out_ref, out_ref)
    print(f"  (sanity: ref vs ref = {snr_ref_only:.2f} dB — should be ~inf)")
