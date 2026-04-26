"""Smoke test for GroupedGemmFP8MXFunc (Phase 1: forward + dgrad + bf16 wgrad fallback).

Compares forward output, grad_a, grad_b against eager bf16 reference.
"""

import sys

import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def grouped_ref(a, b, group_lens):
    """Eager bf16 grouped GEMM: a (total_M, K), b (G, N, K) trans_b=True → out (total_M, N)."""
    out = torch.empty(a.size(0), b.size(1), dtype=a.dtype, device=a.device)
    cum = 0
    for g in range(b.size(0)):
        Mg = int(group_lens[g].item())
        out[cum : cum + Mg] = a[cum : cum + Mg] @ b[g].T
        cum += Mg
    return out


def snr_db(ref, out):
    ref = ref.float()
    out = out.float()
    sig = (ref**2).mean()
    err = ((ref - out) ** 2).mean()
    if err.item() == 0:
        return float("inf")
    return 10.0 * torch.log10(sig / err).item()


def run(G, M_per, N, K, format=Format.E4M3):
    print(f"\n=== G={G}, M_per={M_per}, N={N}, K={K}, fmt={format} ===")
    total_M = G * M_per
    group_lens = torch.tensor([M_per] * G, dtype=torch.int64, device=DEVICE)

    a = torch.randn(total_M, K, dtype=DTYPE, device=DEVICE, requires_grad=True)
    b = torch.randn(G, N, K, dtype=DTYPE, device=DEVICE, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # Reference (bf16)
    out_ref = grouped_ref(a_ref, b_ref, group_lens)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()

    # Turbo MXFP8
    config = Float8QuantConfig(
        format=format,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()

    # Shape check
    assert out.shape == out_ref.shape, f"out shape {out.shape} vs ref {out_ref.shape}"
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    # SNR
    out_snr = snr_db(out_ref, out)
    da_snr = snr_db(a_ref.grad, a.grad)
    db_snr = snr_db(b_ref.grad, b.grad)
    print(f"  out SNR  : {out_snr:7.2f} dB")
    print(f"  dA  SNR  : {da_snr:7.2f} dB")
    print(f"  dB  SNR  : {db_snr:7.2f} dB  (Phase-1 bf16 fallback — should be ~Inf)")

    threshold = 25.0 if format == Format.E4M3 else 20.0
    ok = (out_snr > threshold) and (da_snr > threshold) and (db_snr > threshold)
    if not ok:
        print(f"  ✗ FAIL (threshold={threshold} dB)")
        return False
    print("  ✓ PASS")
    return True


configs = [
    # (G, M_per, N, K, format)
    (2, 256, 512, 384, Format.E4M3),
    (4, 256, 512, 512, Format.E4M3),
    (4, 1024, 2048, 2048, Format.E4M3),
    (4, 2048, 8192, 2048, Format.E4M3),
    (8, 2048, 8192, 2048, Format.E4M3),
    # E5M2 (lower precision, 20 dB threshold)
    (2, 256, 512, 384, Format.E5M2),
    (4, 1024, 2048, 2048, Format.E5M2),
    (4, 2048, 8192, 2048, Format.E5M2),
]

failed = []
for cfg in configs:
    if not run(*cfg):
        failed.append(cfg)

print()
if failed:
    print(f"FAILED configs: {failed}")
    sys.exit(1)
print("All passed.")
