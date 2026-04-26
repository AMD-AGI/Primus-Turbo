"""Smoke test for the new turbo grouped MXFP8 wgrad kernel.

Calls turbo_grouped_gemm_variable_k_fp8 directly (without autograd wrapper) and
compares against bf16 reference dB[g] = grad_out[g]^T @ a[g].
"""
import sys
import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import MX_BLOCK_SIZE, quantize_fp8_with_trans
from primus_turbo.pytorch.core.low_precision import ScalingGranularity

torch.manual_seed(42)
DEVICE = "cuda:0"


def snr_db(ref, out):
    ref = ref.float(); out = out.float()
    sig = (ref**2).mean(); err = ((ref - out) ** 2).mean()
    return float("inf") if err.item() == 0 else 10.0 * torch.log10(sig / err).item()


def run(G, M_per, N, K):
    print(f"\n=== G={G}, M_per={M_per}, N={N}, K={K} ===")
    total_M = G * M_per
    group_lens = torch.tensor([M_per] * G, dtype=torch.int64, device=DEVICE)
    group_offs = grouped_gemm_compute_offs(group_lens)

    a = torch.randn(total_M, K, dtype=torch.bfloat16, device=DEVICE)
    grad_out = torch.randn(total_M, N, dtype=torch.bfloat16, device=DEVICE)

    # Reference bf16 dB[g] = grad_out[g]^T @ a[g]
    ref_dB = torch.empty(G, N, K, dtype=torch.bfloat16, device=DEVICE)
    for g in range(G):
        s, e = g * M_per, (g + 1) * M_per
        ref_dB[g] = (grad_out[s:e].T.float() @ a[s:e].float()).to(torch.bfloat16)
    torch.cuda.synchronize()

    # Quantize a → a_t_fp8 (K, total_M), a_t_scale (K, total_M/32)
    _, _, a_t_fp8, a_t_scale = quantize_fp8_with_trans(
        a, torch.float8_e4m3fn, ScalingGranularity.MX_BLOCKWISE, block_size=MX_BLOCK_SIZE
    )
    # Quantize grad_out → g_t_fp8 (N, total_M)
    _, _, g_t_fp8, g_t_scale = quantize_fp8_with_trans(
        grad_out, torch.float8_e4m3fn, ScalingGranularity.MX_BLOCKWISE, block_size=MX_BLOCK_SIZE
    )
    print(f"  a_t_fp8 shape   : {a_t_fp8.shape}, dtype={a_t_fp8.dtype}")
    print(f"  a_t_scale shape : {a_t_scale.shape}, dtype={a_t_scale.dtype}")
    print(f"  g_t_fp8 shape   : {g_t_fp8.shape}, dtype={g_t_fp8.dtype}")
    print(f"  g_t_scale shape : {g_t_scale.shape}, dtype={g_t_scale.dtype}")

    # LHS = grad_out^T (N, total_M), RHS = a^T (K, total_M)
    db = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_variable_k_fp8(
        g_t_fp8, g_t_scale, a_t_fp8, a_t_scale, group_lens, group_offs,
        torch.bfloat16, "MX_BLOCKWISE",
    )
    torch.cuda.synchronize()

    print(f"  ref_dB shape: {ref_dB.shape}, db shape: {db.shape}")
    s = snr_db(ref_dB, db)
    per_g = [snr_db(ref_dB[g], db[g]) for g in range(G)]
    print(f"  total dB SNR: {s:.2f} dB")
    print(f"  per-group   : {[f'{x:.2f}' for x in per_g]}")
    return s > 25


configs = [
    # (G, M_per, N, K) — M_per >= 384 (= 3*128) for kernel's prologue+epilogue pattern
    (2, 512, 512, 384),
    (2, 512, 512, 512),
    (4, 512, 512, 512),
    (4, 1024, 2048, 2048),
    (4, 2048, 8192, 2048),
    (8, 2048, 8192, 2048),
]

failed = []
for cfg in configs:
    if not run(*cfg):
        failed.append(cfg)

print()
if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
print("All passed.")
