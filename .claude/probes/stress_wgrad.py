"""1000-iter stress for the new turbo grouped MXFP8 wgrad kernel."""
import sys
import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import grouped_gemm_compute_offs
from primus_turbo.pytorch.ops.quantization import MX_BLOCK_SIZE, quantize_fp8_with_trans
from primus_turbo.pytorch.core.low_precision import ScalingGranularity

torch.manual_seed(42)
DEVICE = "cuda:0"
N_ITERS = 1000


def snr(ref, out):
    ref = ref.float(); out = out.float()
    sig = (ref**2).mean(); err = ((ref - out) ** 2).mean()
    return float("inf") if err.item() == 0 else 10.0 * torch.log10(sig / err).item()


def run(G, M_per, N, K):
    total_M = G * M_per
    group_lens = torch.tensor([M_per] * G, dtype=torch.int64, device=DEVICE)
    group_offs = grouped_gemm_compute_offs(group_lens)
    a = torch.randn(total_M, K, dtype=torch.bfloat16, device=DEVICE)
    grad_out = torch.randn(total_M, N, dtype=torch.bfloat16, device=DEVICE)

    # Quantize once (these are stable across iters)
    _, _, a_t_fp8, a_t_scale = quantize_fp8_with_trans(
        a, torch.float8_e4m3fn, ScalingGranularity.MX_BLOCKWISE, block_size=MX_BLOCK_SIZE
    )
    _, _, g_t_fp8, g_t_scale = quantize_fp8_with_trans(
        grad_out, torch.float8_e4m3fn, ScalingGranularity.MX_BLOCKWISE, block_size=MX_BLOCK_SIZE
    )

    # Reference: run kernel once, take its result as reference (since the same call
    # should be deterministic; any race would manifest as drift from this baseline).
    ref_db = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_variable_k_fp8(
        g_t_fp8, g_t_scale, a_t_fp8, a_t_scale, group_lens, group_offs,
        torch.bfloat16, "MX_BLOCKWISE",
    )
    torch.cuda.synchronize()

    bad = 0
    for _ in range(N_ITERS):
        db = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_variable_k_fp8(
            g_t_fp8, g_t_scale, a_t_fp8, a_t_scale, group_lens, group_offs,
            torch.bfloat16, "MX_BLOCKWISE",
        )
        torch.cuda.synchronize()
        d = (ref_db.float() - db.float()).abs()
        if d.max().item() > 1.0:
            bad += 1
    print(f"  G={G:1d} M_per={M_per:5d} N={N:5d} K={K:5d}: {bad:4d}/{N_ITERS} BAD")
    return bad


total = 0
for cfg in [
    (4, 1024, 2048, 2048),
    (4, 2048, 8192, 2048),
    (8, 2048, 8192, 2048),
    (4, 1024, 4096, 2048),
]:
    total += run(*cfg)

print(f"\nTOTAL: {total} BAD / {len([1])*1000} samples per config")
sys.exit(1 if total else 0)
