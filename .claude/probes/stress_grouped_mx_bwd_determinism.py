"""Determinism stress for grouped MXFP8 forward + backward.

For each fixed input, repeatedly runs grouped_gemm_fp8 forward and backward and
compares out/dA/dB against the first run. This catches whole-path nondeterminism
that a one-shot SNR smoke test can miss.
"""

import os
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


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
N_ITERS = int(os.environ.get("N_ITERS", "200"))
BAD_THRESH = float(os.environ.get("BAD_THRESH", "1.0"))


def run_once(a_base, b_base, group_lens, grad_out, config):
    a = a_base.detach().clone().requires_grad_(True)
    b = b_base.detach().clone().requires_grad_(True)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    out.backward(grad_out)
    torch.cuda.synchronize()
    return out.detach().clone(), a.grad.detach().clone(), b.grad.detach().clone()


def run(G, M_per, N, K, fmt=Format.E4M3, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total_m = G * M_per
    group_lens = torch.full((G,), M_per, dtype=torch.int64, device=DEVICE)
    a_base = torch.randn(total_m, K, dtype=DTYPE, device=DEVICE)
    b_base = torch.randn(G, N, K, dtype=DTYPE, device=DEVICE)
    grad_out = torch.randn(total_m, N, dtype=DTYPE, device=DEVICE)
    config = Float8QuantConfig(
        format=fmt,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )

    ref_out, ref_da, ref_db = run_once(a_base, b_base, group_lens, grad_out, config)
    bad = 0
    print(f"G={G} M_per={M_per} N={N} K={K} fmt={fmt.name} seed={seed} iters={N_ITERS}")
    for i in range(N_ITERS):
        out, da, db = run_once(a_base, b_base, group_lens, grad_out, config)
        out_diff = (ref_out.float() - out.float()).abs().max().item()
        da_diff = (ref_da.float() - da.float()).abs().max().item()
        db_diff = (ref_db.float() - db.float()).abs().max().item()
        max_diff = max(out_diff, da_diff, db_diff)
        if max_diff > BAD_THRESH:
            bad += 1
            print(
                f"BAD iter={i} max={max_diff:.3f} "
                f"out={out_diff:.3f} dA={da_diff:.3f} dB={db_diff:.3f}"
            )
            if bad >= int(os.environ.get("STOP_AFTER_BAD", "20")):
                break

    print(f"  result: {bad}/{N_ITERS} BAD")
    return bad


configs = [
    (4, 256, 512, 512, Format.E4M3),
    (4, 1024, 2048, 2048, Format.E4M3),
    (4, 2048, 8192, 2048, Format.E4M3),
    (8, 2048, 8192, 2048, Format.E4M3),
    (4, 1024, 2048, 2048, Format.E5M2),
]

total = 0
for cfg in configs:
    total += run(*cfg)

print(f"\nTOTAL: {total} BAD")
sys.exit(1 if total else 0)
