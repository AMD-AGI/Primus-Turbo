"""Locate intermittent drift in turbo grouped MXFP8 wgrad.

This keeps one quantized input fixed, compares repeated wgrad launches against
the first launch, and prints the group/tile location of any drift.
"""

from collections import Counter
import os
import sys

import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.ops.quantization import MX_BLOCK_SIZE, quantize_fp8_with_trans


DEVICE = "cuda:0"
N_ITERS = int(os.environ.get("N_ITERS", "10000"))
BAD_THRESH = float(os.environ.get("BAD_THRESH", "1.0"))


def run(G: int, M_per: int, N: int, K: int, seed: int) -> int:
    torch.manual_seed(seed)
    total_m = G * M_per
    group_lens = torch.full((G,), M_per, dtype=torch.int64, device=DEVICE)
    group_offs = grouped_gemm_compute_offs(group_lens)
    a = torch.randn(total_m, K, dtype=torch.bfloat16, device=DEVICE)
    grad_out = torch.randn(total_m, N, dtype=torch.bfloat16, device=DEVICE)

    _, _, a_t_fp8, a_t_scale = quantize_fp8_with_trans(
        a, torch.float8_e4m3fn, ScalingGranularity.MX_BLOCKWISE, block_size=MX_BLOCK_SIZE
    )
    _, _, g_t_fp8, g_t_scale = quantize_fp8_with_trans(
        grad_out, torch.float8_e4m3fn, ScalingGranularity.MX_BLOCKWISE, block_size=MX_BLOCK_SIZE
    )

    ref = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_variable_k_fp8(
        g_t_fp8, g_t_scale, a_t_fp8, a_t_scale, group_lens, group_offs,
        torch.bfloat16, "MX_BLOCKWISE",
    )
    torch.cuda.synchronize()

    bad = 0
    group_hits: Counter[int] = Counter()
    tile_hits: Counter[tuple[int, int, int]] = Counter()
    print(f"G={G} M_per={M_per} N={N} K={K} seed={seed} iters={N_ITERS}")
    for i in range(N_ITERS):
        out = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_variable_k_fp8(
            g_t_fp8, g_t_scale, a_t_fp8, a_t_scale, group_lens, group_offs,
            torch.bfloat16, "MX_BLOCKWISE",
        )
        torch.cuda.synchronize()
        diff = (ref.float() - out.float()).abs()
        max_diff = diff.max().item()
        if max_diff <= BAD_THRESH:
            continue

        bad += 1
        flat = int(diff.view(-1).argmax().item())
        g = flat // (N * K)
        rem = flat % (N * K)
        n_idx = rem // K
        k_idx = rem % K
        tile = (g, n_idx // 256, k_idx // 256)
        group_hits[g] += 1
        tile_hits[tile] += 1
        print(
            f"BAD iter={i} max={max_diff:.3f} "
            f"g={g} n={n_idx} k={k_idx} tile={tile}"
        )
        if bad >= int(os.environ.get("STOP_AFTER_BAD", "20")):
            break

    print(f"bad={bad}/{N_ITERS}")
    print(f"group_hits={dict(group_hits)}")
    print(f"tile_hits={dict(tile_hits)}")
    return bad


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    if args:
        if len(args) != 5:
            raise SystemExit("usage: probe_wgrad_race_detail.py G M_per N K seed")
        total_bad = run(*args)
    else:
        total_bad = 0
        for cfg in [
            (4, 1024, 2048, 2048, 42),
            (8, 2048, 8192, 2048, 42),
        ]:
            total_bad += run(*cfg)
    raise SystemExit(1 if total_bad else 0)
