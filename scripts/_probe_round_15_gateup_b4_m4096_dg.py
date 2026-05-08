#!/usr/bin/env python3
"""R15: GateUP-B4-M4096 dgrad-via-H4 num_slots × chunk_size probe.

Cell: tiles_n=11 + k=5760 + tiles_m=16 + m_total=16384
R8 rule: (gm=1, xcds=4), no num_slots override → slots=NUM_CUS=256

Tile-steps = 16*11*4 / 256 = 2.75 ws/CU. Borderline (R3 threshold ≈ 2.5).
R10 set num_slots=200 for the SIBLING cell (M=2048, tiles_m=8) at
1.5 ws/CU — the M=4096 cell hasn't been audited.

Plan:
1. Sweep num_slots ∈ {0, 192, 200, 208, 220, 240, 256} at default cs=64
2. If any slots wins, also test chunk_size at the new (xcds=4, slots=N) cell
   to find the clean-partition optimum (slots/xcds).
"""
import os
import sys
import statistics
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module
import _metric_hk_ratio as hk_ratio

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _bench_p20(fn, warmup=20, iters=2000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


def _patch_rcr(num_slots=None, chunk_size=None):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale

    def wrapped(*args, **kwargs):
        if num_slots is not None:
            kwargs["num_slots"] = num_slots
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore_rcr(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_dgrad(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_impl(
            g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call)
    return flops / (ms * 1e9), ms


def run_grid(label, B, M, N, K, cells, seeds=(42, 137, 2024)):
    """cells: list of (slots, chunk_size) tuples (None = default)."""
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'slots':>5} {'cs':>3}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for (sl, cs) in cells:
        orig = _patch_rcr(num_slots=sl, chunk_size=cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_dgrad(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[(sl, cs)] = (med, min(seed_meds), max(seed_meds))
        _restore_rcr(orig)

    base_key = (None, None)  # default
    base_ms = results[base_key][0]
    sorted_cells = sorted(cells, key=lambda c: results[c][0])
    for (sl, cs) in sorted_cells:
        med, lo, hi = results[(sl, cs)]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        sl_str = "def" if sl is None else str(sl)
        cs_str = "def" if cs is None else str(cs)
        marker = " *base" if (sl, cs) == base_key else ""
        print(f"  {sl_str:>5} {cs_str:>3}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    best = sorted_cells[0]
    best_med = results[best][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: cell={best}  ({lift:+.2f}% over default)")
    return results, best, lift


if __name__ == "__main__":
    print(f"[probe] R15 GateUP-B4-M4096 dgrad-via-H4 num_slots × chunk_size probe")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    # Phase 1: sweep slots at cs=default (64)
    cells_phase1 = [
        (None, None),  # default (slots=256, cs=64) → R8 baseline
        (192, None),
        (200, None),
        (208, None),
        (220, None),
        (240, None),
    ]
    # Phase 2: at slots=200 (R10 sibling lever value), test chunk_size
    # cells_phase2 = ... will be added if phase 1 shows a slot win

    # All cells in one sweep
    cells = [
        (None, None),       # baseline R8 (slots=256, cs=64)
        (192, None),
        (192, 96),  # block=4*96=384, limit=0 → swizzle no-op
        (192, 48),  # block=192=slots, all chunked
        (200, None),
        (200, 50),  # block=200=slots, all chunked
        (200, 100), # block=400, limit=0 → no-op
        (208, None),
        (208, 52),  # block=208=slots, all chunked
        (220, None),
        (240, None),
        (256, 32),  # 2-chunk interleave at default slots
        (256, 16),  # 4-chunk interleave
    ]

    t0 = time.monotonic()
    res, best, lift = run_grid(
        "GateUP-B4-M4096 dgrad-via-H4",
        B=4, M=4096, N=5760, K=2880,
        cells=cells,
    )
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")
