#!/usr/bin/env python3
"""R20 — wide (gm, xcds) sweep on Down-B32-M2048 wgrad.

Down-B32-M2048 wgrad ratio = 0.597 (1673 T) — second-worst wgrad cell
after Down-B4-M2048 wgrad. R1 (this run) tested only ``{(4, 4), (8, 4)}``
and shipped (8, 4) for ``m_total == 65536 AND k==n==2880``. The xcds=2
column was never tested on this shape, despite winning on the B4 family
sibling (R11 (1, 2) for Down-B4-M2048 wgrad with the SAME per-group
[N=2880, K=2880] tile geometry).

Per-group output for Down-B32-M2048 wgrad: [N_fwd=2880, K_fwd=2880] ⇒
tiles_n=11, tiles_k=11 = 121 tile-steps per group × 32 groups = 3872
tile-steps over NUM_CUS=256 persistent slots ≈ 15 wave-steps. Compare
B4 family: 4 groups × 121 = 484 tile-steps ≈ 2 wave-steps. The 7.5×
deeper grid changes the L2 / chiplet trade-off — R10/R11 specifically
flagged this as the wave-step amortisation cutoff. xcds=2 might still
win on B32, or might lose to xcds=4 — needs a probe.

Method: 250 iters × 7 trials × 3 seeds, p20 per seed → median across
seeds. Direct kernel call to ``grouped_variable_k_crr_dscale``.
"""
import os, sys, statistics, time
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
from primus_turbo.pytorch.kernels import hipkitten as hipkitten
from primus_turbo.pytorch.ops.quantization import quantize_fp8

_FP8_DTYPE = torch.float8_e4m3fn  # MI355X OCP-FP8
_GRAN = ScalingGranularity.TENSORWISE


def _bench(fn, warmup=50, iters=250, trials=7):
    """Time fn() in iters × trials. Return median p20 across trials."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    p20s = []
    for _ in range(trials):
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
        p20s.append(times[len(times) // 5])
    p20s.sort()
    return p20s[len(p20s) // 2]


def main():
    print(f"[probe] R20 — Down-B32-M2048 wgrad (gm, xcds) wide sweep")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    print()

    # Down-B32-M4096 wgrad shape: per-group [N_fwd=2880, K_fwd=2880],
    # B=32 groups, M_per_group=4096 ⇒ m_total=131072.
    B, M, N, K = 32, 4096, 2880, 2880
    flops = 2.0 * B * M * N * K  # FP8 grouped GEMM FLOPs
    print(f"  shape: B={B} M={M} N={N} K={K}  m_total={B*M}")
    print(f"  FLOPs/call = {flops/1e9:.0f} GF")
    print()

    hk = hipkitten.load_fp8()

    rules = [
        (4, 4, "R30 baseline (current)"),
        (8, 4, "R39 universal"),
        (1, 4, ""),
        (2, 4, ""),
        (16, 4, ""),
        (32, 4, ""),
        (1, 2, ""),
        (2, 2, ""),
        (4, 2, ""),
        (8, 2, ""),
        (16, 2, ""),
        (12, 4, "R25 DSV3 winner"),
    ]

    seeds = [42, 137, 2024]
    all_meds = {}

    for seed in seeds:
        print(f"  --- seed {seed} ---")
        torch.manual_seed(seed)

        x = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
        grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
        x_fp8, x_s = quantize_fp8(x, _FP8_DTYPE, _GRAN)
        g_fp8, g_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)
        out = torch.empty((B, N, K), dtype=torch.bfloat16, device="cuda")

        g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
        g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
        g_offs[1:] = torch.cumsum(g_lens, dim=0)

        for gm, xc, note in rules:
            def call():
                hk.grouped_variable_k_crr_dscale(
                    g_fp8, x_fp8, out, g_s, x_s, g_offs,
                    group_m=gm, num_xcds=xc, num_slots=0,
                )
            ms = _bench(call, warmup=30, iters=250, trials=5)
            tflops = flops / (ms * 1e9)
            cell = (gm, xc)
            all_meds.setdefault(cell, []).append(ms)
            tag = f"  ← {note}" if note else ""
            print(f"  cell ({gm:2d}, {xc}) ms_med={ms:.4f}  TFLOPS={tflops:.1f}{tag}")
        print()

    print(f"  --- 3-seed summary (median ms / median TFLOPS / Δ vs (8,4)) ---")
    base_med = statistics.median(all_meds[(4, 4)])
    rows = []
    for (gm, xc), mss in all_meds.items():
        med = statistics.median(mss)
        tflops = flops / (med * 1e9)
        delta = (base_med - med) / base_med * 100  # +% means faster
        spread_pp = (max(mss) - min(mss)) / med * 100
        rows.append((gm, xc, med, tflops, delta, spread_pp))
    rows.sort(key=lambda r: -r[4])
    for gm, xc, med, tflops, delta, spread in rows:
        marker = "★" if (gm, xc) == (4, 4) else " "
        print(f"  {marker} ({gm:2d}, {xc})   ms_med={med:.4f}  TFLOPS={tflops:.1f}   Δ={delta:+.2f}%  spread={spread:.2f}%")


if __name__ == "__main__":
    main()
