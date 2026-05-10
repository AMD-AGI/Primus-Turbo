#!/usr/bin/env python3
"""R45 drift re-audit on GateUP-B4-M2048 fwd + dgrad-via-H4.

R44 forward-pointed to a (gm, xcds) drift re-audit of the GateUP-B4-M2048
RCR rule (config.py:1418-1482, currently gm=1, xcd=4, slots=0 (default
NUM_CUS=256), chunk_size=0 (default 64)). The rule was tuned in R23
(comment block at config.py:1418-1470) under the metric-aligned per-iter-
sync regime; many kernel rebuilds and (slots/cs) wirings have landed
since (R9 num_slots, R14 chunk_size, R26-R28 var-K barrier audit, R34-dm
interleave, etc.). Drift re-audit checks whether the (gm=1, xcd=4)
optimum still holds.

Hypothesis: same as R44 — kernel codegen drift since R23 may have
shifted the (gm, xcds) optimum. R23-current sweep ranking was:
    (1, 4) > (1, 2) > (3, 4) > (2, 8) > (4, 4) > (2, 4) > {(1,8), (1,1), (1,16)}

This probe sweeps {(gm, xcd)} on GateUP-B4-M2048 fwd + dgrad-via-H4 with
slots=0 + cs=0 held FIXED (rule has no slots/cs override; defaults
preserved). Cells mirror R23's sweep + R44's diversity:

Cells (gm, xcd):
  (1, 4)*  — current rule (R23 winner; baseline)
  (1, 2)   — R23's #2 (-5.11 TF in R23-current); test if drift flipped
  (1, 8)   — R23 marginal (-41.35 TF); defensive control (xcd=8)
  (3, 4)   — R23 #3 (-8.25 TF); test gm=3 drift
  (4, 4)   — R23 #5 (-11.37 TF); test gm=4 (default)
  (1, 1)   — R23 worst (-43.02 TF); xcd=1 control
  (2, 4)   — R23 #6 (-12.25 TF); R68 default; test if drift flipped

Methodology: 5 seeds × 2000-iter p20 per cell, kernel-only timing via
direct grouped_rcr_dscale call. SHIP if best cell wins by >= 1.0% over
baseline on BOTH fwd and dgrad-via-H4 medians AND signal > noise
(med-gap > spread).

Outcome (per R44 BACKUP plan): if no cell wins >= 1.0% on both sections
(i.e. baseline (1, 4) still optimum), this round's FALSIFICATION pairs
with R44's Down-B4-M2048 FALSIFICATION to establish two consecutive
empirical drift FALSIFICATIONS, justifying the R10-R45 audit summary
marking the dispatcher levers feature-complete on the structural ~700
plateau.
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

# Held FIXED per the GateUP-B4-M2048 rule defaults: no slots/cs override
# (kernel uses NUM_CUS=256 and chunk_size=64).
SLOTS = 0
CHUNK = 0


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


def _patch_hk_dscale(gm, xcds):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale

    def wrapped(*args, **kwargs):
        # The Python dispatcher passes group_m as positional arg 7
        # (a, b, c, sa, sb, offs, group_m, ...). Strip it and re-pass as kwarg.
        if len(args) >= 7:
            args = args[:6]
        kwargs["group_m"] = gm
        kwargs["num_xcds"] = xcds
        kwargs["num_slots"] = SLOTS
        kwargs["chunk_size"] = CHUNK
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore_hk_dscale(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_fwd(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_impl(
            a_fp8, b_fp8, a_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call)
    return flops / (ms * 1e9), ms


def time_dgrad(B, M, N, K):
    # dgrad-via-H4: trans_b=False -> reroutes through fp8_transpose_3d -> RCR
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


def run(label, time_fn, B, M, N, K, cells, baseline_cell, seeds=(42, 137, 2024, 99, 1234)):
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'cell':>10}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'spread%':>7}  {'TFLOPS':>8}  {'delta%':>7}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for cell in cells:
        gm, xcd = cell
        orig = _patch_hk_dscale(gm, xcd)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_fn(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cell] = (med, min(seed_meds), max(seed_meds))
        _restore_hk_dscale(orig)

    base_ms = results[baseline_cell][0]
    for cell in cells:
        med, lo, hi = results[cell]
        tflops = flops / (med * 1e9)
        spread_pp = (hi - lo) / med * 100
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *base" if cell == baseline_cell else ""
        print(f"  {str(cell):>10}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {spread_pp:>6.2f}%  {tflops:>8.1f}  {delta_pp:+6.2f}%{marker}")
    best_cell = min(cells, key=lambda c: results[c][0])
    best_med = results[best_cell][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: cell={best_cell}  ({lift:+.2f}% over baseline {baseline_cell})")
    return results, best_cell, lift


if __name__ == "__main__":
    print(f"[probe] R45 GateUP-B4-M2048 (gm, xcds) drift re-audit")
    print(f"[probe] holding slots={SLOTS}, chunk_size={CHUNK} FIXED (rule defaults)")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    cells = [(1, 4), (1, 2), (1, 8), (3, 4), (4, 4), (1, 1), (2, 4)]
    baseline = (1, 4)

    t0 = time.monotonic()
    res_fwd, best_fwd, lift_fwd = run(
        "GateUP-B4-M2048 fwd", time_fwd,
        B=4, M=2048, N=5760, K=2880,
        cells=cells, baseline_cell=baseline,
    )
    res_dg, best_dg, lift_dg = run(
        "GateUP-B4-M2048 dgrad-via-H4", time_dgrad,
        B=4, M=2048, N=5760, K=2880,
        cells=cells, baseline_cell=baseline,
    )
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")

    print(f"\n[probe] SUMMARY:")
    print(f"  fwd   baseline (1,4): {2.0 * 4 * 2048 * 5760 * 2880 / (res_fwd[(1, 4)][0] * 1e9):.1f} T")
    print(f"  fwd   best {best_fwd}: {2.0 * 4 * 2048 * 5760 * 2880 / (res_fwd[best_fwd][0] * 1e9):.1f} T  ({lift_fwd:+.2f}%)")
    print(f"  dgrad baseline (1,4): {2.0 * 4 * 2048 * 5760 * 2880 / (res_dg[(1, 4)][0] * 1e9):.1f} T")
    print(f"  dgrad best {best_dg}: {2.0 * 4 * 2048 * 5760 * 2880 / (res_dg[best_dg][0] * 1e9):.1f} T  ({lift_dg:+.2f}%)")

    # SHIP gate: same winner on both, lift >= 1.0% on both, signal > spread on both
    if best_fwd == best_dg and best_fwd != baseline:
        gm_w, xcd_w = best_fwd
        fwd_med, fwd_lo, fwd_hi = res_fwd[best_fwd]
        dg_med, dg_lo, dg_hi = res_dg[best_dg]
        fwd_spread_pp = (fwd_hi - fwd_lo) / fwd_med * 100
        dg_spread_pp = (dg_hi - dg_lo) / dg_med * 100
        ship = (lift_fwd >= 1.0 and lift_dg >= 1.0
                and lift_fwd > fwd_spread_pp and lift_dg > dg_spread_pp)
        print(f"\n[probe] SHIP gate: best={best_fwd} on both sections")
        print(f"        lift_fwd={lift_fwd:.2f}% (spread {fwd_spread_pp:.2f}%)")
        print(f"        lift_dg ={lift_dg:.2f}% (spread {dg_spread_pp:.2f}%)")
        print(f"        DECISION: {'SHIP' if ship else 'FALSIFIED-noise'}")
    else:
        print(f"\n[probe] no consistent winner across sections; baseline likely still best")
        print(f"        DECISION: FALSIFIED")
