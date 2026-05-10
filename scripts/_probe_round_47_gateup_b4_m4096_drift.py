#!/usr/bin/env python3
"""R47 drift audit on GateUP-B4-M4096 dgrad-via-H4 (PRIMARY) + fwd (diversity).

Per R46 forward pointer (analysis/_notes/round-46-r45-PRIMARY-OBSOLETE-...md
Section "R47 forward pointer"): the GateUP-B4-M4096 dgrad-via-H4 RCR rule
at config.py:2603-2617 (`tiles_m == 16` branch, currently
`group_m=1, num_xcds=4, fuse_ktail_off=0`, no slots/cs override) has NEVER
been (gm, xcds) swept under fixed production levers. This is the LARGEST
B=4 cell (m_total=16384, 1.6× the M=2048 sibling). R15-4 audit found "NO
chunk_size win there"; only chunk_size was swept previously.

Methodology: mirror R45 (5 seeds × 2000-iter p20). The rule has no
slots/cs override, so SLOTS=0/CHUNK=0 in the kwarg-patch maps to kernel
defaults (NUM_CUS=256, chunk=64) — production levers FIXED.

Cells per R46 spec:
  (1, 4)*current    R23/R43-class baseline (current rule)
  (1, 8)   xcds drift control (default)
  (1, 2)   xcds drift control (low)
  (1, 1)   xcds=1 floor
  (2, 4)   gm drift (R68 default)
  (4, 4)   gm drift (R23 default)
  (8, 4)   gm drift (large)
  (16, 4)  gm drift (very large; tiles_m=16 means 16 m-tiles/group)

Outcome (per R46 R47 plan): if no cell wins >= 1.0% on dgrad-via-H4 with
wmin_beats_lmax, R44-R47 chain forms 4 consecutive (gm, xcds) drift
audits on B=4 RCR with no win → definitive closure of (gm, xcds) lever
class on B=4. SKILL.md NEW DIRECTION D (SALU coord-decode) becomes the
recommended next-task transition.
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

# GateUP-B4-M4096 dgrad-via-H4 rule at config.py:2603-2617 has no
# slots/cs override; kernel defaults apply (NUM_CUS=256, chunk=64).
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
    print(f"  {'cell':>10}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'spread%':>7}  {'TFLOPS':>8}  {'delta%':>7}  {'wmin>lmax':>10}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    seed_lists = {}
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
        seed_lists[cell] = sorted(seed_meds)
        _restore_hk_dscale(orig)

    base_ms = results[baseline_cell][0]
    base_max = max(seed_lists[baseline_cell])
    for cell in cells:
        med, lo, hi = results[cell]
        tflops = flops / (med * 1e9)
        spread_pp = (hi - lo) / med * 100
        delta_pp = (base_ms - med) / base_ms * 100
        cell_min = min(seed_lists[cell])
        wmin_beats_lmax = "YES" if cell_min < base_max and cell != baseline_cell else ("base" if cell == baseline_cell else "no")
        marker = " *base" if cell == baseline_cell else ""
        print(f"  {str(cell):>10}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {spread_pp:>6.2f}%  {tflops:>8.1f}  {delta_pp:+6.2f}%  {wmin_beats_lmax:>10}{marker}")
    best_cell = min(cells, key=lambda c: results[c][0])
    best_med = results[best_cell][0]
    lift = (base_ms - best_med) / base_ms * 100
    # wmin_beats_lmax: every seed of best beats every seed of baseline
    best_max = max(seed_lists[best_cell])
    base_min = min(seed_lists[baseline_cell])
    wmin_strict = best_max < base_min if best_cell != baseline_cell else False
    print(f"  BEST: cell={best_cell}  ({lift:+.2f}% over baseline {baseline_cell})  wmin_strict={wmin_strict}")
    return results, seed_lists, best_cell, lift, wmin_strict


if __name__ == "__main__":
    print(f"[probe] R47 GateUP-B4-M4096 (gm, xcds) drift audit (dgrad-via-H4 PRIMARY)")
    print(f"[probe] holding slots={SLOTS}, chunk_size={CHUNK} FIXED (rule defaults = NUM_CUS=256, cs=64)")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    cells = [(1, 4), (1, 8), (1, 2), (1, 1), (2, 4), (4, 4), (8, 4), (16, 4)]
    baseline = (1, 4)

    t0 = time.monotonic()
    res_fwd, sl_fwd, best_fwd, lift_fwd, wmin_fwd = run(
        "GateUP-B4-M4096 fwd (diversity)", time_fwd,
        B=4, M=4096, N=5760, K=2880,
        cells=cells, baseline_cell=baseline,
    )
    res_dg, sl_dg, best_dg, lift_dg, wmin_dg = run(
        "GateUP-B4-M4096 dgrad-via-H4 (PRIMARY)", time_dgrad,
        B=4, M=4096, N=5760, K=2880,
        cells=cells, baseline_cell=baseline,
    )
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")

    flops = 2.0 * 4 * 4096 * 5760 * 2880
    print(f"\n[probe] SUMMARY:")
    print(f"  fwd   baseline (1,4): {flops / (res_fwd[(1, 4)][0] * 1e9):.1f} T")
    print(f"  fwd   best {best_fwd}: {flops / (res_fwd[best_fwd][0] * 1e9):.1f} T  ({lift_fwd:+.2f}%, wmin_strict={wmin_fwd})")
    print(f"  dgrad baseline (1,4): {flops / (res_dg[(1, 4)][0] * 1e9):.1f} T")
    print(f"  dgrad best {best_dg}: {flops / (res_dg[best_dg][0] * 1e9):.1f} T  ({lift_dg:+.2f}%, wmin_strict={wmin_dg})")

    # SHIP gate per R46 plan: PRIMARY is dgrad-via-H4. Lift >= 1.0% AND
    # wmin_beats_lmax (every seed of winner < every seed of baseline).
    # Fwd is diversity-only — log but not gating (fwd uses different rule).
    if best_dg != baseline and lift_dg >= 1.0 and wmin_dg:
        print(f"\n[probe] SHIP gate: dgrad-via-H4 best={best_dg}, lift={lift_dg:.2f}%, wmin_strict=YES")
        print(f"        DECISION: SHIP-CANDIDATE (rerun for confirmation per R43 protocol)")
    elif best_dg != baseline and lift_dg >= 1.0:
        print(f"\n[probe] dgrad-via-H4 best={best_dg}, lift={lift_dg:.2f}% but wmin_strict=NO")
        print(f"        DECISION: NEEDS-RERUN (lift candidate, noise gating)")
    else:
        print(f"\n[probe] dgrad-via-H4 best={best_dg}, lift={lift_dg:+.2f}%")
        print(f"        DECISION: FALSIFIED (rule at local optimum on dgrad-via-H4)")
