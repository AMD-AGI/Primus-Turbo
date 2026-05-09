#!/usr/bin/env python3
"""Round-3 probe — GateUP-B32-M4096 var-K wgrad: ship the R46 lever class.

Context
-------
R46 (commit 12782d4) shipped `(gm=4, xcds=8, slots=256, cs=32)` for
Down-B32-M4096 var-K wgrad after the round_1 fleet probe found +0.77%
on a 7-seed wmin>lmax tight verify against an OLD (gm=4, xcds=4)
baseline. The R46 commit message explicitly notes:

    "Same xcds=8 lever class as the GateUP-B32-M4096 round_0 +1.17%
     finding; the lever is M=4096-specific (M=2048 sibling tests on
     Down/GateUP both LOSS at same cell, so it is gated to `else`
     branch only)."

i.e. the GateUP-B32-M4096 twin lever was identified in the round_0
fleet probe but NEVER SHIPPED. The round_1 tight-verify in
`tuning_results/round_1/gpu0_result.json` shows:

    GateUP-B32-M4096 wgrad var-K
        (4, 4)cur     2195.1 T   baseline (wmin/lmax=1.9751/1.9904)
        (4, 8, 256, 32) 2212.1 T   +0.77 %  wmin>lmax=True (cleanest sig)
        (2, 8, 256, 32) 2193.2 T   -0.08 %
        (8, 8, 256, 32) 2190.6 T   -0.21 %
        (4, 8, 256, 64) 2129.0 T   -3.10 %  no-op partition
        (4, 8, 224, 28) 2038.5 T   -7.68 %
        (4, 8, 224, 56) 1990.3 T  -10.29 %  no-op partition
        (4, 8, 192, 24) 1894.5 T  -15.86 %  too tight

(4, 8, 256, 32) is the unique winner; partition math is clean
block = xcds * cs = 8 * 32 = 256 = slots → 1 clean chiplet-pair
partition. wmin > lmax → every seed of (4,8,256,32) beats every
seed of (4,4) on the round_1 7-seed × 2000-iter probe.

The unresolved question for shipping: round_1 baseline was the OLD
(gm=4, xcds=4) cell, but the CURRENT shipped rule (R31 at config.py
~line 1322 of the if-elif chain in grouped_gemm_fp8_impl.py) is
(gm=1, xcds=4, slots=0, cs=0). R31's evidence reported (gm=1, xcds=4)
beats R30's (gm=8, xcds=4) by +1.07 % on this shape, but did not
directly compare against (gm=4, xcds=4). So we cannot infer the
relation between (gm=1, xcds=4) [current] and (gm=4, xcds=8, 256, 32)
[candidate] from the existing data.

This probe runs that direct A/B on remote MI355X.

Methodology
-----------
- Shape: GateUP B=32 M=4096 N=5760 K=2880 (gpt_oss family wgrad cell).
- Section: var-K wgrad (CRR layout via `grouped_gemm_fp8_variable_k_impl`).
- Cells (gm, xcd, slots, cs):
    (1, 4, 0,   0)*  — current shipped (R31), baseline
    (4, 8, 256, 32)  — round_1 fleet winner (R46-class lever)
    (4, 4, 0,   0)   — round_1 fleet baseline (R30 cell, sanity bridge)
    (1, 8, 256, 32)  — defensive control: keep R31 gm=1 + add xcds=8 + (slots=256, cs=32)
- 7 seeds × 2000-iter p20 each cell, kernel-only timing via direct
  `grouped_variable_k_crr_dscale` call with monkey-patched
  (gm, xcds, slots, cs).
- SHIP gate: candidate beats baseline by ≥+0.5 % AND wmin > lmax
  (every seed of candidate < every seed of baseline). +0.5 % on a
  single shape ≈ +1 score point (under the metric's noise floor of
  ~3-5, but R46 shipped on the same per-shape evidence quality).
- Sibling regression check: NOT in this probe. The lever is gated to
  `m_total == 131072` only — the M=2048 sibling (m_total=65536) keeps
  the existing R31 (gm=1, xcds=4) rule. R46 already verified M=2048
  REGRESSES at the same cell on Down-B32, so we trust the same
  M-axis behaviour on GateUP-B32 (consistent persistent-grid topology
  argument: both have ws/CU ~ 31 vs ~15 — M=4096 has 2x deeper
  per-CTA work to amortise the cross-chiplet L2 cost of xcds=8).

Decision matrix
---------------
- If (4, 8, 256, 32) wins ≥+0.5 % over (1, 4, 0, 0) AND wmin>lmax:
    SHIP the rule (add a new `m_total == 131072` carve-out branch
    in the `(a==2880 AND b==5760)` GateUP-B32 elif at
    grouped_gemm_fp8_impl.py:~1319).
- If (1, 8, 256, 32) wins similarly: ship that instead (preserve R31
  gm=1 + just add the slots/cs/xcds=8 lever).
- If neither candidate clears +0.5 %: FALSIFIED — document under
  analysis/_notes/round-3-gateup-b32-m4096-wgrad-xcds8-FALSIFIED.md.

Wall budget: ~25 s (4 cells × 7 seeds × 2000 iter × ~0.5 ms / iter ÷ 256
slots ≈ 6 s/cell + warmup).
"""
import os
import statistics
import sys
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: F401, E402
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType  # noqa: E402
from primus_turbo.pytorch.core.low_precision import ScalingGranularity  # noqa: E402
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module  # noqa: E402
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa: E402
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa: E402

import _metric_hk_ratio as hk_ratio  # noqa: E402

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


def _patch_var_k(gm, xcds, slots, cs):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_variable_k_crr_dscale

    def wrapped(*args, **kwargs):
        kwargs["group_m"] = gm
        kwargs["num_xcds"] = xcds
        kwargs["num_slots"] = slots
        kwargs["chunk_size"] = cs
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_variable_k_crr_dscale", wrapped)
    return orig


def _restore_var_k(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_variable_k_crr_dscale", orig)


def time_wgrad(B, M, N, K):
    """var-K wgrad timing.

    The autograd dB call is:
        out [B, N, K] = grad_out.T @ x
    where grad_out is [B*M, N] and x is [B*M, K]. The HK var-K backend's
    `execute(a, b, ...)` takes a=x (=K_fwd dim), b=grad_out (=N_fwd dim),
    so a.shape[1]=K=2880 and b.shape[1]=N=5760 for GateUP.
    """
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    x = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    x_fp8, x_s = quantize_fp8(x, _FP8_DTYPE, _GRAN)
    g_fp8, g_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_variable_k_impl(
            x_fp8, g_fp8, x_s, g_s, g_lens, g_offs,
            trans_a=True, trans_b=False, trans_c=True,
            out_dtype=torch.bfloat16, granularity=_GRAN.value,
            num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call)
    return flops / (ms * 1e9), ms


def run(label, B, M, N, K, cells, baseline_cell, seeds=(42, 137, 2024, 7, 99, 1234, 31337)):
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'cell':>20}  {'med ms':>9}  {'min ms':>9}  {'max ms':>9}  {'spread%':>7}  {'TFLOPS':>8}  {'Δ%':>7}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for cell in cells:
        gm, xcd, slots, cs = cell
        orig = _patch_var_k(gm, xcd, slots, cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_wgrad(B, M, N, K)
            seed_meds.append(ms)
        seed_meds.sort()
        med = seed_meds[len(seed_meds) // 2]
        results[cell] = (med, seed_meds[0], seed_meds[-1], seed_meds)
        _restore_var_k(orig)

    base_med, base_lo, base_hi, base_seeds = results[baseline_cell]
    for cell in cells:
        med, lo, hi, _seeds = results[cell]
        tflops = flops / (med * 1e9)
        spread_pp = (hi - lo) / med * 100
        delta_pp = (base_med - med) / base_med * 100
        marker = " *base" if cell == baseline_cell else ""
        print(f"  {str(cell):>20}  {med:>9.5f}  {lo:>9.5f}  {hi:>9.5f}  {spread_pp:>6.2f}%  {tflops:>8.1f}  {delta_pp:+6.2f}%{marker}")
    best_cell = min(cells, key=lambda c: results[c][0])
    best_med, best_lo, best_hi, best_seeds = results[best_cell]
    lift = (base_med - best_med) / base_med * 100
    # wmin_beats_lmax: every seed of best beats every seed of baseline
    wmin_beats_lmax = best_hi < base_lo
    print(f"  BEST: cell={best_cell}  ({lift:+.3f}% over baseline {baseline_cell})  wmin_beats_lmax={wmin_beats_lmax}")
    return results, best_cell, lift, wmin_beats_lmax


if __name__ == "__main__":
    print("[probe] R3 GateUP-B32-M4096 var-K wgrad — ship R46-class xcds=8 lever")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    cells = [
        (1, 4, 0,   0),    # current shipped R31 baseline
        (4, 8, 256, 32),   # round_1 fleet winner (R46-class)
        (4, 4, 0,   0),    # round_1 fleet baseline (sanity bridge)
        (1, 8, 256, 32),   # defensive: preserve R31 gm=1 + add xcds=8 lever
    ]
    baseline = (1, 4, 0, 0)

    t0 = time.monotonic()
    res, best, lift, wmin = run(
        "GateUP-B32-M4096 wgrad", B=32, M=4096, N=5760, K=2880,
        cells=cells, baseline_cell=baseline,
    )
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")

    print("\n[probe] DECISION:")
    if best == baseline:
        print("        baseline still best → FALSIFIED")
    else:
        ship = (lift >= 0.5) and wmin
        verdict = "SHIP" if ship else f"FALSIFIED (lift {lift:+.2f}%, wmin>lmax={wmin})"
        print(f"        candidate {best}  lift {lift:+.3f}%  wmin>lmax={wmin}  → {verdict}")
