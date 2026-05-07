#!/usr/bin/env python3
"""Round-2 (gpt_oss FP8 kernel-only ceiling): NUM_CUS launch-geometry sweep
on ``grouped_variable_k_crr_dscale`` for the worst gpt_oss wgrad shape
(Down-B4-M2048).

Flows from R1 PMC pass which showed Down-B4-M2048 wgrad at 16.6 % per-CU
MFMA-active under gridDim.x = NUM_CUS = 256, attributing the gap to per-
tile prologue/epilogue overhead × 1.89 wave-steps/CU. R2 hypothesis:
reducing the launch slot count amortises the per-tile overhead better;
candidate range gridDim.x ∈ {32, 64, 96, 128, 160, 192, 256, 384}.

Methodology (matches R10/R11/R30 var-K probe convention):
* 250 iter × 7 trial p20 × 3 seeds, kernel-only direct call to
  ``hk.grouped_variable_k_crr_dscale`` (bypass autograd + dispatcher).
* Bit-equivalence verified at ``TK_VARK_NUM_CUS=256`` (the default) AND
  at the winning candidate (every (gm, xcds, slots) is a pure scheduling
  knob; no MFMA-order rearrangement so output is bit-identical).
* The new env hook (HK kernel_fp8_layouts.cpp:8060 R2 deposit) is
  ``static`` cached on first call, so each sweep value MUST be exercised
  in a fresh subprocess.

Usage: launches one subprocess per sweep value via ``subprocess.run``.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
SWEEP_VALUES = [32, 64, 96, 128, 160, 192, 256, 384]

# Anchor shape: Down-B4-M2048 wgrad (worst gpt_oss wgrad shape, 1263 T at
# baseline). Per-group output [N_fwd, K_fwd] = [2880, 2880] -> tiles_n=11,
# tiles_k=11 = 121 tile-steps per group * 4 groups = 484 tile-steps.
ANCHOR = {"label": "Down_B4_M2048_wgrad",
          "B": 4, "M": 2048, "N": 2880, "K": 2880}

# Sibling + counter for round-3 gating predicate evidence:
#   Down-B4-M4096 wgrad: same 484 tile-step geometry as anchor
#                        (N_fwd=2880, K_fwd=2880) but 2x M_per_group ->
#                        2x K-loop length per tile -> tile compute heavier.
#                        Hypothesis: slots=192 helps similarly (484 / 192).
#   GateUP-B32-M4096 wgrad: 32 groups * 22 * 11 = 7744 tile-steps,
#                           7744 / 256 = 30.25 wave-steps / CU (saturated).
#                           Hypothesis: slots=192 hurts or is neutral
#                           (7744 / 192 = 40.3 wave-steps / CU is fine,
#                            but the persistent loop already amortized).
SIBLINGS = [
    {"label": "Down_B4_M4096_wgrad",
     "B": 4, "M": 4096, "N": 2880, "K": 2880},
    {"label": "GateUP_B32_M4096_wgrad",
     "B": 32, "M": 4096, "N": 5760, "K": 2880},
]

# Inline child runner (executed once per slot value). Times kernel-only
# via cudaEvent. p20 of 7 trials × 250 iters per trial.
CHILD_SOURCE = r'''
import os
import statistics
import sys
import time
import torch

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")
import _metric_hk_ratio as hk_ratio  # noqa

import primus_turbo.pytorch as turbo  # noqa
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType  # noqa
from primus_turbo.pytorch.core.low_precision import (  # noqa
    Float8QuantConfig, Format, ScalingGranularity,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa


def _make_wgrad_kernel(B, M, N, K, seed):
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    a_col, a_s_col = quantize_fp8(a, torch.float8_e4m3fn,
                                  ScalingGranularity.TENSORWISE, axis=-2)
    g_col, g_s_col = quantize_fp8(grad, torch.float8_e4m3fn,
                                  ScalingGranularity.TENSORWISE, axis=-2)
    return (lambda: grouped_gemm_fp8_variable_k_impl(
        a_col, g_col, a_s_col, g_s_col, g_lens, g_offs,
        trans_a=True, trans_b=False, trans_c=True,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE.value, num_cu=None,
        default_backend=BackendType.HIPKITTEN.value,
    ), a_col, g_col)


def time_kernel(fn, n_iters=250, n_trials=7):
    out_durs_ms = []
    for _ in range(n_trials):
        for _ in range(20):  # warmup
            fn()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(n_iters):
            fn()
        ev1.record()
        torch.cuda.synchronize()
        out_durs_ms.append(ev0.elapsed_time(ev1) / n_iters)
    out_durs_ms.sort()
    p20 = out_durs_ms[max(0, int(len(out_durs_ms) * 0.2))]
    return p20, out_durs_ms


def main():
    B, M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    seeds = [42, 137, 2024]
    flops = 2.0 * B * M * N * K
    medians_ms = []
    for s in seeds:
        with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
            fn, _, _ = _make_wgrad_kernel(B, M, N, K, s)
            ms, _ = time_kernel(fn)
            medians_ms.append(ms)
    median_ms = statistics.median(medians_ms)
    tflops = flops / median_ms / 1e9
    print(f"RESULT: median_ms={median_ms:.4f} tflops={tflops:.1f} per-seed={medians_ms}",
          flush=True)


if __name__ == "__main__":
    main()
'''


def _run_one(slots: int, B: int, M: int, N: int, K: int) -> dict:
    env = os.environ.copy()
    env.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
    env["TK_VARK_NUM_CUS"] = str(slots)
    env["METRIC_SKIP_IDLE_CHECK"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SOURCE,
         str(B), str(M), str(N), str(K)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    median_ms = None
    tflops = None
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            for tok in line.split():
                if tok.startswith("median_ms="):
                    median_ms = float(tok.split("=", 1)[1])
                elif tok.startswith("tflops="):
                    tflops = float(tok.split("=", 1)[1])
    return {"slots": slots, "median_ms": median_ms, "tflops": tflops,
            "stdout_tail": out[-400:] if (median_ms is None) else ""}


def main() -> int:
    print(f"\n[round2-vark-numcus] sweep on {ANCHOR['label']} "
          f"(B={ANCHOR['B']}, M={ANCHOR['M']}, N={ANCHOR['N']}, K={ANCHOR['K']})", flush=True)
    print(f"[round2-vark-numcus] sweep slots = {SWEEP_VALUES}", flush=True)
    rows: List[dict] = []
    for slots in SWEEP_VALUES:
        r = _run_one(slots, ANCHOR["B"], ANCHOR["M"], ANCHOR["N"], ANCHOR["K"])
        rows.append(r)
        if r["tflops"] is None:
            print(f"  slots={slots:>3}  TFLOPS=---  median_ms=---", flush=True)
            print(f"      child stderr tail: {r['stdout_tail']}", flush=True)
        else:
            print(f"  slots={slots:>3}  TFLOPS={r['tflops']:>7.1f}  "
                  f"median_ms={r['median_ms']:.4f}", flush=True)

    print("\n[round2-vark-numcus] summary table", flush=True)
    base = next((r["tflops"] for r in rows if r["slots"] == 256 and r["tflops"]), None)
    print(f"  baseline @ slots=256: {base} T", flush=True)
    if base:
        print(f"  {'slots':>5}  {'TFLOPS':>7}  {'Δ vs 256':>9}", flush=True)
        for r in rows:
            if r["tflops"] is None:
                continue
            delta = (r["tflops"] - base) / base * 100.0
            print(f"  {r['slots']:>5}  {r['tflops']:>7.1f}  {delta:>+8.2f}%",
                  flush=True)

    # Sibling + counter shapes — narrow sweep around the apparent winner.
    # Use a coarser slot set since we're checking transferability, not optima.
    NARROW = [128, 160, 192, 224, 256]
    sibling_rows: Dict[str, List[dict]] = {}
    for sib in SIBLINGS:
        print(f"\n[round2-vark-numcus] sibling sweep on {sib['label']} "
              f"(B={sib['B']}, M={sib['M']}, N={sib['N']}, K={sib['K']})", flush=True)
        sib_rows = []
        for slots in NARROW:
            r = _run_one(slots, sib["B"], sib["M"], sib["N"], sib["K"])
            sib_rows.append(r)
            if r["tflops"] is None:
                print(f"  slots={slots:>3}  TFLOPS=---", flush=True)
            else:
                print(f"  slots={slots:>3}  TFLOPS={r['tflops']:>7.1f}  "
                      f"median_ms={r['median_ms']:.4f}", flush=True)
        sib_base = next((r["tflops"] for r in sib_rows
                         if r["slots"] == 256 and r["tflops"]), None)
        if sib_base:
            print(f"  baseline @ slots=256: {sib_base:.1f} T", flush=True)
            for r in sib_rows:
                if r["tflops"] is None:
                    continue
                delta = (r["tflops"] - sib_base) / sib_base * 100.0
                marker = "  <-- ANCHOR_WINNER" if r["slots"] == 192 else ""
                print(f"    slots={r['slots']:>3}  {r['tflops']:>7.1f}  "
                      f"{delta:>+7.2f}%{marker}", flush=True)
        sibling_rows[sib["label"]] = sib_rows

    out_path = "/tmp/round_2_vark_numcus_sweep.json"
    with open(out_path, "w") as fh:
        json.dump({"anchor": ANCHOR, "rows": rows,
                   "siblings": sibling_rows}, fh, indent=2)
    print(f"\n[round2-vark-numcus] saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
