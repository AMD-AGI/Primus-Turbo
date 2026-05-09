#!/usr/bin/env python3
"""Round-10 (gpt_oss FP8 kernel-only ceiling task) — PMC re-measurement
on the post-R9 (closed-form decode) binary.

Executes the same R21 PMC scaffold counter set on the R9 binary and
prints aggregate medians for direct comparison to R21's pre-R9 numbers.

R9 (`HipKittens b3a5c8db` + `Primus ae98d226`) replaced the 6-iter
binary search over `s_cum_tiles[]` with the O(1) closed-form
`group_idx = gt / tiles_per_group`. The hypothesis was R21's measured
SALU/SQ_busy = 85 % was driven (in part) by per-tile coord decode.
The metric showed NEUTRAL (median 694, mean 693.9 over 8 samples;
indistinguishable from R29 noise floor).

This R10 round runs the same PMC pass (Down_B4_M2048 wgrad var-K) on
the R9 binary to attribute whether decode SALU dropped at the counter
level even when the metric did not see it. Two outcomes:

  (A) SALUBusy / SQ_INSTS_SALU dropped — R9 worked partially; the
      remaining SALU floor is in the K-loop body, and R11+ should
      attack per-K-iter SALU contributors (tic/toc hoist; offset
      caching) per R9 forward-pointer items 1-2.

  (B) SALUBusy / SQ_INSTS_SALU unchanged — R9 saved zero observable
      SALU; per-tile decode was already DCE'd or otherwise
      sub-resolution. Direction D as a category should ROTATE to
      A1' variant-2 K-split (R7-R8 forward pointer) because the
      "SALU is the bottleneck" framing is structurally wrong.

This is an OBSERVABILITY round — no kernel change, no metric move
expected. Daemon will record the same ~694 the R9 binary produced.

Usage:
    bash scripts/dbg_remote.sh \
        'python3 scripts/_probe_round_10_vark_pmc_compare.py'
"""
from __future__ import annotations

import csv
import glob
import os
import statistics
import subprocess
import sys


# Re-use the R21 scaffold to actually run rocprofv3.
_HERE = os.path.abspath(os.path.dirname(__file__))
R21_SCAFFOLD = os.path.join(_HERE, "_probe_round_21_vark_pmc_scaffold.py")

SHAPE = "Down_B4_M2048"
ROOT = f"/tmp/r21_vark_pmc_{SHAPE}"

# R21 reference numbers (pre-R9 binary), copied from
# `analysis/_notes/round-21-vark-pmc-mfma-underfeed-IDENTIFIED.md` headline
# table. Per-invocation medians, n=70 across 5 batches × ~14 invocations.
R21 = {
    "MfmaUtil":           32.10,    # %
    "SALUBusy":            9.10,    # %
    "MemUnitStalled":      0.20,    # %
    "FetchSize":           88.0e6,  # bytes (88 MB)
    "SQ_INSTS_VALU_MFMA_F8": 2.36e6,
    "SQ_BUSY_CYCLES":     6.28e6,
    "SQ_INSTS_VALU":      None,     # not in R21 headline; will print R10 only
    "SQ_INSTS_SALU":      None,
    "SQ_INSTS_LDS":       None,
    "SQ_INSTS_SMEM":      None,
    "TCC_HIT_sum":        None,
    "TCC_MISS_sum":       None,
}


def maybe_run_pmc(force: bool = False) -> int:
    """Re-run the R21 scaffold to (re)populate /tmp/r21_vark_pmc_*. Skip
    if the derived/ batch CSV already exists (idempotent)."""
    derived_csv = glob.glob(f"{ROOT}/derived/*/chi*/*counter_collection.csv")
    if derived_csv and not force:
        print(f"[r10] reusing existing PMC under {ROOT}", flush=True)
        return 0
    print(f"[r10] launching R21 scaffold to populate {ROOT}", flush=True)
    return subprocess.call([sys.executable, R21_SCAFFOLD,
                            "--shape", SHAPE, "--n-calls", "50"])


def aggregate() -> dict:
    """Aggregate per-counter medians across all batches/invocations."""
    rows = []
    for csvf in glob.glob(f"{ROOT}/*/*/chi*/*counter_collection.csv"):
        with open(csvf) as f:
            for row in csv.DictReader(f):
                if "grouped_var_k_kernel_fp8" in row.get("Kernel_Name", ""):
                    rows.append(row)
    out = {}
    for k in R21:
        vals = [float(r["Counter_Value"]) for r in rows
                if r["Counter_Name"] == k]
        if vals:
            out[k] = (statistics.median(vals), len(vals))
    return out


def fmt(v: float) -> str:
    if v is None:
        return "—"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.2f}k"
    return f"{v:.2f}"


def main() -> int:
    rc = maybe_run_pmc(force=False)
    if rc != 0:
        return rc
    medians = aggregate()
    print()
    print(f"{'counter':30}  {'R21 (pre-R9)':>14}  "
          f"{'R10 (post-R9)':>14}  {'Δ':>8}  n")
    print("-" * 82)
    for k in R21:
        r21 = R21[k]
        if k not in medians:
            print(f"{k:30}  {fmt(r21):>14}  {'NO DATA':>14}  {'-':>8}  -")
            continue
        r10, n = medians[k]
        delta = "—" if r21 is None else f"{((r10 - r21) / r21 * 100):+.1f}%"
        print(f"{k:30}  {fmt(r21):>14}  {fmt(r10):>14}  {delta:>8}  {n}")
    print()
    print("[verdict] If SALU counters (SALUBusy, SQ_INSTS_SALU) dropped "
          "→ R9 partially worked, K-loop body is residual.")
    print("[verdict] If SALU counters unchanged → R9 was DCE'd; "
          "Direction D rotates to A1' variant-2 K-split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
