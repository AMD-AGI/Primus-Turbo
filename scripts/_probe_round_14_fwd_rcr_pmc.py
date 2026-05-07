#!/usr/bin/env python3
"""Round-14 (gpt_oss FP8 kernel-only ceiling): PMC pass on fwd RCR kernel
for the worst gpt_oss fwd shape (Down_B4_M2048) — characterise dominant
stall on the file-scope ``grouped_rcr_kernel``. Mirrors R1's wgrad
methodology but targets the fwd persistent kernel (R4 falsified the
launch-geometry lever for fwd RCR; this PMC pass diagnoses what is
actually the bottleneck).

Approach: launch the existing kernel-only child probe under
``rocprofv3 --pmc`` with a curated counter list, parse the
``counter_collection.csv`` for ``grouped_rcr_kernel`` dispatches, and
report aggregate ratios (MFMA busy %, MFMA-coexec %, VMEM-cycles %,
SQ-busy %, LDS wait fraction).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
CHILD = os.path.join(SCRIPTS, "_probe_fp8_kernel_rocprof.py")

OUT_DIR = "/tmp/rocprof_round_14_fwd_rcr_pmc"
INPUT_TXT = "/tmp/_round_14_fwd_rcr_pmc_counters.txt"

# Counter selection — same set as R1 (fits gfx950 single-pass PMC budget).
COUNTERS = [
    "GRBM_GUI_ACTIVE",
    "SQ_BUSY_CYCLES",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_VALU_MFMA_COEXEC_CYCLES",
    "SQ_INSTS_VALU_MFMA_F8",
    "SQ_INST_CYCLES_VMEM_RD",
    "SQ_WAIT_INST_LDS",
]

# Worst fwd shape + comparison shapes (Down-B4-M4096 = scaled-up version,
# GateUP-B32-M4096 = saturated counter).
SHAPES: List[Tuple[str, str, int, int, int, int]] = [
    # (label, section, B, M, N, K)
    ("Down_B4_M2048_fwd",   "fwd",  4, 2048, 2880, 2880),  # worst (1482 T, 1.5 wave-steps/CU)
    ("Down_B4_M4096_fwd",   "fwd",  4, 4096, 2880, 2880),  # scaled (1895 T, 3.0 wave-steps/CU)
    ("GateUP_B32_M4096_fwd","fwd", 32, 4096, 5760, 2880),  # counter (2058 T, 46 wave-steps/CU)
]


def _write_counter_input(path: str) -> None:
    with open(path, "w") as fh:
        fh.write("pmc:\n")
        fh.write("  - " + "\n  - ".join(COUNTERS) + "\n")


def _profile_one(label: str, section: str, B: int, M: int, N: int, K: int,
                 n_calls: int = 30) -> Dict[str, float]:
    out_dir = os.path.join(OUT_DIR, label)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
    env["TURBO_BENCH_FORCE_BACKEND"] = "HIPKITTEN"
    env["METRIC_SKIP_IDLE_CHECK"] = "1"

    cmd = [
        "rocprofv3",
        "--pmc", *COUNTERS,
        "--output-format", "csv",
        "-d", out_dir,
        "-o", "trace",
        "--", sys.executable, CHILD,
        section, str(B), str(M), str(N), str(K), str(n_calls),
    ]
    print(f"[round14-pmc] launching {label}: {' '.join(cmd[:8])} ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240)
    if proc.returncode != 0:
        print(f"[round14-pmc] {label} child stderr (tail):\n{proc.stderr[-1500:]}",
              flush=True)

    csvs: List[str] = []
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".csv") and "counter" in f:
                csvs.append(os.path.join(root, f))
    if not csvs:
        print(f"[round14-pmc] {label}: no counter CSV found in {out_dir} -- "
              f"contents: {os.listdir(out_dir)}", flush=True)
        return {}

    # Sum counters across all kernels matching ``grouped_rcr_kernel`` (covers
    # the 4 template specs <0,{false,true},{false,true}> for n_aligned x
    # FUSED_KTAIL). Tail kernel is skipped on K_REM=64+fuse-eligible shapes
    # (per dispatch_grouped_rcr lines ~7388-7402); for the gpt_oss anchors
    # the entire fwd time is in this one kernel template family.
    sums: Dict[str, float] = defaultdict(float)
    rows = 0
    matched = 0
    target_substr = "grouped_rcr_kernel"
    matched_kernels: Dict[str, int] = defaultdict(int)
    for path in csvs:
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows += 1
                kn = row.get("Kernel_Name", "")
                if target_substr not in kn:
                    continue
                matched += 1
                matched_kernels[kn] += 1
                cn = row.get("Counter_Name", "")
                cv = row.get("Counter_Value", "0")
                try:
                    sums[cn] += float(cv)
                except ValueError:
                    pass
    print(f"[round14-pmc] {label}: rows={rows} matched={matched} "
          f"unique_kernels={len(matched_kernels)}", flush=True)
    if matched_kernels:
        for kn, cnt in matched_kernels.items():
            print(f"    kernel: {kn[:120]}  rows={cnt}", flush=True)
    return dict(sums)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_counter_input(INPUT_TXT)
    results: Dict[str, Dict[str, float]] = {}
    for label, section, B, M, N, K in SHAPES:
        results[label] = _profile_one(label, section, B, M, N, K, n_calls=30)

    print("\n=== ROUND-14 FWD RCR PMC SUMMARY ===\n", flush=True)
    print(f"{'shape':<28}  {'GUI_active':>14}  {'SQ_busy/GUI':>10}  "
          f"{'MFMA_busy/SQ':>12}  {'COEXEC/MFMA':>11}  {'VMEMRD_cyc/GUI':>14}  "
          f"{'LDS_wait/GUI':>12}  {'F8_mfma':>10}", flush=True)
    rows_summary = []
    for label, sums in results.items():
        if not sums:
            print(f"{label:<28}  --- no counters ---", flush=True)
            continue
        gui = sums.get("GRBM_GUI_ACTIVE", 0.0)
        sq_busy = sums.get("SQ_BUSY_CYCLES", 0.0)
        mfma_busy = sums.get("SQ_VALU_MFMA_BUSY_CYCLES", 0.0)
        mfma_coexec = sums.get("SQ_VALU_MFMA_COEXEC_CYCLES", 0.0)
        f8_mfma = sums.get("SQ_INSTS_VALU_MFMA_F8", 0.0)
        vmem_rd_cyc = sums.get("SQ_INST_CYCLES_VMEM_RD", 0.0)
        lds_wait = sums.get("SQ_WAIT_INST_LDS", 0.0)
        sq_p = (sq_busy / gui) if gui else 0.0
        mfma_p = (mfma_busy / sq_busy) if sq_busy else 0.0
        coexec_p = (mfma_coexec / mfma_busy) if mfma_busy else 0.0
        vmem_p = (vmem_rd_cyc / gui) if gui else 0.0
        lds_p = (lds_wait / gui) if gui else 0.0
        rows_summary.append({
            "shape": label, "gui_active": gui, "sq_busy": sq_busy,
            "mfma_busy": mfma_busy, "mfma_coexec": mfma_coexec,
            "f8_mfma": f8_mfma, "vmem_rd_cyc": vmem_rd_cyc,
            "lds_wait": lds_wait,
            "sq_p": sq_p, "mfma_p": mfma_p, "coexec_p": coexec_p,
            "vmem_p": vmem_p, "lds_p": lds_p,
        })
        print(f"{label:<28}  {gui:>14.0f}  {sq_p*100:>9.1f}%  "
              f"{mfma_p*100:>11.1f}%  {coexec_p*100:>10.1f}%  "
              f"{vmem_p*100:>13.1f}%  {lds_p*100:>11.1f}%  "
              f"{f8_mfma:>10.0f}", flush=True)

    json_path = os.path.join(OUT_DIR, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(rows_summary, fh, indent=2)
    print(f"\n[round14-pmc] summary saved to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
