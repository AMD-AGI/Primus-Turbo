#!/usr/bin/env python3
"""Round-1 (gpt_oss FP8 kernel-only ceiling): PMC pass on wgrad var-K kernel
for the worst gpt_oss shape (Down_B4_M2048) — characterise dominant stall to
pick the next kernel-level lever per HipKittens Round-G G1 plan.

Approach: launch the existing kernel-only child probe under
``rocprofv3 --pmc`` with a curated list of counters, parse the
``counter_collection.csv`` for the ``grouped_var_k_kernel_fp8`` dispatches,
and report aggregate ratios (MFMA busy %, MFMA-coexec %, VMEM-cycles %,
SQ-busy %, LDS wait fraction).

Single hardware pass — counters chosen to fit the gfx950 PMC budget so
``rocprofv3`` does not split the run into multiple invocations (which
would dilute aggregate ratios across kernel instances).
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

OUT_DIR = "/tmp/rocprof_round_1_wgrad_pmc"
INPUT_TXT = "/tmp/_round_1_wgrad_pmc_counters.txt"

# Counter selection — chosen to fit a single gfx950 PMC pass.  These cover
# the dominant-stall question for an MFMA-heavy CRR kernel:
#   * GRBM_GUI_ACTIVE        — global active cycles (denom for utilisation)
#   * SQ_BUSY_CYCLES         — at least one wave running on SQ
#   * SQ_VALU_MFMA_BUSY_CYCLES — MFMA actively executing
#   * SQ_VALU_MFMA_COEXEC_CYCLES — MFMA + non-MFMA co-issued (good!)
#   * SQ_INSTS_VALU_MFMA_F8  — count of FP8 MFMAs issued
#   * SQ_INST_CYCLES_VMEM_RD — cycles spent on VMEM-RD
#   * SQ_WAIT_INST_LDS       — cycles waiting on LDS
COUNTERS = [
    "GRBM_GUI_ACTIVE",
    "SQ_BUSY_CYCLES",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_VALU_MFMA_COEXEC_CYCLES",
    "SQ_INSTS_VALU_MFMA_F8",
    "SQ_INST_CYCLES_VMEM_RD",
    "SQ_WAIT_INST_LDS",
]

# Shape under inspection (worst-case wgrad shape per metric Round-1 baseline).
SHAPES: List[Tuple[str, str, int, int, int, int]] = [
    # (label, section, B, M, N, K)
    ("Down_B4_M2048_wgrad",   "wgrad",  4, 2048, 2880, 2880),
    ("Down_B4_M4096_wgrad",   "wgrad",  4, 4096, 2880, 2880),
    ("GateUP_B32_M4096_wgrad","wgrad", 32, 4096, 5760, 2880),  # comparison: best wgrad shape
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
    print(f"[round1-pmc] launching {label}: {' '.join(cmd[:8])} ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240)
    if proc.returncode != 0:
        print(f"[round1-pmc] {label} child stderr (tail):\n{proc.stderr[-1500:]}",
              flush=True)

    # rocprofv3 writes counters to a counter_collection.csv (and possibly
    # individual ``<pid>_counter_collection.csv``).  Locate any file with
    # 'counter' in the name.
    csvs: List[str] = []
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".csv") and "counter" in f:
                csvs.append(os.path.join(root, f))
    if not csvs:
        print(f"[round1-pmc] {label}: no counter CSV found in {out_dir} -- "
              f"contents: {os.listdir(out_dir)}", flush=True)
        return {}

    # rocprofv3 v1.x csv layout (one row per (kernel_dispatch, counter)):
    # Agent_Id, Queue_Id, Process_Id, Thread_Id, Grid_Size, Kernel_Id,
    # Kernel_Name, Workgroup_Size, LDS_Block_Size, Scratch_Size,
    # Arch_VGPR, Accum_VGPR, SGPR, Counter_Name, Counter_Value
    sums: Dict[str, float] = defaultdict(float)
    rows = 0
    matched = 0
    target_substr = "grouped_var_k_kernel_fp8"
    for path in csvs:
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows += 1
                kn = row.get("Kernel_Name", "")
                if target_substr not in kn:
                    continue
                matched += 1
                cn = row.get("Counter_Name", "")
                cv = row.get("Counter_Value", "0")
                try:
                    sums[cn] += float(cv)
                except ValueError:
                    pass
    print(f"[round1-pmc] {label}: rows={rows} matched={matched} "
          f"counters={dict(sums)}", flush=True)
    return dict(sums)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_counter_input(INPUT_TXT)
    results: Dict[str, Dict[str, float]] = {}
    for label, section, B, M, N, K in SHAPES:
        results[label] = _profile_one(label, section, B, M, N, K, n_calls=30)

    print("\n=== ROUND-1 PMC SUMMARY ===\n", flush=True)
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
    print(f"\n[round1-pmc] summary saved to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
