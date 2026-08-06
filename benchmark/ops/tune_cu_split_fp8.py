###############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Offline scanner for the dispatch kernel's COMM/PRESHUFFLE CU split.

Produces ``primus_turbo/flydsl/mega/fp8/cu_split_table.json``, the table
``dispatch_grouped_gemm_mxfp8`` reads when the caller leaves the split at None.

The split is scanned here, out of process, rather than on the kernel's first call, because a
first-call search measures the wrong thing: the ranks are still compiling and are stepping through
different candidates, so the two forward candidates land within 0.8% of each other while their real
steady-state gap is 5% -- the winner ends up decided by noise. This runs the ordinary bench, which
warms up and times in steady state with all ranks pinned to the same candidate.

Shape keys are not rebuilt here; the kernel prints the key it resolved (PT_MEGA_CU_SPLIT_DEBUG=1)
and this harvests it, so the table cannot drift from the kernel's own keying.

Each candidate is a full bench process (one compile per candidate per shape, a few minutes each).
Runs the whole matrix and only then writes, so an interrupted scan leaves the committed table alone.

    # scan the shapes the DSv3 EP8 forward and L2 dgrad use, 3 process reps each
    python benchmark/ops/tune_cu_split_fp8.py --reps 3

    # one shape, show what would be written
    python benchmark/ops/tune_cu_split_fp8.py --stages l1 --reps 1 --dry-run
"""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmark" / "ops" / "training" / "bench_mega_moe_fp8.py"
TABLE = REPO / "primus_turbo" / "flydsl" / "mega" / "fp8" / "cu_split_table.json"

# Kernel-reported "cu split (d, p) from <src> for <key>" -- the key comes from the kernel itself
# (PT_MEGA_CU_SPLIT_DEBUG=1) so the scanner never has to re-derive pool size / rank count.
RE_KEY = re.compile(r"\[mega fp8\] cu split \((\d+), (\d+)\) from \w+ for (\S+)")
# "  fp8  L1 :    5.105 ms | ..." / "  fp8  :    2.031 ms | ..."
RE_MS = re.compile(r"fp8\s+(?:L1|L2|fwd)?\s*:\s*([\d.]+)\s*ms")

# Which bench stage exercises the kernel, and the GEMM N it runs at (--inter = I).
STAGES = {
    "l1": lambda inter: 2 * inter,  # forward dispatch(x) + fc1, N = 2I
    "dispatch_fc2_dgrad": lambda inter: inter,  # backward dispatch(dy) + fc2 dgrad, N = I
}


def run_candidate(stage, split, args):
    """Time one (stage, split) in a fresh bench process; return (ms, kernel-reported shape key)."""
    env = dict(os.environ)
    env["PT_MEGA_CU_SPLIT_DEBUG"] = "1"  # makes the kernel report the shape key it resolved
    # The bench's own --dispatch-cu / --preshuffle-cu pin only the stage under test, so the L1 the
    # dgrad stage runs as setup keeps whatever the table already says (PT_MEGA_CU_SPLIT would pin
    # both, and the setup call is not what is being timed).
    cmd = [
        sys.executable, str(BENCH),
        "--dispatch-cu", str(split[0]),
        "--preshuffle-cu", str(split[1]),
        "--stage", stage,
        "--mode", args.mode,
        "--num-processes", str(args.num_processes),
        "--hidden", str(args.hidden),
        "--inter", str(args.inter),
        "--num-experts", str(args.num_experts),
        "--num-topk", str(args.num_topk),
        "--num-tokens", str(args.num_tokens),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out[-4000:])
        raise SystemExit(f"bench failed for {stage} {split} (exit {proc.returncode})")

    # A run touches more than one shape (the dgrad stage warms up through an L1), so pick the key
    # whose N belongs to the stage under test.
    want_n = STAGES[stage](args.inter)
    keys = {k for _d, _p, k in RE_KEY.findall(out) if int(k.split(",")[0]) == want_n}
    ms = [float(m) for m in RE_MS.findall(out)]
    if len(keys) != 1 or not ms:
        print(out[-4000:])
        raise SystemExit(
            f"could not read {stage} {split}: keys at N={want_n}={sorted(keys)}, times={ms}"
        )
    return min(ms), keys.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", nargs="+", choices=sorted(STAGES), default=sorted(STAGES))
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="splits to scan as d,p (default: the kernel's _CU_SPLIT_CANDIDATES)")
    ap.add_argument("--reps", type=int, default=3,
                    help="bench processes per candidate; the median is compared (process-level noise)")
    ap.add_argument("--mode", choices=["load_balanced", "round_robin"], default="load_balanced")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)  # DeepSeek-V3
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="print the table instead of writing it")
    args = ap.parse_args()

    if args.candidates:
        cands = [tuple(int(v) for v in c.split(",")) for c in args.candidates]
    else:
        sys.path.insert(0, str(REPO))
        from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
            _CU_SPLIT_CANDIDATES,
        )
        cands = list(_CU_SPLIT_CANDIDATES)

    winners, report = {}, []
    for stage in args.stages:
        timed = {}
        for split in cands:
            reps = []
            for r in range(args.reps):
                ms, key = run_candidate(stage, split, args)
                reps.append(ms)
                print(f"  {stage:20s} {str(split):9s} rep{r} {ms:8.3f} ms  [{key}]", flush=True)
            timed[split] = (statistics.median(reps), key)
        best = min(timed, key=lambda s: timed[s][0])
        key = timed[best][1]
        winners[key] = list(best)
        report.append((stage, key, best, timed))

    print(f"\n{'=' * 78}\nCU split scan ({args.mode}, T={args.num_tokens}, EP{args.num_processes})\n{'=' * 78}")
    for stage, key, best, timed in report:
        ref = timed[best][0]
        spread = " ".join(
            f"{s}={timed[s][0]:.3f}({(timed[s][0] / ref - 1) * 100:+.1f}%)" for s in cands
        )
        print(f"  {stage:20s} -> {best}  {spread}\n{' ' * 24}key {key}")

    table = {}
    if TABLE.exists():
        table = json.loads(TABLE.read_text())
    table.update(winners)
    if args.dry_run:
        print(f"\n[dry-run] would write {TABLE}:\n{json.dumps(table, indent=2, sort_keys=True)}")
        return
    tmp = TABLE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    tmp.replace(TABLE)
    print(f"\nwrote {TABLE} ({len(table)} shapes)")


if __name__ == "__main__":
    main()
