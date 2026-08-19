#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pull the mega-MoE per-stage numbers out of bench_mega_moe.py logs and diff
them against a baseline.

The two fused operators cover five kernel stages in total:

    dispatch_grouped_gemm : fwd (nt), bwd dgrad (nn), bwd wgrad dW1 (tn)
    grouped_gemm_combine  : fwd (nt), bwd dgrad (nn)

Both sides of the diff can be either a run directory / log file produced by
report/bench_mega_moe.sh, or a baseline JSON (report/baseline/*.json).

Usage:
    python report/extract_performance.py [RUN]
    python report/extract_performance.py report/runs/20260819-014138 \
        --baseline report/baseline/mi355x_19fe104.json

    RUN                 run dir, log file, or several log files.
                        Default: the newest directory in report/runs.
    --baseline PATH     baseline JSON, run dir, or log file.
                        Default: report/baseline/mi355x_19fe104.json
    --case NAME         MoE case to report. Default: the only / first one found.
    --json PATH         also dump the parsed current run + diff as JSON.

Stdlib only: it runs both inside the ROCm container and on a bare host.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASELINE = os.path.join(_HERE, "baseline", "mi355x_19fe104.json")
RUNS_DIR = os.path.join(_HERE, "runs")

# canonical stage order + display labels; keys match the baseline JSON
STAGES = [
    ("dispatch/fwd", "dispatch  fwd (nt)"),
    ("dispatch/bwd", "dispatch  bwd dgrad (nn)"),
    ("dispatch/wgrad", "dispatch  bwd wgrad dW1 (tn)"),
    ("combine/fwd", "combine   fwd (nt)"),
    ("combine/bwd", "combine   bwd dgrad (nn)"),
]
LABELS = dict(STAGES)

# ------------------------------------------------------------------ parsing --
# [dispatch] DeepSeek-V3 MI350X EP8 T=8192 H=7168 I=2048 E=256 K=8 (max over ranks)
HEADER_RE = re.compile(
    r"^\[(?P<mode>dispatch|combine)\]\s+(?P<case>\S+)\s+(?P<gpu>\S+)\s+EP(?P<ep>\d+)\s+"
    r"T=(?P<tokens>\d+)\s+H=(?P<hidden>\d+)\s+I=(?P<inter>\d+)\s+E=(?P<experts>\d+)\s+K=(?P<topk>\d+)"
)
#   ----------------------  backward dgrad (NN, = dispatch_grouped_0)
SUBHEADER_RE = re.compile(r"^\s*-{10,}\s*(?P<label>\S.*?)\s*$")
#   gemm_only    :    4.055 ms |  1004.1 TFLOPS
GEMM_RE = re.compile(r"^\s*gemm_only\s*:\s*(?P<ms>[\d.]+)\s*ms\s*\|\s*(?P<tflops>[\d.]+)\s*TFLOPS")
#   dispatch_only:    1.643 ms |   331.2 GB/s (XGMI)
COMM_RE = re.compile(
    r"^\s*(?:dispatch_only|combine_only)\s*:\s*(?P<ms>[\d.]+)\s*ms\s*\|\s*(?P<gbps>[\d.]+)\s*GB/s"
)
#   fused        :    4.353 ms |   935.4 TFLOPS | roofline (...) = 93.2% | speedup vs serial = 1.31x
FUSED_RE = re.compile(
    r"^\s*fused\s*:\s*(?P<ms>[\d.]+)\s*ms\s*\|\s*(?P<tflops>[\d.]+)\s*TFLOPS\s*\|\s*"
    r"roofline[^=]*=\s*(?P<roofline>[\d.]+)%\s*\|\s*speedup vs serial\s*=\s*(?P<speedup>[\d.]+)x"
)
#   [check] fwd fused (nt)              : cos=1.00000 rel=0.0000 (worst rank=0) PASS
CHECK_RE = re.compile(r"^\s*\[check\]\s+(?P<what>.+?)\s*:\s*.*\s(?P<verdict>PASS|FAIL)\s*$")


def _stage_key_from_subheader(mode: str, label: str) -> str:
    """Map a stage sub-header line to a canonical <mode>/<stage> key."""
    low = label.lower()
    if "wgrad" in low:
        return f"{mode}/wgrad"
    if "dgrad" in low:
        return f"{mode}/bwd"
    return f"{mode}/fwd"


def parse_log(path: str) -> dict:
    """Parse one bench_mega_moe.py log; returns {case: {"meta": ..., "stages": {key: metrics}}}."""
    cases: dict = {}
    mode = case = stage_key = None
    checks: list = []

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            check = CHECK_RE.match(line)
            if check:
                checks.append((check.group("what"), check.group("verdict")))
                continue

            header = HEADER_RE.match(line)
            if header:
                mode = header.group("mode")
                case = header.group("case")
                entry = cases.setdefault(case, {"meta": {}, "stages": {}})
                entry["meta"].update(
                    gpu=header.group("gpu"),
                    ep=int(header.group("ep")),
                    tokens_per_rank=int(header.group("tokens")),
                    hidden=int(header.group("hidden")),
                    inter=int(header.group("inter")),
                    experts=int(header.group("experts")),
                    topk=int(header.group("topk")),
                )
                # the block right after the header is the forward stage (it has no sub-header)
                stage_key = f"{mode}/fwd"
                # checks are printed before the header of the case they belong to
                entry.setdefault("checks", []).extend(checks)
                checks = []
                continue

            if case is None:
                continue  # autotune / warm-up chatter before the first case header

            sub = SUBHEADER_RE.match(line)
            if sub:
                stage_key = _stage_key_from_subheader(mode, sub.group("label"))
                cases[case]["stages"].setdefault(stage_key, {})["label"] = sub.group("label")
                continue

            metrics = cases[case]["stages"].setdefault(stage_key, {})
            gemm = GEMM_RE.match(line)
            if gemm:
                metrics["gemm_ms"] = float(gemm.group("ms"))
                metrics["gemm_tflops"] = float(gemm.group("tflops"))
                continue
            comm = COMM_RE.match(line)
            if comm:
                metrics["comm_ms"] = float(comm.group("ms"))
                metrics["comm_gbps"] = float(comm.group("gbps"))
                continue
            fused = FUSED_RE.match(line)
            if fused:
                metrics["fused_ms"] = float(fused.group("ms"))
                metrics["fused_tflops"] = float(fused.group("tflops"))
                metrics["roofline_pct"] = float(fused.group("roofline"))
                metrics["speedup"] = float(fused.group("speedup"))
    return cases


def collect_logs(paths: list) -> list:
    """Expand run dirs into their *.log files; keep explicit files as given."""
    logs = []
    for path in paths:
        if os.path.isdir(path):
            logs.extend(sorted(glob.glob(os.path.join(path, "*.log"))))
        else:
            logs.append(path)
    return logs


def load_run(paths: list) -> dict:
    """Merge every log of a run into one {case: {...}} dict."""
    merged: dict = {}
    logs = collect_logs(paths)
    if not logs:
        sys.exit(f"error: no log files found in {', '.join(paths)}")
    for log in logs:
        for case, data in parse_log(log).items():
            entry = merged.setdefault(case, {"meta": {}, "stages": {}, "checks": []})
            entry["meta"].update(data["meta"])
            entry["stages"].update(data["stages"])
            entry["checks"].extend(data.get("checks", []))
    return merged


def load_side(path: str, case_hint: str | None) -> tuple:
    """Load one side of the diff: a baseline JSON, or a run dir / log file(s).

    Returns (name, case, stages, meta).
    """
    if os.path.isfile(path) and path.endswith(".json"):
        with open(path) as handle:
            data = json.load(handle)
        meta = dict(data.get("config", {}))
        meta["gpu"] = data.get("gpu", "?")
        return data.get("name", path), data.get("case", "?"), data["stages"], meta

    run = load_run([path])
    case = pick_case(run, case_hint)
    entry = run[case]
    return os.path.relpath(path), case, entry["stages"], entry["meta"]


def pick_case(run: dict, case_hint: str | None) -> str:
    if case_hint:
        if case_hint not in run:
            sys.exit(f"error: case {case_hint} not in the logs (found: {', '.join(run) or 'none'})")
        return case_hint
    if not run:
        sys.exit("error: no benchmark case found in the logs")
    return sorted(run)[0]


def newest_run() -> str:
    dirs = [d for d in glob.glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(d)]
    if not dirs:
        sys.exit(f"error: no run directories under {RUNS_DIR}; pass one explicitly")
    return max(dirs, key=os.path.getmtime)


# ------------------------------------------------------------------ report ---
def render(headers: list, rows: list, aligns: str) -> str:
    """Minimal fixed-width table renderer ('l'/'r' per column)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells):
        out = []
        for i, cell in enumerate(cells):
            text = str(cell)
            out.append(text.ljust(widths[i]) if aligns[i] == "l" else text.rjust(widths[i]))
        return "  ".join(out).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def delta(now: float | None, base: float | None) -> str:
    if now is None or base is None:
        return "n/a"
    return f"{now - base:+.3f}"


def delta_pct(now: float | None, base: float | None) -> str:
    if now is None or base is None or not base:
        return "n/a"
    return f"{(now - base) / base * 100:+.1f}%"


def fmt(value, spec="{:.3f}"):
    return "n/a" if value is None else spec.format(value)


def summarize(now_name, now_case, now_stages, now_meta, base_name, base_case, base_stages, base_meta):
    out = []
    out.append(f"case      : {now_case} (baseline: {base_case})")
    out.append(f"current   : {now_name}  [{now_meta.get('gpu', '?')}]")
    out.append(f"baseline  : {base_name}  [{base_meta.get('gpu', '?')}]")
    out.append("")

    # 1) per-kernel fused latency vs baseline
    rows = []
    missing = []
    for key, label in STAGES:
        now = now_stages.get(key)
        base = base_stages.get(key)
        if now is None:
            missing.append(key)
        now_ms = (now or {}).get("fused_ms")
        base_ms = (base or {}).get("fused_ms")
        rows.append(
            [
                label,
                fmt(base_ms),
                fmt(now_ms),
                delta(now_ms, base_ms),
                delta_pct(now_ms, base_ms),
                fmt((base or {}).get("roofline_pct"), "{:.1f}%"),
                fmt((now or {}).get("roofline_pct"), "{:.1f}%"),
                fmt((base or {}).get("fused_tflops"), "{:.1f}"),
                fmt((now or {}).get("fused_tflops"), "{:.1f}"),
            ]
        )

    def total(stages, field):
        values = [stages.get(k, {}).get(field) for k, _ in STAGES]
        return None if any(v is None for v in values) else sum(values)

    base_total = total(base_stages, "fused_ms")
    now_total = total(now_stages, "fused_ms")
    rows.append(["-" * 28] + ["-" * 6] * 8)
    rows.append(
        [
            "TOTAL (5 kernels)",
            fmt(base_total),
            fmt(now_total),
            delta(now_total, base_total),
            delta_pct(now_total, base_total),
            "",
            "",
            "",
            "",
        ]
    )
    out.append("fused latency per kernel (ms)")
    out.append(
        render(
            ["kernel", "base", "now", "Δ ms", "Δ %", "roof base", "roof now", "TF/s base", "TF/s now"],
            rows,
            "lrrrrrrrr",
        )
    )
    out.append("")

    # 2) which leg moved: GEMM vs comm
    leg_rows = []
    for key, label in STAGES:
        now = now_stages.get(key, {})
        base = base_stages.get(key, {})
        leg_rows.append(
            [
                label,
                fmt(base.get("gemm_ms")),
                fmt(now.get("gemm_ms")),
                delta(now.get("gemm_ms"), base.get("gemm_ms")),
                fmt(base.get("comm_ms")),
                fmt(now.get("comm_ms")),
                delta(now.get("comm_ms"), base.get("comm_ms")),
            ]
        )
    base_gemm, now_gemm = total(base_stages, "gemm_ms"), total(now_stages, "gemm_ms")
    base_comm, now_comm = total(base_stages, "comm_ms"), total(now_stages, "comm_ms")
    leg_rows.append(["-" * 28] + ["-" * 6] * 6)
    leg_rows.append(
        [
            "TOTAL (5 kernels)",
            fmt(base_gemm),
            fmt(now_gemm),
            delta(now_gemm, base_gemm),
            fmt(base_comm),
            fmt(now_comm),
            delta(now_comm, base_comm),
        ]
    )
    out.append("leg breakdown (ms): GEMM-only / comm-only baselines")
    out.append(
        render(
            ["kernel", "gemm base", "gemm now", "Δ", "comm base", "comm now", "Δ"],
            leg_rows,
            "lrrrrrr",
        )
    )
    out.append("")

    # 3) the bottom line
    if base_total is not None and now_total is not None:
        gap = now_total - base_total
        if abs(gap) < 5e-4:
            out.append("on par with the baseline over the 5 kernels.")
        elif gap > 0:
            out.append(
                f"gap to baseline: {gap:+.3f} ms over the 5 kernels "
                f"({gap / base_total * 100:+.1f}%), i.e. still {gap:.3f} ms to claw back."
            )
        else:
            out.append(
                f"ahead of baseline: {gap:.3f} ms over the 5 kernels ({gap / base_total * 100:+.1f}%)."
            )
        if None not in (base_gemm, now_gemm, base_comm, now_comm):
            out.append(
                f"  of which GEMM legs {now_gemm - base_gemm:+.3f} ms, "
                f"comm legs {now_comm - base_comm:+.3f} ms."
            )
        worst = [
            (
                now_stages[k]["fused_ms"] - base_stages[k]["fused_ms"],
                LABELS[k],
            )
            for k, _ in STAGES
            if k in now_stages and k in base_stages and "fused_ms" in now_stages[k]
        ]
        worst.sort(reverse=True)
        if worst:
            top = ", ".join(f"{label.strip()} {d:+.3f} ms" for d, label in worst[:3])
            out.append(f"  biggest offenders: {top}")

    if missing:
        out.append(f"warning: missing from the current run: {', '.join(missing)}")

    failed = [what for what, verdict in now_meta.get("checks", []) if verdict != "PASS"]
    if failed:
        out.append(f"warning: accuracy checks FAILED: {', '.join(failed)}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Extract mega-MoE per-kernel performance from bench logs and diff vs a baseline"
    )
    parser.add_argument("run", nargs="*", help="run dir or log file(s); default = newest report/runs/*")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="baseline JSON, run dir, or log")
    parser.add_argument("--case", default=None, help="MoE case name (default: the first one found)")
    parser.add_argument("--json", dest="json_out", default=None, help="dump parsed metrics as JSON")
    args = parser.parse_args()

    run_paths = args.run or [newest_run()]
    run = load_run(run_paths)
    case = pick_case(run, args.case)
    entry = run[case]
    now_meta = dict(entry["meta"])
    now_meta["checks"] = entry.get("checks", [])
    now_name = os.path.relpath(run_paths[0]) if len(run_paths) == 1 else f"{len(run_paths)} logs"

    base_name, base_case, base_stages, base_meta = load_side(args.baseline, args.case)

    report = summarize(
        now_name,
        case,
        entry["stages"],
        now_meta,
        base_name,
        base_case,
        base_stages,
        base_meta,
    )
    print(report)

    if args.json_out:
        payload = {
            "case": case,
            "current": {"name": now_name, "meta": entry["meta"], "stages": entry["stages"]},
            "baseline": {"name": base_name, "case": base_case, "stages": base_stages},
        }
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
