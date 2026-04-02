#!/usr/bin/env python3
"""Performance regression checker.

Compares current benchmark results against saved baselines and reports
any regressions that exceed defined thresholds.

Usage:
    python3 tools/check_regression.py --baseline benchmark/baselines/ --current output/results.csv
    python3 tools/check_regression.py --baseline benchmark/baselines/ --current output/
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

WARN_THRESHOLD = 0.05   # 5% per-config regression → warning
FAIL_THRESHOLD = 0.10   # 10% per-config regression → failure
GEOMEAN_THRESHOLD = 0.02  # 2% geomean regression → failure


def load_baseline(baseline_dir: str) -> dict[str, dict]:
    """Load all baseline JSON files from a directory."""
    baselines = {}
    baseline_path = Path(baseline_dir)
    if not baseline_path.exists():
        return baselines
    for f in baseline_path.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        op = data.get("operator", f.stem)
        baselines[op] = data
    return baselines


def load_current_results(path: str) -> dict[str, list[dict]]:
    """Load current benchmark results from CSV files.

    Returns {operator: [{label, mean_ms, ...}, ...]}.
    """
    import csv

    results = {}
    p = Path(path)
    files = list(p.glob("*.csv")) if p.is_dir() else [p]

    for f in files:
        op = f.stem.replace("bench_", "").replace("_turbo", "")
        rows = []
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "mean_ms" in row or "Mean(ms)" in row:
                    label = row.get("label", row.get("Case", row.get("config", str(row))))
                    mean_ms = float(row.get("mean_ms", row.get("Mean(ms)", 0)))
                    rows.append({"label": label, "mean_ms": mean_ms})
        if rows:
            results[op] = rows
    return results


def geomean(values: list[float]) -> float:
    """Compute geometric mean of positive values."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def check_operator(op: str, baseline: dict, current: list[dict]) -> tuple[str, list[str]]:
    """Check one operator for regressions.

    Returns (status, messages) where status is 'pass', 'warn', or 'fail'.
    """
    messages = []
    status = "pass"

    baseline_configs = {c["label"]: c["mean_ms"] for c in baseline.get("configs", [])}
    baseline_geomean = baseline.get("geomean_ms", 0)

    current_times = []
    for cfg in current:
        label = cfg["label"]
        cur_ms = cfg["mean_ms"]
        current_times.append(cur_ms)

        if label in baseline_configs:
            base_ms = baseline_configs[label]
            if base_ms > 0:
                regression = (cur_ms - base_ms) / base_ms
                if regression > FAIL_THRESHOLD:
                    messages.append(f"  FAIL  {label}: {base_ms:.3f} → {cur_ms:.3f} ms (+{regression:.1%})")
                    status = "fail"
                elif regression > WARN_THRESHOLD:
                    messages.append(f"  WARN  {label}: {base_ms:.3f} → {cur_ms:.3f} ms (+{regression:.1%})")
                    if status != "fail":
                        status = "warn"

    if current_times and baseline_geomean > 0:
        cur_geomean = geomean(current_times)
        geo_regression = (cur_geomean - baseline_geomean) / baseline_geomean
        messages.insert(0, f"  Geomean: {baseline_geomean:.3f} → {cur_geomean:.3f} ms ({geo_regression:+.1%})")
        if geo_regression > GEOMEAN_THRESHOLD:
            status = "fail"
            messages.insert(1, f"  FAIL  Geomean regression exceeds {GEOMEAN_THRESHOLD:.0%} threshold")

    return status, messages


def main():
    parser = argparse.ArgumentParser(description="Performance regression checker")
    parser.add_argument("--baseline", required=True, help="Baseline directory (benchmark/baselines/)")
    parser.add_argument("--current", required=True, help="Current results CSV file or directory")
    args = parser.parse_args()

    baselines = load_baseline(args.baseline)
    if not baselines:
        print(f"No baselines found in {args.baseline}. Run update_baseline.py first.")
        print("SKIP: No baselines to compare against.")
        sys.exit(0)

    current = load_current_results(args.current)
    if not current:
        print(f"No results found in {args.current}.")
        sys.exit(1)

    overall_status = "pass"
    print("=" * 60)
    print("  Performance Regression Check")
    print("=" * 60)

    for op, cur_data in sorted(current.items()):
        if op not in baselines:
            print(f"\n[{op}] No baseline — SKIP")
            continue

        status, msgs = check_operator(op, baselines[op], cur_data)
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[status]
        print(f"\n[{op}] {icon} {status.upper()}")
        for m in msgs:
            print(m)

        if status == "fail":
            overall_status = "fail"
        elif status == "warn" and overall_status != "fail":
            overall_status = "warn"

    print(f"\n{'=' * 60}")
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[overall_status]
    print(f"  Overall: {icon} {overall_status.upper()}")
    print("=" * 60)

    sys.exit(1 if overall_status == "fail" else 0)


if __name__ == "__main__":
    main()
