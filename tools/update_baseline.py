#!/usr/bin/env python3
"""Update performance baselines from benchmark results.

Usage:
    python3 tools/update_baseline.py --results output/bench_attention_turbo.csv --operator attention
    python3 tools/update_baseline.py --results output/ --auto
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch


def get_gpu_info() -> str:
    """Get current GPU name."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        match = re.search(r"(MI\d+[A-Za-z]*)", name)
        return match.group(1) if match else name
    return "unknown"


def geomean(values: list[float]) -> float:
    """Compute geometric mean of positive values."""
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    log_sum = sum(math.log(v) for v in positive)
    return math.exp(log_sum / len(positive))


def parse_csv(csv_path: str) -> list[dict]:
    """Parse a benchmark CSV into config dicts."""
    configs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", row.get("Case", row.get("config", "")))
            mean_ms = float(row.get("mean_ms", row.get("Mean(ms)", 0)))
            std_ms = float(row.get("std_ms", row.get("Std(ms)", 0)))
            if mean_ms > 0:
                configs.append({
                    "label": label,
                    "mean_ms": round(mean_ms, 4),
                    "std_ms": round(std_ms, 4),
                })
    return configs


def infer_operator(filename: str) -> str:
    """Infer operator name from CSV filename."""
    name = Path(filename).stem
    name = name.replace("bench_", "").replace("_turbo", "")
    return name


def create_baseline(operator: str, configs: list[dict], gpu: str) -> dict:
    """Create a baseline JSON structure."""
    times = [c["mean_ms"] for c in configs]
    return {
        "operator": operator,
        "gpu": gpu,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "num_configs": len(configs),
        "geomean_ms": round(geomean(times), 4),
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser(description="Update performance baselines")
    parser.add_argument("--results", required=True, help="Results CSV file or directory")
    parser.add_argument("--output", default="benchmark/baselines/", help="Output baseline directory")
    parser.add_argument("--operator", default=None, help="Operator name (auto-inferred if not given)")
    parser.add_argument("--auto", action="store_true", help="Process all CSVs in directory")
    args = parser.parse_args()

    gpu = get_gpu_info()
    os.makedirs(args.output, exist_ok=True)

    results_path = Path(args.results)
    if results_path.is_dir():
        csv_files = list(results_path.glob("*.csv"))
    else:
        csv_files = [results_path]

    if not csv_files:
        print(f"No CSV files found in {args.results}")
        sys.exit(1)

    for csv_file in csv_files:
        operator = args.operator or infer_operator(csv_file.name)
        configs = parse_csv(str(csv_file))

        if not configs:
            print(f"  SKIP {csv_file.name}: no valid data")
            continue

        baseline = create_baseline(operator, configs, gpu)
        out_path = os.path.join(args.output, f"{operator}.json")
        with open(out_path, "w") as f:
            json.dump(baseline, f, indent=2)

        print(f"  ✅ {operator}: {len(configs)} configs, geomean={baseline['geomean_ms']:.3f} ms → {out_path}")

    print(f"\nBaselines updated in {args.output}")


if __name__ == "__main__":
    main()
