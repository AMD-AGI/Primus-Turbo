"""Quick correctness + benchmark template for representative shapes.

This is the reference template referenced by SKILL.md. The kernel-optimize
loop copies it to ``<campaign_dir>/quick_test_bench.py`` during
PREPARE_ENVIRONMENT, sets ``<target_backend>``, and fills ``SHAPES`` after
BASELINE. Adapt the op/config for your target, but keep the ``--summary-csv``
writer and the ``SUMMARY_CSV_HEADER`` column schema verbatim — BASELINE
(round-1) and every VALIDATE round read it back with the same CSV reader every
round (the optional ``mcp__turbo__parse_bench_csv`` MCP helper if it is
configured, otherwise a plain ``csv.DictReader``); any column drift silently
disables the per-shape regression gate.

This harness reuses the same ``a`` / ``b`` / ``grad_out`` tensors across the
timing loop (the benchmark idiom), so it CANNOT detect a W2/W3 ``id(...)``-keyed
activation/grad_out cache — that pattern looks fast here but has a ~0 hit rate
in real training. Gate it with iteration_rules.mdc Rule 11's fresh-tensor
8-step simulation, not with this script.

Usage:
    python quick_test_bench.py [--repeats N] [--iters-per-repeat M]
        [--summary-csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("PRIMUS_TURBO_GEMM_BACKEND", "<target_backend>")
if os.environ["PRIMUS_TURBO_GEMM_BACKEND"].startswith("<"):
    raise SystemExit(
        "Replace <target_backend> in quick_test_bench.py with the campaign "
        "backend (e.g. TRITON / CK / TURBO) during PREPARE_ENVIRONMENT, or set "
        "PRIMUS_TURBO_GEMM_BACKEND in the environment before running."
    )

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

SHAPES: List[Dict[str, Any]] = [
    # Fill from representative_shapes in manifest after BASELINE
]

CONFIG = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.BLOCKWISE,
    block_size=128,
)
SNR_THRESHOLD = 25.0

# Canonical CSV header used by BASELINE and every VALIDATE. Do NOT
# reorder or rename — the round-comparison step reads these columns
# verbatim (via the optional mcp__turbo__parse_bench_csv helper, or a
# plain csv.DictReader) and any drift silently disables the per-shape
# regression gate between rounds. "B" is a vestigial batch dim kept for
# schema stability (always 1 for this GEMM template).
#
# NOTE: these columns are GEMM-class specific (M/N/K, *TFLOPS,
# out_snr/da_snr/db_snr). For a non-GEMM op, swap the GEMM-only columns
# for the ones your primary_metric needs (e.g. "Forward GB/s" /
# "Backward GB/s" for a memory-bound op), keep label/Check/*_stddev,
# then freeze the schema at BASELINE and update whichever CSV reader you
# use to match.
SUMMARY_CSV_HEADER: List[str] = [
    "label",
    "B",
    "M",
    "N",
    "K",
    "Check",
    "Forward TFLOPS",
    "Forward TFLOPS_stddev",
    "Backward TFLOPS",
    "Backward TFLOPS_stddev",
    "Forward Time (ms)",
    "Backward Time (ms)",
    "out_snr",
    "da_snr",
    "db_snr",
]


def compute_snr(ref: torch.Tensor, test: torch.Tensor) -> float:
    noise = test.float() - ref.float()
    signal_power = ref.float().norm() ** 2
    noise_power = noise.norm() ** 2
    if noise_power.item() == 0.0:
        return float("inf")
    return float(10 * torch.log10(signal_power / noise_power).item())


def _timed_ms(fn, iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_one(shape: Dict[str, Any], repeats: int, iters_per_repeat: int) -> Dict[str, Any]:
    torch.manual_seed(0)  # stable inputs across rounds
    M = int(shape["M"])
    N = int(shape["N"])
    K = int(shape["K"])
    dtype = shape.get("dtype", torch.bfloat16)
    label = str(shape.get("label", f"M{M}_N{N}_K{K}"))
    device = "cuda"

    a = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.gemm_fp8(a, b, trans_b=True, config=CONFIG)
    grad_out = torch.randn_like(out)
    with torch.no_grad():
        ref = a.detach().float() @ b.detach().float().T
    out_snr = compute_snr(ref.to(dtype), out.detach())
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    (a_ref.float() @ b_ref.float().T).backward(grad_out.float())
    out.backward(grad_out, retain_graph=True)
    da_snr = compute_snr(a_ref.grad, a.grad)
    db_snr = compute_snr(b_ref.grad, b.grad)
    a.grad = None
    b.grad = None
    correct = all(s > SNR_THRESHOLD for s in (out_snr, da_snr, db_snr))

    fwd_fn = lambda: turbo.ops.gemm_fp8(a, b, trans_b=True, config=CONFIG)
    out_for_bwd = fwd_fn()
    bwd_fn = lambda: out_for_bwd.backward(grad_out, retain_graph=True)

    fwd_ms_samples, bwd_ms_samples = [], []
    total_repeats = max(2, repeats + 1)  # always have >=1 warm-up + >=1 timed
    for r in range(total_repeats):
        fwd_ms = _timed_ms(fwd_fn, iters_per_repeat)
        bwd_ms = _timed_ms(bwd_fn, iters_per_repeat)
        if r == 0:
            continue  # warm-up
        fwd_ms_samples.append(fwd_ms)
        bwd_ms_samples.append(bwd_ms)

    fwd_flops = 2.0 * M * N * K
    bwd_flops = 2.0 * fwd_flops
    fwd_tflops_samples = [fwd_flops / (t * 1e-3) / 1e12 for t in fwd_ms_samples]
    bwd_tflops_samples = [bwd_flops / (t * 1e-3) / 1e12 for t in bwd_ms_samples]

    def mean_std(xs):
        m = statistics.fmean(xs)
        s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return m, s

    fwd_ms_mean, _ = mean_std(fwd_ms_samples)
    bwd_ms_mean, _ = mean_std(bwd_ms_samples)
    fwd_tflops_mean, fwd_tflops_std = mean_std(fwd_tflops_samples)
    bwd_tflops_mean, bwd_tflops_std = mean_std(bwd_tflops_samples)
    return {
        "label": label,
        "B": 1,
        "M": M,
        "N": N,
        "K": K,
        "Check": "PASS" if correct else "FAIL",
        "Forward TFLOPS": fwd_tflops_mean,
        "Forward TFLOPS_stddev": fwd_tflops_std,
        "Backward TFLOPS": bwd_tflops_mean,
        "Backward TFLOPS_stddev": bwd_tflops_std,
        "Forward Time (ms)": fwd_ms_mean,
        "Backward Time (ms)": bwd_ms_mean,
        "out_snr": out_snr,
        "da_snr": da_snr,
        "db_snr": db_snr,
    }


def _failed_row(shape: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Emit a canonical-schema FAIL row so BASELINE still sees the shape."""
    return {
        "label": str(shape.get("label", "?")),
        "B": int(shape.get("B", 1)),
        "M": int(shape.get("M", 0)),
        "N": int(shape.get("N", 0)),
        "K": int(shape.get("K", 0)),
        "Check": "FAIL",
        "Forward TFLOPS": float("nan"),
        "Forward TFLOPS_stddev": float("nan"),
        "Backward TFLOPS": float("nan"),
        "Backward TFLOPS_stddev": float("nan"),
        "Forward Time (ms)": float("nan"),
        "Backward Time (ms)": float("nan"),
        "out_snr": float("nan"),
        "da_snr": float("nan"),
        "db_snr": float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters-per-repeat", type=int, default=50)
    ap.add_argument("--summary-csv", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm device required.", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for s in SHAPES:
        try:
            rows.append(run_one(s, args.repeats, args.iters_per_repeat))
        except Exception as e:  # noqa: BLE001
            print(f"{s.get('label', str(s))}: ERROR {e}", file=sys.stderr)
            rows.append(_failed_row(s, str(e)))

    print(f"{'label':<22} {'Check':<5} {'Fwd TFLOPS':>12} {'Bwd TFLOPS':>12}")
    for r in rows:
        print(
            f"{r['label']:<22} {r['Check']:<5} " f"{r['Forward TFLOPS']:>12.2f} {r['Backward TFLOPS']:>12.2f}"
        )

    pass_fwd = [r["Forward TFLOPS"] for r in rows if r["Check"] == "PASS"]
    pass_bwd = [r["Backward TFLOPS"] for r in rows if r["Check"] == "PASS"]

    def geomean(xs):
        xs = [x for x in xs if x > 0 and not math.isnan(x)]
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0

    print(f"Geomean Fwd TFLOPS: {geomean(pass_fwd):.2f}")
    print(f"Geomean Bwd TFLOPS: {geomean(pass_bwd):.2f}")

    if args.summary_csv:
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADER)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in SUMMARY_CSV_HEADER})
        print(f"Summary CSV: {args.summary_csv}")

    all_pass = all(r["Check"] == "PASS" for r in rows)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
