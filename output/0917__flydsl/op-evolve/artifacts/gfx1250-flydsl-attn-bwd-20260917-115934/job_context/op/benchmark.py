#!/usr/bin/env python3
"""Measure one or more implementations on one or more shapes. It decides nothing.

    python3 benchmark.py --arms baseline,beat --shapes fast,proxy,prod
    python3 benchmark.py --arm-path cand=/abs/path/to/rounds/007/op --arms beat

Prints one parseable RESULT line per (shape, arm) and nothing that resembles a verdict:
no pass, no fail, no "faster", no threshold. `validation.py` is the only thing that judges,
and it gets its numbers by calling this module.

Method, fixed here and not negotiable per-run (op.shape.mode = sweep; the card is
VR-throttled to 1100 MHz and drifts 1100 -> 967 MHz inside a timing window):

  statistic    MEDIAN of per-iteration CUDA-event times. Chosen once, written down here,
               never mixed with best-of-N in one comparison.
  iterations   --iters, default 51 timed iterations per arm per shape. 20 was tried first
               and is NOT enough: at the `fast` shape (1.8 ms, launch-bound) two copies of
               the SAME directory disagreed by 7.8% at 20 iterations and by 0.011% at 101,
               so a 20-iteration median would have handed the evolve loop a phantom 8%
               regression to chase. 51 is where the smallest shape settles.
  warmup       --warmup-seconds of CONTINUOUS load per arm, seconds not iterations, with
               no sleeps anywhere: a paced or short warmup reads up to 40% off and is not
               uniform across candidates, which is enough to invert a comparison.
  ordering     PALINDROMIC. Iteration i runs the arms forward, i+1 backward, so every arm
               has the same mean position. Repeating `A B C` instead convicts whichever
               arm runs first.
  inputs       one realistic input per shape, shared by every arm. Arms are compared, data
               never is.
  clock        sclk is read before and after every shape and printed on every line, so a
               figure can never be quoted without the clock it was taken at.
  L2           flushed between timed iterations, outside the event window.

FLOP and byte counts come from tools/op_flops.py, imported from the shipped tools/
directory -- never recomputed inline and never copied into this job.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ut"))
sys.path.insert(0, str(HERE.parent.parent.parent.parent / "tools"))   # op-evolve/tools

import torch  # noqa: E402

import op_flops  # noqa: E402  -- the shipped tool, imported, not copied
from common import SHAPES, forward_reference, load_impl, make_inputs  # noqa: E402

STATISTIC = "median"
L2_FLUSH_MB = 256
ROCM_SMI = "/opt/venv/bin/rocm-smi"


def sclk_mhz():
    """Current shader clock, or -1 if it cannot be read. Never inferred from a profile."""
    try:
        out = subprocess.run([ROCM_SMI, "--showclocks"], capture_output=True, text=True,
                             timeout=30).stdout
        for line in out.splitlines():
            if "sclk" in line.lower() and "Mhz" in line:
                return int(line.split("(")[-1].split("Mhz")[0].strip())
    except Exception:
        pass
    return -1


def counts(shape):
    b, sq, skv, hq, hkv, d = SHAPES[shape]
    return op_flops.attention(batch=b, heads=hq, seqlen_q=sq, seqlen_kv=skv, head_dim=d,
                              kv_heads=hkv, causal="bottom-right", dtype="bf16",
                              backward=True)


def resolve_arms(names, paths):
    arms = []
    for spec in paths:
        label, _, p = spec.partition("=")
        arms.append((label, Path(p).resolve()))
    for name in names:
        arms.append((name, (HERE / name).resolve()))
    return arms


def measure(shape, arms, iters, warmup_seconds, causal=True):
    q, k, v, do = make_inputs(shape, seed=0)
    o, lse = forward_reference(q, k, v, causal=causal)
    torch.cuda.synchronize()
    flush = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

    fns = {label: load_impl(path) for label, path in arms}
    labels = [label for label, _ in arms]

    def call(label):
        return fns[label](do, q, k, v, o, lse, causal=causal)

    for label in labels:                       # build once, outside every timing window
        call(label)
    torch.cuda.synchronize()

    for label in labels:                       # continuous load, seconds, no pauses
        t_end = time.perf_counter() + warmup_seconds
        while time.perf_counter() < t_end:
            call(label)
        torch.cuda.synchronize()

    sclk0 = sclk_mhz()
    times = {label: [] for label in labels}
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for i in range(iters):
        order = labels if i % 2 == 0 else labels[::-1]
        for label in order:
            flush.zero_()
            ev0.record()
            call(label)
            ev1.record()
            ev1.synchronize()
            times[label].append(ev0.elapsed_time(ev1))
    sclk1 = sclk_mhz()

    c = counts(shape)
    rows = []
    for label in labels:
        ts = sorted(times[label])
        med = ts[len(ts) // 2] if len(ts) % 2 else 0.5 * (ts[len(ts) // 2 - 1]
                                                          + ts[len(ts) // 2])
        secs = med / 1e3
        rows.append({
            "shape": shape, "arm": label, "stat": STATISTIC, "iters": iters,
            "latency_ms": med, "min_ms": ts[0], "max_ms": ts[-1],
            "tflops": c.flop / secs / 1e12, "bw_gbs": c.bytes_min / secs / 1e9,
            "flop": c.flop, "bytes_min": c.bytes_min,
            "sclk_start": sclk0, "sclk_end": sclk1, "causal": bool(causal),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="", help="comma-separated names under op/")
    ap.add_argument("--arm-path", action="append", default=[],
                    help="LABEL=/abs/dir -- an arm addressed by PATH, repeatable")
    ap.add_argument("--shapes", default="fast,proxy,prod")
    ap.add_argument("--iters", type=int, default=51)
    ap.add_argument("--warmup-seconds", type=float, default=3.0)
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument("--json", help="also write every row here")
    args = ap.parse_args()

    arms = resolve_arms([a for a in args.arms.split(",") if a], args.arm_path)
    if not arms:
        ap.error("no arms given")
    for label, path in arms:
        if not (path / "impl.py").is_file():
            ap.error(f"arm {label!r}: no impl.py under {path}")
        print(f"# arm {label} -> {path}")
    print(f"# statistic {STATISTIC}  iters {args.iters}  "
          f"warmup {args.warmup_seconds}s continuous  order palindromic  "
          f"device {torch.cuda.get_device_properties(0).gcnArchName}")

    rows = []
    for shape in [s for s in args.shapes.split(",") if s]:
        rows += measure(shape, arms, args.iters, args.warmup_seconds,
                        causal=not args.non_causal)
        for r in rows[-len(arms):]:
            print("RESULT " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
