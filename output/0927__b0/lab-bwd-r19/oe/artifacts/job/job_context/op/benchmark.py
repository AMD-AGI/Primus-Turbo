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
  ordering     BLOCKED (default --block 9 --lead 4): per round each arm runs `lead` untimed
               then `block` timed calls back to back; rounds are PALINDROMIC over arms
               (round r forward, r+1 backward), so every arm has the same mean position.
               --block 1 restores the old call-by-call palindromic interleave, where at
               prod a call inherits the card state (power/clock controller, ms-scale
               memory) left by the previous few calls of OTHER arms; identical code then
               reads differently by position and by which other arms share the process
               (B0 ruler audit, output/0927__b0/ruler/REPORT.md and ruler/bwd/REPORT.md).
               With --block 1, --iters is the timed call count per arm as before; with
               blocking it is rounded UP to a whole number of blocks.
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
import hashlib
import json
import os
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
    """Current shader clock of the card under test, or -1 if it cannot be read.

    B0 runs one container per card (fa-gN, OE_PHYS_GPU=N). rocm-smi reads sysfs for every
    card and its first sclk line is GPU[0], so read this card's pp_dpm_sclk directly.
    """
    phys = os.environ.get("OE_PHYS_GPU")
    if phys is not None:
        try:
            with open(f"/sys/class/drm/renderD{128 + 8 * int(phys)}/device/pp_dpm_sclk") as f:
                for line in f:
                    if line.rstrip().endswith("*"):
                        return int(line.split(":")[1].strip().split("Mhz")[0])
        except Exception:
            pass
        return -1
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


def tree_md5(path):
    """md5 over the arm's *.py AS LOADED: a path (e.g. op/current) is not an identity --
    the fwd job's op/current was re-installed mid-session once and a round-to-round code
    change read as a fake "bimodality"."""
    h = hashlib.md5()
    for f in sorted(Path(path).glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


def measure(shape, arms, fns, iters, warmup_seconds, causal=True, block=9, lead=4):
    q, k, v, do = make_inputs(shape, seed=0)
    # Was forward_reference(...). That is an fp32 Tensile GEMM at every shape including prod,
    # in a subprocess that runs every round, and it is the dispatch that cost a power cycle
    # on 2026-09-22. cached_forward falls back to it whenever the cache does not apply.
    from refcache_util import cached_forward
    o, lse = cached_forward(shape, q, k, v, causal=causal)
    torch.cuda.synchronize()
    flush = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

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

    def timed(label):
        flush.zero_()
        ev0.record()
        call(label)
        ev1.record()
        ev1.synchronize()
        return ev0.elapsed_time(ev1)

    if block <= 1:                             # legacy: arms interleaved call by call
        for i in range(iters):
            order = labels if i % 2 == 0 else labels[::-1]
            for label in order:
                times[label].append(timed(label))
    else:
        # BLOCKED: every timed call is preceded by >= `lead` calls of the SAME arm. At prod
        # a call carries the state left by the previous few calls; interleaving call by call
        # hands that to the neighbour arm, so identical code reads by position and by arm
        # set (ruler/REPORT.md, ruler/bwd/REPORT.md). Rounds stay palindromic over arms.
        rounds = -(-iters // block)
        for r in range(rounds):
            for label in (labels if r % 2 == 0 else labels[::-1]):
                for _ in range(lead):
                    timed(label)
                for _ in range(block):
                    times[label].append(timed(label))
    sclk1 = sclk_mhz()

    c = counts(shape)
    rows = []
    for label in labels:
        ts = sorted(times[label])
        med = ts[len(ts) // 2] if len(ts) % 2 else 0.5 * (ts[len(ts) // 2 - 1]
                                                          + ts[len(ts) // 2])
        secs = med / 1e3
        rows.append({
            "shape": shape, "arm": label, "stat": STATISTIC, "iters": len(ts),
            "latency_ms": med, "min_ms": ts[0], "max_ms": ts[-1],
            "tflops": c.flop / secs / 1e12, "bw_gbs": c.bytes_min / secs / 1e9,
            "flop": c.flop, "bytes_min": c.bytes_min,
            "sclk_start": sclk0, "sclk_end": sclk1, "causal": bool(causal),
            "order": "interleaved" if block <= 1 else f"blocked{block}+lead{lead}",
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
    ap.add_argument("--block", type=int, default=9,
                    help="timed calls per arm per round (<=1: legacy call-by-call interleave)")
    ap.add_argument("--lead", type=int, default=4, help="untimed same-arm calls before each block")
    ap.add_argument("--json", help="also write every row here")
    args = ap.parse_args()

    arms = resolve_arms([a for a in args.arms.split(",") if a], args.arm_path)
    if not arms:
        ap.error("no arms given")
    for label, path in arms:
        if not (path / "impl.py").is_file():
            ap.error(f"arm {label!r}: no impl.py under {path}")
        print(f"# arm {label} -> {path} md5={tree_md5(path)}")
    print(f"# statistic {STATISTIC}  iters {args.iters}  "
          f"warmup {args.warmup_seconds}s continuous  order "
          f"{'palindromic' if args.block <= 1 else f'blocked {args.block}+lead {args.lead}, palindromic rounds'}  "
          f"device {torch.cuda.get_device_properties(0).gcnArchName}")

    # r15 -- ONE module per arm for the WHOLE process, not one per shape.
    # Reloading inside measure() replaced sys.modules (ut/common.py:99-103, and
    # impl.py:17-22 for kernels/_env), which dropped the previous shape's flydsl
    # artifact chain: JitFunction -> _mem_cache -> CompiledArtifact -> GpuJitModule.
    # GpuJitModule.__del__ calls mgpuModuleUnload == hipModuleUnload
    # (flydsl/compiler/jit_executor.py:102,107). Python modules are reference CYCLES, so
    # that unload does not happen here -- it happens at the next generational GC pass,
    # which can land in the middle of the NEXT shape's launches. That is exactly the
    # intermittent fast->proxy fault: unique to the shape boundary, different address
    # every run, and it convicted two innocent arms in round 14.
    # The control was already in the tree: validation.py:117 loads the impl ONCE and
    # loops all three shapes with MORE allocator churn (empty_cache at :169) and does
    # not fault.
    fns = {label: load_impl(path) for label, path in arms}

    rows = []
    for shape in [s for s in args.shapes.split(",") if s]:
        rows += measure(shape, arms, fns, args.iters, args.warmup_seconds,
                        causal=not args.non_causal, block=args.block, lead=args.lead)
        for r in rows[-len(arms):]:
            print("RESULT " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
