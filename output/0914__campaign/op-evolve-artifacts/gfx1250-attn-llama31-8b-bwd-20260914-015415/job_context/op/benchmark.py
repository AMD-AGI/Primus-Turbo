"""Measure one implementation on one shape. Print numbers. Decide nothing.

    python op/benchmark.py --impl baseline --shape b4_s8192_hq32_hkv8_d128
    python op/benchmark.py --impl beat --shape all --iters 50 --json

`--impl` takes `baseline`, `current`, `beat`, `eager`, or a path to a directory
containing an `impl.py`.

THIS FILE PRINTS NO VERDICT. No pass, no fail, no "faster", no target
comparison. `op/validation.py` is the only thing in this job that decides
anything, and it decides by calling this file. The reason for the split is that
this file is also run under rocprofv3, on deliberately corrupted input, and at
iteration counts chosen to probe stability -- in every one of those a verdict
would be meaningless, and a verdict that is meaningless in three of its four
callers is a bug waiting to be quoted.

It also does no correctness checking, deliberately: a benchmark that refuses to
run on corrupted input cannot be used to measure the effect of corrupting input.
Correctness lives in `op/ut/` and is enforced by `op/validation.py`, which gates
every number it reads on the check for the SAME run.

MEASUREMENT DEFINITION -- fixed here, in the benchmark, rather than in the
operator's head, and not to be changed mid-sweep (a change of measurement
definition moves the score and must be reported separately from a speedup):

  statistic   MEDIAN of `--iters` timed repetitions. Chosen once. Median and
              best-of-N are both drift-robust and are NOT interchangeable;
              mixing them across a comparison invents differences.
  warmup      CONTINUOUS load for `--warmup-s` SECONDS, not a count of
              iterations. Small workloads are dominated by clock and pipeline
              transients and a cold start can read 40% off. Seconds, because
              what has to be reached is a clock state, not a cache state.
  no pauses   No sleep and no host synchronise inside the timed loop. Sleeping
              to cool the device makes readings both lower and less stable, and
              a per-iteration sync serialises enqueue against execution and
              produces a bimodal distribution that reads like a governor effect.
  fwd/bwd     Timed SEPARATELY, as `op.reference.api` requires. The backward is
              73-76% of this op; a combined figure hides which half moved, and
              every candidate in this job moves one half at a time.
  FLOP/bytes  From `tools/op_flops.py`, IMPORTED, never reimplemented and never
              copied. One file, one place. Backward FLOP is that file's
              `backward=True`, which is 2.5x the forward and is BACKWARD ONLY --
              it is not a fwd+bwd total, and the two bases differ by 1.4x.

Ordering across arms is the CALLER's job and it matters: run the arms
palindromically (A B C C B A) so no arm is always first. `--impl all` does that.
"""

import argparse
import json
import os
import statistics
import sys
import time

# Hazard 3: hipBLASLt measures 91.5 TFLOP/s against Triton's 1002.7 on this image.
# It must never be on a path that produces a number here. Set before torch imports.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import torch  # noqa: E402

_OP = os.path.dirname(os.path.abspath(__file__))
_JOB_CONTEXT = os.path.dirname(_OP)
# The shipped tools/ directory, the one dependency outside job_context that is allowed.
_TOOLS = os.path.abspath(os.path.join(_JOB_CONTEXT, "..", "..", "..", "tools"))
sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.join(_OP, "ut"))

import op_flops  # noqa: E402
from shapes import ALL_SHAPES  # noqa: E402

STATISTIC = "median"
DEFAULT_ITERS = 30
DEFAULT_WARMUP_S = 3.0

BUILTIN_IMPLS = ("baseline", "current", "beat", "eager")


def resolve_impl_dir(name):
    return name if os.path.sep in name else os.path.join(_OP, name)


def load_impl(name):
    import importlib.util

    d = resolve_impl_dir(name)
    path = os.path.join(d, "impl.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no impl.py under {d}")
    mod_name = "impl_" + os.path.basename(os.path.normpath(d))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_shape(name):
    for s in ALL_SHAPES:
        if s["name"] == name:
            return s
    raise KeyError(f"unknown shape {name!r}; have {[s['name'] for s in ALL_SHAPES]}")


def counts(shape):
    """FLOP and byte counts, from tools/op_flops.py. Imported, never reimplemented."""
    common = dict(
        batch=shape["batch"],
        heads=shape["heads_q"],
        seqlen_q=shape["seqlen_q"],
        seqlen_kv=shape["seqlen_kv"],
        head_dim=shape["head_dim"],
        kv_heads=shape["heads_kv"],
        causal="bottom-right" if shape["causal"] else None,
        dtype="bf16",
        window_left=shape.get("window_left", -1),
    )
    return op_flops.attention(backward=False, **common), op_flops.attention(backward=True, **common)


def make_inputs(shape, device="cuda", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    b, d = shape["batch"], shape["head_dim"]
    sq, skv = shape["seqlen_q"], shape["seqlen_kv"]
    hq, hkv = shape["heads_q"], shape["heads_kv"]
    kw = dict(device=device, dtype=torch.bfloat16, generator=gen)
    q = torch.randn(b, sq, hq, d, **kw).requires_grad_(True)
    k = torch.randn(b, skv, hkv, d, **kw).requires_grad_(True)
    v = torch.randn(b, skv, hkv, d, **kw).requires_grad_(True)
    do = torch.randn(b, sq, hq, d, **kw)
    return q, k, v, do


def _time_continuous(fn, iters):
    """Median ms over `iters`, timed with CUDA events, no host sync inside the loop.

    Events are recorded back to back so the device never drains between
    repetitions -- the loop is the continuous load, not a sequence of cold
    starts. The single synchronise is after the last record.
    """
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _warm(fn, seconds):
    """Continuous load for wall-clock seconds, so the clock reaches steady state."""
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        for _ in range(4):
            fn()
            n += 1
        torch.cuda.synchronize()
    return n


def measure(impl_name, shape, iters=DEFAULT_ITERS, warmup_s=DEFAULT_WARMUP_S, seed=0):
    mod = load_impl(impl_name)
    q, k, v, do = make_inputs(shape, seed=seed)
    causal = shape["causal"]

    fwd = lambda: mod.attention(q, k, v, causal=causal)  # noqa: E731
    # First call builds/compiles. Outside every timed region and outside warmup.
    out = fwd()
    torch.cuda.synchronize()

    # torch.autograd.grad, NOT out.backward(). Measured at op setup on the job
    # shape: `.backward()` after clearing .grad reads a median of 16.6 ms with
    # outliers to 132 ms, while autograd.grad on the identical graph reads 10.21
    # ms on a flat plateau. The difference is not the op -- it is grad
    # allocation and accumulation churn in the timed region, which the unsynced
    # loop lets pile up. autograd.grad returns the gradients and drops them, so
    # the allocator reaches a steady state and what is left is the kernel.
    def bwd():
        torch.autograd.grad(out, (q, k, v), do, retain_graph=True)

    bwd()
    torch.cuda.synchronize()

    warm_fwd = _warm(fwd, warmup_s)
    fwd_ms = _time_continuous(fwd, iters)
    warm_bwd = _warm(bwd, warmup_s)
    bwd_ms = _time_continuous(bwd, iters)

    fwd_c, bwd_c = counts(shape)
    f_ms, b_ms = statistics.median(fwd_ms), statistics.median(bwd_ms)

    def spread(xs):
        """(p90 - p10) / median, in percent.

        Robust rather than max-min: the first few repetitions of an unsynced
        loop are still settling and a single outlier would otherwise dominate
        the number. min and max are reported separately so nothing is hidden by
        the choice.
        """
        ys = sorted(xs)
        m = statistics.median(ys)
        p10 = ys[max(0, int(0.10 * len(ys)) - 1)]
        p90 = ys[min(len(ys) - 1, int(0.90 * len(ys)))]
        return (p90 - p10) / m * 100.0 if m else float("nan")

    fingerprint = mod.fingerprint() if hasattr(mod, "fingerprint") else {}

    return {
        "impl": impl_name,
        "shape": shape["name"],
        "statistic": STATISTIC,
        "iters": iters,
        "warmup_s": warmup_s,
        "warmup_calls_fwd": warm_fwd,
        "warmup_calls_bwd": warm_bwd,
        "fwd_ms": f_ms,
        "bwd_ms": b_ms,
        "total_ms": f_ms + b_ms,
        "fwd_spread_pct": spread(fwd_ms),
        "bwd_spread_pct": spread(bwd_ms),
        "fwd_min_ms": min(fwd_ms),
        "fwd_max_ms": max(fwd_ms),
        "bwd_min_ms": min(bwd_ms),
        "bwd_max_ms": max(bwd_ms),
        "fwd_tflops": fwd_c.flop / (f_ms * 1e-3) / 1e12,
        "bwd_tflops": bwd_c.flop / (b_ms * 1e-3) / 1e12,
        "fwd_gbps": fwd_c.bytes_min / (f_ms * 1e-3) / 1e9,
        "bwd_gbps": bwd_c.bytes_min / (b_ms * 1e-3) / 1e9,
        "fwd_flop": fwd_c.flop,
        "bwd_flop": bwd_c.flop,
        "fwd_bytes": fwd_c.bytes_min,
        "bwd_bytes": bwd_c.bytes_min,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "fingerprint": fingerprint,
    }


def print_row(r):
    """Parseable key=value, one metric per line. No verdict."""
    print(f"[bench] impl={r['impl']} shape={r['shape']}")
    print(f"[bench]   statistic={r['statistic']} iters={r['iters']} warmup_s={r['warmup_s']}")
    print(f"[bench]   fwd_ms={r['fwd_ms']:.4f} fwd_tflops={r['fwd_tflops']:.2f} fwd_gbps={r['fwd_gbps']:.1f} fwd_spread_pct={r['fwd_spread_pct']:.2f}")
    print(f"[bench]   bwd_ms={r['bwd_ms']:.4f} bwd_tflops={r['bwd_tflops']:.2f} bwd_gbps={r['bwd_gbps']:.1f} bwd_spread_pct={r['bwd_spread_pct']:.2f}")
    print(f"[bench]   fwd_min_ms={r['fwd_min_ms']:.4f} fwd_max_ms={r['fwd_max_ms']:.4f} bwd_min_ms={r['bwd_min_ms']:.4f} bwd_max_ms={r['bwd_max_ms']:.4f}")
    print(f"[bench]   total_ms={r['total_ms']:.4f} peak_mem_gib={r['peak_mem_gib']:.2f}")
    if r["fingerprint"]:
        print(f"[bench]   fingerprint={json.dumps(r['fingerprint'], sort_keys=True)}")


def main():
    ap = argparse.ArgumentParser(description="Measure one implementation on one shape.")
    ap.add_argument("--impl", default="baseline",
                    help="baseline | current | beat | eager | path to a dir with impl.py; "
                         "or 'all' for baseline,beat run palindromically")
    ap.add_argument("--shape", default="b4_s8192_hq32_hkv8_d128",
                    help="shape name, or 'all' for every shape in op/ut/shapes.py")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="also emit one JSON object per row")
    args = ap.parse_args()

    shapes = ALL_SHAPES if args.shape == "all" else [get_shape(args.shape)]
    if args.impl == "all":
        # Palindromic: every arm has the same mean position, so drift cannot
        # convict whichever one ran first.
        impls = ["baseline", "beat", "beat", "baseline"]
    else:
        impls = [args.impl]

    for shape in shapes:
        for impl in impls:
            r = measure(impl, shape, args.iters, args.warmup_s, args.seed)
            print_row(r)
            if args.json:
                print("[json] " + json.dumps(r, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
