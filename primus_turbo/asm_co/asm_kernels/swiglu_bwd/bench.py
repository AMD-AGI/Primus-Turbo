#!/usr/bin/env python3
"""Benchmark swiglu_with_mask_bwd_kernel vs Triton reference.

Reports latency (µs), effective bandwidth (GB/s), GFLOPS, and speedup vs Triton.

Kernel shape: N=131072, H=4096  (gpt-oss MoE 20B)

Effective bytes/call (reads + writes):
  Reads:  grad_out[N,H]*2 + x[N,2H]*2 + probs[N,1]*4 + mask[N]*8
  Writes: grad_x[N,2H]*2 + grad_probs[N]*4
  Total:  N * (H*2 + 2H*2 + 4 + 8 + 2H*2 + 4)  =  N * (10H + 16)

FLOPs/call ~ 17 * N * H  (silu deriv + mul chain)

Usage:
    python3 bench.py
    python3 bench.py final.co
    python3 bench.py intermediate.co --iters 500
    python3 bench.py --no-triton
"""
import argparse
import ctypes
import os
import pathlib
import statistics
import struct
import sys

import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

HIP = ctypes.CDLL("libamdhip64.so")
HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)

KERNEL_NAME = b"swiglu_with_mask_bwd_kernel"
N, H        = 131072, 4096
BLOCK_SIZE  = 8192

BYTES_PER_CALL = N * (H * 2 + 2 * H * 2 + 4 + 8 + 2 * H * 2 + 4)
FLOPS_PER_CALL = 17 * N * H


def hip_check(rc, op=""):
    if rc != 0:
        raise RuntimeError(f"HIP {op} failed (rc={rc})")


def load_co(path):
    mod = ctypes.c_void_p()
    hip_check(HIP.hipModuleLoad(ctypes.byref(mod), str(path).encode()), "ModuleLoad")
    func = ctypes.c_void_p()
    hip_check(HIP.hipModuleGetFunction(ctypes.byref(func), mod, KERNEL_NAME), "GetFunction")
    return mod, func


def make_kernarg(go, x, p, rm, gx, gp):
    buf = ctypes.create_string_buffer(80)
    struct.pack_into("<QQQQQQ", buf, 0,
                     go.data_ptr(), x.data_ptr(), p.data_ptr(),
                     rm.data_ptr(), gx.data_ptr(), gp.data_ptr())
    struct.pack_into("<iii", buf, 48, go.stride(0), x.stride(0), gx.stride(0))
    struct.pack_into("<I",   buf, 60, 0)
    struct.pack_into("<QQ",  buf, 64, 0, 0)
    return buf


def launch(func, kernarg):
    arg_size = ctypes.c_size_t(80)
    config = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(kernarg, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    hip_check(
        HIP.hipModuleLaunchKernel(func, BLOCK_SIZE, 1, 1, 256, 1, 1, 16,
                                  None, None, config),
        "LaunchKernel",
    )


def make_tensors(seed=42):
    torch.manual_seed(seed)
    go = torch.randn(N, H,     dtype=torch.bfloat16, device="cuda")
    x  = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    p  = torch.rand(N, 1,      dtype=torch.float32,  device="cuda")
    rm = torch.ones(N,         dtype=torch.int64,    device="cuda")
    return go, x, p, rm


def measure(label, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    mean_us = t0.elapsed_time(t1) / iters * 1000

    SAMPLE = min(iters, 100)
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(SAMPLE)]
    for s, e in evs:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    sample_us = [s.elapsed_time(e) * 1000 for s, e in evs]

    median_us = statistics.median(sample_us)
    min_us    = min(sample_us)
    return {
        "label": label,
        "mean_us": mean_us,
        "median_us": median_us,
        "min_us": min_us,
        "stdev_us": statistics.stdev(sample_us) if len(sample_us) > 1 else 0.0,
        "bw_gbs":      BYTES_PER_CALL / (median_us * 1e-6) / 1e9,
        "bw_peak_gbs": BYTES_PER_CALL / (min_us    * 1e-6) / 1e9,
        "gflops":      FLOPS_PER_CALL / (median_us * 1e-6) / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("co_file", nargs="?",
                        default=str(SCRIPT_DIR / "final.co"),
                        help=".co file to benchmark (default: final.co)")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iters",  type=int, default=10000)
    parser.add_argument("--no-triton", action="store_true")
    args = parser.parse_args()

    co_path = pathlib.Path(args.co_file)
    if not co_path.is_absolute():
        co_path = SCRIPT_DIR / co_path

    bytes_gb = BYTES_PER_CALL / 1e9
    gflops_t = FLOPS_PER_CALL / 1e9
    print(f"\nswiglu_with_mask_bwd_kernel — performance benchmark")
    print(f"  N={N}  H={H}  BLOCK_SIZE={BLOCK_SIZE}")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print(f"  Bytes/call={bytes_gb:.3f} GB  FLOPs/call={gflops_t:.1f} GFLOP")
    print()

    go, x, p, rm = make_tensors()
    results = []

    if not args.no_triton:
        from primus_turbo.pytorch.kernels.activation.swiglu_impl import swiglu_bwd_with_probs
        print("── Triton JIT baseline ──")
        print(f"  Warmup ({args.warmup}) ...", end="", flush=True)
        def triton_fn():
            swiglu_bwd_with_probs(go, x, p, rm)
        results.append(measure("Triton JIT", triton_fn, args.warmup, args.iters))
        print(" done.\n")

    if co_path.exists():
        print(f"── {co_path.name} ──")
        mod, func = load_co(co_path)
        gx = torch.empty(N, 2 * H, dtype=torch.bfloat16, device="cuda")
        gp = torch.empty(N,        dtype=torch.float32,  device="cuda")
        kernarg = make_kernarg(go, x, p, rm, gx, gp)
        print(f"  Warmup ({args.warmup}) ...", end="", flush=True)
        def co_fn():
            launch(func, kernarg)
        results.append(measure(co_path.name[:18], co_fn, args.warmup, args.iters))
        print(" done.\n")
    else:
        print(f"  SKIP: {co_path} not found  (run: bash build.sh)")

    if not results:
        return

    BAR = "=" * 84
    COL = 18
    print(f"\n{BAR}")
    print(f"  BENCHMARK — swiglu_with_mask_bwd_kernel")
    print(f"  Shape: N={N}, H={H}  |  Bytes/call={bytes_gb:.3f} GB")
    print(BAR)
    hdrs = [f"{'Metric':<22}"] + [f"{r['label'][:COL]:>{COL}}" for r in results]
    print("  " + "  ".join(hdrs))
    print(f"  {'-'*76}")
    def prow(name, vals):
        print(f"  {name:<22}  " + "  ".join(f"{v:>{COL}}" for v in vals))
    prow("Latency mean",    [f"{r['mean_us']:.2f} µs"       for r in results])
    prow("Latency median",  [f"{r['median_us']:.2f} µs"     for r in results])
    prow("Latency min",     [f"{r['min_us']:.2f} µs"        for r in results])
    prow("Latency stdev",   [f"{r['stdev_us']:.2f} µs"      for r in results])
    print(f"  {'-'*22}  " + "  ".join(["-" * COL] * len(results)))
    prow("Eff. BW (median)", [f"{r['bw_gbs']:.1f} GB/s"      for r in results])
    prow("Eff. BW (peak)",   [f"{r['bw_peak_gbs']:.1f} GB/s" for r in results])
    prow("Throughput",       [f"{r['gflops']:.1f} GFLOPS"    for r in results])
    print(BAR)
    if len(results) >= 2:
        ref = results[0]; test = results[-1]
        ratio = ref["median_us"] / test["median_us"]
        pct   = (ratio - 1.0) * 100
        print(f"  SPEEDUP ({test['label']} vs {ref['label']}): "
              f"{ratio:.4f}x ({'+' if pct >= 0 else ''}{pct:.2f}%)")
        print(f"    {ref['bw_gbs']:.1f} GB/s -> {test['bw_gbs']:.1f} GB/s  "
              f"({test['bw_gbs'] - ref['bw_gbs']:+.1f} GB/s)")
    print()


if __name__ == "__main__":
    main()
