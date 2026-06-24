#!/usr/bin/env python3
"""Benchmark fused_scaling_group_sum_routing_kernel vs Triton reference.

Softmax + argsort top-k over [s=32768, e=32] expert logits.
Reports latency (µs), throughput (calls/s), effective bandwidth (GB/s),
and speedup vs Triton.

Effective bytes/call:
  Reads:  input_logit[S,E]*4
  Writes: output_scores[S,E]*4 + output_topk_idx[S,K]*8 +
          output_raw_topk_logits[S,K]*4 + output_probs[S,E]*4 +
          output_routing_map[S,E]*4

Usage:
    python3 bench.py
    python3 bench.py --co final.co
    python3 bench.py --ref base.hsaco --co final.co
    python3 bench.py --no-triton --warmup 100 --iters 500
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import statistics
import struct
import sys

import torch

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

S = 32768; E = 32; G = 1; K = 4
SELECTED_GROUPS = 1
SCORE_FUNCTION  = "softmax"
SCALING_FACTOR  = 1.0
THREADS         = 256
LDS_BYTES       = 0
KERNEL_NAME     = "fused_scaling_group_sum_routing_kernel"

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)
_HIP = None


def _bytes_per_call():
    reads  = S * E * 4
    writes = S * E * 4 + S * K * 8 + S * K * 4 + S * E * 4 + S * E * 4
    return reads + writes


def _get_hip():
    global _HIP
    if _HIP is None:
        for name in ("libamdhip64.so", "libamdhip64.so.6"):
            try:
                _HIP = ctypes.CDLL(name); break
            except OSError:
                continue
        if _HIP is None:
            raise OSError("Cannot load libamdhip64.so")
    return _HIP


def _hip_check(rc, op):
    if rc != 0:
        raise RuntimeError(f"{op} failed (rc={rc})")


def load_co(path: pathlib.Path):
    hip = _get_hip()
    mod = ctypes.c_void_p()
    _hip_check(hip.hipModuleLoad(ctypes.byref(mod), str(path).encode()), "hipModuleLoad")
    func = ctypes.c_void_p()
    _hip_check(
        hip.hipModuleGetFunction(ctypes.byref(func), mod, KERNEL_NAME.encode()),
        f"hipModuleGetFunction({KERNEL_NAME})",
    )
    return mod, func


def _pack_kernargs(input_logit, output_scores, output_topk_idx,
                   output_raw_topk_logits, output_probs, output_routing_map,
                   scaling_factor, grid_x):
    buf = ctypes.create_string_buffer(328)
    struct.pack_into("<QQQQQQ", buf, 0,
                     input_logit.data_ptr(), output_scores.data_ptr(),
                     output_topk_idx.data_ptr(), output_raw_topk_logits.data_ptr(),
                     output_probs.data_ptr(), output_routing_map.data_ptr())
    struct.pack_into("<f",   buf, 48, scaling_factor)
    struct.pack_into("<QQ",  buf, 56, 0, 0)
    struct.pack_into("<III", buf, 72, grid_x, 1, 1)
    struct.pack_into("<HHH", buf, 84, THREADS, 1, 1)
    struct.pack_into("<HHH", buf, 90, 0, 0, 0)
    struct.pack_into("<QQQ", buf, 112, 0, 0, 0)
    struct.pack_into("<H",   buf, 136, 1)
    return buf


def launch_co(func, input_logit, outputs):
    output_scores, output_topk_idx, output_raw_topk_logits, output_probs, output_routing_map = outputs
    hip = _get_hip()
    args_buf = _pack_kernargs(input_logit, output_scores, output_topk_idx,
                              output_raw_topk_logits, output_probs,
                              output_routing_map, SCALING_FACTOR, S)
    arg_size = ctypes.c_size_t(328)
    config = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.cast(args_buf, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    _hip_check(hip.hipModuleLaunchKernel(
        func, S, 1, 1, THREADS, 1, 1, LDS_BYTES, None, None, config,
    ), "hipModuleLaunchKernel")


def make_outputs():
    return (
        torch.empty(S, E, dtype=torch.float32, device="cuda"),
        torch.ones( S, K, dtype=torch.int64,   device="cuda"),
        torch.empty(S, K, dtype=torch.float32, device="cuda"),
        torch.zeros(S, E, dtype=torch.float32, device="cuda"),
        torch.zeros(S, E, dtype=torch.int32,   device="cuda"),
    )


def run_triton(input_logit):
    from primus_turbo.pytorch.kernels.moe.fused_moe_router_impl import fused_moe_router_fwd
    return fused_moe_router_fwd(input_logit, S, E, G, K, SELECTED_GROUPS,
                                SCORE_FUNCTION, SCALING_FACTOR)


def measure(label, fn, warmup, iters):
    print(f"  Warmup  [{label}] ({warmup}) ...", end="", flush=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    print(" done.")
    print(f"  Measure [{label}] ({iters}) ...", end="", flush=True)
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
    print(" done.")
    bpc = _bytes_per_call()
    median_us = statistics.median(sample_us)
    min_us    = min(sample_us)
    return {
        "label": label,
        "mean_us": mean_us,
        "median_us": median_us,
        "min_us": min_us,
        "stdev_us": statistics.stdev(sample_us) if len(sample_us) > 1 else 0.0,
        "bw_gbs":      bpc / (median_us * 1e-6) / 1e9,
        "bw_peak_gbs": bpc / (min_us    * 1e-6) / 1e9,
        "calls_per_s": 1e6 / median_us,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--co",  type=pathlib.Path,
                        default=SCRIPT_DIR / "final.co",
                        help="Test .co to benchmark (default: final.co)")
    parser.add_argument("--ref", type=pathlib.Path,
                        default=SCRIPT_DIR / "base.hsaco",
                        help="Reference .hsaco (default: base.hsaco)")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iters",  type=int, default=10000)
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    bpc_gb = _bytes_per_call() / 1e9
    print(f"\nFused MoE Router kernel — performance benchmark")
    print(f"  s={S}  e={E}  k={K}  groups={G}  score_function={SCORE_FUNCTION}")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print(f"  Bytes/call={bpc_gb:.4f} GB")
    print()

    input_logit = torch.randn(S, E, dtype=torch.float32, device="cuda")
    results = []

    if not args.no_triton:
        # Triton JIT warmup compile
        _ = run_triton(input_logit); torch.cuda.synchronize()
        results.append(measure("Triton JIT", lambda: run_triton(input_logit),
                                args.warmup, args.iters))
        print()

    if args.ref.exists():
        _, ref_func = load_co(args.ref)
        ref_outs = make_outputs()
        results.append(measure(args.ref.name[:18],
                                lambda: (launch_co(ref_func, input_logit, ref_outs),
                                         torch.cuda.synchronize()),
                                args.warmup, args.iters))
        print()

    if not args.baselines_only and args.co.exists():
        _, co_func = load_co(args.co)
        co_outs = make_outputs()
        results.append(measure(args.co.name[:18],
                                lambda: (launch_co(co_func, input_logit, co_outs),
                                         torch.cuda.synchronize()),
                                args.warmup, args.iters))
        print()
    elif not args.baselines_only:
        print(f"  SKIP: {args.co} not found  (run: bash build.sh)")

    if not results:
        return

    BAR = "=" * 92
    COL = 18
    print(f"\n{BAR}")
    print(f"  BENCHMARK — fused_scaling_group_sum_routing_kernel")
    print(f"  s={S}  e={E}  k={K}  groups={G}  score={SCORE_FUNCTION}")
    print(f"  Bytes/call={bpc_gb:.4f} GB")
    print(BAR)
    hdrs = [f"{'Metric':<22}"] + [f"{r['label'][:COL]:>{COL}}" for r in results]
    print("  " + "  ".join(hdrs))
    print(f"  {'-'*80}")
    def prow(name, vals):
        print(f"  {name:<22}  " + "  ".join(f"{v:>{COL}}" for v in vals))
    prow("Latency mean",    [f"{r['mean_us']:.2f} µs"     for r in results])
    prow("Latency median",  [f"{r['median_us']:.2f} µs"   for r in results])
    prow("Latency min",     [f"{r['min_us']:.2f} µs"      for r in results])
    prow("Latency stdev",   [f"{r['stdev_us']:.2f} µs"    for r in results])
    print(f"  {'-'*22}  " + "  ".join(["-" * COL] * len(results)))
    prow("Eff. BW (median)", [f"{r['bw_gbs']:.2f} GB/s"      for r in results])
    prow("Eff. BW (peak)",   [f"{r['bw_peak_gbs']:.2f} GB/s" for r in results])
    prow("Throughput",       [f"{r['calls_per_s']:.0f} calls/s" for r in results])
    print(BAR)
    if len(results) >= 2:
        ref = results[0]
        for r in results[1:]:
            ratio = ref["median_us"] / r["median_us"]
            pct   = (ratio - 1.0) * 100
            print(f"  SPEEDUP ({r['label']} vs {ref['label']}): "
                  f"{ratio:.4f}x ({'+' if pct >= 0 else ''}{pct:.2f}%)")
            print(f"    {ref['bw_gbs']:.2f} GB/s -> {r['bw_gbs']:.2f} GB/s  "
                  f"({r['bw_gbs'] - ref['bw_gbs']:+.2f} GB/s)")
    print()


if __name__ == "__main__":
    main()
