#!/usr/bin/env python3
"""Benchmark _grouped_fp8_persistent_gemm_kernel (FWD, trans_b=False) vs Triton.

Shapes (gpt-oss MoE 20B):
  gate_up_fwd: a[131072, 2880] @ b[32, 2880, 5760] -> out[131072, 5760]
  down_fwd:    a[131072, 5760] @ b[32, 5760, 2880] -> out[131072, 2880]

Reports latency (ms), TFLOPS mean/median/peak, and speedup vs Triton.
Uses final_*.hsaco if present, falls back to base_*.hsaco.

Usage:
    python3 bench.py
    python3 bench.py --shape gate_up_fwd
    python3 bench.py --warmup 100 --iters 500
    python3 bench.py --no-triton
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

E       = 32
M_TOTAL = 131_072

ASM_KERNEL_NAME = "_grouped_fp8_persistent_gemm_kernel"
ASM_THREADS     = 512
ASM_LDS_BYTES   = 131_072

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)
_HIP = None


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
        hip.hipModuleGetFunction(ctypes.byref(func), mod, ASM_KERNEL_NAME.encode()),
        "hipModuleGetFunction",
    )
    return mod, func


def launch_fwd(func, a, b, out, a_scale, b_scale, group_offs, K, N):
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import _compute_tile_cumsum_kernel
    tcs = torch.empty(E + 1, device=a.device, dtype=torch.int32)
    _compute_tile_cumsum_kernel[(1,)](group_offs, tcs, E, (N + 255) // 256, BLOCK_SIZE_M=256)
    hip = _get_hip()
    buf = ctypes.create_string_buffer(96)
    struct.pack_into("<QQQQQQQ", buf, 0,
                     a.data_ptr(), b.data_ptr(), out.data_ptr(),
                     a_scale.data_ptr(), b_scale.data_ptr(),
                     group_offs.data_ptr(), tcs.data_ptr())
    struct.pack_into("<iiiiii", buf, 56, E, N, K, a.stride(0), b.stride(0), out.stride(0))
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    arg_size = ctypes.c_size_t(96)
    config = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.cast(buf, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    _hip_check(hip.hipModuleLaunchKernel(
        func, num_cu, 1, 1, ASM_THREADS, 1, 1, ASM_LDS_BYTES,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), None, config,
    ), "hipModuleLaunchKernel")
    torch.cuda.synchronize()


def make_inputs(K, N):
    torch.manual_seed(42)
    from primus_turbo.pytorch.core.low_precision import float8_e4m3
    a = torch.empty((M_TOTAL, K), dtype=float8_e4m3, device="cuda")
    b = torch.empty((E, K, N),    dtype=float8_e4m3, device="cuda")
    a.view(torch.uint8).random_(0, 64)
    b.view(torch.uint8).random_(0, 64)
    a_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    b_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    group_offs = torch.tensor([i * (M_TOTAL // E) for i in range(E + 1)],
                              dtype=torch.int64, device="cuda")
    return a, b, a_scale, b_scale, group_offs


def _tflops(K, N, ms):
    return 2 * M_TOTAL * K * N / (ms * 1e-3) / 1e12


def measure(label, fn, warmup, iters, K, N):
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
    mean_ms = t0.elapsed_time(t1) / iters
    SAMPLE = min(iters, 100)
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(SAMPLE)]
    for s, e in evs:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    sample_ms = [s.elapsed_time(e) for s, e in evs]
    print(" done.")
    return {
        "label": label,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(sample_ms),
        "min_ms": min(sample_ms),
        "tflops_mean": _tflops(K, N, mean_ms),
        "tflops_median": _tflops(K, N, statistics.median(sample_ms)),
        "tflops_peak": _tflops(K, N, min(sample_ms)),
    }


# shape_name -> (K, N, final_co, base_co)
SHAPES = {
    "gate_up_fwd": (2880, 5760, "final_gate_up_5760.hsaco", "base_gate_up_5760.hsaco"),
    "down_fwd":    (5760, 2880, "final_down_2880.hsaco",    "base_down_2880.hsaco"),
}


def print_results(shape_name, K, N, results):
    BAR = "=" * 88
    COL = 18
    flops = 2 * M_TOTAL * K * N / 1e12
    print(f"\n{BAR}")
    print(f"  BENCHMARK -- {shape_name}  K={K}  N={N}")
    print(f"  FLOPs/call={flops:.4f} TFLOPS  E={E}  M={M_TOTAL}")
    print(BAR)
    hdrs = [f"{'Metric':<20}"] + [f"{r['label'][:COL]:>{COL}}" for r in results]
    print("  " + "  ".join(hdrs))
    print(f"  {'-'*80}")
    def prow(name, vals):
        print(f"  {name:<20}  " + "  ".join(f"{v:>{COL}}" for v in vals))
    prow("Latency mean",    [f"{r['mean_ms']:.4f} ms"   for r in results])
    prow("Latency median",  [f"{r['median_ms']:.4f} ms" for r in results])
    prow("Latency min",     [f"{r['min_ms']:.4f} ms"    for r in results])
    print(f"  {'-'*20}  " + "  ".join(["-" * COL] * len(results)))
    prow("TFLOPS (mean)",   [f"{r['tflops_mean']:.2f}"   for r in results])
    prow("TFLOPS (median)", [f"{r['tflops_median']:.2f}" for r in results])
    prow("TFLOPS (peak)",   [f"{r['tflops_peak']:.2f}"   for r in results])
    print(BAR)
    if len(results) >= 2:
        ref = results[0]; test = results[-1]
        ratio = ref["mean_ms"] / test["mean_ms"]
        pct = (ratio - 1) * 100
        print(f"  SPEEDUP ({test['label']} vs {ref['label']}): "
              f"{ratio:.4f}x ({'+' if pct >= 0 else ''}{pct:.2f}%)")
        print(f"    {ref['tflops_mean']:.2f} TFLOPS -> {test['tflops_mean']:.2f} TFLOPS")
    print(BAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=list(SHAPES) + ["all"], default="all")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iters",  type=int, default=10000)
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--no-asm",    action="store_true")
    args = parser.parse_args()

    shapes = list(SHAPES.items()) if args.shape == "all" else [(args.shape, SHAPES[args.shape])]

    print(f"\nGrouped GEMM FP8 FWD benchmark")
    print(f"  warmup={args.warmup}  iters={args.iters}\n")

    asm_info = {}
    if not args.no_asm:
        for name, (K, N, final_co, base_co) in SHAPES.items():
            co_path = SCRIPT_DIR / final_co
            label = final_co
            if not co_path.exists():
                co_path = SCRIPT_DIR / base_co
                label = base_co + " (base)"
            if co_path.exists():
                print(f"Loading ASM: {co_path.name} ...")
                _, func = load_co(co_path)
                asm_info[name] = (func, label)
            else:
                print(f"  SKIP ASM {name}: not found")
        print()

    for shape_name, (K, N, _, _) in shapes:
        a, b, a_scale, b_scale, group_offs = make_inputs(K, N)
        out_asm = torch.empty((M_TOTAL, N), dtype=torch.bfloat16, device="cuda")
        results = []

        if not args.no_triton:
            from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
                grouped_gemm_fp8_tensorwise_triton_kernel,
            )
            def triton_fn():
                grouped_gemm_fp8_tensorwise_triton_kernel(
                    a, b, a_scale, b_scale, group_offs, trans_b=False,
                    out_dtype=torch.bfloat16,
                )
            results.append(measure("Triton", triton_fn, args.warmup, args.iters, K, N))

        if shape_name in asm_info:
            func, label = asm_info[shape_name]
            def asm_fn():
                launch_fwd(func, a, b, out_asm, a_scale, b_scale, group_offs, K, N)
            results.append(measure(label[:18], asm_fn, args.warmup, args.iters, K, N))

        if results:
            print_results(shape_name, K, N, results)


if __name__ == "__main__":
    main()
