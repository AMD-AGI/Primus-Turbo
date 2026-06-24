#!/usr/bin/env python3
"""Benchmark grouped_variable_k_dot_scaled_kernel (WGRAD, FP8 dot-scaled).

Compares Triton (beta=0 and beta=1) vs ASM final.co and intermediate.co.
Reports latency (ms), TFLOPS mean/peak, and speedup vs Triton.

Shapes (gpt-oss MoE 20B):
  OUT_M=2880 x OUT_N=2880   (down proj WGRAD)
  OUT_M=2880 x OUT_N=5760   (gate/up proj WGRAD)

Usage:
    python3 bench.py
    python3 bench.py --co final.co --shape 2880x2880
    python3 bench.py --warmup 50 --iters 200
    python3 bench.py --no-triton
"""
import argparse
import ctypes
import pathlib
import struct
import sys

import torch

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
_HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
_HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)

KERNEL_NAME = "grouped_variable_k_dot_scaled_kernel"
THREADS     = 1024
LDS_BYTES   = 65536

G            = 32
TOTAL_TOKENS = 131_072


def load_co(path: pathlib.Path):
    hip = ctypes.CDLL("libamdhip64.so")
    mod = ctypes.c_void_p()
    rc = hip.hipModuleLoad(ctypes.byref(mod), str(path).encode())
    if rc != 0:
        return None, None, None
    func = ctypes.c_void_p()
    rc = hip.hipModuleGetFunction(ctypes.byref(func), mod, KERNEL_NAME.encode())
    if rc != 0:
        return None, None, None
    return hip, mod, func


def launch(hip, func, lhs, rhs, ls, rs, offs, out):
    """Launch ASM kernel. ls/rs should already be pre-adjusted (x2.0) by caller for pipelined use."""
    g = offs.shape[0] - 1
    out_m, out_n = lhs.shape[1], rhs.shape[1]
    buf = ctypes.create_string_buffer(96)
    struct.pack_into("<QQQQQQ", buf, 0,
                     lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(),
                     ls.data_ptr(), rs.data_ptr(),
                     offs.data_ptr())
    struct.pack_into("<iiiiiiii", buf, 48,
                     g, out_m, out_n, lhs.stride(0), rhs.stride(0),
                     out.stride(0), out.stride(1), out.stride(2))
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    arg_size = ctypes.c_size_t(96)
    config = (ctypes.c_void_p * 5)(
        _HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(buf, ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_END,
    )
    stream = torch.cuda.current_stream().cuda_stream
    rc = hip.hipModuleLaunchKernel(
        func, num_cu, 1, 1, THREADS, 1, 1, LDS_BYTES,
        ctypes.c_void_p(stream), None, config,
    )
    if rc != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed rc={rc}")


def make_inputs(out_m, out_n):
    torch.manual_seed(42)
    from primus_turbo.pytorch.core.low_precision import float8_e4m3
    offs = torch.zeros(G + 1, dtype=torch.int64, device="cuda")
    offs[1:] = torch.arange(1, G + 1, dtype=torch.int64, device="cuda") * (TOTAL_TOKENS // G)
    lhs_bf16 = torch.randn(TOTAL_TOKENS, out_m, device="cuda", dtype=torch.bfloat16)
    rhs_bf16 = torch.randn(TOTAL_TOKENS, out_n, device="cuda", dtype=torch.bfloat16)
    ls = lhs_bf16.abs().max().float() / 448.0
    rs = rhs_bf16.abs().max().float() / 448.0
    lhs = (lhs_bf16.float() / ls).to(float8_e4m3)
    rhs = (rhs_bf16.float() / rs).to(float8_e4m3)
    return lhs, rhs, ls.view(1), rs.view(1), offs


def tflops(out_m, out_n, ms):
    flops = G * 2.0 * (TOTAL_TOKENS // G) * out_m * out_n
    return flops / (ms * 1e-3) / 1e12


def measure(label, fn, warmup, iters):
    print(f"  Warmup [{label}] ({warmup}) ...", end="", flush=True)
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
    print(" done.")
    return t0.elapsed_time(t1) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed-co", type=pathlib.Path,
                        default=SCRIPT_DIR / "intermediate.co",
                        help="ASM .co (beta=0, cbsz fixed) — default: intermediate.co")
    parser.add_argument("--beta1-co", type=pathlib.Path,
                        default=SCRIPT_DIR / "final.co",
                        help="ASM .co (beta=1) — default: final.co")
    parser.add_argument("--shape", choices=["2880x2880", "2880x5760", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-triton", action="store_true")
    args = parser.parse_args()

    shapes = [(2880, 2880), (2880, 5760)] if args.shape == "all" else [
        tuple(int(x) for x in args.shape.split("x"))
    ]

    print(f"\n{'='*72}")
    print("  WGRAD grouped GEMM benchmark: grouped_variable_k_dot_scaled_kernel")
    print(f"  ASM fixed (beta=0): {args.fixed_co}")
    print(f"  ASM beta=1:         {args.beta1_co}")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print(f"{'='*72}\n")

    # Load ASM fixed (beta=0) = intermediate.co
    hip_b0 = func_b0 = None
    if args.fixed_co.exists():
        hip_b0, _, func_b0 = load_co(args.fixed_co)
        if func_b0:
            print(f"  Loaded ASM fixed (beta=0): {args.fixed_co.name}")
        else:
            print(f"  WARNING: could not load {args.fixed_co}")
    else:
        print(f"  SKIP ASM fixed: {args.fixed_co} not found  (run: bash build.sh)")

    # Load ASM beta=1 = final.co
    hip_b1 = func_b1 = None
    if args.beta1_co.exists():
        hip_b1, _, func_b1 = load_co(args.beta1_co)
        if func_b1:
            print(f"  Loaded ASM beta=1:         {args.beta1_co.name}")
        else:
            print(f"  WARNING: could not load {args.beta1_co}")
    else:
        print(f"  SKIP ASM beta=1: {args.beta1_co} not found  (run: bash build.sh)")

    for out_m, out_n in shapes:
        print(f"\n{'─'*72}")
        print(f"  Shape: OUT_M={out_m}  OUT_N={out_n}  G={G}  tokens={TOTAL_TOKENS}")
        t_flops = G * 2.0 * (TOTAL_TOKENS // G) * out_m * out_n / 1e12
        print(f"  FLOPs: {t_flops:.4f} TFLOPS per call")
        print(f"{'─'*72}")

        lhs, rhs, ls, rs, offs = make_inputs(out_m, out_n)
        ls_adj = ls * 2.0
        rs_adj = rs * 2.0

        ms_triton_b0 = ms_triton_b1 = None
        ms_asm_b0 = ms_asm_b1 = None

        if not args.no_triton:
            from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
                grouped_gemm_fp8_tensorwise_variable_k_triton_kernel as triton_fn,
            )
            def triton_b0_fn():
                triton_fn(lhs, rhs, ls, rs, offs, out_dtype=torch.bfloat16)
            ms_triton_b0 = measure("Triton (beta=0)", triton_b0_fn, args.warmup, args.iters)

            C_buf = torch.randn(G, out_m, out_n, device="cuda", dtype=torch.bfloat16) * 0.01
            def triton_b1_fn():
                triton_fn(lhs, rhs, ls, rs, offs, out=C_buf, beta=1.0)
            ms_triton_b1 = measure("Triton (beta=1)", triton_b1_fn, args.warmup, args.iters)

        if func_b0:
            out_b0 = torch.empty((G, out_m, out_n), device="cuda", dtype=torch.bfloat16)
            def asm_b0_fn():
                launch(hip_b0, func_b0, lhs, rhs, ls_adj, rs_adj, offs, out_b0)
            ms_asm_b0 = measure("ASM fixed (beta=0)", asm_b0_fn, args.warmup, args.iters)

        if func_b1:
            out_b1 = torch.randn(G, out_m, out_n, device="cuda", dtype=torch.bfloat16) * 0.01
            def asm_b1_fn():
                launch(hip_b1, func_b1, lhs, rhs, ls_adj, rs_adj, offs, out_b1)
            ms_asm_b1 = measure("ASM beta=1", asm_b1_fn, args.warmup, args.iters)

        print(f"\n{'='*72}")
        print(f"  RESULTS  OUT_M={out_m} OUT_N={out_n}")
        print(f"{'='*72}")
        rows = [
            ("Triton (beta=0)", ms_triton_b0),
            ("ASM fixed (beta=0)", ms_asm_b0),
            ("Triton (beta=1)", ms_triton_b1),
            ("ASM beta=1", ms_asm_b1),
        ]
        for label, ms in rows:
            if ms is None:
                continue
            tf = tflops(out_m, out_n, ms)
            print(f"  {label:<22}  {ms:.4f} ms  {tf:.2f} TFLOPS")

        print(f"  {'─'*60}")
        if ms_triton_b0 and ms_asm_b0:
            ratio = ms_triton_b0 / ms_asm_b0
            pct = (ratio - 1) * 100
            print(f"  ASM fixed vs Triton (beta=0):  {ratio:.4f}x ({'+' if pct >= 0 else ''}{pct:.2f}%)")
        if ms_triton_b1 and ms_asm_b1:
            ratio = ms_triton_b1 / ms_asm_b1
            pct = (ratio - 1) * 100
            print(f"  ASM beta=1 vs Triton (beta=1): {ratio:.4f}x ({'+' if pct >= 0 else ''}{pct:.2f}%)")
        print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
