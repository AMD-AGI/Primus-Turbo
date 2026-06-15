#!/usr/bin/env python3
"""Performance benchmark: _grouped_fp8_persistent_gemm_kernel (DGRAD, trans_b=True).

Compares final .hsaco outputs against Triton baseline.

Shapes:
  down_dgrad:    a[131072, 2880] @ b[32, 2880, 2880]^T -> out[131072, 2880]
  gate_up_dgrad: a[131072, 5760] @ b[32, 2880, 5760]^T -> out[131072, 2880]

Usage:
    python3 bench.py
    python3 bench.py --shape down_dgrad
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
from dataclasses import dataclass

import torch

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

E = 32
M_TOTAL = 131_072

ASM_KERNEL_NAME = "_grouped_fp8_persistent_gemm_kernel"
ASM_THREADS     = 512
ASM_LDS_BYTES   = 131_072

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)
_HIP = None


@dataclass
class DgradShape:
    name: str
    K: int
    N: int

    @property
    def flops(self) -> int:
        return 2 * M_TOTAL * self.K * self.N

    @property
    def default_co(self) -> pathlib.Path:
        suffix = "down_2880" if self.K == 2880 else "gate_up_5760"
        return SCRIPT_DIR / f"final_{suffix}.hsaco"


SHAPES = {
    "down_dgrad":    DgradShape("down_dgrad",    K=2880, N=2880),
    "gate_up_dgrad": DgradShape("gate_up_dgrad", K=5760, N=2880),
}


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
        f"hipModuleGetFunction({ASM_KERNEL_NAME})",
    )
    return mod, func


def launch_dgrad(func, a, b, out, a_scale, b_scale, group_offs, shape: DgradShape):
    """Kernarg layout (96 bytes) — matches launch_asm_co_fwd_dgrad in launcher.py."""
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import _compute_tile_cumsum_kernel
    blk_m, blk_n = 256, 256
    num_pid_n = (shape.N + blk_n - 1) // blk_n
    tile_cumsum = torch.empty(E + 1, device=a.device, dtype=torch.int32)
    group_offs_int = group_offs.to(torch.int64)
    _compute_tile_cumsum_kernel[(1,)](group_offs_int, tile_cumsum, E, num_pid_n,
                                      BLOCK_SIZE_M=blk_m)
    hip = _get_hip()
    buf = ctypes.create_string_buffer(96)
    struct.pack_into("<QQQQQQQ", buf, 0,
                     a.data_ptr(), b.data_ptr(), out.data_ptr(),
                     a_scale.data_ptr(), b_scale.data_ptr(),
                     group_offs_int.data_ptr(), tile_cumsum.data_ptr())
    struct.pack_into("<iiiiiii", buf, 56,
                     E, shape.N, shape.K,
                     a.stride(0),
                     b.stride(0),
                     b.stride(1),
                     out.stride(0))
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    arg_size = ctypes.c_size_t(96)
    config = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.cast(buf, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    stream = torch.cuda.current_stream().cuda_stream
    _hip_check(_get_hip().hipModuleLaunchKernel(
        func, num_cu, 1, 1, ASM_THREADS, 1, 1, ASM_LDS_BYTES,
        ctypes.c_void_p(stream), None, config,
    ), "hipModuleLaunchKernel(dgrad)")
    torch.cuda.synchronize()


def make_inputs(shape: DgradShape):
    torch.manual_seed(42)
    from primus_turbo.pytorch.core.low_precision import float8_e4m3
    a = torch.empty((M_TOTAL, shape.K), dtype=float8_e4m3, device="cuda")
    b = torch.empty((E, shape.N, shape.K), dtype=float8_e4m3, device="cuda")
    a.view(torch.uint8).random_(0, 64)
    b.view(torch.uint8).random_(0, 64)
    a_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    b_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    group_offs = torch.tensor(
        [i * (M_TOTAL // E) for i in range(E + 1)], dtype=torch.int64, device="cuda",
    )
    return a, b, a_scale, b_scale, group_offs


def run_triton(shape: DgradShape, a, b, a_scale, b_scale, group_offs):
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
        grouped_gemm_fp8_tensorwise_triton_kernel,
    )
    return grouped_gemm_fp8_tensorwise_triton_kernel(
        a, b, a_scale, b_scale, group_offs, trans_b=True, out_dtype=torch.bfloat16,
    )


def _tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def measure(label, fn, warmup, iters, flops):
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
        "stdev_ms": statistics.stdev(sample_ms) if len(sample_ms) > 1 else 0.0,
        "tflops_mean":   _tflops(flops, mean_ms),
        "tflops_median": _tflops(flops, statistics.median(sample_ms)),
        "tflops_peak":   _tflops(flops, min(sample_ms)),
    }


def print_results(shape, triton_r, asm_r):
    BAR = "=" * 80
    print(f"\n{BAR}")
    print(f"  DGRAD {shape.name}  a[{M_TOTAL},{shape.K}] @ b[{E},{shape.N},{shape.K}]^T")
    print(f"  FLOPs: {shape.flops/1e12:.4f} T  |  E={E}  M={M_TOTAL}")
    print(BAR)
    for r in [triton_r, asm_r]:
        if r is None:
            continue
        print(f"  [{r['label']}]")
        print(f"    mean={r['mean_ms']:.4f} ms  median={r['median_ms']:.4f} ms  "
              f"min={r['min_ms']:.4f} ms  stdev={r['stdev_ms']:.4f} ms")
        print(f"    TFLOPS mean={r['tflops_mean']:.2f}  median={r['tflops_median']:.2f}  "
              f"peak={r['tflops_peak']:.2f}")
    if triton_r and asm_r:
        ratio = triton_r["mean_ms"] / asm_r["mean_ms"]
        delta = (ratio - 1.0) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  SPEEDUP  ASM vs Triton: {ratio:.4f}x ({sign}{delta:.2f}%)  "
              f"{triton_r['tflops_mean']:.2f} -> {asm_r['tflops_mean']:.2f} TFLOPS")
    print(BAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=list(SHAPES) + ["all"], default="all")
    parser.add_argument("--co", type=pathlib.Path, default=None,
                        help="Override .hsaco path (used for both shapes if set)")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iters",  type=int, default=10000)
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--no-asm",    action="store_true")
    args = parser.parse_args()

    shapes = list(SHAPES.values()) if args.shape == "all" else [SHAPES[args.shape]]

    print(f"\nGrouped GEMM FP8 DGRAD benchmark  warmup={args.warmup}  iters={args.iters}")

    for shape in shapes:
        co_path = args.co or shape.default_co
        print(f"\n{'━'*80}")
        print(f"  {shape.name}  K={shape.K}  N={shape.N}  co={co_path.name}")
        print(f"{'━'*80}")

        a, b, a_scale, b_scale, group_offs = make_inputs(shape)
        out_asm = torch.empty((M_TOTAL, shape.N), dtype=torch.bfloat16, device="cuda")

        triton_r = asm_r = None

        if not args.no_triton:
            def triton_fn():
                run_triton(shape, a, b, a_scale, b_scale, group_offs)
            triton_r = measure("Triton", triton_fn, args.warmup, args.iters, shape.flops)

        if not args.no_asm:
            if not co_path.exists():
                print(f"  SKIP ASM: {co_path} not found. Run: bash build.sh")
            else:
                _, func = load_co(co_path)
                def asm_fn():
                    launch_dgrad(func, a, b, out_asm, a_scale, b_scale, group_offs, shape)
                asm_r = measure(co_path.name, asm_fn, args.warmup, args.iters, shape.flops)

        print_results(shape, triton_r, asm_r)

    sys.exit(0)


if __name__ == "__main__":
    main()
