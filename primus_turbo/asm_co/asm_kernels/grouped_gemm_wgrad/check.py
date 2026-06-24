#!/usr/bin/env python3
"""Correctness check: grouped_variable_k_dot_scaled_kernel (WGRAD, FP8 dot-scaled).

Compares ASM .co output against Triton reference using cosine similarity.
Tests both production shapes: OUT_M=2880xOUT_N=2880 and OUT_M=2880xOUT_N=5760.

Usage:
    python3 check.py
    python3 check.py --co final.co
    python3 check.py --co intermediate.co --seeds 42 43 44
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
COS_SIM_THRESHOLD = 0.999


def load_co(path: pathlib.Path):
    hip = ctypes.CDLL("libamdhip64.so")
    mod = ctypes.c_void_p()
    rc = hip.hipModuleLoad(ctypes.byref(mod), str(path).encode())
    if rc != 0:
        raise RuntimeError(f"hipModuleLoad failed rc={rc}: {path}")
    func = ctypes.c_void_p()
    rc = hip.hipModuleGetFunction(ctypes.byref(func), mod, KERNEL_NAME.encode())
    if rc != 0:
        raise RuntimeError(f"hipModuleGetFunction failed rc={rc}")
    return hip, func


def launch_wgrad(hip, func, lhs, rhs, lhs_scale, rhs_scale, group_offs):
    out_m = lhs.shape[1]
    out_n = rhs.shape[1]
    g = group_offs.shape[0] - 1
    out = torch.empty((g, out_m, out_n), device=lhs.device, dtype=torch.bfloat16)

    # Implicit 2x scale compensation matching ASM launcher convention
    lhs_scale_adj = lhs_scale * 2.0
    rhs_scale_adj = rhs_scale * 2.0

    buf = ctypes.create_string_buffer(96)
    struct.pack_into("<QQQQQQ", buf, 0,
                     lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(),
                     lhs_scale_adj.data_ptr(), rhs_scale_adj.data_ptr(),
                     group_offs.data_ptr())
    struct.pack_into("<iiiiiiii", buf, 48,
                     g, out_m, out_n,
                     lhs.stride(0), rhs.stride(0),
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
    torch.cuda.synchronize()
    return out


def make_inputs(out_m, out_n, seed=42, g=32, total_tokens=131072):
    torch.manual_seed(seed)
    from primus_turbo.pytorch.core.low_precision import float8_e4m3
    tokens_per_group = total_tokens // g
    group_offs = torch.zeros(g + 1, dtype=torch.int64, device="cuda")
    group_offs[1:] = torch.arange(1, g + 1, dtype=torch.int64, device="cuda") * tokens_per_group

    lhs_bf16 = torch.randn(total_tokens, out_m, device="cuda", dtype=torch.bfloat16)
    rhs_bf16 = torch.randn(total_tokens, out_n, device="cuda", dtype=torch.bfloat16)
    ls = lhs_bf16.abs().max().float() / 448.0
    rs = rhs_bf16.abs().max().float() / 448.0
    lhs_fp8 = (lhs_bf16.float() / ls).to(float8_e4m3)
    rhs_fp8 = (rhs_bf16.float() / rs).to(float8_e4m3)
    return lhs_fp8, rhs_fp8, ls.view(1), rs.view(1), group_offs


def run_triton(lhs, rhs, ls, rs, group_offs):
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
        grouped_gemm_fp8_tensorwise_variable_k_triton_kernel,
    )
    out = grouped_gemm_fp8_tensorwise_variable_k_triton_kernel(
        lhs, rhs, ls, rs, group_offs, out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    return out


def cos_sim(a, b):
    af = a.flatten().float()
    bf = b.flatten().float()
    return torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()


def check_shape(hip, func, out_m, out_n, seeds):
    print(f"\n  Shape: OUT_M={out_m}  OUT_N={out_n}")
    all_pass = True
    for seed in seeds:
        lhs, rhs, ls, rs, offs = make_inputs(out_m, out_n, seed)
        triton_out = run_triton(lhs, rhs, ls, rs, offs)
        asm_out = launch_wgrad(hip, func, lhs, rhs, ls, rs, offs)

        c = cos_sim(asm_out, triton_out)
        max_diff = (asm_out.float() - triton_out.float()).abs().max().item()
        has_nan = asm_out.isnan().any().item()
        passed = c >= COS_SIM_THRESHOLD and not has_nan
        sym = "PASS" if passed else "FAIL"
        print(f"    seed={seed}  cos_sim={c:.8f}  max_abs={max_diff:.2e}  "
              f"nan={has_nan}  [{sym}]")
        if not passed:
            all_pass = False
        del lhs, rhs, ls, rs, offs, triton_out, asm_out
        torch.cuda.empty_cache()
    return all_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--co", type=pathlib.Path, default=SCRIPT_DIR / "final.co")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  Correctness: grouped_variable_k_dot_scaled_kernel (WGRAD)")
    print(f"  .co: {args.co}")
    print(f"  threshold={COS_SIM_THRESHOLD}  seeds={args.seeds}")
    print(f"{'='*72}")

    if not args.co.exists():
        print(f"ERROR: .co not found: {args.co}")
        print("  Run:  bash build.sh  first.")
        sys.exit(1)

    hip, func = load_co(args.co)
    print(f"  Loaded '{KERNEL_NAME}' from {args.co.name}")

    shapes = [(2880, 2880), (2880, 5760)]
    results = {}
    for out_m, out_n in shapes:
        passed = check_shape(hip, func, out_m, out_n, args.seeds)
        results[(out_m, out_n)] = passed

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    all_pass = True
    for (m, n), passed in results.items():
        sym = "PASS" if passed else "FAIL"
        print(f"  [{sym}]  OUT_M={m}  OUT_N={n}")
        if not passed:
            all_pass = False
    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
