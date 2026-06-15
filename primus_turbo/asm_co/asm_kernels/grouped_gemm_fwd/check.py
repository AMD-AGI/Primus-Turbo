#!/usr/bin/env python3
"""Correctness check: _grouped_fp8_persistent_gemm_kernel (FWD, trans_b=False).

Compares ASM .hsaco output against Triton baseline using cosine similarity.

Shapes:
  gate_up_fwd: a[131072, 2880] @ b[32, 2880, 5760] -> out[131072, 5760]
  down_fwd:    a[131072, 5760] @ b[32, 5760, 2880] -> out[131072, 2880]

Checks final_*.hsaco if present, otherwise falls back to base_*.hsaco.

Usage:
    python3 check.py
    python3 check.py --shape gate_up_fwd
    python3 check.py --co-gate-up final_gate_up_5760.hsaco --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import struct
import sys

import torch

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

E       = 32
M_TOTAL = 131_072
COS_SIM_THRESHOLD = 0.999990

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


def make_inputs(K, N, seed):
    torch.manual_seed(seed)
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


def run_triton(a, b, a_scale, b_scale, group_offs):
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
        grouped_gemm_fp8_tensorwise_triton_kernel,
    )
    out = grouped_gemm_fp8_tensorwise_triton_kernel(
        a, b, a_scale, b_scale, group_offs, trans_b=False, out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    return out


# shape_name -> (K, N, final_co, base_co)
SHAPES = {
    "gate_up_fwd": (2880, 5760, "final_gate_up_5760.hsaco", "base_gate_up_5760.hsaco"),
    "down_fwd":    (5760, 2880, "final_down_2880.hsaco",    "base_down_2880.hsaco"),
}


def check_shape(func, K, N, seeds, verbose) -> bool:
    all_pass = True
    for seed in seeds:
        a, b, a_sc, b_sc, offs = make_inputs(K, N, seed)
        triton_out = run_triton(a, b, a_sc, b_sc, offs)
        asm_out = torch.empty((M_TOTAL, N), dtype=torch.bfloat16, device="cuda")
        launch_fwd(func, a, b, asm_out, a_sc, b_sc, offs, K, N)
        tf = triton_out.float().flatten()
        af = asm_out.float().flatten()
        cos = torch.nn.functional.cosine_similarity(tf.unsqueeze(0), af.unsqueeze(0)).item()
        max_d = (tf - af).abs().max().item()
        passed = cos >= COS_SIM_THRESHOLD
        sym = "PASS" if passed else "FAIL"
        print(f"    seed={seed}  cos_sim={cos:.8f}  max_abs={max_d:.2e}  [{sym}]")
        if not passed:
            all_pass = False
        del a, b, a_sc, b_sc, offs, triton_out, asm_out
        torch.cuda.empty_cache()
    return all_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=list(SHAPES) + ["all"], default="all")
    parser.add_argument("--co-gate-up", type=pathlib.Path, default=None)
    parser.add_argument("--co-down",    type=pathlib.Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    co_override = {"gate_up_fwd": args.co_gate_up, "down_fwd": args.co_down}
    shapes = list(SHAPES.items()) if args.shape == "all" else [(args.shape, SHAPES[args.shape])]

    print(f"\n{'='*72}")
    print("  Correctness: _grouped_fp8_persistent_gemm_kernel (FWD)")
    print(f"  threshold={COS_SIM_THRESHOLD}  seeds={args.seeds}")
    print(f"{'='*72}\n")

    print("Warming up Triton ...")
    s0_K, s0_N, _, _ = list(SHAPES.values())[0]
    a, b, a_sc, b_sc, offs = make_inputs(s0_K, s0_N, 0)
    _ = run_triton(a, b, a_sc, b_sc, offs)
    del a, b, a_sc, b_sc, offs
    print("  done.\n")

    all_pass = True
    for shape_name, (K, N, final_co, base_co) in shapes:
        # Prefer final.co if it exists, else fall back to base.co
        override = co_override.get(shape_name)
        if override:
            co_path = override
        elif (SCRIPT_DIR / final_co).exists():
            co_path = SCRIPT_DIR / final_co
        else:
            co_path = SCRIPT_DIR / base_co
            print(f"  NOTE: {final_co} not found, using base binary")

        print(f"{'─'*72}")
        print(f"  Shape: {shape_name}  K={K}  N={N}")
        print(f"  .hsaco: {co_path.name}")
        if not co_path.exists():
            print(f"  SKIP: not found"); continue
        _, func = load_co(co_path)
        passed = check_shape(func, K, N, args.seeds, args.verbose)
        all_pass = all_pass and passed
        print(f"  -> {'PASS' if passed else 'FAIL'}\n")

    print(f"{'='*72}")
    verdict = "PASS" if all_pass else "FAIL"
    print(f"  VERDICT: {verdict}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
