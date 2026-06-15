#!/usr/bin/env python3
"""Correctness check: _grouped_fp8_persistent_gemm_kernel (DGRAD, trans_b=True).

Compares final .hsaco outputs against Triton baseline using cosine similarity.

Shapes:
  down_dgrad:    a[131072, 2880] @ b[32, 2880, 2880]^T -> out[131072, 2880]
  gate_up_dgrad: a[131072, 5760] @ b[32, 2880, 5760]^T -> out[131072, 2880]

Usage:
    python3 check.py
    python3 check.py --shape down_dgrad
    python3 check.py --co final_down_2880.hsaco --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import struct
import sys
from dataclasses import dataclass

import torch

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

E = 32
M_TOTAL = 131_072
COS_SIM_THRESHOLD = 0.999990

ASM_KERNEL_NAME = "_grouped_fp8_persistent_gemm_kernel"
ASM_THREADS = 512
ASM_LDS_BYTES = 131_072

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)
_HIP: ctypes.CDLL | None = None


@dataclass
class DgradShape:
    name: str
    K: int
    N: int


SHAPES = {
    "down_dgrad":    DgradShape("down_dgrad",    K=2880, N=2880),
    "gate_up_dgrad": DgradShape("gate_up_dgrad", K=5760, N=2880),
}

DEFAULT_CO = {
    "down_dgrad":    SCRIPT_DIR / "final_down_2880.hsaco",
    "gate_up_dgrad": SCRIPT_DIR / "final_gate_up_5760.hsaco",
}


def _get_hip() -> ctypes.CDLL:
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


def _hip_check(rc: int, op: str):
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
                     a.stride(0),     # stride_am
                     b.stride(0),     # stride_bg
                     b.stride(1),     # stride_bn (trans_b=True, b=[G,N,K])
                     out.stride(0))   # stride_cm

    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    arg_size = ctypes.c_size_t(96)
    config = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(buf, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    stream = torch.cuda.current_stream().cuda_stream
    _hip_check(
        hip.hipModuleLaunchKernel(
            func, num_cu, 1, 1, ASM_THREADS, 1, 1, ASM_LDS_BYTES,
            ctypes.c_void_p(stream), None, config,
        ),
        "hipModuleLaunchKernel(dgrad)",
    )
    torch.cuda.synchronize()


def make_inputs(shape: DgradShape, seed: int):
    torch.manual_seed(seed)
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
    out = grouped_gemm_fp8_tensorwise_triton_kernel(
        a, b, a_scale, b_scale, group_offs, trans_b=True, out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    return out


def check_shape(func, shape: DgradShape, seeds: list[int], verbose: bool) -> bool:
    all_pass = True
    for seed in seeds:
        a, b, a_scale, b_scale, group_offs = make_inputs(shape, seed)
        triton_out = run_triton(shape, a, b, a_scale, b_scale, group_offs)
        asm_out = torch.empty((M_TOTAL, shape.N), dtype=torch.bfloat16, device="cuda")
        launch_dgrad(func, a, b, asm_out, a_scale, b_scale, group_offs, shape)

        t_f = triton_out.float().flatten()
        a_f = asm_out.float().flatten()
        cos = torch.nn.functional.cosine_similarity(
            t_f.unsqueeze(0), a_f.unsqueeze(0),
        ).item()
        max_diff = (t_f - a_f).abs().max().item()
        passed = cos >= COS_SIM_THRESHOLD

        sym = "PASS" if passed else "FAIL"
        print(f"    seed={seed}  cos_sim={cos:.8f}  max_abs={max_diff:.2e}  [{sym}]")
        if verbose and not passed:
            bad = torch.where((t_f - a_f).abs() > 0.01)[0]
            for idx in bad[:5]:
                print(f"      [{idx.item():8d}]  triton={t_f[idx].item():+.6f}  "
                      f"asm={a_f[idx].item():+.6f}")
        if not passed:
            all_pass = False
        del a, b, a_scale, b_scale, group_offs, triton_out, asm_out
        torch.cuda.empty_cache()
    return all_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--co", type=pathlib.Path, default=None,
                        help="Path to .hsaco to check (default: shape-specific default)")
    parser.add_argument("--shape", choices=list(SHAPES) + ["all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    shapes = list(SHAPES.values()) if args.shape == "all" else [SHAPES[args.shape]]

    print(f"\n{'='*72}")
    print("  Correctness: _grouped_fp8_persistent_gemm_kernel (DGRAD)")
    print(f"  threshold={COS_SIM_THRESHOLD}  seeds={args.seeds}")
    print(f"{'='*72}\n")

    all_pass = True

    print("Warming up Triton kernel ...")
    dummy_shape = shapes[0]
    a, b, a_scale, b_scale, offs = make_inputs(dummy_shape, 0)
    _ = run_triton(dummy_shape, a, b, a_scale, b_scale, offs)
    del a, b, a_scale, b_scale, offs
    print("  done.\n")

    for shape in shapes:
        co_path = args.co if args.co else DEFAULT_CO[shape.name]
        print(f"{'─'*72}")
        print(f"  Shape: {shape.name}  K={shape.K}  N={shape.N}")
        print(f"  .hsaco: {co_path}")
        if not co_path.exists():
            print(f"  SKIP: file not found"); continue
        _, func = load_co(co_path)
        passed = check_shape(func, shape, args.seeds, args.verbose)
        all_pass = all_pass and passed
        print(f"  -> {'PASS' if passed else 'FAIL'}\n")

    print(f"{'='*72}")
    verdict = "PASS" if all_pass else "FAIL"
    print(f"  VERDICT: {verdict}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
