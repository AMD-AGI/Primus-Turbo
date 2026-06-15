#!/usr/bin/env python3
"""Correctness check: swiglu_with_mask_bwd_kernel.

Compares .co output against Triton reference (swiglu_bwd_with_probs).
Tests both grad_x and grad_probs with cosine similarity and max absolute error.

Kernel shape: N=131072, H=4096  (gpt-oss MoE 20B)

Usage:
    python3 check.py final.co
    python3 check.py final.co --seeds 42 43 44
    python3 check.py intermediate.co --seeds 42
"""
import argparse
import ctypes
import os
import pathlib
import struct
import sys

import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

HIP = ctypes.CDLL("libamdhip64.so")
HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)

KERNEL_NAME       = b"swiglu_with_mask_bwd_kernel"
N, H              = 131072, 4096
BLOCK_SIZE        = 8192
COS_SIM_THRESHOLD = 0.999990


def hip_check(rc, op=""):
    if rc != 0:
        raise RuntimeError(f"HIP {op} failed (rc={rc})")


def load_co(path):
    mod = ctypes.c_void_p()
    hip_check(HIP.hipModuleLoad(ctypes.byref(mod), str(path).encode()), "ModuleLoad")
    func = ctypes.c_void_p()
    hip_check(HIP.hipModuleGetFunction(ctypes.byref(func), mod, KERNEL_NAME), "GetFunction")
    return mod, func


def launch_co(func, go, x, p, rm, gx, gp):
    buf = ctypes.create_string_buffer(80)
    struct.pack_into("<QQQQQQ", buf, 0,
                     go.data_ptr(), x.data_ptr(), p.data_ptr(),
                     rm.data_ptr(), gx.data_ptr(), gp.data_ptr())
    struct.pack_into("<iii", buf, 48, go.stride(0), x.stride(0), gx.stride(0))
    struct.pack_into("<I",   buf, 60, 0)
    struct.pack_into("<QQ",  buf, 64, 0, 0)
    arg_size = ctypes.c_size_t(80)
    config = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(buf, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    hip_check(
        HIP.hipModuleLaunchKernel(func, BLOCK_SIZE, 1, 1, 256, 1, 1, 16,
                                  None, None, config),
        "LaunchKernel",
    )
    torch.cuda.synchronize()


def chunked_cos_sim(a, b, chunk=2**20):
    dot, norm_a, norm_b = 0.0, 0.0, 0.0
    a_flat, b_flat = a.flatten(), b.flatten()
    for i in range(0, len(a_flat), chunk):
        af = a_flat[i:i + chunk].float()
        bf = b_flat[i:i + chunk].float()
        dot   += (af * bf).sum().item()
        norm_a += (af * af).sum().item()
        norm_b += (bf * bf).sum().item()
    return dot / (norm_a ** 0.5 * norm_b ** 0.5 + 1e-12)


def check_seed(func, seed):
    torch.manual_seed(seed)
    go = torch.randn(N, H,     dtype=torch.bfloat16, device="cuda")
    x  = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    p  = torch.rand(N, 1,      dtype=torch.float32,  device="cuda")
    rm = torch.ones(N,         dtype=torch.int64,    device="cuda")

    from primus_turbo.pytorch.kernels.activation.swiglu_impl import swiglu_bwd_with_probs
    ref_gx, ref_gp = swiglu_bwd_with_probs(go, x, p, rm)

    gx = torch.zeros(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    gp = torch.zeros(N,        dtype=torch.float32,  device="cuda")
    launch_co(func, go, x, p, rm, gx, gp)

    cos_gx = chunked_cos_sim(ref_gx, gx)
    cos_gp = chunked_cos_sim(ref_gp, gp)
    max_gx = (ref_gx.float() - gx.float()).abs().max().item()
    max_gp = (ref_gp.float() - gp.float()).abs().max().item()
    passed = cos_gx >= COS_SIM_THRESHOLD and cos_gp >= COS_SIM_THRESHOLD

    del go, x, p, rm, ref_gx, ref_gp, gx, gp
    torch.cuda.empty_cache()
    return passed, {"cos_gx": cos_gx, "cos_gp": cos_gp,
                    "max_gx": max_gx, "max_gp": max_gp}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("co_file", nargs="?",
                        default=str(SCRIPT_DIR / "final.co"),
                        help=".co file to check (default: final.co)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    co_path = pathlib.Path(args.co_file)
    if not co_path.is_absolute():
        co_path = SCRIPT_DIR / co_path

    print(f"\n{'='*72}")
    print(f"  Correctness: swiglu_with_mask_bwd_kernel")
    print(f"  .co: {co_path}")
    print(f"  N={N}  H={H}  threshold={COS_SIM_THRESHOLD}  seeds={args.seeds}")
    print(f"{'='*72}\n")

    if not co_path.exists():
        print(f"ERROR: .co not found: {co_path}")
        print("  Run:  bash build.sh  first.")
        sys.exit(1)

    mod, func = load_co(co_path)
    all_pass = True

    for seed in args.seeds:
        print(f"── seed={seed} ──")
        passed, d = check_seed(func, seed)
        sym = "PASS" if passed else "FAIL"
        print(f"    grad_x:     cos_sim={d['cos_gx']:.8f}  max_abs={d['max_gx']:.2e}")
        print(f"    grad_probs: cos_sim={d['cos_gp']:.8f}  max_abs={d['max_gp']:.2e}")
        print(f"  -> {sym}\n")
        if not passed:
            all_pass = False

    HIP.hipModuleUnload(mod)

    print(f"{'='*72}")
    verdict = "PASS" if all_pass else "FAIL"
    print(f"  VERDICT: {verdict}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
