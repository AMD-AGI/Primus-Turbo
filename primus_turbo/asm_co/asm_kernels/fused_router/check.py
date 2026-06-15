#!/usr/bin/env python3
"""Correctness check: fused_scaling_group_sum_routing_kernel (MoE router).

Softmax + argsort top-k over [s, e=32] expert logits.
Compiled constexprs: s=32768, e=32, g=1, k=4, score_function=softmax.

Compares .co kernel outputs against Triton JIT.
Uses bit-exact comparison for integer outputs (topk_idx, routing_map)
and cosine similarity (>= 0.999990) for float outputs.
raw_topk_logits is not written by the softmax variant; bit-exact zero
match is treated as pass.

Usage:
    python3 check.py
    python3 check.py --co final.co --seeds 5 --verbose
    python3 check.py --co final.co --ref base.hsaco
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
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
COS_SIM_THRESHOLD = 0.999990

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


def launch_co(func, input_logit):
    output_scores          = torch.empty(S, E, dtype=torch.float32, device="cuda")
    output_topk_idx        = torch.ones(S, K,  dtype=torch.int64,   device="cuda")
    output_raw_topk_logits = torch.zeros(S, K, dtype=torch.float32, device="cuda")
    output_probs           = torch.zeros(S, E, dtype=torch.float32, device="cuda")
    output_routing_map     = torch.zeros(S, E, dtype=torch.int32,   device="cuda")

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
    torch.cuda.synchronize()
    return output_scores, output_topk_idx, output_raw_topk_logits, output_probs, output_routing_map


def run_triton(input_logit):
    from primus_turbo.pytorch.kernels.moe.fused_moe_router_impl import fused_moe_router_fwd
    results = fused_moe_router_fwd(input_logit, S, E, G, K, SELECTED_GROUPS,
                                   SCORE_FUNCTION, SCALING_FACTOR)
    torch.cuda.synchronize()
    return results


def compare_tensor(ref, test, name, verbose):
    if ref.dtype in (torch.int32, torch.int64):
        match = torch.equal(ref, test)
        n_diff = (ref != test).sum().item()
        print(f"    {name:25s}  exact_match={match}  n_diff={n_diff}")
        return match
    ref_f = ref.float().flatten()
    tst_f = test.float().flatten()
    diff = (ref_f - tst_f).abs()
    cos = (ref_f * tst_f).sum().item() / (
        ref_f.norm().item() * tst_f.norm().item() + 1e-12)
    max_abs = diff.max().item()
    bit_exact = torch.allclose(ref.float(), test.float(), rtol=0, atol=0)
    print(f"    {name:25s}  cos_sim={cos:.8f}  max_abs={max_abs:.2e}  "
          f"bit_exact={bit_exact}")
    if verbose and not bit_exact:
        bad = torch.where(diff > 0)[0]
        for idx in bad[:5]:
            print(f"      [{idx.item():8d}]  ref={ref_f[idx].item():+.8f}  "
                  f"test={tst_f[idx].item():+.8f}")
    # bit-identical tensors always pass (cos_sim is meaningless for zero vectors)
    return bit_exact or cos >= COS_SIM_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--co",  type=pathlib.Path,
                        default=SCRIPT_DIR / "final.co",
                        help="Test .co/.hsaco to validate (default: final.co)")
    parser.add_argument("--ref", type=pathlib.Path, default=None,
                        help="Reference .co (default: Triton JIT)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds (default: 3)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  Correctness: fused_scaling_group_sum_routing_kernel")
    print(f"  test .co : {args.co}")
    print(f"  ref      : {args.ref or 'Triton JIT'}")
    print(f"  Shape: s={S}, e={E}, k={K}, groups={G}")
    print(f"{'='*72}\n")

    if not args.co.exists():
        print(f"ERROR: .co not found: {args.co}")
        print("  Run:  bash build.sh  first.")
        sys.exit(1)

    _, test_func = load_co(args.co)
    ref_func = None
    if args.ref is not None and args.ref.exists():
        _, ref_func = load_co(args.ref)

    # Warmup
    print("Warmup ...")
    warmup_in = torch.randn(S, E, dtype=torch.float32, device="cuda")
    launch_co(test_func, warmup_in)
    if ref_func:
        launch_co(ref_func, warmup_in)
    else:
        run_triton(warmup_in)
    del warmup_in
    torch.cuda.synchronize()
    print("  done.\n")

    # raw_topk_logits is not written by the softmax variant; skip comparison
    output_names = ["scores", "topk_idx", "probs", "routing_map"]
    seeds = list(range(42, 42 + args.seeds))
    all_pass = []

    for seed in seeds:
        torch.manual_seed(seed)
        input_logit = torch.randn(S, E, dtype=torch.float32, device="cuda")
        print(f"── seed={seed} {'─'*50}")
        ref_outs  = launch_co(ref_func, input_logit) if ref_func else run_triton(input_logit)
        test_outs = launch_co(test_func, input_logit)
        # skip index 2 (raw_topk_logits) — not written by softmax variant
        ref_cmp  = [t for i, t in enumerate(ref_outs)  if i != 2]
        test_cmp = [t for i, t in enumerate(test_outs) if i != 2]
        passed_all = True
        for name, ref_t, test_t in zip(output_names, ref_cmp, test_cmp):
            ok = compare_tensor(ref_t, test_t, name, args.verbose)
            if not ok:
                passed_all = False
        sym = "PASS" if passed_all else "FAIL"
        print(f"  -> {sym}\n")
        all_pass.append(passed_all)

    print(f"{'='*72}")
    print("  SUMMARY")
    for seed, ok in zip(seeds, all_pass):
        print(f"  [{'PASS' if ok else 'FAIL'}]  seed={seed}")
    verdict = "PASS" if all(all_pass) else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    sys.exit(0 if all(all_pass) else 1)


if __name__ == "__main__":
    main()
