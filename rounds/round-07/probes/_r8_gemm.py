#!/usr/bin/env python3
"""Time every dense-fp8 candidate arm for the proj unit's six GEMM shapes.

The race in `_pick_dense_candidate` samples each arm twice on a shared node; this
prints the whole distribution so a pick can be judged rather than trusted, and adds
K-padded variants of the two K=2880 shapes, where `K % 128 != 0` currently locks the
4-wave whole loop out of the race.
"""
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as GK  # noqa: E402

DEV = "cuda"
F8 = torch.float8_e4m3fn
REC = []


def _time(fn, iters=40, reps=6):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    return best


_orig_pick = GK._pick_dense_candidate


def _pick(cands, args):
    n = len(cands)
    ts = [float("inf")] * n
    order = list(range(n))
    for _ in range(200):
        cands[0][2](*args)
    torch.cuda.synchronize()
    for r in range(3):
        for i in order if r % 2 == 0 else order[::-1]:
            ts[i] = min(ts[i], _time(lambda c=cands[i][2]: c(*args), iters=20, reps=2))
    best = min(order, key=ts.__getitem__)
    REC.append([(str(cands[i][1]), round(ts[i] * 1000, 2)) for i in order] + [f"best={cands[best][1]}"])
    return cands[best]


GK._pick_dense_candidate = _pick


def q(*shape):
    return (torch.randn(*shape, device=DEV) * 0.3).to(F8)


def sc():
    return torch.ones(1, device=DEV, dtype=torch.float32)


CASES = [
    # (tag, a_shape, b_shape, trans_a, trans_b)
    ("qkv_fwd_NT   M32768 N5120 K2880", (32768, 2880), (5120, 2880), False, True),
    ("qkv_fwd_NT_p M32768 N5120 K2944", (32768, 2944), (5120, 2944), False, True),
    ("o_fwd_NT     M32768 N2880 K4096", (32768, 4096), (2880, 4096), False, True),
    ("o_dg_NN      M32768 N4096 K2880", (32768, 2880), (2880, 4096), False, False),
    ("o_dg_NN_p    M32768 N4096 K2944", (32768, 2944), (2944, 4096), False, False),
    ("qkv_dg_NN    M32768 N2880 K5120", (32768, 5120), (5120, 2880), False, False),
    ("o_wg_TN      M4096  N2880 K32768", (32768, 4096), (32768, 2880), True, False),
    ("qkv_wg_TN    M2880  N5120 K32768", (32768, 2880), (32768, 5120), True, False),
]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, ash, bsh, ta, tb in CASES:
        if only and only not in tag:
            continue
        a, b = q(*ash), q(*bsh)
        sa, sb = sc(), sc()
        REC.clear()
        f = lambda: GK.gemm_fp8_tensorwise_flydsl_kernel(  # noqa: E731
            a, sa, b, sb, trans_a=ta, trans_b=tb, out_dtype=torch.bfloat16
        )
        f()
        torch.cuda.synchronize()
        us = _time(f) * 1000
        m = ash[1] if ta else ash[0]
        k = ash[0] if ta else ash[1]
        n = bsh[0] if tb else bsh[1]
        tf = 2 * m * n * k / us / 1e6
        print(f"== {tag}: {us:.2f} us  {tf:.0f} TF/s")
        for r in REC:
            print("     ", r)
        del a, b
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
