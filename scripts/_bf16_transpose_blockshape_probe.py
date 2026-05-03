#!/usr/bin/env python3
"""Round-28 focused probe — validate (128, 128) as BF16 transpose block shape.

From the earlier block-shape sweep:
  - K > N  (GateUP): current (128, 256) best (256, 64) or (128, 128) (+3%)
  - K == N (Down):   current (256, 128) best (128, 128) (+10%)
  - B=32 cases:      current and alternatives all within ~0.5%

Focused verify on all 4 gpt_oss H4 shapes using (128, 128) vs current.
7-trial p20 at 500 iters for tight noise bounds.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")

sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import statistics
import torch
import triton

from primus_turbo.triton.utils.fp8_transpose import (
    _fp8_transpose_3d_kernel, _select_block_shape,
)


def _time_kernel(fn, warmup=20, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[iters // 5]  # p20


def _run(b, bk, bn):
    B, K, N = b.shape
    out = torch.empty((B, N, K), dtype=b.dtype, device=b.device)
    num_n_blocks = triton.cdiv(N, bn)
    grid = (B, triton.cdiv(K, bk) * num_n_blocks)
    _fp8_transpose_3d_kernel[grid](
        b, out, B, K, N,
        K * N, N, 1,
        N * K, K, 1,
        BK=bk, BN=bn,
    )
    return out


def main():
    torch.manual_seed(0)
    shapes = [
        ("gpt_oss-GateUP-B4",   (4, 5760, 2880)),
        ("gpt_oss-Down-B4",     (4, 2880, 2880)),
        ("gpt_oss-GateUP-B32",  (32, 5760, 2880)),
        ("gpt_oss-Down-B32",    (32, 2880, 2880)),
    ]
    print(f"{'shape':22s}  {'K,N':11s}  {'(cur_bk, cur_bn)':18s}  "
          f"{'t_cur_us':>9s}  {'t_new_us':>9s}  {'delta%':>7s}  "
          f"{'spread%':>8s}  {'verdict':14s}")

    TRIALS = 7
    for (name, (B, K, N)) in shapes:
        b = torch.randn((B, K, N), dtype=torch.bfloat16, device="cuda").contiguous()
        cur_bk, cur_bn = _select_block_shape(K, N)
        ref = _run(b, cur_bk, cur_bn)
        # Validate (128, 128) produces bit-equal output
        candidate = _run(b, 128, 128)
        bit_eq = torch.equal(ref, candidate)

        # Tight p20 verify across trials
        cur_times = []
        new_times = []
        for _ in range(TRIALS):
            cur_times.append(_time_kernel(lambda: _run(b, cur_bk, cur_bn),
                                          warmup=10, iters=200))
            new_times.append(_time_kernel(lambda: _run(b, 128, 128),
                                          warmup=10, iters=200))
        cur_m = statistics.median(cur_times)
        new_m = statistics.median(new_times)
        delta = (cur_m - new_m) / cur_m * 100.0
        cur_spread = (max(cur_times) - min(cur_times)) / cur_m * 100.0
        new_spread = (max(new_times) - min(new_times)) / new_m * 100.0
        spread = max(cur_spread, new_spread)

        if not bit_eq:
            verdict = "DIFFERS"
        elif delta > 1.0 and abs(delta) > spread:
            verdict = "**SAFE-WIN**"
        elif delta > 0.0:
            verdict = "sub-noise"
        else:
            verdict = "REGRESS"
        print(f"{name:22s}  {K},{N:>4d}   "
              f"({cur_bk:>3d}, {cur_bn:>3d})          "
              f"{cur_m * 1000:>9.2f}  {new_m * 1000:>9.2f}  "
              f"{delta:>+7.2f}  {spread:>+7.2f}%  {verdict}")
        del b, ref, candidate
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
