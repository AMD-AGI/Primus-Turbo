#!/usr/bin/env python3
"""R20 probe: BF16 dA RRR config sweep on DSV3-Down + Qwen3-GateUP.

Continuation of R18 (Qwen3-Down tiles_n=6 LANDED) / R19 (DSV3-GateUP
tiles_n=28 NOISE-BOUND, real signal). R20 explores the remaining 8
uncovered BF16 dA RRR shapes:
  - DSV3-Down  (tiles_n=8, N_fwd=7168, K_fwd=2048): 4 shapes
  - Qwen3-GateUP (tiles_n=16, N_fwd=3072, K_fwd=4096): 4 shapes

For each, sweep 11 (group_m, num_xcds) cells. 5-trial median × 100 iters
per cell. Bit-equivalent correctness check at default (4, 0) vs (16, 4).
"""

import os
import sys
import statistics

sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "HIPKITTEN")

from primus_turbo.pytorch.kernels.hipkitten import loader as hipkitten  # noqa: E402

hk = hipkitten.load_bf16()
rrr_fn = hk.grouped_rrr

device = "cuda"


def make_tensors(G, M_per_group, N_rrr, K_rrr):
    M_total = G * M_per_group
    torch.manual_seed(42)
    a = torch.randn(M_total, K_rrr, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, K_rrr, N_rrr, dtype=torch.bfloat16, device=device)
    c = torch.empty(M_total, N_rrr, dtype=torch.bfloat16, device=device)
    group_offs = torch.tensor(
        [i * M_per_group for i in range(G + 1)], dtype=torch.int64, device=device
    )
    return a, b, c, group_offs


def time_one(a, b, c, group_offs, group_m, num_xcds, M_per_group, iters=100):
    for _ in range(10):
        rrr_fn(a, b, c, group_offs, group_m, num_xcds, M_per_group)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        rrr_fn(a, b, c, group_offs, group_m, num_xcds, M_per_group)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def correctness_check(a, b, group_offs, group_m, num_xcds, M_per_group):
    c_default = torch.zeros(a.shape[0], b.shape[2], dtype=a.dtype, device=device)
    rrr_fn(a, b, c_default, group_offs, 4, 0, M_per_group)
    c_test = torch.zeros_like(c_default)
    rrr_fn(a, b, c_test, group_offs, group_m, num_xcds, M_per_group)
    max_abs = (c_default.float() - c_test.float()).abs().max().item()
    bit_eq = torch.equal(c_default.view(torch.int16), c_test.view(torch.int16))
    return max_abs, bit_eq


def sweep_shape(G, M_per_group, N_fwd, K_fwd, label):
    K_rrr = N_fwd
    N_rrr = K_fwd
    M_total = G * M_per_group
    a, b, c, group_offs = make_tensors(G, M_per_group, N_rrr, K_rrr)
    flops = 2 * M_total * N_rrr * K_rrr

    candidates = [
        (1, 0),  (2, 0),  (4, 0),  (8, 0), (16, 0),
        (4, 2),  (4, 4),  (2, 4),  (8, 4), (16, 4),  (1, 4),
    ]

    print(f"\n=== {label} ===")
    print(f"  fwd=(M_per_g={M_per_group}, N={N_fwd}, K={K_fwd}), "
          f"RRR=(k={K_rrr}, n={N_rrr}), M_total={M_total}, "
          f"tiles_n={N_rrr // 256}")

    ma, be = correctness_check(a, b, group_offs, 16, 4, M_per_group)
    print(f"  correctness (default vs (16,4)): max_abs={ma:.6e} bit_eq={be}")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = [
            time_one(a, b, c, group_offs, gm, xcd, M_per_group, iters=100)
            for _ in range(5)
        ]
        median_ms = statistics.median(ms_trials)
        tflops = flops / (median_ms * 1e9)
        results.append((gm, xcd, median_ms, tflops))

    default_tflops = next(r[3] for r in results if r[0] == 4 and r[1] == 0)
    results.sort(key=lambda r: r[3], reverse=True)

    print(f"  {'(gm,xcd)':>10s}  {'med_ms':>8s}  {'tflops':>7s}  Δ vs default")
    for r in results:
        delta = (r[3] - default_tflops) / default_tflops * 100
        d = " *def*" if (r[0] == 4 and r[1] == 0) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}  {r[2]:7.4f}   "
              f"{r[3]:6.1f}   {delta:+5.2f}%{d}")


def main():
    print("=" * 60)
    print("DSV3-Down (tiles_n=8): 4 shapes")
    print("=" * 60)
    sweep_shape(G=16, M_per_group=2048, N_fwd=7168, K_fwd=2048,
                label="DSV3-Down B=16 M=2048 (m_total=32768)")
    sweep_shape(G=16, M_per_group=4096, N_fwd=7168, K_fwd=2048,
                label="DSV3-Down B=16 M=4096 (m_total=65536)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=7168, K_fwd=2048,
                label="DSV3-Down B=32 M=2048 (m_total=65536)")
    sweep_shape(G=32, M_per_group=4096, N_fwd=7168, K_fwd=2048,
                label="DSV3-Down B=32 M=4096 (m_total=131072)")

    print()
    print("=" * 60)
    print("Qwen3-GateUP (tiles_n=16): 4 shapes")
    print("=" * 60)
    sweep_shape(G=16, M_per_group=2048, N_fwd=3072, K_fwd=4096,
                label="Qwen3-GateUP B=16 M=2048 (m_total=32768)")
    sweep_shape(G=16, M_per_group=4096, N_fwd=3072, K_fwd=4096,
                label="Qwen3-GateUP B=16 M=4096 (m_total=65536)")
    sweep_shape(G=32, M_per_group=2048, N_fwd=3072, K_fwd=4096,
                label="Qwen3-GateUP B=32 M=2048 (m_total=65536)")
    sweep_shape(G=32, M_per_group=4096, N_fwd=3072, K_fwd=4096,
                label="Qwen3-GateUP B=32 M=4096 (m_total=131072)")


if __name__ == "__main__":
    main()
