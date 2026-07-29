"""Price the NT half-N boundary tile: mx vs tw at N with and without a half block.

N=2816 (11.0 blocks) / 2944 (11.5 -> one half tile) / 3072 (12.0), same M/K/G as the
campaign `down` config. Marginal cost of the 12th block tells us whether the `down`
deficit lives in the boundary tile or in the full-tile body.
"""
import statistics
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
    grouped_gemm_fp8_tensorwise_flydsl_kernel as fly_tw,
)
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel as fly_mx,
)
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G = 32
M = 131072
K = 2944
ONE = torch.ones(1, dtype=torch.float32, device=DEV)


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def sc(*s):
    return torch.full(s, 127, dtype=torch.uint8, device=DEV)


def ab(fa, fb, warmup=20, reps=40):
    for _ in range(warmup):
        fa()
        fb()
    torch.cuda.synchronize()
    ta, tb = [], []
    for _ in range(reps):
        for fn, acc in ((fa, ta), (fb, tb)):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            acc.append(e0.elapsed_time(e1))
    return statistics.median(ta), statistics.median(tb)


def offs():
    u = M // G
    return torch.tensor([i * u for i in range(G + 1)], dtype=torch.int64, device=DEV)


o = offs()
a = f8(M, K)
a_s = sc(M, K // 32)
res = {}
for N in (2816, 2944, 3072):
    w, w_s = f8(G, N, K), sc(G, N, K // 32)
    tw = lambda: fly_tw(a, w, ONE, ONE, o, trans_b=True, out_dtype=torch.bfloat16, num_cu=-1)
    fly_mx(a, a_s, w, w_s, o, N, K, out_dtype=torch.bfloat16, num_cu=-1)
    torch.cuda.synchronize()
    mx = lambda: fly_mx(a, a_s, w, w_s, o, N, K, out_dtype=torch.bfloat16, num_cu=-1, preshuffle=False)
    t_tw, t_mx = ab(tw, mx)
    res[N] = (t_tw, t_mx)
    print(f"N={N:5d} blocks={N/256:5.2f} | tw {t_tw:.4f}ms | mx {t_mx:.4f}ms | tw/mx {t_tw/t_mx:.4f}",
          flush=True)
    del w, w_s

print()
for tag, i in (("tw", 0), ("mx", 1)):
    full = res[3072][i] - res[2816][i]
    half = res[2944][i] - res[2816][i]
    print(f"{tag}: 11-block {res[2816][i]:.4f}ms | +1 full block {full*1000:.1f}us"
          f" | +1 HALF block {half*1000:.1f}us | half/full {half/full:.3f}")
sys.stdout.flush()
