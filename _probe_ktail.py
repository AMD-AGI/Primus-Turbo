"""K-sweep affine decomposition of the grouped mxfp8 NT kernel: us/round = F + t*k_iters.

Fixing M/N/G holds the tile count constant (24 rounds of 256 CUs), so sweeping only K
separates the steady-state K loop (slope t) from the per-tile fixed cost (intercept F).
Both arms are timed interleaved rep-by-rep inside one process, the same drift-immune
ruler the campaign bench uses -- a cross-process sweep cannot resolve the ~1% effects.
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
G, M, N = 32, 131072, 2944
KS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1024, 1536, 2048, 2944, 4096, 5760]
ROUNDS = (M // 256) * ((N + 255) // 256) / 256.0  # tiles / CUs
ONE = torch.ones(1, dtype=torch.float32, device=DEV)
o = torch.tensor([i * (M // G) for i in range(G + 1)], dtype=torch.int64, device=DEV)


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def ab(fa, fb, warmup=15, reps=30):
    for _ in range(warmup):
        fa(); fb()
    torch.cuda.synchronize()
    ta, tb = [], []
    for _ in range(reps):
        for fn, acc in ((fa, ta), (fb, tb)):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            acc.append(e0.elapsed_time(e1))
    return statistics.median(ta), statistics.median(tb)


def fit(xs, ys):
    n = len(xs)
    mx_, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx_) ** 2 for x in xs)
    t = sxy / sxx
    return t, my - t * mx_


rows = []
for K in KS:
    a, w = f8(M, K), f8(G, N, K)
    a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device=DEV)
    w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device=DEV)
    mx = lambda: fly_mx(a, a_s, w, w_s, o, N, K, out_dtype=torch.bfloat16, num_cu=-1, preshuffle=False)
    tw = lambda: fly_tw(a, w, ONE, ONE, o, trans_b=True, out_dtype=torch.bfloat16, num_cu=-1)
    fly_mx(a, a_s, w, w_s, o, N, K, out_dtype=torch.bfloat16, num_cu=-1)
    torch.cuda.synchronize()
    t_mx, t_tw = ab(mx, tw)
    u_mx, u_tw = t_mx * 1e3 / ROUNDS, t_tw * 1e3 / ROUNDS
    rows.append((K // 128, u_mx, u_tw))
    print(f"K={K:5d} iters={K // 128:3d} | mx {u_mx:8.3f} us/round | tw {u_tw:8.3f} | tw/mx {u_tw / u_mx:.4f}",
          flush=True)
    del a, w, a_s, w_s
    torch.cuda.empty_cache()

it = [r[0] for r in rows]
t_m, f_m = fit(it, [r[1] for r in rows])
t_t, f_t = fit(it, [r[2] for r in rows])
print(f"\n         per_k_iter   fixed_per_tile")
print(f"mx       {t_m:8.4f} us   {f_m:8.3f} us")
print(f"tw       {t_t:8.4f} us   {f_t:8.3f} us")
print(f"mx/tw    {t_m / t_t:8.4f}     {f_m / f_t:8.4f}   (fixed delta {f_m - f_t:+.3f} us/tile)")
for n_it in (23, 45):
    print(f"  @{n_it}iter: mx fixed share {f_m / (f_m + t_m * n_it) * 100:.1f}% | "
          f"predicted tw/mx {(f_t + t_t * n_it) / (f_m + t_m * n_it):.4f}")
