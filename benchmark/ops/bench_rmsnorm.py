###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compare RMSNorm fwd+bwd throughput between:
  - primus_turbo.pytorch.ops.normalization.rmsnorm   (this PR)
  - torch.nn.functional.rms_norm                     (eager PyTorch baseline)

Memory-bandwidth bound op, so we report GB/s of effective traffic in addition
to wallclock. Effective traffic (fwd) = 2*N*C*sizeof(T)  (read x, write y),
gamma is amortized; (bwd) = 4*N*C*sizeof(T)  (read x, dy; write dx; reduce
dgamma).

Usage:
    python bench_rmsnorm.py
    python bench_rmsnorm.py --dtype bfloat16 --warmup 25 --iters 200
    python bench_rmsnorm.py --output rmsnorm_bench.csv
"""

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from config import compute_snr, get_platform_info
from tabulate import tabulate

from primus_turbo.pytorch.ops.normalization import rmsnorm as turbo_rmsnorm

# (tokens, hidden) pairs. Tokens span small (decode-like) to large (training);
# hidden spans common LLM widths plus a couple of small ones that exercise the
# warp-per-row fast path (<=512 for bf16/fp16).
SHAPES = [
    # warp-per-row regime (fast path)
    (8192, 128),
    (8192, 512),
    # block-per-row, small/medium
    (8192, 1024),
    (8192, 2048),
    (8192, 4096),
    # production LLM widths
    (4096, 5120),
    (8192, 5120),
    (16384, 4096),
    (16384, 8192),
    (32768, 4096),
    # large hidden
    (4096, 8192),
    (4096, 12288),
    (4096, 16384),
]

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _bytes(dtype):
    return torch.finfo(dtype).bits // 8


def _make_tensors(N, C, dtype, device, requires_grad):
    x = torch.randn(N, C, device=device, dtype=dtype, requires_grad=requires_grad)
    g = torch.randn(C, device=device, dtype=dtype, requires_grad=requires_grad)
    return x, g


def _time(stmt_fn, warmup, iters):
    # Use torch's benchmark.Timer — it handles GPU clock-ramp via adaptive
    # autorange. The `iters` arg becomes the floor on the number of inner runs
    # per measurement; blocked_autorange picks the actual count based on the
    # observed per-call latency so even tiny kernels get a stable µs reading.
    timer = benchmark.Timer(stmt="fn()", globals={"fn": stmt_fn})
    for _ in range(warmup):
        stmt_fn()
    torch.cuda.synchronize()
    m = timer.blocked_autorange(min_run_time=1.0)
    return m.mean * 1e3  # ms


def _correctness(N, C, dtype, device, atol, rtol):
    x = torch.randn(N, C, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(C, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    g_ref = g.detach().clone().requires_grad_()

    y = turbo_rmsnorm(x, g, 1e-6)
    y_ref = F.rms_norm(x_ref, [C], g_ref, 1e-6)
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    return (
        compute_snr(y_ref, y),
        compute_snr(x_ref.grad, x.grad),
        compute_snr(g_ref.grad, g.grad),
    )


def _bench_turbo(N, C, dtype, device, warmup, iters):
    x, g = _make_tensors(N, C, dtype, device, requires_grad=True)

    def fwd():
        return turbo_rmsnorm(x, g, 1e-6)

    fwd_ms = _time(fwd, warmup, iters)

    y = fwd()
    grad_out = torch.randn_like(y)

    def fwd_bwd():
        x.grad = None
        g.grad = None
        out = turbo_rmsnorm(x, g, 1e-6)
        out.backward(grad_out, retain_graph=False)

    fwdbwd_ms = _time(fwd_bwd, warmup, iters)
    return fwd_ms, fwdbwd_ms - fwd_ms


def _bench_torch(N, C, dtype, device, warmup, iters):
    x = torch.randn(N, C, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(C, device=device, dtype=dtype, requires_grad=True)

    def fwd():
        return F.rms_norm(x, [C], g, 1e-6)

    fwd_ms = _time(fwd, warmup, iters)

    y = fwd()
    grad_out = torch.randn_like(y)

    def fwd_bwd():
        x.grad = None
        g.grad = None
        out = F.rms_norm(x, [C], g, 1e-6)
        out.backward(grad_out, retain_graph=False)

    fwdbwd_ms = _time(fwd_bwd, warmup, iters)
    return fwd_ms, fwdbwd_ms - fwd_ms


def _bandwidth_gbps(N, C, dtype, ms, kind):
    nbytes = _bytes(dtype)
    if kind == "fwd":
        # read x, write y
        total = 2 * N * C * nbytes
    else:
        # read x, dy; write dx; reduce dg (counted once per col)
        total = 4 * N * C * nbytes
    return (total / (ms * 1e-3)) / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    device = "cuda"
    platform, gpu = get_platform_info()

    print(f"GPU: {gpu} ({platform}) | dtype: {args.dtype} | " f"warmup={args.warmup} iters={args.iters}")

    rows = []
    for N, C in SHAPES:
        # correctness check vs F.rms_norm (signal-to-noise ratio in dB)
        snr_y, snr_dx, snr_dg = _correctness(N, C, dtype, device, atol=0, rtol=0)

        t_fwd, t_bwd = _bench_turbo(N, C, dtype, device, args.warmup, args.iters)
        py_fwd, py_bwd = _bench_torch(N, C, dtype, device, args.warmup, args.iters)

        row = {
            "Tokens": N,
            "Hidden": C,
            "SNR_y/dx/dg": f"{snr_y:.0f}/{snr_dx:.0f}/{snr_dg:.0f}",
            "Turbo fwd µs": f"{t_fwd*1e3:.1f}",
            "Turbo bwd µs": f"{t_bwd*1e3:.1f}",
            "Turbo fwd GB/s": f"{_bandwidth_gbps(N, C, dtype, t_fwd, 'fwd'):.0f}",
            "Turbo bwd GB/s": f"{_bandwidth_gbps(N, C, dtype, t_bwd, 'bwd'):.0f}",
            "torch fwd µs": f"{py_fwd*1e3:.1f}",
            "torch bwd µs": f"{py_bwd*1e3:.1f}",
            "fwd speedup": f"{py_fwd / t_fwd:.2f}x",
            "bwd speedup": f"{py_bwd / t_bwd:.2f}x",
        }
        rows.append(row)
        print(
            f"  N={N:6d} C={C:5d}  turbo fwd {t_fwd*1e3:7.1f}µs / bwd {t_bwd*1e3:7.1f}µs"
            f"  torch fwd {py_fwd*1e3:7.1f} / bwd {py_bwd*1e3:7.1f}"
            f"  speedup fwd {py_fwd/t_fwd:.2f}x bwd {py_bwd/t_bwd:.2f}x"
        )

    df = pd.DataFrame(rows)
    print()
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
