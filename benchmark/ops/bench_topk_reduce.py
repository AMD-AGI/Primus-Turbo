###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Standalone single-GPU precision + bandwidth bench for ``topk_reduce_only`` (role 2).

The reduce is pure streaming (memory-bound): per token it reads ``topk`` combine rows and
writes one output row, so traffic = ``T * H * 2 * (topk + 1)`` bytes. Sweeps ``num_reduce_cu``
(= grid blocks) to find the minimum that saturates HBM (~6.5 TB/s on MI355X).

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_topk_reduce.py
"""

import argparse

import torch
from mega_utils import topk_reduce_only
from tabulate import tabulate

_HBM_TBPS = 6.5  # MI355X achievable HBM bandwidth (the saturation target)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


_L2_FLUSH = None


def _l2_flush():
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH.zero_()


def _bench(fn, *, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(iters):
        _l2_flush()  # evict so every iter reads comb from HBM
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        total += e0.elapsed_time(e1)
    return total / iters  # ms/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=8192)  # DSv3 per-rank
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--cus", type=str, default="16,32,48,64,96,128,192,256", help="comma-separated num_reduce_cu sweep"
    )
    args = ap.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    T, H, topk, E = args.num_tokens, args.hidden, args.num_topk, args.num_experts
    slots = T * topk

    comb = torch.randn(slots, H, device="cuda", dtype=torch.bfloat16) / 8  # [T*topk, H]
    out = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")
    barrier = torch.ones(slots, dtype=torch.int32, device="cuda")  # all slots ready
    topk_idx = torch.zeros(slots, dtype=torch.int32, device="cuda")  # all valid (0 < E)
    ntok = torch.tensor([T], dtype=torch.int32, device="cuda")

    bytes_moved = T * H * 2 * (topk + 1)  # read topk rows + write 1 row, bf16
    print(
        f"[cfg] T={T} H={H} topk={topk} E={E} | comb {slots*H*2/1e6:.0f} MB  "
        f"traffic {bytes_moved/1e9:.2f} GB/call  target {_HBM_TBPS} TB/s"
    )

    # ---- precision (full-CU run) ----
    cu_max = max(int(c) for c in args.cus.split(","))
    topk_reduce_only(
        out, comb, barrier, topk_idx, ntok, slots, topk=topk, num_experts=E, rank=0, num_reduce_cu=cu_max
    )
    torch.cuda.synchronize()
    ref = comb.float().view(T, topk, H).sum(1)
    print(f"[precision] cos(out, sum_topk) = {_cos(out, ref):.6f}")

    # torch reduction as the practical ceiling for this [T,topk,H] -> [T,H] pattern
    comb3 = comb.view(T, topk, H)
    t_torch = _bench(lambda: torch.sum(comb3, dim=1), iters=args.iters)
    print(
        f"[ref] torch sum(dim=1): {t_torch*1e3:.1f} us | "
        f"{bytes_moved/(t_torch*1e-3)/1e12:.2f} TB/s (strided-reduce ceiling)"
    )
    # contiguous copy of the same bytes -> the pure-streaming HBM ceiling
    src = torch.empty(bytes_moved // 2 // 2, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    t_copy = _bench(lambda: dst.copy_(src), iters=args.iters)
    print(
        f"[ref] contiguous copy : {t_copy*1e3:.1f} us | "
        f"{2*src.numel()*2/(t_copy*1e-3)/1e12:.2f} TB/s (pure-stream R+W ceiling)"
    )
    # read-only ceiling (the reduce is ~8:1 read-heavy): full-buffer sum, reads comb, ~0 write
    t_rd = _bench(lambda: torch.sum(comb), iters=args.iters)
    print(
        f"[ref] read-only sum   : {t_rd*1e3:.1f} us | "
        f"{slots*H*2/(t_rd*1e-3)/1e12:.2f} TB/s (read-only ceiling)\n"
    )

    # ---- bandwidth sweep over num_reduce_cu ----
    rows = []
    best = (0, 0.0)
    min_sat = None
    for cu in [int(c) for c in args.cus.split(",")]:
        t_ms = _bench(
            lambda: topk_reduce_only(
                out, comb, barrier, topk_idx, ntok, slots, topk=topk, num_experts=E, rank=0, num_reduce_cu=cu
            ),
            iters=args.iters,
        )
        tbps = bytes_moved / (t_ms * 1e-3) / 1e12
        pct = 100.0 * tbps / _HBM_TBPS
        rows.append([cu, f"{t_ms*1e3:.1f}", f"{tbps:.2f}", f"{pct:.1f}%"])
        if tbps > best[1]:
            best = (cu, tbps)
        if min_sat is None and tbps >= 0.95 * _HBM_TBPS:
            min_sat = cu

    print(tabulate(rows, headers=["num_reduce_cu", "us", "TB/s", "% of 6.5"], tablefmt="github"))
    print(f"\n[peak] {best[1]:.2f} TB/s @ {best[0]} CU")
    if min_sat is not None:
        print(f"[min CU for >=95% of {_HBM_TBPS} TB/s] {min_sat}")
    else:
        print(f"[saturation] never reached 95% of {_HBM_TBPS} TB/s in the swept range")


if __name__ == "__main__":
    main()
