"""Single-GPU equivalence + perf for the DG wave-scheduled grouped GEMM.

The wave scheduler must cover exactly the same (block_m, block_n) tile set as the
baseline GROUP_M scheduler, so outputs match bit-for-bit. We build a synthetic
expert-grouped pool layout, run both, and compare. Then sweep num_expert_per_wave.

  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_gemm_bf16_dg_wave.py
"""

import argparse

import torch

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_dg_kernel import (
    grouped_gemm_bf16_dg_only,
)
from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (
    grouped_gemm_bf16_only,
)


def _build_layout(epr, W, bm, tokens_per_expert, K, N, seed=0):
    """Expert-grouped dense pool + wave_block_start for `epr` experts, W per wave."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    assert epr % W == 0
    nblk = [((t + bm - 1) // bm) for t in tokens_per_expert]  # blocks per expert
    block_off = [0]
    for nb in nblk:
        block_off.append(block_off[-1] + nb)
    total_blocks = block_off[-1]
    total_rows = total_blocks * bm
    tile_to_expert = torch.empty(total_blocks, dtype=torch.int32, device="cuda")
    for e in range(epr):
        tile_to_expert[block_off[e] : block_off[e + 1]] = e
    num_waves = epr // W
    wbs = [block_off[w * W] for w in range(num_waves)] + [total_blocks]
    wave_block_start = torch.tensor(wbs, dtype=torch.int32, device="cuda")
    num_tile_blocks = torch.tensor([total_blocks], dtype=torch.int32, device="cuda")
    pool = (torch.randn(total_rows, K, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    weight = (torch.randn(epr, N, K, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    return pool, weight, tile_to_expert, wave_block_start, num_tile_blocks, total_rows


def _bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epr", type=int, default=8)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--group-m", type=int, default=4)
    ap.add_argument("--K", type=int, default=7168)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--tokens-per-expert", type=int, default=512)
    args = ap.parse_args()
    torch.cuda.set_device(0)

    epr, bm, K, N = args.epr, args.bm, args.K, args.N
    tpe = [args.tokens_per_expert] * epr  # balanced; tweak for imbalance
    print(
        f"[cfg] epr={epr} bm={bm} bn={args.bn} GROUP_M={args.group_m} K={K} N={N} tpe={args.tokens_per_expert}"
    )

    pool, weight, ttg, _, ntb, total_rows = _build_layout(epr, epr, bm, tpe, K, N)
    out_base = torch.zeros(pool.size(0), N, dtype=torch.bfloat16, device="cuda")
    grouped_gemm_bf16_only(
        pool, weight, out_base, ttg, ntb, layout="nt", BM=bm, BN=args.bn, GROUP_M=args.group_m
    )
    torch.cuda.synchronize()

    print(f"{'W':>4} {'num_waves':>10} {'max_abs_diff':>14} {'us':>9}  match")
    base_us = _bench(
        lambda: grouped_gemm_bf16_only(
            pool, weight, out_base, ttg, ntb, layout="nt", BM=bm, BN=args.bn, GROUP_M=args.group_m
        )
    )
    print(f"{'base':>4} {'(GROUP_M)':>10} {0.0:14.6f} {base_us:9.2f}  -")

    for W in [w for w in range(1, epr + 1) if epr % w == 0]:
        _, _, ttg_w, wbs, ntb_w, _ = _build_layout(epr, W, bm, tpe, K, N)
        out_dg = torch.zeros(pool.size(0), N, dtype=torch.bfloat16, device="cuda")
        grouped_gemm_bf16_dg_only(
            pool, weight, out_dg, ttg_w, wbs, ntb_w, layout="nt", BM=bm, BN=args.bn, GROUP_M=args.group_m
        )
        torch.cuda.synchronize()
        diff = (out_dg.float() - out_base.float()).abs().max().item()
        match = "OK" if diff == 0.0 else ("close" if diff < 1e-2 else "MISMATCH")
        us = _bench(
            lambda: grouped_gemm_bf16_dg_only(
                pool, weight, out_dg, ttg_w, wbs, ntb_w, layout="nt", BM=bm, BN=args.bn, GROUP_M=args.group_m
            )
        )
        print(f"{W:>4} {epr // W:>10} {diff:14.6f} {us:9.2f}  {match}")


if __name__ == "__main__":
    main()
