###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU re-measurement of the mega-MoE GEMM leg (grouped GEMM vs dense roofline).

bench_mega_moe.py runs the GEMM-only baseline on all 8 ranks at once, so every
GPU is hot and the whole node downclocks -> the "GEMM leg" numbers are pessimistic.
The GEMM leg has no cross-rank dependency, so it can be replayed on ONE GPU with
the exact same shapes.

Geometry (M_eff / tile_to_expert) is reproduced by simulating the EP8 routing of
all `world` ranks and keeping what lands on this rank's local experts, then
padding each expert to a BLOCK_M multiple -- identical to what the prologue
produces in the distributed run.

Only gemm_only + dense_gemm are measured here; comm / fused need the real EP group.

    python3 bench_mega_moe_gemm_only_1gpu.py --models DeepSeek-V3
"""

import argparse
import os
import sys

import torch
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_mega_moe as B  # noqa: E402
from config import gen_moe_test_cases  # noqa: E402

BM = 256
BN = 256

# Models benched for the blog that are not (or no longer) in config.MoEModelConfigs.
EXTRA_MODELS = {
    "DeepSeek-V4-Flash": {"hidden": 4096, "inter": 2048, "num_experts": 256, "num_topk": 6},
}


def resolve_cases(names):
    """config models first, then the local extras, in the order the user asked for."""
    from_config = {c["Case"]: c for c in gen_moe_test_cases(None)}
    cases = []
    for name in names:
        if name in from_config:
            cases.append(from_config[name])
        elif name in EXTRA_MODELS:
            cases.append({"Case": name, **EXTRA_MODELS[name]})
        else:
            raise SystemExit(f"unknown model: {name}")
    return cases


def make_geometry(T, E, K, world, rank, *, block_m=BM, seed=0):
    """Replay EP routing on one GPU -> (M_eff, tile_to_expert i32, num_tile_blocks i32[1], group_offs i32)."""
    experts_per_rank = E // world
    lo, hi = rank * experts_per_rank, (rank + 1) * experts_per_rank
    torch.manual_seed(seed)
    counts = torch.zeros(experts_per_rank, dtype=torch.int64, device="cuda")
    for _ in range(world):  # each peer rank routes T tokens; keep the ones aimed at us
        topk_idx, _ = B.generate_routing(T, K, E, device="cuda")
        sel = topk_idx[(topk_idx >= lo) & (topk_idx < hi)] - lo
        counts += torch.bincount(sel.flatten(), minlength=experts_per_rank)
    tiles = (counts + block_m - 1) // block_m  # expert-major pool: pad each expert to BLOCK_M
    tile_to_expert = torch.repeat_interleave(
        torch.arange(experts_per_rank, device="cuda", dtype=torch.int32), tiles
    ).contiguous()
    real_tiles = int(tiles.sum().item())
    num_tile_blocks = torch.tensor([real_tiles], dtype=torch.int32, device="cuda")
    group_offs = torch.zeros(experts_per_rank + 1, dtype=torch.int32, device="cuda")
    group_offs[1:] = (tiles * block_m).to(torch.int32).cumsum(0)
    return real_tiles * block_m, tile_to_expert, num_tile_blocks, group_offs


def make_variable_k_only(lhs, rhs, group_offs, out, *, num_xcd=1):
    """Grouped TN wgrad (variable-K) closure, no per-call output alloc.

    Mirrors bench_mega_moe.grouped_gemm_variable_k_only but against the current
    (gather-capable) launcher signature. Operands are swapped (as the bench does)
    so C = rhs^T @ lhs lands as [G, 2I, H] with a coalesced store.
    """
    from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
        _compile_grouped_variable_k_bf16,
        _ptr_only_view,
    )

    # trans_c: swap operands so C^T = rhs^T @ lhs stores coalesced
    lhs_e, rhs_e = rhs, lhs
    OUT_M_e, OUT_N_e = lhs_e.shape[1], rhs_e.shape[1]
    G = group_offs.numel() - 1
    offsets_i64 = group_offs.to(torch.int64)
    masked_k_i64 = (offsets_i64[1:] - offsets_i64[:-1]).contiguous()
    launch = _compile_grouped_variable_k_bf16(
        OUT_M_e, OUT_N_e, G, BLOCK_M=BM, BLOCK_N=BN, num_xcd=num_xcd, out_fp16=False
    )
    slot_src = masked_k_i64.view(torch.int32)  # unused placeholder table (no gather)
    gemm_args = (
        _ptr_only_view(lhs_e),
        _ptr_only_view(rhs_e),
        B.flyc.from_torch_tensor(out),
        offsets_i64,
        masked_k_i64,
        B.flyc.from_torch_tensor(slot_src),
        slot_src.numel(),
        OUT_M_e,
        OUT_N_e,
        torch.cuda.current_stream(),
    )
    compiled = B.flyc.compile(launch, *gemm_args)
    return lambda: compiled(*gemm_args)


def measure(name, *, gemm_fn, dense_dims, flops, iters):
    """Time the grouped GEMM and the same-FLOPs dense roofline; return one report row."""
    t_gemm = B.bench(gemm_fn, iters=iters)
    t_dense, dense_gm = B.dense_gemm_peak_ms(*dense_dims, BM, BN, iters, group_m_cands=(4,))
    gemm_tf = flops / (t_gemm * 1e-3) / 1e12
    dense_tf = flops / (t_dense * 1e-3) / 1e12
    return {
        "stage": name,
        "dense (ms)": round(t_dense, 3),
        "dense (TFLOPS)": round(dense_tf, 1),
        "gemm_only (ms)": round(t_gemm, 3),
        "gemm_only (TFLOPS)": round(gemm_tf, 1),
        "grouped/dense": f"{gemm_tf / dense_tf * 100:.1f}%",
        "GROUP_M(dense)": dense_gm,
    }


def bf16(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) / 8


def run_case(case, args):
    H, I, E, K = case["hidden"], case["inter"], case["num_experts"], case["num_topk"]
    world, T = args.world, args.num_tokens
    Er = E // world
    M_eff, tile_to_expert, num_tile_blocks, group_offs = make_geometry(
        T, E, K, world, args.rank, seed=args.seed
    )
    print(
        f"\n=== {case['Case']}  H={H} I={I} E={E} topk={K} | EP{world} T={T} -> M_eff={M_eff} ===", flush=True
    )

    rows = []
    W1 = bf16(Er, 2 * I, H)  # [G, 2I, H]: NT weight for L1, NN weight (K=2I,N=H) for combine bwd
    W2 = bf16(Er, H, I)  # [G, H, I]: NN weight (K=H,N=I) for L1 dgrad, NT weight (N=H,K=I) for L2

    # ---- dispatch_grouped_gemm leg -------------------------------------------------
    pool = bf16(M_eff, H)
    l1_out = torch.empty(M_eff, 2 * I, device="cuda", dtype=torch.bfloat16)
    rows.append(
        measure(
            "dispatch forward (nt)",
            gemm_fn=lambda: B.grouped_gemm_bf16_only(
                pool, W1, l1_out, tile_to_expert, num_tile_blocks, BLOCK_M=BM, BLOCK_N=BN
            ),
            dense_dims=(M_eff, 2 * I, H),
            flops=2.0 * M_eff * (2 * I) * H,
            iters=args.iters,
        )
    )
    d_swiglu = torch.empty(M_eff, I, device="cuda", dtype=torch.bfloat16)
    rows.append(
        measure(
            "dispatch bwd dgrad (nn)",
            gemm_fn=lambda: B.grouped_gemm_bf16_only(
                pool, W2, d_swiglu, tile_to_expert, num_tile_blocks, layout="nn", BLOCK_M=BM, BLOCK_N=BN
            ),
            dense_dims=(M_eff, I, H),
            flops=2.0 * M_eff * I * H,
            iters=args.iters,
        )
    )
    d_swiglu = None  # drop the ref so the next alloc can reuse it
    grad_pool = bf16(M_eff, 2 * I)
    dW1 = torch.empty(Er, 2 * I, H, device="cuda", dtype=torch.bfloat16)  # trans_c -> [G, 2I, H]
    rows.append(
        measure(
            "dispatch bwd wgrad (tn)",
            gemm_fn=make_variable_k_only(pool, grad_pool, group_offs, dW1),
            dense_dims=(H, 2 * I, M_eff),
            flops=2.0 * M_eff * H * (2 * I),
            iters=args.iters,
        )
    )
    pool = l1_out = dW1 = None
    torch.cuda.empty_cache()

    # ---- grouped_gemm_combine leg --------------------------------------------------
    act = bf16(M_eff, I)
    l2_out = torch.empty(M_eff, H, device="cuda", dtype=torch.bfloat16)
    rows.append(
        measure(
            "combine forward (nt)",
            gemm_fn=lambda: B.grouped_gemm_bf16_only(
                act, W2, l2_out, tile_to_expert, num_tile_blocks, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=8
            ),
            dense_dims=(M_eff, H, I),
            flops=2.0 * M_eff * H * I,
            iters=args.iters,
        )
    )
    act = None
    rows.append(
        measure(
            "combine bwd dgrad (nn)",
            gemm_fn=lambda: B.grouped_gemm_bf16_only(
                grad_pool,
                W1,
                l2_out,
                tile_to_expert,
                num_tile_blocks,
                layout="nn",
                BLOCK_M=BM,
                BLOCK_N=BN,
                GROUP_M=8,
            ),
            dense_dims=(M_eff, H, 2 * I),
            flops=2.0 * M_eff * H * (2 * I),
            iters=args.iters,
        )
    )
    grad_pool = l2_out = W1 = W2 = None
    torch.cuda.empty_cache()

    for r in rows:
        r["Case"] = case["Case"]
        r["M_eff"] = M_eff
    print(tabulate(rows, headers="keys", tablefmt="github"), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="Single-GPU mega-MoE GEMM-leg benchmark")
    p.add_argument("--models", nargs="+", default=["DeepSeek-V3", "DeepSeek-V4-Pro", "DeepSeek-V4-Flash"])
    p.add_argument("--num-tokens", type=int, default=8192)
    p.add_argument("--world", type=int, default=8, help="EP size to emulate (geometry only)")
    p.add_argument("--rank", type=int, default=0, help="which rank's local experts to emulate")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", "-o", type=str, default=None)
    args = p.parse_args()

    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)} (single GPU, no EP group)", flush=True)

    all_rows = []
    for case in resolve_cases(args.models):
        if case["Case"] in B.UNSUPPORTED:
            print(f"skip {case['Case']} (unsupported tiling)")
            continue
        all_rows.extend(run_case(case, args))

    print("\n===== summary =====")
    print(tabulate(all_rows, headers="keys", tablefmt="github"))
    if args.output:
        import pandas as pd

        pd.DataFrame(all_rows).to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
