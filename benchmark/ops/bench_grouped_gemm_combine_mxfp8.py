###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-level EP benchmark: FUSED mxfp8 L2 GEMM + combine (+reduce) vs decoupled vs bf16.

The L2 (down-projection) mirror of ``bench_dispatch_grouped_gemm_mxfp8.py`` — compares the
forward L2 stage (grouped GEMM -> cross-rank combine PUSH -> weighted top-k reduce):

  * fp8 fused    -- ``grouped_gemm_combine_mxfp8`` (3-role: mxfp8 GEMM + combine + reduce, one kernel)
  * fp8 decoupled -- ``grouped_gemm_mxfp8_flydsl_kernel`` + ``combine_only`` + ``topk_reduce_only``
                     (three separate kernels, no overlap)
  * bf16 fused   -- ``grouped_gemm_combine_bf16`` (reference)

Reports ms/TFLOPS per path + accuracy (fused vs decoupled). Synthetic pool-grouped activation;
routing/origin tables from the dispatch prologue. Quant done once outside the timing loop.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo>:<repo>/benchmark/ops \
      python benchmark/ops/bench_grouped_gemm_combine_mxfp8.py --num-processes 8 --mode load_balanced
"""

import argparse
import datetime
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "training")))

from config import get_platform_info  # noqa: E402
from mega_utils import (  # noqa: E402
    bench,
    dispatch_prologue,
    gate3,
    generate_routing,
    get_symm_buffer_for_mega_moe,
    global_weights,
)

import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega.fp8.grouped_gemm_combine_mxfp8_kernel import (  # noqa: E402
    grouped_gemm_combine_mxfp8,
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (  # noqa: E402
    grouped_gemm_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.mega.fp8.quant import (  # noqa: E402
    quantize_grouped_weight_mxfp8,
    quantize_rowwise_mxfp8,
)
from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (  # noqa: E402
    combine_only,
    grouped_gemm_combine_bf16,
    topk_reduce_only,
)

_HANDLE_GROUP_OFFS = 10


def _all_max(group, v):
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())


def _all_min(group, v):
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return float(t.item())


def profile(group, args, mode, W2):
    rank, world = group.rank(), group.size()
    BM, BN, H, I = args.bm, args.bn, args.hidden, args.inter
    E, K, T, ncu = args.num_experts, args.num_topk, args.num_tokens, args.num_combine_cu
    epr = E // world

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_weight = generate_routing(T, K, E, mode, seed=100 + rank)

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    handle = tuple(
        dispatch_prologue(
            topk_idx, topk_weight, sym_layout=sym_layout, num_tokens=T, num_topk=K,
            num_experts=E, world_size=world, rank=rank, experts_per_rank=epr,
            block_m=BM, num_max_pool_tokens=symm.num_max_pool_tokens, no_cpu_sync=True,
        )
    )
    group_offs = handle[_HANDLE_GROUP_OFFS]
    num_tile_blocks = symm.meta_scalars[1:2]
    real_tiles = int(num_tile_blocks[0].item())
    M = int(symm.num_max_pool_tokens)
    M_eff = real_tiles * BM
    flops = 2.0 * M_eff * H * I  # L2: [M_eff, I] @ [H, I]^T -> [M_eff, H]

    # synthetic pool-grouped activation [M, I] + this rank's L2 weights [G, H, I]; quant ONCE
    act = (torch.randn((M, I), device="cuda", dtype=torch.float32) / (I**0.25)).bfloat16()
    aq, as_ = quantize_rowwise_mxfp8(act)
    w2q, w2s = quantize_grouped_weight_mxfp8(W2)
    topk_idx64 = topk_idx.to(torch.int64).contiguous().view(-1)
    topk_i32 = topk_idx.to(torch.int32).contiguous().view(-1)
    tw = topk_weight.to(torch.float32).contiguous().view(-1)
    combine_slots = int(sym_layout.combine_slots)
    num_experts = int(sym_layout.num_experts)
    topk = int(sym_layout.num_topk)

    def _sb_reset():
        symm.sb_l2.zero_()
        symm.barrier_local.fill_(-1)

    def _synced(fn, sb=False):
        torch.cuda.synchronize(); group.barrier()
        if sb:
            _sb_reset(); torch.cuda.synchronize(); group.barrier()
        out = fn(); torch.cuda.synchronize(); group.barrier()
        return out

    # ── fp8 decoupled parts: L2 gemm-only, combine-only, reduce-only ──
    t_gemm = bench(
        lambda: grouped_gemm_mxfp8_flydsl_kernel(aq, as_, w2q, w2s, group_offs, out_dtype=torch.bfloat16),
        iters=args.iters,
    )
    # pre-fill l2_token_buffer (combine PUSH source) then bench combine_only
    l2 = grouped_gemm_mxfp8_flydsl_kernel(aq, as_, w2q, w2s, group_offs, out_dtype=torch.bfloat16)
    symm.l2_token_buffer.copy_(l2)
    torch.cuda.synchronize(); group.barrier()
    t_combine = bench(lambda: combine_only(group, BM=BM, num_combine_cu=ncu), iters=args.iters)
    torch.cuda.synchronize(); group.barrier()
    y = torch.empty((int(symm.num_tokens), H), dtype=torch.bfloat16, device="cuda")
    symm.barrier_local.zero_()
    t_reduce = bench(
        lambda: topk_reduce_only(
            y, symm.comb, symm.barrier_local, topk_i32, symm.num_tokens_per_rank,
            combine_slots, topk=topk, num_experts=num_experts, rank=rank, topk_weights=tw,
        ),
        iters=args.iters,
    )
    decoupled = t_gemm + t_combine + t_reduce

    # ── fp8 fused: mxfp8 GEMM + combine + reduce (one kernel) ──
    t_fp8_fused = bench(
        lambda: grouped_gemm_combine_mxfp8(
            aq, as_, w2q, w2s, handle, group, topk_indices=topk_idx64, topk_weights=tw,
            BM=BM, BN=BN, num_combine_cu=ncu,
        ),
        iters=args.iters, reset=_sb_reset, group=group,
    )

    # ── bf16 fused (reference): same 3-role kernel, bf16 GEMM ──
    t_bf16_fused = bench(
        lambda: grouped_gemm_combine_bf16(
            act, W2, handle, group, topk_indices=topk_idx64, topk_weights=tw,
            BM=BM, BN=BN, num_combine_cu=ncu,
        )[0],
        iters=args.iters, reset=_sb_reset, group=group,
    )

    # ── accuracy: fp8 fused vs decoupled (established-correct) ──
    fused_y = _synced(
        lambda: grouped_gemm_combine_mxfp8(
            aq, as_, w2q, w2s, handle, group, topk_indices=topk_idx64, topk_weights=tw,
            BM=BM, BN=BN, num_combine_cu=ncu,
        ),
        sb=True,
    )
    dec_l2 = _synced(lambda: grouped_gemm_mxfp8_flydsl_kernel(aq, as_, w2q, w2s, group_offs, out_dtype=torch.bfloat16))
    symm.l2_token_buffer.copy_(dec_l2)
    _synced(lambda: combine_only(group, BM=BM, num_combine_cu=ncu))
    symm.barrier_local.zero_()
    y_dec = torch.empty((int(symm.num_tokens), H), dtype=torch.bfloat16, device="cuda")
    _synced(lambda: topk_reduce_only(
        y_dec, symm.comb, symm.barrier_local, topk_i32, symm.num_tokens_per_rank,
        combine_slots, topk=topk, num_experts=num_experts, rank=rank, topk_weights=tw,
    ))
    cos, _, _ = gate3(fused_y, y_dec)

    symm.destroy()
    return {
        "flops": flops, "M_eff": M_eff,
        "fp8_gemm": t_gemm, "fp8_combine": t_combine, "fp8_reduce": t_reduce, "fp8_decoupled": decoupled,
        "fp8_fused": t_fp8_fused, "bf16_fused": t_bf16_fused, "cos": cos,
    }


def _line(label, ms, tf=None, extra=""):
    s = f"  {label:<16}: {ms:8.3f} ms"
    if tf is not None:
        s += f" | {tf:8.1f} TFLOPS"
    return s + extra


def benchmark(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8492"))
    torch.cuda.set_device(local_rank)
    _timeout_s = int(os.getenv("MEGA_BENCH_TIMEOUT_S", "600"))
    dist.init_process_group(
        "nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank,
        timeout=datetime.timedelta(seconds=_timeout_s),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    _, gpu_name = get_platform_info()

    epr = args.num_experts // world
    _W1, W2_global = global_weights(args.num_experts, args.inter, args.hidden, "cuda")
    W2 = W2_global[rank * epr : (rank + 1) * epr].contiguous()
    del W2_global, _W1

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    try:
        for mode in modes:
            r = profile(group, args, mode, W2)
            gmax = lambda k: _all_max(group, r[k])
            flops = r["flops"]
            fp8_gemm, fp8_combine, fp8_reduce = gmax("fp8_gemm"), gmax("fp8_combine"), gmax("fp8_reduce")
            fp8_decoupled, fp8_fused, bf16_fused = gmax("fp8_decoupled"), gmax("fp8_fused"), gmax("bf16_fused")
            cos = _all_min(group, r["cos"])
            if rank != 0:
                torch.cuda.synchronize(); group.barrier(); continue

            tf = lambda ms: flops / (ms * 1e-3) / 1e12
            print(f"\n{'='*76}")
            print(f"[L2 GEMM+combine  fp8 fused vs decoupled vs bf16 fused]  {gpu_name} EP{world} "
                  f"T={args.num_tokens} H={args.hidden} I={args.inter} E={args.num_experts} "
                  f"K={args.num_topk} mode={mode} ncu={args.num_combine_cu} (max over ranks)")
            print(f"{'='*76}")
            print("  --- fp8 decoupled (gemm + combine + reduce, separate) ---")
            print(_line("gemm_only", fp8_gemm, tf(fp8_gemm)))
            print(_line("combine_only", fp8_combine))
            print(_line("reduce_only", fp8_reduce))
            print(_line("decoupled", fp8_decoupled, tf(fp8_decoupled), " (= gemm+combine+reduce)"))
            print("  --- fp8 fused (3-role: mxfp8 GEMM + combine + reduce) ---")
            print(_line("fused", fp8_fused, tf(fp8_fused),
                        f" | speedup vs decoupled = {fp8_decoupled / fp8_fused:.2f}x"))
            print("  --- bf16 fused (reference) ---")
            print(_line("fused", bf16_fused, tf(bf16_fused)))
            print(f"  --- fp8 fused = {fp8_fused:.3f} ms : {bf16_fused / fp8_fused:.2f}x vs bf16 fused "
                  f"({bf16_fused:.3f}), {fp8_decoupled / fp8_fused:.2f}x vs fp8 decoupled ({fp8_decoupled:.3f}) ---")
            print(f"  [acc] fp8 fused vs decoupled: cos={cos:.5f}  {'PASS' if cos >= 0.99 else 'FAIL'}")
            torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EP fused mxfp8 L2 GEMM+combine vs decoupled vs bf16 fused")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)  # DeepSeek-V3
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--num-combine-cu", type=int, default=64)
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="load_balanced")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.multiprocessing.spawn(benchmark, args=(args.num_processes, args), nprocs=args.num_processes)
