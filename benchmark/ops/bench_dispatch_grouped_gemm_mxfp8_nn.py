###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-level EP benchmark: FUSED fp8 dispatch(dy)+dgrad (NN, backward STEP1) vs bf16.

Backward STEP1 of the fused mega MoE: cross-rank dispatch PUSH of ``dy`` + grouped NN
dgrad ``grad_swiglu = dispatch(dy) @ w2`` on the SAME prologue-generated routing:

  * bf16 fused -- ``dispatch_grouped_gemm_bf16(layout="nn")`` (push bf16 dy + grouped bf16
                  NN dgrad); this is what the current backward STEP1 runs.
  * fp8 fused  -- ``dispatch_grouped_gemm_mxfp8_nn`` (3-stage clean-push raw fp8 dy + E8M0
                  -> preshuffle role -> preshuffled grouped mxfp8 NN dgrad, overlapped).

Reports the fused STEP1 latency per precision + an accuracy gate (fp8 vs bf16 grad_swiglu).
Quantization of dy/w2 is done ONCE outside the timing loop. Mirror of the forward L1
benchmark ``bench_dispatch_grouped_gemm_mxfp8.py``.

Run inside the dev container (8 GPUs):
  python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_nn.py --num-processes 8 --mode load_balanced
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
from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (  # noqa: E402
    dispatch_grouped_gemm_bf16,
)
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (  # noqa: E402
    dispatch_grouped_gemm_mxfp8,
)
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_bwd_kernel import (  # noqa: E402
    dispatch_grouped_gemm_mxfp8_bwd,
)
from primus_turbo.flydsl.mega.fp8.quant import (  # noqa: E402
    quantize_grouped_weight_mxfp8,
    quantize_rowwise_mxfp8,
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
    E, K, T, ndcu = args.num_experts, args.num_topk, args.num_tokens, args.num_dispatch_cu
    pscu = args.num_preshuffle_cu
    epr = E // world

    from primus_turbo.pytorch.core.low_precision import float8_e4m3, float8_e5m2

    dy_fmt = float8_e5m2 if args.dy_e5m2 else float8_e4m3
    w2_fmt = float8_e4m3

    # STEP1-backward optimization target: the dedicated bwd fork (dispatch_grouped_gemm_mxfp8_bwd),
    # which carries the preshuffle-fence tuning switches. PT_MXFP8_BWD_FORK=0 falls back to the
    # forward NT-reuse kernel (dispatch_grouped_gemm_mxfp8) for A/B comparison. Both are drop-in
    # (identical signature, both return the L1 out tensor) and both compute dgrad = dispatch(dy)@w2.
    _use_fork = os.environ.get("PT_MXFP8_BWD_FORK", "1") != "0"
    _fp8_fn = dispatch_grouped_gemm_mxfp8_bwd if _use_fork else dispatch_grouped_gemm_mxfp8

    torch.manual_seed(7 + rank)
    dy = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
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
    num_tile_blocks = symm.meta_scalars[1:2]

    # quantize ONCE (outside timing): fp8 dy + E8M0. NT-reuse dgrad: transpose w2 ->
    # [G, I, H] (b=[N=I,K=H]) and quantize along K=H (last dim) so the FORWARD L1 fused
    # NT kernel computes dgrad = dispatch(dy) @ w2 as a @ b^T.
    dyq, dys = quantize_rowwise_mxfp8(dy, dy_fmt)
    w2t = W2.transpose(1, 2).contiguous()  # [G, I, H]
    w2tq, w2ts = quantize_grouped_weight_mxfp8(w2t, w2_fmt)  # [G,I,H] / [G,I,H//32]

    real_tiles = int(num_tile_blocks[0].item())
    M_eff, N = real_tiles * BM, I
    flops = 2.0 * M_eff * N * H
    dest_cpu, count_cpu = handle[0].cpu(), handle[2].cpu()
    remote_rows = int(count_cpu[dest_cpu != rank].sum().item())
    xgmi_bf16 = remote_rows * H * 2
    xgmi_fp8 = remote_rows * (H + H // 32)

    def _synced(fn, reset_sb=False):
        torch.cuda.synchronize(); group.barrier()
        if reset_sb:
            symm.scoreboard.zero_(); torch.cuda.synchronize(); group.barrier()
        out = fn(); torch.cuda.synchronize(); group.barrier()
        return out

    # ── bf16 fused NN (the current backward STEP1) ──────────────────────────────
    t_bf16_fused = bench(
        lambda: dispatch_grouped_gemm_bf16(
            dy, W2, group, handle=handle, layout="nn", BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        iters=args.iters,
    )

    # ── fp8 fused NT-reuse (forward L1 NT kernel + transposed w2 => dgrad as a @ b^T) ──
    t_fp8_nt = bench(
        lambda: _fp8_fn(
            dyq, dys, w2tq, w2ts, handle, sym_layout, symm, BM=BM, BN=BN,
            num_dispatch_cu=ndcu, num_preshuffle_cu=pscu,
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )

    # ── accuracy: fp8 NT-reuse grad_swiglu vs bf16 fused grad_swiglu (real rows) ──
    # COHERENCE GATE (PT_MXFP8_ACC_FRESH, default 1): compare on a FRESH dy the timing loop NEVER
    # pushed. The timing loop above pushed `dyq` 30x, warming L2 with its pool_scale; pushing a
    # DIFFERENT dy here forces the preshuffle/gemm acquire to actually observe the freshly-pushed
    # scale. A missing/insufficient scale fence reads the stale (timing-loop) scale from the small,
    # L2-resident scale region -> cos collapses. Run the fp8 path FIRST, before the big bf16 push
    # evicts the warm scale lines. Set PT_MXFP8_ACC_FRESH=0 for the old (reuse-dyq) weak gate.
    _acc_fresh = os.environ.get("PT_MXFP8_ACC_FRESH", "1") != "0"
    if _acc_fresh:
        dy_acc = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
        dyq_acc, dys_acc = quantize_rowwise_mxfp8(dy_acc, dy_fmt)
    else:
        dy_acc, dyq_acc, dys_acc = dy, dyq, dys
    nt_out = _synced(
        lambda: _fp8_fn(
            dyq_acc, dys_acc, w2tq, w2ts, handle, sym_layout, symm, BM=BM, BN=BN,
            num_dispatch_cu=ndcu, num_preshuffle_cu=pscu,
        ),
        reset_sb=True,
    )
    bf16_out = _synced(
        lambda: dispatch_grouped_gemm_bf16(
            dy_acc, W2, group, handle=handle, layout="nn", BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        reset_sb=True,
    )[0]
    cos_nt, rel_nt, _ = gate3(nt_out[:M_eff], bf16_out[:M_eff])

    symm.destroy()
    return {
        "flops": flops, "M_eff": M_eff, "xgmi_bf16": xgmi_bf16, "xgmi_fp8": xgmi_fp8,
        "bf16_fused": t_bf16_fused, "fp8_nt": t_fp8_nt,
        "cos_nt": cos_nt, "rel_nt": rel_nt,
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
    _, W2_global = global_weights(args.num_experts, args.inter, args.hidden, "cuda")
    W2 = W2_global[rank * epr : (rank + 1) * epr].contiguous()
    del W2_global

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    try:
        for mode in modes:
            r = profile(group, args, mode, W2)
            gmax = lambda k: _all_max(group, r[k])
            flops = r["flops"]
            bf16_fused, fp8_nt = gmax("bf16_fused"), gmax("fp8_nt")
            cos_nt = _all_min(group, r["cos_nt"]); rel_nt = _all_max(group, r["rel_nt"])
            if rank != 0:
                torch.cuda.synchronize(); group.barrier(); continue

            tf = lambda ms: flops / (ms * 1e-3) / 1e12
            _kern = "bwd-fork" if os.environ.get("PT_MXFP8_BWD_FORK", "1") != "0" else "fwd-NTreuse"
            _acc = "fresh-dy" if os.environ.get("PT_MXFP8_ACC_FRESH", "1") != "0" else "reuse-dy"
            _kern = f"{_kern} acc={_acc}"
            print(f"\n{'='*80}")
            print(f"[bwd STEP1  dispatch(dy)+fc2 dgrad  fp8 NT-reuse vs bf16 fused]  {gpu_name} EP{world} "
                  f"T={args.num_tokens} H={args.hidden} I={args.inter} E={args.num_experts} "
                  f"K={args.num_topk} mode={mode} ndcu={args.num_dispatch_cu} pscu={args.num_preshuffle_cu} "
                  f"dy_fmt={'E5M2' if args.dy_e5m2 else 'E4M3'} kern={_kern} (max over ranks)")
            print(f"{'='*80}")
            print(_line("bf16 fused", bf16_fused, tf(bf16_fused)))
            print(_line("fp8 NT-reuse", fp8_nt, tf(fp8_nt),
                        f" | {bf16_fused / fp8_nt:.2f}x vs bf16 | cos={cos_nt:.5f} rel={rel_nt:.4f} "
                        f"{'PASS' if cos_nt >= 0.99 else 'FAIL'}"))
            torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EP fused fp8 backward STEP1 (NN dgrad) vs bf16 fused")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)  # DeepSeek-V3
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--num-dispatch-cu", type=int, default=16)
    ap.add_argument("--num-preshuffle-cu", type=int, default=16)
    ap.add_argument("--dy-e5m2", action="store_true", help="quantize dy as E5M2 (default E4M3)")
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="load_balanced")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.multiprocessing.spawn(benchmark, args=(args.num_processes, args), nprocs=args.num_processes)
