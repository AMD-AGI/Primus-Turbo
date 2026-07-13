###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-level EP benchmark: FUSED fp8 dispatch+GEMM (role pipeline) vs bf16 fused vs
decoupled fp8.

Apples-to-apples comparison of the forward L1 stage (cross-rank dispatch PUSH + grouped
GEMM) on the SAME prologue-generated routing:

  * bf16 fused    -- ``dispatch_grouped_gemm_bf16`` (push bf16 tokens + grouped bf16 NT GEMM)
  * fp8 decoupled -- ``dispatch_fp8_push_launch`` (push fp8 + raw E8M0) then
                     ``grouped_gemm_mxfp8_flydsl_kernel`` (preshuffle + grouped mxfp8 GEMM),
                     as two separate kernels (no comm/compute overlap).
  * fp8 fused     -- ``dispatch_grouped_gemm_mxfp8`` (3-stage pipeline: clean-push raw fp8 +
                     E8M0 -> preshuffle role -> preshuffled grouped mxfp8 GEMM, overlapped).

Reports ``gemm_only`` / ``dispatch_only`` / ``fused`` per precision + an accuracy gate.
This is the kernel-latency number (one dispatch+L1 GEMM), NOT the whole MoE forward.
Quantization of tokens/weights is done ONCE outside the timing loop.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo>:<repo>/benchmark/ops \
      python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8.py --num-processes 8 --mode load_balanced
"""

import argparse
import datetime
import os
import sys

import torch
import torch.distributed as dist

_VERBOSE = os.getenv("MEGA_BENCH_VERBOSE", "0") == "1"


def _v(rank, msg):
    if _VERBOSE:
        print(f"[rank{rank}] {msg}", flush=True)


_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "training")))

from config import get_platform_info  # noqa: E402
from mega_utils import (  # noqa: E402
    bench,
    dense_gemm_peak_ms,
    dispatch_only,
    dispatch_prologue,
    gate3,
    generate_routing,
    get_symm_buffer_for_mega_moe,
    global_weights,
    grouped_gemm_bf16_only,
)

import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (  # noqa: E402
    dispatch_grouped_gemm_bf16,
)
from primus_turbo.flydsl.mega.fp8.dispatch_fp8_push_kernel import (  # noqa: E402
    dispatch_fp8_push_launch,
)
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (  # noqa: E402
    dispatch_grouped_gemm_mxfp8,
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (  # noqa: E402
    grouped_gemm_mxfp8_flydsl_kernel,
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


def profile(group, args, mode, W1):
    rank, world = group.rank(), group.size()
    BM, BN, H, I = args.bm, args.bn, args.hidden, args.inter
    E, K, T, ndcu = args.num_experts, args.num_topk, args.num_tokens, args.num_dispatch_cu
    pscu = args.num_preshuffle_cu
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
    tile_to_expert, expected = handle[7], handle[8]
    num_tile_blocks = symm.meta_scalars[1:2]
    group_offs = handle[_HANDLE_GROUP_OFFS]

    # quantize ONCE (outside timing): fp8 tokens + E8M0, fp8 weights + E8M0
    xq, xs = quantize_rowwise_mxfp8(x)
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)

    real_tiles = int(num_tile_blocks[0].item())
    M_eff, N = real_tiles * BM, 2 * I
    flops = 2.0 * M_eff * N * H
    dest_cpu, count_cpu = handle[0].cpu(), handle[2].cpu()
    remote_rows = int(count_cpu[dest_cpu != rank].sum().item())
    xgmi_bf16 = remote_rows * H * 2
    xgmi_fp8 = remote_rows * (H + H // 32)  # fp8 token byte + E8M0 scale byte

    def _synced(fn, reset_sb=False):
        torch.cuda.synchronize(); group.barrier()
        if reset_sb:
            symm.scoreboard.zero_(); torch.cuda.synchronize(); group.barrier()
        out = fn(); torch.cuda.synchronize(); group.barrier()
        return out

    # ── bf16: gemm_only / dispatch_only / fused ────────────────────────────────
    l1_bf16 = torch.empty((symm.num_max_pool_tokens, N), dtype=torch.bfloat16, device="cuda")
    _synced(lambda: dispatch_only(x, handle, symm.pool, symm.pool_ptrs, num_dispatch_cu=ndcu))
    t_bf16_gemm = bench(
        lambda: grouped_gemm_bf16_only(symm.pool, W1, l1_bf16, tile_to_expert, num_tile_blocks, BM=BM, BN=BN),
        iters=args.iters,
    )
    t_bf16_disp = bench(
        lambda: dispatch_only(x, handle, symm.pool, symm.pool_ptrs, num_dispatch_cu=ndcu), iters=args.iters
    )
    t_bf16_fused = bench(
        lambda: dispatch_grouped_gemm_bf16(
            x, W1, group, handle=handle, layout="nt", BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        iters=args.iters,
    )

    # ── fp8 decoupled: push (dispatch_only) + preshuffle+GEMM (gemm_only), no overlap ──
    torch.cuda.synchronize(); group.barrier()
    dispatch_fp8_push_launch(xq, xs, handle, sym_layout, symm.num_max_pool_tokens, world)
    torch.cuda.synchronize(); group.barrier()
    t_fp8_gemm = bench(
        lambda: grouped_gemm_mxfp8_flydsl_kernel(
            symm.pool_fp8, symm.pool_scale, w1q, w1s, group_offs, out_dtype=torch.bfloat16
        ),
        iters=args.iters,
    )
    t_fp8_disp = bench(
        lambda: dispatch_fp8_push_launch(xq, xs, handle, sym_layout, symm.num_max_pool_tokens, world),
        iters=args.iters,
    )

    # ── fp8 fused: 3-stage clean-push -> preshuffle role -> preshuffled GEMM pipeline ──
    _v(rank, "bench fp8 fused (role pipeline) ...")
    t_fp8_fused = bench(
        lambda: dispatch_grouped_gemm_mxfp8(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN,
            num_dispatch_cu=ndcu, num_preshuffle_cu=pscu,
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    _v(rank, "fp8 fused done")

    # ── accuracy on a FRESH x the timing loop NEVER pushed (stale-read / coherence gate) ──
    # PT_MXFP8_ACC_FRESH (default 1): push a DIFFERENT x here so the preshuffle/gemm acquire must
    # observe the freshly-pushed scale; a missing/insufficient scale fence reads the stale
    # (timing-loop xq) scale from the small L2-resident scale region -> cos collapses. Run the fp8
    # fused path FIRST (L2 still warm with the timing xq), then the decoupled ref + bf16.
    _acc_fresh = os.environ.get("PT_MXFP8_ACC_FRESH", "1") != "0"
    if _acc_fresh:
        x_acc = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
        xq_acc, xs_acc = quantize_rowwise_mxfp8(x_acc)
    else:
        x_acc, xq_acc, xs_acc = x, xq, xs
    fp8_out = _synced(
        lambda: dispatch_grouped_gemm_mxfp8(
            xq_acc, xs_acc, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN,
            num_dispatch_cu=ndcu, num_preshuffle_cu=pscu,
        ),
        reset_sb=True,
    )
    ref_fp8 = _synced(  # decoupled mxfp8 GEMM over the pool fp8_out just pushed (same x_acc)
        lambda: grouped_gemm_mxfp8_flydsl_kernel(
            symm.pool_fp8, symm.pool_scale, w1q, w1s, group_offs, out_dtype=torch.bfloat16
        )
    )
    bf16_out = _synced(
        lambda: dispatch_grouped_gemm_bf16(
            x_acc, W1, group, handle=handle, layout="nt", BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        reset_sb=True,
    )[0]
    cos_vs_ref, _, _ = gate3(fp8_out[:M_eff], ref_fp8[:M_eff])
    cos_vs_bf16, _, _ = gate3(fp8_out[:M_eff], bf16_out[:M_eff])

    symm.destroy()
    return {
        "flops": flops, "M_eff": M_eff, "xgmi_bf16": xgmi_bf16, "xgmi_fp8": xgmi_fp8,
        "bf16_gemm": t_bf16_gemm, "bf16_disp": t_bf16_disp, "bf16_fused": t_bf16_fused,
        "fp8_gemm": t_fp8_gemm, "fp8_disp": t_fp8_disp, "fp8_fused": t_fp8_fused,
        "cos_vs_ref": cos_vs_ref, "cos_vs_bf16": cos_vs_bf16,
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
    W1_global, _ = global_weights(args.num_experts, args.inter, args.hidden, "cuda")
    W1 = W1_global[rank * epr : (rank + 1) * epr].contiguous()
    del W1_global

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    try:
        for mode in modes:
            r = profile(group, args, mode, W1)
            gmax = lambda k: _all_max(group, r[k])
            flops = r["flops"]
            bf16_gemm, bf16_disp, bf16_fused = gmax("bf16_gemm"), gmax("bf16_disp"), gmax("bf16_fused")
            fp8_gemm, fp8_disp, fp8_fused = gmax("fp8_gemm"), gmax("fp8_disp"), gmax("fp8_fused")
            cos_ref = _all_min(group, r["cos_vs_ref"])
            cos_bf16 = _all_min(group, r["cos_vs_bf16"])
            dense_ms, dgm = dense_gemm_peak_ms(
                r["M_eff"], 2 * args.inter, args.hidden, args.bm, args.bn, args.iters
            ) if rank == 0 else (0.0, 0)
            if rank != 0:
                torch.cuda.synchronize(); group.barrier(); continue

            tf = lambda ms: flops / (ms * 1e-3) / 1e12
            bw = lambda by, ms: by / (ms * 1e-3) / 1e9
            fp8_decoupled = fp8_gemm + fp8_disp
            print(f"\n{'='*76}")
            print(f"[dispatch+L1 GEMM  fp8 fused vs bf16 fused vs decoupled]  {gpu_name} EP{world} "
                  f"T={args.num_tokens} H={args.hidden} I={args.inter} E={args.num_experts} "
                  f"K={args.num_topk} mode={mode} ndcu={args.num_dispatch_cu} pscu={args.num_preshuffle_cu} "
                  f"(max over ranks)")
            print(f"{'='*76}")
            print(_line("dense_gemm", dense_ms, tf(dense_ms) if dense_ms else None,
                        f" (single-weight roofline, GROUP_M={dgm})"))
            print("  --- bf16 (push bf16 tokens) ---")
            print(_line("gemm_only", bf16_gemm, tf(bf16_gemm)))
            print(_line("dispatch_only", bf16_disp, extra=f" | {bw(r['xgmi_bf16'], bf16_disp):7.1f} GB/s XGMI"))
            print(_line("fused", bf16_fused, tf(bf16_fused),
                        f" | speedup vs serial = {(bf16_gemm + bf16_disp) / bf16_fused:.2f}x"))
            print("  --- fp8 decoupled (push + preshuffle+GEMM, no overlap) ---")
            print(_line("gemm_only", fp8_gemm, tf(fp8_gemm)))
            print(_line("dispatch_only", fp8_disp, extra=f" | {bw(r['xgmi_fp8'], fp8_disp):7.1f} GB/s XGMI "
                        f"({r['xgmi_bf16'] / r['xgmi_fp8']:.2f}x fewer bytes vs bf16)"))
            print(_line("decoupled", fp8_decoupled, tf(fp8_decoupled), " (= push + gemm, serial)"))
            print("  --- fp8 fused (3-stage: clean-push -> preshuffle role -> gemm) ---")
            print(_line("fused", fp8_fused, tf(fp8_fused),
                        f" | speedup vs decoupled = {fp8_decoupled / fp8_fused:.2f}x"))
            print(f"  --- fp8 fused = {fp8_fused:.3f} ms : {bf16_fused / fp8_fused:.2f}x vs bf16 fused "
                  f"({bf16_fused:.3f}), {fp8_decoupled / fp8_fused:.2f}x vs fp8 decoupled ({fp8_decoupled:.3f}) ---")
            print(f"  [acc] fp8 fused vs decoupled-mxfp8 ref: cos={cos_ref:.5f}  vs bf16: cos={cos_bf16:.5f}  "
                  f"{'PASS' if cos_ref >= 0.99 else 'FAIL'}")
            torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EP fused fp8 (role pipeline) vs bf16 fused vs decoupled")
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
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="load_balanced")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.multiprocessing.spawn(benchmark, args=(args.num_processes, args), nprocs=args.num_processes)
