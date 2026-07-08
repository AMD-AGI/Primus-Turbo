###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-level EP benchmark: FUSED fp8 dispatch+GEMM vs FUSED bf16 dispatch+GEMM.

Apples-to-apples comparison of the forward L1 stage (cross-rank dispatch PUSH +
grouped GEMM, overlapped in ONE kernel), on the SAME prologue-generated routing:

  * bf16  -- ``dispatch_grouped_gemm_bf16`` (push bf16 tokens + grouped bf16 NT GEMM)
  * fp8   -- ``dispatch_grouped_gemm_mxfp8`` (push fp8 tokens + E8M0 scales + grouped
             per-1x32 mxfp8 NT GEMM). fp8 halves the dispatch bytes.

For each precision it reports ``gemm_only`` / ``dispatch_only`` / ``fused`` (the same
metric template as ``bench_dispatch_grouped_gemm.py``) plus an accuracy gate. This is
the CORRECT kernel-latency number (one dispatch+L1 GEMM), NOT the whole MoE forward
(which additionally re-quantizes weights, runs SwiGLU + L2 + combine + host syncs).

Quantization of the tokens/weights is done ONCE outside the timing loop (in a real
training step the weights are quantized once per step, not per micro-op).

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo>:<repo>/benchmark/ops:<repo>/benchmark/ops/training \
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
    dispatch_grouped_gemm_mxfp8_cleanpush,
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (  # noqa: E402
    grouped_gemm_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.mega.fp8.quant import (  # noqa: E402
    quantize_grouped_weight_mxfp8,
    quantize_rowwise_mxfp8,
)
from primus_turbo.flydsl.mega.fp8.quant_flydsl import (  # noqa: E402
    quantize_rowwise_mxfp8_flydsl,
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

    # ── fp8: gemm_only / dispatch_only / fused ─────────────────────────────────
    # fill pool_fp8 / pool_scale once (peers synced) for the gemm-only baseline
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
    t_fp8_fused = bench(
        lambda: dispatch_grouped_gemm_mxfp8(
            x, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        iters=args.iters,
    )
    # DIAG: fused with the L2 wb/inv coherence fences disabled (wrong answer; isolates the
    # fence serialization cost vs the raw-scale gemm cost within the fused kernel).
    t_fp8_fused_nofence = bench(
        lambda: dispatch_grouped_gemm_mxfp8(
            x, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu, no_fence=True
        ),
        iters=args.iters,
    )
    # DIAG: fused GEMM role ONLY (comm disabled, pool pre-filled) = the fused kernel's
    # internal RAW-scale grouped GEMM in isolation. If ~= fused_nofence, comm is fully
    # overlapped and the raw-scale gemm is the whole cost (vs the preshuffled gemm_only).
    torch.cuda.synchronize(); group.barrier()
    dispatch_fp8_push_launch(xq, xs, handle, sym_layout, symm.num_max_pool_tokens, world)
    torch.cuda.synchronize(); group.barrier()
    t_fp8_fused_nogate = bench(
        lambda: dispatch_grouped_gemm_mxfp8(
            x, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu, no_gate=True
        ),
        iters=args.iters,
    )

    # ── DIAG: decompose the comm role (quant-in-push) that sits on the fused critical path.
    #   (a) quant_local  : FlyDSL rowwise mxfp8 quant of the T local tokens -> fp8 + broadcast
    #                      E8M0 scale, written to LOCAL hbm (no cross-rank push). Isolates the
    #                      quant compute + local write (the work fused adds on top of a plain push).
    #   (b) comm_only    : the fused kernel's comm role ALONE (read bf16 -> quant -> PUSH fp8 +
    #                      broadcast scale to peer pool), gemm grid removed. This is "pure push +
    #                      quant". comm_only - fp8_disp = the quant/broadcast-scale tax over a
    #                      plain fp8 copy-push (which reads pre-quant fp8 + pushes the raw scale).
    torch.cuda.synchronize(); group.barrier()
    _v(rank, "bench quant_local ...")
    t_fp8_quant_local = bench(
        lambda: quantize_rowwise_mxfp8_flydsl(x, preshuffle=True), iters=args.iters
    )
    _v(rank, "bench comm_only ...")
    t_fp8_comm_only = bench(
        lambda: dispatch_grouped_gemm_mxfp8(
            x, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu, comm_only=True
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    _v(rank, "comm_only done")
    # broadcast-layout scale is 4x the raw scale bytes (K128*16 vs K128*4 per row)
    xgmi_fp8_ps = remote_rows * (H + (H // 128) * 16)
    quant_bytes = T * H * 2 + T * H + T * (H // 128) * 16  # read bf16 + write fp8 + write bcast scale

    # ── CLEAN-PUSH fused: comm role copies PRE-QUANTIZED fp8 + RAW scale (coalesced, like
    #   dispatch_fp8_push -> XGMI-saturating), gemm role reads the raw scale on-the-fly
    #   (ScaleS2RRaw). Isolates whether decoupling quant from the push recovers the comm BW
    #   (quant-in-push collapsed it to ~78-148 GB/s). Tokens pre-quantized ONCE outside.
    torch.cuda.synchronize(); group.barrier()
    dispatch_fp8_push_launch(xq, xs, handle, sym_layout, symm.num_max_pool_tokens, world)
    torch.cuda.synchronize(); group.barrier()
    _v(rank, "bench cleanpush gemm-only / comm-only / fused ...")
    t_fp8_cp_nogate = bench(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu, no_gate=True
        ),
        iters=args.iters,
    )
    t_fp8_cp_comm_only = bench(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu, comm_only=True
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    t_fp8_cp_fused = bench(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    _v(rank, "cleanpush done")

    # ── clean-push + PRESHUFFLED gemm (Option D): comm copies pre-quant fp8 (coalesced) +
    #   writes the E8M0 scale in broadcast layout to pool_scale_ps; gemm reads it preshuffled
    #   (fast MMA). Goal: fast comm (no quant) + fast gemm (preshuffled) overlapped -> ~2ms.
    _v(rank, "bench cleanpush_ps comm-only / fused ...")
    t_fp8_cpps_comm_only = bench(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu,
            gemm_preshuffled=True, comm_only=True,
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    t_fp8_cpps_fused = bench(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu,
            gemm_preshuffled=True,
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    _v(rank, "cleanpush_ps done")

    # ── Option C': clean push RAW scale (coalesced/fast comm) + gemm role locally preshuffles
    #   its tile's A-scale raw->broadcast before the fast preshuffled MMA. Fast comm (hides) +
    #   fast gemm -> targets ~2ms. comm is the same as fp8_cp_comm_only (raw push).
    _v(rank, "bench cleanpush_lp fused ...")
    t_fp8_lp_fused = bench(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu,
            gemm_preshuffled=True, local_preshuffle=True,
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    _v(rank, "cleanpush_lp done")

    # ── Option "role": 3-stage pipeline comm(raw push) -> preshuffle role(transpose once/block_m,
    #   non-redundant) -> gemm(preshuffled). Targets ~2ms (fast comm hides, fast gemm, cheap preshuffle).
    _v(rank, "bench cleanpush_role fused ...")
    t_fp8_role_fused = bench(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu,
            preshuffle_role=True, num_preshuffle_cu=args.num_preshuffle_cu,
        ),
        iters=args.iters, reset=lambda: symm.scoreboard.zero_(), group=group,
    )
    _v(rank, "cleanpush_role done")

    # ── accuracy: fp8 fused vs the decoupled mxfp8 GEMM over the fused-filled pool,
    #    and vs the bf16 fused output (fp8-quant noise). Both on real (non-pad) rows.
    bf16_out = _synced(
        lambda: dispatch_grouped_gemm_bf16(
            x, W1, group, handle=handle, layout="nt", BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        reset_sb=True,
    )[0]
    fp8_out = _synced(
        lambda: dispatch_grouped_gemm_mxfp8(
            x, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        reset_sb=True,
    )
    ref_fp8 = _synced(
        lambda: grouped_gemm_mxfp8_flydsl_kernel(
            symm.pool_fp8, symm.pool_scale, w1q, w1s, group_offs, out_dtype=torch.bfloat16
        )
    )
    cos_vs_ref, rel_vs_ref, _ = gate3(fp8_out[:M_eff], ref_fp8[:M_eff])
    cos_vs_bf16, rel_vs_bf16, _ = gate3(fp8_out[:M_eff], bf16_out[:M_eff])
    # clean-push fused: fills pool_fp8 + pool_scale (raw), then reads them (raw loader).
    cp_out = _synced(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu
        ),
        reset_sb=True,
    )
    ref_cp = _synced(
        lambda: grouped_gemm_mxfp8_flydsl_kernel(
            symm.pool_fp8, symm.pool_scale, w1q, w1s, group_offs, out_dtype=torch.bfloat16
        )
    )
    cos_cp_ref, _, _ = gate3(cp_out[:M_eff], ref_cp[:M_eff])
    cos_cp_bf16, _, _ = gate3(cp_out[:M_eff], bf16_out[:M_eff])
    # clean-push + preshuffled gemm (Option D): comm fills pool_fp8 + pool_scale_ps (broadcast).
    cpps_out = _synced(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu,
            gemm_preshuffled=True,
        ),
        reset_sb=True,
    )
    cos_cpps_bf16, _, _ = gate3(cpps_out[:M_eff], bf16_out[:M_eff])
    # Option C': clean push raw + gemm-role local preshuffle -> preshuffled MMA.
    lp_out = _synced(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu,
            gemm_preshuffled=True, local_preshuffle=True,
        ),
        reset_sb=True,
    )
    cos_lp_bf16, _, _ = gate3(lp_out[:M_eff], bf16_out[:M_eff])
    # Option "role": 3-stage pipeline (comm raw -> preshuffle role -> gemm).
    role_out = _synced(
        lambda: dispatch_grouped_gemm_mxfp8_cleanpush(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN, num_dispatch_cu=ndcu,
            preshuffle_role=True, num_preshuffle_cu=args.num_preshuffle_cu,
        ),
        reset_sb=True,
    )
    cos_role_bf16, _, _ = gate3(role_out[:M_eff], bf16_out[:M_eff])

    symm.destroy()
    return {
        "flops": flops, "M_eff": M_eff, "xgmi_bf16": xgmi_bf16, "xgmi_fp8": xgmi_fp8,
        "xgmi_fp8_ps": xgmi_fp8_ps, "quant_bytes": quant_bytes, "T": T,
        "bf16_gemm": t_bf16_gemm, "bf16_disp": t_bf16_disp, "bf16_fused": t_bf16_fused,
        "fp8_gemm": t_fp8_gemm, "fp8_disp": t_fp8_disp, "fp8_fused": t_fp8_fused,
        "fp8_fused_nofence": t_fp8_fused_nofence, "fp8_fused_nogate": t_fp8_fused_nogate,
        "fp8_quant_local": t_fp8_quant_local, "fp8_comm_only": t_fp8_comm_only,
        "fp8_cp_fused": t_fp8_cp_fused, "fp8_cp_nogate": t_fp8_cp_nogate,
        "fp8_cp_comm_only": t_fp8_cp_comm_only,
        "fp8_cpps_fused": t_fp8_cpps_fused, "fp8_cpps_comm_only": t_fp8_cpps_comm_only,
        "fp8_lp_fused": t_fp8_lp_fused, "fp8_role_fused": t_fp8_role_fused,
        "cos_vs_ref": cos_vs_ref, "rel_vs_ref": rel_vs_ref,
        "cos_vs_bf16": cos_vs_bf16, "rel_vs_bf16": rel_vs_bf16,
        "cos_cp_ref": cos_cp_ref, "cos_cp_bf16": cos_cp_bf16, "cos_cpps_bf16": cos_cpps_bf16,
        "cos_lp_bf16": cos_lp_bf16, "cos_role_bf16": cos_role_bf16,
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
            # bottleneck = slowest rank; accuracy = worst (min cos) rank
            flops = r["flops"]
            bf16_gemm, bf16_disp, bf16_fused = gmax("bf16_gemm"), gmax("bf16_disp"), gmax("bf16_fused")
            fp8_gemm, fp8_disp, fp8_fused = gmax("fp8_gemm"), gmax("fp8_disp"), gmax("fp8_fused")
            fp8_fused_nf = gmax("fp8_fused_nofence")
            fp8_fused_ng = gmax("fp8_fused_nogate")
            fp8_quant_local = gmax("fp8_quant_local")
            fp8_comm_only = gmax("fp8_comm_only")
            fp8_cp_fused, fp8_cp_ng, fp8_cp_co = gmax("fp8_cp_fused"), gmax("fp8_cp_nogate"), gmax("fp8_cp_comm_only")
            fp8_cpps_fused, fp8_cpps_co = gmax("fp8_cpps_fused"), gmax("fp8_cpps_comm_only")
            fp8_lp_fused, fp8_role_fused = gmax("fp8_lp_fused"), gmax("fp8_role_fused")
            cos_ref = _all_min(group, r["cos_vs_ref"])
            cos_bf16 = _all_min(group, r["cos_vs_bf16"])
            cos_cp_ref = _all_min(group, r["cos_cp_ref"])
            cos_cp_bf16 = _all_min(group, r["cos_cp_bf16"])
            cos_cpps_bf16 = _all_min(group, r["cos_cpps_bf16"])
            cos_lp_bf16 = _all_min(group, r["cos_lp_bf16"])
            cos_role_bf16 = _all_min(group, r["cos_role_bf16"])
            # dense single-weight roofline of the same M_eff x 2I x H (rank 0 only, local)
            dense_ms, dgm = dense_gemm_peak_ms(
                r["M_eff"], 2 * args.inter, args.hidden, args.bm, args.bn, args.iters
            ) if rank == 0 else (0.0, 0)
            if rank != 0:
                torch.cuda.synchronize(); group.barrier(); continue

            tf = lambda ms: flops / (ms * 1e-3) / 1e12
            bw = lambda by, ms: by / (ms * 1e-3) / 1e9
            print(f"\n{'='*76}")
            print(f"[dispatch+L1 GEMM  bf16 vs fp8_fused]  {gpu_name} EP{world} "
                  f"T={args.num_tokens} H={args.hidden} I={args.inter} E={args.num_experts} "
                  f"K={args.num_topk} mode={mode} (max over ranks)")
            print(f"{'='*76}")
            print(_line("dense_gemm", dense_ms, tf(dense_ms) if dense_ms else None,
                        f" (single-weight roofline, GROUP_M={dgm})"))
            print("  --- bf16 (push bf16 tokens) ---")
            print(_line("gemm_only", bf16_gemm, tf(bf16_gemm)))
            print(_line("dispatch_only", bf16_disp, extra=f" | {bw(r['xgmi_bf16'], bf16_disp):7.1f} GB/s XGMI"))
            print(_line("fused", bf16_fused, tf(bf16_fused),
                        f" | speedup vs serial = {(bf16_gemm + bf16_disp) / bf16_fused:.2f}x"))
            print("  --- fp8_fused (push fp8 tokens + E8M0) ---")
            print(_line("gemm_only", fp8_gemm, tf(fp8_gemm)))
            print(_line("dispatch_only", fp8_disp, extra=f" | {bw(r['xgmi_fp8'], fp8_disp):7.1f} GB/s XGMI "
                        f"({r['xgmi_bf16'] / r['xgmi_fp8']:.2f}x fewer bytes vs bf16)"))
            print(_line("fused", fp8_fused, tf(fp8_fused),
                        f" | speedup vs serial = {(fp8_gemm + fp8_disp) / fp8_fused:.2f}x"))
            print(_line("fused(no-fence)", fp8_fused_nf, tf(fp8_fused_nf),
                        f" | DIAG: fence cost = {fp8_fused - fp8_fused_nf:.3f} ms (wrong answer)"))
            print(_line("fused(gemm-only)", fp8_fused_ng, tf(fp8_fused_ng),
                        f" | DIAG: raw-scale gemm alone (comm off); comm-overlap gap = {fp8_fused_nf - fp8_fused_ng:.3f} ms"))
            print("  --- comm role breakdown (quant-in-push on the fused critical path) ---")
            print(_line("quant_local", fp8_quant_local,
                        extra=f" | {bw(r['quant_bytes'], fp8_quant_local):7.1f} GB/s hbm "
                              f"(T={r['T']} rows bf16->fp8+bcast scale, no push)"))
            print(_line("push_only(fp8)", fp8_disp,
                        extra=f" | {bw(r['xgmi_fp8'], fp8_disp):7.1f} GB/s XGMI (copy pre-quant fp8 + raw scale)"))
            print(_line("comm_only(q+push)", fp8_comm_only,
                        extra=f" | {bw(r['xgmi_fp8_ps'], fp8_comm_only):7.1f} GB/s XGMI (read bf16 -> quant -> push fp8 + bcast scale)"))
            print(f"      quant-in-push tax vs plain push = {fp8_comm_only - fp8_disp:+.3f} ms "
                  f"({fp8_comm_only / fp8_disp:.2f}x) ; comm_only vs fused = {fp8_comm_only:.3f} / {fp8_fused:.3f} ms "
                  f"({100.0 * fp8_comm_only / fp8_fused:.0f}% of fused, exposed on critical path)")
            print("  --- fp8_cleanpush (push PRE-QUANT fp8 + RAW scale; gemm reads raw scale) ---")
            print(_line("gemm_only", fp8_cp_ng, tf(fp8_cp_ng),
                        " | DIAG: raw-scale gemm alone (comm off)"))
            print(_line("comm_only(push)", fp8_cp_co,
                        extra=f" | {bw(r['xgmi_fp8'], fp8_cp_co):7.1f} GB/s XGMI (copy pre-quant fp8 + raw scale + signal)"))
            print(_line("fused", fp8_cp_fused, tf(fp8_cp_fused),
                        f" | comm-overlap gap = {fp8_cp_fused - fp8_cp_ng:.3f} ms"))
            print(f"      clean push comm_only = {fp8_cp_co:.3f} ms ({bw(r['xgmi_fp8'], fp8_cp_co) / bw(r['xgmi_fp8'], fp8_disp):.2f}x push_only BW) "
                  f"vs quant-in-push comm_only {fp8_comm_only:.3f} ms ({fp8_comm_only / fp8_cp_co:.2f}x)")
            print("  --- fp8_cleanpush + PRESHUFFLED gemm (Option D: comm writes broadcast scale) ---")
            print(_line("comm_only(push+bc)", fp8_cpps_co,
                        extra=f" | {bw(r['xgmi_fp8_ps'], fp8_cpps_co):7.1f} GB/s XGMI (copy fp8 + broadcast scale, no quant)"))
            print(_line("fused", fp8_cpps_fused, tf(fp8_cpps_fused),
                        f" | preshuffled gemm ({fp8_gemm:.3f}) + broadcast comm ({fp8_cpps_co:.3f}) overlapped"))
            print("  --- fp8_cleanpush + LOCAL-PRESHUFFLE (Option C': raw push + gemm-role preshuffle) ---")
            print(_line("fused", fp8_lp_fused, tf(fp8_lp_fused),
                        f" | raw comm ({fp8_cp_co:.3f}, hides) + preshuffled gemm + per-tile preshuffle"))
            print("  --- fp8_cleanpush + PRESHUFFLE ROLE (3-stage: comm raw -> preshuffle role -> gemm) ---")
            print(_line("fused", fp8_role_fused, tf(fp8_role_fused),
                        f" | raw comm ({fp8_cp_co:.3f}) + non-redundant preshuffle role + preshuffled gemm, overlapped"))
            _best = min(fp8_fused, fp8_cp_fused, fp8_cpps_fused, fp8_lp_fused, fp8_role_fused)
            print(f"  --- BEST fused fp8 = {_best:.3f} ms (quant-in-push {fp8_fused:.3f} / cp-raw {fp8_cp_fused:.3f} "
                  f"/ cp-broadcast {fp8_cpps_fused:.3f} / cp-localpreshuf {fp8_lp_fused:.3f} / role {fp8_role_fused:.3f}) "
                  f"vs bf16 fused {bf16_fused:.3f} ({bf16_fused / _best:.2f}x) "
                  f"vs decoupled {fp8_gemm + fp8_disp:.3f} ms ({(fp8_gemm + fp8_disp) / _best:.2f}x) ---")
            print(f"  [acc vs bf16] fused={cos_bf16:.5f} cp-raw={cos_cp_bf16:.5f} "
                  f"cp-bc={cos_cpps_bf16:.5f} cp-lp={cos_lp_bf16:.5f} role={cos_role_bf16:.5f}  "
                  f"{'PASS' if min(cos_ref, cos_cp_ref, cos_cpps_bf16, cos_lp_bf16, cos_role_bf16) >= 0.99 else 'FAIL'}")
            torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EP fused fp8 vs bf16 dispatch+GEMM kernel benchmark")
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
