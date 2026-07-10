###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for the mega-MoE EP benchmarks (dispatch+GEMM / GEMM+combine).

Single source of truth for: routing/weight generation, the dense-GEMM roofline,
the CUDA-event bench helper, the prologue-driven input builders for both kernels,
and the per-stage metric/print template so both benchmarks print identically.
"""

import functools
import math
import os
import statistics
from types import SimpleNamespace
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
import torch.distributed as dist
from flydsl.expr import arith
from flydsl.expr.buffer_ops import (
    buffer_load,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)
from flydsl.expr.typing import AddressSpace, PointerType

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
    _compile_dense_nt,
    _compile_grouped_variable_k_bf16,
    _get_compiled_dense,
    _i64,
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.mega import (
    dispatch_prologue_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.flydsl.mega.ep_intranode import (
    _BLOCK_THREADS,
    _NUM_WARPS,
    _PVEC,
    combine_bf16_tile,
    dispatch_bf16_tile,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.symm_buffer import SymLayout, get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.gemm_helper import (
    ceildiv,
    make_value_attrs,
    xcd_remap_pid,
)

# re-export so benchmarks can import everything mega from one place
__all__ = [
    "generate_routing",
    "dense_gemm_peak_ms",
    "bench",
    "gate3",
    "check_accuracy",
    "AccuracyCheck",
    "generate_input",
    "compute_stage_metrics",
    "BF16_BYTES",
    "sync_ranks",
    "reduce_across_ranks",
    "aggregate_stage_metrics",
    "stage_columns",
    "turbo_csv_row",
    "checks_verdict",
    "apply_case",
    "all_ranks_ok",
    "print_header",
    "print_stage",
    "dispatch_only",
    "dispatch_prologue_flydsl_kernel",
    "grouped_gemm_bf16_only",
    "grouped_gemm_variable_k_only",
    "compile_grouped_gemm_bf16",
    "get_symm_buffer_for_mega_moe",
    "swiglu_flydsl_kernel",
    "combine_only",
    "topk_reduce_only",
]


# --------------------------------------------------------------------------- #
# Per-model sweep helpers: both mega benchmarks iterate config.gen_moe_test_cases,
# sizing each run from one MoE model and skipping cases the kernel can't run.
# --------------------------------------------------------------------------- #
def apply_case(args, case):
    """Set per-model geometry (H/I/E/K) on args from a gen_moe_test_cases entry."""
    args.hidden = case["hidden"]
    args.inter = case["inter"]
    args.num_experts = case["num_experts"]
    args.num_topk = case["num_topk"]


def all_ranks_ok(group, ok):
    """Collective AND of a per-rank bool; lets every rank skip an unsupported case together."""
    flags = [None] * group.size()
    dist.all_gather_object(flags, ok, group=group)
    return all(flags)


# --------------------------------------------------------------------------- #
# Bench-only baselines (moved here from the kernel module so the kernel file keeps
# only the fused dispatch_grouped_gemm_bf16_flydsl_kernel): the grouped-GEMM / dispatch-only /
# variable-K TN wgrad compute peaks, plus their thin host wrappers.
# --------------------------------------------------------------------------- #
# Grouped GEMM-only launcher (the compute-peak baseline). Dense XCD-swizzle
# scheduler + GROUP_M, per-expert B slab via tile_to_expert.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=256)
def compile_grouped_gemm_bf16(
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    nt_vmcnt=4,  # swept: vmcnt=4 > 3 (~1% on L1 NT, gfx950)
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    layout="nt",
):
    """Compile (cached) the grouped BF16 GEMM launcher for one (K, tile, layout)
    combo. Grid over-launched to the padded pool; each block early-exits past the
    real tile range. Returns the flyc launch callable."""
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    # per-tile GEMM closure by layout (NT forward, NN dgrad, TN wgrad); grouped via b_group_base
    gemm_tile = functools.partial(gemm_bf16_tile, layout)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def grouped_gemm_k(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        group_res = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        # The pool is over-allocated (capacity >> real tiles), so the launch grid is
        # mostly padding. Map/XCD-swizzle over the REAL tile range only (front-loaded),
        # so real work stays DENSE across CUs with L2 reuse and padding tiles early-exit
        # at the tail. (Swizzling over the full pool grid scatters real tiles into the
        # padding -> ~half-idle waves -> ~2x slower; that was the gemm_only perf bug.)
        real_grid = real_tiles * n_blocks

        def _emit():
            pid = xcd_remap_pid(fx.block_idx.x, real_grid, num_xcd)
            num_pid_m = real_tiles
            num_pid_in_group = GROUP_M * n_blocks
            group_id = pid // num_pid_in_group
            pid_in_group = pid % num_pid_in_group
            first_pid_m = group_id * GROUP_M
            remaining_m = num_pid_m - first_pid_m
            group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
            block_m = first_pid_m + (pid_in_group % group_size_m)
            block_n = pid_in_group // group_size_m
            g_idx = buffer_load(group_res, block_m, vec_width=1, dtype=fx.T.i32())
            gbase = g_idx * fx.Int32(K) * c_n
            # Worst-case pool (cap*K > 2^31): rebase A/C per tile in int64 so each
            # tile's buffer resource spans only BLOCK_M rows (int32 in-resource
            # offset, base advanced in int64). Mirrors the fused nt/nn path. A is
            # [M,K] row-major for nt/nn -> advance by block_m*BLOCK_M rows.
            if layout in ("nt", "nn"):
                pool_ptr_ty = PointerType.get(
                    elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
                )
                a_byte_off = _i64(block_m) * fx.Int64(BLOCK_M * K * 2)
                c_byte_off = _i64(block_m * fx.Int32(BLOCK_M)) * _i64(c_n) * fx.Int64(2)
                a_base = fx.arith.ArithValue(arith.index_cast(fx.T.i64(), extract_base_index(A)), signed=True)
                c_base = fx.arith.ArithValue(arith.index_cast(fx.T.i64(), extract_base_index(C)), signed=True)
                A_tile = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, a_base + a_byte_off), fx.make_layout(BLOCK_M * K, 1)
                )
                C_tile = fx.make_view(
                    fx.inttoptr(pool_ptr_ty, c_base + c_byte_off),
                    fx.make_layout(fx.Int32(BLOCK_M) * c_n, 1),
                )
                gemm_tile(
                    A_tile,
                    B,
                    C_tile,
                    fx.Int32(BLOCK_M),
                    c_n,
                    lds,
                    fx.Int32(0),
                    block_n,
                    K=K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16,
                    nt_vmcnt=nt_vmcnt,
                    b_group_base=gbase,
                )
            else:
                gemm_tile(
                    A,
                    B,
                    C,
                    c_m,
                    c_n,
                    lds,
                    block_m,
                    block_n,
                    K=K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    out_fp16=out_fp16,
                    nt_vmcnt=nt_vmcnt,
                    b_group_base=gbase,
                )

        if fx.block_idx.x < real_grid:
            _emit()

    @flyc.jit
    def launch(A, B, C, TILE_TO_GROUP, NUM_TILE_BLOCKS, c_m: int, c_n: int, stream: fx.Stream):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        grouped_gemm_k(
            A,
            B,
            C,
            TILE_TO_GROUP,
            NUM_TILE_BLOCKS,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


# --------------------------------------------------------------------------- #
# Dispatch-PUSH-only launcher (no GEMM, no scoreboard) — raw dispatch bandwidth.
# Every block pushes a round-robin share of the comm tasks to peer pools over XGMI.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=256)
def _compile_dispatch_only(hidden_size, num_max_pool_tokens, num_dispatch_cu, num_comm, waves_per_eu=2):
    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_only_k(
        INPUT_TOKENS: fx.Tensor,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_layout: SymLayout,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        input_res = create_buffer_resource(INPUT_TOKENS, max_size=True)
        expert_send_dst_rank_res = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_res = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_res = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_res = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dispatched_token_idx_res = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)

        if block_index < comm_block_count:
            local_count = (
                fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
            ) // comm_block_count
            for local_iter in range(local_count):
                dispatch_bf16_tile(
                    sym_layout,
                    thread_index=thread_index,
                    hidden_size=hidden_size,
                    input_res=input_res,
                    expert_send_dst_rank_res=expert_send_dst_rank_res,
                    expert_send_dst_row_res=expert_send_dst_row_res,
                    expert_send_count_res=expert_send_count_res,
                    expert_send_offset_res=expert_send_offset_res,
                    dispatched_token_idx_res=dispatched_token_idx_res,
                    task_index=block_index + local_iter * comm_block_count,
                    signal=False,
                )

    @flyc.jit
    def launch(
        INPUT_TOKENS,
        EXPERT_SEND_DST_RANK,
        EXPERT_SEND_DST_ROW,
        EXPERT_SEND_COUNT,
        EXPERT_SEND_OFFSET,
        DISPATCHED_TOKEN_IDX,
        sym_layout,
        stream: fx.Stream,
    ):
        dispatch_only_k(
            INPUT_TOKENS,
            EXPERT_SEND_DST_RANK,
            EXPERT_SEND_DST_ROW,
            EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET,
            DISPATCHED_TOKEN_IDX,
            sym_layout,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_dispatch_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# --------------------------------------------------------------------------- #
# Grouped TN wgrad (variable-K) GEMM-only baseline over dispatched pools:
# dW[g] = lhs_pool[g]^T @ rhs_pool[g], out [G, OUT_M, OUT_N]. The compute core is
# the canonical _compile_grouped_variable_k_bf16 (gemm_bf16_kernel); the fused
# dispatch+wgrad path lives in dispatch_grouped_gemm_bf16_flydsl_kernel(layout="tn").
# --------------------------------------------------------------------------- #
def grouped_gemm_variable_k_only(
    lhs_pool,  # [M_pool, OUT_M] bf16   dispatched lhs (e.g. recomputed activation)
    rhs_pool,  # [M_pool, OUT_N] bf16   dispatched rhs (e.g. dY)
    num_tokens_per_expert_prefix,  # [G+1] int64   per-group token boundaries in the pool
    out_dw,  # [G, OUT_M, OUT_N] bf16 ([G, OUT_N, OUT_M] if trans_c)  C = dW
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=1,  # xcd=1 maximizes L2 reuse on the variable-K M-reduction (+14% vs 8, +6% vs 4; swept)
    trans_c=False,
    waves_per_eu=2,
):
    """GEMM-only grouped TN wgrad over dispatched pools (no comm). dW[g] =
    lhs_pool[offs[g]:offs[g+1]]^T @ rhs_pool[offs[g]:offs[g+1]] (transposed if trans_c)."""
    G = num_tokens_per_expert_prefix.numel() - 1
    OUT_M = lhs_pool.shape[1]
    OUT_N = rhs_pool.shape[1]
    out_fp16 = out_dw.dtype == torch.float16
    # prefix offsets must be int64
    prefix_i64 = (
        num_tokens_per_expert_prefix
        if num_tokens_per_expert_prefix.dtype == torch.int64
        else num_tokens_per_expert_prefix.to(torch.int64)
    )
    # trans_c: produce C^T = rhs^T @ lhs by swapping operands (both K-major, symmetric)
    # so the result lands [G, OUT_N, OUT_M] via the FAST coalesced normal store.
    if trans_c:
        lhs_e, rhs_e, OUT_M_e, OUT_N_e = rhs_pool, lhs_pool, OUT_N, OUT_M
    else:
        lhs_e, rhs_e, OUT_M_e, OUT_N_e = lhs_pool, rhs_pool, OUT_M, OUT_N
    launch = _compile_grouped_variable_k_bf16(
        OUT_M_e,
        OUT_N_e,
        G,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        waves_per_eu=waves_per_eu,
    )
    # lhs/rhs pools as 2-D (flat view(-1) overflows the int32 shape ABI at
    # worst-case pool); the wgrad kernel rebases per group via extract_base_index.
    args = (
        lhs_e.contiguous(),
        rhs_e.contiguous(),
        out_dw.view(-1),  # output: view (not contiguous) so writes hit the real buffer
        prefix_i64,
        OUT_M_e,
        OUT_N_e,
        torch.cuda.current_stream(),
    )
    # shared shape/dtype-keyed compile cache (same helper as dense_gemm_peak_ms)
    _get_compiled_dense(launch, args)(*args)
    return out_dw


def grouped_gemm_bf16_only(
    pool,  # [M, K] bf16   A operand (pre-filled activation)
    weight,  # [G, N, K] bf16   per-expert B
    output,  # [M, N] bf16   C
    tile_to_expert,  # [n_mblk] i32   expert id per BLOCK_M pool block
    num_tile_blocks,  # [1] i32        real tile-block count (device)
    *,
    layout="nt",
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=4,
    # xcd=1 maximizes L2 reuse of shared weight slabs (swept: +3% NT, +11% NN, +15% TN vs xcd=8)
    num_xcd=1,
    nt_vmcnt=4,  # swept: vmcnt=4 > 3 (~1% on L1 NT, gfx950)
    waves_per_eu=2,
    agpr_alloc=0,
):
    """Pure grouped BF16 GEMM (no dispatch) — the compute-peak baseline.

    ``pool`` is A=[M,K] bf16, ``output`` is [M,N] bf16, ``tile_to_expert`` maps each
    BLOCK_M pool block -> expert. Weight layout: NT (forward) ``weight`` [G,N,K];
    NN (dgrad) / TN (wgrad) ``weight`` [G,K,N]. ``num_tile_blocks`` is the real
    tile-block count (runtime over-launch self-bound)."""
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    assert pool.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    if layout == "tn":  # A is [K, M] (K-major)
        hidden_size, c_m = pool.shape
    else:  # A is [M, K]
        c_m, hidden_size = pool.shape
    if layout == "nt":  # weight [G, N, K]
        G, N, K = weight.shape
        weight_flat = weight.reshape(G * N, K).contiguous().view(-1)
    else:  # NN / TN: weight [G, K, N]
        G, K, N = weight.shape
        weight_flat = weight.reshape(G * K, N).contiguous().view(-1)
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    launch = compile_grouped_gemm_bf16(
        K=hidden_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        nt_vmcnt=int(nt_vmcnt),
        waves_per_eu=int(waves_per_eu),
        agpr_alloc=int(agpr_alloc),
        layout=layout,
    )
    # Pass A/C as 2-D (each dim < 2^31): flat view(-1) overflows the int32 shape
    # ABI at worst-case pool. The kernel rebases per tile via extract_base_index.
    launch(
        pool.contiguous(),
        weight_flat,
        output,
        tile_to_expert,
        num_tile_blocks,
        c_m,
        out_features,
        stream=torch.cuda.current_stream(),
    )
    return output


def dispatch_only(
    x,  # [num_src_tokens, K] bf16
    handle,  # dispatch handle (DeepEP-style; handle[:5] = the send ABI)
    symm,  # SymmBuffer owning the peer pool + delta tables
    *,
    num_dispatch_cu=32,
):
    """Cross-rank dispatch PUSH only (no GEMM) — pushes ``x`` token rows to peer
    pools over XGMI via the two-heap delta addressing (matches the fused kernel).
    Bytes pushed per rank = (sum expert_send_count) * hidden * 2."""
    expert_send_dst_rank, expert_send_dst_row, expert_send_count, expert_send_offset, dispatched_token_idx = (
        handle[:5]
    )
    num_comm = expert_send_dst_rank.numel()
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(1)
    pool_capacity = symm.num_max_pool_tokens
    x_i32 = x.contiguous().view(torch.int32).view(-1)
    launch = _compile_dispatch_only(hidden_size, pool_capacity, int(num_dispatch_cu), int(num_comm))
    launch(
        x_i32,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        symm.get_sym_layout(),
        stream=torch.cuda.current_stream(),
    )


# --------------------------------------------------------------------------- #
# Routing + weights
# --------------------------------------------------------------------------- #
def generate_routing(num_tokens, num_topk, num_experts, *, device="cuda", seed=0):
    """(topk_idx[T,K] int64, topk_weight[T,K] f32), load-balanced routing."""
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, generator=g, device=device).abs() + 1
    topk_weight, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)
    return topk_idx.to(torch.int64), topk_weight.to(torch.float32)


# --------------------------------------------------------------------------- #
# Bench helper: warmup, one L2 flush before timing, then CUDA-event timing.
# Cross-rank variants (group set) barrier around an optional scoreboard reset.
# --------------------------------------------------------------------------- #
_L2_FLUSH_BYTES = 256 * 1024 * 1024  # > gfx950 L2 -> zeroing it evicts the cache
_L2_FLUSH_BUF = None


def _l2_flush():
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH_BUF.zero_()


def bench(fn, *, warmup=20, iters=30, reset=None, group=None):
    """Mean ms/call (CUDA events).

    Cross-rank discipline (group set): barrier BEFORE reset so every rank has
    finished the previous fn (no in-flight peer signals), zero, then a second
    barrier so all state is clean before any rank's fn pushes/signals a peer.
    Skipping the pre-reset barrier races the scoreboard and can deadlock."""
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def _iter(start_event=None, end_event=None):
        if group is not None:
            torch.cuda.synchronize()
            group.barrier()  # all ranks done with previous fn
        if reset is not None:
            reset()
        if group is not None:
            torch.cuda.synchronize()
            group.barrier()  # all state clean before any fn
        if start_event is None:
            fn()
            return
        start_event.record()
        fn()
        end_event.record()

    for _ in range(warmup):
        _iter()
    _l2_flush()
    for i in range(iters):
        _iter(start_events[i], end_events[i])
    torch.cuda.synchronize()

    # drop the first timed iter: residual cold-start after the single L2 flush above
    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]
    return np.average(times)


# --------------------------------------------------------------------------- #
# Accuracy check: single source of truth for the gate-3 (cos / rel_rmse / ok)
# correctness check shared by every mega-MoE benchmark stage. Mirrors the e2e
# test's _gate3 so a bench PASS means the same thing as the test PASS.
# --------------------------------------------------------------------------- #
def gate3(out, ref, *, cos_thresh=0.99, rel_thresh=0.05):
    """(cos, rel_rmse, ok) of a kernel output vs its reference (flattened, fp32)."""
    g, r = out.float().flatten(), ref.float().flatten()
    cos = float(torch.dot(g, r) / (g.norm() * r.norm() + 1e-12))
    rel = float((g - r).norm() / (r.norm() + 1e-12))
    return cos, rel, (cos >= cos_thresh and rel <= rel_thresh)


class AccuracyCheck(NamedTuple):
    """Global gate-3 accuracy check for one stage (index access kept for back-compat)."""

    cos: float
    rel: float
    ok: bool


def check_accuracy(group, name, out, ref, *, cos_thresh=0.99, rel_thresh=0.05):
    """Gate-3 one stage across all ranks; rank 0 prints the worst (min-cos) rank,
    every rank returns the global AccuracyCheck(cos, rel, ok) so the caller can
    record it. None out/ref -> skipped (returns None)."""
    if out is None or ref is None:
        if group.rank() == 0:
            print(f"  [check] {name:<28}: (skipped — no reference)")
        return None
    cos, rel, ok = gate3(out, ref, cos_thresh=cos_thresh, rel_thresh=rel_thresh)
    world = group.size()
    gathered = [None] * world
    dist.all_gather_object(gathered, (group.rank(), cos, rel, ok), group=group)
    worst_rank, worst_cos, worst_rel, _ = min(gathered, key=lambda t: t[1])  # lowest cos = worst rank
    all_ok = all(rank_ok for _, _, _, rank_ok in gathered)
    if group.rank() == 0:
        print(
            f"  [check] {name:<28}: cos={worst_cos:.5f} rel={worst_rel:.4f} "
            f"(worst rank={worst_rank}) {'PASS' if all_ok else 'FAIL'}"
        )
    return AccuracyCheck(worst_cos, worst_rel, all_ok)


def dense_gemm_peak_ms(M, N, K, BLOCK_M, BLOCK_N, iters, *, group_m_cands=(4,)):
    """Dense NT bf16 GEMM (gemm_bf16_kernel) of the SAME M x N x K as the grouped
    GEMM -> the single-weight compute roofline. autotune sweeps GROUP_M {1,4,8};
    default off uses the single best (GROUP_M=4). Returns (best_ms, best_group_m)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 8
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 8  # NT: B [N,K]
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    dense_args = (a.view(-1), b.view(-1), c.view(-1), M, N, torch.cuda.current_stream())
    best_ms, best_group_m = float("inf"), None
    for group_m in group_m_cands:
        launch = _compile_dense_nt(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_M=group_m, num_xcd=8)
        compiled = _get_compiled_dense(launch, dense_args)
        ms = bench(lambda c=compiled: c(*dense_args), iters=iters)
        if ms < best_ms:
            best_ms, best_group_m = ms, group_m
    del a, b, c
    return best_ms, best_group_m


# --------------------------------------------------------------------------- #
# Input builders (consistent with the EP test): build the SymmBuffer, run the
# prologue to produce the dispatch handle, then fill the real pool/activation.
# `kind` selects which kernel's inputs to materialize.
# --------------------------------------------------------------------------- #
def _build_symm_and_plan(group, *, T, H, I, E, K, BLOCK_M, BLOCK_N, base_seed):
    """Shared prologue: SymmBuffer + routing + dispatch handle (no pool fill yet)."""
    rank = group.rank()
    torch.manual_seed(base_seed + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_weight = generate_routing(T, K, E, device="cuda", seed=100 + rank)

    # one symmetric allocation for every cross-rank + scratch buffer (production arena)
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
    )
    # prologue -> the flat dispatch handle (same path as the test); resets scoreboard +
    # barrier_local in-kernel and ends with a cross-rank barrier. The handle IS the full
    # prologue tuple (tile_to_expert / tile_expected / group_offs ride at fixed indices).
    handle = dispatch_prologue_flydsl_kernel(
        topk_idx,
        topk_weight,
        sym_layout=symm.get_sym_layout(),
        num_tokens=T,
        num_topk=K,
        num_experts=E,
        num_ranks=symm.world,
        rank=symm.rank,
        experts_per_rank=E // symm.world,
        block_m=symm.block_m,
        num_max_pool_tokens=symm.num_max_pool_tokens,
    )
    tile_to_expert, expected = handle[5], handle[6]
    num_tile_blocks = symm.meta_scalars[1:2]  # device real-tile count
    symm.assert_capacity()  # fail loudly rather than silently drop rows
    return symm, x, topk_idx, topk_weight, handle, tile_to_expert, expected, num_tile_blocks


def _dispatch_and_settle(group, x, handle, symm, *, num_dispatch_cu):
    """Cross-rank dispatch PUSH bracketed by barriers: all ranks quiet before the
    push, all peer pools filled before any caller reads them."""
    torch.cuda.synchronize()
    group.barrier()
    dispatch_only(x, handle, symm, num_dispatch_cu=num_dispatch_cu)
    torch.cuda.synchronize()
    group.barrier()


def generate_input(group, *, kind, T, H, I, E, K, BLOCK_M, BLOCK_N, num_dispatch_cu=16, base_seed=7):
    """Build the real inputs for one of the mega kernels over the production SymmBuffer.

    This rank's expert weights (W1/W2) are generated here (deterministic global set,
    sliced to this rank), so callers never build or slice the weights themselves.

    kind="dispatch": pool is filled by a single dispatch PUSH (real A for the L1 GEMM).
    kind="combine":  prologue + dispatch + L1 GEMM + SwiGLU -> act (real L2 input), and
                     l2_token_buffer is filled once so combine_only has real rows."""
    symm, x, topk_idx, topk_weight, handle, tile_to_expert, expected, num_tile_blocks = _build_symm_and_plan(
        group, T=T, H=H, I=I, E=E, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, base_seed=base_seed
    )
    # this rank's expert weights: deterministic global set (identical on every rank via
    # the fixed seed), sliced to this rank so the slice is consistent across ranks
    g = torch.Generator(device="cuda").manual_seed(1234)
    W1_global = torch.randn((E, 2 * I, H), generator=g, device="cuda", dtype=torch.bfloat16) * (
        2.0 / math.sqrt(H)
    )
    W2_global = torch.randn((E, H, I), generator=g, device="cuda", dtype=torch.bfloat16) * (
        2.0 / math.sqrt(I)
    )
    experts_per_rank = E // group.size()
    local = slice(group.rank() * experts_per_rank, (group.rank() + 1) * experts_per_rank)
    W1, W2 = W1_global[local].contiguous(), W2_global[local].contiguous()

    if kind == "dispatch":
        # L1 GEMM output (2*inter wide) has no slot in the arena -> local scratch
        l1_out = torch.empty((symm.num_max_pool_tokens, 2 * I), dtype=torch.bfloat16, device="cuda")
        destination, count = handle[0], handle[2]  # dst_rank, expert_send_count
        # fill the pool (real A) via dispatch_only (peers synced by the prologue)
        _dispatch_and_settle(group, x, handle, symm, num_dispatch_cu=num_dispatch_cu)
        return SimpleNamespace(
            symm=symm,
            x=x,
            handle=handle,
            l1_out=l1_out,
            tile_to_expert=tile_to_expert,
            expected=expected,
            num_tile_blocks=num_tile_blocks,
            destination=destination,
            count=count,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            W1=W1,
            W2=W2,
        )

    if kind == "combine":
        # combine scoreboard is now the monotonic parity combine_flag (never reset).
        # cross-rank dispatch PUSH (fills the pool), then grouped L1 GEMM (NT):
        # pool[M,H] @ W1[g,2I,H] -> l1_out[M,2I] (local scratch, no arena slot)
        _dispatch_and_settle(group, x, handle, symm, num_dispatch_cu=num_dispatch_cu)
        l1_out = torch.empty((symm.num_max_pool_tokens, 2 * I), dtype=torch.bfloat16, device="cuda")
        grouped_gemm_bf16_only(
            symm.dispatch_token_pool,
            W1,
            l1_out,
            tile_to_expert,
            num_tile_blocks,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        # fused SwiGLU activation -> act (L2 GEMM input)
        act = swiglu_flydsl_kernel(l1_out)
        # fill l2_token_buffer once with the real rows (for combine_only)
        grouped_gemm_bf16_only(
            act, W2, symm.l2_token_buffer, tile_to_expert, num_tile_blocks, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        torch.cuda.synchronize()
        group.barrier()
        return SimpleNamespace(
            symm=symm,
            x=x,  # raw input tokens (for the turbo full-forward reference)
            act=act,
            handle=handle,
            num_tile_blocks=num_tile_blocks,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            tile_to_expert=tile_to_expert,
            W1=W1,
            W2=W2,
        )

    raise ValueError(f"unknown kind: {kind}")


# --------------------------------------------------------------------------- #
# Metric + print template: identical layout for dispatch and combine, fwd/bwd.
# --------------------------------------------------------------------------- #
def compute_stage_metrics(*, gemm_ms, dense_ms, dense_gm, comm_ms, fused_ms, flops, xgmi):
    """Derive the standard per-stage metrics shared by both benchmarks.

    grouped/dense = grouped GEMM vs the dense single-weight roofline.
    hidden  = serial(gemm+comm) - fused      (comm time hidden under GEMM)
    speedup = serial / fused                 (fused vs serial)
    roofline= max(gemm,comm) / fused         (overlap floor: the slower leg)"""
    serial_ms = gemm_ms + comm_ms
    return SimpleNamespace(
        flops=flops,
        gemm_ms=gemm_ms,
        gemm_tf=flops / (gemm_ms * 1e-3) / 1e12,
        dense_ms=dense_ms,
        dense_tf=flops / (dense_ms * 1e-3) / 1e12,
        dense_gm=dense_gm,
        comm_ms=comm_ms,
        comm_bw=xgmi / (comm_ms * 1e-3) / 1e9,
        fused_ms=fused_ms,
        fused_tf=flops / (fused_ms * 1e-3) / 1e12,
        grouped_eff_pct=(flops / (gemm_ms * 1e-3)) / (flops / (dense_ms * 1e-3)) * 100.0,
        serial_ms=serial_ms,
        hidden_ms=serial_ms - fused_ms,
        speedup=serial_ms / fused_ms,
        roofline_pct=max(gemm_ms, comm_ms) / fused_ms * 100.0,
    )


# --------------------------------------------------------------------------- #
# Cross-rank aggregation + CSV column helpers shared by both benchmarks. Each
# per-rank result is {"stages": {stage: StageMetrics}, ...}; StageMetrics carries
# gemm_ms / dense_ms / dense_gm / comm_ms / fused_ms / flops.
# --------------------------------------------------------------------------- #
BF16_BYTES = 2  # bytes per bf16 element (XGMI push volume, dense-roofline sizing)


def sync_ranks(group):
    """Quiesce every rank: flush this rank's GPU work, then barrier so all ranks line up."""
    torch.cuda.synchronize()
    group.barrier()


def reduce_across_ranks(per_rank, stage, field, *, reduce_fn=max):
    """Reduce one StageMetrics field across ranks (bottleneck = max, work = mean)."""
    return reduce_fn([getattr(res["stages"][stage], field) for res in per_rank])


def aggregate_stage_metrics(per_rank, stage, xgmi):
    """Cross-rank metric bundle for one stage; latencies=slowest rank, flops=mean, dense_gm=rank0."""
    return compute_stage_metrics(
        gemm_ms=reduce_across_ranks(per_rank, stage, "gemm_ms"),
        dense_ms=reduce_across_ranks(per_rank, stage, "dense_ms"),
        dense_gm=per_rank[0]["stages"][stage].dense_gm,
        comm_ms=reduce_across_ranks(per_rank, stage, "comm_ms"),
        fused_ms=reduce_across_ranks(per_rank, stage, "fused_ms"),
        flops=reduce_across_ranks(per_rank, stage, "flops", reduce_fn=statistics.mean),
        xgmi=xgmi,
    )


def stage_columns(prefix, m, check, *, comm_label, comm_short, dense_ms=False, xgmi=False, hidden=False):
    """One stage's CSV columns in canonical order; comm_label/comm_short name the comm leg, flags gate optional cols."""
    cols = {}
    if dense_ms:
        cols[f"{prefix}dense_gemm (ms)"] = f"{m.dense_ms:.3f}"
    cols[f"{prefix}dense_gemm (TFLOPS)"] = f"{m.dense_tf:.1f}"
    cols[f"{prefix}gemm_only (ms)"] = f"{m.gemm_ms:.3f}"
    cols[f"{prefix}gemm_only (TFLOPS)"] = f"{m.gemm_tf:.1f}"
    cols[f"{prefix}grouped/dense"] = f"{m.grouped_eff_pct:.1f}%"
    cols[f"{prefix}{comm_label} (ms)"] = f"{m.comm_ms:.3f}"
    if xgmi:
        cols[f"{prefix}{comm_label} (XGMI GB/s)"] = f"{m.comm_bw:.1f}"
    cols[f"{prefix}fused (ms)"] = f"{m.fused_ms:.3f}"
    cols[f"{prefix}fused (TFLOPS)"] = f"{m.fused_tf:.1f}"
    # accuracy check -> 'cos/PASS|FAIL' (or 'n/a' when the stage had no reference)
    cols[f"{prefix}fused accuracy (cos/ok)"] = (
        "n/a" if check is None else f"{check.cos:.5f}/{'PASS' if check.ok else 'FAIL'}"
    )
    if hidden:
        cols[f"{prefix}comm_hidden (ms)"] = f"{m.hidden_ms:.3f}"
    cols[f"{prefix}speedup (vs serial)"] = f"{m.speedup:.2f}x"
    cols[f"{prefix}roofline (max(gemm,{comm_short})/fused)"] = f"{m.roofline_pct:.1f}%"
    return cols


def checks_verdict(checks):
    """Reduce a stage->AccuracyCheck dict to one PASS/FAIL (None checks ignored)."""
    graded = [c for c in checks.values() if c is not None]
    if not graded:
        return "n/a"
    return "PASS" if all(c.ok for c in graded) else "FAIL"


def turbo_csv_row(platform, gpu_name, world, args, *, case, check, fwd, bwd_stages):
    """Compact gemm_turbo-style CSV row: config + split Forward/Backward Time+TFLOPS.

    case       : model case name (shared mega model, e.g. DeepSeek-V3).
    fwd        : forward stage metrics namespace.
    bwd_stages : backward stage namespaces summed into one Backward figure
                 (dispatch: dgrad + wgrad; combine: dgrad only)."""
    bwd_ms = sum(m.fused_ms for m in bwd_stages)
    bwd_flops = sum(m.flops for m in bwd_stages)
    bwd_tf = bwd_flops / (bwd_ms * 1e-3) / 1e12 if bwd_ms > 0 else 0.0
    return {
        "Platform": platform,
        "GPU": gpu_name,
        "Case": case,
        "EP": world,
        "T": args.num_tokens,
        "H": args.hidden,
        "I": args.inter,
        "E": args.num_experts,
        "K": args.num_topk,
        "Check": check,
        "Forward Time (ms)": f"{fwd.fused_ms:.3f}",
        "Forward TFLOPS": f"{fwd.fused_tf:.1f}",
        "Backward Time (ms)": f"{bwd_ms:.3f}",
        "Backward TFLOPS": f"{bwd_tf:.1f}",
    }


def print_header(tag, gpu_name, world, args, *, case=None):
    """Common '===' header line shared by both benchmarks."""
    case_tag = f" {case}" if case else ""
    print(
        f"\n{'='*72}\n[{tag}]{case_tag} {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
        f"I={args.inter} E={args.num_experts} K={args.num_topk} "
        f"(max over ranks)\n{'=' * 72}"
    )


def print_stage(m, *, comm_label, comm_unit, comm_tag, comm_extra="", fused_extra="", sub_header=None):
    """Print one stage (forward or backward) in the shared 4-line layout.

    comm_label : 'dispatch_only' | 'combine_only'   (left column)
    comm_unit  : e.g. 'GB/s (XGMI, nodeup)' | 'GB/s (XGMI)'
    comm_tag   : 'disp' | 'comb'                     (roofline formula text)
    comm_extra / fused_extra : kernel-specific suffixes (e.g. CU sweep strings)."""
    if sub_header is not None:
        print(f"  {'-' * 68}  {sub_header}")
    print(
        f"  dense_gemm   : {m.dense_ms:8.3f} ms | {m.dense_tf:7.1f} TFLOPS "
        f"(single-weight roofline, GROUP_M={m.dense_gm})"
    )
    print(
        f"  gemm_only    : {m.gemm_ms:8.3f} ms | {m.gemm_tf:7.1f} TFLOPS | "
        f"grouped/dense = {m.grouped_eff_pct:.1f}%"
    )
    print(f"  {comm_label:<13}: {m.comm_ms:8.3f} ms | {m.comm_bw:7.1f} {comm_unit}{comm_extra}")
    print(
        f"  fused        : {m.fused_ms:8.3f} ms | {m.fused_tf:7.1f} TFLOPS | "
        f"roofline (max(gemm,{comm_tag})/fused) = {m.roofline_pct:.1f}% | "
        f"speedup vs serial = {m.speedup:.2f}x{fused_extra}"
    )


@functools.lru_cache(maxsize=256)
def _compile_reduce_only(
    out_features,
    num_combine_slots,
    num_reduce_cu,
    topk,
    num_experts,
    rank,
    waves_per_eu=2,
    apply_weights=False,
):
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert topk >= 1, "topk must be >= 1"

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def reduce_only_k(
        OUTPUT: fx.Tensor,
        COMB_LOCAL: fx.Tensor,
        BARRIER_LOCAL: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comb_local_res = create_buffer_resource(COMB_LOCAL, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        barrier_base = extract_base_index(BARRIER_LOCAL, address_space=1)
        topk_reduce_bf16_tile(
            False,
            apply_weights,
            False,
            thread_index,
            block_index,
            fx.Int32(num_reduce_cu * _NUM_WARPS),
            topk,
            out_features,
            num_experts,
            rank,
            comb_local_res,
            output_res,
            topk_indices_res,
            num_tokens_res,
            barrier_base,
            fx.Int32(0),
            topk_weights_res,
            None,
            None,
            fx.Int64(0),
        )

    @flyc.jit
    def launch(
        OUTPUT,
        COMB_LOCAL,
        BARRIER_LOCAL,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        TOPK_WEIGHTS,
        stream: fx.Stream,
    ):
        reduce_only_k(
            OUTPUT,
            COMB_LOCAL,
            BARRIER_LOCAL,
            TOPK_INDICES,
            NUM_TOKENS_PER_RANK,
            TOPK_WEIGHTS,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_reduce_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


@functools.lru_cache(maxsize=256)
def _compile_combine_only_task(out_features, num_experts, num_combine_cu, waves_per_eu=2, with_gate=False):
    """Task-based combine push: grid strides over num_experts recv-segments; a warp
    sustains ONE peer per segment (mirror of dispatch) -> sustained XGMI link."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def combine_task_k(
        GRAD_GATE: fx.Tensor,
        sym_layout: SymLayout,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        seg_bytes = num_experts * 4
        recv_dst_rank_res = create_buffer_resource_from_addr(
            sym_layout.combine_recv_dst_rank, num_records_bytes=seg_bytes
        )
        recv_start_row_res = create_buffer_resource_from_addr(
            sym_layout.combine_recv_start_row, num_records_bytes=seg_bytes
        )
        recv_count_res = create_buffer_resource_from_addr(
            sym_layout.combine_recv_count, num_records_bytes=seg_bytes
        )
        origin_slot_res = create_buffer_resource_from_addr(
            sym_layout.pool_src_slot, num_records_bytes=int(sym_layout.num_max_pool_tokens) * 4
        )
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None

        local_count = (fx.Int32(num_experts) - block_index + combine_cu - fx.Int32(1)) // combine_cu
        for local_iter in range(local_count):
            combine_bf16_tile(
                sym_layout,
                thread_index=thread_index,
                task_index=block_index + local_iter * combine_cu,
                recv_dst_rank_res=recv_dst_rank_res,
                recv_start_row_res=recv_start_row_res,
                recv_count_res=recv_count_res,
                origin_slot_res=origin_slot_res,
                grad_gate_res=grad_gate_res,
                with_gate=with_gate,
            )

    @flyc.jit
    def launch(GRAD_GATE, sym_layout, stream: fx.Stream):
        combine_task_k(
            GRAD_GATE,
            sym_layout,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_combine_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def combine_only(
    group,
    *,
    BLOCK_M=256,
    num_combine_cu=None,
    grad_gate=None,
):
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.get_sym_layout()
    out_features = int(sym_layout.hidden)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    if num_combine_cu is None:
        num_combine_cu = num_max_pool_tokens // BLOCK_M
    num_tile_blocks = symm.meta_scalars[1:2]
    with_gate = grad_gate is not None
    waves = int(os.environ.get("MEGA_COMB_WAVES") or "2")  # combine push occupancy knob
    grad_gate_arg = grad_gate.contiguous().view(-1) if with_gate else num_tile_blocks
    # task-based push (sustained per-peer): strides over num_experts recv-segments
    launch = _compile_combine_only_task(
        out_features,
        int(sym_layout.num_experts),
        int(num_combine_cu),
        waves_per_eu=waves,
        with_gate=with_gate,
    )
    launch(grad_gate_arg, sym_layout, stream=torch.cuda.current_stream())
    return symm.l2_token_buffer


def topk_reduce_only(
    output,
    comb_local,
    barrier_local,
    topk_indices,
    num_tokens_per_rank,
    num_combine_slots,
    *,
    topk,
    num_experts,
    rank=0,
    num_reduce_cu=32,
    topk_weights=None,
):
    assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"
    out_features = output.size(1)
    apply_weights = topk_weights is not None
    launch = _compile_reduce_only(
        out_features,
        int(num_combine_slots),
        int(num_reduce_cu),
        int(topk),
        int(num_experts),
        int(rank),
        apply_weights=apply_weights,
    )
    topk_weights_d = topk_weights.contiguous().view(-1) if apply_weights else num_tokens_per_rank
    launch(
        output.view(-1),
        comb_local.contiguous().view(-1),
        barrier_local,
        topk_indices.contiguous().view(-1),
        num_tokens_per_rank,
        topk_weights_d,
        stream=torch.cuda.current_stream(),
    )
    return output
