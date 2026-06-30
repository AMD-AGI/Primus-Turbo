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
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
import torch.distributed as dist
from flydsl.expr import arith
from flydsl.expr.buffer_ops import buffer_load, create_buffer_resource

from primus_turbo.flydsl.common.tile_spec import _emit_if_then
from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
from primus_turbo.flydsl.mega.ep_intranode import _BLOCK_THREADS, dispatch_bf16_tile
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (
    _compile_dense_nt,
    _compile_grouped_tn_wgrad_bf16,
    _get_compiled_dense,
    _make_shared_storage,
    gemm_bf16_nn_tile,
    gemm_bf16_nt_tile,
    gemm_bf16_tn_tile,
)
from primus_turbo.flydsl.mega.swiglu_kernel import swiglu
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.gemm_helper import (
    ceildiv,
    make_value_attrs,
    xcd_remap_pid,
)

# per-tile GEMM closure by layout (NT forward, NN dgrad, TN wgrad); grouped via b_group_base
_GEMM_TILE = {"nt": gemm_bf16_nt_tile, "nn": gemm_bf16_nn_tile, "tn": gemm_bf16_tn_tile}

# re-export so benchmarks can import everything mega from one place
__all__ = [
    "generate_routing",
    "global_weights",
    "dense_gemm_peak_ms",
    "bench",
    "gate3",
    "check_accuracy",
    "generate_input",
    "compute_stage_metrics",
    "print_header",
    "print_stage",
    "dispatch_only",
    "dispatch_prologue",
    "grouped_gemm_bf16_only",
    "grouped_gemm_tn_wgrad_only",
    "compile_grouped_gemm_bf16",
    "get_symm_buffer_for_mega_moe",
    "swiglu",
]


# --------------------------------------------------------------------------- #
# Bench-only baselines (moved here from the kernel module so the kernel file keeps
# only the fused dispatch_grouped_gemm_bf16): the grouped-GEMM / dispatch-only /
# variable-K TN wgrad compute peaks, plus their thin host wrappers.
# --------------------------------------------------------------------------- #
def _bf16_flat(t):
    return t.contiguous().view(-1)


# ───────────────────────────────────────────────────────────────────────
# Grouped GEMM-only launcher (the compute-peak baseline). Dense XCD-swizzle
# scheduler + GROUP_M, per-expert B slab via tile_to_expert.
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def compile_grouped_gemm_bf16(
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    layout="nt",
):
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    gemm_tile = _GEMM_TILE[layout]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped(
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
        ntb = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(ntb, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
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

        _emit_if_then(fx.block_idx.x < real_grid, _emit)

    @flyc.jit
    def launch(
        A, B, C, TILE_TO_GROUP, NUM_TILE_BLOCKS, c_m: int, c_n: int, stream: fx.Stream = fx.Stream(None)
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_grouped(
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


# ───────────────────────────────────────────────────────────────────────
# Dispatch-PUSH-only launcher (no GEMM, no scoreboard) — raw dispatch bandwidth.
# Every block pushes a round-robin share of the comm tasks to peer pools over XGMI.
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_dispatch_only(hidden_size, num_max_pool_tokens, num_dispatch_cu, num_comm, waves_per_eu=2):
    # split each task across blocks_per_task blocks when tasks < blocks (saturate XGMI)
    blocks_per_task = max(1, num_dispatch_cu // num_comm)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_only_k(
        INPUT_TOKENS: fx.Tensor,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        POOL_PTRS: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        pool_address_resource = create_buffer_resource(POOL_PTRS, max_size=True)

        dispatch_tile = dispatch_bf16_tile(
            thread_index=thread_index,
            hidden_size=hidden_size,
            num_max_pool_tokens=num_max_pool_tokens,
            input_resource=input_resource,
            expert_send_dst_rank_resource=expert_send_dst_rank_resource,
            expert_send_dst_row_resource=expert_send_dst_row_resource,
            expert_send_count_resource=expert_send_count_resource,
            expert_send_offset_resource=expert_send_offset_resource,
            dispatched_token_idx_resource=dispatched_token_idx_resource,
            pool_address_resource=pool_address_resource,
            signal=False,
        )

        if blocks_per_task > 1:
            task_index = block_index // fx.Int32(blocks_per_task)
            sub = block_index % fx.Int32(blocks_per_task)
            if task_index < fx.Int32(num_comm):
                dispatch_tile(task_index, sub, blocks_per_task)
        else:
            if block_index < comm_block_count:
                local_task_count = (
                    fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)
                ) // comm_block_count
                for it in range(local_task_count):
                    dispatch_tile(block_index + it * comm_block_count, fx.Int32(0), 1)

    @flyc.jit
    def launch(
        INPUT_TOKENS,
        EXPERT_SEND_DST_RANK,
        EXPERT_SEND_DST_ROW,
        EXPERT_SEND_COUNT,
        EXPERT_SEND_OFFSET,
        DISPATCHED_TOKEN_IDX,
        POOL_PTRS,
        stream: fx.Stream = fx.Stream(None),
    ):
        dispatch_only_k(
            INPUT_TOKENS,
            EXPERT_SEND_DST_RANK,
            EXPERT_SEND_DST_ROW,
            EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET,
            DISPATCHED_TOKEN_IDX,
            POOL_PTRS,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_dispatch_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# ───────────────────────────────────────────────────────────────────────
# Grouped TN wgrad (variable-K) GEMM-only baseline over dispatched pools:
# dW[g] = lhs_pool[g]^T @ rhs_pool[g], out [G, OUT_M, OUT_N]. The compute core is
# the canonical _compile_grouped_tn_wgrad_bf16 (gemm_bf16_kernel); the fused
# dispatch+wgrad path lives in dispatch_grouped_gemm_bf16(layout="tn").
# ───────────────────────────────────────────────────────────────────────
_WGRAD_DG_COMPILED = {}


def grouped_gemm_tn_wgrad_only(
    lhs_pool,  # [M_pool, OUT_M] bf16   dispatched lhs (e.g. recomputed activation)
    rhs_pool,  # [M_pool, OUT_N] bf16   dispatched rhs (e.g. dY)
    group_offs,  # [G+1] int32           per-group token boundaries in the pool
    out_dw,  # [G, OUT_M, OUT_N] bf16 ([G, OUT_N, OUT_M] if trans_c)  C = dW
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=1,  # xcd=1 maximizes L2 reuse on the variable-K M-reduction (+14% vs 8, +6% vs 4; swept)
    trans_c=False,
    waves_per_eu=2,
):
    """GEMM-only grouped TN wgrad over dispatched pools (no comm). dW[g] =
    lhs_pool[offs[g]:offs[g+1]]^T @ rhs_pool[offs[g]:offs[g+1]] (transposed if trans_c)."""
    G = group_offs.numel() - 1
    OUT_M = lhs_pool.shape[1]
    OUT_N = rhs_pool.shape[1]
    out_fp16 = out_dw.dtype == torch.float16
    go32 = group_offs.to(torch.int32) if group_offs.dtype != torch.int32 else group_offs
    # trans_c: produce C^T = rhs^T @ lhs by swapping operands (both K-major, symmetric)
    # so the result lands [G, OUT_N, OUT_M] via the FAST coalesced normal store.
    if trans_c:
        lhs_e, rhs_e, OUT_M_e, OUT_N_e = rhs_pool, lhs_pool, OUT_N, OUT_M
    else:
        lhs_e, rhs_e, OUT_M_e, OUT_N_e = lhs_pool, rhs_pool, OUT_M, OUT_N
    launch = _compile_grouped_tn_wgrad_bf16(
        OUT_M_e,
        OUT_N_e,
        G,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        waves_per_eu=waves_per_eu,
    )
    args = (
        _bf16_flat(lhs_e),
        _bf16_flat(rhs_e),
        out_dw.view(-1),
        go32,
        OUT_M_e,
        OUT_N_e,
        torch.cuda.current_stream(),
    )
    key = (OUT_M_e, OUT_N_e, G, BLOCK_M, BLOCK_N, out_fp16, waves_per_eu)
    compiled = _WGRAD_DG_COMPILED.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _WGRAD_DG_COMPILED[key] = compiled
    compiled(*args)
    return out_dw


def grouped_gemm_bf16_only(
    pool,  # [M, K] bf16   A operand (pre-filled activation)
    weight,  # [G, N, K] bf16   per-expert B
    output,  # [M, N] bf16   C
    tile_to_expert,  # [n_mblk] i32   expert id per BM pool block
    num_tile_blocks,  # [1] i32        real tile-block count (device)
    *,
    layout="nt",
    BM=256,
    BN=256,
    GROUP_M=4,
    # xcd=1 maximizes L2 reuse of shared weight slabs (swept: +3% NT, +11% NN, +15% TN vs xcd=8)
    num_xcd=1,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
):
    """Pure grouped BF16 GEMM (no dispatch) — the compute-peak baseline.

    ``pool`` is A=[M,K] bf16, ``output`` is [M,N] bf16, ``tile_to_expert`` maps each
    BM pool block -> expert. Weight layout: NT (forward) ``weight`` [G,N,K];
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
        BLOCK_M=BM,
        BLOCK_N=BN,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        nt_vmcnt=int(nt_vmcnt),
        waves_per_eu=int(waves_per_eu),
        agpr_alloc=int(agpr_alloc),
        layout=layout,
    )
    launch(
        _bf16_flat(pool),
        weight_flat,
        _bf16_flat(output),
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
    pool,  # [pool_capacity, K] bf16   local landing pool
    pool_ptrs,  # [world] i64   peer pool base ptrs
    *,
    num_dispatch_cu=32,
):
    """Cross-rank dispatch PUSH only (no GEMM) — pushes ``x`` token rows to peer
    pools over XGMI. Bytes pushed per rank = (sum expert_send_count) * hidden * 2."""
    expert_send_dst_rank, expert_send_dst_row, expert_send_count, expert_send_offset, dispatched_token_idx = (
        handle[:5]
    )
    num_comm = expert_send_dst_rank.numel()
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(1)
    pool_capacity = pool.size(0)
    x_i32 = x.contiguous().view(torch.int32).view(-1)
    launch = _compile_dispatch_only(hidden_size, pool_capacity, int(num_dispatch_cu), int(num_comm))
    launch(
        x_i32,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        pool_ptrs,
        stream=torch.cuda.current_stream(),
    )


# --------------------------------------------------------------------------- #
# Routing + weights
# --------------------------------------------------------------------------- #
def generate_routing(num_tokens, num_topk, num_experts, mode, *, device="cuda", seed=0):
    """(topk_idx[T,K] int64, topk_weight[T,K] f32) for load_balanced / round_robin modes."""
    g = torch.Generator(device=device).manual_seed(seed)
    if mode == "load_balanced":
        scores = torch.rand(num_tokens, num_experts, generator=g, device=device).abs() + 1
        topk_weight, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)
        topk_idx = topk_idx.to(torch.int64)
        topk_weight = topk_weight.to(torch.float32)
    elif mode == "round_robin":
        topk_idx = (
            torch.arange(num_tokens * num_topk, device=device).view(num_tokens, num_topk) % num_experts
        ).to(torch.int64)
        topk_weight = (
            torch.rand(num_tokens, num_topk, generator=g, device=device).softmax(-1).to(torch.float32)
        )
    else:
        raise ValueError(f"unknown routing mode: {mode}")
    return topk_idx, topk_weight


def global_weights(E, I, H, device):
    """Deterministic global expert weights (identical on every rank, then sliced)."""
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    return W1, W2


# --------------------------------------------------------------------------- #
# Bench helper: warmup, one L2 flush before timing, then CUDA-event timing.
# Cross-rank variants (group set) barrier around an optional scoreboard reset.
# --------------------------------------------------------------------------- #
_L2_FLUSH_BUF = None


def _l2_flush():
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
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

    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]
    return np.average(times)


# --------------------------------------------------------------------------- #
# Accuracy check: single source of truth for the gate-3 (cos / rel_rmse / ok)
# correctness verdict shared by every mega-MoE benchmark stage. Mirrors the e2e
# test's _gate3 so a bench PASS means the same thing as the test PASS.
# --------------------------------------------------------------------------- #
def gate3(out, ref, *, cos_thresh=0.99, rel_thresh=0.05):
    """(cos, rel_rmse, ok) of a kernel output vs its reference (flattened, fp32)."""
    g, r = out.float().flatten(), ref.float().flatten()
    cos = float(torch.dot(g, r) / (g.norm() * r.norm() + 1e-12))
    rel = float((g - r).norm() / (r.norm() + 1e-12))
    return cos, rel, (cos >= cos_thresh and rel <= rel_thresh)


def check_accuracy(group, name, out, ref, *, cos_thresh=0.99, rel_thresh=0.05):
    """Gate-3 one stage across all ranks; rank 0 prints the worst (min-cos) rank,
    every rank returns the global verdict (worst_cos, worst_rel, all_ok) so the
    caller can record it. None out/ref -> skipped (returns None)."""
    if out is None or ref is None:
        if group.rank() == 0:
            print(f"  [check] {name:<28}: (skipped — no reference)")
        return None
    cos, rel, ok = gate3(out, ref, cos_thresh=cos_thresh, rel_thresh=rel_thresh)
    world = group.size()
    gathered = [None] * world
    dist.all_gather_object(gathered, (group.rank(), cos, rel, ok), group=group)
    worst = min(gathered, key=lambda t: t[1])  # lowest cos = worst rank
    all_ok = all(g[3] for g in gathered)
    if group.rank() == 0:
        worst_r, worst_cos, worst_rel, _ = worst
        print(
            f"  [check] {name:<28}: cos={worst_cos:.5f} rel={worst_rel:.4f} "
            f"(worst rank={worst_r}) {'PASS' if all_ok else 'FAIL'}"
        )
    return worst[1], worst[2], all_ok


def dense_gemm_peak_ms(M, N, K, BM, BN, iters, *, group_m_cands=(4,)):
    """Dense NT bf16 GEMM (gemm_bf16_kernel) of the SAME M x N x K as the grouped
    GEMM -> the single-weight compute roofline. autotune sweeps GROUP_M {1,4,8};
    default off uses the single best (GROUP_M=4). Returns (best_ms, best_group_m)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 8
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 8  # NT: B [N,K]
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    dense_args = (a.view(-1), b.view(-1), c.view(-1), M, N, torch.cuda.current_stream())
    best_ms, best_group_m = float("inf"), None
    for group_m in group_m_cands:
        launch = _compile_dense_nt(K=K, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=group_m, num_xcd=8)
        compiled = _get_compiled_dense(launch, dense_args)
        ms = bench(lambda: compiled(*dense_args), iters=iters)
        if ms < best_ms:
            best_ms, best_group_m = ms, group_m
    del a, b, c
    return best_ms, best_group_m


# --------------------------------------------------------------------------- #
# Input builders (consistent with the EP test): build the SymmBuffer, run the
# prologue to produce the dispatch handle, then fill the real pool/activation.
# `kind` selects which kernel's inputs to materialize.
# --------------------------------------------------------------------------- #
def _build_symm_and_plan(group, *, T, H, I, E, K, BM, BN, mode, base_seed):
    """Shared prologue: SymmBuffer + routing + dispatch handle (no pool fill yet)."""
    rank = group.rank()
    torch.manual_seed(base_seed + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_weight = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)

    # one symmetric allocation for every cross-rank + scratch buffer (production arena)
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
        block_m=BM,
        block_n=BN,
    )
    # prologue -> the flat dispatch handle (same path as the test); resets scoreboard +
    # barrier_local in-kernel and ends with a cross-rank barrier. The handle IS the full
    # prologue tuple (tile_to_expert / tile_expected / group_offs ride at fixed indices).
    handle = dispatch_prologue(
        topk_idx,
        topk_weight,
        sym_layout=symm.make_sym_layout(),
        num_tokens=T,
        num_topk=K,
        num_experts=E,
        world_size=symm.world,
        rank=symm.rank,
        experts_per_rank=E // symm.world,
        block_m=symm.block_m,
        num_max_pool_tokens=symm.num_max_pool_tokens,
        no_cpu_sync=True,
    )
    tile_to_expert, expected = handle[7], handle[8]
    num_tile_blocks = symm.meta_scalars[1:2]  # device real-tile count
    symm.assert_capacity()  # fail loudly rather than silently drop rows
    return symm, x, topk_idx, topk_weight, handle, tile_to_expert, expected, num_tile_blocks


def generate_input(group, *, kind, mode, T, H, I, E, K, BM, BN, W1, W2, num_dispatch_cu=16, base_seed=7):
    """Build the real inputs for one of the mega kernels over the production SymmBuffer.

    kind="dispatch": pool is filled by a single dispatch PUSH (real A for the L1 GEMM).
    kind="combine":  prologue + dispatch + L1 GEMM + SwiGLU -> act (real L2 input), and
                     l2_token_buffer is filled once so combine_only has real rows."""
    group.rank()
    symm, x, topk_idx, topk_weight, handle, tile_to_expert, expected, num_tile_blocks = _build_symm_and_plan(
        group, T=T, H=H, I=I, E=E, K=K, BM=BM, BN=BN, mode=mode, base_seed=base_seed
    )

    if kind == "dispatch":
        # L1 GEMM output (2*inter wide) has no slot in the arena -> local scratch
        l1_out = torch.empty((symm.num_max_pool_tokens, 2 * I), dtype=torch.bfloat16, device="cuda")
        destination, count = handle[0], handle[2]  # dst_rank, expert_send_count
        # fill the pool (real A) via dispatch_only (peers synced by the prologue)
        torch.cuda.synchronize()
        group.barrier()
        dispatch_only(x, handle, symm.pool, symm.pool_ptrs, num_dispatch_cu=num_dispatch_cu)
        torch.cuda.synchronize()
        group.barrier()
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
        # sb_l2 is a same-rank scoreboard accumulator -> zero it (no barrier needed).
        symm.sb_l2.zero_()
        # cross-rank dispatch PUSH (fills the pool), then grouped L1 GEMM (NT):
        # pool[M,H] @ W1[g,2I,H] -> l1_out[M,2I] (local scratch, no arena slot)
        torch.cuda.synchronize()
        group.barrier()
        dispatch_only(x, handle, symm.pool, symm.pool_ptrs, num_dispatch_cu=32)
        torch.cuda.synchronize()
        group.barrier()
        l1_out = torch.empty((symm.num_max_pool_tokens, 2 * I), dtype=torch.bfloat16, device="cuda")
        grouped_gemm_bf16_only(symm.pool, W1, l1_out, tile_to_expert, num_tile_blocks, BM=BM, BN=BN)
        # fused SwiGLU activation -> act (L2 GEMM input)
        act = swiglu(l1_out)
        # fill l2_token_buffer once with the real rows (for combine_only)
        grouped_gemm_bf16_only(act, W2, symm.l2_token_buffer, tile_to_expert, num_tile_blocks, BM=BM, BN=BN)
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


def print_header(tag, gpu_name, world, args, mode):
    """Common '===' header line shared by both benchmarks."""
    print(
        f"\n{'='*72}\n[{tag}] {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
        f"I={args.inter} E={args.num_experts} K={args.num_topk} mode={mode} "
        f"(max over ranks)\n{'='*72}"
    )


def print_stage(m, *, comm_label, comm_unit, comm_tag, comm_extra="", fused_extra="", sub_header=None):
    """Print one stage (forward or backward) in the shared 4-line layout.

    comm_label : 'dispatch_only' | 'combine_only'   (left column)
    comm_unit  : e.g. 'GB/s (XGMI, nodeup)' | 'GB/s (XGMI)'
    comm_tag   : 'disp' | 'comb'                     (roofline formula text)
    comm_extra / fused_extra : kernel-specific suffixes (e.g. CU sweep strings)."""
    if sub_header is not None:
        print(f"  {'-'*68}  {sub_header}")
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
