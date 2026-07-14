###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 mega MoE kernels (EP, intra-node).

Two modes, selected with --mode:
  dispatch_grouped_gemm : fused dispatch + grouped GEMM
  grouped_gemm_combine  : fused grouped GEMM + combine

Self-contained: the shared reference ops (routing/weight generation, the
dense-GEMM roofline, the CUDA-event bench helper, the prologue-driven input
builders, and the per-stage metric/print template) live in this file too.
"""

import argparse
import functools
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tabulate import tabulate

# config + this module's helpers are same-dir (auto on the script-dir path)
from config import gen_moe_test_cases, get_platform_info

# repo root (primus_turbo) on the path
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import arith  # noqa: E402
from flydsl.expr.buffer_ops import (  # noqa: E402
    buffer_load,
    create_buffer_resource,
    extract_base_index,
)
from flydsl.expr.typing import AddressSpace, PointerType  # noqa: E402

# primus_turbo.flydsl.* imported before primus_turbo.pytorch (kept from the
# original mega_utils order; the two fused kernels below need pytorch first).
from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (  # noqa: E402
    _compile_dense_nt,
    _compile_grouped_variable_k_bf16,
    _get_compiled_dense,
    _i64,
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.mega import (  # noqa: E402
    dispatch_prologue_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.flydsl.mega.ep_intranode import (  # noqa: E402
    _BLOCK_THREADS,
    _NUM_WARPS,
    _PVEC,
    combine_bf16_tile,
    dispatch_bf16_tile,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    BLOCK_M as _POOL_BLOCK_M,
)
from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    TOKEN_DTYPE,
    SymBuffer,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.utils.gemm_helper import (  # noqa: E402
    ceildiv,
    make_value_attrs,
    xcd_remap_pid,
)

# import primus_turbo.pytorch first to dodge the mega kernels' circular import
import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega import (  # noqa: E402
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
)
from primus_turbo.pytorch.ops import grouped_gemm as turbo_grouped_gemm  # noqa: E402


# --------------------------------------------------------------------------- #
# Per-model sweep helpers: iterate config.gen_moe_test_cases, skip unrunnable cases.
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
# Bench-only baselines: grouped-GEMM / dispatch-only / variable-K TN wgrad peaks + host wrappers.
# --------------------------------------------------------------------------- #
# Grouped GEMM-only launcher (compute-peak baseline): dense XCD-swizzle + GROUP_M, per-expert B slab.
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
        # XCD-swizzle over the REAL tile range only (front-loaded); swizzling the full padded pool scatters real tiles -> ~2x slower.
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
            # Worst-case pool (cap*K > 2^31): rebase A/C per tile in int64, int32 in-resource offset. Mirrors fused nt/nn.
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
# Dispatch-PUSH-only launcher (no GEMM/scoreboard): raw dispatch bandwidth over XGMI.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=256)
def _compile_dispatch_only(
    hidden_size,
    num_max_pool_tokens,
    num_dispatch_cu,
    num_comm,
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    waves_per_eu=2,
):
    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_only_k(
        INPUT_TOKENS: fx.Tensor,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_buffer: SymBuffer,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comm_block_count = fx.Int32(num_dispatch_cu)
        # build workspace (hoist heap-derived ptrs before dynamic control flow)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden_size,
            token_dtype=TOKEN_DTYPE,
        )
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
                    sym_buffer,
                    workspace,
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
        sym_buffer,
        stream: fx.Stream,
    ):
        dispatch_only_k(
            INPUT_TOKENS,
            EXPERT_SEND_DST_RANK,
            EXPERT_SEND_DST_ROW,
            EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET,
            DISPATCHED_TOKEN_IDX,
            sym_buffer,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_dispatch_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# --------------------------------------------------------------------------- #
# Grouped TN wgrad (variable-K) GEMM-only baseline: dW[g] = lhs_pool[g]^T @ rhs_pool[g] -> [G, OUT_M, OUT_N].
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
    # trans_c: C^T = rhs^T @ lhs by swapping operands -> [G, OUT_N, OUT_M] via fast coalesced store.
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
    # lhs/rhs pools as 2-D (flat view(-1) overflows int32 shape ABI); kernel rebases per group.
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
    # Pass A/C as 2-D (flat view(-1) overflows int32 shape ABI); kernel rebases per tile.
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
    launch = _compile_dispatch_only(
        hidden_size,
        pool_capacity,
        int(num_dispatch_cu),
        int(num_comm),
        int(symm.world),
        int(symm.num_experts),
        int(symm.num_max_tokens_per_rank),
        int(symm.num_topk),
    )
    launch(
        x_i32,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        symm.get_sym_buffer(),
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
# Bench helper
# --------------------------------------------------------------------------- #


def bench(fn, *, warmup=20, iters=30):
    """Mean ms/call (CUDA events), copied from deep_ep utils.bench."""
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(warmup):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]
    return np.average(times)


# --------------------------------------------------------------------------- #
# Accuracy check: gate-3 (cos / rel_rmse / ok) shared by all stages; mirrors the e2e test's _gate3.
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
# Input builders (like the EP test): build SymmBuffer, run prologue for the handle, fill pool/activation; `kind` picks the kernel.
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
    # prologue -> flat dispatch handle (same as test); resets scoreboard+barrier, cross-rank barrier. Handle IS the full prologue tuple.
    handle = dispatch_prologue_flydsl_kernel(
        topk_idx,
        topk_weight,
        sym_buffer=symm.get_sym_buffer(),
        num_tokens=T,
        num_topk=K,
        num_experts=E,
        num_ranks=symm.world,
        rank=symm.rank,
        experts_per_rank=E // symm.world,
        block_m=_POOL_BLOCK_M,
        num_max_pool_tokens=symm.num_max_pool_tokens,
        hidden=symm.hidden,
        num_max_tokens_per_rank=symm.num_max_tokens_per_rank,
    )
    # pool_src_slot is cross-rank (symm); ride it on the handle like the production path
    handle = tuple(handle) + (symm.pool_src_slot,)
    tile_to_expert = handle[5]
    num_tile_blocks = handle[8]  # device real-tile count (prologue-written, per-forward)
    return symm, x, topk_idx, topk_weight, handle, tile_to_expert, num_tile_blocks


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
    symm, x, topk_idx, topk_weight, handle, tile_to_expert, num_tile_blocks = _build_symm_and_plan(
        group, T=T, H=H, I=I, E=E, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, base_seed=base_seed
    )
    # this rank's expert weights: deterministic global set (fixed seed) sliced per rank -> consistent across ranks.
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
            num_tile_blocks=num_tile_blocks,
            destination=destination,
            count=count,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            W1=W1,
            W2=W2,
        )

    if kind == "combine":
        # combine scoreboard = monotonic combine_flag (never reset). dispatch PUSH fills pool, then grouped L1 GEMM (NT): pool[M,H]@W1 -> l1_out[M,2I].
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
        act = swiglu_flydsl_kernel(l1_out, num_tile_blocks)
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
# Cross-rank aggregation + CSV helpers; per-rank result {"stages": {stage: StageMetrics}} carries gemm/dense/comm/fused ms + flops.
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
        f"\n{'=' * 72}\n[{tag}]{case_tag} {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
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
def _compile_combine_only_task(
    out_features,
    num_experts,
    num_combine_cu,
    num_ranks,
    num_max_tokens_per_rank,
    num_topk,
    waves_per_eu=2,
    with_gate=False,
):
    """Task-based combine push: grid strides over num_experts recv-segments; a warp
    sustains ONE peer per segment (mirror of dispatch) -> sustained XGMI link."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def combine_task_k(
        GRAD_GATE: fx.Tensor,
        RECV_DST_RANK: fx.Tensor,
        RECV_START_ROW: fx.Tensor,
        RECV_COUNT: fx.Tensor,
        POOL_SRC_SLOT: fx.Tensor,
        sym_buffer: SymBuffer,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        # build workspace (hoist heap-derived ptrs before dynamic control flow)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            out_features,
            token_dtype=TOKEN_DTYPE,
        )
        # recv-segment table + origin slots ride the handle (per-forward local copies)
        recv_dst_rank_res = create_buffer_resource(RECV_DST_RANK, max_size=True)
        recv_start_row_res = create_buffer_resource(RECV_START_ROW, max_size=True)
        recv_count_res = create_buffer_resource(RECV_COUNT, max_size=True)
        origin_slot_res = create_buffer_resource(POOL_SRC_SLOT, max_size=True)
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None

        local_count = (fx.Int32(num_experts) - block_index + combine_cu - fx.Int32(1)) // combine_cu
        for local_iter in range(local_count):
            combine_bf16_tile(
                sym_buffer,
                workspace,
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
    def launch(
        GRAD_GATE,
        RECV_DST_RANK,
        RECV_START_ROW,
        RECV_COUNT,
        POOL_SRC_SLOT,
        sym_buffer,
        stream: fx.Stream,
    ):
        combine_task_k(
            GRAD_GATE,
            RECV_DST_RANK,
            RECV_START_ROW,
            RECV_COUNT,
            POOL_SRC_SLOT,
            sym_buffer,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_combine_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def combine_only(
    group,
    *,
    handle,
    BLOCK_M=256,
    num_combine_cu=None,
    grad_gate=None,
):
    symm = get_symm_buffer_for_mega_moe()
    sym_buffer = symm.get_sym_buffer()  # pure addressing handle
    out_features = int(symm.hidden)  # dims live on SymmBuffer, not the handle
    num_max_pool_tokens = int(symm.num_max_pool_tokens)
    if num_combine_cu is None:
        num_combine_cu = num_max_pool_tokens // BLOCK_M
    # recv-segment table + origin slots ride the handle (per-forward, not shared symm)
    recv_dst_rank, recv_start_row, recv_count, pool_src_slot = (
        handle[9],
        handle[10],
        handle[11],
        handle[12],
    )
    with_gate = grad_gate is not None
    waves = int(os.environ.get("MEGA_COMB_WAVES") or "2")  # combine push occupancy knob
    grad_gate_arg = grad_gate.contiguous().view(-1) if with_gate else recv_count
    # task-based push (sustained per-peer): strides over num_experts recv-segments
    launch = _compile_combine_only_task(
        out_features,
        int(symm.num_experts),
        int(num_combine_cu),
        int(symm.world),
        int(symm.num_max_tokens_per_rank),
        int(symm.num_topk),
        waves_per_eu=waves,
        with_gate=with_gate,
    )
    launch(
        grad_gate_arg,
        recv_dst_rank,
        recv_start_row,
        recv_count,
        pool_src_slot,
        sym_buffer,
        stream=torch.cuda.current_stream(),
    )
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


# Fast-skip flydsl tiling gaps: moe_intermediate_size not a multiple of GEMM tile (BN=256) -> autotune fails. TODO(mega): add tail N tile.
UNSUPPORTED = {"DeepSeek-V2-Lite", "MoE-1T"}


###############################################################################
# Shared scaffolding (both modes)
###############################################################################


@dataclass
class StageMetrics:
    """Raw per-rank timings + work for one stage (fields match compute_stage_metrics kwargs)."""

    gemm_ms: float
    dense_ms: float
    dense_gm: int
    comm_ms: float
    fused_ms: float
    flops: float


@dataclass
class StageSpec:
    """Declarative spec for one benchmarked stage; fused_fn is reused for timing + accuracy check."""

    name: str
    flops: float
    dense_dims: tuple[int, int, int]
    gemm_fn: Callable
    comm_fn: Callable  # comm baseline (dispatch_only / combine_only)
    fused_fn: Callable
    ref_fn: Callable
    acc_slice: slice = slice(None)


class StageRunner:
    """Binds per-run context so each stage passes only its own knobs; run does the shared template."""

    def __init__(self, group, args, synced_fn, group_m_cands):
        self.group = group
        self.args = args
        self.synced_fn = synced_fn
        self.group_m_cands = group_m_cands

    def run(self, spec):
        """Time gemm / dense roofline / comm_only / fused, then gate accuracy under a synced bracket."""
        args = self.args
        dense_m, dense_n, dense_k = spec.dense_dims
        t_gemm = bench(spec.gemm_fn, iters=args.iters)
        t_dense, dense_gm = dense_gemm_peak_ms(
            dense_m, dense_n, dense_k, 256, 256, args.iters, group_m_cands=self.group_m_cands
        )
        t_comm = bench(spec.comm_fn, iters=args.iters)
        t_fused = bench(spec.fused_fn, iters=args.iters)
        metrics = StageMetrics(
            gemm_ms=t_gemm,
            dense_ms=t_dense,
            dense_gm=dense_gm,
            comm_ms=t_comm,
            fused_ms=t_fused,
            flops=spec.flops,
        )
        # accuracy: fused vs ref over the SAME symm state (both under a synced bracket)
        ref_out = self.synced_fn(spec.ref_fn)
        out = self.synced_fn(spec.fused_fn)[0]
        check = check_accuracy(self.group, spec.name, out[spec.acc_slice], ref_out[spec.acc_slice])
        return metrics, check


def _make_runner(group, args):
    """Build a StageRunner with a synced-bracket probe wrapper (shared by both modes)."""

    # synced bracket isolates accuracy probes from timing; scoreboard is never-reset parity
    def _synced(run_fn):
        sync_ranks(group)
        out = run_fn()
        sync_ranks(group)
        return out

    group_m_cands = (args.dense_group_m,)
    return StageRunner(group, args, _synced, group_m_cands)


###############################################################################
# Mode: dispatch_grouped_gemm
###############################################################################


@dataclass
class DispatchContext:
    """Shared per-run state built by _dispatch_make_context: config, input/symm buffers, derived geometry."""

    # collective + CLI config
    group: Any
    args: Any
    rank: int
    # dispatched input namespace + symmetric buffers (own x / handle / pool / cap)
    inp: Any
    symm: Any
    # derived geometry: expert-major pool, each expert padded to a BM multiple
    M_eff: int  # total padded pool rows this rank GEMMs
    experts_per_rank: int
    padded_group_lens: Any  # per-expert BM-padded row counts, sum == M_eff
    group_offs: Any  # cumulative padded boundaries [experts_per_rank + 1]


def _dispatch_make_context(group, args):
    """Build input + symm buffers and precompute shared geometry (weights built in generate_input)."""
    inp = generate_input(
        group,
        kind="dispatch",
        T=args.num_tokens,
        H=args.hidden,
        I=args.inter,
        E=args.num_experts,
        K=args.num_topk,
        BLOCK_M=256,
        BLOCK_N=256,
    )
    experts_per_rank = args.num_experts // group.size()
    # BM-padded expert-major per-expert row counts (padding rows zero) so turbo gg matches the layout
    real_tiles = int(inp.num_tile_blocks[0].item())
    M_eff = real_tiles * 256
    tile_experts = inp.tile_to_expert[:real_tiles].to(torch.int64)
    counts = torch.bincount(tile_experts, minlength=experts_per_rank)[:experts_per_rank]
    padded_group_lens = (counts * 256).to(torch.int64)  # sum == M_eff
    # block_m-padded per-expert boundaries for the variable-K wgrads
    group_offs = torch.zeros(experts_per_rank + 1, dtype=torch.int32, device="cuda")
    group_offs[1:] = padded_group_lens.to(torch.int32).cumsum(0)
    return DispatchContext(
        group=group,
        args=args,
        rank=group.rank(),
        inp=inp,
        symm=inp.symm,
        M_eff=M_eff,
        experts_per_rank=experts_per_rank,
        padded_group_lens=padded_group_lens,
        group_offs=group_offs,
    )


def _dispatch_make_fused_call(ctx, lhs, rhs, layout, *, trans_c=False):
    """Build the fused dispatch+GEMM call; stages differ only in operands / layout / trans_c."""
    return lambda: dispatch_grouped_gemm_bf16_flydsl_kernel(
        lhs,
        rhs,
        ctx.group,
        handle=ctx.inp.handle,
        layout=layout,
        BM=256,
        BN=256,
        trans_c=trans_c,
    )


def _dispatch_make_comm_call(ctx, operand):
    """Build the dispatch-only call (the comm baseline for one stage)."""
    return lambda: dispatch_only(operand, ctx.inp.handle, ctx.symm)


def _dispatch_stage_fwd(runner, ctx):
    """forward (NT): N=2I, K=H; returns (metrics, check, xgmi_bytes)."""
    inp, args = ctx.inp, ctx.args
    pool = ctx.symm.dispatch_token_pool
    M_eff, N_fwd, K = ctx.M_eff, 2 * args.inter, args.hidden
    flops = 2.0 * M_eff * N_fwd * K
    # XGMI push bytes per rank = remote rows (dest != rank) x hidden x bf16
    dest_cpu, count_cpu = inp.destination.cpu(), inp.count.cpu()
    remote_rows = int(count_cpu[dest_cpu != ctx.rank].sum().item())
    xgmi_bytes = remote_rows * args.hidden * BF16_BYTES

    spec = StageSpec(
        name="fwd fused (nt)",
        flops=flops,
        dense_dims=(M_eff, N_fwd, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            pool,
            inp.W1,
            inp.l1_out,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            BLOCK_M=256,
            BLOCK_N=256,
        ),
        comm_fn=_dispatch_make_comm_call(ctx, inp.x),
        fused_fn=_dispatch_make_fused_call(ctx, inp.x, inp.W1, "nt"),
        ref_fn=lambda: turbo_grouped_gemm(
            pool[:M_eff].contiguous(), inp.W1, ctx.padded_group_lens, trans_b=True
        ),
        acc_slice=slice(0, M_eff),
    )
    metrics, check = runner.run(spec)
    return metrics, check, xgmi_bytes


def _dispatch_stage_bwd_dgrad(runner, ctx):
    """backward dgrad (NN): dispatch dy + L2 dgrad pool[M,H] @ w2 -> d_swiglu[M,I]; N=I, K=H."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    pool = symm.dispatch_token_pool
    M_eff, N_bwd, K = ctx.M_eff, args.inter, args.hidden
    flops = 2.0 * M_eff * N_bwd * K
    dy = torch.ones(args.num_tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    d_swiglu = torch.empty(symm.num_max_pool_tokens, N_bwd, device="cuda", dtype=torch.bfloat16)
    # fill the pool with dy once so the gemm-only baseline contracts the right rows
    sync_ranks(ctx.group)
    dispatch_only(dy, inp.handle, symm)
    sync_ranks(ctx.group)

    # accuracy: fused dispatch+GEMM(NN) vs turbo grouped_gemm. NN -> trans_b=False.
    spec = StageSpec(
        name="bwd dgrad fused (nn)",
        flops=flops,
        dense_dims=(M_eff, N_bwd, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            pool,
            inp.W2,
            d_swiglu,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            layout="nn",
            BLOCK_M=256,
            BLOCK_N=256,
        ),
        comm_fn=_dispatch_make_comm_call(ctx, dy),
        fused_fn=_dispatch_make_fused_call(ctx, dy, inp.W2, "nn"),
        ref_fn=lambda: turbo_grouped_gemm(
            pool[:M_eff].contiguous(), inp.W2, ctx.padded_group_lens, trans_b=False
        ),
        acc_slice=slice(0, M_eff),
    )
    return runner.run(spec)


def _dispatch_stage_bwd_wgrad(runner, ctx):
    """backward wgrad dW1 (TN, variable-K): dW1 = pool(x)^T @ grad_l1; OUT_M=H, OUT_N=2I."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    pool = symm.dispatch_token_pool
    cap = symm.num_max_pool_tokens
    M_out, N_out = args.hidden, 2 * args.inter  # dW1: lhs feature = H (pool x), rhs feature = 2I (grad_l1)
    flops = 2.0 * ctx.M_eff * M_out * N_out
    x_pool = torch.ones(cap, M_out, device="cuda", dtype=torch.bfloat16)
    grad_pool = torch.ones(cap, N_out, device="cuda", dtype=torch.bfloat16)
    # trans_c: dW1 stored transposed [G, N_out, M_out] = [G, 2I, H] = W1-native layout
    dW1 = torch.empty(ctx.experts_per_rank, N_out, M_out, device="cuda", dtype=torch.bfloat16)

    ref_dW1 = torch.zeros_like(dW1)  # empty groups -> 0 (matches the fused padded output)
    offs_cpu = ctx.group_offs.tolist()

    def _ref():
        dispatch_only(x_pool, inp.handle, symm)
        sync_ranks(ctx.group)
        for expert in range(ctx.experts_per_rank):
            start, end = offs_cpu[expert], offs_cpu[expert + 1]
            if end > start:  # padded rows are zero -> contract to 0 (skip is equivalent)
                ref_dW1[expert] = (grad_pool[start:end].float().T @ pool[start:end].float()).to(dW1.dtype)
        return ref_dW1

    spec = StageSpec(
        name="wgrad dW1 fused (tn)",
        flops=flops,
        # dense roofline of the same total FLOPs (one [H,2I] GEMM contracting M_eff rows)
        dense_dims=(M_out, N_out, ctx.M_eff),
        gemm_fn=lambda: grouped_gemm_variable_k_only(
            x_pool, grad_pool, ctx.group_offs, dW1, BLOCK_M=256, BLOCK_N=256, trans_c=True
        ),
        comm_fn=_dispatch_make_comm_call(ctx, x_pool),
        fused_fn=_dispatch_make_fused_call(ctx, x_pool, grad_pool, "tn", trans_c=True),
        ref_fn=_ref,
        acc_slice=slice(None),
    )
    return runner.run(spec)


def _dispatch_profile(group, args):
    ctx = _dispatch_make_context(group, args)
    runner = _make_runner(group, args)
    checks = {}
    try:
        fwd_metrics, checks["fwd"], xgmi_bytes = _dispatch_stage_fwd(runner, ctx)
        bwd_metrics, checks["bwd"] = _dispatch_stage_bwd_dgrad(runner, ctx)
        wgrad_metrics, checks["wgrad"] = _dispatch_stage_bwd_wgrad(runner, ctx)
    finally:
        ctx.symm.destroy()  # always free symmetric buffers
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "stages": {"fwd": fwd_metrics, "bwd": bwd_metrics, "wgrad": wgrad_metrics},
        "xgmi_bytes": xgmi_bytes,
        "checks": checks,
    }


###############################################################################
# Mode: grouped_gemm_combine
###############################################################################


@dataclass
class CombineContext:
    """Shared per-run state built by _combine_make_context: config, input/symm buffers, derived geometry + tables."""

    # collective + CLI config
    group: Any
    args: Any
    rank: int
    # input namespace + symmetric buffers (own act / handle / l2 buffers, read off inp/symm)
    inp: Any
    symm: Any
    # derived geometry + tables (computed once, shared by both stages)
    M_eff: int  # total padded pool rows this rank GEMMs
    xgmi_bytes: int  # combine push bytes per rank (same fwd/bwd, H-wide)
    num_tokens: int
    topk_idx_flat: Any  # int32 [T*K], drives the per-token reduce
    topk_w_flat: Any  # f32 [T*K], forward routing weights
    reduce_ready: Any  # standalone reduce ready flags (0 == ready)


def _combine_make_context(group, args):
    """Build input + symm buffers and precompute shared geometry + topk tables (weights built in generate_input)."""
    inp = generate_input(
        group,
        kind="combine",
        T=args.num_tokens,
        H=args.hidden,
        I=args.inter,
        E=args.num_experts,
        K=args.num_topk,
        BLOCK_M=256,
        BLOCK_N=256,
    )
    symm = inp.symm
    rank = group.rank()
    real_tiles = int(inp.num_tile_blocks.item())
    M_eff = real_tiles * 256
    # combine push bytes per rank = remote rows (origin_rank != rank, valid) x H x bf16
    origin = symm.pool_src_rank
    remote_rows = int(((origin != rank) & (origin >= 0)).sum().item())
    xgmi_bytes = remote_rows * args.hidden * BF16_BYTES
    topk_idx_flat = inp.topk_idx.to(torch.int32).contiguous().view(-1)
    topk_w_flat = inp.topk_weight.to(torch.float32).contiguous().view(-1)
    reduce_ready = torch.zeros(int(symm.num_combine_slots), dtype=torch.int32, device="cuda")
    return CombineContext(
        group=group,
        args=args,
        rank=rank,
        inp=inp,
        symm=symm,
        M_eff=M_eff,
        xgmi_bytes=xgmi_bytes,
        num_tokens=int(symm.num_tokens),
        topk_idx_flat=topk_idx_flat,
        topk_w_flat=topk_w_flat,
        reduce_ready=reduce_ready,
    )


def _combine_make_fused_call(ctx, lhs, rhs, *, layout="nt", topk_weights):
    """Build the fused GEMM + combine PUSH + topk-reduce call (3-role); stages differ in operands / layout / weights."""
    return lambda: grouped_gemm_combine_bf16_flydsl_kernel(
        lhs,
        rhs,
        ctx.inp.handle,
        topk_indices=ctx.topk_idx_flat,
        topk_weights=topk_weights,
        layout=layout,
        BM=256,
        BN=256,
    )


def _combine_make_comm_call(ctx):
    """Build the combine-only call (the comm baseline; CU count autotuned)."""
    return lambda: combine_only(ctx.group, handle=ctx.inp.handle)


def _combine_stage_fwd(runner, ctx):
    """forward (e2e step 4, NT): L2 GEMM N=H, K=I; 3-role fused -> weighted y[T,H]."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    N, K = args.hidden, args.inter
    flops = 2.0 * ctx.M_eff * N * K
    ref_y = torch.empty(ctx.num_tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

    # reference: decoupled gemm_only(nt) + combine_only + weighted topk_reduce, over the
    # SAME handle/buffers as fused_fn -> never-reset scoreboard drift cancels between them.
    def _ref_fwd():
        grouped_gemm_bf16_only(
            inp.act,
            inp.W2,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            BLOCK_M=256,
            BLOCK_N=256,
            GROUP_M=8,
        )
        sync_ranks(ctx.group)
        combine_only(ctx.group, handle=ctx.inp.handle)
        sync_ranks(ctx.group)
        ctx.reduce_ready.zero_()  # 0 == ready (reduce_only does not spin)
        topk_reduce_only(
            ref_y,
            symm.combine_token_buffer,
            ctx.reduce_ready,
            ctx.topk_idx_flat,
            symm.num_tokens_per_rank,
            int(symm.num_combine_slots),
            topk=int(symm.num_topk),
            num_experts=int(symm.num_experts),
            rank=ctx.rank,
            topk_weights=ctx.topk_w_flat,  # weighted (forward routing weights)
        )
        return ref_y

    # L2 has small K=I -> GROUP_M=8 reuses the per-expert weight across more M-tiles (best here)
    spec = StageSpec(
        name="fwd fused (nt)",
        flops=flops,
        dense_dims=(ctx.M_eff, N, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            inp.act,
            inp.W2,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            BLOCK_M=256,
            BLOCK_N=256,
            GROUP_M=8,
        ),
        comm_fn=_combine_make_comm_call(ctx),
        fused_fn=_combine_make_fused_call(ctx, inp.act, inp.W2, topk_weights=ctx.topk_w_flat),
        ref_fn=_ref_fwd,
    )
    return runner.run(spec)


def _combine_stage_bwd(runner, ctx):
    """backward (e2e step 3, NN): L1 dgrad grad_l1[M,2I] @ w1 -> grad_pool[M,H]; 3-role fused -> dx[T,H]."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    N, K = args.hidden, 2 * args.inter  # weight [G, K=2I, N=H]
    flops = 2.0 * ctx.M_eff * N * K
    grad_l1 = torch.randn(symm.num_max_pool_tokens, K, device="cuda", dtype=torch.bfloat16) / 8
    ref_dx = torch.empty(ctx.num_tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

    # reference: decoupled gemm_only(nn) + combine_only + unweighted topk_reduce (weight rides grad_l1)
    def _ref_bwd():
        grouped_gemm_bf16_only(
            grad_l1,
            inp.W1,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            layout="nn",
            BLOCK_M=256,
            BLOCK_N=256,
            GROUP_M=8,
        )
        sync_ranks(ctx.group)
        combine_only(ctx.group, handle=ctx.inp.handle)
        sync_ranks(ctx.group)
        ctx.reduce_ready.zero_()  # 0 == ready (reduce_only does not spin)
        topk_reduce_only(
            ref_dx,
            symm.combine_token_buffer,
            ctx.reduce_ready,
            ctx.topk_idx_flat,
            symm.num_tokens_per_rank,
            int(symm.num_combine_slots),
            topk=int(symm.num_topk),
            num_experts=int(symm.num_experts),
            rank=ctx.rank,
            topk_weights=None,  # unweighted (weight rides grad_l1)
        )
        return ref_dx

    spec = StageSpec(
        name="bwd dgrad fused (nn)",
        flops=flops,
        dense_dims=(ctx.M_eff, N, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            grad_l1,
            inp.W1,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            layout="nn",
            BLOCK_M=256,
            BLOCK_N=256,
            GROUP_M=8,
        ),
        comm_fn=_combine_make_comm_call(ctx),
        fused_fn=_combine_make_fused_call(ctx, grad_l1, inp.W1, layout="nn", topk_weights=None),
        ref_fn=_ref_bwd,
    )
    return runner.run(spec)


def _combine_profile(group, args):
    """Forward = e2e step 4 (NT weighted); backward = e2e step 3 (NN unweighted). Both report the 3-role fused kernel."""
    ctx = _combine_make_context(group, args)
    runner = _make_runner(group, args)
    checks = {}
    try:
        fwd_metrics, checks["fwd"] = _combine_stage_fwd(runner, ctx)
        bwd_metrics, checks["bwd"] = _combine_stage_bwd(runner, ctx)
        xgmi_bytes = ctx.xgmi_bytes
    finally:
        ctx.symm.destroy()  # always free the symmetric buffer
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "stages": {"fwd": fwd_metrics, "bwd": bwd_metrics},
        "xgmi_bytes": xgmi_bytes,
        "checks": checks,
    }


###############################################################################
# Mode registry + shared reporting / driver
###############################################################################


@dataclass
class StageReport:
    """Per-stage reporting descriptor: which result key, its column prefix / flags, print sub_header."""

    key: str  # per_rank["stages"] / checks key
    col_prefix: str  # stage_columns name prefix ("", "bwd ", "wgrad ")
    col_flags: dict = field(default_factory=dict)  # extra stage_columns switches (dense_ms / xgmi / hidden)
    sub_header: str = ""  # print_stage sub_header ("" = none, used by the fwd stage)


@dataclass
class ModeSpec:
    """Everything the shared driver needs to run + report one mode."""

    profile: Callable  # (group, args) -> {"stages", "xgmi_bytes", "checks"}
    port: int  # default MASTER_PORT for this mode's spawn
    prefix: str  # output CSV filename prefix
    header_kind: str  # print_header kind label
    comm_label: str  # comm baseline name (dispatch_only / combine_only)
    comm_unit: str  # comm bandwidth unit string
    comm_tag: str  # short comm tag (disp / comb); reused as stage_columns comm_short
    stages: list  # ordered StageReport; [0] is the fwd (csv fwd), the rest are bwd_stages


# fwd / bwd share the same column flags across modes; only labels + the extra wgrad stage differ
_FWD_REPORT = StageReport("fwd", "", {"dense_ms": True, "xgmi": True, "hidden": True})

MODES = {
    "dispatch_grouped_gemm": ModeSpec(
        profile=_dispatch_profile,
        port=8481,
        prefix="dispatch_grouped_gemm",
        header_kind="dispatch",
        comm_label="dispatch_only",
        comm_unit="GB/s (XGMI, nodeup)",
        comm_tag="disp",
        stages=[
            _FWD_REPORT,
            StageReport("bwd", "bwd ", {"xgmi": True}, "backward dgrad (NN, = dispatch_grouped_0)"),
            StageReport(
                "wgrad", "wgrad ", {}, "backward wgrad dW1 (TN, = dispatch + variable-K wgrad)"
            ),
        ],
    ),
    "grouped_gemm_combine": ModeSpec(
        profile=_combine_profile,
        port=8483,
        prefix="grouped_gemm_combine",
        header_kind="combine",
        comm_label="combine_only",
        comm_unit="GB/s (XGMI)",
        comm_tag="comb",
        stages=[
            _FWD_REPORT,
            StageReport("bwd", "bwd ", {"xgmi": True}, "backward dgrad (NN, = mega_moe_fused STEP 3)"),
        ],
    ),
}


def _report_case(mode, platform, gpu_name, world, args, case_name, per_rank):
    """rank-0: aggregate one case's per-rank results -> (rich_row, csv_row); prints the detail block."""
    # distributed bottleneck = slowest rank (max latency); work ~ uniform (mean)
    xgmi = statistics.mean([res["xgmi_bytes"] for res in per_rank])
    rank0_checks = per_rank[0]["checks"]
    aggs = {s.key: aggregate_stage_metrics(per_rank, s.key, xgmi) for s in mode.stages}

    print_header(mode.header_kind, gpu_name, world, args, case=case_name)
    for s in mode.stages:
        extra = {"sub_header": s.sub_header} if s.sub_header else {}
        print_stage(
            aggs[s.key],
            comm_label=mode.comm_label,
            comm_unit=mode.comm_unit,
            comm_tag=mode.comm_tag,
            **extra,
        )
    rich_row = {
        "Platform": platform,
        "GPU": gpu_name,
        "Case": case_name,
        "EP": world,
        "T": args.num_tokens,
        "H": args.hidden,
        "I": args.inter,
        "E": args.num_experts,
        "K": args.num_topk,
    }
    for s in mode.stages:
        rich_row.update(
            stage_columns(
                s.col_prefix,
                aggs[s.key],
                rank0_checks.get(s.key),
                comm_label=mode.comm_label,
                comm_short=mode.comm_tag,
                **s.col_flags,
            )
        )
    # saved CSV follows the gemm_turbo convention; fwd = stages[0], backward = the rest
    csv_row = turbo_csv_row(
        platform,
        gpu_name,
        world,
        args,
        case=case_name,
        check=checks_verdict(rank0_checks),
        fwd=aggs[mode.stages[0].key],
        bwd_stages=[aggs[s.key] for s in mode.stages[1:]],
    )
    return rich_row, csv_row


def _init_dist(local_rank, world, default_port):
    """Bring up NCCL + the collective group for this spawn; returns the process group."""
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", str(default_port)))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{master_addr}:{port}", world_size=world, rank=local_rank
    )
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    return dist.new_group(list(range(world)))


def _benchmark(local_rank, world, args):
    """One spawn sweeps every MoE model for args.mode; unsupported / failing cases are skipped."""
    mode = MODES[args.mode]
    group = _init_dist(local_rank, world, mode.port)
    rank = dist.get_rank()
    platform, gpu_name = get_platform_info()

    try:
        rich_rows, csv_rows = [], []
        for case in gen_moe_test_cases(args.models):
            name = case["Case"]
            if name in UNSUPPORTED:
                if rank == 0:
                    print(f"[skip] {name}: unsupported by mega {args.mode} (see UNSUPPORTED TODO)")
                continue
            apply_case(args, case)
            sync_ranks(group)
            ok, result = True, None
            try:
                result = mode.profile(group, args)
            except Exception as e:  # noqa: BLE001  probe: skip cases the kernel can't run
                ok = False
                if rank == 0:
                    print(f"[skip] {name}: {e!r}")
            # collective agreement so every rank skips a failed case together (no hang)
            if not all_ranks_ok(group, ok):
                sync_ranks(group)
                continue
            per_rank = [None] * world  # all_gather_object fills per_rank[i] from rank i
            dist.all_gather_object(per_rank, result, group=group)
            if rank == 0:
                rich_row, csv_row = _report_case(mode, platform, gpu_name, world, args, name, per_rank)
                rich_rows.append(rich_row)
                csv_rows.append(csv_row)
            sync_ranks(group)

        if rank == 0 and csv_rows:
            print("\nFinal Results:")
            print(tabulate(pd.DataFrame(rich_rows), headers="keys", tablefmt="grid", showindex=False))
            out_file = args.output or f"{mode.prefix}_{datetime.now():%Y%m%d}_{gpu_name}.csv"
            pd.DataFrame(csv_rows).to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")
        sync_ranks(group)
    finally:
        dist.destroy_process_group()


def _run(args):
    """Driver: one spawn iterates all cases in-process for the selected mode."""
    torch.multiprocessing.spawn(_benchmark, args=(args.num_processes, args), nprocs=args.num_processes)


###############################################################################
# Entrypoint
###############################################################################


def _build_parser():
    parser = argparse.ArgumentParser(description="Benchmark the fused BF16 mega MoE kernels")
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        required=True,
        help="which fused mega kernel to benchmark",
    )
    parser.add_argument("--num-processes", type=int, default=8)
    # H/I/E/K come from each MoE model case (config.gen_moe_test_cases); no CLI knob
    parser.add_argument("--num-tokens", type=int, default=8192)
    # GROUP_M for the dense roofline reference
    parser.add_argument("--dense-group-m", type=int, default=4)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", "-o", type=str, default=None)
    # restrict the sweep to these MoE model names (default = all)
    parser.add_argument("--models", nargs="+", default=None)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _run(args)
