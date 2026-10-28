###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 Mega MoE kernels on gfx1250 (MI455X), EP intranode.

The gfx1250 counterpart of bench_mega_moe.py's per-stage view; that bench drives the gfx950 kernels,
whose direct-to-LDS loads do not lower on gfx1250. Per MoE model it times the six GEMM stages of one
fwd+bwd step as ``fused_mega_moe`` runs them on gfx1250, then the whole step through the op:

  L1 fwd     dispatch_gemm (NT): dispatch x + GateUp GEMM
  L2 fwd     gemm_combine (NT): Down GEMM + combine + top-k reduce
  L2 dgrad   dispatch_gemm (NN): dispatch dy + Down dgrad
  dW2        variable-K GEMM reading the dispatched dy in place (no comm of its own)
  L1 dgrad   gemm_combine (NN) with the routing-weight gradient, while peer copies on a side stream
             re-dispatch x for dW1
  dW1        variable-K GEMM on the re-dispatched x

Per stage: gemm_only (the gfx1250 FlyDSL grouped GEMM on the dispatched pool, the unfused path's
GEMM), comm_only (the stage's communication alone), fused (the production launch), hidden =
gemm_only + comm_only - fused, roofline = max(gemm_only, comm_only) / fused, and dense (the same
M x N x K as one group, the grouped GEMM's ceiling). Every fused output is checked against its
unfused reference (cos / rel-RMSE, worst rank). Times are the slowest rank; stage FLOPs count the
pool rows the GEMMs run (padding included), averaged over ranks, and the step row's count
tokens x topk.

Routing is the softmax top-k of N(0, 1) logits with ``--skew`` added to rank 0's experts
(0 = random routing). Each model runs in its own spawn: a second symmetric-buffer shape inside
one EP4 process can hang its rendezvous.

Usage::

    python benchmark/ops/training/bench_mega_moe_gfx1250.py --num-processes 4
    python benchmark/ops/training/bench_mega_moe_gfx1250.py --num-processes 4 --num-experts 128 --skew 1 -o out.csv
"""

import argparse
import functools
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from config import gen_moe_test_cases, get_platform_info
from tabulate import tabulate

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

# import primus_turbo.pytorch first to dodge the mega kernels' circular import
import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (  # noqa: E402
    grouped_gemm_bf16_nn_flydsl_kernel,
    grouped_gemm_bf16_nt_flydsl_kernel,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel import (  # noqa: E402
    dispatch_gemm_nt_gfx1250,
    dispatch_only_gfx1250,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import (  # noqa: E402
    gemm_combine_nt_gfx1250,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.variable_k import (  # noqa: E402
    grouped_gemm_bf16_variable_k_flydsl_kernel,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.x_redispatch_kernel import (  # noqa: E402
    get_x_redispatch,
)
from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (  # noqa: E402
    run_dispatch_prologue,
)
from primus_turbo.flydsl.mega.gfx1250.ep_intranode import (  # noqa: E402
    _BLOCK_THREADS,
    _NUM_WARPS,
    combine_bf16_tile,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.gfx1250.prims import WAVE  # noqa: E402
from primus_turbo.flydsl.mega.gfx1250.swiglu_kernel import (  # noqa: E402
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.flydsl.mega.gfx1250.symm_buffer import (  # noqa: E402
    BLOCK_M,
    TOKEN_DTYPE,
    SymBuffer,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.utils.buffer_ops import (  # noqa: E402
    buffer_load,
    create_buffer_resource,
    extract_base_index,
)
from primus_turbo.pytorch.core.utils import is_gfx1250  # noqa: E402
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe  # noqa: E402

# dispatch_gemm_nt_gfx1250's default: the workgroups its launch gives the dispatch
_DISPATCH_CU = 64
# the gfx1250 tiles take K in whole k-tiles of 64 at least, and at least two of them
_MIN_K = 128


###############################################################################
# Communication-only baselines: the combine push and the top-k reduce alone
###############################################################################


@functools.lru_cache(maxsize=None)
def _compile_combine_only(
    hidden, num_experts, num_ranks, num_max_tokens_per_rank, num_topk, num_combine_cu, has_grad_gate
):
    """Combine push: the grid strides over the num_experts recv segments, one fold-row push per warp."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def combine_only_k(
        GRAD_GATE: fx.Tensor,
        RECV_DST_RANK: fx.Tensor,
        RECV_START_ROW: fx.Tensor,
        RECV_FOLD_COUNT: fx.Tensor,
        POOL_ROW_TO_ROUTE: fx.Tensor,
        POOL_ROW_TO_RECV_TOKEN: fx.Tensor,
        RECV_TOKEN_TO_POOL_ROWS: fx.Tensor,
        L2_TOKENS: fx.Tensor,
        sym_buffer: SymBuffer,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        l2_ptr = fx.arith.ArithValue(
            fx.arith.index_cast(fx.T.i64, extract_base_index(L2_TOKENS)), signed=True
        )
        # hoist the heap-derived pointers before the dynamic loop (the rewriter can't carry Workspace)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            token_dtype=TOKEN_DTYPE,
        )
        recv_dst_rank_res = create_buffer_resource(RECV_DST_RANK, max_size=True)
        recv_start_row_res = create_buffer_resource(RECV_START_ROW, max_size=True)
        recv_fold_count_res = create_buffer_resource(RECV_FOLD_COUNT, max_size=True)
        pool_row_to_route_res = create_buffer_resource(POOL_ROW_TO_ROUTE, max_size=True)
        pool_row_to_recv_token_res = create_buffer_resource(POOL_ROW_TO_RECV_TOKEN, max_size=True)
        recv_token_to_pool_rows_res = create_buffer_resource(RECV_TOKEN_TO_POOL_ROWS, max_size=True)
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if has_grad_gate else None

        local_count = (fx.Int32(num_experts) - block_index + combine_cu - fx.Int32(1)) // combine_cu
        for local_iter in range(local_count):
            segment_idx = block_index + local_iter * combine_cu
            # epoch 0 never satisfies a fused reduce wait, so these flag stores are inert
            combine_bf16_tile(
                sym_buffer,
                workspace,
                thread_index=thread_index,
                segment_idx=segment_idx,
                first_fold_row=buffer_load(recv_start_row_res, segment_idx, vec_width=1, dtype=fx.T.i32),
                num_fold_rows=buffer_load(recv_fold_count_res, segment_idx, vec_width=1, dtype=fx.T.i32),
                recv_dst_rank_res=recv_dst_rank_res,
                pool_row_to_route_res=pool_row_to_route_res,
                pool_row_to_recv_token_res=pool_row_to_recv_token_res,
                recv_token_to_pool_rows_res=recv_token_to_pool_rows_res,
                l2_ptr=l2_ptr,
                epoch=fx.Int64(0),
                reduce_bank=fx.Int32(0),
                grad_gate_res=grad_gate_res,
                has_grad_gate=has_grad_gate,
            )

    @flyc.jit
    def launch(
        GRAD_GATE,
        RECV_DST_RANK,
        RECV_START_ROW,
        RECV_FOLD_COUNT,
        POOL_ROW_TO_ROUTE,
        POOL_ROW_TO_RECV_TOKEN,
        RECV_TOKEN_TO_POOL_ROWS,
        L2_TOKENS,
        sym_buffer,
        stream: fx.Stream,
    ):
        combine_only_k(
            GRAD_GATE,
            RECV_DST_RANK,
            RECV_START_ROW,
            RECV_FOLD_COUNT,
            POOL_ROW_TO_ROUTE,
            POOL_ROW_TO_RECV_TOKEN,
            RECV_TOKEN_TO_POOL_ROWS,
            L2_TOKENS,
            sym_buffer,
        ).launch(grid=(num_combine_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def combine_only(handle, l2_tokens, grad_gate=None):
    """Push every pool row of ``l2_tokens`` to its token's primary route, as the fused combine does."""
    symm = get_symm_buffer_for_mega_moe()
    launch = _compile_combine_only(
        int(symm.hidden),
        int(symm.num_experts),
        int(symm.num_ranks),
        int(symm.num_max_tokens_per_rank),
        int(symm.num_topk),
        int(symm.num_max_pool_tokens) // BLOCK_M,
        grad_gate is not None,
    )
    launch(
        grad_gate.contiguous().view(-1) if grad_gate is not None else handle.num_tile_blocks,
        handle.combine_recv_dst_rank,
        handle.combine_recv_start_row,
        handle.combine_recv_fold_count,
        handle.pool_row_to_route,
        handle.pool_row_to_recv_token,
        handle.recv_token_to_pool_rows,
        l2_tokens,
        symm.get_sym_buffer(),
        stream=torch.cuda.current_stream(),
    )


@functools.lru_cache(maxsize=None)
def _compile_reduce_only(
    hidden, num_max_routes, num_reduce_cu, num_topk, num_experts, num_experts_per_rank, rank
):
    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def reduce_only_k(
        OUTPUT: fx.Tensor,
        COMBINE_TOKENS: fx.Tensor,
        REDUCE_FLAGS: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
    ):
        block_index, _b, _c = fx.block_idx
        # exact size: the reduce points non-primary routes past num_records to read 0
        combine_token_res = create_buffer_resource(
            COMBINE_TOKENS, max_size=False, num_records_bytes=num_max_routes * hidden * 2
        )
        topk_reduce_bf16_tile(
            thread_index=fx.thread_idx.x,
            reduce_block_idx=block_index,
            num_reduce_warps=num_reduce_cu * _NUM_WARPS,
            num_topk=num_topk,
            hidden=hidden,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_max_routes=num_max_routes,
            rank=rank,
            combine_token_res=combine_token_res,
            output_res=create_buffer_resource(OUTPUT, max_size=True),
            topk_indices_res=create_buffer_resource(TOPK_INDICES, max_size=True),
            num_tokens_res=create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True),
            reduce_flag_base=extract_base_index(REDUCE_FLAGS, address_space=1),
            reduce_bank=fx.Int32(0),
            epoch=fx.Int64(0),
        )

    @flyc.jit
    def launch(OUTPUT, COMBINE_TOKENS, REDUCE_FLAGS, TOPK_INDICES, NUM_TOKENS_PER_RANK, stream: fx.Stream):
        reduce_only_k(OUTPUT, COMBINE_TOKENS, REDUCE_FLAGS, TOPK_INDICES, NUM_TOKENS_PER_RANK).launch(
            grid=(num_reduce_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch


def topk_reduce_only(output, reduce_flags, topk_idx, *, num_experts_per_rank, rank, num_reduce_cu=32):
    """Sum each token's primary-route combine rows; ``reduce_flags`` is all zero, so no wait blocks."""
    symm = get_symm_buffer_for_mega_moe()
    launch = _compile_reduce_only(
        output.size(1),
        int(symm.num_max_routes),
        num_reduce_cu,
        int(symm.num_topk),
        int(symm.num_experts),
        num_experts_per_rank,
        rank,
    )
    launch(
        output.view(-1),
        symm.combine_token_buffer.contiguous().view(-1),
        reduce_flags,
        topk_idx.to(torch.int64).contiguous().view(-1),
        symm.num_tokens_per_rank,
        stream=torch.cuda.current_stream(),
    )
    return output


###############################################################################
# Timing, accuracy and routing helpers
###############################################################################


def bench(fn, *, warmup=5, iters=20):
    """Mean ms per call (CUDA events), after warmup and one 256 MB L2 flush; the first call is dropped."""
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    for _ in range(warmup):
        fn()
    cache.zero_()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(start, end):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return float(np.mean([s.elapsed_time(e) for s, e in zip(start, end)][1:]))


def _over_ranks(value, op):
    t = torch.tensor([float(value)], device="cuda")
    dist.all_reduce(t, op=op)
    return t.item()


def timed(fn, iters):
    """Slowest rank's mean ms of ``fn``; every rank starts together."""
    torch.cuda.synchronize()
    dist.barrier()
    ms = bench(fn, iters=iters)
    torch.cuda.synchronize()
    dist.barrier()
    return _over_ranks(ms, dist.ReduceOp.MAX)


def worst_check(out, ref, *, cos_thresh=0.99, rel_thresh=0.05):
    """(cos, rel_rmse, ok) of ``out`` against ``ref``, the worst over ranks."""
    o, r = out.flatten(), ref.flatten()
    # chunked fp32 products, fp64 sums: a wgrad is ~2^31 elements, past torch.dot's int32 numel
    dot = oo = rr = dd = 0.0
    for i in range(0, o.numel(), 1 << 28):
        oc, rc = o[i : i + (1 << 28)].float(), r[i : i + (1 << 28)].float()
        dot += float((oc * rc).sum(dtype=torch.float64))
        oo += float((oc * oc).sum(dtype=torch.float64))
        rr += float((rc * rc).sum(dtype=torch.float64))
        dd += float(((oc - rc) ** 2).sum(dtype=torch.float64))
    cos = dot / (math.sqrt(oo) * math.sqrt(rr) + 1e-12)
    rel = math.sqrt(dd) / (math.sqrt(rr) + 1e-12)
    t = torch.tensor([cos, -rel, float(cos >= cos_thresh and rel <= rel_thresh)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t[0].item(), -t[1].item(), bool(t[2].item())


def generate_routing(num_tokens, num_topk, num_experts, num_experts_per_rank, skew):
    """(topk_idx [T, K] int64, topk_weights [T, K] f32) from N(0, 1) logits, rank 0's experts raised by skew."""
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    logits[:, :num_experts_per_rank] += skew
    weights, idx = logits.softmax(-1).topk(num_topk, dim=-1)
    return idx.to(torch.int64), weights.float()


def dense_ms(M, N, K, iters):
    """The grouped NT kernel on one group of the same M x N x K: the ceiling the grouped GEMM is held to."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 8
    b = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16) / 8
    offs = torch.tensor([0, M], device="cuda", dtype=torch.int64)
    return timed(lambda: grouped_gemm_bf16_nt_flydsl_kernel(a, b, offs), iters)


def _per_expert_wgrad(lhs, rhs, offs, num_experts_per_rank):
    """ref[e] = lhs[e].T @ rhs[e] over each expert's pool rows, fp32 accumulate."""
    ref = torch.zeros(num_experts_per_rank, lhs.shape[1], rhs.shape[1], device="cuda", dtype=torch.bfloat16)
    for e in range(num_experts_per_rank):
        s, t = int(offs[e]), int(offs[e + 1])
        if t > s:
            ref[e] = (lhs[s:t].T.float() @ rhs[s:t].float()).to(torch.bfloat16)
    return ref


###############################################################################
# One model case
###############################################################################


def _bench_case(group, args, case):
    rank, world = group.rank(), group.size()
    T, H, I, K = args.num_tokens, case["hidden"], case["inter"], case["num_topk"]
    E = args.num_experts or case["num_experts"]
    EPR = E // world
    iters = args.iters
    torch.manual_seed(1234 + rank)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)

    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5)
    w2 = randn(EPR, H, I, scale=2 / I**0.5)
    x, dy = randn(T, H), randn(T, H)
    idx, tw = generate_routing(T, K, E, EPR, args.skew)

    # the production step's tensors, stage by stage
    handle, dispatch_weights = run_dispatch_prologue(x, w1, group, idx, tw)
    symm = get_symm_buffer_for_mega_moe()
    real_tiles = int(handle.num_tile_blocks[0].item())
    M_eff = real_tiles * BLOCK_M
    offs = torch.zeros(EPR + 1, dtype=torch.int64, device="cuda")
    offs[1:] = (
        torch.bincount(handle.tile_to_expert[:real_tiles].long(), minlength=EPR)[:EPR] * BLOCK_M
    ).cumsum(0)
    recv_idx = handle.pool_row_to_recv_token[:M_eff].long()
    real = recv_idx < int(symm.num_max_recv_tokens)
    prefix, tok_counts = handle.num_tokens_per_expert_prefix, handle.num_tokens_per_expert
    token_buffer = symm.dispatch_token_buffer

    def gather_pool():
        # pool rows past the real ones point at recv index num_max_recv_tokens: the zero pad row
        return F.pad(token_buffer, (0, 0, 0, 1))[recv_idx].contiguous()

    def sync():
        torch.cuda.synchronize()
        dist.barrier()

    l1_out = dispatch_gemm_nt_gfx1250(x, w1, handle)
    sync()
    pool_x = gather_pool()
    act = swiglu_flydsl_kernel(
        l1_out,
        num_tile_blocks=handle.num_tile_blocks,
        scale=dispatch_weights,
        tile_to_expert=handle.tile_to_expert,
    )
    grad_act = dispatch_gemm_nt_gfx1250(dy, w2, handle, b_nn=True)
    sync()
    pool_dy = gather_pool()
    grad_l1, grad_gate, act_w = swiglu_backward_flydsl_kernel(
        grad_act,
        l1_out,
        scale=dispatch_weights,
        return_gate=True,
        return_act_w=True,
        num_tile_blocks=handle.num_tile_blocks,
        tile_to_expert=handle.tile_to_expert,
    )
    sync()

    def remote_rows(counts, dst):
        return int(counts[dst != rank].sum().item())

    disp_bytes = remote_rows(handle.expert_send_count, handle.expert_send_dst_rank) * H * 2
    comb_bytes = remote_rows(handle.combine_recv_fold_count, handle.combine_recv_dst_rank) * H * 2
    x_copy_bytes = T * H * 2 * (world - 1)
    l2_rows = torch.zeros(handle.pool_row_to_route.numel(), H, device="cuda", dtype=torch.bfloat16)
    reduce_flags = torch.zeros(int(symm.num_max_routes), dtype=torch.int64, device="cuda")
    x_redispatch = get_x_redispatch(symm)

    def combine_ref(gemm_rows, grad_gate_=None):
        """Unfused reference: GEMM rows -> combine push -> top-k reduce."""
        l2_rows[:M_eff] = gemm_rows
        sync()
        combine_only(handle, l2_rows, grad_gate_)
        sync()
        out = torch.empty(int(symm.num_max_tokens_per_rank), H, device="cuda", dtype=torch.bfloat16)
        topk_reduce_only(out, reduce_flags, idx, num_experts_per_rank=EPR, rank=rank)
        sync()
        return out[:T].clone()

    flops_rows = _over_ranks(M_eff, dist.ReduceOp.SUM) / world
    stages = []

    def record(name, n, k, gemm, comm, fused, check, comm_bytes=0, dense_dims=None):
        stages.append(
            dict(
                stage=name,
                flops=2.0 * flops_rows * n * k,
                dense=dense_ms(*(dense_dims or (M_eff, n, k)), iters),
                gemm=gemm,
                comm=comm,
                fused=fused,
                check=check,
                comm_bytes=comm_bytes,
            )
        )

    # 1. L1 fwd: dispatch x + pool @ w1.T (NT, N = 2I, K = H)
    ref = grouped_gemm_bf16_nt_flydsl_kernel(pool_x, w1, offs)
    got = dispatch_gemm_nt_gfx1250(x, w1, handle)[:M_eff]
    record(
        "L1 fwd dispatch_gemm (NT)",
        2 * I,
        H,
        timed(lambda: grouped_gemm_bf16_nt_flydsl_kernel(pool_x, w1, offs), iters),
        timed(lambda: dispatch_only_gfx1250(x, handle, num_dispatch_cu=_DISPATCH_CU), iters),
        timed(lambda: dispatch_gemm_nt_gfx1250(x, w1, handle), iters),
        worst_check(got[real], ref[real]),
        disp_bytes,
    )

    # 2. L2 fwd: act @ w2.T + combine + top-k reduce (NT, N = H, K = I)
    ref = combine_ref(grouped_gemm_bf16_nt_flydsl_kernel(act[:M_eff].contiguous(), w2, offs))
    got = gemm_combine_nt_gfx1250(act, w2, handle, idx)
    record(
        "L2 fwd gemm_combine (NT)",
        H,
        I,
        timed(lambda: grouped_gemm_bf16_nt_flydsl_kernel(act[:M_eff], w2, offs), iters),
        timed(lambda: combine_only(handle, l2_rows), iters),
        timed(lambda: gemm_combine_nt_gfx1250(act, w2, handle, idx), iters),
        worst_check(got, ref),
        comb_bytes,
    )

    # 3. L2 dgrad: dispatch dy + pool @ w2 (NN, N = I, K = H); leaves dy in the token buffer for dW2
    ref = grouped_gemm_bf16_nn_flydsl_kernel(pool_dy, w2, offs)
    t_gemm = timed(lambda: grouped_gemm_bf16_nn_flydsl_kernel(pool_dy, w2, offs), iters)
    t_comm = timed(lambda: dispatch_only_gfx1250(dy, handle, num_dispatch_cu=_DISPATCH_CU), iters)
    t_fused = timed(lambda: dispatch_gemm_nt_gfx1250(dy, w2, handle, b_nn=True), iters)
    got = dispatch_gemm_nt_gfx1250(dy, w2, handle, b_nn=True)[:M_eff]
    record(
        "L2 dgrad dispatch_gemm (NN)",
        I,
        H,
        t_gemm,
        t_comm,
        t_fused,
        worst_check(got[real], ref[real]),
        disp_bytes,
    )

    # 4. dW2[e] = dy_pool[e].T @ act_w[e] (variable-K over the dispatched dy, read in place)
    def dw2():
        return grouped_gemm_bf16_variable_k_flydsl_kernel(
            token_buffer, act_w, prefix, masked_k=tok_counts, a_index=handle.pool_row_to_recv_token
        )

    t_gemm = timed(dw2, iters)
    ref = _per_expert_wgrad(pool_dy * real[:, None], act_w, offs, EPR)
    record("dW2 variable-K", H, I, t_gemm, 0.0, t_gemm, worst_check(dw2(), ref), dense_dims=(H, I, M_eff))

    # 5. L1 dgrad: grad_l1 @ w1 + combine with the gate grad + reduce (NN, N = H, K = 2I); x for dW1
    #    re-dispatched by peer copies on a side stream meanwhile, as in production
    def l1_dgrad_fused():
        x_redispatch.push(x)
        out = gemm_combine_nt_gfx1250(grad_l1, w1, handle, idx, grad_gate=grad_gate, b_nn=True)
        x_redispatch.wait()
        return out

    ref = combine_ref(grouped_gemm_bf16_nn_flydsl_kernel(grad_l1[:M_eff].contiguous(), w1, offs), grad_gate)
    got = l1_dgrad_fused()[0]
    t_combine = timed(lambda: combine_only(handle, l2_rows, grad_gate), iters)
    t_x_copy = timed(lambda: (x_redispatch.push(x), x_redispatch.wait()), iters)
    record(
        "L1 dgrad gemm_combine (NN) + x re-dispatch",
        H,
        2 * I,
        timed(lambda: grouped_gemm_bf16_nn_flydsl_kernel(grad_l1[:M_eff], w1, offs), iters),
        t_combine + t_x_copy,
        timed(l1_dgrad_fused, iters),
        worst_check(got, ref),
        comb_bytes + x_copy_bytes,
    )

    # 6. dW1[e] = grad_l1[e].T @ x_pool[e] (variable-K over the re-dispatched x)
    x_recv = x_redispatch.wait()

    def dw1():
        return grouped_gemm_bf16_variable_k_flydsl_kernel(
            grad_l1, x_recv, prefix, masked_k=tok_counts, b_index=handle.pool_row_to_recv_token
        )

    t_gemm = timed(dw1, iters)
    ref = _per_expert_wgrad(grad_l1[:M_eff] * real[:, None], pool_x, offs, EPR)
    record(
        "dW1 variable-K", 2 * I, H, t_gemm, 0.0, t_gemm, worst_check(dw1(), ref), dense_dims=(2 * I, H, M_eff)
    )
    del ref, got

    # the whole step through the op, for what lies outside the six GEMM stages
    xg, twg = x.clone().requires_grad_(), tw.clone().requires_grad_()
    w1g, w2g = w1.clone().requires_grad_(), w2.clone().requires_grad_()

    def step():
        for t in (xg, twg, w1g, w2g):
            t.grad = None
        fused_mega_moe(group, xg, idx, twg, w1g, w2g).backward(dy)

    t_step = timed(step, iters)
    step_flops = 3 * 2.0 * T * K * 3 * H * I

    rows = []
    head = dict(model=case["Case"], EP=world, tokens=T, experts_per_rank=EPR, topk=K, skew=args.skew)
    for s in stages:
        hidden = s["gemm"] + s["comm"] - s["fused"]
        rows.append(
            dict(
                head,
                stage=s["stage"],
                dense_ms=s["dense"],
                gemm_only_ms=s["gemm"],
                gemm_only_tflops=s["flops"] / (s["gemm"] * 1e-3) / 1e12,
                grouped_vs_dense_pct=100 * s["dense"] / s["gemm"],
                comm_only_ms=s["comm"],
                comm_gbps=s["comm_bytes"] / (s["comm"] * 1e-3) / 1e9 if s["comm"] > 0 else float("nan"),
                fused_ms=s["fused"],
                fused_tflops=s["flops"] / (s["fused"] * 1e-3) / 1e12,
                hidden_ms=hidden,
                hidden_of_comm_pct=100 * hidden / s["comm"] if s["comm"] > 0 else float("nan"),
                roofline_pct=100 * max(s["gemm"], s["comm"]) / s["fused"] if s["comm"] > 0 else float("nan"),
                check=f"{'PASS' if s['check'][2] else 'FAIL'} cos={s['check'][0]:.5f} rel={s['check'][1]:.1e}",
            )
        )
    fused_sum = sum(s["fused"] for s in stages)
    rows.append(
        dict(
            head,
            stage="fused_mega_moe fwd+bwd step",
            gemm_only_ms=sum(s["gemm"] for s in stages),
            comm_only_ms=sum(s["comm"] for s in stages),
            fused_ms=t_step,
            fused_tflops=step_flops / (t_step * 1e-3) / 1e12,
            check=f"6 stages {fused_sum:.3f} ms + rest {t_step - fused_sum:.3f} ms",
        )
    )
    if rank == 0:
        print(
            f"\n[{case['Case']}] EP{world} tokens/rank={T} H={H} I={I} experts/rank={EPR} topk={K} "
            f"skew={args.skew} pool tiles={real_tiles} (rank 0), x re-dispatch alone {t_x_copy:.3f} ms",
            flush=True,
        )
    return rows


###############################################################################
# Driver: one spawn per model case
###############################################################################


def _supported(case, args, world):
    E = args.num_experts or case["num_experts"]
    if E % world:
        return f"{E} experts do not split over {world} ranks"
    if case["num_topk"] > WAVE:
        return f"top-{case['num_topk']} exceeds the combine's {WAVE}-lane wave"
    if any(k % 64 or k < _MIN_K for k in (case["hidden"], case["inter"])):
        return "hidden / intermediate size not a multiple of 64 of at least 128"
    return None


def _worker(local_rank, world, args, case, port, queue):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{port}",
        world_size=world,
        rank=local_rank,
    )
    group = dist.group.WORLD  # the symm-mem rendezvous needs WORLD (a fresh subgroup fails "send fd")
    try:
        rows = _bench_case(group, args, case)
        if local_rank == 0:
            queue.put(rows)
    finally:
        try:
            get_symm_buffer_for_mega_moe().destroy()
        except RuntimeError:
            pass  # none was ever built
        dist.destroy_process_group()


def main(args):
    if not is_gfx1250():
        sys.exit("bench_mega_moe_gfx1250.py runs the gfx1250 kernels; use bench_mega_moe.py on gfx950")
    world = args.num_processes
    _, gpu_name = get_platform_info()
    queue = mp.get_context("spawn").SimpleQueue()
    base_port = int(os.getenv("MASTER_PORT", "29531"))
    rows = []
    for i, case in enumerate(gen_moe_test_cases(args.models)):
        reason = _supported(case, args, world)
        if reason:
            print(f"[skip] {case['Case']}: {reason}")
            continue
        try:
            mp.spawn(_worker, args=(world, args, case, base_port + i, queue), nprocs=world)
        except mp.ProcessRaisedException as e:
            print(f"[skip] {case['Case']}: {e}")
            continue
        rows += queue.get()
    if not rows:
        return
    df = pd.DataFrame(rows)
    shown = [
        "model",
        "stage",
        "gemm_only_ms",
        "comm_only_ms",
        "fused_ms",
        "fused_tflops",
        "hidden_ms",
        "roofline_pct",
        "grouped_vs_dense_pct",
        "check",
    ]
    print("\nFinal Results:")
    print(tabulate(df[shown], headers="keys", tablefmt="grid", showindex=False, floatfmt=".3f"))
    out_file = args.output or f"mega_moe_gfx1250_{datetime.now():%Y%m%d}_{gpu_name}.csv"
    df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")


def _build_parser():
    parser = argparse.ArgumentParser(description="Benchmark the fused BF16 Mega MoE kernels on gfx1250")
    parser.add_argument("--num-processes", type=int, default=4, help="EP size, one GPU per process")
    parser.add_argument("--num-tokens", type=int, default=8192, help="tokens per rank")
    # H / I / K come from each MoE model case (config.gen_moe_test_cases); E too unless given here
    parser.add_argument("--num-experts", type=int, default=None, help="override the model's routed experts")
    parser.add_argument("--skew", type=float, default=0.0, help="logit bias of rank 0's experts (0 = random)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=["DeepSeek-V3"])
    parser.add_argument("--output", "-o", type=str, default=None)
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
