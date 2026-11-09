###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Ulysses GemmA2A: a dense GEMM whose output columns are head-sharded over the group, scattered
straight into the post-all-to-all layout (pull mode).

GEMM blocks compute ``C = A @ B`` (``A`` = this rank's tokens) into this rank's symmetric staging
bank and publish one flag per tile. Pull blocks of every rank wait for the tiles that cover their
own column range ``[rank * N_r, (rank + 1) * N_r)`` on each source rank, read them over XGMI and
write the rows into ordinary q / k / v (or dO) tensors at the source rank's sequence offset.

A rank-column block holds ``G`` groups of ``P = nq + nk + nv`` head-dim pieces: the q heads, then
the k and v heads of the group, which covers Megatron's interleaved QKV rows (``G = ng / n``,
``nq = hq / ng``, ``nk = nv = 1``), a rank-major concat (``G = 1``) and a q-only output (``nk = nv = 0``).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr

from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import (
    grouped_gemm_bf16_variable_k_tile,
)
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    NUM_WARPS,
    SYS_NT,
    VEC_BF16,
    VEC_I32,
    WARP,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
    spin_flag,
)
from primus_turbo.flydsl.ulysses.workspace import (
    FLAG_BYTES,
    FLAG_REGION_BYTES,
    FLAGS_PER_REGION,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
)
from primus_turbo.flydsl.utils.prims import (
    ceildiv,
    ld,
    read_clock,
    st,
)

# head pieces whose loads are issued before their stores
_PIECE_BATCH = 8
# variable-K wgrad tile and its LDS chunk stride
_WGRAD_BM = 256
_WGRAD_BN = 256
_WGRAD_LDS_CHUNK_STRIDE = 1152


def piece_dst(j, nq, nk, nv):
    """(tensor id, local head) of head piece ``j`` of a rank-column block (compile time)."""
    P = nq + nk + nv
    g, jj = divmod(j, P)
    if jj < nq:
        return 0, g * nq + jj
    if jj < nq + nk:
        return 1, g * nk + jj - nq
    return 2, g * nv + jj - nq - nk


@ASTRewriter.transform
def _pull_unit(
    thread_index,
    src,
    m_tile,
    src_heap,
    bank_offset,
    out_res,
    epoch,
    *,
    M,
    M_pad,
    N,
    N_r,
    BM,
    BN,
    head_dim,
    G,
    nq,
    nk,
    nv,
    seq_local,
    is_bshd,
    rank,
    num_ranks,
    flag_region,
    trace_base=None,
    trace_idx=None,
):
    """Copy rows [m_tile*BM, +BM) of ``src``'s tile columns of this rank into the q/k/v outputs."""
    flag_base = src_heap + fx.Int64(flag_region * FLAG_REGION_BYTES)
    n_tiles = ceildiv(N, BN)
    first_n = rank * N_r // BN
    last_n = ((rank + 1) * N_r - 1) // BN
    if thread_index == fx.Int32(0):
        for tile_n in range_constexpr(first_n, last_n + 1):
            spin_flag(
                flag_base, m_tile * fx.Int32(n_tiles) + fx.Int32(tile_n), epoch, rank, src, "gemm_a2a pull"
            )
        if const_expr(trace_base is not None):
            st(trace_base, trace_idx, read_clock())
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()

    stage_res = create_buffer_resource_from_addr(src_heap + bank_offset, num_records_bytes=M_pad * N * 2)
    lanes_per_row = head_dim // VEC_BF16
    rows_per_warp = WARP // lanes_per_row
    rows_per_iter = NUM_WARPS * rows_per_warp
    lane = thread_index % fx.Int32(WARP)
    warp = thread_index // fx.Int32(WARP)
    lane_row = lane // fx.Int32(lanes_per_row)
    lane_word = (lane % fx.Int32(lanes_per_row)) * fx.Int32(VEC_I32)
    num_pieces = G * (nq + nk + nv)
    heads = (G * nq, G * nk, G * nv)
    for it in range(BM // rows_per_iter):
        t = m_tile * fx.Int32(BM) + fx.Int32(it * rows_per_iter) + warp * fx.Int32(rows_per_warp) + lane_row
        valid = t < fx.Int32(M)
        if const_expr(is_bshd):
            bi = t // fx.Int32(seq_local)
            dst_row = t + bi * fx.Int32(seq_local * (num_ranks - 1)) + src * fx.Int32(seq_local)
        else:
            dst_row = src * fx.Int32(M) + t
        src_word = t * fx.Int32(N // 2) + fx.Int32(rank * N_r // 2) + lane_word
        for first in range_constexpr(0, num_pieces, _PIECE_BATCH):
            batch = list(range(first, min(first + _PIECE_BATCH, num_pieces)))
            vals = [
                buffer_load(
                    stage_res,
                    src_word + fx.Int32(j * head_dim // 2),
                    vec_width=VEC_I32,
                    dtype=fx.T.i32,
                    cache_modifier=SYS_NT,
                )
                for j in batch
            ]
            for j, val in zip(batch, vals):
                tensor_id, head = piece_dst(j, nq, nk, nv)
                dst_word = (dst_row * fx.Int32(heads[tensor_id]) + fx.Int32(head)) * fx.Int32(
                    head_dim // 2
                ) + lane_word
                buffer_store(val, out_res[tensor_id], dst_word, mask=valid)


@ASTRewriter.transform
def _pull_loop(
    thread_index,
    state_base,
    claim_slot,
    peers_base,
    bank_offset,
    out_res,
    epoch,
    trace_base,
    trace_units_at,
    **consts,
):
    """Claim units in readiness order (m-major, local source first) until none are left."""
    num_ranks = consts["num_ranks"]
    rank = consts["rank"]
    m_tiles = ceildiv(consts["M"], consts["BM"])
    num_units = num_ranks * m_tiles
    unit = claim_unit(thread_index, state_base, claim_slot)
    while unit < fx.Int32(num_units):
        m_tile = unit // fx.Int32(num_ranks)
        src = (fx.Int32(rank) + unit % fx.Int32(num_ranks)) % fx.Int32(num_ranks)
        src_heap = ld(peers_base, src, dtype=fx.T.i64)
        if const_expr(trace_base is not None):
            trace_idx = fx.Int32(trace_units_at) + unit * fx.Int32(3)
            if thread_index == fx.Int32(0):
                st(trace_base, trace_idx, read_clock())
            _pull_unit(
                thread_index,
                src,
                m_tile,
                src_heap,
                bank_offset,
                out_res,
                epoch,
                trace_base=trace_base,
                trace_idx=trace_idx + 1,
                **consts,
            )
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            if thread_index == fx.Int32(0):
                st(trace_base, trace_idx + 2, read_clock())
        else:
            _pull_unit(thread_index, src, m_tile, src_heap, bank_offset, out_res, epoch, **consts)
        # the next unit's flag wait must not overtake this unit's reads of the staging bank
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        unit = claim_unit(thread_index, state_base, claim_slot)


@functools.lru_cache(maxsize=64)
def _make_kernel(
    M,
    K,
    N,
    BM,
    BN,
    layout,
    rank,
    num_ranks,
    num_pull_cu,
    head_dim,
    G,
    nq,
    nk,
    nv,
    seq_local,
    is_bshd,
    flag_region,
    nt_vmcnt,
    bank_bytes,
    num_lead=0,
    num_tail_pull=0,
    wgrad_n=0,
    wgrad_at=0,
    wgrad_bm=256,
    trace=False,
):
    assert N % num_ranks == 0
    N_r = N // num_ranks
    assert N_r == G * (nq + nk + nv) * head_dim, "rank-column block must hold G * (nq + nk + nv) heads"
    assert head_dim % VEC_BF16 == 0 and WARP % (head_dim // VEC_BF16) == 0
    m_tiles = ceildiv(M, BM)
    M_pad = m_tiles * BM
    n_tiles = ceildiv(N, BN)
    num_gemm_tiles = m_tiles * n_tiles
    assert num_gemm_tiles <= FLAGS_PER_REGION, "too many tiles for one flag region"
    assert M_pad * N < 2**31 and M * K < 2**31 and K * N < 2**31, "i32 element offsets overflow"
    num_lead = min(num_lead, num_gemm_tiles)
    # wgrad_n > 0: blocks dispatched after GEMM tile ``wgrad_at`` also compute DW = WA^T @ WB
    # ([K, wgrad_n], over M), one variable-K tile each; they wait for nothing
    wm_tiles = K // wgrad_bm if wgrad_n else 0
    num_wtiles = wm_tiles * (wgrad_n // _WGRAD_BN)
    assert not wgrad_n or (K % wgrad_bm == 0 and wgrad_n % _WGRAD_BN == 0)
    wgrad_at = min(max(wgrad_at, num_lead), num_gemm_tiles)
    # tail pull blocks follow every GEMM block in dispatch order, so none of them can hold a CU a GEMM
    # tile they wait on still needs; they take the CUs the last GEMM wave leaves idle
    grid = num_gemm_tiles + num_pull_cu + num_wtiles + num_tail_pull
    gemm_tile = functools.partial(gemm_bf16_tile, layout, pair_n=True)
    # one extra i32: the block-wide broadcast of a claimed pull unit
    SharedStorage = (
        _make_shared_storage(
            max(BM, wgrad_bm), BN, chunk_stride=_WGRAD_LDS_CHUNK_STRIDE, num_lds_row_idx_entries=1
        )
        if num_wtiles
        else _make_shared_storage(BM, BN, num_lds_row_idx_entries=1)
    )

    @flyc.kernel(name="ulysses_gemm_a2a_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        OUT_Q: fx.Tensor,
        OUT_K: fx.Tensor,
        OUT_V: fx.Tensor,
        WA: fx.Tensor,
        WB: fx.Tensor,
        DW: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
        TRACE: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.lds_row_idx.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        # debug timeline (s_memrealtime): [start, end] per GEMM tile, then [claim, ready, end] per pull unit
        trace_base = i64_base(TRACE) if trace else None
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = ld(peers_base, fx.Int32(rank), dtype=fx.T.i64)

        out_res = [
            create_buffer_resource(t, num_records_bytes=num_ranks * M * h * G * head_dim * 2)
            for t, h in zip((OUT_Q, OUT_K, OUT_V), (nq, nk, nv))
        ]

        consts = dict(
            M=M,
            M_pad=M_pad,
            N=N,
            N_r=N_r,
            BM=BM,
            BN=BN,
            head_dim=head_dim,
            G=G,
            nq=nq,
            nk=nk,
            nv=nv,
            seq_local=seq_local,
            is_bshd=is_bshd,
            rank=rank,
            num_ranks=num_ranks,
            flag_region=flag_region,
        )

        # dispatch order: lead GEMM | pull | GEMM up to wgrad_at | wgrad | rest of GEMM | tail pull
        w_first = wgrad_at + num_pull_cu
        is_pull = ((block_index >= fx.Int32(num_lead)) & (block_index < fx.Int32(num_lead + num_pull_cu))) | (
            block_index >= fx.Int32(num_gemm_tiles + num_pull_cu + num_wtiles)
        )
        is_wgrad = (block_index >= fx.Int32(w_first)) & (block_index < fx.Int32(w_first + num_wtiles))
        if const_expr(num_wtiles > 0):
            if is_wgrad:
                w = block_index - fx.Int32(w_first)
                grouped_gemm_bf16_variable_k_tile(
                    WA,
                    WB,
                    DW,
                    fx.Int32(0),
                    w // fx.Int32(wgrad_n // _WGRAD_BN),
                    w % fx.Int32(wgrad_n // _WGRAD_BN),
                    fx.Int32(0),
                    fx.Int32(M),
                    lds,
                    fx.Int32(K),
                    fx.Int32(wgrad_n),
                    G=1,
                    OUT_M=K,
                    OUT_N=wgrad_n,
                    BLOCK_M=wgrad_bm,
                    BLOCK_N=_WGRAD_BN,
                    lds_chunk_stride=_WGRAD_LDS_CHUNK_STRIDE,
                )
        if is_pull:
            _pull_loop(
                thread_index,
                state_base,
                claim_slot,
                peers_base,
                bank_offset,
                out_res,
                epoch,
                trace_base,
                2 * num_gemm_tiles,
                **consts,
            )
        is_gemm = (block_index < fx.Int32(num_lead)) | (
            (block_index >= fx.Int32(num_lead + num_pull_cu))
            & (block_index < fx.Int32(num_gemm_tiles + num_pull_cu + num_wtiles))
            & ((block_index < fx.Int32(w_first)) | (block_index >= fx.Int32(w_first + num_wtiles)))
        )
        if is_gemm:
            gemm_index = arith.select(
                block_index < fx.Int32(num_lead),
                block_index,
                arith.select(
                    block_index < fx.Int32(w_first),
                    block_index - fx.Int32(num_pull_cu),
                    block_index - fx.Int32(num_pull_cu + num_wtiles),
                ),
            )
            tile_m = gemm_index // fx.Int32(n_tiles)
            tile_n = gemm_index % fx.Int32(n_tiles)
            if const_expr(trace):
                if thread_index == fx.Int32(0):
                    st(trace_base, (tile_m * fx.Int32(n_tiles) + tile_n) * fx.Int32(2), read_clock())
            A_t = make_bf16_fp16_tile_tensor(i64_base(A), fx.Int64(0), M * K)
            B_t = make_bf16_fp16_tile_tensor(i64_base(B), fx.Int64(0), K * N)
            C_t = make_bf16_fp16_tile_tensor(my_base, bank_offset, M_pad * N)
            gemm_tile(
                A_t,
                B_t,
                C_t,
                fx.Int32(M_pad),
                fx.Int32(N),
                lds,
                tile_m,
                tile_n,
                K=K,
                BLOCK_M=BM,
                BLOCK_N=BN,
                out_fp16=False,
                nt_vmcnt=nt_vmcnt,
                n_tail=N % BN,
                c_cache_modifier=SYS_NT,
            )
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            # keep a separator: LLVM folds adjacent barriers, but two rendezvous are required
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            if thread_index == fx.Int32(0):
                flag_base = my_base + fx.Int64(flag_region * FLAG_REGION_BYTES)
                st(flag_base, tile_m * fx.Int32(n_tiles) + tile_n, epoch, scope="sys", order="release")
                if const_expr(trace):
                    st(trace_base, (tile_m * fx.Int32(n_tiles) + tile_n) * fx.Int32(2) + 1, read_clock())

        exit_and_bump_epoch(thread_index, state_base, epoch, grid)

    return kernel, grid


@flyc.jit
def _launch(
    A,
    B,
    OUT_Q,
    OUT_K,
    OUT_V,
    WA,
    WB,
    DW,
    PEER_PTRS,
    STATE,
    TRACE,
    bank_bytes: fx.Constexpr[int],
    M: fx.Constexpr[int],
    K: fx.Constexpr[int],
    N: fx.Constexpr[int],
    BM: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    layout_code: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    num_pull_cu: fx.Constexpr[int],
    head_dim: fx.Constexpr[int],
    G: fx.Constexpr[int],
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    nv: fx.Constexpr[int],
    seq_local: fx.Constexpr[int],
    is_bshd: fx.Constexpr[bool],
    flag_region: fx.Constexpr[int],
    nt_vmcnt: fx.Constexpr[int],
    num_lead: fx.Constexpr[int],
    num_tail_pull: fx.Constexpr[int],
    wgrad_n: fx.Constexpr[int],
    wgrad_at: fx.Constexpr[int],
    wgrad_bm: fx.Constexpr[int],
    trace: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel, grid = _make_kernel(
        M,
        K,
        N,
        BM,
        BN,
        ("nt", "nn")[layout_code],
        rank,
        num_ranks,
        num_pull_cu,
        head_dim,
        G,
        nq,
        nk,
        nv,
        seq_local,
        is_bshd,
        flag_region,
        nt_vmcnt,
        bank_bytes,
        num_lead,
        num_tail_pull,
        wgrad_n,
        wgrad_at,
        wgrad_bm,
        trace,
    )
    kernel(
        A,
        B,
        OUT_Q,
        OUT_K,
        OUT_V,
        WA,
        WB,
        DW,
        PEER_PTRS,
        STATE,
        TRACE,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def gemm_a2a_bank_bytes(M, N, BM):
    return ceildiv(M, BM) * BM * N * 2


def ulysses_gemm_a2a(
    a,
    b,
    outs,
    workspace,
    *,
    layout,
    head_dim,
    groups,
    nq,
    nk,
    nv,
    seq_local,
    qkv_format,
    flag_region,
    BM=128,
    BN=256,
    num_pull_cu=16,
    num_lead=None,
    num_tail_pull=16,
    wgrad=None,
    wgrad_at=None,
    wgrad_bm=256,
    trace=None,
):
    """``a`` [M, K] this rank's tokens; ``b`` [N, K] (``nt``) or [K, N] (``nn``), columns rank-major.

    ``outs`` = (q, k, v) ordinary tensors ``[rows, heads, head_dim]`` with ``rows = n * M`` laid out as
    ``qkv_format`` (``sbhd``: [s, b], ``bshd``: [b, s]); a q-only output repeats one tensor.

    ``wgrad=(a, wb, dw)`` also computes ``dw = a^T @ wb`` (``wb`` [M, N_w], ``dw`` [K, N_w]) in the
    same launch, on ``wgrad_bm x 256`` tiles dispatched after GEMM tile ``wgrad_at`` (needs
    ``K % wgrad_bm == 0``, ``N_w % 256 == 0``).
    """
    M, K = a.shape
    N = b.shape[0] if layout == "nt" else b.shape[1]
    assert a.is_contiguous() and b.is_contiguous() and all(o.is_contiguous() for o in outs)
    if wgrad is not None:
        wa, wb, dw = wgrad
        assert wa is a and wb.is_contiguous() and wb.shape[0] == M
        assert dw.is_contiguous() and tuple(dw.shape) == (K, wb.shape[1])
    if num_lead is None:
        # pull blocks dispatch once the first GEMM wave retires: nothing is ready to copy before that
        num_lead = torch.cuda.get_device_properties(a.device).multi_processor_count
    if wgrad is None:
        wgrad_at = 0
    elif wgrad_at is None:
        # behind the first GEMM waves (the pulls need their tiles first); queued last, the wgrad tiles
        # would hold the CUs the tail pulls need and run unhidden. Swept flat over 1/16..1/4 at n = 4.
        # More wgrad tiles than CUs (GLM-4.7: 960, MiniMax-M2.5: 288) would delay the GEMM tiles
        # behind them by whole waves: those go last. Qwen3-30B (128) is 13 % slower that way.
        gemm_tiles = ceildiv(M, BM) * ceildiv(N, BN)
        num_wtiles = K // wgrad_bm * (wgrad[1].shape[1] // 256)
        many_wtiles = num_wtiles > torch.cuda.get_device_properties(a.device).multi_processor_count
        wgrad_at = gemm_tiles if many_wtiles else num_lead + gemm_tiles // 8
    run_launcher(
        _launch,
        a,
        b,
        outs[0],
        outs[1],
        outs[2],
        a,
        a if wgrad is None else wgrad[1],
        a if wgrad is None else wgrad[2],
        workspace.peer_ptrs,
        workspace.state,
        workspace.state if trace is None else trace,
        workspace.bank_bytes,
        M,
        K,
        N,
        BM,
        BN,
        ("nt", "nn").index(layout),
        workspace.rank,
        workspace.num_ranks,
        num_pull_cu,
        head_dim,
        groups,
        nq,
        nk,
        nv,
        seq_local,
        qkv_format == "bshd",
        flag_region,
        3,
        num_lead,
        num_tail_pull,
        0 if wgrad is None else wgrad[1].shape[1],
        wgrad_at,
        wgrad_bm,
        trace is not None,
        torch.cuda.current_stream(),
    )
