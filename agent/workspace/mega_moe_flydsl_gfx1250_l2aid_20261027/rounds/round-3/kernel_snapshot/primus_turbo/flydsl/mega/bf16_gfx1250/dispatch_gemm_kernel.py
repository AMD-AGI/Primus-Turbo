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

"""Mega MoE fused dispatch + L1 GEMM for gfx1250: ``out[p] = x[token(p)] @ w1[e(p)].T``.

One launch, two roles, as in the gfx950 ``dispatch_grouped_gemm_bf16_kernel``: the first
``num_dispatch_cu`` workgroups push this rank's tokens into every destination rank's dispatch
token buffer and bump that rank's per-expert flag; every later workgroup is one
``tile_m x tile_n`` tile of the pool, which waits for its expert's flag and then runs
``nt_gemm_tile`` with A gathered from the dispatch token buffer by pool row.

The role split is a branch around the comm code only. The GEMM tile is reached by every
workgroup in straight-line code and masked off with ``live`` where it has nothing to do (comm
workgroups, pool blocks past this call's tile count): a dynamic branch over the TDM pipeline
is what the grouped kernel's ``dead_tile_exit`` found to fault intermittently.

Every workgroup is ``m_warp * n_warp`` wave32 waves; the comm role runs the shared
``dispatch_bf16_tile`` on that many threads (64-lane copy warps).
"""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import Constexpr, T

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _as_ptr,
    _i32_table,
    _require_gfx1250,
)
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import DispatchHandle
from primus_turbo.flydsl.mega.bf16.ep_intranode import (
    dispatch_bf16_tile,
    spin_until_flag_reaches,
)
from primus_turbo.flydsl.mega.bf16.symm_buffer import (
    BLOCK_M,
    TOKEN_DTYPE,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_nt_tile import (
    EB,
    GFX1250_LDS_BYTES,
    NUM_BUFFERS,
    WAVE,
    aid_half_index,
    check_gather_rows,
    nt_gemm_tile,
    tile_geometry,
)
from primus_turbo.flydsl.utils.buffer_ops import create_buffer_resource_from_addr
from primus_turbo.flydsl.utils.prims import cast, ld, st


def _addr(p):
    return fx.Int64(fx.ptrtoint(p))


@flyc.jit
def _launch_dispatch_gemm_nt(
    arg_out: fx.Pointer,
    arg_x: fx.Pointer,
    arg_w: fx.Pointer,
    arg_token_buffer: fx.Pointer,
    arg_row_idx: fx.Pointer,
    arg_tile_to_expert: fx.Pointer,
    arg_num_tile_blocks: fx.Pointer,
    arg_send_dst_rank: fx.Pointer,
    arg_send_count: fx.Pointer,
    arg_send_offset: fx.Pointer,
    arg_dispatched_token_idx: fx.Pointer,
    arg_chunk_counter: fx.Pointer,
    arg_parity: fx.Pointer,
    arg_expected: fx.Pointer,
    sym_buffer,
    stream: fx.Stream,
    grid_size: fx.Int32,
    N: Constexpr[int],
    HIDDEN: Constexpr[int],
    N_BLOCKS_N: Constexpr[int],
    NUM_DISPATCH_CU: Constexpr[int],
    NUM_CHUNKS: Constexpr[int],
    NUM_RANKS: Constexpr[int],
    RANK: Constexpr[int],
    NUM_EXPERTS: Constexpr[int],
    NUM_MAX_TOKENS_PER_RANK: Constexpr[int],
    NUM_TOPK: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    group_m: Constexpr[int],
    idx_bits: Constexpr[int],
    DISPATCH_ONLY: Constexpr[bool],
    B_NN: Constexpr[bool],
    EXPERTS_PER_PUBLISH: Constexpr[int],
    AID_SPLIT: Constexpr[int],
):
    block = m_warp * n_warp * WAVE
    arena_bytes = tile_geometry(tile_m, tile_n, tile_k, B_NN)[3]
    if arena_bytes > GFX1250_LDS_BYTES:
        raise ValueError(f"LDS arena {arena_bytes} B exceeds {GFX1250_LDS_BYTES} B")
    num_experts_per_rank = NUM_EXPERTS // NUM_RANKS
    if DISPATCH_ONLY and num_experts_per_rank > block:
        raise ValueError(f"the dispatch-only wait gives each of {num_experts_per_rank} experts a lane of {block}")

    @flyc.kernel(known_block_size=[64, 1, 1])
    def epoch_bump(arg_parity: fx.Pointer, arg_expected: fx.Pointer):
        """Flip the parity and raise the new bank's expected flag value by one per sender rank."""
        if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
            parity_addr, expected_addr = _addr(arg_parity), _addr(arg_expected)
            new_parity = ld(parity_addr, fx.Int32(0), dtype=fx.T.i64) ^ fx.Int64(1)
            st(parity_addr, fx.Int32(0), new_parity)
            bank = cast(new_parity, fx.T.i32)
            st(expected_addr, bank, ld(expected_addr, bank, dtype=fx.T.i64) + fx.Int64(NUM_RANKS))

    @flyc.kernel(known_block_size=[block, 1, 1])
    def dispatch_gemm_nt(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_token_buffer: fx.Pointer,
        arg_row_idx: fx.Pointer,
        arg_tile_to_expert: fx.Pointer,
        arg_num_tile_blocks: fx.Pointer,
        arg_send_dst_rank: fx.Pointer,
        arg_send_count: fx.Pointer,
        arg_send_offset: fx.Pointer,
        arg_dispatched_token_idx: fx.Pointer,
        arg_chunk_counter: fx.Pointer,
        arg_parity: fx.Pointer,
        arg_expected: fx.Pointer,
        sym_buffer,
    ):
        arena = fx.SharedAllocator(static=False)
        arena.allocate(arena_bytes)
        base_ptr = arena.base_ptr
        tid = fx.thread_idx.x
        bid = fx.Int32(fx.block_idx.x)

        # Workspace-derived addresses are hoisted above the branch (the rewriter can't carry it).
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            NUM_RANKS,
            NUM_EXPERTS,
            NUM_MAX_TOKENS_PER_RANK,
            NUM_TOPK,
            HIDDEN,
            token_dtype=TOKEN_DTYPE,
        )
        dispatch_flag_base = workspace.get_dispatch_flag_ptr()
        num_pool_blocks = int(workspace.num_max_pool_blocks)
        a_rows = fx.Int32(int(workspace.num_max_recv_tokens))
        # this launch's epoch, already bumped: parity picks the flag bank, expected is the target
        parity = cast(ld(_addr(arg_parity), fx.Int32(0), dtype=fx.T.i64), fx.T.i32)
        expected = ld(_addr(arg_expected), parity, dtype=fx.T.i64)

        if bid < fx.Int32(NUM_CHUNKS * NUM_RANKS):
            dispatch_bf16_tile(
                sym_buffer,
                workspace,
                thread_index=tid,
                block_index=bid,
                hidden_size=HIDDEN,
                input_res=create_buffer_resource_from_addr(_addr(arg_x)),
                expert_send_dst_rank_res=create_buffer_resource_from_addr(_addr(arg_send_dst_rank)),
                expert_send_count_res=create_buffer_resource_from_addr(_addr(arg_send_count)),
                expert_send_offset_res=create_buffer_resource_from_addr(_addr(arg_send_offset)),
                dispatched_token_idx_res=create_buffer_resource_from_addr(_addr(arg_dispatched_token_idx)),
                dispatch_parity=parity,
                dispatch_chunk_counter_ptr=_addr(arg_chunk_counter),
                num_chunks=NUM_CHUNKS,
                num_ranks=NUM_RANKS,
                num_experts_per_rank=num_experts_per_rank,
                rank=RANK,
                block_threads=block,
                experts_per_publish=EXPERTS_PER_PUBLISH,
            )

        # ---- GEMM role: straight-line for every workgroup, masked by `live` ----
        pid = bid - fx.Int32(NUM_DISPATCH_CU)
        real_tiles = rocdl.readfirstlane(T.i32, _i32_table(arg_num_tile_blocks)[fx.Int32(0)])
        if const_expr(DISPATCH_ONLY):
            # No GEMM tile waits on the expert flags, so the one workgroup past the comm ones
            # holds the launch open until every sender has pushed every expert of this rank: the
            # caller reads the token buffer next, and a peer's rows may still be in flight.
            if (pid == fx.Int32(0)) & (fx.Int32(tid) < fx.Int32(num_experts_per_rank)):
                spin_until_flag_reaches(
                    dispatch_flag_base,
                    parity * fx.Int32(num_pool_blocks) + fx.Int32(tid),
                    expected,
                    "sys",
                    RANK,
                    "dispatch_only",
                )
            real_tiles = fx.Int32(0)
        live = (pid >= fx.Int32(0)) & (pid < real_tiles * fx.Int32(N_BLOCKS_N))
        pid = live.select(pid, fx.Int32(0))
        num_pid_in_group = fx.Int32(group_m * N_BLOCKS_N)
        first_pid_m = (pid // num_pid_in_group) * fx.Int32(group_m)
        remaining_m = real_tiles - first_pid_m
        group_size_m = (remaining_m < fx.Int32(group_m)).select(remaining_m, fx.Int32(group_m))
        group_size_m = (group_size_m > fx.Int32(0)).select(group_size_m, fx.Int32(1))
        pid_in_group = pid % num_pid_in_group
        if const_expr(AID_SPLIT > 0):
            # decoded m fastest, each AID's half of the super-tile is half its N blocks: only that
            # half of every expert's weight enters the AID's GL2, but all of the super-tile's pool
            # rows do. Mode 2 splits only when the pool has fewer blocks than this rank's weights.
            split = aid_half_index(pid_in_group, group_size_m * fx.Int32(N_BLOCKS_N), bid)
            if const_expr(AID_SPLIT == 2):
                split = (real_tiles < fx.Int32(num_experts_per_rank * N_BLOCKS_N)).select(split, pid_in_group)
            pid_in_group = split
        block_m = rocdl.readfirstlane(T.i32, live.select(first_pid_m + pid_in_group % group_size_m, fx.Int32(0)))
        block_n = rocdl.readfirstlane(T.i32, live.select(pid_in_group // group_size_m, fx.Int32(0)))
        expert = rocdl.readfirstlane(T.i32, _i32_table(arg_tile_to_expert)[block_m])
        expert = live.select(expert, fx.Int32(0))

        # every sender has pushed this expert (and, by push order, every lower one); a masked
        # tile waits for 0, which any flag already meets
        if fx.Int32(tid) == fx.Int32(0):
            spin_until_flag_reaches(
                dispatch_flag_base,
                parity * fx.Int32(num_pool_blocks) + expert,
                live.select(expected, fx.Int64(0)),
                "sys",
                RANK,
                "dispatch_gemm",
            )
        fx.gpu.barrier()

        blk_n = block_n * fx.Int32(tile_n)
        m_row = block_m * fx.Int32(tile_m)
        c_addr = _addr(fx.recast_iter(fx.Int8, arg_out)) + (
            fx.Int64(m_row) * fx.Int64(N) + fx.Int64(blk_n)
        ) * fx.Int64(EB)
        nt_gemm_tile(
            base_ptr,
            gA_base=fx.recast_iter(fx.Int8, arg_token_buffer),
            lda=HIDDEN,
            a_row0=m_row,
            a_rows=a_rows,
            row_idx_table=_i32_table(arg_row_idx),
            row_idx_base=m_row,
            gB_base=fx.recast_iter(fx.Int8, arg_w),
            ldb=N if B_NN else HIDDEN,
            b_slab=fx.Int64(expert) * fx.Int64(N * HIDDEN * EB),
            blk_n=blk_n,
            c_addr=c_addr,
            ldc=N,
            m_rows=fx.Int32(tile_m),
            n_valid=fx.Int32(N) - blk_n,
            i32_k=fx.Int32(HIDDEN),
            live=live,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            gather=True,
            idx_bits=idx_bits,
            b_nn=B_NN,
            i32_n=fx.Int32(N),
        )

    epoch_bump(arg_parity, arg_expected).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    dispatch_gemm_nt(
        arg_out,
        arg_x,
        arg_w,
        arg_token_buffer,
        arg_row_idx,
        arg_tile_to_expert,
        arg_num_tile_blocks,
        arg_send_dst_rank,
        arg_send_count,
        arg_send_offset,
        arg_dispatched_token_idx,
        arg_chunk_counter,
        arg_parity,
        arg_expected,
        sym_buffer,
    ).launch(grid=(grid_size, 1, 1), block=(block, 1, 1), stream=stream)


_launch_dispatch_gemm_nt.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}
_CONSTEXPR0 = 17


def dispatch_gemm_nt_gfx1250(
    x: torch.Tensor,
    w1: torch.Tensor,
    handle: DispatchHandle,
    *,
    num_dispatch_cu: int = 64,
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 0,
    m_warp: int = 4,
    n_warp: int = 2,
    group_m: int = 16,
    b_nn: bool = False,
) -> torch.Tensor:
    """Dispatch ``x`` by ``handle`` and return the L1 output over this rank's pool rows.

    ``handle`` comes from ``dispatch_prologue_flydsl_kernel`` on the active Mega symmetric
    buffer; the output is ``[pool rows, N]`` in pool order, padding rows undefined. ``w1`` is
    ``[experts, N, hidden]``, or ``[experts, hidden, N]`` with ``b_nn`` (``out = x @ w1[e]``).
    ``tile_k`` 0 is 128 when it divides hidden, else 64."""
    return _run(x, w1, handle, False, b_nn, num_dispatch_cu, tile_m, tile_n, tile_k, m_warp, n_warp, group_m)


def dispatch_only_gfx1250(
    x: torch.Tensor, handle: DispatchHandle, *, num_dispatch_cu: int | None = None
) -> torch.Tensor:
    """Dispatch ``x`` by ``handle`` and return the dispatch token buffer it lands in, in receive
    order (``handle.pool_row_to_recv_token`` maps pool rows to it). ``num_dispatch_cu`` defaults
    to every CU but the one the launch's waiting workgroup holds: no GEMM shares the launch."""
    if num_dispatch_cu is None:
        num_dispatch_cu = torch.cuda.get_device_properties(x.device).multi_processor_count - 1
    symm = get_symm_buffer_for_mega_moe()
    G = handle.expert_send_dst_rank.numel() // int(symm.num_ranks)
    # only the GEMM's shape is read: the launch holds the comm workgroups alone
    shape_only = x.new_empty(x.shape[1]).as_strided((G, 256, x.shape[1]), (0, 0, 1))
    _run(x, shape_only, handle, True, False, num_dispatch_cu, 256, 256, 0, 4, 2, 16)
    return symm.dispatch_token_buffer


def _run(x, w1, handle, dispatch_only, b_nn, num_dispatch_cu, tile_m, tile_n, tile_k, m_warp, n_warp, group_m):
    _require_gfx1250(x.device)
    symm = get_symm_buffer_for_mega_moe()
    G, N, K = w1.shape
    if b_nn:
        K, N = N, K
        assert N >= tile_n, "an NN block backs its B read off into the row, so N >= tile_n"
    tile_k = tile_k or (128 if K % 128 == 0 else 64)
    assert x.dtype == w1.dtype == torch.bfloat16 and x.is_contiguous()
    assert dispatch_only or w1.is_contiguous()
    assert K == x.shape[1] == int(symm.hidden), "w1 K, x hidden and the symmetric buffer's hidden must agree"
    assert tile_m == BLOCK_M, "a GEMM tile is one Mega pool block"
    assert K % tile_k == 0 and K // tile_k >= NUM_BUFFERS, "K must be a whole number of >= 2 k-tiles"
    num_ranks = int(symm.num_ranks)
    num_comm = handle.expert_send_dst_rank.numel()
    assert num_comm % num_ranks == 0 and num_dispatch_cu >= num_ranks, "every dst rank needs a comm block"
    assert num_comm // num_ranks == G, "w1 holds this rank's experts"
    a_rows = int(symm.num_max_recv_tokens)
    check_gather_rows(a_rows, K, tile_m, tile_k)
    # the prologue marks padding rows with a_rows, out of range for the A gather
    idx_bits = 16 if a_rows < 0xFFFF else 32
    out = x.new_empty((0, N) if dispatch_only else (handle.pool_row_to_route.numel(), N))
    n_blocks_n = (N + tile_n - 1) // tile_n
    args = (
        _as_ptr(out),
        _as_ptr(x),
        _as_ptr(w1),
        _as_ptr(symm.dispatch_token_buffer),
        _as_ptr(handle.pool_row_to_recv_token),
        _as_ptr(handle.tile_to_expert),
        _as_ptr(handle.num_tile_blocks),
        _as_ptr(handle.expert_send_dst_rank),
        _as_ptr(handle.expert_send_count),
        _as_ptr(handle.expert_send_offset),
        _as_ptr(handle.dispatched_token_idx),
        _as_ptr(symm._dispatch_chunk_counter),
        _as_ptr(symm._dispatch_parity),
        _as_ptr(symm._dispatch_expected),
        symm.get_sym_buffer(),
        torch.cuda.current_stream(),
        # the comm workgroups, then the GEMM tiles of this call's pool (the prologue's
        # num_tile_blocks), or dispatch-only's one waiter: masked worst-case tiles would drain last
        num_dispatch_cu + (1 if dispatch_only else handle.pool_row_to_route.numel() // BLOCK_M * n_blocks_n),
        N,
        K,
        n_blocks_n,
        num_dispatch_cu,
        num_dispatch_cu // num_ranks,
        num_ranks,
        int(symm.rank),
        int(symm.num_experts),
        int(symm.num_max_tokens_per_rank),
        int(symm.num_topk),
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        group_m,
        idx_bits,
        dispatch_only,
        b_nn,
        # GEMM tiles start on their expert's flag; with no tile to wait, a comm block publishes
        # once, sparing a remote store-retire round trip per expert
        G if dispatch_only else 1,
        int(os.environ.get("MEGA_AID_SPLIT_DG", "0")),
    )
    key = args[_CONSTEXPR0:]
    fn = _COMPILED.get(key)
    if fn is None:
        fn = flyc.compile(_launch_dispatch_gemm_nt, *args)
        if fn is None:
            return out
        _COMPILED[key] = fn
    fn(*args)
    return out
