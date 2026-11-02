###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""DeepEP intranode notify / cached notify (FlyDSL), shared by the V1 and V2 dispatches.

``notify_dispatch`` exchanges how many tokens each rank sends to every other rank and to each of
their experts, and returns the rank / channel prefix matrices and this rank's receive counts;
``cached_notify_dispatch`` restores the rank prefix matrix of an earlier dispatch. Both use the start
of every rank's heap, in int32 words with ``R`` ranks:

- ``[0, R * R)``: the rank prefix matrix (``full_rank_matrix`` fills every column, else only this
  rank's), which the dispatch that follows reads;
- ``[R * R, R * R + num_experts)``: the expert counts, during the notify only;
- ``[R * R, R * R + num_memset_int)``: cleared before the notify returns, for the dispatch that
  follows (V1 clears its channel queues there, V2 passes 0).

A dispatch's own heap regions start after this: V1's at ``R * R`` words, V2's at ``META_BYTES``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from primus_turbo.flydsl.deep_ep.common import (
    WARP_SIZE,
    _channel_task_range,
    _ld_sys,
    _load_lane_ptr,
    _load_ptr,
    _load_u8,
    _st_sys,
    barrier_block,
    ld_lds,
    lds_base,
    load_i32,
    st_lds,
)
from primus_turbo.flydsl.utils.buffer_ops import buffer_store, create_buffer_resource

NOTIFY_THREADS = 256
# butterfly-reduction lane offsets; a 32-lane offset on wave32 would lower to a wave64 permlane
_WARP_XOR_OFFSETS = tuple(WARP_SIZE >> i for i in range(1, WARP_SIZE.bit_length()))


def _make_notify_dispatch(
    num_ranks, num_experts, num_channels, num_memset_int, expert_alignment, full_rank_matrix
):
    """``full_rank_matrix`` fills every column of the rank prefix matrix (V2 needs where it sits in
    every rank's receive order); otherwise only this rank's column is valid, as in the HIP kernel."""
    num_experts_per_rank = num_experts // num_ranks

    @flyc.kernel(name="deep_ep_notify_dispatch_kernel", known_block_size=[NOTIFY_THREADS, 1, 1])
    def notify_dispatch_kernel(
        NUM_TOKENS_PER_RANK: fx.Tensor,
        NUM_TOKENS_PER_EXPERT: fx.Tensor,
        IS_TOKEN_IN_RANK: fx.Tensor,
        CHANNEL_PREFIX_MATRIX: fx.Tensor,
        RANK_PREFIX_MATRIX: fx.Tensor,
        RECV_COUNTERS: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        BARRIER_PTRS: fx.Tensor,
        num_tokens: fx.Int32,
        rank: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        block_id = fx.block_idx.x
        lane_id = thread_id % fx.Int32(WARP_SIZE)
        warp_id = thread_id // fx.Int32(WARP_SIZE)
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
        barrier_ptrs_res = create_buffer_resource(BARRIER_PTRS, max_size=True)
        lds = lds_base()

        if block_id == fx.Int32(0):
            barrier_block(barrier_ptrs_res, rank, num_ranks, True, "notify_dispatch/enter")

            # peer[rank, t] = tokens we send to t; peer's expert rows likewise
            if thread_id < fx.Int32(num_ranks):
                tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
                experts_res = create_buffer_resource(NUM_TOKENS_PER_EXPERT, max_size=True)
                peer = _load_lane_ptr(buffer_ptrs_res, thread_id)
                if fx.const_expr(full_rank_matrix):
                    for t in range_constexpr(num_ranks):
                        _st_sys(
                            peer, rank * fx.Int32(num_ranks) + fx.Int32(t), load_i32(tokens_res, fx.Int32(t))
                        )
                else:
                    _st_sys(
                        peer,
                        rank * fx.Int32(num_ranks) + thread_id,
                        load_i32(tokens_res, thread_id),
                    )
                for i in range(num_experts_per_rank):
                    count = load_i32(experts_res, thread_id * fx.Int32(num_experts_per_rank) + fx.Int32(i))
                    _st_sys(
                        peer,
                        fx.Int32(num_ranks * num_ranks) + rank * fx.Int32(num_experts_per_rank) + fx.Int32(i),
                        count,
                    )
            barrier_block(barrier_ptrs_res, rank, num_ranks, False, "notify_dispatch/gathered")

            local = _load_ptr(buffer_ptrs_res, rank)
            counters_res = create_buffer_resource(RECV_COUNTERS, max_size=True)
            # inclusive prefix over source ranks, one column per thread
            if thread_id < fx.Int32(num_ranks):
                running = _ld_sys(local, thread_id)
                for i in range(1, num_ranks):
                    running = running + _ld_sys(local, fx.Int32(i * num_ranks) + thread_id)
                    _st_sys(local, fx.Int32(i * num_ranks) + thread_id, running)
                if thread_id == rank:
                    buffer_store(running, counters_res, fx.Int32(0))
            expert = thread_id
            while expert < fx.Int32(num_experts_per_rank):
                total = fx.Int32(0)
                for i in range(num_ranks):
                    total = total + _ld_sys(
                        local, fx.Int32(num_ranks * num_ranks + i * num_experts_per_rank) + expert
                    )
                total = (
                    (total + fx.Int32(expert_alignment - 1))
                    // fx.Int32(expert_alignment)
                    * fx.Int32(expert_alignment)
                )
                buffer_store(total, counters_res, fx.Int32(1) + expert)
                expert = expert + fx.Int32(NOTIFY_THREADS)
            fx.gpu.barrier()

            rank_prefix_res = create_buffer_resource(RANK_PREFIX_MATRIX, max_size=True)
            i = thread_id
            while i < fx.Int32(num_ranks * num_ranks):
                buffer_store(_ld_sys(local, i), rank_prefix_res, i)
                i = i + fx.Int32(NOTIFY_THREADS)
            # clean the channel offsets / queue head and tail for the dispatch
            i = thread_id
            while i < fx.Int32(num_memset_int):
                _st_sys(local, fx.Int32(num_ranks * num_ranks) + i, fx.Int32(0))
                i = i + fx.Int32(NOTIFY_THREADS)
            barrier_block(barrier_ptrs_res, rank, num_ranks, False, "notify_dispatch/cleaned")
        else:
            # per-channel token counts to one destination, then the prefix over channels
            dst_rank = block_id - fx.Int32(1)
            in_rank_res = create_buffer_resource(IS_TOKEN_IN_RANK, max_size=True)
            channel = warp_id
            while channel < fx.Int32(num_channels):
                token_start, token_end = _channel_task_range(num_tokens, num_channels, channel)
                count = fx.Int32(0)
                token = token_start + lane_id
                while token < token_end:
                    count = count + _load_u8(in_rank_res, token * fx.Int32(num_ranks) + dst_rank)
                    token = token + fx.Int32(WARP_SIZE)
                for offset in _WARP_XOR_OFFSETS:
                    count = count + fx.shuffle_xor(count, offset, WARP_SIZE)
                if lane_id == fx.Int32(0):
                    st_lds(lds, channel, count)
                channel = channel + fx.Int32(NOTIFY_THREADS // WARP_SIZE)
            fx.gpu.barrier()
            if thread_id == fx.Int32(0):
                prefix_res = create_buffer_resource(CHANNEL_PREFIX_MATRIX, max_size=True)
                running = fx.Int32(0)
                for c in range(num_channels):
                    running = running + ld_lds(lds, c)
                    buffer_store(running, prefix_res, dst_rank * fx.Int32(num_channels) + fx.Int32(c))

    return notify_dispatch_kernel


@flyc.jit
def launch_notify_dispatch(
    num_tokens_per_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    channel_prefix_matrix,
    rank_prefix_matrix,
    recv_counters,
    buffer_ptrs,
    barrier_ptrs,
    num_tokens: fx.Int32,
    rank: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_channels: fx.Constexpr[int],
    num_memset_int: fx.Constexpr[int],
    expert_alignment: fx.Constexpr[int],
    full_rank_matrix: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel = _make_notify_dispatch(
        num_ranks, num_experts, num_channels, num_memset_int, expert_alignment, full_rank_matrix
    )
    kernel(
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        channel_prefix_matrix,
        rank_prefix_matrix,
        recv_counters,
        buffer_ptrs,
        barrier_ptrs,
        num_tokens,
        rank,
    ).launch(grid=(1 + num_ranks, 1, 1), block=(NOTIFY_THREADS, 1, 1), stream=stream, smem=num_channels * 4)


def _make_cached_notify_dispatch(num_ranks, num_memset_int):
    @flyc.kernel(name="deep_ep_cached_notify_dispatch_kernel", known_block_size=[NOTIFY_THREADS, 1, 1])
    def cached_notify_dispatch_kernel(
        RANK_PREFIX_MATRIX: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        BARRIER_PTRS: fx.Tensor,
        rank: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
        barrier_ptrs_res = create_buffer_resource(BARRIER_PTRS, max_size=True)
        barrier_block(barrier_ptrs_res, rank, num_ranks, True, "cached_notify_dispatch/enter")
        local = _load_ptr(buffer_ptrs_res, rank)
        rank_prefix_res = create_buffer_resource(RANK_PREFIX_MATRIX, max_size=True)
        i = thread_id
        while i < fx.Int32(num_ranks * num_ranks):
            _st_sys(local, i, load_i32(rank_prefix_res, i))
            i = i + fx.Int32(NOTIFY_THREADS)
        i = thread_id
        while i < fx.Int32(num_memset_int):
            _st_sys(local, fx.Int32(num_ranks * num_ranks) + i, fx.Int32(0))
            i = i + fx.Int32(NOTIFY_THREADS)
        barrier_block(barrier_ptrs_res, rank, num_ranks, False, "cached_notify_dispatch/cleaned")

    return cached_notify_dispatch_kernel


@flyc.jit
def launch_cached_notify_dispatch(
    rank_prefix_matrix,
    buffer_ptrs,
    barrier_ptrs,
    rank: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_memset_int: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_cached_notify_dispatch(num_ranks, num_memset_int)
    kernel(rank_prefix_matrix, buffer_ptrs, barrier_ptrs, rank).launch(
        grid=(1, 1, 1), block=(NOTIFY_THREADS, 1, 1), stream=stream
    )
