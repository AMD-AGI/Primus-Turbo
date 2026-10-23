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

"""DeepEP intranode dispatch layout (FlyDSL): one thread per token."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, range_constexpr

from primus_turbo.flydsl.deep_ep.common import (
    ld_lds,
    lds_base,
    load_index,
    st_lds,
    tensor_addr,
)
from primus_turbo.flydsl.utils.buffer_ops import buffer_store, create_buffer_resource
from primus_turbo.flydsl.utils.launch_cache import run_launcher
from primus_turbo.flydsl.utils.prims import atomic_add

_THREADS = 256


def _make_layout_kernel(num_topk: int, num_ranks: int, num_experts: int):
    num_counts = num_experts + num_ranks
    num_experts_per_rank = num_experts // num_ranks
    # experts past num_ranks * num_experts_per_rank count per expert but reach no rank
    rank_expert_end = num_ranks * num_experts_per_rank
    pack_words = num_ranks % 4 == 0

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def dispatch_layout_kernel(
        TOPK_IDX: fx.Tensor,
        COUNTS: fx.Tensor,
        IS_TOKEN_IN_RANK: fx.Tensor,
        num_tokens: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        block_id = fx.block_idx.x
        # LDS: [num_experts] expert counts, then [num_ranks] rank counts
        lds = lds_base()
        counts_base = tensor_addr(COUNTS)
        topk_res = create_buffer_resource(TOPK_IDX, max_size=True)
        in_rank_res = create_buffer_resource(IS_TOKEN_IN_RANK, max_size=True)

        i = thread_id
        while i < fx.Int32(num_counts):
            st_lds(lds, i, fx.Int32(0))
            i = i + fx.Int32(_THREADS)
        fx.gpu.barrier()

        token = block_id * fx.Int32(_THREADS) + thread_id
        if token < num_tokens:
            rank_of = []
            for j in range_constexpr(num_topk):
                expert = load_index(topk_res, token * fx.Int32(num_topk) + fx.Int32(j), TOPK_IDX.element_type)
                if (expert >= fx.Int32(0)) & (expert < fx.Int32(num_experts)):
                    atomic_add(lds, expert, fx.Int32(1), "workgroup", 3)
                is_rank_expert = (expert >= fx.Int32(0)) & (expert < fx.Int32(rank_expert_end))
                rank_of.append(
                    arith.select(is_rank_expert, expert // fx.Int32(num_experts_per_rank), fx.Int32(-1))
                )

            # a token counts once per rank, on the first of its routes to it
            for j in range_constexpr(num_topk):
                is_first = rank_of[j] >= fx.Int32(0)
                for k in range_constexpr(j):
                    is_first = is_first & (rank_of[k] != rank_of[j])
                if is_first:
                    atomic_add(lds, fx.Int32(num_experts) + rank_of[j], fx.Int32(1), "workgroup", 3)

            def in_rank(r):
                hit = rank_of[0] == fx.Int32(r)
                for j in range_constexpr(1, num_topk):
                    hit = hit | (rank_of[j] == fx.Int32(r))
                return hit

            if fx.const_expr(pack_words):
                # 4 ranks per 32-bit store; rows stay 4-byte aligned since num_ranks % 4 == 0
                for w in range_constexpr(num_ranks // 4):
                    word = fx.Int32(0)
                    for b in range_constexpr(4):
                        word = word | arith.select(in_rank(4 * w + b), fx.Int32(1 << (8 * b)), fx.Int32(0))
                    buffer_store(word, in_rank_res, token * fx.Int32(num_ranks // 4) + fx.Int32(w))
            else:
                for r in range_constexpr(num_ranks):
                    byte = arith.select(in_rank(r), fx.Int8(1), fx.Int8(0))
                    buffer_store(byte, in_rank_res, token * fx.Int32(num_ranks) + fx.Int32(r))
        fx.gpu.barrier()

        # one global add per non-zero block count; integer adds keep the result deterministic
        i = thread_id
        while i < fx.Int32(num_counts):
            count = ld_lds(lds, i)
            if count != fx.Int32(0):
                atomic_add(counts_base, i, count, "agent", 1)
            i = i + fx.Int32(_THREADS)

    return dispatch_layout_kernel


@flyc.jit
def _launch_dispatch_layout(
    topk_idx,
    counts,
    is_token_in_rank,
    num_tokens: fx.Int32,
    num_blocks: fx.Int32,
    num_topk: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_layout_kernel(num_topk, num_ranks, num_experts)
    kernel(topk_idx, counts, is_token_in_rank, num_tokens).launch(
        grid=(num_blocks, 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
        smem=(num_experts + num_ranks) * 4,
    )


def get_dispatch_layout_flydsl(topk_idx: torch.Tensor, num_ranks: int, num_experts: int):
    """Return ``(num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank)`` like DeepEP's layout kernel."""
    assert topk_idx.dim() == 2 and topk_idx.is_contiguous()
    assert topk_idx.dtype in (torch.int32, torch.int64), topk_idx.dtype
    num_tokens, num_topk = topk_idx.shape
    counts = torch.zeros(num_experts + num_ranks, dtype=torch.int32, device=topk_idx.device)
    is_token_in_rank = torch.empty((num_tokens, num_ranks), dtype=torch.bool, device=topk_idx.device)
    if num_tokens > 0 and num_topk > 0:
        in_rank_bytes = is_token_in_rank.view(torch.uint8)
        run_launcher(
            _launch_dispatch_layout,
            topk_idx,
            counts,
            in_rank_bytes.view(torch.int32) if num_ranks % 4 == 0 else in_rank_bytes,
            num_tokens,
            (num_tokens + _THREADS - 1) // _THREADS,
            num_topk,
            num_ranks,
            num_experts,
            torch.cuda.current_stream(),
        )
    elif num_tokens > 0:
        is_token_in_rank.zero_()
    return counts[num_experts:], counts[:num_experts], is_token_in_rank
