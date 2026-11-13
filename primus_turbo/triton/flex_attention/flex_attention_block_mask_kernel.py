###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Classify (query-block, key-block) tiles of a ``@triton.jit`` mask_mod.

``create_block_mask`` runs the same mask_mod the attention kernel later inlines, so one
predicate drives both the block lists and the per-element masking of partial blocks.
"""

import triton
import triton.language as tl

from .flex_attention_utils import apply_mask_mod

BLOCK_EMPTY = 0
BLOCK_PARTIAL = 1
BLOCK_FULL = 2

# Kernel-side copies: a @triton.jit function only reads globals declared constexpr.
_EMPTY = tl.constexpr(BLOCK_EMPTY)
_PARTIAL = tl.constexpr(BLOCK_PARTIAL)
_FULL = tl.constexpr(BLOCK_FULL)


@triton.jit
def classify_blocks_kernel(
    OUT,
    stride_ob,
    stride_oh,
    stride_oq,
    Q_LEN,
    KV_LEN,
    CU_SEQLENS_Q,
    CU_SEQLENS_K,
    MASK_MOD: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_AUX0=None,
    MASK_AUX1=None,
    MASK_AUX2=None,
    MASK_AUX3=None,
    NUM_MASK_AUX: tl.constexpr = 0,
):
    """One program per tile: OUT[b, h, q_block, kv_block] = EMPTY / PARTIAL / FULL.

    Positions past the sequence end count as masked, so a tile straddling the end is
    never FULL: the attention kernel's FULL pass skips its bounds checks.
    """
    kv_block = tl.program_id(0)
    q_block = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    if IS_VARLEN:
        seqlen_q = tl.load(CU_SEQLENS_Q + b + 1) - tl.load(CU_SEQLENS_Q + b)
        seqlen_k = tl.load(CU_SEQLENS_K + b + 1) - tl.load(CU_SEQLENS_K + b)
    else:
        seqlen_q = Q_LEN
        seqlen_k = KV_LEN
    q_idx = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
    kv_idx = kv_block * BLOCK_KV + tl.arange(0, BLOCK_KV)[None, :]
    in_bounds = (q_idx < seqlen_q) & (kv_idx < seqlen_k)
    keep = apply_mask_mod(
        MASK_MOD, b, h, q_idx, kv_idx, MASK_AUX0, MASK_AUX1, MASK_AUX2, MASK_AUX3, NUM_MASK_AUX
    )
    keep = (keep & in_bounds).to(tl.int32)
    any_kept = tl.max(tl.max(keep, axis=1), axis=0)
    all_kept = tl.min(tl.min(keep, axis=1), axis=0)
    category = tl.where(all_kept == 1, _FULL, tl.where(any_kept == 1, _PARTIAL, _EMPTY))
    tl.store(OUT + b * stride_ob + h * stride_oh + q_block * stride_oq + kv_block, category.to(tl.int8))
