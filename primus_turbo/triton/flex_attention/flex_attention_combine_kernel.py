###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Split-KV combine (``num_splits > 1``).

Splitting the KV loop across programs raises occupancy when the natural grid
(``batch * nheads * q_blocks``) is too small to fill the GPU -- the short-query /
long-context case. Each split attends a slice of the keys and writes a partial output
plus its own LSE; this reduces them.

For split ``s`` over key subset ``S_s``::

    lse_s = log sum_{j in S_s} exp(x_j)
    o_s   = sum_{j in S_s} exp(x_j) v_j / exp(lse_s)

so with ``lse = logsumexp_s(lse_s)``::

    o = sum_s o_s * exp(lse_s - lse)

A split that saw no keys carries ``lse_s = -inf`` and drops out on its own, since
``exp(-inf - lse) == 0``. A row where *every* split is empty yields zero output.

AITER has a ``fwd_combine`` entry point but it raises NotImplementedError, and its
split-K machinery lives in the decode kernel, which does not carry this project's
score_mod / mask_mod / sink support. So the split is driven from the prefill kernel and
reduced here.
"""

import triton
import triton.language as tl

__all__ = ["combine_splits"]


@triton.jit
def _combine_kernel(
    OutPartial,  # [num_splits, B, S, H, Dv] fp32
    LsePartial,  # [num_splits, B, H, S]     fp32
    Out,  # [B, S, H, Dv] (out dtype)
    Lse,  # [B, H, S]     fp32
    stride_op_s,
    stride_op_b,
    stride_op_m,
    stride_op_h,
    stride_op_d,
    stride_lp_s,
    stride_lp_b,
    stride_lp_h,
    stride_lp_m,
    stride_o_b,
    stride_o_m,
    stride_o_h,
    stride_o_d,
    stride_l_b,
    stride_l_h,
    stride_l_m,
    seqlen_q,
    nheads,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    lse_base = LsePartial + off_b * stride_lp_b + off_h * stride_lp_h
    # running max over splits, so the exponentials below cannot overflow
    lse_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        v = tl.load(lse_base + s * stride_lp_s + offs_m * stride_lp_m, mask=mask_m, other=float("-inf"))
        lse_max = tl.maximum(lse_max, v)
    # all-empty rows: pin to 0 so the arithmetic stays finite; output stays zero
    all_empty = lse_max == float("-inf")
    lse_max_safe = tl.where(all_empty, 0.0, lse_max)

    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    o_base = (
        OutPartial
        + off_b * stride_op_b
        + off_h * stride_op_h
        + offs_m[:, None] * stride_op_m
        + offs_d[None, :] * stride_op_d
    )
    for s in tl.static_range(NUM_SPLITS):
        lse_s = tl.load(lse_base + s * stride_lp_s + offs_m * stride_lp_m, mask=mask_m, other=float("-inf"))
        w = tl.exp(lse_s - lse_max_safe)
        w = tl.where(lse_s == float("-inf"), 0.0, w)
        denom += w
        o_s = tl.load(o_base + s * stride_op_s, mask=mask_m[:, None], other=0.0)
        acc += o_s * w[:, None]

    denom_safe = tl.where(denom == 0.0, 1.0, denom)
    acc = acc / denom_safe[:, None]

    out_ptrs = (
        Out
        + off_b * stride_o_b
        + off_h * stride_o_h
        + offs_m[:, None] * stride_o_m
        + offs_d[None, :] * stride_o_d
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

    lse = lse_max_safe + tl.log(denom_safe)
    lse = tl.where(all_empty, float("-inf"), lse)
    tl.store(Lse + off_b * stride_l_b + off_h * stride_l_h + offs_m * stride_l_m, lse, mask=mask_m)


def combine_splits(out_partial, lse_partial, out, lse, block_m: int = 64):
    """Reduce ``[num_splits, ...]`` partials into ``out`` / ``lse`` in place."""
    num_splits, batch, seqlen_q, nheads, head_dim = out_partial.shape
    grid = (triton.cdiv(seqlen_q, block_m), batch * nheads)
    _combine_kernel[grid](
        out_partial,
        lse_partial,
        out,
        lse,
        *out_partial.stride(),
        *lse_partial.stride(),
        *out.stride(),
        *lse.stride(),
        seqlen_q,
        nheads,
        NUM_SPLITS=num_splits,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
    )
    return out, lse
