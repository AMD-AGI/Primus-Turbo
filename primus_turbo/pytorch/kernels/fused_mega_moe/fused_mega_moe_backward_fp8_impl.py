###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 backward: conjugate of forward via Dispatch<->Combine duality (FlyDSL).

Unlike the bf16 sibling this is a plain orchestration function (no custom_op / dispatcher): it
reuses the forward's live symmetric buffer.

The L1 dgrad combine and dW1 MUST stay on the default stream back-to-back; dual-stream overlap is
unsupported.
"""

from typing import Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    colwise_grouped_meta,
    colwise_requant_fp8in_and_quant_bf16_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    quantize_grouped_weight_mxfp8_flydsl,
    swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e5m2
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_weight_prep_fp8 import (
    prepare_dispatch_weight_fp8,
    prepare_w2_fp8,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)

__all__ = [
    "fused_mega_moe_backward_fp8_impl",
    "prepare_dw1_pool_operand_fp8",
    "prepare_w2t_dgrad_fp8",
]

# dW1/dW2 wgrad fp8 encoding. E4M3 measured a slightly higher dW SNR than E5M2 at DSv3 magnitudes.
_DW_FP8_FORMAT = torch.float8_e4m3fn

# dispatch_prologue handle layout: [9]=num_tokens_per_expert, [10]=its prefix into the padded pool.
# These MUST be the REAL unpadded lengths -- the variable-K wgrads mask each group at group_lens, so
# a padded length would fold the tail padding rows into dW.
_HANDLE_GROUP_LENS = 9
_HANDLE_GROUP_OFFS = 10

_W2T_PREP_ATTR = "_mega_fp8_w2t_prep"
_W1T_COMBINE_PREP_ATTR = "_mega_fp8_w1t_combine_prep"


def prepare_w2t_dgrad_fp8(w2: torch.Tensor) -> tuple:
    """``w2^T`` [G,I,H] prepped for the L2 dgrad's dispatch GEMM -> ``(wq, ws, flat, b_sp)``.

    The L2 dgrad runs NT via the transposed weight, so w2 must be quantized along H (its
    contraction axis). Static weight prep; the transpose never runs inside the kernel.
    """
    return prepare_dispatch_weight_fp8(w2.transpose(1, 2).contiguous())  # [G,I,H]


def _w2t_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed cache of :func:`prepare_w2t_dgrad_fp8`, stashed on the weight tensor."""
    v = getattr(w2, "_version", 0)
    ent = getattr(w2, _W2T_PREP_ATTR, None)
    if ent is None or ent[0] != v:
        with torch.no_grad():
            out = prepare_w2t_dgrad_fp8(w2)
        ent = (v, out)
        setattr(w2, _W2T_PREP_ATTR, ent)
    return ent[1]


def _w1t_combine_fp8_cached(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed cache of the fc1^T combine weight prep (same format as the forward w2)."""
    v = getattr(w1, "_version", 0)
    ent = getattr(w1, _W1T_COMBINE_PREP_ATTR, None)
    if ent is None or ent[0] != v:
        with torch.no_grad():
            out = prepare_w2_fp8(w1.transpose(1, 2).contiguous())  # [G, H, 2I]
        ent = (v, out)
        setattr(w1, _W1T_COMBINE_PREP_ATTR, ent)
    return ent[1]


def _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2, group, handle, *, w2t_fp8=None,
                                           num_dispatch_cu=None, num_preshuffle_cu=None):
    """Fp8 dispatch(dy) PUSH + fc2 dgrad: ``grad_swiglu = dy @ w2`` and rowwise-fp8 pool for dW2 ``a``."""
    w2t = w2t_fp8 if w2t_fp8 is not None else _w2t_fp8_cached(w2)
    # None lets the kernel look this shape's split up; the dgrad's N=I wants a different one than the
    # forward's N=2I, which is what a per-shape table gets right without pinned constants here.
    grad_swiglu, _, _, pool_fp8_handle = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        dy, w2t, group, handle=handle,
        num_dispatch_cu=num_dispatch_cu, num_preshuffle_cu=num_preshuffle_cu,
    )
    return grad_swiglu, pool_fp8_handle


def _mxfp8_variable_k_wgrad_dw2(a_fp8, b_bf16, group_lens, group_offs, meta=None):
    """dW2 = ``a^T @ b`` (variable-K over the dispatched pool tokens), MXFP8 -> ``[G, H, *]`` bf16.

    ``a`` is the L2-dgrad dispatched-dy pool, requantized colwise directly from fp8 with no bf16
    round-trip; ``b`` is ``act_weighted``. One dual-launch produces both transposed operands."""
    pool_fp8, pool_scale = a_fp8
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=pool_fp8.shape[0])
    a_t, a_ts, b_t, b_ts, lens_pc, offs_pc = colwise_requant_fp8in_and_quant_bf16_grouped_flydsl(
        pool_fp8, pool_scale, b_bf16, _DW_FP8_FORMAT, meta=meta,
    )
    return grouped_gemm_fp8_variable_k_impl(
        a_t, b_t,
        a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64), offs_pc.to(torch.int64),
        trans_a=False, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None, default_backend=BackendType.FLYDSL.value,
    )


def _l1_dgrad_combine_mxfp8_flydsl_kernel(
    w1, group, handle, *, grad_l1_rowwise_fp8, grad_gate, topk_idx, num_tokens, num_topk,
    w1t_fp8=None,
):
    """L1 dgrad (``grad_l1 @ w1^T``) + combine PUSH + reduce + grad_gate scatter
    -> ``(dx [num_tokens, H] bf16, grad_topk_weights [num_tokens, num_topk] f32)``."""
    w1tf = w1t_fp8 if w1t_fp8 is not None else _w1t_combine_fp8_cached(w1)
    dx, d_topk_w_flat = grouped_gemm_combine_mxfp8_flydsl_kernel(
        None, w1tf, list(handle), group,
        topk_indices=topk_idx.contiguous().view(-1), grad_gate=grad_gate,
        x_fp8_rowwise=grad_l1_rowwise_fp8,
        num_combine_cu=28,  # unified w/ fwd L2 (task-based push; T=8192)
    )
    grad_topk_weights = d_topk_w_flat[: num_tokens * num_topk].view(num_tokens, num_topk)
    return dx, grad_topk_weights


def prepare_dw1_pool_operand_fp8(
    pool_x_fp8: Tuple[torch.Tensor, torch.Tensor], handle: tuple,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], dict]:
    """Turn the forward's fc1-input pool into dW1's ``b`` operand -> ``(pool_colwise, meta)``.

    Called from the forward while ``pool_x_fp8`` is still a live view of the symm pool: requantizing
    it there consumes the view instead of the clone the backward would need, and keeps the requant
    off the backward critical path. ``meta`` is reused by the dual-quant, dW1 and dW2."""
    meta = colwise_grouped_meta(
        handle[_HANDLE_GROUP_LENS], handle[_HANDLE_GROUP_OFFS], pool_rows=pool_x_fp8[0].shape[0]
    )
    pool_colwise = colwise_requant_mxfp8_grouped_fp8in_flydsl(
        pool_x_fp8[0], pool_x_fp8[1], _DW_FP8_FORMAT, meta=meta,
    )[:2]
    return pool_colwise, meta


def _mxfp8_variable_k_wgrad_dw1(
    a_colwise_fp8,
    pool_x_operand: Tuple[torch.Tensor, torch.Tensor],
    meta,
    *,
    pool_x_is_colwise: bool = True,
):
    """dW1 = ``a^T @ b`` (variable-K over the pool tokens), MXFP8 -> ``[G, 2I, H]`` bf16.

    ``a`` = ``grad_l1``, pre-quantized colwise by the fused dual-quant; ``b`` = the fc1-input pool,
    already colwise-fp8 from the forward, or requantized here when ``pool_x_is_colwise=False``
    (isolated benches that time the requant as part of dW1).

    This GEMM is LOCAL: it contracts over pool tokens already gathered on this rank, so unlike the
    bf16 path -- which re-dispatches ``saved_x`` cross-rank -- it needs no transfer at all."""
    a_t, a_ts = a_colwise_fp8
    lens_pc, offs_pc = meta["lens_pc"], meta["offs_pc"]
    if pool_x_is_colwise:
        b_t, b_ts = pool_x_operand
    else:
        pool_fp8, pool_scale = pool_x_operand
        b_t, b_ts, _, _ = colwise_requant_mxfp8_grouped_fp8in_flydsl(
            pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta,
        )
    return grouped_gemm_fp8_variable_k_impl(
        a_t, b_t,
        a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64), offs_pc.to(torch.int64),
        trans_a=False, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None, default_backend=BackendType.FLYDSL.value,
    )


def fused_mega_moe_backward_fp8_impl(
    grad_y: torch.Tensor,
    l1: torch.Tensor,
    dispatch_weights: torch.Tensor,
    pool_x_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: tuple,
    group,
    num_tokens: int,
    num_topk: int,
    colwise_meta: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mxfp8 MoE backward (conjugate of forward via the Dispatch<->Combine duality).

    ``pool_x_colwise_fp8`` / ``colwise_meta`` come straight from the forward (see
    :func:`prepare_dw1_pool_operand_fp8`). Returns ``(dx, grad_topk_weights, dW1, dW2)`` with
    dW1/dW2 cast back to the weight dtypes.
    """
    group_lens = handle[_HANDLE_GROUP_LENS]
    group_offs = handle[_HANDLE_GROUP_OFFS]
    dy = grad_y.contiguous().to(torch.bfloat16)

    # L2 dgrad: dispatch(dy) + fc2 -> grad_swiglu + the dispatched-dy pool (the dW2 `a` operand)
    grad_swiglu, dispatch_l2_grad_fp8 = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2, group, handle)

    # SwiGLU^T (re-inject routing weight) + gate grad, fused with grad_l1's rowwise (L1 dgrad) and
    # colwise (dW1) quant -- grad_l1 has no other consumer, so it never reaches HBM
    (
        gl1_q_row, gl1_a_sp, gl1_q_col, gl1_s_col, grad_gate, act_weighted,
    ) = swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl(
        grad_swiglu, l1, dispatch_weights, _DW_FP8_FORMAT, meta=colwise_meta,
    )

    # dW2 = dispatch_l2_grad^T @ act_weighted -- must run before anything overwrites symm.pool_fp8
    dW2 = _mxfp8_variable_k_wgrad_dw2(
        dispatch_l2_grad_fp8, act_weighted, group_lens, group_offs, meta=colwise_meta,
    )

    # L1 dgrad + combine, then dW1 -- serial on the default stream (dual-stream overlap forbidden)
    dx, grad_topk_weights = _l1_dgrad_combine_mxfp8_flydsl_kernel(
        w1, group, handle,
        grad_l1_rowwise_fp8=(gl1_q_row, gl1_a_sp),
        grad_gate=grad_gate, topk_idx=topk_idx, num_tokens=num_tokens, num_topk=num_topk,
    )
    dW1 = _mxfp8_variable_k_wgrad_dw1((gl1_q_col, gl1_s_col), pool_x_colwise_fp8, colwise_meta)

    return dx, grad_topk_weights, dW1.to(w1.dtype), dW2.to(w2.dtype)
