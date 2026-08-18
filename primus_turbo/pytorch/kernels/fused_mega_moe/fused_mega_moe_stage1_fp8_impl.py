###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage mega MoE stage1 (gate-up) MXFP8 FlyDSL kernel composition.

Stage1 owns the forward dispatch + fc1 and the fc1-input pool requant for dW1; on the backward, the
L1 dgrad and the dW1 wgrad. Every call below is a helper the fused path already uses.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    colwise_grouped_meta,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    combine_l1_dgrad_mxfp8_flydsl_kernel,
    dispatch_l1_fwd_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_fp8_weights import (
    _DW_FP8_FORMAT,
    _w1_fp8_cached,
    _w1t_combine_fp8_cached,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)

__all__ = [
    "fused_mega_moe_stage1_forward_fp8_impl",
    "fused_mega_moe_stage1_backward_fp8_impl",
]


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
            pool_fp8,
            pool_scale,
            _DW_FP8_FORMAT,
            meta=meta,
        )
    return grouped_gemm_fp8_variable_k_impl(
        a_t,
        b_t,
        a_ts.view(torch.float8_e8m0fnu),
        b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64),
        offs_pc.to(torch.int64),
        trans_a=False,
        trans_b=False,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None,
        default_backend=BackendType.FLYDSL.value,
    )


def _l1_dgrad_combine_mxfp8_flydsl_kernel(
    w1,
    group,
    handle,
    *,
    grad_l1_rowwise_fp8,
    grad_gate,
    topk_idx,
    num_tokens,
    num_topk,
    w1t_fp8=None,
):
    """L1 dgrad (``grad_l1 @ w1^T``) + combine PUSH + reduce + grad_gate scatter
    -> ``(dx [num_tokens, H] bf16, grad_topk_weights [num_tokens, num_topk] f32)``."""
    w1tf = w1t_fp8 if w1t_fp8 is not None else _w1t_combine_fp8_cached(w1)
    dx, d_topk_w_flat = combine_l1_dgrad_mxfp8_flydsl_kernel(
        w1tf,
        list(handle),
        topk_indices=topk_idx.contiguous().view(-1),
        grad_gate=grad_gate,
        x_fp8_rowwise=grad_l1_rowwise_fp8,
        num_combine_cu=28,  # unified w/ fwd L2 (task-based push; T=8192)
    )
    grad_topk_weights = d_topk_w_flat[: num_tokens * num_topk].view(num_tokens, num_topk)
    return dx, grad_topk_weights


def prepare_dw1_pool_operand_fp8(
    pool_x_fp8: Tuple[torch.Tensor, torch.Tensor],
    handle: tuple,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], dict]:
    """Turn the forward's fc1-input pool into dW1's ``b`` operand -> ``(pool_colwise, meta)``.

    Called from the forward while ``pool_x_fp8`` is still a live view of the symm pool: requantizing
    it there consumes the view instead of the clone the backward would need, and keeps the requant
    off the backward critical path. ``meta`` is reused by the dual-quant, dW1 and dW2."""
    meta = colwise_grouped_meta(
        handle[_HANDLE_GROUP_LENS], handle[_HANDLE_GROUP_OFFS], pool_rows=pool_x_fp8[0].shape[0]
    )
    pool_colwise = colwise_requant_mxfp8_grouped_fp8in_flydsl(
        pool_x_fp8[0],
        pool_x_fp8[1],
        _DW_FP8_FORMAT,
        meta=meta,
    )[:2]
    return pool_colwise, meta


_HANDLE_GROUP_LENS = 9  # fp8 dispatch handle index: per-local-expert real token counts
_HANDLE_GROUP_OFFS = 10  # ... and their prefix over the block_m-padded pool

# fp8 dispatch handle: dispatch_prologue's 11 tables + num_tile_blocks appended by the L1 kernel.
_HANDLE_LEN = 14


def fused_mega_moe_stage1_forward_fp8_impl(
    x: torch.Tensor,
    w1: torch.Tensor,
    group,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    save_bwd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, tuple, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
    """dispatch + grouped fc1 GEMM (nt, mxfp8).

    Returns ``(l1, dispatch_weights, handle, pool_x_colwise, colwise_meta)``. ``dispatch_weights`` is
    cloned because stage2's backward needs it as the SwiGLU^T scale long after later stages overwrite
    the symm buffer. ``pool_x_colwise`` / ``colwise_meta`` (None unless ``save_bwd``) are dW1's ``b``
    operand, requantized here while the fc1-input pool is still live in symm.
    """
    # int64 end-to-end (combine reads topk i64)
    topk_idx = topk_idx.to(torch.int64)
    l1, handle, dispatch_weights, pool_x_fp8 = dispatch_l1_fwd_mxfp8_flydsl_kernel(
        x,
        _w1_fp8_cached(w1),
        group,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )
    assert len(handle) == _HANDLE_LEN, f"fp8 dispatch handle len {len(handle)} != {_HANDLE_LEN}; ABI changed"

    pool_x_colwise, colwise_meta = (
        prepare_dw1_pool_operand_fp8(pool_x_fp8, handle) if save_bwd else (None, None)
    )
    return l1, dispatch_weights.clone(), tuple(handle), pool_x_colwise, colwise_meta


def fused_mega_moe_stage1_backward_fp8_impl(
    grad_l1_rowwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    grad_l1_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    grad_gate: torch.Tensor,
    pool_x_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    colwise_meta: dict,
    w1: torch.Tensor,
    handle: tuple,
    group,
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L1 dgrad combine + dW1 (variable-K wgrad).

    Returns ``(dx, grad_topk_weights, dW1)``. Both grad_l1 operands arrive pre-quantized from
    stage2's fused SwiGLU^T dual-quant, which is why they come through the state side channel and
    not the ``l1`` gradient slot. The two calls stay serial on the default stream.
    """
    dx, grad_topk_weights = _l1_dgrad_combine_mxfp8_flydsl_kernel(
        w1,
        group,
        handle,
        grad_l1_rowwise_fp8=grad_l1_rowwise_fp8,
        grad_gate=grad_gate,
        topk_idx=topk_idx,
        num_tokens=num_tokens,
        num_topk=num_topk,
    )
    dW1 = _mxfp8_variable_k_wgrad_dw1(grad_l1_colwise_fp8, pool_x_colwise_fp8, colwise_meta)
    return dx, grad_topk_weights, dW1
