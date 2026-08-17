###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mega MoE autograd ops with an MXFP8 forward and a partial-fp8 backward.

The fp8 counterpart of ``fused_mega_moe``, with the same two entry points:

  * ``fused_mega_moe_fp8`` -- one fully fused op (dispatch+fc1+SwiGLU+fc2+combine).
  * ``fused_mega_moe_fp8_stage1`` / ``fused_mega_moe_fp8_stage2`` -- the same math split into two
    autograd edges at ``l1`` (pre-SwiGLU), so w1 and w2 can be separate DDP gradient boundaries.
    Stage state rides a ``_Fp8StageState`` side channel rather than the ops' args/returns.

Pass ``w1`` / ``w2`` as the high-precision weights; their mxfp8 quant is maintained inside the
impls, keyed on ``w._version``. The op is NOT CUDA-graph capturable.
"""

from typing import Optional

import torch
from torch.distributed import ProcessGroup

from primus_turbo.pytorch.kernels.fused_mega_moe import (
    fused_mega_moe_stage1_backward_fp8_impl,
    fused_mega_moe_stage1_forward_fp8_impl,
    fused_mega_moe_stage2_backward_fp8_impl,
    fused_mega_moe_stage2_forward_fp8_impl,
)

# This op file exports only its own final API (the autograd Function + its wrapper). Everything else
# (the per-stage ``_mxfp8_*`` helpers, ``_DW_FP8_FORMAT``, and the ``prepare_w1t/w2t_dgrad_fp8``
# weight-prep) lives in the kernels layer -- callers import those from
# ``primus_turbo.pytorch.kernels.fused_mega_moe`` directly (weight prep lives in
# ``mega_moe_fp8_weights``, the per-stage helpers in the stage modules), not via
# this file.
__all__ = [
    "FusedMegaMoEFP8Stage1Function",
    "FusedMegaMoEFP8Stage2Function",
    "fused_mega_moe_fp8_stage1",
    "fused_mega_moe_fp8_stage2",
]


class _Fp8StageState:
    """Side channel carrying non-differentiable fp8 operands between stage1 and stage2.

    The bf16 split threads everything through op args/returns, but the fp8 backward cannot: its
    fused SwiGLU^T emits grad_l1 ONLY as the two quantized operands the L1 dgrad and dW1 consume
    ``((q_row, a_sp), (q_col, s_col))``, uint8/int32 tensors of a different shape than ``l1``, which
    no gradient slot accepts. So stage2.backward parks them here, returns ``None`` on the ``l1``
    slot, and stage1.backward picks them up -- safe because autograd's topological order runs
    stage2.backward strictly first. ``grad_gate`` still rides the ``dispatch_weights`` gradient slot,
    the same channel bf16 uses.

    The forward half (``pool_x_colwise`` / ``colwise_meta``, produced by stage1's pool requant) rides
    along because stage2's dual-quant and dW2 need the same grouped meta.

    One instance per stage1 call; both stages hold it on their ``ctx``.
    """

    __slots__ = ("pool_x_colwise", "colwise_meta", "grad_l1_rowwise_fp8", "grad_l1_colwise_fp8")

    def __init__(self):
        self.pool_x_colwise = None
        self.colwise_meta = None
        self.grad_l1_rowwise_fp8 = None
        self.grad_l1_colwise_fp8 = None


class FusedMegaMoEFP8Stage1Function(torch.autograd.Function):
    """Stage1 gate-up (MXFP8): dispatch + fc1, owns w1. Output is pre-SwiGLU ``l1``."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        group: ProcessGroup,
        state: _Fp8StageState,
    ):
        with torch.profiler.record_function("mega_moe_fp8_stage1_forward"):
            assert x.dim() == 2 and x.is_cuda and x.dtype == torch.bfloat16, (
                f"x must be 2D bf16 CUDA, got {tuple(x.shape)}/{x.dtype}"
            )
            assert w1.is_cuda and w1.dim() == 3 and w1.dtype == torch.bfloat16, (
                "w1 must be a 3D bf16 CUDA tensor"
            )

            # int64 end-to-end (combine reads topk i64)
            topk_idx = topk_idx.to(torch.int64)
            ctx.set_materialize_grads(False)
            save_bwd = any(ctx.needs_input_grad)

            (
                l1,
                dispatch_weights,
                handle,
                pool_x_colwise,
                colwise_meta,
            ) = fused_mega_moe_stage1_forward_fp8_impl(
                x,
                w1,
                group,
                topk_idx,
                topk_weights,
                save_bwd=save_bwd,
            )
            state.pool_x_colwise = pool_x_colwise
            state.colwise_meta = colwise_meta

            if save_bwd:
                # Unlike bf16, dW1 contracts over the pool requant stage1 already produced, so x
                # is NOT saved -- only w1 is. Non-tensor / non-graph state goes on ctx.
                ctx.group = group
                ctx.state = state
                ctx.handle = handle
                ctx.topk_idx = topk_idx
                ctx.num_tokens = x.shape[0]
                ctx.num_topk = topk_idx.shape[-1]
                ctx.save_for_backward(w1)
            # handle tensors are non-differentiable index/table tensors
            ctx.mark_non_differentiable(*handle)
            return (l1, dispatch_weights, *handle)

    @staticmethod
    @torch.no_grad()
    def backward(
        ctx, grad_l1: Optional[torch.Tensor], grad_dispatch_weights: Optional[torch.Tensor], *grad_handle
    ):
        """dx / grad_topk_weights + dW1. ``grad_l1`` is always None -- the real grad_l1 arrives
        pre-quantized through ``ctx.state``; ``grad_dispatch_weights`` couriers ``grad_gate``."""
        with torch.profiler.record_function("mega_moe_fp8_stage1_backward"):
            state = ctx.state
            if state.grad_l1_rowwise_fp8 is None:  # stage2.backward never ran -> nothing to do
                return (None,) * 6
            (w1,) = ctx.saved_tensors

            dx, grad_topk_weights, dW1 = fused_mega_moe_stage1_backward_fp8_impl(
                state.grad_l1_rowwise_fp8,
                state.grad_l1_colwise_fp8,
                grad_dispatch_weights,
                state.pool_x_colwise,
                state.colwise_meta,
                w1,
                ctx.handle,
                ctx.group,
                ctx.topk_idx,
                ctx.num_tokens,
                ctx.num_topk,
            )
            state.grad_l1_rowwise_fp8 = state.grad_l1_colwise_fp8 = None

            # grads for (x, topk_idx, topk_weights, w1, group, state)
            return dx, None, grad_topk_weights, dW1.to(w1.dtype), None, None


class FusedMegaMoEFP8Stage2Function(torch.autograd.Function):
    """Stage2 gate-down (MXFP8): SwiGLU + fc2 + combine, owns w2. Output is the MoE result y."""

    @staticmethod
    def forward(
        ctx,
        l1: torch.Tensor,
        dispatch_weights: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w2: torch.Tensor,
        group: ProcessGroup,
        state: _Fp8StageState,
        *handle,
    ) -> torch.Tensor:
        with torch.profiler.record_function("mega_moe_fp8_stage2_forward"):
            assert w2.is_cuda and w2.dim() == 3 and w2.dtype == torch.bfloat16, (
                "w2 must be a 3D bf16 CUDA tensor"
            )
            handle = tuple(handle)

            y = fused_mega_moe_stage2_forward_fp8_impl(
                l1,
                w2,
                handle,
                group,
                topk_idx,
                topk_weights,
            )

            ctx.set_materialize_grads(False)
            if any(ctx.needs_input_grad):
                ctx.group = group
                ctx.state = state
                ctx.handle = handle
                # dispatch_weights is unused in forward; saved only as the SwiGLU^T scale in backward
                ctx.save_for_backward(l1, dispatch_weights, w2)
            return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]):
        """dW2 + grad_gate (on the ``dispatch_weights`` slot). The quantized grad_l1 pair goes to
        stage1 through ``ctx.state``, so the ``l1`` slot returns None."""
        with torch.profiler.record_function("mega_moe_fp8_stage2_backward"):
            handle = ctx.handle
            n_in = 7 + len(handle)
            if grad_y is None:
                return (None,) * n_in
            l1, dispatch_weights, w2 = ctx.saved_tensors
            state = ctx.state

            (
                grad_l1_rowwise_fp8,
                grad_l1_colwise_fp8,
                grad_gate,
                dW2,
            ) = fused_mega_moe_stage2_backward_fp8_impl(
                grad_y,
                l1,
                dispatch_weights,
                w2,
                handle,
                ctx.group,
                state.colwise_meta,
            )
            state.grad_l1_rowwise_fp8 = grad_l1_rowwise_fp8
            state.grad_l1_colwise_fp8 = grad_l1_colwise_fp8

            # grads for (l1, dispatch_weights, topk_idx, topk_weights, w2, group, state, *handle)
            return (
                None,
                grad_gate,
                None,
                None,
                dW2.to(w2.dtype),
                None,
                None,
                *((None,) * len(handle)),
            )


def fused_mega_moe_fp8_stage1(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    group: ProcessGroup,
):
    """Stage1 gate-up (MXFP8). Returns ``(l1, dispatch_weights, handle, state)`` to feed stage2.

    ``dispatch_weights``, ``handle`` and ``state`` are opaque forward state; pass them straight into
    :func:`fused_mega_moe_fp8_stage2`. Pass ``w1`` directly -- its mxfp8 quant is maintained
    internally by a version-keyed cache.
    """
    state = _Fp8StageState()
    l1, dispatch_weights, *handle = FusedMegaMoEFP8Stage1Function.apply(
        x,
        topk_idx,
        topk_weights,
        w1,
        group,
        state,
    )
    return l1, dispatch_weights, tuple(handle), state


def fused_mega_moe_fp8_stage2(
    l1: torch.Tensor,
    dispatch_weights: torch.Tensor,
    handle: tuple,
    state: _Fp8StageState,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w2: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Stage2 gate-down (MXFP8). Consumes stage1's forward state; returns y."""
    return FusedMegaMoEFP8Stage2Function.apply(
        l1,
        dispatch_weights,
        topk_idx,
        topk_weights,
        w2,
        group,
        state,
        *handle,
    )
