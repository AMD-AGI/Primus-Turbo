###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mega MoE forward/backward autograd ops over a single symmetric buffer.

Two entry points share the same kernel sequence (all FlyDSL composition lives in
``primus_turbo.pytorch.kernels.fused_mega_moe``; this module holds only autograd):

  * ``fused_mega_moe`` -- one fully fused op (dispatch+GEMM1+SwiGLU+GEMM2+combine).
  * ``fused_mega_moe_stage1`` / ``fused_mega_moe_stage2`` -- the same math split
    into two autograd edges so the module tree can mirror Megatron's native
    ``experts.linear_fc1`` (owns w1) / ``experts.linear_fc2`` (owns w2), which
    restores the DDP comm/compute overlap granularity. Boundary is at ``l1_out``
    (pre-SwiGLU). State is threaded via the ops' args/returns -- no shared
    context object: stage1 returns ``(l1_out, dispatch_weights, handle)``; ``grad_gate``
    (stage2's SwiGLU^T -> stage1's combine) rides back on the gradient slot of
    ``dispatch_weights`` (a differentiable stage1 output / stage2 input), so stage2.backward
    returns it as d/d(dispatch_weights) and autograd delivers it to stage1.backward.
"""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.fused_mega_moe import (
    fused_mega_moe_backward_impl,
    fused_mega_moe_forward_impl,
    fused_mega_moe_stage1_backward_impl,
    fused_mega_moe_stage1_forward_impl,
    fused_mega_moe_stage2_backward_impl,
    fused_mega_moe_stage2_forward_impl,
)

__all__ = [
    "FusedMegaMoEFunction",
    "fused_mega_moe",
    "fused_mega_moe_stage1",
    "fused_mega_moe_stage2",
]


class FusedMegaMoEFunction(torch.autograd.Function):
    """Wraps the fused mega MoE forward so its output joins the autograd graph."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        group: ProcessGroup,
    ) -> torch.Tensor:
        with torch.profiler.record_function("fused_mega_moe_forward"):
            # Validate at the op boundary so failures point here, not deep in FlyDSL.
            assert x.dim() == 2 and x.is_cuda and x.dtype == torch.bfloat16, (
                f"x must be 2D bf16 CUDA, got {x.shape}/{x.dtype}"
            )
            assert w1.is_cuda and w2.is_cuda, "w1/w2 must be CUDA tensors"
            assert x.device == w1.device == w2.device == topk_idx.device == topk_weights.device, (
                "all inputs must share one device"
            )
            assert topk_idx.shape[0] == topk_weights.shape[0] == x.shape[0], (
                f"token count mismatch: x={x.shape[0]} idx={topk_idx.shape[0]} w={topk_weights.shape[0]}"
            )
            assert topk_idx.shape[-1] == topk_weights.shape[-1], (
                f"num_topk mismatch: idx={topk_idx.shape[-1]} w={topk_weights.shape[-1]}"
            )

            num_tokens = x.shape[0]
            num_topk = topk_idx.shape[-1]
            # int64 end-to-end (combine reads topk i64)
            topk_idx = topk_idx.to(torch.int64)

            ctx.set_materialize_grads(False)

            # fused MoE forward: dispatch grouped L1 GEMM (NT) + SwiGLU + grouped L2 GEMM combine (NT)
            y, l1_out, dispatch_weights_in_buf, handle = fused_mega_moe_forward_impl(
                x,
                w1,
                w2,
                group,
                topk_idx,
                topk_weights,
                "nt",
                BackendType.FLYDSL.value,
            )

            # stash everything backward needs
            if any(ctx.needs_input_grad):
                # clone before a later layer overwrites the shared buffer
                dispatch_weights_in_buf = dispatch_weights_in_buf.clone()

                ctx.group = group
                ctx.num_tokens = num_tokens
                ctx.num_topk = num_topk
                ctx.save_for_backward(
                    x,
                    l1_out,
                    dispatch_weights_in_buf,
                    w1,
                    w2,
                    topk_idx,
                    *handle,
                )
            return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        """Conjugate of forward via Dispatch<->Combine duality; grads for x / w1 / w2 / topk_w."""
        with torch.profiler.record_function("fused_mega_moe_backward"):
            # grad_y is None when the output got no grad
            if grad_y is None:
                return (None,) * 6
            saved_x, l1_out, dispatch_weights_in_buf, w1, w2, topk_idx, *handle = ctx.saved_tensors
            handle = tuple(handle)

            # fused MoE backward: L2 dgrad (nn) + SwiGLU^T + dW2 + L1 dgrad combine (nn) + dW1 (tn)
            dx, grad_topk_weights, dW1, dW2 = fused_mega_moe_backward_impl(
                grad_y,
                saved_x,
                l1_out,
                dispatch_weights_in_buf,
                w1,
                w2,
                topk_idx,
                handle,
                ctx.group,
                ctx.num_tokens,
                ctx.num_topk,
                BackendType.FLYDSL.value,
            )

            # grads for (x, topk_idx, topk_weights, w1, w2, group)
            return (
                dx,
                None,
                grad_topk_weights,
                dW1,
                dW2,
                None,
            )


def fused_mega_moe(
    group: ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """One fully fused mega MoE forward; the symmetric buffer is fetched (and cached) internally."""
    return FusedMegaMoEFunction.apply(x, topk_idx, topk_weights, w1, w2, group)


class FusedMegaMoEStage1Function(torch.autograd.Function):
    """Stage1 gate-up: dispatch + grouped GEMM1 (owns w1). Output is pre-SwiGLU l1_out."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        group: ProcessGroup,
    ):
        with torch.profiler.record_function("fused_mega_moe_stage1_forward"):
            assert x.dim() == 2 and x.is_cuda and x.dtype == torch.bfloat16, (
                f"x must be 2D bf16 CUDA, got {tuple(x.shape)}/{x.dtype}"
            )
            assert w1.is_cuda and w1.dim() == 3, "w1 must be a 3D CUDA tensor"

            l1_out, dispatch_weights, handle = fused_mega_moe_stage1_forward_impl(
                x, w1, group, topk_idx, topk_weights
            )

            ctx.set_materialize_grads(False)
            if any(ctx.needs_input_grad):
                ctx.group = group
                ctx.handle = handle
                ctx.num_tokens = x.shape[0]
                ctx.num_topk = topk_idx.shape[-1]
                ctx.topk_idx = topk_idx
                ctx.save_for_backward(x, w1)
            # handle tensors are non-differentiable index/table tensors
            ctx.mark_non_differentiable(*handle)
            return (l1_out, dispatch_weights, *handle)

    @staticmethod
    @torch.no_grad()
    def backward(
        ctx, grad_l1: Optional[torch.Tensor], grad_dispatch_weights: Optional[torch.Tensor], *grad_handle
    ):
        """dx / grad_topk_weights + dW1. ``grad_dispatch_weights`` couriers ``grad_gate`` from stage2."""
        with torch.profiler.record_function("fused_mega_moe_stage1_backward"):
            if grad_l1 is None:
                return (None,) * 5
            x, w1 = ctx.saved_tensors
            dx, grad_topk_weights, dW1 = fused_mega_moe_stage1_backward_impl(
                grad_l1,
                x,
                w1,
                ctx.handle,
                ctx.group,
                ctx.topk_idx,
                grad_dispatch_weights,  # courier: grad w.r.t. dispatch_weights slot == grad_gate
                ctx.num_tokens,
                ctx.num_topk,
            )
            # grads for (x, topk_idx, topk_weights, w1, group)
            return dx, None, grad_topk_weights, dW1.to(w1.dtype), None


class FusedMegaMoEStage2Function(torch.autograd.Function):
    """Stage2 gate-down: SwiGLU + grouped GEMM2 + combine (owns w2). Output is the MoE result y."""

    @staticmethod
    def forward(
        ctx,
        l1_out: torch.Tensor,
        dispatch_weights: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w2: torch.Tensor,
        group: ProcessGroup,
        *handle,
    ) -> torch.Tensor:
        with torch.profiler.record_function("fused_mega_moe_stage2_forward"):
            assert w2.is_cuda and w2.dim() == 3, "w2 must be a 3D CUDA tensor"
            handle = tuple(handle)

            y = fused_mega_moe_stage2_forward_impl(l1_out, w2, handle, topk_idx, topk_weights)

            ctx.set_materialize_grads(False)
            if any(ctx.needs_input_grad):
                ctx.group = group
                ctx.handle = handle
                # dispatch_weights is unused in forward; saved only as the SwiGLU^T scale in backward
                ctx.save_for_backward(l1_out, dispatch_weights, w2)
            return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]):
        """grad l1_out (via L2 dgrad + SwiGLU^T) + dW2; grad_gate rides the dispatch_weights slot."""
        with torch.profiler.record_function("fused_mega_moe_stage2_backward"):
            handle = ctx.handle
            n_in = 6 + len(handle)
            if grad_y is None:
                return (None,) * n_in
            l1_out, dispatch_weights, w2 = ctx.saved_tensors

            grad_l1, grad_gate, dW2 = fused_mega_moe_stage2_backward_impl(
                grad_y, l1_out, dispatch_weights, w2, handle, ctx.group
            )

            # grads for (l1_out, dispatch_weights, topk_idx, topk_weights, w2, group, *handle)
            # dispatch_weights slot carries grad_gate back to stage1.backward.
            return (
                grad_l1,
                grad_gate,
                None,
                None,
                dW2.to(w2.dtype),
                None,
                *((None,) * len(handle)),
            )


def fused_mega_moe_stage1(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    group: ProcessGroup,
):
    """Stage1 gate-up. Returns ``(l1_out, dispatch_weights, handle)`` to feed stage2.

    ``dispatch_weights`` and ``handle`` are opaque forward state; pass them straight into
    :func:`fused_mega_moe_stage2`.
    """
    l1_out, dispatch_weights, *handle = FusedMegaMoEStage1Function.apply(x, topk_idx, topk_weights, w1, group)
    return l1_out, dispatch_weights, tuple(handle)


def fused_mega_moe_stage2(
    l1_out: torch.Tensor,
    dispatch_weights: torch.Tensor,
    handle: tuple,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w2: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Stage2 gate-down. Consumes stage1's ``(l1_out, dispatch_weights, handle)``; returns y."""
    return FusedMegaMoEStage2Function.apply(
        l1_out, dispatch_weights, topk_idx, topk_weights, w2, group, *handle
    )
