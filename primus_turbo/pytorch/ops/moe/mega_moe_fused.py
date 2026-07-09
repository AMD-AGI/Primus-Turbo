###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fully fused mega MoE forward/backward (FlyDSL) over a single symmetric buffer."""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.symm_buffer import SymmBuffer
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.mega_moe import (
    mega_moe_backward_impl,
    mega_moe_forward_impl,
)

__all__ = [
    "SymmBuffer",
    "MegaMoEFusedFunction",
    "mega_moe_fused",
]


class MegaMoEFusedFunction(torch.autograd.Function):
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
        with torch.profiler.record_function("mega_moe_forward"):
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
            y, l1_out, dispatch_weights_in_buf, handle = mega_moe_forward_impl(
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
        with torch.profiler.record_function("mega_moe_backward"):
            # grad_y is None when the output got no grad
            if grad_y is None:
                return (None,) * 6
            saved_x, l1_out, dispatch_weights_in_buf, w1, w2, topk_idx, *handle = ctx.saved_tensors
            handle = tuple(handle)

            # fused MoE backward: L2 dgrad (nn) + SwiGLU^T + dW2 + L1 dgrad combine (nn) + dW1 (tn)
            dx, grad_topk_weights, dW1, dW2 = mega_moe_backward_impl(
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


def mega_moe_fused(
    group: ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """One fully fused mega MoE forward; the symmetric buffer is fetched (and cached) internally."""
    return MegaMoEFusedFunction.apply(x, topk_idx, topk_weights, w1, w2, group)
