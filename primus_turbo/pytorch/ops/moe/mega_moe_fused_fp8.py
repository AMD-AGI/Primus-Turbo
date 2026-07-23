###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trainable mega MoE with MXFP8 forward + partial-fp8 backward (autograd Function).

Thin ``torch.autograd.Function`` that validates at the op boundary and delegates the actual
orchestration to ``mega_moe_forward_fp8_impl`` / ``mega_moe_backward_fp8_impl`` in
``pytorch/kernels/mega_moe``. Those impls are PLAIN functions, NOT the ``custom_op`` /
``AutoKernelDispatcher`` layer: the fp8 path carries state the custom_op schema can't hold --
reuse of the forward's live symmetric buffer in backward and derived non-tensor handles (the
bf16 mega MoE op is a plain autograd.Function for the same reason). The comm gates self-reset via
a device epoch (no host synchronize()+barrier() rendezvous), which removed the large-T reset-race
deadlock -- but the cross-rank symm-memory PUSH + device spin-wait handshake still does not compose
with CUDA-graph capture, so the op is NOT graph-captured.

Forward = L1 fused mxfp8 dispatch+fc1 NT -> SwiGLU -> L2 fp8 combine. Backward = STEP1
dispatch(dy)+fc2 dgrad -> STEP2 SwiGLU^T -> dW2 variable-K wgrad -> STEP3 fc1 dgrad + combine
(fp8-PUSH) -> dW1 variable-K wgrad (LOCAL).

The differentiable ``w1`` / ``w2`` weights stay the high-precision inputs; their mxfp8 (and
w1^T/w2^T dgrad) quant is maintained INSIDE the impls by version-keyed caches keyed on ``w._version``
(no caller prequant args).
"""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.pytorch.kernels.mega_moe import (
    mega_moe_backward_fp8_impl,
    mega_moe_forward_fp8_impl,
)

# This op file exports only its own final API (the autograd Function + its wrapper). Everything else
# (the per-stage ``_mxfp8_*`` helpers, ``_DW_FP8_FORMAT``, and the ``prepare_w1t/w2t_dgrad_fp8``
# weight-prep) lives in the kernels layer -- callers import those from
# ``primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl`` directly, not via this file.
__all__ = [
    "MegaMoEFusedFP8Function",
    "mega_moe_fused_fp8",
]


class MegaMoEFusedFP8Function(torch.autograd.Function):
    """Fused mega MoE, MXFP8 forward + fp8-dW1/dW2 backward. Joins the autograd graph."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        group: ProcessGroup,
        block_m: int,
        block_n: int,
    ) -> torch.Tensor:
        with torch.profiler.record_function("mega_moe_fp8_forward"):
            # Validate at the op boundary so failures point here, not deep in FlyDSL. w1 / w2 are the
            # differentiable weights; their mxfp8 quant is maintained INSIDE the impl by a
            # version-keyed cache (re-quantized only when a weight changes, reused across grad-accum).
            assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
            assert w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16, "w1/w2 must be bf16"

            num_topk = topk_idx.shape[-1]
            # int64 end-to-end (combine reads topk i64)
            topk_idx = topk_idx.to(torch.int64)
            ctx.set_materialize_grads(False)
            save_bwd = any(ctx.needs_input_grad)

            y, l1, dispatch_weights, pool_x_fp8, handle = mega_moe_forward_fp8_impl(
                x, topk_idx, topk_weights, w1, w2, group, block_m, block_n,
            )

            if save_bwd:
                # dispatch_weights / pool_x_fp8 are LIVE views into the shared symm pool -> clone
                # before a later stage (backward STEP1 dispatch(dy), or the next forward) overwrites
                # it. l1 is a fresh dispatch-GEMM output (not a symm view), so it needs no clone.
                dispatch_weights = dispatch_weights.clone()
                pool_x_fp8 = (pool_x_fp8[0].clone(), pool_x_fp8[1].clone())

                ctx.group = group
                ctx.num_tokens = x.shape[0]
                ctx.num_topk = num_topk
                ctx.block_m = block_m
                ctx.block_n = block_n
                ctx.handle_len = len(handle)
                # pool_x_fp8 (dW1's LOCAL fc1-input pool): a non-diff derived tensor pair -> stash on
                # ctx (not save_for_backward, which is for graph-tracked tensors). w1/w2 stay the
                # differentiable weights; backward re-derives their w1^T/w2^T dgrad quant version-keyed.
                ctx.pool_x_fp8 = pool_x_fp8
                ctx.save_for_backward(*handle, l1, dispatch_weights, w1, w2, topk_idx)
            return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        with torch.profiler.record_function("mega_moe_fp8_backward"):
            # grad_y is None when the output got no grad
            if grad_y is None:
                return (None,) * 8
            saved = ctx.saved_tensors
            handle = tuple(saved[: ctx.handle_len])
            l1, dispatch_weights, w1, w2, topk_idx = saved[ctx.handle_len :]

            dx, grad_topk_weights, dW1, dW2 = mega_moe_backward_fp8_impl(
                grad_y,
                l1,
                dispatch_weights,
                ctx.pool_x_fp8,
                w1,
                w2,
                topk_idx,
                handle,
                ctx.group,
                ctx.num_tokens,
                ctx.num_topk,
                ctx.block_m,
                ctx.block_n,
            )

            # grads align with forward inputs (x, topk_idx, topk_weights, w1, w2, group, block_m, block_n)
            return (dx, None, grad_topk_weights, dW1, dW2, None, None, None)


def mega_moe_fused_fp8(
    group: ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    block_m: int = 256,
    block_n: int = 256,
) -> torch.Tensor:
    """One fully fused mega MoE forward (MXFP8) that joins autograd; backward fp8-izes dW1/dW2.

    Pass the ``w1`` / ``w2`` weights directly -- the op maintains their mxfp8 quant internally with a
    version-keyed cache (re-quantized only on ``optim.step``), so there are no weight-prequant args.
    """
    return MegaMoEFusedFP8Function.apply(
        x, topk_idx, topk_weights, w1, w2, group, block_m, block_n,
    )
