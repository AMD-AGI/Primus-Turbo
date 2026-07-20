###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trainable mega MoE with MXFP8 forward + partial-fp8 backward (autograd Function).

SKELETON. This is the fp8 sibling of the bf16 ``mega_moe_fused`` (``mega_moe_fused.py``). Unlike
bf16 it does NOT go through the ``pytorch/kernels/mega_moe`` custom_op / AutoKernelDispatcher
layer: the orchestration is inlined here directly on the FlyDSL fp8 kernels, because the fp8 path
carries state the custom_op schema can't hold -- optional caller-supplied weight-prequant tuples
(``w1_fp8`` / ``w2_fp8`` / ``w2t_fp8`` / ``w1t_fp8``), reuse of the forward's live symmetric buffer
in backward, and host ``synchronize()`` + ``group.barrier()`` rendezvous.

Pipeline (to be ported from ``/perf_apps/xiaoming/Primus-Turbo``
``primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py``):
  * Forward: L1 fused mxfp8 dispatch+fc1 (NT) -> SwiGLU (bf16) -> L2 fp8 combine.
  * Backward (Dispatch<->Combine duality):
      STEP1  dispatch(dy)+fc2 dgrad (NN) in MXFP8 -> grad_swiglu + rowwise-fp8 dispatched-dy pool.
      STEP2  SwiGLU^T (bf16), re-inject routing weight, gate grad.
      dW2    variable-K wgrad (MXFP8), a-operand requant-fused from the STEP1 fp8 pool.
      STEP3  fc1 dgrad (fp8 GEMM) + combine/reduce (bf16, byte-bound).
      dW1    variable-K wgrad (MXFP8), LOCAL -- reuses the FORWARD-dispatched fc1-input pool.

The bf16 ``w1`` / ``w2`` stay the differentiable inputs; the ``*_fp8`` args are non-diff derived
weight preps a stateful holder (``MegaMoEFP8``) maintains version-keyed and hands in.
"""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

# --- PORT: uncomment as the fp8 flydsl kernels + core deps land (see fp8/__init__.py) ---
# from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue_flydsl_kernel
# from primus_turbo.flydsl.mega.swiglu_kernel import (
#     swiglu_flydsl_kernel,
#     swiglu_backward_flydsl_kernel,
# )
# from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
# from primus_turbo.flydsl.mega.fp8 import (
#     dispatch_grouped_gemm_mxfp8,
#     dispatch_grouped_gemm_mxfp8_bwd,
#     grouped_gemm_combine_fp8,
#     grouped_gemm_combine_mxfp8_bwd,
#     quantize_grouped_weight_mxfp8,
#     quantize_grouped_weight_mxfp8_cached,
#     quantize_rowwise_mxfp8,
#     quantize_rowwise_mxfp8_flydsl,
#     colwise_grouped_meta,
#     colwise_quant_mxfp8_grouped_flydsl,
#     colwise_requant_mxfp8_grouped_fp8in_flydsl,
# )
# from primus_turbo.pytorch.core.backend import BackendType
# from primus_turbo.pytorch.core.low_precision import (
#     ScalingGranularity,
#     float8_e4m3,
#     float8_e5m2,
# )
# from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
#     grouped_gemm_fp8_variable_k_impl,
# )

__all__ = [
    "MegaMoEFusedFP8Function",
    "mega_moe_fused_fp8",
    "prepare_w1t_dgrad_fp8",
    "prepare_w2t_dgrad_fp8",
]

_NOT_PORTED = (
    "mega MoE fp8 kernels not ported yet; see "
    "/perf_apps/xiaoming/Primus-Turbo/primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py"
)


def prepare_w2t_dgrad_fp8(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 quant of ``w2^T`` (``[G,I,H]``) for the backward STEP1 fc2-dgrad NT-reuse GEMM.

    Returns ``(w2tq [G,I,H] fp8, w2ts [G,I,H//32] raw E8M0)``. STATIC weight prep -- a stateful
    holder (``MegaMoEFP8``) runs it once per ``w2._version`` and passes it via the op's ``w2t_fp8``
    arg so the transpose+quant never runs inside backward.
    """
    raise NotImplementedError(_NOT_PORTED)


def prepare_w1t_dgrad_fp8(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 quant of ``w1^T`` (``[G,H,2I]``) for the backward STEP3 fc1-dgrad NT-reuse.

    Returns ``(w1tq [G,H,2I] fp8, w1ts [G,H,2I//32] raw E8M0)``. Mirrors
    :func:`prepare_w2t_dgrad_fp8`; owned version-keyed by ``MegaMoEFP8`` and passed via ``w1t_fp8``.
    """
    raise NotImplementedError(_NOT_PORTED)


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
        w1_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        w2_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        w2t_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        w1t_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # PORT: L1 fused mxfp8 dispatch+fc1 -> SwiGLU -> L2 fp8 combine; stash handle + l1 +
        # dispatch_weights + w1/w2/topk_idx + (w2t_fp8/w1t_fp8/pool_x_fp8 on ctx) for backward.
        raise NotImplementedError(_NOT_PORTED)

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        # PORT: STEP1 fp8 fork; dW2 + dW1 MXFP8 variable-K (LOCAL); STEP3 fc1-dgrad(fp8)+combine(bf16).
        # grads align with forward inputs: (x, topk_idx, topk_weights, w1, w2, group, block_m,
        # block_n, w1_fp8, w2_fp8, w2t_fp8, w1t_fp8) -- the *_fp8 / block_* / group are None.
        raise NotImplementedError(_NOT_PORTED)


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
    w1_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    w2_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    w2t_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    w1t_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """One fully fused mega MoE forward (MXFP8) that joins autograd; backward fp8-izes dW1/dW2.

    fp8 sibling of :func:`primus_turbo.pytorch.ops.moe.mega_moe_fused.mega_moe_fused`. The optional
    ``w1_fp8`` / ``w2_fp8`` / ``w2t_fp8`` / ``w1t_fp8`` are caller-owned grouped mxfp8 weight preps
    ``(w_fp8, w_scale)`` -- pass them from a stateful holder (``MegaMoEFP8``, version-keyed) so the
    forward/backward skip re-quantizing weights. Leave ``None`` to prepare lazily inside.
    """
    return MegaMoEFusedFP8Function.apply(
        x, topk_idx, topk_weights, w1, w2, group, block_m, block_n,
        w1_fp8, w2_fp8, w2t_fp8, w1t_fp8,
    )
