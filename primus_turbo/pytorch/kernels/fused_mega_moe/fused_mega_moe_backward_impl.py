###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE backward custom op: conjugate of forward via Dispatch<->Combine duality (FlyDSL)."""

from typing import List, Optional, Tuple

import torch
from torch.distributed.distributed_c10d import _resolve_process_group

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import (
    grouped_gemm_bf16_variable_k_flydsl_kernel,
)
from primus_turbo.flydsl.mega import (
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
)
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import DispatchHandle
from primus_turbo.flydsl.utils.glu_activation import (
    GLUActivation,
    activation_constexpr,
)
from primus_turbo.flydsl.utils.swiglu_kernel import swiglu_backward_flydsl_kernel
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    KernelBackend,
    TuneCache,
)

_SUPPORTED_DTYPES = (torch.bfloat16,)


def compute_dW2(
    dispatch_l2_grad: torch.Tensor, act_weighted: torch.Tensor, handle: DispatchHandle
) -> torch.Tensor:
    """dW2 = dispatched(dy)^T @ act_weighted, contracted over each expert's unpadded rows."""
    return grouped_gemm_bf16_variable_k_flydsl_kernel(
        dispatch_l2_grad,
        act_weighted,
        handle.num_tokens_per_expert_prefix,
        masked_k=handle.num_tokens_per_expert,
        a_row_idx=handle.pool_row_to_recv_token,
    )


class FusedMegaMoEBackwardFlyDSLBackend(KernelBackend):
    """FlyDSL fused MoE backward: L2 dgrad (nn) + SwiGLU^T + dW2 + L1 dgrad combine (nn) + dW1 (tn)."""

    @staticmethod
    def can_handle(
        grad_y: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= grad_y.dim() == 2 and w1.dim() == 3 and w2.dim() == 3
        supported &= grad_y.dtype in _SUPPORTED_DTYPES
        supported &= w1.dtype in _SUPPORTED_DTYPES and w2.dtype in _SUPPORTED_DTYPES
        return supported

    @staticmethod
    def execute(
        grad_y: torch.Tensor,
        saved_x: torch.Tensor,
        l1_out: torch.Tensor,
        dispatch_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_idx: torch.Tensor,
        handle: list,
        group,
        num_tokens: int,
        num_topk: int,
        activation: Optional[List[float]] = None,
        **kwargs,
    ):
        handle = DispatchHandle(*handle)
        dy = grad_y.contiguous().to(torch.bfloat16)

        # L2 dgrad: cross-rank dispatch PUSH + grouped GEMM (nn)
        grad_swiglu, dispatch_l2_grad, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
            dy,
            w2,
            group,
            handle=handle,
            layout="nn",
        )

        # SwiGLU^T (re-inject routing weight) + gate grad
        grad_l1, grad_gate, act_weighted = swiglu_backward_flydsl_kernel(
            grad_swiglu,
            l1_out,
            scale=dispatch_weights,
            return_gate=True,
            return_act_w=True,
            # bound by THIS handle's tile count (per-forward, not shared symm)
            num_tile_blocks=handle.num_tile_blocks,
            activation=activation,
        )

        dW2 = compute_dW2(dispatch_l2_grad, act_weighted, handle)

        # L1 dgrad (grad_l1 @ w1, nn) + combine PUSH + dx reduce + grad_gate scatter
        dx, grad_topk_weights_flat = grouped_gemm_combine_bf16_flydsl_kernel(
            grad_l1,
            w1,
            handle,
            topk_indices=topk_idx.contiguous().view(-1),
            grad_gate=grad_gate,
            layout="nn",
        )

        # dW1 = pool(x)^T @ grad_l1 (variable-K tn wgrad; re-dispatch saved x)
        dW1, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
            saved_x,
            grad_l1,
            group,
            handle=handle,
            layout="tn",
            trans_c=True,
        )

        # reshape the combine-reduce gate output to [num_tokens, num_topk]
        grad_topk_weights = grad_topk_weights_flat.view(num_tokens, num_topk)
        return dx, grad_topk_weights, dW1.to(w1.dtype), dW2.to(w2.dtype)


_FUSED_MEGA_MOE_BACKWARD_BACKENDS = {
    # autotune is kernel-internal; skip framework-level backend profiling
    BackendType.FLYDSL: BackendEntry(FusedMegaMoEBackwardFlyDSLBackend, autotune=False),
}


class FusedMegaMoEBackwardKernelDispatcher(AutoKernelDispatcher):
    _backends = _FUSED_MEGA_MOE_BACKWARD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, grad_y, w1, w2, num_tokens, num_topk, **kwargs):
        G = w1.shape[0]
        # w1: [G, 2I, K], w2: [G, N, I]
        N1, K1 = w1.shape[1], w1.shape[2]
        N2 = w2.shape[1]
        return (G, N1, K1, N2, num_tokens, num_topk, grad_y.dtype)


_torch_custom_op_wrapper = torch.library.custom_op

# Kernel writes only the active symm workspace via raw pointers (untracked, not args); outputs are fresh


@_torch_custom_op_wrapper(
    "primus_turbo::fused_mega_moe_backward",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mega_moe_backward(
    grad_y: torch.Tensor,
    saved_x: torch.Tensor,
    l1_out: torch.Tensor,
    dispatch_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: List[torch.Tensor],
    group_name: str,
    num_tokens: int,
    num_topk: int,
    default_backend: int,
    activation: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _resolve_process_group(group_name)
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))

    kwargs = dict(
        grad_y=grad_y,
        saved_x=saved_x,
        l1_out=l1_out,
        dispatch_weights=dispatch_weights,
        w1=w1,
        w2=w2,
        topk_idx=topk_idx,
        handle=handle,
        group=group,
        num_tokens=num_tokens,
        num_topk=num_topk,
        activation=activation,
    )
    return FusedMegaMoEBackwardKernelDispatcher.dispatch(default_backend_choice, None, **kwargs)


@_fused_mega_moe_backward.register_fake
def _fused_mega_moe_backward_meta(
    grad_y: torch.Tensor,
    saved_x: torch.Tensor,
    l1_out: torch.Tensor,
    dispatch_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: List[torch.Tensor],
    group_name: str,
    num_tokens: int,
    num_topk: int,
    default_backend: int,
    activation: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # eager-only path (EP rendezvous can't be traced); approximate meta for completeness.
    hidden = grad_y.shape[1]
    dx = grad_y.new_empty((num_tokens, hidden), dtype=torch.bfloat16)
    grad_topk_weights = grad_y.new_empty((num_tokens, num_topk), dtype=torch.float32)
    dW1 = w1.new_empty(w1.shape, dtype=w1.dtype)
    dW2 = w2.new_empty(w2.shape, dtype=w2.dtype)
    return dx, grad_topk_weights, dW1, dW2


def fused_mega_moe_backward_impl(
    grad_y: torch.Tensor,
    saved_x: torch.Tensor,
    l1_out: torch.Tensor,
    dispatch_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: DispatchHandle,
    group: torch.distributed.group,
    num_tokens: int,
    num_topk: int,
    default_backend: int,
    activation: Optional[GLUActivation] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused MoE backward (conjugate of forward via Dispatch<->Combine duality).

    Returns (dx, grad_topk_weights, dW1, dW2).
    """
    return _fused_mega_moe_backward(
        grad_y,
        saved_x,
        l1_out,
        dispatch_weights,
        w1,
        w2,
        topk_idx,
        list(handle),
        group.group_name,
        num_tokens,
        num_topk,
        default_backend,
        None if activation is None else list(activation_constexpr(activation)),
    )
