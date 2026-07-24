###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE backward custom op: conjugate of forward via Dispatch<->Combine duality (FlyDSL)."""

from typing import List, Tuple

import torch
from torch.distributed.distributed_c10d import _resolve_process_group

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import grouped_gemm_variable_k_bf16
from primus_turbo.flydsl.mega import (
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
    swiglu_backward_flydsl_kernel,
)
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    KernelBackend,
    TuneCache,
)

_SUPPORTED_DTYPES = (torch.bfloat16,)

# dispatch handle layout (see dispatch_prologue return + pool_src_slot snapshot):
# 0-5 send/dispatch tables + tile_to_expert, 6 real_count_per_expert,
# 7 num_tokens_per_expert_prefix, 8 num_tile_blocks, 9-11 combine_recv_*, 12 pool_src_slot.
_HANDLE_LEN = 13
_H_NUM_TILE_BLOCKS = 8
_H_REAL_COUNT_PER_EXPERT = 6
_H_NUM_TOKENS_PER_EXPERT_PREFIX = 7


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
        dispatch_weights_in_buf: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_idx: torch.Tensor,
        handle: list,
        group,
        num_tokens: int,
        num_topk: int,
        **kwargs,
    ):

        # ABI guard: catch a kernel return-order change loudly.
        assert len(handle) == _HANDLE_LEN, f"dispatch handle len {len(handle)} != {_HANDLE_LEN}; ABI changed"
        real_count_per_expert = handle[_H_REAL_COUNT_PER_EXPERT]
        num_tokens_per_expert_prefix = handle[_H_NUM_TOKENS_PER_EXPERT_PREFIX]
        in_handle = tuple(handle)

        # int64 end-to-end (combine reads topk i64)
        topk_idx = topk_idx.to(torch.int64)
        dy = grad_y.contiguous().to(torch.bfloat16)

        # L2 dgrad: cross-rank dispatch PUSH + grouped GEMM (nn)
        grad_swiglu, dispatch_l2_grad, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
            dy,
            w2,
            group,
            handle=in_handle,
            layout="nn",
        )

        # SwiGLU^T (re-inject routing weight) + gate grad
        grad_l1, grad_gate, act_weighted = swiglu_backward_flydsl_kernel(
            grad_swiglu,
            l1_out,
            scale=dispatch_weights_in_buf,
            return_gate=True,
            return_act_w=True,
            # bound by THIS handle's tile count (per-forward, not shared symm)
            num_tile_blocks=handle[_H_NUM_TILE_BLOCKS],
        )

        dW2 = grouped_gemm_variable_k_bf16(
            dispatch_l2_grad,
            act_weighted,
            num_tokens_per_expert_prefix,
            masked_k=real_count_per_expert,
            trans_c=False,
        )

        # L1 dgrad (grad_l1 @ w1, nn) + combine PUSH + dx reduce + grad_gate scatter
        dx, grad_topk_weights_flat = grouped_gemm_combine_bf16_flydsl_kernel(
            grad_l1,
            w1,
            handle,
            topk_indices=topk_idx.contiguous().view(-1),
            topk_weights=None,
            grad_gate=grad_gate,
            layout="nn",
        )

        # dW1 = pool(x)^T @ grad_l1 (variable-K tn wgrad; re-dispatch saved x)
        dW1, _, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
            saved_x,
            grad_l1,
            group,
            handle=in_handle,
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
    dispatch_weights_in_buf: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: List[torch.Tensor],
    group_name: str,
    num_tokens: int,
    num_topk: int,
    default_backend: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _resolve_process_group(group_name)
    default_backend_enum = BackendType(default_backend)

    kwargs = dict(
        grad_y=grad_y,
        saved_x=saved_x,
        l1_out=l1_out,
        dispatch_weights_in_buf=dispatch_weights_in_buf,
        w1=w1,
        w2=w2,
        topk_idx=topk_idx,
        handle=handle,
        group=group,
        num_tokens=num_tokens,
        num_topk=num_topk,
    )
    return FusedMegaMoEBackwardKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)


@_fused_mega_moe_backward.register_fake
def _fused_mega_moe_backward_meta(
    grad_y: torch.Tensor,
    saved_x: torch.Tensor,
    l1_out: torch.Tensor,
    dispatch_weights_in_buf: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: List[torch.Tensor],
    group_name: str,
    num_tokens: int,
    num_topk: int,
    default_backend: int,
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
    dispatch_weights_in_buf: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: tuple,
    group: torch.distributed.group,
    num_tokens: int,
    num_topk: int,
    default_backend: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused MoE backward (conjugate of forward via Dispatch<->Combine duality).

    Returns (dx, grad_topk_weights, dW1, dW2).
    """
    return _fused_mega_moe_backward(
        grad_y,
        saved_x,
        l1_out,
        dispatch_weights_in_buf,
        w1,
        w2,
        topk_idx,
        list(handle),
        group.group_name,
        num_tokens,
        num_topk,
        default_backend,
    )
