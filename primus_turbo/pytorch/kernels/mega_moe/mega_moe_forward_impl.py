###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE forward custom op: dispatch grouped GEMM + SwiGLU + grouped GEMM combine (FlyDSL)."""

from typing import List, Tuple

import torch
from torch.distributed.distributed_c10d import _resolve_process_group

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    KernelBackend,
    TuneCache,
)

_SUPPORTED_DTYPES = (torch.bfloat16,)

# dispatch handle layout (see dispatch_prologue_kernel.py return order).
_HANDLE_LEN = 9
_H_NUM_TOKENS_PER_EXPERT = 7
_H_NUM_TOKENS_PER_EXPERT_PREFIX = 8


class MegaMoEForwardFlyDSLBackend(KernelBackend):
    """FlyDSL fused MoE forward: dispatch grouped GEMM (nt) + SwiGLU + grouped GEMM combine (nt)."""

    @staticmethod
    def can_handle(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= x.dim() == 2 and w1.dim() == 3 and w2.dim() == 3
        supported &= x.dtype in _SUPPORTED_DTYPES
        supported &= w1.dtype in _SUPPORTED_DTYPES and w2.dtype in _SUPPORTED_DTYPES
        return supported

    @staticmethod
    def execute(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        group,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        layout: str,
        **kwargs,
    ):
        # lazy import — keep this module importable when FlyDSL is absent
        from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (
            dispatch_grouped_gemm_bf16,
        )
        from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (
            grouped_gemm_combine_bf16,
        )
        from primus_turbo.flydsl.mega.swiglu_kernel import swiglu

        # int64 end-to-end (combine reads topk i64)
        topk_idx = topk_idx.to(torch.int64)

        # fused prologue + cross-rank dispatch PUSH + grouped L1 GEMM (nt)
        l1_out, _, dispatch_weights_in_buf, handle = dispatch_grouped_gemm_bf16(
            x,
            w1,
            group,
            handle=None,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layout=layout,
        )

        act = swiglu(l1_out)

        # fused grouped L2 GEMM + combine PUSH + topk reduce
        y, _ = grouped_gemm_combine_bf16(
            act,
            w2,
            handle,
            topk_indices=topk_idx.contiguous().view(-1),
            topk_weights=topk_weights.to(torch.float32).contiguous().view(-1),
            layout=layout,
        )

        # ABI guard: catch a kernel return-order change loudly.
        assert len(handle) == _HANDLE_LEN, f"dispatch handle len {len(handle)} != {_HANDLE_LEN}; ABI changed"
        # clone: also returned inside handle -> break aliasing (custom-op forbids aliased returns)
        num_tokens_per_expert = handle[_H_NUM_TOKENS_PER_EXPERT].clone()
        num_tokens_per_expert_prefix = handle[_H_NUM_TOKENS_PER_EXPERT_PREFIX].clone()
        return (
            y,
            l1_out,
            dispatch_weights_in_buf,
            num_tokens_per_expert,
            num_tokens_per_expert_prefix,
            list(handle),
        )


_MEGA_MOE_FORWARD_BACKENDS = {
    # autotune is kernel-internal; skip framework-level backend profiling
    BackendType.FLYDSL: BackendEntry(MegaMoEForwardFlyDSLBackend, autotune=False),
}


class MegaMoEForwardKernelDispatcher(AutoKernelDispatcher):
    _backends = _MEGA_MOE_FORWARD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, x, w1, w2, layout, **kwargs):
        num_tokens = x.shape[0]
        num_topk = kwargs["topk_idx"].shape[-1]
        G = w1.shape[0]
        # w1: [G, 2I, K], w2: [G, N, I] for nt layout
        n1_idx, k1_idx = (1, 2) if layout == "nt" else (2, 1)
        N1, K1 = w1.shape[n1_idx], w1.shape[k1_idx]
        n2_idx = 1 if layout == "nt" else 2
        N2 = w2.shape[n2_idx]
        return (G, N1, K1, N2, num_tokens, num_topk, layout, x.dtype)


_torch_custom_op_wrapper = torch.library.custom_op

# Kernel writes only the active symm workspace via raw pointers (untracked, not args); outputs are fresh


@_torch_custom_op_wrapper(
    "primus_turbo::mega_moe_forward",
    mutates_args=(),
    device_types="cuda",
)
def _mega_moe_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_name: str,
    default_backend: int,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    layout: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    group = _resolve_process_group(group_name)
    default_backend_enum = BackendType(default_backend)

    kwargs = dict(
        x=x,
        w1=w1,
        w2=w2,
        group=group,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        layout=layout,
    )
    return MegaMoEForwardKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)


@_mega_moe_forward.register_fake
def _mega_moe_forward_meta(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_name: str,
    default_backend: int,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    layout: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    # eager-only path (EP rendezvous can't be traced); approximate meta for completeness.
    N2 = w2.shape[1] if layout == "nt" else w2.shape[2]
    two_I = w1.shape[1] if layout == "nt" else w1.shape[2]
    num_tokens = x.shape[0]
    y = x.new_empty((num_tokens, N2), dtype=torch.bfloat16)
    l1_out = x.new_empty((0, two_I), dtype=torch.bfloat16)
    dispatch_weights_in_buf = x.new_empty((0,), dtype=torch.float32)
    num_tokens_per_expert = x.new_empty((0,), dtype=torch.int32)
    num_tokens_per_expert_prefix = x.new_empty((0,), dtype=torch.int32)
    return y, l1_out, dispatch_weights_in_buf, num_tokens_per_expert, num_tokens_per_expert_prefix, []


def mega_moe_forward_impl(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group: torch.distributed.group,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    layout: str,
    default_backend: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Fused MoE forward (dispatch grouped GEMM + SwiGLU + grouped GEMM combine).

    Returns (y, l1_out, dispatch_weights_in_buf, num_tokens_per_expert,
    num_tokens_per_expert_prefix, handle).
    """
    (
        y,
        l1_out,
        dispatch_weights_in_buf,
        num_tokens_per_expert,
        num_tokens_per_expert_prefix,
        handle,
    ) = _mega_moe_forward(
        x,
        w1,
        w2,
        group.group_name,
        default_backend,
        topk_idx,
        topk_weights,
        layout,
    )
    return (
        y,
        l1_out,
        dispatch_weights_in_buf,
        num_tokens_per_expert,
        num_tokens_per_expert_prefix,
        tuple(handle),
    )
