###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused grouped GEMM + combine PUSH + topk reduce custom op (wraps grouped_gemm_combine_bf16)."""

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    KernelBackend,
    TuneCache,
)

_SUPPORTED_DTYPES = (torch.bfloat16,)


def _flydsl_kernel():
    """Lazy import — keep this module importable when FlyDSL is absent."""
    from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (
        grouped_gemm_combine_bf16,
    )

    return grouped_gemm_combine_bf16


def _active_symm():
    """Lazy import — fetch the active symmetric workspace (no-group call)."""
    from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

    return get_symm_buffer_for_mega_moe()


class GroupedGEMMCombineFlyDSLBackend(KernelBackend):
    """FlyDSL fused grouped BF16 GEMM + combine PUSH + topk reduce (3-role)."""

    @staticmethod
    def can_handle(
        act: torch.Tensor,
        weight: torch.Tensor,
        layout: str,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= act.dim() == 2 and weight.dim() == 3
        supported &= act.dtype in _SUPPORTED_DTYPES and weight.dtype in _SUPPORTED_DTYPES
        supported &= layout in ("nt", "nn", "tn")
        # K (contraction) must match between act and weight per layout
        act_k = act.shape[0] if layout == "tn" else act.shape[1]
        w_k = weight.shape[2] if layout == "nt" else weight.shape[1]
        supported &= act_k == w_k
        return supported

    @staticmethod
    def execute(
        act: torch.Tensor,
        weight: torch.Tensor,
        handle: list,
        topk_indices: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        grad_gate: Optional[torch.Tensor],
        layout: str,
        BM: int,
        BN: int,
        num_combine_cu: int,
        num_reduce_cu: int,
        autotune: bool,
        **kwargs,
    ):
        kernel = _flydsl_kernel()
        # handle[5] = tile_to_expert; symm buffer + autotune are kernel-internal.
        # num_combine_cu/num_reduce_cu are @autotune-swept inside the kernel, not forwarded here.
        return kernel(
            act,
            weight,
            handle,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            grad_gate=grad_gate,
            layout=layout,
            BM=int(BM),
            BN=int(BN),
        )


_GROUPED_GEMM_COMBINE_BACKENDS = {
    # autotune is kernel-internal; skip framework-level backend profiling
    BackendType.FLYDSL: BackendEntry(GroupedGEMMCombineFlyDSLBackend, autotune=False),
}


class GroupedGEMMCombineKernelDispatcher(AutoKernelDispatcher):
    _backends = _GROUPED_GEMM_COMBINE_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, act, weight, layout, BM, BN, num_reduce_cu, **kwargs):
        G = weight.shape[0]
        n_idx, k_idx = (1, 2) if layout == "nt" else (2, 1)
        N, K = weight.shape[n_idx], weight.shape[k_idx]
        num_max_pool_tokens = act.shape[1] if layout == "tn" else act.shape[0]
        return (G, N, K, num_max_pool_tokens, BM, BN, int(num_reduce_cu), layout, act.dtype)


_torch_custom_op_wrapper = torch.library.custom_op

# Kernel writes only the active symm workspace via raw pointers (untracked, not args); outputs are fresh


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_combine_impl",
    mutates_args=(),
    device_types="cuda",
)
def grouped_gemm_combine_impl(
    act: torch.Tensor,
    weight: torch.Tensor,
    handle: list[torch.Tensor],
    default_backend: int,
    topk_indices: torch.Tensor,
    topk_weights: Optional[torch.Tensor] = None,
    grad_gate: Optional[torch.Tensor] = None,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    num_combine_cu: int = 64,
    num_reduce_cu: int = 0,
    autotune: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused grouped BF16 GEMM + combine PUSH + topk reduce; returns (output, d_topk_w)."""
    default_backend_enum = BackendType(default_backend)

    # num_combine_cu/num_reduce_cu forwarded to the kernel (0 -> per-layout default);
    # MEGA_GGC_* env still overrides for sweeps. autotune is kernel-internal.
    kwargs = dict(
        act=act,
        weight=weight,
        handle=handle,
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        grad_gate=grad_gate,
        layout=layout,
        BM=BM,
        BN=BN,
        num_combine_cu=num_combine_cu,
        num_reduce_cu=num_reduce_cu,
        autotune=autotune,
    )

    output, d_topk_w = GroupedGEMMCombineKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)
    # custom-op returns can't be Optional -> [0] placeholder when the gate is off
    if d_topk_w is None:
        d_topk_w = act.new_empty((0,), dtype=torch.float32)
    return output, d_topk_w


@grouped_gemm_combine_impl.register_fake
def grouped_gemm_combine_impl_meta(
    act: torch.Tensor,
    weight: torch.Tensor,
    handle: list[torch.Tensor],
    default_backend: int,
    topk_indices: torch.Tensor,
    topk_weights: Optional[torch.Tensor] = None,
    grad_gate: Optional[torch.Tensor] = None,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    num_combine_cu: int = 64,
    num_reduce_cu: int = 0,
    autotune: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert act.dim() == 2, f"act must be 2D, got {act.shape}"
    assert weight.dim() == 3, f"weight must be 3D, got {weight.shape}"
    assert act.dtype in _SUPPORTED_DTYPES, f"act must be bf16, got {act.dtype}"
    assert weight.dtype in _SUPPORTED_DTYPES, f"weight must be bf16, got {weight.dtype}"
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    # output [num_tokens, N] + d_topk_w [num_combine_slots or 0] sized from active symm (N from weight)
    N = weight.shape[1] if layout == "nt" else weight.shape[2]
    symm = _active_symm()
    output = act.new_empty((int(symm.num_tokens), N), dtype=torch.bfloat16)
    gate_slots = int(symm.num_combine_slots) if grad_gate is not None else 0
    d_topk_w = act.new_empty((gate_slots,), dtype=torch.float32)
    return output, d_topk_w
