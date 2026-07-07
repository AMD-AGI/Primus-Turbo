###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH + grouped BF16 GEMM custom op (wraps dispatch_grouped_gemm_bf16)."""

from typing import List, Optional, Tuple

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


def _flydsl_kernel():
    """Lazy import — keep this module importable when FlyDSL is absent."""
    from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (
        dispatch_grouped_gemm_bf16,
    )

    return dispatch_grouped_gemm_bf16


class DispatchGroupedGEMMFlyDSLBackend(KernelBackend):
    """FlyDSL fused cross-rank dispatch PUSH + grouped BF16 GEMM (nt / nn / tn)."""

    @staticmethod
    def can_handle(
        x: torch.Tensor,
        l1_weights: torch.Tensor,
        layout: str,
        **kwargs,
    ) -> bool:
        supported = True
        # tn wgrad passes a 2D activation as l1_weights (rhs); nt/nn pass 3D weights
        weight_dim = 2 if layout == "tn" else 3
        supported &= x.dim() == 2 and l1_weights.dim() == weight_dim
        supported &= x.dtype in _SUPPORTED_DTYPES and l1_weights.dtype in _SUPPORTED_DTYPES
        supported &= layout in ("nt", "nn", "tn")
        return supported

    @staticmethod
    def execute(
        x: torch.Tensor,
        l1_weights: torch.Tensor,
        group,
        handle: List[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        layout: str,
        num_dispatch_cu: int,
        trans_c: bool,
        out_dtype: torch.dtype,
        **kwargs,
    ):
        kernel = _flydsl_kernel()
        # forward: handle=None builds the symm workspace; reuse: re-feed the tuple.
        in_handle = tuple(handle) if len(handle) > 0 else None
        out, dispatch_x_in_buf, dispatch_weights_in_buf, full_handle = kernel(
            x,
            l1_weights,
            group,
            handle=in_handle,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layout=layout,
            num_dispatch_cu=int(num_dispatch_cu),
            trans_c=trans_c,
            out_dtype=out_dtype,
        )
        # forward built a fresh handle -> return it; reuse handle is an input -> return [] (no alias)
        handle_out = list(full_handle) if in_handle is None else []
        return out, dispatch_x_in_buf, dispatch_weights_in_buf, handle_out


_DISPATCH_GROUPED_GEMM_BACKENDS = {
    # autotune is kernel-internal; skip framework-level backend profiling
    BackendType.FLYDSL: BackendEntry(DispatchGroupedGEMMFlyDSLBackend, autotune=False),
}


class DispatchGroupedGEMMKernelDispatcher(AutoKernelDispatcher):
    _backends = _DISPATCH_GROUPED_GEMM_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, x, l1_weights, layout, num_dispatch_cu, **kwargs):
        num_tokens = x.shape[0]
        if layout == "tn":
            # tn: l1_weights is a 2D activation (pool_rows, N)
            pool_rows, N = l1_weights.shape
            return (pool_rows, N, num_tokens, int(num_dispatch_cu), layout, x.dtype)
        G = l1_weights.shape[0]
        n_idx, k_idx = (1, 2) if layout == "nt" else (2, 1)
        N, K = l1_weights.shape[n_idx], l1_weights.shape[k_idx]
        return (G, N, K, num_tokens, int(num_dispatch_cu), layout, x.dtype)


_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper(
    "primus_turbo::dispatch_grouped_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _dispatch_grouped_gemm(
    x: torch.Tensor,
    l1_weights: torch.Tensor,
    group_name: str,
    default_backend: int,
    handle: List[torch.Tensor],
    topk_idx: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    layout: str,
    num_dispatch_cu: int,
    trans_c: bool,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    group = _resolve_process_group(group_name)
    default_backend_enum = BackendType(default_backend)

    kwargs = dict(
        x=x,
        l1_weights=l1_weights,
        group=group,
        handle=handle,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        layout=layout,
        num_dispatch_cu=num_dispatch_cu,
        trans_c=trans_c,
        out_dtype=out_dtype,
    )
    return DispatchGroupedGEMMKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)


@_dispatch_grouped_gemm.register_fake
def _dispatch_grouped_gemm_meta(
    x: torch.Tensor,
    l1_weights: torch.Tensor,
    group_name: str,
    default_backend: int,
    handle: List[torch.Tensor],
    topk_idx: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    layout: str,
    num_dispatch_cu: int,
    trans_c: bool,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    # eager-only path (EP rendezvous can't be traced); approximate meta for completeness.
    if layout == "tn":
        N = l1_weights.shape[1]  # 2D activation (pool_rows, N)
    else:
        N = l1_weights.shape[1] if layout == "nt" else l1_weights.shape[2]
    out = x.new_empty((0, N), dtype=x.dtype)
    dispatch_x_in_buf = x.new_empty((0, x.shape[1]), dtype=x.dtype)
    dispatch_weights_in_buf = x.new_empty((0,), dtype=torch.float32)
    return out, dispatch_x_in_buf, dispatch_weights_in_buf, []


def dispatch_grouped_gemm_impl(
    x: torch.Tensor,
    l1_weights: torch.Tensor,
    group: torch.distributed.group,
    default_backend: int,
    *,
    topk_idx: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    handle: tuple | None = None,
    layout: str = "nt",
    num_dispatch_cu: int = 16,
    trans_c: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Fused dispatch PUSH + grouped BF16 GEMM; returns (output, dispatch_x_in_buf, dispatch_weights_in_buf, handle)."""
    out, dispatch_x_in_buf, dispatch_weights_in_buf, handle_out = _dispatch_grouped_gemm(
        x,
        l1_weights,
        group.group_name,
        default_backend,
        list(handle) if handle is not None else [],
        topk_idx,
        topk_weights,
        layout,
        num_dispatch_cu,
        trans_c,
        out_dtype,
    )
    # forward: use the kernel-built handle; reuse: keep the caller's handle (op returned [])
    return out, dispatch_x_in_buf, dispatch_weights_in_buf, (handle_out if handle is None else handle)
