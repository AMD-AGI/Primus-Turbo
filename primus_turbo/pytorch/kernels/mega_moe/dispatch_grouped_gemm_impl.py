###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH + grouped BF16 GEMM as a torch custom op.

Thin wrapper over the FlyDSL kernel ``dispatch_grouped_gemm_bf16``. The kernel owns the
whole symmetric workspace: forward (no handle) builds it from ``group`` + tensor shapes,
runs the fused prologue, and returns the handle (flat prologue tuple [0..10] plus this
layer's origin/meta snapshot tail [11..13]); reuse (handle given) re-feeds that tuple and
the kernel restores the snapshot tail device-side.

``_dispatch_grouped_gemm`` is a ``torch.library.custom_op``. A custom op cannot take a
``ProcessGroup``, so the public wrapper passes the group **by name** and the op resolves it
back. It returns ``(output, dispatch_weight_in_buf, handle)``: ``dispatch_weight_in_buf`` is
the dispatch-side symm buffer the caller consumes next (forward nt -> ``weight_recv_buf``
routing weight; backward nn -> ``pool`` = ``d_l2y``; tn -> unused). The built handle rides
the return on forward; on reuse it is ``[]`` (the input handle would alias the output) and
the caller keeps its own handle. Tiling is fixed at the kernel default block_m/n = 256.
"""

from typing import List, Optional, Tuple

import torch

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (
    dispatch_grouped_gemm_bf16,
)

_SUPPORTED_DTYPES = (torch.bfloat16,)


def _pg_name(group) -> str:
    """Stable registered name of a ProcessGroup (custom ops can't take the PG itself)."""
    return group.group_name


def _resolve_pg(group_name: str):
    from torch.distributed.distributed_c10d import _resolve_process_group

    return _resolve_process_group(group_name)


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
    autotune: bool,
    trans_c: bool,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Custom-op core: resolve the PG by name, run the FlyDSL kernel, return the standard
    4-output API ``(output, dispatch_x_in_buf, dispatch_weights_in_buf, handle)`` (handle only
    on forward; ``[]`` on reuse). ``dispatch_x_in_buf`` = the dispatched pool (nn dgrad: d_l2y);
    ``dispatch_weights_in_buf`` = the per-pool-row routing weight (fwd swiglu scale)."""
    group = _resolve_pg(group_name)
    in_handle = tuple(handle) if len(handle) > 0 else None
    out, dispatch_x_in_buf, dispatch_weights_in_buf, full_handle = dispatch_grouped_gemm_bf16(
        x,
        l1_weights,
        group,
        handle=in_handle,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        layout=layout,
        num_dispatch_cu=int(num_dispatch_cu),
        autotune=bool(autotune),
        trans_c=trans_c,
        out_dtype=out_dtype,
    )
    # forward built a fresh handle -> return it; reuse handle is an input -> return [] (no alias)
    handle_out = list(full_handle) if in_handle is None else []
    return out, dispatch_x_in_buf, dispatch_weights_in_buf, handle_out


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
    autotune: bool,
    trans_c: bool,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    # eager-only path (EP rendezvous can't be traced); approximate meta for completeness.
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
    autotune: bool = False,
    trans_c: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Fused cross-rank dispatch PUSH + grouped BF16 GEMM (nt / nn / tn).

    Forward (``handle=None``): the kernel builds the active symm workspace from ``group`` +
    shapes, runs the fused prologue (incl. the origin/meta snapshot tail), and returns the
    handle. Reuse (``handle`` given): the backward NN dgrad / TN wgrads ride the forward's
    plan; the kernel restores the snapshot tail device-side.

    Returns the standard 4-output API ``(output, dispatch_x_in_buf, dispatch_weights_in_buf,
    handle)`` (tiling fixed at block_m/n = 256)."""
    out, dispatch_x_in_buf, dispatch_weights_in_buf, handle_out = _dispatch_grouped_gemm(
        x,
        l1_weights,
        _pg_name(group),
        default_backend,
        list(handle) if handle is not None else [],
        topk_idx,
        topk_weights,
        layout,
        num_dispatch_cu,
        autotune,
        trans_c,
        out_dtype,
    )
    # forward: use the kernel-built handle; reuse: keep the caller's handle (op returned [])
    return out, dispatch_x_in_buf, dispatch_weights_in_buf, (handle_out if handle is None else handle)
