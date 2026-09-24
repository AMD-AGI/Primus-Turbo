###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MoE permute / unpermute autograd ops; permute and unpermute must use the same backend."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.moe.moe_permute_impl import (
    moe_permute_impl,
    moe_permute_process_impl,
    moe_unpermute_impl,
)

__all__ = ["moe_permute", "moe_unpermute"]


def _default_backend(
    pad_multiple: int,
    use_fp8: bool = False,
    scaling_factor: Optional[torch.Tensor] = None,
) -> BackendType:
    """Padding / FP8 need TURBO; Triton handles the rest. Must agree across permute+unpermute."""
    if pad_multiple != 0 or use_fp8 or scaling_factor is not None:
        return BackendType.TURBO
    return BackendType.TRITON


class _MoEPermute(torch.autograd.Function):
    """Forward: preprocessing + permute. Backward: unpermute (+ probs)."""

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        routing_map: Optional[torch.Tensor],
        topk_indices: Optional[torch.Tensor],
        num_local_experts: int,
        num_topk: int,
        pad_multiple: int,
        num_permuted_tokens: int,
        scaling_factor: Optional[torch.Tensor],
        probs: Optional[torch.Tensor],
        scales_per_token: int,
        use_fp8: bool,
        probs_topk_stride: int,
        backend: BackendType,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # Unused grads reach backward as None, not zeros; dynamo cannot trace this ctx call.
        if not torch.compiler.is_compiling():
            ctx.set_materialize_grads(False)

        hidden_size = int(tokens.shape[-1])
        num_dispatched = int(tokens.shape[0])

        probs_topk_stride = int(probs_topk_stride) if probs is not None else 0
        input_probs_topk_stride = probs_topk_stride

        if use_fp8 and scaling_factor is not None:
            assert scales_per_token > 0, "scales_per_token must be > 0 when use_fp8=True"
        # needs_input_grad, not is_grad_enabled: grad mode is already off here.
        assert not (use_fp8 and ctx.needs_input_grad[0]), "moe_permute: FP8 backward is unsupported"

        (
            row_id_map,
            tokens_per_expert,
            overflow_flag,
            num_dispatched_tokens,
            num_permuted,
            backend_probs,
            backend_probs_topk_stride,
            indices_position_map,
        ) = moe_permute_process_impl(
            backend,
            routing_map,
            topk_indices,
            num_dispatched,
            num_local_experts,
            num_topk,
            pad_multiple,
            num_permuted_tokens,
            probs,
            probs_topk_stride,
        )
        permuted_tokens, permuted_scaling_factor, permuted_probs = moe_permute_impl(
            backend,
            tokens,
            row_id_map,
            num_dispatched_tokens,
            num_permuted,
            num_local_experts,
            hidden_size,
            pad_multiple,
            scaling_factor,
            backend_probs,
            scales_per_token,
            use_fp8,
            backend_probs_topk_stride,
        )

        ctx.backend = backend
        ctx.num_dispatched = num_dispatched
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.num_permuted = num_permuted
        ctx.use_fp8 = use_fp8
        ctx.has_probs = probs is not None
        ctx.probs_topk_stride = input_probs_topk_stride
        ctx.tokens_dtype = tokens.dtype
        # Kept only for backward; the dispatched-token bound is not public API.
        ctx.save_for_backward(row_id_map, num_dispatched_tokens, indices_position_map)
        return (
            permuted_tokens,
            row_id_map,
            tokens_per_expert,
            overflow_flag,
            permuted_scaling_factor,
            permuted_probs,
        )

    @staticmethod
    def backward(
        ctx,
        grad_permuted_tokens: torch.Tensor,
        row_id_map_grad: Optional[torch.Tensor],
        tokens_per_expert_grad: Optional[torch.Tensor],
        overflow_flag_grad: Optional[torch.Tensor],
        permuted_scaling_factor_grad: Optional[torch.Tensor],
        permuted_probs_grad: Optional[torch.Tensor],
    ):
        row_id_map, num_dispatched_tokens, indices_position_map = ctx.saved_tensors
        # None when only the probs output was used; the kernel needs a buffer.
        if grad_permuted_tokens is None:
            grad_permuted_tokens = torch.zeros(
                (ctx.num_permuted, ctx.hidden_size),
                dtype=ctx.tokens_dtype,
                device=row_id_map.device,
            )
        grad_permuted_tokens = grad_permuted_tokens.contiguous()

        if ctx.has_probs and permuted_probs_grad is not None:
            permuted_probs_grad = permuted_probs_grad.contiguous()
        else:
            permuted_probs_grad = None

        assert not ctx.use_fp8, "moe_permute backward: FP8 backward not supported"
        # Permute backward is unpermute.
        grad_tokens, grad_probs = moe_unpermute_impl(
            ctx.backend,
            grad_permuted_tokens,
            row_id_map,
            num_dispatched_tokens,
            ctx.num_dispatched,
            ctx.num_local_experts,
            ctx.hidden_size,
            permuted_probs_grad,
            ctx.probs_topk_stride,
            indices_position_map,
        )

        return (
            grad_tokens,
            None,  # routing_map
            None,  # topk_indices
            None,  # num_local_experts
            None,  # num_topk
            None,  # pad_multiple
            None,  # num_permuted_tokens
            None,  # scaling_factor
            grad_probs,  # probs
            None,  # scales_per_token
            None,  # use_fp8
            None,  # probs_topk_stride
            None,  # backend
        )


class _MoEUnpermute(torch.autograd.Function):
    """Forward: unpermute. Backward: permute (+ probs)."""

    @staticmethod
    def forward(
        ctx,
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        num_dispatched_tokens_tensor: Optional[torch.Tensor],
        restore_shape: torch.Size,
        num_local_experts: int,
        permuted_probs: Optional[torch.Tensor],
        probs_topk_stride: int,
        pad_multiple: int,
        backend: BackendType,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Unused grads reach backward as None, not zeros; dynamo cannot trace this ctx call.
        if not torch.compiler.is_compiling():
            ctx.set_materialize_grads(False)

        probs_topk_stride = int(probs_topk_stride) if permuted_probs is not None else 0

        # A 3-D shape would silently unpermute the wrong extent.
        assert len(restore_shape) == 2, (
            f"moe_unpermute: restore_shape must be 2-D, got {tuple(restore_shape)}"
        )
        num_dispatched, hidden_size = int(restore_shape[0]), int(restore_shape[1])
        assert int(permuted_tokens.shape[-1]) == hidden_size, (
            f"moe_unpermute: permuted_tokens hidden {int(permuted_tokens.shape[-1])} "
            f"!= restore_shape hidden {hidden_size}"
        )

        ctx.backend = backend
        ctx.num_dispatched = num_dispatched
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.num_permuted = int(permuted_tokens.shape[0])
        ctx.pad_multiple = pad_multiple
        ctx.has_probs = permuted_probs is not None
        ctx.probs_topk_stride = probs_topk_stride
        ctx.tokens_dtype = permuted_tokens.dtype

        outputs = moe_unpermute_impl(
            backend,
            permuted_tokens,
            row_id_map,
            num_dispatched_tokens_tensor,
            num_dispatched,
            num_local_experts,
            hidden_size,
            permuted_probs,
            probs_topk_stride,
            None,
        )
        ctx.save_for_backward(row_id_map, num_dispatched_tokens_tensor)
        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_unpermuted_tokens: torch.Tensor,
        unpermuted_probs_grad: Optional[torch.Tensor],
    ):
        row_id_map, num_dispatched_tokens_tensor = ctx.saved_tensors
        # None when only the probs output was used; the kernel needs a buffer.
        if grad_unpermuted_tokens is None:
            grad_unpermuted_tokens = torch.zeros(
                (ctx.num_dispatched, ctx.hidden_size),
                dtype=ctx.tokens_dtype,
                device=row_id_map.device,
            )
        grad_unpermuted_tokens = grad_unpermuted_tokens.contiguous()

        if ctx.has_probs and unpermuted_probs_grad is not None:
            unpermuted_probs_grad = unpermuted_probs_grad.contiguous()
        else:
            unpermuted_probs_grad = None

        # Unpermute backward is permute; the row_id_map is already built.
        grad_permuted, _, grad_permuted_probs = moe_permute_impl(
            ctx.backend,
            grad_unpermuted_tokens,
            row_id_map,
            num_dispatched_tokens_tensor,
            ctx.num_permuted,
            ctx.num_local_experts,
            ctx.hidden_size,
            ctx.pad_multiple,
            None,  # scaling_factor
            unpermuted_probs_grad,
            0,  # scales_per_token
            False,  # use_fp8
            ctx.probs_topk_stride,
        )

        return (
            grad_permuted,
            None,  # row_id_map
            None,  # num_dispatched_tokens_tensor
            None,  # restore_shape
            None,  # num_local_experts
            grad_permuted_probs,  # permuted_probs
            None,  # probs_topk_stride
            None,  # pad_multiple
            None,  # backend
        )


def moe_permute(
    tokens: torch.Tensor,
    *,
    routing_map: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    num_local_experts: int,
    num_topk: int = 0,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
    scaling_factor: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    probs_layout: str = "topk",
    scales_per_token: int = 0,
    use_fp8: bool = False,
    backend: Optional[BackendType] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Fused preprocessing + permute; returns (permuted_tokens, row_id_map, tokens_per_expert, overflow_flag, permuted_scaling_factor, permuted_probs).

    ``num_permuted_tokens`` is a capacity on TURBO (extra rows dropped, flagged in
    ``overflow_flag``) but only an upper bound on TRITON -- too small writes OOB.
    TRITON never drops, so its ``overflow_flag`` is a host-side zero tensor.
    """
    if routing_map is None and topk_indices is None:
        raise ValueError("moe_permute: one of routing_map / topk_indices must be provided")
    if probs_layout not in ("routing_map", "topk"):
        raise ValueError(f"moe_permute: probs_layout must be 'routing_map' or 'topk', got {probs_layout!r}")
    if probs is not None and probs_layout == "topk":
        if num_topk <= 0:
            raise ValueError("moe_permute: probs_layout='topk' requires num_topk > 0")
        if topk_indices is None:
            raise ValueError("moe_permute: probs_layout='topk' requires topk_indices")
        if num_topk != int(topk_indices.shape[1]):
            raise ValueError(
                f"moe_permute: num_topk={num_topk} disagrees with "
                f"topk_indices width {int(topk_indices.shape[1])}"
            )

    probs_topk_stride = num_topk if (probs is not None and probs_layout == "topk") else 0

    if backend is None:
        backend = _default_backend(pad_multiple, use_fp8, scaling_factor)

    return _MoEPermute.apply(
        tokens,
        routing_map,
        topk_indices,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
        probs_topk_stride,
        backend,
    )


def moe_unpermute(
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens_tensor: Optional[torch.Tensor] = None,
    *,
    restore_shape: torch.Size,
    num_local_experts: int,
    backend: Optional[BackendType] = None,
    permuted_probs: Optional[torch.Tensor] = None,
    probs_topk_stride: int = 0,
    pad_multiple: int = 0,
    use_fp8: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpermute back into ``restore_shape`` using the matching permute backend.

    ``num_dispatched_tokens_tensor`` is an optional device-side row bound letting
    TURBO skip the unrouted tail; defaults to ``restore_shape[0]``. TRITON ignores it.
    """
    if backend is None:
        backend = _default_backend(pad_multiple, use_fp8)
    if backend is BackendType.TRITON and permuted_probs is not None and probs_topk_stride > 0:
        # Silently switching to TURBO here would read a foreign row_id_map layout.
        raise ValueError(
            "moe_unpermute: TRITON cannot emit top-k-aligned probs -- the forward "
            "position map only lives inside moe_permute's backward. Pass "
            "probs_topk_stride=0, or run both permute and unpermute on TURBO."
        )
    return _MoEUnpermute.apply(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens_tensor,
        restore_shape,
        num_local_experts,
        permuted_probs,
        probs_topk_stride,
        pad_multiple,
        backend,
    )
