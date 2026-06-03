###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = ["moe_permute", "moe_unpermute"]


class _MoEPermute(torch.autograd.Function):
    """Forward: permute_preprocessing + permute. Backward: unpermute (+ probs)."""

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        expert_map: torch.Tensor,
        num_local_experts: int,
        num_topk: int,
        pad_multiple: int = 0,
        num_permuted_tokens: int = -1,
        scaling_factor: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        scales_per_token: int = 0,
        use_fp8: bool = False,
        probs_topk_stride: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        device = tokens.device
        hidden_size = int(tokens.shape[-1])
        num_dispatched = int(tokens.shape[0])

        probs_topk_stride = int(probs_topk_stride) if probs is not None else 0
        probs_row_width = probs_topk_stride if probs_topk_stride > 0 else num_local_experts

        ctx.num_dispatched = num_dispatched
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.use_fp8 = use_fp8
        ctx.with_probs = probs is not None
        ctx.probs_dtype = probs.dtype if probs is not None else None
        ctx.probs_topk_stride = probs_topk_stride
        ctx.probs_row_width = probs_row_width

        if use_fp8 and scaling_factor is not None:
            assert scales_per_token > 0, "scales_per_token must be > 0 when use_fp8=True"
        # backward asserts not use_fp8; catch the unsupported combo at the forward boundary.
        assert not (
            use_fp8 and probs is not None and tokens.requires_grad
        ), "moe_permute: FP8 + probs backward is unsupported"

        # Fast path: preprocessing kernel asserts num_dispatched > 0.
        if num_dispatched == 0:
            int_opts = dict(dtype=torch.int32, device=device)
            row_id_map = torch.zeros((pad_multiple, 2 * num_local_experts + 1), **int_opts)
            tokens_per_expert = torch.zeros((num_local_experts,), dtype=torch.int64, device=device)
            overflow_flag = torch.zeros((1,), **int_opts)
            num_dispatched_tokens = torch.zeros((1,), **int_opts)
            permuted_tokens = tokens.new_empty((0, hidden_size))
            permuted_scaling_factor = (
                scaling_factor.new_empty((0, scales_per_token))
                if use_fp8 and scaling_factor is not None
                else None
            )
            permuted_probs = probs.new_zeros((0,)) if probs is not None else None
            ctx.save_for_backward(row_id_map, num_dispatched_tokens)
            return (
                permuted_tokens,
                row_id_map,
                tokens_per_expert,
                overflow_flag,
                num_dispatched_tokens,
                permuted_scaling_factor,
                permuted_probs,
            )

        row_id_map, tokens_per_expert, overflow_flag, num_dispatched_tokens = (
            torch.ops.primus_turbo_cpp_extension.permute_preprocessing(
                expert_map,
                num_local_experts,
                num_topk,
                pad_multiple,
                num_permuted_tokens,
                probs_topk_stride,
            )
        )

        if num_permuted_tokens > 0:
            num_permuted_alloc = int(num_permuted_tokens)
        else:
            num_permuted_alloc = int(tokens_per_expert.sum().item())

        permuted_tokens = torch.empty((num_permuted_alloc, hidden_size), dtype=tokens.dtype, device=device)
        if use_fp8 and scaling_factor is not None:
            permuted_scaling_factor = torch.empty(
                (num_permuted_alloc, scales_per_token),
                dtype=scaling_factor.dtype,
                device=device,
            )
        else:
            permuted_scaling_factor = None
        permuted_probs = (
            torch.empty((num_permuted_alloc,), dtype=probs.dtype, device=device)
            if probs is not None
            else None
        )

        torch.ops.primus_turbo_cpp_extension.permute(
            tokens,
            permuted_tokens,
            scaling_factor,
            permuted_scaling_factor,
            probs,
            permuted_probs,
            row_id_map,
            num_dispatched_tokens,
            pad_multiple,
            num_local_experts,
            hidden_size,
            scales_per_token,
            use_fp8,
            probs is not None,
            num_permuted_alloc,
            probs_topk_stride,
        )

        ctx.save_for_backward(row_id_map, num_dispatched_tokens)
        return (
            permuted_tokens,
            row_id_map,
            tokens_per_expert,
            overflow_flag,
            num_dispatched_tokens,
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
        num_dispatched_tokens_grad: Optional[torch.Tensor],
        permuted_scaling_factor_grad: Optional[torch.Tensor],
        permuted_probs_grad: Optional[torch.Tensor],
    ):
        # unpermute kernel only accepts bf16 / fp16.
        assert not ctx.use_fp8, "_MoEPermute.backward: FP8 backward not supported"

        row_id_map, num_dispatched_tokens = ctx.saved_tensors
        grad_permuted_tokens = grad_permuted_tokens.contiguous()
        device = grad_permuted_tokens.device
        grad_tokens = torch.empty(
            (ctx.num_dispatched, ctx.hidden_size),
            dtype=grad_permuted_tokens.dtype,
            device=device,
        )

        # Grad probs row width matches the forward input.
        if ctx.with_probs and permuted_probs_grad is not None:
            permuted_probs_grad = permuted_probs_grad.contiguous()
            grad_probs: Optional[torch.Tensor] = torch.empty(
                (ctx.num_dispatched, ctx.probs_row_width),
                dtype=ctx.probs_dtype,
                device=device,
            )
        else:
            permuted_probs_grad = None
            grad_probs = None

        if ctx.num_dispatched > 0:
            torch.ops.primus_turbo_cpp_extension.unpermute(
                grad_permuted_tokens,
                grad_tokens,
                permuted_probs_grad,
                grad_probs,
                row_id_map,
                num_dispatched_tokens,
                ctx.num_local_experts,
                ctx.hidden_size,
                grad_probs is not None,
                ctx.probs_topk_stride,
            )

        return (
            grad_tokens,
            None,  # expert_map
            None,  # num_local_experts
            None,  # num_topk
            None,  # pad_multiple
            None,  # num_permuted_tokens
            None,  # scaling_factor
            grad_probs,  # probs
            None,  # scales_per_token
            None,  # use_fp8
            None,  # probs_topk_stride
        )


class _MoEUnpermute(torch.autograd.Function):
    """Forward: unpermute. Backward: permute (+ probs)."""

    @staticmethod
    def forward(
        ctx,
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        num_dispatched_tokens_tensor: torch.Tensor,
        restore_shape: torch.Size,
        num_local_experts: int,
        permuted_probs: Optional[torch.Tensor],
        probs_topk_stride: int,
        pad_multiple: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = permuted_tokens.device
        num_permuted = int(permuted_tokens.shape[0])
        # restore_shape == (num_dispatched, hidden_size); see moe_unpermute().
        num_dispatched, hidden_size = int(restore_shape[0]), int(restore_shape[1])

        # Must match the row_id_map produced by forward-permute (0 = multihot, >0 = topk-aligned).
        probs_row_width = probs_topk_stride if probs_topk_stride > 0 else num_local_experts

        ctx.num_permuted = num_permuted
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.with_probs = permuted_probs is not None
        ctx.permuted_probs_dtype = permuted_probs.dtype if permuted_probs is not None else None
        ctx.probs_topk_stride = probs_topk_stride
        ctx.probs_row_width = probs_row_width
        ctx.pad_multiple = pad_multiple

        unpermuted_tokens = torch.empty(
            (num_dispatched, hidden_size), dtype=permuted_tokens.dtype, device=device
        )
        unpermuted_probs = (
            torch.empty(
                (num_dispatched, probs_row_width),
                dtype=permuted_probs.dtype,
                device=device,
            )
            if permuted_probs is not None
            else None
        )

        # Buffers are already zero; skip the kernel when there's nothing to do.
        if num_permuted == 0 or num_dispatched == 0:
            ctx.save_for_backward(row_id_map, num_dispatched_tokens_tensor)
            return unpermuted_tokens, unpermuted_probs

        torch.ops.primus_turbo_cpp_extension.unpermute(
            permuted_tokens,
            unpermuted_tokens,
            permuted_probs,
            unpermuted_probs,
            row_id_map,
            num_dispatched_tokens_tensor,
            num_local_experts,
            hidden_size,
            permuted_probs is not None,
            probs_topk_stride,
        )

        ctx.save_for_backward(row_id_map, num_dispatched_tokens_tensor)
        return unpermuted_tokens, unpermuted_probs

    @staticmethod
    def backward(
        ctx,
        grad_unpermuted_tokens: torch.Tensor,
        unpermuted_probs_grad: Optional[torch.Tensor],
    ):
        row_id_map, num_dispatched_tokens_tensor = ctx.saved_tensors
        grad_unpermuted_tokens = grad_unpermuted_tokens.contiguous()
        device = grad_unpermuted_tokens.device
        # No pre-fill: kernel zeros padded slots itself (see moe_permute.hip is_padding_token).
        grad_permuted = torch.empty(
            (ctx.num_permuted, ctx.hidden_size),
            dtype=grad_unpermuted_tokens.dtype,
            device=device,
        )

        if ctx.with_probs and unpermuted_probs_grad is not None:
            unpermuted_probs_grad = unpermuted_probs_grad.contiguous()
            grad_permuted_probs: Optional[torch.Tensor] = torch.empty(
                (ctx.num_permuted,), dtype=ctx.permuted_probs_dtype, device=device
            )
        else:
            unpermuted_probs_grad = None
            grad_permuted_probs = None

        if ctx.num_permuted > 0 and grad_unpermuted_tokens.shape[0] > 0:
            torch.ops.primus_turbo_cpp_extension.permute(
                grad_unpermuted_tokens,
                grad_permuted,
                None,  # scaling_factor
                None,  # output_scaling_factor
                unpermuted_probs_grad,
                grad_permuted_probs,
                row_id_map,
                num_dispatched_tokens_tensor,
                ctx.pad_multiple,
                ctx.num_local_experts,
                ctx.hidden_size,
                0,  # scales_per_token
                False,  # use_fp8
                grad_permuted_probs is not None,
                ctx.num_permuted,
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
        )


def moe_permute(
    tokens: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    num_local_experts: int,
    num_topk: int = 0,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
    scaling_factor: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    probs_layout: str = "topk",
    scales_per_token: int = 0,
    use_fp8: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Fused preprocessing + permute.

    Returns (permuted_tokens, row_id_map, tokens_per_expert, overflow_flag,
    num_dispatched_tokens, permuted_scaling_factor, permuted_probs). Pass
    ``num_dispatched_tokens`` straight to ``moe_unpermute``.
    """
    if probs_layout not in ("routing_map", "topk"):
        raise ValueError(f"moe_permute: probs_layout must be 'routing_map' or 'topk', got {probs_layout!r}")
    if probs_layout == "topk" and num_topk <= 0:
        raise ValueError("moe_permute: probs_layout='topk' requires num_topk > 0")

    probs_topk_stride = num_topk if probs_layout == "topk" else 0

    return _MoEPermute.apply(
        tokens,
        expert_map,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
        probs_topk_stride,
    )


def moe_unpermute(
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens_tensor: torch.Tensor,
    *,
    restore_shape: torch.Size,
    num_local_experts: int,
    permuted_probs: Optional[torch.Tensor] = None,
    probs_topk_stride: int = 0,
    pad_multiple: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpermute back into ``restore_shape`` (= original ``tokens.shape``).

    ``probs_topk_stride`` picks the ``unpermuted_probs`` row width: ``0`` =>
    multihot ``[T, num_local_experts]``, ``>0`` => topk-aligned ``[T, stride]``.
    Must match the value used in forward ``moe_permute``.

    ``pad_multiple`` must match the value used by the forward ``moe_permute``
    that produced ``row_id_map``. It is only consumed by backward; passing 0
    when the input was padded leaves padded slots of the input-grad buffer
    uninitialized.
    """
    return _MoEUnpermute.apply(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens_tensor,
        restore_shape,
        num_local_experts,
        permuted_probs,
        probs_topk_stride,
        pad_multiple,
    )
