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
        num_dispatched_token_tensor: torch.Tensor,
        num_local_experts: int,
        num_topk: int,
        pad_multiple: int = 0,
        num_permuted_tokens: int = -1,
        scaling_factor: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        scales_per_token: int = 0,
        use_fp8: bool = False,
    ) -> Tuple[
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

        ctx.num_dispatched = num_dispatched
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.use_fp8 = use_fp8
        ctx.with_probs = probs is not None
        ctx.probs_dtype = probs.dtype if probs is not None else None

        if use_fp8 and scaling_factor is not None:
            assert scales_per_token > 0, "scales_per_token must be > 0 when use_fp8=True"

        # Fast path: preprocessing kernel asserts num_dispatched > 0.
        if num_dispatched == 0:
            int_opts = dict(dtype=torch.int32, device=device)
            row_id_map = torch.zeros((pad_multiple, 2 * num_local_experts + 1), **int_opts)
            tokens_per_expert = torch.zeros((num_local_experts,), **int_opts)
            overflow_flag = torch.zeros((1,), **int_opts)
            permuted_tokens = tokens.new_empty((0, hidden_size))
            permuted_scaling_factor = (
                scaling_factor.new_empty((0, scales_per_token))
                if use_fp8 and scaling_factor is not None
                else None
            )
            permuted_probs = probs.new_zeros((0,)) if probs is not None else None
            ctx.save_for_backward(row_id_map, num_dispatched_token_tensor)
            return (
                permuted_tokens,
                row_id_map,
                tokens_per_expert,
                overflow_flag,
                permuted_scaling_factor,
                permuted_probs,
            )

        row_id_map, tokens_per_expert, overflow_flag = (
            torch.ops.primus_turbo_cpp_extension.permute_preprocessing(
                expert_map,
                num_dispatched_token_tensor,
                num_local_experts,
                num_topk,
                pad_multiple,
                num_permuted_tokens,
            )
        )

        # Prefer caller-provided cap to avoid host sync.
        if num_permuted_tokens >= 0:
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
        # zeros: trailing [Σ real + pad, alloc) is never written when caller over-allocates.
        permuted_probs = (
            torch.zeros((num_permuted_alloc,), dtype=probs.dtype, device=device)
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
            num_dispatched_token_tensor,
            pad_multiple,
            num_local_experts,
            hidden_size,
            scales_per_token,
            use_fp8,
            probs is not None,
            num_permuted_alloc,
        )

        ctx.save_for_backward(row_id_map, num_dispatched_token_tensor)
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
        # unpermute kernel only accepts bf16 / fp16.
        assert not ctx.use_fp8, "_MoEPermute.backward: FP8 backward not supported"

        row_id_map, num_dispatched_token_tensor = ctx.saved_tensors
        grad_permuted_tokens = grad_permuted_tokens.contiguous()
        device = grad_permuted_tokens.device
        grad_tokens = torch.empty(
            (ctx.num_dispatched, ctx.hidden_size),
            dtype=grad_permuted_tokens.dtype,
            device=device,
        )

        # empty: unpermute kernel zeros every probs[t, :] slot up front.
        if ctx.with_probs and permuted_probs_grad is not None:
            permuted_probs_grad = permuted_probs_grad.contiguous()
            grad_probs: Optional[torch.Tensor] = torch.empty(
                (ctx.num_dispatched, ctx.num_local_experts),
                dtype=ctx.probs_dtype,
                device=device,
            )
        else:
            permuted_probs_grad = None
            grad_probs = None

        # Fast path: kernel only writes matched rows; zero unwritten outputs explicitly.
        if ctx.num_dispatched == 0 or grad_permuted_tokens.shape[0] == 0:
            grad_tokens.zero_()
            if grad_probs is not None:
                grad_probs.zero_()
        else:
            torch.ops.primus_turbo_cpp_extension.unpermute(
                grad_permuted_tokens,
                grad_tokens,
                permuted_probs_grad,
                grad_probs,
                row_id_map,
                num_dispatched_token_tensor,
                ctx.num_local_experts,
                ctx.hidden_size,
                grad_probs is not None,
            )

        return (
            grad_tokens,
            None,  # expert_map
            None,  # num_dispatched_token_tensor
            None,  # num_local_experts
            None,  # num_topk
            None,  # pad_multiple
            None,  # num_permuted_tokens
            None,  # scaling_factor
            grad_probs,  # probs
            None,  # scales_per_token
            None,  # use_fp8
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = permuted_tokens.device
        num_permuted = int(permuted_tokens.shape[0])
        # restore_shape == (num_dispatched, hidden_size); see moe_unpermute().
        num_dispatched, hidden_size = int(restore_shape[0]), int(restore_shape[1])

        ctx.num_permuted = num_permuted
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.with_probs = permuted_probs is not None
        ctx.permuted_probs_dtype = permuted_probs.dtype if permuted_probs is not None else None

        unpermuted_tokens = torch.empty(
            (num_dispatched, hidden_size), dtype=permuted_tokens.dtype, device=device
        )
        unpermuted_probs = (
            torch.zeros(
                (num_dispatched, num_local_experts),
                dtype=permuted_probs.dtype,
                device=device,
            )
            if permuted_probs is not None
            else None
        )

        # Fast path: kernel won't run — zero the activation buffer.
        if num_permuted == 0 or num_dispatched == 0:
            unpermuted_tokens.zero_()
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
        # zeros: backward-permute uses pad_multiple=0, so per-expert padded slots
        # are unwritten; pre-zero matches the forward-emitted padded data.
        grad_permuted = torch.zeros(
            (ctx.num_permuted, ctx.hidden_size),
            dtype=grad_unpermuted_tokens.dtype,
            device=device,
        )

        if ctx.with_probs and unpermuted_probs_grad is not None:
            unpermuted_probs_grad = unpermuted_probs_grad.contiguous()
            grad_permuted_probs: Optional[torch.Tensor] = torch.zeros(
                (ctx.num_permuted,), dtype=ctx.permuted_probs_dtype, device=device
            )
        else:
            unpermuted_probs_grad = None
            grad_permuted_probs = None

        # Buffers are already zero; only launch the kernel when there's work.
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
                0,  # pad_multiple
                ctx.num_local_experts,
                ctx.hidden_size,
                0,  # scales_per_token
                False,  # use_fp8
                grad_permuted_probs is not None,
                ctx.num_permuted,
            )

        return (
            grad_permuted,
            None,  # row_id_map
            None,  # num_dispatched_tokens_tensor
            None,  # restore_shape
            None,  # num_local_experts
            grad_permuted_probs,  # permuted_probs
        )


def moe_permute(
    tokens: torch.Tensor,
    expert_map: torch.Tensor,
    num_dispatched_token_tensor: torch.Tensor,
    *,
    num_local_experts: int,
    num_topk: int = 0,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
    scaling_factor: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    scales_per_token: int = 0,
    use_fp8: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Fused preprocessing + permute.

    Returns (permuted_tokens, row_id_map, tokens_per_expert, overflow_flag,
    permuted_scaling_factor, permuted_probs).
    """
    return _MoEPermute.apply(
        tokens,
        expert_map,
        num_dispatched_token_tensor,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
    )


def moe_unpermute(
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens_tensor: torch.Tensor,
    *,
    restore_shape: torch.Size,
    num_local_experts: int,
    permuted_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpermute back into ``restore_shape`` (= original ``tokens.shape``).

    Returns (unpermuted_tokens, unpermuted_probs) shaped ``restore_shape``
    and ``[restore_shape[0], num_local_experts]``.
    """
    return _MoEUnpermute.apply(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens_tensor,
        restore_shape,
        num_local_experts,
        permuted_probs,
    )
