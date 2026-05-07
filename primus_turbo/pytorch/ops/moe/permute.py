###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch

__all__ = [
    "moe_permute",
    "moe_unpermute",
]


@lru_cache(maxsize=1)
def _get_local_rank_info() -> Tuple[int, int]:
    """Return ``(local_rank, num_ranks_per_node)`` from the runtime env."""
    local_rank_env = os.environ.get("LOCAL_RANK")
    local_world_env = os.environ.get("LOCAL_WORLD_SIZE")
    if local_rank_env is not None and local_world_env is not None:
        return int(local_rank_env), int(local_world_env)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


class _MoEPermute(torch.autograd.Function):
    """Forward = ``permute_preprocessing`` + ``permute``.

    The forward saves ``row_id_map`` (and ``num_dispatched_token_tensor``) so
    the backward can run a single ``unpermute`` over ``grad_permuted_tokens``
    to recover the gradient of the dispatched ``tokens``.

    Returns
    -------
    permuted_tokens : torch.Tensor
        ``[num_permuted, hidden]`` (same dtype as ``tokens``).
    row_id_map : torch.Tensor
        ``int32 [max_num_dispatched_tokens + pad_multiple, 2 * E + 1]`` map
        emitted by preprocessing — re-usable by a downstream ``_MoEUnpermute``.
    tokens_per_expert : torch.Tensor
        ``int32 [num_local_experts]``.
    overflow_flag : torch.Tensor
        ``int32 [1]`` — non-zero if preprocessing dropped tokens.
    permuted_scaling_factor : Optional[torch.Tensor]
        Only populated when ``use_fp8`` and ``scaling_factor`` is given.
    permuted_probs : Optional[torch.Tensor]
        Only populated when ``probs`` is given.
    """

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        expert_map: torch.Tensor,
        num_dispatched_token_tensor: torch.Tensor,
        max_num_dispatched_tokens: int,
        num_local_experts: int,
        num_topk: int,
        pad_multiple: int = 0,
        num_permuted_tokens: int = -1,
        scaling_factor: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        scales_per_token: int = 0,
        use_fp8: bool = False,
        num_blocks_permute: int = 0,
        num_blocks_unpermute: int = 0,
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
        local_rank, num_ranks_per_node = _get_local_rank_info()

        # 1) Preprocessing: row_id_map / tokens_per_expert / overflow_flag.
        row_id_map, tokens_per_expert, overflow_flag = (
            torch.ops.primus_turbo_cpp_extension.permute_preprocessing(
                expert_map,
                num_dispatched_token_tensor,
                max_num_dispatched_tokens,
                num_local_experts,
                num_topk,
                pad_multiple,
                num_permuted_tokens,
            )
        )

        # 2) Pick the permuted-output row count.
        # Prefer the caller-provided cap to avoid a host sync. Fall back to a
        # `.item()` pull only when the cap is unknown (``num_permuted_tokens < 0``).
        if num_permuted_tokens is not None and num_permuted_tokens >= 0:
            num_permuted_alloc = int(num_permuted_tokens)
        else:
            num_permuted_alloc = int(tokens_per_expert.sum().item())

        # 3) Allocate outputs.
        permuted_tokens = torch.empty((num_permuted_alloc, hidden_size), dtype=tokens.dtype, device=device)
        if use_fp8 and scaling_factor is not None:
            assert scales_per_token > 0, "_MoEPermute: scales_per_token must be > 0 when use_fp8=True"
            permuted_scaling_factor: Optional[torch.Tensor] = torch.empty(
                (num_permuted_alloc, scales_per_token),
                dtype=scaling_factor.dtype,
                device=device,
            )
        else:
            permuted_scaling_factor = None
        if probs is not None:
            permuted_probs: Optional[torch.Tensor] = torch.empty(
                (num_permuted_alloc,), dtype=probs.dtype, device=device
            )
        else:
            permuted_probs = None
        with_probs = (probs is not None) and (permuted_probs is not None)

        # 4) Run the permute kernel.
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
            local_rank,
            num_ranks_per_node,
            use_fp8,
            with_probs,
            num_permuted_alloc,
            num_blocks_permute,
        )

        # 5) Save state for backward (unpermute).
        ctx.save_for_backward(row_id_map, num_dispatched_token_tensor)
        ctx.num_dispatched = num_dispatched
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.num_blocks_unpermute = num_blocks_unpermute
        ctx.use_fp8 = use_fp8

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
        # The HIP unpermute kernel is bf16-only; we don't currently support
        # FP8 backward (gradients are typically bf16 anyway).
        assert not ctx.use_fp8, (
            "_MoEPermute.backward: FP8 backward is not supported "
            "(unpermute kernel only accepts bfloat16 inputs)."
        )

        row_id_map, num_dispatched_token_tensor = ctx.saved_tensors
        local_rank, num_ranks_per_node = _get_local_rank_info()
        grad_permuted_tokens = grad_permuted_tokens.contiguous()
        grad_tokens = torch.empty(
            (ctx.num_dispatched, ctx.hidden_size),
            dtype=grad_permuted_tokens.dtype,
            device=grad_permuted_tokens.device,
        )

        torch.ops.primus_turbo_cpp_extension.unpermute(
            grad_permuted_tokens,
            grad_tokens,
            None,  # permuted_probs
            None,  # output_probs
            row_id_map,
            num_dispatched_token_tensor,
            ctx.num_local_experts,
            ctx.hidden_size,
            local_rank,
            num_ranks_per_node,
            False,  # with_probs
            ctx.num_blocks_unpermute,
        )

        # Match the 14 forward inputs; only ``tokens`` receives a gradient.
        return (
            grad_tokens,
            None,  # expert_map
            None,  # num_dispatched_token_tensor
            None,  # max_num_dispatched_tokens
            None,  # num_local_experts
            None,  # num_topk
            None,  # pad_multiple
            None,  # num_permuted_tokens
            None,  # scaling_factor
            None,  # probs
            None,  # scales_per_token
            None,  # use_fp8
            None,  # num_blocks_permute
            None,  # num_blocks_unpermute
        )


class _MoEUnpermute(torch.autograd.Function):
    """Forward = ``unpermute``.

    Saves ``row_id_map`` so the backward can re-run ``permute`` on
    ``grad_unpermuted_tokens``.
    """

    @staticmethod
    def forward(
        ctx,
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        num_dispatched_tokens_tensor: torch.Tensor,
        num_dispatched: int,
        num_local_experts: int,
        permuted_probs: Optional[torch.Tensor],
        pad_multiple: int,
        num_blocks_unpermute: int,
        num_blocks_permute: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = permuted_tokens.device
        hidden_size = int(permuted_tokens.shape[-1])
        num_permuted = int(permuted_tokens.shape[0])
        local_rank, num_ranks_per_node = _get_local_rank_info()

        unpermuted_tokens = torch.empty(
            (num_dispatched, hidden_size),
            dtype=permuted_tokens.dtype,
            device=device,
        )
        unpermuted_probs: Optional[torch.Tensor] = None
        if permuted_probs is not None:
            unpermuted_probs = torch.empty(
                (num_dispatched, num_local_experts),
                dtype=permuted_probs.dtype,
                device=device,
            )
        with_probs = (permuted_probs is not None) and (unpermuted_probs is not None)

        torch.ops.primus_turbo_cpp_extension.unpermute(
            permuted_tokens,
            unpermuted_tokens,
            permuted_probs,
            unpermuted_probs,
            row_id_map,
            num_dispatched_tokens_tensor,
            num_local_experts,
            hidden_size,
            local_rank,
            num_ranks_per_node,
            with_probs,
            num_blocks_unpermute,
        )

        ctx.save_for_backward(row_id_map, num_dispatched_tokens_tensor)
        ctx.num_permuted = num_permuted
        ctx.hidden_size = hidden_size
        ctx.num_local_experts = num_local_experts
        ctx.pad_multiple = pad_multiple
        ctx.num_blocks_permute = num_blocks_permute

        return unpermuted_tokens, unpermuted_probs

    @staticmethod
    def backward(
        ctx,
        grad_unpermuted_tokens: torch.Tensor,
        unpermuted_probs_grad: Optional[torch.Tensor],
    ):
        row_id_map, num_dispatched_tokens_tensor = ctx.saved_tensors
        local_rank, num_ranks_per_node = _get_local_rank_info()
        grad_unpermuted_tokens = grad_unpermuted_tokens.contiguous()
        grad_permuted = torch.empty(
            (ctx.num_permuted, ctx.hidden_size),
            dtype=grad_unpermuted_tokens.dtype,
            device=grad_unpermuted_tokens.device,
        )

        torch.ops.primus_turbo_cpp_extension.permute(
            grad_unpermuted_tokens,
            grad_permuted,
            None,  # scaling_factor
            None,  # output_scaling_factor
            None,  # probs
            None,  # output_probs
            row_id_map,
            num_dispatched_tokens_tensor,
            ctx.pad_multiple,
            ctx.num_local_experts,
            ctx.hidden_size,
            0,  # scales_per_token
            local_rank,
            num_ranks_per_node,
            False,  # use_fp8
            False,  # with_probs
            ctx.num_permuted,
            ctx.num_blocks_permute,
        )

        # Match the 9 forward inputs; only ``permuted_tokens`` gets a gradient.
        return (
            grad_permuted,
            None,  # row_id_map
            None,  # num_dispatched_tokens_tensor
            None,  # num_dispatched
            None,  # num_local_experts
            None,  # permuted_probs
            None,  # pad_multiple
            None,  # num_blocks_unpermute
            None,  # num_blocks_permute
        )


# -----------------------------------------------------------------------------
# User-facing autograd-aware wrappers.
# -----------------------------------------------------------------------------


def moe_permute(
    tokens: torch.Tensor,
    expert_map: torch.Tensor,
    num_dispatched_token_tensor: torch.Tensor,
    max_num_dispatched_tokens: int,
    *,
    num_local_experts: int,
    num_topk: int = 0,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
    scaling_factor: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    scales_per_token: int = 0,
    use_fp8: bool = False,
    num_blocks_permute: int = 0,
    num_blocks_unpermute: int = 0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Autograd-aware fused ``preprocessing + permute``.

    Returns ``(permuted_tokens, row_id_map, tokens_per_expert, overflow_flag,
    permuted_scaling_factor, permuted_probs)``. The backward routes only the
    activation gradient (``permuted_tokens.grad``) back to ``tokens``.
    """
    return _MoEPermute.apply(
        tokens,
        expert_map,
        num_dispatched_token_tensor,
        max_num_dispatched_tokens,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
        num_blocks_permute,
        num_blocks_unpermute,
    )


def moe_unpermute(
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens_tensor: torch.Tensor,
    num_dispatched: int,
    *,
    num_local_experts: int,
    permuted_probs: Optional[torch.Tensor] = None,
    pad_multiple: int = 0,
    num_blocks_unpermute: int = 0,
    num_blocks_permute: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Autograd-aware ``unpermute`` (forward) + ``permute`` (backward).

    Returns ``(unpermuted_tokens, unpermuted_probs)``.
    """
    return _MoEUnpermute.apply(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens_tensor,
        num_dispatched,
        num_local_experts,
        permuted_probs,
        pad_multiple,
        num_blocks_unpermute,
        num_blocks_permute,
    )
