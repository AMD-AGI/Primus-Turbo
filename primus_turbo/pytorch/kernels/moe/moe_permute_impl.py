###############################################################################
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from NVIDIA Megatron-LM (https://github.com/NVIDIA/Megatron-LM),
#   file megatron/core/transformer/moe/moe_utils.py.
# Modified by the Primus-Turbo team.
#
# This file is distributed under the 3-clause BSD license used by Megatron-LM,
# not the MIT license that covers the rest of Primus-Turbo. Both texts are in
# LICENSE.
###############################################################################

"""MoE permute / unpermute backends; ``row_id_map`` layouts are backend-specific and not interchangeable."""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple, Type, runtime_checkable

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.moe.fused_moe_indices_converter_impl import (
    fused_moe_indices_converter_backward_impl,
    fused_moe_indices_converter_forward_impl,
)
from primus_turbo.triton.moe import permutation as triton_permutation

__all__ = [
    "MoEPermuteBackend",
    "moe_permute_process_impl",
    "moe_permute_impl",
    "moe_unpermute_impl",
]


@runtime_checkable
class MoEPermuteBackend(Protocol):
    """Structural interface for a permute/unpermute backend pair."""

    @staticmethod
    def preprocess(
        routing_map: Optional[torch.Tensor],
        topk_indices: Optional[torch.Tensor],
        num_tokens: int,
        num_local_experts: int,
        num_topk: int,
        pad_multiple: int,
        num_permuted_tokens: int,
        probs: Optional[torch.Tensor],
        probs_topk_stride: int,
    ) -> Tuple[
        torch.Tensor,  # row_id_map
        torch.Tensor,  # tokens_per_expert
        torch.Tensor,  # overflow_flag (TURBO: device; TRITON: host zero, never drops)
        Optional[torch.Tensor],  # num_dispatched_tokens (TURBO only; None on TRITON)
        int,  # num_permuted (rows to allocate)
        Optional[torch.Tensor],  # multihot probs, set only when the backend converted them
        Optional[torch.Tensor],  # top-k position map for backward
    ]: ...

    @staticmethod
    def permute(
        tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        num_dispatched_tokens: Optional[torch.Tensor],
        num_permuted: int,
        num_local_experts: int,
        hidden_size: int,
        pad_multiple: int,
        scaling_factor: Optional[torch.Tensor],
        probs: Optional[torch.Tensor],
        scales_per_token: int,
        use_fp8: bool,
        probs_topk_stride: int,
    ) -> Tuple[
        torch.Tensor,  # permuted_tokens
        Optional[torch.Tensor],  # permuted_scaling_factor
        Optional[torch.Tensor],  # permuted_probs
    ]: ...

    @staticmethod
    def unpermute(
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        num_dispatched_tokens: Optional[torch.Tensor],
        num_dispatched: int,
        num_local_experts: int,
        hidden_size: int,
        permuted_probs: Optional[torch.Tensor],
        probs_topk_stride: int,
        indices_position_map: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,  # unpermuted_tokens
        Optional[torch.Tensor],  # unpermuted_probs
    ]: ...


def _assert_mask_map_layout(row_id_map: torch.Tensor, num_local_experts: int) -> None:
    """Reject a row_id_map width the mask-map kernels would index out of bounds."""
    expected = 2 * num_local_experts + 1
    assert row_id_map.dim() == 2 and int(row_id_map.shape[1]) == expected, (
        f"row_id_map must be [num_tokens, {expected}] for {num_local_experts} "
        f"experts, got {tuple(row_id_map.shape)}"
    )


class TritonMaskMapBackend:
    """Triton path: unpadded 0-based mask-map layout, no capacity / FP8 support."""

    @staticmethod
    def preprocess(
        routing_map,
        topk_indices,
        num_tokens,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        probs,
        probs_topk_stride,
    ):
        assert pad_multiple == 0, "TRITON moe_permute does not support pad_multiple; use TURBO"
        indices_position_map = None
        multihot_probs = None
        if routing_map is None:
            assert topk_indices is not None, "TRITON moe_permute requires routing_map or topk_indices"
            convert_topk_probs = probs is not None and probs_topk_stride > 0
            routing_map, multihot_probs, indices_position_map = fused_moe_indices_converter_forward_impl(
                topk_indices,
                probs if convert_topk_probs else None,
                num_local_experts,
            )
        else:
            assert probs_topk_stride == 0, (
                "TRITON moe_permute only accepts topk-aligned probs with topk_indices-only input"
            )
        device = routing_map.device
        num_experts = int(routing_map.shape[1])
        assert num_experts == num_local_experts, (
            f"routing_map has {num_experts} experts, expected {num_local_experts}"
        )
        assert int(routing_map.shape[0]) == num_tokens, (
            f"routing_map has {int(routing_map.shape[0])} rows, expected {num_tokens}"
        )
        # Kernels launch a grid of num_tokens blocks; nothing to do at zero.
        if num_tokens == 0:
            return (
                torch.zeros((0, 2 * num_experts + 1), dtype=torch.int32, device=device),
                torch.zeros((num_experts,), dtype=torch.int64, device=device),
                torch.zeros((1,), dtype=torch.int32),
                None,
                0,
                multihot_probs,
                indices_position_map,
            )

        row_id_map, tokens_per_expert = triton_permutation.make_row_id_map(
            routing_map, num_tokens, num_experts, True
        )
        # No capacity kernel here, so nothing is dropped; keep it on host, no sync.
        overflow_flag = torch.zeros((1,), dtype=torch.int32)

        # Must be an upper bound: the kernel stores at dst row without clamping.
        if num_permuted_tokens > 0:
            num_permuted = int(num_permuted_tokens)
        else:
            num_permuted = int(tokens_per_expert.sum().item())
        return (
            row_id_map,
            tokens_per_expert,
            overflow_flag,
            None,
            num_permuted,
            multihot_probs,
            indices_position_map,
        )

    @staticmethod
    def permute(
        tokens,
        row_id_map,
        num_dispatched_tokens,
        num_permuted,
        num_local_experts,
        hidden_size,
        pad_multiple,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
        probs_topk_stride,
    ):
        _assert_mask_map_layout(row_id_map, num_local_experts)
        assert pad_multiple == 0, "TRITON moe_permute does not support pad_multiple; use TURBO"
        assert not use_fp8 and scaling_factor is None, "TRITON moe_permute does not support FP8"
        assert probs_topk_stride == 0, "TRITON probs must be converted to multihot during preprocessing"
        num_tokens = int(tokens.shape[0])
        if num_tokens == 0 or num_permuted == 0:
            permuted_tokens = tokens.new_zeros((num_permuted, hidden_size))
            permuted_probs = probs.new_zeros((num_permuted,)) if probs is not None else None
            return permuted_tokens, None, permuted_probs

        permuted_tokens, _, permuted_probs = triton_permutation.permute_with_mask_map(
            tokens,
            row_id_map,
            probs,
            None,  # fp8 scale
            num_tokens,
            num_local_experts,
            num_permuted,
            hidden_size,
            None,  # scale_hidden_dim
        )
        return permuted_tokens, None, permuted_probs

    @staticmethod
    def unpermute(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens,
        num_dispatched,
        num_local_experts,
        hidden_size,
        permuted_probs,
        probs_topk_stride,
        indices_position_map,
    ):
        _assert_mask_map_layout(row_id_map, num_local_experts)
        if num_dispatched == 0 or permuted_tokens.shape[0] == 0:
            unpermuted_tokens = permuted_tokens.new_zeros((num_dispatched, hidden_size))
            unpermuted_probs = (
                permuted_probs.new_zeros((num_dispatched, num_local_experts))
                if permuted_probs is not None
                else None
            )
        else:
            unpermuted_tokens, unpermuted_probs = triton_permutation.unpermute_with_mask_map(
                permuted_tokens,
                row_id_map,
                None,  # merging_probs
                permuted_probs,
                num_dispatched,
                num_local_experts,
                hidden_size,
            )

        if unpermuted_probs is not None and probs_topk_stride > 0:
            assert indices_position_map is not None, (
                "TRITON topk-aligned probs require the forward position map"
            )
            # moe_permute backward must match the original top-k probs shape.
            unpermuted_probs = fused_moe_indices_converter_backward_impl(
                unpermuted_probs,
                indices_position_map,
                probs_topk_stride,
            )
        return unpermuted_tokens, unpermuted_probs


class TurboBackend:
    """HIP C++ path: ``row_id_map`` is [T+pad, 2E+1], signed one-based; supports padding / FP8."""

    @staticmethod
    def preprocess(
        routing_map,
        topk_indices,
        num_tokens,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        probs,
        probs_topk_stride,
    ):
        # topk_indices wins: it is the only format supporting topk-aligned probs.
        expert_map = topk_indices if topk_indices is not None else routing_map
        assert expert_map is not None, "TurboBackend requires routing_map or topk_indices"
        device = expert_map.device

        # Preprocessing kernel asserts num_tokens > 0.
        if num_tokens == 0:
            return (
                torch.zeros((pad_multiple, 2 * num_local_experts + 1), dtype=torch.int32, device=device),
                torch.zeros((num_local_experts,), dtype=torch.int64, device=device),
                torch.zeros((1,), dtype=torch.int32, device=device),
                torch.zeros((1,), dtype=torch.int32, device=device),
                0,
                None,
                None,
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
            num_permuted = int(num_permuted_tokens)
        else:
            num_permuted = int(tokens_per_expert.sum().item())
        return (
            row_id_map,
            tokens_per_expert,
            overflow_flag,
            num_dispatched_tokens,
            num_permuted,
            None,
            None,
        )

    @staticmethod
    def permute(
        tokens,
        row_id_map,
        num_dispatched_tokens,
        num_permuted,
        num_local_experts,
        hidden_size,
        pad_multiple,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
        probs_topk_stride,
    ):
        device = tokens.device
        # Kernel writes every output row (padding included), so empty is safe when it runs.
        run_kernel = num_permuted > 0 and tokens.shape[0] > 0
        alloc = torch.empty if run_kernel else torch.zeros

        permuted_tokens = alloc((num_permuted, hidden_size), dtype=tokens.dtype, device=device)
        if use_fp8 and scaling_factor is not None:
            permuted_scaling_factor = alloc(
                (num_permuted, scales_per_token), dtype=scaling_factor.dtype, device=device
            )
        else:
            permuted_scaling_factor = None
        permuted_probs = (
            alloc((num_permuted,), dtype=probs.dtype, device=device) if probs is not None else None
        )

        if run_kernel:
            # Kernel early-exit row limit; the allocated row count is always safe.
            if num_dispatched_tokens is None:
                num_dispatched_tokens = torch.full((1,), tokens.shape[0], dtype=torch.int32, device=device)
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
                num_permuted,
                probs_topk_stride,
            )
        return permuted_tokens, permuted_scaling_factor, permuted_probs

    @staticmethod
    def unpermute(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens,
        num_dispatched,
        num_local_experts,
        hidden_size,
        permuted_probs,
        probs_topk_stride,
        indices_position_map,
    ):
        device = permuted_tokens.device
        probs_row_width = probs_topk_stride if probs_topk_stride > 0 else num_local_experts

        # Kernel skipped here, so zero instead of empty.
        if permuted_tokens.shape[0] == 0 or num_dispatched == 0:
            unpermuted_tokens = permuted_tokens.new_zeros((num_dispatched, hidden_size))
            unpermuted_probs = (
                permuted_probs.new_zeros((num_dispatched, probs_row_width))
                if permuted_probs is not None
                else None
            )
            return unpermuted_tokens, unpermuted_probs

        unpermuted_tokens = torch.empty(
            (num_dispatched, hidden_size), dtype=permuted_tokens.dtype, device=device
        )
        unpermuted_probs = (
            torch.empty((num_dispatched, probs_row_width), dtype=permuted_probs.dtype, device=device)
            if permuted_probs is not None
            else None
        )
        # Kernel early-exit row limit; the allocated row count is always safe.
        if num_dispatched_tokens is None:
            num_dispatched_tokens = torch.full((1,), num_dispatched, dtype=torch.int32, device=device)
        torch.ops.primus_turbo_cpp_extension.unpermute(
            permuted_tokens,
            unpermuted_tokens,
            permuted_probs,
            unpermuted_probs,
            row_id_map,
            num_dispatched_tokens,
            num_local_experts,
            hidden_size,
            permuted_probs is not None,
            probs_topk_stride,
        )
        return unpermuted_tokens, unpermuted_probs


_BACKEND_REGISTRY: Dict[BackendType, Type[MoEPermuteBackend]] = {
    BackendType.TRITON: TritonMaskMapBackend,
    BackendType.TURBO: TurboBackend,
}


def get_moe_permute_backend_cls(backend: BackendType) -> Type[MoEPermuteBackend]:
    cls = _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise ValueError(f"Unknown moe_permute backend {backend}")
    return cls


_torch_custom_op_wrapper = torch.library.custom_op

# Schemas are spelled out: infer_schema cannot express optional tensor returns.
# ``backend`` crosses the boundary as ``BackendType.value``; shape-derived sizes are SymInt.
_PREPROCESS_SCHEMA = (
    "(Tensor? routing_map, Tensor? topk_indices, SymInt num_tokens, int num_local_experts, "
    "int num_topk, int pad_multiple, int num_permuted_tokens, Tensor? probs, "
    "int probs_topk_stride, int backend) "
    "-> (Tensor, Tensor, Tensor, Tensor?, SymInt, Tensor?, Tensor?)"
)
_PERMUTE_SCHEMA = (
    "(Tensor tokens, Tensor row_id_map, Tensor? num_dispatched_tokens, SymInt num_permuted, "
    "int num_local_experts, SymInt hidden_size, int pad_multiple, Tensor? scaling_factor, "
    "Tensor? probs, int scales_per_token, bool use_fp8, int probs_topk_stride, int backend) "
    "-> (Tensor, Tensor?, Tensor?)"
)
_UNPERMUTE_SCHEMA = (
    "(Tensor permuted_tokens, Tensor row_id_map, Tensor? num_dispatched_tokens, "
    "SymInt num_dispatched, int num_local_experts, SymInt hidden_size, Tensor? permuted_probs, "
    "int probs_topk_stride, Tensor? indices_position_map, int backend) -> (Tensor, Tensor?)"
)


@_torch_custom_op_wrapper(
    "primus_turbo::moe_permute_preprocess",
    mutates_args=(),
    device_types="cuda",
    schema=_PREPROCESS_SCHEMA,
)
def _moe_permute_preprocess(
    routing_map,
    topk_indices,
    num_tokens,
    num_local_experts,
    num_topk,
    pad_multiple,
    num_permuted_tokens,
    probs,
    probs_topk_stride,
    backend,
):
    return get_moe_permute_backend_cls(BackendType(backend)).preprocess(
        routing_map,
        topk_indices,
        num_tokens,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        probs,
        probs_topk_stride,
    )


@_moe_permute_preprocess.register_fake
def _moe_permute_preprocess_meta(
    routing_map,
    topk_indices,
    num_tokens,
    num_local_experts,
    num_topk,
    pad_multiple,
    num_permuted_tokens,
    probs,
    probs_topk_stride,
    backend,
):
    expert_map = routing_map if routing_map is not None else topk_indices
    device = expert_map.device
    is_triton = BackendType(backend) is BackendType.TRITON

    # TRITON forbids pad_multiple, so the padded height covers both backends.
    row_id_map = torch.empty(
        (num_tokens + pad_multiple, 2 * num_local_experts + 1), dtype=torch.int32, device=device
    )
    tokens_per_expert = torch.empty((num_local_experts,), dtype=torch.int64, device=device)
    if is_triton:
        # TRITON never drops: host flag, and no dispatched-token bound.
        overflow_flag = torch.empty((1,), dtype=torch.int32)
        num_dispatched_tokens = None
    else:
        overflow_flag = torch.empty((1,), dtype=torch.int32, device=device)
        num_dispatched_tokens = torch.empty((1,), dtype=torch.int32, device=device)

    # Only the caller-provided capacity is known at trace time.
    if num_permuted_tokens > 0:
        num_permuted = num_permuted_tokens
    else:
        num_permuted = torch.library.get_ctx().new_dynamic_size()

    multihot_probs, indices_position_map = None, None
    if is_triton and routing_map is None:
        indices_position_map = torch.empty((num_tokens, num_local_experts), dtype=torch.int32, device=device)
        if probs is not None and probs_topk_stride > 0:
            multihot_probs = probs.new_empty((num_tokens, num_local_experts))
    return (
        row_id_map,
        tokens_per_expert,
        overflow_flag,
        num_dispatched_tokens,
        num_permuted,
        multihot_probs,
        indices_position_map,
    )


@_torch_custom_op_wrapper(
    "primus_turbo::moe_permute",
    mutates_args=(),
    device_types="cuda",
    schema=_PERMUTE_SCHEMA,
)
def _moe_permute(
    tokens,
    row_id_map,
    num_dispatched_tokens,
    num_permuted,
    num_local_experts,
    hidden_size,
    pad_multiple,
    scaling_factor,
    probs,
    scales_per_token,
    use_fp8,
    probs_topk_stride,
    backend,
):
    return get_moe_permute_backend_cls(BackendType(backend)).permute(
        tokens,
        row_id_map,
        num_dispatched_tokens,
        num_permuted,
        num_local_experts,
        hidden_size,
        pad_multiple,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
        probs_topk_stride,
    )


@_moe_permute.register_fake
def _moe_permute_meta(
    tokens,
    row_id_map,
    num_dispatched_tokens,
    num_permuted,
    num_local_experts,
    hidden_size,
    pad_multiple,
    scaling_factor,
    probs,
    scales_per_token,
    use_fp8,
    probs_topk_stride,
    backend,
):
    permuted_tokens = tokens.new_empty((num_permuted, hidden_size))
    permuted_scaling_factor = (
        scaling_factor.new_empty((num_permuted, scales_per_token))
        if use_fp8 and scaling_factor is not None
        else None
    )
    permuted_probs = probs.new_empty((num_permuted,)) if probs is not None else None
    return permuted_tokens, permuted_scaling_factor, permuted_probs


@_torch_custom_op_wrapper(
    "primus_turbo::moe_unpermute",
    mutates_args=(),
    device_types="cuda",
    schema=_UNPERMUTE_SCHEMA,
)
def _moe_unpermute(
    permuted_tokens,
    row_id_map,
    num_dispatched_tokens,
    num_dispatched,
    num_local_experts,
    hidden_size,
    permuted_probs,
    probs_topk_stride,
    indices_position_map,
    backend,
):
    return get_moe_permute_backend_cls(BackendType(backend)).unpermute(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens,
        num_dispatched,
        num_local_experts,
        hidden_size,
        permuted_probs,
        probs_topk_stride,
        indices_position_map,
    )


@_moe_unpermute.register_fake
def _moe_unpermute_meta(
    permuted_tokens,
    row_id_map,
    num_dispatched_tokens,
    num_dispatched,
    num_local_experts,
    hidden_size,
    permuted_probs,
    probs_topk_stride,
    indices_position_map,
    backend,
):
    unpermuted_tokens = permuted_tokens.new_empty((num_dispatched, hidden_size))
    # Top-k-aligned probs are narrowed back by the converter; multihot keeps one column per expert.
    probs_row_width = probs_topk_stride if probs_topk_stride > 0 else num_local_experts
    unpermuted_probs = (
        permuted_probs.new_empty((num_dispatched, probs_row_width)) if permuted_probs is not None else None
    )
    return unpermuted_tokens, unpermuted_probs


def moe_permute_process_impl(
    backend: BackendType,
    routing_map: Optional[torch.Tensor],
    topk_indices: Optional[torch.Tensor],
    num_tokens: int,
    num_local_experts: int,
    num_topk: int,
    pad_multiple: int,
    num_permuted_tokens: int,
    probs: Optional[torch.Tensor],
    probs_topk_stride: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    int,
    Optional[torch.Tensor],
    int,
    Optional[torch.Tensor],
]:
    (
        row_id_map,
        tokens_per_expert,
        overflow_flag,
        num_dispatched_tokens,
        num_permuted,
        multihot_probs,
        indices_position_map,
    ) = _moe_permute_preprocess(
        routing_map,
        topk_indices,
        num_tokens,
        num_local_experts,
        num_topk,
        pad_multiple,
        num_permuted_tokens,
        probs,
        probs_topk_stride,
        backend.value,
    )
    # Resolved here, not in the op: a custom op may not return one of its own inputs.
    if multihot_probs is not None:
        probs, probs_topk_stride = multihot_probs, 0
    return (
        row_id_map,
        tokens_per_expert,
        overflow_flag,
        num_dispatched_tokens,
        num_permuted,
        probs,
        probs_topk_stride,
        indices_position_map,
    )


def moe_permute_impl(
    backend: BackendType,
    tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens: Optional[torch.Tensor],
    num_permuted: int,
    num_local_experts: int,
    hidden_size: int,
    pad_multiple: int,
    scaling_factor: Optional[torch.Tensor],
    probs: Optional[torch.Tensor],
    scales_per_token: int,
    use_fp8: bool,
    probs_topk_stride: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    return _moe_permute(
        tokens,
        row_id_map,
        num_dispatched_tokens,
        num_permuted,
        num_local_experts,
        hidden_size,
        pad_multiple,
        scaling_factor,
        probs,
        scales_per_token,
        use_fp8,
        probs_topk_stride,
        backend.value,
    )


def moe_unpermute_impl(
    backend: BackendType,
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens: Optional[torch.Tensor],
    num_dispatched: int,
    num_local_experts: int,
    hidden_size: int,
    permuted_probs: Optional[torch.Tensor],
    probs_topk_stride: int,
    indices_position_map: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return _moe_unpermute(
        permuted_tokens,
        row_id_map,
        num_dispatched_tokens,
        num_dispatched,
        num_local_experts,
        hidden_size,
        permuted_probs,
        probs_topk_stride,
        indices_position_map,
        backend.value,
    )
