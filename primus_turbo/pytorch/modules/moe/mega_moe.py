###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""High-level ``nn.Module`` wrapper around the fused Mega MoE FFN.

Provides a torch ``Module`` that owns:
    * the symmetric memory buffer (one per group, lazily allocated),
    * the kernel-native weight layout,
    * a forward() that copies inputs into the symm buffer and invokes
      :func:`primus_turbo.pytorch.ops.fp8_mega_moe`.

This wrapper plays a similar role to ``DeepEPTokenDispatcher`` — it
exposes a stable nn.Module surface while hiding the symmetric memory
lifecycle.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from primus_turbo.pytorch.ops.mega_moe import (
    MegaMoESymmBuffer,
    fp8_mega_moe,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
)

__all__ = ["FusedMegaMoE"]


class FusedMegaMoE(nn.Module):
    """Single-call fused MoE FFN.

    Args:
        num_experts: Global number of experts (must divide world size).
        num_topk: Top-k routing count.
        hidden: Hidden size of the model.
        intermediate_hidden: Intermediate hidden size (per gate / up branch).
        ep_group: Process group across which experts are sharded.
        num_max_tokens_per_rank: Upper bound on per-rank token count.
            Used to size the symmetric memory buffer.
        use_fp8_dispatch: Use FP8 (E4M3) dispatch traffic.  Currently
            the only supported mode.
        activation: Activation function.  Only ``swiglu`` is supported.
        activation_clamp: Optional SwiGLU clamp value.
        fast_math: Enable fast-math approximations.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        ep_group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
        activation_clamp: Optional[float] = None,
        fast_math: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.ep_group = ep_group
        self.num_max_tokens_per_rank = int(num_max_tokens_per_rank)
        self.use_fp8_dispatch = bool(use_fp8_dispatch)
        self.activation = activation
        self.activation_clamp = activation_clamp
        self.fast_math = fast_math

        if num_experts % ep_group.size() != 0:
            raise ValueError(
                f"FusedMegaMoE: num_experts ({num_experts}) must be divisible by "
                f"ep_group.size() ({ep_group.size()})"
            )
        self.num_experts_per_rank = num_experts // ep_group.size()

        # Weights are populated externally via ``load_weights`` so that
        # users can integrate the module with arbitrary weight
        # checkpoints.  We do not allocate parameters here because the
        # expected dtype (FP4) is not a native torch parameter dtype.
        self._l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._symm_buffer: Optional[MegaMoESymmBuffer] = None

    # ------------------------------------------------------------------
    #  Symmetric buffer lifecycle
    # ------------------------------------------------------------------

    def _ensure_symm_buffer(self) -> MegaMoESymmBuffer:
        if self._symm_buffer is None:
            self._symm_buffer = get_symm_buffer_for_mega_moe(
                self.ep_group,
                num_experts=self.num_experts,
                num_max_tokens_per_rank=self.num_max_tokens_per_rank,
                num_topk=self.num_topk,
                hidden=self.hidden,
                intermediate_hidden=self.intermediate_hidden,
                use_fp8_dispatch=self.use_fp8_dispatch,
                activation=self.activation,
            )
        return self._symm_buffer

    def destroy(self) -> None:
        if self._symm_buffer is not None:
            self._symm_buffer.destroy()
            self._symm_buffer = None

    # ------------------------------------------------------------------
    #  Weight management
    # ------------------------------------------------------------------

    def load_weights(
        self,
        l1_weights: Tuple[torch.Tensor, torch.Tensor],
        l2_weights: Tuple[torch.Tensor, torch.Tensor],
        *,
        already_transformed: bool = False,
    ) -> None:
        """Stash FP4 weights, optionally pre-transforming them."""
        if not already_transformed:
            l1_weights, l2_weights = transform_weights_for_mega_moe(l1_weights, l2_weights)
        self._l1_weights = l1_weights
        self._l2_weights = l2_weights

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._l1_weights is None or self._l2_weights is None:
            raise RuntimeError(
                "FusedMegaMoE.forward: weights have not been loaded; " "call `load_weights` before `forward`."
            )

        buffer = self._ensure_symm_buffer()

        num_tokens = int(x_fp8.size(0))
        if num_tokens > buffer.num_max_tokens_per_rank:
            raise ValueError(
                f"FusedMegaMoE.forward: num_tokens ({num_tokens}) exceeds "
                f"num_max_tokens_per_rank ({buffer.num_max_tokens_per_rank})"
            )

        # Stage inputs into the symmetric buffer so peers can pull from
        # it during the dispatch phase.  These copies may later be
        # fused into the kernel that produced ``x_fp8`` / ``x_sf``.
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, self.hidden), dtype=torch.bfloat16, device=x_fp8.device)
        fp8_mega_moe(
            y,
            self._l1_weights,
            self._l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            activation=self.activation,
            activation_clamp=self.activation_clamp,
            fast_math=self.fast_math,
        )
        return y
