###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Public API for the fused Mega MoE FFN kernel.

Mirrors DeepGEMM's ``deep_gemm.mega`` API surface 1:1 so existing call
sites in DeepSeek-style MoE pipelines can switch backends with minimal
diff:

* :class:`SymmBuffer`                    — symmetric memory + tensor views
* :func:`get_symm_buffer_for_mega_moe`   — factory matching the DG name
* :func:`transform_weights_for_mega_moe` — weight pre-processing
* :func:`fp8_fp4_mega_moe`               — fused dispatch + GG1 + SwiGLU + GG2 + combine

The underlying GPU kernel is a stub in this revision; the Python layer
already wires up symmetric memory rendezvous, buffer slicing and shape
validation so that downstream integrations can be written against the
final API.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from primus_turbo.pytorch.core.symm_mem import SymmetricMemory
from primus_turbo.pytorch.kernels.mega_moe import (
    SymmBufferLayout,
    fp8_fp4_mega_moe_impl,
    get_symm_buffer_size_for_mega_moe,
    get_token_alignment_for_mega_moe,
    transform_l1_weights_for_mega_moe,
    transform_l2_weights_for_mega_moe,
)

__all__ = [
    "SymmBuffer",
    "fp8_fp4_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "transform_weights_for_mega_moe",
]


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Symmetric buffer wrapper
# ---------------------------------------------------------------------------


class SymmBuffer:
    """Container holding the symmetric memory allocation + tensor views.

    Equivalent of DeepGEMM's ``deep_gemm.mega.SymmBuffer``.  Each rank
    in ``group`` calls into this class with the same arguments; the
    constructor rendezvous'es IPC handles internally and exposes
    per-region tensor views that the caller can copy inputs into
    before invoking :func:`fp8_fp4_mega_moe`.

    The exposed views mirror DG's ``slice_input_buffers`` exactly:
    ``(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts,
    l2_acts_sf)``.  Internal regions (per-expert weight columns,
    BF16 combine buffer) are not exposed — the kernel addresses them
    directly through ``sym_buffer_ptrs``.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
    ) -> None:
        if activation != "swiglu":
            raise NotImplementedError(f"SymmBuffer only supports activation='swiglu', got {activation!r}")

        self.group = group
        self.num_experts = int(num_experts)
        self.num_max_tokens_per_rank = int(num_max_tokens_per_rank)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.use_fp8_dispatch = bool(use_fp8_dispatch)
        self.activation = activation

        # Per-rank token count must be aligned.
        self.num_max_tokens_per_rank = _align_up(
            self.num_max_tokens_per_rank, get_token_alignment_for_mega_moe()
        )

        # Compute the layout (offsets + total bytes).
        self.layout: SymmBufferLayout = get_symm_buffer_size_for_mega_moe(
            num_ranks=group.size(),
            num_experts=self.num_experts,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            num_topk=self.num_topk,
            hidden=self.hidden,
            intermediate_hidden=self.intermediate_hidden,
            use_fp8_dispatch=self.use_fp8_dispatch,
        )

        # Allocate symmetric memory.  ``SymmetricMemory`` handles IPC
        # rendezvous and exposes per-rank device pointers.
        self.symm_mem = SymmetricMemory(group, alloc_size=int(self.layout.total_bytes))
        # ``buffer`` is a flat int8 view over this rank's local slice.
        self.buffer: torch.Tensor = self.symm_mem.get_buffer(
            rank=self.symm_mem.rank,
            sizes=[int(self.layout.total_bytes)],
            dtype=torch.int8,
        )

        # Create per-region tensor views (mirrors DG's 8-tuple).
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
        ) = self._slice_views()

        # Initialise to zero and barrier so peers see a clean buffer.
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    #  Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_ranks(self) -> int:
        return self.symm_mem.world_size

    @property
    def rank(self) -> int:
        return self.symm_mem.rank

    @property
    def buffer_ptrs(self) -> list[int]:
        """Per-rank base device pointers used by the launcher."""
        return list(self.symm_mem.buffer_ptrs)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _view_at(
        self,
        offset_bytes: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a typed view over the local symmetric buffer."""
        if offset_bytes % dtype.itemsize != 0:
            raise ValueError(
                f"SymmBuffer: offset {offset_bytes} is not aligned to "
                f"dtype {dtype} (itemsize={dtype.itemsize})"
            )
        storage_offset = offset_bytes // dtype.itemsize
        return self.symm_mem.get_buffer(
            rank=self.rank, sizes=list(shape), dtype=dtype, storage_offset=storage_offset
        )

    def _slice_views(
        self,
    ) -> Tuple[
        torch.Tensor,  # x          FP8   [N, H]
        torch.Tensor,  # x_sf       int   [N, H/128]
        torch.Tensor,  # topk_idx   int64 [N, K]
        torch.Tensor,  # topk_w     f32   [N, K]
        torch.Tensor,  # l1_acts    FP8   [P, H]
        torch.Tensor,  # l1_acts_sf int   [P_sf, H/128]
        torch.Tensor,  # l2_acts    FP8   [P, I]
        torch.Tensor,  # l2_acts_sf int   [P_sf, I/128]
    ]:
        layout = self.layout

        x_dtype = torch.float8_e4m3fnuz if self.use_fp8_dispatch else torch.bfloat16

        x = self._view_at(
            layout.input_x_offset,
            shape=(self.num_max_tokens_per_rank, self.hidden),
            dtype=x_dtype,
        )
        x_sf = self._view_at(
            layout.input_x_sf_offset,
            shape=(self.num_max_tokens_per_rank, self.hidden // 128),
            dtype=torch.int32,
        )
        topk_idx = self._view_at(
            layout.input_topk_idx_offset,
            shape=(self.num_max_tokens_per_rank, self.num_topk),
            dtype=torch.int64,
        )
        topk_weights = self._view_at(
            layout.input_topk_weights_offset,
            shape=(self.num_max_tokens_per_rank, self.num_topk),
            dtype=torch.float32,
        )

        l1_acts = self._view_at(
            layout.l1_pool_x_offset,
            shape=(layout.num_max_pool_tokens, self.hidden),
            dtype=x_dtype,
        )
        l1_acts_sf = self._view_at(
            layout.l1_pool_x_sf_offset,
            shape=(layout.num_padded_sf_pool_tokens, self.hidden // 128),
            dtype=torch.int32,
        )
        l2_acts = self._view_at(
            layout.l2_pool_x_offset,
            shape=(layout.num_max_pool_tokens, self.intermediate_hidden),
            dtype=x_dtype,
        )
        l2_acts_sf = self._view_at(
            layout.l2_pool_x_sf_offset,
            shape=(layout.num_padded_sf_pool_tokens, self.intermediate_hidden // 128),
            dtype=torch.int32,
        )
        return (
            x,
            x_sf,
            topk_idx,
            topk_weights,
            l1_acts,
            l1_acts_sf,
            l2_acts,
            l2_acts_sf,
        )

    def destroy(self) -> None:
        """Release the symmetric memory + IPC handles."""
        # Drop tensor refs so torch can release backing storage.
        for attr in (
            "x",
            "x_sf",
            "topk_idx",
            "topk_weights",
            "l1_acts",
            "l1_acts_sf",
            "l2_acts",
            "l2_acts_sf",
            "buffer",
        ):
            setattr(self, attr, None)
        if self.symm_mem is not None:
            self.symm_mem.destroy()
            self.symm_mem = None


# ---------------------------------------------------------------------------
# Factories / public ops
# ---------------------------------------------------------------------------


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    *,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
) -> SymmBuffer:
    """Allocate the symmetric buffer used by :func:`fp8_fp4_mega_moe`.

    Mirrors DG's ``deep_gemm.mega.get_symm_buffer_for_mega_moe``.
    """

    return SymmBuffer(
        group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        use_fp8_dispatch=use_fp8_dispatch,
        activation=activation,
    )


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Transform (weight, scaling-factor) pairs into the kernel-native layout.

    * L1: interleave gate / up along N then permute the SF tensor.
    * L2: permute the SF tensor only.

    Match DeepGEMM's ``transform_weights_for_mega_moe`` semantics so
    pre-trained weight checkpoints can be shared.
    """

    return (
        transform_l1_weights_for_mega_moe(l1_weights),
        transform_l2_weights_for_mega_moe(l2_weights),
    )


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    *,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
) -> None:
    """Run the fused mega-MoE FFN (dispatch + GG1 + SwiGLU + GG2 + combine).

    FP8 activations × FP4 weights variant.  Mirrors DG's
    ``deep_gemm.mega.fp8_fp4_mega_moe`` API surface 1:1.

    Args:
        y: BF16 output tensor of shape ``[num_tokens, hidden]``.  This
            rank's combined per-token output is written in-place.
        l1_weights: ``(weight, sf)`` pair for the up-projection.  Must
            already be in the kernel-native layout (use
            :func:`transform_weights_for_mega_moe`).
        l2_weights: ``(weight, sf)`` pair for the down-projection.
        sym_buffer: A :class:`SymmBuffer` previously allocated by
            :func:`get_symm_buffer_for_mega_moe`.  The caller is
            expected to copy ``x``, ``x_sf``, ``topk_idx`` and
            ``topk_weights`` into the buffer views before invoking.
        cumulative_local_expert_recv_stats: Optional int32 tensor of
            shape ``[num_experts_per_rank]`` accumulating the number of
            received tokens per local expert across calls.
        recipe: SF granularity ``(M, N, K)``.  Only ``(1, 1, 32)`` is
            currently supported.
        activation: Activation function name.  Only ``swiglu`` is
            currently supported.
        activation_clamp: Optional clamp value for SwiGLU stability.
            ``None`` disables clamping.
        fast_math: Enable fast-math approximations inside the activation.
    """

    if recipe != (1, 1, 32):
        raise NotImplementedError(f"fp8_fp4_mega_moe currently supports recipe=(1, 1, 32), got {recipe}")
    if y.dim() != 2:
        raise ValueError(f"fp8_fp4_mega_moe: y must be 2D, got shape {tuple(y.shape)}")
    if y.size(1) != sym_buffer.hidden:
        raise ValueError(
            f"fp8_fp4_mega_moe: y hidden dim ({y.size(1)}) does not match buffer hidden "
            f"({sym_buffer.hidden})"
        )

    num_tokens = int(y.size(0))
    if num_tokens > sym_buffer.num_max_tokens_per_rank:
        raise ValueError(
            f"fp8_fp4_mega_moe: num_tokens ({num_tokens}) exceeds "
            f"num_max_tokens_per_rank ({sym_buffer.num_max_tokens_per_rank})"
        )

    fp8_fp4_mega_moe_impl(
        y,
        l1_weights,
        l2_weights,
        sym_buffer=sym_buffer.buffer,
        sym_buffer_ptrs=sym_buffer.buffer_ptrs,
        rank_idx=sym_buffer.rank,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_experts=sym_buffer.num_experts,
        num_topk=sym_buffer.num_topk,
        num_tokens=num_tokens,
        hidden=sym_buffer.hidden,
        intermediate_hidden=sym_buffer.intermediate_hidden,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
