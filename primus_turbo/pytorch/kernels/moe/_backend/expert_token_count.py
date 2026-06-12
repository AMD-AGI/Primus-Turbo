###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Per-expert token counting (Mori dispatch post-processing)."""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl


def _compute_expert_token_info_configs() -> List[triton.Config]:
    """Autotune space for :func:`compute_expert_token_info_kernel`."""
    configs: List[triton.Config] = []
    for block_m in (64, 128, 256, 512, 1024):
        for num_warps in (1, 2, 4, 8):
            # Wave=64, cap threads/CTA to 1024 (HW limit on CDNA).
            if num_warps * 64 > 1024:
                continue
            # Keep at least one element per thread in the M dimension.
            if block_m < num_warps * 64:
                continue
            for num_stages in (1, 2):
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE_M": block_m},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=_compute_expert_token_info_configs(),
    key=["num_tokens", "num_topk"],
    reset_to_zero=["num_recv_tokens_per_expert_ptr"],
)
@triton.jit
def compute_expert_token_info_kernel(
    recv_topk_idx_ptr,
    deepep_topk_idx_ptr,
    num_recv_tokens_per_expert_ptr,
    total_recv_ptr,
    num_tokens: tl.int32,
    num_topk: tl.int32,
    num_local_experts: tl.int32,
    expert_base: tl.int32,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HAS_TOTAL_RECV: tl.constexpr,
):
    """Count per-expert received tokens and emit a DeepEP-format topk_idx."""
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offs = tl.arange(0, BLOCK_SIZE_K)
    bin_offs = tl.arange(0, BLOCK_SIZE_N)

    if HAS_TOTAL_RECV:
        total_recv = tl.load(total_recv_ptr)
        row_limit = tl.minimum(total_recv, num_tokens)
    else:
        row_limit = num_tokens

    in_footprint = row_offs < num_tokens
    in_valid_rows = row_offs < row_limit
    col_mask = col_offs < num_topk

    footprint_mask = in_footprint[:, None] & col_mask[None, :]
    load_mask = in_valid_rows[:, None] & col_mask[None, :]

    safe_row = tl.where(in_footprint, row_offs, 0)
    expert_offs = safe_row[:, None] * num_topk + col_offs[None, :]

    topk_experts = tl.load(
        recv_topk_idx_ptr + expert_offs,
        mask=load_mask,
        other=0,
    )

    local_experts = topk_experts - expert_base
    valid = load_mask & (local_experts >= 0) & (local_experts < num_local_experts)
    # Clamp invalid lanes to a legal bin; the histogram mask excludes them.
    safe_experts = tl.where(valid, local_experts, 0)

    # ``tl.histogram`` requires a flat 1D input; reshape the 2D tile.
    flat_experts = tl.reshape(safe_experts, [BLOCK_SIZE_M * BLOCK_SIZE_K])
    flat_mask = tl.reshape(valid, [BLOCK_SIZE_M * BLOCK_SIZE_K])
    local_counts = tl.histogram(flat_experts, BLOCK_SIZE_N, mask=flat_mask)

    # Flush CTA-local accumulator to global memory with one atomic per bin.
    bin_mask = (bin_offs < num_local_experts) & (local_counts > 0)
    tl.atomic_add(
        num_recv_tokens_per_expert_ptr + bin_offs,
        local_counts,
        sem="relaxed",
        scope="gpu",
        mask=bin_mask,
    )

    deepep_experts = tl.where(valid, local_experts, -1).to(topk_experts.dtype)
    tl.store(deepep_topk_idx_ptr + expert_offs, deepep_experts, mask=footprint_mask)


def compute_expert_token_info(
    recv_topk_idx: torch.Tensor,
    num_local_experts: int,
    *,
    expert_base: int = 0,
    total_recv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Count per-expert tokens and return recv_topk_idx in DeepEP layout."""
    assert recv_topk_idx.is_cuda, "recv_topk_idx must be a CUDA tensor"
    assert recv_topk_idx.dim() == 2, "recv_topk_idx must be 2D [num_tokens, num_topk]"

    device = recv_topk_idx.device
    num_tokens, num_topk = recv_topk_idx.shape

    num_recv_tokens_per_expert = torch.zeros(num_local_experts, dtype=recv_topk_idx.dtype, device=device)
    deepep_like_topk_idx = torch.empty_like(recv_topk_idx)

    if num_tokens == 0 or num_topk == 0 or num_local_experts == 0:
        # Nothing to count; fill with the DeepEP padding sentinel.
        if deepep_like_topk_idx.numel() > 0:
            deepep_like_topk_idx.fill_(-1)
        return num_recv_tokens_per_expert, deepep_like_topk_idx

    has_total_recv = total_recv is not None
    if has_total_recv:
        assert total_recv.is_cuda, "total_recv must be a CUDA tensor"
        assert total_recv.numel() >= 1, "total_recv must have at least 1 element"
        # Triton expects an int32 pointer; coerce if needed.
        if total_recv.dtype != torch.int32:
            total_recv = total_recv.to(torch.int32)
        total_recv_ptr = total_recv
    else:
        # Dummy pointer; unused when HAS_TOTAL_RECV=False.
        total_recv_ptr = recv_topk_idx

    # ``tl.histogram`` requires ``num_bins`` to be a power of 2 on AMD Triton.
    block_size_n = triton.next_power_of_2(num_local_experts)

    # ``BLOCK_SIZE_M`` is picked by the autotuner; the grid size depends on it.
    def grid(META):
        return (triton.cdiv(num_tokens, META["BLOCK_SIZE_M"]),)  # noqa: E731

    compute_expert_token_info_kernel[grid](
        recv_topk_idx,
        deepep_like_topk_idx,
        num_recv_tokens_per_expert,
        total_recv_ptr,
        num_tokens,
        num_topk,
        num_local_experts,
        expert_base,
        BLOCK_SIZE_K=triton.next_power_of_2(num_topk),
        BLOCK_SIZE_N=block_size_n,
        HAS_TOTAL_RECV=has_total_recv,
    )
    return num_recv_tokens_per_expert, deepep_like_topk_idx
