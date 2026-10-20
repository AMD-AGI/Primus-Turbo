###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Expert-bias gradients for the Mega MoE stage2 backward.

The forward folds both biases into existing kernels (fc1's into the activation, fc2's into the
combine push); their gradients are pool-row reductions, run here as two-phase Triton kernels so
they come out in a fixed order.
"""

import torch
import triton

from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import DispatchHandle
from primus_turbo.flydsl.mega.bf16.symm_buffer import BLOCK_M as POOL_BLOCK
from primus_turbo.triton.moe.mega_moe_bias_kernel import (
    expert_partials_sum_kernel,
    expert_rows_sum_partial_kernel,
    fc2_bias_backward_partial_kernel,
)

_BLOCK_C = 256
# rows a fc2-bias program owns: a whole number per pool block, so every row has exactly one owner
_FC2_BLOCK_R = 64


def _sum_partials_per_expert(partial: torch.Tensor, handle: DispatchHandle, parts_per_tile: int, dtype):
    num_experts = handle.num_tokens_per_expert.numel()
    cols = partial.shape[1]
    out = torch.empty((num_experts, cols), dtype=dtype, device=partial.device)
    expert_partials_sum_kernel[(num_experts, triton.cdiv(cols, _BLOCK_C))](
        partial,
        handle.num_tokens_per_expert_prefix,
        out,
        cols,
        POOL_BLOCK=POOL_BLOCK,
        PARTS_PER_TILE=parts_per_tile,
        BLOCK_C=_BLOCK_C,
    )
    return out


def fc1_bias_backward(grad_l1: torch.Tensor, handle: DispatchHandle, dtype: torch.dtype) -> torch.Tensor:
    """d(fc1 bias)[e] = the sum of expert e's real rows of ``grad_l1`` [rows, 2I]."""
    rows, cols = grad_l1.shape
    assert rows % POOL_BLOCK == 0, f"grad_l1 spans whole pool blocks, got {rows} rows"
    num_tiles = rows // POOL_BLOCK
    partial = torch.empty((num_tiles, cols), dtype=torch.float32, device=grad_l1.device)
    if num_tiles:
        expert_rows_sum_partial_kernel[(num_tiles, triton.cdiv(cols, _BLOCK_C))](
            grad_l1,
            handle.tile_to_expert,
            handle.num_tokens_per_expert_prefix,
            handle.num_tokens_per_expert,
            partial,
            cols,
            grad_l1.stride(0),
            POOL_BLOCK=POOL_BLOCK,
            BLOCK_R=32,
            BLOCK_C=_BLOCK_C,
        )
    return _sum_partials_per_expert(partial, handle, 1, dtype)


def fc1_bias_backward_from_block_sums(
    block_sums: torch.Tensor, offs_pc: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """d(fc1 bias) from the mxfp8 dual-quant's per-32-row-block column sums ``[n_pblk, 2I]``:
    group g owns blocks ``[offs_pc[g] / 32, offs_pc[g + 1] / 32)``, its rows padded to 128."""
    num_experts = offs_pc.numel() - 1
    cols = block_sums.shape[1]
    out = torch.empty((num_experts, cols), dtype=dtype, device=block_sums.device)
    expert_partials_sum_kernel[(num_experts, triton.cdiv(cols, _BLOCK_C))](
        block_sums,
        offs_pc,
        out,
        cols,
        POOL_BLOCK=32,
        PARTS_PER_TILE=1,
        BLOCK_C=_BLOCK_C,
    )
    return out


def fc2_bias_backward(
    dispatched_dy: torch.Tensor,
    handle: DispatchHandle,
    dispatch_weights: torch.Tensor,
    fc2_bias: torch.Tensor,
    grad_gate: torch.Tensor,
) -> torch.Tensor:
    """d(fc2 bias)[e] = sum_r p_r dy_r over expert e's rows; adds dy_r . b_e into ``grad_gate`` in place.

    ``dispatched_dy`` is the output gradient as the L2 dgrad dispatched it, one row per received
    token; a pool row reads its token's row through ``handle.pool_row_to_recv_token``."""
    rows = dispatch_weights.numel()
    hidden = fc2_bias.shape[1]
    assert dispatched_dy.shape[1] == hidden and dispatched_dy.stride(1) == 1
    assert grad_gate.numel() == rows and grad_gate.dtype == torch.float32
    num_tiles = rows // POOL_BLOCK
    groups = POOL_BLOCK // _FC2_BLOCK_R
    partial = torch.empty((num_tiles * groups, hidden), dtype=torch.float32, device=fc2_bias.device)
    if num_tiles:
        fc2_bias_backward_partial_kernel[(num_tiles, groups)](
            dispatched_dy,
            handle.pool_row_to_recv_token,
            dispatch_weights,
            handle.tile_to_expert,
            handle.num_tokens_per_expert_prefix,
            handle.num_tokens_per_expert,
            fc2_bias.contiguous(),
            grad_gate,
            partial,
            hidden,
            dispatched_dy.stride(0),
            POOL_BLOCK=POOL_BLOCK,
            BLOCK_R=_FC2_BLOCK_R,
            BLOCK_H=_BLOCK_C,
        )
    return _sum_partials_per_expert(partial, handle, groups, fc2_bias.dtype)
