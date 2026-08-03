###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage mega MoE stage1 (gate-up) MXFP8 FlyDSL kernel composition.

fp8 sibling of ``fused_mega_moe_stage1_impl``: the same kernels in the same order as the fused
``fused_mega_moe_forward_fp8_impl`` / ``fused_mega_moe_backward_fp8_impl``, cut at ``l1`` (pre-SwiGLU)
so w1 and w2 can sit in separate modules and become independent DDP gradient boundaries. No new
kernel is introduced here -- every call below is a helper the fused path already uses.

Stage1 owns: forward dispatch + fc1 (NT mxfp8) and the fc1-input pool requant for dW1; backward
STEP3 (fc1 dgrad + combine -> dx) and the dW1 variable-K wgrad.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import dispatch_grouped_gemm_mxfp8_flydsl_kernel
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import (
    _l1_dgrad_combine_mxfp8_flydsl_kernel,
    _mxfp8_variable_k_wgrad_dw1,
    prepare_dw1_pool_operand_fp8,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_forward_fp8_impl import (
    _L1_NUM_DISPATCH_CU,
    _L1_NUM_PRESHUFFLE_CU,
    _w1_fp8_cached,
)

__all__ = [
    "fused_mega_moe_stage1_forward_fp8_impl",
    "fused_mega_moe_stage1_backward_fp8_impl",
]

# fp8 dispatch handle: dispatch_prologue's 11 tables + num_tile_blocks appended by the L1 kernel.
_HANDLE_LEN = 12


def fused_mega_moe_stage1_forward_fp8_impl(
    x: torch.Tensor,
    w1: torch.Tensor,
    group,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    block_m: int,
    block_n: int,
    save_bwd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, tuple, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
    """Fused mxfp8 dispatch + fc1 (NT).

    Returns ``(l1, dispatch_weights, handle, pool_x_colwise, colwise_meta)``. ``l1`` is the fc1
    output [P, 2I] that stage2 SwiGLUs; ``dispatch_weights`` is cloned out of the live symm pool
    because stage2's backward needs it as the SwiGLU^T scale long after later stages overwrite the
    buffer. ``pool_x_colwise`` / ``colwise_meta`` (None unless ``save_bwd``) are dW1's ``b`` operand,
    requantized here while the fc1-input pool is still live in symm -- the fused path does the same,
    only after the L2 combine instead of right after dispatch.
    """
    # int64 end-to-end (combine reads topk i64)
    topk_idx = topk_idx.to(torch.int64)
    w1q, w1s = _w1_fp8_cached(w1)

    l1, handle, dispatch_weights, pool_x_fp8 = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        x,
        w1q, w1s,
        group,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_dispatch_cu=_L1_NUM_DISPATCH_CU,
        num_preshuffle_cu=_L1_NUM_PRESHUFFLE_CU,
        BM=block_m, BN=block_n,
    )
    assert len(handle) == _HANDLE_LEN, f"fp8 dispatch handle len {len(handle)} != {_HANDLE_LEN}; ABI changed"

    pool_x_colwise, colwise_meta = (
        prepare_dw1_pool_operand_fp8(pool_x_fp8, handle) if save_bwd else (None, None)
    )
    return l1, dispatch_weights.clone(), tuple(handle), pool_x_colwise, colwise_meta


def fused_mega_moe_stage1_backward_fp8_impl(
    grad_l1_rowwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    grad_l1_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    grad_gate: torch.Tensor,
    pool_x_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    colwise_meta: dict,
    w1: torch.Tensor,
    handle: tuple,
    group,
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_topk: int,
    block_m: int,
    block_n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """STEP3 (fc1 dgrad + combine + reduce -> dx, grad_gate scatter) then the dW1 variable-K wgrad.

    Both grad_l1 operands come pre-quantized from stage2's fused SwiGLU^T dual-quant -- the fp8 path
    never materializes a bf16 grad_l1, which is why they arrive through the state side-channel
    rather than the ``l1`` gradient slot. STEP3 and dW1 stay serial on the default stream.

    Returns ``(dx, grad_topk_weights, dW1)`` with dW1 in the wgrad's bf16 accumulate dtype.
    """
    dx, grad_topk_weights = _l1_dgrad_combine_mxfp8_flydsl_kernel(
        w1, group, handle, block_m, block_n,
        grad_l1_rowwise_fp8=grad_l1_rowwise_fp8,
        grad_gate=grad_gate,
        topk_idx=topk_idx,
        num_tokens=num_tokens,
        num_topk=num_topk,
    )
    dW1 = _mxfp8_variable_k_wgrad_dw1(grad_l1_colwise_fp8, pool_x_colwise_fp8, colwise_meta)
    return dx, grad_topk_weights, dW1
