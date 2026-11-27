###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The bf16 Mega MoE stage2 on gfx1250 with a per-expert post norm between the L2 GEMM and the combine.

A model whose expert output is ``p * RMSNorm(round((act @ w2^T) * (1 + w)))`` (Muse Spark's latent MoE) cannot
take the routing weight inside the activation, as :func:`fused_mega_moe_gfx1250_impl.stage2_forward` does: the
norm would cancel it. Here stage2 is the unweighted activation, the L2 grouped GEMM over the pool, the norm with
the routing weight per pool row, and the combine of ``gemm_combine_nt_gfx1250`` alone (``combine_only_gfx1250``).
Stage1 (dispatch + L1 GEMM) is unchanged; the routing-weight gradient comes out of the norm's backward and goes
back to stage1 as its ``grad_gate``, as the fused stage2's does.
"""

from typing import Optional

import torch
import triton

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_dispatch import (
    grouped_gemm_bf16_nn,
    grouped_gemm_bf16_nt,
    grouped_gemm_bf16_variable_k,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel import dispatch_only_gfx1250
from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import (
    combine_only_gfx1250,
    gemm_combine_nt_gfx1250,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.variable_k import (
    grouped_gemm_bf16_variable_k_flydsl_kernel as _variable_k,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.x_redispatch_kernel import get_x_redispatch
from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import DispatchHandle
from primus_turbo.flydsl.mega.gfx1250.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.gfx1250.swiglu_kernel import (
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.triton.moe.mega_post_expert_norm_kernel import (
    mega_post_norm_bwd_kernel,
    mega_post_norm_fwd_kernel,
)

# backward launch: rows per tile capped so a tile stays near 4096 elements, programs per CU for the dw partials
_BWD_TILE_ELEMS = 4096
_MAX_BWD_ROWS = 4
_BWD_PROGRAMS_PER_CU = 4


def _norm_launch(hidden: int):
    block = triton.next_power_of_2(hidden)
    return block, max(1, min(_MAX_BWD_ROWS, _BWD_TILE_ELEMS // block)), max(1, min(16, block // 256))


def stage2_post_norm_forward(
    l1_out: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    topk_idx: torch.Tensor,
    dispatch_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    activation=None,
):
    """Activation + L2 GEMM + post norm with the routing weight + combine.

    Returns ``(y [tokens, hidden], act, l2, rstd)``: ``act`` (the unweighted activation, dW2's operand), ``l2``
    (the L2 output before the norm) and ``rstd`` (per pool row) for :func:`stage2_post_norm_backward`."""
    act = swiglu_flydsl_kernel(
        l1_out, num_tile_blocks=handle.num_tile_blocks, scale=None, activation=activation
    )
    l2 = grouped_gemm_bf16_nt(act, w2, handle.num_tokens_per_expert_prefix)
    rows, hidden = l2.shape
    num_recv = int(_dispatch_token_rows())
    out = torch.empty_like(l2)
    rstd = torch.empty(rows, dtype=torch.float32, device=l2.device)
    if rows:
        block, _, warps = _norm_launch(hidden)
        mega_post_norm_fwd_kernel[(rows,)](
            l2,
            norm_weight.contiguous(),
            dispatch_weights,
            handle.pool_row_to_recv_token,
            out,
            rstd,
            num_recv,
            hidden,
            l2.stride(0),
            out.stride(0),
            float(eps),
            BLOCK_H=block,
            num_warps=warps,
        )
    y = combine_only_gfx1250(out, handle, topk_idx)
    return y, act, l2, rstd


def stage2_post_norm_backward(
    grad_y: torch.Tensor,
    l1_out: torch.Tensor,
    act: torch.Tensor,
    l2: torch.Tensor,
    rstd: torch.Tensor,
    dispatch_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    activation=None,
    need_norm_weight_grad: bool = True,
    dw2_out: Optional[torch.Tensor] = None,
):
    """dy dispatch + post-norm backward + L2 dgrad + activation backward + dW2.

    Returns ``(grad_l1, grad_gate, dW2, d_norm_weight)``: ``grad_gate`` (f32 per pool row) is the routing-weight
    gradient stage1's backward combines into ``grad_topk_weights``. With ``dw2_out`` (bf16, w2's shape) dW2 is
    added into it and returned as None."""
    token_buffer = dispatch_only_gfx1250(grad_y.contiguous().to(torch.bfloat16), handle)
    rows, hidden = l2.shape
    num_recv = token_buffer.shape[0]
    dl2 = torch.empty_like(l2)
    grad_gate = torch.empty(rows, dtype=torch.float32, device=l2.device)
    n_prog = max(
        1, min(rows, _BWD_PROGRAMS_PER_CU * torch.cuda.get_device_properties(l2.device).multi_processor_count)
    )
    dwp = None
    if need_norm_weight_grad:
        dwp = (torch.empty if rows else torch.zeros)((n_prog, hidden), dtype=torch.float32, device=l2.device)
    if rows:
        block, block_t, warps = _norm_launch(hidden)
        mega_post_norm_bwd_kernel[(n_prog,)](
            l2,
            norm_weight.contiguous(),
            dispatch_weights,
            handle.pool_row_to_recv_token,
            token_buffer,
            rstd,
            dl2,
            grad_gate,
            dwp if need_norm_weight_grad else rstd,
            rows,
            num_recv,
            hidden,
            triton.cdiv(triton.cdiv(rows, block_t), n_prog),
            l2.stride(0),
            token_buffer.stride(0),
            dl2.stride(0),
            BLOCK_T=block_t,
            BLOCK_H=block,
            HAS_DW=need_norm_weight_grad,
            num_warps=warps,
        )
    d_norm_weight = dwp.sum(0).to(norm_weight.dtype) if need_norm_weight_grad else None
    prefix, counts = handle.num_tokens_per_expert_prefix, handle.num_tokens_per_expert
    grad_act = grouped_gemm_bf16_nn(dl2, w2, prefix)
    grad_l1 = swiglu_backward_flydsl_kernel(
        grad_act, l1_out, handle.num_tile_blocks, scale=None, activation=activation
    )
    # dW2[e] = dl2[e].T @ act[e] -> [E, hidden, inter], over expert e's routed rows only
    if dw2_out is not None and dw2_out.dtype == torch.float32:
        grouped_gemm_bf16_variable_k(dl2, act, prefix, masked_k=counts, out=dw2_out, accumulate=1)
        return grad_l1, grad_gate, None, d_norm_weight
    dW2 = _variable_k(dl2, act, prefix, masked_k=counts, out=dw2_out, accumulate=int(dw2_out is not None))
    return grad_l1, grad_gate, None if dw2_out is not None else dW2.to(w2.dtype), d_norm_weight


def stage1_backward_fp32_accum(grad_l1, saved_x, w1, handle, topk_idx, grad_gate, dw1_out):
    """:func:`fused_mega_moe_gfx1250_impl.stage1_backward` with dW1 added into an fp32 ``dw1_out``.

    The bf16 Mega variable-K accumulates into bf16 only; the generic gfx1250 variable-K accumulates into fp32 but
    reads its operands in pool order, so the redispatched x is gathered into pool order first (padding rows take
    an arbitrary row: the per-expert counts leave them out of the contraction). Returns ``(dx, grad_topk_weights)``.
    """
    x_redispatch = get_x_redispatch(get_symm_buffer_for_mega_moe())
    x_redispatch.push(saved_x.contiguous())
    dx, grad_topk_weights = gemm_combine_nt_gfx1250(
        grad_l1, w1, handle, topk_idx, grad_gate=grad_gate, b_nn=True
    )
    x_recv = x_redispatch.wait()
    rows = grad_l1.shape[0]
    idx = handle.pool_row_to_recv_token[:rows].clamp_max(x_recv.shape[0] - 1).long()
    x_pool = x_recv.index_select(0, idx)
    # dW1[e] = grad_l1[e].T @ x_pool[e] -> [E, 2 * inter, hidden]
    grouped_gemm_bf16_variable_k(
        grad_l1,
        x_pool,
        handle.num_tokens_per_expert_prefix,
        masked_k=handle.num_tokens_per_expert,
        out=dw1_out,
        accumulate=1,
    )
    return dx, grad_topk_weights


def _dispatch_token_rows() -> int:
    return get_symm_buffer_for_mega_moe().dispatch_token_buffer.shape[0]
