###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Forward / backward launch for flex attention.

Tensors arrive in the kernels' layouts: ``bshd`` views for dense (strided views are
fine, the kernels take strides) and packed ``thd`` for varlen. Outputs are allocated by
the caller so the op layer can hand the kernels a view of ``[B, H, S, D]`` storage.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.kernels.flex_attention.flex_attention_block_mask_utils import (
    BlockPlan,
    backward_block_plans,
)
from primus_turbo.triton.flex_attention.flex_attention_bwd_kernel import (
    attention_backward_triton_impl,
)
from primus_turbo.triton.flex_attention.flex_attention_combine_kernel import (
    combine_splits,
)
from primus_turbo.triton.flex_attention.flex_attention_fwd_kernel import (
    attention_forward_prefill_triton_impl,
)

__all__ = ["flex_attention_backward_impl", "flex_attention_forward_impl"]


def _layout(cu_seqlens_q) -> str:
    return "thd" if cu_seqlens_q is not None else "bshd"


def _batch_heads(q: torch.Tensor, cu_seqlens_q) -> Tuple[int, int]:
    if cu_seqlens_q is not None:
        return cu_seqlens_q.numel() - 1, q.shape[1]
    return q.shape[0], q.shape[2]


def flex_attention_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window: Tuple[int, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    score_mod=None,
    mask_mod=None,
    block_plan: Optional[BlockPlan] = None,
    num_splits: int = 1,
    aux_tensors=None,
    mask_aux_tensors=None,
) -> None:
    """Fill ``out`` and the natural-log ``lse`` in place.

    ``lse`` is ``[B, H, Sq]`` for dense and ``[H, total_q]`` for varlen. ``num_splits > 1``
    slices the KV loop across programs (dense only) and reduces the partials.
    ``block_plan`` may have size-1 batch/head dims; they are broadcast here.
    """
    if block_plan is not None:
        block_plan = block_plan.expand(*_batch_heads(q, cu_seqlens_q))
    out_partial = lse_partial = None
    kernel_out = out
    if num_splits > 1:
        batch, seqlen_q, heads, head_dim_v = out.shape
        out_partial = torch.empty(
            num_splits, batch, seqlen_q, heads, head_dim_v, dtype=torch.float32, device=q.device
        )
        lse_partial = torch.empty(num_splits, batch, heads, seqlen_q, dtype=torch.float32, device=q.device)
        # The kernel writes each split with the output's strides, so they must match a
        # contiguous partial; combine_splits then writes ``out`` honouring its strides.
        if not out.is_contiguous():
            kernel_out = torch.empty(out.shape, dtype=out.dtype, device=out.device)
    attention_forward_prefill_triton_impl(
        q,
        k,
        v,
        kernel_out,
        lse,
        scale,
        causal,
        window[0],
        window[1],
        _layout(cu_seqlens_q),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        True,  # use_exp2
        score_mod=score_mod,
        mask_mod=mask_mod,
        block_sparse=block_plan,
        num_splits=num_splits,
        out_partial=out_partial,
        lse_partial=lse_partial,
        aux_tensors=aux_tensors,
        mask_aux_tensors=mask_aux_tensors,
    )
    if num_splits > 1:
        combine_splits(out_partial, lse_partial, out, lse)


def flex_attention_backward_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window: Tuple[int, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    score_mod=None,
    mask_mod=None,
    score_mod_bwd=None,
    block_plan: Optional[BlockPlan] = None,
    aux_tensors=None,
    mask_aux_tensors=None,
    score_grad_hook=None,
    score_grad: Optional[torch.Tensor] = None,
    dlse: Optional[torch.Tensor] = None,
) -> None:
    """Fill ``dq``/``dk``/``dv`` (and ``score_grad`` through the hook) in place.

    Deterministic: every output tile is written by a single program, with no atomics.
    """
    bs_dq = bs_dkdv = None
    if block_plan is not None:
        # Derived from the caller's (unexpanded, reused) plan so the memoization hits.
        bs_dq, bs_dkdv = backward_block_plans(block_plan)
        batch, heads = _batch_heads(q, cu_seqlens_q)
        bs_dq, bs_dkdv = bs_dq.expand(batch, heads), bs_dkdv.expand(batch, heads)
    delta = torch.zeros_like(lse)
    attention_backward_triton_impl(
        do=do,
        q=q,
        k=k,
        v=v,
        o=out,
        softmax_lse=lse,
        dq=dq,
        dk=dk,
        dv=dv,
        delta=delta,
        sm_scale=scale,
        causal=causal,
        layout=_layout(cu_seqlens_q),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q if cu_seqlens_q is not None else q.shape[1],
        max_seqlen_k=max_seqlen_k if cu_seqlens_q is not None else k.shape[1],
        use_exp2=True,
        window_size_left=window[0],
        window_size_right=window[1],
        score_mod=score_mod,
        mask_mod=mask_mod,
        score_mod_bwd=score_mod_bwd,
        block_sparse_dkdv=bs_dkdv,
        block_sparse_dq=bs_dq,
        aux_tensors=aux_tensors,
        mask_aux_tensors=mask_aux_tensors,
        score_grad_hook=score_grad_hook,
        score_grad=score_grad,
        dlse=dlse,
    )
