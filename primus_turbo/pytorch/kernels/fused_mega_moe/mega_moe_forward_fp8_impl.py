###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 forward orchestration (FlyDSL): L1 fused dispatch+fc1 (NT) + SwiGLU + L2 fp8 combine.

fp8 sibling of the bf16 ``mega_moe_forward_impl`` (``mega_moe_forward_impl.py``). Unlike bf16 this is
a PLAIN orchestration function, NOT a ``torch.library.custom_op`` / ``AutoKernelDispatcher`` backend:
the fp8 path carries state the custom_op schema can't hold -- reuse of the forward's live symmetric
buffer in backward, host ``synchronize()`` + ``group.barrier()`` rendezvous, and derived non-tensor
handles. The bf16 ``w1`` / ``w2`` stay the differentiable inputs; their mxfp8 quant is maintained
here by version-keyed caches keyed on ``w._version`` (no caller-supplied prequant tuples).
"""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    grouped_gemm_combine_fp8,
    prepare_w2_fp8,
    quantize_grouped_weight_mxfp8_cached,
    swiglu,
)

__all__ = [
    "mega_moe_forward_fp8_impl",
]

_W2_PREP_ATTR = "_mega_fp8_w2_prep"


def _host_rendezvous(group) -> None:
    """Cross-rank publish barrier: drain this rank's GPU work, then all-rank barrier, so a
    scoreboard/flag reset is visible on every peer before any rank signals it. (Full mode;
    the source op gates these behind PT_MEGA_BARRIER_MODE -- kept always-on here for safety.)"""
    torch.cuda.synchronize()
    group.barrier()


def _w2_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed cache of :func:`prepare_w2_fp8` (fc2 weight quant + scale preshuffle + int8
    flat) stashed ON the weight tensor: re-prep only when ``w2`` changes (``optim.step`` bumps
    ``_version``), reused across a grad-accum window. The op-layer analog of the w1 cache
    (``quantize_grouped_weight_mxfp8_cached``); the combine kernel stays pure compute."""
    v = getattr(w2, "_version", 0)
    ent = getattr(w2, _W2_PREP_ATTR, None)
    if ent is None or ent[0] != v:
        with torch.no_grad():
            out = prepare_w2_fp8(w2)
        ent = (v, out)
        setattr(w2, _W2_PREP_ATTR, ent)
    return ent[1]


def mega_moe_forward_fp8_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group: ProcessGroup,
    block_m: int,
    block_n: int,
    *,
    save_bwd: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], tuple]:
    """Fused mxfp8 MoE forward: L1 (dispatch + fc1, NT) -> SwiGLU (bf16) -> L2 fp8 combine.

    ``topk_idx`` must already be int64 (the op layer converts). Returns
    ``(y, l1, dispatch_weights, pool_x_fp8, handle)`` where -- when ``save_bwd`` -- ``dispatch_weights``
    and ``pool_x_fp8`` are the CLONED backward inputs (see below), else ``None``:
      * ``l1``               : fc1 output ``[P, 2I]`` (the SwiGLU / swiglu_backward operand).
      * ``dispatch_weights`` : per-pool-row routing weight (prologue-scattered) -- swiglu_backward
        re-injects it as the SwiGLU^T scale + gate grad.
      * ``pool_x_fp8``       : THIS dispatched fc1-input pool in native rowwise-fp8 -- lets dW1 be a
        LOCAL variable-K wgrad (grad_l1^T @ pool_x) with NO cross-rank re-dispatch (mirrors dW2's
        reuse of the STEP1 pool). ``(pool_fp8 [P,H] fp8, pool_scale [P,H//32] E8M0)``.
      * ``handle``           : the dispatch prologue handle tuple (backward reuses it).
    """
    G, world = w1.shape[0], group.size()
    T, H = x.shape
    I = w1.shape[1] // 2
    K = topk_idx.shape[-1]

    # ── L1: fused mxfp8 dispatch + fc1 (token quant folded inside via the bf16-x path) ──
    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=G * world, num_max_tokens_per_rank=T, num_topk=K,
        hidden=H, intermediate_hidden=I, block_m=block_m, block_n=block_n, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    handle = tuple(
        dispatch_prologue(
            topk_idx, topk_weights, sym_layout=sym_layout, num_tokens=T, num_topk=K,
            num_experts=G * world, world_size=world, rank=symm.rank, experts_per_rank=G,
            block_m=block_m, num_max_pool_tokens=symm.num_max_pool_tokens,
        )
    )
    w1q, w1s = quantize_grouped_weight_mxfp8_cached(w1)  # version-keyed cache on w1._version
    # publish scoreboard=0 cross-rank before the L1 comm signals (per-pool-block sentinel handoff)
    _host_rendezvous(group)
    symm.scoreboard.zero_()
    _host_rendezvous(group)
    l1 = dispatch_grouped_gemm_mxfp8(
        x, None, w1q, w1s, handle, sym_layout, symm, BM=block_m, BN=block_n
    )
    # backward saves (clone BEFORE backward STEP1's dispatch(dy) overwrites the symm pool)
    dispatch_weights = symm.weight_recv_buf.clone() if save_bwd else None
    pool_x_fp8 = None
    if save_bwd:
        _Px, _Hx = symm.pool_fp8.shape
        pool_x_fp8 = (symm.pool_fp8.clone(), symm.pool_scale.reshape(_Px, _Hx // 32).clone())

    act = swiglu(l1)

    # w2 fp8 prep (quant + scale preshuffle), version-keyed here at the op layer -- the combine
    # is a pure-compute kernel and takes the prepared weight in (symmetric with w1 at L1).
    w2_fp8 = _w2_fp8_cached(w2)

    # ── L2: fp8 combine (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + bf16-out dequant reduce) ──
    _host_rendezvous(group)
    symm.sb_l2.zero_()
    symm.barrier_local.fill_(-1)
    _host_rendezvous(group)
    y = grouped_gemm_combine_fp8(
        act, w2_fp8, list(handle), group,
        topk_indices=topk_idx, topk_weights=topk_weights.to(torch.float32),
        BM=block_m, BN=block_n, num_combine_cu=48,
    )
    return y, l1, dispatch_weights, pool_x_fp8, handle
