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
    _host_rendezvous,
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_fp8,
    prepare_w2_fp8,
    quantize_grouped_weight_mxfp8_cached,
    swiglu,
)

__all__ = [
    "mega_moe_forward_fp8_impl",
]

_W2_PREP_ATTR = "_mega_fp8_w2_prep"


def _w1_fp8_cached(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed fc1 weight prep -> ``(w1q [G,2I,H] fp8, w1s [G,2I,H//32] raw E8M0)``. The L1
    dispatch GEMM takes the raw quant and preshuffles the weight scale internally (cached), so this
    is just the grouped mxfp8 quant. Re-quantized only when ``w1`` changes (``optim.step`` bumps
    ``_version``); the cache is kept on the weight by ``quantize_grouped_weight_mxfp8_cached``."""
    return quantize_grouped_weight_mxfp8_cached(w1)


def _w2_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed fc2 weight prep -> ``(w2q, w2s)`` = ``prepare_w2_fp8(w2)`` = ``(weight_flat int8
    [G*H*I], b_sp int32 preshuffled scale)``. Unlike w1, the L2 combine is pure-compute (no internal
    quant/preshuffle), so quant + ScaleBComb preshuffle + int8-flat are baked here. Stashed ON the
    weight tensor: re-prepped only when ``w2`` changes (``optim.step`` bumps ``_version``)."""
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
    # w1 fp8 prep -> (w1q, w1s), version-keyed on w1._version -- symmetric with the w2 prep below.
    w1q, w1s = _w1_fp8_cached(w1)

    # ── L1: fused mxfp8 dispatch + fc1 (one self-contained unit) ──
    l1, handle, symm, dispatch_weights, pool_x_fp8 = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        x, w1q, w1s, group, topk_idx=topk_idx, topk_weights=topk_weights,
        BM=block_m, BN=block_n, save_bwd=save_bwd,
    )

    act = swiglu(l1)

    # w2 fp8 prep -> (w2q, w2s), version-keyed here at the op layer -- symmetric with w1 at L1.
    w2q, w2s = _w2_fp8_cached(w2)

    # ── L2: fp8 combine (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + bf16-out dequant reduce) ──
    _host_rendezvous(group)
    symm.sb_l2.zero_()
    symm.barrier_local.fill_(-1)
    _host_rendezvous(group)
    y = grouped_gemm_combine_fp8(
        act, (w2q, w2s), list(handle), group,
        topk_indices=topk_idx, topk_weights=topk_weights.to(torch.float32),
        BM=block_m, BN=block_n, num_combine_cu=48,
    )
    return y, l1, dispatch_weights, pool_x_fp8, handle
