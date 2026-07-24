###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 forward (FlyDSL): L1 dispatch+fc1 (NT) -> SwiGLU -> L2 fp8 combine.

A plain orchestration function (not a custom_op): the fp8 path carries state the schema can't hold
(live symm buffer reused in backward, non-tensor handles). The comm gates now self-reset via a
device epoch (no host synchronize()+barrier() rendezvous). ``w1`` / ``w2`` stay the differentiable
inputs; their mxfp8 quant is version-keyed on ``w._version``.
"""

from typing import Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega import swiglu_flydsl_kernel
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.mega_moe.weight_prep_fp8 import prepare_w1_fp8, prepare_w2_fp8

__all__ = [
    "mega_moe_forward_fp8_impl",
]

_W1_PREP_ATTR = "_mega_fp8_w1_prep"
_W2_PREP_ATTR = "_mega_fp8_w2_prep"
_H_NUM_TILE_BLOCKS = 11  # fp8 dispatch handle index of num_tile_blocks (device real-tile count)


def _version_keyed_weight_prep(w: torch.Tensor, attr: str, prep):
    """Cache ``prep(w)`` ON the weight tensor, keyed by ``w._version`` -- the single place the fp8
    weight-prep caching lives (not scattered across the quant helpers). Recomputes only when the
    weight changed in place (``optim.step()`` bumps ``_version``); otherwise returns the stash.

    Storing on the tensor (vs a global dict) makes it per-weight: auto-scales to many layers with no
    size cap / LRU thrash, freed with the weight. Correctness-safe (``_version`` guards stale
    weights) and transfer-safe (keyed off the weight's own version, never an activation id -- Rule
    11). Reuse pays off across a grad-accum window (all micro-steps share one ``_version``)."""
    v = getattr(w, "_version", 0)
    ent = getattr(w, attr, None)
    if ent is not None and ent[0] == v:
        return ent[1]
    with torch.no_grad():
        out = prep(w)
    try:
        setattr(w, attr, (v, out))
    except (AttributeError, RuntimeError):
        pass  # can't stash on this tensor (rare) -> return freshly computed, no caching
    return out


def _w1_fp8_cached(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed L1 (fc1) weight prep -> ``(w1q [G,2I,H] fp8, w1s [G,2I,H//32] raw E8M0)``. The L1
    dispatch GEMM takes the raw quant and preshuffles the weight scale internally, so this is just
    the grouped mxfp8 (E4M3) quant."""
    return _version_keyed_weight_prep(w1, _W1_PREP_ATTR, prepare_w1_fp8)


def _w2_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed L2 (fc2) weight prep -> ``(weight_flat int8 [G*H*I], b_sp int32 preshuffled scale)``.
    Unlike w1, the L2 combine is pure-compute (no internal quant/preshuffle), so quant + ScaleBComb
    preshuffle + int8-flat are baked here by ``prepare_w2_fp8``."""
    return _version_keyed_weight_prep(w2, _W2_PREP_ATTR, prepare_w2_fp8)


def mega_moe_forward_fp8_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group: ProcessGroup,
    block_m: int,
    block_n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], tuple]:
    """Fused mxfp8 MoE forward: L1 (dispatch + fc1, NT) -> SwiGLU (bf16) -> L2 fp8 combine.

    ``topk_idx`` must already be int64 (the op layer converts). Returns
    ``(y, l1, dispatch_weights, pool_x_fp8, handle)``: ``l1`` = L1 (fc1) output [P, 2I];
    ``dispatch_weights`` = per-pool-row routing weight; ``pool_x_fp8`` = L1-input pool in rowwise-fp8
    ``(pool_fp8 [P,H], pool_scale [P,H//32])`` for a LOCAL dW1 wgrad; ``handle`` = dispatch prologue
    tuple. The backward operands are LIVE symm-pool views (not cloned)."""
    # w1 fp8 prep -> (w1q, w1s), version-keyed on w1._version -- symmetric with the w2 prep below.
    w1q, w1s = _w1_fp8_cached(w1)

    # ── L1: fused mxfp8 dispatch + fc1 (one self-contained unit) ──
    # NOTE: the isolated l1 bench favoured 24/8, but that was a back-to-back-prologue artifact; the
    # per-forward op path (e2e) is insensitive to this split, so keep the 16/16 default.
    l1, handle, dispatch_weights, pool_x_fp8 = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        x, 
        w1q, w1s, 
        group, 
        topk_idx=topk_idx, 
        topk_weights=topk_weights,
        BM=block_m, BN=block_n,
    )

    act = swiglu_flydsl_kernel(l1, handle[_H_NUM_TILE_BLOCKS])

    # w2 fp8 prep -> (w2q, w2s), version-keyed here at the op layer -- symmetric with w1 at L1.
    w2q, w2s = _w2_fp8_cached(w2)

    # ── L2: fp8 combine (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + bf16-out dequant reduce) ──
    # grouped_gemm_combine_mxfp8_flydsl_kernel is self-contained: it resets the L2 scoreboard/flags cross-rank.
    y, _ = grouped_gemm_combine_mxfp8_flydsl_kernel(
        act, (w2q, w2s), list(handle), group,
        topk_indices=topk_idx, topk_weights=topk_weights.to(torch.float32),
        BM=block_m, BN=block_n, num_combine_cu=32,  # retuned 48->32 for epoch comm (T=8192, +4.6%)
    )
    return y, l1, dispatch_weights, pool_x_fp8, handle
