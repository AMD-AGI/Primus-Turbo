###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 forward (FlyDSL): L1 dispatch+fc1 (NT) -> SwiGLU+mxfp8 quant -> L2 fp8 combine.

A plain orchestration function, not a custom_op: the fp8 path carries state no schema can hold --
a live symm buffer the backward reuses, plus non-tensor handles.
"""

from typing import Tuple

import functools

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import (
    prepare_dw1_pool_operand_fp8,
)
from primus_turbo.flydsl.mega.fp8 import weight_generation
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_weight_prep_fp8 import (
    prepare_w1_fp8,
    prepare_w2_fp8,
)

__all__ = [
    "fused_mega_moe_forward_fp8_impl",
]

_W1_PREP_ATTR = "_mega_fp8_w1_prep"
_W2_PREP_ATTR = "_mega_fp8_w2_prep"
_H_NUM_TILE_BLOCKS = 11  # fp8 dispatch handle index of num_tile_blocks (device real-tile count)

# Retuned CU splits for EP8 T=8192 DSv3 (see bench_mega_moe_fp8 sweeps). The isolated L1 bench
# favours 24/8, but that is a back-to-back-prologue artifact and e2e is insensitive, so keep 16/16.
# L2 combine 32 beats 48 by ~5%. Pinned so prod skips the autotune cold-start sweep.
_L1_NUM_DISPATCH_CU = 16
_L1_NUM_PRESHUFFLE_CU = 16
_L2_NUM_COMBINE_CU = 32


_PREP_BUFFERS: dict = {}
_PREP_FRESH: dict = {}
_PREP_STATE = {"warned": False}


def _version_keyed_weight_prep(w: torch.Tensor, attr: str, prep):
    """Quantize ``w`` once per optimizer step, into buffers that live for the whole run.

    The quantized weight has a fixed shape, so it gets one allocation per weight and is rewritten in
    place -- the same footprint the original never-refreshing cache had. Handing back a NEW tensor
    each step is what made this leak: the old one is released only if nothing else references it, and
    a live autograd graph does, so the copies piled up a step at a time (+41 GB by iteration 17, then
    HIP OOM). ``prep`` still allocates a temporary, freed as soon as it is copied in, so the peak is
    one persistent set plus one transient rather than one set per step.

    Rewriting in place is safe only because every microbatch backward of step N finishes before the
    first forward of step N+1, which is when the refresh happens, so no saved tensor from a live
    graph can still point at these bytes. ``_version`` stays in the key so an in-place write that
    does bump it still invalidates."""
    key = (attr, w.data_ptr(), tuple(w.shape))
    gen = (weight_generation(), getattr(w, "_version", 0))
    buf = _PREP_BUFFERS.get(key)
    if buf is not None and _PREP_FRESH.get(key) == gen:
        return buf
    if buf is not None and weight_generation() == 0 and w.grad is not None and not _PREP_STATE["warned"]:
        # Reuse is only safe while something advances the generation. A whole backward has run and
        # the generation never moved, so this is about to serve step-0 weights for the rest of the
        # run -- invisible in the loss at first, then a model that stops learning.
        _PREP_STATE["warned"] = True
        print(
            "[mega fp8] WARNING: the fp8 weight caches were never invalidated, so the experts are "
            "about to keep training on their step-0 weights. Whoever owns the expert module must "
            "call advance_weight_generation() once per optimizer step.",
            flush=True,
        )
    with torch.no_grad():
        out = prep(w)
    if buf is None:
        _PREP_BUFFERS[key] = buf = out
    else:
        for dst, src in zip(buf, out):
            dst.copy_(src)
        del out  # release the temporary before returning, so steady-state stays one set
    _PREP_FRESH[key] = gen
    return buf


def _w1_fp8_cached(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> ``(w1q [G,2I,H] fp8, w1s [G,2I,H//32] raw E8M0)``. The scale stays raw because the L1
    dispatch GEMM preshuffles it internally."""
    return _version_keyed_weight_prep(w1, _W1_PREP_ATTR, prepare_w1_fp8)


def _w2_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> ``(weight_flat int8 [G*H*I], b_sp int32 preshuffled scale)``. Unlike w1, the L2 combine is
    pure-compute, so quant + ScaleBComb preshuffle + int8-flat are all baked in here."""
    return _version_keyed_weight_prep(w2, _W2_PREP_ATTR, prepare_w2_fp8)


def fused_mega_moe_forward_fp8_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group: ProcessGroup,
    block_m: int,
    block_n: int,
    save_bwd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict, tuple]:
    """Fused mxfp8 MoE forward: L1 (dispatch + fc1, NT) -> SwiGLU+mxfp8 quant -> L2 fp8 combine.

    Returns ``(y, l1, dispatch_weights, pool_x_colwise, colwise_meta, handle)``, where ``l1`` is the
    fc1 output [P, 2I] and ``handle`` the dispatch prologue tuple. ``topk_idx`` must already be
    int64 (the op layer converts). ``dispatch_weights`` is a LIVE symm-pool view -- clone it before
    a later stage overwrites it.

    ``pool_x_colwise`` / ``colwise_meta`` are dW1's ``b`` operand (None unless ``save_bwd``), built
    here because the fc1-input pool is still live in the symm buffer: requantizing it colwise now
    consumes that view instead of the clone the backward would otherwise need."""
    w1q, w1s = _w1_fp8_cached(w1)

    # ── L1: fused mxfp8 dispatch + fc1 ──
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

    act_fp8, act_a_sp = swiglu_mxfp8_flydsl_kernel(l1, handle[_H_NUM_TILE_BLOCKS])

    w2q, w2s = _w2_fp8_cached(w2)

    # ── L2: fp8 combine (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + bf16-out dequant reduce) ──
    y, _ = grouped_gemm_combine_mxfp8_flydsl_kernel(
        None, (w2q, w2s), list(handle), group,
        topk_indices=topk_idx,
        topk_weights=topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32),
        x_fp8=(act_fp8, act_a_sp),
        BM=block_m, BN=block_n,
        num_combine_cu=_L2_NUM_COMBINE_CU,
    )

    pool_x_colwise, colwise_meta = (
        prepare_dw1_pool_operand_fp8(pool_x_fp8, handle) if save_bwd else (None, None)
    )
    return y, l1, dispatch_weights, pool_x_colwise, colwise_meta, handle
