###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 backward orchestration (FlyDSL): conjugate of forward via Dispatch<->Combine.

fp8 sibling of the bf16 ``mega_moe_backward_impl`` (``mega_moe_backward_impl.py``). Like the fp8
forward this is a PLAIN orchestration function (no custom_op / dispatcher): it reuses the forward's
live symmetric buffer and maintains the version-keyed w1^T / w2^T dgrad quant inside the L2/L1 dgrad
helpers (no ctx-passed prequant). The STEP1 dispatch(dy) and STEP3 combine gates now self-reset via
a device epoch (no host synchronize()+barrier() rendezvous).

Backward (conjugate via the Dispatch<->Combine duality; L2 = fc2, L1 = fc1, as in bf16):
  * L2 dgrad: dispatch(dy) + fc2 GEMM (NT via w2^T) MXFP8 -> grad_swiglu + rowwise-fp8 dispatched-dy pool.
  * SwiGLU^T (bf16), re-inject routing weight, gate grad, act_weighted (dW2 b-operand).
  * dW2   variable-K wgrad (MXFP8), a-operand requant-fused from the L2-dgrad fp8 pool.
  * L1 dgrad: fc1 GEMM (fp8) + combine/reduce (fp8-PUSH). The epoch self-reset flags removed the
    host reset-race that used to stall this combine at large T (now stable through T=8192).
  * dW1   variable-K wgrad (MXFP8), LOCAL -- reuses the FORWARD-dispatched fc1-input pool.
"""

from typing import Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    colwise_grouped_meta,
    colwise_quant_mxfp8_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    quantize_grouped_weight_mxfp8,
    rowcol_dual_quant_mxfp8_grouped_flydsl,
)
from primus_turbo.flydsl.mega import swiglu_backward_flydsl_kernel
from primus_turbo.pytorch.kernels.mega_moe.weight_prep_fp8 import prepare_w2_fp8
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e5m2
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)

__all__ = [
    "mega_moe_backward_fp8_impl",
    "prepare_w1t_dgrad_fp8",
    "prepare_w2t_dgrad_fp8",
]

# dW2/dW1 wgrad fp8 encoding: E5M2 (gradient dynamic range). E4M3 measured slightly higher SNR at
# DSv3 magnitudes -- flip to float8_e4m3 to compare (gate by dW SNR + few-step training loss).
_DW_FP8_FORMAT = float8_e5m2

# The L2 dgrad reuses the FORWARD dispatch+GEMM kernel with this CU split (the dgrad shape N=I
# prefers more comm CUs / fewer preshuffle CUs than the forward's N=2I default of 16/16).
_L2_DGRAD_NUM_DISPATCH_CU = 24
_L2_DGRAD_NUM_PRESHUFFLE_CU = 8

# dispatch_prologue handle layout (see dispatch_prologue return): [7]=tile_to_expert,
# [9]=num_tokens_per_expert (block_m-padded group_lens), [10]=its prefix (group_offs). The
# variable-K wgrads (dW1/dW2) take group_lens/offs; the combine takes tile_to_expert.
_HANDLE_GROUP_LENS = 9
_HANDLE_GROUP_OFFS = 10
_HANDLE_NUM_TILE_BLOCKS = 11  # device real-tile count (SwiGLU epilogue row bound)

_W2T_PREP_ATTR = "_mega_fp8_w2t_prep"
_W1T_COMBINE_PREP_ATTR = "_mega_fp8_w1t_combine_prep"

_BWD_NOT_PORTED = (
    "mega MoE fp8 backward not ported yet (forward works); next port step. Reference: "
    "/perf_apps/xiaoming/Primus-Turbo/primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py"
)


def prepare_w2t_dgrad_fp8(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 quant of ``w2^T`` (``[G,I,H]``) for the backward L2 dgrad NT-reuse GEMM.

    Backward L2 dgrad ``grad_act = dispatched_dy @ w2`` is done NT via the transposed weight, so w2
    must be quantized in the ``[G,I,H]`` layout (along H = the dgrad contraction axis). Returns
    ``(w2tq [G,I,H] fp8, w2ts [G,I,H//32] raw E8M0)``. STATIC weight prep (maintained version-keyed
    at the op layer); the transpose+quant never runs inside the kernel.
    """
    return quantize_grouped_weight_mxfp8(w2.transpose(1, 2).contiguous())  # [G,I,H]


def prepare_w1t_dgrad_fp8(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 quant of ``w1^T`` (``[G,H,2I]``) for the backward L1 dgrad NT-reuse.

    Returns ``(w1tq [G,H,2I] fp8, w1ts [G,H,2I//32] raw E8M0)``. Mirrors
    :func:`prepare_w2t_dgrad_fp8`; owned version-keyed by ``MegaMoEFP8`` and passed via ``w1t_fp8``.
    """
    raise NotImplementedError(_BWD_NOT_PORTED)


def _w2t_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed cache of :func:`prepare_w2t_dgrad_fp8` (w2^T grouped mxfp8, backward fc2
    dgrad) stashed ON the weight tensor -- same discipline as the forward ``_w2_fp8_cached``."""
    v = getattr(w2, "_version", 0)
    ent = getattr(w2, _W2T_PREP_ATTR, None)
    if ent is None or ent[0] != v:
        with torch.no_grad():
            out = prepare_w2t_dgrad_fp8(w2)
        ent = (v, out)
        setattr(w2, _W2T_PREP_ATTR, ent)
    return ent[1]


def _w1t_combine_fp8_cached(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed cache of ``prepare_w2_fp8(w1^T [G,H,2I])`` -- the fc1^T COMBINE weight prep
    (quant + scale preshuffle + int8 flat, same format as the forward w2) for the backward L1 dgrad
    fp8 combine. Cached ON the weight tensor, re-prepped only when ``w1`` changes."""
    v = getattr(w1, "_version", 0)
    ent = getattr(w1, _W1T_COMBINE_PREP_ATTR, None)
    if ent is None or ent[0] != v:
        with torch.no_grad():
            out = prepare_w2_fp8(w1.transpose(1, 2).contiguous())  # [G, H, 2I]
        ent = (v, out)
        setattr(w1, _W1T_COMBINE_PREP_ATTR, ent)
    return ent[1]


def _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2, group, handle, block_m, block_n, *, w2t_fp8=None,
                                           num_dispatch_cu=None, num_preshuffle_cu=None):
    """Fused fp8 dispatch(dy) PUSH + L2 (fc2) dgrad (the winnable fp8 backward fork).

    Quantizes ``dy`` rowwise mxfp8, PUSHes it cross-rank into the pool (fp8, byte-halved), and
    computes ``grad_swiglu = dispatched_dy @ w2`` as a grouped mxfp8 NT GEMM against ``w2^T``
    (``w2t_fp8``; version-keyed here). Returns ``(grad_swiglu [P, I] bf16, pool_fp8_handle)`` where
    ``pool_fp8_handle = (pool_fp8 [P,H] fp8, pool_scale [P,H//32] E8M0)`` is the dispatched-dy pool
    in native rowwise-fp8 -- the dW2 ``a`` operand for the (later) variable-K wgrad.

    Needs the live mxfp8 symm buffer (the forward's global buffer) and the same cross-rank
    scoreboard gate the forward L1 uses; the caller need not have kept anything but the handle."""
    w2tq, w2ts = w2t_fp8 if w2t_fp8 is not None else _w2t_fp8_cached(w2)
    # The L2 dgrad IS the forward L1 op: same NT dispatch-PUSH + grouped mxfp8 GEMM wrapper (handle-
    # reuse path -> reuses the live symm buffer + does the scoreboard rendezvous internally). Here
    # A = dy (bf16 -> quantized + PUSHed inside), weight = w2^T [G,I,H] -> grad_swiglu [P, I]; only
    # the CU split differs. pool_x_fp8 (the left-populated dispatched-dy pool) is the dW2 `a` operand
    # for the (later) variable-K wgrad; dispatch_weights is unused here.
    grad_swiglu, _, _, pool_fp8_handle = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        dy, w2tq, w2ts, group, handle=handle,
        num_dispatch_cu=_L2_DGRAD_NUM_DISPATCH_CU if num_dispatch_cu is None else num_dispatch_cu,
        num_preshuffle_cu=_L2_DGRAD_NUM_PRESHUFFLE_CU if num_preshuffle_cu is None else num_preshuffle_cu,
        BM=block_m, BN=block_n,
    )
    return grad_swiglu, pool_fp8_handle


def _mxfp8_variable_k_wgrad(a_fp8, b_bf16, group_lens, group_offs, meta=None):
    """dW2 = ``a^T @ b`` (variable-K over the dispatched pool tokens) in MXFP8 -> ``[G, H, b.dim1]`` bf16.

    L2 (fc2) weight grad: ``a`` = the L2-dgrad dispatched-dy pool ``(pool_fp8 [P,H], pool_scale [P,H//32])`` (the
    ``dispatch_l2_grad`` operand, native rowwise-fp8) -- requant COLWISE DIRECTLY from fp8 (fused
    dequant->requant, no bf16 round-trip, LDS-transposed coalesced write); ``b`` = ``act_weighted``
    (bf16, from ``swiglu_backward``) -- colwise-quantized. Both emit only the transposed operand
    the wgrad needs + raw E8M0; the grouped meta (one D2H of the padded per-group offsets) is shared.
    Then the FlyDSL mxfp8 variable-K grouped GEMM. dW2 is the large backward GEMM (H*I over the pool)."""
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs)
    pool_fp8, pool_scale = a_fp8
    a_t, a_ts, lens_pc, offs_pc = colwise_requant_mxfp8_grouped_fp8in_flydsl(
        pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta
    )
    b_t, b_ts, _, _ = colwise_quant_mxfp8_grouped_flydsl(b_bf16, _DW_FP8_FORMAT, meta=meta)
    return grouped_gemm_fp8_variable_k_impl(
        a_t, b_t,
        a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64), offs_pc.to(torch.int64),
        trans_a=False, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None, default_backend=BackendType.FLYDSL.value,
    )


def _l1_dgrad_combine_mxfp8_flydsl_kernel(
    grad_l1, w1, group, handle, block_m, block_n, *, grad_gate, topk_idx, num_tokens, num_topk,
    w1t_fp8=None, grad_l1_rowwise_fp8=None,
):
    """Backward L1 dgrad: fp8 fc1-dgrad (``grad_l1 @ w1^T``) + combine PUSH + unweighted reduce +
    grad_gate scatter -> ``(dx [num_tokens, H] bf16, grad_topk_weights [num_tokens, num_topk] f32)``.

    Backward mirror of the forward L2 (compute -> comm, combine-bound). fc1^T is prepped
    version-keyed at the op layer (``_w1t_combine_fp8_cached``); the combine kernel is pure compute
    and self-resets its epoch flags on device (no host scoreboard/flag reset rendezvous).

    ``grad_l1_rowwise_fp8`` = precomputed ``(aq e4m3, a_sp preshuffled)`` for ``grad_l1`` (from the
    fused dual-quant); when given, the combine skips its internal rowwise quant (one grad_l1 read)."""
    w1tf = w1t_fp8 if w1t_fp8 is not None else _w1t_combine_fp8_cached(w1)
    dx, d_topk_w_flat = grouped_gemm_combine_mxfp8_flydsl_kernel(
        grad_l1, w1tf, list(handle), group,
        topk_indices=topk_idx.contiguous().view(-1), grad_gate=grad_gate,
        x_fp8_rowwise=grad_l1_rowwise_fp8,
        BM=block_m, BN=block_n, num_combine_cu=24,  # retuned 16->24 for epoch comm (T=8192, +5.2%)
    )
    grad_topk_weights = d_topk_w_flat[: num_tokens * num_topk].view(num_tokens, num_topk)
    return dx, grad_topk_weights


def _mxfp8_variable_k_wgrad_dw1(a_colwise_fp8, b_fp8, meta):
    """dW1 = ``a^T @ b`` (variable-K over the pool tokens) in MXFP8, LOCAL -> ``[G, 2I, H]`` bf16.

    L1 (fc1) weight grad: ``a`` = ``grad_l1`` [P, 2I] colwise-quantized -- passed in PRE-QUANTIZED as
    ``a_colwise_fp8 = (q_col [2I,Mpad] fp8, s_col E8M0)`` from the fused dual-quant (one grad_l1 read
    shared with STEP3's rowwise operand); ``b`` = ``pool_x`` [P, H] = the FORWARD-dispatched fc1-input
    pool kept in native rowwise-fp8 (cloned in the forward before the L2 dgrad overwrites the symm
    pool) -- requant COLWISE directly from fp8. LOCAL: dW1 contracts over pool tokens already gathered
    on THIS rank, so (unlike the bf16 fused path that re-dispatches ``saved_x`` cross-rank) it needs NO
    cross-rank transfer -- that is where dW1's fp8 win comes from. ``meta`` (grouped padded offsets) is
    shared with the fused quant + dW2 + the pool requant."""
    a_t, a_ts = a_colwise_fp8
    lens_pc, offs_pc = meta["lens_pc"], meta["offs_pc"]
    pool_fp8, pool_scale = b_fp8
    b_t, b_ts, _, _ = colwise_requant_mxfp8_grouped_fp8in_flydsl(
        pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta
    )
    return grouped_gemm_fp8_variable_k_impl(
        a_t, b_t,
        a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64), offs_pc.to(torch.int64),
        trans_a=False, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None, default_backend=BackendType.FLYDSL.value,
    )


def mega_moe_backward_fp8_impl(
    grad_y: torch.Tensor,
    l1: torch.Tensor,
    dispatch_weights: torch.Tensor,
    pool_x_fp8: Tuple[torch.Tensor, torch.Tensor],
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: tuple,
    group,
    num_tokens: int,
    num_topk: int,
    block_m: int,
    block_n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mxfp8 MoE backward (conjugate of forward via the Dispatch<->Combine duality).

    L2 dgrad: dispatch(dy)+fc2 (fp8 fork) -> SwiGLU^T (re-inject routing weight + gate grad)
    -> dW2 variable-K wgrad -> L1 dgrad: fc1 + combine (fp8-PUSH) -> dW1 variable-K wgrad (LOCAL).
    The version-keyed w1^T / w2^T dgrad quant is maintained inside the L2/L1 dgrad helpers.

    Returns ``(dx, grad_topk_weights, dW1, dW2)`` with dW1/dW2 cast back to the weight dtypes.
    """
    group_lens = handle[_HANDLE_GROUP_LENS]
    group_offs = handle[_HANDLE_GROUP_OFFS]
    dy = grad_y.contiguous().to(torch.bfloat16)
    # grouped padded-offset meta shared across the fused grad_l1 dual-quant, dW1, dW2, pool requant.
    meta = colwise_grouped_meta(group_lens, group_offs)

    # L2 dgrad (fp8 fork): dispatch(dy) + fc2 -> grad_swiglu + the dispatched-dy pool in native
    # rowwise-fp8 (the dW2 `a` operand `dispatch_l2_grad`, requant colwise directly from fp8).
    grad_swiglu, dispatch_l2_grad_fp8 = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2, group, handle, block_m, block_n)

    # SwiGLU^T (bf16) -> grad_l1 [P,2I]; re-inject routing weight; grad_gate [P]; and
    # act_weighted [P,I] = fwd-act * weight (the dW2 `b` operand, folding host saved_act*weight).
    grad_l1, grad_gate, act_weighted = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, handle[_HANDLE_NUM_TILE_BLOCKS],
        scale=dispatch_weights, return_gate=True, return_act_w=True,
    )

    # grad_l1 is quantized TWICE downstream: rowwise-preshuffled E4M3 (STEP3 fc1-dgrad A operand) +
    # colwise-grouped (dW1 `a` operand). Fuse both into ONE read of grad_l1 (byte-exact to the two
    # shipped kernels): q_row/a_sp feed the combine; q_col/s_col feed dW1. Saves ~0.09ms/backward.
    gl1_q_row, gl1_a_sp, gl1_q_col, gl1_s_col = rowcol_dual_quant_mxfp8_grouped_flydsl(
        grad_l1, _DW_FP8_FORMAT, meta=meta,
    )

    # dW2 (MXFP8 variable-K): dispatch_l2_grad^T @ act_weighted; `a` requant-fused directly from
    # the L2-dgrad rowwise-fp8 pool. Run before anything else overwrites symm.pool_fp8.
    dW2 = _mxfp8_variable_k_wgrad(dispatch_l2_grad_fp8, act_weighted, group_lens, group_offs, meta=meta)

    # L1 dgrad (fp8-PUSH): fc1 dgrad + combine + unweighted reduce + grad_gate scatter -> dx +
    # grad_topk_weights. Uses the pre-quantized rowwise grad_l1 (fused above; combine skips its
    # internal quant). (epoch self-reset flags -> no host reset-race stall; stable through T=8192.)
    dx, grad_topk_weights = _l1_dgrad_combine_mxfp8_flydsl_kernel(
        grad_l1, w1, group, handle, block_m, block_n,
        grad_gate=grad_gate, topk_idx=topk_idx, num_tokens=num_tokens, num_topk=num_topk,
        grad_l1_rowwise_fp8=(gl1_q_row, gl1_a_sp),
    )

    # dW1 (MXFP8 variable-K, LOCAL): grad_l1^T @ pool_x -- `a` = pre-quantized colwise grad_l1 (fused
    # above); `b` = the FORWARD-dispatched fc1-input pool (rowwise-fp8, cloned on ctx), NO re-dispatch.
    dW1 = _mxfp8_variable_k_wgrad_dw1((gl1_q_col, gl1_s_col), pool_x_fp8, meta)

    return dx, grad_topk_weights, dW1.to(w1.dtype), dW2.to(w2.dtype)
