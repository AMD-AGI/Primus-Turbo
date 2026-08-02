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
  * STEP3: fc1 dgrad + combine + reduce (fused) -> dx; then dW1 variable-K wgrad (serial).

STEP3 and dW1 MUST stay on the default stream back-to-back (no dual-stream overlap).
``PT_MEGA_BWD_DUAL_STREAM`` and similar hooks are intentionally unsupported.
"""

from typing import Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    colwise_grouped_meta,
    colwise_requant_fp8in_and_quant_bf16_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    quantize_grouped_weight_mxfp8_flydsl,
    swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl,
)
from primus_turbo.pytorch.kernels.mega_moe.weight_prep_fp8 import prepare_w2_fp8
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e5m2
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)

__all__ = [
    "mega_moe_backward_fp8_impl",
    "prepare_dw1_pool_operand_fp8",
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
# [9]=num_tokens_per_expert (REAL unpadded group_lens), [10]=its prefix into the block_m-padded
# pool (group_offs). The variable-K wgrads (dW1/dW2) take group_lens/offs and mask each group at
# group_lens, so a padded length would fold the tail padding rows into dW; the combine takes
# tile_to_expert.
_HANDLE_GROUP_LENS = 9
_HANDLE_GROUP_OFFS = 10

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
    return quantize_grouped_weight_mxfp8_flydsl(w2.transpose(1, 2).contiguous())  # [G,I,H]


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
    """Fp8 dispatch(dy) PUSH + fc2 dgrad: ``grad_swiglu = dy @ w2`` and rowwise-fp8 pool for dW2 ``a``."""
    w2tq, w2ts = w2t_fp8 if w2t_fp8 is not None else _w2t_fp8_cached(w2)
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
    (bf16, from ``swiglu_backward``) -- colwise-quantized. Both are produced by
    ``colwise_requant_fp8in_and_quant_bf16_grouped_flydsl`` (dual-launch: independent requant +
    quant grids, shared ``meta``, one stream). Both emit only the transposed operand
    the wgrad needs + raw E8M0; the grouped meta (one D2H of the padded per-group offsets) is shared.
    Then the FlyDSL mxfp8 variable-K grouped GEMM. dW2 is the large backward GEMM (H*I over the pool)."""
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs)
    pool_fp8, pool_scale = a_fp8
    a_t, a_ts, b_t, b_ts, lens_pc, offs_pc = colwise_requant_fp8in_and_quant_bf16_grouped_flydsl(
        pool_fp8, pool_scale, b_bf16, _DW_FP8_FORMAT, meta=meta,
    )
    return grouped_gemm_fp8_variable_k_impl(
        a_t, b_t,
        a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64), offs_pc.to(torch.int64),
        trans_a=False, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None, default_backend=BackendType.FLYDSL.value,
    )


def _l1_dgrad_combine_mxfp8_flydsl_kernel(
    w1, group, handle, block_m, block_n, *, grad_l1_rowwise_fp8, grad_gate, topk_idx, num_tokens, num_topk,
    w1t_fp8=None,
):
    """Backward L1 dgrad: fp8 fc1-dgrad (``grad_l1 @ w1^T``) + combine PUSH + unweighted reduce +
    grad_gate scatter -> ``(dx [num_tokens, H] bf16, grad_topk_weights [num_tokens, num_topk] f32)``.

    ``grad_l1_rowwise_fp8`` = ``(q_row e4m3, a_sp preshuffled)`` from
    ``swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl``."""
    w1tf = w1t_fp8 if w1t_fp8 is not None else _w1t_combine_fp8_cached(w1)
    dx, d_topk_w_flat = grouped_gemm_combine_mxfp8_flydsl_kernel(
        None, w1tf, list(handle), group,
        topk_indices=topk_idx.contiguous().view(-1), grad_gate=grad_gate,
        x_fp8_rowwise=grad_l1_rowwise_fp8,
        BM=block_m, BN=block_n, num_combine_cu=28,  # unified w/ fwd L2 (task-based push; T=8192)
    )
    grad_topk_weights = d_topk_w_flat[: num_tokens * num_topk].view(num_tokens, num_topk)
    return dx, grad_topk_weights


def prepare_dw1_pool_operand_fp8(
    pool_x_fp8: Tuple[torch.Tensor, torch.Tensor], handle: tuple,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], dict]:
    """Turn the forward's fc1-input pool into dW1's ``b`` operand -> ``(pool_colwise, meta)``.

    Called from ``mega_moe_forward_fp8_impl`` while ``pool_x_fp8`` is still a live rowwise-fp8 view
    of the symm pool. Requantizing it colwise there consumes the view in place of the clone the
    backward would otherwise need, and takes the requant off the backward critical path. ``meta``
    is returned alongside so backward reuses it for the dual-quant / dW1 / dW2."""
    meta = colwise_grouped_meta(handle[_HANDLE_GROUP_LENS], handle[_HANDLE_GROUP_OFFS])
    pool_colwise = colwise_requant_mxfp8_grouped_fp8in_flydsl(
        pool_x_fp8[0], pool_x_fp8[1], _DW_FP8_FORMAT, meta=meta,
    )[:2]
    return pool_colwise, meta


def _mxfp8_variable_k_wgrad_dw1(
    a_colwise_fp8,
    pool_x_operand: Tuple[torch.Tensor, torch.Tensor],
    meta,
    *,
    pool_x_is_colwise: bool = True,
):
    """dW1 = ``a^T @ b`` (variable-K over the pool tokens) in MXFP8, LOCAL -> ``[G, 2I, H]`` bf16.

    L1 (fc1) weight grad: ``a`` = ``grad_l1`` [P, 2I] colwise-quantized -- passed in PRE-QUANTIZED as
    ``a_colwise_fp8 = (q_col [2I,Mpad] fp8, s_col E8M0)`` from the fused dual-quant (one grad_l1 read
    shared with STEP3's rowwise operand); ``b`` = ``pool_x`` = the FORWARD-dispatched fc1-input pool,
    colwise-fp8 ``(q [H,Mpad], s [H,n_pblk])`` already requantized in the forward (production path),
    or native rowwise-fp8 ``(pool_fp8 [P,H], pool_scale [P,H//32])`` requantized here when
    ``pool_x_is_colwise=False`` (isolated benches that time the requant as part of dW1).

    LOCAL: dW1 contracts over pool tokens already gathered on THIS rank, so (unlike the bf16 fused
    path that re-dispatches ``saved_x`` cross-rank) it needs NO cross-rank transfer -- that is where
    dW1's fp8 win comes from. ``meta`` (grouped padded offsets) is shared with the fused quant +
    dW2 + the pool requant."""
    a_t, a_ts = a_colwise_fp8
    lens_pc, offs_pc = meta["lens_pc"], meta["offs_pc"]
    if pool_x_is_colwise:
        b_t, b_ts = pool_x_operand
    else:
        pool_fp8, pool_scale = pool_x_operand
        b_t, b_ts, _, _ = colwise_requant_mxfp8_grouped_fp8in_flydsl(
            pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta,
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
    pool_x_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    handle: tuple,
    group,
    num_tokens: int,
    num_topk: int,
    block_m: int,
    block_n: int,
    colwise_meta: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mxfp8 MoE backward (conjugate of forward via the Dispatch<->Combine duality).

    L2 dgrad: dispatch(dy)+fc2 (fp8 fork) -> SwiGLU^T (re-inject routing weight + gate grad)
    -> dW2 variable-K wgrad -> STEP3 (fused fc1 dgrad+combine -> dx, then serial dW1 wgrad).
    The version-keyed w1^T / w2^T dgrad quant is maintained inside the L2/L1 dgrad helpers.

    ``pool_x_colwise_fp8`` / ``colwise_meta`` come straight from ``mega_moe_forward_fp8_impl``,
    which requants the fc1-input pool while it is still live in the symm buffer instead of cloning
    it for a backward requant (see ``prepare_dw1_pool_operand_fp8``).

    Returns ``(dx, grad_topk_weights, dW1, dW2)`` with dW1/dW2 cast back to the weight dtypes.
    """
    group_lens = handle[_HANDLE_GROUP_LENS]
    group_offs = handle[_HANDLE_GROUP_OFFS]
    dy = grad_y.contiguous().to(torch.bfloat16)

    # L2 dgrad (fp8 fork): dispatch(dy) + fc2 -> grad_swiglu + the dispatched-dy pool in native
    # rowwise-fp8 (the dW2 `a` operand `dispatch_l2_grad`, requant colwise directly from fp8).
    grad_swiglu, dispatch_l2_grad_fp8 = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2, group, handle, block_m, block_n)

    # SwiGLU^T with the routing weight re-injected, fused with the rowwise (STEP3) and colwise
    # (dW1) mxfp8 quant of grad_l1 -- which has no other consumer, so it never reaches HBM.
    # Also emits grad_gate [P] and act_weighted [P,I] (the dW2 `b` operand, folding
    # host saved_act*weight). pack=4 a_sp fused in launch().
    (
        gl1_q_row, gl1_a_sp, gl1_q_col, gl1_s_col, grad_gate, act_weighted,
    ) = swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl(
        grad_swiglu, l1, dispatch_weights, _DW_FP8_FORMAT, meta=colwise_meta,
    )

    # dW2 (MXFP8 variable-K): dispatch_l2_grad^T @ act_weighted; `a` requant-fused directly from
    # the L2-dgrad rowwise-fp8 pool. Run before anything else overwrites symm.pool_fp8.
    dW2 = _mxfp8_variable_k_wgrad(
        dispatch_l2_grad_fp8, act_weighted, group_lens, group_offs, meta=colwise_meta,
    )

    # STEP3 then dW1 — serial on the default stream (dual-stream overlap forbidden).
    dx, grad_topk_weights = _l1_dgrad_combine_mxfp8_flydsl_kernel(
        w1, group, handle, block_m, block_n,
        grad_l1_rowwise_fp8=(gl1_q_row, gl1_a_sp),
        grad_gate=grad_gate, topk_idx=topk_idx, num_tokens=num_tokens, num_topk=num_topk,
    )
    dW1 = _mxfp8_variable_k_wgrad_dw1((gl1_q_col, gl1_s_col), pool_x_colwise_fp8, colwise_meta)

    return dx, grad_topk_weights, dW1.to(w1.dtype), dW2.to(w2.dtype)
