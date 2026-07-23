###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 backward orchestration (FlyDSL): conjugate of forward via Dispatch<->Combine.

fp8 sibling of the bf16 ``mega_moe_backward_impl`` (``mega_moe_backward_impl.py``). Like the fp8
forward this is a PLAIN orchestration function (no custom_op / dispatcher): it reuses the forward's
live symmetric buffer, does host ``synchronize()`` + ``group.barrier()`` rendezvous, and maintains
the version-keyed w1^T / w2^T dgrad quant inside the STEP1 / STEP3 helpers (no ctx-passed prequant).

Backward (conjugate via the Dispatch<->Combine duality):
  * STEP1 dispatch(dy)+fc2 dgrad (NN) MXFP8 -> grad_swiglu + rowwise-fp8 dispatched-dy pool.
  * STEP2 SwiGLU^T (bf16), re-inject routing weight, gate grad, act_weighted (dW2 b-operand).
  * dW2   variable-K wgrad (MXFP8), a-operand requant-fused from the STEP1 fp8 pool.
  * STEP3 fc1 dgrad (fp8 GEMM) + combine/reduce (fp8-PUSH). KNOWN: the fp8-PUSH combine has an
    intermittent cross-rank reduce-flag liveness stall at large T (stable at T<=2048; robust fix
    pending -- candidate: split into fused GEMM+push then a barrier'd standalone reduce).
  * dW1   variable-K wgrad (MXFP8), LOCAL -- reuses the FORWARD-dispatched fc1-input pool.
"""

from typing import Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    _host_rendezvous,
    colwise_grouped_meta,
    colwise_quant_mxfp8_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    dispatch_grouped_gemm_mxfp8,
    get_symm_buffer_for_mega_moe,
    grouped_gemm_combine_mxfp8_flydsl_kernel_bwd,
    quantize_grouped_weight_mxfp8,
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

# STEP1 dgrad reuses the FORWARD dispatch+GEMM kernel with this CU split (the dgrad shape N=I
# prefers more comm CUs / fewer preshuffle CUs than the forward's N=2I default of 16/16).
_STEP1_NUM_DISPATCH_CU = 24
_STEP1_NUM_PRESHUFFLE_CU = 8

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
    """Grouped mxfp8 quant of ``w2^T`` (``[G,I,H]``) for the backward STEP1 fc2-dgrad NT-reuse GEMM.

    Backward fc2 dgrad ``grad_act = dispatched_dy @ w2`` is done NT via the transposed weight, so w2
    must be quantized in the ``[G,I,H]`` layout (along H = the dgrad contraction axis). Returns
    ``(w2tq [G,I,H] fp8, w2ts [G,I,H//32] raw E8M0)``. STATIC weight prep (maintained version-keyed
    at the op layer); the transpose+quant never runs inside the kernel.
    """
    return quantize_grouped_weight_mxfp8(w2.transpose(1, 2).contiguous())  # [G,I,H]


def prepare_w1t_dgrad_fp8(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 quant of ``w1^T`` (``[G,H,2I]``) for the backward STEP3 fc1-dgrad NT-reuse.

    Returns ``(w1tq [G,H,2I] fp8, w1ts [G,H,2I//32] raw E8M0)``. Mirrors
    :func:`prepare_w2t_dgrad_fp8`; owned version-keyed by ``MegaMoEFP8`` and passed via ``w1t_fp8``.
    """
    raise NotImplementedError(_BWD_NOT_PORTED)


def _w2t_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed cache of :func:`prepare_w2t_dgrad_fp8` (w2^T grouped mxfp8, backward STEP1 fc2
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
    (quant + scale preshuffle + int8 flat, same format as the forward w2) for the backward STEP3
    fp8 combine. Cached ON the weight tensor, re-prepped only when ``w1`` changes."""
    v = getattr(w1, "_version", 0)
    ent = getattr(w1, _W1T_COMBINE_PREP_ATTR, None)
    if ent is None or ent[0] != v:
        with torch.no_grad():
            out = prepare_w2_fp8(w1.transpose(1, 2).contiguous())  # [G, H, 2I]
        ent = (v, out)
        setattr(w1, _W1T_COMBINE_PREP_ATTR, ent)
    return ent[1]


def _mxfp8_step1_dispatch_dgrad(dy, w2, group, handle, block_m, block_n, *, w2t_fp8=None):
    """Backward STEP1 (the winnable fp8 fork): fused fp8 dispatch(dy) PUSH + fc2 dgrad (NT-reuse).

    Quantizes ``dy`` rowwise mxfp8, PUSHes it cross-rank into the pool (fp8, byte-halved), and
    computes ``grad_swiglu = dispatched_dy @ w2`` as a grouped mxfp8 NT GEMM against ``w2^T``
    (``w2t_fp8``; version-keyed here). Returns ``(grad_swiglu [P, I] bf16, pool_fp8_handle)`` where
    ``pool_fp8_handle = (pool_fp8 [P,H] fp8, pool_scale [P,H//32] E8M0)`` is the dispatched-dy pool
    in native rowwise-fp8 -- the dW2 ``a`` operand for the (later) variable-K wgrad.

    Needs the live mxfp8 symm buffer (the forward's global buffer) and the same cross-rank
    scoreboard gate the forward L1 uses; the caller need not have kept anything but the handle."""
    symm = get_symm_buffer_for_mega_moe()  # live buffer from the forward
    sym_layout = symm.make_sym_layout()
    w2tq, w2ts = w2t_fp8 if w2t_fp8 is not None else _w2t_fp8_cached(w2)
    # publish scoreboard=0 cross-rank before the fp8 comm signals (mirror the forward L1 gate)
    _host_rendezvous(group)
    symm.scoreboard.zero_()
    _host_rendezvous(group)
    # STEP1 dgrad IS the forward L1 op (generic fp8 dispatch PUSH + grouped mxfp8 NT GEMM): here
    # A = dy (bf16 -> quantized + PUSHed inside), weight = w2^T [G,I,H] -> grad_swiglu [P, I]. Reuses
    # the exact forward kernel (no separate bwd fork); only the CU split differs (STEP1 tuning). The
    # dispatched-dy pool (symm.pool_fp8/pool_scale) is left populated for the dW2 wgrad.
    grad_swiglu = dispatch_grouped_gemm_mxfp8(
        dy, None, w2tq, w2ts, handle, sym_layout, symm,
        num_dispatch_cu=_STEP1_NUM_DISPATCH_CU, num_preshuffle_cu=_STEP1_NUM_PRESHUFFLE_CU,
        BM=block_m, BN=block_n,
    )
    P, H = symm.pool_fp8.shape
    pool_fp8_handle = (symm.pool_fp8, symm.pool_scale.reshape(P, H // 32))
    return grad_swiglu, pool_fp8_handle


def _mxfp8_variable_k_wgrad(a_fp8, b_bf16, group_lens, group_offs):
    """dW2 = ``a^T @ b`` (variable-K over the dispatched pool tokens) in MXFP8 -> ``[G, H, b.dim1]`` bf16.

    fc2 weight grad: ``a`` = STEP1's dispatched-dy pool ``(pool_fp8 [P,H], pool_scale [P,H//32])`` (the
    ``dispatch_l2_grad`` operand, native rowwise-fp8) -- requant COLWISE DIRECTLY from fp8 (fused
    dequant->requant, no bf16 round-trip, LDS-transposed coalesced write); ``b`` = ``act_weighted``
    (bf16, from STEP2 ``swiglu_backward``) -- colwise-quantized. Both emit only the transposed operand
    the wgrad needs + raw E8M0; the grouped meta (one D2H of the padded per-group offsets) is shared.
    Then the FlyDSL mxfp8 variable-K grouped GEMM. dW2 is the large backward GEMM (H*I over the pool)."""
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


def _mxfp8_step3_fc1_dgrad_combine(
    grad_l1, w1, group, handle, block_m, block_n, *, grad_gate, topk_idx, num_tokens, num_topk,
    w1t_fp8=None,
):
    """Backward STEP3: fp8 fc1-dgrad (``grad_l1 @ w1^T``) + combine PUSH + unweighted reduce +
    grad_gate scatter -> ``(dx [num_tokens, H] bf16, grad_topk_weights [num_tokens, num_topk] f32)``.

    Backward mirror of the forward L2 (compute -> comm, combine-bound). fc1^T is prepped
    version-keyed at the op layer (``_w1t_combine_fp8_cached``); the combine kernel is pure compute.
    Resets the L2 scoreboard / reduce flags / combine_gate cross-rank (barrier-bracketed) first."""
    symm = get_symm_buffer_for_mega_moe()
    w1tf = w1t_fp8 if w1t_fp8 is not None else _w1t_combine_fp8_cached(w1)
    _host_rendezvous(group)
    symm.sb_l2.zero_()
    symm.barrier_local.fill_(-1)
    symm.combine_gate.zero_()
    _host_rendezvous(group)
    dx, d_topk_w_flat = grouped_gemm_combine_mxfp8_flydsl_kernel_bwd(
        grad_l1, w1tf, list(handle), group,
        topk_indices=topk_idx.contiguous().view(-1), grad_gate=grad_gate,
        BM=block_m, BN=block_n, num_combine_cu=48,
    )
    grad_topk_weights = d_topk_w_flat[: num_tokens * num_topk].view(num_tokens, num_topk)
    return dx, grad_topk_weights


def _mxfp8_variable_k_wgrad_dw1(a_bf16, b_fp8, group_lens, group_offs):
    """dW1 = ``a^T @ b`` (variable-K over the pool tokens) in MXFP8, LOCAL -> ``[G, 2I, H]`` bf16.

    fc1 weight grad: ``a`` = ``grad_l1`` [P, 2I] (bf16, from STEP2) -- colwise-quantized; ``b`` =
    ``pool_x`` [P, H] = the FORWARD-dispatched fc1-input pool kept in native rowwise-fp8 (cloned in
    the forward before STEP1 overwrites the symm pool) -- requant COLWISE directly from fp8. LOCAL:
    dW1 contracts over pool tokens already gathered on THIS rank, so (unlike the bf16 fused path that
    re-dispatches ``saved_x`` cross-rank) it needs NO cross-rank transfer -- that is where dW1's fp8
    win comes from. Mirrors :func:`_mxfp8_variable_k_wgrad` (dW2) with the fp8 operand on ``b``."""
    meta = colwise_grouped_meta(group_lens, group_offs)
    a_t, a_ts, lens_pc, offs_pc = colwise_quant_mxfp8_grouped_flydsl(a_bf16, _DW_FP8_FORMAT, meta=meta)
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

    STEP1 dispatch(dy)+fc2 dgrad (fp8 fork) -> STEP2 SwiGLU^T (re-inject routing weight + gate grad)
    -> dW2 variable-K wgrad -> STEP3 fc1 dgrad + combine (fp8-PUSH) -> dW1 variable-K wgrad (LOCAL).
    The version-keyed w1^T / w2^T dgrad quant is maintained inside the STEP1 / STEP3 helpers.

    Returns ``(dx, grad_topk_weights, dW1, dW2)`` with dW1/dW2 cast back to the weight dtypes.
    """
    group_lens = handle[_HANDLE_GROUP_LENS]
    group_offs = handle[_HANDLE_GROUP_OFFS]
    dy = grad_y.contiguous().to(torch.bfloat16)

    # STEP1 (fp8 fork): dispatch(dy) + fc2 dgrad -> grad_swiglu + the dispatched-dy pool in native
    # rowwise-fp8 (the dW2 `a` operand `dispatch_l2_grad`, requant colwise directly from fp8).
    grad_swiglu, dispatch_l2_grad_fp8 = _mxfp8_step1_dispatch_dgrad(dy, w2, group, handle, block_m, block_n)

    # STEP2 (bf16): SwiGLU^T -> grad_l1 [P,2I]; re-inject routing weight; grad_gate [P]; and
    # act_weighted [P,I] = fwd-act * weight (the dW2 `b` operand, folding host saved_act*weight).
    grad_l1, grad_gate, act_weighted = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, handle[_HANDLE_NUM_TILE_BLOCKS],
        scale=dispatch_weights, return_gate=True, return_act_w=True,
    )

    # dW2 (MXFP8 variable-K): dispatch_l2_grad^T @ act_weighted; `a` requant-fused directly from
    # the STEP1 rowwise-fp8 pool. Run before anything else overwrites symm.pool_fp8.
    dW2 = _mxfp8_variable_k_wgrad(dispatch_l2_grad_fp8, act_weighted, group_lens, group_offs)

    # STEP3 (fp8-PUSH): fc1 dgrad + combine + unweighted reduce + grad_gate scatter -> dx +
    # grad_topk_weights. (fp8-PUSH combine has a known intermittent cross-rank reduce-flag stall
    # at large T -- robust fix pending; see the STEP3 combine notes.)
    dx, grad_topk_weights = _mxfp8_step3_fc1_dgrad_combine(
        grad_l1, w1, group, handle, block_m, block_n,
        grad_gate=grad_gate, topk_idx=topk_idx, num_tokens=num_tokens, num_topk=num_topk,
    )

    # dW1 (MXFP8 variable-K, LOCAL): grad_l1^T @ pool_x -- reuses the FORWARD-dispatched
    # fc1-input pool (rowwise-fp8, cloned on ctx), so NO cross-rank re-dispatch.
    dW1 = _mxfp8_variable_k_wgrad_dw1(grad_l1, pool_x_fp8, group_lens, group_offs)

    return dx, grad_topk_weights, dW1.to(w1.dtype), dW2.to(w2.dtype)
