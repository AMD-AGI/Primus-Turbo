###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 backward: conjugate of forward via Dispatch<->Combine duality (FlyDSL).

Unlike the bf16 sibling this is a plain orchestration function (no custom_op / dispatcher): it
reuses the forward's live symmetric buffer.

Also the fp8 path's weight-prep home, for both directions. It sits here because every other fp8
module in this package already imports from this one, so one cache can serve all four prepared
weights without an import cycle -- and one cache is the point: when the forward and the transposed
dgrad weights had separate caches, only the forward's ever noticed an optimizer step.

The L1 dgrad combine and dW1 MUST stay on the default stream back-to-back; dual-stream overlap is
unsupported.
"""

from typing import Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    colwise_grouped_meta,
    colwise_requant_fp8in_and_quant_bf16_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    preshuffle_b_scale,
    quantize_grouped_weight_mxfp8_flydsl,
    swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e5m2
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)

__all__ = [
    "advance_weight_generation",
    "fused_mega_moe_backward_fp8_impl",
    "prepare_dispatch_weight_fp8",
    "prepare_dw1_pool_operand_fp8",
    "prepare_w1_fp8",
    "prepare_w1t_combine_fp8",
    "prepare_w2_fp8",
    "prepare_w2t_dgrad_fp8",
    "weight_generation",
]

# dW1/dW2 wgrad fp8 encoding. E4M3 measured a slightly higher dW SNR than E5M2 at DSv3 magnitudes.
_DW_FP8_FORMAT = torch.float8_e4m3fn

# dispatch_prologue handle layout: [9]=num_tokens_per_expert, [10]=its prefix into the padded pool.
# These MUST be the REAL unpadded lengths -- the variable-K wgrads mask each group at group_lens, so
# a padded length would fold the tail padding rows into dW.
_HANDLE_GROUP_LENS = 9
_HANDLE_GROUP_OFFS = 10

# ─────────────────────────── fp8 weight prep and its freshness ───────────────────────────
# Composition of the FlyDSL primitives (grouped mxfp8 quant + scale preshuffle) into the operands
# each GEMM contracts, plus the one cache that keeps them current. This lives with the backward
# because the backward is what every other fp8 module here already imports; the kernels themselves
# take prepared weights and hold no weight state at all.


def prepare_dispatch_weight_fp8(w: torch.Tensor) -> tuple:
    """Prepare a grouped weight ``[G, N, K]`` for the fp8 dispatch GEMM -> ``(wq, ws, flat, b_sp)``.

    Grouped mxfp8 quant + int8 flat + scale preshuffle (ScaleBComb, ``pack=1``): every weight
    derivative the dispatch GEMM contracts, so the kernel does no per-call weight work. ``flat`` is a
    view of ``wq``, kept alongside it because the kernel still reads ``wq`` for shape and dtype.
    """
    G, N, K = w.shape
    wq, ws = quantize_grouped_weight_mxfp8_flydsl(w)
    flat = wq.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    return wq, ws, flat, preshuffle_b_scale(ws, G, N, K, pack=1)


def prepare_w1_fp8(w1: torch.Tensor) -> tuple:
    """The L1 fc1 weight ``[G, 2I, H]`` prepped for the dispatch GEMM. Thin alias of
    :func:`prepare_dispatch_weight_fp8`, so both weights prep through one layer."""
    return prepare_dispatch_weight_fp8(w1)


def prepare_w2_fp8(l2_weights: torch.Tensor) -> tuple:
    """Prepare a grouped combine-GEMM weight ``[G, N, K]`` -> ``(weight_flat int8 [G*N*K], b_sp
    int32)``, exactly the two operands the mxfp8 combine GEMM consumes: grouped mxfp8 quant + scale
    preshuffle (ScaleBComb, ``pack=4``) + int8 flat, so the combine does NO per-call weight quant or
    preshuffle. Used for the forward fc2 weight and, transposed, the L1 dgrad fc1^T combine weight."""
    G, N, K = l2_weights.shape
    w2q, w2s = quantize_grouped_weight_mxfp8_flydsl(l2_weights)
    b_sp = preshuffle_b_scale(w2s, G, N, K, pack=4)
    weight_flat = w2q.reshape(G * N, K).contiguous().view(torch.int8).reshape(-1)
    return weight_flat, b_sp


def prepare_w2t_dgrad_fp8(w2: torch.Tensor) -> tuple:
    """``w2^T`` [G,I,H] prepped for the L2 dgrad's dispatch GEMM -> ``(wq, ws, flat, b_sp)``.

    The L2 dgrad runs NT via the transposed weight, so w2 must be quantized along H (its
    contraction axis). Static weight prep; the transpose never runs inside the kernel.
    """
    return prepare_dispatch_weight_fp8(w2.transpose(1, 2).contiguous())  # [G,I,H]


def prepare_w1t_combine_fp8(w1: torch.Tensor) -> tuple:
    """``w1^T`` [G, H, 2I] prepped for the L1 dgrad's combine GEMM (same format as the forward w2)."""
    return prepare_w2_fp8(w1.transpose(1, 2).contiguous())


_WEIGHT_GENERATION = [0]
_W1_PREP_ATTR = "_mega_fp8_w1_prep"
_W2_PREP_ATTR = "_mega_fp8_w2_prep"
_W2T_PREP_ATTR = "_mega_fp8_w2t_prep"
_W1T_COMBINE_PREP_ATTR = "_mega_fp8_w1t_combine_prep"

_PREP_BUFFERS: dict = {}
_PREP_FRESH: dict = {}
_PREP_STATE = {"warned": False}


def advance_weight_generation() -> None:
    """Invalidate every prepared fp8 weight. Call once per optimizer step.

    The cache below cannot detect a weight update on its own. ``w._version`` is the obvious signal
    and it does not work: an optimizer lands its update through the parameter's ``.data`` view, which
    shares the storage but not the version counter, so ``_version`` was measured at 0 on every call
    of every iteration. Neither does the identity of the prepared tensors, which are rewritten in
    place, and which kept their address even back when they were reallocated -- the allocator hands
    back the block it just freed. Missing the update is why the fp8 experts once trained on their
    step-0 weights.

    Megatron already publishes the right signal: the pipeline schedule calls
    ``model.set_is_first_microbatch()`` on the first microbatch of each step, which is exactly when
    the weight has changed and no microbatch of this step has consumed it yet.
    """
    _WEIGHT_GENERATION[0] += 1


def weight_generation() -> int:
    """The current weight generation; include it in any cache key derived from a weight."""
    return _WEIGHT_GENERATION[0]


def _version_keyed_weight_prep(w: torch.Tensor, attr: str, prep):
    """Run ``prep(w)`` once per optimizer step, into buffers that live for the whole run.

    The prepared weight has a fixed shape, so it gets one allocation per weight and is rewritten in
    place. Handing back a NEW tensor each step is what made this leak: the old one is released only if
    nothing else references it, and a live autograd graph does, so the copies piled up a step at a
    time (+41 GB by iteration 17, then HIP OOM). ``prep`` still allocates a temporary, freed as soon
    as it is copied in, so the peak is one persistent set plus one transient rather than one per step.

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


# All four prepared weights go through the one cache above. They used to be split: the two forward
# ones were generation-keyed while the two transposed dgrad ones keyed on ``_version`` alone and
# stashed their entry on the weight tensor, which never invalidated -- the backward computed dx from
# step-0 weights for the whole run.
def _w1_fp8_cached(w1: torch.Tensor) -> tuple:
    """-> the dispatch GEMM's 4-tuple ``(w1q, w1s, flat, b_sp)``; see ``prepare_dispatch_weight_fp8``."""
    return _version_keyed_weight_prep(w1, _W1_PREP_ATTR, prepare_w1_fp8)


def _w2_fp8_cached(w2: torch.Tensor) -> tuple:
    """-> ``(weight_flat int8 [G*H*I], b_sp int32)`` for the forward fc2 combine."""
    return _version_keyed_weight_prep(w2, _W2_PREP_ATTR, prepare_w2_fp8)


def _w2t_fp8_cached(w2: torch.Tensor) -> tuple:
    """-> ``w2^T`` prepped for the L2 dgrad's dispatch GEMM."""
    return _version_keyed_weight_prep(w2, _W2T_PREP_ATTR, prepare_w2t_dgrad_fp8)


def _w1t_combine_fp8_cached(w1: torch.Tensor) -> tuple:
    """-> ``w1^T`` prepped for the L1 dgrad's combine GEMM."""
    return _version_keyed_weight_prep(w1, _W1T_COMBINE_PREP_ATTR, prepare_w1t_combine_fp8)


def _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2, group, handle, *, w2t_fp8=None,
                                           num_dispatch_cu=None, num_preshuffle_cu=None):
    """Fp8 dispatch(dy) PUSH + fc2 dgrad: ``grad_swiglu = dy @ w2`` and rowwise-fp8 pool for dW2 ``a``."""
    w2t = w2t_fp8 if w2t_fp8 is not None else _w2t_fp8_cached(w2)
    # None lets the kernel look this shape's split up; the dgrad's N=I wants a different one than the
    # forward's N=2I, which is what a per-shape table gets right without pinned constants here.
    grad_swiglu, _, _, pool_fp8_handle = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        dy, w2t, group, handle=handle,
        num_dispatch_cu=num_dispatch_cu, num_preshuffle_cu=num_preshuffle_cu,
    )
    return grad_swiglu, pool_fp8_handle


def _mxfp8_variable_k_wgrad_dw2(a_fp8, b_bf16, group_lens, group_offs, meta=None):
    """dW2 = ``a^T @ b`` (variable-K over the dispatched pool tokens), MXFP8 -> ``[G, H, *]`` bf16.

    ``a`` is the L2-dgrad dispatched-dy pool, requantized colwise directly from fp8 with no bf16
    round-trip; ``b`` is ``act_weighted``. One dual-launch produces both transposed operands."""
    pool_fp8, pool_scale = a_fp8
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=pool_fp8.shape[0])
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
    w1, group, handle, *, grad_l1_rowwise_fp8, grad_gate, topk_idx, num_tokens, num_topk,
    w1t_fp8=None,
):
    """L1 dgrad (``grad_l1 @ w1^T``) + combine PUSH + reduce + grad_gate scatter
    -> ``(dx [num_tokens, H] bf16, grad_topk_weights [num_tokens, num_topk] f32)``."""
    w1tf = w1t_fp8 if w1t_fp8 is not None else _w1t_combine_fp8_cached(w1)
    dx, d_topk_w_flat = grouped_gemm_combine_mxfp8_flydsl_kernel(
        w1tf, list(handle), group,
        topk_indices=topk_idx.contiguous().view(-1), grad_gate=grad_gate,
        x_fp8_rowwise=grad_l1_rowwise_fp8,
        num_combine_cu=28,  # unified w/ fwd L2 (task-based push; T=8192)
    )
    grad_topk_weights = d_topk_w_flat[: num_tokens * num_topk].view(num_tokens, num_topk)
    return dx, grad_topk_weights


def prepare_dw1_pool_operand_fp8(
    pool_x_fp8: Tuple[torch.Tensor, torch.Tensor], handle: tuple,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], dict]:
    """Turn the forward's fc1-input pool into dW1's ``b`` operand -> ``(pool_colwise, meta)``.

    Called from the forward while ``pool_x_fp8`` is still a live view of the symm pool: requantizing
    it there consumes the view instead of the clone the backward would need, and keeps the requant
    off the backward critical path. ``meta`` is reused by the dual-quant, dW1 and dW2."""
    meta = colwise_grouped_meta(
        handle[_HANDLE_GROUP_LENS], handle[_HANDLE_GROUP_OFFS], pool_rows=pool_x_fp8[0].shape[0]
    )
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
    """dW1 = ``a^T @ b`` (variable-K over the pool tokens), MXFP8 -> ``[G, 2I, H]`` bf16.

    ``a`` = ``grad_l1``, pre-quantized colwise by the fused dual-quant; ``b`` = the fc1-input pool,
    already colwise-fp8 from the forward, or requantized here when ``pool_x_is_colwise=False``
    (isolated benches that time the requant as part of dW1).

    This GEMM is LOCAL: it contracts over pool tokens already gathered on this rank, so unlike the
    bf16 path -- which re-dispatches ``saved_x`` cross-rank -- it needs no transfer at all."""
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


def fused_mega_moe_backward_fp8_impl(
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
    colwise_meta: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mxfp8 MoE backward (conjugate of forward via the Dispatch<->Combine duality).

    ``pool_x_colwise_fp8`` / ``colwise_meta`` come straight from the forward (see
    :func:`prepare_dw1_pool_operand_fp8`). Returns ``(dx, grad_topk_weights, dW1, dW2)`` with
    dW1/dW2 cast back to the weight dtypes.
    """
    group_lens = handle[_HANDLE_GROUP_LENS]
    group_offs = handle[_HANDLE_GROUP_OFFS]
    dy = grad_y.contiguous().to(torch.bfloat16)

    # L2 dgrad: dispatch(dy) + fc2 -> grad_swiglu + the dispatched-dy pool (the dW2 `a` operand)
    grad_swiglu, dispatch_l2_grad_fp8 = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2, group, handle)

    # SwiGLU^T (re-inject routing weight) + gate grad, fused with grad_l1's rowwise (L1 dgrad) and
    # colwise (dW1) quant -- grad_l1 has no other consumer, so it never reaches HBM
    (
        gl1_q_row, gl1_a_sp, gl1_q_col, gl1_s_col, grad_gate, act_weighted,
    ) = swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl(
        grad_swiglu, l1, dispatch_weights, _DW_FP8_FORMAT, meta=colwise_meta,
    )

    # dW2 = dispatch_l2_grad^T @ act_weighted -- must run before anything overwrites symm.pool_fp8
    dW2 = _mxfp8_variable_k_wgrad_dw2(
        dispatch_l2_grad_fp8, act_weighted, group_lens, group_offs, meta=colwise_meta,
    )

    # L1 dgrad + combine, then dW1 -- serial on the default stream (dual-stream overlap forbidden)
    dx, grad_topk_weights = _l1_dgrad_combine_mxfp8_flydsl_kernel(
        w1, group, handle,
        grad_l1_rowwise_fp8=(gl1_q_row, gl1_a_sp),
        grad_gate=grad_gate, topk_idx=topk_idx, num_tokens=num_tokens, num_topk=num_topk,
    )
    dW1 = _mxfp8_variable_k_wgrad_dw1((gl1_q_col, gl1_s_col), pool_x_colwise_fp8, colwise_meta)

    return dx, grad_topk_weights, dW1.to(w1.dtype), dW2.to(w2.dtype)
