###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

import torch

from primus_turbo.flydsl.mega.fp8 import (
    preshuffle_b_scale,
    quantize_grouped_weight_mxfp8_flydsl,
)

__all__ = [
    "advance_weight_generation",
    "prepare_dispatch_weight_fp8",
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
_PREP_STATE = {"stale_serves": 0}


def advance_weight_generation() -> None:
    """Invalidate every prepared fp8 weight. The caller owns the step boundary: call this once after
    the weights have been updated, and the next forward re-preps them.

    The cache below cannot detect a weight update on its own. ``w._version`` is the obvious signal
    and it does not work: an optimizer lands its update through the parameter's ``.data`` view, which
    shares the storage but not the version counter, so ``_version`` was measured at 0 on every call
    of every iteration. Neither does the identity of the prepared tensors, which are rewritten in
    place, and which kept their address even back when they were reallocated -- the allocator hands
    back the block it just freed. Missing the update is why the fp8 experts once trained on their
    step-0 weights.

    Counting forwards in the expert module is not a substitute: activation recompute, pipeline warm-up
    and eval all forward without stepping, so any count drifts off the optimizer. Under Megatron the
    signal comes from wrapping ``train_step`` -- see Primus's ``patch_mega_moe_weight_generation``,
    which also records why ``set_is_first_microbatch()`` cannot carry it: that hook returns early
    unless Megatron's own ``config.fp8``/``fp4``/``kitchen`` recipe is on, and MegaMoE selects mxfp8
    through ``turbo_mega_moe_precision`` while Megatron's recipe stays off.
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

    Rewriting in place is safe because the generation advances after a step completes while the
    refresh happens on the next forward, and nothing reads these buffers in between: step N's
    microbatch backwards have all finished, so no saved tensor from a live graph still points at
    these bytes. ``_version`` stays in the key so an in-place write that does bump it still
    invalidates."""
    key = (attr, w.data_ptr(), tuple(w.shape))
    gen = (weight_generation(), getattr(w, "_version", 0))
    buf = _PREP_BUFFERS.get(key)
    if buf is not None and _PREP_FRESH.get(key) == gen:
        return buf
    # Reuse is only safe while something advances the generation. A whole backward has run (grads
    # exist) and the generation never moved, so every serve from here on hands back step-0 weights --
    # invisible in the loss at first, then a model that stops learning. Megatron's distributed
    # optimizer accumulates into ``main_grad`` and may leave ``.grad`` as None, so check both or this
    # never fires where it matters. Repeated on every serve, not once: the run is broken and the count
    # is the evidence of how long it has been.
    if (
        buf is not None
        and weight_generation() == 0
        and (w.grad is not None or getattr(w, "main_grad", None) is not None)
    ):
        _PREP_STATE["stale_serves"] += 1
        print(
            f"[mega fp8] WARNING (#{_PREP_STATE['stale_serves']}): the fp8 weight caches were never "
            "invalidated, so the experts are training on their step-0 weights. Whoever owns the "
            "expert module must call advance_weight_generation() once per optimizer step.",
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
