###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op-layer MXFP8 weight prep for the mega MoE GEMMs, and the state that keeps it fresh.

The FlyDSL layer provides the basic primitives (grouped mxfp8 quant + scale preshuffle); this module
composes them into the operands each GEMM consumes, and owns the "quantize once per optimizer step"
bookkeeping around them. Both halves live here so that the kernels stay pure computation and the
forward, the backward L1 dgrad, the stage entry points and the standalone benches all share one
notion of a prepared weight instead of reaching into whichever file happened to define it.
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
    "prepare_w2_fp8",
    "weight_generation",
]


def prepare_dispatch_weight_fp8(w: torch.Tensor):
    """Prepare a grouped weight ``[G, N, K]`` for the fp8 dispatch GEMM -> ``(wq, ws, flat, b_sp)``.

    Grouped mxfp8 quant + int8 flat + scale preshuffle (ScaleBComb, ``pack=1``): every weight
    derivative the dispatch GEMM contracts, so the kernel does no per-call weight work and needs no
    cache of its own. ``flat`` is a view of ``wq``, kept alongside it because the kernel still reads
    ``wq`` for shape and dtype. Static per weight version, so a version-keyed holder computes this
    once per ``optim.step``.
    """
    G, N, K = w.shape
    wq, ws = quantize_grouped_weight_mxfp8_flydsl(w)
    flat = wq.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    return wq, ws, flat, preshuffle_b_scale(ws, G, N, K, pack=1)


def prepare_w1_fp8(w1: torch.Tensor):
    """The L1 fc1 weight ``[G, 2I, H]`` prepped for the dispatch GEMM.

    Thin alias of :func:`prepare_dispatch_weight_fp8`; parallels :func:`prepare_w2_fp8` so both
    weights prep through one layer."""
    return prepare_dispatch_weight_fp8(w1)


def prepare_w2_fp8(l2_weights: torch.Tensor):
    """Prepare a grouped combine-GEMM weight ``[G, N, K]`` for the fp8 combine: grouped mxfp8 quant
    (FlyDSL) + scale preshuffle (ScaleBComb layout) + int8 flat -> ``(weight_flat int8 [G*N*K],
    b_sp int32)``, exactly the two operands the mxfp8 combine GEMM consumes. Static per weight
    version, so a stateful holder computes this ONCE per ``optim.step`` and passes it as ``w2_fp8``
    -- the combine then does NO per-call weight quant OR preshuffle. Used for the forward fc2 weight
    and, transposed, the backward L1 dgrad fc1^T combine weight."""
    G, N, K = l2_weights.shape
    w2q, w2s = quantize_grouped_weight_mxfp8_flydsl(l2_weights)
    b_sp = preshuffle_b_scale(w2s, G, N, K, pack=4)
    weight_flat = w2q.reshape(G * N, K).contiguous().view(torch.int8).reshape(-1)
    return weight_flat, b_sp


_WEIGHT_GENERATION = [0]
_W1_PREP_ATTR = "_mega_fp8_w1_prep"
_W2_PREP_ATTR = "_mega_fp8_w2_prep"

_PREP_BUFFERS: dict = {}
_PREP_FRESH: dict = {}
_PREP_STATE = {"warned": False}


def advance_weight_generation() -> None:
    """Invalidate every prepared fp8 weight. Call once per optimizer step.

    The cache below cannot detect a weight update on its own. ``w._version`` is the obvious signal
    and it does not work: Megatron's precision-aware optimizer never bumps it. Neither does the
    identity of the prepared tensors, which are rewritten in place, and which kept their address even
    back when they were reallocated -- the allocator hands back the block it just freed. Missing the
    update is why the fp8 experts once trained on their step-0 weights.

    Megatron already publishes the right signal: the pipeline schedule calls
    ``model.set_is_first_microbatch()`` on the first microbatch of each step, which is exactly when
    the weight has changed and no microbatch of this step has consumed it yet.
    """
    _WEIGHT_GENERATION[0] += 1


def weight_generation() -> int:
    """The current weight generation; include it in any cache key derived from a weight."""
    return _WEIGHT_GENERATION[0]


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


def _w1_fp8_cached(w1: torch.Tensor) -> tuple:
    """-> the dispatch GEMM's 4-tuple ``(w1q, w1s, flat, b_sp)``; see ``prepare_dispatch_weight_fp8``."""
    return _version_keyed_weight_prep(w1, _W1_PREP_ATTR, prepare_w1_fp8)


def _w2_fp8_cached(w2: torch.Tensor) -> tuple:
    """-> ``(weight_flat int8 [G*H*I], b_sp int32 preshuffled scale)``. Unlike w1, the L2 combine is
    pure-compute, so quant + ScaleBComb preshuffle + int8-flat are all baked in here."""
    return _version_keyed_weight_prep(w2, _W2_PREP_ATTR, prepare_w2_fp8)
