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
    "fp8_padded_dims",
    "pad_w1_fp8_dims",
    "pad_w2_fp8_dims",
    "prepare_dispatch_weight_fp8",
    "prepare_w1_fp8",
    "prepare_w1t_combine_fp8",
    "prepare_w2_fp8",
    "prepare_w2t_dgrad_fp8",
    "unpad_w1_grad",
    "unpad_w2_grad",
    "weight_generation",
]

# dW1/dW2 wgrad fp8 encoding. E4M3 measured a slightly higher dW SNR than E5M2 at DSv3 magnitudes.
_DW_FP8_FORMAT = torch.float8_e4m3fn

# ─────────────────────────── padding to the fp8 kernels' alignment ───────────────────────────
# The fp8 token push and reduce move whole 1 KiB rows (hidden % 1024), and the fp8 GEMM tiles take
# neither a K nor an N tail: the intermediate is a K of the fc2 GEMM and an N of the L2 dgrad
# (% 256). An expert shape off that grid (GPT-OSS: hidden = intermediate = 2880) runs zero-padded to
# it. That is exact: a padded gate column is 0 before and after the activation, every padded weight
# row/column is 0, and the padded outputs and gradients are sliced off.
_FP8_HIDDEN_ALIGN = 1024
_FP8_INTER_ALIGN = 256


def fp8_padded_dims(hidden: int, inter: int) -> tuple:
    """``(hidden, intermediate)`` rounded up to what the fp8 kernels run; unchanged when aligned."""
    return (
        -(-hidden // _FP8_HIDDEN_ALIGN) * _FP8_HIDDEN_ALIGN,
        -(-inter // _FP8_INTER_ALIGN) * _FP8_INTER_ALIGN,
    )


def pad_w1_fp8_dims(w1: torch.Tensor) -> torch.Tensor:
    """fc1 ``[G, 2I, H]`` -> ``[G, 2Ip, Hp]``; the gate and up halves are padded separately."""
    G, two_i, hidden = w1.shape
    inter = two_i // 2
    hp, ip = fp8_padded_dims(hidden, inter)
    if (hp, ip) == (hidden, inter):
        return w1
    out = w1.new_zeros((G, 2 * ip, hp))
    out[:, :inter, :hidden] = w1[:, :inter]
    out[:, ip : ip + inter, :hidden] = w1[:, inter:]
    return out


def pad_w2_fp8_dims(w2: torch.Tensor) -> torch.Tensor:
    """fc2 ``[G, H, I]`` -> ``[G, Hp, Ip]``."""
    G, hidden, inter = w2.shape
    hp, ip = fp8_padded_dims(hidden, inter)
    if (hp, ip) == (hidden, inter):
        return w2
    out = w2.new_zeros((G, hp, ip))
    out[:, :hidden, :inter] = w2
    return out


def unpad_w1_grad(dw1: torch.Tensor, hidden: int, inter: int) -> torch.Tensor:
    """The padded fc1 gradient ``[G, 2Ip, Hp]`` back to ``[G, 2I, H]``."""
    ip = dw1.shape[1] // 2
    if (dw1.shape[2], ip) == (hidden, inter):
        return dw1
    return torch.cat((dw1[:, :inter, :hidden], dw1[:, ip : ip + inter, :hidden]), dim=1)


def pad_b1_fp8_dims(b1: torch.Tensor, padded_inter: int) -> torch.Tensor:
    """fc1 bias ``[G, 2I]`` -> ``[G, 2Ip]``, gate and up halves padded separately like the weight."""
    G, two_i = b1.shape
    inter = two_i // 2
    if inter == padded_inter:
        return b1
    out = b1.new_zeros((G, 2 * padded_inter))
    out[:, :inter] = b1[:, :inter]
    out[:, padded_inter : padded_inter + inter] = b1[:, inter:]
    return out


def unpad_b1_grad(db1: torch.Tensor, inter: int) -> torch.Tensor:
    """The padded fc1 bias gradient ``[G, 2Ip]`` back to ``[G, 2I]``."""
    ip = db1.shape[1] // 2
    if ip == inter:
        return db1
    return torch.cat((db1[:, :inter], db1[:, ip : ip + inter]), dim=1)


def unpad_w2_grad(dw2: torch.Tensor, hidden: int, inter: int) -> torch.Tensor:
    """The padded fc2 gradient ``[G, Hp, Ip]`` back to ``[G, H, I]``."""
    if tuple(dw2.shape[1:]) == (hidden, inter):
        return dw2
    return dw2[:, :hidden, :inter].contiguous()


# dispatch_prologue handle layout: [9]=num_tokens_per_expert, [10]=its prefix into the padded pool.
# These MUST be the REAL unpadded lengths -- the variable-K wgrads mask each group at group_lens, so
# a padded length would fold the tail padding rows into dW.

# ─────────────────────────── fp8 weight prep and its freshness ───────────────────────────
# Composition of the FlyDSL primitives (grouped mxfp8 quant + scale preshuffle) into the operands
# each GEMM contracts, plus the one cache that keeps them current. This lives with the backward
# because the backward is what every other fp8 module here already imports; the kernels themselves
# take prepared weights and hold no weight state at all.


def prepare_dispatch_weight_fp8(w: torch.Tensor) -> tuple:
    """Prepare a grouped weight ``[G, N, K]`` for the fp8 dispatch GEMM -> ``(wq, flat, b_sp)``.

    Grouped mxfp8 quant + int8 flat + scale preshuffle (ScaleBComb, ``pack=1``): every weight
    derivative the dispatch GEMM contracts, so the kernel does no per-call weight work. ``flat`` is a
    view of ``wq``, kept alongside it because the kernel still reads ``wq`` for shape and dtype. The
    raw E8M0 scales are dropped once preshuffled: the GEMM reads ``b_sp`` only.
    """
    G, N, K = w.shape
    wq, ws = quantize_grouped_weight_mxfp8_flydsl(w)
    flat = wq.contiguous().reshape(G * N, K).view(torch.int8).reshape(-1)
    return wq, flat, preshuffle_b_scale(ws, G, N, K, pack=1)


def prepare_w1_fp8(w1: torch.Tensor) -> tuple:
    """The L1 fc1 weight ``[G, 2I, H]`` prepped (padded to :func:`fp8_padded_dims`) for the dispatch
    GEMM, through :func:`prepare_dispatch_weight_fp8` so both weights prep through one layer."""
    return prepare_dispatch_weight_fp8(pad_w1_fp8_dims(w1))


def prepare_w2_fp8(l2_weights: torch.Tensor) -> tuple:
    """Prepare a grouped combine-GEMM weight ``[G, N, K]`` -> ``(weight_flat int8 [G*N*K], b_sp
    int32)``, exactly the two operands the mxfp8 combine GEMM consumes: grouped mxfp8 quant + scale
    preshuffle (ScaleBComb, ``pack=4``) + int8 flat, so the combine does NO per-call weight quant or
    preshuffle. Used for the forward fc2 weight and, transposed, the L1 dgrad fc1^T combine weight.
    Takes the weight as the GEMM runs it; :func:`prepare_w2_fp8_padded` pads the forward fc2 first."""
    G, N, K = l2_weights.shape
    w2q, w2s = quantize_grouped_weight_mxfp8_flydsl(l2_weights)
    b_sp = preshuffle_b_scale(w2s, G, N, K, pack=4)
    weight_flat = w2q.reshape(G * N, K).contiguous().view(torch.int8).reshape(-1)
    return weight_flat, b_sp


def prepare_w2_fp8_padded(w2: torch.Tensor) -> tuple:
    """The forward fc2 weight ``[G, H, I]``, padded to :func:`fp8_padded_dims`, prepped for the combine."""
    return prepare_w2_fp8(pad_w2_fp8_dims(w2))


def prepare_w2t_dgrad_fp8(w2: torch.Tensor) -> tuple:
    """``w2^T`` [G,Ip,Hp] prepped for the L2 dgrad's dispatch GEMM -> ``(wq, flat, b_sp)``.

    The L2 dgrad runs NT via the transposed weight, so w2 must be quantized along H (its
    contraction axis). Static weight prep; the transpose never runs inside the kernel.
    """
    return prepare_dispatch_weight_fp8(pad_w2_fp8_dims(w2).transpose(1, 2).contiguous())  # [G,Ip,Hp]


def prepare_w1t_combine_fp8(w1: torch.Tensor) -> tuple:
    """``w1^T`` [G, Hp, 2Ip] prepped for the L1 dgrad's combine GEMM (same format as the forward w2)."""
    return prepare_w2_fp8(pad_w1_fp8_dims(w1).transpose(1, 2).contiguous())


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
    """-> the dispatch GEMM's 3-tuple ``(w1q, flat, b_sp)``; see ``prepare_dispatch_weight_fp8``."""
    return _version_keyed_weight_prep(w1, _W1_PREP_ATTR, prepare_w1_fp8)


def _w2_fp8_cached(w2: torch.Tensor) -> tuple:
    """-> ``(weight_flat int8 [G*Hp*Ip], b_sp int32)`` for the forward fc2 combine."""
    return _version_keyed_weight_prep(w2, _W2_PREP_ATTR, prepare_w2_fp8_padded)


def _w2t_fp8_cached(w2: torch.Tensor) -> tuple:
    """-> ``w2^T`` prepped for the L2 dgrad's dispatch GEMM."""
    return _version_keyed_weight_prep(w2, _W2T_PREP_ATTR, prepare_w2t_dgrad_fp8)


def _w1t_combine_fp8_cached(w1: torch.Tensor) -> tuple:
    """-> ``w1^T`` prepped for the L1 dgrad's combine GEMM."""
    return _version_keyed_weight_prep(w1, _W1T_COMBINE_PREP_ATTR, prepare_w1t_combine_fp8)
