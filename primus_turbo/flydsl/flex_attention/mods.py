###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ready-made FlyDSL score / mask mods, with FA4's signatures:

    score_mod(score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors) -> score
    score_mod_bwd(dscore, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors) -> dscore
    mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors) -> bool

A trainable aux tensor gets its gradient from a ``score_grad_hook``, called once per
(query, key) pair in the backward's dQ sweep with the gradient of the score_mod output:

    score_grad_hook(dscore, valid, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors)
        -> (flat index into the gradient, value, store?)

The kernel stores ``value`` there when ``store?`` (mods have no side effects of their
own); ``valid`` is false for pairs past either sequence's end.

They are plain Python functions traced into the kernel, one element at a time:
``score`` is an ``fx.Float32`` already multiplied by ``softmax_scale``, the indices are
``fx.Int32`` and sequence-local, and ``aux_tensors`` is a tuple of buffer tensors.
Because a mod is traced rather than AST-rewritten, branch with ``.select`` rather than
a Python ``if`` on runtime values.
"""

import functools
import math

import flydsl.expr as fx


def tanh(x):
    """``tanh`` from ``exp2``; ``math.tanh`` has no AMDGPU lowering (no ``ftanh`` libcall)."""
    t = fx.math.exp2(fx.math.absf(x) * fx.Float32(-2.0 * math.log2(math.e)))
    return fx.math.copysign((fx.Float32(1.0) - t) / (fx.Float32(1.0) + t), x)


def identity_score_mod_bwd(dscore, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    """VJP of any additive score_mod (``score + f(...)``)."""
    return dscore


def noop_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    return kv_idx == kv_idx


def causal_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    """``kv_idx <= q_idx`` (top-left aligned, as PyTorch's ``causal_mask``)."""
    return kv_idx <= q_idx


@functools.lru_cache(maxsize=None)
def make_sliding_window_mask(window_left: int, window_right: int):
    """``q_idx - window_left <= kv_idx <= q_idx + window_right`` (top-left aligned)."""
    left, right = int(window_left), int(window_right)

    def sliding_window_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        rel = q_idx - kv_idx
        return (rel <= fx.Int32(left)) & (rel >= fx.Int32(-right))

    return sliding_window_mask


@functools.lru_cache(maxsize=None)
def make_softcap_mods(softcap: float):
    """``(score_mod, score_mod_bwd)`` for ``cap * tanh(score / cap)``, as FA4 implements
    ``softcap``. Cached so one value maps to one function (and one compiled kernel)."""
    cap = float(softcap)
    inv = 1.0 / cap

    def softcap_score_mod(score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return fx.Float32(cap) * tanh(score * fx.Float32(inv))

    def softcap_score_mod_bwd(dscore, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        t = tanh(score * fx.Float32(inv))
        return dscore * (fx.Float32(1.0) - t * t)

    return softcap_score_mod, softcap_score_mod_bwd


@functools.lru_cache(maxsize=None)
def make_rel_bias_mods(num_heads: int, rel_extent: int):
    """``(score_mod, score_grad_hook)`` for ``score += rel_logits[t, h, q_idx - kv_idx]``,
    ``0 <= q_idx - kv_idx < rel_extent``, with aux tensors ``(rel_logits [T, H, RE] fp32,
    cu_seqlens)``: ``t = cu_seqlens[b] + q_idx`` is the query's packed row (varlen). The row
    comes from ``seqlen_info`` (the kernel's per-sequence offset and length, from the same
    cu_seqlens), not a per-score reload of ``cu_seqlens``."""
    H, RE = int(num_heads), int(rel_extent)

    def rel_index(q_idx, kv_idx, seqlen_info):
        rd = q_idx - kv_idx
        keep = (rd >= fx.Int32(0)) & (rd < fx.Int32(RE)) & (q_idx < seqlen_info.seqlen_q)
        return seqlen_info.offset_q + q_idx, rd, keep

    def rel_bias_score_mod(score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        row, rd, keep = rel_index(q_idx, kv_idx, seqlen_info)
        zero = fx.Int32(0)
        bias = fx.Float32(fx.slice(aux_tensors[0], (keep.select(row, zero), head_idx, None))[keep.select(rd, zero)])
        return score + keep.select(bias, fx.Float32(0.0))

    def rel_bias_grad_hook(dscore, valid, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        row, rd, keep = rel_index(q_idx, kv_idx, seqlen_info)
        return row * fx.Int32(H * RE) + head_idx * fx.Int32(RE) + rd, dscore, keep & valid

    return rel_bias_score_mod, rel_bias_grad_hook
