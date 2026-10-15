###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ready-made score mods for flex attention.

A ``score_mod(score, b, h, q_idx, kv_idx [, *aux]) -> score`` is a ``@triton.jit``
function inlined into the kernel; it runs before masking, on the scaled score. Its VJP
``score_mod_bwd(dscore, score, b, h, q_idx, kv_idx) -> dscore`` is written by hand.

Relative-position bias (Inkling): ``score += rel_logits[t, h, q_idx - kv_idx]`` for
``0 <= q_idx - kv_idx < RE``, where ``t`` is the query's packed row. For a fixed query
and head every distance maps to exactly one key, so the gradient has one contributing
pair per element and the score-gradient hook writes it with plain stores in the dQ
sweep -- no atomics and no second pass over K/V.
"""

import functools

import torch
import triton

from primus_turbo.pytorch.kernels.flex_attention.flex_attention_jit_utils import (
    build_jit,
    literal_tag,
)
from primus_turbo.pytorch.ops.attention.flex_attention_interface import (
    flex_attention_varlen,
)
from primus_turbo.pytorch.ops.attention.flex_attention_masks import (
    causal_mask,
    create_block_mask_varlen,
    sliding_window_mask,
)

__all__ = [
    "flex_attention_varlen_rel_bias",
    "identity_score_mod_bwd",
    "make_rel_bias_mods",
    "make_softcap_score_mod",
]


@triton.jit
def identity_score_mod_bwd(dscore, score, b, h, q_idx, kv_idx):
    """VJP of any additive score_mod (``score + f(b, h, q_idx, kv_idx)``)."""
    return dscore


@functools.lru_cache(maxsize=None)
def make_softcap_score_mod(softcap: float):
    """``(score_mod, score_mod_bwd)`` for logit softcapping, ``cap * tanh(score / cap)``."""
    if softcap <= 0:
        raise ValueError(f"softcap must be positive, got {softcap}")
    cap = float(softcap)
    tag = literal_tag(cap)
    fwd_name, bwd_name = f"softcap_score_mod_{tag}", f"softcap_score_mod_bwd_{tag}"
    src = f"""
@triton.jit
def {fwd_name}(score, b, h, q_idx, kv_idx):
    return {cap!r} * libdevice.tanh(score * {1.0 / cap!r})


@triton.jit
def {bwd_name}(dscore, score, b, h, q_idx, kv_idx):
    t = libdevice.tanh(score * {1.0 / cap!r})
    return dscore * (1.0 - t * t)
"""
    return build_jit(src, (fwd_name, bwd_name), f"softcap_{tag}")


@functools.lru_cache(maxsize=None)
def make_rel_bias_mods(num_heads: int, rel_extent: int):
    """``(score_mod, score_grad_hook)`` for ``rel_logits`` ``[T, num_heads, rel_extent]``.

    Both take the aux tensors ``(rel_logits, cu_seqlens)``: ``b`` is the sequence index
    under varlen, so ``cu_seqlens[b]`` turns the sequence-local ``q_idx`` into the packed
    row. Rows past the sequence end are masked so they never touch another sequence's rows.
    """
    H, RE = int(num_heads), int(rel_extent)
    mod_name, hook_name = f"rel_bias_score_mod_{H}_{RE}", f"rel_bias_grad_hook_{H}_{RE}"
    src = f"""
@triton.jit
def _rel_offsets_{H}_{RE}(b, h, q_idx, kv_idx, CU_SEQLENS):
    seq_start = tl.load(CU_SEQLENS + b)
    seq_end = tl.load(CU_SEQLENS + b + 1)
    rd = q_idx - kv_idx
    row = (seq_start + q_idx).to(tl.int64)
    keep = (rd >= 0) & (rd < {RE}) & (row < seq_end)
    return row * {H * RE} + h * {RE} + rd, keep


@triton.jit
def {mod_name}(score, b, h, q_idx, kv_idx, REL, CU_SEQLENS):
    offset, keep = _rel_offsets_{H}_{RE}(b, h, q_idx, kv_idx, CU_SEQLENS)
    return score + tl.load(REL + offset, mask=keep, other=0.0).to(tl.float32)


@triton.jit
def {hook_name}(dscore, valid, b, h, q_idx, kv_idx, DREL, REL, CU_SEQLENS):
    offset, keep = _rel_offsets_{H}_{RE}(b, h, q_idx, kv_idx, CU_SEQLENS)
    tl.store(DREL + offset, dscore.to(DREL.dtype.element_ty), mask=keep & valid)
"""
    return build_jit(src, (mod_name, hook_name), f"rel_bias_{H}_{RE}")


def flex_attention_varlen_rel_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rel_logits: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale=None,
    window_left=None,
):
    """Causal varlen self-attention with a relative-position bias table.

    query ``[T, Hq, D]``, key/value ``[T, Hkv, D]`` (GQA allowed), rel_logits
    ``[T, Hq, RE]`` contiguous (Inkling uses fp32). Keys are kept when
    ``0 <= q_idx - kv_idx <= window_left`` (no window when None). Gradients flow to
    query, key, value and rel_logits.
    """
    T, H, _ = query.shape
    if rel_logits.dim() != 3 or rel_logits.shape[:2] != (T, H):
        raise ValueError(f"rel_logits must be [T, Hq, RE] = [{T}, {H}, RE], got {tuple(rel_logits.shape)}")
    if not rel_logits.is_contiguous():
        raise ValueError("rel_logits must be contiguous")
    score_mod, grad_hook = make_rel_bias_mods(H, rel_logits.shape[-1])
    mask = causal_mask if window_left is None else sliding_window_mask(window_left)
    return flex_attention_varlen(
        query,
        key,
        value,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        score_mod=score_mod,
        block_mask=create_block_mask_varlen(mask, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen),
        scale=scale,
        enable_gqa=True,
        score_mod_bwd=identity_score_mod_bwd,
        aux_tensors=[rel_logits.detach(), cu_seqlens],
        score_grad_hook=grad_hook,
        score_grad_target=rel_logits,
    )
