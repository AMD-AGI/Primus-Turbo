###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FlyDSL hd64 flash-attention operators (Meta shape family), gfx950 / MI355X.

Two independent operators (not fused into a single autograd op):

* ``flash_attn_varlen_flydsl_forward_impl`` — native dual-wave software-pipelined
  FlyDSL forward (``flash_attn_gfx950``). Returns O (and optionally LSE).
* ``flash_attn_varlen_flydsl_backward_impl`` — deterministic 16x16x32 FlyDSL
  backward (``flash_attn_bwd_rect16_kernel`` via ``flydsl_varlen_backward``).
  Returns dQ / dK / dV.

Constraints (both operators): THD/varlen packed layout with UNIFORM per-batch
seqlens, bottom-right causal, GQA, D in {64, 128}, bf16. The softmax scale is
baked to 1/sqrt(D) by the forward kernel; the backward takes it explicitly.
These mirror the ``attention_aiter_impl`` / ``attention_triton_impl`` impl layer;
higher-level dispatch/autograd wiring is intentionally left to the caller.
"""

import functools
import math

import torch

from primus_turbo.flydsl.attention.flash_attn_bwd import (
    flydsl_varlen_backward,
)
from primus_turbo.flydsl.attention.flash_attn_fwd import (
    build_flash_attn_dualwave_swp_module,
)


def _uniform_seqlen(cu_seqlens: "torch.Tensor"):
    """(batch, S) from a cumulative-seqlens vector; asserts every segment is equal
    (the FlyDSL varlen kernels are compiled for a uniform per-batch length)."""
    seg = cu_seqlens[1:] - cu_seqlens[:-1]
    S = int(seg[0].item())
    assert bool((seg == S).all().item()), "flydsl flash-attn requires uniform seqlens"
    return cu_seqlens.numel() - 1, S


@functools.lru_cache(maxsize=64)
def _fwd_module(Hq, Hkv, D, causal, cross_seqlen, emit_lse, window_left):
    return build_flash_attn_dualwave_swp_module(
        num_heads=Hq,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=Hkv,
        varlen=True,
        cross_seqlen=cross_seqlen,
        emit_lse=emit_lse,
        window_left=window_left,
    )


def flash_attn_varlen_flydsl_forward_impl(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    return_lse=False,
):
    """Native FlyDSL forward. q:[B*Sq,Hq,D], k/v:[B*Skv,Hkv,D] bf16 (THD packed).
    Returns O:[B*Sq,Hq,D] (and LSE:[B*Sq,Hq] fp32 when ``return_lse``)."""
    assert causal, "flydsl flash-attn forward is bottom-right causal only"
    assert q.dtype == torch.bfloat16, "flydsl flash-attn forward is bf16 only"
    B, Sq = _uniform_seqlen(cu_seqlens_q)
    Bk, Skv = _uniform_seqlen(cu_seqlens_k)
    assert B == Bk, f"q/k batch mismatch ({B} vs {Bk})"
    Hq, D = q.shape[1], q.shape[2]
    Hkv = k.shape[1]
    assert D in (64, 128), f"flydsl flash-attn forward supports D in (64,128), got {D}"
    if softmax_scale is not None:
        assert abs(softmax_scale - 1.0 / math.sqrt(D)) < 1e-6, (
            "flydsl flash-attn forward bakes softmax_scale=1/sqrt(D)"
        )
    wl, wr = window_size
    assert wr in (0, -1), "only left-window (W,0) / full (-1,-1) supported"
    window_left = wl if wl >= 0 else -1

    mod = _fwd_module(Hq, Hkv, D, True, Sq != Skv, bool(return_lse), window_left)
    out = torch.empty_like(q)
    stream = torch.cuda.current_stream()
    kw = dict(seq_len_kv=Skv, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k, stream=stream)
    lse = None
    if return_lse:
        # LSE flows through the DebugCounts slot; kernel layout is [total_q, Hq] fp32.
        lse = torch.zeros((B * Sq, Hq), device=q.device, dtype=torch.float32)
        kw["debug_counts"] = lse
    mod(q, k, v, out, B, Sq, **kw)
    return (out, lse) if return_lse else out


def flash_attn_varlen_flydsl_backward_impl(
    dout,
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    fast_exp2=False,
):
    """Deterministic 16x16x32 FlyDSL backward. ``lse`` is the natural-log softmax
    LSE in [B, Hq, Sq] fp32 (the backward prescales it internally). Returns
    dQ:[B*Sq,Hq,D], dK/dV:[B*Skv,Hkv,D]."""
    assert causal, "flydsl flash-attn backward is bottom-right causal only"
    B, Sq = _uniform_seqlen(cu_seqlens_q)
    Bk, Skv = _uniform_seqlen(cu_seqlens_k)
    assert B == Bk, f"q/k batch mismatch ({B} vs {Bk})"
    Hq, D = q.shape[1], q.shape[2]
    Hkv = k.shape[1]
    assert D in (64, 128), f"flydsl flash-attn backward supports D in (64,128), got {D}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    wl, wr = window_size
    assert wr in (0, -1), "only left-window (W,0) / full (-1,-1) supported"
    window_left = wl if wl >= 0 else -1

    return flydsl_varlen_backward(
        dout.contiguous(),
        q,
        k,
        v,
        out,
        lse,
        B,
        Sq,
        Skv,
        Hq,
        Hkv,
        D,
        softmax_scale,
        fast_exp2=fast_exp2,
        window_left=window_left,
    )
