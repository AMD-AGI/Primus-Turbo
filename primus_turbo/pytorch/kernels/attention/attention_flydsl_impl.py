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


def _uniform_shape(cu_seqlens: "torch.Tensor", max_seqlen, total):
    """(batch, S) iff every segment is exactly ``max_seqlen``, else None.

    Routes to the rectangular rect16 fast path, which is compiled for a uniform
    per-batch length; the ragged / block-causal path reads per-segment boundaries
    from cu_seqlens inside the kernel instead.

    The test is host-only on purpose. Every segment is <= max_seqlen (the caller's
    contract, which the grid already relies on) and they sum to ``total``, so
    total == batch*max_seqlen forces all of them to equal max_seqlen. Deriving it
    from cu_seqlens instead costs a .item() -- a full device sync plus a handful of
    tiny reduce kernels -- on EVERY backward call, which at the gpt-oss prefill
    shape measured ~1.1% of the backward wall and blocks the host from running the
    launch sequence ahead of the GPU.
    """
    B = cu_seqlens.numel() - 1
    S = int(max_seqlen)
    return (B, S) if B * S == int(total) else None


@functools.lru_cache(maxsize=64)
def _fwd_module(Hq, Hkv, D, causal, cross_seqlen, emit_lse, window_left, sbhd=False, has_sink=False):
    # D in (64,128) use the tuned stagger-off config (block_m=128, waves_per_eu=2):
    # stagger-off lifts MFMA utilization for both. The raw 8-wave build default halves
    # occupancy; this is the same point D64 was tuned to. Other head dims keep the default.
    cfg = {}
    if D in (64, 128):
        cfg = dict(waves_per_eu=2, dualwave_swp_enable_stagger=False, block_m=128)
    return build_flash_attn_dualwave_swp_module(
        num_heads=Hq,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=Hkv,
        varlen=not sbhd,
        cross_seqlen=cross_seqlen,
        emit_lse=emit_lse,
        window_left=window_left,
        sbhd=sbhd,
        has_sink=has_sink,
        **cfg,
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
    sink=None,
):
    """Native FlyDSL forward. q:[total_q,Hq,D], k/v:[total_kv,Hkv,D] bf16 (THD packed).
    Segment boundaries come from ``cu_seqlens_q/k`` -- ragged (block-causal / document
    masking) is supported natively: the kernel reads per-segment [tok_base,tok_end)
    from cu_seqlens and applies a per-segment bottom-right causal (offset =
    seglen_kv-seglen_q). Uniform seqlens are just the special case where every segment
    is equal. Grid tiles by ``max_seqlen_q``; out-of-segment tiles early-exit.
    ``sink`` (optional [Hq] fp32) is a learned per-q-head attention sink folded into the
    softmax denominator by the kernel. Returns O:[total_q,Hq,D] (and LSE:[total_q,Hq]
    fp32 when ``return_lse`` -- sink-inclusive when ``sink`` is given)."""
    assert causal, "flydsl flash-attn forward is bottom-right causal only"
    assert q.dtype == torch.bfloat16, "flydsl flash-attn forward is bf16 only"
    B = cu_seqlens_q.numel() - 1
    Bk = cu_seqlens_k.numel() - 1
    assert B == Bk, f"q/k batch mismatch ({B} vs {Bk})"
    # Grid tiles by the max per-segment length; equal-length is max_seqlen == per-seg length.
    Sq, Skv = int(max_seqlen_q), int(max_seqlen_k)
    total_q = q.shape[0]
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
    if sink is not None:
        assert sink.dtype == torch.float32 and sink.numel() == Hq, "sink must be fp32 [Hq]"
        sink = sink.contiguous()

    mod = _fwd_module(Hq, Hkv, D, True, Sq != Skv, bool(return_lse), window_left, has_sink=sink is not None)
    out = torch.empty_like(q)
    stream = torch.cuda.current_stream()
    kw = dict(seq_len_kv=Skv, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k, sink=sink, stream=stream)
    lse = None
    if return_lse:
        # LSE flows through the DebugCounts slot; kernel layout is [total_q, Hq] fp32.
        lse = torch.zeros((total_q, Hq), device=q.device, dtype=torch.float32)
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
    sink=None,
):
    """Deterministic 16x16x32 FlyDSL backward. ``lse`` is packed [total_q,Hq] fp32 (as
    emitted by the forward) for both paths; the uniform path transposes it to [B,Hq,Sq]
    internally. Returns dQ:[total_q,Hq,D], dK/dV:[total_kv,Hkv,D] (and dsink:[Hq] fp32
    when ``sink`` is given -- uniform path only). ``sink`` (optional [Hq] fp32) is the
    learned per-q-head attention sink; dQ/dK/dV are sink-agnostic (the saved LSE is
    already sink-inclusive), only dsink is added. Ragged/block-causal has no sink."""
    assert causal, "flydsl flash-attn backward is bottom-right causal only"
    Hq, D = q.shape[1], q.shape[2]
    Hkv = k.shape[1]
    assert D in (64, 128), f"flydsl flash-attn backward supports D in (64,128), got {D}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    wl, wr = window_size
    assert wr in (0, -1), "only left-window (W,0) / full (-1,-1) supported"
    window_left = wl if wl >= 0 else -1
    if sink is not None:
        assert sink.dtype == torch.float32 and sink.numel() == Hq, "sink must be fp32 [Hq]"
        sink = sink.contiguous()

    uq = _uniform_shape(cu_seqlens_q, max_seqlen_q, q.shape[0])
    uk = _uniform_shape(cu_seqlens_k, max_seqlen_k, k.shape[0])
    if uq is not None and uk is not None:
        B, Sq = uq
        Bk, Skv = uk
        assert B == Bk, f"q/k batch mismatch ({B} vs {Bk})"
        # rect16 wants head-major [B,Hq,Sq]; left non-contiguous so the -log2e prescale
        # inside the backward materialises it in ONE pass instead of copy-then-scale.
        lse_bhsq = lse.reshape(B, Sq, Hq).permute(0, 2, 1)
        return flydsl_varlen_backward(
            dout.contiguous(),
            q,
            k,
            v,
            out,
            lse_bhsq,
            B,
            Sq,
            Skv,
            Hq,
            Hkv,
            D,
            softmax_scale,
            window_left=window_left,
            sink=sink,
        )

    # Sparse block-diagonal (CP): empty q-segs give zero dk/dv but still launch early-exit
    # WGs; when non-empty segs are sparse (~<=1/8), run only those as rect16 sub-problems.
    cq = cu_seqlens_q.cpu().tolist()
    n_seg = len(cq) - 1
    nonempty = [s for s in range(n_seg) if cq[s + 1] > cq[s]]
    n_ne = len(nonempty)
    if sink is None and n_ne * 8 <= n_seg and (n_seg - n_ne) >= 8:
        ck = cu_seqlens_k.cpu().tolist()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        for s in nonempty:
            q0, q1 = cq[s], cq[s + 1]
            k0, k1 = ck[s], ck[s + 1]
            lq, lk = q1 - q0, k1 - k0
            # lse is packed [total_q,Hq]; rect16 wants head-major [B=1,Hq,Sq].
            lse_bhsq = lse[q0:q1].reshape(1, lq, Hq).permute(0, 2, 1)
            dqs, dks, dvs = flydsl_varlen_backward(
                dout[q0:q1].contiguous(),
                q[q0:q1],
                k[k0:k1],
                v[k0:k1],
                out[q0:q1],
                lse_bhsq,
                1,
                lq,
                lk,
                Hq,
                Hkv,
                D,
                softmax_scale,
                window_left=window_left,
            )
            dq[q0:q1] = dqs
            dk[k0:k1] = dks
            dv[k0:k1] = dvs
        return dq, dk, dv

    # Ragged / block-causal: per-segment [tok_base,tok_end) from cu_seqlens.
    B = cu_seqlens_q.numel() - 1
    Bk = cu_seqlens_k.numel() - 1
    assert B == Bk, f"q/k batch mismatch ({B} vs {Bk})"
    max_sq, max_skv = int(max_seqlen_q), int(max_seqlen_k)
    return flydsl_varlen_backward(
        dout.contiguous(),
        q,
        k,
        v,
        out,
        lse,
        B,
        max_sq,
        max_skv,
        Hq,
        Hkv,
        D,
        softmax_scale,
        window_left=window_left,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=max_sq,
        max_seqlen_kv=max_skv,
    )


def flash_attn_sbhd_flydsl_forward_impl(
    q,
    k,
    v,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    return_lse=False,
    sink=None,
):
    """Native SBHD FlyDSL forward. q:[Sq,B,Hq,D], k/v:[Skv,B,Hkv,D] bf16 (the trace
    layout -- batch interleaved inside the seq axis). NO permute/copy: the kernel
    addresses SBHD directly via a compile-time SBHD trait + runtime seq-step stride
    (B*H*D). ``sink`` (optional [Hq] fp32) is folded into the softmax denominator.
    Returns O:[Sq,B,Hq,D] (and LSE:[B*Sq,Hq] fp32, sink-inclusive, when ``return_lse``)."""
    assert causal, "flydsl flash-attn forward is bottom-right causal only"
    assert q.dtype == torch.bfloat16, "flydsl flash-attn forward is bf16 only"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "SBHD tensors must be contiguous"
    Sq, B, Hq, D = q.shape
    Skv, Bk, Hkv, Dk = k.shape
    assert B == Bk and D == Dk, f"q/k shape mismatch ({q.shape} vs {k.shape})"
    assert D in (64, 128), f"flydsl flash-attn forward supports D in (64,128), got {D}"
    if softmax_scale is not None:
        assert abs(softmax_scale - 1.0 / math.sqrt(D)) < 1e-6, (
            "flydsl flash-attn forward bakes softmax_scale=1/sqrt(D)"
        )
    wl, wr = window_size
    assert wr in (0, -1), "only left-window (W,0) / full (-1,-1) supported"
    window_left = wl if wl >= 0 else -1
    if sink is not None:
        assert sink.dtype == torch.float32 and sink.numel() == Hq, "sink must be fp32 [Hq]"
        sink = sink.contiguous()

    mod = _fwd_module(
        Hq, Hkv, D, True, Sq != Skv, bool(return_lse), window_left, sbhd=True, has_sink=sink is not None
    )
    out = torch.empty_like(q)
    stream = torch.cuda.current_stream()
    # SBHD seq-step strides live in the runtime stride args; the SBHD trait fixes the
    # per-batch base to H*D.
    kw = dict(
        seq_len_kv=Skv,
        stride_q_n=B * Hq * D,
        stride_kv_n=B * Hkv * D,
        sink=sink,
        stream=stream,
    )
    lse = None
    if return_lse:
        # LSE is batch-major [B*Sq, Hq] fp32 (layout-independent of SBHD q/k/v).
        lse = torch.zeros((B * Sq, Hq), device=q.device, dtype=torch.float32)
        kw["debug_counts"] = lse
    mod(q, k, v, out, B, Sq, **kw)
    return (out, lse) if return_lse else out


def flash_attn_sbhd_flydsl_backward_impl(
    dout,
    q,
    k,
    v,
    out,
    lse,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    sink=None,
):
    """Native SBHD deterministic 16x16x32 FlyDSL backward. q/dout/out:[Sq,B,Hq,D],
    k/v:[Skv,B,Hkv,D] bf16; ``lse`` is natural-log softmax LSE in [B,Hq,Sq] fp32.
    NO permute/copy: SBHD is addressed natively and the dk/dv workspace is laid out
    so the slot reduction is contiguous. Returns dQ:[Sq,B,Hq,D], dK/dV:[Skv,B,Hkv,D]
    (and dsink:[Hq] fp32 when ``sink`` is given). ``sink`` (optional [Hq] fp32) is the
    learned per-q-head attention sink; dQ/dK/dV are sink-agnostic (saved LSE is already
    sink-inclusive), only dsink is added."""
    assert causal, "flydsl flash-attn backward is bottom-right causal only"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "SBHD tensors must be contiguous"
    Sq, B, Hq, D = q.shape
    Skv, Bk, Hkv, Dk = k.shape
    assert B == Bk and D == Dk, f"q/k shape mismatch ({q.shape} vs {k.shape})"
    assert D in (64, 128), f"flydsl flash-attn backward supports D in (64,128), got {D}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    wl, wr = window_size
    assert wr in (0, -1), "only left-window (W,0) / full (-1,-1) supported"
    window_left = wl if wl >= 0 else -1
    if sink is not None:
        assert sink.dtype == torch.float32 and sink.numel() == Hq, "sink must be fp32 [Hq]"
        sink = sink.contiguous()

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
        window_left=window_left,
        sbhd=True,
        sink=sink,
    )
