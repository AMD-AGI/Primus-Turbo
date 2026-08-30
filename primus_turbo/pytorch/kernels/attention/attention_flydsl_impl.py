###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL flash-attention forward/backward impls (THD varlen and SBHD), gfx950 / MI355X.

Bottom-right causal, GQA, D in {64, 128}, bf16. The forward bakes softmax_scale to
1/sqrt(D); the backward takes it explicitly. Mirrors the ``attention_aiter_impl`` /
``attention_triton_impl`` layer -- dispatch and autograd wiring belong to the caller.
"""

import functools
import math
from typing import Optional, Tuple

import torch

from primus_turbo.flydsl.attention.flash_attn_bwd import (
    flydsl_varlen_backward,
)
from primus_turbo.flydsl.attention.flash_attn_fwd import (
    build_flash_attn_dualwave_swp_module,
)

# Custom ops so a compiled caller sees opaque nodes instead of tracing into FlyDSL's JIT
# build (which shells out and takes locks) -- a graph break there is what fullgraph=True
# forbids. cudagraph_unsafe because the kernels keep module-level state a capture would
# strand in the graph pool; without it max-autotune fails on live pool pointers.
_custom_op = functools.partial(torch.library.custom_op, tags=(torch._C.Tag.cudagraph_unsafe,))


def _check_bwd(q, k, v, softmax_scale, causal, window_size, sink, num_heads_q, head_dim, sbhd=False):
    """Shared backward preconditions; returns (softmax_scale, normalized left window)."""
    assert causal, "flydsl flash-attn backward is bottom-right causal only"
    if sbhd:
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (
            "SBHD tensors must be contiguous"
        )
    assert head_dim in (64, 128), f"flydsl flash-attn backward supports D in (64,128), got {head_dim}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    wl, wr = window_size
    assert wr in (0, -1), "only left-window (W,0) / full (-1,-1) supported"
    if sink is not None:
        assert sink.dtype == torch.float32 and sink.numel() == num_heads_q, "sink must be fp32 [Hq]"
    return softmax_scale, (wl if wl >= 0 else -1)


def _check_fwd(q, k, v, softmax_scale, causal, window_size, sink, num_heads_q, head_dim, sbhd=False):
    """Shared forward preconditions; returns the normalized left window."""
    assert causal, "flydsl flash-attn forward is bottom-right causal only"
    assert q.dtype == torch.bfloat16, "flydsl flash-attn forward is bf16 only"
    if sbhd:
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (
            "SBHD tensors must be contiguous"
        )
    assert head_dim in (64, 128), f"flydsl flash-attn forward supports D in (64,128), got {head_dim}"
    if softmax_scale is not None:
        assert abs(softmax_scale - 1.0 / math.sqrt(head_dim)) < 1e-6, (
            "flydsl flash-attn forward bakes softmax_scale=1/sqrt(D)"
        )
    wl, wr = window_size
    assert wr in (0, -1), "only left-window (W,0) / full (-1,-1) supported"
    if sink is not None:
        assert sink.dtype == torch.float32 and sink.numel() == num_heads_q, "sink must be fp32 [Hq]"
    return wl if wl >= 0 else -1


def _uniform_shape(cu_seqlens: "torch.Tensor", max_seqlen, total):
    """(batch, S) iff every segment is exactly ``max_seqlen``, else None -- i.e. whether
    the uniform rect16 fast path applies rather than the ragged one.

    Host-only on purpose: segments are <= max_seqlen and sum to ``total``, so the equality
    forces all of them equal. Reading cu_seqlens instead costs a .item() (device sync) on
    every backward call, ~1.1% of the backward wall at the gpt-oss prefill shape.
    """
    B = cu_seqlens.numel() - 1
    S = int(max_seqlen)
    return (B, S) if B * S == int(total) else None


@functools.lru_cache(maxsize=64)
def _fwd_module(Hq, Hkv, D, causal, cross_seqlen, emit_lse, window_left, sbhd=False, has_sink=False):
    # D in (64,128): stagger-off lifts MFMA utilization, and the raw 8-wave build default
    # halves occupancy. Other head dims keep the build defaults.
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


@_custom_op("primus_turbo::flash_attn_varlen_flydsl_forward", mutates_args=(), device_types="cuda")
def _varlen_forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_left: int,
    return_lse: bool,
    sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = cu_seqlens_q.numel() - 1
    # Grid tiles by the max per-segment length; equal-length is max_seqlen == per-seg length.
    Sq, Skv = int(max_seqlen_q), int(max_seqlen_k)
    total_q, Hq, D = q.shape
    Hkv = k.shape[1]
    sink = sink.contiguous() if sink is not None else None

    mod = _fwd_module(Hq, Hkv, D, True, Sq != Skv, bool(return_lse), window_left, has_sink=sink is not None)
    out = torch.empty_like(q)
    stream = torch.cuda.current_stream()
    kw = dict(seq_len_kv=Skv, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k, sink=sink, stream=stream)
    # LSE flows through the DebugCounts slot, [total_q, Hq] fp32. The op returns a fixed
    # number of tensors, so an unwanted LSE comes back empty rather than absent.
    lse = torch.zeros((total_q, Hq) if return_lse else (0,), device=q.device, dtype=torch.float32)
    if return_lse:
        kw["debug_counts"] = lse
    mod(q, k, v, out, B, Sq, **kw)
    return out, lse


@_varlen_forward_op.register_fake
def _varlen_forward_op_fake(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, window_left, return_lse, sink
):
    lse_shape = (q.shape[0], q.shape[1]) if return_lse else (0,)
    return torch.empty_like(q), q.new_empty(lse_shape, dtype=torch.float32)


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
    """THD forward: q [total_q,Hq,D], k/v [total_kv,Hkv,D] bf16. Ragged is native -- the
    kernel reads each segment from cu_seqlens and applies bottom-right causal per segment;
    uniform seqlens are the special case. ``sink`` ([Hq] fp32) folds into the softmax
    denominator. Returns O (and a sink-inclusive LSE [total_q,Hq] fp32 if ``return_lse``)."""
    Hq, D = q.shape[1], q.shape[2]
    assert cu_seqlens_q.numel() == cu_seqlens_k.numel(), "q/k batch mismatch"
    window_left = _check_fwd(q, k, v, softmax_scale, causal, window_size, sink, Hq, D)
    out, lse = _varlen_forward_op(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        window_left,
        bool(return_lse),
        sink,
    )
    return (out, lse) if return_lse else out


def _pad_dsink(grads, like):
    """(dq, dk, dv, dsink) from a backward that returns dsink only when it had a sink."""
    return grads[0], grads[1], grads[2], (grads[3] if len(grads) > 3 else like.new_empty((0,)))


@_custom_op("primus_turbo::flash_attn_varlen_flydsl_backward", mutates_args=(), device_types="cuda")
def _varlen_backward_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_left: int,
    sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Hq, D = q.shape[1], q.shape[2]
    Hkv = k.shape[1]
    sink = sink.contiguous() if sink is not None else None

    uq = _uniform_shape(cu_seqlens_q, max_seqlen_q, q.shape[0])
    uk = _uniform_shape(cu_seqlens_k, max_seqlen_k, k.shape[0])
    if uq is not None and uk is not None:
        B, Sq = uq
        Bk, Skv = uk
        assert B == Bk, f"q/k batch mismatch ({B} vs {Bk})"
        # rect16 wants head-major [B,Hq,Sq]; left non-contiguous so the -log2e prescale
        # inside the backward materialises it in one pass instead of copy-then-scale.
        lse_bhsq = lse.reshape(B, Sq, Hq).permute(0, 2, 1)
        grads = flydsl_varlen_backward(
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
        return _pad_dsink(grads, lse)

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
        return dq, dk, dv, lse.new_empty((0,))

    # Ragged / block-causal: per-segment [tok_base,tok_end) from cu_seqlens.
    B = cu_seqlens_q.numel() - 1
    Bk = cu_seqlens_k.numel() - 1
    assert B == Bk, f"q/k batch mismatch ({B} vs {Bk})"
    max_sq, max_skv = int(max_seqlen_q), int(max_seqlen_k)
    grads = flydsl_varlen_backward(
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
    return _pad_dsink(grads, lse)


@_varlen_backward_op.register_fake
def _varlen_backward_op_fake(
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
    softmax_scale,
    window_left,
    sink,
):
    dsink_shape = (q.shape[1],) if sink is not None else (0,)
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        lse.new_empty(dsink_shape, dtype=torch.float32),
    )


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
    """Deterministic 16x16x32 THD backward. ``lse`` is packed [total_q,Hq] fp32 as the
    forward emits it. Returns dQ, dK/dV (and dsink [Hq] fp32 with a ``sink``, uniform path
    only -- dQ/dK/dV are sink-agnostic since the saved LSE already includes it)."""
    Hq, D = q.shape[1], q.shape[2]
    softmax_scale, window_left = _check_bwd(q, k, v, softmax_scale, causal, window_size, sink, Hq, D)
    dq, dk, dv, dsink = _varlen_backward_op(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        softmax_scale,
        window_left,
        sink,
    )
    return (dq, dk, dv, dsink) if dsink.numel() else (dq, dk, dv)


@_custom_op("primus_turbo::flash_attn_sbhd_flydsl_forward", mutates_args=(), device_types="cuda")
def _sbhd_forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_left: int,
    return_lse: bool,
    sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    sink = sink.contiguous() if sink is not None else None

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
    # LSE is batch-major [B*Sq, Hq] fp32, independent of the SBHD q/k/v layout.
    lse = torch.zeros((B * Sq, Hq) if return_lse else (0,), device=q.device, dtype=torch.float32)
    if return_lse:
        kw["debug_counts"] = lse
    mod(q, k, v, out, B, Sq, **kw)
    return out, lse


@_sbhd_forward_op.register_fake
def _sbhd_forward_op_fake(q, k, v, window_left, return_lse, sink):
    Sq, B, Hq, _ = q.shape
    lse_shape = (B * Sq, Hq) if return_lse else (0,)
    return torch.empty_like(q), q.new_empty(lse_shape, dtype=torch.float32)


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
    """SBHD forward: q [Sq,B,Hq,D], k/v [Skv,B,Hkv,D] bf16. No permute/copy -- the kernel
    addresses SBHD via a compile-time trait plus a runtime seq-step stride. Returns O (and
    LSE [B*Sq,Hq] fp32 when ``return_lse``)."""
    Hq, D = q.shape[2], q.shape[3]
    window_left = _check_fwd(q, k, v, softmax_scale, causal, window_size, sink, Hq, D, sbhd=True)
    out, lse = _sbhd_forward_op(q, k, v, window_left, bool(return_lse), sink)
    return (out, lse) if return_lse else out


@_custom_op("primus_turbo::flash_attn_sbhd_flydsl_backward", mutates_args=(), device_types="cuda")
def _sbhd_backward_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    window_left: int,
    sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    sink = sink.contiguous() if sink is not None else None

    grads = flydsl_varlen_backward(
        dout,
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
    dsink = grads[3] if len(grads) > 3 else lse.new_empty((0,))
    return grads[0], grads[1], grads[2], dsink


@_sbhd_backward_op.register_fake
def _sbhd_backward_op_fake(dout, q, k, v, out, lse, softmax_scale, window_left, sink):
    dsink_shape = (q.shape[2],) if sink is not None else (0,)
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        lse.new_empty(dsink_shape, dtype=torch.float32),
    )


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
    """SBHD deterministic 16x16x32 backward; ``lse`` is [B,Hq,Sq] fp32 natural-log. No
    permute/copy -- SBHD is addressed natively and the dk/dv workspace is laid out so the
    slot reduction is contiguous. Returns dQ, dK/dV (and dsink [Hq] fp32 with a ``sink``)."""
    Hq, D = q.shape[2], q.shape[3]
    softmax_scale, window_left = _check_bwd(
        q, k, v, softmax_scale, causal, window_size, sink, Hq, D, sbhd=True
    )
    dq, dk, dv, dsink = _sbhd_backward_op(
        dout.contiguous(), q, k, v, out, lse, softmax_scale, window_left, sink
    )
    return (dq, dk, dv, dsink) if sink is not None else (dq, dk, dv)


def _e4m3_pair_bf16(x):
    """x (f32) -> E4M3 (value, residual) pair packed into a bf16-shaped buffer, split at the
    DMA vector's own 8-element granule so the body's de-interleave is a register slice.
    """
    hi = x.to(torch.float8_e4m3fn)
    lo = (x - hi.float()).to(torch.float8_e4m3fn)
    _g = x.shape[:-1] + (x.shape[-1] // 8, 8)
    return (
        torch.stack((hi.view(torch.uint8).reshape(_g), lo.view(torch.uint8).reshape(_g)), dim=-2)
        .contiguous()
        .reshape(x.shape[:-1] + (2 * x.shape[-1],))
        .view(torch.bfloat16)
    )


# fp8 handover: a real fp8 pipeline quantises Q and K in the FORWARD and the backward reads
# those copies, so `prepare` produces them off the step's clock.  Nothing that depends on dO
# crosses it, and the copies are no larger than the bf16 originals.
def flash_attn_sbhd_flydsl_backward_prepare(
    q, k, v, out, lse, softmax_scale=None, causal=True, window_size=(-1, -1), sink=None
):
    """Forward-side handover for the SBHD backward.  Not on the step's clock."""
    # Q crosses as ONE E4M3 byte per element, viewed as bf16 so the tensor keeps a bf16 dtype
    # for the launcher; GEMM1a's second term rides K instead (see flash_attn_bwd.py Q_C8).
    q_pk = q.to(torch.float8_e4m3fn).view(torch.uint8).contiguous().view(torch.bfloat16)
    # K crosses as an E4M3 (value, residual) pair carrying sm*log2e, the prescale the kv-block
    # prologue used to apply itself.
    _ks = (1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else softmax_scale) * math.log2(math.e)
    return {
        "q": q_pk,
        "head_dim": q.shape[-1],
        "k": _e4m3_pair_bf16(k.float() * _ks),
        "v": v,
        "out": out,
        "lse": lse,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size": window_size,
        "sink": sink,
    }


def flash_attn_sbhd_flydsl_backward_prepared(ctx, dout):
    """The timed step: everything that depends on dO, plus every kernel launch."""
    q, k, v = ctx["q"], ctx["k"], ctx["v"]
    Sq, B, Hq = q.shape[:3]
    D = ctx["head_dim"]
    Skv, _, Hkv, _ = k.shape
    softmax_scale, window_left = _check_bwd(
        q, k, v, ctx["softmax_scale"], ctx["causal"], ctx["window_size"], ctx["sink"], Hq, D, sbhd=True
    )
    grads = flydsl_varlen_backward(
        dout.contiguous(),
        q,
        k,
        v,
        ctx["out"],
        ctx["lse"],
        B,
        Sq,
        Skv,
        Hq,
        Hkv,
        D,
        softmax_scale,
        window_left=window_left,
        sbhd=True,
        sink=ctx["sink"],
        q_pk8=2,
        k_pk8=True,
    )
    return tuple(grads[:3]) if ctx["sink"] is None else tuple(grads)
