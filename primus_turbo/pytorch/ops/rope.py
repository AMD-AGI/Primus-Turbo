###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused QKV rotary position embedding.

Public API:
    - ``fused_qkv_rope(qkv, q_freqs, k_freqs, qkv_split_arg_list) -> (q, k, v)``
    - ``fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, eps) -> (q, k, v[, gate])``
    - ``dit_qk_norm_rope(qkv, w_q, w_k, cos, sin, eps, ...) -> (q, k, v)`` and its tables,
      ``dit_qk_norm_rope_cos_sin(freqs, dtype) -> (cos, sin)``
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.rope.dit_qk_norm_rope_impl import (
    check_operands,
    dit_qk_norm_rope_backward_op,
    dit_qk_norm_rope_op,
    source_layout,
    split_streams,
)
from primus_turbo.pytorch.kernels.rope.qk_rmsnorm_rope_impl import (
    qk_rmsnorm_rope_bwd_impl,
    qk_rmsnorm_rope_fwd_impl,
    qk_rmsnorm_rope_shape_error,
)
from primus_turbo.pytorch.kernels.rope.rope_impl import (
    rope_bwd_impl,
    rope_fwd_impl,
    rope_shape_error,
)

__all__ = [
    "fused_qkv_rope",
    "fused_qkv_rmsnorm_rope",
    "dit_qk_norm_rope",
    "dit_qk_norm_rope_cos_sin",
]


class _FusedQKVRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, q_freqs, k_freqs, qkv_split_arg_list):
        # The kernels index the tables as fp32 and stride them contiguously.
        q_freqs = q_freqs.float().contiguous()
        k_freqs = k_freqs.float().contiguous()
        q, k, v = rope_fwd_impl(qkv, q_freqs, k_freqs, qkv_split_arg_list)
        ctx.save_for_backward(q_freqs, k_freqs)
        ctx.qkv_split_arg_list = qkv_split_arg_list
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        q_freqs, k_freqs = ctx.saved_tensors
        dqkv = rope_bwd_impl(dq, dk, dv, q_freqs, k_freqs, ctx.qkv_split_arg_list)
        return dqkv, None, None, None


def fused_qkv_rope(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotary position embedding over a packed QKV tensor, split on the way out.

    Args:
        qkv: ``[S, B, H, q + k + v]`` bfloat16, contiguous. ``S * B`` must be a
            multiple of the kernels' row group.
        q_freqs: ``[S, 1, 1, D]`` rotation angles for the q heads.
        k_freqs: ``[S, 1, 1, D]`` rotation angles for the k heads.
        qkv_split_arg_list: ``[q, k, v]`` widths of the last dim; k and v are one
            head each, q a whole multiple of it.

    Returns:
        ``(q, k, v)``, q and k rotated and v a plain copy.

    Raises:
        ValueError: if the tensors are not a shape the kernels accept.
    """
    why = rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list)
    if why is not None:
        raise ValueError(f"fused_qkv_rope: unsupported input ({why})")
    return _FusedQKVRoPEFunction.apply(qkv, q_freqs, k_freqs, qkv_split_arg_list)


class _FusedQKVRMSNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, eps, backend):
        if freqs is not None:
            freqs = freqs.float().contiguous()
        chosen, q, k, v, gate, saved = qk_rmsnorm_rope_fwd_impl(
            qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, eps, backend
        )
        ctx.save_for_backward(qkv, q_gamma, k_gamma, freqs, *saved)
        ctx.args = (chosen, tuple(qkv_split_arg_list), eps)
        return (q, k, v) if gate is None else (q, k, v, gate)

    @staticmethod
    def backward(ctx, dq, dk, dv, dgate=None):
        qkv, q_gamma, k_gamma, freqs, *saved = ctx.saved_tensors
        chosen, split, eps = ctx.args
        dqkv, dq_gamma, dk_gamma = qk_rmsnorm_rope_bwd_impl(
            chosen, dq, dk, dv, dgate, qkv, q_gamma, k_gamma, freqs, tuple(saved), split, eps
        )
        return dqkv, dq_gamma, dk_gamma, None, None, None, None


def fused_qkv_rmsnorm_rope(
    qkv: torch.Tensor,
    q_gamma: Optional[torch.Tensor],
    k_gamma: Optional[torch.Tensor],
    freqs: Optional[torch.Tensor],
    qkv_split_arg_list: Sequence[int],
    eps: float,
    backend: Optional[BackendType] = None,
) -> Tuple[torch.Tensor, ...]:
    """Split a packed QKV while fusing per-head Q/K RMSNorm and rotate-half RoPE.

    ``qkv`` is Megatron's packed ``[S, B, NG, W]``; per query group ``q: NPG x D | k: D | v: D`` (GPT-OSS, Qwen3),
    or with ``qkv_split_arg_list = [q, gate, k, v]`` ``q: NPG x D | gate: NPG x D | k: D | v: D`` (Megatron's
    ``attention_output_gate``, e.g. Muse Spark).

    ``q = rope(norm(q, q_gamma))`` and ``k = rope(norm(k, k_gamma))`` with ``norm(x, g) = round(x * rstd * g)``
    (``round(x * rstd)`` when both gammas are None: a weightless norm) and ``rope(n)`` from fp32 cos / sin of
    ``freqs``, rounded once per output (none when ``freqs`` is None: a NoPE layer). The normalization result and
    the inverse-RoPE gradient are rounded to BF16 in registers, preserving the numerical boundaries of the unfused
    training path.

    Backends: FlyDSL (gfx942 / gfx950; weighted norms with RoPE, no gate, head_dim 64 / 128, its row tiling) and
    Triton (any arch; every case above, any heads per group, head_dim a power of 2 >= 16). ``backend=None`` takes
    FlyDSL whenever it supports the input, else Triton; ``BackendType.FLYDSL`` / ``BackendType.TRITON`` force one.

    Args:
        qkv: ``[S, B, NG, W]`` contiguous bfloat16.
        q_gamma, k_gamma: ``[D]`` contiguous bfloat16 RMSNorm weights, or both None.
        freqs: ``[S, 1, 1, D]`` rotation angles (each RoPE pair's angle twice), or None for no RoPE.
        qkv_split_arg_list: ``[q, k, v]`` or ``[q, gate, k, v]`` widths; k and v are one head (D) each, q (and
            the gate) a whole multiple of D.
        eps: RMSNorm epsilon.
        backend: None (auto), ``BackendType.FLYDSL`` or ``BackendType.TRITON``.

    Returns:
        ``(q [S, B, NG * NPG, D], k [S, B, NG, D], v [S, B, NG, D])``, plus ``gate [S, B, NG, NPG * D]`` with a gate
        block. The Triton backend returns v and the gate as views of ``qkv`` (the backward writes their
        gradients into d(qkv)); FlyDSL returns v as a copy.

    For diffusion transformers (one q / k / v per head, interleaved RoPE, both
    token orders, a text and an image stream) see ``dit_qk_norm_rope``.
    """
    why = qk_rmsnorm_rope_shape_error(qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, backend)
    if why is not None:
        raise ValueError(f"fused_qkv_rmsnorm_rope: unsupported input ({why})")
    return _FusedQKVRMSNormRoPEFunction.apply(
        qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, float(eps), backend
    )


_V_OUT = ("copy", "none")


def dit_qk_norm_rope_cos_sin(freqs: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos / sin tables as ``dit_qk_norm_rope`` takes them: ``[rows, D]``, contiguous, in ``dtype``.

    ``freqs`` holds the angles with each RoPE pair's angle twice (Megatron's and diffusers' layout), shaped
    ``[S, 1, 1, D]`` / ``[S, D]`` (per position: rows = S) or ``[S, B, 1, D]`` (per position and batch element,
    as Megatron's Flux builds them for per-image positions: rows = S * B, row = s * B + b).
    """
    D = freqs.shape[-1]
    return (
        torch.cos(freqs).to(dtype).reshape(-1, D).contiguous(),
        torch.sin(freqs).to(dtype).reshape(-1, D).contiguous(),
    )


def _v_view(qkv: torch.Tensor, D: int, layout: str) -> torch.Tensor:
    m = source_layout(qkv, D, layout)
    if qkv.dim() == 4:
        return qkv[..., m["VO"] : m["VO"] + D]
    return qkv[..., m["VO"] : m["VO"] + m["H"] * D].unflatten(-1, (m["H"], D))


class _DitQkNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, bias, txt_bias, eps, layout, rope,
                round_normed, v_out, txt_seq_len):  # fmt: skip
        v_copy = v_out == "copy"
        q, k, v = dit_qk_norm_rope_op(
            qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, bias, txt_bias, eps, layout, rope, round_normed,
            v_copy, txt_seq_len,
        )  # fmt: skip
        if not v_copy:  # one stream, or both in one joint buffer: v is a view of qkv's v columns
            v = _v_view(qkv, w_q.shape[0], layout)
        ctx.two_streams = txt_qkv is not None or txt_seq_len > 0
        ctx.args = (eps, layout, rope, round_normed, txt_seq_len)
        ctx.save_for_backward(qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, bias, txt_bias)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, bias, txt_bias = ctx.saved_tensors
        eps, layout, rope, round_normed, txt_seq_len = ctx.args
        dq, dk, dv = (
            torch.zeros_like(dq if dq is not None else dk) if g is None else g for g in (dq, dk, dv)
        )
        bias_grad = bias is not None and (ctx.needs_input_grad[8] or ctx.needs_input_grad[9])
        out = dit_qk_norm_rope_backward_op(
            dq, dk, dv, qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, bias, txt_bias, eps, layout, rope,
            round_normed, bias_grad, txt_seq_len,
        )  # fmt: skip
        D = w_q.shape[0]
        red = out[-1]
        d_txt, d_qkv = (out[0], out[1]) if txt_qkv is not None else (None, out[0])
        r_txt, r_img = (red[0], red[1]) if ctx.two_streams else (None, red[0])

        def bias_grad_of(r, b):
            return r[2 * D :].view(b.shape) if bias_grad and b is not None else None

        return (
            d_qkv,
            r_img[:D],
            r_img[D : 2 * D],
            None,
            None,
            d_txt,
            None if r_txt is None else r_txt[:D],
            None if r_txt is None else r_txt[D : 2 * D],
            bias_grad_of(r_img, bias),
            None if r_txt is None else bias_grad_of(r_txt, txt_bias),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dit_qk_norm_rope(
    qkv: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    *,
    layout: str = "sbhd",
    txt_qkv: Optional[torch.Tensor] = None,
    w_txt_q: Optional[torch.Tensor] = None,
    w_txt_k: Optional[torch.Tensor] = None,
    txt_seq_len: int = 0,
    rope: str = "interleaved",
    round_normed: bool = True,
    v_out: str = "copy",
    bias: Optional[torch.Tensor] = None,
    txt_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v from the QKV GEMM output(s): q and k RMS-normalised over head_dim and rotated, v copied.

    Diffusion transformers (Flux, SD3 / MMDiT) RMS-normalise q and k per head with learned weights, rotate them,
    and, in their dual-stream blocks, concatenate a text and an image stream along the sequence. This op does
    that from the QKV GEMM output(s), read in place, to q / k / v in attention's layout:

        q, k, v = dit_qk_norm_rope(qkv, w_q, w_k, cos, sin, eps, layout="sbhd")                  # Megatron
        q, k, v = dit_qk_norm_rope(img_qkv, w_q, w_k, cos, sin, eps, layout="bshd",
                                   txt_qkv=txt_qkv, w_txt_q=w_txt_q, w_txt_k=w_txt_k)            # diffusers, dual

    How it relates to ``fused_qkv_rmsnorm_rope`` (#556): both RMS-normalise q and k of a packed QKV and apply
    RoPE, forward and backward. That op is the LLM GQA one: ``[S, B, NG, (NPG + 2) * D]`` groups, half-split RoPE
    from fp32 ``freqs``, FlyDSL on gfx950. This one is for DiTs: one q / k / v per head, both token orders,
    interleaved (Flux) or half-split RoPE from cos / sin tables, a text and an image stream written into one
    joint sequence, the QKV bias added and its gradient fused, Triton on every arch and a FlyDSL backward on
    gfx950.

    Args:
        qkv: the (image) stream's QKV GEMM output. ``layout="sbhd"``: ``[S, B, H, 3*D]`` (Megatron, per head
            q | k | v) or ``[S, B, 3*H*D]``; ``"bshd"``: ``[B, S, 3*H*D]`` (diffusers, q heads | k heads | v heads)
            or ``[B, S, H, 3*D]``. Read in place at its strides (unit last stride, one uniform token stride).
            With ``txt_seq_len``: both streams' output in one ``[S_txt + S, B, ..]`` buffer, text first.
        w_q, w_k: RMSNorm weights ``[D]``, qkv's dtype.
        cos, sin: ``[S_out, D]`` (per position) or ``[S_out * B, D]`` (per position and batch element, row
            ``s * B + b``), each RoPE pair's angle twice; see ``dit_qk_norm_rope_cos_sin``.
        eps: RMSNorm epsilon.
        layout: token order of qkv and of the outputs, ``"sbhd"`` or ``"bshd"``.
        txt_qkv, w_txt_q, w_txt_k: a text stream (dual-stream blocks), same form as qkv; its tokens come first in
            the joint sequence ``S_out = S_txt + S``, in one launch each way.
        txt_seq_len: ``S_txt`` when ``qkv`` holds both streams (``layout="sbhd"``, ``txt_qkv`` None, the text
            weights given): its first ``S_txt`` positions are the text stream. The backward returns one gradient
            for the whole buffer. 0 (default): ``qkv`` is one stream.
        rope: ``"interleaved"`` pairs (2i, 2i+1) (Flux, Megatron's ``rotary_interleaved``); ``"half"`` pairs
            (i, i + D/2).
        round_normed: round the normalised, weighted q / k to qkv's dtype before RoPE, and the inverse-RoPE
            gradient before the RMSNorm backward. True is what an RMSNorm module followed by RoPE computes
            (diffusers, #556); False keeps both in fp32 (one rounding per output; closer to fp64). With a bias,
            True also rounds ``qkv + bias`` before the norm, as a separate bias add does.
        v_out: ``"copy"`` writes v into a fresh ``[.., .., H, D]`` tensor; ``"none"`` returns v as a strided view of
            the source instead (no copy), and its gradient is written into d(qkv) by this op's backward. With a
            text stream the view spans both streams as one joint sequence, so both must come in one buffer
            (``txt_seq_len``). Not with a bias (the source does not hold it).
        bias, txt_bias: the QKV GEMMs' biases, ``[3*H*D]`` in the source's column order, qkv's dtype; the GEMMs
            then run without them. The op adds them to q, k and v and returns their gradients (the column sums
            of d(qkv)), summed in the same pass. With a text stream, both or neither.

    Returns:
        q, k, v: ``[S_out, B, H, D]`` (``"sbhd"``) or ``[B, S_out, H, D]`` (``"bshd"``).

    The backward returns d(qkv) in qkv's layout (dv in its v columns), dw_q / dw_k per stream and, with
    ``bias``, the bias gradients. The weight and bias gradients are fp32 partial sums reduced in a fixed order:
    the result is deterministic. Kernels: Triton forward; backward FlyDSL on gfx950 (bf16, D = 128, interleaved
    RoPE, 16 to 64 heads, each stream's token count a multiple of 256), Triton otherwise.
    """
    if v_out not in _V_OUT:
        raise ValueError(f"dit_qk_norm_rope: v_out must be one of {_V_OUT}, got {v_out!r}")
    if v_out == "none" and txt_qkv is not None:
        raise ValueError(
            "dit_qk_norm_rope: v_out='none' with a text stream needs both streams in one buffer: pass the joint "
            "[S_txt + S, B, ..] QKV output as qkv with txt_seq_len=S_txt, or use v_out='copy'"
        )
    if v_out == "none" and bias is not None:
        raise ValueError(
            "dit_qk_norm_rope: v_out='none' returns v as a view of qkv, which has no bias; use 'copy'"
        )
    txt_seq_len = int(txt_seq_len)
    txt, img = split_streams(qkv, txt_qkv, txt_seq_len, layout)
    check_operands(img, w_q, w_k, cos, sin, txt, w_txt_q, w_txt_k, layout, rope, bias, txt_bias)
    return _DitQkNormRope.apply(
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, bias, txt_bias, float(eps), layout, rope,
        bool(round_normed), v_out, txt_seq_len,
    )  # fmt: skip
