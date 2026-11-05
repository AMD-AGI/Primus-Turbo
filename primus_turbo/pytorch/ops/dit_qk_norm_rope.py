###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DiT q / k RMSNorm + RoPE: between the QKV GEMM and attention, one kernel each way.

Diffusion transformers (Flux, SD3 / MMDiT) RMS-normalise q and k per head with learned weights, rotate them, and,
in their dual-stream blocks, concatenate a text and an image stream along the sequence. This op does that from
the QKV GEMM output(s), read in place, to q / k / v in attention's layout:

    q, k, v = dit_qk_norm_rope(qkv, w_q, w_k, cos, sin, eps, layout="sbhd")                  # Megatron
    q, k, v = dit_qk_norm_rope(img_qkv, w_q, w_k, cos, sin, eps, layout="bshd",
                               txt_qkv=txt_qkv, w_txt_q=w_txt_q, w_txt_k=w_txt_k)            # diffusers, dual

How it relates to ``fused_qkv_rmsnorm_rope`` (#556, ``primus_turbo.pytorch.ops.rope``): both RMS-normalise q and
k of a packed QKV and apply RoPE, forward and backward. That op is the LLM GQA one: ``[S, B, NG, (NPG + 2) * D]``
groups, half-split RoPE from fp32 ``freqs``, FlyDSL on gfx950. This one is for DiTs: one q / k / v per head,
both token orders, interleaved (Flux) or half-split RoPE from cos / sin tables, a text and an image stream
written into one joint sequence, an optional fused QKV-bias gradient, Triton on every arch and a FlyDSL backward
on gfx950.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.kernels.rope.dit_qk_norm_rope_impl import (
    check_operands,
    dit_qk_norm_rope_backward_op,
    dit_qk_norm_rope_op,
    source_layout,
)

__all__ = ["dit_qk_norm_rope", "dit_qk_norm_rope_cos_sin"]

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
                round_normed, v_out):  # fmt: skip
        v_copy = v_out == "copy"
        q, k, v = dit_qk_norm_rope_op(
            qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, eps, layout, rope, round_normed, v_copy
        )
        if not v_copy:
            v = _v_view(qkv, w_q.shape[0], layout)
        ctx.has_txt = txt_qkv is not None
        ctx.bias_grad = bias is not None
        ctx.bias_meta = [(b.shape, b.dtype) if b is not None else None for b in (bias, txt_bias)]
        ctx.args = (eps, layout, rope, round_normed)
        ctx.save_for_backward(qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k = ctx.saved_tensors
        eps, layout, rope, round_normed = ctx.args
        dq, dk, dv = (
            torch.zeros_like(dq if dq is not None else dk) if g is None else g for g in (dq, dk, dv)
        )
        out = dit_qk_norm_rope_backward_op(
            dq, dk, dv, qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, eps, layout, rope, round_normed,
            ctx.bias_grad,
        )  # fmt: skip
        D = w_q.shape[0]
        red = out[-1]
        if ctx.has_txt:
            d_txt, d_qkv = out[0], out[1]
            r_txt, r_img = red[0], red[1]
        else:
            d_txt, d_qkv = None, out[0]
            r_txt, r_img = None, red[0]

        def bias_of(r, meta):
            return None if meta is None or r is None else r[2 * D :].view(meta[0]).to(meta[1])

        db = bias_of(r_img, ctx.bias_meta[0]) if ctx.bias_grad else None
        db_txt = bias_of(r_txt, ctx.bias_meta[1]) if ctx.bias_grad else None
        return (
            d_qkv,
            r_img[:D],
            r_img[D : 2 * D],
            None,
            None,
            d_txt,
            None if r_txt is None else r_txt[:D],
            None if r_txt is None else r_txt[D : 2 * D],
            db,
            db_txt,
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
    rope: str = "interleaved",
    round_normed: bool = True,
    v_out: str = "copy",
    bias: Optional[torch.Tensor] = None,
    txt_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v from the QKV GEMM output(s): q and k RMS-normalised over head_dim and rotated, v copied.

    Args:
        qkv: the (image) stream's QKV GEMM output. ``layout="sbhd"``: ``[S, B, H, 3*D]`` (Megatron, per head
            q | k | v) or ``[S, B, 3*H*D]``; ``"bshd"``: ``[B, S, 3*H*D]`` (diffusers, q heads | k heads | v heads)
            or ``[B, S, H, 3*D]``. Read in place at its strides (unit last stride, one uniform token stride).
        w_q, w_k: RMSNorm weights ``[D]``, qkv's dtype.
        cos, sin: ``[S_out, D]`` (per position) or ``[S_out * B, D]`` (per position and batch element, row
            ``s * B + b``), each RoPE pair's angle twice; see ``dit_qk_norm_rope_cos_sin``.
        eps: RMSNorm epsilon.
        layout: token order of qkv and of the outputs, ``"sbhd"`` or ``"bshd"``.
        txt_qkv, w_txt_q, w_txt_k: a text stream (dual-stream blocks), same form as qkv; its tokens come first in
            the joint sequence ``S_out = S_txt + S``, in one launch each way.
        rope: ``"interleaved"`` pairs (2i, 2i+1) (Flux, Megatron's ``rotary_interleaved``); ``"half"`` pairs
            (i, i + D/2).
        round_normed: round the normalised, weighted q / k to qkv's dtype before RoPE, and the inverse-RoPE
            gradient before the RMSNorm backward. True is what an RMSNorm module followed by RoPE computes
            (diffusers, #556); False keeps both in fp32 (one rounding per output; closer to fp64).
        v_out: ``"copy"`` writes v into a fresh ``[.., .., H, D]`` tensor; ``"none"`` (one stream only) returns v as
            a strided view of qkv, and its gradient is written into d(qkv) by this op's backward.
        bias, txt_bias: the QKV GEMMs' biases, when the GEMM adds them without differentiating them (e.g. a
            detached bias). The backward then returns their gradients (the column sums of d(qkv)), summed in the
            same pass.

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
        raise ValueError("dit_qk_norm_rope: v_out='none' needs one stream (the joint v is a concatenation)")
    if txt_bias is not None and (bias is None or txt_qkv is None):
        raise ValueError("dit_qk_norm_rope: txt_bias needs txt_qkv and bias")
    if bias is not None and txt_qkv is not None and txt_bias is None:
        raise ValueError("dit_qk_norm_rope: with a text stream, pass both biases or neither")
    metas, D, _, _ = check_operands(
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, layout, rope, bias_grad=bias is not None
    )
    for name, b, m in (("bias", bias, metas[-1]), ("txt_bias", txt_bias, metas[0])):
        if b is not None and b.numel() != m["ROW_W"]:
            raise ValueError(f"dit_qk_norm_rope: {name} must have {m['ROW_W']} elements, got {list(b.shape)}")
    return _DitQkNormRope.apply(
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k, bias, txt_bias, float(eps), layout, rope,
        bool(round_normed), v_out,
    )  # fmt: skip
