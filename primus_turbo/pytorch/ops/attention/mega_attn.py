###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaAttn: Ulysses attention with the projections, the all-to-alls and attention merged into as few
kernels as possible (FlyDSL, gfx950, single node).

``mega_attn_func``: causal self-attention whose forward runs as a persistent kernel (one CTA per CU
claiming work items), bitwise equal to ``flash_attn_func``'s FlyDSL sbhd path; FlyDSL sbhd backward.

``ulysses_qkv_attn_func``: QKV projection + Ulysses all-to-all + causal attention forward in one kernel
(``flydsl.mega_attn.qkv_attn_kernel``), bitwise equal to ``ulysses_qkv_gemm_a2a`` followed by
``flash_attn_func``; backward = FlyDSL attention backward, then the QKV A2AGemm.

``ulysses_attn_layer_func``: the same kernel continued by the reverse all-to-all and the output
projection + residual (``ulysses_a2a_o_gemm``); backward = o_proj GemmA2A, attention backward, QKV A2AGemm.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist

from primus_turbo.flydsl.mega_attn.attn_fwd_persistent import attn_fwd_persistent
from primus_turbo.flydsl.mega_attn.qkv_attn_kernel import (
    qkv_attn_bank_bytes,
    ulysses_qkv_attn,
)
from primus_turbo.flydsl.ulysses.gemm_a2a_kernel import gemm_a2a_bank_bytes
from primus_turbo.flydsl.ulysses.workspace import get_ulysses_workspace
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
)
from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
    _check,
    _check_seq_chunk,
    _o_gemm_backward,
    _qkv_backward,
    _qkv_rank_block,
)

__all__ = ["mega_attn_func", "ulysses_qkv_attn_func", "ulysses_attn_layer_func"]


def _attn_backward(dout, q, k, v, out, lse, deterministic=False):
    """FlyDSL sbhd backward; ``lse`` as the forward kernel writes it, ``[b * s, hq]``."""
    Sq, B, Hq, D = q.shape
    return flash_attn_sbhd_flydsl_backward_impl(
        dout,
        q,
        k,
        v,
        out,
        lse.view(B, Sq, Hq).permute(0, 2, 1),
        softmax_scale=D**-0.5,
        causal=True,
        deterministic=deterministic,
    )


class MegaAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, lse = attn_fwd_persistent(q, k, v)
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):
        return _attn_backward(dout, *ctx.saved_tensors)


def mega_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q`` [s, b, hq, d], ``k`` / ``v`` [s, b, hkv, d] contiguous bf16 (sbhd), causal, scale 1/sqrt(d)."""
    return MegaAttnFunction.apply(q, k, v)


class UlyssesQKVAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_qkv, hq, hkv, d, group, qkv_format, seq_chunk, deterministic):
        n = group.size()
        s_local, hidden = (x.shape[0], x.shape[2]) if qkv_format == "sbhd" else (x.shape[1], x.shape[2])
        seq = s_local * n
        a = x.reshape(s_local, hidden).contiguous()
        w = w_qkv.contiguous()
        q, k, v = (torch.empty(seq, 1, h // n, d, dtype=x.dtype, device=x.device) for h in (hq, hkv, hkv))
        o = torch.empty_like(q)
        lse = torch.empty(seq, hq // n, dtype=torch.float32, device=x.device)
        workspace = get_ulysses_workspace(group, gemm_a2a_bank_bytes(s_local, w.shape[0], 256))
        G, nq, _, _ = _qkv_rank_block(hq, hkv, d, n, "interleaved")
        ulysses_qkv_attn(
            a,
            w,
            q,
            k,
            v,
            o,
            lse,
            workspace,
            head_dim=d,
            groups=G,
            nq=nq,
            seq_local=s_local,
            seq_chunk=seq_chunk,
        )
        ctx.save_for_backward(x, w_qkv, q, k, v, o, lse)
        ctx.hq, ctx.hkv, ctx.d, ctx.group = hq, hkv, d, group
        ctx.qkv_format, ctx.seq_chunk, ctx.s_local = qkv_format, seq_chunk, s_local
        ctx.deterministic = deterministic
        return o if qkv_format == "sbhd" else o.view(1, seq, hq // n, d)

    @staticmethod
    def backward(ctx, do):
        x, w_qkv, q, k, v, o, lse = ctx.saved_tensors
        do = do.reshape(o.shape).contiguous()
        dq, dk, dv = _attn_backward(do, q, k, v, o, lse, ctx.deterministic)
        qkv_ctx = SimpleNamespace(
            saved_tensors=(x, w_qkv),
            needs_input_grad=ctx.needs_input_grad,
            hq=ctx.hq,
            hkv=ctx.hkv,
            d=ctx.d,
            group=ctx.group,
            weight_layout="interleaved",
            s_local=ctx.s_local,
            b=1,
            qkv_format=ctx.qkv_format,
            seq_chunk=ctx.seq_chunk,
        )
        dx, dw = _qkv_backward(qkv_ctx, dq, dk, dv)
        return dx, dw, None, None, None, None, None, None, None


def _qkv_ctx(ctx, x, w_qkv):
    return SimpleNamespace(
        saved_tensors=(x, w_qkv),
        needs_input_grad=ctx.needs_input_grad,
        hq=ctx.hq,
        hkv=ctx.hkv,
        d=ctx.d,
        group=ctx.group,
        weight_layout="interleaved",
        s_local=ctx.s_local,
        b=1,
        qkv_format=ctx.qkv_format,
        seq_chunk=ctx.seq_chunk,
    )


class UlyssesAttnLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_qkv, w_o, residual, hq, hkv, d, group, qkv_format, seq_chunk, deterministic):
        n = group.size()
        s_local, hidden = (x.shape[0], x.shape[2]) if qkv_format == "sbhd" else (x.shape[1], x.shape[2])
        seq = s_local * n
        a = x.reshape(s_local, hidden).contiguous()
        w = w_qkv.contiguous()
        q, k, v = (torch.empty(seq, 1, h // n, d, dtype=x.dtype, device=x.device) for h in (hq, hkv, hkv))
        o = torch.empty_like(q)
        lse = torch.empty(seq, hq // n, dtype=torch.float32, device=x.device)
        N_o = w_o.shape[0]
        y = torch.empty(s_local, N_o, dtype=x.dtype, device=x.device)
        o_gathered = (
            torch.empty(s_local, hq * d, dtype=x.dtype, device=x.device) if w_o.requires_grad else None
        )
        workspace = get_ulysses_workspace(group, qkv_attn_bank_bytes(s_local, w.shape[0], hq * d))
        G, nq, _, _ = _qkv_rank_block(hq, hkv, d, n, "interleaved")
        ulysses_qkv_attn(
            a,
            w,
            q,
            k,
            v,
            o,
            lse,
            workspace,
            head_dim=d,
            groups=G,
            nq=nq,
            seq_local=s_local,
            seq_chunk=seq_chunk,
            w_o=w_o.contiguous(),
            residual=residual.reshape(s_local, N_o).contiguous(),
            y=y,
            o_gathered=o_gathered,
        )
        ctx.save_for_backward(x, w_qkv, w_o, q, k, v, o, lse, o_gathered)
        ctx.hq, ctx.hkv, ctx.d, ctx.group = hq, hkv, d, group
        ctx.qkv_format, ctx.seq_chunk, ctx.s_local = qkv_format, seq_chunk, s_local
        ctx.deterministic = deterministic
        return y.view(residual.shape)

    @staticmethod
    def backward(ctx, dy):
        x, w_qkv, w_o, q, k, v, o, lse, o_gathered = ctx.saved_tensors
        n = ctx.group.size()
        dy2d = dy.reshape(ctx.s_local, w_o.shape[0]).contiguous()
        do, dw_o = _o_gemm_backward(
            dy2d,
            w_o,
            o_gathered,
            ctx.group,
            ctx.hq // n,
            ctx.d,
            ctx.s_local,
            1,
            ctx.qkv_format,
            ctx.seq_chunk,
            True,
            ctx.needs_input_grad[2],
        )
        dq, dk, dv = _attn_backward(do.reshape(o.shape).contiguous(), q, k, v, o, lse, ctx.deterministic)
        dx, dw = _qkv_backward(_qkv_ctx(ctx, x, w_qkv), dq, dk, dv)
        d_residual = dy if ctx.needs_input_grad[3] else None
        return dx, dw, dw_o, d_residual, None, None, None, None, None, None, None


def ulysses_qkv_attn_func(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group: dist.ProcessGroup,
    qkv_format: str = "sbhd",
    seq_chunk: int = 0,
    deterministic: bool = False,
) -> torch.Tensor:
    """``flash_attn_func(*ulysses_qkv_gemm_a2a(x, w_qkv, ..., "interleaved", seq_chunk), causal=True,
    deterministic=deterministic)`` with the forward in one kernel. Batch 1, interleaved (Megatron) QKV
    weight rows.

    Returns:
        the attention output for the local heads over the full sequence, ``[s, 1, hq/n, d]`` (sbhd)
        or ``[1, s, hq/n, d]`` (bshd), sequence in order.
    """
    _check(group, num_q_heads, num_kv_heads, head_dim, qkv_format, "interleaved")
    batch = x.shape[1] if qkv_format == "sbhd" else x.shape[0]
    assert batch == 1, "ulysses_qkv_attn_func supports batch 1"
    _check_seq_chunk(seq_chunk, x.shape[0 if qkv_format == "sbhd" else 1])
    return UlyssesQKVAttnFunction.apply(
        x, w_qkv, num_q_heads, num_kv_heads, head_dim, group, qkv_format, seq_chunk, deterministic
    )


def ulysses_attn_layer_func(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    residual: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group: dist.ProcessGroup,
    qkv_format: str = "sbhd",
    seq_chunk: int = 0,
    deterministic: bool = False,
) -> torch.Tensor:
    """``ulysses_a2a_o_gemm(ulysses_qkv_attn_func(x, w_qkv, ...), w_o, residual, ...)`` with the forward in
    one kernel. Batch 1, interleaved QKV weight rows, ``seq_chunk`` a multiple of 128 (or 0 with
    ``s/n % 128 == 0``), ``hidden % 256 == 0``.

    Returns:
        ``O_gathered @ w_o^T + residual``, shaped like ``residual`` (``[s/n, 1, hidden]`` sbhd,
        ``[1, s/n, hidden]`` bshd).
    """
    _check(group, num_q_heads, num_kv_heads, head_dim, qkv_format, "interleaved")
    s_local = x.shape[0 if qkv_format == "sbhd" else 1]
    batch = x.shape[1] if qkv_format == "sbhd" else x.shape[0]
    assert batch == 1, "ulysses_attn_layer_func supports batch 1"
    _check_seq_chunk(seq_chunk, s_local)
    assert (seq_chunk or s_local) % 128 == 0, "the o_proj push needs whole 128-token q blocks per chunk"
    assert residual.shape == x.shape[:2] + (w_o.shape[0],)
    return UlyssesAttnLayerFunction.apply(
        x,
        w_qkv,
        w_o,
        residual,
        num_q_heads,
        num_kv_heads,
        head_dim,
        group,
        qkv_format,
        seq_chunk,
        deterministic,
    )
