###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses causal attention with the all-to-alls fused around it (FlyDSL, single node, batch 1).

``ulysses_mega_attn`` replaces ``all-to-all -> flash_attn -> all-to-all`` between the projections: it
takes this rank's tokens with every head and returns the attention output for those tokens with every
head. The forward is one kernel (pull q / k / v, attention, push O), and so is the backward (pull dO,
the FlyDSL SBHD backward on this rank's heads, push dq / dk / dv to the token owners). A shape the fused
backward does not cover runs it as an in-kernel all-to-all, the FlyDSL backward and a second all-to-all.
MLA head dims (q / k 192, v 128) run the gfx950 MLA forward tile and, unfused, its backward.

Sequence split: with ``seq_chunk`` cs, global chunk ``c`` (cs tokens) lives on rank ``c % n``;
``seq_chunk=0`` is the contiguous split (rank r holds the r-th chunk of ``S / n`` tokens). A striped
split balances causal attention across ranks and lets the attention start before the exchange ends.
"""

from typing import Optional

import torch
import torch.distributed as dist

from primus_turbo.flydsl.mega_attn.a2a_attn_bwd_kernel import _plan as _bwd_plan
from primus_turbo.flydsl.mega_attn.a2a_attn_bwd_kernel import (
    a2a_attn_bwd_bank_bytes,
    ulysses_a2a_attn_bwd,
)
from primus_turbo.flydsl.mega_attn.a2a_attn_kernel import (
    a2a_attn_fwd_bank_bytes,
    is_mla,
    ulysses_a2a_attn_fwd,
)
from primus_turbo.flydsl.mega_attn.a2a_kernel import (
    a2a_bank_bytes,
    ulysses_a2a_heads_to_seq,
    ulysses_a2a_seq_to_heads,
)
from primus_turbo.flydsl.ulysses.workspace import get_ulysses_workspace
from primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx950_mla_impl import (
    flash_attn_flydsl_gfx950_mla_backward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
)

__all__ = ["ulysses_mega_attn"]


def _bank_bytes(M, n, hq_r, hkv_r, d, d_v):
    banks = [
        a2a_attn_fwd_bank_bytes(M, n, hq_r, hkv_r, d, d_v),
        a2a_bank_bytes(M, n, (hq_r, hkv_r, hkv_r), (d, d, d_v)),
    ]
    if d == d_v:
        banks.append(a2a_attn_bwd_bank_bytes(M, n, hq_r, hkv_r, d))
    return max(banks)


def _fused_bwd_ok(S, hq_r, hkv_r, d, d_v, cs):
    """Whether the one-kernel backward runs this shape: the stock plan must be the a16 dQ image with
    256-row bands (dense causal, ``S`` a multiple of the q-split tile)."""
    if d != d_v or cs % 128 or S % 256:
        return False
    try:
        _bwd_plan(S, hq_r, hkv_r, d)
    except AssertionError:
        return False
    return True


class UlyssesMegaAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, seq_chunk, num_pull, softmax_scale):
        n = group.size()
        M, hq, d = q.shape
        hkv, d_v = k.shape[1], v.shape[-1]
        hq_r, hkv_r = hq // n, hkv // n
        S = M * n
        ws = get_ulysses_workspace(group, _bank_bytes(M, n, hq_r, hkv_r, d, d_v), name="mega_attn")
        q_h = q.new_empty(S, hq_r, d)
        k_h = q.new_empty(S, hkv_r, d)
        v_h = q.new_empty(S, hkv_r, d_v)
        o_h = q.new_empty(S, hq_r, d_v)
        lse = q.new_empty(S, hq_r, dtype=torch.float32)
        out = q.new_empty(M, hq, d_v)
        ulysses_a2a_attn_fwd(
            q,
            k,
            v,
            q_h,
            k_h,
            v_h,
            o_h,
            lse,
            out,
            ws,
            seq_chunk=seq_chunk,
            num_pull=num_pull,
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(q_h, k_h, v_h, o_h, lse)
        ctx.group, ctx.seq_chunk, ctx.softmax_scale = group, seq_chunk, softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q_h, k_h, v_h, o_h, lse = ctx.saved_tensors
        group, cs = ctx.group, ctx.seq_chunk
        n = group.size()
        S, hq_r, d = q_h.shape
        hkv_r, d_v = k_h.shape[1], v_h.shape[-1]
        M = S // n
        ws = get_ulysses_workspace(group, _bank_bytes(M, n, hq_r, hkv_r, d, d_v), name="mega_attn")
        dq = q_h.new_empty(M, n * hq_r, d)
        dk = q_h.new_empty(M, n * hkv_r, d)
        dv = q_h.new_empty(M, n * hkv_r, d_v)
        grads = (dq, dk, dv, None, None, None, None)
        if _fused_bwd_ok(S, hq_r, hkv_r, d, d_v, cs or M):
            ulysses_a2a_attn_bwd(dout.contiguous(), q_h, k_h, v_h, o_h, lse, dq, dk, dv, ws, seq_chunk=cs)
            return grads
        do_h = torch.empty_like(o_h)
        ulysses_a2a_seq_to_heads([dout.contiguous()], [do_h], ws, seq_chunk=cs)
        if is_mla(d, d_v):
            dq_h, dk_h, dv_h = flash_attn_flydsl_gfx950_mla_backward_impl(
                do_h[:, None], q_h[:, None], k_h[:, None], v_h[:, None], o_h[:, None], lse, ctx.softmax_scale
            )
        else:
            dq_h, dk_h, dv_h = flash_attn_sbhd_flydsl_backward_impl(
                do_h[:, None],
                q_h[:, None],
                k_h[:, None],
                v_h[:, None],
                o_h[:, None],
                lse.view(1, S, hq_r).permute(0, 2, 1),
                causal=True,
            )
        ulysses_a2a_heads_to_seq(
            [dq_h.reshape(S, hq_r, d), dk_h.reshape(S, hkv_r, d), dv_h.reshape(S, hkv_r, d_v)],
            [dq, dk, dv],
            ws,
            seq_chunk=cs,
        )
        return grads


def ulysses_mega_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    seq_chunk: int = 0,
    num_pull: int = 32,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Causal attention over the Ulysses group's sequence for this rank's tokens.

    Args:
        q: ``[M, Hq, D]`` or SBHD ``[M, 1, Hq, D]`` bf16, this rank's tokens with every head.
        k, v: ``[M, Hkv, D]`` / ``[M, Hkv, D_v]`` (or SBHD), like ``q``. ``D_v != D`` only for MLA
            (``D`` 192, ``D_v`` 128, ``Hq == Hkv``).
        group: the Ulysses group (one node); ``Hq`` and ``Hkv`` must divide by its size, and the
            local ``Hq / Hkv`` group must be even (except MLA).
        seq_chunk: the striped split's chunk (a multiple of 128 dividing ``M``, of 256 for MLA);
            0 is contiguous.
        num_pull: forward blocks that stage and pull q / k / v before joining the attention.
        softmax_scale: MLA only (e.g. YaRN's ``mscale ** 2 / sqrt(D)``); default ``D ** -0.5``.

    Returns:
        The attention output for this rank's tokens, ``[M, Hq, D_v]`` (or SBHD).
    """
    sbhd = q.dim() == 4
    if sbhd:
        assert q.shape[1] == 1, "batch 1 only"
        q, k, v = q[:, 0], k[:, 0], v[:, 0]
    if softmax_scale is not None and abs(softmax_scale * q.shape[-1] ** 0.5 - 1.0) <= 1e-12:
        softmax_scale = None
    out = UlyssesMegaAttnFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), group, seq_chunk, num_pull, softmax_scale
    )
    return out[:, None] if sbhd else out
