###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses causal attention with the all-to-alls fused around it (FlyDSL, single node, batch 1).

``ulysses_mega_attn`` replaces ``all-to-all -> flash_attn -> all-to-all`` between the projections: it
takes this rank's tokens with every head and returns the attention output for those tokens with every
head. The forward is one kernel (pull q / k / v, attention, push O). The backward pulls dO with an
in-kernel all-to-all, runs the FlyDSL SBHD backward on this rank's heads and pushes dq / dk / dv back
to the token owners with a second one.

Sequence split: with ``seq_chunk`` cs, global chunk ``c`` (cs tokens) lives on rank ``c % n``;
``seq_chunk=0`` is the contiguous split (rank r holds the r-th chunk of ``S / n`` tokens). A striped
split balances causal attention across ranks and lets the attention start before the exchange ends.
"""

import torch
import torch.distributed as dist

from primus_turbo.flydsl.mega_attn.a2a_attn_kernel import (
    a2a_attn_fwd_bank_bytes,
    ulysses_a2a_attn_fwd,
)
from primus_turbo.flydsl.mega_attn.a2a_kernel import (
    a2a_bank_bytes,
    ulysses_a2a_heads_to_seq,
    ulysses_a2a_seq_to_heads,
)
from primus_turbo.flydsl.ulysses.workspace import get_ulysses_workspace
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
)

__all__ = ["ulysses_mega_attn"]


def _bank_bytes(M, n, hq_r, hkv_r, d):
    return max(
        a2a_attn_fwd_bank_bytes(M, n, hq_r, hkv_r, d),
        a2a_bank_bytes(M, n, (hq_r, hkv_r, hkv_r), d),
    )


class UlyssesMegaAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, seq_chunk, num_pull):
        n = group.size()
        M, hq, d = q.shape
        hkv = k.shape[1]
        hq_r, hkv_r = hq // n, hkv // n
        S = M * n
        ws = get_ulysses_workspace(group, _bank_bytes(M, n, hq_r, hkv_r, d), name="mega_attn")
        q_h = q.new_empty(S, hq_r, d)
        k_h = q.new_empty(S, hkv_r, d)
        v_h = q.new_empty(S, hkv_r, d)
        o_h = q.new_empty(S, hq_r, d)
        lse = q.new_empty(S, hq_r, dtype=torch.float32)
        out = torch.empty_like(q)
        ulysses_a2a_attn_fwd(
            q, k, v, q_h, k_h, v_h, o_h, lse, out, ws, seq_chunk=seq_chunk, num_pull=num_pull
        )
        ctx.save_for_backward(q_h, k_h, v_h, o_h, lse)
        ctx.group, ctx.seq_chunk = group, seq_chunk
        return out

    @staticmethod
    def backward(ctx, dout):
        q_h, k_h, v_h, o_h, lse = ctx.saved_tensors
        group, cs = ctx.group, ctx.seq_chunk
        n = group.size()
        S, hq_r, d = q_h.shape
        hkv_r = k_h.shape[1]
        M = S // n
        ws = get_ulysses_workspace(group, _bank_bytes(M, n, hq_r, hkv_r, d), name="mega_attn")
        do_h = torch.empty_like(o_h)
        ulysses_a2a_seq_to_heads([dout.contiguous()], [do_h], ws, seq_chunk=cs)
        dq_h, dk_h, dv_h = flash_attn_sbhd_flydsl_backward_impl(
            do_h[:, None],
            q_h[:, None],
            k_h[:, None],
            v_h[:, None],
            o_h[:, None],
            lse.view(1, S, hq_r).permute(0, 2, 1),
            causal=True,
        )
        dq = q_h.new_empty(M, n * hq_r, d)
        dk = q_h.new_empty(M, n * hkv_r, d)
        dv = q_h.new_empty(M, n * hkv_r, d)
        ulysses_a2a_heads_to_seq(
            [dq_h.view(S, hq_r, d), dk_h.view(S, hkv_r, d), dv_h.view(S, hkv_r, d)],
            [dq, dk, dv],
            ws,
            seq_chunk=cs,
        )
        return dq, dk, dv, None, None, None


def ulysses_mega_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    seq_chunk: int = 0,
    num_pull: int = 32,
) -> torch.Tensor:
    """Causal attention over the Ulysses group's sequence for this rank's tokens.

    Args:
        q: ``[M, Hq, D]`` or SBHD ``[M, 1, Hq, D]`` bf16, this rank's tokens with every head.
        k, v: ``[M, Hkv, D]`` (or SBHD), like ``q``.
        group: the Ulysses group (one node); ``Hq`` and ``Hkv`` must divide by its size, and the
            local ``Hq / Hkv`` group must be even.
        seq_chunk: the striped split's chunk (a multiple of 128 dividing ``M``); 0 is contiguous.
        num_pull: forward blocks that stage and pull q / k / v before joining the attention.

    Returns:
        The attention output for this rank's tokens, shaped like ``q`` (softmax scale ``D ** -0.5``).
    """
    sbhd = q.dim() == 4
    if sbhd:
        assert q.shape[1] == 1, "batch 1 only"
        q, k, v = q[:, 0], k[:, 0], v[:, 0]
    out = UlyssesMegaAttnFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), group, seq_chunk, num_pull
    )
    return out[:, None] if sbhd else out
