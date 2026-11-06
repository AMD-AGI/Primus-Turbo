###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses projections with the all-to-all fused into the GEMM (FlyDSL, single node).

``ulysses_qkv_gemm_a2a`` replaces ``linear_qkv -> pack -> all-to-all -> unpack`` and returns q / k / v
with the local heads over the full sequence; ``ulysses_a2a_o_gemm`` replaces
``pack -> all-to-all -> unpack -> linear_proj + residual``. Attention between them is the plain
``flash_attn_func``. Sequence order matches ``flash_attn_usp_func``: rank r holds the r-th contiguous
chunk of the sequence.
"""

from typing import Tuple

import torch
import torch.distributed as dist

from primus_turbo.flydsl.ulysses.gemm_a2a_kernel import (
    gemm_a2a_bank_bytes,
    ulysses_gemm_a2a,
)
from primus_turbo.flydsl.ulysses.workspace import (
    REGION_QKV_GEMM_A2A,
    get_ulysses_workspace,
)

__all__ = ["ulysses_qkv_gemm_a2a"]

_QKV_FORMATS = ("sbhd", "bshd")
_WEIGHT_LAYOUTS = ("interleaved", "concat")


def _rank_major_concat_rows(hq, hkv, d, n, device):
    """Row order of a [q; k; v] weight that makes each rank's q, k, v heads one contiguous block."""
    q, k, v = (torch.arange(h * d, device=device).view(n, h // n * d) for h in (hq, hkv, hkv))
    return torch.cat((q, k + hq * d, v + (hq + hkv) * d), dim=1).reshape(-1)


_PERMUTED_WEIGHTS = {}


def _rank_major_weight(w, hq, hkv, d, n):
    """``w[perm]`` for a concat-layout weight, rebuilt only when the weight changes."""
    key = (w.data_ptr(), hq, hkv, d, n)
    cached = _PERMUTED_WEIGHTS.get(key)
    if cached is not None and cached[0] == w._version:
        return cached[1]
    permuted = w.detach().index_select(0, _rank_major_concat_rows(hq, hkv, d, n, w.device))
    _PERMUTED_WEIGHTS[key] = (w._version, permuted)
    return permuted


def _qkv_rank_block(hq, hkv, d, n, weight_layout):
    """(G, nq, nk, nv) of one rank's QKV column block, see ``gemm_a2a_kernel``."""
    if weight_layout == "interleaved":
        return hkv // n, hq // hkv, 1, 1
    return 1, hq // n, hkv // n, hkv // n


def _check(group, hq, hkv, d, qkv_format, weight_layout):
    n = group.size()
    assert qkv_format in _QKV_FORMATS, f"qkv_format must be one of {_QKV_FORMATS}"
    assert weight_layout in _WEIGHT_LAYOUTS, f"weight_layout must be one of {_WEIGHT_LAYOUTS}"
    assert hq % n == 0 and hkv % n == 0 and hq % hkv == 0, f"hq={hq}, hkv={hkv} must divide by n={n}"
    assert d in (64, 128), f"head_dim {d} is not supported"


def _qkv_gemm_a2a_forward(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
    n = group.size()
    if qkv_format == "sbhd":
        s_local, b, hidden = x.shape
    else:
        b, s_local, hidden = x.shape
    a = x.reshape(s_local * b, hidden)
    if not a.is_contiguous():
        a = a.contiguous()
    w = w_qkv if weight_layout == "interleaved" else _rank_major_weight(w_qkv, hq, hkv, d, n)
    w = w.contiguous()
    M, N = a.shape[0], w.shape[0]
    assert N == (hq + 2 * hkv) * d and w.shape[1] == hidden

    seq = s_local * n
    shape = (seq, b) if qkv_format == "sbhd" else (b, seq)
    q, k, v = (torch.empty(seq * b, h // n, d, dtype=x.dtype, device=x.device) for h in (hq, hkv, hkv))
    workspace = get_ulysses_workspace(group, gemm_a2a_bank_bytes(M, N, 256))
    G, nq, nk, nv = _qkv_rank_block(hq, hkv, d, n, weight_layout)
    ulysses_gemm_a2a(
        a,
        w,
        (q, k, v),
        workspace,
        layout="nt",
        head_dim=d,
        groups=G,
        nq=nq,
        nk=nk,
        nv=nv,
        seq_local=s_local,
        qkv_format=qkv_format,
        flag_region=REGION_QKV_GEMM_A2A,
    )
    return tuple(t.view(*shape, *t.shape[1:]) for t in (q, k, v))


def _qkv_rows_to_weight_layout(dq, dk, dv, hq, hkv, d, weight_layout):
    """[t, h, d] per-tensor grads -> [t, (hq + 2 * hkv) * d] in the weight's row order."""
    t = dq.shape[0]
    if weight_layout == "interleaved":
        ng = hkv
        return torch.cat(
            (dq.view(t, ng, hq // ng, d), dk.view(t, ng, 1, d), dv.view(t, ng, 1, d)), dim=2
        ).view(t, -1)
    return torch.cat((dq.reshape(t, -1), dk.reshape(t, -1), dv.reshape(t, -1)), dim=1)


def _qkv_backward_unfused(ctx, dq, dk, dv):
    from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
        get_attention_cp_a2a_helper,
    )

    x, w_qkv = ctx.saved_tensors
    hq, hkv, d, group = ctx.hq, ctx.hkv, ctx.d, ctx.group
    n = group.size()
    s_local, b = ctx.s_local, ctx.b
    seq_dim = 0 if ctx.qkv_format == "sbhd" else 1
    helper = get_attention_cp_a2a_helper(b, s_local * n, hq, hkv, d, d, seq_dim, n)
    packed = helper.combine_dqkv_before_a2a(dq.contiguous(), dk.contiguous(), dv.contiguous())
    out = torch.empty_like(packed)
    dist.all_to_all_single(out, packed, group=group)
    dq, dk, dv = (t.reshape(s_local * b, -1, d) for t in helper.split_dqkv_after_a2a(out))
    dqkv = _qkv_rows_to_weight_layout(dq, dk, dv, hq, hkv, d, ctx.weight_layout)
    x2d = x.reshape(s_local * b, -1)
    dx = (dqkv @ w_qkv).view_as(x)
    dw = dqkv.t() @ x2d
    return dx, dw


class UlyssesQKVGemmA2AFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
        q, k, v = _qkv_gemm_a2a_forward(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout)
        ctx.save_for_backward(x, w_qkv)
        ctx.hq, ctx.hkv, ctx.d, ctx.group = hq, hkv, d, group
        ctx.qkv_format, ctx.weight_layout = qkv_format, weight_layout
        if qkv_format == "sbhd":
            ctx.s_local, ctx.b = x.shape[0], x.shape[1]
        else:
            ctx.b, ctx.s_local = x.shape[0], x.shape[1]
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        dx, dw = _qkv_backward_unfused(ctx, dq, dk, dv)
        return dx, dw, None, None, None, None, None, None


def ulysses_qkv_gemm_a2a(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group: dist.ProcessGroup,
    qkv_format: str = "sbhd",
    weight_layout: str = "interleaved",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV projection fused with the Ulysses all-to-all.

    Args:
        x: this rank's tokens, ``[s/n, b, hidden]`` (sbhd) or ``[b, s/n, hidden]`` (bshd).
        w_qkv: ``[(hq + 2 * hkv) * d, hidden]``; ``interleaved`` is Megatron's ``[ng, hq/ng + 2, d]``
            row order, ``concat`` is ``[q; k; v]``.

    Returns:
        q, k, v: local heads over the full sequence, ``[s, b, h/n, d]`` (sbhd) or ``[b, s, h/n, d]``.
    """
    _check(group, num_q_heads, num_kv_heads, head_dim, qkv_format, weight_layout)
    return UlyssesQKVGemmA2AFunction.apply(
        x, w_qkv, num_q_heads, num_kv_heads, head_dim, group, qkv_format, weight_layout
    )
