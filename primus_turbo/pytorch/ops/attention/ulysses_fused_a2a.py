###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from primus_turbo.flydsl.ulysses.a2a_gemm_kernel import (
    a2a_gemm_bank_bytes,
    ulysses_a2a_gemm,
)
from primus_turbo.flydsl.ulysses.gemm_a2a_kernel import (
    gemm_a2a_bank_bytes,
    ulysses_gemm_a2a,
)
from primus_turbo.flydsl.ulysses.workspace import (
    REGION_O_A2A_GEMM,
    REGION_O_GEMM_A2A,
    REGION_QKV_A2A_GEMM,
    REGION_QKV_GEMM_A2A,
    get_ulysses_workspace,
)

__all__ = ["ulysses_qkv_gemm_a2a", "ulysses_a2a_o_gemm", "GemmA2ATuning", "A2AGemmTuning"]

_QKV_FORMATS = ("sbhd", "bshd")
_WEIGHT_LAYOUTS = ("interleaved", "concat")


@dataclass(frozen=True)
class GemmA2ATuning:
    """Launch knobs of a GemmA2A (forward QKV, backward o_proj); ``None`` keeps the default.

    Attributes:
        BM: GEMM tile rows (128); smaller tiles finish, and get pulled, earlier.
        num_lead: GEMM blocks dispatched ahead of the pull blocks (CU count, one wave).
        num_pull_cu: pull blocks queued after the lead wave (16; forward QKV 24).
        num_tail_pull: pull blocks queued after all GEMM blocks, for the last wave (16; forward QKV 8).
        wgrad_at: GEMM tile after which the in-kernel wgrad tiles are dispatched (backward o_proj
            only; ``num_lead + gemm_tiles // 8``, or last when there are more wgrad tiles than CUs).
    """

    BM: Optional[int] = None
    num_lead: Optional[int] = None
    num_pull_cu: Optional[int] = None
    num_tail_pull: Optional[int] = None
    wgrad_at: Optional[int] = None


@dataclass(frozen=True)
class A2AGemmTuning:
    """Launch knobs of an A2AGemm (forward o_proj, backward QKV); ``None`` keeps the shape heuristic.

    Attributes:
        BM: GEMM tile rows, 128 or 256 (256 when every tile fits on the CUs or wgrad tiles are many).
        num_push: push blocks, which run the last GEMM tiles once done (32; 16 with few wgrad tiles).
        push_split: sub-tiles a ``BM``-row m-tile is pushed in, so its rows land in pieces (8).
        wgrad_at: claim slot of the in-kernel wgrad tiles (backward QKV only; 0, or after every
            dgrad tile when there are more than 2x CU wgrad tiles).
    """

    BM: Optional[int] = None
    num_push: Optional[int] = None
    push_split: Optional[int] = None
    wgrad_at: Optional[int] = None


def _launch_kwargs(tuning, **defaults):
    if tuning is None:
        return defaults
    return {**defaults, **{k: v for k, v in asdict(tuning).items() if v is not None}}


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


def _qkv_gemm_a2a_forward(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, tuning=None):
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
        **_launch_kwargs(tuning, num_pull_cu=24, num_tail_pull=8),
    )
    return tuple(t.view(*shape, *t.shape[1:]) for t in (q, k, v))


def _qkv_backward(ctx, dq, dk, dv):
    """A2AGemm: the local heads' grads over the full sequence are pushed into the sequence owner's
    recv bank in the (rank-major) weight row order; ``dx = dqkv @ w_qkv`` and ``dw = dqkv^T @ x``
    consume them as they land, in the same kernel (shapes it cannot tile copy ``dqkv`` out for a
    separate wgrad)."""
    x, w_qkv = ctx.saved_tensors
    hq, hkv, d, group, weight_layout = ctx.hq, ctx.hkv, ctx.d, ctx.group, ctx.weight_layout
    n = group.size()
    s_local, b = ctx.s_local, ctx.b
    M, hidden = s_local * b, w_qkv.shape[1]
    K = (hq + 2 * hkv) * d
    concat = weight_layout != "interleaved"
    w = _rank_major_weight(w_qkv, hq, hkv, d, n) if concat else w_qkv.contiguous()
    parts = tuple(t.reshape(s_local * n * b, -1, d).contiguous() for t in (dq, dk, dv))
    need_dx, need_dw = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
    dx = torch.empty(M, hidden, dtype=x.dtype, device=x.device)
    # dw tiles inside the kernel: whole 256-row m-tiles, one source's columns per 256-row dw tile
    fused_wgrad = need_dw and M % 256 == 0 and (K // n) % 256 == 0 and hidden % 256 == 0
    x2d = x.reshape(M, hidden).contiguous() if need_dw else None
    dqkv = torch.empty(M, K, dtype=x.dtype, device=x.device) if need_dw and not fused_wgrad else None
    dw_rank_major = torch.empty(K, hidden, dtype=x.dtype, device=x.device) if fused_wgrad else None
    workspace = get_ulysses_workspace(group, a2a_gemm_bank_bytes(M, K))
    G = _qkv_rank_block(hq, hkv, d, n, weight_layout)[0]
    ulysses_a2a_gemm(
        parts,
        w,
        None,
        dx,
        dqkv,
        workspace,
        seq_local=s_local,
        qkv_format=ctx.qkv_format,
        flag_region=REGION_QKV_A2A_GEMM,
        layout="nn",
        groups=G,
        wgrad=(x2d, dw_rank_major) if fused_wgrad else None,
        **_launch_kwargs(ctx.bwd_tuning),
    )
    dw = None
    if need_dw:
        if not fused_wgrad:
            dw_rank_major = dqkv.t() @ x2d
        if concat:
            dw = torch.empty_like(dw_rank_major)
            dw.index_copy_(0, _rank_major_concat_rows(hq, hkv, d, n, x.device), dw_rank_major)
        else:
            dw = dw_rank_major
    return (dx.view_as(x) if need_dx else None), dw


class UlyssesQKVGemmA2AFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, fwd_tuning, bwd_tuning):
        q, k, v = _qkv_gemm_a2a_forward(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, fwd_tuning)
        ctx.save_for_backward(x, w_qkv)
        ctx.hq, ctx.hkv, ctx.d, ctx.group = hq, hkv, d, group
        ctx.qkv_format, ctx.weight_layout, ctx.bwd_tuning = qkv_format, weight_layout, bwd_tuning
        if qkv_format == "sbhd":
            ctx.s_local, ctx.b = x.shape[0], x.shape[1]
        else:
            ctx.b, ctx.s_local = x.shape[0], x.shape[1]
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        dx, dw = _qkv_backward(ctx, dq, dk, dv)
        return dx, dw, None, None, None, None, None, None, None, None


def ulysses_qkv_gemm_a2a(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group: dist.ProcessGroup,
    qkv_format: str = "sbhd",
    weight_layout: str = "interleaved",
    fwd_tuning: Optional[GemmA2ATuning] = None,
    bwd_tuning: Optional[A2AGemmTuning] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV projection fused with the Ulysses all-to-all.

    Args:
        x: this rank's tokens, ``[s/n, b, hidden]`` (sbhd) or ``[b, s/n, hidden]`` (bshd).
        w_qkv: ``[(hq + 2 * hkv) * d, hidden]``; ``interleaved`` is Megatron's ``[ng, hq/ng + 2, d]``
            row order, ``concat`` is ``[q; k; v]``.
        fwd_tuning: launch knobs of the forward GemmA2A.
        bwd_tuning: launch knobs of the backward A2AGemm (dgrad + in-kernel wgrad).

    Returns:
        q, k, v: local heads over the full sequence, ``[s, b, h/n, d]`` (sbhd) or ``[b, s, h/n, d]``.
    """
    _check(group, num_q_heads, num_kv_heads, head_dim, qkv_format, weight_layout)
    return UlyssesQKVGemmA2AFunction.apply(
        x,
        w_qkv,
        num_q_heads,
        num_kv_heads,
        head_dim,
        group,
        qkv_format,
        weight_layout,
        fwd_tuning,
        bwd_tuning,
    )


def _o_gemm_forward(o, w_o, residual, group, qkv_format, save_o, tuning=None):
    n = group.size()
    if qkv_format == "sbhd":
        seq, b, heads, d = o.shape
    else:
        b, seq, heads, d = o.shape
    s_local = seq // n
    M, K_r = s_local * b, heads * d
    N, K = w_o.shape
    assert K == n * K_r, f"w_o has {K} input features, expected {n} * {K_r}"
    y_shape = (s_local, b, N) if qkv_format == "sbhd" else (b, s_local, N)
    if residual is not None:
        assert tuple(residual.shape) == y_shape, f"residual {tuple(residual.shape)} != {y_shape}"
        residual = residual.reshape(M, N).contiguous()
    y = torch.empty(M, N, dtype=o.dtype, device=o.device)
    o_gathered = torch.empty(M, K, dtype=o.dtype, device=o.device) if save_o else None
    workspace = get_ulysses_workspace(group, a2a_gemm_bank_bytes(M, K))
    ulysses_a2a_gemm(
        o.reshape(seq * b, K_r).contiguous(),
        w_o.contiguous(),
        residual,
        y,
        o_gathered,
        workspace,
        seq_local=s_local,
        qkv_format=qkv_format,
        flag_region=REGION_O_A2A_GEMM,
        **_launch_kwargs(tuning),
    )
    return y.view(y_shape), o_gathered


def _o_gemm_dgrad(dy2d, w_o, group, heads, d, s_local, b, qkv_format, wgrad=None, tuning=None):
    """``dO = (dY @ w_o)`` scattered back to the local heads over the full sequence (GemmA2A);
    ``wgrad=(o_gathered, dw)`` also computes ``dw = dY^T @ o_gathered`` in the same kernel."""
    n = group.size()
    M, K = dy2d.shape[0], w_o.shape[1]
    seq = s_local * n
    do = torch.empty(seq * b, heads, d, dtype=dy2d.dtype, device=dy2d.device)
    workspace = get_ulysses_workspace(group, gemm_a2a_bank_bytes(M, K, 256))
    ulysses_gemm_a2a(
        dy2d,
        w_o.contiguous(),
        (do, do, do),
        workspace,
        layout="nn",
        head_dim=d,
        groups=1,
        nq=heads,
        nk=0,
        nv=0,
        seq_local=s_local,
        qkv_format=qkv_format,
        flag_region=REGION_O_GEMM_A2A,
        wgrad=None if wgrad is None else (dy2d, *wgrad),
        **_launch_kwargs(tuning),
    )
    shape = (seq, b) if qkv_format == "sbhd" else (b, seq)
    return do.view(*shape, heads, d)


def _o_gemm_backward(
    dy2d, w_o, o_gathered, group, heads, d, s_local, b, qkv_format, need_do, need_dw, tuning=None
):
    """``(dO, dW_o)``: dO by GemmA2A, ``dW_o = dY^T @ O_gathered``."""
    do = dw = None
    hidden, K = w_o.shape
    dgrad_args = (dy2d, w_o, group, heads, d, s_local, b, qkv_format)
    if need_do and need_dw and hidden % 256 == 0 and K % 256 == 0:
        # wgrad does not depend on the exchange: its tiles fill the GemmA2A launch behind the GEMM tiles
        dw = torch.empty_like(w_o)
        do = _o_gemm_dgrad(*dgrad_args, wgrad=(o_gathered, dw), tuning=tuning)
    elif need_do and need_dw:
        do = _o_gemm_dgrad(*dgrad_args, tuning=tuning)
        dw = dy2d.t() @ o_gathered
    elif need_do:
        do = _o_gemm_dgrad(*dgrad_args, tuning=tuning)
    elif need_dw:
        dw = dy2d.t() @ o_gathered
    return do, dw


class UlyssesA2AOGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, w_o, residual, group, qkv_format, fwd_tuning, bwd_tuning):
        y, o_gathered = _o_gemm_forward(
            o, w_o, residual, group, qkv_format, save_o=w_o.requires_grad, tuning=fwd_tuning
        )
        ctx.save_for_backward(w_o, o_gathered)
        ctx.group, ctx.qkv_format, ctx.has_residual = group, qkv_format, residual is not None
        ctx.bwd_tuning = bwd_tuning
        ctx.o_shape = o.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        w_o, o_gathered = ctx.saved_tensors
        qkv_format = ctx.qkv_format
        if qkv_format == "sbhd":
            s_local, b, N = dy.shape
        else:
            b, s_local, N = dy.shape
        heads, d = ctx.o_shape[2], ctx.o_shape[3]
        dy2d = dy.reshape(s_local * b, N).contiguous()
        do, dw = _o_gemm_backward(
            dy2d,
            w_o,
            o_gathered,
            ctx.group,
            heads,
            d,
            s_local,
            b,
            qkv_format,
            *ctx.needs_input_grad[:2],
            tuning=ctx.bwd_tuning,
        )
        d_residual = dy if ctx.has_residual and ctx.needs_input_grad[2] else None
        return do, dw, d_residual, None, None, None, None


def ulysses_a2a_o_gemm(
    o_local_heads: torch.Tensor,
    w_o: torch.Tensor,
    residual: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    qkv_format: str = "sbhd",
    fwd_tuning: Optional[A2AGemmTuning] = None,
    bwd_tuning: Optional[GemmA2ATuning] = None,
) -> torch.Tensor:
    """Output projection fused with the reverse Ulysses all-to-all, plus ``residual``.

    Args:
        o_local_heads: attention output for the local heads over the full sequence,
            ``[s, b, h/n, d]`` (sbhd) or ``[b, s, h/n, d]`` (bshd).
        w_o: ``[hidden, h * d]``.
        residual: ``[s/n, b, hidden]`` (sbhd) / ``[b, s/n, hidden]`` (bshd), or ``None``.
        fwd_tuning: launch knobs of the forward A2AGemm.
        bwd_tuning: launch knobs of the backward GemmA2A (dgrad + in-kernel wgrad).

    Returns:
        y = o_gathered @ w_o^T + residual, shaped like ``residual``. The K accumulation order is
        fixed, so the result is bitwise reproducible.
    """
    assert qkv_format in _QKV_FORMATS, f"qkv_format must be one of {_QKV_FORMATS}"
    n = group.size()
    assert o_local_heads.shape[-1] in (64, 128), f"head_dim {o_local_heads.shape[-1]} is not supported"
    seq_dim = 0 if qkv_format == "sbhd" else 1
    assert o_local_heads.shape[seq_dim] % n == 0, "sequence length must divide by the group size"
    return UlyssesA2AOGemmFunction.apply(
        o_local_heads, w_o, residual, group, qkv_format, fwd_tuning, bwd_tuning
    )
