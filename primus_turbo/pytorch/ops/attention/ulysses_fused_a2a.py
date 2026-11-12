###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses projections with the all-to-all fused into the GEMM (single node).

``ulysses_qkv_gemm_a2a`` replaces ``linear_qkv -> pack -> all-to-all -> unpack`` and returns q / k / v
with the local heads over the full sequence; ``ulysses_a2a_o_gemm`` replaces
``pack -> all-to-all -> unpack -> linear_proj + residual``. Attention between them is the plain
``flash_attn_func``. Sequence order matches ``flash_attn_usp_func``: rank r holds the r-th contiguous
chunk of the sequence.

Both passes run on the backend chosen in ``kernels/attention/ulysses_a2a_impl.py`` (``FLYDSL`` by
default, or ``TENSILELITE``; ``GlobalBackendManager.set_ulysses_a2a_backend`` or
``PRIMUS_TURBO_ULYSSES_A2A_BACKEND``), picked once in forward and kept for that call's backward. A call
TENSILELITE cannot take runs on FLYDSL, with a warning.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from primus_turbo.pytorch.kernels.attention.ulysses_a2a_impl import (
    ulysses_a2a_o_gemm_backend,
    ulysses_a2a_o_gemm_bwd_impl,
    ulysses_a2a_o_gemm_fwd_impl,
    ulysses_qkv_gemm_a2a_backend,
    ulysses_qkv_gemm_a2a_bwd_impl,
    ulysses_qkv_gemm_a2a_fwd_impl,
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


def _check(group, hq, hkv, d, qkv_format, weight_layout):
    n = group.size()
    assert qkv_format in _QKV_FORMATS, f"qkv_format must be one of {_QKV_FORMATS}"
    assert weight_layout in _WEIGHT_LAYOUTS, f"weight_layout must be one of {_WEIGHT_LAYOUTS}"
    assert hq % n == 0 and hkv % n == 0 and hq % hkv == 0, f"hq={hq}, hkv={hkv} must divide by n={n}"
    assert d in (64, 128), f"head_dim {d} is not supported"


class UlyssesQKVGemmA2AFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, fwd_tuning, bwd_tuning):
        # one backend for the forward and the backward of this call
        need_bwd = x.requires_grad or w_qkv.requires_grad
        ctx.backend = ulysses_qkv_gemm_a2a_backend(x, w_qkv, hq, hkv, d, group, need_bwd)
        q, k, v = ulysses_qkv_gemm_a2a_fwd_impl(
            x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, fwd_tuning, backend=ctx.backend
        )
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
        x, w_qkv = ctx.saved_tensors
        dx, dw = ulysses_qkv_gemm_a2a_bwd_impl(
            x,
            w_qkv,
            dq,
            dk,
            dv,
            ctx.hq,
            ctx.hkv,
            ctx.d,
            ctx.group,
            ctx.qkv_format,
            ctx.weight_layout,
            ctx.s_local,
            ctx.b,
            *ctx.needs_input_grad[:2],
            ctx.bwd_tuning,
            backend=ctx.backend,
        )
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
        fwd_tuning: launch knobs of the forward FlyDSL GemmA2A (the TENSILELITE backend ignores them).
        bwd_tuning: launch knobs of the backward FlyDSL A2AGemm (dgrad + in-kernel wgrad; TENSILELITE ignores them).

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


class UlyssesA2AOGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, w_o, residual, group, qkv_format, fwd_tuning, bwd_tuning):
        ctx.backend = ulysses_a2a_o_gemm_backend(o, w_o, residual, group, need_do=o.requires_grad)
        y, o_gathered = ulysses_a2a_o_gemm_fwd_impl(
            o,
            w_o,
            residual,
            group,
            qkv_format,
            save_o=w_o.requires_grad,
            tuning=fwd_tuning,
            backend=ctx.backend,
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
        do, dw = ulysses_a2a_o_gemm_bwd_impl(
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
            backend=ctx.backend,
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
        fwd_tuning: launch knobs of the forward FlyDSL A2AGemm (the TENSILELITE backend ignores them).
        bwd_tuning: launch knobs of the backward FlyDSL GemmA2A (dgrad + in-kernel wgrad; TENSILELITE ignores them).

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
