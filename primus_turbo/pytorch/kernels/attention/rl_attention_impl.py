###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend selection for RL attention.

RL needs the rollout forward and the training forward to produce the same bits, so both must run
the same backend with the same reduction order. That rules out what other ops do here: no
per-shape auto-tuning across backends and no silent fallback. The backend is the ``BACKEND``
kernel option (TRITON by default, or FLYDSL on gfx950), and a backend that cannot take a call
raises. The two backends give different bits, so rollout and training must use the same one.

Each backend provides ``can_handle(**kwargs)``, ``fwd(**kwargs)`` and ``bwd(**kwargs)``; a new one
is an entry of ``_BACKENDS``. The forward receives the resolved kernel options as
keyword arguments (``tile_size``, ``segment_size``, ``block_m``, ``num_warps``, ``num_stages``,
``waves_per_eu``, ``mode``; 0 / "auto" = backend default).
"""

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.attention.rl_attention_triton_impl import (
    SUPPORTED_DTYPES,
    rl_attention_bwd_triton,
    rl_attention_fwd_triton,
)


class RLAttentionTritonBackend:
    @staticmethod
    def can_handle(q, k, **kwargs) -> bool:
        return q.dtype in SUPPORTED_DTYPES and k.dtype == q.dtype and q.shape[-1] <= 256

    fwd = staticmethod(rl_attention_fwd_triton)
    bwd = staticmethod(rl_attention_bwd_triton)


class RLAttentionFlydslBackend:
    """gfx950 FlyDSL forward (bf16, D 64 / 128) with the Triton backward. Imported on first use, so
    the Triton backend works without FlyDSL installed."""

    @staticmethod
    def can_handle(**kwargs) -> bool:
        from primus_turbo.pytorch.kernels.attention import rl_attention_flydsl_impl

        return rl_attention_flydsl_impl.can_handle(**kwargs)

    @staticmethod
    def fwd(**kwargs):
        from primus_turbo.pytorch.kernels.attention import rl_attention_flydsl_impl

        return rl_attention_flydsl_impl.rl_attention_fwd_flydsl(**kwargs)

    @staticmethod
    def bwd(**kwargs):
        return rl_attention_bwd_triton(**kwargs)


_BACKENDS = {BackendType.TRITON: RLAttentionTritonBackend, BackendType.FLYDSL: RLAttentionFlydslBackend}
BACKEND_NAMES = tuple(b.name for b in _BACKENDS)


def _select(backend: str, **kwargs):
    if backend not in BACKEND_NAMES:
        raise ValueError(f"RL attention BACKEND must be one of {BACKEND_NAMES}, got {backend!r}")
    impl = _BACKENDS[BackendType[backend]]
    if not impl.can_handle(**kwargs):
        raise ValueError(f"RL attention backend {backend} cannot handle these inputs")
    return impl


@torch.library.custom_op("primus_turbo::rl_attention_fwd_impl", mutates_args=(), device_types="cuda")
def rl_attention_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_left: int,
    vis_lo: Optional[torch.Tensor],
    vis_hi: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    return_lse: bool,
    mode: str,
    tile_size: int,
    segment_size: int,
    block_m: int,
    num_warps: int,
    num_stages: int,
    waves_per_eu: int,
    backend: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """out [num_tokens, Hq, Dv] and the natural-log LSE [num_tokens, Hq] fp32 (shape [0] unless
    ``return_lse``)."""
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seq_lens=seq_lens,
        block_table=block_table,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        window_left=window_left,
        vis_lo=vis_lo,
        vis_hi=vis_hi,
        sinks=sinks,
        return_lse=return_lse,
        mode=mode,
        tile_size=tile_size,
        segment_size=segment_size,
        block_m=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    return _select(backend, **kwargs).fwd(**kwargs)


@rl_attention_fwd_impl.register_fake
def _rl_attention_fwd_impl_meta(
    q, k, v, cu_seqlens_q, cu_seqlens_k, seq_lens, block_table, max_seqlen_q, max_seqlen_k,
    softmax_scale, window_left, vis_lo, vis_hi, sinks, return_lse, mode, tile_size, segment_size,
    block_m, num_warps, num_stages, waves_per_eu, backend,
):  # fmt: skip
    num_tokens, num_q_heads, _ = q.shape
    lse_shape = (num_tokens, num_q_heads) if return_lse else (0,)
    return q.new_empty((num_tokens, num_q_heads, v.shape[-1])), q.new_empty(lse_shape, dtype=torch.float32)


@torch.library.custom_op("primus_turbo::rl_attention_bwd_impl", mutates_args=(), device_types="cuda")
def rl_attention_bwd_impl(
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
    vis_lo: Optional[torch.Tensor],
    vis_hi: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    backend: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dq, dk, dv, and dsink [Hq] fp32 (shape [0] without sinks, since the schema returns a fixed
    number of tensors)."""
    kwargs = dict(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        window_left=window_left,
        vis_lo=vis_lo,
        vis_hi=vis_hi,
        sinks=sinks,
    )
    dq, dk, dv, dsink = _select(backend, **kwargs).bwd(**kwargs)
    if dsink is None:
        dsink = q.new_empty((0,), dtype=torch.float32)
    return dq, dk, dv, dsink


@rl_attention_bwd_impl.register_fake
def _rl_attention_bwd_impl_meta(
    dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale, window_left, vis_lo, vis_hi, sinks, backend,
):  # fmt: skip
    dsink_shape = (q.shape[1],) if sinks is not None else (0,)
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        q.new_empty(dsink_shape, dtype=torch.float32),
    )
