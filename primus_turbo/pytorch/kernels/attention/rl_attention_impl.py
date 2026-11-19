###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend selection for RL attention.

RL needs the rollout forward and the training forward to produce the same bits, so both must run
the same backend with the same reduction order. That rules out what other ops do here: no
per-shape auto-tuning across backends and no silent fallback. The backend is one process-wide
choice (``set_backend``), TRITON by default, and a backend that cannot take a call raises.

Each backend provides ``can_handle(**kwargs)``, ``fwd(**kwargs)`` and ``bwd(**kwargs)``; a new one
(e.g. FLYDSL) is added to ``_BACKENDS``.
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


_BACKENDS = {BackendType.TRITON: RLAttentionTritonBackend}
_backend = BackendType.TRITON


def set_backend(backend: BackendType) -> None:
    global _backend
    if backend not in _BACKENDS:
        raise ValueError(
            f"RL attention has no {backend.name} backend; available: {[b.name for b in _BACKENDS]}"
        )
    _backend = backend


def get_backend() -> BackendType:
    return _backend


def _select(**kwargs):
    impl = _BACKENDS[_backend]
    if not impl.can_handle(**kwargs):
        raise ValueError(f"RL attention backend {_backend.name} cannot handle these inputs")
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
    )
    return _select(**kwargs).fwd(**kwargs)


@rl_attention_fwd_impl.register_fake
def _rl_attention_fwd_impl_meta(
    q, k, v, cu_seqlens_q, cu_seqlens_k, seq_lens, block_table, max_seqlen_q, max_seqlen_k,
    softmax_scale, window_left, vis_lo, vis_hi, sinks, return_lse, mode,
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
    dq, dk, dv, dsink = _select(**kwargs).bwd(**kwargs)
    if dsink is None:
        dsink = q.new_empty((0,), dtype=torch.float32)
    return dq, dk, dv, dsink


@rl_attention_bwd_impl.register_fake
def _rl_attention_bwd_impl_meta(
    dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale, window_left, vis_lo, vis_hi, sinks,
):  # fmt: skip
    dsink_shape = (q.shape[1],) if sinks is not None else (0,)
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        q.new_empty(dsink_shape, dtype=torch.float32),
    )
