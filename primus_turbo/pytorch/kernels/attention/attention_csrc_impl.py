###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch
from aiter.ops.mha import _flash_attn_backward, _flash_attn_forward, maybe_contiguous

_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper(
    "primus_turbo::attention_aiter_csrc_forward_impl", mutates_args=(), device_types="cuda"
)
def attention_aiter_csrc_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
    )
    return out_padded, softmax_lse, S_dmask, rng_state


def round_multiple(x, m):
    return (x + m - 1) // m * m


@attention_aiter_csrc_forward_impl.register_fake
def _attention_aiter_csrc_forward_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = torch.empty_like(q)
    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout
    )
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        p = torch.empty(
            (batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)),
            dtype=q.dtype,
            device=q.device,
            layout=q.layout,
        )
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return out, softmax_lse, p, rng_state


@_torch_custom_op_wrapper(
    "primus_turbo::attention_aiter_csrc_backward_impl", mutates_args=("dq", "dk", "dv"), device_types="cuda"
)
def attention_aiter_csrc_backward_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dbias: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
) -> torch.Tensor:
    return _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dbias,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        bias,
        alibi_slopes,
        deterministic,
        rng_state,
        is_v3_atomic_fp32,
        how_v3_bf16_cvt,
    )


@attention_aiter_csrc_backward_impl.register_fake
def _attention_aiter_csrc_backward_impl_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dbias: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    batch_size, seqlen_q, num_heads, _ = q.shape
    softmax_d = torch.empty(
        (batch_size, num_heads, round_multiple(seqlen_q, 128)), device=q.device, dtype=torch.float32
    )

    return softmax_d
