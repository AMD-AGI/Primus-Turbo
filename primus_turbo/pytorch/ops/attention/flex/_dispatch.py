###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lowering of a recognised variant onto the varlen (THD) backend.

Kept separate from the routing layer: routing decides *which* backend, this decides
*how* the recognised document-packed variant is handed to ``flash_attn_varlen_func``.
"""

from typing import Any, Dict, Optional

import torch


def _dispatch_document_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seg_lens: list,
    *,
    scale: Optional[float],
    alibi_slopes: Optional[torch.Tensor],
    dropout_p: float,
    sink: Optional[torch.Tensor],
    deterministic: bool = False,
    return_bshd: bool = False,
    causal: bool = True,
    window_size: tuple = (-1, -1),
) -> torch.Tensor:
    """Run a recognised document-packed *dense* call through the varlen backend.

    ``query``/``key``/``value`` are bhsd ``[B, H, S, D]`` sharing the same per-batch
    document structure ``seg_lens`` (``sum(seg_lens) == S``; batch/head independence of
    the mask is already verified by the classifier). They are packed to THD, dispatched
    block-diagonally via ``flash_attn_varlen_func`` -- which honours document boundaries
    through ``cu_seqlens`` rather than attending across them -- and the packed output is
    unpacked back to bhsd. ``sink`` is threaded only when supplied (newer-backend
    feature; a no-op default otherwise).

    ``causal`` and ``window_size`` are the *within-document* pattern recovered by the
    classifier: the varlen kernel applies them per segment, so bidirectional packing
    (``causal=False``, what diffusion models use) and packing plus a local window both
    come out of the same call. They default to the autoregressive unwindowed shape.
    """
    bsz, hq, sq, _ = query.shape
    dv = value.shape[-1]

    def _pack(t: torch.Tensor) -> torch.Tensor:  # (B,H,S,D) -> (B*S, H, D)
        # THD wants the tokens of a sequence adjacent, so this copy is real (bhsd bytes
        # are head-major). ``reshape`` on the non-contiguous transpose already produces a
        # contiguous result, so no further ``.contiguous()`` is needed.
        b, h, s, d = t.shape
        return t.transpose(1, 2).reshape(b * s, h, d)

    q_thd = _pack(query)
    k_thd = _pack(key)
    v_thd = _pack(value)

    # Replicate the per-batch document boundaries across the packed batch dimension.
    seglens_all = list(seg_lens) * bsz
    cu = torch.zeros(len(seglens_all) + 1, dtype=torch.int32, device=query.device)
    cu[1:] = torch.tensor(seglens_all, dtype=torch.int32, device=query.device).cumsum(0)
    max_s = int(max(seg_lens))

    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_varlen_func

    call_kwargs: Dict[str, Any] = dict(
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=False,
    )
    if sink is not None:
        call_kwargs["sink"] = sink

    out_thd = flash_attn_varlen_func(q_thd, k_thd, v_thd, cu, cu, max_s, max_s, **call_kwargs)
    out_bshd = out_thd.reshape(bsz, sq, hq, dv)  # (B*S, Hq, Dv) -> (B, S, Hq, Dv)
    if return_bshd:
        return out_bshd
    return out_bshd.transpose(1, 2).contiguous()  # -> (B, Hq, S, Dv)
