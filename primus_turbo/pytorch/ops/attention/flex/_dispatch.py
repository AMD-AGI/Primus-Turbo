###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lowering of a recognised variant onto the varlen (THD) backend.

Kept separate from the routing layer: routing decides *which* backend, this decides
*how* the recognised document-packed variant is handed to ``flash_attn_varlen_func``.
"""

from typing import Any, Callable, Dict, Optional

import torch

from ._routing import _backend_accepts


def require_varlen_sink_support(fn: Callable) -> None:
    """Raise unless this build's varlen entry can actually carry an attention sink.

    Both routes that lower a sink onto packed varlen go through this, so the reason
    lives in one place.

    The thing being probed is the *wrapper* ``flash_attn_varlen_func``, not whichever
    kernel it would eventually select. That distinction is the whole point: aiter's own
    varlen forward and backward do take a sink on recent builds, and the FlyDSL varlen
    backend takes one for equal-length segments, so reasoning about backends suggests
    the sink can get through. It cannot, because Turbo's varlen wrapper has no ``sink``
    parameter to pass it to -- the call dies with a bare ``TypeError`` from the binding
    before any backend is chosen. Dropping the sink instead is not an option: it changes
    the softmax denominator of every query and silently produces different numbers.
    """
    if _backend_accepts(fn, "sink"):
        return
    raise NotImplementedError(
        "Turbo flex compat layer: an attention sink was supplied together with document "
        "packing, but this Primus-Turbo build's flash_attn_varlen_func has no 'sink' "
        "parameter, so the sink logits cannot reach the packed kernel. Dropping them "
        "would change the softmax denominator for every query and silently produce "
        "different numbers, so this raises instead. Use the dense entries "
        "(flex_attention / flex_attention_bshd), whose backend does take a sink, or "
        "upgrade Primus-Turbo to a build whose varlen entry forwards one."
    )


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
    unpacked back to bhsd. A ``sink`` is threaded only after
    :func:`require_varlen_sink_support` confirms the varlen entry has somewhere to put
    it; builds whose wrapper predates the parameter are rejected, not degraded.

    ``causal`` and ``window_size`` are the *within-document* pattern recovered by the
    classifier: the varlen kernel applies them per segment, so bidirectional packing
    (``causal=False``, what diffusion models use) and packing plus a local window both
    come out of the same call. They default to the autoregressive unwindowed shape.

    The pack/unpack copies are not free, and on this route they are the dominant cost:
    measured on MI355 (gfx950, B=2 H=32 D=128, bf16) they add 0.09 / 0.21 / 0.33 ms at
    S=1024 / 4096 / 8192 on top of a varlen kernel taking 0.05 / 0.16 / 0.41 ms -- so
    roughly a doubling. They are real copies rather than views because bhsd is
    head-major while THD needs a sequence's tokens adjacent. A caller who already holds
    THD data should use ``flex_attention_varlen`` and skip this entirely.
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
        # Enforced here rather than only at the call sites: this is the single point
        # where a sink is actually attached to the varlen call, so a future caller
        # cannot route around the check.
        require_varlen_sink_support(flash_attn_varlen_func)
        call_kwargs["sink"] = sink

    out_thd = flash_attn_varlen_func(q_thd, k_thd, v_thd, cu, cu, max_s, max_s, **call_kwargs)
    out_bshd = out_thd.reshape(bsz, sq, hq, dv)  # (B*S, Hq, Dv) -> (B, S, Hq, Dv)
    if return_bshd:
        return out_bshd
    return out_bshd.transpose(1, 2).contiguous()  # -> (B, Hq, S, Dv)
