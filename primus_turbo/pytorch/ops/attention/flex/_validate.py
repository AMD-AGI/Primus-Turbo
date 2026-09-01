###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Argument validation for the flex compat entry points.

Every rejection here is deliberate and loud: the failure mode this layer exists to
prevent is a parameter that is accepted in a signature and then silently dropped,
which runs clean and trains a different model than the config asked for.
"""

import math
from typing import Any, Tuple

import torch

from ._config import _SUPPORTED_DTYPES


def _validate_explicit_alibi_slopes(
    alibi_slopes: Any,
    *,
    hq: int,
    device: Any,
) -> torch.Tensor:
    """Validate a caller-supplied explicit ``alibi_slopes`` and align its device.

    Requirements (matches ``flash_attn_func``'s per-head convention): a 1D fp32
    tensor of length ``Hq`` (query heads). Returns the tensor moved onto ``device``
    (q's device). Raises ``ValueError`` with a clear message otherwise.
    """
    if not isinstance(alibi_slopes, torch.Tensor):
        raise ValueError(
            "Turbo flex compat layer requires explicit alibi_slopes to be a torch.Tensor, "
            f"got {type(alibi_slopes).__name__}."
        )
    if alibi_slopes.ndim != 1:
        raise ValueError(
            "Turbo flex compat layer requires explicit alibi_slopes to be a 1D tensor, "
            f"got ndim={alibi_slopes.ndim} (shape={tuple(alibi_slopes.shape)})."
        )
    if alibi_slopes.shape[0] != hq:
        raise ValueError(
            "Turbo flex compat layer requires len(alibi_slopes) to equal the query head count "
            f"Hq={hq}, got length={alibi_slopes.shape[0]}."
        )
    if alibi_slopes.dtype != torch.float32:
        raise ValueError(
            "Turbo flex compat layer requires explicit alibi_slopes to be fp32 (torch.float32), "
            f"got dtype={alibi_slopes.dtype}."
        )
    return alibi_slopes.to(device=device)


def _normalise_explicit_softcap(softcap: Any) -> float:
    """Coerce the explicit ``softcap`` arg to a non-negative float (0.0 == disabled).

    ``None`` -> ``0.0`` (disabled). A negative or NaN cap is rejected. The
    ``softcap > 0`` gating itself lives at a single point in :func:`flex_attention`.
    """
    if softcap is None:
        return 0.0
    try:
        cap = float(softcap)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Turbo flex compat layer requires explicit softcap to be a float or None; "
            f"cannot convert {softcap!r}: {exc}"
        ) from exc
    if math.isnan(cap):
        raise ValueError("Turbo flex compat layer's explicit softcap cannot be NaN.")
    if cap < 0.0:
        raise ValueError(
            f"Turbo flex compat layer requires explicit softcap >= 0 (0/None disables it), got {cap}."
        )
    return cap


def _is_power_of_two(n: int) -> bool:
    """Local copy of the backend's power-of-two check (see attention_aiter_impl).

    Kept here so validating ``sink``'s head-dim constraint does not force the heavy
    kernel module to import during pure classification / this module's import.
    """
    return n > 0 and (n & (n - 1)) == 0


def _validate_dropout_p(dropout_p: Any) -> float:
    """Validate the Turbo-extension ``dropout_p`` (attention dropout probability).

    Requires ``0 <= p < 1`` (``p == 0`` disables dropout, the drop-in default). The value
    is threaded straight to ``flash_attn_func(dropout_p=...)``; as in flash-attn / torch
    ``scaled_dot_product_attention`` it takes effect whenever ``p > 0`` (the training
    convention -- dropout is applied unconditionally on the score matrix, so callers must
    pass ``0`` for eval). ``dropout_p > 0`` composes with ``return_lse`` (the LSE is still
    returned) and the compat layer always dispatches with ``deterministic=False``, so there
    is no dropout/determinism conflict to reject. Returns the validated float.
    """
    try:
        p = float(dropout_p)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Turbo flex compat layer requires dropout_p to be a float; cannot convert {dropout_p!r}: {exc}"
        ) from exc
    if math.isnan(p):
        raise ValueError("Turbo flex compat layer's dropout_p cannot be NaN.")
    if not (0.0 <= p < 1.0):
        raise ValueError(
            f"Turbo flex compat layer requires dropout_p in [0, 1) (0 disables dropout), got {p}."
        )
    return p


def _validate_explicit_sink(
    sink: Any,
    *,
    hq: int,
    head_dim_qk: int,
    head_dim_v: int,
    device: Any,
) -> torch.Tensor:
    """Validate a caller-supplied attention ``sink`` and align its device.

    Mirrors the backend sink constraints (see
    ``attention_aiter_impl.AttnFwdAiterBackend.can_handle`` and ``tests/.../test_attention.py``):
    a 1D fp32 tensor of length ``Hq`` (query heads -- one learned sink logit per query head),
    and the sink kernel path additionally requires ``head_dim_qk == head_dim_v`` with a
    power-of-two head dim. The value is threaded straight to ``flash_attn_func(sink=...)``.
    Returns the tensor moved onto ``device`` (q's device); raises ``ValueError`` with a clear
    message otherwise.
    """
    if not isinstance(sink, torch.Tensor):
        raise ValueError(
            f"Turbo flex compat layer requires sink to be a torch.Tensor, got {type(sink).__name__}."
        )
    if sink.ndim != 1:
        raise ValueError(
            "Turbo flex compat layer requires sink to be a 1D tensor (one sink value per query head), "
            f"got ndim={sink.ndim} (shape={tuple(sink.shape)})."
        )
    if sink.shape[0] != hq:
        raise ValueError(
            "Turbo flex compat layer requires len(sink) to equal the query head count "
            f"Hq={hq}, got length={sink.shape[0]}."
        )
    if sink.dtype != torch.float32:
        raise ValueError(
            f"Turbo flex compat layer requires sink to be fp32 (torch.float32), got dtype={sink.dtype}."
        )
    if head_dim_qk != head_dim_v:
        raise ValueError(
            "Turbo flex compat layer's sink path requires head_dim_qk == head_dim_v (backend "
            f"constraint), got head_dim_qk={head_dim_qk}, head_dim_v={head_dim_v}."
        )
    if not _is_power_of_two(head_dim_qk):
        raise ValueError(
            "Turbo flex compat layer's sink path requires head_dim to be a power of two (backend "
            f"constraint), got head_dim={head_dim_qk}."
        )
    return sink.to(device=device)


def _validate_and_adapt_bias(
    bias: Any,
    *,
    sq: int,
    skv: int,
    dtype: Any,
    device: Any,
) -> torch.Tensor:
    """Validate a caller-supplied additive attention ``bias`` and adapt it to the backend.

    The aiter dense kernel accepts a single **2D** bias of shape ``[Sq, Skv]`` in q's
    dtype (fp16/bf16), added to the pre-softmax logits and *shared across batch and
    heads*. This is an empirically pinned constraint: a 4D /
    per-head bias raises ``RuntimeError: bias shape should be [sq, sk]`` and an fp32 bias
    yields ``NaN``; only a ``[Sq, Skv]`` bias in q's dtype is numerically correct (fwd &
    bwd). We therefore accept ``[Sq, Skv]`` or a leading-singleton broadcast of it
    (``[1, Sq, Skv]`` / ``[1, 1, Sq, Skv]``) and reject a genuine per-batch / per-head
    bias with a clear message; the value is cast to q's dtype and moved onto q's device.
    Returns the adapted contiguous ``[Sq, Skv]`` tensor.
    """
    if not isinstance(bias, torch.Tensor):
        raise ValueError(
            f"Turbo flex compat layer requires bias to be a torch.Tensor, got {type(bias).__name__}."
        )
    b = bias
    if b.ndim == 4:
        if b.shape[0] != 1 or b.shape[1] != 1:
            raise ValueError(
                "Turbo flex compat layer's bias backend only supports a single [Sq,Skv] shared across "
                "batch/head (AITER dense constraint); a bias that varies per batch/head is not "
                f"supported. Got shape={tuple(bias.shape)} (for per-head/per-sample bias use the "
                "codegen path)."
            )
        b = b.reshape(b.shape[2], b.shape[3])
    elif b.ndim == 3:
        if b.shape[0] != 1:
            raise ValueError(
                "Turbo flex compat layer's bias backend only supports a single shared [Sq,Skv] "
                f"(AITER dense constraint), got 3D shape={tuple(bias.shape)} (leading dim must be 1)."
            )
        b = b.reshape(b.shape[1], b.shape[2])
    elif b.ndim != 2:
        raise ValueError(
            "Turbo flex compat layer requires bias to be 2D [Sq,Skv] (or a broadcastable shape with "
            f"leading singletons: [1,Sq,Skv]/[1,1,Sq,Skv]), got ndim={b.ndim} "
            f"(shape={tuple(bias.shape)})."
        )
    if tuple(b.shape) != (sq, skv):
        raise ValueError(
            "Turbo flex compat layer requires the last two dims of bias to be "
            "[Sq={0}, Skv={1}] (AITER dense constraint), got {2}.".format(sq, skv, tuple(b.shape))
        )
    if not b.is_floating_point():
        raise ValueError(
            "Turbo flex compat layer requires bias to be a floating-point tensor (it will be cast "
            f"to q's dtype), got dtype={b.dtype}."
        )
    # Adapt precision to q's dtype: the kernel needs bias in q's dtype (fp16/bf16); an
    # fp32 bias produces NaN. Cast + move to q's device, contiguous for the kernel.
    return b.to(dtype=dtype, device=device).contiguous()


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "Turbo flex compat layer requires q/k/v to be 4D [B,H,S,D] (bhsd) tensors; "
            f"got ndim=({query.ndim},{key.ndim},{value.ndim})."
        )
    for name, t in (("query", query), ("key", key), ("value", value)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError(
                "Turbo flex compat layer currently supports only fp16/bf16 inputs; "
                f"{name}.dtype={t.dtype}, fall back to torch flex_attention for other dtypes."
            )
    if query.device != key.device or query.device != value.device:
        raise ValueError("Turbo flex compat layer requires q/k/v to be on the same device.")
    if not (query.shape[0] == key.shape[0] == value.shape[0]):
        raise ValueError("Turbo flex compat layer requires q/k/v to share the same batch dim.")
    if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
        raise ValueError("Turbo flex compat layer requires key/value to share head count and seqlen.")


def _validate_qkv_varlen(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    """Validate packed THD ``[total_tokens, H, D]`` q/k/v for the varlen entry.

    Unlike the dense entry (``bhsd``), the varlen backend consumes the THD packed
    layout directly (sequences concatenated along dim 0), so no transpose happens
    here. GQA/MQA is supported natively (``Hq % Hkv == 0``); q/k share the
    ``head_dim_qk`` while v may carry a different ``head_dim_v``.
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "Turbo flex varlen entry requires q/k/v to be 3D [total_tokens, H, D] (THD packed) "
            f"tensors; got ndim=({query.ndim},{key.ndim},{value.ndim})."
        )
    for name, t in (("query", query), ("key", key), ("value", value)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError(
                "Turbo flex varlen entry currently supports only fp16/bf16 inputs; "
                f"{name}.dtype={t.dtype}, fall back to the torch reference implementation."
            )
    if query.device != key.device or query.device != value.device:
        raise ValueError("Turbo flex varlen entry requires q/k/v to be on the same device.")
    if key.shape[0] != value.shape[0]:
        raise ValueError(
            "Turbo flex varlen entry requires key/value to have the same total_tokens, "
            f"got key={key.shape[0]}, value={value.shape[0]}."
        )
    if key.shape[1] != value.shape[1]:
        raise ValueError(
            "Turbo flex varlen entry requires key/value to have the same head count, "
            f"got Hk={key.shape[1]}, Hv={value.shape[1]}."
        )
    if query.shape[2] != key.shape[2]:
        raise ValueError(
            "Turbo flex varlen entry requires query/key to have the same head_dim (head_dim_qk), "
            f"got Dq={query.shape[2]}, Dk={key.shape[2]}."
        )
    hq, hkv = query.shape[1], key.shape[1]
    if hq != hkv and (hkv <= 0 or hq % hkv != 0):
        raise ValueError(
            f"Turbo flex varlen entry requires Hq divisible by Hkv (GQA/MQA), got Hq={hq}, Hkv={hkv}."
        )


def _validate_window_size(window_size: Any) -> Tuple[int, int]:
    """Coerce/validate a ``(left, right)`` window into a 2-int tuple.

    ``(-1, -1)`` means full attention; a left window ``(W, 0)`` mirrors the dense
    classifier's sliding-window-causal mapping (per segment in the varlen case).
    """
    if isinstance(window_size, torch.Tensor) or not isinstance(window_size, (tuple, list)):
        raise ValueError(
            "Turbo flex varlen entry requires window_size to be a length-2 (left, right) tuple/list, "
            f"got {type(window_size).__name__}."
        )
    if len(window_size) != 2:
        raise ValueError(
            "Turbo flex varlen entry requires window_size to have length 2 (left, right), "
            f"got length={len(window_size)}."
        )
    left, right = window_size
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (left, right)):
        raise ValueError(
            f"Turbo flex varlen entry requires both window_size elements to be int, got {window_size!r}."
        )
    return (int(left), int(right))


def _validate_max_seqlen(name: str, value: Any) -> int:
    """Validate a ``max_seqlen_*`` argument (a positive Python int)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Turbo flex varlen entry requires {name} to be a Python int, "
            f"got {type(value).__name__}={value!r}."
        )
    if value <= 0:
        raise ValueError(f"Turbo flex varlen entry requires {name} to be a positive int, got {value}.")
    return int(value)


def _validate_cu_seqlens(
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    *,
    total_q: int,
    total_k: int,
    max_seqlen_q: Any,
    max_seqlen_k: Any,
    causal: bool,
    device: Any,
) -> Tuple[int, int]:
    """Validate the varlen cumulative-sequence-length descriptors.

    Requirements (matching ``flash_attn_varlen_func``): both are 1D **int32**
    tensors on q's device, starting at ``0``, monotonically non-decreasing, with a
    final element equal to ``total_q`` / ``total_k`` (the packed token counts) and a
    matching number of segments (``len(cu_seqlens_q) == len(cu_seqlens_k)``). The
    longest per-segment length must not exceed the supplied ``max_seqlen``. When
    ``causal`` is set, document-internal causal masking is only well-defined when
    every segment has ``q_len == k_len`` (the kernel's bottom-right alignment would
    otherwise silently shift the mask), so ``cu_seqlens_q`` must equal
    ``cu_seqlens_k``. Raises ``ValueError`` with a clear message on any violation.
    Returns the validated ``(max_seqlen_q, max_seqlen_k)`` ints.
    """
    max_seqlen_q = _validate_max_seqlen("max_seqlen_q", max_seqlen_q)
    max_seqlen_k = _validate_max_seqlen("max_seqlen_k", max_seqlen_k)

    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if not isinstance(cu, torch.Tensor):
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to be a torch.Tensor, got {type(cu).__name__}."
            )
        if cu.dtype != torch.int32:
            raise ValueError(f"Turbo flex varlen entry requires {name} to be int32, got dtype={cu.dtype}.")
        if cu.ndim != 1:
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to be a 1D [num_seqs+1] tensor, "
                f"got ndim={cu.ndim} (shape={tuple(cu.shape)})."
            )
        if cu.numel() < 2:
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to have at least 2 elements ([0, total]), "
                f"got numel={cu.numel()}."
            )
        if cu.device != device:
            raise ValueError(
                f"Turbo flex varlen entry requires {name} to be on the same device as q, "
                f"got {cu.device} vs {device}."
            )
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "Turbo flex varlen entry requires cu_seqlens_q and cu_seqlens_k to have the same number "
            f"of segments (equal len), got {cu_seqlens_q.numel()} vs {cu_seqlens_k.numel()}."
        )

    # These descriptors are tiny; inspect on CPU in int64 (avoids int32 overflow in
    # the diff and keeps the value reads off the classifier's hot path).
    q_cpu = cu_seqlens_q.detach().to(device="cpu", dtype=torch.int64)
    k_cpu = cu_seqlens_k.detach().to(device="cpu", dtype=torch.int64)

    if int(q_cpu[0]) != 0 or int(k_cpu[0]) != 0:
        raise ValueError(
            "Turbo flex varlen entry requires the first cu_seqlens element to be 0, "
            f"got cu_seqlens_q[0]={int(q_cpu[0])}, cu_seqlens_k[0]={int(k_cpu[0])}."
        )

    q_seg = q_cpu[1:] - q_cpu[:-1]
    k_seg = k_cpu[1:] - k_cpu[:-1]
    if bool((q_seg < 0).any()) or bool((k_seg < 0).any()):
        raise ValueError(
            "Turbo flex varlen entry requires cu_seqlens to be monotonically non-decreasing (every "
            f"segment length >= 0), got cu_seqlens_q={q_cpu.tolist()}, cu_seqlens_k={k_cpu.tolist()}."
        )

    if int(q_cpu[-1]) != int(total_q):
        raise ValueError(
            "Turbo flex varlen entry requires the last cu_seqlens_q element to equal query's "
            f"total_tokens, got cu_seqlens_q[-1]={int(q_cpu[-1])}, total_q={int(total_q)}."
        )
    if int(k_cpu[-1]) != int(total_k):
        raise ValueError(
            "Turbo flex varlen entry requires the last cu_seqlens_k element to equal key/value's "
            f"total_tokens, got cu_seqlens_k[-1]={int(k_cpu[-1])}, total_k={int(total_k)}."
        )

    q_max_seg = int(q_seg.max()) if q_seg.numel() > 0 else 0
    k_max_seg = int(k_seg.max()) if k_seg.numel() > 0 else 0
    if max_seqlen_q < q_max_seg:
        raise ValueError(
            "Turbo flex varlen entry requires max_seqlen_q >= the longest query segment length, "
            f"got max_seqlen_q={max_seqlen_q}, actual longest segment={q_max_seg}."
        )
    if max_seqlen_k < k_max_seg:
        raise ValueError(
            "Turbo flex varlen entry requires max_seqlen_k >= the longest key segment length, "
            f"got max_seqlen_k={max_seqlen_k}, actual longest segment={k_max_seg}."
        )

    if causal and not torch.equal(q_cpu, k_cpu):
        raise ValueError(
            "Turbo flex varlen entry with causal=True requires q_len == k_len per segment "
            "(in-document causal is bottom-right aligned, so mismatched segment lengths silently "
            "misalign); make cu_seqlens_q == cu_seqlens_k, or use causal=False for cross-attention."
        )

    return max_seqlen_q, max_seqlen_k
