###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 HipKittens Project Contributors
#
# Adapted from HipKittens (https://github.com/HazyResearch/HipKittens).
# Modified by the Primus-Turbo team.
###############################################################################

"""HipKittens flash attention for gfx950 (MI355X), forward and backward.

The kernels are compiled into ``primus_turbo.pytorch._C`` like every other turbo kernel and
reached through ``torch.ops.primus_turbo_cpp_extension.hk_attn_*``. They are built in every
arch configuration but each body is guarded on ``__gfx950__``, so on any other card they
compile to an assert-and-return; nothing in the build stops them being called, which is what
this module is for. It checks the envelope, pads the sequence axes to the tile sizes the
kernels report, and owns every workspace allocation.

WHAT THESE KERNELS ADMIT, all enforced by :func:`hipkittens_attn_supported`:

* gfx950 only, and bf16 only.
* SBHD ``[s, b, h, d]``, contiguous.
* head dim 64 or 128.
* bottom-right causal, optionally with a left window. Non-causal is not implemented.
* ``Sq <= Skv``. See the note on that check for why the other direction is refused.
* no learned sink, no varlen/THD packing, no dropout, no bias, no alibi.

Anything outside that must not reach the kernels: they read out of bounds or return
uninitialized memory rather than failing, so the checks here are load-bearing rather than
defensive.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import is_gfx950

__all__ = [
    "SUPPORTED_HEAD_DIMS",
    "hipkittens_attn_supported",
    "hipkittens_attn_forward",
    "hipkittens_attn_backward",
]

SUPPORTED_HEAD_DIMS = (64, 128)

# A buffer descriptor's num_records is 32-bit, so a global tensor at or above 4 GiB wraps --
# and at an exact multiple it wraps to ZERO, silently discarding every store with no fault.
# This bounds the split-K workspaces below. It is a correctness limit, not a memory budget.
_WS_LIMIT = 2**32

# The fused single-pass backward holds one whole dQ per kv band, which is far larger than the
# dK/dV partials. Cap it at a fraction of the card rather than at the descriptor limit alone.
_FUSED_WS_FRACTION = 8
# Below this Skv:Sq ratio the fused path's fifth GEMM walks more band than it saves.
_FUSED_MIN_KV_RATIO = 1.0

# The launcher has one `switch` case per split, per head dim. Sizing the partials tensor for a
# split the launcher has no case for makes it run the NSPLIT == 1 kernel against a grid built
# for n_split, decoding kv blocks far past the end of K/V -- so this must stay in lockstep
# with that switch.
_DKDV_SPLIT_VALUES = {64: (1, 2, 4, 8), 128: (1, 2, 4, 5, 8)}


def _ops():
    """The op namespace. These are ordinary primus_turbo_cpp_extension ops.

    The kernels are compiled into every arch configuration -- each body is guarded on
    __gfx950__ and asserts on any other device pass -- so there is no separate extension to
    import and no build in which these ops are absent. What keeps them off the wrong card is
    the is_gfx950() check in hipkittens_attn_supported, not the build.
    """
    return torch.ops.primus_turbo_cpp_extension


@functools.lru_cache(maxsize=None)
def _blocks(head_dim: int) -> dict:
    """The tile sizes the kernels were compiled with, read from the kernels themselves.

    Not restated here: the sequence axes are padded up to these, so a table that drifted from
    the build would pad to the wrong multiple and read past the end of a tensor.
    """
    b = _ops().hk_attn_block_sizes(head_dim)
    keys = (
        "FWD_Q",
        "FWD_KV",
        "DQ_Q",
        "DQ_KV",
        "DKDV_Q",
        "DKDV_KV",
        "PREP_Q",
        "FUSED_Q",
        "FUSED_KV",
    )
    return dict(zip(keys, (int(x) for x in b)))


def _q_align(head_dim: int) -> int:
    b = _blocks(head_dim)
    return math.lcm(b["FWD_Q"], b["DQ_Q"], b["DKDV_Q"], b["PREP_Q"], b["FUSED_Q"])


def _kv_align(head_dim: int) -> int:
    b = _blocks(head_dim)
    return math.lcm(b["FWD_KV"], b["DQ_KV"], b["DKDV_KV"], b["FUSED_KV"])


def hipkittens_attn_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size: Tuple[int, int] = (-1, -1),
    sink: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> Tuple[bool, str]:
    """Whether these kernels can take this call, and if not, why.

    Returns the reason as well as the verdict so a pinned-backend caller can say what it
    refused rather than silently handing the work to someone else.

    ``q`` is ``[Sq, B, Hq, D]`` and ``k``/``v`` are ``[Skv, B, Hkv, D]``, i.e. SBHD.
    """
    if not (torch.cuda.is_available() and is_gfx950()):
        return False, "hipkittens attention is gfx950-only"
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        # The kernels declare gl<bf16, ...>, so any other 16-bit dtype is reinterpreted
        # bit-for-bit rather than converted or rejected -- garbage with no error.
        return False, f"hipkittens attention is bf16-only, got {q.dtype}"
    if not causal:
        return False, "hipkittens attention implements causal masking only"
    if sink is not None:
        return False, "hipkittens attention does not implement a learned sink"
    if dropout_p != 0.0:
        return False, "hipkittens attention does not implement dropout"
    if bias is not None or alibi_slopes is not None:
        return False, "hipkittens attention does not implement bias / alibi"
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False, "hipkittens attention takes dense 4-D SBHD tensors, not varlen/THD packing"
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False, "hipkittens attention needs contiguous [s, b, h, d]; it derives strides from the shape"

    if len(window_size) > 1 and int(window_size[1]) not in (0, -1):
        return False, "hipkittens attention implements a left window only"

    Sq, B, Hq, D = q.shape
    Skv, Bk, Hkv, Dk = k.shape
    if (B, D) != (Bk, Dk) or v.shape != k.shape:
        return False, f"shape mismatch q={tuple(q.shape)} k={tuple(k.shape)}"
    if D not in SUPPORTED_HEAD_DIMS:
        return False, f"hipkittens attention supports head dim {SUPPORTED_HEAD_DIMS}, got {D}"
    if Hq % Hkv != 0:
        return False, f"Hq={Hq} must be a multiple of Hkv={Hkv}"
    if Sq > Skv:
        # The mask is bottom-right aligned, so the leading Sq - Skv query rows attend to no
        # key. The forward kernel takes an early return for exactly those waves, before its
        # store, leaving those output rows unwritten -- uninitialized memory, not a NaN. The
        # backward handles the same case correctly, but a shape that works one way and is
        # silently wrong the other is worse than one refused on both.
        return False, f"hipkittens attention does not support Sq > Skv yet (Sq={Sq}, Skv={Skv})"
    return True, ""


def _require_supported(q, k, v, *, causal, window_size, sink=None, **kw) -> None:
    ok, why = hipkittens_attn_supported(
        q, k, v, causal=causal, window_size=window_size, sink=sink, **kw
    )
    if not ok:
        raise NotImplementedError(why)


def _pad_seq(t: torch.Tensor, target: int) -> torch.Tensor:
    if t.shape[0] == target:
        return t
    out = t.new_zeros((target, *t.shape[1:]))
    out[: t.shape[0]] = t
    return out


def hipkittens_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``q/k/v`` SBHD and contiguous. Returns ``(out, lse)``, lse ``[B, Hq, 1, Sq]`` natural log."""
    _require_supported(q, k, v, causal=causal, window_size=window_size)

    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    window_left = int(window_size[0])
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    # The kernels work in whole tiles and read past a ragged tail as unspecified bytes, which
    # the MFMA then turns into NaN even where the mask has already zeroed P. So a ragged
    # length has to be zero-padded rather than merely masked. Sq/Skv stay the true lengths in
    # the launch arguments, so the mask still excludes every padded position.
    b = _blocks(D)
    Sq_pad = -(-Sq // b["FWD_Q"]) * b["FWD_Q"]
    Skv_pad = -(-Skv // b["FWD_KV"]) * b["FWD_KV"]
    q_k, k_k, v_k = _pad_seq(q, Sq_pad), _pad_seq(k, Skv_pad), _pad_seq(v, Skv_pad)

    out = torch.empty_like(q_k)
    # LSE is written a whole Q tile at a time along its innermost axis, so a ragged Sq would
    # spill the last tile into the next (b, h) slice; pad and hand back a view.
    lse_pad = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)

    op = _ops().hk_attn_fwd_d64 if D == 64 else _ops().hk_attn_fwd_d128
    op(q_k, k_k, v_k, out, lse_pad, Sq, Skv, B, Hq, Hkv, window_left, float(softmax_scale))
    return out[:Sq], lse_pad[..., :Sq]


@functools.lru_cache(maxsize=None)
def _dkdv_split(Sq, Skv, B, Hq, Hkv, window_left, D, Skv_pad) -> int:
    """How many ways to split the GQA head group for dK/dV.

    The kernel's own selector decides, but the answer is memoised and bounded here because
    this layer allocates the partials tensor: a disagreement between the two reads past the
    end of it rather than failing.
    """
    n = int(_ops().hk_attn_dkdv_head_split(D, Sq, Skv, B, Hq, Hkv, window_left))
    if n not in _DKDV_SPLIT_VALUES[D]:
        return 1
    while n > 1 and (n * Skv_pad * B * Hkv * D * 2) >= _WS_LIMIT:
        n //= 2
    return n


def _use_fused(D, window_left, Sq, Sq_pad, Skv_pad, B, Hq) -> bool:
    """Whether to take the fused single-pass backward instead of the split dq + dkdv pair."""
    if D != 64 or window_left >= 0:
        return False
    if Skv_pad < _FUSED_MIN_KV_RATIO * Sq:
        return False
    n_bands = Skv_pad // _blocks(D)["FUSED_KV"]
    ws_bytes = n_bands * Sq_pad * B * Hq * D * 2
    if ws_bytes >= _WS_LIMIT:
        return False
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return ws_bytes <= total // _FUSED_WS_FRACTION


def hipkittens_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``dout/q/out`` are ``[Sq, B, Hq, D]``, ``k/v`` ``[Skv, B, Hkv, D]``, all SBHD bf16.

    ``lse`` is the forward's natural-log LSE, ``[B, Hq, 1, Sq]`` or ``[B, Hq, Sq]``, fp32.
    Returns ``(dq, dk, dv)`` in the layouts of q, k and v.
    """
    _require_supported(q, k, v, causal=causal, window_size=window_size)
    if dout.dtype is not torch.bfloat16 or out.dtype is not torch.bfloat16:
        raise NotImplementedError("hipkittens attention is bf16-only")
    if dout.shape != q.shape or out.shape != q.shape:
        raise ValueError("hipkittens attention backward: dout/out must have q's shape")
    for name, t in (("dout", dout), ("out", out)):
        if not t.is_contiguous():
            raise ValueError(f"hipkittens attention backward: {name} must be contiguous")

    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    window_left = int(window_size[0])
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    Sq_pad = -(-Sq // _q_align(D)) * _q_align(D)
    Skv_pad = -(-Skv // _kv_align(D)) * _kv_align(D)
    q_k, k_k, v_k = _pad_seq(q, Sq_pad), _pad_seq(k, Skv_pad), _pad_seq(v, Skv_pad)
    o_k, do_k = _pad_seq(out, Sq_pad), _pad_seq(dout, Sq_pad)

    lse_flat = lse.reshape(B, Hq, 1, -1)
    if lse_flat.shape[-1] != Sq:
        raise ValueError(f"hipkittens attention backward: lse last axis {lse_flat.shape[-1]} != Sq {Sq}")
    lse_k = torch.zeros((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)
    lse_k[..., :Sq] = lse_flat

    dq = torch.empty_like(q_k)
    dk = torch.empty_like(k_k)
    dv = torch.empty_like(v_k)
    # delta and lneg are written by prep over exactly [0, Sq_pad) -- Sq_pad is a multiple of
    # PREP_Q_BLOCK, so its grid covers every element -- which is why these are empty and not
    # zeros. A padded row carries delta = 0 and lse = 0 and contributes exactly zero to dK/dV.
    delta = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)
    lneg = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)

    args = dict(
        q=q_k, k=k_k, v=v_k, o=o_k, dO=do_k, dq=dq, dk=dk, dv=dv, lse=lse_k,
        Sq=Sq, Skv=Skv, B=B, Hq=Hq, Hkv=Hkv,
        window_left=window_left, softmax_scale=float(softmax_scale),
    )
    ops = _ops()

    if _use_fused(D, window_left, Sq, Sq_pad, Skv_pad, B, Hq):
        n_bands = Skv_pad // _blocks(D)["FUSED_KV"]
        ws = torch.empty((n_bands * Sq_pad, B, Hq, D), device=q.device, dtype=q_k.dtype)
        op = ops.hk_attn_bwd_fused_d64 if D == 64 else ops.hk_attn_bwd_fused_d128
        op(args["q"], args["k"], args["v"], args["o"], args["dO"], dq, dk, dv, ws,
           args["lse"], delta, Sq, Skv, B, Hq, Hkv, window_left, float(softmax_scale))
    else:
        n_split = _dkdv_split(Sq, Skv, B, Hq, Hkv, window_left, D, Skv_pad)
        if n_split > 1:
            wsk = torch.empty((n_split * Skv_pad, B, Hkv, D), device=q.device, dtype=k_k.dtype)
            wsv = torch.empty_like(wsk)
        else:
            # Never dereferenced at n_split == 1, and the op schema takes tensors rather than
            # optionals, so alias them onto dk.
            wsk = wsv = dk
        op = ops.hk_attn_bwd_d64 if D == 64 else ops.hk_attn_bwd_d128
        op(args["q"], args["k"], args["v"], args["o"], args["dO"], dq, dk, dv, args["lse"],
           delta, lneg, wsk, wsv, Sq, Skv, B, Hq, Hkv, window_left, float(softmax_scale),
           n_split)

    return dq[:Sq], dk[:Skv], dv[:Skv]
