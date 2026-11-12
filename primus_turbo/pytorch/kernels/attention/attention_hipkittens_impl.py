###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 HipKittens Project Contributors
#
# Adapted from HipKittens (https://github.com/HazyResearch/HipKittens).
# Modified by the Primus-Turbo team.
###############################################################################

"""HipKittens flash attention for gfx950 (MI355X), forward and backward.

The kernels are compiled into ``primus_turbo.pytorch._C`` from ``csrc/kernels/attention/hipkittens`` and reached through
``torch.ops.primus_turbo_cpp_extension.hk_attn_*``. Their sources carry the ``_gfx950`` suffix, so the build leaves them
out unless gfx950 is among the offload archs, and each body is guarded on ``__gfx950__``. This module checks the
envelope they admit (``hipkittens_attn_supported_impl``), pads the sequence axes to their tiles and owns every workspace
allocation. The gfx1250 (MI455X) MLA backward on HipKittens kernels is ``attention_hipkittens_gfx1250_mla_impl``.
"""

import functools
import math
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.utils import is_gfx950

__all__ = [
    "hipkittens_attn_supported_impl",
    "flash_attn_sbhd_hipkittens_forward_impl",
    "flash_attn_sbhd_hipkittens_backward_impl",
]

# =============================================================================
# gfx950: dense attention
#
# The kernels admit only what hipkittens_attn_supported_impl accepts: bf16, contiguous SBHD [s, b, h, d], head dim
# 64 or 128, bottom-right causal with an optional left window, Sq <= Skv, no sink, varlen, dropout, bias or alibi.
# Outside that they read out of bounds or leave output unwritten instead of failing, so the checks are load-bearing.
# =============================================================================

SUPPORTED_HEAD_DIMS = (64, 128)

# A buffer descriptor's num_records is 32-bit, so a global tensor at or above 4 GiB wraps -- and at an exact multiple
# it wraps to ZERO, silently discarding every store. This bounds the split-K workspaces below.
_WS_LIMIT = 2**32

# The backward always takes the split dq + dkdv pair: the fused single-pass variant in the kernels is 2.2x to 3.6x
# slower wherever its workspace fits, and silently wrong (_WS_LIMIT) where it does not.

# The launcher has one `switch` case per split and head dim; a split it has no case for runs the NSPLIT == 1 kernel
# against a grid built for n_split. Must stay in lockstep with that switch.
_DKDV_SPLIT_VALUES = {64: (1, 2, 4, 8), 128: (1, 2, 4, 5, 8)}


def _ops():
    return torch.ops.primus_turbo_cpp_extension


@functools.lru_cache(maxsize=None)
def _ops_built() -> bool:
    """Whether this build carries the kernels: they are named *_gfx950.cu, so a build whose offload archs exclude
    gfx950 never registers the ops (is_gfx950() reads the card, not the build)."""
    return hasattr(_ops(), "hk_attn_fwd_d64")


@functools.lru_cache(maxsize=None)
def _blocks(head_dim: int) -> dict:
    """The tile sizes the kernels were compiled with; the sequence axes are padded to these."""
    b = _ops().hk_attn_block_sizes(head_dim)
    # Positional (the kernel decides the order); FUSED_* belong to the undispatched fused backward.
    keys = ("FWD_Q", "FWD_KV", "DQ_Q", "DQ_KV", "DKDV_Q", "DKDV_KV", "PREP_Q", "FUSED_Q", "FUSED_KV")
    return dict(zip(keys, (int(x) for x in b)))


def _q_align(head_dim: int) -> int:
    b = _blocks(head_dim)
    return math.lcm(b["FWD_Q"], b["DQ_Q"], b["DKDV_Q"], b["PREP_Q"])


def _kv_align(head_dim: int) -> int:
    b = _blocks(head_dim)
    return math.lcm(b["FWD_KV"], b["DQ_KV"], b["DKDV_KV"])


def hipkittens_attn_supported_impl(
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
    """Whether the gfx950 kernels take SBHD ``q`` [Sq, B, Hq, D], ``k`` / ``v`` [Skv, B, Hkv, D], and if not, why."""
    if not (torch.cuda.is_available() and is_gfx950()):
        return False, "hipkittens attention is gfx950-only"
    if not _ops_built():
        return False, "hipkittens attention was not built (gfx950 is not among the offload archs)"
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        # The kernels declare gl<bf16, ...>: another 16-bit dtype would be reinterpreted, not converted.
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
        # The leading Sq - Skv query rows attend to no key: the forward returns early for them and leaves those
        # output rows uninitialized, so the shape is refused on both passes.
        return False, f"hipkittens attention does not support Sq > Skv yet (Sq={Sq}, Skv={Skv})"
    return True, ""


def _require_supported(q, k, v, *, causal, window_size) -> None:
    ok, why = hipkittens_attn_supported_impl(q, k, v, causal=causal, window_size=window_size)
    if not ok:
        raise NotImplementedError(why)


def _pad_seq(t: torch.Tensor, target: int) -> torch.Tensor:
    if t.shape[0] == target:
        return t
    out = t.new_zeros((target, *t.shape[1:]))
    out[: t.shape[0]] = t
    return out


def flash_attn_sbhd_hipkittens_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    return_lse: bool = True,
):
    """SBHD forward. Returns ``(out, lse)``, lse ``[B, Hq, 1, Sq]`` natural log, or just ``out``."""
    _require_supported(q, k, v, causal=causal, window_size=window_size)
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    window_left = int(window_size[0])
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    # The kernels read a ragged tail tile as unspecified bytes, which the MFMA turns into NaN even where the mask has
    # zeroed P, so ragged lengths are zero-padded; Sq / Skv stay the true lengths in the launch, so the mask still
    # excludes every padded position.
    b = _blocks(D)
    Sq_pad = -(-Sq // b["FWD_Q"]) * b["FWD_Q"]
    Skv_pad = -(-Skv // b["FWD_KV"]) * b["FWD_KV"]
    q_k, k_k, v_k = _pad_seq(q, Sq_pad), _pad_seq(k, Skv_pad), _pad_seq(v, Skv_pad)

    out = torch.empty_like(q_k)
    # LSE is written a whole Q tile at a time: a ragged Sq would spill into the next (b, h) slice.
    lse_pad = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)

    op = _ops().hk_attn_fwd_d64 if D == 64 else _ops().hk_attn_fwd_d128
    op(q_k, k_k, v_k, out, lse_pad, Sq, Skv, B, Hq, Hkv, window_left, float(softmax_scale))
    out, lse = out[:Sq], lse_pad[..., :Sq]
    return (out, lse) if return_lse else out


@functools.lru_cache(maxsize=None)
def _dkdv_split(Sq, Skv, B, Hq, Hkv, window_left, D, Skv_pad) -> int:
    """How many ways to split the GQA head group for dK/dV: the kernel's selector, bounded here because this layer
    allocates the partials (a disagreement would read past their end)."""
    n = int(_ops().hk_attn_dkdv_head_split(D, Sq, Skv, B, Hq, Hkv, window_left))
    if n not in _DKDV_SPLIT_VALUES[D]:
        return 1
    while n > 1 and (n * Skv_pad * B * Hkv * D * 2) >= _WS_LIMIT:
        n //= 2
    return n


def flash_attn_sbhd_hipkittens_backward_impl(
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
    """SBHD backward: ``dout/q/out`` [Sq, B, Hq, D], ``k/v`` [Skv, B, Hkv, D] bf16, ``lse`` the forward's
    ``[B, Hq, 1, Sq]`` (or ``[B, Hq, Sq]``) fp32. Returns ``(dq, dk, dv)`` in the layouts of q, k and v."""
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
    # Prep writes delta and lneg over exactly [0, Sq_pad); a padded row carries delta = 0 and lse = 0 and adds exactly
    # zero to dK/dV.
    delta = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)
    lneg = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)

    n_split = _dkdv_split(Sq, Skv, B, Hq, Hkv, window_left, D, Skv_pad)
    if n_split > 1:
        wsk = torch.empty((n_split * Skv_pad, B, Hkv, D), device=q.device, dtype=k_k.dtype)
        wsv = torch.empty_like(wsk)
    else:
        wsk = wsv = dk  # never dereferenced at n_split == 1; the op schema takes tensors, not optionals

    op = _ops().hk_attn_bwd_d64 if D == 64 else _ops().hk_attn_bwd_d128
    op(
        q_k,
        k_k,
        v_k,
        o_k,
        do_k,
        dq,
        dk,
        dv,
        lse_k,
        delta,
        lneg,
        wsk,
        wsv,
        Sq,
        Skv,
        B,
        Hq,
        Hkv,
        window_left,
        float(softmax_scale),
        n_split,
    )
    return dq[:Sq], dk[:Skv], dv[:Skv]
