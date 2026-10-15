###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Host side of the gfx1250 MLA flash-attention backward: shape rules, launch plan, entry point."""

from typing import Optional, Tuple

import flydsl.compiler as flyc
import torch

from . import bwd_kernels as _k

_NUM_CU = 256  # MI455X
# A gradient chain whose one-wave grid has fewer workgroups than this launches its one-wave kernel
# (k_dkdv; one k_dqg over all queries): on small grids two-wave workgroups only crowd fewer CUs.
# Tests patch this table to force either launch set.
SMALL_GRID_WAVES = {"dkdv": 4 * _NUM_CU, "dq": 8 * _NUM_CU}
# Heads per grid row of the head-grouped launches: fewer heads per row keep more tiles of one head
# in flight, so a Q/dO or K/V tile fetched by one workgroup is more often still in L2 for the next.
_HEAD_GROUP = 64
_DQ_SPLITS = (0, 32, 64)  # one of these leaves a multiple of 96 queries for any Sq % 32 == 0

# flyc.compile's CompiledFunction per (launcher, tensor dtypes/ranks, device). Launching through it
# skips FlyDSL's per-call key derivation; all shape-dependent kernel arguments are runtime scalars.
_COMPILED = {}


def shape_unsupported_reason(batch: int, seqlen_q: int, seqlen_kv: int, num_heads: int) -> Optional[str]:
    """Why the backward cannot run these MHA sizes, or None."""
    if batch <= 0 or num_heads <= 0:
        return f"batch and num_heads must be positive, got {batch} / {num_heads}"
    if seqlen_q <= 0 or seqlen_q % 64:  # k_dkdv walks query pairs of 32 rows
        return f"seqlen_q must be a positive multiple of 64, got {seqlen_q}"
    kv_mult = _k.BLOCK_KV * _k.DKDV_NW
    if seqlen_kv <= 0 or seqlen_kv % kv_mult:
        return f"seqlen_kv must be a positive multiple of {kv_mult}, got {seqlen_kv}"
    if seqlen_q > seqlen_kv:
        # bottom-right causal leaves the first seqlen_q - seqlen_kv queries without keys
        return f"causal needs seqlen_q <= seqlen_kv, got {seqlen_q} > {seqlen_kv}"
    # k_dqg's fixed descriptor extents (1 GiB q/do/dq, 256 MiB lse/delta), k_dkdv's int32 k/v sizes
    rows_q, rows_kv = batch * seqlen_q * num_heads, batch * seqlen_kv * num_heads
    if rows_q * _k.D_QK * 2 > (1 << 30) or rows_q * 4 > (1 << 28) or rows_kv * _k.D_QK * 2 >= (1 << 31):
        return "q/k/lse too large for the backward's 32-bit descriptor extents"
    return None


def _head_group(nh):
    """Heads per grid row: nh up to _HEAD_GROUP, else the largest multiple of 8 (one XCD per head
    under round-robin dispatch) that divides nh and is at most _HEAD_GROUP, else nh."""
    if nh <= _HEAD_GROUP:
        return nh
    for d in range(_HEAD_GROUP, 0, -8):
        if nh % d == 0:
            return d
    return nh


def _dq_split(sq):
    """(q_split, n32, n96): k_dqg covers queries [0, q_split) in n32 tiles of 32, k_dqg96 covers
    [q_split, sq) in n96 tiles of 96."""
    for qs in _DQ_SPLITS:
        if qs <= sq and (sq - qs) % _k.DQ_BQW96 == 0:
            return qs, qs // _k.DQ_BQW, (sq - qs) // _k.DQ_BQW96
    raise ValueError(f"no dQ split for seqlen_q {sq}")


def _launches(dout, q, k, v, out, lse, softmax_scale, causal, stream):
    """(launcher, args) of every kernel launch in issue order, and the (dq, dk, dv) they write."""
    b, sq, h, _ = q.shape
    skv = k.shape[1]
    hg = _head_group(h)
    ngz = b * h // hg
    scale = float(softmax_scale)
    c = int(bool(causal))
    delta = torch.empty((b, h, sq), device=q.device, dtype=torch.float32)
    nl = torch.empty_like(delta)  # -LOG2E * lse
    nd = torch.empty_like(delta)  # -scale * delta
    dq, dk, dv = q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)
    launches = [
        (
            _k.launch_delta,
            (dout, out, delta, lse, nl, nd, scale, sq, h, b * sq * h, sq // _k.ROWS_DELTA, b * h, stream),
        )
    ]
    # Hq, Hkv, G = h, h, 1: MHA
    dkdv = (q, k, v, dout, nl, nd, dv, dk, scale, sq, skv, h, h, 1, sq // 16, skv - sq, c)
    if b * h * (skv // _k.BLOCK_KV) < SMALL_GRID_WAVES["dkdv"]:
        launches.append((_k.launch_dkdv, dkdv + (skv // _k.BLOCK_KV, h, b, stream)))
    else:
        nblk = skv // (_k.BLOCK_KV * _k.DKDV_NW)
        launches.append((_k.launch_dkdv64, dkdv + (nblk, hg, ngz, b, stream)))
    dqa = (q, k, v, dout, out, lse, delta, dq, scale, sq, skv, h, h, 1, skv // _k.KV_STEP, skv - sq, c)
    if b * h * (sq // _k.DQ_BQW) < SMALL_GRID_WAVES["dq"]:
        launches.append((_k.launch_dqg, dqa + (0, sq // _k.DQ_BQW, hg, ngz, stream)))
    else:
        # k_dqg96 over [q_split, sq) longest tiles first, then k_dqg over the short head [0, q_split)
        q_split, n32, n96 = _dq_split(sq)
        if n96:
            launches.append((_k.launch_dqg96, dqa + (q_split, n96, hg, ngz, stream)))
        if n32:
            launches.append((_k.launch_dqg, dqa + (0, n32, hg, ngz, stream)))
    return launches, (dq, dk, dv)


def _launch(launcher, args):
    key = (
        launcher,
        tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor)),
        args[0].device.index,
    )
    fn = _COMPILED.get(key)
    if fn is not None:
        fn(*args)
        return
    fn = flyc.compile(launcher, *args)  # compiles and issues this launch
    if fn is not None:  # None under COMPILE_ONLY
        _COMPILED[key] = fn


def flash_attn_mla_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients (dq, dk, dv) of MLA flash attention on gfx1250, laid out like q, k, v.

    q [B, Sq, H, 192], k [B, Skv, H, 192], v [B, Skv, H, 128], out/dout [B, Sq, H, 128], contiguous
    bf16 on one device; lse [B, H, Sq] fp32 natural log, as the forward returns it. Causal only,
    bottom-right.
    """
    if not causal:
        raise ValueError("the gfx1250 MLA attention backward supports causal attention only")
    named = (
        ("dout", dout, _k.D_V),
        ("q", q, _k.D_QK),
        ("k", k, _k.D_QK),
        ("v", v, _k.D_V),
        ("out", out, _k.D_V),
    )
    for name, t, dim in named:
        if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[-1] != dim or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [B, S, H, {dim}] bfloat16 tensor")
    if any(t.device != q.device for t in (dout, k, v, out, lse)):
        raise ValueError("dout/q/k/v/out/lse must be on one device")
    b, sq, h, _ = q.shape
    skv = k.shape[1]
    if k.shape[:3] != (b, skv, h) or v.shape[:3] != (b, skv, h):
        raise ValueError(
            f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}: batch and heads must match (MHA)"
        )
    if out.shape[:3] != (b, sq, h) or dout.shape != out.shape or lse.shape != (b, h, sq):
        raise ValueError(
            f"out/dout/lse shapes {tuple(out.shape)}/{tuple(dout.shape)}/{tuple(lse.shape)} "
            f"do not match q {tuple(q.shape)}"
        )
    reason = shape_unsupported_reason(b, sq, skv, h)
    if reason is not None:
        raise ValueError(reason)
    lse = lse.contiguous().float()
    launches, grads = _launches(dout, q, k, v, out, lse, softmax_scale, causal, torch.cuda.current_stream())
    for launcher, args in launches:
        _launch(launcher, args)
    return grads
