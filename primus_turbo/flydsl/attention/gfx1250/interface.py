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

"""Host entry points of the gfx1250 flash-attention forward and backward.

Both take BSHD tensors: q/o/do [B, Sq, Hq, D], k/v [B, Skv, Hkv, D], bf16 and contiguous,
with Hq a multiple of Hkv (GQA). The forward returns lse as [B, Hq, Sq] fp32 in natural
log, which is exactly what the backward consumes. causal is bottom-right: query i attends
keys j <= i + (Skv - Sq).
"""

import math
import os
from typing import Optional

import flydsl.compiler as flyc
import torch

from . import flash_attn_bwd_kernel as _bwd
from . import flash_attn_fwd_kernel as _fwd
from . import flash_attn_fwd_small_grid_kernel as _fwd_small

HEAD_DIM = _bwd.D
# k_dq consumes query tiles of BLOCK_Q and its launch grid is sq // BLOCK_Q, so a
# remainder would leave a tile uncomputed and write past the end; k_dkdv owns kv tiles
# of BLOCK_KV and contracts dS over KV_STEP.
SEQLEN_Q_MULTIPLE = _bwd.BLOCK_Q
SEQLEN_KV_MULTIPLE = max(_bwd.BLOCK_KV, _bwd.KV_STEP)
# The forward packs the G = Hq / Hkv query heads of a kv head into the rows of one tile, so
# a wave's 32 rows must hold whole head groups (G | 32) and the 16-row output tile at least
# one sequence position (G <= 16).
GQA_RATIOS = (1, 2, 4, 8, 16)


def unsupported_reason(q_shape, k_shape, v_shape, dtype, causal) -> Optional[str]:
    """Why the gfx1250 kernels cannot run a BSHD problem, or None when they can.

    Covers forward AND backward: the forward alone also takes qk head dims 192/256, but
    the autograd path always needs both.
    """
    if dtype != torch.bfloat16:
        return f"dtype must be bfloat16, got {dtype}"
    b, sq, hq, d = q_shape
    bk, skv, hkv, dk = k_shape
    if tuple(v_shape) != tuple(k_shape) or bk != b:
        return f"k {tuple(k_shape)} and v {tuple(v_shape)} must match, with q's batch {b}"
    if d != HEAD_DIM or dk != HEAD_DIM:
        return f"head_dim must be {HEAD_DIM}, got q {d} / k {dk}"
    if hkv == 0 or hq % hkv or hq // hkv not in GQA_RATIOS:
        return f"heads_q / heads_kv must be one of {GQA_RATIOS}, got {hq} / {hkv}"
    if sq == 0 or sq % SEQLEN_Q_MULTIPLE:
        return f"seqlen_q must be a positive multiple of {SEQLEN_Q_MULTIPLE}, got {sq}"
    if skv == 0 or skv % SEQLEN_KV_MULTIPLE:
        return f"seqlen_kv must be a positive multiple of {SEQLEN_KV_MULTIPLE}, got {skv}"
    if causal and sq > skv:
        # Bottom-right causal leaves the first sq - skv queries with no key at all.
        return f"causal needs seqlen_q <= seqlen_kv, got {sq} > {skv}"
    return None


def _check(tensors, q, k, v, causal):
    for name, t in tensors:
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous bfloat16 tensor")
    reason = unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal)
    if reason is not None:
        raise ValueError(reason)


_NUM_CU = {}


def _num_cu(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_CU:
        _NUM_CU[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NUM_CU[idx]


def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """Returns (o [B, Sq, Hq, D] bf16, lse [B, Hq, Sq] fp32 natural log)."""
    _check((("q", q), ("k", k), ("v", v)), q, k, v, causal)
    b, sq, hq, d = q.shape
    hkv = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    # An under-filled grid is bound by its heaviest workgroup's latency; the small-grid
    # variant halves the rows per wave (BLOCK_M 64 instead of 256) to spread the work.
    grid = -(-sq * (hq // hkv) // _fwd.BLOCK_M) * hkv * b
    kern = _fwd_small if grid < _num_cu(q.device) else _fwd
    return kern.flash_attn_batch_m32x8(
        q, k, v, softmax_scale=float(softmax_scale), causal=bool(causal), return_lse=True
    )


_COMPILED = {}


def _launch(name, launcher, args):
    """Launch `launcher(*args)`, via flyc.compile's fast path after the first call.

    A plain @flyc.jit call re-derives the whole cache key on every launch, a flat
    ~0.27 ms of host time; the CompiledFunction flyc.compile returns only refreshes the
    argument storage. Every shape-dependent argument of these launchers is a runtime
    Int32, so one compiled object serves every shape.
    """
    key = (
        name,
        tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor)),
        args[0].device.index,
    )
    if os.environ.get("COMPILE_ONLY") == "1":
        # Build only; never memoize, or a later call would launch through the memo.
        flyc.compile(launcher, *args)
        return
    fn = _COMPILED.get(key)
    if fn is not None:
        fn(*args)
        return
    # flyc.compile() issues this launch itself, so it must not be repeated here.
    _COMPILED[key] = flyc.compile(launcher, *args)


# Split-K over k_dq's kv loop is capped at 8: at 16 the per-split loop is about one trip,
# so the prologue and the fp32 partial store stop being amortised (measured 4/8/16 on
# gfx1250; 8 is the interior optimum).
_NSP_Q_CAP = 8
# Enough workgroups for two dispatch waves: one wave ends when its longest (most causal
# work) workgroup ends; two let the dispatcher balance the causal skew.
_TARGET_WGS = 2048


def _split(wgs, cap):
    nsp = 1
    while wgs * nsp < _TARGET_WGS and nsp < cap:
        nsp *= 2
    return nsp


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Returns (dq, dk, dv) in q/k/v's dtype and layout; dk/dv have Hkv heads.

    The GQA reduction happens inside k_dkdv (one workgroup streams every q head of its
    kv head), so every output element is written exactly once: deterministic, no atomics.
    """
    _check((("do", do), ("q", q), ("k", k), ("v", v), ("o", o)), q, k, v, causal)
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    lse = lse.contiguous().float()
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    scale = float(softmax_scale)
    g = hq // hkv
    cz = int(bool(causal))
    stream = torch.cuda.current_stream()
    n_rows = b * sq * hq

    nsp_q = _split(sq // _bwd.BLOCK_Q * hq * b, _NSP_Q_CAP)
    # The grouped k_dq variant covers the unsplit case.
    use_g = nsp_q == 1 and hq % _bwd.DQ_NW == 0 and sq % _bwd.DQ_BQW == 0

    delta = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    _launch("delta", _bwd.launch_delta, (do, o, delta, sq, hq, n_rows, n_rows // _bwd.ROWS_DELTA, stream))

    # Split-K over k_dkdv's q loop when its grid ((Skv/BLOCK_KV)*Hkv*B workgroups, one
    # wave32 each) is too small to fill the machine twice; partials are folded in a fixed
    # order by k_redsp, so the result stays deterministic.
    nsp = _split(skv // _bwd.BLOCK_KV * hkv * b, 16)
    dk = torch.empty((b, skv, hkv, d), device=k.device, dtype=k.dtype)
    dv = torch.empty_like(dk)
    if nsp == 1:
        _launch(
            "dkdv",
            _bwd.launch_dkdv,
            (
                q,
                k,
                v,
                do,
                lse,
                delta,
                dv,
                dk,
                scale,
                sq,
                skv,
                hq,
                hkv,
                g,
                sq // 16,
                skv - sq,
                cz,
                skv // _bwd.BLOCK_KV,
                hkv,
                b,
                stream,
            ),
        )
    else:
        dkp = torch.empty((nsp, b, skv, hkv, d), device=k.device, dtype=torch.float32)
        dvp = torch.empty_like(dkp)
        _launch(
            "dkdv_sp",
            _bwd.launch_dkdv_sp,
            (
                q,
                k,
                v,
                do,
                lse,
                delta,
                dvp,
                dkp,
                scale,
                sq,
                skv,
                hq,
                hkv,
                g,
                sq // 16,
                skv - sq,
                cz,
                skv // _bwd.BLOCK_KV,
                hkv,
                b,
                nsp,
                hkv * nsp,
                stream,
            ),
        )
        n_vec = dk.numel() // _bwd.RED_VEC
        _launch(
            "redsp",
            _bwd.launch_redsp,
            (dkp, dvp, dk, dv, n_vec, nsp, -(-n_vec // _bwd.RED_THREADS), stream),
        )

    dq = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
    if use_g:
        _launch(
            "dqg",
            _bwd.launch_dqg,
            (
                q,
                k,
                v,
                do,
                o,
                lse,
                delta,
                dq,
                scale,
                sq,
                skv,
                hq,
                hkv,
                g,
                skv // _bwd.KV_STEP,
                skv - sq,
                cz,
                sq // _bwd.DQ_BQW,
                hq // _bwd.DQ_NW,
                b,
                stream,
            ),
        )
    elif nsp_q == 1:
        _launch(
            "dq",
            _bwd.launch_dq,
            (
                q,
                k,
                v,
                do,
                lse,
                delta,
                dq,
                scale,
                sq,
                skv,
                hq,
                hkv,
                g,
                skv // _bwd.KV_STEP,
                skv - sq,
                cz,
                sq // _bwd.BLOCK_Q,
                hq,
                b,
                stream,
            ),
        )
    else:
        dqp = torch.empty((nsp_q, b, sq, hq, d), device=q.device, dtype=torch.float32)
        _launch(
            "dq_sp",
            _bwd.launch_dq_sp,
            (
                q,
                k,
                v,
                do,
                lse,
                delta,
                dqp,
                scale,
                sq,
                skv,
                hq,
                hkv,
                g,
                skv // _bwd.KV_STEP,
                skv - sq,
                cz,
                sq // _bwd.BLOCK_Q,
                hq,
                b,
                nsp_q,
                hq * nsp_q,
                stream,
            ),
        )
        n_vec = dq.numel() // _bwd.RED_VEC
        _launch(
            "redsp_q",
            _bwd.launch_redsp_q,
            (dqp, dq, n_vec, nsp_q, -(-n_vec // _bwd.RED_THREADS), stream),
        )
    return dq, dk, dv
