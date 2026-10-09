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

Both take BSHD tensors on one device: q/o/do [B, Sq, Hq, D], k/v [B, Skv, Hkv, D], bf16 and
contiguous, with Hq a multiple of Hkv (GQA). The forward returns lse as [B, Hq, Sq] fp32 in
natural log, which is exactly what the backward consumes. causal is bottom-right: query i
attends keys j <= i + (Skv - Sq).

The supported range is ``unsupported_reason``'s: head_dim 128, Hq / Hkv in ``GQA_RATIOS``,
Sq a multiple of 64 and Skv of 32, no empty dimension, q and k each at most 1 GiB
(``MAX_TENSOR_BYTES``), and Sq <= Skv when causal. Both entry points raise ValueError
outside it.

Importing this module raises ImportError unless flydsl satisfies
``flydsl_version.FLYDSL_REQUIREMENT`` (``>=0.3.4.1,<0.3.5``).
"""

import math
from typing import Optional

import torch

from .flydsl_version import require_flydsl

require_flydsl()

from . import flash_attn_bwd_kernel as _bwd
from . import flash_attn_fwd_kernel as _fwd
from .flash_attn_utils import _run_compiled

HEAD_DIM = _bwd.D
if _fwd.HEAD_DIM != HEAD_DIM:
    raise ImportError(f"forward head_dim {_fwd.HEAD_DIM} != backward head_dim {HEAD_DIM}")
# The dQ kernels consume query tiles of BLOCK_Q (k_dq_sp) / DQ_BQW (k_dqg) and launch
# sq // tile of them, so a remainder would leave a tile uncomputed and write past the end;
# k_dkdv streams query pairs of 32 rows. k_dkdv owns kv tiles of BLOCK_KV and the dQ
# kernels step over kv in blocks of KV_STEP.
SEQLEN_Q_MULTIPLE = max(_bwd.BLOCK_Q, _bwd.DQ_BQW)
SEQLEN_KV_MULTIPLE = max(_bwd.BLOCK_KV, _bwd.KV_STEP)
# The forward packs the G = Hq / Hkv query heads of a kv head into the rows of one tile, so
# a wave's 32 rows must hold whole head groups (G | 32) and the 16-row output tile at least
# one sequence position (G <= 16).
GQA_RATIOS = (1, 2, 4, 8, 16)
# k_dqg and k_dq_sp take no batch count, so their q/k/v/do buffer descriptors (and k_dqg's dq)
# carry a flat 1 GiB extent: past it, loads would read zeros and stores would be dropped.
# k_dq_sp's fp32 dq workspace has its true extent.
MAX_TENSOR_BYTES = 1 << 30


def unsupported_reason(q_shape, k_shape, v_shape, dtype, causal) -> Optional[str]:
    """Why the gfx1250 kernels cannot run a BSHD problem, or None when they can.

    Covers forward AND backward: the autograd path always needs both.
    """
    if dtype != torch.bfloat16:
        return f"dtype must be bfloat16, got {dtype}"
    if len(q_shape) != 4 or len(k_shape) != 4:
        return f"q and k must be 4-D [B, S, H, D], got q {tuple(q_shape)} / k {tuple(k_shape)}"
    if 0 in (*q_shape, *k_shape):
        # An empty problem would launch zero-size grids.
        return f"q {tuple(q_shape)} and k {tuple(k_shape)} must have no empty dimension"
    b, sq, hq, d = q_shape
    bk, skv, hkv, dk = k_shape
    if tuple(v_shape) != tuple(k_shape) or bk != b:
        return f"k {tuple(k_shape)} and v {tuple(v_shape)} must match, with q's batch {b}"
    if d != HEAD_DIM or dk != HEAD_DIM:
        return f"head_dim must be {HEAD_DIM}, got q {d} / k {dk}"
    if hq % hkv or hq // hkv not in GQA_RATIOS:
        return f"heads_q / heads_kv must be one of {GQA_RATIOS}, got {hq} / {hkv}"
    if sq % SEQLEN_Q_MULTIPLE:
        return f"seqlen_q must be a multiple of {SEQLEN_Q_MULTIPLE}, got {sq}"
    if skv % SEQLEN_KV_MULTIPLE:
        return f"seqlen_kv must be a multiple of {SEQLEN_KV_MULTIPLE}, got {skv}"
    nbytes = 2 * d * max(b * sq * hq, b * skv * hkv)
    if nbytes > MAX_TENSOR_BYTES:
        return f"q and k must each fit in {MAX_TENSOR_BYTES} bytes, got {nbytes}"
    if causal and sq > skv:
        # Bottom-right causal leaves the first sq - skv queries with no key at all.
        return f"causal needs seqlen_q <= seqlen_kv, got {sq} > {skv}"
    return None


def _check(q, k, v, causal, do=None, o=None, lse=None):
    """Raise ValueError unless the kernels can take these tensors.

    The backward passes do, o and lse too. The kernels address every tensor from its base
    pointer with the shapes taken from q and k (the dQ kernels through flat 1 GiB buffer
    descriptors), so a smaller do / o / lse would be read past its end instead of failing.
    """
    bf16 = {"q": q, "k": k, "v": v} if do is None else {"do": do, "q": q, "k": k, "v": v, "o": o}
    named = bf16 if lse is None else {**bf16, "lse": lse}
    for name, t in named.items():
        if t.device != q.device:
            raise ValueError(
                f"all tensors must be on one device, got q on {q.device} and {name} on {t.device}"
            )
    for name, t in bf16.items():
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous bfloat16 tensor")
    reason = unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal)
    if reason is not None:
        raise ValueError(reason)
    for name, t in (("do", do), ("o", o)):
        if t is not None and t.shape != q.shape:
            raise ValueError(f"{name} must have q's shape {tuple(q.shape)}, got {tuple(t.shape)}")
    if lse is not None:
        b, sq, hq, _ = q.shape
        if lse.dtype != torch.float32 or tuple(lse.shape) != (b, hq, sq):
            raise ValueError(
                f"lse must be float32 of shape {(b, hq, sq)}, got {lse.dtype} {tuple(lse.shape)}"
            )


_NUM_CU = {}


def _num_cu(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_CU:
        _NUM_CU[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NUM_CU[idx]


def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """Returns (o [B, Sq, Hq, D] bf16, lse [B, Hq, Sq] fp32 natural log)."""
    _check(q, k, v, causal)
    b, sq, hq, d = q.shape
    hkv = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    # An under-filled grid is bound by its heaviest workgroup's latency; the small-grid
    # tiling uses 64-row workgroups (BLOCK_M 64 instead of 256) to spread the work.
    grid = -(-sq * (hq // hkv) // _fwd.DEFAULT.block_m) * hkv * b
    tiling = _fwd.SMALL_GRID if grid < _num_cu(q.device) else _fwd.DEFAULT
    return _fwd.flash_attn_fwd(q, k, v, tiling, softmax_scale=float(softmax_scale), causal=bool(causal))


def _launch(name, launcher, args):
    """One backward launch, through the package's compiled-function cache (_run_compiled).

    ``name`` labels the launch for the tests that replace this function to record the
    backward's launch plan.
    """
    _run_compiled(launcher, *args)


# Split-K over k_dq_sp's kv loop is capped at 8: at 16 the per-split loop is about one
# trip, so the prologue and the fp32 partial store stop being amortised (measured 4/8/16 on
# gfx1250; 8 is the interior optimum). k_dkdv_sp's q loop is longer and splits up to 16 ways.
_NSP_Q_CAP = 8
_NSP_KV_CAP = 16
# Enough workgroups for two dispatch waves: one wave ends when its longest (most causal
# work) workgroup ends; two let the dispatcher balance the causal skew.
_TARGET_WGS = 2048


def _split(wgs, cap):
    nsp = 1
    while wgs * nsp < _TARGET_WGS and nsp < cap:
        nsp *= 2
    return nsp


# The dQ chain (k_dqg, or k_dq_sp + k_redsp_q) runs on a side stream, concurrently with the
# dK/dV chain (k_dkdv, or k_dkdv_sp + k_redsp) on the caller's stream, so the dQ workgroups
# can fill the dispatch slots the causal tail of k_dkdv leaves idle. Issuing the dQ chain
# after the dK/dV chain on one stream measured about 0.9% slower on MI455X (gfx1250) at
# b4 s8192 hq32 hkv8 causal. Both chains only read q/k/v/do/lse/delta and write disjoint
# outputs, so the results are bitwise the same either way.
_SIDE_STREAMS = {}


def _side_stream(device):
    stream = _SIDE_STREAMS.get(device)
    if stream is None:
        stream = _SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Returns (dq, dk, dv) in q/k/v's dtype and layout; dk/dv have Hkv heads.

    The GQA reduction happens inside k_dkdv (one workgroup streams every q head of its
    kv head), so every output element is written exactly once: deterministic, no atomics.

    Streams: the dQ chain runs on a side stream (see _SIDE_STREAMS) that first waits for the
    caller's current stream, and the current stream waits for the side stream before this
    returns, so the call is ordered like single-stream work on the caller's current stream.
    Tensor.record_stream is not needed: everything the side stream touches is an input or
    was allocated here on the current stream before the fork, and stays referenced until
    the join is enqueued, so a freed block can only be reused by work ordered after the
    join -- for an input from another stream, through the ordering the caller already needs
    to use it on its current stream. record_stream would only hold those blocks (about
    1.1 GiB per call at b4 s8192 hq32) until the side stream's work is seen complete, which
    raises peak memory while the host runs ahead.
    """
    _check(q, k, v, causal, do=do, o=o, lse=lse)
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    lse = lse.contiguous()
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    scale = float(softmax_scale)
    g = hq // hkv
    cz = int(bool(causal))
    stream = torch.cuda.current_stream()
    n_rows = b * sq * hq

    delta = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    _launch("delta", _bwd.launch_delta, (do, o, delta, sq, hq, n_rows, n_rows // _bwd.ROWS_DELTA, stream))

    # Split-K over the dQ kv loop when the dQ grid ((Sq/BLOCK_Q)*Hq*B workgroups, one wave32
    # each) is too small to fill the machine twice; k_redsp_q folds the partials in a fixed
    # order. The unsplit case takes k_dqg. The dQ chain's buffers are allocated before the
    # fork, on the caller's stream.
    nsp_q = _split(sq // _bwd.BLOCK_Q * hq * b, _NSP_Q_CAP)
    dq = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
    dqp = torch.empty((nsp_q, b, sq, hq, d), device=q.device, dtype=torch.float32) if nsp_q > 1 else None

    # Fork: the side stream waits for everything already issued on the caller's stream (the
    # inputs, k_delta_bshd) but not for the dK/dV chain issued next.
    side = _side_stream(q.device)
    side.wait_stream(stream)

    # dK/dV chain, on the caller's stream. Split-K over k_dkdv's q loop when its grid
    # ((Skv/BLOCK_KV)*Hkv*B workgroups) is too small to fill the machine twice; k_redsp
    # folds the partials in a fixed order, so the result stays deterministic.
    nsp = _split(skv // _bwd.BLOCK_KV * hkv * b, _NSP_KV_CAP)
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

    # dQ chain, on the side stream.
    if nsp_q == 1:
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
                hq,
                b,
                side,
            ),
        )
    else:
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
                side,
            ),
        )
        n_vec = dq.numel() // _bwd.RED_VEC
        _launch(
            "redsp_q",
            _bwd.launch_redsp_q,
            (dqp, dq, n_vec, nsp_q, -(-n_vec // _bwd.RED_THREADS), side),
        )

    # Join: whatever the caller issues next on its stream is ordered after the dQ chain.
    stream.wait_stream(side)
    return dq, dk, dv
