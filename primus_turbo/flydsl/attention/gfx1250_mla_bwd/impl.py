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

"""flydsl_attn_bwd: the gfx1250 FlyDSL backward (k_delta, k_dkdv64, k_dqg96 + k_dqg in kernels.py;
k_dkdv + one k_dqg on grids too small to give every SIMD a wave: _geometry)."""

import math
import os

import flydsl.compiler as _flyc
import torch

from primus_turbo.common.logger import logger

from . import kernels as _k

# FlyDSL's @flyc.jit __call__ re-derives the whole cache key on every launch (~0.27 ms of
# host time per call). flyc.compile(launcher, *args) performs the first launch and returns
# a CompiledFunction whose __call__ only refreshes the ctypes argument storage. Every
# shape-dependent argument is a runtime fx.Int32, so one compiled object serves every
# shape; the memo key (dtype and rank of each tensor, device) keeps a different dtype or
# layout from reusing it.
_COMPILED = {}


def _launch(name, launcher, args):
    """Launch `launcher(*args)`, via flyc.compile's fast path after the first call."""
    # id(launcher): one plan name may map to different launchers by shape (k_dqg96 vs k_dqg).
    key = (
        name,
        id(launcher),
        tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor)),
        args[0].device.index,
    )
    fn = _COMPILED.get(key)
    if fn is not None:
        fn(*args)
        return
    # flyc.compile() ISSUES this launch itself, so it must not be repeated here.
    fn = _flyc.compile(launcher, *args)
    if fn is None:  # COMPILE_ONLY builds return None
        launcher(*args)
        return
    _COMPILED[key] = fn


_DEVICES_CHECKED = set()

# The dQ chain (k_dqg) runs on the caller's stream AFTER the dK/dV chain (k_dkdv) by default.
# It may instead run on a SIDE stream concurrently with k_dkdv: both only read
# q/k/v/do/lse/delta (delta is written by k_delta on the main stream BEFORE the fork) and
# write disjoint outputs, so the fork is legal and the outputs are bitwise identical to the
# serial order. For MHA at DeepSeek-V3 shapes each kernel alone fills every SIMD (prod: 32768
# one-wave workgroups each), and running the two concurrently was 28% SLOWER than serial at
# b2 s4096 h128 (7.06 vs 5.52 ms) and no faster at b1 s4096 h64; the side stream paid off
# only for the hd128 GQA shape it was built for. Env switches, read once at import:
#   FLY_BWD_SIDE_STREAM=1    side stream: k_dqg concurrently with k_dkdv.
#   FLY_BWD_RECORD_STREAM=0  with the side stream, skip the record_stream() calls at the
#                            join (every block s2 touches was allocated on the caller's
#                            stream, which waits for s2 before we return).
DQ_SIDE_STREAM = os.environ.get("FLY_BWD_SIDE_STREAM", "0") != "0"
DQ_SIDE_RECORD = os.environ.get("FLY_BWD_RECORD_STREAM", "1") != "0"
# Head-group traversal target of k_dkdv64 / k_dqg96 / k_dqg (kernels.HEAD_GROUP note): a launch
# with more than HEAD_GROUP heads per batch runs as head groups of kernels.head_group(nh, HEAD_GROUP)
# heads. Launch-only (a runtime kernel argument, one binary for every value): FLY_BWD_HEAD_GROUP=0
# restores r3_a's grid for every shape; flydsl_attn_bwd(..., head_group=N) overrides it per call
# (in-process A/B controls).
# Default 64 (r5_hg64 / r5.i4.g18, PROVENANCE.md): prod b2h128 launches (64, tiles, 4), twice the
# resident tiles per head of target 128 (kernels.HEAD_GROUP, r4_a's default); nh <= 64 (proxy, fast,
# toy) launches exactly as before. The default lives here, not in kernels.py: kernels.py is device
# source, and a launch-only change should leave its text (and so every kernel's JIT key) alone.
HEAD_GROUP_DEFAULT = 64
HEAD_GROUP = int(os.environ.get("FLY_BWD_HEAD_GROUP", str(HEAD_GROUP_DEFAULT)))
_SIDE = {}

# Small-grid fallback (bwd_r4_c's host rule, per chain, on the head-group launches). The two-wave
# kernels (k_dkdv64, k_dqg96) put two waves of one workgroup on one CU to halve the TDM bytes per
# FLOP; on a small grid the fetch bandwidth is idle and the pairing only concentrates the work on
# fewer CUs (and the dQ chain pays a second launch for its head). A chain whose ONE-wave kernel has
# fewer than SMALL_GRID_WAVES[chain] workgroups launches that kernel instead: k_dkdv (nw = 1, grid
# (Hkv, Skv/32, B); no head group) for dK/dV, ONE k_dqg over [0, Sq) (grid (hg, Sq/32, B*Hq/hg); the
# binary _plan launches over the head [0, q_split)) for dQ. Thresholds from same-process A/Bs at
# b1 MHA (one-wave workgroups 256 / 512 / 1024 / 2048 per chain; one-wave / two-wave kernel time):
# k_dkdv 0.91 / 0.94 / 1.12 / 1.29, dQ chain 0.63 / 0.64-0.72 / 0.80 / 1.04, so k_dkdv switches
# below 1024 and the dQ chain below 2048. Each chain decides from B*H*S alone, so the fold launch
# [1, S, B*H, D] and the [B, S, H, D] launch of the same tensors decide alike; Megatron's prod fold
# (32768 per chain) keeps the two-wave launches. FLY_BWD_SMALL_GRID=0 turns the fallback off;
# flydsl_attn_bwd(..., small_grid=True/False) forces one set for both chains per call (A/B controls,
# tests). Pure Python from here to _geometry's end (no torch): bounds_proof.py execs it.
N_CU = 256  # MI455X (gfx1250) compute units, 4 SIMDs each
SMALL_GRID_WAVES = {"dkdv": 4 * N_CU, "dq": 8 * N_CU}
if os.environ.get("FLY_BWD_SMALL_GRID", "1") == "0":
    SMALL_GRID_WAVES = {"dkdv": 0, "dq": 0}


def _geometry(b, sq, skv, hq, hkv, small_grid=None):
    """Launch set of both gradient chains for these sizes.

    dkdv: (nw, nblk): nw = DKDV_NW -> k_dkdv64 (nblk = Skv/64), nw = 1 -> k_dkdv (nblk = Skv/32).
    dq:   [(nqw, nwave, q_off, ntile)] in issue order: (NQW48, DQ_NWAVE) -> k_dqg96 (96-query
          tiles), (NQW, 1) -> k_dqg (32-query tiles); the tiles partition [0, Sq).
    small: (dkdv_small, dq_small). small_grid None decides each chain by SMALL_GRID_WAVES,
    True / False forces both chains to the one-wave / two-wave set.
    """
    if small_grid is None:
        dkdv_small = b * hkv * (skv // _k.BLOCK_KV) < SMALL_GRID_WAVES["dkdv"]
        dq_small = b * hq * (sq // _k.DQ_BQW) < SMALL_GRID_WAVES["dq"]
    else:
        dkdv_small = dq_small = bool(small_grid)
    nw = 1 if dkdv_small else _k.DKDV_NW
    if dq_small:
        dq = [(_k.NQW, 1, 0, sq // _k.DQ_BQW)]
    else:
        q_split, n32, n96 = _k.dq_split(sq)
        dq = ([(_k.NQW48, _k.DQ_NWAVE, q_split, n96)] if n96 else []) + ([(_k.NQW, 1, 0, n32)] if n32 else [])
    return {"dkdv": (nw, skv // (_k.BLOCK_KV * nw)), "dq": dq, "small": (dkdv_small, dq_small)}


# Validation only: allocate delta/dq/dk/dv NaN-filled, so an element the kernels never write
# shows up as non-finite instead of as stale memory.
POISON = False


def _alloc(shape, device, dtype):
    if POISON:
        return torch.full(shape, float("nan"), device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype)


def _check_device_once(device):
    """Raise unless `device` is a gfx1250 (the kernels are wave32 WMMA code for it); log the knobs."""
    if device in _DEVICES_CHECKED:
        return
    arch = torch.cuda.get_device_properties(device).gcnArchName
    if "gfx1250" not in arch:
        raise RuntimeError(f"the gfx1250 MLA backward kernels cannot run on {arch}")
    logger.debug(
        f"gfx1250 MLA backward: DQ_SIDE_STREAM={int(DQ_SIDE_STREAM)} DQ_SIDE_RECORD={int(DQ_SIDE_RECORD)} "
        f"HEAD_GROUP={HEAD_GROUP} SMALL_GRID_WAVES={SMALL_GRID_WAVES}",
        once=True,
    )
    _DEVICES_CHECKED.add(device)


def _side_stream(dev):
    s = _SIDE.get(dev)
    if s is None:
        s = torch.cuda.Stream(device=dev)
        _SIDE[dev] = s
    return s


def _check(do, q, k, v, o, lse):
    """Shape/dtype/layout contract of the kernels; returns (b, sq, skv, hq, hkv)."""
    b, sq, hq, dqk = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    dv = v.shape[-1]
    assert (dqk, dv) == (_k.D_QK, _k.D_V), (
        f"these kernels are head dims (qk {_k.D_QK}, v {_k.D_V}) only, got ({dqk}, {dv})"
    )
    assert k.shape == (b, skv, hkv, dqk) and v.shape == (b, skv, hkv, dv), (q.shape, k.shape, v.shape)
    assert o.shape == (b, sq, hq, dv) and do.shape == o.shape, (q.shape, o.shape, do.shape)
    assert lse.shape == (b, hq, sq), f"lse must be [B, Hq, Sq], got {tuple(lse.shape)}"
    assert hq % hkv == 0, f"heads_q {hq} is not a multiple of heads_kv {hkv}"
    # k_dkdv consumes query PAIRS of 32 rows; k_dqg tiles [0, q_split) by DQ_BQW and k_dqg96
    # [q_split, sq) by DQ_BQW96 (kernels.dq_split): a remainder would leave rows uncomputed and
    # write past the end.
    assert sq % 64 == 0 and sq % _k.DQ_BQW == 0, f"seqlen_q must be a multiple of 64, got {sq}"
    # k_dkdv64 consumes DKDV_NW*BLOCK_KV = 64-row kv blocks (one 32-row tile per wave).
    assert skv % _k.KV_STEP == 0 and skv % (_k.BLOCK_KV * _k.DKDV_NW) == 0, (
        f"seqlen_kv must be a multiple of {_k.BLOCK_KV * _k.DKDV_NW}, got {skv}"
    )
    n_rows = b * sq * hq
    assert n_rows % _k.ROWS_DELTA == 0 and sq % _k.ROWS_DELTA == 0, (
        f"seqlen_q must be a multiple of {_k.ROWS_DELTA} (k_delta query blocks), got {sq}"
    )
    for name, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
        assert t.is_contiguous(), f"{name} must be contiguous"
        assert t.dtype == torch.bfloat16, f"{name} must be bf16, got {t.dtype}"
    # Byte extents the kernels compute in 32-bit arithmetic or hard-code as descriptor
    # num_records (k_dqg: 1 GiB for q/do/dq, 256 MiB for lse/delta).
    assert max(q.numel(), o.numel()) * 2 <= (1 << 30), "q/do/dq larger than k_dqg's 1 GiB descriptor extent"
    assert lse.numel() * 4 <= (1 << 28), "lse/delta larger than k_dqg's 256 MiB descriptor extent"
    assert max(k.numel(), v.numel()) * 2 < (1 << 31), (
        "k/v byte extent overflows k_dkdv's int32 descriptor size"
    )
    assert n_rows * _k.D_V * 2 < (1 << 31), "o/do byte extent overflows k_delta's int32 size"
    return b, sq, skv, hq, hkv


def _plan(
    do, q, k, v, o, lse, softmax_scale, causal, stream, dq_stream=None, head_group=None, small_grid=None
):
    """Every launch flydsl_attn_bwd makes for these inputs, in issue order, plus the
    tensors it allocates. Used verbatim by the launcher below and by compile-only builds
    (meta tensors, stream=None, COMPILE_ONLY=1), so the compiled set is exactly the
    launched set.

    launches: [(name, launcher, args, chain)], chain "main" (caller's stream) or "dq".
    head_group: target of kernels.head_group (None: HEAD_GROUP; 0: r3_a's grids).
    small_grid: _geometry's override (None: SMALL_GRID_WAVES decides per chain).
    """
    b, sq, skv, hq, hkv = _check(do, q, k, v, o, lse)
    tgt = HEAD_GROUP if head_group is None else int(head_group)
    # head groups: grid (hg, tiles, b*nh/hg), the same workgroup count as (nh, tiles, b)
    hg_kv, hg_q = _k.head_group(hkv, tgt), _k.head_group(hq, tgt)
    assert hkv % hg_kv == 0 and hq % hg_q == 0, (hkv, hg_kv, hq, hg_q)
    ngz_kv, ngz_q = b * hkv // hg_kv, b * hq // hg_q
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(_k.D_QK)
    g = hq // hkv
    n_rows = b * sq * hq
    c = int(bool(causal))
    dq_stream = stream if dq_stream is None else dq_stream
    delta = _alloc((b, hq, sq), q.device, torch.float32)
    # k_dkdv64's / k_dkdv's per-row softmax constants, written by k_delta (r5.i3.g17): -LOG2E*lse,
    # -scale*delta
    nl = _alloc((b, hq, sq), q.device, torch.float32)
    nd = _alloc((b, hq, sq), q.device, torch.float32)
    dq = _alloc((b, sq, hq, _k.D_QK), q.device, q.dtype)
    dk = _alloc((b, skv, hkv, _k.D_QK), k.device, k.dtype)
    dv = _alloc((b, skv, hkv, _k.D_V), v.device, v.dtype)
    geo = _geometry(b, sq, skv, hq, hkv, small_grid)
    nw, nblk = geo["dkdv"]
    dkdv_args = (
        q,
        k,
        v,
        do,
        nl,
        nd,
        dv,
        dk,
        float(softmax_scale),
        sq,
        skv,
        hq,
        hkv,
        g,
        sq // 16,
        skv - sq,
        c,
        nblk,
    )
    launches = [
        (
            "delta",
            _k.launch_delta,
            (
                do,
                o,
                delta,
                lse,
                nl,
                nd,
                float(softmax_scale),
                sq,
                hq,
                n_rows,
                sq // _k.ROWS_DELTA,
                b * hq,
                stream,
            ),
            "main",
        ),
    ]
    grids = {"delta": (sq // _k.ROWS_DELTA, b * hq, 1)}
    if nw == _k.DKDV_NW:
        # k_dkdv64: 2 waves per 64-row kv block share one Q/dO ring (bwd_r2_a)
        launches.append(("dkdv", _k.launch_dkdv64, dkdv_args + (hg_kv, ngz_kv, b, stream), "main"))
        grids["dkdv"] = (hg_kv, nblk, ngz_kv)
    else:
        # small grid: the one-wave k_dkdv, one 32-row kv tile per workgroup, grid (Hkv, Skv/32, B)
        launches.append(("dkdv", _k.launch_dkdv, dkdv_args + (hkv, b, stream), "main"))
        grids["dkdv"] = (hkv, nblk, b)
    # dQ chain: k_dqg96 (2 waves x 48 queries on one K/V ring, bwd_r2_b) over [q_split, sq),
    # longest-first, then the 32-query k_dqg over the head [0, q_split) (the shortest tiles); on a
    # small grid one k_dqg launch over [0, sq). Plan name "dqg" is the main dQ launch (k_dqg96; k_dqg
    # when there is no 96-query part), "dqg_head" the head launch. Both are single-kernel modules, so
    # the compile-only dump/ISA table sees each kernel on its own.
    q_split, n32, n96 = _k.dq_split(sq)
    assert q_split + n96 * _k.DQ_BQW96 == sq and n32 * _k.DQ_BQW == q_split, (sq, q_split, n32, n96)
    if geo["small"][1]:
        q_split, n32, n96 = sq, sq // _k.DQ_BQW, 0  # k_dqg covers [0, sq)
    dq_args = (
        q,
        k,
        v,
        do,
        o,
        lse,
        delta,
        dq,
        float(softmax_scale),
        sq,
        skv,
        hq,
        hkv,
        g,
        skv // _k.KV_STEP,
        skv - sq,
        c,
    )
    for i, (_, nwave, q_off, ntile) in enumerate(geo["dq"]):
        nm = "dqg" if i == 0 else "dqg_head"
        fn = _k.launch_dqg96 if nwave == _k.DQ_NWAVE else _k.launch_dqg
        launches.append((nm, fn, dq_args + (q_off, ntile, hg_q, ngz_q, dq_stream), "dq"))
        grids[nm] = (hg_q, ntile, ngz_q)
    return {
        "launches": launches,
        "outputs": (dq, dk, dv),
        "delta": delta,
        "nl": nl,
        "nd": nd,
        "grids": grids,
        "dq_split": (q_split, n32, n96),
        "head_group": {"kv": hg_kv, "q": hg_q, "target": tgt},
        "small_grid": geo["small"],
    }


def flydsl_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True, head_group=None, small_grid=None):
    """Flash-attention backward on gfx1250.

    q [B, Sq, Hq, D_QK], o/do [B, Sq, Hq, D_V], k [B, Skv, Hkv, D_QK], v [B, Skv, Hkv, D_V],
    all bf16 and contiguous (DeepSeek-V3 MLA: D_QK = 192, D_V = 128).
    lse     [B, Hq, Sq] fp32, NATURAL log, as the gfx1250 forward emits it.
    Returns (dq, dk, dv) in q/k/v's dtype, laid out like q/k/v.

    causal is BOTTOM-RIGHT: query i attends keys j <= i + (Skv - Sq).
    head_group: launch traversal only (outputs bitwise identical for every value); None =
    HEAD_GROUP, 0 = r3_a's grids. See kernels.HEAD_GROUP.
    small_grid: None = the per-chain SMALL_GRID_WAVES rule (_geometry); True / False force the
    one-wave / two-wave launch set for both chains.
    """
    _check_device_once(q.device)
    lse = lse.contiguous().float()
    stream = torch.cuda.current_stream()
    split = DQ_SIDE_STREAM
    s2 = _side_stream(q.device) if split else stream
    plan = _plan(do, q, k, v, o, lse, softmax_scale, causal, stream, s2, head_group, small_grid)
    for name, fn, args, chain in plan["launches"]:
        _launch(name, fn, args)
        if name == "delta" and split:
            # fork: the dQ chain waits for everything already on the main stream (inputs,
            # lse.float(), k_delta) but not for k_dkdv, which is issued next.
            s2.wait_stream(stream)
    dq, dk, dv = plan["outputs"]
    if split:
        if DQ_SIDE_RECORD:
            for t in (q, k, v, do, o, lse, plan["delta"], dq):
                t.record_stream(s2)
        stream.wait_stream(s2)
    return dq, dk, dv


attn_bwd = flydsl_attn_bwd  # uniform name every loader uses
