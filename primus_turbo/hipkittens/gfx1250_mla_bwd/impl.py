###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hk_attn_bwd: the gfx1250 MLA backward with HIP dK/dV and dQ kernels.

Three kernel launches, all on the caller's current stream, in this order:
  "delta"  HIP k_delta: delta, and the per-row constants the dK/dV kernel reads (64-query workgroups).
           With the manifest's ``nd_fold`` (the generated k_dkdv96ds starts its dP chains from -delta and
           its softmax yields scale * P): NL = -LOG2E*lse + l2s, ND = -delta, l2s = log2(scale) rounded to
           fp32 on the host (k_delta's f32 argument). The FlyDSL k_delta of
           ``primus_turbo.flydsl.attention.gfx1250_mla`` (NL = -LOG2E*lse, ND = -scale*delta, delta bitwise
           equal) runs in its place when bin/k_delta.hsaco is missing or not the pinned one (k_delta is
           the manifest's one ``optional`` kernel), followed by the same fold on the host (``_fold_nl_nd``:
           one fp32 add, one negation), so NL / ND and the gradients are the same bits either way;
  "dkdv"   HIP k_dkdv96ds: dK and dV (96-row kv blocks of two 48-row waves on one 3-stage Q/dO TDM ring,
           then 64-row tail blocks), and every bf16 dS^T tile it feeds the dK WMMA (softmax scale
           included) stored into this call's scratch in dslayout.h v1. With the manifest's ``ticket``
           (-DHKGEN_TICKET=1) a workgroup takes its kv block from a ticket counter (the int 64 B in front of
           the dS layout, zeroed on the host by the "ticket" step right before the launch) instead of its
           block id, in the grid's own order: the dispatcher deals a fixed number of blocks to each XCD, and
           on a slower XCD they start later and take the later, shorter blocks instead of that XCD finishing
           last. The block-to-workgroup assignment only changes which workgroup computes a block, not what
           it computes;
  "dq"     HIP k_dqds_c: dQ = dS K read back from that scratch (384-query workgroups of 4 waves sharing
           one K ring, plus one merged 256-query head tile), so S and dP are never recomputed.
k_dqds_c reads what k_dkdv96ds writes: there is no side stream. Every output element is written by
exactly one workgroup (the ticket's atomic only hands out work items), so the results are deterministic
and do not depend on the head-group targets.

Scratch: guard | layout | guard bf16, from the torch caching allocator on the current stream and
referenced by this call only (4.33 GB at b2 s4096 h128, ``limits.ds_layout_bytes``); when the call
returns the block goes back to the allocator, stream-ordered behind the two kernels that use it.

Scratch budget (``max_scratch_bytes`` / ``MAX_SCRATCH_BYTES``, default None = no limit): when the
whole call's scratch (guards included) exceeds it, the call runs in head chunks: "delta" once, then
("dkdv", "dq") per chunk of hc consecutive heads of one batch, every chunk on the SAME scratch of hc
regions (chunk c's dkdv overwrites it only after chunk c-1's dq, stream order). hc is the largest
multiple of the head group dividing Hq whose scratch fits. A chunk launch is the full launch's
argument list (Hq / Hkv unchanged, so every head and row stride stays right) with B = 1, the grid z
cut to hc / head group, and q/k/v/do/nl/nd/dq/dk/dv passed from the chunk's first (batch, head):
each workgroup computes what the full launch's workgroup of that head computes, so dq/dk/dv are
bitwise identical to the unchunked call (two chunks at b2 s4096 h128: 2.16 GB, +0.75% time). When
not even one head group fits, the call runs the FlyDSL backward instead (same results).
"""

import math
import os
import struct
import threading

import torch

from primus_turbo.common.logger import logger
from primus_turbo.flydsl.attention.gfx1250_mla import bwd as _fly

from . import hiprt, hsaco, limits

_k = _fly._k  # the FlyDSL backward's kernels module: k_delta and the head-dim constants

# Head-group traversal targets (_head_group): launch grids (hg, tiles, B*H/hg). Launch-only,
# outputs bitwise identical for every value the kernels' index proofs replay (the tuples below).
HEAD_GROUP_KV = 32  # k_dkdv96ds
HEAD_GROUP_Q = 32  # k_dqds_c
PROVEN_KV_TARGETS = (0, 8, 16, 24, 32, 64, 96, 128, 256)
PROVEN_Q_TARGETS = (0, 8, 16, 32, 64, 128, 256)
DS_MODE = 1  # k_dkdv96ds: store dS^T with num_records = the wave's true extent (0 drops every store)
DELTA_ROWS = 64  # k_delta queries per workgroup (grid (Sq / DELTA_ROWS, B*Hq), block DELTA_BLOCK)
DELTA_BLOCK = 256
DS_GUARD_B = 65536  # bytes before and after the layout in the scratch
MAX_SCRATCH_BYTES_ENV = "PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES"


def _bytes_setting(name):
    """A positive byte count from the environment (int or float syntax, e.g. 2.2e9), or None if unset."""
    text = os.environ.get(name, "").strip()
    if not text:
        return None
    try:
        n = int(float(text))
    except ValueError:
        n = 0
    if n <= 0:
        raise ValueError(f"{name}={text!r}: expected a positive byte count")
    return n


# dS scratch budget of one call in bytes (guards included), read at import like the adapter's switches;
# None = no limit (one launch pair, the default). hk_attn_bwd(max_scratch_bytes=...) overrides it per call.
MAX_SCRATCH_BYTES = _bytes_setting(MAX_SCRATCH_BYTES_ENV)


class ScratchBudgetError(NotImplementedError):
    """Not even one head group's dS scratch fits the budget (hk_attn_bwd then runs the FlyDSL backward)."""


# Validation only: allocate delta/dq/dk/dv and the whole dS scratch NaN-filled, so an element the
# kernels never write shows up as non-finite instead of as stale memory.
POISON = False

_KERNELS = None
_LOCK = threading.Lock()
_DEVICES_CHECKED = set()


def _head_group(nh, target):
    """Head group of a launch with nh heads per batch: nh itself when nh <= target or target <= 0;
    else the largest multiple of 8 that divides nh and is <= target (a multiple of 8 keeps every
    head's workgroups on one XCD under the round-robin linear-id dispatch), nh when there is none."""
    if target <= 0 or nh <= target:
        return nh
    for d in range(target - target % 8, 7, -8):
        if nh % d == 0:
            return d
    return nh


def _scratch_bytes(b, sq, skv, nh, causal):
    """Bytes _ds_scratch allocates for b batches x nh heads: the layout plus both guards."""
    return limits.ds_layout_bytes(b, sq, skv, nh, causal) + 2 * DS_GUARD_B


def _chunk_heads(b, sq, skv, hq, causal, step, max_bytes):
    """Heads per chunk under a scratch budget: None = no chunking (no budget, or the whole call fits);
    else the largest multiple of ``step`` that divides hq and whose one-batch scratch fits, 0 if none."""
    if max_bytes is None or _scratch_bytes(b, sq, skv, hq, causal) <= max_bytes:
        return None
    for hc in range(hq - hq % step, 0, -step):
        if hq % hc == 0 and _scratch_bytes(1, sq, skv, hc, causal) <= max_bytes:
            return hc
    return 0


def _check_device_once(device):
    """Raise unless ``device`` is a gfx1250 (the code objects are gfx1250 wave32 WMMA code)."""
    if device in _DEVICES_CHECKED:
        return
    arch = torch.cuda.get_device_properties(device).gcnArchName
    if "gfx1250" not in arch:
        raise RuntimeError(f"the HIP MLA backward kernels cannot run on {arch}")
    logger.debug(
        f"gfx1250 HIP MLA backward: HEAD_GROUP_KV={HEAD_GROUP_KV} HEAD_GROUP_Q={HEAD_GROUP_Q}", once=True
    )
    _DEVICES_CHECKED.add(device)


def _check(do, q, k, v, o, lse):
    """Shape/dtype/layout contract of the kernels; returns (b, sq, skv, hq, hkv)."""
    b, sq, hq, dqk = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    if (dqk, v.shape[-1]) != (_k.D_QK, _k.D_V):
        raise ValueError(f"head dims must be qk {_k.D_QK} / v {_k.D_V}, got {dqk} / {v.shape[-1]}")
    if k.shape != (b, skv, hkv, _k.D_QK) or v.shape != (b, skv, hkv, _k.D_V):
        raise ValueError(f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)} do not match")
    if o.shape != (b, sq, hq, _k.D_V) or do.shape != o.shape or lse.shape != (b, hq, sq):
        raise ValueError(
            f"out/dout/lse shapes {tuple(o.shape)}/{tuple(do.shape)}/{tuple(lse.shape)} do not match q"
        )
    if sq % _k.ROWS_DELTA:
        raise ValueError(f"seqlen_q must be a multiple of {_k.ROWS_DELTA} (k_delta query blocks), got {sq}")
    for name, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
        if not t.is_contiguous() or t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a contiguous bfloat16 tensor")
    if b * sq * hq * _k.D_V * 2 >= (1 << 31):
        raise ValueError("o/do byte extent overflows k_delta's int32 size")
    return b, sq, skv, hq, hkv


def kernels():
    """{name: hiprt.Kernel} (code objects sha256-checked on first use; no GPU work here)."""
    global _KERNELS
    if _KERNELS is None:
        with _LOCK:
            if _KERNELS is None:
                _KERNELS = hiprt.load_kernels()
    return _KERNELS


def _alloc(shape, device, dtype):
    if POISON:
        return torch.full(shape, float("nan"), device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype)


def _nd_fold():
    """The manifest's nd_fold: k_dkdv96ds expects the folded NL / ND (see the module docstring)."""
    return bool(hsaco.load_manifest().get("nd_fold", False))


def _ticket():
    """The manifest's ticket: k_dkdv96ds takes its work items from a ticket counter (see the module docstring)."""
    return bool(hsaco.load_manifest().get("ticket", False))


def nd_fold_offset(scale):
    """log2(softmax scale) rounded to fp32: the HIP k_delta's f32 argument and the FlyDSL-delta fold's addend."""
    return struct.unpack("f", struct.pack("f", math.log2(scale)))[0]


def _fold_nl_nd(nl, nd, delta, l2s):
    """After the FlyDSL k_delta (NL = -LOG2E lse, ND = -scale delta): the folded constants the HIP k_delta writes
    (NL + l2s, -delta), bit for bit: the same fp32 add, a negation."""
    nl.add_(l2s)
    torch.neg(delta, out=nd)


def _zero_ticket(ds_full):
    """k_dkdv96ds's ticket counter (the int 64 B in front of the dS layout) = 0, before every dkdv launch."""
    ds_full.view(torch.int32)[(DS_GUARD_B - 64) // 4 : DS_GUARD_B // 4].zero_()


HOST_STEPS = (_fold_nl_nd, _zero_ticket)  # launches that are torch ops on the host side, not kernels


def _hk_delta_reason(sq):
    """None when the HIP k_delta can take seqlen_q ``sq`` and its code object is the pinned one, else why not."""
    if sq % DELTA_ROWS:
        return f"seqlen_q {sq} is not a multiple of {DELTA_ROWS}"
    return hsaco.kernel_unavailable_reason("k_delta")


def _ds_scratch(nbytes, device):
    t = torch.empty(((2 * DS_GUARD_B + nbytes) // 2,), dtype=torch.bfloat16, device=device)
    if POISON and t.device.type != "meta":
        t.fill_(float("nan"))
    return t, t[DS_GUARD_B // 2 : DS_GUARD_B // 2 + nbytes // 2]


def _plan(
    do, q, k, v, o, lse, softmax_scale, causal, stream, head_group=None, max_scratch_bytes=None, hk_delta=None
):
    """Every launch hk_attn_bwd makes for these inputs, in issue order, plus what it allocates.

    launches: [(name, launcher, args)]: "delta" (+ "fold" after the FlyDSL k_delta with nd_fold), then
    ("ticket" with the manifest's ticket) "dkdv" / "dq" once, or once per head chunk; a HIP launch is (kernel
    name, (grid, block, kernel argument values)), a FlyDSL one (launcher, its arguments), a host step (one of
    HOST_STEPS, its arguments).
    hk_delta: "delta" on the HIP k_delta (True) or the FlyDSL k_delta (False); None: HIP when
    _hk_delta_reason(sq) is None. head_group:
    target for both kernels (None: HEAD_GROUP_KV / HEAD_GROUP_Q); one the index proofs do not replay
    is refused (ValueError). max_scratch_bytes: dS scratch budget (None: MAX_SCRATCH_BYTES); over it
    the plan is head-chunked (module docstring), and ScratchBudgetError when no head group fits.
    Works on meta tensors (stream unused until launch).
    """
    b, sq, skv, hq, hkv = _check(do, q, k, v, o, lse)
    c = int(bool(causal))
    reason = limits.shape_unsupported_reason(b, sq, skv, hq, hkv, c)
    if reason is not None:
        raise NotImplementedError(f"hk_attn_bwd: {reason}")
    tkv = HEAD_GROUP_KV if head_group is None else int(head_group)
    tq = HEAD_GROUP_Q if head_group is None else int(head_group)
    if tkv not in PROVEN_KV_TARGETS or tq not in PROVEN_Q_TARGETS:
        raise ValueError(
            f"hk_attn_bwd: head-group targets kv {tkv} / q {tq} are not proven "
            f"(kv {PROVEN_KV_TARGETS}, q {PROVEN_Q_TARGETS})"
        )
    hg_kv, hg_q = _head_group(hkv, tkv), _head_group(hq, tq)
    assert hkv % hg_kv == 0 and hq % hg_q == 0, (hkv, hg_kv, hq, hg_q)
    ngz_kv, ngz_q = b * hkv // hg_kv, b * hq // hg_q
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(_k.D_QK)
    scale = float(softmax_scale)
    g = hq // hkv
    budget = MAX_SCRATCH_BYTES if max_scratch_bytes is None else int(max_scratch_bytes)
    hc = _chunk_heads(b, sq, skv, hq, c, math.lcm(hg_kv, hg_q), budget)
    if hc == 0:
        raise ScratchBudgetError(
            f"hk_attn_bwd: the dS scratch of {math.lcm(hg_kv, hg_q)} heads "
            f"({_scratch_bytes(1, sq, skv, math.lcm(hg_kv, hg_q), c)} B) exceeds max_scratch_bytes {budget}"
        )
    dev = q.device
    delta = _alloc((b, hq, sq), dev, torch.float32)
    nl = _alloc((b, hq, sq), dev, torch.float32)
    nd = _alloc((b, hq, sq), dev, torch.float32)
    dq = _alloc((b, sq, hq, _k.D_QK), dev, q.dtype)
    dk = _alloc((b, skv, hkv, _k.D_QK), dev, k.dtype)
    dv = _alloc((b, skv, hkv, _k.D_V), dev, v.dtype)
    if hc is None:
        chunks = None
        parts = [((q, k, v, do, nl, nd, dv, dk, dq), b, ngz_kv, ngz_q)]
        ds_bytes = limits.ds_layout_bytes(b, sq, skv, hq, c)
    else:
        # head chunks of one batch: views from the chunk's first (batch, head), B = 1 (MHA: k/v heads = q heads)
        assert g == 1 and hc % hg_kv == 0 and hc % hg_q == 0, (g, hc, hg_kv, hg_q)
        chunks = [(bi, h0, hc) for bi in range(b) for h0 in range(0, hq, hc)]
        parts = []
        for bi, h0, _ in chunks:
            bshd = (slice(bi, bi + 1), slice(None), slice(h0, h0 + hc))  # [b, s, h, d] tensors
            bhs = (slice(bi, bi + 1), slice(h0, h0 + hc))  # [b, h, s] tensors
            views = (
                tuple(t[bshd] for t in (q, k, v, do))
                + (nl[bhs], nd[bhs])
                + tuple(t[bshd] for t in (dv, dk, dq))
            )
            parts.append((views, 1, hc // hg_kv, hc // hg_q))
        ds_bytes = limits.ds_layout_bytes(1, sq, skv, hc, c)
    ds_full, ds = _ds_scratch(ds_bytes, dev)
    q_off, nhead, ntile = limits.dq_split(sq)
    if hk_delta is None:
        hk_delta = _hk_delta_reason(sq) is None
    elif hk_delta and sq % DELTA_ROWS:
        raise ValueError(f"hk_attn_bwd: the HIP k_delta needs seqlen_q % {DELTA_ROWS} == 0, got {sq}")
    if hk_delta:
        delta_grid = (sq // DELTA_ROWS, b * hq, 1)
        delta_values = (do, o, delta, lse, nl, nd, nd_fold_offset(scale) if _nd_fold() else scale, sq, hq)
        launches = [("delta", "k_delta", (delta_grid, DELTA_BLOCK, delta_values))]
    else:
        delta_grid = (sq // _k.ROWS_DELTA, b * hq, 1)
        delta_args = (
            do,
            o,
            delta,
            lse,
            nl,
            nd,
            scale,
            sq,
            hq,
            b * sq * hq,
            sq // _k.ROWS_DELTA,
            b * hq,
            stream,
        )
        launches = [("delta", _k.launch_delta, delta_args)]
        if _nd_fold():
            launches.append(("fold", _fold_nl_nd, (nl, nd, delta, nd_fold_offset(scale))))
    for (q_, k_, v_, do_, nl_, nd_, dv_, dk_, dq_), b_, gz_kv, gz_q in parts:
        dkdv_values = (
            q_,
            k_,
            v_,
            do_,
            nl_,
            nd_,
            dv_,
            dk_,
            ds,
            scale,
            sq,
            skv,
            hq,
            hkv,
            g,
            sq // 16,
            skv - sq,
            c,
            b_,
            hg_kv,
            DS_MODE,
        )
        dq_values = (ds, k_, dq_, sq, skv, hq, hkv, g, skv // 32, skv - sq, c, q_off, ntile, hg_q)
        if _ticket():
            launches.append(("ticket", _zero_ticket, (ds_full,)))
        launches.append(("dkdv", "k_dkdv96ds", ((hg_kv, limits.k3_nblk(skv), gz_kv), 64, dkdv_values)))
        launches.append(("dq", "k_dqds_c", ((hg_q, ntile + nhead, gz_q), 128, dq_values)))
    return {
        "launches": launches,
        "outputs": (dq, dk, dv),
        "delta": delta,
        "nl": nl,
        "nd": nd,
        "grids": {
            "delta": delta_grid,
            "dkdv": next(l for l in launches if l[0] == "dkdv")[2][0],
            "dq": next(l for l in launches if l[0] == "dq")[2][0],
        },
        "head_group": {"kv": hg_kv, "q": hg_q},
        "chunks": chunks,  # None, or [(batch, first head, heads)] of every (dkdv, dq) launch pair
        "ds_layout_bytes": ds_bytes,
        "ds_full": ds_full,
    }


def hk_attn_bwd(
    do, q, k, v, o, lse, softmax_scale=None, causal=True, head_group=None, max_scratch_bytes=None
):
    """Flash-attention backward on gfx1250, DeepSeek-V3 MLA head dims (qk 192, v 128), MHA.

    Same contract as ``primus_turbo.flydsl.attention.gfx1250_mla.bwd.flash_attn_mla_bwd``:
    q [B, Sq, H, 192], o/do [B, Sq, H, 128], k [B, Skv, H, 192], v [B, Skv, H, 128], bf16 contiguous;
    lse [B, H, Sq] fp32 natural log; bottom-right causal. Returns (dq, dk, dv) laid out like q/k/v.
    Also needs Sq % 384 in {0, 256} and Skv >= Sq (``limits.shape_unsupported_reason``).
    max_scratch_bytes: dS scratch budget of this call (None: MAX_SCRATCH_BYTES, unset = no limit);
    over it the call runs in head chunks, and on the FlyDSL backward when no head group fits.
    """
    _check_device_once(q.device)
    lse = lse.contiguous().float()
    stream = torch.cuda.current_stream()
    table = kernels()
    hk_delta = "k_delta" in table
    if not hk_delta:
        reason = hsaco.kernel_unavailable_reason("k_delta") or "k_delta is not loaded"
        logger.warning(f"gfx1250 HIP MLA backward: {reason}: delta runs on the FlyDSL k_delta", once=True)
    try:
        plan = _plan(
            do, q, k, v, o, lse, softmax_scale, causal, stream, head_group, max_scratch_bytes, hk_delta
        )
    except ScratchBudgetError as e:
        logger.warning(f"{e}: this backward runs on FlyDSL", once=True)
        scale = 1.0 / math.sqrt(_k.D_QK) if softmax_scale is None else float(softmax_scale)
        return _fly.flash_attn_mla_bwd(do, q, k, v, o, lse, scale, causal)
    for _, fn, args in plan["launches"]:
        if isinstance(fn, str):
            grid, block, values = args
            table[fn].call(grid, block, values, stream)
        elif fn in HOST_STEPS:
            fn(*args)
        else:
            _fly._launch(fn, args)
    return plan["outputs"]
