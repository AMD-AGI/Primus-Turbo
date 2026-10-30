###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hk_attn_bwd: the gfx1250 MLA backward on the HIP kernels k_delta, k_dkdv96ds and k_dqds_c.

Launches, all on the caller's current stream, in this order:
  "delta"   k_delta: delta = rowsum(dO * O) and the row constants k_dkdv96ds reads, NL = -LOG2E * lse + l2s and
            ND = -delta (l2s = log2(scale) rounded to fp32 on the host): k_dkdv96ds starts its dP chains from ND and
            its softmax yields scale * P, so dS = (scale * P) * (dP - delta) is a single multiply;
  "ticket"  a host fill: zero k_dkdv96ds's work-item counter (an int in the scratch's front guard);
  "dkdv"    k_dkdv96ds: dK and dV over 96-row kv blocks of two 48-row waves on a 3-stage Q/dO TDM ring, then 64-row
            tail blocks; every bf16 dS^T tile it feeds the dK WMMA (softmax scale included) is also stored into this
            call's dS scratch. A workgroup takes its kv block from the ticket counter, longest blocks first: slower
            XCDs then take the later, shorter blocks instead of finishing last. The counter only hands out work;
            which workgroup computes a block does not change what it computes;
  "dq"      k_dqds_c: dQ = dS K read back from the scratch (384-query workgroups of 4 waves sharing one K ring, plus
            one merged 256-query head tile), so S and dP are never recomputed.
Every output element is written by exactly one workgroup: the results are deterministic and do not depend on the
head-group launch targets or on head chunking.

Scratch: guard | dS layout | guard, from the torch caching allocator on the current stream (4.33 GB at b2 s4096
h128, ``limits.ds_layout_bytes``); it returns to the allocator stream-ordered behind the kernels that use it. Under a
budget (``max_scratch_bytes``, default ``$PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES``, unset = no limit) the call runs
"delta" once, then ("ticket", "dkdv", "dq") per chunk of consecutive heads of one batch, every chunk on the same
scratch: a chunk launch is the full launch with B = 1, the grid's z cut to the chunk and the tensors offset to its
first (batch, head), so the gradients are bitwise those of the unchunked call (two chunks at b2 s4096 h128: 2.16 GB,
+0.75% time). When not even one head group fits, or the scratch cannot be allocated, the call runs the FlyDSL
backward.
"""

import math
import os
import struct
import threading

import torch

from primus_turbo.common.logger import logger

from . import hiprt, hsaco, limits

# Head-group traversal targets (``_head_group``): launch grids (hg, tiles, B*H/hg). Launch-only: the outputs are
# bitwise identical for every target the kernels' index arithmetic is verified for (the tuples below).
HEAD_GROUP_KV = 32  # k_dkdv96ds
HEAD_GROUP_Q = 32  # k_dqds_c
VERIFIED_KV_TARGETS = (0, 8, 16, 24, 32, 64, 96, 128, 256)
VERIFIED_Q_TARGETS = (0, 8, 16, 32, 64, 128, 256)
DS_MODE = 1  # k_dkdv96ds: store dS^T with num_records = the wave's true extent
DELTA_ROWS = 64  # k_delta queries per workgroup: grid (Sq / DELTA_ROWS, B*Hq), block DELTA_BLOCK
DELTA_BLOCK = 256
DS_GUARD_B = 65536  # bytes before and after the dS layout in the scratch
TICKET_OFFSET_B = DS_GUARD_B - 64  # the ticket counter, inside the front guard
MAX_SCRATCH_BYTES_ENV = "PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES"


def _bytes_setting(name):
    """A positive byte count from the environment (int or float syntax, e.g. 2.2e9), or None if unset."""
    text = os.environ.get(name, "").strip()
    if not text:
        return None
    try:
        n = int(float(text))
    except (ValueError, OverflowError):
        n = 0
    if n <= 0:
        raise ValueError(f"{name}={text!r}: expected a positive byte count")
    return n


# Read when this module is first imported (by the first HIP backward): None = no limit.
MAX_SCRATCH_BYTES = _bytes_setting(MAX_SCRATCH_BYTES_ENV)


class ScratchBudgetError(NotImplementedError):
    """Not even one head group's dS scratch fits the budget (hk_attn_bwd then runs the FlyDSL backward)."""


# Validation only: allocate delta / dq / dk / dv and the dS scratch NaN-filled, so an element the kernels never
# write shows up as non-finite.
POISON = False

_KERNELS = None
_LOCK = threading.Lock()


def kernels():
    """{name: hiprt.Kernel} of the backward (code objects sha256-checked on first use; no GPU work here)."""
    global _KERNELS
    if _KERNELS is None:
        with _LOCK:
            if _KERNELS is None:
                _KERNELS = hiprt.load_kernels(hsaco.BWD_KERNELS)
    return _KERNELS


def _head_group(nh, target):
    """Head group of a launch with nh heads per batch: nh when nh <= target or target <= 0, else the largest
    multiple of 8 that divides nh and is <= target (a multiple of 8 keeps every head's workgroups on one XCD under
    the round-robin dispatch), nh when there is none."""
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
    """Heads per chunk under a scratch budget: None = no chunking (no budget, or the whole call fits); else the
    largest multiple of ``step`` that divides hq and whose one-batch scratch fits, 0 if none."""
    if max_bytes is None or _scratch_bytes(b, sq, skv, hq, causal) <= max_bytes:
        return None
    for hc in range(hq - hq % step, 0, -step):
        if hq % hc == 0 and _scratch_bytes(1, sq, skv, hc, causal) <= max_bytes:
            return hc
    return 0


def _check(do, q, k, v, o, lse):
    """Shape / dtype / layout contract of the kernels; returns (b, sq, skv, hq, hkv)."""
    b, sq, hq, dqk = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    if (dqk, v.shape[-1]) != (limits.D_QK, limits.D_V):
        raise ValueError(f"head dims must be qk {limits.D_QK} / v {limits.D_V}, got {dqk} / {v.shape[-1]}")
    if k.shape != (b, skv, hkv, limits.D_QK) or v.shape != (b, skv, hkv, limits.D_V):
        raise ValueError(f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)} do not match")
    if o.shape != (b, sq, hq, limits.D_V) or do.shape != o.shape or lse.shape != (b, hq, sq):
        raise ValueError(
            f"out/dout/lse shapes {tuple(o.shape)}/{tuple(do.shape)}/{tuple(lse.shape)} do not match q"
        )
    if min(b, sq, skv, hq) <= 0:
        raise ValueError(f"empty problem: q {tuple(q.shape)}, k {tuple(k.shape)}")
    for name, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o), ("lse", lse)):
        if t.device != q.device:
            raise ValueError(f"{name} is on {t.device}, q on {q.device}")
        if name != "lse" and (not t.is_contiguous() or t.dtype != torch.bfloat16):
            raise ValueError(f"{name} must be a contiguous bfloat16 tensor")
    return b, sq, skv, hq, hkv


def _alloc(shape, device, dtype):
    if POISON:
        return torch.full(shape, float("nan"), device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype)


def log2_scale_fp32(scale):
    """log2(softmax scale) rounded to fp32: k_delta's l2s argument."""
    return struct.unpack("f", struct.pack("f", math.log2(scale)))[0]


def _zero_ticket(ds_full):
    """k_dkdv96ds's ticket counter = 0; runs before every dkdv launch."""
    ds_full.view(torch.int32)[TICKET_OFFSET_B // 4 : DS_GUARD_B // 4].zero_()


def _ds_scratch(nbytes, device):
    t = torch.empty(((2 * DS_GUARD_B + nbytes) // 2,), dtype=torch.bfloat16, device=device)
    if POISON and t.device.type != "meta":
        t.fill_(float("nan"))
    return t, t[DS_GUARD_B // 2 : DS_GUARD_B // 2 + nbytes // 2]


def _plan(do, q, k, v, o, lse, softmax_scale, causal, head_group=None, max_scratch_bytes=None):
    """Every launch hk_attn_bwd makes for these inputs, in issue order, plus what it allocates.

    launches: [(name, kernel or host step, args)]: "delta", then ("ticket", "dkdv", "dq") once, or once per head
    chunk. A kernel launch is (kernel name, (grid, block, kernel argument values)), the host step (_zero_ticket,
    its arguments). head_group: the target of both kernels (None: HEAD_GROUP_KV / HEAD_GROUP_Q); one the index
    arithmetic is not verified for raises ValueError. max_scratch_bytes: dS scratch budget (None: MAX_SCRATCH_BYTES);
    over it the plan is head-chunked (module docstring), ScratchBudgetError when no head group fits. Works on meta
    tensors."""
    b, sq, skv, hq, hkv = _check(do, q, k, v, o, lse)
    c = int(bool(causal))
    scale = limits.softmax_scale(softmax_scale)
    reason = limits.bwd_shape_unsupported_reason(b, sq, skv, hq, hkv, bool(causal))
    reason = reason or limits.bwd_scale_unsupported_reason(scale)
    if reason is not None:
        raise NotImplementedError(f"hk_attn_bwd: {reason}")
    tkv = HEAD_GROUP_KV if head_group is None else int(head_group)
    tq = HEAD_GROUP_Q if head_group is None else int(head_group)
    if tkv not in VERIFIED_KV_TARGETS or tq not in VERIFIED_Q_TARGETS:
        raise ValueError(
            f"hk_attn_bwd: head-group targets kv {tkv} / q {tq} are not verified "
            f"(kv {VERIFIED_KV_TARGETS}, q {VERIFIED_Q_TARGETS})"
        )
    hg_kv, hg_q = _head_group(hkv, tkv), _head_group(hq, tq)
    g = hq // hkv
    budget = MAX_SCRATCH_BYTES if max_scratch_bytes is None else int(max_scratch_bytes)
    step = math.lcm(hg_kv, hg_q)
    hc = _chunk_heads(b, sq, skv, hq, c, step, budget)
    if hc == 0:
        raise ScratchBudgetError(
            f"hk_attn_bwd: the dS scratch of {step} heads ({_scratch_bytes(1, sq, skv, step, c)} B) exceeds "
            f"max_scratch_bytes {budget}"
        )
    dev = q.device
    delta, nl, nd = (_alloc((b, hq, sq), dev, torch.float32) for _ in range(3))
    dq = _alloc((b, sq, hq, limits.D_QK), dev, q.dtype)
    dk = _alloc((b, skv, hkv, limits.D_QK), dev, k.dtype)
    dv = _alloc((b, skv, hkv, limits.D_V), dev, v.dtype)
    if hc is None:
        chunks = None
        parts = [((q, k, v, do, nl, nd, dv, dk, dq), b, b * hkv // hg_kv, b * hq // hg_q)]
        ds_bytes = limits.ds_layout_bytes(b, sq, skv, hq, c)
    else:
        # head chunks of one batch: views from the chunk's first (batch, head), B = 1 (MHA: k/v heads = q heads)
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
    delta_values = (do, o, delta, lse, nl, nd, log2_scale_fp32(scale), sq, hq)
    launches = [("delta", "k_delta", ((sq // DELTA_ROWS, b * hq, 1), DELTA_BLOCK, delta_values))]
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
        )
        dkdv_values += (b_, hg_kv, DS_MODE)
        dq_values = (ds, k_, dq_, sq, skv, hq, hkv, g, skv // 32, skv - sq, c, q_off, ntile, hg_q)
        launches.append(("ticket", _zero_ticket, (ds_full,)))
        launches.append(("dkdv", "k_dkdv96ds", ((hg_kv, limits.k3_nblk(skv), gz_kv), 64, dkdv_values)))
        launches.append(("dq", "k_dqds_c", ((hg_q, ntile + nhead, gz_q), 128, dq_values)))
    return {
        "launches": launches,
        "outputs": (dq, dk, dv),
        "delta": delta,
        "nl": nl,
        "nd": nd,
        "head_group": {"kv": hg_kv, "q": hg_q},
        "chunks": chunks,  # None, or [(batch, first head, heads)] of every (dkdv, dq) launch pair
        "ds_layout_bytes": ds_bytes,
        "ds_full": ds_full,
    }


def _flydsl_bwd(reason, do, q, k, v, o, lse, softmax_scale, causal):
    """The FlyDSL backward in place of a call the HIP kernels could not run (``reason``)."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd as fly

    b, sq, hq, _ = q.shape
    why = fly.shape_unsupported_reason(b, sq, k.shape[1], hq)
    if why is not None:
        raise ScratchBudgetError(f"{reason}; the FlyDSL backward cannot take this call either: {why}")
    logger.warning(f"{reason}: this backward runs on FlyDSL", once=True)
    return fly.flash_attn_mla_bwd(do, q, k, v, o, lse, limits.softmax_scale(softmax_scale), causal)


def hk_attn_bwd(
    do, q, k, v, o, lse, softmax_scale=None, causal=True, head_group=None, max_scratch_bytes=None
):
    """Flash-attention backward on gfx1250, DeepSeek-V3 MLA head dims (qk 192, v 128), MHA, bottom-right causal.

    Same contract as ``primus_turbo.flydsl.attention.gfx1250_mla.bwd.flash_attn_mla_bwd``: q [B, Sq, H, 192],
    o / do [B, Sq, H, 128], k [B, Skv, H, 192], v [B, Skv, H, 128], bf16 contiguous; lse [B, H, Sq] fp32 natural log.
    Returns (dq, dk, dv) laid out like q / k / v. The shapes must pass ``limits.bwd_shape_unsupported_reason``.
    max_scratch_bytes: dS scratch budget of this call (module docstring). When no head group fits the budget, or the
    scratch cannot be allocated, the call runs the FlyDSL backward."""
    hiprt.check_device(q.device)
    lse = lse.contiguous().float()
    table = kernels()
    with torch.cuda.device(q.device):
        try:
            plan, reason = (
                _plan(do, q, k, v, o, lse, softmax_scale, causal, head_group, max_scratch_bytes),
                None,
            )
        except ScratchBudgetError as e:
            reason = str(e)
        except torch.OutOfMemoryError:
            reason = f"hk_attn_bwd: out of memory for the dS scratch (cap it with {MAX_SCRATCH_BYTES_ENV})"
        if reason is not None:  # outside the except clause: the failed allocations are freed by now
            return _flydsl_bwd(reason, do, q, k, v, o, lse, softmax_scale, causal)
        stream = torch.cuda.current_stream(q.device)
        for _, fn, args in plan["launches"]:
            if isinstance(fn, str):
                grid, block, values = args
                table[fn].call(grid, block, values, stream)
            else:
                fn(*args)
    return plan["outputs"]
