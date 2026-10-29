###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bwd_hk: launcher of the MLA backward rungs (the ladder's copy of
primus_turbo/hipkittens/gfx1250_mla_bwd/impl.py at Primus-Turbo 71bd0329): the code objects come from a
rung's hkl.build directory (set_hsaco_dir), not from a manifest; no logger, no scratch budget from the
environment. The kernels' ABI is the same in every rung (d3590be4 .. 71bd0329), so one launch plan
serves them all; a rung without k_delta (before 5b19697f) runs the FlyDSL k_delta, as production did.

    configure(delta=None | "hk" | "flydsl")     # None: the HIP k_delta iff the rung built it
    set_hsaco_dir(build/mla_bwd/<rung>)
    dq, dk, dv = flash_attn_bwd(do, q, k, v, o, lse, softmax_scale, causal=True)   # SBHD or BSHD

flash_attn_bwd takes the ladder problem's tensors: q / k [S, B, H, 192], v / o / do [S, B, H, 128] SBHD
(Megatron's layout, contiguous) or [B, S, H, D] BSHD, lse [B, H, Sq]. SBHD with B > 1 is handed to the
kernels as [1, S, B*H, D] (the batch folded into the heads, no copy), exactly as the production adapter
(attention_flydsl_gfx1250_mla_impl.py) does; the gradients come back in the input layout.

The original module documentation follows.

hk_attn_bwd: the gfx1250 MLA backward with HIP dK/dV and dQ kernels.

Three launches, all on the caller's current stream, in this order:
  "delta"  HIP k_delta: delta, and the per-row constants NL = -LOG2E*lse, ND = -scale*delta the dK/dV
           kernel reads (64-query workgroups, bitwise equal to the FlyDSL k_delta of
           ``primus_turbo.flydsl.attention.gfx1250_mla``, which runs in its place when bin/k_delta.hsaco
           is missing or not the pinned one: k_delta is the manifest's one ``optional`` kernel);
  "dkdv"   HIP k_dkdv96ds: dK and dV (96-row kv blocks of two 48-row waves on one 3-stage Q/dO TDM ring,
           then 64-row tail blocks), and every bf16 dS^T tile it feeds the dK WMMA (softmax scale
           included) stored into this call's scratch in dslayout.h v1;
  "dq"     HIP k_dqds_c: dQ = dS K read back from that scratch (384-query workgroups of 4 waves sharing
           one K ring, plus one merged 256-query head tile), so S and dP are never recomputed.
k_dqds_c reads what k_dkdv96ds writes: there is no side stream. Every output element is written by
exactly one workgroup (no atomics), so the results are deterministic and do not depend on the
head-group targets.

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

import hashlib
import json

from . import _turbo_path  # noqa: F401  (puts the Primus-Turbo checkout on sys.path)
from primus_turbo.flydsl.attention.gfx1250_mla import bwd as _fly  # noqa: E402

from . import codeobj, hiprt, limits  # noqa: E402

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
MAX_SCRATCH_BYTES = None  # ladder: never chunk (every rung is measured on the whole call)


class ScratchBudgetError(NotImplementedError):
    """Not even one head group's dS scratch fits the budget (hk_attn_bwd then runs the FlyDSL backward)."""


# Validation only: allocate delta/dq/dk/dv and the whole dS scratch NaN-filled, so an element the
# kernels never write shows up as non-finite instead of as stale memory.
POISON = False

_KERNELS = None
_LOCK = threading.Lock()
_DEVICES_CHECKED = set()
HSACO_DIR = None
DELTA = None  # configure(): None = HIP k_delta iff built, "hk" = required, "flydsl" = never
ND_FOLD = False  # configure(): the rung's k_dkdv96ds folds ND into its dP C init (k_delta -DHK_ND_FOLD: NL += l2s,
#                  ND = -delta, l2s = log2(scale) rounded to fp32 on the host, k_delta's f32 argument)
TICKET = False  # configure(): k_dkdv96ds built with -DHKGEN_TICKET (k_dqds_c -DDQ_TICKET) takes its work item from
#                 a ticket counter (the int 64 / 128 B in front of the dS layout), zeroed before every dkdv launch
DQ_PERSIST = 0  # configure(): k_dqds_c built with -DDQ_PERSIST runs this many persistent workgroups (grid x) on
#                 per-XCD queues (the ints 128 B in front of the dS layout, zeroed with the ticket) + argument nz


def nd_fold_offset(scale):
    """log2(softmax scale) rounded to fp32: the HIP k_delta's f32 argument and the FlyDSL-delta fold's addend."""
    return struct.unpack("f", struct.pack("f", math.log2(scale)))[0]


def _fold_nl_nd(nl, nd, delta, l2s):
    """After the FlyDSL k_delta (NL = -LOG2E lse, ND = -scale delta): the folded constants the HIP k_delta writes
    (NL + l2s, -delta), bit for bit: the same fp32 add, a negation."""
    nl.add_(l2s)
    torch.neg(delta, out=nd)


def _zero_ticket(ds_full):
    """The ticket counters in the 128 B in front of the dS layout = 0 (k_dkdv96ds -DHKGEN_TICKET: the int at -64,
    k_dqds_c -DDQ_TICKET: the int at -128)."""
    ds_full.view(torch.int32)[(DS_GUARD_B - 128) // 4 : DS_GUARD_B // 4].zero_()


HOST_STEPS = (_fold_nl_nd, _zero_ticket)  # launches run on the host side (torch ops), not HIP / FlyDSL kernels
# kernel argument kinds (manifest.json of 71bd0329; identical in every rung)
SPECS = {
    "k_dkdv96ds": {"block": 64, "args": [("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("DO", "ptr"), ("NL", "ptr"),
                   ("ND", "ptr"), ("DV", "ptr"), ("DK", "ptr"), ("DS", "ptr"), ("scale", "f32"), ("Sq", "i32"),
                   ("Skv", "i32"), ("Hq", "i32"), ("Hkv", "i32"), ("G", "i32"), ("nqt", "i32"), ("cshift", "i32"),
                   ("causal", "i32"), ("B", "i32"), ("HG", "i32"), ("ds_mode", "i32")]},
    "k_dqds_c": {"block": 128, "args": [("DS", "ptr"), ("K", "ptr"), ("DQ", "ptr"), ("Sq", "i32"), ("Skv", "i32"),
                 ("Hq", "i32"), ("Hkv", "i32"), ("G", "i32"), ("nkvt", "i32"), ("cshift", "i32"), ("causal", "i32"),
                 ("q_off", "i32"), ("ntile", "i32"), ("HG", "i32")]},
    "k_delta": {"block": 256, "args": [("DO", "ptr"), ("O", "ptr"), ("DEL", "ptr"), ("LSE", "ptr"), ("NL", "ptr"),
                ("ND", "ptr"), ("scale", "f32"), ("S", "i32"), ("H", "i32")]},
}  # fmt: skip


def configure(delta=None, dkdv_block=64, nd_fold=False, ticket=False, dq_persist=0):
    """delta: see DELTA; dkdv_block: the k_dkdv96ds workgroup size of the rung (64; 128 for the 4-wave
    wave-specialized kernel, which supports MHA only); nd_fold: see ND_FOLD; ticket / dq_persist: see TICKET,
    DQ_PERSIST."""
    global DELTA, _KERNELS, ND_FOLD, TICKET, DQ_PERSIST
    ND_FOLD = bool(nd_fold)
    TICKET = bool(ticket)
    DQ_PERSIST = int(dq_persist)
    args = [a for a in SPECS["k_dqds_c"]["args"] if a[0] != "nz"] + ([("nz", "i32")] if DQ_PERSIST else [])
    SPECS["k_dqds_c"] = dict(SPECS["k_dqds_c"], args=args)
    if delta not in (None, "hk", "flydsl"):
        raise ValueError(f"delta must be None, 'hk' or 'flydsl', got {delta!r}")
    if dkdv_block not in (64, 128):
        raise ValueError(f"dkdv_block must be 64 or 128, got {dkdv_block!r}")
    SPECS["k_dkdv96ds"] = dict(SPECS["k_dkdv96ds"], block=dkdv_block)
    with _LOCK:
        DELTA, _KERNELS = delta, None


def set_hsaco_dir(path):
    global HSACO_DIR, _KERNELS
    with _LOCK:
        HSACO_DIR = None if path is None else os.path.abspath(path)
        _KERNELS = None


def _load(name, d=None):
    d = d or HSACO_DIR
    path = os.path.join(d, f"{name}.hsaco")
    mj = os.path.join(d, f"{name}.meta.json")
    with open(mj) as f:
        m = json.load(f)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if not m.get("ok", False) or m.get("hsaco_sha256") != sha:
        raise hiprt.HipError(f"{path}: stale or failed build ({mj})")
    t = codeobj.kernels(path)[name]
    if t["scratch"] or t["vgpr_spill"]:
        raise hiprt.HipError(f"{path}: scratch {t['scratch']} / vgpr spill {t['vgpr_spill']}")
    return hiprt.load_hsaco(path, {name: SPECS[name]}, sha256=sha)[name]


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
    """{name: hiprt.Kernel} of the rung directory (sha256 vs its meta.json; no GPU work here)."""
    global _KERNELS
    if _KERNELS is None:
        with _LOCK:
            if _KERNELS is None:
                if HSACO_DIR is None:
                    raise RuntimeError("bwd_hk: set_hsaco_dir() first")
                table = {n: _load(n) for n in ("k_dkdv96ds", "k_dqds_c")}
                has_delta = os.path.isfile(os.path.join(HSACO_DIR, "k_delta.hsaco"))
                if DELTA == "hk" and not has_delta:
                    raise RuntimeError(f"{HSACO_DIR}: no k_delta code object")
                if has_delta and DELTA != "flydsl":
                    table["k_delta"] = _load("k_delta")
                elif os.environ.get("HKL_MLA_DELTA_DIR") and DELTA != "flydsl":
                    # profiling aid: a rung without k_delta takes the HIP k_delta of another build dir (bitwise equal
                    # to FlyDSL's), because FlyDSL's JIT launch deadlocks under rocprofv3's queue interposition
                    table["k_delta"] = _load("k_delta", os.environ["HKL_MLA_DELTA_DIR"])
                _KERNELS = table
    return _KERNELS


def _alloc(shape, device, dtype):
    if POISON:
        return torch.full(shape, float("nan"), device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype)


def _hk_delta_reason(sq):
    """None when the HIP k_delta can take seqlen_q ``sq`` and its code object is the pinned one, else why not."""
    if sq % DELTA_ROWS:
        return f"seqlen_q {sq} is not a multiple of {DELTA_ROWS}"
    return None if "k_delta" in kernels() else "the rung has no HIP k_delta"


def _ds_scratch(nbytes, device):
    t = torch.empty(((2 * DS_GUARD_B + nbytes) // 2,), dtype=torch.bfloat16, device=device)
    if POISON and t.device.type != "meta":
        t.fill_(float("nan"))
    return t, t[DS_GUARD_B // 2 : DS_GUARD_B // 2 + nbytes // 2]


def _plan(
    do, q, k, v, o, lse, softmax_scale, causal, stream, head_group=None, max_scratch_bytes=None, hk_delta=None
):
    """Every launch hk_attn_bwd makes for these inputs, in issue order, plus what it allocates.

    launches: [(name, launcher, args)]: "delta", then "dkdv" / "dq" once, or once per head chunk; a HIP
    launch is (kernel name, (grid, block, kernel argument values)), a FlyDSL one (launcher, its arguments).
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
    if SPECS["k_dkdv96ds"]["block"] == 128 and g != 1:
        raise NotImplementedError("hk_attn_bwd: the wave-specialized k_dkdv96ds (block 128) supports MHA only")
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
        delta_values = (do, o, delta, lse, nl, nd, nd_fold_offset(scale) if ND_FOLD else scale, sq, hq)
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
        if ND_FOLD:
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
        if TICKET or DQ_PERSIST:
            launches.append(("ticket", _zero_ticket, (ds_full,)))
        launches.append(("dkdv", "k_dkdv96ds", ((hg_kv, limits.k3_nblk(skv), gz_kv), SPECS["k_dkdv96ds"]["block"],
                                                dkdv_values)))
        if DQ_PERSIST:
            launches.append(("dq", "k_dqds_c", ((DQ_PERSIST, 1, 1), 128, dq_values + (gz_q,))))
        else:
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
    try:
        plan = _plan(
            do, q, k, v, o, lse, softmax_scale, causal, stream, head_group, max_scratch_bytes, hk_delta
        )
    except ScratchBudgetError:
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


# ---------------------------------------------------------------- ladder entry point (layout folding)


def _fold(t, sbhd):
    """[S, B, H, D] contiguous -> [1, S, B*H, D] view; BSHD as is."""
    if not sbhd:
        return t
    s_, b_, h_, d_ = t.shape
    return t.view(1, s_, b_ * h_, d_)


def _unfold(t, b, sbhd):
    if not sbhd:
        return t
    _, s_, bh, d_ = t.shape
    return t.view(s_, b, bh // b, d_)


def is_sbhd(q, lse):
    """The ladder's MLA problem is SBHD: q [S, B, H, D] with lse [B, H, Sq] (BSHD has lse.shape[2] == q.shape[1])."""
    return lse.shape[0] == q.shape[1] and lse.shape[2] == q.shape[0] and q.shape[0] != q.shape[1]


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """(dq, dk, dv) in the inputs' layout (see the module docstring)."""
    sbhd = is_sbhd(q, lse)
    b = q.shape[1] if sbhd else q.shape[0]
    f = [_fold(t, sbhd) for t in (do, q, k, v, o)]
    lse_k = lse.reshape(1, -1, lse.shape[-1]) if sbhd else lse
    dq, dk, dv = hk_attn_bwd(*f, lse_k, softmax_scale=softmax_scale, causal=causal)
    return tuple(_unfold(t, b, sbhd) for t in (dq, dk, dv))
