###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launcher of the MLA backward rungs below hk_v0 (llama/hip/src/ladder/m0*.hip; ABIs in ladder/lm.h): qk 192 /
v 128, MHA, bf16, bottom-right causal. Takes the ladder problem's SBHD tensors (or BSHD) like bwd_hk and folds
SBHD to [1, S, B*H, D] (no copy).

    configure(prefix="k_mla_m00", plan="split")
    set_hsaco_dir(build/mla_bwd/m00)
    dq, dk, dv = flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True)

Plans: split  <prefix>_prep (DEL, NL) grid (Sq/64, H, B) block 256; <prefix>_dkdv grid (Skv/64, H, B) block
128; <prefix>_dq grid (Sq/64, H, B) block 128.
       export <prefix>_prep; <prefix>_dkdv (dK, dV, dS^T -> scratch [B, H, Skv, Sq] bf16) grid (Skv/64, H, B);
              <prefix>_dq (dQ = scale dS K from the scratch) grid (Sq/q_per_wg, H, B)
configure(..., q_per_wg=64, kv_per_wg=64): the dq / dkdv workgroup sizes in rows (m03q on: 128 queries, m05w on:
128 kv rows, m07q on: 256 queries; dkdv_block=64: a 2-wave dK / dV workgroup, m08h); sq / skv must be multiples of them, except q_per_wg 256: grid ceil(sq / 256), the
kernel handles a last workgroup of fewer (a multiple of 64) queries.
"""

import hashlib
import json
import math
import os
import threading

import torch

from . import codeobj, hiprt
from .bwd_hk import _fold, _unfold, is_sbhd

__all__ = ["configure", "set_hsaco_dir", "plan", "flash_attn_bwd"]

DQK, DV = 192, 128
I32_LIMIT = 1 << 31
_P, _I, _L, _F = "ptr", "i32", "i64", "f32"
_EXT = [("q_bytes", _L), ("k_bytes", _L), ("v_bytes", _L), ("o_bytes", _L), ("row_bytes", _L)]
_TAIL = [("B", _I), ("Sq", _I), ("Skv", _I), ("H", _I), ("causal", _I), ("c2", _F), ("scale", _F)]
_IN = [("Q", _P), ("K", _P), ("V", _P), ("DO", _P), ("DEL", _P), ("NL", _P)]
ARGS = {
    "split": {
        "prep": [("DO", _P), ("O", _P), ("LSE", _P), ("DEL", _P), ("NL", _P), ("o_bytes", _L), ("row_bytes", _L),
                 ("B", _I), ("Sq", _I), ("H", _I)],
        "dkdv": _IN + [("DK", _P), ("DV", _P)] + _EXT + _TAIL,
        "dq": _IN + [("DQ", _P)] + _EXT + _TAIL,
    },
}  # fmt: skip
ARGS["export"] = {
    "prep": ARGS["split"]["prep"],
    "dkdv": _IN + [("DK", _P), ("DV", _P), ("DS", _P)] + _EXT + [("ds_bytes", _L)] + _TAIL,
    "dq": [("DS", _P), ("K", _P), ("DQ", _P), ("k_bytes", _L), ("q_bytes", _L), ("ds_bytes", _L)] + _TAIL,
}
BLOCKS = {"prep": 256, "dkdv": 128, "dq": 128}

PREFIX = None
PLAN = "split"
Q_PER_WG = 64
KV_PER_WG = 64
DKDV_BLOCK = 128
HSACO_DIR = None
_KERNELS = None
_LOCK = threading.Lock()


def configure(prefix, plan="split", q_per_wg=64, kv_per_wg=64, dkdv_block=128):
    global PREFIX, PLAN, Q_PER_WG, KV_PER_WG, DKDV_BLOCK, _KERNELS
    if plan not in ARGS:
        raise ValueError(f"unknown plan {plan!r}")
    if q_per_wg not in (64, 128, 256) or kv_per_wg not in (64, 128):
        raise ValueError(f"q_per_wg / kv_per_wg must be 64 or 128, got {q_per_wg} / {kv_per_wg}")
    if dkdv_block not in (64, 128):
        raise ValueError(f"dkdv_block must be 64 or 128, got {dkdv_block}")
    with _LOCK:
        PREFIX, PLAN, Q_PER_WG, KV_PER_WG, _KERNELS = str(prefix), plan, int(q_per_wg), int(kv_per_wg), None
        DKDV_BLOCK = int(dkdv_block)


def set_hsaco_dir(path):
    global HSACO_DIR, _KERNELS
    with _LOCK:
        HSACO_DIR = None if path is None else os.path.abspath(path)
        _KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    with _LOCK:
        if PREFIX is None or HSACO_DIR is None:
            raise RuntimeError("bwd_mladder: configure() and set_hsaco_dir() first")
        path = os.path.join(HSACO_DIR, f"{PREFIX}.hsaco")
        with open(os.path.join(HSACO_DIR, f"{PREFIX}.meta.json")) as f:
            m = json.load(f)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if not m.get("ok", False) or m.get("hsaco_sha256") != sha:
            raise hiprt.HipError(f"{path}: stale or failed build")
        table = codeobj.kernels(path)
        for role in ARGS[PLAN]:
            t = table.get(f"{PREFIX}_{role}")
            if t is None or t["scratch"] or t["vgpr_spill"]:
                raise hiprt.HipError(f"{path}: {PREFIX}_{role} missing, or uses scratch / spills")
        specs = {f"{PREFIX}_{r}": {"block": _block(r), "args": a} for r, a in ARGS[PLAN].items()}
        loaded = hiprt.load_hsaco(path, specs, sha256=sha)
        _KERNELS = {r: loaded[f"{PREFIX}_{r}"] for r in ARGS[PLAN]}
        return _KERNELS


def _block(role):
    return DKDV_BLOCK if role == "dkdv" else BLOCKS[role]


def _launch(name, fn, args):
    fn(*args)


def plan(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Launches of the folded BSHD call (works on meta tensors)."""
    b, sq, h, dqk = q.shape
    skv = k.shape[1]
    if (dqk, v.shape[-1], k.shape[2]) != (DQK, DV, h) or o.shape != (b, sq, h, DV) or do.shape != o.shape:
        raise ValueError(f"MLA shapes expected (qk {DQK}, v {DV}, MHA): q {tuple(q.shape)} k {tuple(k.shape)} "
                         f"v {tuple(v.shape)} o {tuple(o.shape)}")  # fmt: skip
    if sq % min(Q_PER_WG, 128) or skv % KV_PER_WG or sq % 64 or skv % 64 or (causal and sq > skv):
        raise NotImplementedError(f"{PREFIX}: sq {sq} / skv {skv} must be multiples of {max(64, min(Q_PER_WG, 128))} / "
                                  f"{max(64, KV_PER_WG)}, causal sq <= skv")
    ds_el = b * h * skv * sq if PLAN == "export" else 0
    # the kernels address the dS scratch per (b, head) (own buffer resource), so only one head's 2 Skv Sq bytes count
    if max(2 * b * sq * h * DQK, 2 * b * skv * h * DQK, 2 * skv * sq) >= I32_LIMIT:
        raise NotImplementedError(f"{PREFIX}: byte extents must be < 2^31 (int32 buffer offsets)")
    scale = 1.0 / math.sqrt(dqk) if softmax_scale is None else float(softmax_scale)
    c2 = scale * math.log2(math.e)
    dev = q.device
    dl = torch.empty((b, h, sq), dtype=torch.float32, device=dev)
    nl = torch.empty_like(dl)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    ext = (2 * q.numel(), 2 * k.numel(), 2 * v.numel(), 2 * o.numel(), 4 * dl.numel())
    tail = (b, sq, skv, h, int(bool(causal)), c2, scale)
    prep = ("prep", (sq // 64, h, b), (do, o, lse, dl, nl, 2 * o.numel(), 4 * dl.numel(), b, sq, h))
    if PLAN == "split":
        launches = [prep, ("dkdv", (skv // 64, h, b), (q, k, v, do, dl, nl, dk, dv) + ext + tail),
                    ("dq", (sq // 64, h, b), (q, k, v, do, dl, nl, dq) + ext + tail)]  # fmt: skip
    else:
        ds = torch.empty((b, h, skv, sq), dtype=torch.bfloat16, device=dev)  # dS^T, only causal-visible blocks written
        launches = [prep, ("dkdv", (skv // KV_PER_WG, h, b), (q, k, v, do, dl, nl, dk, dv, ds) + ext + (2 * ds_el,) + tail),
                    ("dq", (-(-sq // Q_PER_WG), h, b), (ds, k, dq, 2 * k.numel(), 2 * q.numel(), 2 * ds_el) + tail)]  # fmt: skip
    return {"launches": launches, "outputs": (dq, dk, dv)}


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    sbhd = is_sbhd(q, lse)
    bsz = q.shape[1] if sbhd else q.shape[0]
    f = [_fold(t, sbhd) for t in (do, q, k, v, o)]
    lse_k = (lse.reshape(1, -1, lse.shape[-1]) if sbhd else lse).contiguous().float()
    p = plan(*f, lse_k, softmax_scale, causal)
    kern = _kernels()
    stream = torch.cuda.current_stream(q.device)
    for role, grid, values in p["launches"]:
        _launch(role, kern[role].call, (grid, (_block(role), 1, 1), values, stream))
    return tuple(_unfold(t, bsz, sbhd) for t in p["outputs"])
