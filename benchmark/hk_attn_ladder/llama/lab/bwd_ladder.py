###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launcher of the backward rungs below v0 (src/ladder/b0*.hip): head dim 128, bf16 BSHD, GQA, bottom-right
causal or full attention.

    configure(prefix="k_bwd_b00", plan="split", heads_per_wg=1)   # the rung (ladder json "launch.opts")
    set_hsaco_dir(build/llama_bwd/b00)
    dq, dk, dv = flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True)

Plans (every launch on torch's current stream, in this order; ABIs in include/ladder/lb.h):
  split  <prefix>_prep (DEL, NL)          grid (Sq/64, Hq, B), block 256
         <prefix>_dkdv (dK, dV)           grid (Skv/64, Hkv, B), block 128
         <prefix>_dq   (dQ, recomputing S and dP)  grid (Sq/64, Hq / heads_per_wg, B), block 128
  fused  dQ_acc.zero_() (torch), <prefix>_prep,
         <prefix>_dkdv (dK, dV, dQ_acc += dS K by fp32 atomics)    grid (Skv/64, Hkv, B), block 128
         <prefix>_cvt  (dq = bf16(scale dQ_acc))                    grid (B Sq Hq 128 / 2048), block 256
Anything outside the range the kernels are written for raises (no fallback).
"""

import hashlib
import json
import math
import os
import threading

import torch

from . import codeobj, hiprt

__all__ = ["configure", "set_hsaco_dir", "unsupported_reason", "plan", "flash_attn_bwd"]

D = 128
I32_LIMIT = 1 << 31
_P, _I, _L, _F = "ptr", "i32", "i64", "f32"
ARGS = {
    "prep": [("DO", _P), ("O", _P), ("LSE", _P), ("DEL", _P), ("NL", _P), ("o_bytes", _L), ("row_bytes", _L),
             ("B", _I), ("Sq", _I), ("Hq", _I)],
    "dkdv": [("Q", _P), ("K", _P), ("V", _P), ("DO", _P), ("DEL", _P), ("NL", _P), ("DK", _P), ("DV", _P),
             ("q_bytes", _L), ("kv_bytes", _L), ("row_bytes", _L), ("B", _I), ("Sq", _I), ("Skv", _I), ("Hq", _I),
             ("Hkv", _I), ("causal", _I), ("c2", _F), ("scale", _F)],
    "dq": [("Q", _P), ("K", _P), ("V", _P), ("DO", _P), ("DEL", _P), ("NL", _P), ("DQ", _P), ("q_bytes", _L),
           ("kv_bytes", _L), ("row_bytes", _L), ("B", _I), ("Sq", _I), ("Skv", _I), ("Hq", _I), ("Hkv", _I),
           ("causal", _I), ("c2", _F), ("scale", _F)],
}  # fmt: skip
ARGS_FUSED = {
    "prep": ARGS["prep"],
    "dkdv": ARGS["dkdv"][:8] + [("DQA", _P)] + ARGS["dkdv"][8:11] + [("dqa_bytes", _L)] + ARGS["dkdv"][11:],
    "cvt": [("DQA", _P), ("DQ", _P), ("dqa_bytes", _L), ("q_bytes", _L), ("scale", _F)],
}
BLOCKS = {"prep": 256, "dkdv": 128, "dq": 128, "cvt": 256}

PREFIX = None
PLAN = "split"
HEADS_PER_WG = 1
HSACO_DIR = None
_KERNELS = None
_LOCK = threading.Lock()


def configure(prefix, plan="split", heads_per_wg=1):
    global PREFIX, PLAN, HEADS_PER_WG, _KERNELS
    if plan not in ("split", "fused"):
        raise ValueError(f"unknown plan {plan!r}")
    if heads_per_wg not in (1, 4):
        raise ValueError(f"heads_per_wg must be 1 or 4, got {heads_per_wg!r}")
    with _LOCK:
        PREFIX, PLAN, HEADS_PER_WG, _KERNELS = str(prefix), plan, int(heads_per_wg), None


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
            raise RuntimeError("bwd_ladder: configure() and set_hsaco_dir() first")
        path = os.path.join(HSACO_DIR, f"{PREFIX}.hsaco")
        mj = os.path.join(HSACO_DIR, f"{PREFIX}.meta.json")
        with open(mj) as f:
            m = json.load(f)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if not m.get("ok", False) or m.get("hsaco_sha256") != sha:
            raise hiprt.HipError(f"{path}: stale or failed build ({mj})")
        table = codeobj.kernels(path)
        specs = {}
        roles = ARGS if PLAN == "split" else ARGS_FUSED
        for role, args in roles.items():
            name = f"{PREFIX}_{role}"
            t = table.get(name)
            if t is None:
                raise hiprt.HipError(f"{path}: no kernel {name}")
            if t["scratch"] or t["vgpr_spill"] or t["wavefront_size"] != 32:
                raise hiprt.HipError(f"{path}: {name} scratch {t['scratch']} vgpr_spill {t['vgpr_spill']}")
            specs[name] = {"block": BLOCKS[role], "args": args}
        loaded = hiprt.load_hsaco(path, specs, sha256=sha)
        _KERNELS = {role: loaded[f"{PREFIX}_{role}"] for role in roles}
        return _KERNELS


def _launch(name, fn, args):
    """Every kernel launch of this module (a profiler / segment timer can wrap it)."""
    fn(*args)


def unsupported_reason(b, sq, skv, hq, hkv, causal, d=D):
    if d != D:
        return f"head dim must be {D}, got {d}"
    if min(b, sq, skv, hq, hkv) < 1 or hq % hkv:
        return f"bad heads / sizes: b {b} sq {sq} skv {skv} hq {hq} hkv {hkv}"
    if HEADS_PER_WG == 4 and (hq // hkv) % 4:
        return f"G = {hq // hkv} is not a multiple of 4"
    if sq % 64 or skv % 64:
        return f"sq {sq} and skv {skv} must be multiples of 64"
    if causal and sq > skv:
        return f"causal needs sq <= skv (bottom-right), got {sq} > {skv}"
    if max(2 * b * sq * hq * D, 2 * b * skv * hkv * D, 4 * b * hq * sq) >= I32_LIMIT:
        return "byte extents must be < 2^31 (int32 buffer offsets)"
    return None


def plan(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Launches and allocations of flash_attn_bwd (works on meta tensors)."""
    for n, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or not t.is_contiguous():
            raise ValueError(f"{n} must be a contiguous 4-D bf16 BSHD tensor")
    b, sq, hq, d = q.shape
    _, skv, hkv, _ = k.shape
    if tuple(v.shape) != tuple(k.shape) or o.shape != q.shape or do.shape != q.shape or tuple(lse.shape) != (b, hq, sq):
        raise ValueError("shapes do not match")
    c = int(bool(causal))
    reason = unsupported_reason(b, sq, skv, hq, hkv, c, d)
    if reason is not None:
        raise NotImplementedError(f"{PREFIX}: {reason}")
    scale = 1.0 / math.sqrt(d) if softmax_scale is None else float(softmax_scale)
    lse = lse.contiguous().float()
    dev = q.device
    dl = torch.empty((b, hq, sq), dtype=torch.float32, device=dev)
    nl = torch.empty((b, hq, sq), dtype=torch.float32, device=dev)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    qb, kvb, rb = 2 * q.numel(), 2 * k.numel(), 4 * dl.numel()
    c2 = scale * math.log2(math.e)
    common = (q, k, v, do, dl, nl)
    tail = (b, sq, skv, hq, hkv, c, c2, scale)
    if PLAN == "split":
        launches = [
            ("prep", (sq // 64, hq, b), (do, o, lse, dl, nl, qb, rb, b, sq, hq)),
            ("dkdv", (skv // 64, hkv, b), common + (dk, dv, qb, kvb, rb) + tail),
            ("dq", (sq // 64, hq // HEADS_PER_WG, b), common + (dq, qb, kvb, rb) + tail),
        ]
    else:
        dqa = torch.empty((b, sq, hq, d), dtype=torch.float32, device=dev)
        n8 = dqa.numel() // 8
        launches = [
            ("zero", None, (dqa,)),
            ("prep", (sq // 64, hq, b), (do, o, lse, dl, nl, qb, rb, b, sq, hq)),
            ("dkdv", (skv // 64, hkv, b), common + (dk, dv, dqa, qb, kvb, rb, 4 * dqa.numel()) + tail),
            ("cvt", (-(-n8 // 256), 1, 1), (dqa, dq, 4 * dqa.numel(), qb, scale)),
        ]
    return {"launches": launches, "outputs": (dq, dk, dv)}


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """(dq, dk, dv) bf16 in q / k / v's layout, on torch's current stream."""
    p = plan(do, q, k, v, o, lse, softmax_scale, causal)
    kern = _kernels()
    stream = torch.cuda.current_stream(q.device)
    for role, grid, values in p["launches"]:
        if grid is None:  # dQ_acc memset
            _launch(role, torch.Tensor.zero_, values)
        else:
            _launch(role, kern[role].call, (grid, (BLOCKS[role], 1, 1), values, stream))
    return p["outputs"]
