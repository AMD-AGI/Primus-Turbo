###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bwd_v1 -- host side of the HK gfx1250 d128 GQA attention backward v1.

    flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True) -> (dq, dk, dv)

Same contract, shapes, gate and numerics as bwd_v0 (q / o / do [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] bf16
contiguous BSHD, lse fp32 [B, Hq, Sq] natural log, bottom-right causal). Kernels (one code object,
hip/src/k_attn_bwd_v1.hip; design hk/notes/bwd_v1_design.md, implementation note hk/notes/bwd_v1_impl.md), all
on torch.cuda.current_stream(), in this order:

    k_hk_bwd1_prep_z  LD records {NL, delta} + dQ_acc zeroed (HK_BWD1_ZERO=memset: torch memset + k_hk_bwd1_prep)
                                                                               grid (Sq/64, Hq, B), block 256
    k_hk_bwd1_g4      dK, dV, fp32 dQ_acc atomics (G = 4; re-timed expert-mode step, head rotation)
                                                                               grid (Hkv, Skv/128, B), block 128
    k_hk_bwd1_dqcvt   dq = bf16(scale dQ_acc)                                  grid (Sq/32, Hq, B), block 256

The compile-time variant (BWD1_SCHED / ATOM / NWIN / HROT, expert or default scheduling) is a property of the
code-object directory; every variant has the same kernels, arguments and grids.

Lab contract (hk/lab/lab.py): arm `name=hk:<this package dir>#bwd_v1@<hsaco dir>`. Every launch, the memset
included, goes through the module-level _launch(name, fn, args); the code object <dir>/k_attn_bwd_v1.hsaco is
resolved at the first launch from HSACO_DIR (lab / set_hsaco_dir), else env HK_HSACO_DIR, else DEFAULT_HSACO_DIR.
Validation aid: HK_BWD1_POISON=1 NaN-fills ld / dq / dk / dv first.

Production (impl.py, the package default "v1"; manifest entry k_attn_bwd_v1 built with the atom2 flags): impl points
the module at the package's pinned bin/ with set_hsaco_dir before its first launch, so DEFAULT_HSACO_DIR (the lab's
default-variant directory) is never used there.
"""

import math
import os
import threading

import torch

from . import bwd_v0 as _v0
from . import codeobj, hiprt

__all__ = ["flash_attn_bwd", "plan", "host_gate", "unsupported_reason", "set_hsaco_dir", "SPECS"]

DEFAULT_HSACO_DIR = "/home/lihuzhan/projects/2026_1010__llama3.1_attn/hk/build/bwd_v1/default"
HSACO_NAME = "k_attn_bwd_v1.hsaco"
HSACO_DIR = None

D, G, BKV, QS, PREP_ROWS = _v0.D, _v0.G, _v0.BKV, _v0.QS, _v0.PREP_ROWS
LIM = _v0.LIM
Ref = _v0.Ref

_P, _I, _F = "ptr", "i32", "f32"
SPECS = {
    "k_hk_bwd1_prep": {"block": 256, "args": _v0._PREP_ARGS, "lds": 256},
    "k_hk_bwd1_prep_z": {"block": 256, "args": _v0._PREP_ARGS, "lds": 256},
    "k_hk_bwd1_g4": {"block": 128, "args": _v0.SPECS["k_hk_bwd_g4"]["args"], "lds": 266240},
    "k_hk_bwd1_dqcvt": {"block": 256, "args": _v0.SPECS["k_hk_bwd_dqcvt"]["args"], "lds": 8704},
}


def _launch(name, fn, args):
    """Every kernel launch and memset of this module goes through here (lab --seg wraps it)."""
    return fn(*args)


# ---------------------------------------------------------------- shape rules (== v0: same kernels' gate)

GATE_OK, GATE_NAMES = _v0.GATE_OK, _v0.GATE_NAMES


def host_gate(B, Sq, Skv, Hq, Hkv, causal, g=G):
    """Mirror of b0::host_gate (bwd/idx.h; v1 adds no rule: n and nf are multiples of G = 4 for every admitted
    shape, which the x4-unrolled loop needs -- proved by hk/lab/bounds/bwd_v1_check.cpp)."""
    return _v0.host_gate(B, Sq, Skv, Hq, Hkv, causal, g)


def unsupported_reason(q_shape, k_shape, v_shape, dtype, causal):
    """None when bwd_v1 takes the problem (the bwd_v0 rules), else the reason."""
    r = _v0.unsupported_reason(q_shape, k_shape, v_shape, dtype, causal)
    return None if r is None else r.replace("bwd_v0", "bwd_v1")


def _check(do, q, k, v, o, lse, causal):
    try:
        _v0._check(do, q, k, v, o, lse, causal)
    except ValueError as e:
        raise ValueError(str(e).replace("bwd_v0", "bwd_v1")) from None


def zero_mode():
    """'prep' (default: k_hk_bwd1_prep_z zeroes dQ_acc) or 'memset' (env HK_BWD1_ZERO=memset)."""
    m = os.environ.get("HK_BWD1_ZERO", "prep")
    if m not in ("prep", "memset"):
        raise ValueError(f"HK_BWD1_ZERO must be prep or memset, got {m!r}")
    return m


# ---------------------------------------------------------------- launch plan (shapes only; CPU-testable)


def plan(B, Sq, Skv, Hq, Hkv, causal, scale, prep_zero=True):
    """{"work": {name: (shape, dtype)}, "launches": [(name, grid, block, args)]} of one call; grid None = a torch
    memset (args = the tensor). ValueError for every shape the gate rejects: no launch is ever planned for it."""
    rc = host_gate(B, Sq, Skv, Hq, Hkv, int(bool(causal)))
    if rc != GATE_OK:
        raise ValueError(f"bwd_v1 host gate: {GATE_NAMES[rc]}")
    nqp = Sq // QS
    work = {"ld": ((B, Hq, Sq // 4, 8), torch.float32), "dqa": ((B * Hq * nqp * 4096,), torch.float32)}
    L = []
    if not prep_zero:
        L.append(("zero_dqa", None, None, (Ref("dqa"),)))
    pre = (Ref("do"), Ref("o"), Ref("lse"), Ref("ld"), Ref("dqa"), Sq, Hq, B)
    L.append(("k_hk_bwd1_prep_z" if prep_zero else "k_hk_bwd1_prep", (Sq // PREP_ROWS, Hq, B), 256, pre))
    L.append(("k_hk_bwd1_g4", (Hkv, Skv // BKV, B), 128,
              (Ref("q"), Ref("k"), Ref("v"), Ref("do"), Ref("ld"), Ref("dk"), Ref("dv"), Ref("dqa"), float(scale),
               Sq, Skv, Hq, Hkv, B, int(bool(causal)))))  # fmt: skip
    L.append(("k_hk_bwd1_dqcvt", (Sq // QS, Hq, B), 256, (Ref("dqa"), Ref("dq"), float(scale), Sq, Hq, B)))
    for name, grid, block, args in L:
        if grid is None:
            continue
        spec = SPECS[name]
        assert block == spec["block"] and len(args) == len(spec["args"]), name
        for a, (an, kind) in zip(args, spec["args"]):
            assert (kind == _P) == isinstance(a, Ref), (name, an)
            if kind == _I:
                assert isinstance(a, int) and 0 <= a < LIM, (name, an, a)
        assert all(1 <= x < (1 << 31) for x in grid), (name, grid)
    return {"work": work, "launches": L}


# ---------------------------------------------------------------- code object (lazy)

_KERNELS = {}
_LOCK = threading.Lock()


def set_hsaco_dir(path):
    """lab hook: the code-object directory of later launches (resolved at the next launch)."""
    global HSACO_DIR
    HSACO_DIR = None if path is None else os.path.abspath(path)


def hsaco_path():
    d = HSACO_DIR or os.environ.get("HK_HSACO_DIR") or DEFAULT_HSACO_DIR
    return os.path.join(d, HSACO_NAME)


def _kernels():
    path = hsaco_path()
    k = _KERNELS.get(path)
    if k is not None:
        return k
    with _LOCK:
        k = _KERNELS.get(path)
        if k is None:
            if not os.path.isfile(path):
                raise hiprt.HipError(f"bwd_v1: code object {path} not found (build it with hk/lab/cc.sh)")
            table = codeobj.kernels(path)
            for name, spec in SPECS.items():
                t = table.get(name)
                if t is None:
                    raise hiprt.HipError(f"bwd_v1: {name} not in {path}")
                if t["lds"] != spec["lds"] or t["scratch"] != 0 or t["vgpr_spill"] or t["sgpr_spill"]:
                    raise hiprt.HipError(f"bwd_v1: {name} in {path}: lds {t['lds']} (want {spec['lds']}), scratch "
                                         f"{t['scratch']}, spill v{t['vgpr_spill']} s{t['sgpr_spill']}")  # fmt: skip
            k = hiprt.load_hsaco(path, {n: {"block": s["block"], "args": s["args"]} for n, s in SPECS.items()})
            _KERNELS[path] = k
    return k


# ---------------------------------------------------------------- entry point


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True, fallback=None):
    """(dq [B,Sq,Hq,128] bf16, dk [B,Skv,Hkv,128] bf16, dv) of the attention backward; see the module doc.
    fallback: optional callable with flash_attn_bwd's signature for problems outside the gate (else ValueError)."""
    if fallback is not None and (not all(isinstance(t, torch.Tensor) for t in (q, k, v))
                                 or unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal) is not None):  # fmt: skip
        return fallback(do, q, k, v, o, lse, softmax_scale, causal)
    _check(do, q, k, v, o, lse, causal)
    if q.device.type != "cuda":
        raise ValueError(f"bwd_v1 runs on a gfx1250 GPU, got tensors on {q.device}")
    B, Sq, Hq, d = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    scale = float(1.0 / math.sqrt(d) if softmax_scale is None else softmax_scale)
    pl = plan(B, Sq, Skv, Hq, Hkv, causal, scale, prep_zero=zero_mode() == "prep")
    return _run(pl, {"do": do, "q": q, "k": k, "v": v, "o": o, "lse": lse}, _kernels(), torch.cuda.current_stream(),
                poison=os.environ.get("HK_BWD1_POISON") == "1")


def _run(pl, t, kern, stream, poison=False):
    """Allocate the workspace / outputs on the inputs' device and issue the plan's launches in order through
    _launch (kern: {name: object with .call(grid, block, values, stream)}). Returns (dq, dk, dv)."""
    dev = t["q"].device
    for name, (shape, dtype) in pl["work"].items():
        t[name] = torch.empty(shape, device=dev, dtype=dtype)
    t["dq"], t["dk"], t["dv"] = torch.empty_like(t["q"]), torch.empty_like(t["k"]), torch.empty_like(t["v"])
    if poison:
        for n in ("ld", "dq", "dk", "dv"):
            _launch("poison_" + n, torch.Tensor.fill_, (t[n], float("nan")))
    for name, grid, block, args in pl["launches"]:
        vals = [t[a.name] if isinstance(a, Ref) else a for a in args]
        if grid is None:
            _launch(name, torch.Tensor.zero_, (vals[0],))
        else:
            _launch(name, kern[name].call, (grid, (block, 1, 1), vals, stream))
    return t["dq"], t["dk"], t["dv"]
