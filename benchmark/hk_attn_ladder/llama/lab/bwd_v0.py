###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bwd_v0 -- host side of the HK gfx1250 d128 GQA attention backward v0.

    flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True) -> (dq, dk, dv)

q / o / do [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] bf16 contiguous (BSHD), lse fp32 [B, Hq, Sq] natural log,
causal = bottom-right (query i sees keys j <= i + Skv - Sq). dk / dv have Hkv heads. Kernels (one code object,
hip/src/k_attn_bwd_v0.hip; design hk/notes/bwd_v0_design.md, implementation note hk/notes/bwd_v0_impl.md),
all on torch.cuda.current_stream(), in this order:

    zero_dqa        dQ_acc.zero_() (torch memset; HK_BWD_PREP_Z=1: zeroed by the prep kernel instead)
    k_hk_bwd_prep   LD records {NL = -LSE log2e, delta = rowsum(dO o O)}     grid (Sq/64, Hq, B), block 256
    k_hk_bwd_g4     dK, dV, fp32 dQ_acc atomics (G = Hq / Hkv = 4)             grid (Hkv, Skv/128, B), block 128
    k_hk_bwd_dqcvt  dq = bf16(scale dQ_acc)                                    grid (Sq/32, Hq, B), block 256

Shapes the kernels take (host_gate, mirror of bwd/idx.h b0::host_gate, after FlyDSL's unsupported_reason rules):
every dim >= 1, Hq == 4 Hkv, Sq % 64 == 0, Skv % 128 == 0, causal -> Sq <= Skv, byte extents < 2^31. Anything
else raises ValueError before any allocation or launch (or goes to `fallback` when one is passed).

Lab contract (hk/lab/lab.py): arm `name=hk:<this package dir>#bwd_v0@<hsaco dir>`. Every launch, the memset
included, goes through the module-level _launch(name, fn, args); the code object <dir>/k_attn_bwd_v0.hsaco is
resolved at the first launch from HSACO_DIR (set by lab or set_hsaco_dir), else env HK_HSACO_DIR, else
DEFAULT_HSACO_DIR (a cc.sh output directory). Validation aid: HK_BWD_POISON=1 NaN-fills ld / dq / dk / dv first.
"""

import math
import os
import threading

import torch

from . import codeobj, hiprt

__all__ = ["flash_attn_bwd", "plan", "host_gate", "unsupported_reason", "set_hsaco_dir", "SPECS"]

DEFAULT_HSACO_DIR = "/home/lihuzhan/projects/2026_1010__llama3.1_attn/hk/build/bwd_v0"
HSACO_NAME = "k_attn_bwd_v0.hsaco"
HSACO_DIR = None

D, G, BKV, QS, PREP_ROWS = 128, 4, 128, 32, 64
MAX_TENSOR_BYTES = 1 << 30  # FlyDSL's q / k limit
LIM = 1 << 31

_P, _I, _F = "ptr", "i32", "f32"
_PREP_ARGS = [("DO", _P), ("O", _P), ("LSE", _P), ("LD", _P), ("DQA", _P), ("Sq", _I), ("Hq", _I), ("B", _I)]
SPECS = {
    "k_hk_bwd_prep": {"block": 256, "args": _PREP_ARGS, "lds": 256},
    "k_hk_bwd_prep_z": {"block": 256, "args": _PREP_ARGS, "lds": 256},
    "k_hk_bwd_g4": {
        "block": 128,
        "args": [("Q", _P), ("K", _P), ("V", _P), ("DO", _P), ("LD", _P), ("DK", _P), ("DV", _P), ("DQA", _P),
                 ("scale", _F), ("Sq", _I), ("Skv", _I), ("Hq", _I), ("Hkv", _I), ("B", _I), ("causal", _I)],
        "lds": 266240,
    },
    "k_hk_bwd_dqcvt": {"block": 256, "args": [("DQA", _P), ("DQ", _P), ("scale", _F), ("Sq", _I), ("Hq", _I), ("B", _I)],
                       "lds": 8704},
}  # fmt: skip


def _launch(name, fn, args):
    """Every kernel launch and memset of this module goes through here (lab --seg wraps it)."""
    return fn(*args)


# ---------------------------------------------------------------- shape rules


GATE_OK, GATE_EMPTY, GATE_G, GATE_SQ, GATE_SKV, GATE_CAUSAL, GATE_BYTES = range(7)
GATE_NAMES = (
    "ok",
    "empty dimension",
    "Hq != 4 Hkv",
    "Sq % 64",
    "Skv % 128",
    "causal Sq > Skv",
    "byte extent >= 2^31",
)


def host_gate(B, Sq, Skv, Hq, Hkv, causal, g=G):
    """Mirror of b0::host_gate (bwd/idx.h): 0 = the kernels take the shape, else the first failed rule."""
    if min(B, Sq, Skv, Hq, Hkv) < 1:
        return GATE_EMPTY
    if Hq != g * Hkv:
        return GATE_G
    if Sq % 64:
        return GATE_SQ
    if Skv % BKV:
        return GATE_SKV
    if causal and Sq > Skv:
        return GATE_CAUSAL
    if (B * Sq * Hq * 256 >= LIM or B * Hq * Sq * 512 >= LIM or g * Sq * 512 >= LIM or B * Skv * Hkv * 256 >= LIM
            or (BKV - 1) * Hkv * 256 + 256 >= LIM):  # fmt: skip
        return GATE_BYTES
    return GATE_OK


def unsupported_reason(q_shape, k_shape, v_shape, dtype, causal):
    """FlyDSL gfx1250 interface.unsupported_reason's rules (bf16, 4-D, no empty dim, d 128, ...), then host_gate;
    None when bwd_v0 takes the problem."""
    if dtype != torch.bfloat16:
        return f"dtype must be bfloat16, got {dtype}"
    if len(q_shape) != 4 or len(k_shape) != 4:
        return f"q and k must be 4-D [B, S, H, D], got q {tuple(q_shape)} / k {tuple(k_shape)}"
    if 0 in (*q_shape, *k_shape):
        return f"q {tuple(q_shape)} and k {tuple(k_shape)} must have no empty dimension"
    b, sq, hq, d = q_shape
    bk, skv, hkv, dk = k_shape
    if tuple(v_shape) != tuple(k_shape) or bk != b:
        return f"k {tuple(k_shape)} and v {tuple(v_shape)} must match, with q's batch {b}"
    if d != D or dk != D:
        return f"head_dim must be {D}, got q {d} / k {dk}"
    if 2 * d * max(b * sq * hq, b * skv * hkv) > MAX_TENSOR_BYTES:
        return "q and k must each fit in 1 GiB"
    rc = host_gate(b, sq, skv, hq, hkv, int(bool(causal)))
    if rc != GATE_OK:
        return f"bwd_v0 host gate: {GATE_NAMES[rc]} (B {b} Sq {sq} Skv {skv} Hq {hq} Hkv {hkv} causal {bool(causal)})"
    return None


def _check(do, q, k, v, o, lse, causal):
    """ValueError unless the tensors are exactly what the kernels address (types, layouts, shapes, devices)."""
    named = {"do": do, "q": q, "k": k, "v": v, "o": o, "lse": lse}
    for n, t in named.items():
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{n} must be a tensor")
        if t.device != q.device:
            raise ValueError(f"all tensors must be on one device: q on {q.device}, {n} on {t.device}")
        if not t.is_contiguous():
            raise ValueError(f"{n} must be contiguous")
    for n in ("do", "q", "k", "v", "o"):
        if named[n].dtype != torch.bfloat16:
            raise ValueError(f"{n} must be bfloat16, got {named[n].dtype}")
    reason = unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal)
    if reason is not None:
        raise ValueError(reason)
    if do.shape != q.shape or o.shape != q.shape:
        raise ValueError(
            f"do / o must have q's shape {tuple(q.shape)}, got {tuple(do.shape)} / {tuple(o.shape)}"
        )
    b, sq, hq, _ = q.shape
    if lse.dtype != torch.float32 or tuple(lse.shape) != (b, hq, sq):
        raise ValueError(f"lse must be float32 of shape {(b, hq, sq)}, got {lse.dtype} {tuple(lse.shape)}")


# ---------------------------------------------------------------- launch plan (shapes only; CPU-testable)


class Ref:
    """a tensor argument of a planned launch, by name (inputs do/q/k/v/o/lse, outputs dq/dk/dv, workspace ld/dqa)"""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Ref({self.name})"


def plan(B, Sq, Skv, Hq, Hkv, causal, scale, prep_zero=False):
    """{"work": {name: (shape, dtype)}, "launches": [(name, grid, block, args)]} of one call; grid None = a torch
    memset (args = the tensor). ValueError for every shape the gate rejects: no launch is ever planned for it."""
    rc = host_gate(B, Sq, Skv, Hq, Hkv, int(bool(causal)))
    if rc != GATE_OK:
        raise ValueError(f"bwd_v0 host gate: {GATE_NAMES[rc]}")
    nqp = Sq // QS
    work = {"ld": ((B, Hq, Sq // 4, 8), torch.float32), "dqa": ((B * Hq * nqp * 4096,), torch.float32)}
    L = []
    if not prep_zero:
        L.append(("zero_dqa", None, None, (Ref("dqa"),)))
    pre = (Ref("do"), Ref("o"), Ref("lse"), Ref("ld"), Ref("dqa"), Sq, Hq, B)
    L.append(("k_hk_bwd_prep_z" if prep_zero else "k_hk_bwd_prep", (Sq // PREP_ROWS, Hq, B), 256, pre))
    L.append(("k_hk_bwd_g4", (Hkv, Skv // BKV, B), 128,
              (Ref("q"), Ref("k"), Ref("v"), Ref("do"), Ref("ld"), Ref("dk"), Ref("dv"), Ref("dqa"), float(scale),
               Sq, Skv, Hq, Hkv, B, int(bool(causal)))))  # fmt: skip
    L.append(("k_hk_bwd_dqcvt", (Sq // QS, Hq, B), 256, (Ref("dqa"), Ref("dq"), float(scale), Sq, Hq, B)))
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
                raise hiprt.HipError(f"bwd_v0: code object {path} not found (build it with hk/lab/cc.sh)")
            table = codeobj.kernels(path)
            for name, spec in SPECS.items():
                t = table.get(name)
                if t is None:
                    raise hiprt.HipError(f"bwd_v0: {name} not in {path}")
                if t["lds"] != spec["lds"] or t["scratch"] != 0 or t["vgpr_spill"] or t["sgpr_spill"]:
                    raise hiprt.HipError(f"bwd_v0: {name} in {path}: lds {t['lds']} (want {spec['lds']}), scratch "
                                         f"{t['scratch']}, spill v{t['vgpr_spill']} s{t['sgpr_spill']}")  # fmt: skip
            k = hiprt.load_hsaco(
                path, {n: {"block": s["block"], "args": s["args"]} for n, s in SPECS.items()}
            )
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
        raise ValueError(f"bwd_v0 runs on a gfx1250 GPU, got tensors on {q.device}")
    B, Sq, Hq, d = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    scale = float(1.0 / math.sqrt(d) if softmax_scale is None else softmax_scale)
    pl = plan(B, Sq, Skv, Hq, Hkv, causal, scale, prep_zero=os.environ.get("HK_BWD_PREP_Z") == "1")
    return _run(
        pl,
        {"do": do, "q": q, "k": k, "v": v, "o": o, "lse": lse},
        _kernels(),
        torch.cuda.current_stream(),
        poison=os.environ.get("HK_BWD_POISON") == "1",
    )


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
