###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HK gfx1250 flash-attention forward v0 (HIP kernel ``k_attn_fwd_v0``): head dim 128, bf16 BSHD, GQA with
G = Hq / Hkv in (4, 8, 16), bottom-right causal (query i sees keys j <= i + Skv - Sq) or full attention.

    o, lse = flash_attn_fwd(q, k, v, softmax_scale=None, causal=True)
        q [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] bf16 contiguous on one gfx1250 device
        -> o [B, Sq, Hq, 128] bf16, lse [B, Hq, Sq] fp32 (natural log of the scaled row sums)

lab hk contract (hk/lab/lab.py, arm ``hk:<pkgdir>#fwd_v0[@<hsaco_dir>]``): every launch goes through the
module-level ``_launch(name, fn, args)``; the code object is resolved lazily at the first launch from
``HSACO_DIR`` (set by lab or ``set_hsaco_dir``), else the package attribute ``HSACO_DIR``, else env
``HK_HSACO_DIR``, else ``DEFAULT_HSACO_DIR`` (the cc.sh output of hk/lab/cc.sh). Launches go to
``torch.cuda.current_stream()`` through hiprt (explicit kernarg kinds; no hidden arguments).

Host gate (fwd_v0_design section 11): every argument contract is checked and anything outside the proven
range raises (ValueError for a broken contract, NotImplementedError for a valid but unsupported problem,
e.g. G in (1, 2)); there is no silent fallback. The bounds proof hk/lab/bounds/fwd_v0_check.cpp replays
exactly the shapes this gate admits (its ``admitted`` mirrors ``unsupported_reason``).
Build: hk/lab/cc.sh <pkg>/hip/src/k_attn_fwd_v0.hip <dir> -mllvm -structurizecfg-skip-uniform-regions
-mllvm -amdgpu-sched-strategy=coexec (notes: hk/notes/fwd_v0_impl.md).
"""

import hashlib
import json
import math
import os
import sys
import threading

import torch

from . import codeobj, hiprt

__all__ = ["flash_attn_fwd", "unsupported_reason", "set_hsaco_dir", "plan"]

KERNEL = "k_attn_fwd_v0"
D = 128
BLOCK = 128  # 4 waves (1 per SIMD)
LDS_BYTES = 327680  # static: one workgroup per CU
QB = 64  # query positions per workgroup
NH = 4  # q heads per workgroup (one head quad)
KV_ST = 64  # kv rows per ring stage
GQA_RATIOS = (4, 8, 16)
KARG_BYTES = 104
# kernel arguments in order, with explicit kinds (fwd_v0_design section 11)
ARGS = (
    ("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("O", "ptr"), ("LSE", "ptr"),
    ("o_bytes", "i64"), ("lse_bytes", "i64"),
    ("B", "i32"), ("Sq", "i32"), ("Skv", "i32"), ("Hq", "i32"), ("Hkv", "i32"), ("G", "i32"),
    ("nqb", "i32"), ("nhq4", "i32"), ("nkv_st", "i32"), ("cst", "i32"), ("causal", "i32"),
    ("c2", "f32"),
)  # fmt: skip
I32_LIMIT = 1 << 31
DEFAULT_HSACO_DIR = "/home/lihuzhan/projects/2026_1010__llama3.1_attn/hk/build/fwd_v0"

HSACO_DIR = None  # set by lab.py / set_hsaco_dir; resolved at the first launch
_KERNEL = None  # (path, hiprt.Kernel)
_LOCK = threading.Lock()
_DEVICES_CHECKED = set()


def set_hsaco_dir(path):
    """Use the code object in ``path`` (a cc.sh output dir) from the next launch on."""
    global HSACO_DIR, _KERNEL
    with _LOCK:
        HSACO_DIR = None if path is None else os.path.abspath(path)
        _KERNEL = None


def _hsaco_dir():
    if HSACO_DIR:
        return HSACO_DIR
    pkg = sys.modules.get(__package__) if __package__ else None
    d = getattr(pkg, "HSACO_DIR", None) if pkg is not None else None
    return d or os.environ.get("HK_HSACO_DIR") or DEFAULT_HSACO_DIR


def _check_code_object(path):
    """Resources of the code object must be the ones the bounds proof and the launch assume."""
    meta = codeobj.kernels(path).get(KERNEL)
    if meta is None:
        raise RuntimeError(f"{path}: no kernel {KERNEL}")
    want = {"lds": LDS_BYTES, "scratch": 0, "vgpr_spill": 0, "sgpr_spill": 0, "wavefront_size": 32,
            "max_block": BLOCK, "kernarg_size": KARG_BYTES, "dynamic_stack": False}  # fmt: skip
    bad = {k: meta[k] for k, v in want.items() if meta[k] != v}
    if bad:
        raise RuntimeError(f"{path}: {KERNEL} resources {bad} != {want}")
    mj = os.path.splitext(path)[0] + ".meta.json"
    if os.path.isfile(mj):  # cc.sh output: the build must have passed and match these bytes
        with open(mj) as f:
            m = json.load(f)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if not m.get("ok", False) or m.get("hsaco_sha256") != sha:
            raise RuntimeError(f"{path}: stale or failed build ({mj}: ok={m.get('ok')}, sha256 mismatch "
                               f"{m.get('hsaco_sha256', '')[:12]} vs {sha[:12]})")  # fmt: skip
    return meta


def _kernel():
    """hiprt.Kernel of the resolved code object (loaded at the first launch; file reads only until then)."""
    global _KERNEL
    path = os.path.join(_hsaco_dir(), f"{KERNEL}.hsaco")
    k = _KERNEL
    if k is not None and k[0] == path:
        return k[1]
    with _LOCK:
        if _KERNEL is None or _KERNEL[0] != path:
            if not os.path.isfile(path):
                raise RuntimeError(f"{KERNEL}: code object {path} not found (build it with hk/lab/cc.sh)")
            _check_code_object(path)
            spec = {KERNEL: {"block": BLOCK, "args": list(ARGS)}}
            _KERNEL = (path, hiprt.load_hsaco(path, spec)[KERNEL])
        return _KERNEL[1]


def _launch(name, fn, args):
    """Every kernel launch of this module (lab --seg wraps it)."""
    fn(*args)


def unsupported_reason(b, sq, skv, hq, hkv, causal, d=D):
    """Why k_attn_fwd_v0 cannot run this problem (None when it can). Mirrors the bounds proof's gate."""
    if d != D:
        return f"head dim must be {D}, got {d}"
    if min(b, sq, skv, hq, hkv) < 1:
        return f"empty problem: b {b} sq {sq} skv {skv} hq {hq} hkv {hkv}"
    if hq % hkv:
        return f"hq {hq} not a multiple of hkv {hkv}"
    g = hq // hkv
    if g not in GQA_RATIOS:
        return f"G = hq / hkv = {g} not in {GQA_RATIOS} (v0 packs 4 q heads of one kv head per workgroup)"
    if sq % QB or skv % KV_ST:
        return f"sq {sq} and skv {skv} must be multiples of {QB} / {KV_ST}"
    if causal and sq > skv:
        return (
            f"causal needs sq <= skv (bottom-right: the first sq - skv queries see no key), got {sq} > {skv}"
        )
    if 2 * b * sq * hq * D >= I32_LIMIT or 4 * b * hq * sq >= I32_LIMIT:
        return "O / LSE byte extents must be < 2^31 (int32 buffer offsets)"
    if b * (hq // NH) * (sq // QB) >= I32_LIMIT:
        return "grid too large"
    return None


def _check_args(q, k, v, softmax_scale, causal):
    """Contract checks (raise); returns (b, sq, skv, hq, hkv, scale, causal_int)."""
    named = {"q": q, "k": k, "v": v}
    for n, t in named.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{n} must be a torch.Tensor")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{n} must be bfloat16, got {t.dtype}")
        if t.dim() != 4:
            raise ValueError(f"{n} must be 4-D [B, S, H, D], got {tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"{n} must be contiguous (BSHD), strides {t.stride()}")
        if t.device != q.device:
            raise ValueError(f"all tensors on one device: q on {q.device}, {n} on {t.device}")
        if t.device.type != "meta" and t.data_ptr() % 16:
            raise ValueError(f"{n}.data_ptr() must be 16-B aligned")
    b, sq, hq, d = q.shape
    bk, skv, hkv, dk = k.shape
    if tuple(v.shape) != tuple(k.shape) or bk != b or dk != d:
        raise ValueError(f"shapes q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} do not match")
    if not isinstance(causal, (bool, int)) or int(causal) not in (0, 1):
        raise ValueError(f"causal must be a bool, got {causal!r}")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    scale = float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and > 0, got {softmax_scale!r}")
    reason = unsupported_reason(b, sq, skv, hq, hkv, int(causal), d)
    if reason is not None:
        raise NotImplementedError(f"{KERNEL}: {reason}")
    return b, sq, skv, hq, hkv, scale, int(causal)


def plan(q, k, v, softmax_scale=None, causal=True):
    """The launch flash_attn_fwd makes for these inputs (works on meta tensors; touches no GPU):
    {"grid", "block", "values" (kernarg values in ARGS order), "outputs" (o, lse), "c2", "derived"}."""
    b, sq, skv, hq, hkv, scale, c = _check_args(q, k, v, softmax_scale, causal)
    g = hq // hkv
    nqb, nhq4, nkv_st = sq // QB, hq // NH, skv // KV_ST
    cst = (skv - sq) // KV_ST if c else 0
    o = torch.empty_like(q)
    lse = torch.empty((b, hq, sq), dtype=torch.float32, device=q.device)
    o_bytes, lse_bytes = 2 * o.numel(), 4 * lse.numel()
    assert o_bytes == 2 * b * sq * hq * D and lse_bytes == 4 * b * hq * sq
    c2 = scale * math.log2(math.e)  # double; packed as fp32 (RNE) into the kernarg
    values = (q, k, v, o, lse, o_bytes, lse_bytes, b, sq, skv, hq, hkv, g, nqb, nhq4, nkv_st, cst, c, c2)
    assert len(values) == len(ARGS)
    grid = (b * nhq4 * nqb, 1, 1)
    return {"grid": grid, "block": (BLOCK, 1, 1), "values": values, "outputs": (o, lse), "c2": c2,
            "derived": {"G": g, "nqb": nqb, "nhq4": nhq4, "nkv_st": nkv_st, "cst": cst}}  # fmt: skip


def _check_device_once(device):
    if device.type != "cuda":
        raise ValueError(f"{KERNEL} runs on a CUDA/HIP device, got {device}")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _DEVICES_CHECKED:
        return
    arch = torch.cuda.get_device_properties(idx).gcnArchName
    if "gfx1250" not in arch:
        raise RuntimeError(f"{KERNEL} is gfx1250 code; device {idx} is {arch}")
    _DEVICES_CHECKED.add(idx)


def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """(o [B, Sq, Hq, 128] bf16, lse [B, Hq, Sq] fp32 natural log) on torch's current stream."""
    _check_device_once(q.device)
    p = plan(q, k, v, softmax_scale, causal)
    kern = _kernel()
    stream = torch.cuda.current_stream(q.device)
    _launch("fwd", kern.call, (p["grid"], p["block"], p["values"], stream))
    return p["outputs"]
