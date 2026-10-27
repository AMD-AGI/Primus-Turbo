###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launcher of the forward rungs below v0 (src/ladder/f0*.hip, kernels ``k_fwd_f00`` .. ``k_fwd_f08``):
head dim 128, bf16 BSHD, GQA, bottom-right causal or full attention.

    configure(kernel="k_fwd_f03", heads_per_wg=4)   # the rung (ladder json "launch.opts")
    set_hsaco_dir(build/llama_fwd/f03)               # its build directory (hkl.build output)
    o, lse = flash_attn_fwd(q, k, v, softmax_scale=None, causal=True)
        q [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] bf16 contiguous on one gfx1250 device
        -> o [B, Sq, Hq, 128] bf16, lse [B, Hq, Sq] fp32 (natural log of the scaled row sums)

Every rung has the ABI of ladder/lf.h (Q, K, V, O, LSE, q_bytes, kv_bytes, o_bytes, lse_bytes, B, Sq,
Skv, Hq, Hkv, causal, c2) and a 1-D grid of 128-thread workgroups: one per (b, q head, 64-query block)
(``heads_per_wg`` 1) or per (b, head quad, 64-query block) (4); the work order is the kernel's own.
Anything outside the range the kernels are written for raises (no fallback).
"""

import hashlib
import json
import math
import os
import threading

import torch

from . import codeobj, hiprt

__all__ = ["configure", "set_hsaco_dir", "unsupported_reason", "plan", "flash_attn_fwd"]

D = 128
BLOCK = 128
QB = 64
KVB = 64
ARGS = (
    ("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("O", "ptr"), ("LSE", "ptr"),
    ("q_bytes", "i64"), ("kv_bytes", "i64"), ("o_bytes", "i64"), ("lse_bytes", "i64"),
    ("B", "i32"), ("Sq", "i32"), ("Skv", "i32"), ("Hq", "i32"), ("Hkv", "i32"), ("causal", "i32"),
    ("c2", "f32"),
)  # fmt: skip
KARG_BYTES = 100
I32_LIMIT = 1 << 31

KERNEL = None  # configure()
HEADS_PER_WG = 1
HSACO_DIR = None
_KERNEL = None  # (path, kernel name, hiprt.Kernel)
_LOCK = threading.Lock()
_DEVICES_CHECKED = set()


def configure(kernel, heads_per_wg=1):
    """Select the rung: its kernel symbol and how many q heads one workgroup takes (1 or 4)."""
    global KERNEL, HEADS_PER_WG, _KERNEL
    if heads_per_wg not in (1, 4):
        raise ValueError(f"heads_per_wg must be 1 or 4, got {heads_per_wg!r}")
    with _LOCK:
        KERNEL, HEADS_PER_WG, _KERNEL = str(kernel), int(heads_per_wg), None


def set_hsaco_dir(path):
    """Use the code object in ``path`` (an hkl.build rung directory) from the next launch on."""
    global HSACO_DIR, _KERNEL
    with _LOCK:
        HSACO_DIR = None if path is None else os.path.abspath(path)
        _KERNEL = None


def _check_code_object(path):
    meta = codeobj.kernels(path).get(KERNEL)
    if meta is None:
        raise RuntimeError(f"{path}: no kernel {KERNEL}")
    want = {"scratch": 0, "vgpr_spill": 0, "wavefront_size": 32, "max_block": BLOCK,
            "kernarg_size": KARG_BYTES, "dynamic_stack": False}  # fmt: skip
    bad = {k: meta[k] for k, v in want.items() if meta[k] != v}
    if bad:
        raise RuntimeError(f"{path}: {KERNEL} resources {bad} != {want}")
    mj = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.isfile(mj):
        raise RuntimeError(f"{path}: no {mj} (build rungs with hkl.build)")
    with open(mj) as f:
        m = json.load(f)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if not m.get("ok", False) or m.get("hsaco_sha256") != sha:
        raise RuntimeError(f"{path}: stale or failed build ({mj}: ok={m.get('ok')}, sha256 "
                           f"{m.get('hsaco_sha256', '')[:12]} vs {sha[:12]})")  # fmt: skip
    return meta


def _kernel():
    global _KERNEL
    if KERNEL is None or HSACO_DIR is None:
        raise RuntimeError("fwd_ladder: configure() and set_hsaco_dir() first")
    path = os.path.join(HSACO_DIR, f"{KERNEL}.hsaco")
    k = _KERNEL
    if k is not None and k[0] == path and k[1] == KERNEL:
        return k[2]
    with _LOCK:
        if not os.path.isfile(path):
            raise RuntimeError(f"{KERNEL}: code object {path} not found")
        _check_code_object(path)
        kern = hiprt.load_hsaco(path, {KERNEL: {"block": BLOCK, "args": list(ARGS)}})[KERNEL]
        _KERNEL = (path, KERNEL, kern)
        return kern


def _launch(name, fn, args):
    """Every kernel launch of this module (a profiler / segment timer can wrap it)."""
    fn(*args)


def unsupported_reason(b, sq, skv, hq, hkv, causal, d=D):
    """Why the configured rung cannot run this problem (None when it can); mirrors ladder/lf.h's gate."""
    if d != D:
        return f"head dim must be {D}, got {d}"
    if min(b, sq, skv, hq, hkv) < 1:
        return f"empty problem: b {b} sq {sq} skv {skv} hq {hq} hkv {hkv}"
    if hq % hkv:
        return f"hq {hq} not a multiple of hkv {hkv}"
    if HEADS_PER_WG == 4 and (hq // hkv) % 4:
        return f"G = hq / hkv = {hq // hkv} is not a multiple of 4 (a workgroup takes 4 q heads of one kv head)"
    if sq % QB or skv % KVB:
        return f"sq {sq} and skv {skv} must be multiples of {QB} / {KVB}"
    if causal and sq > skv:
        return f"causal needs sq <= skv (bottom-right), got {sq} > {skv}"
    if max(2 * b * sq * hq * D, 2 * b * skv * hkv * D, 4 * b * hq * sq) >= I32_LIMIT:
        return "byte extents must be < 2^31 (int32 buffer offsets)"
    return None


def plan(q, k, v, softmax_scale=None, causal=True):
    """The launch flash_attn_fwd makes (works on meta tensors; touches no GPU)."""
    for n, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or not t.is_contiguous():
            raise ValueError(f"{n} must be a contiguous 4-D bf16 BSHD tensor, got {t.dtype} {tuple(t.shape)}")
        if t.device != q.device:
            raise ValueError(f"all tensors on one device: q on {q.device}, {n} on {t.device}")
    b, sq, hq, d = q.shape
    _, skv, hkv, _ = k.shape
    if tuple(v.shape) != tuple(k.shape) or k.shape[0] != b or k.shape[3] != d:
        raise ValueError(f"shapes q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} do not match")
    c = int(bool(causal))
    reason = unsupported_reason(b, sq, skv, hq, hkv, c, d)
    if reason is not None:
        raise NotImplementedError(f"{KERNEL}: {reason}")
    scale = 1.0 / math.sqrt(d) if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and > 0, got {softmax_scale!r}")
    o = torch.empty_like(q)
    lse = torch.empty((b, hq, sq), dtype=torch.float32, device=q.device)
    c2 = scale * math.log2(math.e)
    values = (q, k, v, o, lse, 2 * q.numel(), 2 * k.numel(), 2 * o.numel(), 4 * lse.numel(),
              b, sq, skv, hq, hkv, c, c2)  # fmt: skip
    grid = (b * (hq // HEADS_PER_WG) * (sq // QB), 1, 1)
    return {"grid": grid, "block": (BLOCK, 1, 1), "values": values, "outputs": (o, lse)}


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
