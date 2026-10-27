###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launcher of the HK MLA forward rungs (k_mla_fwd: X6's pure-HIP kernels, the marker shell + generated regions of
S1a .. E2, and their HIP reference variants): q / k 192, v 128, MHA, bottom-right causal, Sq == Skv.

    configure(nargs=11, persistent=False, layout="sbhd")      # the rung (ladder json "launch.opts")
    set_hsaco_dir(build/mla_fwd/e2)
    o, lse = flash_attn_fwd(q, k, v, softmax_scale, causal=True)

Kernel ABI (a08-1's fwd_ab.py): k_mla_fwd(Q, K, V, O, LSE, scale f32, Sq, Skv, H, B, nqb[, npw]) i32, block 128 (4
waves), kernarg 64 B (11 arguments + tail padding) or 68 B (12: E2). Tensors in the kernel layout [B, S, H, D]: the
problem's SBHD storage [S, B, H, D] goes in folded as [1, S, B*H, D] (FlyDSL's _to_kernel_layout, no copy), BSHD as
is; O [B, Sq, H, 128] bf16 and LSE [B, H, Sq] fp32 (natural log) come back in the problem's layout ([S, B, H, 128]
and [B, H, Sq]). Grid (nqb = ceil(Sq / 256), H, B), one 256-query block per workgroup; E2 (persistent=True) runs
grid (256, 1, 1) with npw = 256 where its static schedule covers the shape (persist_npw), else the same grid with
npw = 0. Outside the kernels' domain (not causal, Sq != Skv, Sq % 64, a byte extent >= 2^31) it raises
NotImplementedError.
"""

import hashlib
import json
import os
import threading

import torch

from . import codeobj, hiprt

__all__ = ["configure", "set_hsaco_dir", "persist_npw", "flash_attn_fwd"]

D_QK, D_V, BM, NPW = 192, 128, 256, 256
NAME = "k_mla_fwd"
_A11 = [("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("O", "ptr"), ("LSE", "ptr"), ("scale", "f32"), ("Sq", "i32"),
        ("Skv", "i32"), ("H", "i32"), ("B", "i32"), ("nqb", "i32")]  # fmt: skip

NARGS = 11
PERSISTENT = False
LAYOUT = "sbhd"
HSACO_DIR = None
_KERNEL = None
_LOCK = threading.Lock()


def configure(nargs=11, persistent=False, layout="sbhd"):
    global NARGS, PERSISTENT, LAYOUT, _KERNEL
    if nargs not in (11, 12) or (persistent and nargs != 12) or layout not in ("sbhd", "bshd"):
        raise ValueError(f"bad configuration nargs={nargs} persistent={persistent} layout={layout}")
    with _LOCK:
        NARGS, PERSISTENT, LAYOUT, _KERNEL = int(nargs), bool(persistent), layout, None


def set_hsaco_dir(path):
    global HSACO_DIR, _KERNEL
    with _LOCK:
        HSACO_DIR = None if path is None else os.path.abspath(path)
        _KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    with _LOCK:
        if HSACO_DIR is None:
            raise RuntimeError("fwd_hk: set_hsaco_dir() first")
        path = os.path.join(HSACO_DIR, f"{NAME}.hsaco")
        with open(os.path.join(HSACO_DIR, f"{NAME}.meta.json")) as f:
            m = json.load(f)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if not m.get("ok", False) or m.get("hsaco_sha256") != sha:
            raise hiprt.HipError(f"{path}: stale or failed build")
        t = codeobj.kernels(path).get(NAME)
        if t is None or t["scratch"] or t["vgpr_spill"]:
            raise hiprt.HipError(f"{path}: {NAME} missing, or uses scratch / spills")
        args = _A11 + ([("npw", "i32")] if NARGS == 12 else [])
        _KERNEL = hiprt.load_hsaco(path, {NAME: {"block": 128, "args": args}}, sha256=sha)[NAME]
        return _KERNEL


def persist_npw(kb, sq, skv, kh):
    """E2's npw for the kernel-layout shape [kb, sq, kh] (a08-1 fwd_ab.persist_npw): Sq == Skv, Sq % 256 == 0, NPW / 8
    workgroups per group a multiple of nqb, H * B % 8 == 0, the heads of a group a whole, even number of steps of
    R = NPW / 8 / nqb heads, at least 2 NPW blocks; else 0 (grid (nqb, H, B))."""
    nqb = (sq + BM - 1) // BM
    g, gyz = NPW // 8, kb * kh
    if sq != skv or sq % BM or g % nqb or gyz % 8:
        return 0
    r = g // nqb
    hx = gyz // 8
    if hx % r or (hx // r) % 2 or nqb * gyz < 2 * NPW:
        return 0
    return NPW


def _to_kernel(t):
    """The problem's tensor -> the kernel layout [B, S, H, D] (SBHD storage folded to [1, S, B*H, D])."""
    if LAYOUT == "bshd":
        return t.contiguous()
    s, b, h, d = t.shape
    return t.contiguous().view(1, s, b * h, d)


def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """(o [problem layout, 128 wide] bf16, lse [B, H, Sq] fp32)."""
    for n, t, d in (("q", q, D_QK), ("k", k, D_QK), ("v", v, D_V)):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[-1] != d:
            raise ValueError(f"{n} must be a 4-D bf16 tensor with head dim {d}")
    if not causal:
        raise NotImplementedError("the HK MLA forward is causal only")
    qk, kk, vk = (_to_kernel(t) for t in (q, k, v))
    kb, sq, kh, _ = qk.shape
    skv = kk.shape[1]
    if sq != skv or sq % 64 or tuple(kk.shape[:3]) != (kb, skv, kh) or tuple(vk.shape[:3]) != (kb, skv, kh):
        raise NotImplementedError(f"outside the HK forward's domain: Sq {sq} Skv {skv} (Sq == Skv, Sq % 64 == 0, MHA)")
    if max(kb * sq * kh * D_QK * 2, kb * kh * sq * 4) >= 2**31:
        raise NotImplementedError("byte extents must be < 2^31 (int32 buffer offsets)")
    scale = D_QK**-0.5 if softmax_scale is None else float(softmax_scale)
    nqb = (sq + BM - 1) // BM
    o = torch.empty((kb, sq, kh, D_V), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((kb, kh, sq), device=q.device, dtype=torch.float32)
    args = [qk, kk, vk, o, lse, scale, sq, skv, kh, kb, nqb]
    grid = (nqb, kh, kb)
    if NARGS == 12:
        npw = persist_npw(kb, sq, skv, kh) if PERSISTENT else 0
        args.append(npw)
        if npw:
            grid = (npw, 1, 1)
    _kernel().call(grid, (128, 1, 1), tuple(args), torch.cuda.current_stream(q.device))
    if LAYOUT == "bshd":
        return o, lse
    s, b, h, _ = q.shape
    return o.view(s, b, h, D_V), lse.view(b, h, sq)
