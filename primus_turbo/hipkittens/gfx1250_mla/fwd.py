###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hk_attn_fwd: the gfx1250 MLA forward on the HIP kernel k_mla_fwd.

k_mla_fwd runs the FlyDSL MLA forward's arithmetic in the same order (fp32 online softmax in log2 units, the same
lazy rescale decision and sum trees), so out (bf16) and lse (fp32, natural log) are bitwise equal to
``primus_turbo.flydsl.attention.gfx1250_mla.fwd.flash_attn_mla_fwd``. Its source is a marker shell
(``hip/src/k_mla_fwd.hip``) with generated tile regions (``tools/fwdgen``).

One launch on the caller's current stream. A workgroup (4 waves, one per SIMD) owns 256 query rows of one head and
runs their causal kv range in 64-row tiles through a 4-stage TDM ring. Grid: one q block per workgroup in FlyDSL's
longest-first, head-grouped order; or, when the static schedule covers the shape (``persistent_workgroups``),
PERSISTENT_WORKGROUPS workgroups that each run an equal share of the (q block, head) pairs. Every output element is
written by exactly one workgroup: the results are deterministic.
"""

import threading

import torch

from . import hiprt, hsaco, limits

BLOCK = 128  # 4 waves
PERSISTENT_WORKGROUPS = 256  # one per CU: 8 XCDs x 32 CUs

_KERNELS = None
_LOCK = threading.Lock()


def kernels():
    """{name: hiprt.Kernel} of the forward (code objects sha256-checked on first use; no GPU work here)."""
    global _KERNELS
    if _KERNELS is None:
        with _LOCK:
            if _KERNELS is None:
                _KERNELS = hiprt.load_kernels(hsaco.FWD_KERNELS)
    return _KERNELS


def persistent_workgroups(b: int, s: int, h: int) -> int:
    """The persistent grid size for [b, s, h] (seqlen_q == seqlen_kv == s), or 0 for one q block per workgroup.

    The static schedule: workgroup w serves the head-batch group w % 8 (one XCD under the round-robin dispatch) and,
    in step t, head-batch 8 (R t + r) + w % 8 with q block i on even steps and nqb - 1 - i on odd ones (w / 8 = r nqb
    + i, R = the heads per group and step), so two consecutive steps cost every workgroup nqb + 1 q blocks' tiles.
    It covers the shape when every group gets a whole, even number of steps."""
    nqb = -(-s // limits.FWD_ROWS)
    per_group, hb = PERSISTENT_WORKGROUPS // 8, b * h
    if s % limits.FWD_ROWS or per_group % nqb or hb % 8:
        return 0
    r = per_group // nqb
    steps, rem = divmod(hb // 8, r)
    if rem or steps % 2:
        return 0
    return PERSISTENT_WORKGROUPS


def _check(q, k, v):
    """Shape / dtype / layout contract of the kernel; returns (b, sq, skv, hq, hkv)."""
    b, sq, hq, dqk = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    if (dqk, v.shape[-1]) != (limits.D_QK, limits.D_V):
        raise ValueError(f"head dims must be qk {limits.D_QK} / v {limits.D_V}, got {dqk} / {v.shape[-1]}")
    if k.shape != (b, skv, hkv, limits.D_QK) or v.shape != (b, skv, hkv, limits.D_V):
        raise ValueError(f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)} do not match")
    if min(b, sq, skv, hq) <= 0:
        raise ValueError(f"empty problem: q {tuple(q.shape)}, k {tuple(k.shape)}")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.device != q.device:
            raise ValueError(f"{name} is on {t.device}, q on {q.device}")
        if not t.is_contiguous() or t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a contiguous bfloat16 tensor")
    return b, sq, skv, hq, hkv


def _plan(q, k, v, softmax_scale, causal):
    """(grid, kernel argument values, out, lse) of the launch; works on meta tensors."""
    b, s, skv, h, hkv = _check(q, k, v)
    reason = limits.fwd_shape_unsupported_reason(b, s, skv, h, hkv, bool(causal))
    if reason is not None:
        raise NotImplementedError(f"hk_attn_fwd: {reason}")
    scale = limits.D_QK**-0.5 if softmax_scale is None else float(softmax_scale)
    nqb = -(-s // limits.FWD_ROWS)
    npw = persistent_workgroups(b, s, h)
    out = torch.empty((b, s, h, limits.D_V), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((b, h, s), device=q.device, dtype=torch.float32)
    grid = (npw, 1, 1) if npw else (nqb, h, b)
    return grid, (q, k, v, out, lse, scale, s, s, h, b, nqb, npw), out, lse


def hk_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """Flash-attention forward on gfx1250, DeepSeek-V3 MLA head dims (qk 192, v 128), MHA, bottom-right causal.

    q / k [B, S, H, 192], v [B, S, H, 128], bf16 contiguous (seqlen_q == seqlen_kv, ``limits``). Returns
    (out [B, S, H, 128] bf16, lse [B, H, S] fp32 natural log)."""
    hiprt.check_device(q.device)
    kernel = kernels()["k_mla_fwd"]
    with torch.cuda.device(q.device):
        grid, values, out, lse = _plan(q, k, v, softmax_scale, causal)
        kernel.call(grid, BLOCK, values, torch.cuda.current_stream(q.device))
    return out, lse
