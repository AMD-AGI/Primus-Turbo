###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hk_attn_bwd: the gfx1250 MLA backward with HIP dK/dV and dQ kernels.

Three launches, all on the caller's current stream, in this order:
  "delta"  FlyDSL k_delta of ``primus_turbo.flydsl.attention.gfx1250_mla_bwd`` (delta, and the per-row
           constants NL = -LOG2E*lse, ND = -scale*delta the dK/dV kernel reads);
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
"""

import math
import threading

import torch

from primus_turbo.flydsl.attention.gfx1250_mla_bwd import impl as _fly

from . import hiprt, limits

_k = _fly._k

# Head-group traversal targets (kernels.head_group): launch grids (hg, tiles, B*H/hg). Launch-only,
# outputs bitwise identical for every value the kernels' index proofs replay (the tuples below).
HEAD_GROUP_KV = 32  # k_dkdv96ds
HEAD_GROUP_Q = 32  # k_dqds_c
PROVEN_KV_TARGETS = (0, 8, 16, 24, 32, 64, 96, 128, 256)
PROVEN_Q_TARGETS = (0, 8, 16, 32, 64, 128, 256)
DS_MODE = 1  # k_dkdv96ds: store dS^T with num_records = the wave's true extent (0 drops every store)
DS_GUARD_B = 65536  # bytes before and after the layout in the scratch

# Validation only: allocate delta/dq/dk/dv and the whole dS scratch NaN-filled, so an element the
# kernels never write shows up as non-finite instead of as stale memory.
POISON = False

_KERNELS = None
_LOCK = threading.Lock()


def kernels():
    """{name: hiprt.Kernel} (code objects sha256-checked on first use; no GPU work here)."""
    global _KERNELS
    if _KERNELS is None:
        with _LOCK:
            if _KERNELS is None:
                _KERNELS = hiprt.load_kernels()
    return _KERNELS


def _alloc(shape, device, dtype):
    if POISON:
        return torch.full(shape, float("nan"), device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype)


def _ds_scratch(nbytes, device):
    t = torch.empty(((2 * DS_GUARD_B + nbytes) // 2,), dtype=torch.bfloat16, device=device)
    if POISON and t.device.type != "meta":
        t.fill_(float("nan"))
    return t, t[DS_GUARD_B // 2 : DS_GUARD_B // 2 + nbytes // 2]


def _plan(do, q, k, v, o, lse, softmax_scale, causal, stream, head_group=None):
    """Every launch hk_attn_bwd makes for these inputs, in issue order, plus what it allocates.

    launches: [(name, launcher, args)]: "delta" (FlyDSL launcher, its arguments), "dkdv" / "dq" (HIP
    kernel name, (grid, block, kernel argument values)). head_group: target for both kernels (None:
    HEAD_GROUP_KV / HEAD_GROUP_Q); one the index proofs do not replay is refused (ValueError).
    Works on meta tensors (stream unused until launch).
    """
    b, sq, skv, hq, hkv = _fly._check(do, q, k, v, o, lse)
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
    hg_kv, hg_q = _k.head_group(hkv, tkv), _k.head_group(hq, tq)
    assert hkv % hg_kv == 0 and hq % hg_q == 0, (hkv, hg_kv, hq, hg_q)
    ngz_kv, ngz_q = b * hkv // hg_kv, b * hq // hg_q
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(_k.D_QK)
    scale = float(softmax_scale)
    g = hq // hkv
    dev = q.device
    delta = _alloc((b, hq, sq), dev, torch.float32)
    nl = _alloc((b, hq, sq), dev, torch.float32)
    nd = _alloc((b, hq, sq), dev, torch.float32)
    dq = _alloc((b, sq, hq, _k.D_QK), dev, q.dtype)
    dk = _alloc((b, skv, hkv, _k.D_QK), dev, k.dtype)
    dv = _alloc((b, skv, hkv, _k.D_V), dev, v.dtype)
    ds_bytes = limits.ds_layout_bytes(b, sq, skv, hq, c)
    ds_full, ds = _ds_scratch(ds_bytes, dev)
    q_off, nhead, ntile = limits.dq_split(sq)
    delta_args = (do, o, delta, lse, nl, nd, scale, sq, hq, b * sq * hq, sq // _k.ROWS_DELTA, b * hq, stream)
    dkdv_values = (
        q,
        k,
        v,
        do,
        nl,
        nd,
        dv,
        dk,
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
        b,
        hg_kv,
        DS_MODE,
    )
    dq_values = (ds, k, dq, sq, skv, hq, hkv, g, skv // 32, skv - sq, c, q_off, ntile, hg_q)
    launches = [
        ("delta", _k.launch_delta, delta_args),
        ("dkdv", "k_dkdv96ds", ((hg_kv, limits.k3_nblk(skv), ngz_kv), 64, dkdv_values)),
        ("dq", "k_dqds_c", ((hg_q, ntile + nhead, ngz_q), 128, dq_values)),
    ]
    return {
        "launches": launches,
        "outputs": (dq, dk, dv),
        "delta": delta,
        "grids": {
            "delta": (sq // _k.ROWS_DELTA, b * hq, 1),
            "dkdv": launches[1][2][0],
            "dq": launches[2][2][0],
        },
        "head_group": {"kv": hg_kv, "q": hg_q},
        "ds_layout_bytes": ds_bytes,
        "ds_full": ds_full,
    }


def hk_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True, head_group=None):
    """Flash-attention backward on gfx1250, DeepSeek-V3 MLA head dims (qk 192, v 128), MHA.

    Same contract as ``primus_turbo.flydsl.attention.gfx1250_mla_bwd.flydsl_attn_bwd``:
    q [B, Sq, H, 192], o/do [B, Sq, H, 128], k [B, Skv, H, 192], v [B, Skv, H, 128], bf16 contiguous;
    lse [B, H, Sq] fp32 natural log; bottom-right causal. Returns (dq, dk, dv) laid out like q/k/v.
    Also needs Sq % 384 in {0, 256} and Skv >= Sq (``limits.shape_unsupported_reason``).
    """
    _fly._check_device_once(q.device)
    lse = lse.contiguous().float()
    stream = torch.cuda.current_stream()
    plan = _plan(do, q, k, v, o, lse, softmax_scale, causal, stream, head_group)
    table = kernels()
    for name, fn, args in plan["launches"]:
        if isinstance(fn, str):
            grid, block, values = args
            table[fn].call(grid, block, values, stream)
        else:
            _fly._launch(name, fn, args)
    return plan["outputs"]
