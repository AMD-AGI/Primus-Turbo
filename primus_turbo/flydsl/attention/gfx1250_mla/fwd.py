###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################


"""Host entry of the gfx1250 MLA attention forward."""

import contextlib
from typing import Tuple

import flydsl.compiler as flyc
import torch

from . import NUM_XCD
from .common import num_cus
from .fwd_kernel import BM, D_QK, D_V, launch_mla_fwd

_COMPILED = {}  # device index -> flyc.compile's CompiledFunction (every shape-dependent argument is a runtime scalar)
# Persistent workgroups (one per CU, each serving the head-batches of its XCD) where the grid allows them; False: one
# workgroup per q block.
PERSIST = True


def launch_grid(q: torch.Tensor, k: torch.Tensor) -> Tuple[int, int]:
    """(npw, nqb) of the kernel for q [B, Sq, H, 192], k [B, Skv, H, 192]: nqb q blocks of 256 rows per (batch, head),
    and npw > 0 workgroups (one per CU) on the static persistent schedule, or 0 for one workgroup per q block. The
    schedule covers seqlen_q == seqlen_kv in whole q blocks when every XCD group of CUs takes a whole number of
    q-block rows and an even number of steps (two steps pair q block i with nqb - 1 - i: equal work per workgroup)."""
    b, sq, h, _ = q.shape
    nqb = -(-sq // BM)
    if not PERSIST or k.shape[1] != sq or sq % BM:
        return 0, nqb
    cus = num_cus(q.device)
    per_group, hb = cus // NUM_XCD, b * h
    if cus % NUM_XCD or per_group % nqb or hb % NUM_XCD:
        return 0, nqb
    steps, rem = divmod(hb // NUM_XCD, per_group // nqb)
    return (cus if rem == 0 and steps % 2 == 0 else 0), nqb


def flash_attn_mla_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal attention forward for q/k head dim 192 and v head dim 128 on gfx1250.

    q [B, Sq, H, 192], k [B, Skv, H, 192], v [B, Skv, H, 128]: contiguous bf16 on one gfx1250
    device, Sq and Skv multiples of 64 (the kernel's kv tiles and its mask's diagonal tiles), Sq <= Skv,
    out under 2 GiB. The mask is bottom-right: query i attends keys j <= i + Skv - Sq. Returns
    o [B, Sq, H, 128] bf16 and lse [B, H, Sq] fp32 (natural log).
    """
    if not causal:
        raise ValueError("the gfx1250 MLA attention forward supports causal attention only")
    for name, t, dim in (("q", q, D_QK), ("k", k, D_QK), ("v", v, D_V)):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[-1] != dim or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [B, S, H, {dim}] bfloat16 tensor")
    if k.device != q.device or v.device != q.device:
        raise ValueError(f"q/k/v must be on one device, got {q.device}/{k.device}/{v.device}")
    b, sq, h, _ = q.shape
    skv = k.shape[1]
    if tuple(k.shape[:3]) != (b, skv, h) or tuple(v.shape[:3]) != (b, skv, h):
        raise ValueError(
            f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}: batch and heads must match (MHA)"
        )
    if b == 0 or h == 0:
        raise ValueError(f"batch and heads must be positive, got {b} / {h}")
    if sq == 0 or sq % 64 or skv % 64 or sq > skv:
        raise ValueError(
            f"seqlen_q and seqlen_kv must be positive multiples of 64 with seqlen_q <= seqlen_kv, got {sq} / {skv}"
        )
    if b * h * sq * D_V * 2 >= 2**31:  # (lse, 4 bytes per row, is then far below its 2 GiB)
        raise ValueError(
            f"out must be under 2 GiB (the kernel's 32-bit byte counts), got {b} * {sq} * {h} rows"
        )

    out = torch.empty((b, sq, h, D_V), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, sq), dtype=torch.float32, device=q.device)
    # q's device as the current one: its stream and compiled kernel
    on_q_device = q.device.type != "cuda" or q.device.index == torch.cuda.current_device()
    with contextlib.nullcontext() if on_q_device else torch.cuda.device(q.device):
        npw, nqb = launch_grid(q, k)
        args = (out, q, k, v, lse, float(softmax_scale), sq, skv, h, b, nqb, npw, torch.cuda.current_stream())
        fn = _COMPILED.get(q.device.index)
        if fn is not None:
            fn(*args)
        else:
            # flyc.compile launches the kernel itself; it returns None under COMPILE_ONLY.
            fn = flyc.compile(launch_mla_fwd, *args)
            if fn is not None:
                _COMPILED[q.device.index] = fn
    return out, lse
