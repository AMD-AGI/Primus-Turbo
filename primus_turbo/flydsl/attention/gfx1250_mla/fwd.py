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

from typing import Tuple

import flydsl.compiler as flyc
import torch

from .fwd_kernel import D_QK, D_V, launch_mla_fwd

_COMPILED = {}


def flash_attn_mla_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal attention forward for q/k head dim 192 and v head dim 128 on gfx1250.

    q [B, Sq, H, 192], k [B, Skv, H, 192], v [B, Skv, H, 128]: contiguous bf16 on one gfx1250
    device, Sq a multiple of 64 and Sq <= Skv. The mask is bottom-right: query i attends keys
    j <= i + Skv - Sq. Returns o [B, Sq, H, 128] bf16 and lse [B, H, Sq] fp32 (natural log).
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
    if sq == 0 or sq % 64 or sq > skv:
        raise ValueError(f"seqlen_q must be a positive multiple of 64 and <= seqlen_kv, got {sq} / {skv}")
    if b * h * sq * 4 >= 2**31:
        raise ValueError("lse is too large for the kernel's 32-bit offsets")

    out = torch.empty((b, sq, h, D_V), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, sq), dtype=torch.float32, device=q.device)
    strides = (
        *(t.stride(1) for t in (q, k, v, out)),
        *(t.stride(2) for t in (q, k, v, out)),
        *(lse.stride(i) for i in (2, 1, 0)),
    )
    # q fills the unused tensor slot; unused_i32 = 0 and window_right = 0 (plain causal).
    args = (
        out,
        q,
        k,
        v,
        lse,
        q,
        float(softmax_scale),
        *strides,
        0,
        0,
        sq,
        skv,
        h,
        b,
        torch.cuda.current_stream(),
    )
    fn = _COMPILED.get(q.device.index)
    if fn is not None:
        fn(*args)
    else:
        # flyc.compile launches the kernel itself; it returns None under COMPILE_ONLY.
        fn = flyc.compile(launch_mla_fwd, *args)
        if fn is not None:
            _COMPILED[q.device.index] = fn
    return out, lse
