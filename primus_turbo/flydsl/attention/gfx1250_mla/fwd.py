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

from .fwd_kernel import BLOCK_M, D_QK, D_V, launch_mla_fwd

_COMPILED = {}
_NUM_CUS = {}
# Ticket counters of the persistent schedule per (device, stream): 8 group counters and a done counter,
# int32s 256 B apart, all 0 between calls (the last workgroup to finish resets them), so calls on one
# stream share them and calls on two streams never do. A launch captured into a CUDA graph gets a buffer
# of its own, zero-filled by the graph before the kernel (graphs share their capture stream).
_COUNTERS = {}
# Persistent workgroups (at most one per CU, taking q blocks from counters) where the grid allows them.
PERSIST = True


def persist_mode(q: torch.Tensor, k: torch.Tensor) -> Tuple[int, int]:
    """(persist_hb, workgroups) of the persistent schedule for q [B, Sq, H, 192] and k [B, Skv, H, 192],
    or (0, 0) for one workgroup per q block. Persistent mode needs full q blocks, seqlen_q == seqlen_kv
    (the kernel takes the next ticket at a fixed tile of the causal block), q blocks and heads * batch
    in multiples of 16 (the grouped dispatch order), a CU count that is a multiple of the 8 ticket
    groups and at least two blocks per CU."""
    b, sq, h, _ = q.shape
    if not PERSIST or sq % BLOCK_M or k.shape[1] != sq or h >= 1 << 16 or b >= 1 << 15:
        return 0, 0
    if q.device.type != "cuda":  # COMPILE_ONLY
        return h | (b << 16), 256
    idx = q.device.index
    if idx not in _NUM_CUS:
        _NUM_CUS[idx] = torch.cuda.get_device_properties(q.device).multi_processor_count
    blocks = (sq // BLOCK_M) * h * b
    if blocks < 2 * _NUM_CUS[idx] or (sq // BLOCK_M) % 16 or (b * h) % 16 or _NUM_CUS[idx] % 8:
        return 0, 0
    return h | (b << 16), _NUM_CUS[idx]


def _counter(q: torch.Tensor, stream) -> torch.Tensor:
    if q.device.type != "cuda":  # COMPILE_ONLY
        return torch.zeros(4, dtype=torch.int32)
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros(9 * 64, dtype=torch.int32, device=q.device)
    key = (q.device.index, stream.cuda_stream)
    c = _COUNTERS.get(key)
    if c is None:
        c = torch.zeros(9 * 64, dtype=torch.int32, device=q.device)
        _COUNTERS[key] = c
    return c


def launch_args(q, k, v, out, lse, softmax_scale):
    """The arguments of ``launch_mla_fwd`` for contiguous q/k/v and out [B, Sq, H, 128], lse [B, H, Sq]."""
    b, sq, h, _ = q.shape
    strides = (
        *(t.stride(1) for t in (q, k, v, out)),
        *(t.stride(2) for t in (q, k, v, out)),
        *(lse.stride(i) for i in (2, 1, 0)),
    )
    persist_hb, wgs = persist_mode(q, k)
    stream = torch.cuda.current_stream()
    # The counter fills the unused tensor slot; window_right = 0 (plain causal).
    return (
        out,
        q,
        k,
        v,
        lse,
        _counter(q, stream),
        float(softmax_scale),
        *strides,
        persist_hb,
        0,
        sq,
        k.shape[1],
        h,
        b,
        wgs,
        stream,
    )


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
    args = launch_args(q, k, v, out, lse, softmax_scale)
    fn = _COMPILED.get(q.device.index)
    if fn is not None:
        fn(*args)
    else:
        # flyc.compile launches the kernel itself; it returns None under COMPILE_ONLY.
        fn = flyc.compile(launch_mla_fwd, *args)
        if fn is not None:
            _COMPILED[q.device.index] = fn
    return out, lse
