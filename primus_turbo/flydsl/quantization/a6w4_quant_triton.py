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

"""A6W4 operand quantizers for FLUX: MXFP6 (E2M3) activations and MXFP4 (E2M1) weights.

Both operands are rotated with the same block-diagonal H16 along K, which cancels in the
product, then quantized per 32 with E8M0 scales. The rounding and scale rules are the ones
the A6W4 emulation in fwddiag used (and converged with):
  activation: scale = 2^(floor(log2 amax) - 2), E2M3 round-half-even, clamp 7.5
  weight:     the FlyDSL MXFP4 packer's scale (amax mantissa rounded at 2^21), E2M1 RNE, clamp 6

quant_act_a6w4 and quant_w_a6w4 return exactly what primus_turbo.flydsl.gemm.gemm_a6w4_kernel
consumes:
  A: FP8-padded packed FP6 [M, K] uint8 (per 32 values, 24 bytes holding a little-endian
     6-bit code stream, then 8 zero bytes), scales in shuffle_scale_w4 layout
  W: packed FP4 (even element in the low nibble) in shuffle_weight_w4(16) layout, scales in
     shuffle_scale_w4 layout
Only the bytes are a contract; the returned tensors' shapes are not.
"""

import torch
import triton
import triton.language as tl

_RNE_MAGIC = tl.constexpr(12582912.0)  # 1.5 * 2^23: (v + M) - M rounds half-to-even for 0 <= v < 2^22
_H32 = {}


def _h32(device):
    """Block-diagonal [32, 32] of two Sylvester H16 / 4, as bf16 (entries +-0.25 are exact)."""
    h = _H32.get(device)
    if h is None:
        h16 = torch.ones(1, 1)
        for _ in range(4):
            h16 = torch.cat((torch.cat((h16, h16), 1), torch.cat((h16, -h16), 1)), 0)
        h = torch.block_diag(h16, h16).mul_(0.25).to(device=device, dtype=torch.bfloat16)
        _H32[device] = h
    return h


@triton.jit
def _rotate_and_scale(x_ptr, h_ptr, g, mask, ROUND_SCALE: tl.constexpr, EMAX: tl.constexpr):
    j = tl.arange(0, 32)
    x = tl.load(x_ptr + g[:, None] * 32 + j[None, :], mask=mask[:, None], other=0.0)
    h = tl.load(h_ptr + j[:, None] * 32 + j[None, :])
    z = tl.dot(x, h, out_dtype=tl.float32)
    bits = tl.max(tl.abs(z), axis=1).to(tl.int32, bitcast=True)
    if ROUND_SCALE:
        bits = bits + (1 << 21)
    e8 = tl.minimum(tl.maximum(((bits >> 23) & 0xFF) - 2, 0), 254)
    inv = tl.exp2((127 - e8).to(tl.float32))
    a = tl.minimum(tl.abs(z) * inv[:, None], EMAX)
    neg = z < 0
    return a, neg, e8


_QUANT_CONFIGS = [triton.Config({"BG": bg}, num_warps=nw) for bg in (64, 128, 256, 512) for nw in (2, 4, 8)]


@triton.autotune(configs=_QUANT_CONFIGS, key=["G"])
@triton.jit
def _quant_act_fp6_kernel(x_ptr, h_ptr, out_ptr, sc_ptr, G, BG: tl.constexpr):
    g = tl.program_id(0) * BG + tl.arange(0, BG)
    mask = g < G
    a, neg, e8 = _rotate_and_scale(x_ptr, h_ptr, g, mask, False, 7.5)
    k1 = (a * 8.0 + _RNE_MAGIC) - _RNE_MAGIC
    k2 = (a * 4.0 + _RNE_MAGIC) - _RNE_MAGIC
    k3 = (a * 2.0 + _RNE_MAGIC) - _RNE_MAGIC
    code = tl.where(a < 2.0, k1, tl.where(a < 4.0, k2 + 8.0, k3 + 16.0)).to(tl.int32)
    code = tl.where(neg, code | 32, code)
    c = tl.reshape(code, (BG, 8, 2, 2))
    p, q = tl.split(c)
    e0, e2 = tl.split(p)
    e1, e3 = tl.split(q)
    b0 = ((e1 & 3) << 6) | e0
    b1 = ((e2 & 15) << 4) | (e1 >> 2)
    b2 = (e3 << 2) | (e2 >> 4)
    i = tl.arange(0, 8)
    base = out_ptr + g[:, None] * 32 + i[None, :] * 3
    m2 = mask[:, None]
    tl.store(base, b0.to(tl.uint8), mask=m2)
    tl.store(base + 1, b1.to(tl.uint8), mask=m2)
    tl.store(base + 2, b2.to(tl.uint8), mask=m2)
    tl.store(out_ptr + g[:, None] * 32 + 24 + i[None, :], tl.zeros((BG, 8), tl.uint8), mask=m2)
    tl.store(sc_ptr + g, e8.to(tl.uint8), mask=mask)


@triton.autotune(configs=_QUANT_CONFIGS, key=["G"])
@triton.jit
def _quant_w_fp4_kernel(x_ptr, h_ptr, out_ptr, sc_ptr, G, BG: tl.constexpr):
    g = tl.program_id(0) * BG + tl.arange(0, BG)
    mask = g < G
    a, neg, e8 = _rotate_and_scale(x_ptr, h_ptr, g, mask, True, 6.0)
    k1 = (a * 2.0 + _RNE_MAGIC) - _RNE_MAGIC
    k2 = (a + _RNE_MAGIC) - _RNE_MAGIC
    k3 = (a * 0.5 + _RNE_MAGIC) - _RNE_MAGIC
    code = tl.where(a < 2.0, k1, tl.where(a < 4.0, k2 + 2.0, k3 + 4.0)).to(tl.int32)
    code = tl.where(neg, code | 8, code)
    lo, hi = tl.split(tl.reshape(code, (BG, 16, 2)))
    i = tl.arange(0, 16)
    tl.store(out_ptr + g[:, None] * 16 + i[None, :], (lo | (hi << 4)).to(tl.uint8), mask=mask[:, None])
    tl.store(sc_ptr + g, e8.to(tl.uint8), mask=mask)


def shuffle_scale(s):
    """FlyDSL's shuffle_scale_w4(s, 1, False) for an [R, K/32] E8M0 tensor (R % 32, K % 256)."""
    r, k = s.shape
    return s.view(r // 32, 2, 16, k // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous()


def shuffle_weight(wq):
    """FlyDSL's shuffle_weight_w4(wq, 16, False, False) for packed FP4 [N, K/2]."""
    n, kp = wq.shape
    return wq.view(n // 16, 16, kp // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def quant_act_fp6(x):
    """bf16 [M, K] -> (FP8-padded packed FP6 [M, K] uint8, unshuffled E8M0 [M, K/32] uint8)."""
    m, k = x.shape
    out = torch.empty(m, k, dtype=torch.uint8, device=x.device)
    sc = torch.empty(m, k // 32, dtype=torch.uint8, device=x.device)
    g = m * k // 32
    _quant_act_fp6_kernel[lambda meta: (triton.cdiv(g, meta["BG"]),)](x, _h32(x.device), out, sc, g)
    return out, sc


def quant_w_fp4(w):
    """bf16 [N, K] -> (packed FP4 [N, K/2] uint8, unshuffled E8M0 [N, K/32] uint8)."""
    n, k = w.shape
    out = torch.empty(n, k // 2, dtype=torch.uint8, device=w.device)
    sc = torch.empty(n, k // 32, dtype=torch.uint8, device=w.device)
    g = n * k // 32
    _quant_w_fp4_kernel[lambda meta: (triton.cdiv(g, meta["BG"]),)](w, _h32(w.device), out, sc, g)
    return out, sc


def quant_act_a6w4(x):
    """bf16 [M, K] contiguous, M % 32 == 0, K % 256 == 0 -> (FP6 A, shuffled A scales) for gemm_a6w4."""
    aq, sa = quant_act_fp6(x)
    return aq, shuffle_scale(sa)


def quant_w_a6w4(w):
    """bf16 [N, K] contiguous, N % 32 == 0, K % 256 == 0 -> (preshuffled FP4 W, shuffled W scales)."""
    wq, sb = quant_w_fp4(w)
    return shuffle_weight(wq), shuffle_scale(sb)
