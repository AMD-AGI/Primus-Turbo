###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Finish a deferred stochastic rounding of FP4 (E2M1) codes, in place.

``quantize_mx_dual_out(..., col_prob=...)`` emits each FP4 code rounded down and, in a second buffer, the probability
``p`` of rounding it up in k bits: k = 4 (sixteenths, nibble for nibble in the codes' layout) or k = 2 (quarters; the
byte of code bytes 2i and 2i + 1 is byte i). ``fp4_prob_round`` draws a uniform k-bit value per code from ``seed`` and
the code's position and adds one to the code's magnitude when the draw is below ``p`` -- so a code rounds up with
probability ``p / 2^k``. One copy of codes and probabilities can thus
serve several receivers, each finishing it with its own seed into an independent draw.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mix(x):
    # the same avalanche hash as the packers' sr_mix
    x = x ^ (x >> 16)
    x = x * 0x7FEB352D
    x = x ^ (x >> 15)
    x = x * 0x846CA68B
    x = x ^ (x >> 16)
    return x


@triton.jit
def _fp4_prob_round_kernel(codes_ptr, probs_ptr, n, seed, BITS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(codes_ptr + offs, mask=mask, other=0).to(tl.uint32)
    r = _mix((offs.to(tl.uint32) * 0x9E3779B1) ^ seed)
    if BITS == 4:
        p = tl.load(probs_ptr + offs, mask=mask, other=0).to(tl.uint32)
        up_lo = ((r >> 28) & 15) < (p & 15)
        up_hi = ((r >> 24) & 15) < ((p >> 4) & 15)
    else:  # 2 bits: code byte i's two probabilities are bits 4 (i % 2) .. +3 of probability byte i / 2
        pb = tl.load(probs_ptr + offs // 2, mask=mask, other=0).to(tl.uint32) >> ((offs % 2).to(tl.uint32) * 4)
        up_lo = ((r >> 30) & 3) < (pb & 3)
        up_hi = ((r >> 28) & 3) < ((pb >> 2) & 3)
    # a code with a nonzero probability is below the top magnitude (7), so the increment stays inside the nibble
    c = c + up_lo.to(tl.uint32) + (up_hi.to(tl.uint32) << 4)
    tl.store(codes_ptr + offs, c.to(tl.uint8), mask=mask)


def fp4_prob_round(codes: torch.Tensor, probs: torch.Tensor, seed: int) -> torch.Tensor:
    """In place on ``codes`` (uint8, two FP4 codes per byte, rounded down) with ``probs`` (uint8): ``codes``' size for
    4-bit probabilities (each nibble the probability of the code nibble at the same position, in sixteenths) or half
    of it for 2-bit ones (quarters; byte i holds code bytes 2i and 2i + 1, low bits first)."""
    assert codes.dtype == torch.uint8 and probs.dtype == torch.uint8
    bits = 4 if probs.numel() == codes.numel() else 2
    assert bits == 4 or probs.numel() * 2 == codes.numel(), "probs: codes' size (4-bit) or half of it (2-bit)"
    assert codes.is_contiguous() and probs.is_contiguous() and codes.device == probs.device
    n = codes.numel()
    if n:
        BLOCK = 4096
        _fp4_prob_round_kernel[(triton.cdiv(n, BLOCK),)](codes, probs, n, seed & 0xFFFFFFFF, BITS=bits, BLOCK=BLOCK)
    return codes
