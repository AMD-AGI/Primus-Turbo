###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Finish a deferred stochastic rounding of FP4 (E2M1) codes, in place.

``quantize_mx_dual_out(..., col_prob=...)`` emits each FP4 code rounded down and, nibble for nibble in a second
buffer of the same layout, the 4-bit probability ``p`` (in sixteenths) of rounding it up. ``fp4_prob_round`` draws a
uniform 4-bit value per code from ``seed`` and the code's position and adds one to the code's magnitude when the
draw is below ``p`` -- so a code rounds up with probability ``p / 16``. One copy of codes and probabilities can thus
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
def _fp4_prob_round_kernel(codes_ptr, probs_ptr, n, seed, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(codes_ptr + offs, mask=mask, other=0).to(tl.uint32)
    p = tl.load(probs_ptr + offs, mask=mask, other=0).to(tl.uint32)
    r = _mix((offs.to(tl.uint32) * 0x9E3779B1) ^ seed)
    up_lo = ((r >> 28) & 15) < (p & 15)
    up_hi = ((r >> 24) & 15) < ((p >> 4) & 15)
    # a code with a nonzero probability is below the top magnitude (7), so the increment stays inside the nibble
    c = c + up_lo.to(tl.uint32) + (up_hi.to(tl.uint32) << 4)
    tl.store(codes_ptr + offs, c.to(tl.uint8), mask=mask)


def fp4_prob_round(codes: torch.Tensor, probs: torch.Tensor, seed: int) -> torch.Tensor:
    """In place on ``codes`` (uint8, two FP4 codes per byte, rounded down) with ``probs`` (uint8, the same layout:
    each nibble the round-up probability of the code nibble at the same position, in sixteenths)."""
    assert codes.dtype == torch.uint8 and probs.dtype == torch.uint8 and codes.numel() == probs.numel()
    assert codes.is_contiguous() and probs.is_contiguous() and codes.device == probs.device
    n = codes.numel()
    if n:
        BLOCK = 4096
        _fp4_prob_round_kernel[(triton.cdiv(n, BLOCK),)](codes, probs, n, seed & 0xFFFFFFFF, BLOCK=BLOCK)
    return codes
