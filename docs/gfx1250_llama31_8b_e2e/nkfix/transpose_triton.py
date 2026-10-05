###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tiled 2-D transpose for nkfix's operand copies (y = x.T, y contiguous).

torch's x.t().contiguous() runs the generic strided copy (elementwise_kernel_manual_unroll
direct_copy): in the 32-layer step it was 449 ms of the 554 ms nkfix overhead (~0.8 TB/s). A tiled
transpose loads a BR x BC tile coalesced along x's unit-stride dim and stores it coalesced along
y's rows. Pure data movement: the result is bit-identical to torch's copy (tprobe checks it).

Bounds: every load/store is masked with (r < R) & (c < C); offsets are int64 (the lm_head operand
has 4.2e9 elements).
"""

import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(x_ptr, y_ptr, R, C, sxr, sxc, BR: tl.constexpr, BC: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = pr.to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)
    c = pc.to(tl.int64) * BC + tl.arange(0, BC).to(tl.int64)
    m = (r[:, None] < R) & (c[None, :] < C)
    v = tl.load(x_ptr + r[:, None] * sxr + c[None, :] * sxc, mask=m)
    # y is (C, R) row-major: y[c, r] = x[r, c]
    tl.store(y_ptr + c[:, None] * R + r[None, :], tl.trans(v), mask=tl.trans(m))


def transpose_into(y, x, BR=64, BC=64, num_warps=4):
    """y (C, R) contiguous <- x (R, C) with any strides. Returns y."""
    R, C = x.shape
    assert y.shape == (C, R) and y.is_contiguous() and y.dtype == x.dtype and y.device == x.device
    grid = (triton.cdiv(R, BR), triton.cdiv(C, BC))
    _transpose_kernel[grid](x, y, R, C, x.stride(0), x.stride(1), BR=BR, BC=BC, num_warps=num_warps)
    return y
