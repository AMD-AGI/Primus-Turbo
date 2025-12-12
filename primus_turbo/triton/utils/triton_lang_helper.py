###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl

tl_extra_shim = triton.language.extra.hip.libdevice


@triton.jit
def pow(x, exponent):
    if x.type.element_ty == tl.bfloat16:
        return tl_extra_shim.pow(x.to(tl.float32), exponent)
    elif x.type.element_ty == tl.float16:
        return tl_extra_shim.pow(x.to(tl.float32), exponent)
    else:
        return tl_extra_shim.pow(x.to(tl.float64), exponent)


@triton.jit
def generate_randval_4x(m, n, philox_seed, philox_offset):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
    r1, r2, r3, r4 = tl.randint4x(philox_seed, rng_offsets)

    return r1, r2, r3, r4
