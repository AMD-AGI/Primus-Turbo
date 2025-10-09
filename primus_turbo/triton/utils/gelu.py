###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl

from primus_turbo.triton.utils.triton_lang_helper import pow, tl_extra_shim

tanh = tl_extra_shim.tanh
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp


@triton.jit
def gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    output = 0.5 * x * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32, 2))))
    return output


@triton.jit
def gelu_none(x):
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    output = 0.5 * x * (1 + erf(x.to(tl.float32) * scale))
    return output


@triton.jit
def gelu_bwd_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * pow(x_fp32, 2)))
    dydx = 0.5 * x_fp32 * ((1 - pow(tanh_out, 2)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2))) + 0.5 * (
        1 + tanh_out
    )
    dx = dydx * dy
    return dx


@triton.jit
def gelu_bwd_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2)) + 0.5 * erf(scale1 * x_fp32) + 0.5
    dx = dydx * dy
    return dx
