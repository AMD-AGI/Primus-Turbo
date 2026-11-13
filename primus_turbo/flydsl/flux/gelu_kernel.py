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

"""FlyDSL kernel for the FLUX MLP's GELU: ``cat([prefix, gelu_tanh(h)], dim=1)`` in one pass.

FLUX's single-stream block feeds ``proj_out`` with the attention output and the MLP's GELU output
side by side. This kernel writes that buffer straight from the up-projection's pre-activation
``h``: the prefix columns copied, the others through the tanh GELU, which it evaluates as
Inductor's ``gelu(approximate="tanh")`` does, operation for operation.

One block of 384 threads covers 3,072 columns, 8 BF16 values per thread; each of the 1,024 blocks
walks its rows two at a time.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
import torch
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
)

__all__ = ["GELU_CAT_TILE", "gelu_cat_supported", "flydsl_gelu_cat_forward"]

_VEC, _THREADS = 8, 384
GELU_CAT_TILE = _VEC * _THREADS
_GRID, _ROWS = 1024, 2


def gelu_cat_supported(m: int, prefix_width: int, n: int) -> bool:
    """Whether the tiling covers ``m`` rows of ``n`` GELU columns after ``prefix_width`` copied ones."""
    # Each of the _GRID blocks walks its rows _ROWS at a time.
    return n % GELU_CAT_TILE == 0 and prefix_width % GELU_CAT_TILE == 0 and m % (_GRID * _ROWS) == 0


def _tanh(v, one, two, two_log2e):
    # fmath.tanh lowers to llvm.tanh, which the AMDGPU backend cannot compile without the device
    # math library. 1 - 2 / (1 + e^(2v)) on the hardware exp2 and reciprocal (v_exp_f32, v_rcp_f32,
    # ~1 ulp each) saturates cleanly at +-1, and its FP32 error is far below the BF16 rounding that
    # follows.
    e = fmath.exp2(fx.arith.mulf(v, two_log2e))
    r = fx.arith.divf(one, fx.arith.addf(one, e), fastmath=fx.FastMathFlags.afn)
    return fx.arith.subf(one, fx.arith.mulf(two, r))


def _make_fwd_kernel(m: int, p: int, n: int, aux_stride: int):
    width, rows_per_block = p + n, m // _GRID
    copy_steps, gelu_steps = p // GELU_CAT_TILE, n // GELU_CAT_TILE

    @flyc.kernel(name="flux_gelu_cat_fwd_kernel", known_block_size=[_THREADS, 1, 1])
    def kernel(AUX: fx.Tensor, PRE: fx.Tensor, OUT: fx.Tensor):
        f32v = fx.T.vec(_VEC, fx.T.f32)
        bf16v = fx.T.vec(_VEC, fx.T.bf16)
        bf16 = fx.T.bf16
        col = fx.thread_idx.x * fx.Int32(_VEC)
        bx, _, _ = fx.block_idx
        aux_r, out_r = (create_buffer_resource(x, max_size=True) for x in (AUX, OUT))
        pre_r = create_buffer_resource(PRE, max_size=True) if p else None
        half, c3, k, one, two, two_log2e = (
            fx.arith.constant_vector(v, f32v)
            for v in (0.5, 0.044715, 0.7978845608028654, 1.0, 2.0, 2.8853900817779268)
        )

        def rows(it):
            # _ROWS rows per iteration, every load issued before any math.
            rs = [bx + (ArithValue(it) * _ROWS + j) * _GRID for j in range(_ROWS)]
            pres = [
                [
                    buffer_load(
                        pre_r, r * fx.Int32(p) + fx.Int32(s * GELU_CAT_TILE) + col, vec_width=_VEC, dtype=bf16
                    )
                    for s in range(copy_steps)
                ]
                for r in rs
            ]
            hs = [
                [
                    buffer_load(
                        aux_r,
                        r * fx.Int32(aux_stride) + fx.Int32(s * GELU_CAT_TILE) + col,
                        vec_width=_VEC,
                        dtype=bf16,
                    )
                    for s in range(gelu_steps)
                ]
                for r in rs
            ]
            for r, pr, hr in zip(rs, pres, hs):
                for s, v in enumerate(pr):
                    buffer_store(v, out_r, r * fx.Int32(width) + fx.Int32(s * GELU_CAT_TILE) + col)
                for s, hb in enumerate(hr):
                    h = fx.arith.extf(f32v, hb)
                    # Inductor's gelu(approximate="tanh"): (h * 0.5) * (tanh((h + h*h*h*0.044715) * k) + 1).
                    inner = fx.arith.mulf(
                        fx.arith.addf(h, fx.arith.mulf(fx.arith.mulf(fx.arith.mulf(h, h), h), c3)), k
                    )
                    y = fx.arith.mulf(
                        fx.arith.mulf(h, half), fx.arith.addf(_tanh(inner, one, two, two_log2e), one)
                    )
                    buffer_store(
                        fx.arith.trunc_f(bf16v, y),
                        out_r,
                        r * fx.Int32(width) + fx.Int32(p + s * GELU_CAT_TILE) + col,
                    )

        for it in range(fx.Int32(rows_per_block // _ROWS)):
            rows(it)

    return kernel


@flyc.jit
def _compiled_fwd(
    AUX,
    PRE,
    OUT,
    M: fx.Constexpr[int],
    P: fx.Constexpr[int],
    N: fx.Constexpr[int],
    AUX_STRIDE: fx.Constexpr[int],
    stream: fx.Stream,
):
    _make_fwd_kernel(M, P, N, AUX_STRIDE)(AUX, PRE, OUT).launch(
        grid=(_GRID, 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


def flydsl_gelu_cat_forward(aux: torch.Tensor, prefix: Optional[torch.Tensor]) -> torch.Tensor:
    """``cat([prefix, gelu_tanh(aux)], dim=1)`` for a 2-D BF16 ``aux`` with unit column stride (its
    rows may be a column slice of a wider buffer) and an optional BF16 ``prefix`` with as many rows.
    The caller checks :func:`gelu_cat_supported`."""
    m, n = aux.shape
    if aux.stride(1) != 1:
        aux = aux.contiguous()
    p = 0 if prefix is None else prefix.shape[1]
    out = torch.empty((m, p + n), device=aux.device, dtype=aux.dtype)
    _compiled_fwd(
        AUX=aux,
        PRE=aux if prefix is None else prefix.contiguous(),
        OUT=out,
        M=m,
        P=p,
        N=n,
        AUX_STRIDE=aux.stride(0),
        stream=torch.cuda.current_stream(aux.device),
    )
    return out
