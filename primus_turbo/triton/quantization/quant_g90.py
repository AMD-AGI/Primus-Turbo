###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""1x128 FP8 quantization for the gfx1250 g90 blockwise GEMM (UE8M0, paper recipe).

One pass over a 2D bf16/fp32 tensor x[M, N] writes
  * row-quantized x (1x128 tiles along N): fp8 [M, N], fp32 scale_inv [M, N/128], E8M0 bytes [M, N/128]
  * token-major transposed copy quantized along M (1x128 tiles along M): fp8 [N, M], E8M0 bytes [N, M/128]
so the wgrad GEMM (reduction over tokens) is a plain NT GEMM with per-row scales on both operands,
and no per-call fp32 -> E8M0 conversion is needed. Numerics match Turbo's POW2 blockwise quant:
scale = pow2_floor(FP8_MAX / max(amax, 1e-4)), q = clamp(x * scale), scale_inv = 1 / scale (exact).
The exponent is taken from the amax bits and the scale applied as an exact power of two (no division,
quant_epi_gfx1250.ue8m0_exp_g90); bytes are identical to the former division form.

Options: ``scale_layout="packed"`` writes the E8M0 bytes in the dsv4up pack_scale BK256 layout (4x per 128
block) instead of one byte per block; ``rowmap`` fuses a row gather (MoE permute / dispatch): row i of the
outputs is x[rowmap[i]] (negative -> a zero row).
"""

import torch
import triton
import triton.language as tl

from primus_turbo.triton.quantization.quant_epi_gfx1250 import (
    SL_BYTE,
    SL_PACKED,
    fp8_max_exp,
    pow2_neg,
    pow2_pos,
    scale_layout_id,
    store_scale,
    ue8m0_exp_g90,
)


@triton.jit
def _pow2_floor(s):
    return (s.to(tl.int32, bitcast=True) & 0x7F800000).to(tl.float32, bitcast=True)


@triton.jit
def _quant_dual_t_kernel(
    x_ptr, qr_ptr, sr_ptr, er_ptr, qt_ptr, et_ptr, map_ptr,
    M, N, stride_x,
    BLOCK: tl.constexpr, FP8_MAX: tl.constexpr, WITH_ROW: tl.constexpr, WITH_T: tl.constexpr,
    MAX_EXP: tl.constexpr = 8, SL: tl.constexpr = 1, HAS_MAP: tl.constexpr = 0,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.cast(pid_m * BLOCK + tl.arange(0, BLOCK), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK + tl.arange(0, BLOCK), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if HAS_MAP:
        src = tl.load(map_ptr + offs_m, mask=offs_m < M, other=-1).to(tl.int64)
        lmask = mask & (src >= 0)[:, None]
        x = tl.load(x_ptr + tl.maximum(src, 0)[:, None] * stride_x + offs_n[None, :], mask=lmask,
                    other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs_m[:, None] * stride_x + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)
    nb_n = tl.cdiv(N, BLOCK)
    nb_m = tl.cdiv(M, BLOCK)
    if WITH_ROW:
        e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [BLOCK] per row
        q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(qr_ptr + offs_m[:, None] * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty), mask=mask)
        rmask = offs_m < M
        tl.store(sr_ptr + offs_m * nb_n + pid_n, pow2_pos(e), mask=rmask)
        store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SL, mask=rmask)
    if WITH_T:
        e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [BLOCK] per column
        q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX)
        qt = tl.trans(q.to(qt_ptr.dtype.element_ty))
        tmask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(qt_ptr + offs_n[:, None] * M + offs_m[None, :], qt, mask=tmask)
        store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SL, mask=offs_n < N)


def quant_dual_t_e8m0(x: torch.Tensor, dtype=torch.float8_e4m3fn, with_row=True, with_t=True, block=128,
                      scale_layout="g90", rowmap: torch.Tensor = None):
    """Returns (q_row [M,N], scale_inv_row fp32 [M,N/128], e8m0_row u8 [M,N/128], q_t [N,M], e8m0_t u8 [N,M/128]).
    Outputs not requested are None. With ``rowmap`` (int32 [M]) row i is x[rowmap[i]] (negative -> zeros).
    ``scale_layout="packed"``: e8m0_row [M/128, N/32*128], e8m0_t [N/128, M/32*128] (dsv4up layout)."""
    assert x.dim() == 2 and x.stride(1) == 1 and (rowmap is not None or x.is_contiguous())
    sl = scale_layout_id(scale_layout)
    N = x.shape[1]
    M = rowmap.shape[0] if rowmap is not None else x.shape[0]
    nb_m, nb_n = triton.cdiv(M, block), triton.cdiv(N, block)
    if sl == SL_PACKED:
        assert block == 128 and M % 128 == 0 and N % 128 == 0
    else:
        assert sl == SL_BYTE
    dev = x.device
    qr = torch.empty((M, N), dtype=dtype, device=dev) if with_row else None
    sr = torch.empty((M, nb_n), dtype=torch.float32, device=dev) if with_row else None
    qt = torch.empty((N, M), dtype=dtype, device=dev) if with_t else None
    if sl == SL_PACKED:
        er = torch.empty((M // 128, N // 32 * 128), dtype=torch.uint8, device=dev) if with_row else None
        et = torch.empty((N // 128, M // 32 * 128), dtype=torch.uint8, device=dev) if with_t else None
    else:
        er = torch.empty((M, nb_n), dtype=torch.uint8, device=dev) if with_row else None
        et = torch.empty((N, nb_m), dtype=torch.uint8, device=dev) if with_t else None
    dummy = x
    fmax = float(torch.finfo(dtype).max)
    _quant_dual_t_kernel[(nb_m, nb_n)](
        x, qr if with_row else dummy, sr if with_row else dummy, er if with_row else dummy,
        qt if with_t else dummy, et if with_t else dummy, rowmap if rowmap is not None else dummy,
        M, N, x.stride(0), BLOCK=block, FP8_MAX=fmax, WITH_ROW=with_row, WITH_T=with_t,
        MAX_EXP=fp8_max_exp(fmax), SL=sl, HAS_MAP=int(rowmap is not None),
        num_warps=8,
    )
    return qr, sr, er, qt, et


def e8m0_of(scale_inv: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two scale_inv -> E8M0 exponent bytes (same shape)."""
    return ((scale_inv.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
