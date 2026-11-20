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
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pow2_floor(s):
    return (s.to(tl.int32, bitcast=True) & 0x7F800000).to(tl.float32, bitcast=True)


@triton.jit
def _quant_dual_t_kernel(
    x_ptr, qr_ptr, sr_ptr, er_ptr, qt_ptr, et_ptr,
    M, N,
    BLOCK: tl.constexpr, FP8_MAX: tl.constexpr, WITH_ROW: tl.constexpr, WITH_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.cast(pid_m * BLOCK + tl.arange(0, BLOCK), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK + tl.arange(0, BLOCK), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)
    nb_n = tl.cdiv(N, BLOCK)
    nb_m = tl.cdiv(M, BLOCK)
    if WITH_ROW:
        sc = _pow2_floor(FP8_MAX / tl.maximum(tl.max(ax, axis=1), 1e-4))  # [BLOCK] per row
        q = tl.clamp(x * sc[:, None], -FP8_MAX, FP8_MAX)
        tl.store(qr_ptr + offs_m[:, None] * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty), mask=mask)
        inv = 1.0 / sc
        rmask = offs_m < M
        tl.store(sr_ptr + offs_m * nb_n + pid_n, inv, mask=rmask)
        tl.store(er_ptr + offs_m * nb_n + pid_n,
                 ((inv.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8), mask=rmask)
    if WITH_T:
        sc = _pow2_floor(FP8_MAX / tl.maximum(tl.max(ax, axis=0), 1e-4))  # [BLOCK] per column
        q = tl.clamp(x * sc[None, :], -FP8_MAX, FP8_MAX)
        qt = tl.trans(q.to(qt_ptr.dtype.element_ty))
        tmask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(qt_ptr + offs_n[:, None] * M + offs_m[None, :], qt, mask=tmask)
        inv = 1.0 / sc
        tl.store(et_ptr + offs_n * nb_m + pid_m,
                 ((inv.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8), mask=offs_n < N)


def quant_dual_t_e8m0(x: torch.Tensor, dtype=torch.float8_e4m3fn, with_row=True, with_t=True, block=128):
    """Returns (q_row [M,N], scale_inv_row fp32 [M,N/128], e8m0_row u8 [M,N/128], q_t [N,M], e8m0_t u8 [N,M/128]).
    Outputs not requested are None."""
    assert x.dim() == 2 and x.is_contiguous()
    M, N = x.shape
    nb_m, nb_n = triton.cdiv(M, block), triton.cdiv(N, block)
    dev = x.device
    qr = torch.empty((M, N), dtype=dtype, device=dev) if with_row else None
    sr = torch.empty((M, nb_n), dtype=torch.float32, device=dev) if with_row else None
    er = torch.empty((M, nb_n), dtype=torch.uint8, device=dev) if with_row else None
    qt = torch.empty((N, M), dtype=dtype, device=dev) if with_t else None
    et = torch.empty((N, nb_m), dtype=torch.uint8, device=dev) if with_t else None
    dummy = x
    _quant_dual_t_kernel[(nb_m, nb_n)](
        x, qr if with_row else dummy, sr if with_row else dummy, er if with_row else dummy,
        qt if with_t else dummy, et if with_t else dummy,
        M, N, BLOCK=block, FP8_MAX=float(torch.finfo(dtype).max), WITH_ROW=with_row, WITH_T=with_t,
        num_warps=8,
    )
    return qr, sr, er, qt, et


def e8m0_of(scale_inv: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two scale_inv -> E8M0 exponent bytes (same shape)."""
    return ((scale_inv.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
