###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Single-scan rowwise FP8 quant in Triton.

Replaces the 2-scan pattern at csrc/kernels/quantization/quantization.cu:221
(quantize_rowwise_row_major_two_scan_kernel) which reads each row twice
(amax pass + quant pass). The 2-scan variant is inherently HBM-limited at
50% (2 reads × 1 write = 3 HBM touches per element instead of 1+1=2).

1-scan design: hold the full row in LDS/registers across the block-reduce
amax, then scale+store in one stream. Requires BLOCK_N >= row_len.

Shape limits: row_len up to the LDS budget. MI355X has 160 KB LDS/CU; at
fp32 compute, BLOCK_N=16384 uses 64 KB/CU, fits with room for reduction
scratch. For row_len > 16384, fall back to the 2-scan CPP kernel (or a
variant that tiles the row and does a 2-phase amax).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def quantize_fp8_rowwise_one_scan_kernel(
    x_ptr,                  # [M, N]  bf16/fp16
    out_fp8_ptr,            # [M, N]  fp8
    out_scale_inv_ptr,      # [M]     fp32 (1/scale)
    M, N,
    stride_x, stride_out,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,  # must be >= N, power of 2
):
    """One program per row; read row once, compute amax via block-reduce, quant in-place."""
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_off = row.to(tl.int64) * stride_x
    x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    scale     = FP8_MAX / amax
    scale_inv = 1.0 / scale

    x_q = tl.clamp(x * scale, -FP8_MAX, FP8_MAX).to(out_fp8_ptr.dtype.element_ty)

    tl.store(out_fp8_ptr + row.to(tl.int64) * stride_out + cols, x_q, mask=mask)
    tl.store(out_scale_inv_ptr + row, scale_inv)


def quantize_fp8_rowwise_one_scan(
    x: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-scan Triton rowwise FP8 quant. Returns (out_fp8, scale_inv).

    Drop-in for ``primus_turbo.pytorch.ops.quantize_fp8(x, dtype, ROWWISE, axis=-1)``
    when x.shape[-1] <= 16384 (LDS-fit limit on MI355X).
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M, N = x.shape
    assert x.is_contiguous(), "x must be contiguous"

    BLOCK_N = max(64, triton.next_power_of_2(N))
    assert BLOCK_N <= 16384, f"N={N} exceeds 1-scan LDS budget; use 2-scan CPP kernel"

    out_fp8 = torch.empty((M, N), dtype=out_dtype, device=x.device)
    scale_inv = torch.empty((M,), dtype=torch.float32, device=x.device)

    fp8_max = torch.finfo(out_dtype).max
    quantize_fp8_rowwise_one_scan_kernel[(M,)](
        x, out_fp8, scale_inv,
        M, N,
        x.stride(0), out_fp8.stride(0),
        FP8_MAX=fp8_max,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out_fp8, scale_inv
