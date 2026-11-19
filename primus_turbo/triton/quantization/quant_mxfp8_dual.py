###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP8 dual (row-wise + col-wise) quantization in Triton, for gfx1250.

The HIP ``quantize_mxfp8_dual`` kernel is written for wave64 CDNA (128x32 blocks,
4 B loads per lane, the 8-lane PK8 converter used half-empty on gfx1250) and runs
at ~25% of the HBM floor on gfx1250; the FlyDSL dual-cast kernel is CDNA4-only.
This kernel reads one [BM, BN] bf16/fp16 tile once and writes both directions:

* row-wise: ``q_row[M, N]`` + E8M0 ``s_row[M, N/32]`` (one scale per 1x32 run along N),
* col-wise: ``q_col[N, M]`` (the transpose) + E8M0 ``s_col[N, M/32]`` (per 32x1 run along M),

with optional 32x32 2D-block scales per direction (``use_2d_block``, used for weights).
The E8M0 rule mirrors the HIP ``compute_tile_scale``: round the amax mantissa to the
target format (round-half-up on the bit pattern), take its exponent, subtract the
target max power of two, clamp to [-127, 128]; values are soft-clamped to +-fp8_max
before the RNE cast. Requires M % BM == 0 and N % BN == 0 (no padding path).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 32


@triton.jit
def _e8m0(amax, VAL_TO_ADD: tl.constexpr, SUB: tl.constexpr):
    bits = amax.to(tl.int32, bitcast=True)
    e = ((bits + VAL_TO_ADD) >> 23) & 0x1FF
    e = e - SUB
    e = tl.minimum(tl.maximum(e, -127), 128)
    return e


@triton.jit
def _quant_mxfp8_dual_kernel(
    x_ptr,
    q_row_ptr,
    s_row_ptr,
    q_col_ptr,
    s_col_ptr,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ROW_2D: tl.constexpr,
    COL_2D: tl.constexpr,
    VAL_TO_ADD: tl.constexpr,
    SUB: tl.constexpr,
    FP8_MAX: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rm64 = rm.to(tl.int64)
    rn64 = rn.to(tl.int64)
    x = tl.load(x_ptr + rm64[:, None] * N + rn64[None, :]).to(tl.float32)
    ax = tl.abs(x)

    # ---- amax per 1x32 (row), 32x1 (col) and 32x32 (2D) block ----
    a_row = tl.max(tl.reshape(ax, (BM, BN // 32, 32)), axis=2)  # [BM, BN/32]
    a_col = tl.max(tl.reshape(ax, (BM // 32, 32, BN)), axis=1)  # [BM/32, BN]
    if ROW_2D or COL_2D:
        a_2d = tl.max(tl.reshape(a_row, (BM // 32, 32, BN // 32)), axis=1)  # [BM/32, BN/32]

    # ---- row-wise ----
    if ROW_2D:
        e_blk = _e8m0(a_2d, VAL_TO_ADD, SUB)  # [BM/32, BN/32]
        e_row = tl.reshape(tl.broadcast_to(e_blk[:, None, :], (BM // 32, 32, BN // 32)), (BM, BN // 32))
    else:
        e_row = _e8m0(a_row, VAL_TO_ADD, SUB)  # [BM, BN/32]
    inv_row = tl.exp2(-e_row.to(tl.float32))
    xr = tl.reshape(x, (BM, BN // 32, 32)) * inv_row[:, :, None]
    xr = tl.minimum(tl.maximum(xr, -FP8_MAX), FP8_MAX)
    q_row = tl.reshape(xr, (BM, BN)).to(OUT_DTYPE)
    tl.store(q_row_ptr + rm64[:, None] * N + rn64[None, :], q_row)
    rs = pid_n * (BN // 32) + tl.arange(0, BN // 32)
    tl.store(s_row_ptr + rm64[:, None] * (N // 32) + rs[None, :], (e_row + 127).to(tl.uint8))

    # ---- col-wise (stored transposed: q_col[N, M], s_col[N, M/32]) ----
    if COL_2D:
        e_blk_c = _e8m0(a_2d, VAL_TO_ADD, SUB)  # [BM/32, BN/32]
        e_col = tl.reshape(tl.broadcast_to(e_blk_c[:, :, None], (BM // 32, BN // 32, 32)), (BM // 32, BN))
    else:
        e_col = _e8m0(a_col, VAL_TO_ADD, SUB)  # [BM/32, BN]
    inv_col = tl.exp2(-e_col.to(tl.float32))
    xc = tl.reshape(x, (BM // 32, 32, BN)) * inv_col[:, None, :]
    xc = tl.minimum(tl.maximum(xc, -FP8_MAX), FP8_MAX)
    q_col = tl.trans(tl.reshape(xc, (BM, BN)).to(OUT_DTYPE))  # [BN, BM]
    tl.store(q_col_ptr + rn64[:, None] * M + rm64[None, :], q_col)
    cs = pid_m * (BM // 32) + tl.arange(0, BM // 32)
    tl.store(s_col_ptr + rn64[:, None] * (M // 32) + cs[None, :], tl.trans((e_col + 127).to(tl.uint8)))


_OUT = {
    torch.float8_e4m3fn: (tl.float8e4nv, 1 << 19, 127 + 8, 448.0),
    torch.float8_e5m2: (tl.float8e5, 1 << 20, 127 + 15, 57344.0),
}


def quant_mxfp8_dual_triton_supported(x: torch.Tensor, out_dtype: torch.dtype, bm: int = 64, bn: int = 128) -> bool:
    return (
        x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and out_dtype in _OUT
        and x.shape[0] % bm == 0
        and x.shape[1] % bn == 0
        and x.shape[0] > 0
        and x.shape[1] > 0
    )


def quant_mxfp8_dual_triton(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    row_2d: bool = False,
    col_2d: bool = False,
    bm: int = 64,
    bn: int = 128,
    num_warps: int = 8,
):
    """Returns ``(q_row [M,N], s_row [M,N/32], q_col [N,M], s_col [N,M/32])``, scales as e8m0."""
    assert quant_mxfp8_dual_triton_supported(x, out_dtype, bm, bn)
    x = x.contiguous()
    M, N = x.shape
    tl_dtype, val_to_add, sub, fp8_max = _OUT[out_dtype]
    q_row = torch.empty((M, N), dtype=out_dtype, device=x.device)
    s_row = torch.empty((M, N // _BLOCK), dtype=torch.uint8, device=x.device)
    q_col = torch.empty((N, M), dtype=out_dtype, device=x.device)
    s_col = torch.empty((N, M // _BLOCK), dtype=torch.uint8, device=x.device)
    grid = (M // bm, N // bn)
    _quant_mxfp8_dual_kernel[grid](
        x,
        q_row,
        s_row,
        q_col,
        s_col,
        M,
        N,
        BM=bm,
        BN=bn,
        ROW_2D=row_2d,
        COL_2D=col_2d,
        VAL_TO_ADD=val_to_add,
        SUB=sub,
        FP8_MAX=fp8_max,
        OUT_DTYPE=tl_dtype,
        num_warps=num_warps,
    )
    e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    return q_row, s_row.view(e8m0), q_col, s_col.view(e8m0)


def quant_mxfp8_dual_triton_config(M: int, N: int, row_2d: bool, col_2d: bool):
    """Tile config where this kernel beats the HIP dual cast on gfx1250, else None.

    From the job2 sweep (bit-exact vs HIP on every shape): the Triton kernel wins above
    4096x4096 elements (1.2-2.5x; e.g. 4096x32768 158 -> 109 us, 129280x4096 2D 535 -> 211 us)
    and loses on the small DSv4 tensors (4096x1024: 19.7 vs 25 us), which stay on HIP.
    """
    two_d = row_2d or col_2d
    if two_d:
        if M * N < 4096 * 4096:
            return None
        return (128, 128, 8) if M * N >= 2**28 else (128, 64, 8)
    if M * N <= 4096 * 4096:
        return None
    return (32, 128, 4)
