###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Standalone Triton implementation of the MXFP8 dequantize ops.

This module bundles both the ``@triton.jit`` dequantize kernels and their Python
launchers. They mirror the public ``torch.ops.primus_turbo_cpp_extension`` MXFP8
dequant ops (``dequantize_mxfp8`` / ``grouped_dequantize_mxfp8``) in tensor
shapes, layouts and E8M0 numerics.

The kernels reproduce the CUDA/HIP kernels in
``csrc/kernels/quantization/dequantization_mxfp8.cu``:
``y = float(x_fp8) * e8m0_to_scale(e8m0)`` with ``e8m0_to_scale(e) = uint_as_float(e << 23)``,
and an out-of-range scale index defaulting to ``e8m0 = 127`` (scale == 1.0).
Colwise output is transposed; the grouped variant gathers compact rows from the
per-group padded layout.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.low_precision import MXFP8_BLOCK_SIZE
from primus_turbo.pytorch.core.utils import is_gfx1250

# FP8 dtypes that use the E5M2 layout (everything else handled here is E4M3FN).
_E5M2_DTYPES = {torch.float8_e5m2, torch.float8_e5m2fnuz}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@triton.jit
def _fp8_to_f32(b, E5M2: tl.constexpr):
    """Decode one FP8 code (held in a uint32 lane) to fp32 with pure integer/fp32 ops.

    NOTE (gfx1250): Triton's native FP8->float path (loading a ``float8_*`` tensor, or
    ``uint8.to(fp8, bitcast=True).to(float32)``) miscompiles on gfx1250 (wave32) and drops
    ~half the elements of the tile (roundtrip rel_err ~= 1/sqrt(2)). Loading the raw bytes as
    ``uint8`` and reconstructing the value with the bit-field math below avoids that broken
    conversion entirely and is bit-exact vs. PyTorch for every finite code (plus Inf/NaN).
    """
    s = (b >> 7) & 0x1
    sign = tl.where(s == 1, -1.0, 1.0)
    if E5M2:
        e = (b >> 2) & 0x1F
        m = b & 0x3
        ef = e.to(tl.float32)
        mf = m.to(tl.float32) * 0.25  # mantissa / 2^2
        norm = tl.exp2(ef - 15.0) * (1.0 + mf)
        sub = tl.exp2(tl.full(b.shape, 1.0 - 15.0, tl.float32)) * mf
        val = tl.where(e == 0, sub, norm)
        is_inf = (e == 0x1F) & (m == 0)
        is_nan = (e == 0x1F) & (m != 0)
        val = tl.where(is_inf, float("inf"), val)
        val = tl.where(is_nan, float("nan"), val)
    else:
        # E4M3FN (OCP): no Inf; only 0x7F / 0xFF (exp==0xF, mant==0x7) are NaN.
        e = (b >> 3) & 0xF
        m = b & 0x7
        ef = e.to(tl.float32)
        mf = m.to(tl.float32) * 0.125  # mantissa / 2^3
        norm = tl.exp2(ef - 7.0) * (1.0 + mf)
        sub = tl.exp2(tl.full(b.shape, 1.0 - 7.0, tl.float32)) * mf
        val = tl.where(e == 0, sub, norm)
        is_nan = (e == 0xF) & (m == 0x7)
        val = tl.where(is_nan, float("nan"), val)
    return sign * val


# ===========================================================================
# Triton kernels
# ===========================================================================
@triton.jit
def dequantize_mxfp8_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    num_rows,
    row_length,
    scale_m,
    scale_n,
    scale_stride,
    BLOCK_SIZE: tl.constexpr,
    ROWWISE: tl.constexpr,
    E5M2: tl.constexpr,
    BITDECODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)

    data_mask = (rows[:, None] < num_rows) & (cols[None, :] < row_length)
    x_off = rows64[:, None] * row_length + cols64[None, :]
    if BITDECODE:
        # gfx1250: x_ptr is the FP8 tensor viewed as uint8; decode manually
        # (native Triton FP8->float miscompiles on gfx1250).
        xb = tl.load(x_ptr + x_off, mask=data_mask, other=0).to(tl.uint32)
        x = _fp8_to_f32(xb, E5M2)
    else:
        # other arches: x_ptr is the FP8 tensor; use Triton's native conversion.
        x = tl.load(x_ptr + x_off, mask=data_mask, other=0.0).to(tl.float32)

    col_block = cols // BLOCK_SIZE
    scale_mask = (rows[:, None] < scale_m) & (col_block[None, :] < scale_n)
    e8m0 = tl.load(
        scale_ptr + rows64[:, None] * scale_stride + col_block[None, :].to(tl.int64),
        mask=scale_mask,
        other=0,
    ).to(tl.uint32)
    e8m0 = tl.where(scale_mask, e8m0, 127)
    scale = (e8m0 << 23).to(tl.float32, bitcast=True)

    y = x * scale

    if ROWWISE:
        tl.store(y_ptr + rows64[:, None] * row_length + cols64[None, :], y, mask=data_mask)
    else:
        # Output is transposed: y_out[c, r] = x[r, c] * scale.
        yT = tl.trans(y)
        out_mask = (cols[:, None] < row_length) & (rows[None, :] < num_rows)
        tl.store(y_ptr + cols64[:, None] * num_rows + rows64[None, :], yT, mask=out_mask)


@triton.jit
def grouped_dequantize_mxfp8_rowwise_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    group_offs_ptr,
    group_offs_padded_ptr,
    G,
    total_M,
    n_cols,
    scale_m,
    scale_n,
    scale_stride,
    BLOCK_SIZE: tl.constexpr,
    E5M2: tl.constexpr,
    BITDECODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)  # compact rows
    cols = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)

    # compact -> padded row
    padded = rows
    for i in range(G):
        s = tl.load(group_offs_ptr + i)
        e = tl.load(group_offs_ptr + i + 1)
        ps = tl.load(group_offs_padded_ptr + i)
        in_g = (rows >= s) & (rows < e)
        padded = tl.where(in_g, ps + (rows - s), padded)

    data_mask = (rows[:, None] < total_M) & (cols[None, :] < n_cols)
    x_off = padded[:, None] * n_cols + cols[None, :]
    if BITDECODE:
        # gfx1250: decode FP8 bytes manually (native Triton FP8->float miscompiles).
        xb = tl.load(x_ptr + x_off, mask=data_mask, other=0).to(tl.uint32)
        x = _fp8_to_f32(xb, E5M2)
    else:
        x = tl.load(x_ptr + x_off, mask=data_mask, other=0.0).to(tl.float32)

    col_block = cols // BLOCK_SIZE
    scale_mask = (padded[:, None] < scale_m) & (col_block[None, :] < scale_n)
    e8m0 = tl.load(
        scale_ptr + padded[:, None] * scale_stride + col_block[None, :],
        mask=scale_mask,
        other=0,
    ).to(tl.uint32)
    e8m0 = tl.where(scale_mask, e8m0, 127)
    scale = (e8m0 << 23).to(tl.float32, bitcast=True)

    y = x * scale
    tl.store(y_ptr + rows[:, None] * n_cols + cols[None, :], y, mask=data_mask)


@triton.jit
def grouped_dequantize_mxfp8_colwise_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    group_offs_ptr,
    group_offs_padded_ptr,
    G,
    total_M,
    num_rows,
    m_padded,
    scale_m,
    scale_n,
    scale_stride,
    BLOCK_SIZE: tl.constexpr,
    E5M2: tl.constexpr,
    BITDECODE: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
):
    # Input x is [num_rows, m_padded]; output y is compact [total_M, num_rows].
    pid_n = tl.program_id(0)  # input rows (== output cols)
    pid_m = tl.program_id(1)  # input cols (padded M)
    ns = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
    ms_pad = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)

    # padded M -> compact M (-1 if padding)
    compact = tl.zeros([BM], tl.int64) - 1
    for i in range(G):
        ps = tl.load(group_offs_padded_ptr + i)
        pe = tl.load(group_offs_padded_ptr + i + 1)
        s = tl.load(group_offs_ptr + i)
        e = tl.load(group_offs_ptr + i + 1)
        len_g = e - s
        local = ms_pad - ps
        is_real = (ms_pad >= ps) & (ms_pad < pe) & (local < len_g)
        compact = tl.where(is_real, s + local, compact)

    data_mask = (ns[:, None] < num_rows) & (ms_pad[None, :] < m_padded)
    x_off = ns[:, None] * m_padded + ms_pad[None, :]
    if BITDECODE:
        # gfx1250: decode FP8 bytes manually (native Triton FP8->float miscompiles).
        xb = tl.load(x_ptr + x_off, mask=data_mask, other=0).to(tl.uint32)
        x = _fp8_to_f32(xb, E5M2)
    else:
        x = tl.load(x_ptr + x_off, mask=data_mask, other=0.0).to(tl.float32)

    col_block = ms_pad // BLOCK_SIZE
    scale_mask = (ns[:, None] < scale_m) & (col_block[None, :] < scale_n)
    e8m0 = tl.load(
        scale_ptr + ns[:, None] * scale_stride + col_block[None, :],
        mask=scale_mask,
        other=0,
    ).to(tl.uint32)
    e8m0 = tl.where(scale_mask, e8m0, 127)
    scale = (e8m0 << 23).to(tl.float32, bitcast=True)

    y = x * scale  # [BN, BM]

    # Scatter transposed: y_out[compact_m, n] = y[n, m].
    out_mask = (ns[:, None] < num_rows) & (compact[None, :] >= 0)
    tl.store(y_ptr + compact[None, :] * num_rows + ns[:, None], y, mask=out_mask)


# ===========================================================================
# Python launchers
# ===========================================================================
def dequantize_mxfp8_triton(
    input: torch.Tensor,
    scale_inv: torch.Tensor,
    axis: int,
    block_size: int,
    dest_dtype: torch.dtype,
) -> torch.Tensor:
    assert input.is_cuda and input.is_contiguous()
    assert input.dim() in (2, 3)
    assert scale_inv.dim() == input.dim()
    assert block_size == MXFP8_BLOCK_SIZE
    assert dest_dtype in (torch.bfloat16, torch.float16, torch.float32)

    is_batched = input.dim() == 3
    if input.dim() == 2:
        assert axis in (0, 1)
        use_rowwise = axis == 1
        num_rows, row_length = input.size(0), input.size(1)
    else:
        assert axis in (1, 2)
        use_rowwise = axis == 2
        num_rows = input.size(0) * input.size(1)
        row_length = input.size(2)
    assert row_length % block_size == 0

    input_2d = input.reshape(num_rows, row_length)
    # gfx1250: pass the FP8 codes as raw uint8 and bit-decode in-kernel (native Triton
    # FP8->float miscompiles there). Other arches: pass the FP8 tensor (native convert).
    gfx1250 = is_gfx1250()
    x_arg = input_2d.view(torch.uint8) if gfx1250 else input_2d
    if is_batched:
        scale_2d = scale_inv.reshape(scale_inv.size(0) * scale_inv.size(1), scale_inv.size(2))
    else:
        scale_2d = scale_inv
    scale_u8 = scale_2d.contiguous().view(torch.uint8)
    scale_m, scale_n = scale_u8.size(0), scale_u8.size(1)

    device = input.device
    if use_rowwise:
        out_2d = torch.empty((num_rows, row_length), device=device, dtype=dest_dtype)
    else:
        out_2d = torch.empty((row_length, num_rows), device=device, dtype=dest_dtype)

    BM, BN = 32, 128
    grid = (_cdiv(num_rows, BM), _cdiv(row_length, BN))
    dequantize_mxfp8_kernel[grid](
        x_arg,
        out_2d,
        scale_u8,
        num_rows,
        row_length,
        scale_m,
        scale_n,
        scale_n,
        BLOCK_SIZE=block_size,
        ROWWISE=use_rowwise,
        E5M2=input.dtype in _E5M2_DTYPES,
        BITDECODE=gfx1250,
        BM=BM,
        BN=BN,
    )

    if is_batched:
        if use_rowwise:
            return out_2d.reshape(input.size(0), input.size(1), row_length)
        return out_2d.reshape(row_length, input.size(0), input.size(1)).permute(1, 2, 0)
    return out_2d


def grouped_dequantize_mxfp8_triton(
    input: torch.Tensor,
    scale_inv: torch.Tensor,
    group_offs: torch.Tensor,
    group_offs_padded: torch.Tensor,
    axis: int,
    block_size: int,
    dest_dtype: torch.dtype,
    total_M: Optional[int] = None,
) -> torch.Tensor:
    assert input.is_cuda and input.is_contiguous() and input.dim() == 2
    assert scale_inv.dim() == 2
    assert block_size == MXFP8_BLOCK_SIZE
    assert dest_dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert axis in (0, 1)

    use_rowwise = axis == 1
    G = group_offs.size(0) - 1
    if total_M is None:
        total_M = int(group_offs[G].item())

    num_rows, row_length = input.size(0), input.size(1)
    assert row_length % block_size == 0

    m_padded = num_rows if use_rowwise else row_length
    n_cols = row_length if use_rowwise else num_rows

    scale_u8 = scale_inv.contiguous().view(torch.uint8)
    scale_m, scale_n = scale_u8.size(0), scale_u8.size(1)
    # gfx1250: pass FP8 codes as raw uint8 and bit-decode in-kernel (native Triton
    # FP8->float miscompiles there). Other arches: pass the FP8 tensor (native convert).
    gfx1250 = is_gfx1250()
    x_arg = input.view(torch.uint8) if gfx1250 else input
    is_e5m2 = input.dtype in _E5M2_DTYPES

    device = input.device
    output = torch.zeros((total_M, n_cols), device=device, dtype=dest_dtype)

    if use_rowwise:
        BM, BN = 32, 128
        grid = (_cdiv(total_M, BM), _cdiv(n_cols, BN))
        grouped_dequantize_mxfp8_rowwise_kernel[grid](
            x_arg,
            output,
            scale_u8,
            group_offs,
            group_offs_padded,
            G,
            total_M,
            n_cols,
            scale_m,
            scale_n,
            scale_n,
            BLOCK_SIZE=block_size,
            E5M2=is_e5m2,
            BITDECODE=gfx1250,
            BM=BM,
            BN=BN,
        )
    else:
        BN, BM = 32, 128
        grid = (_cdiv(num_rows, BN), _cdiv(row_length, BM))
        grouped_dequantize_mxfp8_colwise_kernel[grid](
            x_arg,
            output,
            scale_u8,
            group_offs,
            group_offs_padded,
            G,
            total_M,
            num_rows,
            m_padded,
            scale_m,
            scale_n,
            scale_n,
            BLOCK_SIZE=block_size,
            E5M2=is_e5m2,
            BITDECODE=gfx1250,
            BN=BN,
            BM=BM,
        )
    return output
