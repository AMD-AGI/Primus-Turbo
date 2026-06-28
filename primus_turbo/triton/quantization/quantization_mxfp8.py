###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Standalone Triton implementation of the MXFP8 quantize ops.

The dequantize ops live in ``dequantization_mxfp8.py``.
"""

from typing import List

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    MXFP8_PADDING_ALIGN_SIZE,
)

MX_BLOCK = tl.constexpr(32)

# Per-group M alignment for the grouped layout (matches the C++ constants
# MXFP8_GROUP_M_PADDING_ALIGN_SIZE / MXFP8_K_DIM_PADDING_ALIGN_SIZE).
_GROUP_ROW_ALIGN = 32
_GROUP_COL_ALIGN = 128

E8M0_DTYPE = torch.float8_e8m0fnu

_E5M2_DTYPES = {torch.float8_e5m2, torch.float8_e5m2fnuz}


# ===========================================================================
# Triton kernels
# ===========================================================================
@triton.jit
def _compute_e8m0_scale(
    amax,
    MBITS: tl.constexpr,
    TARGET_MAX_POW2: tl.constexpr,
    ZERO_AMAX_TO_ONE: tl.constexpr,
):
    """Replicate ``compute_tile_scale``: amax (fp32) -> (e8m0 uint32, native fp32 scale)."""
    amax_bits = amax.to(tl.uint32, bitcast=True)
    VAL_TO_ADD: tl.constexpr = 1 << (23 - MBITS - 1)
    EXP_MASK: tl.constexpr = (1 << 9) - 1
    shifted = ((amax_bits + VAL_TO_ADD) >> 23) & EXP_MASK
    pow2 = shifted.to(tl.int32) - 127 - TARGET_MAX_POW2
    e_unbiased = tl.maximum(pow2, -127)
    e_unbiased = tl.minimum(e_unbiased, 128)
    e_biased = e_unbiased + 127  # int32 in [0, 255]
    if ZERO_AMAX_TO_ONE:
        e_biased = tl.where(amax == 0.0, 127, e_biased)
    e_biased_u = e_biased.to(tl.uint32)
    scale_native = (e_biased_u << 23).to(tl.float32, bitcast=True)
    return e_biased_u, scale_native


@triton.jit
def _quantize_vals(x, scale_native, FP8_MAX: tl.constexpr):
    """clamp(x / scale, -FP8_MAX, FP8_MAX); a zero native scale (all-zero block) -> 0."""
    q = tl.where(scale_native == 0.0, 0.0, x / scale_native)
    return tl.clamp(q, -FP8_MAX, FP8_MAX)


@triton.jit
def _f32_to_fp8(x, MBITS: tl.constexpr):
    """Encode fp32 -> FP8 code (uint8 held in a uint32 lane) with pure integer/fp32 ops
    -- the inverse of ``_fp8_to_f32``.

    NOTE (gfx1250): Triton's native f32->FP8 cast (``q.to(float8_*)``) miscompiles on
    gfx1250 (wave32) and drops lanes of the tile, the store-side twin of the
    load-side bug worked around in ``_fp8_to_f32``. Building
    the FP8 byte by hand and bitcast-storing it avoids the broken conversion entirely.
    Normals use round-to-nearest-even on the fraction; subnormals round-to-nearest.
    ``x`` is assumed already clamped to the format's finite range (see ``_quantize_vals``),
    so there is no overflow-to-Inf to handle. Supports E4M3FN (MBITS=3) and E5M2 (MBITS=2).
    """
    BIAS: tl.constexpr = 15 if MBITS == 2 else 7
    EBITS: tl.constexpr = 5 if MBITS == 2 else 4
    # finite max exponent field: E5M2 reserves all-ones for Inf/NaN; E4M3FN has no Inf.
    EFIELD_MAX: tl.constexpr = ((1 << EBITS) - 2) if MBITS == 2 else ((1 << EBITS) - 1)
    MANT_MAX: tl.constexpr = (1 << MBITS) - 1
    NSHIFT: tl.constexpr = 23 - MBITS
    RBIAS: tl.constexpr = (1 << (NSHIFT - 1)) - 1            # round-to-nearest-even bias
    SUB_INV: tl.constexpr = float(1 << (MBITS - 1 + BIAS))   # 1 / subnormal step
    NAN_CODE: tl.constexpr = ((((1 << EBITS) - 1) << MBITS) | 0x1) if MBITS == 2 else 0x7F

    a = tl.abs(x)
    xb = x.to(tl.uint32, bitcast=True)
    ab = a.to(tl.uint32, bitcast=True)
    sign = (xb >> 31) & 0x1
    f32m = ab & 0x7FFFFF
    fe = ((ab >> 23) & 0xFF).to(tl.int32) - 127 + BIAS      # target biased FP8 exponent

    # Normal (fe >= 1): round the 23-bit fraction down to MBITS bits, ties-to-even.
    mr = (f32m + (RBIAS + ((f32m >> NSHIFT) & 1))) >> NSHIFT
    carry = mr == (1 << MBITS)
    fe_n = tl.where(carry, fe + 1, fe)
    mr = tl.where(carry, 0, mr)
    over = fe_n > EFIELD_MAX
    fe_n = tl.where(over, EFIELD_MAX, fe_n)
    mr = tl.where(over, MANT_MAX, mr)
    code_n = (fe_n.to(tl.uint32) << MBITS) | mr.to(tl.uint32)

    # Subnormal (fe <= 0): m = round(|x| / subnormal_step). m == 2^MBITS rolls up to the
    # smallest normal (efield=1, m=0); int() of a non-negative float truncates == floor.
    ms = tl.minimum((a * SUB_INV + 0.5).to(tl.int32), 1 << MBITS)
    code_s = ms.to(tl.uint32)

    code = tl.where(fe >= 1, code_n, code_s)
    code = tl.where(a == 0.0, code * 0, code)
    code = tl.where(a != a, code * 0 + NAN_CODE, code)
    return (code | (sign << 7)).to(tl.uint8)


# ---------------------------------------------------------------------------
# Dense quantize (single direction)
# ---------------------------------------------------------------------------
@triton.jit
def quantize_mxfp8_rowwise_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    x_g_stride,
    out_g_stride,
    scale_g_stride,
    M,
    N,
    N_pad,
    scale_N,
    USE_2D_BLOCK: tl.constexpr,
    MBITS: tl.constexpr,
    TARGET_MAX_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    rows = pid_m * MX_BLOCK + tl.arange(0, MX_BLOCK)
    cols = pid_n * MX_BLOCK + tl.arange(0, MX_BLOCK)
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)

    load_mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(
        x_ptr + g * x_g_stride + rows64[:, None] * N + cols64[None, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)

    amax_row = tl.max(tl.abs(x), axis=1)  # [MX_BLOCK]
    if USE_2D_BLOCK:
        tile_amax = tl.max(amax_row)
        amax_vec = tl.zeros([MX_BLOCK], tl.float32) + tile_amax
    else:
        amax_vec = amax_row

    e8m0, scale_native = _compute_e8m0_scale(amax_vec, MBITS, TARGET_MAX_POW2, False)
    q = _quantize_vals(x, scale_native[:, None], FP8_MAX)

    store_mask = (rows[:, None] < M) & (cols[None, :] < N_pad)
    tl.store(
        out_ptr + g * out_g_stride + rows64[:, None] * N_pad + cols64[None, :],
        _f32_to_fp8(q, MBITS).to(out_ptr.dtype.element_ty, bitcast=True),
        mask=store_mask,
    )
    tl.store(
        scale_ptr + g * scale_g_stride + rows64 * scale_N + pid_n,
        e8m0.to(scale_ptr.dtype.element_ty),
        mask=rows < M,
    )


@triton.jit
def quantize_mxfp8_colwise_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    x_g_stride,
    out_g_stride,
    scale_g_stride,
    M,
    N,
    M_pad,
    scale_N,
    USE_2D_BLOCK: tl.constexpr,
    MBITS: tl.constexpr,
    TARGET_MAX_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)  # M-direction (rows of input)
    pid_n = tl.program_id(1)  # N-direction (cols of input)
    g = tl.program_id(2)

    ms = pid_m * MX_BLOCK + tl.arange(0, MX_BLOCK)
    ns = pid_n * MX_BLOCK + tl.arange(0, MX_BLOCK)
    ms64 = ms.to(tl.int64)
    ns64 = ns.to(tl.int64)

    load_mask = (ms[:, None] < M) & (ns[None, :] < N)
    x = tl.load(
        x_ptr + g * x_g_stride + ms64[:, None] * N + ns64[None, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)

    amax_col = tl.max(tl.abs(x), axis=0)  # [MX_BLOCK] per N column
    if USE_2D_BLOCK:
        tile_amax = tl.max(amax_col)
        amax_vec = tl.zeros([MX_BLOCK], tl.float32) + tile_amax
    else:
        amax_vec = amax_col

    e8m0, scale_native = _compute_e8m0_scale(amax_vec, MBITS, TARGET_MAX_POW2, False)
    q = _quantize_vals(x, scale_native[None, :], FP8_MAX)

    # Transposed store: out[N, M_pad]; out[n, m] = q[m, n].
    qT = tl.trans(q)
    store_mask = (ns[:, None] < N) & (ms[None, :] < M_pad)
    tl.store(
        out_ptr + g * out_g_stride + ns64[:, None] * M_pad + ms64[None, :],
        _f32_to_fp8(qT, MBITS).to(out_ptr.dtype.element_ty, bitcast=True),
        mask=store_mask,
    )
    tl.store(
        scale_ptr + g * scale_g_stride + ns64 * scale_N + pid_m,
        e8m0.to(scale_ptr.dtype.element_ty),
        mask=ns < N,
    )


# ---------------------------------------------------------------------------
# Grouped quantize (single direction)
# ---------------------------------------------------------------------------
@triton.jit
def grouped_quantize_mxfp8_rowwise_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    group_offs_ptr,
    group_offs_padded_ptr,
    G,
    N,
    N_pad,
    scale_N,
    USE_2D_BLOCK: tl.constexpr,
    MBITS: tl.constexpr,
    TARGET_MAX_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = (pid_m * MX_BLOCK).to(tl.int64)

    # Locate the group this 32-row output tile belongs to (group_offs_padded is
    # 32-aligned for rowwise, so the tile lies entirely within one group or beyond
    # the last group -> all-padding).
    co = tl.zeros([], tl.int64)
    cop = tl.zeros([], tl.int64)
    len_g = tl.zeros([], tl.int64)
    found = tl.zeros([], tl.int32)
    for i in range(G):
        ps = tl.load(group_offs_padded_ptr + i)
        pe = tl.load(group_offs_padded_ptr + i + 1)
        in_g = (row0 >= ps) & (row0 < pe)
        gs = tl.load(group_offs_ptr + i)
        ge = tl.load(group_offs_ptr + i + 1)
        co = tl.where(in_g, gs, co)
        cop = tl.where(in_g, ps, cop)
        len_g = tl.where(in_g, ge - gs, len_g)
        found = tl.where(in_g, 1, found)

    rows_pad = row0 + tl.arange(0, MX_BLOCK).to(tl.int64)
    local = rows_pad - cop
    valid = (found != 0) & (local >= 0) & (local < len_g)
    in_rows = co + local

    cols = pid_n * MX_BLOCK + tl.arange(0, MX_BLOCK)
    cols64 = cols.to(tl.int64)

    load_mask = valid[:, None] & (cols[None, :] < N)
    x = tl.load(
        x_ptr + in_rows[:, None] * N + cols64[None, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)

    amax_row = tl.max(tl.abs(x), axis=1)
    if USE_2D_BLOCK:
        tile_amax = tl.max(amax_row)
        amax_vec = tl.zeros([MX_BLOCK], tl.float32) + tile_amax
    else:
        amax_vec = amax_row

    e8m0, scale_native = _compute_e8m0_scale(amax_vec, MBITS, TARGET_MAX_POW2, True)
    q = _quantize_vals(x, scale_native[:, None], FP8_MAX)

    # Only real rows are written (matches the CUDA grouped rowwise kernel).
    store_mask = valid[:, None] & (cols[None, :] < N_pad)
    tl.store(
        out_ptr + rows_pad[:, None] * N_pad + cols64[None, :],
        _f32_to_fp8(q, MBITS).to(out_ptr.dtype.element_ty, bitcast=True),
        mask=store_mask,
    )
    tl.store(
        scale_ptr + rows_pad * scale_N + pid_n,
        e8m0.to(scale_ptr.dtype.element_ty),
        mask=valid,
    )


@triton.jit
def grouped_quantize_mxfp8_colwise_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    group_offs_ptr,
    group_offs_padded_ptr,
    G,
    N,
    M_pad,
    scale_N,
    USE_2D_BLOCK: tl.constexpr,
    MBITS: tl.constexpr,
    TARGET_MAX_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)  # padded M-direction
    pid_n = tl.program_id(1)  # N-direction
    m0 = (pid_m * MX_BLOCK).to(tl.int64)

    co = tl.zeros([], tl.int64)
    cop = tl.zeros([], tl.int64)
    len_g = tl.zeros([], tl.int64)
    found = tl.zeros([], tl.int32)
    for i in range(G):
        ps = tl.load(group_offs_padded_ptr + i)
        pe = tl.load(group_offs_padded_ptr + i + 1)
        in_g = (m0 >= ps) & (m0 < pe)
        gs = tl.load(group_offs_ptr + i)
        ge = tl.load(group_offs_ptr + i + 1)
        co = tl.where(in_g, gs, co)
        cop = tl.where(in_g, ps, cop)
        len_g = tl.where(in_g, ge - gs, len_g)
        found = tl.where(in_g, 1, found)

    ms_pad = m0 + tl.arange(0, MX_BLOCK).to(tl.int64)
    local = ms_pad - cop
    valid_m = (found != 0) & (local >= 0) & (local < len_g)
    in_rows = co + local

    ns = pid_n * MX_BLOCK + tl.arange(0, MX_BLOCK)
    ns64 = ns.to(tl.int64)

    load_mask = valid_m[:, None] & (ns[None, :] < N)
    x = tl.load(
        x_ptr + in_rows[:, None] * N + ns64[None, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)

    amax_col = tl.max(tl.abs(x), axis=0)
    if USE_2D_BLOCK:
        tile_amax = tl.max(amax_col)
        amax_vec = tl.zeros([MX_BLOCK], tl.float32) + tile_amax
    else:
        amax_vec = amax_col

    e8m0, scale_native = _compute_e8m0_scale(amax_vec, MBITS, TARGET_MAX_POW2, True)
    q = _quantize_vals(x, scale_native[None, :], FP8_MAX)

    # Colwise grouped writes the entire padded tile (padding -> 0, scale 1.0).
    qT = tl.trans(q)
    store_mask = (ns[:, None] < N) & (ms_pad[None, :] < M_pad)
    tl.store(
        out_ptr + ns64[:, None] * M_pad + ms_pad[None, :],
        _f32_to_fp8(qT, MBITS).to(out_ptr.dtype.element_ty, bitcast=True),
        mask=store_mask,
    )
    tl.store(
        scale_ptr + ns64 * scale_N + pid_m,
        e8m0.to(scale_ptr.dtype.element_ty),
        mask=ns < N,
    )


# ===========================================================================
# Python launchers
# ===========================================================================
def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _cdiv_tensor(t: torch.Tensor, b: int) -> torch.Tensor:
    return (t + b - 1) // b


def _fp8_params(dest_dtype: torch.dtype):
    """Return (MBITS, TARGET_MAX_POW2, FP8_MAX) for the OCP MXFP8 output dtype."""
    if dest_dtype in _E5M2_DTYPES:
        return 2, 15, 57344.0
    # e4m3 (OCP). MXFP8 is only supported on gfx950+, so no FNUZ target branch.
    return 3, 8, 448.0


def _check_no_shuffle(**flags) -> None:
    for name, val in flags.items():
        if val:
            raise NotImplementedError(
                f"Triton MXFP8 implementation does not support {name}; shuffle layout is unsupported."
            )


def _compute_padded_layout(group_lens: torch.Tensor, align: int):
    group_lens_padded = _cdiv_tensor(group_lens, align) * align
    group_offs_padded = torch.zeros(group_lens.size(0) + 1, device=group_lens.device, dtype=group_lens.dtype)
    group_offs_padded[1:] = torch.cumsum(group_lens_padded, dim=0)
    return group_lens_padded, group_offs_padded


# ---------------------------------------------------------------------------
# Dense quantize
# ---------------------------------------------------------------------------
def quantize_mxfp8_triton(
    input: torch.Tensor,
    dest_dtype: torch.dtype,
    axis: int,
    use_2d_block: bool = False,
    shuffle_scale: bool = False,
    shuffle_out: bool = False,
) -> List[torch.Tensor]:
    _check_no_shuffle(shuffle_scale=shuffle_scale, shuffle_out=shuffle_out)
    assert input.is_cuda and input.is_contiguous()
    assert input.dtype in (torch.bfloat16, torch.float16)

    if input.dim() == 2:
        assert axis in (0, 1), "Axis must be 0 or 1 for 2D input"
        is_rowwise = axis == 1
        G, M, N = 1, input.size(0), input.size(1)
    elif input.dim() == 3:
        assert axis in (1, 2), "Axis must be 1 or 2 for 3D input"
        is_rowwise = axis == 2
        G, M, N = input.size(0), input.size(1), input.size(2)
    else:
        raise ValueError("Input must be 2D or 3D")

    assert N % MXFP8_BLOCK_SIZE == 0, f"N must be divisible by {MXFP8_BLOCK_SIZE}"

    align = MXFP8_PADDING_ALIGN_SIZE
    M_pad = _cdiv(M, align) * align
    N_pad = _cdiv(N, align) * align
    mbits, target, fp8_max = _fp8_params(dest_dtype)

    out_fp8, scale = _quantize_single(
        input.reshape(G, M, N),
        dest_dtype,
        is_rowwise,
        G,
        M,
        N,
        M_pad,
        N_pad,
        use_2d_block,
        mbits,
        target,
        fp8_max,
    )

    if input.dim() == 2:
        out_fp8 = out_fp8.squeeze(0)
        scale = scale.squeeze(0)
    return [out_fp8, scale.view(E8M0_DTYPE)]


def _quantize_single(
    x: torch.Tensor,  # [G, M, N]
    dest_dtype: torch.dtype,
    is_rowwise: bool,
    G: int,
    M: int,
    N: int,
    M_pad: int,
    N_pad: int,
    use_2d_block: bool,
    mbits: int,
    target: int,
    fp8_max: float,
):
    device = x.device
    if is_rowwise:
        out_rows, out_cols = M, N_pad
        scale_N = _cdiv(N_pad, MXFP8_BLOCK_SIZE)
        scale_rows = M
    else:
        out_rows, out_cols = N, M_pad
        scale_N = _cdiv(M_pad, MXFP8_BLOCK_SIZE)
        scale_rows = N

    out_fp8 = torch.zeros((G, out_rows, out_cols), device=device, dtype=dest_dtype)
    scale = torch.zeros((G, scale_rows, scale_N), device=device, dtype=torch.uint8)

    x_g_stride = M * N
    out_g_stride = out_rows * out_cols
    scale_g_stride = scale_rows * scale_N

    if is_rowwise:
        grid = (_cdiv(M, MXFP8_BLOCK_SIZE), N_pad // MXFP8_BLOCK_SIZE, G)
        quantize_mxfp8_rowwise_kernel[grid](
            x,
            out_fp8,
            scale,
            x_g_stride,
            out_g_stride,
            scale_g_stride,
            M,
            N,
            N_pad,
            scale_N,
            USE_2D_BLOCK=use_2d_block,
            MBITS=mbits,
            TARGET_MAX_POW2=target,
            FP8_MAX=fp8_max,
        )
    else:
        grid = (M_pad // MXFP8_BLOCK_SIZE, N // MXFP8_BLOCK_SIZE, G)
        quantize_mxfp8_colwise_kernel[grid](
            x,
            out_fp8,
            scale,
            x_g_stride,
            out_g_stride,
            scale_g_stride,
            M,
            N,
            M_pad,
            scale_N,
            USE_2D_BLOCK=use_2d_block,
            MBITS=mbits,
            TARGET_MAX_POW2=target,
            FP8_MAX=fp8_max,
        )
    return out_fp8, scale


# ---------------------------------------------------------------------------
# Dense dual quantize (rowwise + colwise)
# ---------------------------------------------------------------------------
def quantize_mxfp8_dual_triton(
    input: torch.Tensor,
    dest_dtype: torch.dtype,
    rowwise_use_2d_block: bool = False,
    colwise_use_2d_block: bool = False,
    shuffle_rowwise_scale: bool = False,
    shuffle_rowwise: bool = False,
    shuffle_colwise_scale: bool = False,
    shuffle_colwise: bool = False,
) -> List[torch.Tensor]:
    _check_no_shuffle(
        shuffle_rowwise_scale=shuffle_rowwise_scale,
        shuffle_rowwise=shuffle_rowwise,
        shuffle_colwise_scale=shuffle_colwise_scale,
        shuffle_colwise=shuffle_colwise,
    )
    assert input.is_cuda and input.is_contiguous()
    assert input.dtype in (torch.bfloat16, torch.float16)

    if input.dim() == 2:
        G, M, N = 1, input.size(0), input.size(1)
    elif input.dim() == 3:
        G, M, N = input.size(0), input.size(1), input.size(2)
    else:
        raise ValueError("Input must be 2D or 3D")
    assert N % MXFP8_BLOCK_SIZE == 0, f"N must be divisible by {MXFP8_BLOCK_SIZE}"

    align = MXFP8_PADDING_ALIGN_SIZE
    M_pad = _cdiv(M, align) * align
    N_pad = _cdiv(N, align) * align
    mbits, target, fp8_max = _fp8_params(dest_dtype)
    x3d = input.reshape(G, M, N)

    rw_fp8, rw_scale = _quantize_single(
        x3d,
        dest_dtype,
        True,
        G,
        M,
        N,
        M_pad,
        N_pad,
        rowwise_use_2d_block,
        mbits,
        target,
        fp8_max,
    )
    cw_fp8, cw_scale = _quantize_single(
        x3d,
        dest_dtype,
        False,
        G,
        M,
        N,
        M_pad,
        N_pad,
        colwise_use_2d_block,
        mbits,
        target,
        fp8_max,
    )

    if input.dim() == 2:
        rw_fp8, rw_scale = rw_fp8.squeeze(0), rw_scale.squeeze(0)
        cw_fp8, cw_scale = cw_fp8.squeeze(0), cw_scale.squeeze(0)
    return [
        rw_fp8,
        rw_scale.view(E8M0_DTYPE),
        cw_fp8,
        cw_scale.view(E8M0_DTYPE),
    ]


# ---------------------------------------------------------------------------
# Grouped quantize
# ---------------------------------------------------------------------------
def _grouped_quantize_single(
    input: torch.Tensor,  # [total_M, N]
    group_offs: torch.Tensor,
    group_offs_padded: torch.Tensor,
    dest_dtype: torch.dtype,
    is_rowwise: bool,
    G: int,
    total_M: int,
    N: int,
    N_pad: int,
    M_pad_out: int,
    use_2d_block: bool,
    mbits: int,
    target: int,
    fp8_max: float,
):
    device = input.device
    if is_rowwise:
        out_rows, out_cols = M_pad_out, N_pad
        scale_N = _cdiv(N_pad, MXFP8_BLOCK_SIZE)
        scale_rows = M_pad_out
    else:
        out_rows, out_cols = N, M_pad_out
        scale_N = _cdiv(M_pad_out, MXFP8_BLOCK_SIZE)
        scale_rows = N

    out_fp8 = torch.zeros((out_rows, out_cols), device=device, dtype=dest_dtype)
    scale = torch.zeros((scale_rows, scale_N), device=device, dtype=torch.uint8)

    if is_rowwise:
        grid = (M_pad_out // MXFP8_BLOCK_SIZE, N_pad // MXFP8_BLOCK_SIZE)
        grouped_quantize_mxfp8_rowwise_kernel[grid](
            input,
            out_fp8,
            scale,
            group_offs,
            group_offs_padded,
            G,
            N,
            N_pad,
            scale_N,
            USE_2D_BLOCK=use_2d_block,
            MBITS=mbits,
            TARGET_MAX_POW2=target,
            FP8_MAX=fp8_max,
        )
    else:
        grid = (M_pad_out // MXFP8_BLOCK_SIZE, N // MXFP8_BLOCK_SIZE)
        grouped_quantize_mxfp8_colwise_kernel[grid](
            input,
            out_fp8,
            scale,
            group_offs,
            group_offs_padded,
            G,
            N,
            M_pad_out,
            scale_N,
            USE_2D_BLOCK=use_2d_block,
            MBITS=mbits,
            TARGET_MAX_POW2=target,
            FP8_MAX=fp8_max,
        )
    return out_fp8, scale


def grouped_quantize_mxfp8_triton(
    input: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    dest_dtype: torch.dtype,
    axis: int,
    use_2d_block: bool = False,
    shuffle_scale: bool = False,
    shuffle_out: bool = False,
) -> List[torch.Tensor]:
    _check_no_shuffle(shuffle_scale=shuffle_scale, shuffle_out=shuffle_out)
    assert input.is_cuda and input.is_contiguous() and input.dim() == 2
    assert input.dtype in (torch.bfloat16, torch.float16)
    assert axis in (0, 1)

    is_rowwise = axis == 1
    G = group_lens.size(0)
    total_M, N = input.size(0), input.size(1)
    assert N % MXFP8_BLOCK_SIZE == 0, f"N must be divisible by {MXFP8_BLOCK_SIZE}"

    m_align = _GROUP_ROW_ALIGN if is_rowwise else _GROUP_COL_ALIGN
    M_pad_out = _cdiv(total_M + G * m_align, m_align) * m_align
    N_pad = _cdiv(N, _GROUP_COL_ALIGN) * _GROUP_COL_ALIGN
    mbits, target, fp8_max = _fp8_params(dest_dtype)

    group_lens_padded, group_offs_padded = _compute_padded_layout(group_lens, m_align)

    out_fp8, scale = _grouped_quantize_single(
        input,
        group_offs,
        group_offs_padded,
        dest_dtype,
        is_rowwise,
        G,
        total_M,
        N,
        N_pad,
        M_pad_out,
        use_2d_block,
        mbits,
        target,
        fp8_max,
    )
    return [out_fp8, scale.view(E8M0_DTYPE), group_lens_padded, group_offs_padded]


def grouped_quantize_mxfp8_dual_triton(
    input: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    dest_dtype: torch.dtype,
    rowwise_use_2d_block: bool = False,
    colwise_use_2d_block: bool = False,
    shuffle_rowwise_scale: bool = False,
    shuffle_rowwise: bool = False,
    shuffle_colwise_scale: bool = False,
    shuffle_colwise: bool = False,
) -> List[torch.Tensor]:
    _check_no_shuffle(
        shuffle_rowwise_scale=shuffle_rowwise_scale,
        shuffle_rowwise=shuffle_rowwise,
        shuffle_colwise_scale=shuffle_colwise_scale,
        shuffle_colwise=shuffle_colwise,
    )
    assert input.is_cuda and input.is_contiguous() and input.dim() == 2
    assert input.dtype in (torch.bfloat16, torch.float16)

    G = group_lens.size(0)
    total_M, N = input.size(0), input.size(1)
    assert N % MXFP8_BLOCK_SIZE == 0, f"N must be divisible by {MXFP8_BLOCK_SIZE}"

    M_pad_row = _cdiv(total_M + G * _GROUP_ROW_ALIGN, _GROUP_ROW_ALIGN) * _GROUP_ROW_ALIGN
    M_pad_col = _cdiv(total_M + G * _GROUP_COL_ALIGN, _GROUP_COL_ALIGN) * _GROUP_COL_ALIGN
    N_pad = _cdiv(N, _GROUP_COL_ALIGN) * _GROUP_COL_ALIGN
    mbits, target, fp8_max = _fp8_params(dest_dtype)

    glp_row, gop_row = _compute_padded_layout(group_lens, _GROUP_ROW_ALIGN)
    glp_col, gop_col = _compute_padded_layout(group_lens, _GROUP_COL_ALIGN)

    rw_fp8, rw_scale = _grouped_quantize_single(
        input,
        group_offs,
        gop_row,
        dest_dtype,
        True,
        G,
        total_M,
        N,
        N_pad,
        M_pad_row,
        rowwise_use_2d_block,
        mbits,
        target,
        fp8_max,
    )
    cw_fp8, cw_scale = _grouped_quantize_single(
        input,
        group_offs,
        gop_col,
        dest_dtype,
        False,
        G,
        total_M,
        N,
        N_pad,
        M_pad_col,
        colwise_use_2d_block,
        mbits,
        target,
        fp8_max,
    )
    return [
        rw_fp8,
        rw_scale.view(E8M0_DTYPE),
        cw_fp8,
        cw_scale.view(E8M0_DTYPE),
        glp_row,
        gop_row,
        glp_col,
        gop_col,
    ]
