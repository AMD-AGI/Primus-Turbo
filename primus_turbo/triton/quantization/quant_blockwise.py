###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl


@triton.jit
def compute_scale_and_quant(x_tile, x_tile_abs, axis, FP8_MAX):
    x_tile_max = tl.max(x_tile_abs, axis=axis, keep_dims=True)
    x_tile_max = tl.maximum(x_tile_max, 1e-4)
    x_scales_tile = FP8_MAX / x_tile_max
    x_fp8_tile = x_tile * x_scales_tile
    x_fp8_tile = tl.clamp(x_fp8_tile, min=-FP8_MAX, max=FP8_MAX)
    return x_fp8_tile, x_scales_tile


# Standard blockwise quantize kernel
@triton.jit
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, AXIS, FP8_MAX)

    # Store output
    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    # Store scale
    if AXIS == 1:
        scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:
        scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(
        x_scales_ptr + scale_offs,
        x_scales_tile_inv,
        mask=scale_mask,
    )


@triton.jit
def quant_fp8_blockwise_dual_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_scales_row_ptr,
    x_fp8_col_ptr,
    x_scales_col_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    COL_TRANSPOSED: tl.constexpr = False,
    COL_PRESHUFFLED: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_row_tile, x_scales_row_tile = compute_scale_and_quant(x_tile, x_tile_abs, 1, FP8_MAX)
    # Col-axis (axis=0) reduction routed through a single tl.trans so the max-reduction
    # runs along the contiguous axis (axis=1 of the transposed tile). This swizzles the
    # LDS staging and eliminates the strided cross-lane bank conflicts of the direct
    # axis=0 reduction. The transposed tile (and everything derived from it) stays in
    # [n, m] orientation: abs commutes with transpose so x_tile_abs_t is taken directly
    # from x_tile_t (no second LDS transpose), and the col fp8/scale results are kept
    # transposed and stored from there. Algebraically/byte-identical: same per-column
    # max, scale, and quant.
    x_tile_t = tl.trans(x_tile)
    x_tile_abs_t = tl.abs(x_tile_t)
    x_fp8_col_tile_t, x_scales_col_tile_t = compute_scale_and_quant(x_tile_t, x_tile_abs_t, 1, FP8_MAX)

    x_fp8_row_ptrs = x_fp8_row_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_row_ptrs, x_fp8_row_tile.to(x_fp8_row_ptr.dtype.element_ty), mask=mask)

    # Col-quant data store. With COL_TRANSPOSED the column-quantized tile is
    # written directly into an [N, M] output buffer (element (m, n) -> flat
    # n*M + m) instead of the default [M, N] (element (m, n) -> flat m*N + n).
    # The col results are already in [n, m] (transposed) orientation, so the
    # COL_TRANSPOSED store writes x_fp8_col_tile_t straight out with a transposed
    # pointer/mask -- byte-identical to the previous trans-then-store-into-[N,M]
    # path but with both output transposes removed. The [N, M] layout matches
    # exactly the operand the FlyDSL wgrad kernel needs for grad_out
    # (A = grad_out^T), so the separate elementwise transpose-copy in the wgrad
    # launcher collapses to a zero-cost view. The default [M, N] col path keeps
    # the output transpose. The col-scale layout ([M//128, N], stored below) is
    # unchanged.
    if COL_PRESHUFFLED:
        # Write the column-quantized tile DIRECTLY in the FlyDSL (16, 16) MFMA
        # preshuffled+transposed operand layout, byte-identical to
        # shuffle_b(x_col[M, N].transpose(0, 1).contiguous()) consumed by the
        # wgrad GEMM. This folds the launcher's standalone preshuffle copy (a
        # full M*N fp8 HBM round-trip + a dedicated kernel launch per backward
        # step) into this store. The index math mirrors
        # preshuffle_fp8._preshuffle_b_transposed_kernel exactly with (p=n, q=m)
        # (here the operand is the transpose, so P=N, Q=M; BN=16, BK=32,
        # K_inner=16 for the (16, 16) fp8 layout). Output is contiguous in runs
        # of K_inner along m (last dim), so the store coalesces. The col results
        # are already in [n, m] orientation (x_fp8_col_tile_t), so they store
        # straight out at the permuted offset.
        BN_PS: tl.constexpr = 16
        BK_PS: tl.constexpr = 32
        KIN_PS: tl.constexpr = 16
        i0 = offs_n // BN_PS
        i1 = offs_n % BN_PS
        i2 = offs_m // BK_PS
        i3 = (offs_m % BK_PS) // KIN_PS
        i4 = offs_m % KIN_PS
        ps_off = i4[None, :] + KIN_PS * (
            i1[:, None]
            + BN_PS * (i3[None, :] + (BK_PS // KIN_PS) * (i2[None, :] + (M // BK_PS) * i0[:, None]))
        )
        col_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(x_fp8_col_ptr + ps_off, x_fp8_col_tile_t.to(x_fp8_col_ptr.dtype.element_ty), mask=col_mask)
    elif COL_TRANSPOSED:
        x_fp8_col_ptrs = x_fp8_col_ptr + offs_n[:, None] * M + offs_m[None, :]
        col_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(x_fp8_col_ptrs, x_fp8_col_tile_t.to(x_fp8_col_ptr.dtype.element_ty), mask=col_mask)
    else:
        x_fp8_col_tile = tl.trans(x_fp8_col_tile_t)
        x_fp8_col_ptrs = x_fp8_col_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(x_fp8_col_ptrs, x_fp8_col_tile.to(x_fp8_col_ptr.dtype.element_ty), mask=mask)

    row_scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    row_scale_mask = offs_m < M
    x_scales_row_tile_inv = tl.reshape(1.0 / x_scales_row_tile, BLOCK_SIZE)
    tl.store(
        x_scales_row_ptr + row_scale_offs,
        x_scales_row_tile_inv,
        mask=row_scale_mask,
    )

    if COL_PRESHUFFLED:
        # Matching transposed scale layout [N, M // BLOCK_SIZE] (byte-identical to
        # the wgrad launcher's a_col_scale_inv.transpose(0, 1).contiguous()), so
        # the launcher's scale transpose is also dropped. Element (n, pid_m) at
        # flat n * num_m_blocks + pid_m.
        num_m_blocks = tl.cdiv(M, BLOCK_SIZE)
        col_scale_offs = offs_n * num_m_blocks + pid_m
        col_scale_mask = offs_n < N
    else:
        col_scale_offs = pid_m * N + offs_n
        col_scale_mask = offs_n < N
    # x_scales_col_tile_t is [n, 1]; flattening it yields the per-column scales in
    # the same n-order as flattening the previously-transposed [1, n] tile, so the
    # output scale layout is unchanged while the transpose is dropped.
    x_scales_col_tile_inv = tl.reshape(1.0 / x_scales_col_tile_t, BLOCK_SIZE)
    tl.store(
        x_scales_col_ptr + col_scale_offs,
        x_scales_col_tile_inv,
        mask=col_scale_mask,
    )


# w_ptr         [B, M, N]
# w_fp8_ptr     [B, M, N] FP8
# w_scales_ptr  [B, M // BLOCK_SIZE, N // BLOCK_SIZE] FP32
@triton.jit
def quant_fp8_blockwise_for_weight_kernel(
    w_ptr,
    w_fp8_ptr,
    w_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_offset_w = bid * M * N
    batch_offset_scales = bid * tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    w_ptrs = w_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    w_tile_abs = tl.abs(w_tile)
    w_tile_max = tl.max(w_tile_abs)  # [1]
    w_tile_max = tl.maximum(w_tile_max, 1e-4)
    w_scales = FP8_MAX / w_tile_max
    w_fp8_tile = w_tile * w_scales
    w_fp8_tile = tl.clamp(w_fp8_tile, min=-FP8_MAX, max=FP8_MAX)

    # Store
    w_fp8_ptrs = w_fp8_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile.to(w_fp8_ptr.dtype.element_ty), mask=mask)
    # Store scale
    scale_offs = batch_offset_scales + pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)


# w_ptr           [M, N]            (2D weight; M=rows / out-features, N=cols / in-features)
# w_fp8_ptr       [M, N] FP8        plain block2d (for ctx / dgrad)
# w_fp8_ps_ptr    [M, N] FP8        forward (16, 16) MFMA pre-shuffled (byte-identical to shuffle_b)
# w_fp8_dgrad_ps_ptr [M, N] FP8     dgrad (NN) transposed-preshuffle (byte-identical to
#                                   _shuffle_b_transposed(w_fp8); only written when EMIT_DGRAD_PS)
# w_scales_ptr    [M // BLOCK_SIZE, N // BLOCK_SIZE] FP32
@triton.jit
def quant_fp8_blockwise_for_weight_dual_kernel(
    w_ptr,
    w_fp8_ptr,
    w_fp8_ps_ptr,
    w_fp8_dgrad_ps_ptr,
    w_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EMIT_DGRAD_PS: tl.constexpr = False,
):
    """2D blockwise weight quant that emits BOTH the plain block2d FP8 weight and
    its forward (16, 16) MFMA pre-shuffled copy in a single read of ``w``.

    The pre-shuffled store is byte-identical to ``shuffle_b(w_fp8)`` (the launcher's
    standalone ``permute().contiguous()`` copy of the [M, N] weight), so the FlyDSL
    forward GEMM can consume it directly and skip its standalone pre-shuffle HBM
    round-trip + kernel launch. The plain copy is kept for ``ctx`` / dgrad (which
    applies a different, transposed pre-shuffle). One read of ``w``, two FP8 stores.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    w_ptrs = w_ptr + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    w_tile_abs = tl.abs(w_tile)
    w_tile_max = tl.max(w_tile_abs)  # [1]
    w_tile_max = tl.maximum(w_tile_max, 1e-4)
    w_scales = FP8_MAX / w_tile_max
    w_fp8_tile = w_tile * w_scales
    w_fp8_tile = tl.clamp(w_fp8_tile, min=-FP8_MAX, max=FP8_MAX)
    w_fp8_cast = w_fp8_tile.to(w_fp8_ptr.dtype.element_ty)

    # Plain block2d store (consumed by ctx / dgrad).
    w_fp8_ptrs = w_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_cast, mask=mask)

    # Pre-shuffled store in the FlyDSL forward (16, 16) MFMA layout, byte-identical
    # to shuffle_b(w_fp8) [N_shuf=M (rows), K_shuf=N (cols); BN=16, BK=32,
    # K_inner=16 for 1-byte fp8]. shuffle_b reads element (row, col) at the 5D view
    # index (i0, i1, i2, i3, i4) with row = i0*BN + i1, col = i2*BK + i3*K_inner + i4
    # and writes it at permute(0, 2, 3, 1, 4); the flat offset below reproduces that
    # exactly. Output is contiguous in runs of K_inner along the cols, so the store
    # coalesces.
    BN_PS: tl.constexpr = 16
    BK_PS: tl.constexpr = 32
    KIN_PS: tl.constexpr = 16
    i0 = offs_m // BN_PS
    i1 = offs_m % BN_PS
    i2 = offs_n // BK_PS
    i3 = (offs_n % BK_PS) // KIN_PS
    i4 = offs_n % KIN_PS
    ps_off = i4[None, :] + KIN_PS * (
        i1[:, None]
        + BN_PS * (i3[None, :] + (BK_PS // KIN_PS) * (i2[None, :] + (N // BK_PS) * i0[:, None]))
    )
    tl.store(w_fp8_ps_ptr + ps_off, w_fp8_cast, mask=mask)

    # Dgrad (NN) transposed-preshuffle store in the SAME read of w. Byte-identical
    # to _shuffle_b_transposed(w_fp8) == shuffle_b(w_fp8.transpose(0, 1)), the
    # operand the FlyDSL dgrad GEMM consumes (logical [K, N] from w [M=N_w, N=K_w]).
    # T = w_fp8.T has shape [P=N (=K_w cols), Q=M (=N_w rows)]; shuffle_b reads
    # T[row, col] = w_fp8[col, row] with row over P (=offs_n) and col over Q
    # (=offs_m), so i0/i1 index offs_n (rows of T) and i2/i3/i4 index offs_m (cols
    # of T), with Q = M. The stored tile is tl.trans(w_fp8_cast) -> [offs_n, offs_m].
    # Folding this here eliminates the standalone per-backward-step
    # _preshuffle_b_transposed_kernel; the dgrad GEMM's operand load is unchanged.
    if EMIT_DGRAD_PS:
        i0d = offs_n // BN_PS
        i1d = offs_n % BN_PS
        i2d = offs_m // BK_PS
        i3d = (offs_m % BK_PS) // KIN_PS
        i4d = offs_m % KIN_PS
        dps_off = i4d[None, :] + KIN_PS * (
            i1d[:, None]
            + BN_PS * (i3d[None, :] + (BK_PS // KIN_PS) * (i2d[None, :] + (M // BK_PS) * i0d[:, None]))
        )
        dps_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(w_fp8_dgrad_ps_ptr + dps_off, tl.trans(w_fp8_cast), mask=dps_mask)

    # Store scale (layout unchanged vs the plain weight quant).
    scale_offs = pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)


@triton.jit
def quant_fp8_blockwise_segment_m_row_col_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_fp8_col_padded_ptr,
    x_scales_row_ptr,
    x_scales_col_padded_ptr,
    group_offs_ptr,
    padded_group_offs_ptr,
    M_in,
    N,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Fused row + segment-padded col quant for grouped GEMM (dgrad + variable-K wgrad).

    One bf16 read of x produces: row-wise scaled output (for the fwd/dgrad GEMM) and
    segment-padded col-wise scaled output (for the variable-K wgrad GEMM). Each program
    handles one [BLOCK, BLOCK] tile in padded output space; row outputs are emitted only
    for valid input rows. Row scales are written pre-shuffled as [N_blocks, M_in] so the
    persistent fwd GEMM reads them by stride with no runtime .T.contiguous().
    """
    pid_m = tl.program_id(axis=0)  # padded M index
    pid_n = tl.program_id(axis=1)

    M_padded = tl.load(padded_group_offs_ptr + num_groups)
    block_start = pid_m * BLOCK_SIZE
    if block_start >= M_padded:
        return

    # Find group containing this padded tile
    group_id = 0
    for g in range(num_groups):
        ps = tl.load(padded_group_offs_ptr + g)
        pe = tl.load(padded_group_offs_ptr + g + 1)
        if block_start >= ps and block_start < pe:
            group_id = g

    orig_start = tl.load(group_offs_ptr + group_id)
    orig_end = tl.load(group_offs_ptr + group_id + 1)
    pad_start = tl.load(padded_group_offs_ptr + group_id)

    offs_m_out = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_m_in = orig_start + (offs_m_out - pad_start)

    mask_in = (offs_m_in[:, None] >= orig_start) & (offs_m_in[:, None] < orig_end) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_m_in[:, None] * N + offs_n[None, :], mask=mask_in, other=0.0).to(tl.float32)
    x_abs = tl.abs(x)

    col_max = tl.maximum(tl.max(x_abs, axis=0, keep_dims=True), 1e-4)
    col_scale = FP8_MAX / col_max
    x_fp8_col = tl.clamp(x * col_scale, min=-FP8_MAX, max=FP8_MAX).to(x_fp8_col_padded_ptr.dtype.element_ty)

    row_max = tl.maximum(tl.max(x_abs, axis=1, keep_dims=True), 1e-4)
    row_scale = FP8_MAX / row_max
    x_fp8_row = tl.clamp(x * row_scale, min=-FP8_MAX, max=FP8_MAX).to(x_fp8_row_ptr.dtype.element_ty)

    out_mask_pad = (offs_m_out[:, None] < M_padded) & (offs_n[None, :] < N)
    tl.store(
        x_fp8_col_padded_ptr + offs_m_out[:, None] * N + offs_n[None, :],
        x_fp8_col,
        mask=out_mask_pad,
    )

    tl.store(
        x_fp8_row_ptr + offs_m_in[:, None] * N + offs_n[None, :],
        x_fp8_row,
        mask=mask_in,
    )

    col_scale_inv = tl.reshape(1.0 / col_scale, BLOCK_SIZE)
    col_scale_mask = (pid_m < tl.cdiv(M_padded, BLOCK_SIZE)) & (offs_n < N)
    tl.store(
        x_scales_col_padded_ptr + pid_m * N + offs_n,
        col_scale_inv,
        mask=col_scale_mask,
    )

    row_scale_inv = tl.reshape(1.0 / row_scale, BLOCK_SIZE)
    row_scale_mask = (offs_m_in >= orig_start) & (offs_m_in < orig_end) & (offs_m_in < M_in)
    # Pshuffled [N_blocks, M_in]: matches the persistent fwd GEMM's scale order, so the
    # GEMM reads scales coalesced with no runtime .T.contiguous(). The HIP fast path
    # (quantize_fp8_blockwise_segment_m_row_col) emits the same layout.
    tl.store(
        x_scales_row_ptr + pid_n * M_in + offs_m_in,
        row_scale_inv,
        mask=row_scale_mask,
    )
