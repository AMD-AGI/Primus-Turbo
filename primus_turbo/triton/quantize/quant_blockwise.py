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


# Blockwise quantize
@triton.jit
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AIXS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, AIXS, FP8_MAX)

    # Store
    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    # Store scale
    if AIXS == 1:
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


# w_ptr         [M, N]
# w_fp8_ptr     [M, N] FP8
# w_scales_ptr  [M // BLOCK_SIZE, N // BLOCK_SIZE] FP32
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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

    # Store
    w_fp8_ptrs = w_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile.to(w_fp8_ptr.dtype.element_ty), mask=mask)
    # Store scale
    scale_offs = pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)


# x_ptr             [M, N]
# x_fp8_row_ptr     [M, N] FP8
# x_fp8_col_ptr     [M, N] FP8
# x_scales_row_ptr  [M, N // BLOCK_SIZE] FP32
# x_scales_col_ptr  [M // BLOCK_SIZE, N] FP32
@triton.jit
def quant_fp8_blockwise_for_act_grad_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_scales_row_ptr,
    x_fp8_col_ptr,
    x_scales_col_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    # Row-wise quantization
    x_fp8_tile_row, x_scales_tile_row = compute_scale_and_quant(x_tile, x_tile_abs, 1, FP8_MAX)

    # Col-wise quantization
    x_fp8_tile_col, x_scales_tile_col = compute_scale_and_quant(x_tile, x_tile_abs, 0, FP8_MAX)

    # Store
    x_fp8_row_ptrs = x_fp8_row_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_fp8_col_ptrs = x_fp8_col_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_row_ptrs, x_fp8_tile_row.to(x_fp8_row_ptr.dtype.element_ty), mask=mask)
    tl.store(x_fp8_col_ptrs, x_fp8_tile_col.to(x_fp8_col_ptr.dtype.element_ty), mask=mask)

    # Store row-wise scales inverse: [M, N // BLOCK_SIZE]
    row_scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    x_scales_tile_row_inv = tl.reshape(1.0 / x_scales_tile_row, BLOCK_SIZE)
    tl.store(
        x_scales_row_ptr + row_scale_offs,
        x_scales_tile_row_inv,
        mask=offs_m < M,
    )

    # Store col-wise scales inverse: [M // BLOCK_SIZE, N]
    col_scale_offs = pid_m * N + offs_n
    x_scales_tile_col_inv = tl.reshape(1.0 / x_scales_tile_col, BLOCK_SIZE)
    tl.store(
        x_scales_col_ptr + col_scale_offs,
        x_scales_tile_col_inv,
        mask=offs_n < N,
    )
