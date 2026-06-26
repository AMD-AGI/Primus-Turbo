###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import torch
import triton

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP4_PADDING_ALIGN_SIZE,
    MXFP8_BLOCK_SIZE,
    MXFP8_PADDING_ALIGN_SIZE,
    ScalingRecipe,
    check_mxfp4_support,
    check_mxfp8_support,
)
from primus_turbo.triton.quantization.quant_blockwise import (
    quant_fp8_blockwise_dual_kernel,
    quant_fp8_blockwise_for_weight_dual_kernel,
    quant_fp8_blockwise_for_weight_kernel,
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_row_col_kernel,
)


def ceil_div(a, b):
    return (a + b - 1) // b


def quantize_fp8_tensorwise_impl(
    x: torch.Tensor, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Tensor-Wise
    """
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, out_dtype, None)
    return x_fp8, scale_inv


def quantize_fp8_rowwise_impl(
    x: torch.Tensor, out_dtype: torch.dtype, axis: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Row-Wise
    """
    assert x.is_contiguous(), "The x tensor must be contiguous."
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_rowwise(x, out_dtype, axis, None)

    return x_fp8, scale_inv


def dequantize_fp8_tensorwise_impl(x: torch.Tensor, out_dtype: torch.dtype, scale_inv: torch.Tensor):
    """
    DeQuantize FP8 Tensor-Wise
    """
    return torch.ops.primus_turbo_cpp_extension.dequantize_fp8_tensorwise(x, scale_inv, out_dtype)


def dequantize_fp8_rowwise_impl(x: torch.Tensor, out_dtype: torch.dtype, axis: int, scale_inv: torch.Tensor):
    """
    DeQuantize FP8 Row-Wise
    """
    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert scale_inv.is_contiguous(), "The scale_inv tensor must be contiguous."

    return torch.ops.primus_turbo_cpp_extension.dequantize_fp8_rowwise(x, scale_inv, axis, out_dtype)


@torch.library.custom_op("primus_turbo::quant_fp8_blockwise_impl", mutates_args=())
def quant_fp8_blockwise_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise.

    Quantizes a 2D tensor using blockwise scale along the specified axis.
    Assumes `x` is contiguous and 2D.

    Args:
        x: Input tensor to quantize.
        dtype: FP8 dtype for output.
        axis: Axis along which to compute blockwise scales (0 or 1).
        block_size: Block size for quantization.

    Returns:
        x_fp8: FP8-quantized tensor.
        x_scales: Per-block scale tensor in float32.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2

    M, N = x.shape

    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (M, triton.cdiv(N, block_size)) if axis == 1 else (triton.cdiv(M, block_size), N)
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_kernel[grid](
        x,
        x_fp8,
        x_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
        axis,
    )
    return x_fp8, x_scales


@quant_fp8_blockwise_impl.register_fake
def quant_fp8_blockwise_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2

    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (M, triton.cdiv(N, block_size)) if axis == 1 else (triton.cdiv(M, block_size), N)
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    return x_fp8, x_scales


@torch.library.custom_op("primus_turbo::quant_fp8_blockwise_dual_impl", mutates_args=())
def quant_fp8_blockwise_dual_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
    col_transposed: bool = False,
    col_preshuffled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor in both blockwise row and column modes in one pass.

    When ``col_transposed`` is True the column-quantized FP8 output is stored
    directly in transposed ``[N, M]`` layout (byte-identical to
    ``x_fp8_col[M,N].transpose(0,1).contiguous()``) so a downstream TN GEMM that
    needs ``x^T`` (the FlyDSL wgrad path consuming grad_out) can use it without a
    separate elementwise transpose-copy. The col-scale shape ``[M//128, N]`` is
    unchanged. The row output and row scale are unaffected.

    When ``col_preshuffled`` is True the column-quantized FP8 output is stored
    directly in the FlyDSL (16, 16) MFMA preshuffled+transposed operand layout
    and the col-scale is stored transposed ``[N, M//128]``, so the wgrad
    launcher's standalone preshuffle copy + scale transpose collapse to zero-cost
    views. The two flags are mutually exclusive.

    NOTE: This op is registered as ``torch.library.custom_op`` (opaque to
    inductor) instead of ``triton_op`` + ``wrap_triton``. On the AMD MI300
    (gfx942) + Triton 3.7 combination, the analysis pass
    ``identify_mutated_tensors`` raises ``IndexError('Function argument index
    out of range')`` while generating TTIR. PyTorch then falls back to the
    conservative "all inputs mutated" path, but in this PyTorch build that
    fallback only checks ``isinstance(value, Tensor)`` and misses ``TensorBox``
    values seen during inductor lowering, so the kernel call is silently
    eliminated by DCE and the returned ``torch.empty`` buffers leak as the
    "outputs". The opaque ``custom_op`` registration sidesteps the
    ``identify_mutated_tensors`` path entirely and matches the pattern already
    used by ``quant_fp8_blockwise_impl`` and
    ``quant_fp8_blockwise_for_weight_impl``.
    """
    assert not (col_transposed and col_preshuffled), "col_transposed and col_preshuffled are mutually exclusive"
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"

    M, N = x.shape
    row_scales_shape = (M, triton.cdiv(N, block_size))
    # Col-scale buffer: transposed [N, M // block] when preshuffled (matches the
    # wgrad operand's transposed scale), else [M // block, N].
    col_scales_shape = (
        (N, triton.cdiv(M, block_size)) if col_preshuffled else (triton.cdiv(M, block_size), N)
    )

    x_fp8_row = torch.empty((M, N), dtype=dtype, device=x.device)
    x_scales_row = torch.empty(row_scales_shape, dtype=torch.float32, device=x.device)
    # Col FP8 buffer shape. With col_preshuffled the buffer holds the (16, 16)
    # MFMA preshuffled+transposed operand bytes (logical [N, M]) but is allocated
    # with the un-transposed [M, N] shape so the downstream GEMM dispatch (keyed
    # on operand shape) is byte-for-byte unchanged; the wgrad launcher reshapes
    # it to [N, M] and skips its standalone preshuffle copy. With col_transposed
    # the buffer is [N, M]; otherwise [M, N].
    if col_preshuffled:
        col_fp8_shape = (M, N)
    elif col_transposed:
        col_fp8_shape = (N, M)
    else:
        col_fp8_shape = (M, N)
    x_fp8_col = torch.empty(col_fp8_shape, dtype=dtype, device=x.device)
    x_scales_col = torch.empty(col_scales_shape, dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_dual_kernel[grid](
        x,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
        COL_TRANSPOSED=col_transposed,
        COL_PRESHUFFLED=col_preshuffled,
    )
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


@quant_fp8_blockwise_dual_impl.register_fake
def quant_fp8_blockwise_dual_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
    col_transposed: bool = False,
    col_preshuffled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    M, N = x.shape
    row_scales_shape = (M, triton.cdiv(N, block_size))
    col_scales_shape = (
        (N, triton.cdiv(M, block_size)) if col_preshuffled else (triton.cdiv(M, block_size), N)
    )
    x_fp8_row = torch.empty((M, N), dtype=dtype, device=x.device)
    x_scales_row = torch.empty(row_scales_shape, dtype=torch.float32, device=x.device)
    if col_preshuffled:
        col_fp8_shape = (M, N)
    elif col_transposed:
        col_fp8_shape = (N, M)
    else:
        col_fp8_shape = (M, N)
    x_fp8_col = torch.empty(col_fp8_shape, dtype=dtype, device=x.device)
    x_scales_col = torch.empty(col_scales_shape, dtype=torch.float32, device=x.device)
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


@torch.library.custom_op("primus_turbo::quant_fp8_blockwise_for_weight_impl", mutates_args=())
def quant_fp8_blockwise_for_weight_impl(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise (weight).

    Quantizes a 2D or 3D weight tensor using blockwise scales along both axes.
    Assumes `w` is contiguous and 2D or 3D.

    Returns:
        w_fp8: FP8-quantized weight tensor.
        w_scales: Per-block scale tensor in float32.
    """

    assert w.dim() in (2, 3)
    if not w.is_contiguous():
        w = w.contiguous()

    # HIP fast path (single C++ call → lower host overhead, identical output layout);
    # only when the C++ op is built and block_size == 128 (its only supported size),
    # else fall through to the Triton kernel. Benefits both the grouped and
    # non-grouped blockwise paths that quantize weights through here.
    if block_size == 128 and hasattr(
        torch.ops.primus_turbo_cpp_extension, "quantize_fp8_blockwise_for_weight"
    ):
        return torch.ops.primus_turbo_cpp_extension.quantize_fp8_blockwise_for_weight(w, dtype, block_size)

    ori_dims = w.dim()
    if ori_dims == 2:
        B, M, N = 1, *w.shape
        w = w.unsqueeze(0)
    else:
        B, M, N = w.shape
    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (B, triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_for_weight_kernel[grid](
        w,
        w_fp8,
        w_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )

    if ori_dims == 2:
        w_fp8 = w_fp8.squeeze(0)
        w_scales = w_scales.squeeze(0)
    return w_fp8, w_scales


@quant_fp8_blockwise_for_weight_impl.register_fake
def quant_fp8_blockwise_for_weight_impl_meta(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert w.dim() in (2, 3)
    ori_dims = w.dim()
    if ori_dims == 2:
        B, M, N = 1, *w.shape
        w = w.unsqueeze(0)
    else:
        B, M, N = w.shape
    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    if ori_dims == 2:
        w_fp8 = w_fp8.squeeze(0)
        w_scales = w_scales.squeeze(0)
    return w_fp8, w_scales


@torch.library.custom_op("primus_turbo::quant_fp8_blockwise_for_weight_dual_impl", mutates_args=())
def quant_fp8_blockwise_for_weight_dual_impl(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D blockwise weight quant emitting plain + forward-preshuffled FP8 in one pass.

    Returns ``(w_fp8, w_fp8_ps, w_scales)`` where:
      - ``w_fp8``    : plain block2d FP8 [M, N]   (for ctx / dgrad)
      - ``w_fp8_ps`` : forward (16, 16) MFMA pre-shuffled FP8 [M, N], byte-identical
                       to ``shuffle_b(w_fp8)`` (consumed directly by the FlyDSL
                       forward GEMM, folding away its standalone pre-shuffle copy)
      - ``w_scales`` : per-block scale [M // block, N // block] fp32 (layout
                       unchanged vs the plain weight quant)

    Triton-only (the HIP C++ fast path has no pre-shuffled store), 2D weights only.
    """
    assert w.dim() == 2, "Input must be 2D"
    if not w.is_contiguous():
        w = w.contiguous()

    M, N = w.shape
    w_fp8 = torch.empty((M, N), dtype=dtype, device=w.device)
    w_fp8_ps = torch.empty((M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_for_weight_dual_kernel[grid](
        w,
        w_fp8,
        w_fp8_ps,
        w_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )
    return w_fp8, w_fp8_ps, w_scales


@quant_fp8_blockwise_for_weight_dual_impl.register_fake
def quant_fp8_blockwise_for_weight_dual_impl_meta(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert w.dim() == 2, "Input must be 2D"
    M, N = w.shape
    w_fp8 = torch.empty((M, N), dtype=dtype, device=w.device)
    w_fp8_ps = torch.empty((M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    return w_fp8, w_fp8_ps, w_scales


@torch.library.custom_op("primus_turbo::quant_fp8_blockwise_segment_m_row_col_impl", mutates_args=())
def quant_fp8_blockwise_segment_m_row_col_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    gemm_other_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused row + segment-padded col quant of a grouped tensor in one pass.

    A single bf16 read of `x` produces the row-wise scaled tensor (fwd/dgrad GEMM)
    and the segment-padded col-wise scaled tensor (variable-K wgrad). Row scales are
    pshuffled [N_blocks, M] to match the persistent GEMM's coalesced scale reads.

    Dispatches to the HIP fast path when the surrounding GEMM is small enough that its
    lower host overhead wins; otherwise launches the Triton kernel. Pass `gemm_other_dim`
    (the GEMM dim not visible from x.shape) to enable the dispatch; default = Triton.

    Registered as ``torch.library.custom_op`` (opaque to inductor) to sidestep the AMD
    MI300 / Triton 3.7 ``identify_mutated_tensors`` DCE bug, matching the other blockwise
    quant impls.

    Returns: (x_fp8_row [M,N], x_fp8_col_padded [M_padded_max,N],
              x_scales_row [N_blocks,M], x_scales_col_padded [M_padded_blocks,N],
              var_k_group_lens [B], var_k_group_offs [B+1]).
    """
    assert x.is_contiguous() and x.dim() == 2
    # HIP fast path: only when the C++ op is built and block_size == 128 (its
    # only supported size); otherwise fall through to the Triton kernel below.
    # HIP wins on small surrounding GEMMs (single C++ call → low host overhead) but
    # loses on big ones (vreg pressure → 1 block/CU bottlenecks the next GEMM), so it
    # is gated by a GEMM-FLOPs threshold. TODO: this empirical threshold (calibrated
    # across LFM2 / Qwen3 / DeepSeek-V3 / GPT-OSS BLOCKWISE shapes) is a heuristic —
    # replace with a principled cost model / autotune signal when available.
    hip_gemm_flops_threshold = 70_000_000_000
    if (
        gemm_other_dim is not None
        and block_size == 128
        and hasattr(torch.ops.primus_turbo_cpp_extension, "quantize_fp8_blockwise_segment_m_row_col")
    ):
        gemm_flops = x.size(0) * x.size(1) * gemm_other_dim
        if gemm_flops <= hip_gemm_flops_threshold:
            return torch.ops.primus_turbo_cpp_extension.quantize_fp8_blockwise_segment_m_row_col(
                x, dtype, block_size, group_lens, group_offs
            )

    M, N = x.shape
    num_groups = group_lens.size(0)
    # Segment-padded group offsets (each segment rounded up to block_size). Only the
    # Triton fallback (large GEMMs) reaches here, where this host cost is negligible;
    # the HIP fast path computes them in-kernel.
    var_k_group_lens = ((group_lens + block_size - 1) // block_size) * block_size
    var_k_group_offs = torch.zeros(num_groups + 1, dtype=torch.int64, device=x.device)
    var_k_group_offs[1:] = torch.cumsum(var_k_group_lens, dim=0)
    M_padded_max = M + num_groups * block_size
    # Kernel mask-writes cover all positions read downstream; skip zero-init.
    # x_scales_row in pshuffled [N_blocks, M] matches the fwd GEMM scale order.
    x_fp8_row = torch.empty((M, N), dtype=dtype, device=x.device)
    x_fp8_col_padded = torch.empty((M_padded_max, N), dtype=dtype, device=x.device)
    x_scales_row = torch.empty((triton.cdiv(N, block_size), M), dtype=torch.float32, device=x.device)
    x_scales_col_padded = torch.empty(
        (triton.cdiv(M_padded_max, block_size), N), dtype=torch.float32, device=x.device
    )
    grid = (triton.cdiv(M_padded_max, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_segment_m_row_col_kernel[grid](
        x,
        x_fp8_row,
        x_fp8_col_padded,
        x_scales_row,
        x_scales_col_padded,
        group_offs,
        var_k_group_offs,
        M,
        N,
        num_groups,
        block_size,
        torch.finfo(dtype).max,
    )
    return (
        x_fp8_row,
        x_fp8_col_padded,
        x_scales_row,
        x_scales_col_padded,
        var_k_group_lens,
        var_k_group_offs,
    )


@quant_fp8_blockwise_segment_m_row_col_impl.register_fake
def quant_fp8_blockwise_segment_m_row_col_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    gemm_other_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    num_groups = group_lens.size(0)
    M_padded_max = M + num_groups * block_size
    return (
        torch.empty((M, N), dtype=dtype, device=x.device),
        torch.empty((M_padded_max, N), dtype=dtype, device=x.device),
        torch.empty((triton.cdiv(N, block_size), M), dtype=torch.float32, device=x.device),
        torch.empty((triton.cdiv(M_padded_max, block_size), N), dtype=torch.float32, device=x.device),
        torch.empty(num_groups, dtype=torch.int64, device=x.device),
        torch.empty(num_groups + 1, dtype=torch.int64, device=x.device),
    )


def quantize_mxfp8_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    axis: Union[int, None],
    block_size: int,
    with_trans: bool = False,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    # NOTE: quantize fp8 kernel use the ISA which only available on cdna4.
    mxfp8_support, reason = check_mxfp8_support()
    assert mxfp8_support, reason

    assert block_size == MXFP8_BLOCK_SIZE, f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"

    scaling_recipe = ScalingRecipe() if scaling_recipe is None else scaling_recipe
    if with_trans:
        scaling_recipe_for_trans = (
            ScalingRecipe() if scaling_recipe_for_trans is None else scaling_recipe_for_trans
        )
    else:
        scaling_recipe_for_trans = scaling_recipe

    if not with_trans:
        assert axis in (0, 1), "The axis must be 0 or 1 when with_trans is False."
    else:
        assert axis is None, "The axis must be None when with_trans is True."

    if with_trans:
        return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8_dual(
            x,
            out_dtype,
            MXFP8_PADDING_ALIGN_SIZE,
            scaling_recipe.use_2d_block,
            scaling_recipe_for_trans.use_2d_block,
            scaling_recipe.shuffle_scale,
            scaling_recipe.shuffle_out,
            scaling_recipe_for_trans.shuffle_scale,
            scaling_recipe_for_trans.shuffle_out,
        )
    else:
        return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(
            x,
            out_dtype,
            axis,
            MXFP8_PADDING_ALIGN_SIZE,
            scaling_recipe.use_2d_block,
            scaling_recipe.shuffle_scale,
            scaling_recipe.shuffle_out,
        )


def grouped_quantize_mxfp8_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """MXFP8 dual (rowwise + colwise) quantization fused with per-group M-axis
    zero-padding.

    Each per-group region of ``x`` (defined by ``group_lens`` / ``group_offs``)
    is virtually zero-padded along M: rowwise to 32 and colwise to 128.  The
    padded per-group layouts are computed on GPU (no D2H sync).
    """
    mxfp8_support, reason = check_mxfp8_support()
    assert mxfp8_support, reason

    assert block_size == MXFP8_BLOCK_SIZE, f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"

    scaling_recipe = ScalingRecipe() if scaling_recipe is None else scaling_recipe
    scaling_recipe_for_trans = (
        ScalingRecipe() if scaling_recipe_for_trans is None else scaling_recipe_for_trans
    )

    return torch.ops.primus_turbo_cpp_extension.grouped_quantize_mxfp8_dual(
        x,
        group_lens,
        group_offs,
        out_dtype,
        scaling_recipe.use_2d_block,
        scaling_recipe_for_trans.use_2d_block,
        scaling_recipe.shuffle_scale,
        scaling_recipe.shuffle_out,
        scaling_recipe_for_trans.shuffle_scale,
        scaling_recipe_for_trans.shuffle_out,
    )


def dequantize_mxfp8_impl(
    x: torch.Tensor, out_dtype: torch.dtype, axis: int, block_size: int, scale_inv: torch.Tensor
) -> torch.Tensor:
    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert x.dim() == 2, "The x must be 2D tensor."
    assert scale_inv.dim() == 2, "The scale_inv must be 2D tensor."
    assert scale_inv.is_contiguous(), "The scale_inv tensor must be contiguous."
    assert axis in (0, 1), "The axis must be 0 or 1."
    SUPPORTED_OUT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    assert (
        out_dtype in SUPPORTED_OUT_DTYPES
    ), f"The out dtype must be one of {SUPPORTED_OUT_DTYPES} but got {out_dtype}."

    _, row_length = x.size()
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size."

    return torch.ops.primus_turbo_cpp_extension.dequantize_mxfp8(x, scale_inv, axis, block_size, out_dtype)


def quantize_mxfp4_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    axis: Union[int, None],
    block_size: int,
    with_trans: bool = False,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    # NOTE: quantize fp4 kernel use the ISA which only available on cdna4.
    mxfp4_support, reason = check_mxfp4_support()
    assert mxfp4_support, reason

    assert block_size == MXFP4_BLOCK_SIZE, f"The block size must be {MXFP4_BLOCK_SIZE} for MXFP4 quantization"

    scaling_recipe = ScalingRecipe() if scaling_recipe is None else scaling_recipe
    if with_trans:
        scaling_recipe_for_trans = (
            ScalingRecipe() if scaling_recipe_for_trans is None else scaling_recipe_for_trans
        )
    else:
        scaling_recipe_for_trans = scaling_recipe

    if not with_trans:
        assert axis in (0, 1), "The axis must be 0 or 1 when with_trans is False."
    else:
        assert axis is None, "The axis must be None when with_trans is True."

    assert x.is_contiguous(), "The x tensor must be contiguous."

    if with_trans:
        return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
            x,
            out_dtype,
            MXFP4_PADDING_ALIGN_SIZE,
            scaling_recipe.use_2d_block,
            scaling_recipe.use_sr,
            scaling_recipe.use_rht,
            scaling_recipe_for_trans.use_2d_block,
            scaling_recipe_for_trans.use_sr,
            scaling_recipe_for_trans.use_rht,
            scaling_recipe.shuffle_scale,
            scaling_recipe.shuffle_out,
            scaling_recipe_for_trans.shuffle_scale,
            scaling_recipe_for_trans.shuffle_out,
        )
    else:
        return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4(
            x,
            out_dtype,
            axis,
            MXFP4_PADDING_ALIGN_SIZE,
            scaling_recipe.use_2d_block,
            scaling_recipe.use_sr,
            scaling_recipe.use_rht,
            scaling_recipe.shuffle_scale,
            scaling_recipe.shuffle_out,
        )


def dequantize_mxfp4_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    axis: int,
    block_size: int,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert x.dim() == 2, "The x must be 2D tensor."
    assert scale_inv.dim() == 2, "The scale_inv must be 2D tensor."
    assert scale_inv.is_contiguous(), "The scale_inv tensor must be contiguous."
    assert axis in (
        0,
        1,
    ), "The axis must be 0 or 1."
    assert x.dtype == torch.float4_e2m1fn_x2, f"The x dtype must be torch.float4_e2m1fn_x2 but got {x.dtype}."
    SUPPORTED_OUT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    assert (
        out_dtype in SUPPORTED_OUT_DTYPES
    ), f"The out dtype must be one of {SUPPORTED_OUT_DTYPES} but got {out_dtype}."

    num_rows, row_length = x.size()
    # NOTE: x is packed in last dimension
    row_length = row_length * 2
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size."

    return torch.ops.primus_turbo_cpp_extension.dequantize_mxfp4(x, scale_inv, axis, block_size, out_dtype)
