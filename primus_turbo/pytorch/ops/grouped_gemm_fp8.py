###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)
from primus_turbo.pytorch.ops.quantization import (
    grouped_quantize_fp8_with_trans,
    quantize_fp8_with_trans,
)

__all__ = [
    "grouped_gemm_fp8",
]


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


class FP8GroupedGemmBlockFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size in [128], "Only block_size 128 is supported currently."
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        assert group_lens.size(0) == b.size(0), "group_lens size must match b size(0)."
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)

        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
            a, a_dtype, axis=1, block_size=config.block_size
        )

        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8,
            a_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
        )

        a_fp8_col, a_scale_inv_col, _, _ = quant_fp8_blockwise_segment_m_impl(
            a, a_dtype, config.block_size, group_lens, group_offs
        )

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)

        (
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors
        block_size = ctx.config.block_size
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in row-wise for dgrad
        grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
            grad_out, grad_out_dtype, axis=1, block_size=block_size
        )

        # grad_a: grad_out @ b^T
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8,
            grad_out_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # Quantize grad_out with segment padding for wgrad (colwise quantization)
        grad_out_fp8_col, grad_out_scale_inv_col, var_k_group_lens, var_k_group_offs = (
            quant_fp8_blockwise_segment_m_impl(
                grad_out,
                grad_out_dtype,
                block_size,
                group_lens,
                group_offs,
            )
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            var_k_group_lens,
            var_k_group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        return grad_a, grad_b, None, None, None, None, None, None


class FP8GroupedGemmRowFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.ROWWISE

        # --- A side: [total_m, k] grouped activation, row-wise scale on axis=-1 (K) ---
        if isinstance(a, QuantizedTensor):
            assert a._is_grouped_tensor, "A QuantizedTensor input must be a grouped tensor"
            check_quantized_tensor(a, config, axis=-1)
            quantized_a = a
            group_offs = a.group_offs
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                group_lens=group_lens,
            )

        # --- B side: 3D weight, row axis is K-direction, col axis is N-direction ---
        # trans_b=True  -> layout [G, N, K]: K is axis=-1, N is axis=-2
        # trans_b=False -> layout [G, K, N]: K is axis=-2, N is axis=-1
        b_row_axis = -1 if trans_b else -2
        b_col_axis = -2 if trans_b else -1
        if isinstance(b, QuantizedTensor):
            assert not b._is_grouped_tensor, "B QuantizedTensor input must not be a grouped tensor"
            check_quantized_tensor(b, config, axis=b_row_axis)
            quantized_b = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=b_row_axis,
                block_size=config.block_size,
            )

        out = grouped_gemm_fp8_impl(
            quantized_a.qdata,
            quantized_b.qdata,
            quantized_a.scale_inv,
            quantized_b.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # Col-wise trans cache for backward. If the caller pre-quantized this
        # and passed it via ``a_t`` / ``b_t``, reuse it directly; otherwise
        # derive it (dequantize + re-quantize along the other axis), mirroring
        # FP8GemmRowFunction in gemm_fp8.py.
        if a_t is not None:
            quantized_a_t = a_t
        else:
            quantized_a_t = QuantizedTensor.quantize(
                quantized_a.dequantize(),
                quantized_a.real_dtype,
                config.granularity,
                axis=-2,
                block_size=config.block_size,
                group_lens=group_lens,
            )

        if b_t is not None:
            quantized_b_t = b_t
        else:
            quantized_b_t = QuantizedTensor.quantize(
                quantized_b.dequantize(),
                quantized_b.real_dtype,
                config.granularity,
                axis=b_col_axis,
                block_size=config.block_size,
            )

        ctx.save_for_backward(
            quantized_a_t.qdata,
            quantized_b_t.qdata,
            quantized_a_t.scale_inv,
            quantized_b_t.scale_inv,
            group_lens,
            group_offs,
        )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # grad_out row-wise (axis=-1) for grad_a
        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
        )

        grad_a = grouped_gemm_fp8_impl(
            quantized_grad_out.qdata,
            b_fp8_col,
            quantized_grad_out.scale_inv,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # grad_out col-wise (axis=-2) for grad_b
        quantized_grad_out_t = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-2,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            quantized_grad_out_t.qdata,
            a_scale_inv_col,
            quantized_grad_out_t.scale_inv,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # Grads correspond to forward args:
        #   (a, b, a_t, b_t, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
        return grad_a, grad_b, None, None, None, None, None, None, None, None


class FP8GroupedGemmTensorFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.TENSORWISE

        # TENSORWISE has a single scalar scale, so the same buffers feed both
        # forward and backward (no separate trans cache needed).
        if isinstance(a, QuantizedTensor):
            assert a._is_grouped_tensor, "A QuantizedTensor input must be a grouped tensor"
            check_quantized_tensor(a, config)
            quantized_a = a
            group_offs = a.group_offs
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                group_lens=group_lens,
            )

        if isinstance(b, QuantizedTensor):
            assert not b._is_grouped_tensor, "B QuantizedTensor input must not be a grouped tensor"
            check_quantized_tensor(b, config)
            quantized_b = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        out = grouped_gemm_fp8_impl(
            quantized_a.qdata,
            quantized_b.qdata,
            quantized_a.scale_inv,
            quantized_b.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
            maybe_pre_sync=True,
        )

        ctx.save_for_backward(
            quantized_a.qdata,
            quantized_b.qdata,
            quantized_a.scale_inv,
            quantized_b.scale_inv,
            group_lens,
            group_offs,
        )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
        )

        grad_a = grouped_gemm_fp8_impl(
            quantized_grad_out.qdata,
            b_fp8,
            quantized_grad_out.scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8,
            quantized_grad_out.qdata,
            a_scale_inv,
            quantized_grad_out.scale_inv,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        return grad_a, grad_b, None, None, None, None, None, None


class FP8GroupedGemmMXFunc(torch.autograd.Function):
    """MXFP8 grouped GEMM autograd (MX_BLOCKWISE).

    Forward / dgrad / wgrad all land on the turbo MXFP8 backends.

    Only the NT layout is supported: ``trans_b`` must be ``True`` and ``b``
    is expected to already be ``(G, N, K)`` (the forward asserts this).
    Per-group M_g is zero-padded along the M axis to
    ``MXFP8_GROUP_M_PADDING_ALIGN_SIZE`` (32) for the rowwise fwd/dgrad path
    and to 128 for the colwise wgrad path.  Forward / dgrad outputs are
    sliced back to the user-visible shape; the wgrad output ``(G, N, K)``
    does not depend on the padding.
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [G,] int64
        group_offs: torch.Tensor,  # [G+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), "A and B must be torch.Tensor"
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2, "A must be 2D [total_M, K]."
        assert b.ndim == 3, "B must be 3D."
        assert trans_b, "MX_BLOCKWISE grouped GEMM only supports NT layout (trans_b=True)."

        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        # Kernel only supports NT; b is already (G, N, K) (asserted above).
        G, _, _ = b.size()
        a_dtype = b_dtype = _get_fp8_dtype(config.format, True)

        # Fused grouped quant for A: produces row + col MXFP8 outputs
        # in the padded layout AND computes the padded per-group layout
        # on GPU (no D2H sync).
        # rowwise A is 32-padded (read via group_offs_padded_rowwise); colwise A is
        # 128-padded (wgrad, read via group_offs_padded_colwise).
        (
            a_fp8_row,
            a_scale_inv_row,
            a_fp8_col,
            a_scale_inv_col,
            _,
            group_offs_padded_rowwise,
            _,
            _,
        ) = grouped_quantize_fp8_with_trans(
            a,
            a_dtype,
            config.granularity,
            group_lens,
            group_offs,
            block_size=config.block_size,
            scaling_recipe=ScalingRecipe(),
            scaling_recipe_for_trans=ScalingRecipe(),
        )
        # Per-group dual quant for B: rowwise (G, N, K_pad) for fwd,
        # colwise (G, K, N_pad) for dgrad, both ready for GEMM.
        b_fp8_row, b_scale_inv_row, b_fp8_col, b_scale_inv_col = quantize_fp8_with_trans(
            b,
            b_dtype,
            ScalingGranularity.MX_BLOCKWISE,
            block_size=config.block_size,
            scaling_recipe=ScalingRecipe(use_2d_block=True),
            scaling_recipe_for_trans=ScalingRecipe(use_2d_block=True),
        )

        # Inputs read from the padded layout via ``group_offs_padded``;
        # output is written directly to the unpadded (total_M, N) buffer
        # via ``group_offs_out``.  When balanced + aligned the two layouts
        # match and the write is a pure contiguous store.
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=num_cu,
            default_backend=BackendType.TURBO.value,
            group_offs_out=group_offs,
        )

        # ``out`` is written into the padded-input row buffer (a_fp8_row may be
        # over-allocated to an upper bound to avoid a D2H sync), but only the
        # first ``total_M`` rows hold valid unpadded data (group_offs_out maps each
        # group into the tight output layout). Slice back to the user-visible shape.
        total_m = a.shape[0]
        out = out[:total_m]

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_inv_col,
            b_fp8_col,
            b_scale_inv_col,
            group_lens,
            group_offs,
        )
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.total_m = total_m

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (
            a_fp8_col,
            a_scale_inv_col,
            b_fp8_col,
            b_scale_inv_col,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Fused grouped quant for grad_out (same op as fwd).
        (
            grad_out_fp8_row,
            grad_out_scale_inv_row,
            grad_out_t_fp8,
            grad_out_t_scale_inv,
            _,
            group_offs_padded_rowwise,
            group_lens_padded_colwise,
            group_offs_padded_colwise,
        ) = grouped_quantize_fp8_with_trans(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            group_lens,
            group_offs,
            block_size=ctx.config.block_size,
            scaling_recipe=ScalingRecipe(),
            scaling_recipe_for_trans=ScalingRecipe(),
        )

        # dgrad: read grad_out from 32-padded rowwise layout, write tight via group_offs_out.
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TURBO.value,
            group_offs_out=group_offs,
        )

        # Same padded-buffer slice-back as forward: dgrad is written into the
        # padded-input row buffer; only the first ``total_M`` rows are valid and
        # must match the user-visible ``a`` shape.
        grad_a = grad_a[: ctx.total_m]

        # wgrad: dB = dC^T @ A via fixed-NT variable-K turbo.  Operates
        # on the padded layout end-to-end (output is (G, N, K), no row
        # compression needed).
        grad_b = grouped_gemm_fp8_variable_k_impl(
            grad_out_t_fp8,
            a_fp8_col,
            grad_out_t_scale_inv,
            a_scale_inv_col,
            group_lens_padded_colwise,
            group_offs_padded_colwise,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TURBO.value,
        )

        return grad_a, grad_b, None, None, None, None, None


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Grouped FP8 GEMM constructs (Grouped)QuantizedTensor wrapper subclasses "
        "inside its autograd.Function.forward and reads their inner tensors "
        "(data / scale_inv / group_lens / group_offs). Dynamo cannot recover Python "
        "sources for those graph-internal inner tensors, tripping gb0116 "
        "('SourcelessBuilder.create cannot wrap FakeTensor'). "
    ),
)
def grouped_gemm_fp8(
    a: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    b: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    group_lens: torch.Tensor,
    group_offs: Union[torch.Tensor, None] = None,
    trans_b: bool = True,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """Grouped GEMM with FP8 quantization.

    This function automatically quantizes input tensors to FP8 based on the config,
    performs grouped matrix multiplication, and returns the result in the original dtype.

    Args:
        a: Input tensor A with shape [bs * m, k] (float16 or bfloat16).
            Can also be a pre-quantized :class:`QuantizedTensor` (grouped), or
            a :class:`QuantizedTensorPair` carrying both ``data`` (row-wise) and
            the backward-direction ``data_t`` (col-wise) for ROWWISE granularity.
        b: Input tensor B with shape [bs, k, n] or [bs, n, k] if trans_b (float16 or bfloat16).
            Same pre-quantized variants as ``a`` are accepted.
        group_lens: Group lengths tensor [bs] (int64)
        trans_b: Whether B is transposed (default: True)
        out_dtype: Output dtype (default: None, inferred from input dtypes)
        config: FP8 quantization config. If None, uses default (TENSORWISE, E4M3, DYNAMIC)
        num_cu: Number of compute units. If None, uses default (-1)

    Returns:
        Output tensor with shape [m, n] (same dtype as input)
    """
    if config is None:
        config = Float8QuantConfig()

    if group_offs is None:
        group_offs = group_offs_from_lens(group_lens)
    if isinstance(a, QuantizedTensorPair):
        a_data, a_data_t = a.data, a.data_t
    else:
        a_data, a_data_t = a, None

    if isinstance(b, QuantizedTensorPair):
        b_data, b_data_t = b.data, b.data_t
    else:
        b_data, b_data_t = b, None

    if out_dtype is None:
        out_dtype = torch.promote_types(a_data.dtype, b_data.dtype)

    if config.granularity == ScalingGranularity.TENSORWISE:
        # TENSORWISE has a single scalar scale (no col-wise trans cache needed);
        # the inner ``data_t`` is ignored if provided.
        return FP8GroupedGemmTensorFunc.apply(
            a_data, b_data, group_lens, group_offs, trans_b, out_dtype, config, num_cu
        )
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GroupedGemmRowFunc.apply(
            a_data,
            b_data,
            a_data_t,
            b_data_t,
            group_lens,
            group_offs,
            trans_b,
            out_dtype,
            config,
            num_cu,
        )
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        # BLOCKWISE only accepts raw tensors today; preserve existing assertion
        # behaviour in ``FP8GroupedGemmBlockFunc.forward``.
        return FP8GroupedGemmBlockFunc.apply(a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GroupedGemmMXFunc.apply(a, b, group_lens, group_offs, trans_b, config, num_cu)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
