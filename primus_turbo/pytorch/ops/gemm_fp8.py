###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp8_support,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.core.utils import is_gfx942
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    gemm_fp8_accum_impl,
    gemm_fp8_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_dual_impl,
    quantize_mxfp8_impl,
)
from primus_turbo.pytorch.ops.quantization import (
    quantize_fp8,
    quantize_fp8_with_trans,
)
from primus_turbo.pytorch.ops.utils import (
    _get_dummy_wgrad,
    _get_fp8_dtype,
    _setup_fused_grad_accum,
)

__all__ = ["gemm_fp8"]


def _deter_use_nt_layout_gemm_in_bwd(trans_a: bool, trans_b: bool):
    if is_gfx942():
        return False

    # NOTE: the non-NT layout gemm is not optimized for mi350/mi450.
    # Force to use NT layout GEMM in backward for now.
    return trans_a == False and trans_b == True


def _bgrad_gemm_fp8_impl_wrapper(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    granularity: int,
    default_backend: int,
    inplace_add_to_out: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the wgrad GEMM, accumulating into ``out`` when asked to.

    Returns the weight gradient for autograd, or a dummy buffer when the wgrad went
    straight into ``out``: forward already flagged the weight, so the training
    framework's own accumulation step stands down. Megatron still expects a tensor
    rather than None there, so its backward hooks stay on the main thread; the
    contents are never read. It is handed back in the weight's own dtype, since a
    mismatch would make autograd allocate and cast a full-size copy.
    """
    inputs = (a, a_scale_inv, trans_a, b, b_scale_inv, trans_b)
    options = dict(
        out_dtype=out_dtype,
        trans_c=trans_c,
        granularity=granularity,
    )

    if not inplace_add_to_out:
        return gemm_fp8_impl(*inputs, default_backend=default_backend, **options)

    assert out is not None, "out should not be None when inplace_add_to_out is True"
    # The wgrad keeps the caller's default backend: hipBLASLt and Triton both carry the
    # beta=1 epilogue. Backends without it report `inplace_add_to_out` as unsupported,
    # which keeps an explicitly pinned backend or auto-tune from silently landing
    # somewhere that ignores `out`.
    gemm_fp8_accum_impl(*inputs, out=out, default_backend=default_backend, **options)

    return _get_dummy_wgrad(out.shape, out_dtype)


class FP8GemmTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        use_nt_layout_gemm_in_bwd = _deter_use_nt_layout_gemm_in_bwd(trans_a, trans_b)
        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)

        if isinstance(a, QuantizedTensor):
            quantized_a = a
            check_quantized_tensor(quantized_a, config)
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        if use_nt_layout_gemm_in_bwd:
            if a_t is not None and isinstance(a_t, QuantizedTensor):
                quantized_a_t = a_t
            else:
                quantized_a_t = quantized_a.t().contiguous()

        if isinstance(b, QuantizedTensor):
            quantized_b = b
            check_quantized_tensor(quantized_b, config)
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        if use_nt_layout_gemm_in_bwd:
            if b_t is not None and isinstance(b_t, QuantizedTensor):
                quantized_b_t = b_t
            else:
                quantized_b_t = quantized_b.t().contiguous()

        out = gemm_fp8_impl(
            quantized_a.qdata,
            quantized_a.scale_inv,
            trans_a,
            quantized_b.qdata,
            quantized_b.scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        if use_nt_layout_gemm_in_bwd:
            ctx.save_for_backward(
                quantized_a_t.qdata,
                quantized_a_t.scale_inv,
                quantized_b_t.qdata,
                quantized_b_t.scale_inv,
            )
        else:
            ctx.save_for_backward(
                quantized_a.qdata, quantized_a.scale_inv, quantized_b.qdata, quantized_b.scale_inv
            )

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.use_nt_layout_gemm_in_bwd = use_nt_layout_gemm_in_bwd
        ctx.fuse_bgrad_accum = fuse_bgrad_accum
        ctx.main_grad = main_grad

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        if ctx.use_nt_layout_gemm_in_bwd:
            a_fp8_t, a_t_scale_inv, b_fp8_t, b_t_scale_inv = ctx.saved_tensors
        else:
            a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
        )

        if ctx.use_nt_layout_gemm_in_bwd:
            a_grad = gemm_fp8_impl(
                quantized_grad_out.qdata,
                quantized_grad_out.scale_inv,
                False,
                b_fp8_t,
                b_t_scale_inv,
                True,
                ctx.out_dtype,
                False,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )
        else:
            a_grad = gemm_fp8_impl(
                quantized_grad_out.qdata,
                quantized_grad_out.scale_inv,
                False,
                b_fp8,
                b_scale_inv,
                not ctx.trans_b,
                ctx.out_dtype,
                ctx.trans_a,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )

        if ctx.use_nt_layout_gemm_in_bwd:
            quantized_grad_out_t = quantized_grad_out.t().contiguous()

            b_grad = _bgrad_gemm_fp8_impl_wrapper(
                a_fp8_t,
                a_t_scale_inv,
                False,
                quantized_grad_out_t.qdata,
                quantized_grad_out_t.scale_inv,
                True,
                ctx.out_dtype,
                ctx.trans_b,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
                inplace_add_to_out=ctx.fuse_bgrad_accum,
                out=ctx.main_grad,
            )
        else:
            b_grad = _bgrad_gemm_fp8_impl_wrapper(
                a_fp8,
                a_scale_inv,
                not ctx.trans_a,
                quantized_grad_out.qdata,
                quantized_grad_out.scale_inv,
                False,
                ctx.out_dtype,
                ctx.trans_b,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
                inplace_add_to_out=ctx.fuse_bgrad_accum,
                out=ctx.main_grad,
            )

        return (
            a_grad,  # a
            b_grad,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # fuse_bgrad_accum_pattern
        )


class FP8GemmRowFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"

        if isinstance(a, QuantizedTensor):
            quantized_a = a
            check_quantized_tensor(quantized_a, config, axis=-1)
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        if a_t is None:
            quantized_a_t = QuantizedTensor.quantize(
                quantized_a.dequantize(),
                quantized_a.real_dtype,
                config.granularity,
                axis=-2,
                block_size=config.block_size,
            )
        else:
            assert isinstance(a_t, QuantizedTensor)
            quantized_a_t = a_t

        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, axis=-1 if trans_b else -2)
            quantized_b = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=-1 if trans_b else -2,
                block_size=config.block_size,
            )

        if b_t is None:
            quantized_b_t = QuantizedTensor.quantize(
                quantized_b.dequantize(),
                quantized_b.real_dtype,
                config.granularity,
                axis=-2 if trans_b else -1,
                block_size=config.block_size,
            )
        else:
            assert isinstance(b_t, QuantizedTensor)
            quantized_b_t = b_t

        out = gemm_fp8_impl(
            quantized_a.qdata,
            quantized_a.scale_inv,
            trans_a,
            quantized_b.qdata,
            quantized_b.scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        ctx.save_for_backward(
            quantized_a_t.qdata,
            quantized_a_t.scale_inv,
            quantized_b_t.qdata,
            quantized_b_t.scale_inv,
        )
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        a_fp8_t, a_t_scale_inv, b_fp8_t, b_t_scale_inv = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
        )

        # NT
        a_grad = gemm_fp8_impl(
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            b_fp8_t,
            b_t_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            ctx.trans_a,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        quantized_grad_out_t = QuantizedTensor.quantize(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2, block_size=ctx.config.block_size
        )

        # TN
        b_grad = gemm_fp8_impl(
            a_fp8_t,
            a_t_scale_inv,
            not ctx.trans_a,
            quantized_grad_out_t.qdata,
            quantized_grad_out_t.scale_inv,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        return (
            a_grad,  # a
            b_grad,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
        )


class FP8GemmBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],  # not used
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        flydsl_blockwise = GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.FLYDSL
        if flydsl_blockwise and config.block_size != 128:
            raise ValueError("FlyDSL BLOCKWISE GEMM requires block_size=128")
        scale_a_k_major = False
        if flydsl_blockwise:
            from primus_turbo.flydsl.gemm.blockscale_fp8_gemm import (
                select_blockscale_fp8_forward_kernel,
            )

            a_shape = a.qdata.shape if isinstance(a, QuantizedTensor) else a.shape
            b_shape = b.qdata.shape if isinstance(b, QuantizedTensor) else b.shape
            if trans_b:
                scale_a_k_major = select_blockscale_fp8_forward_kernel(
                    a_shape[0],
                    b_shape[0],
                    a_shape[1],
                )["scale_a_k_major"]

        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, axis=-1)
            a_row, a_row_scale = a.qdata, a.scale_inv
            if scale_a_k_major:
                a_row_scale = a_row_scale.T.contiguous()
            if a_t is None:
                a_t = QuantizedTensor.quantize(
                    a.dequantize(),
                    a_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                )

            a_col, a_col_scale = a_t.qdata, a_t.scale_inv
        else:
            if flydsl_blockwise and trans_b and a_dtype == float8_e4m3:
                if (
                    a.dtype == torch.bfloat16
                    and a.shape[0] % config.block_size == 0
                    and a.shape[1] % config.block_size == 0
                    and a.numel() * a.element_size() <= 0xFFFFFFFF
                ):
                    from primus_turbo.flydsl.quantization.blockwise_fp8_quant_poc import (
                        quantize_blockwise_fp8_dual,
                    )

                    a_row, a_row_scale, a_col, a_col_scale = quantize_blockwise_fp8_dual(
                        a,
                        row_scale_transposed=scale_a_k_major,
                    )
                else:
                    a_row, a_row_scale, a_col, a_col_scale = quant_fp8_blockwise_dual_impl(
                        a,
                        a_dtype,
                        config.block_size,
                        col_transposed=True,
                        row_scale_transposed=scale_a_k_major,
                    )
                a_col = a_col.transpose(0, 1)
            else:
                (
                    a_row,
                    a_row_scale,
                    a_col,
                    a_col_scale,
                ) = quantize_fp8_with_trans(
                    a,
                    a_dtype,
                    config.granularity,
                    block_size=config.block_size,
                )
        if scale_a_k_major:
            expected_scale_shape = (a_row.shape[1] // config.block_size, a_row.shape[0])
            if tuple(a_row_scale.shape) != expected_scale_shape:
                a_row_scale = a_row_scale.T.contiguous()

        # --- B side: 2D-block weight, reused unchanged in fwd + bwd. ---
        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, scaling_recipe=b_scaling_recipe)
            b_row, b_row_scale = b.qdata, b.scale_inv
        else:
            if (
                flydsl_blockwise
                and b_dtype == float8_e4m3
                and b.dtype == torch.bfloat16
                and b.dim() == 2
                and b.is_contiguous()
                and b.shape[1] % 16 == 0
                and b.numel() * b.element_size() <= 0xFFFFFFFF
            ):
                from primus_turbo.flydsl.quantization.blockwise_fp8_quant_poc import (
                    quantize_blockwise_fp8_weight,
                )

                b_row, b_row_scale = quantize_blockwise_fp8_weight(b)
            else:
                b_row, b_row_scale = quantize_fp8(
                    b,
                    b_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    scaling_recipe=b_scaling_recipe,
                )

        b_col, b_col_scale = b_row, b_row_scale

        out = gemm_fp8_impl(
            a_row,
            a_row_scale,
            trans_a,
            b_row,
            b_row_scale,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale)

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.flydsl_blockwise = flydsl_blockwise

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        flydsl_blockwise = getattr(ctx, "flydsl_blockwise", False)
        dgrad_contract = b_col.shape[0]
        pad_dgrad_contract = flydsl_blockwise and ctx.trans_b and dgrad_contract % ctx.config.block_size != 0

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        wgrad_col_transposed = flydsl_blockwise and ctx.trans_b
        if wgrad_col_transposed:
            if (
                grad_out_dtype == float8_e4m3
                and grad_out.dtype == torch.bfloat16
                and not pad_dgrad_contract
                and grad_out.shape[0] % ctx.config.block_size == 0
                and grad_out.shape[1] % ctx.config.block_size == 0
                and grad_out.numel() * grad_out.element_size() <= 0xFFFFFFFF
            ):
                from primus_turbo.flydsl.quantization.blockwise_fp8_quant_poc import (
                    quantize_blockwise_fp8_dual,
                )

                g_row, g_row_scale, g_col, g_col_scale = quantize_blockwise_fp8_dual(
                    grad_out,
                    row_scale_transposed=False,
                )
            else:
                g_row, g_row_scale, g_col, g_col_scale = quant_fp8_blockwise_dual_impl(
                    grad_out,
                    grad_out_dtype,
                    ctx.config.block_size,
                    col_transposed=True,
                    row_pad_to_block=pad_dgrad_contract,
                )
            # Present the transposed producer storage as the logical [M, N]
            # operand expected by dispatch; the FlyDSL launcher transposes it
            # back to its contiguous [N, M] storage without a copy.
            g_col = g_col.transpose(0, 1)
        else:
            (
                g_row,
                g_row_scale,
                g_col,
                g_col_scale,
            ) = quantize_fp8_with_trans(
                grad_out,
                grad_out_dtype,
                ctx.config.granularity,
                block_size=ctx.config.block_size,
            )

        dgrad_b = b_col
        dgrad_b_scale = b_col_scale
        if flydsl_blockwise and ctx.trans_b:
            if pad_dgrad_contract:
                dgrad_m = (dgrad_contract + ctx.config.block_size - 1) // ctx.config.block_size
                dgrad_m *= ctx.config.block_size
                dgrad_b_padded = b_col.new_zeros((dgrad_m, b_col.shape[1]))
                dgrad_b_padded[:dgrad_contract, :] = b_col
                dgrad_b = dgrad_b_padded

        if flydsl_blockwise and not ctx.trans_b:
            from primus_turbo.flydsl.gemm.blockscale_fp8_gemm import (
                select_blockscale_fp8_forward_kernel,
            )

            dgrad_config = select_blockscale_fp8_forward_kernel(
                g_row.shape[0],
                dgrad_b.shape[0],
                g_row.shape[1],
            )
            if dgrad_config["scale_a_k_major"]:
                g_row_scale = g_row_scale.T.contiguous()

        grad_a = gemm_fp8_impl(
            g_row,
            g_row_scale,
            False,
            dgrad_b,
            dgrad_b_scale,
            not ctx.trans_b,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )
        grad_b = gemm_fp8_impl(
            a_col,
            a_col_scale,
            not ctx.trans_a,
            g_col,
            g_col_scale,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
        )


class FP8GemmMXFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        supported_mxfp8_backend, reason = check_mxfp8_support()
        assert supported_mxfp8_backend, reason

        assert trans_a == False and trans_b == True, "trans_a has to be False and trans_b has to be True"

        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)

        # Scale preshuffle is NOT done here: the quant emits raw E8M0 [dim, K//32]
        # scales and each GEMM backend implicitly preshuffles right before its own
        # kernel (FlyDSL fuses it into the gemm launch / turbo 16x4 / hipBLASLt vendor).
        fp8_dtype = _get_fp8_dtype(config.format, True)
        granularity = config.granularity
        block_size = config.block_size

        a_scaling_recipe = ScalingRecipe()
        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, axis=-1, scaling_recipe=a_scaling_recipe)
            a_row, a_row_scale = a.qdata, a.scale_inv
            if a_t is None:
                a_t = QuantizedTensor.quantize(
                    a.dequantize(),
                    a.real_dtype,
                    granularity,
                    axis=-2,
                    block_size=block_size,
                    scaling_recipe=a_scaling_recipe,
                )
            a_col, a_col_scale = a_t.qdata, a_t.scale_inv
        else:
            a_row, a_row_scale, a_col, a_col_scale = quantize_mxfp8_impl(
                a,
                fp8_dtype,
                None,
                block_size,
                with_trans=True,
                scaling_recipe=a_scaling_recipe,
                scaling_recipe_for_trans=a_scaling_recipe,
            )

        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, axis=-1, scaling_recipe=b_scaling_recipe)
            b_row, b_row_scale = b.qdata, b.scale_inv
            if b_t is None:
                b_t = QuantizedTensor.quantize(
                    b.dequantize(),
                    b.real_dtype,
                    granularity,
                    axis=-2,
                    block_size=block_size,
                    scaling_recipe=b_scaling_recipe,
                )
            b_col, b_col_scale = b_t.qdata, b_t.scale_inv
        else:
            b_row, b_row_scale, b_col, b_col_scale = quantize_mxfp8_impl(
                b,
                fp8_dtype,
                None,
                block_size,
                with_trans=True,
                scaling_recipe=b_scaling_recipe,
                scaling_recipe_for_trans=b_scaling_recipe,
            )

        # NT layout
        out = gemm_fp8_impl(
            a_row,
            a_row_scale,
            False,
            b_row,
            b_row_scale,
            True,
            out_dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.FLYDSL.value,
        )

        ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale)
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.fuse_bgrad_accum = fuse_bgrad_accum
        ctx.main_grad = main_grad

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        # reshape (not view): the dual-cast quant kernel asserts a contiguous input;
        # a strided grad_out (e.g. a transpose/slice upstream) would trip it.
        grad_out = grad_out.reshape(grad_out.shape[0], -1).contiguous()

        # Dual-cast grad_out: one kernel emits both the row-wise (grad_a, NN->NT) and
        # col-wise (grad_b, TN->NT) directions -- no second single-direction pass.
        g_row, g_row_scale, g_col, g_col_scale = quantize_mxfp8_impl(
            grad_out,
            grad_out_dtype,
            None,
            ctx.config.block_size,
            with_trans=True,
            scaling_recipe=ScalingRecipe(),
            scaling_recipe_for_trans=ScalingRecipe(),
        )

        grad_a = gemm_fp8_impl(
            g_row,
            g_row_scale,
            False,
            b_col,
            b_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.FLYDSL.value,
        )

        grad_b = _bgrad_gemm_fp8_impl_wrapper(
            g_col,
            g_col_scale,
            False,
            a_col,
            a_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=(
                BackendType.HIPBLASLT.value if ctx.fuse_bgrad_accum else BackendType.FLYDSL.value
            ),
            inplace_add_to_out=ctx.fuse_bgrad_accum,
            out=ctx.main_grad,
        )

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # fuse_bgrad_accum_pattern
        )


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "FP8 GEMM constructs QuantizedTensor wrapper subclasses inside its "
        "autograd.Function.forward and reads their inner tensors (data / scale_inv). "
        "Dynamo cannot recover Python sources for those graph-internal inner tensors, "
    ),
)
def gemm_fp8(
    a: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    b: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float8QuantConfig, None] = None,
    fuse_bgrad_accum_pattern: Union[None, str] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) with FP8 quantization, supporting autograd.

    Automatically quantizes inputs to FP8 format during forward and backward passes
    to accelerate training and inference.

    Args:
        a: Input matrix A with shape (M, K), must be 2D tensor
        b: Input matrix B with shape (K, N) or (N, K), must be 2D tensor
        trans_a: Whether to transpose matrix A
        trans_b: Whether to transpose matrix B, if True B shape is (N, K)
        out_dtype: Output data type, defaults to None (auto-inferred)
        config: FP8 quantization config, defaults to None (uses TENSORWISE + E4M3)
        fuse_bgrad_accum_pattern: Enables fusing the weight-gradient accumulation
            into the wgrad GEMM epilogue, so backward writes ``b.main_grad``
            directly instead of returning a gradient the framework then adds.
            ``"megatron"`` is the only supported pattern; ``b`` must carry
            ``main_grad`` / ``grad_added_to_main_grad``. TENSORWISE and
            MX_BLOCKWISE only. Defaults to None (no fusion).

    Returns:
        torch.Tensor: Output matrix with shape (M, N)

    Scaling Granularity (config.granularity):
        - TENSORWISE
        - ROWWISE
        - BLOCKWISE
        - MX_BLOCKWISE

    FP8 Format (config.format):
        - E4M3
        - E5M2

    Example::

        >>> # Basic usage
        >>> a = torch.randn(128, 512, device='cuda')
        >>> b = torch.randn(512, 256, device='cuda')
        >>> out = gemm_fp8(a, b)
        >>>
        >>> # ROWWISE quantization
        >>> config = Float8QuantConfig(
        ...     format=Format.E4M3,
        ...     granularity=ScalingGranularity.ROWWISE
        ... )
        >>> out = gemm_fp8(a, b, trans_b=True, config=config)

    """
    if config is None:
        config = Float8QuantConfig()

    if isinstance(a, QuantizedTensorPair):
        a_data, a_data_t = a.data, a.data_t
    else:
        a_data, a_data_t = a, None

    if isinstance(b, QuantizedTensorPair):
        b_data, b_data_t = b.data, b.data_t
    else:
        b_data, b_data_t = b, None

    assert a_data.ndim == 2, "Only 2D tensors are supported"
    assert b_data.ndim == 2, "Only 2D tensors are supported"

    if out_dtype is None:
        out_dtype = torch.promote_types(a_data.dtype, b_data.dtype)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return FP8GemmTensorFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config, fuse_bgrad_accum_pattern
        )
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GemmMXFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config, fuse_bgrad_accum_pattern
        )

    # The remaining granularities silently ignore the flag rather than fusing, which
    # would leave main_grad unwritten and the weight without a gradient.
    assert fuse_bgrad_accum_pattern is None, (
        f"fuse_bgrad_accum_pattern is not supported for {config.granularity}"
    )
    if config.granularity == ScalingGranularity.ROWWISE:
        return FP8GemmRowFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GemmBlockFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
