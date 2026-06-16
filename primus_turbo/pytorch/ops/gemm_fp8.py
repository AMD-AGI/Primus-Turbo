###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.flydsl.gemm.mxfp8_gemm_kernel import (
    _mx_pack,
    gemm_mxfp8_flydsl_kernel,
    peek_mxfp8_cfg,
)
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp8_support,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_dual_impl,
    quantize_mxfp8_impl,
)

__all__ = ["gemm_fp8"]


def _flydsl_mxfp8_nt_ok(M: int, N: int, K: int) -> bool:
    """FlyDSL MXFP8 NT kernel shape constraints (mirror GEMMFP8FlyDSLBackend)."""
    return K % 128 == 0 and K >= 256 and M % 64 == 0 and N % 256 == 0 and M * K < 2**31 and N * K < 2**31


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


class FP8GemmTensorFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        trans_a: bool,  # trans_a has to be False
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
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
        ctx.save_for_backward(
            quantized_a.qdata, quantized_a.scale_inv, quantized_b.qdata, quantized_b.scale_inv
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
        a_fp8_data, a_scale_inv, b_fp8_data, b_scale_inv = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
        )

        a_grad = gemm_fp8_impl(
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            b_fp8_data,
            b_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            ctx.trans_a,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        b_grad = gemm_fp8_impl(
            a_fp8_data,
            a_scale_inv,
            not ctx.trans_a,
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        return (a_grad, b_grad, None, None, None, None)


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
            # B's row-wise axis is (-1 if trans_b else -2); the col-wise / trans
            # cache used by backward is the other axis.
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
            default_backend=BackendType.HIPBLASLT.value,
        )

        # a_fp8.qdata = axis=-1 (row-wise), a_fp8.t() = axis=-2 (col-wise / transposed)
        ctx.save_for_backward(
            quantized_a_t.qdata, quantized_a_t.scale_inv, quantized_b_t.qdata, quantized_b_t.scale_inv
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

        # Quantize grad_out row-wise (axis=-1), then transpose to derive col-wise (axis=-2) version.
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
            default_backend=BackendType.HIPBLASLT.value,
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
            default_backend=BackendType.HIPBLASLT.value,
        )

        # Grads correspond to forward args:
        #   (a, b, a_t, b_t, trans_a, trans_b, out_dtype, config)
        return (a_grad, b_grad, None, None, None, None, None, None)


# TODO(ruibin): Add support for quantized tensor
class FP8GemmBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        from primus_turbo.pytorch.ops.quantization import (
            quant_fp8_blockwise_for_weight_impl,
            quant_fp8_blockwise_impl,
        )

        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), "a and b must be torch.Tensor"
        assert trans_a == False, "trans_a has to be False"
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_dtype_bwd = _get_fp8_dtype(config.format, False)

        # When forward and backward share the same FP8 dtype (i.e. non-HYBRID
        # formats), fuse forward row-quant with column-quant of the same
        # activation in a single dual kernel pass and save the column result
        # into ctx so the backward can skip its own column-quant launch.
        # This is a kernel fusion (single tensor `a` is read once instead of
        # twice) and does NOT depend on tensor identity across iterations.
        fuse_a_dual = a_dtype == a_dtype_bwd and a.is_contiguous()

        if fuse_a_dual:
            (
                a_fp8_row,
                a_scale_inv_row,
                a_fp8_col,
                a_scale_inv_col,
            ) = quant_fp8_blockwise_dual_impl(a, a_dtype, config.block_size)
        else:
            a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
                a, a_dtype, axis=1, block_size=config.block_size
            )
            a_fp8_col = None
            a_scale_inv_col = None

        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            trans_a,
            b_fp8,
            b_scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.CK.value,
        )
        if fuse_a_dual:
            ctx.save_for_backward(b_fp8, b_scale_inv, a_fp8_col, a_scale_inv_col)
            ctx.has_prequantized_a_col = True
        else:
            ctx.save_for_backward(a, b_fp8, b_scale_inv)
            ctx.has_prequantized_a_col = False
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        from primus_turbo.pytorch.ops.quantization import quant_fp8_blockwise_impl

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        if ctx.has_prequantized_a_col:
            b_fp8, b_scale_inv, a_fp8_col, a_scale_inv_col = ctx.saved_tensors
        else:
            a, b_fp8, b_scale_inv = ctx.saved_tensors
            a_dtype = _get_fp8_dtype(ctx.config.format, False)
            a_fp8_col, a_scale_inv_col = quant_fp8_blockwise_impl(
                a, a_dtype, axis=0, block_size=ctx.config.block_size
            )
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        (
            grad_out_fp8_row,
            grad_out_scale_inv_row,
            grad_out_fp8_col,
            grad_out_scale_inv_col,
        ) = quant_fp8_blockwise_dual_impl(grad_out, grad_out_dtype, ctx.config.block_size)

        a_grad = gemm_fp8_impl(
            grad_out_fp8_row,
            grad_out_scale_inv_row,
            False,
            b_fp8,
            b_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.CK.value,
        )

        b_grad = gemm_fp8_impl(
            a_fp8_col,
            a_scale_inv_col,
            not ctx.trans_a,
            grad_out_fp8_col,
            grad_out_scale_inv_col,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.CK.value,
        )

        return a_grad, b_grad, None, None, None, None


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
    ):
        supported_mxfp8_backend, reason = check_mxfp8_support()
        assert supported_mxfp8_backend, reason

        assert trans_a == False and trans_b == True, "trans_a has to be False and trans_b has to be True"

        fp8_dtype = _get_fp8_dtype(config.format, True)
        a_recipe = ScalingRecipe()
        b_recipe = ScalingRecipe(use_2d_block=True)

        # TE-style dual cast: produce the row-wise (forward) AND col-wise (transposed,
        # saved for backward) MXFP8 tensors directly from the high-precision input in
        # ONE pass, instead of dequantize() + requantize(axis=-2). This drops the extra
        # dequant/quant kernels and the double-quantization error. The col-wise output
        # orientation matches the old axis=-2 quant, so the NT-fold backward is unchanged.
        def _row_col(x, x_t, recipe):
            if isinstance(x, QuantizedTensor):
                check_quantized_tensor(x, config, axis=-1, scaling_recipe=recipe)
                if x_t is not None:
                    return x.qdata, x.scale_inv, x_t.qdata, x_t.scale_inv
                qt = QuantizedTensor.quantize(
                    x.dequantize(),
                    x.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    scaling_recipe=recipe,
                )
                return x.qdata, x.scale_inv, qt.qdata, qt.scale_inv
            return quantize_mxfp8_impl(x, fp8_dtype, None, config.block_size, True, recipe, recipe)

        # FLYDSL fused-preshuffle fast path: the dual-cast quant emits the scale already
        # in the gemm's preshuffled layout (opsel byte-pack: A=layout 2, B=layout 4;
        # pack==1 -> broadcast 1/3), skipping the host preshuffle that otherwise eats
        # FlyDSL's kernel edge over turbo. Row path -> fwd gemm (cfg-gated for the A
        # fanout); col path (at/bt) -> bwd B-operands (no bm dep) preshuffled here so
        # the backward also skips its (much larger) host preshuffle. Only when the user
        # selects FlyDSL, for dynamic torch.Tensor inputs with supported shapes.
        use_flydsl = (
            GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.FLYDSL
            and not isinstance(a, QuantizedTensor)
            and not isinstance(b, QuantizedTensor)
            and a.dim() == 2
            and b.dim() == 2
            and _flydsl_mxfp8_nt_ok(a.shape[0], b.shape[0], a.shape[1])
        )
        # bwd FlyDSL needs e4m3 grads (the mxfp8 kernel is e4m3-only); HYBRID grads stay
        # turbo, so their col operands (at/bt) must NOT be preshuffled.
        flydsl_bwd = use_flydsl and _get_fp8_dtype(config.format, False) == float8_e4m3

        if use_flydsl:
            M_, K_ = a.shape
            N_ = b.shape[0]
            cfg = peek_mxfp8_cfg(M_, N_, K_, out_dtype, "nt")
            if flydsl_bwd:
                # at = a-col = grad_b's B (contract M); bt = b-col = grad_a's B (contract N)
                at_recipe = ScalingRecipe(preshuffle_layout=(4 if _mx_pack(M_) > 1 else 3))
                bt_recipe = ScalingRecipe(use_2d_block=True, preshuffle_layout=(4 if _mx_pack(N_) > 1 else 3))
            else:
                at_recipe, bt_recipe = a_recipe, b_recipe
            if cfg is not None:
                packed = _mx_pack(K_) > 1
                a_row_layout, b_row_layout = (2, 4) if packed else (1, 3)
                a_qd, a_sc, at_qd, at_sc = quantize_mxfp8_impl(
                    a,
                    fp8_dtype,
                    None,
                    config.block_size,
                    True,
                    ScalingRecipe(preshuffle_layout=a_row_layout, preshuffle_n_tiles=cfg[0] // 64),
                    at_recipe,
                )
                b_qd, b_sc, bt_qd, bt_sc = quantize_mxfp8_impl(
                    b,
                    fp8_dtype,
                    None,
                    config.block_size,
                    True,
                    ScalingRecipe(use_2d_block=True, preshuffle_layout=b_row_layout),
                    bt_recipe,
                )
                out = gemm_mxfp8_flydsl_kernel(
                    a_qd,
                    a_sc,
                    b_qd,
                    b_sc,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=out_dtype,
                    scales_preshuffled=True,
                )
            else:
                # First call: raw row scale (gemm host-preshuffles + autotunes, caching
                # the cfg) while the col operands are already fused for the backward.
                a_qd, a_sc, at_qd, at_sc = quantize_mxfp8_impl(
                    a, fp8_dtype, None, config.block_size, True, a_recipe, at_recipe
                )
                b_qd, b_sc, bt_qd, bt_sc = quantize_mxfp8_impl(
                    b, fp8_dtype, None, config.block_size, True, b_recipe, bt_recipe
                )
                out = gemm_mxfp8_flydsl_kernel(
                    a_qd,
                    a_sc,
                    b_qd,
                    b_sc,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=out_dtype,
                )
        else:
            a_qd, a_sc, at_qd, at_sc = _row_col(a, a_t, a_recipe)
            b_qd, b_sc, bt_qd, bt_sc = _row_col(b, b_t, b_recipe)
            out = gemm_fp8_impl(
                a_qd,
                a_sc,
                False,
                b_qd,
                b_sc,
                True,
                out_dtype,
                False,
                granularity=config.granularity.value,
                default_backend=BackendType.TURBO.value,
            )

        ctx.save_for_backward(at_qd, at_sc, bt_qd, bt_sc)
        ctx.flydsl_bwd = flydsl_bwd

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_fp8_t, a_t_scale_inv, b_fp8_t, b_t_scale_inv = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out = grad_out.view(grad_out.shape[0], -1)

        # FLYDSL fused backward: the saved col operands (at/bt = B in the grad GEMMs)
        # were already preshuffled at fwd-time (no BLOCK_M dep), and grad_out's row/col
        # dual-cast emits the A-scale preshuffled here -> both grad GEMMs skip the host
        # preshuffle, which in bwd is ~4x fwd's (large contract dim -> big scale).
        #   grad_a = NT(go_row[M,N], bt[K,N]) -> [M,K], cfg key (M, K, N)
        #   grad_b = NT(go_col[N,M], at[K,M]) -> [N,K], cfg key (N, K, M)
        if getattr(ctx, "flydsl_bwd", False):
            M = grad_out.shape[0]
            N = grad_out.shape[1]
            K = b_fp8_t.shape[0]
            cfg_a = peek_mxfp8_cfg(M, K, N, ctx.out_dtype, "nt")
            cfg_b = peek_mxfp8_cfg(N, K, M, ctx.out_dtype, "nt")
            if cfg_a is not None and cfg_b is not None:
                # go_row = grad_a's A (contract N); go_col = grad_b's A (contract M)
                go_row, go_row_sc, go_col, go_col_sc = quantize_mxfp8_impl(
                    grad_out,
                    grad_out_dtype,
                    None,
                    ctx.config.block_size,
                    True,
                    ScalingRecipe(
                        preshuffle_layout=(2 if _mx_pack(N) > 1 else 1),
                        preshuffle_n_tiles=cfg_a[0] // 64,
                    ),
                    ScalingRecipe(
                        preshuffle_layout=(2 if _mx_pack(M) > 1 else 1),
                        preshuffle_n_tiles=cfg_b[0] // 64,
                    ),
                )
                grad_a = gemm_mxfp8_flydsl_kernel(
                    go_row,
                    go_row_sc,
                    b_fp8_t,
                    b_t_scale_inv,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=ctx.out_dtype,
                    scales_preshuffled=True,
                )
                grad_b = gemm_mxfp8_flydsl_kernel(
                    go_col,
                    go_col_sc,
                    a_fp8_t,
                    a_t_scale_inv,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=ctx.out_dtype,
                    scales_preshuffled=True,
                )
            else:
                # First bwd: raw grad_out (gemm host-preshuffles A + autotunes -> caches
                # cfg) with the already-preshuffled B operands (at/bt).
                recipe = ScalingRecipe()
                go_row, go_row_sc, go_col, go_col_sc = quantize_mxfp8_impl(
                    grad_out, grad_out_dtype, None, ctx.config.block_size, True, recipe, recipe
                )
                grad_a = gemm_mxfp8_flydsl_kernel(
                    go_row,
                    go_row_sc,
                    b_fp8_t,
                    b_t_scale_inv,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=ctx.out_dtype,
                    b_scales_preshuffled=True,
                )
                grad_b = gemm_mxfp8_flydsl_kernel(
                    go_col,
                    go_col_sc,
                    a_fp8_t,
                    a_t_scale_inv,
                    trans_a=False,
                    trans_b=True,
                    out_dtype=ctx.out_dtype,
                    b_scales_preshuffled=True,
                )
            return grad_a, grad_b, None, None, None, None, None, None

        # TE-style dual cast of grad_out: row-wise (for grad_a) + col-wise (for grad_b)
        # MXFP8 in ONE pass from the high-precision grad, instead of two separate quants.
        recipe = ScalingRecipe()
        go_row, go_row_sc, go_col, go_col_sc = quantize_mxfp8_impl(
            grad_out, grad_out_dtype, None, ctx.config.block_size, True, recipe, recipe
        )

        # NT-fold (MXFP8 dense is NT-only): grad_a = NN->NT, grad_b = TN->NT.
        grad_a = gemm_fp8_impl(
            go_row,
            go_row_sc,
            False,
            b_fp8_t,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        grad_b = gemm_fp8_impl(
            go_col,
            go_col_sc,
            False,
            a_fp8_t,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        # Grads correspond to forward args:
        #   (a, b, a_t, b_t, trans_a, trans_b, out_dtype, config)
        return grad_a, grad_b, None, None, None, None, None, None


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
        return FP8GemmTensorFunction.apply(a_data, b_data, trans_a, trans_b, out_dtype, config)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GemmRowFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        # BLOCKWISE does not yet support pre-quantized inputs; preserve the
        # existing assertion behaviour in ``FP8GemmBlockFunction.forward``.
        return FP8GemmBlockFunction.apply(a, b, trans_a, trans_b, out_dtype, config)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GemmMXFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
