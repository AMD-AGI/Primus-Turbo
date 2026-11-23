###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core import fp8blk_weight_cache
from primus_turbo.pytorch.core import overlap as _overlap
from primus_turbo.pytorch.core.backend import (
    BackendType,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp8_support,
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
from primus_turbo.pytorch.kernels.quantization.quantization_impl import quantize_mxfp8_impl
from primus_turbo.pytorch.ops.quantization import (
    quantize_fp8,
    quantize_fp8_with_trans,
)
from primus_turbo.pytorch.ops.utils import (
    _get_dummy_wgrad,
    _get_fp8_dtype,
    _setup_fused_grad_accum,
    main_grad_first_write,
)

__all__ = ["gemm_fp8"]

# Linear sites ("f<K>", K = the activation width) whose forward keeps a separate weight-scale pack launch
# instead of packing inside the activation quant launch.
_PACK_IN_QUANT_SKIP = frozenset({"f4096"})


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
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        assert trans_a == False, "trans_a has to be False"
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        # Fused wgrad accumulation (g90 path: the kernel's beta=1 epilogue reads/writes main_grad, or beta=0 on the
        # step's first write; other paths: GEMM then one add / copy into main_grad). The weight object is kept for
        # the first-write policy, which is asked in backward (actual write order).
        ctx.fuse_bgrad_accum, ctx.main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)
        ctx.wparam = b if ctx.fuse_bgrad_accum else None

        # gfx1250 dense path (PRIMUS_TURBO_FP8_BLOCK_KERNEL = dsv4up or g90): all three passes as NT GEMMs,
        # 1x128 quant + token-major transposed copy in one pass, E8M0 scale bytes straight from the quant.
        ctx.g90 = False
        from primus_turbo.pytorch.kernels.gemm import gemm_fp8_flydsl_gfx1250_impl

        if (
            gemm_fp8_flydsl_gfx1250_impl.enabled()
            and not isinstance(a, QuantizedTensor)
            and not isinstance(b, QuantizedTensor)
            and a.dim() == 2
            and trans_b
            and a.shape[0] % 512 == 0
            and a.shape[1] % 512 == 0
            and b.shape[0] % 512 == 0
            and a_dtype == torch.float8_e4m3fn
            and _get_fp8_dtype(config.format, False) == torch.float8_e4m3fn
            and envs.PRIMUS_TURBO_FP8_BLOCK_UE8M0  # weight scales must be powers of two
        ):
            from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl
            from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel as _qg

            a = a.contiguous()
            plan = None
            if gemm_fp8_gluon_gfx1250_impl.enabled():
                M_, K_ = a.shape
                N_ = b.shape[0]
                plan = (
                    gemm_fp8_gluon_gfx1250_impl.use(M_, N_, K_, "fwd"),
                    gemm_fp8_gluon_gfx1250_impl.use(M_, K_, N_, "dgrad"),
                    gemm_fp8_gluon_gfx1250_impl.use(N_, K_, M_, "wgrad"),
                )
            # With a ready weight cache the weight-scale packing of the dsv4up passes runs inside the
            # activation quant launch (same bytes, one launch less per Linear forward).
            mbpk = None
            if plan is not None and f"f{K_}" not in _PACK_IN_QUANT_SKIP:
                wc = fp8blk_weight_cache.lookup(b, b_dtype)
                if wc is not None:
                    (a_row, _, a_re, a_t, a_te), mbpk = _qg.quant_dual_pack(
                        a,
                        a_dtype,
                        "packed" if plan[0] else "g90",
                        "packed" if plan[2] else "g90",
                        [
                            (wc.e8m0, 128, N_, K_) if plan[0] else None,
                            (wc.e8m0.t(), 128, K_, N_) if plan[1] else None,
                        ],
                        site="f",
                    )
            if mbpk is None:
                # The quant writes each activation scale in its consumer's layout (dsv4up packed for the dsv4up
                # fwd / wgrad passes, g90 bytes otherwise): no activation pack_scales job.
                a_row, _, a_re, a_t, a_te = _qg.quant_dual_t_e8m0_v2(
                    a,
                    a_dtype,
                    row_layout="packed" if plan and plan[0] else "g90",
                    t_layout="packed" if plan and plan[2] else "g90",
                    site="f",
                )
            # W, W^T and E8M0 scales written by the optimizer step (FP8 weight cache), else quantized here.
            wc = fp8blk_weight_cache.lookup(b, b_dtype)
            if wc is not None:
                b_row, b_e, b_t, b_te = wc.w, wc.e8m0, wc.wt, wc.e8m0_t
            else:
                b_row, b_row_scale = quantize_fp8(
                    b,
                    b_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    scaling_recipe=ScalingRecipe(use_2d_block=True),
                )
                b_e = _qg.e8m0_of(b_row_scale)
                b_t, b_te = None, None
            # dsv4up (Gluon) per pass where the shape allows, else g90. The weight scales of every dsv4up pass that
            # only needs forward-time scales are packed in one launch here.
            ctx.dsv = None
            if plan is not None:
                if mbpk is not None:  # weight scales packed by the quant launch
                    pb, pbt = mbpk
                else:
                    _, pb, pbt, _ = gemm_fp8_gluon_gfx1250_impl.pack_scales(
                        [
                            None,
                            (b_e, 128, N_, K_) if plan[0] else None,
                            (b_e.t(), 128, K_, N_) if plan[1] else None,  # transposed view, no copy
                            None,
                        ]
                    )
                pa = a_re if plan[0] else None  # activation scales are already packed by the quant
                pat = a_te if plan[2] else None
                ctx.dsv = (plan, pbt, pat)
                if plan[0]:
                    out = gemm_fp8_gluon_gfx1250_impl.gemm_nt(a_row, b_row, pa, pb, out_dtype)
                else:
                    out = gemm_fp8_flydsl_gfx1250_impl.gemm_nt(a_row, b_row, a_re, b_e, out_dtype)
            else:
                out = gemm_fp8_flydsl_gfx1250_impl.gemm_nt(a_row, b_row, a_re, b_e, out_dtype)
            ctx.save_for_backward(a_t, a_te, b_row, b_e, b_t, b_te)
            ctx.g90 = True
            ctx.trans_a, ctx.trans_b, ctx.out_dtype, ctx.config = trans_a, trans_b, out_dtype, config
            return out

        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, axis=-1)
            a_row, a_row_scale = a.qdata, a.scale_inv
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
            (
                a_row,
                a_row_scale,
                a_col,
                a_col_scale,
            ) = quantize_fp8_with_trans(a, a_dtype, config.granularity, block_size=config.block_size)

        # --- B side: 2D-block weight, reused unchanged in fwd + bwd. ---
        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        wc = None if isinstance(b, QuantizedTensor) else fp8blk_weight_cache.lookup(b, b_dtype)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, scaling_recipe=b_scaling_recipe)
            b_row, b_row_scale = b.qdata, b.scale_inv
        elif wc is not None:  # quantized by the optimizer step (FP8 weight cache)
            b_row, b_row_scale = wc.w, wc.scale_inv
        else:
            b_row, b_row_scale = quantize_fp8(
                b, b_dtype, config.granularity, block_size=config.block_size, scaling_recipe=b_scaling_recipe
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

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.g90:
            from primus_turbo.pytorch.kernels.gemm import gemm_fp8_flydsl_gfx1250_impl
            from primus_turbo.triton.quantization.quantization_fp8blk_dual_kernel import quant_dual_t_e8m0_v2

            a_t, a_te, b_row, b_e, b_t, b_te = ctx.saved_tensors
            g_dtype = _get_fp8_dtype(ctx.config.format, False)
            pd, pw = ctx.dsv[0][1:] if ctx.dsv is not None else (False, False)
            g_row, _, g_re, g_t, g_te = quant_dual_t_e8m0_v2(
                grad_out.contiguous(),
                g_dtype,
                row_layout="packed" if pd else "g90",
                t_layout="packed" if pw else "g90",
                site="b",
            )
            # Per-pass plan made in forward. The dY scales come out of the quant in the dsv4up layout.
            use_d = use_w = False
            if ctx.dsv is not None:
                from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl

                (_, use_d, use_w), pbt, pat = ctx.dsv
                ctx.dsv = None
                pg, pgt = (g_re if use_d else None), (g_te if use_w else None)

            def wgemm(out_dtype, out=None, accumulate=False):
                if use_w:
                    return gemm_fp8_gluon_gfx1250_impl.gemm_nt(
                        g_t, a_t, pgt, pat, out_dtype, out=out, accumulate=accumulate
                    )
                return gemm_fp8_flydsl_gfx1250_impl.gemm_nt(
                    g_t, a_t, g_te, a_te, out_dtype, out=out, accumulate=accumulate
                )

            if (
                b_t is None
            ):  # no optimizer-written W^T: transpose the FP8 copy (128x128 blocks are transpose-invariant)
                b_t, b_te = b_row.t().contiguous(), b_e.t().contiguous()

            def fused_wgrad():
                # wgrad: dW[N,K] = dY^T . X  ->  NT on the token-major copies, straight into main_grad
                mg = ctx.main_grad
                overwrite = main_grad_first_write(ctx.wparam)
                if gemm_fp8_flydsl_gfx1250_impl.accum_supported(mg, b_row.shape[0], a_t.shape[0]) and not (
                    overwrite and mg.dtype != ctx.out_dtype
                ):
                    # beta 0: plain epilogue straight into main_grad (= the unfused GEMM's output bits);
                    # beta 1: old main_grad + acc in fp32, rounded once into main_grad's dtype
                    run = lambda: wgemm(mg.dtype, out=mg, accumulate=not overwrite)  # noqa: E731
                else:

                    def run():
                        gb = wgemm(ctx.out_dtype)
                        mg.copy_(gb) if overwrite else mg.add_(gb)

                ctx.main_grad = ctx.wparam = None
                return mg, run

            # With overlap on, the fused wgrad goes to the side stream first (it only feeds the optimizer), so
            # it runs next to the dgrad chain; same kernel, same inputs -> same main_grad bits.
            deferred = None
            if ctx.fuse_bgrad_accum and _overlap.enabled("WGRAD"):
                deferred, run = fused_wgrad()
                _overlap.defer(
                    run, deferred, g_t, a_t, g_te, a_te, pgt if use_w else None, pat if use_w else None
                )
            # dgrad: dX = dY . W  ->  NT with the FP8 copy of W^T (block scales transposed)
            if use_d:
                grad_a = gemm_fp8_gluon_gfx1250_impl.gemm_nt(g_row, b_t, pg, pbt, ctx.out_dtype)
            else:
                grad_a = gemm_fp8_flydsl_gfx1250_impl.gemm_nt(g_row, b_t, g_re, b_te, ctx.out_dtype)
            if deferred is not None:
                grad_b = _get_dummy_wgrad(list(deferred.shape), ctx.out_dtype)
            elif ctx.fuse_bgrad_accum:
                mg, run = fused_wgrad()
                run()
                grad_b = _get_dummy_wgrad(list(mg.shape), ctx.out_dtype)
            else:
                grad_b = wgemm(ctx.out_dtype)
            return grad_a, grad_b, None, None, None, None, None, None, None

        a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out = grad_out.contiguous()

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        (
            g_row,
            g_row_scale,
            g_col,
            g_col_scale,
        ) = quantize_fp8_with_trans(
            grad_out, grad_out_dtype, ctx.config.granularity, block_size=ctx.config.block_size
        )

        grad_a = gemm_fp8_impl(
            g_row,
            g_row_scale,
            False,
            b_col,
            b_col_scale,
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
        if ctx.fuse_bgrad_accum:
            # no BLOCKWISE Triton kernel has a beta=1 epilogue: one add (or copy on the step's first write)
            mg = ctx.main_grad
            mg.copy_(grad_b) if main_grad_first_write(ctx.wparam) else mg.add_(grad_b)
            ctx.main_grad = ctx.wparam = None
            grad_b = _get_dummy_wgrad(list(mg.shape), ctx.out_dtype)

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

    if config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GemmBlockFunction.apply(
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
