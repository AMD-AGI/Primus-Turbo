###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Optional, Union

import torch

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
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_accum_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_segment_m_row_col_impl,
)
from primus_turbo.pytorch.ops.quantization import (
    grouped_quantize_fp8_with_trans,
    quantize_fp8_with_trans,
)
from primus_turbo.pytorch.ops.utils import (
    _ensure_contiguous_grad_out,
    _get_dummy_wgrad,
    _get_fp8_dtype,
    _setup_fused_grad_accum,
)

__all__ = [
    "grouped_gemm_fp8",
]


def _grouped_gemm_fp8_variable_k_impl_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: Optional[int],
    default_backend: int,
    inplace_add_to_out: bool = False,
    out: Optional[torch.Tensor] = None,
    m_real: Optional[int] = None,
    n_real: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Run the variable-K wgrad GEMM, accumulating into ``out`` when asked. Returns the
    weight grad, or a dummy buffer (weight dtype, never read) when it accumulated in
    place -- forward flagged the weight so the framework's own accum stands down.
    """
    inputs = (a, b, a_scales, b_scales, group_lens, group_offs)
    options = dict(
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        out_dtype=out_dtype,
        granularity=granularity,
        num_cu=num_cu,
        default_backend=default_backend,
    )
    tight = m_real is not None or n_real is not None

    if not inplace_add_to_out:
        return grouped_gemm_fp8_variable_k_impl(*inputs, m_real=m_real, n_real=n_real, **options)

    assert out is not None, "out should not be None when inplace_add_to_out is True"

    if tight:
        # tight output + beta=1 is not a supported kernel path: emit a fresh tight
        # grad at beta=0 and fold it into the accumulation buffer on the host.
        grad = grouped_gemm_fp8_variable_k_impl(*inputs, m_real=m_real, n_real=n_real, **options)
        out.add_(grad)
        return _get_dummy_wgrad(out.shape, out_dtype)

    grouped_gemm_fp8_variable_k_accum_impl(*inputs, out=out, **options)

    return _get_dummy_wgrad(out.shape, out_dtype)


class FP8GroupedGemmBlockFunc(torch.autograd.Function):
    """BLOCKWISE grouped GEMM autograd"""

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],  # not used
        b_t: Optional[QuantizedTensor],  # not used
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)

        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size in [128], "Only block_size 128 is supported currently."
        if isinstance(a, QuantizedTensor):
            # TODO(ruibin): grouped BLOCKWISE emits fused pre-shuffled row / segment-padded col scales, from the quant kernel, it is not compatible with the QuantizedTensor."
            raise NotImplementedError(
                "FP8GroupedGemmBlockFunc does not support a pre-quantized activation `a`"
            )
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        assert group_lens.size(0) == b.size(0), "group_lens size must match b size(0)."
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)

        # --- A side: fused row + segment-padded col grouped quant in one bf16 read.
        gemm_n = b.size(-2) if trans_b else b.size(-1)
        a_fp8_row, a_fp8_col, a_scale_row, a_scale_col, _, _ = quant_fp8_blockwise_segment_m_row_col_impl(
            a, a_dtype, config.block_size, group_lens, group_offs, gemm_other_dim=gemm_n
        )

        # --- B side: 2D-block weight, reused unchanged in fwd + bwd. If the caller
        # pre-quantized it as a QuantizedTensor, reuse its buffers directly. ---
        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, scaling_recipe=b_scaling_recipe)
            b_fp8, b_scale = b.qdata, b.scale_inv
        else:
            b_fp8, b_scale = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8,
            a_scale_row,
            b_scale,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
        )

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_col,
            b_fp8,
            b_scale,
            group_lens,
            group_offs,
        )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.fuse_bgrad_accum = fuse_bgrad_accum
        ctx.main_grad = main_grad

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)

        (
            a_fp8_col,
            a_scale_col,
            b_fp8,
            b_scale,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors
        block_size = ctx.config.block_size
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # --- grad_out: fused row + segment-padded col grouped quant in one bf16 read.
        gemm_k = b_fp8.size(-1) if ctx.trans_b else b_fp8.size(-2)
        (
            grad_out_fp8_row,
            grad_out_fp8_col,
            grad_out_scale_row,
            grad_out_scale_col,
            var_k_group_lens,
            var_k_group_offs,
        ) = quant_fp8_blockwise_segment_m_row_col_impl(
            grad_out, grad_out_dtype, block_size, group_lens, group_offs, gemm_other_dim=gemm_k
        )

        # grad_a: grad_out @ b^T
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8,
            grad_out_scale_row,
            b_scale,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        grad_b = _grouped_gemm_fp8_variable_k_impl_wrapper(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_col,
            grad_out_scale_col,
            var_k_group_lens,
            var_k_group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
            inplace_add_to_out=ctx.fuse_bgrad_accum,
            out=ctx.main_grad,
        )

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
            None,  # fuse_bgrad_accum_pattern
        )


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
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)

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
        ctx.fuse_bgrad_accum = fuse_bgrad_accum
        ctx.main_grad = main_grad
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

        grad_b = _grouped_gemm_fp8_variable_k_impl_wrapper(
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
            inplace_add_to_out=ctx.fuse_bgrad_accum,
            out=ctx.main_grad,
        )

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
            None,  # fuse_bgrad_accum_pattern
        )


class FP8GroupedGemmTensorFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],  # not used
        b_t: Optional[QuantizedTensor],  # not used
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)

        assert config.granularity == ScalingGranularity.TENSORWISE

        # Raw-tensor NT path pads both operands' K to Kp=ceil128(K) and the weight's HIDDEN=N to
        # Np=ceil128(N) at cast time; pad rows/cols are exact zero so the real extents come back
        # untouched (copy-free) and n_real recovers the real output width inside the backend.
        raw_nt = (
            trans_b
            and not isinstance(a, QuantizedTensor)
            and not isinstance(b, QuantizedTensor)
            and a.dim() == 2
            and b.dim() == 3
        )
        if raw_nt:
            from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
                quantize_fp8_tensorwise_pad_impl,
            )

            go = group_offs if group_offs is not None else group_offs_from_lens(group_lens)
            fp8_dtype = _get_fp8_dtype(config.format, True)
            k_real = a.shape[-1]  # real contraction K before the K-pad
            n_real = b.shape[1]  # real HIDDEN=N before the N-pad
            # Request a tight output only when the pad actually shrinks a dim; an aligned shape is a
            # bit-exact no-op and stays a plain generic GEMM (every backend eligible, CK/hbl included).
            tight = (k_real % 128 != 0) or (n_real % 128 != 0)
            a_q, a_sc = quantize_fp8_tensorwise_pad_impl(a, fp8_dtype)  # [M, Kp]
            b_q, b_sc = quantize_fp8_tensorwise_pad_impl(b, fp8_dtype, pad_n=True)  # [G, Np, Kp]
            out = grouped_gemm_fp8_impl(
                a_q,
                b_q,
                a_sc,
                b_sc,
                group_lens,
                go,
                trans_a=False,
                trans_b=True,
                out_dtype=out_dtype,
                granularity=config.granularity.value,
                num_cu=num_cu,
                default_backend=(BackendType.FLYDSL.value if tight else BackendType.TRITON.value),
                maybe_pre_sync=True,
                n_real=(n_real if tight else None),
            )  # [M, N]
            ctx.save_for_backward(a_q, b_q, a_sc, b_sc, group_lens, go)
            ctx.trans_a = False
            ctx.trans_b = trans_b
            ctx.config = config
            ctx.out_dtype = out_dtype
            ctx.num_cu = num_cu
            ctx.k_real = k_real
            # This path returns early, so it sets the grad-accum context the shared backward reads.
            ctx.fuse_bgrad_accum = fuse_bgrad_accum
            ctx.main_grad = main_grad
            ctx.n_real = n_real
            ctx.tight = tight
            return out

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
        ctx.fuse_bgrad_accum = fuse_bgrad_accum
        # Kept off save_for_backward on purpose: the wgrad GEMM writes into this
        # buffer in place, which would bump the version counter that saved tensors
        # are checked against.
        ctx.main_grad = main_grad

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors
        # Set by the raw-tensor NT forward (None on the pre-quantized path); ``tight`` only when the
        # pad actually grew a dim.
        k_real = getattr(ctx, "k_real", None)  # real K = dgrad output free dim
        n_real = getattr(ctx, "n_real", None)  # real N = dgrad contraction / wgrad OUT_M
        raw = k_real is not None
        tight = getattr(ctx, "tight", False)
        default_backend = BackendType.FLYDSL.value if tight else BackendType.TRITON.value

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        if raw:
            from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
                quantize_fp8_tensorwise_pad_impl,
            )

            # dgrad NN contracts grad_out's last dim (=N) against b_fp8's penultimate dim (=Np), so
            # grad_out is widened N->Np inside the quant kernel (k_align=128); the zero pad adds
            # exact 0, no F.pad / host slice.
            go_qdata, go_scale_inv = quantize_fp8_tensorwise_pad_impl(grad_out, grad_out_dtype, k_align=128)
            assert go_qdata.shape[-1] == b_fp8.shape[-2]
        else:
            quantized_grad_out = QuantizedTensor.quantize(
                grad_out,
                grad_out_dtype,
                ctx.config.granularity,
                axis=-1,
                block_size=ctx.config.block_size,
                group_lens=group_lens,
            )
            go_qdata, go_scale_inv = quantized_grad_out.qdata, quantized_grad_out.scale_inv

        # dgrad NN: contract N; n_real=k_real shrinks the output free dim Kp->K so grad_a is tight.
        grad_a = grouped_gemm_fp8_impl(
            go_qdata,
            b_fp8,
            go_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            n_real=(k_real if tight else None),
        )

        # wgrad TN variable-K: m_real=N / n_real=K shrink the [G, Np, Kp] output to a tight
        # [G, N, K]. Fused accum is folded inside the wrapper (tight + beta=1 has no kernel path).
        grad_b = _grouped_gemm_fp8_variable_k_impl_wrapper(
            a_fp8,
            go_qdata,
            a_scale_inv,
            go_scale_inv,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            inplace_add_to_out=ctx.fuse_bgrad_accum,
            out=ctx.main_grad,
            m_real=(n_real if tight else None),
            n_real=(k_real if tight else None),
        )

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
            None,  # fuse_bgrad_accum_pattern
        )


class FP8GroupedGemmMXFunc(torch.autograd.Function):
    """MXFP8 grouped GEMM autograd (MX_BLOCKWISE), Triton backend.

    Same interface as the hip path; only the backend differs
    (default_backend=TRITON).  A / grad_out use grouped dual-quant (padded
    per-group M, dense E8M0 scale); B uses per-group dual-quant.  fwd / dgrad
    read the padded layout (group_offs_padded_rowwise); the output is
    over-allocated to the padded rows, group_offs_out packs each group tight,
    and the caller slices [:total_m].  wgrad output (G, N, K) is
    padding-independent.  When hip MX kernels land, only default_backend
    needs to flip to TURBO.
    """

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
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)

        supported_mxfp8_backend, reason = check_mxfp8_support()
        assert supported_mxfp8_backend, reason

        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2 and b.ndim == 3
        assert trans_b, "MXFP8 grouped GEMM only supports trans_b=True (NT layout)."

        a_dtype = b_dtype = _get_fp8_dtype(config.format, True)

        a_scaling_recipe = ScalingRecipe()
        if not isinstance(a, QuantizedTensor):
            # NOTE: If a is not a QuantizedTensor use grouped_quantize_fp8_with_trans to avoid call dequantize.
            (
                a_fp8_row,
                a_scale_row,
                a_fp8_col,
                a_scale_col,
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
                scaling_recipe=a_scaling_recipe,
                scaling_recipe_for_trans=a_scaling_recipe,
            )
        else:
            quantized_a = a
            check_quantized_tensor(quantized_a, config, axis=-1, scaling_recipe=a_scaling_recipe)

            if a_t is None:
                quantized_a_t = QuantizedTensor.quantize(
                    quantized_a.dequantize(),
                    quantized_a.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    scaling_recipe=a_scaling_recipe,
                    group_lens=group_lens,
                )
            else:
                assert isinstance(a_t, QuantizedTensor)
                quantized_a_t = a_t

            a_fp8_row = quantized_a.qdata
            a_scale_row = quantized_a.scale_inv
            a_fp8_col = quantized_a_t.qdata
            a_scale_col = quantized_a_t.scale_inv

            group_offs_padded_rowwise = quantized_a.group_offs

        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if not isinstance(b, QuantizedTensor):
            # NOTE: If b is not a QuantizedTensor use quantize_fp8_with_trans to avoid call dequantize.

            b_fp8_row, b_scale_row, b_fp8_col, b_scale_col = quantize_fp8_with_trans(
                b,
                b_dtype,
                config.granularity,
                block_size=config.block_size,
                scaling_recipe=ScalingRecipe(use_2d_block=True),
                scaling_recipe_for_trans=ScalingRecipe(use_2d_block=True),
            )
        else:
            quantized_b = b
            check_quantized_tensor(quantized_b, config, axis=-1, scaling_recipe=b_scaling_recipe)

            if b_t is None:
                quantized_b_t = QuantizedTensor.quantize(
                    quantized_b.dequantize(),
                    quantized_b.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    scaling_recipe=b_scaling_recipe,
                )
            else:
                assert isinstance(b_t, QuantizedTensor)
                quantized_b_t = b_t

            b_fp8_row = quantized_b.qdata
            b_scale_row = quantized_b.scale_inv
            b_fp8_col = quantized_b_t.qdata
            b_scale_col = quantized_b_t.scale_inv

        total_m = int(a.size(0))
        # fwd: read rowwise-padded layout (group_offs_padded_rowwise); the output
        # is over-allocated to the padded rows, group_offs_out packs each group
        # tight, then slice [:total_m] back to the user-visible shape.
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_row,
            b_scale_row,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )
        out = out[:total_m]

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_col,
            b_fp8_col,
            b_scale_col,
            group_lens,
            group_offs,
        )
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.total_m = total_m
        ctx.fuse_bgrad_accum = fuse_bgrad_accum
        ctx.main_grad = main_grad
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (a_fp8_col, a_scale_col, b_fp8_col, b_scale_col, group_lens, group_offs) = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        (
            grad_out_fp8_row,
            grad_out_scale_row,
            grad_out_t_fp8,
            grad_out_t_scale,
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

        # dgrad: grad_a = grad_out @ b_col^T  (same single NT op as fwd)
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_row,
            b_scale_col,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )
        grad_a = grad_a[: ctx.total_m]

        # wgrad: grad_b[g] = grad_out_col[g] @ a_col[g]^T  (variable-K over colwise-128 M_g)
        # FlyDSL is the default for the ordinary path, where it is faster. Its beta=1
        # epilogue only writes 16-bit, so the fused path defaults to Triton, which also
        # covers the fp32 main_grad Megatron allocates by default; pin FlyDSL through
        # GlobalBackendManager when main_grad matches the weight's own bf16/fp16 dtype.
        grad_b = _grouped_gemm_fp8_variable_k_impl_wrapper(
            grad_out_t_fp8,
            a_fp8_col,
            grad_out_t_scale,
            a_scale_col,
            group_lens_padded_colwise,
            group_offs_padded_colwise,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=(BackendType.TRITON.value if ctx.fuse_bgrad_accum else BackendType.FLYDSL.value),
            inplace_add_to_out=ctx.fuse_bgrad_accum,
            out=ctx.main_grad,
        )
        # NT-only: wgrad already produces grad_b as (G, N, K) matching b.
        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
            None,  # fuse_bgrad_accum_pattern
        )


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
    fuse_bgrad_accum_pattern: Union[None, str] = None,
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
        num_cu: Cap on the number of compute units the grouped GEMM may use
            (limits the persistent-kernel grid). If None, uses all CUs on the device.

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
            fuse_bgrad_accum_pattern,
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
            fuse_bgrad_accum_pattern,
        )
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        # BLOCKWISE accepts a pre-quantized 2D-block weight (``b``); the activation
        # ``a`` must stay a raw tensor (fused pre-shuffled quant). ``a_data_t`` /
        # ``b_data_t`` are unused by ``FP8GroupedGemmBlockFunc``.
        return FP8GroupedGemmBlockFunc.apply(
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
            fuse_bgrad_accum_pattern,
        )
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GroupedGemmMXFunc.apply(
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
            fuse_bgrad_accum_pattern,
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
