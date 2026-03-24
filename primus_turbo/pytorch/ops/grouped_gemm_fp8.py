###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import os
import weakref
from typing import Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    _BlockwiseBufferCache,
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_cached,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_cached,
    quant_fp8_blockwise_segment_m_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.triton.quantization.quant_fp8_tensorwise import (
    quantize_fp8_tensorwise as _quant_fp8_tw,
    _BufferCache,
)

__all__ = [
    "grouped_gemm_fp8",
]

_tw_cache_a = _BufferCache()
_tw_cache_b = _BufferCache()
_tw_cache_grad = _BufferCache()
_bw_cache_a_row = _BlockwiseBufferCache()
_bw_cache_grad_row = _BlockwiseBufferCache()
_bw_cache_a_seg = _BlockwiseBufferCache()
_bw_cache_grad_seg = _BlockwiseBufferCache()
_bw_aux_streams: dict[int, torch.cuda.Stream] = {}
_tw_weight_cache: dict[tuple[int, int, int, tuple[int, ...], tuple[int, ...], torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
_bw_weight_cache: dict[
    int,
    tuple[
        weakref.ReferenceType[torch.Tensor],
        tuple[int, int, tuple[int, ...], tuple[int, ...], int],
        torch.Tensor,
        torch.Tensor,
    ],
] = {}
_group_offs_cache: dict[
    int,
    tuple[
        weakref.ReferenceType[torch.Tensor],
        tuple[int, int, tuple[int, ...]],
        torch.Tensor,
    ],
] = {}


def _use_triton_grouped_gemm_backend() -> bool:
    return os.environ.get("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "").upper() == "TRITON"


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


def _get_blockwise_weight_cache_key(
    w: torch.Tensor,
    block_size: int,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int]:
    return (
        int(w.data_ptr()),
        int(getattr(w, "_version", 0)),
        tuple(w.shape),
        tuple(w.stride()),
        int(block_size),
    )


def _get_tensorwise_weight_cache_key(
    w: torch.Tensor,
) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...], torch.dtype]:
    return (
        id(w),
        int(w.data_ptr()),
        int(getattr(w, "_version", 0)),
        tuple(w.shape),
        tuple(w.stride()),
        w.dtype,
    )


def _tensorwise_weight_reuse_enabled() -> bool:
    return os.environ.get("PRIMUS_TURBO_FP8_TW_REUSE_B", "0") == "1"


def _get_group_lens_cache_key(group_lens: torch.Tensor) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(group_lens.data_ptr()),
        int(getattr(group_lens, "_version", 0)),
        tuple(group_lens.shape),
    )


def _get_cached_group_offs(group_lens: torch.Tensor) -> torch.Tensor:
    cache_key = _get_group_lens_cache_key(group_lens)
    cached_entry = _group_offs_cache.get(id(group_lens))
    if (
        cached_entry is not None
        and cached_entry[0]() is group_lens
        and cached_entry[1] == cache_key
    ):
        return cached_entry[2]

    group_offs = grouped_gemm_compute_offs(group_lens)
    if len(_group_offs_cache) >= 64:
        _group_offs_cache.clear()
    _group_offs_cache[id(group_lens)] = (weakref.ref(group_lens), cache_key, group_offs)
    return group_offs


def _get_blockwise_aux_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _bw_aux_streams.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream()
        _bw_aux_streams[device_index] = stream
    return stream


def _should_overlap_blockwise_bwd(
    a: torch.Tensor,
    needs_grad_a: bool,
    needs_grad_b: bool,
) -> bool:
    if os.environ.get("PRIMUS_TURBO_BW_BWD_OVERLAP", "1") == "0":
        return False
    # Overlap only helps when both backward branches run and K is large enough to
    # hide the segment-padding quantization behind dgrad GEMM.
    return (
        needs_grad_a
        and needs_grad_b
        and a.is_cuda
        and a.shape[1] > 2048
    )


class GroupedGemmFP8BlockFunc(torch.autograd.Function):
    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size in [128], "Only block_size 128 is supported currently."
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        assert group_lens.size(0) == b.size(0), "group_lens size must match b size(0)."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(config.format, True)
        use_triton_layout = _use_triton_grouped_gemm_backend()
        needs_grad_a = ctx.needs_input_grad[0]
        needs_grad_b = ctx.needs_input_grad[1]

        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_cached(
            a,
            a_dtype,
            axis=1,
            block_size=config.block_size,
            scales_transposed=use_triton_layout,
            buf_cache=_bw_cache_a_row,
        )

        weight_key = _get_blockwise_weight_cache_key(b, config.block_size)
        cached_entry = _bw_weight_cache.get(id(b))
        if (
            cached_entry is not None
            and cached_entry[0]() is b
            and cached_entry[1] == weight_key
        ):
            b_fp8, b_scale_inv = cached_entry[2], cached_entry[3]
        else:
            b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(
                b, b_dtype, block_size=config.block_size
            )
            if len(_bw_weight_cache) >= 32:
                _bw_weight_cache.clear()
            _bw_weight_cache[id(b)] = (weakref.ref(b), weight_key, b_fp8, b_scale_inv)

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
            default_backend=BackendType.CK.value,
        )

        ctx.save_for_backward(
            a,
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
        ctx.needs_grad_a = needs_grad_a
        ctx.needs_grad_b = needs_grad_b

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)

        (
            a,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors
        block_size = ctx.config.block_size
        grad_out_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(ctx.config.format, False)
        use_triton_layout = _use_triton_grouped_gemm_backend()
        grad_a = None
        grad_b = None
        overlap_seg_quant = _should_overlap_blockwise_bwd(a, ctx.needs_grad_a, ctx.needs_grad_b)
        aux_stream = None
        a_fp8_col = None
        a_scale_inv_col = None
        grad_out_fp8_col = None
        grad_out_scale_inv_col = None
        var_k_group_lens = None
        var_k_group_offs = None

        if overlap_seg_quant:
            main_stream = torch.cuda.current_stream(device=a.device)
            aux_stream = _get_blockwise_aux_stream(a.device)
            aux_stream.wait_stream(main_stream)
            with torch.cuda.stream(aux_stream):
                a_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(ctx.config.format, True)
                a_fp8_col, a_scale_inv_col, _, _ = quant_fp8_blockwise_segment_m_cached(
                    a, a_dtype, block_size, group_lens, group_offs, buf_cache=_bw_cache_a_seg
                )

                grad_out_fp8_col, grad_out_scale_inv_col, var_k_group_lens, var_k_group_offs = (
                    quant_fp8_blockwise_segment_m_cached(
                        grad_out,
                        grad_out_dtype,
                        block_size,
                        group_lens,
                        group_offs,
                        buf_cache=_bw_cache_grad_seg,
                    )
                )

        if ctx.needs_grad_a:
            # Quantize grad_out in row-wise for dgrad
            grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_cached(
                grad_out,
                grad_out_dtype,
                axis=1,
                block_size=block_size,
                scales_transposed=use_triton_layout,
                buf_cache=_bw_cache_grad_row,
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
                default_backend=BackendType.CK.value,
            )

        if ctx.needs_grad_b:
            if overlap_seg_quant:
                torch.cuda.current_stream(device=a.device).wait_stream(aux_stream)
            else:
                a_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(ctx.config.format, True)
                a_fp8_col, a_scale_inv_col, _, _ = quant_fp8_blockwise_segment_m_cached(
                    a, a_dtype, block_size, group_lens, group_offs, buf_cache=_bw_cache_a_seg
                )

                # Quantize grad_out with segment padding for wgrad (colwise quantization)
                grad_out_fp8_col, grad_out_scale_inv_col, var_k_group_lens, var_k_group_offs = (
                    quant_fp8_blockwise_segment_m_cached(
                        grad_out,
                        grad_out_dtype,
                        block_size,
                        group_lens,
                        group_offs,
                        buf_cache=_bw_cache_grad_seg,
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
                default_backend=BackendType.CK.value,
            )

        return grad_a, grad_b, None, None, None, None, None


class GroupedGemmFP8RowFunc(torch.autograd.Function):
    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.ROWWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)
        a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, config.granularity, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-1 if trans_b else -2)
        )
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.CK.value,
        )

        # we need a/b do col quant for backward.
        a_fp8_col, a_scale_inv_col = quantize_fp8(a, a_dtype, config.granularity, axis=-2)
        b_fp8_col, b_scale_inv_col = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-2 if trans_b else -1)
        )

        ctx.save_for_backward(a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs)
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

        # For grad_a
        grad_out_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-1
        )

        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        # For grad_b
        grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class GroupedGemmFP8TensorFunc(torch.autograd.Function):

    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        elif format == Format.HYBRID:
            return float8_e4m3 if is_fwd_stage else float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):

        assert config.granularity == ScalingGranularity.TENSORWISE
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        a_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)

        a_fp8, a_scale_inv = _quant_fp8_tw(a, a_dtype, buf_cache=_tw_cache_a)
        if _tensorwise_weight_reuse_enabled():
            weight_key = _get_tensorwise_weight_cache_key(b)
            cached_weight = _tw_weight_cache.get(weight_key)
            if cached_weight is None:
                b_fp8, b_scale_inv = _quant_fp8_tw(b, b_dtype, buf_cache=_tw_cache_b)
                if len(_tw_weight_cache) >= 32:
                    _tw_weight_cache.clear()
                _tw_weight_cache[weight_key] = (b_fp8, b_scale_inv)
            else:
                b_fp8, b_scale_inv = cached_weight
        else:
            b_fp8, b_scale_inv = _quant_fp8_tw(b, b_dtype, buf_cache=_tw_cache_b)

        out = grouped_gemm_fp8_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.CK.value,
        )

        ctx.save_for_backward(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = a.dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        grad_out_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8, grad_out_scale_inv = _quant_fp8_tw(
            grad_out, grad_out_dtype, buf_cache=_tw_cache_grad
        )

        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8,
            b_fp8,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8,
            grad_out_fp8,
            a_scale_inv,
            grad_out_scale_inv,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        return grad_a, grad_b, None, None, None, None, None


def grouped_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = True,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """ """
    supported_dtypes = [torch.bfloat16, torch.float16]
    assert a.dtype in supported_dtypes, f"Unsupported dtype {a.dtype}, expected one of {supported_dtypes}"
    assert b.dtype in supported_dtypes, f"Unsupported dtype {b.dtype}, expected one of {supported_dtypes}"

    if group_offs is None:
        group_offs = _get_cached_group_offs(group_lens)
    if config is None:
        config = Float8QuantConfig()

    args = (a, b, group_lens, group_offs, trans_b, config, num_cu)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return GroupedGemmFP8TensorFunc.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return GroupedGemmFP8RowFunc.apply(*args)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return GroupedGemmFP8BlockFunc.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
