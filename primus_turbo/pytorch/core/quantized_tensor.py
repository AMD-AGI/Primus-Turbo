###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP4_PADDING_ALIGN_SIZE,
    MXFP8_BLOCK_SIZE,
    MXFP8_PADDING_ALIGN_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    check_mxfp8_support,
    float4_e2m1fn_x2,
    float8_e4m3,
    float8_e5m2,
)

_SUPPORTED_QUANTIZED_DTYPES = [float8_e4m3, float8_e5m2, float4_e2m1fn_x2]
_SUPPORTED_QUANTIZED_GRANS = [
    ScalingGranularity.ROWWISE,
    ScalingGranularity.TENSORWISE,
    ScalingGranularity.MX_BLOCKWISE,
]
_SUPPORTED_GROUPED_QUANTIZED_GRANS = [ScalingGranularity.ROWWISE, ScalingGranularity.TENSORWISE]


def _pad_inner_dim(shape, align):
    """Pad the last dimension of *shape* to a multiple of *align*,
    mirroring the C++ ``cdiv(N, ALIGN) * ALIGN`` logic used by
    ``quantize_mxfp8_dual`` / ``quantize_mxfp4_dual``."""
    if len(shape) == 0:
        return shape
    # Tensorwise / rowwise paths do not require inner-dim padding.
    # Treat align<=1 as a no-op to avoid divide-by-zero for align==0.
    if align <= 1:
        return shape
    last = shape[-1]
    padded = (last + align - 1) // align * align
    if padded == last:
        return shape
    return (*shape[:-1], padded)


def _get_padding_align_size(tensor: QuantizedTensor):
    """Return the padding alignment for *tensor* based on its granularity."""
    assert isinstance(tensor, QuantizedTensor), "tensor must be a QuantizedTensor"
    if (
        tensor._granularity == ScalingGranularity.TENSORWISE
        or tensor._granularity == ScalingGranularity.ROWWISE
    ):
        return 0
    else:
        if tensor._dest_dtype == float4_e2m1fn_x2:
            return MXFP4_PADDING_ALIGN_SIZE
        else:
            assert tensor._dest_dtype == float8_e4m3 or tensor._dest_dtype == float8_e5m2
            return MXFP8_PADDING_ALIGN_SIZE


def _get_packing_factor(tensor: QuantizedTensor):
    """fp4 packs 2 values per element; fp8 has 1 value per element."""
    assert isinstance(tensor, QuantizedTensor), "tensor must be a QuantizedTensor"
    if tensor._dest_dtype == float4_e2m1fn_x2:
        return 2
    return 1


def _normalize_axis(axis: Optional[int], ndim: int) -> Optional[int]:
    """Resolve a possibly-negative ``axis`` against ``ndim``.

    The wrapper APIs (``QuantizedTensor._quantize`` etc.) expose ``axis=-1`` /
    ``axis=-2`` so that 2D and 3D inputs share the same call sites
    (``-1`` = inner-K, ``-2`` = inner-M).  Several low-level kernels — most
    notably the C++ MXFP8/MXFP4 ones — strictly require a non-negative axis,
    so we normalise *before* dispatching downwards.
    """
    if axis is None:
        return None
    return axis if axis >= 0 else axis + ndim


def _compute_scale_shape(data_shape, block_size: int, packing_factor: int):
    """Derive the scale tensor shape from the data tensor shape.

    Per quantization.cpp (rowwise, no shuffle)::

        N_pad  = data_last_dim * packing_factor
        scale  = [*data_leading_dims, cdiv(N_pad, block_size)]
    """
    n_pad = data_shape[-1] * packing_factor
    scale_last = (n_pad + block_size - 1) // block_size
    return (*data_shape[:-1], scale_last)


def check_quantized_tensor(
    quantized_tensor: QuantizedTensor,
    config: Any,
    axis: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
) -> None:
    """Assert a QuantizedTensor's granularity / block_size (and optionally
    scaling recipes) match the given quant config.

    ``config`` is duck-typed: any object exposing ``granularity`` and
    ``block_size`` attributes (e.g. ``Float8QuantConfig`` / ``Float4QuantConfig``)
    is accepted.
    """
    assert quantized_tensor.granularity == config.granularity, (
        f"QuantizedTensor granularity {quantized_tensor.granularity} does not match config "
        f"granularity {config.granularity}"
    )
    assert quantized_tensor.block_size == config.block_size, (
        f"QuantizedTensor block_size {quantized_tensor.block_size} does not match config "
        f"block_size {config.block_size}"
    )

    if axis is not None:
        normalized_axis = _normalize_axis(axis, quantized_tensor.qdata.ndim)
        assert (
            quantized_tensor.quantized_axis == normalized_axis
        ), f"QuantizedTensor quantized_axis {quantized_tensor.quantized_axis} does not match axis={normalized_axis}"

    if scaling_recipe is not None:
        assert quantized_tensor.scaling_recipe == scaling_recipe, (
            f"QuantizedTensor scaling_recipe {quantized_tensor.scaling_recipe} does not match config "
            f"scaling_recipe {scaling_recipe}"
        )


class QuantizedTensor(torch.Tensor):
    """Wrapper subclass that carries low-precision quantized data, scale_inv."""

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        scale_inv: Optional[torch.Tensor],
        *,
        shape: torch.Size,
        orig_dtype: torch.dtype,
        dest_dtype: torch.dtype,
        granularity: ScalingGranularity,
        group_lens: Optional[torch.Tensor] = None,
        group_offs: Optional[torch.Tensor] = None,
        is_grouped_tensor: bool = False,
        block_size: Optional[int] = None,
        scaling_recipe: Optional[ScalingRecipe] = None,
        quantized_axis: Optional[int] = None,
        requires_grad: bool = False,
    ):
        assert dest_dtype in _SUPPORTED_QUANTIZED_DTYPES, "Unsupported quantized dtype"
        assert data.is_contiguous(), "data must be contiguous"
        assert scale_inv.is_contiguous(), "scale_inv must be contiguous"

        if is_grouped_tensor:
            assert group_lens is not None, "group_lens must be provided for grouped tensors"
            assert group_lens.is_cuda, "group_lens must be on CUDA"

            # Materialise ``group_offs`` from ``group_lens`` only if it was not
            # supplied (e.g. by ``__tensor_unflatten__`` / ``_make_like`` which
            # already carry a valid cached offsets tensor).
            if group_offs is None:
                from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
                    group_offs_from_lens,
                )

                group_offs = group_offs_from_lens(group_lens)

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=requires_grad,
            device=data.device,
        )
        self._orig_dtype = orig_dtype
        self._dest_dtype = dest_dtype

        self._scaling_recipe = scaling_recipe
        self._granularity = granularity
        self._block_size = block_size

        self._group_lens = group_lens
        self._group_offs = group_offs
        self._is_grouped_tensor = is_grouped_tensor
        self._quantized_axis = quantized_axis

        self._data, self._scale_inv = data, scale_inv

        return self

    # ------------------------------------------------------------------
    # Factory: quantize a high-precision tensor and wrap the result.
    # ------------------------------------------------------------------
    @classmethod
    def quantize(
        cls,
        hp_tensor: Union[torch.Tensor, torch.nn.Parameter],
        dest_dtype: torch.dtype,
        granularity: ScalingGranularity,
        *,
        axis: Optional[int] = None,
        group_lens: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
        scaling_recipe: Optional[ScalingRecipe] = None,
    ) -> "QuantizedTensor":
        """Quantize *hp_tensor* and return a ``QuantizedTensor`` wrapping it.

        This is the standard way to construct a ``QuantizedTensor`` from a
        high-precision input.  Use ``__new__`` directly only when you already
        hold quantized buffers (e.g. inside a custom kernel's autograd
        Function output, de-serialisation, or wrapper-preserving views).
        """
        assert hp_tensor.ndim in (2, 3), f"data must be a 2D or 3D tensor, got {hp_tensor.ndim}D"
        assert dest_dtype in _SUPPORTED_QUANTIZED_DTYPES, "Unsupported quantized dtype"

        is_grouped_tensor = group_lens is not None
        if is_grouped_tensor:
            assert (
                granularity in _SUPPORTED_GROUPED_QUANTIZED_GRANS
            ), f"Unsupported grouped granularity {granularity}"
            assert (
                hp_tensor.ndim == 2
            ), f"Grouped quantized tensor expects a 2D packed-M tensor, got {hp_tensor.ndim}D"
            assert (
                dest_dtype in _SUPPORTED_QUANTIZED_DTYPES and dest_dtype != float4_e2m1fn_x2
            ), "Unsupported quantized dtype (FP4 not supported for grouped activations)"
        else:
            assert granularity in _SUPPORTED_QUANTIZED_GRANS, f"Unsupported granularity {granularity}"

            if granularity == ScalingGranularity.MX_BLOCKWISE:
                assert dest_dtype in [
                    float8_e4m3,
                    float8_e5m2,
                    float4_e2m1fn_x2,
                ], "Unsupported quantized dtype for MX_BLOCKWISE"

                if dest_dtype == float4_e2m1fn_x2:
                    supported, reason = check_mxfp4_support()
                    assert supported, reason
                    assert (
                        block_size == MXFP4_BLOCK_SIZE
                    ), "block_size must be MXFP4_BLOCK_SIZE for MX_BLOCKWISE"
                elif dest_dtype in [float8_e4m3, float8_e5m2]:
                    supported, reason = check_mxfp8_support()
                    assert (
                        block_size == MXFP8_BLOCK_SIZE
                    ), "block_size must be MXFP8_BLOCK_SIZE for MX_BLOCKWISE"
                    assert supported, reason

                assert scaling_recipe is not None, "scaling_recipe must be provided for MX_BLOCKWISE"

        if granularity != ScalingGranularity.TENSORWISE:
            assert axis is not None, "axis must be provided for non-TENSORWISE granularity"

        data, scale_inv = cls._quantize(
            hp_tensor,
            dest_dtype,
            granularity,
            block_size=block_size,
            axis=axis,
            scaling_recipe=scaling_recipe,
        )

        return cls(
            data,
            scale_inv,
            # NOTE: quantize will do padding internally
            # To keep original shape here is to workaround autograd issues.
            shape=hp_tensor.size(),
            orig_dtype=hp_tensor.dtype,
            dest_dtype=dest_dtype,
            granularity=granularity,
            block_size=block_size,
            scaling_recipe=scaling_recipe,
            requires_grad=hp_tensor.requires_grad,
            quantized_axis=axis,
            # grouped tensor related
            group_lens=group_lens,
            is_grouped_tensor=is_grouped_tensor,
        )

    @classmethod
    @torch.no_grad()
    def _quantize(
        cls,
        data: torch.Tensor,
        dest_dtype: torch.dtype,
        granularity: ScalingGranularity,
        block_size: Optional[int] = None,
        axis: Optional[int] = None,
        scaling_recipe: Optional[ScalingRecipe] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from primus_turbo.pytorch.ops.quantization import quantize_fp4, quantize_fp8

        axis = _normalize_axis(axis, data.ndim)

        if dest_dtype in [float8_e4m3, float8_e5m2]:
            data_, scale_inv = quantize_fp8(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                axis=axis,
                scaling_recipe=scaling_recipe,
            )
            return data_, scale_inv
        else:
            assert dest_dtype == float4_e2m1fn_x2
            assert granularity == ScalingGranularity.MX_BLOCKWISE
            # MXFP4 single-direction kernel only supports 2D input with axis in {0, 1};
            assert data.ndim == 2, (
                f"FP4 single-direction quantization requires a 2D tensor, " f"got {data.ndim}D."
            )

            data_, scale_inv = quantize_fp4(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                axis=axis,
                scaling_recipe=scaling_recipe,
            )
            return data_, scale_inv

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def qdata(self) -> torch.Tensor:
        """Quantized inner data buffer."""
        assert self._data is not None, "qdata is None. Please call quantize() to quantize the data."

        return self._data

    @property
    def scale_inv(self) -> torch.Tensor:
        assert self._scale_inv is not None, "scale_inv is None. Please call quantize() to quantize the data."

        return self._scale_inv

    @property
    def group_lens(self) -> torch.Tensor:
        assert self._is_grouped_tensor, "This is not a grouped tensor."

        return self._group_lens

    @property
    def group_offs(self) -> torch.Tensor:
        assert self._is_grouped_tensor, "This is not a grouped tensor."

        return self._group_offs

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        return self._orig_dtype

    @property
    def real_dtype(self) -> torch.dtype:
        return self._data.dtype

    @property
    def granularity(self) -> ScalingGranularity:
        return self._granularity

    @property
    def block_size(self) -> Union[int, None]:
        return self._block_size

    @property
    def scaling_recipe(self) -> ScalingRecipe:
        return self._scaling_recipe

    @property
    def quantized_axis(self) -> int:
        return _normalize_axis(self._quantized_axis, self._data.ndim)

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """Dequantize back to the original high-precision dtype."""
        from primus_turbo.pytorch.ops.quantization import dequantize_fp4, dequantize_fp8

        axis = _normalize_axis(self._quantized_axis, self._data.ndim)

        # TODO(ruibi): add support for grouped dequantization
        if self._dest_dtype in [float8_e4m3, float8_e5m2]:
            out = dequantize_fp8(
                self._data,
                self._orig_dtype,
                self._granularity,
                block_size=self._block_size,
                axis=axis,
                scale_inv=self._scale_inv,
                scaling_recipe=self._scaling_recipe,
            )
        elif self._dest_dtype == float4_e2m1fn_x2:
            out = dequantize_fp4(
                self._data,
                self._orig_dtype,
                self._granularity,
                block_size=self._block_size,
                axis=axis,
                scale_inv=self._scale_inv,
                scaling_recipe=self._scaling_recipe,
            )
        else:
            assert False, "Unsupported dtype"

        # TODO(ruibin): fused unpad in dequantize kernel
        if out.ndim == 2:
            out = out[:, : self.size(-1)].contiguous()
        elif out.ndim == 3:
            out = out[:, :, : self.size(-1)].contiguous()
        else:
            assert False, "Unsupported ndim"

        return out

    # ------------------------------------------------------------------
    # Serialisation (torch.compile / FSDP)
    # ------------------------------------------------------------------
    def __tensor_flatten__(self):
        tensors = {"_data": self._data, "_scale_inv": self._scale_inv}
        if self._group_lens is not None:
            tensors["_group_lens"] = self._group_lens
        if self._group_offs is not None:
            tensors["_group_offs"] = self._group_offs
        metadata = {
            "_orig_dtype": self._orig_dtype,
            "_dest_dtype": self._dest_dtype,
            "_granularity": self._granularity,
            "_block_size": self._block_size,
            "_scaling_recipe": self._scaling_recipe,
            "_is_grouped_tensor": self._is_grouped_tensor,
            "_quantized_axis": self._quantized_axis,
        }
        return list(tensors.keys()), metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        # ``outer_stride`` is provided by PyTorch wrapper-subclass unflatten
        # hooks (e.g. FSDP / torch.compile rebuild paths).
        _ = outer_stride
        return QuantizedTensor(
            inner_tensors["_data"],
            inner_tensors["_scale_inv"],
            shape=outer_size,
            orig_dtype=metadata["_orig_dtype"],
            dest_dtype=metadata["_dest_dtype"],
            granularity=metadata["_granularity"],
            block_size=metadata["_block_size"],
            scaling_recipe=metadata["_scaling_recipe"],
            group_lens=inner_tensors.get("_group_lens"),
            group_offs=inner_tensors.get("_group_offs"),
            is_grouped_tensor=metadata["_is_grouped_tensor"],
            quantized_axis=metadata.get("_quantized_axis"),
        )

    # ------------------------------------------------------------------
    # View / reshape helpers
    # ------------------------------------------------------------------
    @classmethod
    def _make_like(
        cls,
        tensor: "QuantizedTensor",
        *,
        data: torch.Tensor,
        scale_inv: torch.Tensor,
        shape: torch.Size,
    ) -> "QuantizedTensor":
        """Construct a new ``QuantizedTensor`` sharing metadata with *tensor*
        but using the supplied *data*, *scale_inv* and *shape*.  Used by
        view / reshape dispatch to avoid a dequantize round-trip.

        Grouped metadata (``group_lens`` / ``group_offs`` / ``is_grouped_tensor``)
        is propagated unchanged so packed-M views/reshapes retain their
        grouped identity.
        """
        return cls(
            data,
            scale_inv,
            shape=shape,
            orig_dtype=tensor._orig_dtype,
            dest_dtype=tensor._dest_dtype,
            granularity=tensor._granularity,
            block_size=tensor._block_size,
            scaling_recipe=tensor._scaling_recipe,
            group_lens=tensor._group_lens,
            group_offs=tensor._group_offs,
            is_grouped_tensor=tensor._is_grouped_tensor,
            quantized_axis=tensor._quantized_axis,
            requires_grad=data.requires_grad,
        )

    @staticmethod
    def _view_data_and_scale_inv(tensor: QuantizedTensor, target_shape: torch.Size, op: Callable):
        """Apply *op* (``view`` or ``reshape``) to ``_data``, ``_scale_inv``"""
        assert len(target_shape) == tensor.qdata.ndim

        out_shape = torch.Size(target_shape)
        wrapper_shape = tensor.shape

        if out_shape == wrapper_shape:
            return tensor._data, tensor._scale_inv, out_shape

        align = _get_padding_align_size(tensor)
        padded_target_shape = _pad_inner_dim(target_shape, align)

        packing = _get_packing_factor(tensor)
        assert padded_target_shape[-1] % packing == 0, (
            f"padded inner dim {padded_target_shape[-1]} must be divisible by " f"packing factor {packing}"
        )
        data_target_shape = (
            *padded_target_shape[:-1],
            padded_target_shape[-1] // packing,
        )

        out_data = op(tensor._data, *data_target_shape)

        out_scale_inv = tensor._scale_inv
        if tensor._granularity == ScalingGranularity.MX_BLOCKWISE and tensor._scale_inv is not None:
            block_size = tensor._block_size
            scale_target_shape = _compute_scale_shape(data_target_shape, block_size, packing)
            out_scale_inv = op(tensor._scale_inv, *scale_target_shape)

        return out_data, out_scale_inv, out_shape

    def view(self, *shape) -> "QuantizedTensor":
        """View without dequantizing (autograd-aware)."""
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape) -> "QuantizedTensor":
        """Reshape without dequantizing (autograd-aware)."""
        return _ReshapeFunc.apply(self, shape)

    # ------------------------------------------------------------------
    # Dispatch hooks
    # ------------------------------------------------------------------
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
            tensor = args[0]
            if isinstance(tensor, QuantizedTensor):
                out_data, out_sinv, out_shape = cls._view_data_and_scale_inv(
                    tensor, args[1], torch.Tensor.view
                )
                return cls._make_like(
                    tensor,
                    data=out_data,
                    scale_inv=out_sinv,
                    shape=out_shape,
                )

        if func == torch.ops.aten.reshape.default:
            tensor = args[0]
            if isinstance(tensor, QuantizedTensor):
                out_data, out_sinv, out_shape = cls._view_data_and_scale_inv(
                    tensor, args[1], torch.Tensor.reshape
                )
                return cls._make_like(
                    tensor,
                    data=out_data,
                    scale_inv=out_sinv,
                    shape=out_shape,
                )

        if func == torch.ops.aten.select.int:
            tensor = args[0]
            if isinstance(tensor, QuantizedTensor):
                return func(tensor._data, *args[1:], **(kwargs or {}))

        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            tensor = args[0]
            if isinstance(tensor, QuantizedTensor):
                return cls._make_like(
                    tensor,
                    data=tensor._data.detach(),
                    scale_inv=tensor._scale_inv.detach(),
                    shape=tensor.shape,
                )

        raise NotImplementedError(f"Unsupported dispatch: {func}")

    __torch_function__ = torch._C._disabled_torch_function_impl


class _ViewFunc(torch.autograd.Function):
    """View a QuantizedTensor without dequantizing."""

    @staticmethod
    def forward(ctx, tensor: QuantizedTensor, shape: Tuple[int, ...]) -> QuantizedTensor:
        ctx.shape = tensor.shape
        out_data, out_sinv, out_shape = QuantizedTensor._view_data_and_scale_inv(
            tensor, shape, torch.Tensor.view
        )
        # ``_make_like`` propagates all wrapper metadata (including grouped
        # fields ``group_lens`` / ``group_offs`` / ``is_grouped_tensor``), so
        # packed-M wrappers keep their grouped identity through the view.
        return type(tensor)._make_like(
            tensor,
            data=out_data,
            scale_inv=out_sinv,
            shape=out_shape,
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        return grad.reshape(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape a QuantizedTensor without dequantizing."""

    @staticmethod
    def forward(ctx, tensor: QuantizedTensor, shape: Tuple[int, ...]) -> QuantizedTensor:
        ctx.shape = tensor.shape
        out_data, out_sinv, out_shape = QuantizedTensor._view_data_and_scale_inv(
            tensor, shape, torch.Tensor.reshape
        )
        # See ``_ViewFunc.forward`` — ``_make_like`` preserves grouped metadata.
        return type(tensor)._make_like(
            tensor,
            data=out_data,
            scale_inv=out_sinv,
            shape=out_shape,
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        return grad.reshape(ctx.shape), None


class QuantizedTensorPair(NamedTuple):
    """
    Wrapper for quantized tensors.

    Args:
        data: Row-major quantized tensor.
        data_t: Transposed quantized tensor.
    """

    data: QuantizedTensor
    data_t: Optional[QuantizedTensor] = None
