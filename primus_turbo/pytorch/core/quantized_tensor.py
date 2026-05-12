###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP4_PADDING_ALIGN_SIZE,
    MXFP8_BLOCK_SIZE,
    MXFP8_PADDING_ALIGN_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
    float8_e4m3,
    float8_e5m2,
)

_SUPPORTED_QUANTIZED_DTYPES = [float8_e4m3, float8_e5m2, float4_e2m1fn_x2]


def _pad_inner_dim(shape, align):
    """Pad the last dimension of *shape* to a multiple of *align*,
    mirroring the C++ ``cdiv(N, ALIGN) * ALIGN`` logic used by
    ``quantize_mxfp8_dual`` / ``quantize_mxfp4_dual``."""
    if len(shape) == 0:
        return shape
    last = shape[-1]
    padded = (last + align - 1) // align * align
    if padded == last:
        return shape
    return (*shape[:-1], padded)


def _get_padding_align_size(tensor: Union[QuantizedTensor, GroupedQuantizedTensor]):
    """Return the padding alignment for *tensor* based on its granularity.

    Accepts any wrapper exposing ``_granularity`` and ``_dest_dtype``
    (i.e. ``QuantizedTensor`` or ``GroupedQuantizedTensor``).
    """
    assert isinstance(tensor, QuantizedTensor) or isinstance(
        tensor, GroupedQuantizedTensor
    ), "tensor must be a QuantizedTensor or GroupedQuantizedTensor"
    if (
        tensor._granularity == ScalingGranularity.TENSORWISE
        or tensor._granularity == ScalingGranularity.ROWWISE
        or tensor._granularity == ScalingGranularity.BLOCKWISE
    ):
        return 0
    else:
        if tensor._dest_dtype == float4_e2m1fn_x2:
            return MXFP4_PADDING_ALIGN_SIZE
        else:
            assert tensor._dest_dtype == float8_e4m3 or tensor._dest_dtype == float8_e5m2
            return MXFP8_PADDING_ALIGN_SIZE


def _get_packing_factor(tensor: Union[QuantizedTensor, GroupedQuantizedTensor]):
    """fp4 packs 2 values per element; fp8 has 1 value per element.

    Accepts any wrapper exposing ``_dest_dtype``.
    """
    assert isinstance(tensor, QuantizedTensor) or isinstance(
        tensor, GroupedQuantizedTensor
    ), "tensor must be a QuantizedTensor or GroupedQuantizedTensor"
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
    quantized_tensor: Union[QuantizedTensor, GroupedQuantizedTensor],
    config: Any,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
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

    if scaling_recipe is not None:
        assert quantized_tensor.scaling_recipe == scaling_recipe, (
            f"QuantizedTensor scaling_recipe {quantized_tensor.scaling_recipe} does not match config "
            f"scaling_recipe {scaling_recipe}"
        )
    if scaling_recipe_for_trans is not None:
        assert quantized_tensor.scaling_recipe_for_trans == scaling_recipe_for_trans, (
            f"QuantizedTensor scaling_recipe_for_trans {quantized_tensor.scaling_recipe_for_trans} does not match config "
            f"scaling_recipe_for_trans {scaling_recipe_for_trans}"
        )


def check_grouped_quantized_tensor(
    grouped_quantized_tensor: GroupedQuantizedTensor,
    config: Any,
    group_lens: torch.Tensor,
) -> None:
    check_quantized_tensor(grouped_quantized_tensor, config)

    assert torch.equal(
        grouped_quantized_tensor.group_lens, group_lens
    ), "group_lens must match the given group_lens"


class QuantizedTensor(torch.Tensor):
    """Wrapper subclass that carries low-precision quantized data, scale_inv"""

    @staticmethod
    def __new__(
        cls,
        data: Union[torch.Tensor, torch.nn.Parameter],
        dest_dtype: torch.dtype,
        granularity: ScalingGranularity,
        block_size: Optional[int] = None,
        scaling_recipe: Optional[ScalingRecipe] = None,
        scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
        keep_trans_cache: bool = False,
    ):
        # 2D: standard activation/weight; 3D: batched weight (e.g. MoE expert
        # stacks, shape [G, K, N]).  Underlying low-level quant kernels decide
        # what they actually accept — the wrapper only plumbs through.
        assert data.ndim in (2, 3), f"data must be a 2D or 3D tensor, got {data.ndim}D"
        assert dest_dtype in _SUPPORTED_QUANTIZED_DTYPES, "Unsupported quantized dtype"

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            if dest_dtype in [float8_e4m3, float8_e5m2]:
                assert block_size == MXFP8_BLOCK_SIZE, "block_size must be MXFP8_BLOCK_SIZE for MX_BLOCKWISE"
            elif dest_dtype == float4_e2m1fn_x2:
                assert block_size == MXFP4_BLOCK_SIZE, "block_size must be MXFP4_BLOCK_SIZE for MX_BLOCKWISE"

            assert scaling_recipe is not None, "scaling_recipe must be provided for MX_BLOCKWISE"
            assert (
                scaling_recipe_for_trans is not None
            ), "scaling_recipe_for_trans must be provided for MX_BLOCKWISE"

        orig_dtype = data.dtype

        data_, scale_inv, data_t, scale_inv_t = None, None, None, None

        if keep_trans_cache:
            data_, scale_inv, data_t, scale_inv_t = QuantizedTensor._quantize_with_trans(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                scaling_recipe=scaling_recipe,
                scaling_recipe_for_trans=scaling_recipe_for_trans,
            )
        else:
            data_, scale_inv = QuantizedTensor._quantize(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                axis=-1,
                scaling_recipe=scaling_recipe,
            )

        # NOTE: use the *original* tensor's size / stride for the wrapper.
        # Some quantization kernels (e.g. MXFP8) pad the inner-dim to a
        # multiple of block_size, which would otherwise make the wrapper
        # shape diverge from the high-precision source tensor — breaking
        # autograd's gradient-shape check.
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._orig_dtype = orig_dtype
        self._dest_dtype = dest_dtype
        # Pinning the original ndim lets view/reshape forbid 2D <-> 3D
        # round-trips, which would otherwise silently break the
        # layout-dependent scale shape and transpose-cache semantics.
        self._orig_ndim = data.ndim

        self._keep_trans_cache = keep_trans_cache
        self._scaling_recipe = scaling_recipe
        self._granularity = granularity
        self._block_size = block_size
        self._scaling_recipe_for_trans = scaling_recipe_for_trans

        self._data, self._scale_inv = data_, scale_inv
        self._data_t, self._scale_inv_t = data_t, scale_inv_t

        return self

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
                f"FP4 single-direction quantization requires a 2D tensor, "
                f"got {data.ndim}D. Use keep_trans_cache=True for batched layouts."
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

    @classmethod
    @torch.no_grad()
    def _quantize_with_trans(
        cls,
        data: torch.Tensor,
        dest_dtype: torch.dtype,
        granularity: ScalingGranularity,
        block_size: Optional[int] = None,
        scaling_recipe: Optional[ScalingRecipe] = None,
        scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from primus_turbo.pytorch.ops.quantization import (
            quantize_fp4_with_trans,
            quantize_fp8,
            quantize_fp8_with_trans,
        )

        axis_inner = _normalize_axis(-1, data.ndim)
        axis_outer = _normalize_axis(-2, data.ndim)

        if dest_dtype in [float8_e4m3, float8_e5m2]:
            if granularity == ScalingGranularity.MX_BLOCKWISE:
                data_, scale_inv, data_t, scale_inv_t = quantize_fp8_with_trans(
                    data,
                    dest_dtype,
                    granularity,
                    block_size=block_size,
                    scaling_recipe=scaling_recipe,
                    scaling_recipe_for_trans=scaling_recipe_for_trans,
                )
            else:
                data_, scale_inv = quantize_fp8(
                    data,
                    dest_dtype,
                    granularity,
                    block_size=block_size,
                    axis=axis_inner,
                    scaling_recipe=scaling_recipe,
                )
                data_t, scale_inv_t = quantize_fp8(
                    data,
                    dest_dtype,
                    granularity,
                    block_size=block_size,
                    axis=axis_outer,
                    scaling_recipe=scaling_recipe_for_trans,
                )
        else:
            assert dest_dtype == float4_e2m1fn_x2
            assert granularity == ScalingGranularity.MX_BLOCKWISE

            data_, scale_inv, data_t, scale_inv_t = quantize_fp4_with_trans(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                scaling_recipe=scaling_recipe,
                scaling_recipe_for_trans=scaling_recipe_for_trans,
            )

        return data_, scale_inv, data_t, scale_inv_t

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def scale_inv(self) -> torch.Tensor:
        return self._scale_inv

    @property
    def orig_ndim(self) -> int:
        """The ndim of the high-precision source tensor at construction time.

        Used to enforce that downstream ``view`` / ``reshape`` keep the same
        rank (e.g. 2D-to-2D or 3D-to-3D), since cross-rank reshapes would
        invalidate the layout-dependent scale shape and the transpose cache.
        """
        return self._orig_ndim

    def t(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(data_t, scale_inv_t)`` — the column-quantized counterpart
        sharing the *same physical layout* as ``_data`` / ``_scale_inv``.

        ``_scale_inv`` is reduced along ``axis=-1`` (each scale spans a "row"
        of the last two dims); ``_scale_inv_t`` is reduced along ``axis=-2``
        (each scale spans a "column").  The data buffer's layout is unchanged
        — only the scale-reduction axis differs.

        Despite the method name, the returned tensors are NOT physically
        transposed.  This matches what ``_quantize_with_trans`` produces
        when ``keep_trans_cache=True``, so the eager and lazy construction
        paths produce semantically identical buffers.
        """
        if self._data_t is None:
            # TODO(ruibin): eliminate the dequant->requant by a low-precision
            # in-place col-quantize kernel.
            orig_data = self.dequantize()

            data_t, scale_inv_t = QuantizedTensor._quantize(
                orig_data,
                self._dest_dtype,
                self._granularity,
                block_size=self._block_size,
                axis=-2,
                scaling_recipe=self._scaling_recipe_for_trans,
            )

            if self._keep_trans_cache:
                # cache the transposed data and scale_inv
                self._data_t = data_t
                self._scale_inv_t = scale_inv_t
            else:
                return data_t, scale_inv_t

        return self._data_t, self._scale_inv_t

    @property
    def T(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.t()

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
    def scaling_recipe_for_trans(self) -> ScalingRecipe:
        return self._scaling_recipe_for_trans

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """Dequantize back to the original high-precision dtype."""
        from primus_turbo.pytorch.ops.quantization import dequantize_fp4, dequantize_fp8

        axis = _normalize_axis(-1, self._data.ndim)

        if self._dest_dtype in [float8_e4m3, float8_e5m2]:
            return dequantize_fp8(
                self._data,
                self._orig_dtype,
                self._granularity,
                block_size=self._block_size,
                axis=axis,
                scale_inv=self._scale_inv,
                scaling_recipe=self._scaling_recipe,
            )
        elif self._dest_dtype == float4_e2m1fn_x2:
            return dequantize_fp4(
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

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        has_trans = self._data_t is not None
        return (
            f"QuantizedTensor(shape={list(self.shape)}, orig_dtype={self._orig_dtype}, "
            f"real_dtype={self._data.dtype}, granularity={self._granularity.name}"
            f"{', has_trans_cache' if has_trans else ''})"
        )

    # ------------------------------------------------------------------
    # Serialisation (torch.compile / FSDP)
    # ------------------------------------------------------------------
    def __tensor_flatten__(self):
        tensors = {"_data": self._data, "_scale_inv": self._scale_inv}
        if self._data_t is not None:
            tensors["_data_t"] = self._data_t
        if self._scale_inv_t is not None:
            tensors["_scale_inv_t"] = self._scale_inv_t
        metadata = {
            "_orig_dtype": self._orig_dtype,
            "_dest_dtype": self._dest_dtype,
            "_orig_ndim": self._orig_ndim,
            "_granularity": self._granularity,
            "_block_size": self._block_size,
            "_keep_trans_cache": self._keep_trans_cache,
            "_scaling_recipe": self._scaling_recipe,
            "_scaling_recipe_for_trans": self._scaling_recipe_for_trans,
        }
        return list(tensors.keys()), metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        data = inner_tensors["_data"]
        self = torch.Tensor._make_wrapper_subclass(
            QuantizedTensor,
            outer_size,
            strides=outer_stride,
            dtype=metadata["_orig_dtype"],
            device=data.device,
        )
        self._orig_dtype = metadata["_orig_dtype"]
        self._dest_dtype = metadata["_dest_dtype"]
        self._orig_ndim = metadata["_orig_ndim"]
        self._granularity = metadata["_granularity"]
        self._block_size = metadata["_block_size"]
        self._keep_trans_cache = metadata["_keep_trans_cache"]
        self._scaling_recipe = metadata["_scaling_recipe"]
        self._scaling_recipe_for_trans = metadata["_scaling_recipe_for_trans"]
        self._data = data
        self._scale_inv = inner_tensors["_scale_inv"]
        self._data_t = inner_tensors.get("_data_t")
        self._scale_inv_t = inner_tensors.get("_scale_inv_t")
        return self

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
        data_t: Optional[torch.Tensor] = None,
        scale_inv_t: Optional[torch.Tensor] = None,
    ) -> "QuantizedTensor":
        """Construct a new ``QuantizedTensor`` sharing metadata with *tensor*
        but using the supplied *data*, *scale_inv* and *shape*.  Used by
        view / reshape dispatch to avoid a dequantize round-trip."""
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=tensor._orig_dtype,
            device=tensor.device,
            requires_grad=data.requires_grad,
        )
        out._orig_dtype = tensor._orig_dtype
        out._dest_dtype = tensor._dest_dtype
        out._orig_ndim = tensor._orig_ndim
        out._keep_trans_cache = tensor._keep_trans_cache
        out._scaling_recipe = tensor._scaling_recipe
        out._granularity = tensor._granularity
        out._block_size = tensor._block_size
        out._scaling_recipe_for_trans = tensor._scaling_recipe_for_trans
        out._data = data
        out._scale_inv = scale_inv
        out._data_t = data_t
        out._scale_inv_t = scale_inv_t
        return out

    @staticmethod
    def _view_data_and_transpose(tensor: QuantizedTensor, target_shape: torch.Size, op: Callable):
        """Apply *op* (``view`` or ``reshape``) to ``_data``, ``_scale_inv``
        and, when compatible, to the transpose cache.  Returns
        ``(out_data, out_scale_inv, out_shape, out_data_t, out_scale_inv_t)``.

        MX_BLOCKWISE quantization kernels may pad ``_data``'s inner
        dimension to a multiple of the format-specific padding alignment,
        so ``_data.shape`` can differ from the wrapper's logical shape.
        This method transparently computes a padded target shape for
        ``_data`` (and its col-quant counterpart) in that case.  The scale
        shape is derived from the new data shape using the C++ formula::

            scale = [*data_leading_dims, cdiv(data_last * packing, BLOCK_SIZE)]

        Supports both 2D wrappers and 3D batched wrappers (e.g. stacked
        weights ``[G, K, N]``).  ``_data_t`` shares the same physical layout
        as ``_data`` (it is a col-quantized version, not a transposed one),
        so the same target shape is applied to both.

        The target shape must keep the same rank as the wrapper's original
        ndim (``2D-to-2D`` or ``3D-to-3D``); cross-rank reshapes are
        rejected because they would invalidate the layout-dependent scale
        shape and the transpose cache.
        """
        assert len(target_shape) == tensor._orig_ndim, (
            f"view/reshape must preserve the original ndim "
            f"({tensor._orig_ndim}D), got target ndim {len(target_shape)}D"
        )

        out_shape = torch.Size(target_shape)
        wrapper_shape = tensor.shape

        if out_shape == wrapper_shape:
            return (
                tensor._data,
                tensor._scale_inv,
                out_shape,
                tensor._data_t,
                tensor._scale_inv_t,
            )

        align = _get_padding_align_size(tensor)
        padded_target_shape = _pad_inner_dim(target_shape, align)

        assert (
            tensor._data.numel() == torch.Size(padded_target_shape).numel()
        ), "data numel and padded_target_shape must have the same number of elements"

        out_data = op(tensor._data, *padded_target_shape)

        out_scale_inv = tensor._scale_inv
        if (
            tensor._granularity == ScalingGranularity.MX_BLOCKWISE
            or tensor._granularity == ScalingGranularity.BLOCKWISE
        ) and tensor._scale_inv is not None:
            block_size = tensor._block_size
            packing = _get_packing_factor(tensor)
            scale_target_shape = _compute_scale_shape(padded_target_shape, block_size, packing)
            out_scale_inv = op(tensor._scale_inv, *scale_target_shape)

        out_data_t = tensor._data_t
        out_scale_inv_t = tensor._scale_inv_t
        if out_data_t is not None:
            # ``_data_t`` is the col-quantized counterpart at the SAME
            # physical layout as ``_data`` — apply the identical target
            # shape, no last-two-dim swap.
            assert (
                out_data_t.numel() == torch.Size(padded_target_shape).numel()
            ), "data_t numel and padded_target_shape must have the same number of elements"

            out_data_t = op(out_data_t, *padded_target_shape)
            if (
                tensor._granularity == ScalingGranularity.MX_BLOCKWISE
                or tensor._granularity == ScalingGranularity.BLOCKWISE
            ) and out_scale_inv_t is not None:
                # TODO(ruibin): ``_compute_scale_shape`` assumes axis=-1
                # chunked scale layout (row-quant).  ``_scale_inv_t`` is
                # chunked along axis=-2 (col-quant), so its shape differs.
                # Wrapper-input BLOCKWISE/MX_BLOCKWISE is not supported yet
                # (see grouped_gemm_fp8 BlockFunc), so we keep the existing
                # call here as a placeholder; revisit when that path lands.
                scale_t_target_shape = _compute_scale_shape(
                    padded_target_shape, tensor._block_size, _get_packing_factor(tensor)
                )
                out_scale_inv_t = op(out_scale_inv_t, *scale_t_target_shape)

        return out_data, out_scale_inv, out_shape, out_data_t, out_scale_inv_t

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
                out_data, out_sinv, out_shape, out_data_t, out_sinv_t = cls._view_data_and_transpose(
                    tensor, args[1], torch.Tensor.view
                )
                return cls._make_like(
                    tensor,
                    data=out_data,
                    scale_inv=out_sinv,
                    shape=out_shape,
                    data_t=out_data_t,
                    scale_inv_t=out_sinv_t,
                )

        if func == torch.ops.aten.reshape.default:
            tensor = args[0]
            if isinstance(tensor, QuantizedTensor):
                out_data, out_sinv, out_shape, out_data_t, out_sinv_t = cls._view_data_and_transpose(
                    tensor, args[1], torch.Tensor.reshape
                )
                return cls._make_like(
                    tensor,
                    data=out_data,
                    scale_inv=out_sinv,
                    shape=out_shape,
                    data_t=out_data_t,
                    scale_inv_t=out_sinv_t,
                )

        def unwrap(t):
            if isinstance(t, QuantizedTensor):
                return t.dequantize()
            return t

        args_unwrapped = torch.utils._pytree.tree_map(unwrap, args)
        kwargs = kwargs or {}
        kwargs_unwrapped = torch.utils._pytree.tree_map(unwrap, kwargs)
        return func(*args_unwrapped, **kwargs_unwrapped)

    __torch_function__ = torch._C._disabled_torch_function_impl


class GroupedQuantizedTensor(QuantizedTensor):
    """Packed-M FP8 wrapper for grouped GEMM activations.

    Layout
    ------
    ``data`` is 2D ``[sum(group_lens), K]``; ``group_lens[g]`` is the number
    of tokens routed to expert ``g``.  This is the standard *packed-M*
    activation layout consumed by grouped GEMM kernels (e.g.
    ``grouped_gemm_fp8``).

    Quantization semantics
    ----------------------
    Today's quantization is *group-agnostic* — group boundaries do not
    influence the scales:

    - ``ROWWISE`` / ``BLOCKWISE``: scales are inherently per-row / per-block,
      so packed-M is a drop-in for plain 2D.
    - ``TENSORWISE``: a *single global* scale is used (matches existing
      ``grouped_gemm_fp8`` behaviour).  Per-group TENSORWISE amax (one scale
      per group, requires a custom kernel) is on the roadmap; the API is
      structured so that switching to it later is a kernel-only change.

    The class therefore exists primarily to (a) carry ``group_lens`` /
    ``group_offs`` metadata alongside the FP8 buffers, (b) preserve packed-M
    semantics through view / reshape / transpose, and (c) provide a single
    ``isinstance(..., GroupedQuantizedTensor)`` hook for downstream kernels
    to dispatch on.

    All quantization / dequantization / transpose / dispatch logic is
    inherited unchanged from :class:`QuantizedTensor`; only the constructor,
    serialization helpers, and ``_make_like`` are specialised here so that
    ``group_lens`` / ``group_offs`` flow through correctly.
    """

    @staticmethod
    def __new__(
        cls,
        data: Union[torch.Tensor, torch.nn.Parameter],
        dest_dtype: torch.dtype,
        granularity: ScalingGranularity,
        group_lens: torch.Tensor,
        block_size: Optional[int] = None,
        scaling_recipe: Optional[ScalingRecipe] = None,
        scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
        keep_trans_cache: bool = False,
    ):
        assert data.ndim == 2, f"GroupedQuantizedTensor expects a 2D packed-M tensor, got {data.ndim}D"
        assert (
            dest_dtype in _SUPPORTED_QUANTIZED_DTYPES and dest_dtype != float4_e2m1fn_x2
        ), "Unsupported quantized dtype (FP4 not supported for grouped activations)"
        assert group_lens.ndim == 1, "group_lens must be a 1D tensor of shape [G]"

        assert granularity in [
            ScalingGranularity.ROWWISE,
            ScalingGranularity.TENSORWISE,
        ], "GroupedQuantizedTensor only supports ROWWISE and TENSORWISE granularity"

        orig_dtype = data.dtype

        data_, scale_inv, data_t, scale_inv_t = None, None, None, None
        if keep_trans_cache:
            data_, scale_inv, data_t, scale_inv_t = QuantizedTensor._quantize_with_trans(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                scaling_recipe=scaling_recipe,
                scaling_recipe_for_trans=scaling_recipe_for_trans,
            )
        else:
            data_, scale_inv = QuantizedTensor._quantize(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                axis=-1,
                scaling_recipe=scaling_recipe,
            )

        # See QuantizedTensor.__new__ for why we use the *original* tensor's
        # size / stride for the wrapper (kernels may pad inner-dim).
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._orig_dtype = orig_dtype
        self._dest_dtype = dest_dtype
        # Always 2 here (asserted above); kept for symmetry with the base
        # class and so view/reshape can use the same _orig_ndim check.
        self._orig_ndim = data.ndim
        self._keep_trans_cache = keep_trans_cache
        self._scaling_recipe = scaling_recipe
        self._granularity = granularity
        self._block_size = block_size
        self._scaling_recipe_for_trans = scaling_recipe_for_trans

        self._data, self._scale_inv = data_, scale_inv
        self._data_t, self._scale_inv_t = data_t, scale_inv_t

        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
            group_offs_from_lens,
        )

        self._group_lens = group_lens
        self._group_offs = group_offs_from_lens(group_lens)

        return self

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def group_lens(self) -> torch.Tensor:
        return self._group_lens

    @property
    def group_offs(self) -> torch.Tensor:
        return self._group_offs

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # type: ignore[override]
        has_trans = self._data_t is not None
        return (
            f"GroupedQuantizedTensor(shape={list(self.shape)}, "
            f"group_lens={self._group_lens.tolist()}, orig_dtype={self._orig_dtype}, "
            f"real_dtype={self._data.dtype}, granularity={self._granularity.name}"
            f"{', has_trans_cache' if has_trans else ''})"
        )

    # ------------------------------------------------------------------
    # Serialisation (torch.compile / FSDP)
    # ------------------------------------------------------------------
    def __tensor_flatten__(self):
        keys, metadata = super().__tensor_flatten__()
        metadata["_group_lens"] = self._group_lens
        return keys, metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
            group_offs_from_lens,
        )

        data = inner_tensors["_data"]
        self = torch.Tensor._make_wrapper_subclass(
            GroupedQuantizedTensor,
            outer_size,
            strides=outer_stride,
            dtype=metadata["_orig_dtype"],
            device=data.device,
        )
        self._orig_dtype = metadata["_orig_dtype"]
        self._dest_dtype = metadata["_dest_dtype"]
        self._orig_ndim = metadata["_orig_ndim"]
        self._granularity = metadata["_granularity"]
        self._block_size = metadata["_block_size"]
        self._keep_trans_cache = metadata["_keep_trans_cache"]
        self._scaling_recipe = metadata["_scaling_recipe"]
        self._scaling_recipe_for_trans = metadata["_scaling_recipe_for_trans"]
        self._data = data
        self._scale_inv = inner_tensors["_scale_inv"]
        self._data_t = inner_tensors.get("_data_t")
        self._scale_inv_t = inner_tensors.get("_scale_inv_t")

        # _group_offs is derived; recompute to keep the wrapper self-consistent
        # after torch.compile / FSDP round-trips.
        self._group_lens = metadata["_group_lens"]
        self._group_offs = group_offs_from_lens(metadata["_group_lens"])
        return self

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
        data_t: Optional[torch.Tensor] = None,
        scale_inv_t: Optional[torch.Tensor] = None,
    ) -> "QuantizedTensor":
        """Same as :meth:`QuantizedTensor._make_like`, but additionally
        propagates ``group_lens`` / ``group_offs``.

        ``tensor`` is typed as ``QuantizedTensor`` to satisfy override
        compatibility with the base class; at runtime it is asserted to be a
        ``GroupedQuantizedTensor`` so we can read its group metadata.  The
        return type is similarly widened to ``QuantizedTensor`` even though
        the underlying ``_make_wrapper_subclass(cls, ...)`` produces a
        ``GroupedQuantizedTensor`` instance.
        """
        assert isinstance(
            tensor, GroupedQuantizedTensor
        ), "GroupedQuantizedTensor._make_like requires a GroupedQuantizedTensor"
        # NOTE: must go through ``super()`` (not ``QuantizedTensor._make_like``
        # directly) so that the classmethod ``cls`` stays bound to
        # ``GroupedQuantizedTensor`` — otherwise ``_make_wrapper_subclass(cls,
        # ...)`` inside the parent would construct a plain ``QuantizedTensor``.
        out = super()._make_like(
            tensor,
            data=data,
            scale_inv=scale_inv,
            shape=shape,
            data_t=data_t,
            scale_inv_t=scale_inv_t,
        )
        out._group_lens = tensor._group_lens
        out._group_offs = tensor._group_offs
        return out


class _ViewFunc(torch.autograd.Function):
    """View a QuantizedTensor without dequantizing."""

    @staticmethod
    def forward(ctx, tensor: QuantizedTensor, shape: Tuple[int, ...]) -> QuantizedTensor:
        ctx.shape = tensor.shape
        out_data, out_sinv, out_shape, out_data_t, out_sinv_t = QuantizedTensor._view_data_and_transpose(
            tensor, shape, torch.Tensor.view
        )
        # Dispatch on the runtime type so subclasses (e.g.
        # ``GroupedQuantizedTensor``) keep their identity — and their
        # extra metadata such as ``group_lens`` / ``group_offs`` — through
        # the view.
        return type(tensor)._make_like(
            tensor,
            data=out_data,
            scale_inv=out_sinv,
            shape=out_shape,
            data_t=out_data_t,
            scale_inv_t=out_sinv_t,
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        return grad.reshape(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape a QuantizedTensor without dequantizing."""

    @staticmethod
    def forward(ctx, tensor: QuantizedTensor, shape: Tuple[int, ...]) -> QuantizedTensor:
        ctx.shape = tensor.shape
        out_data, out_sinv, out_shape, out_data_t, out_sinv_t = QuantizedTensor._view_data_and_transpose(
            tensor, shape, torch.Tensor.reshape
        )
        # See ``_ViewFunc.forward`` — preserve the subclass identity.
        return type(tensor)._make_like(
            tensor,
            data=out_data,
            scale_inv=out_sinv,
            shape=out_shape,
            data_t=out_data_t,
            scale_inv_t=out_sinv_t,
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        return grad.reshape(ctx.shape), None
