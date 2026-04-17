###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP8_BLOCK_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.ops.quantization import (
    dequantize_fp8,
    quantize_fp8,
    quantize_fp8_with_trans,
)

_SUPPORTED_QUANTIZED_DTYPES = [float8_e4m3, float8_e5m2, float4_e2m1fn_x2]


class QuantizedTensor(torch.Tensor):
    """Wrapper subclass that carries low-precision quantized data, scale_inv"""

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        dest_dtype: torch.dtype,
        granularity: ScalingGranularity,
        block_size: Optional[int] = None,
        scaling_recipe: Optional[ScalingRecipe] = None,
        scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
        keep_trans_cache: bool = False,
    ):
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
                    axis=1,
                    scaling_recipe=scaling_recipe,
                )
                data_t, scale_inv_t = quantize_fp8(
                    data,
                    dest_dtype,
                    granularity,
                    block_size=block_size,
                    axis=0,
                    scaling_recipe=scaling_recipe_for_trans,
                )
        else:
            data_, scale_inv = quantize_fp8(
                data,
                dest_dtype,
                granularity,
                block_size=block_size,
                axis=1,
                scaling_recipe=scaling_recipe,
            )

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data_.size(),
            strides=data_.stride(),
            storage_offset=data_.storage_offset(),
            dtype=orig_dtype,
            layout=data_.layout,
            requires_grad=data.requires_grad,
            device=data_.device,
        )
        self._orig_dtype = orig_dtype
        self._dest_dtype = dest_dtype

        self._keep_trans_cache = keep_trans_cache
        self._scaling_recipe = scaling_recipe
        self._granularity = granularity
        self._block_size = block_size
        self._scaling_recipe_for_trans = scaling_recipe_for_trans

        self._data, self._scale_inv = data_, scale_inv
        self._data_t, self._scale_inv_t = data_t, scale_inv_t

        return self

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def scale_inv(self) -> torch.Tensor:
        return self._scale_inv

    def t(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._data_t is None:
            # TODO(ruibin): eliminate the transpose operation by low precision transpose kernel.
            orig_data = self.dequantize()
            orig_data_t = orig_data.t().contiguous()

            data_t, scale_inv_t = quantize_fp8(
                orig_data_t,
                self._orig_dtype,
                self._granularity,
                block_size=self._block_size,
                axis=1,
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

    def dequantize(self) -> torch.Tensor:
        """Dequantize back to the original high-precision dtype."""
        return dequantize_fp8(
            self._data,
            self._orig_dtype,
            self._granularity,
            block_size=self._block_size,
            axis=1,
            scale_inv=self._scale_inv,
            scaling_recipe=self._scaling_recipe,
        )

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
    # Dispatch hooks
    # ------------------------------------------------------------------
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, QuantizedTensor):
                return t.dequantize()
            return t

        args_unwrapped = torch.utils._pytree.tree_map(unwrap, args)
        kwargs = kwargs or {}
        kwargs_unwrapped = torch.utils._pytree.tree_map(unwrap, kwargs)
        return func(*args_unwrapped, **kwargs_unwrapped)

    __torch_function__ = torch._C._disabled_torch_function_impl
