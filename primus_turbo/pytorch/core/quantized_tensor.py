###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    MXScalingRecipe,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops.quantization import (
    dequantize_fp8,
    quantize_fp8,
    quantize_fp8_with_trans,
)

SHUFFLE_LAYOUT = [16, 16]


class QuantizedTensor(torch.Tensor):
    """Wrapper subclass that carries low-precision quantized data, scale, and
    optionally their hardware-shuffled counterparts.

    Attributes
    ----------
    data            : Quantized low-precision data (e.g. fp8).
    scale            : Per-block / per-tensor inverse scale.
    data_t           : Transposed quantized data.
    scale_t          : Transposed per-block / per-tensor inverse scale.
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        dest_dtype: torch.dtype,
        config: Float8QuantConfig,
        with_trans: bool = False,
        scaling_recipe: Optional[MXScalingRecipe] = None,
        scaling_recipe_for_trans: Optional[MXScalingRecipe] = None,
    ):
        assert isinstance(config, Float8QuantConfig), "config must be a Float8QuantConfig"

        orig_dtype = data.dtype

        data_, scale, data_t, scale_t = None, None, None, None
        if with_trans:
            if config.granularity == ScalingGranularity.MX_BLOCKWISE:
                data_, scale, data_t, scale_t = quantize_fp8_with_trans(
                    data,
                    dest_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    scaling_recipe=scaling_recipe,
                    scaling_recipe_for_trans=scaling_recipe_for_trans,
                )
            else:
                data_, scale = quantize_fp8(
                    data,
                    dest_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    axis=1,
                    scaling_recipe=scaling_recipe,
                )
                data_t, scale_t = quantize_fp8(
                    data.t(),
                    dest_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    axis=0,
                    scaling_recipe=scaling_recipe_for_trans,
                )
        else:
            data_, scale = quantize_fp8(
                data,
                dest_dtype,
                config.granularity,
                block_size=config.block_size,
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

        self._config = config
        self._scaling_recipe = scaling_recipe
        self._scaling_recipe_for_trans = scaling_recipe_for_trans

        self._data, self._scale = data_, scale
        self._data_t, self._scale_t = data_t, scale_t

        return self

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def t(self) -> Union[torch.Tensor, torch.Tensor]:
        assert self._data_t is not None, "data_t is not quantized"
        assert self._scale_t is not None, "scale_t is not quantized"

        return self._data_t, self._scale_t

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        return self._orig_dtype

    @property
    def real_dtype(self) -> torch.dtype:
        return self._data.dtype

    def dequantize(self) -> torch.Tensor:
        """Dequantize back to the original high-precision dtype."""
        return dequantize_fp8(
            self._data,
            self._orig_dtype,
            self._config.granularity,
            block_size=self._config.block_size,
            axis=1,
            scale_inv=self._scale,
            scaling_recipe=self._scaling_recipe,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        has_trans = self._data_t is not None
        return (
            f"QuantizedTensor(shape={list(self.shape)}, orig_dtype={self._orig_dtype}, "
            f"real_dtype={self._data.dtype}, granularity={self._config.granularity.name}"
            f"{', has_transpose' if has_trans else ''})"
        )

    # ------------------------------------------------------------------
    # Serialisation (torch.compile / FSDP)
    # ------------------------------------------------------------------
    def __tensor_flatten__(self):
        tensors = {"_data": self._data, "_scale": self._scale}
        if self._data_t is not None:
            tensors["_data_t"] = self._data_t
        if self._scale_t is not None:
            tensors["_scale_t"] = self._scale_t
        metadata = {
            "_orig_dtype": self._orig_dtype,
            "_config": self._config,
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
        self._config = metadata["_config"]
        self._scaling_recipe = metadata["_scaling_recipe"]
        self._scaling_recipe_for_trans = metadata["_scaling_recipe_for_trans"]
        self._data = data
        self._scale = inner_tensors["_scale"]
        self._data_t = inner_tensors.get("_data_t")
        self._scale_t = inner_tensors.get("_scale_t")
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
