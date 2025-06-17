from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class Float8ScalingGranularity(Enum):
    TENSORWISE = auto()
    ROWWISE = auto()
    BLOCKWISE = auto()


class Float8ScalingStrategy(Enum):
    DYNAMIC = auto()
    # DELAYED_SCALING = auto() # TODO: undetermined


@dataclass
class Float8QuantConfig:
    dtype: torch.dtype = torch.float8_e4m3fn  # TODO: HIP
    granularity: Float8ScalingGranularity = Float8ScalingGranularity.TENSORWISE
    strategy: Float8ScalingStrategy = Float8ScalingStrategy.DYNAMIC
    block_size: Optional[int] = None


class Float8Tensor(torch.Tensor):
    """
    Float8 Tensor.
    Contains:
    * `_data`: quantized float8 data
    * `_scale`: scale tensor
    * `_orig_dtype`: original data type
    * `_config`: quantization config
    """

    __slots__ = ["_data", "_scale", "_orig_dtype", "_config"]

    def __repr__(self) -> str:
        return (
            f"Float8Tensor(shape={self.shape}, dtype={self._orig_dtype}, "
            f"float8_dtype={self._data.dtype}, config={self._config})"
        )

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
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
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype
        self._config = config
        return self

    def dequantize(self) -> torch.Tensor:
        # TODO
        raise NotImplementedError("dequantize is not implemented yet.")

    def __tensor_flatten__(self):
        """
        Return the list of tensor attributes and non-tensor metadata for serialization.
        """
        return ["_data", "_scale"], {
            "_orig_dtype": self._orig_dtype,
            "_config": self._config,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        """
        Reconstruct the Float8Tensor from tensors and metadata.
        """
        return Float8Tensor(
            data=inner_tensors["_data"],
            scale=inner_tensors["_scale"],
            orig_dtype=metadata["_orig_dtype"],
            config=metadata["_config"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Placeholder for future Float8-aware operator support.
        """
        raise NotImplementedError(f"{func} is not yet supported for Float8Tensor.")

    __torch_function__ = torch._C._disabled_torch_function_impl
