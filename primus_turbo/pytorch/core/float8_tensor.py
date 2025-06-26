import torch

from primus_turbo.pytorch.core.float8 import Float8QuantConfig


# TODO: This Class is still develop.
class Float8Tensor(torch.Tensor):
    """
    Float8 Tensor.
    Contains:
    * `_data`: quantized float8 data
    * `_scale`: scale tensor
    * `_orig_dtype`: original data type
    * `_fp8_dtype`: fp8 data type
    * `_config`: quantization config
    """

    __slots__ = ["_data", "_scale", "_orig_dtype", "_fp8_dtype", "_config"]

    def __repr__(self) -> str:
        return (
            f"Float8Tensor(shape={self.shape}, dtype={self._orig_dtype}, "
            f"float8_dtype={self._fp8_dtype}, config={self._config})"
        )

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        fp8_dtype: torch.dtype,
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
        self._fp8_dtype = fp8_dtype
        self._config = config
        return self

    @staticmethod
    def from_tensor(x: torch.Tensor, fp8_dtype: torch.dtype, config: Float8QuantConfig):
        # TODO
        raise NotImplementedError("from_tensor is not implemented yet.")

    def dequantize(self) -> torch.Tensor:
        # TODO
        raise NotImplementedError("dequantize is not implemented yet.")

    def __tensor_flatten__(self):
        """
        Return the list of tensor attributes and non-tensor metadata for serialization.
        """
        return ["_data", "_scale"], {
            "_orig_dtype": self._orig_dtype,
            "_fp8_dtype": self._fp8_dtype,
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
            fp8_dtype=metadata["_fp8_dtype"],
            config=metadata["_config"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Placeholder for future Float8-aware operator support.
        """
        raise NotImplementedError(f"{func} is not yet supported for Float8Tensor.")

    __torch_function__ = torch._C._disabled_torch_function_impl
