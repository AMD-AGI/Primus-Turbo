###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.core.float8 import Float8QuantConfig, ScalingGranularity
from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8

__all__ = ["Float8Linear"]


class Float8Linear(torch.nn.Linear):
    """
    Linear layer with FP8 quantization.

    Supports multiple quantization granularities: TENSORWISE, ROWWISE, BLOCKWISE, MX_BLOCKWISE.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias term (default: True)
        config: Float8QuantConfig for quantization settings (default: TENSORWISE)
        **kwargs: Additional args passed to torch.nn.Linear (device, dtype, etc.)

    Examples:
        >>> # Default tensorwise quantization
        >>> linear = Float8Linear(512, 256)
        >>>
        >>> # Custom config
        >>> config = Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128)
        >>> linear = Float8Linear(512, 256, config=config)
        >>>
        >>> # Convert from existing Linear (in-place)
        >>> fp8_linear = Float8Linear.from_float(torch_linear, config)
        >>>
        >>> # Convert without modifying original
        >>> fp8_linear = Float8Linear.from_float(torch_linear, config, inplace=False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[Float8QuantConfig] = None,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        if config is None:
            config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE)
        self.config = config

    def forward(self, x):
        flatten_shapes = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)

        out_2d = gemm_fp8(
            x_2d, self.weight, trans_a=False, trans_b=True, out_dtype=x.dtype, config=self.config
        )

        if self.bias is not None:
            out_2d = out_2d + self.bias

        return out_2d.view(*flatten_shapes, self.out_features)

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod: torch.nn.Linear,
        config: Optional[Float8QuantConfig] = None,
        inplace: bool = True,
    ):
        """
        Convert a torch.nn.Linear module to Float8Linear.

        Args:
            mod: The Linear module to convert
            config: FP8 quantization config (default: TENSORWISE)
            inplace: If True, modify mod in-place; if False, create new module (default: True)

        Returns:
            Float8Linear: The converted module
        """
        assert isinstance(mod, torch.nn.Linear), f"Expected torch.nn.Linear, got {type(mod)}"

        if config is None:
            config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE)

        if inplace:
            # In-place conversion
            mod.__class__ = Float8Linear
            mod.config = config
            return mod
        else:
            # Create new module
            fp8_linear = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                config=config,
                device=mod.weight.device,
                dtype=mod.weight.dtype,
            )
            fp8_linear.weight.copy_(mod.weight)
            if mod.bias is not None:
                fp8_linear.bias.copy_(mod.bias)
            return fp8_linear

    def extra_repr(self):
        return f"{super().extra_repr()}, config={getattr(self, 'config', None)}"
