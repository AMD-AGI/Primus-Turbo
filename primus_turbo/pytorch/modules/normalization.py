###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import numbers
from typing import Optional, Union

import torch
from torch import Size
from torch.nn.parameter import Parameter

from primus_turbo.pytorch.ops.normalization import rmsnorm as _rmsnorm

__all__ = ["RMSNorm"]


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        normalized_shape: Union[int, list[int], Size],
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        zero_centered_gamma: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        if len(self.normalized_shape) != 1:
            raise ValueError(
                "primus_turbo RMSNorm currently only supports a 1-D normalized_shape "
                f"(got {self.normalized_shape}). Use the last hidden dimension."
            )
        self.eps = eps if eps is not None else 1e-6
        self.elementwise_affine = elementwise_affine
        # zero_centered_gamma: the effective gain is (1 + weight), so the weight is
        # initialized to zeros (effective gain == 1).
        self.zero_centered_gamma = zero_centered_gamma
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            if self.zero_centered_gamma:
                torch.nn.init.zeros_(self.weight)
            else:
                torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rmsnorm(x, self.weight, self.eps, self.zero_centered_gamma)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, "
            "zero_centered_gamma={zero_centered_gamma}".format(**self.__dict__)
        )
