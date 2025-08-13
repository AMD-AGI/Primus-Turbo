###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import numbers
from typing import Optional, Union

import torch
from torch import Size
from torch.nn import functional as F
from torch.nn.parameter import Parameter

__all__ = ["RMSNorm"]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, list[int], Size],
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Extra information about the module.
        """
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )
