import math

import torch
import torch.nn as nn

from primus_turbo.pytorch.ops import gemm

__all__ = ["Linear"]


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flatten_shapes = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        out_2d = gemm(x_2d, self.weight, False, True, self.dtype)
        if self.bias is not None:
            out_2d = out_2d + self.bias
        return out_2d.view(*flatten_shapes, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        )
