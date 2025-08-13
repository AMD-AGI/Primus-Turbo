###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math

import torch
import torch.nn as nn

from primus_turbo.pytorch.ops import grouped_gemm

__all__ = ["GroupedLinear"]


class GroupedLinear(torch.nn.Module):
    def __init__(
        self,
        group_num: int,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_num = group_num
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((group_num, self.out_features, self.in_features), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return grouped_gemm(x, self.weight, group_lens, group_offs, trans_b=True)

    def extra_repr(self) -> str:
        return f"group_num={self.group_num},in_features={self.in_features}, out_features={self.out_features}"
