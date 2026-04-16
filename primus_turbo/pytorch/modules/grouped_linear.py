###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GroupedLinear module.

Mirrors TransformerEngine's ``GroupedLinear`` API but with Primus-Turbo's
contiguous-weight convention: a **single** ``[num_gemms, out_features,
in_features]`` parameter instead of TE's per-GEMM discrete weight tensors.

# TODO: support FP8 weight quantisation (fp8 weight)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm

__all__ = ["GroupedLinear"]


class GroupedLinear(torch.nn.Module):
    """Applies grouped linear transformations ``y_g = x_g A_g^T + b_g``.

    Unlike TransformerEngine which stores one ``[out, in]`` parameter per GEMM,
    Primus-Turbo keeps a **single contiguous** weight of shape
    ``[num_gemms, out_features, in_features]``.

    Parameters
    ----------
    num_gemms : int
        Number of GEMMs (groups) to perform simultaneously.
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default = True
        If ``False`` the layer will not learn an additive bias.
    device : torch.device | str | None
        Device for parameter allocation.
    dtype : torch.dtype | None
        Data type for parameter allocation.
    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(
            torch.empty(
                (num_gemms, out_features, in_features),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight[0]
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return grouped_gemm(
            x,
            self.weight,
            group_lens,
            group_offs,
            trans_b=True,
            bias=self.bias,
        )

    def extra_repr(self) -> str:
        return (
            f"num_gemms={self.num_gemms}, "
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
