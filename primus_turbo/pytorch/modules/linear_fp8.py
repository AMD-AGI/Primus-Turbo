###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.core.float8 import MXQuantConfig
from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8_blockwise

__all__ = ["MXLinear"]


class MXLinear(torch.nn.Linear):
    """ """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[MXQuantConfig] = None,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        if config is None:
            config = MXQuantConfig()
        self.config = config

    def forward(self, x):
        out = gemm_fp8_blockwise(
            x,
            self.weight,
            transA=False,
            transB=True,
            out_dtype=x.dtype,
            config=self.config,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        config: Optional[MXQuantConfig] = None,
    ):
        if config is None:
            config = MXQuantConfig()
        assert isinstance(mod, torch.nn.Linear), f"unsupported type(mod) {type(mod)}"
        assert isinstance(config, MXQuantConfig)
        mod.__class__ = MXLinear
        mod.config = config
        return mod

    def extra_repr(self):
        return f"{super().extra_repr()}, config={getattr(self, 'config', None)}"


# TODO: For tensorwise and rowwise FP8 GEMM
class Float8Linear(torch.nn.Linear):
    pass
