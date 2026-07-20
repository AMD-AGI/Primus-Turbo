###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``MegaMoEFP8``: standalone module for all-fp8 (mxfp8) fused mega-MoE expert compute.

Minimal standalone module: it owns only the bf16 expert weights + the EP group and exposes
``expert_compute``. Routing / shared-expert / token bookkeeping are the caller's (training
framework's) responsibility.

Weight fp8 quantization is NOT maintained here -- :func:`mega_moe_fused_fp8` does it internally
with a version-keyed cache (each static weight is re-quantized only when its ``_version`` changes,
i.e. after ``optim.step``, and reused across a gradient-accumulation window). So this module just
holds the bf16 ``Parameter`` s (the differentiable inputs) and forwards to the op.
"""

import torch
from torch.distributed import ProcessGroup

from primus_turbo.pytorch.ops.moe.mega_moe_fused_fp8 import mega_moe_fused_fp8

__all__ = ["MegaMoEFP8"]


class MegaMoEFP8(torch.nn.Module):
    """Standalone all-fp8 (mxfp8) fused mega-MoE expert compute.

    Args:
        w1: bf16 grouped fc1 weight ``[G, 2I, H]`` (up/gate). Held as a ``Parameter``.
        w2: bf16 grouped fc2 weight ``[G, H, I]`` (down). Held as a ``Parameter``.
        ep_group: the expert-parallel process group.
        block_m / block_n: FlyDSL tiling for the fused kernels.

    Shape requirements of the mxfp8 op (asserted inside the op): ``hidden % 1024 == 0`` (fp8 push)
    / ``% 256`` (weight N), ``intermediate % 1024 == 0`` (SwiGLU). Otherwise use the bf16 path.
    """

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        ep_group: ProcessGroup,
        *,
        block_m: int = 256,
        block_n: int = 256,
    ) -> None:
        super().__init__()
        assert w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16, "w1/w2 must be bf16"
        self.w1 = torch.nn.Parameter(w1)
        self.w2 = torch.nn.Parameter(w2)
        self.ep_group = ep_group
        self.block_m = block_m
        self.block_n = block_n

    def expert_compute(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fused mxfp8 dispatch -> L1 -> SwiGLU -> L2 fp8 combine -> reduce; returns y ``[T, hidden]``.

        Only token-dependent work goes to the op; the invariant weight quant is maintained inside
        the op (version-keyed), so nothing is prepared here.
        """
        return mega_moe_fused_fp8(
            self.ep_group, x, topk_idx, topk_weights, self.w1, self.w2,
            block_m=self.block_m, block_n=self.block_n,
        )

    def forward(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.expert_compute(x, topk_idx, topk_weights)
