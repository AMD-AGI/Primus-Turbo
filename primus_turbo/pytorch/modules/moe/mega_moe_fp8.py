###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``MegaMoEFP8``: standalone stateful holder for all-fp8 (mxfp8) fused mega-MoE expert compute.

SKELETON. Unlike the Primus-Turbo source (where ``MegaMoEFP8`` subclasses a bf16 ``MegaMoE`` base
module), this repo has no such base module, so this is a *standalone minimal* module: it owns only
the expert weights + EP group + the version-keyed fp8 weight-quant cache, and exposes
``expert_compute``. Routing / shared-expert / token bookkeeping are the caller's (training
framework's) responsibility.

Why this module exists: fp8 perf depends on quantizing each static weight ONCE per ``optim.step``
and reusing it across a gradient-accumulation window. This module maintains that state
(``_w1_fp8`` / ``_w2_fp8`` / ``_w2t_fp8`` / ``_w1t_fp8``), re-quantizing a weight only when its
``_version`` changes, and hands the preps to :func:`mega_moe_fused_fp8`. ``w1`` / ``w2`` stay bf16
``Parameter`` s and remain the differentiable inputs.
"""

from typing import Callable, Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.pytorch.ops.moe.mega_moe_fused_fp8 import (
    mega_moe_fused_fp8,
    prepare_w1t_dgrad_fp8,
    prepare_w2t_dgrad_fp8,
)

# --- PORT: the forward weight-prep producers land with the fp8 flydsl kernels ---
# from primus_turbo.flydsl.mega.fp8 import (
#     prepare_w2_fp8,                        # w2 -> (weight_flat, b_sp): quant + scale preshuffle (combine)
#     quantize_grouped_weight_mxfp8_flydsl,  # w1 -> (w1q, w1s): grouped quant (dispatch)
# )

__all__ = ["MegaMoEFP8"]

_NOT_PORTED = (
    "mega MoE fp8 weight producers not ported yet; see "
    "/perf_apps/xiaoming/Primus-Turbo/primus_turbo/pytorch/modules/moe/mega_moe_fp8.py"
)


class MegaMoEFP8(torch.nn.Module):
    """Standalone all-fp8 (mxfp8) fused mega-MoE expert compute with version-keyed weight quant.

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
        # version-keyed fp8 weight-prep cache entries: (attr) -> (w._version, produced_tuple)
        self._w1_fp8: Optional[Tuple[int, Tuple]] = None
        self._w2_fp8: Optional[Tuple[int, Tuple]] = None
        self._w2t_fp8: Optional[Tuple[int, Tuple]] = None
        self._w1t_fp8: Optional[Tuple[int, Tuple]] = None

    def _cached_weight(self, w: torch.Tensor, attr: str, produce: Callable) -> Tuple:
        """Maintain a static weight's fp8 prep as module state, recomputed only when ``w._version``
        changes (``optim.step()`` bump) and reused across a grad-accum window. Forward-only."""
        v = getattr(w, "_version", 0)
        ent = getattr(self, attr, None)
        if ent is None or ent[0] != v:
            with torch.no_grad():
                out = produce(w)
            ent = (v, out)
            setattr(self, attr, ent)
        return ent[1]

    def expert_compute(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fused mxfp8 dispatch -> L1 -> SwiGLU -> L2 fp8 combine -> reduce; returns y ``[T, hidden]``.

        Weight fp8 prep is maintained here (module-owned, version-keyed): ``w1`` grouped quant for
        L1 dispatch, ``w2`` quant + scale preshuffle for the L2 combine, and -- when training -- the
        ``w2^T`` / ``w1^T`` grouped quant for the backward fc2 / fc1 dgrad. All computed once per
        ``optim.step`` and reused across grad-accum; only token-dependent work goes to the op.
        """
        # PORT: wire the producers once the fp8 flydsl kernels land, e.g.
        #   w1_fp8 = self._cached_weight(self.w1, "_w1_fp8", quantize_grouped_weight_mxfp8_flydsl)
        #   w2_fp8 = self._cached_weight(self.w2, "_w2_fp8", prepare_w2_fp8)
        #   if torch.is_grad_enabled():
        #       w2t_fp8 = self._cached_weight(self.w2, "_w2t_fp8", prepare_w2t_dgrad_fp8)
        #       w1t_fp8 = self._cached_weight(self.w1, "_w1t_fp8", prepare_w1t_dgrad_fp8)
        #   else:
        #       w2t_fp8 = w1t_fp8 = None
        #   return mega_moe_fused_fp8(
        #       self.ep_group, x, topk_idx, topk_weights, self.w1, self.w2,
        #       block_m=self.block_m, block_n=self.block_n,
        #       w1_fp8=w1_fp8, w2_fp8=w2_fp8, w2t_fp8=w2t_fp8, w1t_fp8=w1t_fp8,
        #   )
        raise NotImplementedError(_NOT_PORTED)

    def forward(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.expert_compute(x, topk_idx, topk_weights)
