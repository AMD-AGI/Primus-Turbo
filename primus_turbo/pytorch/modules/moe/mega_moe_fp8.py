###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``MegaMoEFP8``: MegaMoE variant whose fused expert compute runs in all-fp8 (mxfp8)."""

from typing import Callable, Tuple

import torch

from primus_turbo.flydsl.mega.fp8.grouped_gemm_combine_fp8_kernel import prepare_w2_fp8
from primus_turbo.flydsl.mega.fp8.quant import quantize_grouped_weight_mxfp8_flydsl
from primus_turbo.pytorch.modules.moe.mega_moe import MegaMoE
from primus_turbo.pytorch.ops.moe.mega_moe_fused_mxfp8 import mega_moe_fused_mxfp8

__all__ = ["MegaMoEFP8"]


class MegaMoEFP8(MegaMoE):
    """MegaMoE with all-fp8 (mxfp8) fused expert compute.

    Identical routing, weights, shared-expert and ``forward`` as :class:`MegaMoE`. It swaps the
    expert compute (dispatch + fc1 / fc2 + combine) to the fused mxfp8 op ``mega_moe_fused_mxfp8``,
    AND owns the STATIC weight quantization: ``w1`` / ``w2`` stay bf16 Parameters, and this module
    maintains their per-1x32 E8M0 fp8 grouped quant as module state (``_w1_fp8`` / ``_w2_fp8``),
    re-quantizing a weight only when its ``_version`` changes (i.e. after ``optim.step()``). The op
    consumes the pre-quantized weights; the bf16 weights remain the differentiable inputs for the
    (bf16, dW2-optionally-fp8) backward. So per training step each weight is quantized once and
    reused across a gradient-accumulation window (all micro-steps share one ``_version``).

    Only token-dependent work (activation quant, dispatch prologue, dispatch+GEMM, swiglu, combine)
    runs every forward; the invariant weight quant is hoisted here. Forward-only fp8 for now.

    Shape requirements of the mxfp8 op (asserted inside): ``hidden % 1024 == 0`` (fp8 push) /
    ``% 256`` (weight N), ``intermediate % 1024 == 0`` (SwiGLU). Otherwise use :class:`MegaMoE` (bf16).
    """

    def _cached_weight(self, w: torch.Tensor, attr: str, produce: Callable) -> Tuple:
        """Maintain a static weight's fp8 prep as module state, recomputed only when ``w._version``
        changes (``optim.step()`` bump); reused across a grad-accum window. Forward-only (no grad).
        ``produce(w)`` returns the op-consumed form: w1 -> (w1q, w1s) [dispatch]; w2 ->
        (weight_flat, b_sp) [combine, quant+preshuffle already applied]."""
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
        """Fused mxfp8 dispatch -> L1 -> SwiGLU -> L2 -> fp8 combine -> reduce; y [T, hidden].

        Weight fp8 prep is maintained here (module-owned, version-keyed, FlyDSL): w1 grouped
        quant for L1 dispatch, and w2 quant + scale preshuffle for the L2 combine -- both computed
        once per optim.step and reused (grad-accum), so the op does NO per-call weight prep. Only
        token-dependent work goes to the op."""
        w1_fp8 = self._cached_weight(self.w1, "_w1_fp8", quantize_grouped_weight_mxfp8_flydsl)
        w2_fp8 = self._cached_weight(self.w2, "_w2_fp8", prepare_w2_fp8)
        return mega_moe_fused_mxfp8(
            self.ep_group, x, topk_idx, topk_weights, self.w1, self.w2,
            w1_fp8=w1_fp8, w2_fp8=w2_fp8,
        )
