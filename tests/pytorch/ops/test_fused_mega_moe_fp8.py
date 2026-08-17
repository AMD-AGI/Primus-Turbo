###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Accuracy tests for the MXFP8 mega MoE against the turbo DeepEP MoE.

The fp8 sibling of ``test_fused_mega_moe.py``, and deliberately the same shape: same reference, same
metrics, same tensors compared, so the two files diff cleanly and an fp8-only regression shows up as a
threshold gap rather than a difference in method. The reference helpers are imported from that module
rather than copied, so both suites provably measure against the identical bf16 reference.

This drives the STAGED pair, because that is what training drives: Primus' ``MegaMoEFP8Experts``
(megatron/core/extensions/mega_moe.py) calls ``fused_mega_moe_fp8_stage1`` then
``fused_mega_moe_fp8_stage2``, and never the single fused ``fused_mega_moe_fp8``.

What this covers that the other fp8 tests do not: an END-TO-END accuracy gate against an INDEPENDENT
implementation. ``tests/pytorch/modules/test_mega_moe_mxfp8.py`` checks one stage (L1) against a torch
dequant GEMM, checks staged-against-fused -- that is, the op against ITSELF -- and checks a training
loop for finiteness only. None of those would catch the whole op drifting together.

The single fused entry is NOT covered here, and not by oversight. It defers the dW1 pool requant
(``prepare_dw1_pool_operand_fp8``) until after the L2 combine, while stage1 does it immediately after
L1. The pool is ``symm.pool_fp8``, which peers write cross-rank, and the combine is a cross-rank PUSH,
so in the fused order a peer's combine lands in the pool before this rank has snapshotted it: over 16
runs its dW1 came out at 22.5 dB (correct) or 2.9-8.9 dB (corrupted) about a third of the time, with
~8200 of 139264 pool rows -- one peer's worth -- observed to change between L1 and the requant. A test
that fails a third of the time teaches people to ignore failures, so the fused path stays out until
the requant moves up. Training is unaffected: the staged order has no such window, and its dW1
measured 22.52 dB on six consecutive runs.
"""

from __future__ import annotations

import unittest

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from primus_turbo.pytorch.core.low_precision import check_mxfp8_support
from primus_turbo.pytorch.core.utils import is_gfx1250

# The flydsl mega path is not supported on gfx1250; skip before importing the flydsl-backed modules.
if is_gfx1250():
    pytest.skip("mega_moe_fused is not supported on gfx1250", allow_module_level=True)

from primus_turbo.flydsl.mega.fp8.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (  # noqa: E402
    fused_mega_moe_fp8_stage1,
    fused_mega_moe_fp8_stage2,
)

# Import the FUNCTIONS only: pulling in the bf16 module's TestCase class would make pytest collect and
# re-run that suite from this module's namespace as well.
from tests.pytorch.ops.test_fused_mega_moe import (  # noqa: E402
    baseline_reference,
    generate_inputs,
)
from tests.pytorch.test_utils import compute_snr  # noqa: E402

# Measured on 8 x gfx950 (EP8, DSv3 shape), min over ranks, stable to the digit across runs:
#   forward 22.31  dx 21.96  dW1 22.52  dW2 22.96  dtw 23.06  dB
#   cosine: 0.99712 / 0.99687 / 0.99725 / 0.99751 / 0.99755
# One floor for all five, ~2 dB under the weakest: room for a shape or routing change, no room for a
# structural failure (the fused path's corrupted dW1 lands at 2.9-8.9 dB).
_SNR_FLOOR_DB = 20.0
_COSINE_FLOOR = 0.995

# Same as the bf16 suite: match the small tensor-level grad norm seen under real loss scaling.
_GRAD_OUT_NORM = 1e-3

skip_unless_mxfp8 = unittest.skipUnless(
    torch.cuda.is_available() and check_mxfp8_support()[0], "mxfp8 mega MoE requires gfx950"
)


@instantiate_parametrized_tests
class FusedMegaMoEFp8Test(MultiProcContinuousTest):
    """EP8 accuracy for the staged mxfp8 op vs turbo DeepEP; PG comes up once per class."""

    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _metrics(self, actual, ref):
        """SNR (dB) and cosine vs ref, reduced MIN so the weakest rank governs the EP group."""
        cos = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), ref.float().flatten(), dim=0, eps=1e-12
        )
        m = torch.tensor([compute_snr(ref, actual), float(cos)], device=self.device)
        dist.all_reduce(m, op=dist.ReduceOp.MIN)
        return float(m[0]), float(m[1])

    @skip_unless_mxfp8
    @skip_if_lt_x_gpu(8)
    @parametrize(
        "hidden, inter, num_experts, num_topk, num_tokens",
        [
            (7168, 2048, 256, 8, 8192),
        ],
    )
    def test_staged_forward_backward(self, hidden, inter, num_experts, num_topk, num_tokens):
        """stage1 + stage2 fwd+bwd vs the bf16 turbo DeepEP reference, on identical inputs."""
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)
        group = dist.group.WORLD

        x, l1_weight, l2_weight, topk_idx, topk_weight = generate_inputs(
            self.rank,
            self.world_size,
            num_tokens=num_tokens,
            hidden=hidden,
            inter=inter,
            num_experts=num_experts,
            num_topk=num_topk,
            device=self.device,
        )
        symm = get_symm_buffer_for_mega_moe(
            group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=inter,
            use_mxfp8=True,
        )
        try:
            _gy = torch.randn(x.shape, device=x.device, dtype=torch.float32)
            grad_y = (_gy / (_gy.norm() + 1e-12) * _GRAD_OUT_NORM).bfloat16()

            # the shape Primus' MegaMoEFP8Experts.forward runs: w1, stage1, w2, stage2
            x_m = x.detach().requires_grad_(True)
            l1_m = l1_weight.detach().requires_grad_(True)
            l2_m = l2_weight.detach().requires_grad_(True)
            tw_m = topk_weight.detach().requires_grad_(True)
            l1_out, dwib, handle, state = fused_mega_moe_fp8_stage1(x_m, topk_idx, tw_m, l1_m, group)
            y_m = fused_mega_moe_fp8_stage2(l1_out, dwib, handle, state, topk_idx, tw_m, l2_m, group)
            dx_m, dl1_m, dl2_m, dtw_m = torch.autograd.grad(y_m, [x_m, l1_m, l2_m, tw_m], grad_y)
            torch.cuda.synchronize()
            group.barrier()

            x_t = x.detach().requires_grad_(True)
            l1_t = l1_weight.detach().requires_grad_(True)
            l2_t = l2_weight.detach().requires_grad_(True)
            tw_t = topk_weight.detach().requires_grad_(True)
            y_t = baseline_reference(
                group,
                x_t,
                topk_idx,
                tw_t,
                l1_t,
                l2_t,
                num_experts=num_experts,
                num_topk=num_topk,
            )
            dx_t, dl1_t, dl2_t, dtw_t = torch.autograd.grad(y_t, [x_t, l1_t, l2_t, tw_t], grad_y)
            torch.cuda.synchronize()
            group.barrier()

            results = [
                ("forward", y_m, y_t),
                ("dx", dx_m, dx_t),
                ("dl1_weight", dl1_m, dl1_t),
                ("dl2_weight", dl2_m, dl2_t),
                ("dtw", dtw_m, dtw_t),
            ]
        finally:
            symm.destroy()

        measured = [(tag, *self._metrics(a, r)) for tag, a, r in results]
        if self.rank == 0:
            print(f"\n{'=' * 72}")
            print(f"[staged fp8 mega MoE vs turbo DeepEP]  EP{self.world_size} T={num_tokens}")
            print(f"{'=' * 72}")
            for tag, snr, cos in measured:
                print(f"  {tag:<12}: min SNR = {snr:7.2f} dB  min cos = {cos:.5f}", flush=True)
        for tag, snr, cos in measured:
            self.assertGreaterEqual(snr, _SNR_FLOOR_DB, f"[{tag}] SNR {snr:.2f} dB < {_SNR_FLOOR_DB}")
            self.assertGreaterEqual(cos, _COSINE_FLOOR, f"[{tag}] cosine {cos:.5f} < {_COSINE_FLOOR}")


if __name__ == "__main__":
    run_tests()
