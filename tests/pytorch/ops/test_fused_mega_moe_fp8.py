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
from primus_turbo.pytorch.kernels.fused_mega_moe import (  # noqa: E402
    advance_weight_generation,
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
            # deepseek_v2's expert shape. Its MegaMoE mxfp8 arm trains to loss 11.81 at iteration
            # 20 where its bf16 arm reaches 10.39, with iteration-1 loss matching -- forward fine,
            # backward not. inter=1536 is the only structural difference from the shape above.
            (5120, 1536, 160, 6, 8192),
            # Same, with hidden held at the known-good 7168, so a failure here pins it on inter
            # alone rather than on the hidden/inter pair.
            (7168, 1536, 256, 8, 8192),
            # The shape above with topk 6 instead of 8. In training this is the configuration that
            # blows up: a topk-sweep on deepseek_v3 measured iteration-1 grad norm 0.81 at topk 4
            # and 1.43 at topk 8, against 2245 / 1529 / 1290 at topk 5 / 6 / 7 -- only powers of
            # two survive, and only with mxfp8 experts (bf16 experts report 1.09 at topk 6). The
            # layer's dx is correct there, so it is the weight gradients that are inflated.
            (7168, 2048, 256, 6, 8192),
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


    @skip_unless_mxfp8
    @skip_if_lt_x_gpu(8)
    @parametrize("num_topk", [8, 6])
    def test_multi_layer_grad_accumulation(self, num_topk):
        """Several layers per pass and several accumulated passes -- the shape of a training step.

        The single-pass test above reuses the symmetric workspace once; a training step reuses it
        ``num_layers * micro_batches * 2`` times, and two cross-rank handoffs in the fp8 combine
        were only correct on a cold workspace:

          * ``combine_gate`` (the gate gradient a peer scatters into this rank's CACHED main heap)
            was pushed and read without the system-scope cache bits, so every layer but the last
            read the gate gradient the NEXT layer's backward had left in L2 -- dtw fell from 23 dB
            to 6 dB on layers 0..L-2 while the last layer, whose backward runs with nothing in
            between, stayed correct.
          * the ``comb`` payload was pushed and read the same way, so a handful of slots per call
            were reduced from the FORWARD's y still sitting in the buffer. dx is ~1e-6 and y is
            ~1e0, so a few stale slots inflated dx by ~1e6 -- iteration-1 grad norm 1300 instead
            of 1.4, clip_grad turning every update into a 1000x smaller learning rate.

        Both need the workspace to be warm, which is why they are invisible to a single pass, and
        both are silent: no NaN, no hang, forward and loss unchanged.

        What this test needs that the earlier one did not: DISTINCT data per pass. ``generate_inputs``
        seeds its own generator, so passes built without a ``seed`` are the same micro-batch twice,
        and a buffer left holding the previous pass's values is indistinguishable from a correct one.

        dx and dtw are checked, not just the weight gradients: dx is where the 1e6 inflation shows,
        and dtw is where the gate staleness does. topk 8 runs alongside 6 as the control -- the
        training sweep found only powers of two surviving, and the timing of the comb race follows
        the tile count, so the two values exercise different schedules of the same code.
        """
        torch.cuda.set_device(self.device)
        # The prepared-fp8-weight cache keys on (data_ptr, shape) plus the generation, so a tensor
        # allocated at an address a previous test just freed, at the same shape, is served that
        # test's quantized weights. Advancing the generation is what an optimizer step does and is
        # the cache's documented contract; without it this test reads the previous parametrization's
        # w1/w2 and reports cos ~ 0 at unchanged magnitude.
        advance_weight_generation()
        group = dist.group.WORLD
        # Same shape as the single-pass test above, deliberately: this class runs every test in
        # one process, and compiling the combine for a second token count there trips a FlyDSL
        # trace error in whichever test recompiles next.
        hidden, inter, num_experts, num_tokens = 7168, 2048, 256, 8192
        num_layers, num_passes = 2, 2

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
            shape = dict(
                num_tokens=num_tokens,
                hidden=hidden,
                inter=inter,
                num_experts=num_experts,
                num_topk=num_topk,
                device=self.device,
            )
            # One weight pair per layer, one (x, routing, grad_y) per pass, all distinct.
            weights = [
                generate_inputs(self.rank, self.world_size, seed=7700 + 13 * lyr, **shape)[1:3]
                for lyr in range(num_layers)
            ]
            passes = []
            for p in range(num_passes):
                x, _, _, topk_idx, topk_weight = generate_inputs(
                    self.rank, self.world_size, seed=4200 + 91 * p, **shape
                )
                _g = torch.randn(x.shape, device=self.device, dtype=torch.float32)
                passes.append((x, topk_idx, topk_weight, (_g / (_g.norm() + 1e-12) * _GRAD_OUT_NORM).bfloat16()))

            def accumulate(run_one):
                """All layers per pass, then one backward -- so each layer's backward runs with
                other layers' calls between it and its own forward, as in a real step."""
                ws = [
                    (l1.detach().clone().requires_grad_(True), l2.detach().clone().requires_grad_(True))
                    for l1, l2 in weights
                ]
                dx_acc = [torch.zeros(num_tokens, hidden, device=self.device) for _ in range(num_layers)]
                dtw_acc = [torch.zeros(num_tokens, num_topk, device=self.device) for _ in range(num_layers)]
                for x, topk_idx, topk_weight, grad_y in passes:
                    xs, tws, ys = [], [], []
                    for l1, l2 in ws:
                        xs.append(x.detach().requires_grad_(True))
                        tws.append(topk_weight.detach().requires_grad_(True))
                        ys.append(run_one(xs[-1], topk_idx, tws[-1], l1, l2))
                    # NO synchronize()/barrier() between the calls: a rendezvous there is not
                    # something a training step does, and it hides what this test is for.
                    sum(ys).backward(grad_y)
                    for lyr in range(num_layers):
                        dx_acc[lyr] += xs[lyr].grad.float()
                        dtw_acc[lyr] += tws[lyr].grad.float()
                return [
                    (l1.grad, l2.grad, dx_acc[lyr], dtw_acc[lyr]) for lyr, (l1, l2) in enumerate(ws)
                ]

            def mega(x, topk_idx, tw, l1, l2):
                l1_out, dwib, handle, state = fused_mega_moe_fp8_stage1(x, topk_idx, tw, l1, group)
                return fused_mega_moe_fp8_stage2(l1_out, dwib, handle, state, topk_idx, tw, l2, group)

            def reference(x, topk_idx, tw, l1, l2):
                return baseline_reference(
                    group, x, topk_idx, tw, l1, l2, num_experts=num_experts, num_topk=num_topk
                )

            got = accumulate(mega)
            torch.cuda.synchronize()
            group.barrier()
            want = accumulate(reference)
        finally:
            symm.destroy()

        tags = ("dl1_weight", "dl2_weight", "dx", "dtw")
        measured = [
            (f"layer{lyr} {tag}", got[lyr][i], want[lyr][i])
            for lyr in range(num_layers)
            for i, tag in enumerate(tags)
        ]
        results = [(name, *self._metrics(a, r)) for name, a, r in measured]
        if self.rank == 0:
            print(
                f"\n[{num_layers}-layer x {num_passes}-pass accumulated fp8 mega MoE]  "
                f"EP{self.world_size} topk={num_topk}"
            )
            for (name, snr, cos), (_, a, r) in zip(results, measured):
                # Scale is the symptom: the training failure is a ~1e6 inflation, not noise.
                ratio = float(a.float().norm() / r.float().norm())
                print(
                    f"  {name:<20}: min SNR = {snr:7.2f} dB  min cos = {cos:.5f}  "
                    f"||mega||/||ref|| = {ratio:.3f}",
                    flush=True,
                )
        for name, snr, cos in results:
            self.assertGreaterEqual(snr, _SNR_FLOOR_DB, f"[{name}] SNR {snr:.2f} dB < {_SNR_FLOOR_DB}")
            self.assertGreaterEqual(cos, _COSINE_FLOOR, f"[{name}] cosine {cos:.5f} < {_COSINE_FLOOR}")


if __name__ == "__main__":
    run_tests()
