###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE in MXFP8 on gfx1250 across every visible GPU (EP2+). Written for a multi-GPU node; skipped below two GPUs.

Every call uses fresh tokens, routing and weights, and between calls the dispatch and combine token buffers are
filled with NaN bytes, so a wait the protocol lacks reads NaN instead of a right value an earlier call left; some
cases hold one rank back before its launches. The per-row stages are checked bit for bit on every rank
(``check_stages`` of the EP1 suite), the ops against the fp32 reference of the single-GPU bf16 suite (20 dB, the MX
recipe gives ~23 dB; every row of y / dx within 0.5 relative), and determinism and two interleaved layers bitwise.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.core.utils import is_gfx1250

if not is_gfx1250():
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation  # noqa: E402
from tests.pytorch.ops.mega_gfx1250_ep_utils import (  # noqa: E402
    assert_rows_close_all_ranks,
    assert_snr_all_ranks,
    gather_rows,
    late_rank,
    poison_mega_heap,
)
from tests.pytorch.ops.test_mega_bf16_gfx1250_fused_moe import (  # noqa: E402
    _reference,
    _snr,
)
from tests.pytorch.ops.test_mega_mxfp8_gfx1250 import (  # noqa: E402
    _op_forward,
    _op_grads,
    check_stages,
)

SWIGLU = GLUActivation.swiglu()


class FusedMegaMoEMXFP8Gfx1250EPTest(MultiProcContinuousTest):
    timeout = timedelta(seconds=900)

    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _inputs(self, num_tokens, hidden, inter, experts_per_rank, num_topk, skew=0.0, seed=0):
        gen = torch.Generator(device=self.device).manual_seed(1234 + 17 * seed + self.rank)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, device=self.device, generator=gen) * scale).to(torch.bfloat16)

        num_experts = experts_per_rank * self.world_size
        x, dy = randn(num_tokens, hidden), randn(num_tokens, hidden)
        logits = torch.randn(num_tokens, num_experts, device=self.device, generator=gen)
        logits[:, :experts_per_rank] += skew  # crowd rank 0's experts
        topk_w, topk_idx = logits.softmax(-1).topk(num_topk, dim=-1)
        w1 = randn(experts_per_rank, 2 * inter, hidden, scale=2 / hidden**0.5)
        w2 = randn(experts_per_rank, hidden, inter, scale=2 / inter**0.5)
        return x, topk_idx, topk_w, w1, w2, dy

    def _forward(self, x, topk_idx, topk_w, w1, w2, activation, fuse=False, delay_rank=None):
        """The EP1 suite's staged forward, ``(y, leaves)``, as a step of its own (the MX weight cache is keyed by
        address, which fresh weight clones may reuse) and after holding ``delay_rank`` back."""
        advance_weight_generation()
        late_rank(delay_rank)
        return _op_forward(dist.group.WORLD, x, topk_idx, topk_w, w1, w2, activation, fuse)

    def _grads(self, y, leaves, dy, fuse=False, delay_rank=None):
        late_rank(delay_rank)
        return _op_grads(y, leaves, dy, fuse)

    def _step(self, x, topk_idx, topk_w, w1, w2, dy, activation, fuse=False, delay_rank=None):
        """One staged forward + backward; ``(y, dx, d topk_w, dW1, dW2)`` (the wgrads from main_grad with ``fuse``)."""
        y, leaves = self._forward(x, topk_idx, topk_w, w1, w2, activation, fuse, delay_rank)
        return self._grads(y, leaves, dy, fuse, delay_rank)

    def _want(self, x, topk_idx, topk_w, w1, w2, dy, activation):
        """The fp32 reference over every rank's tokens and experts: this rank's ``(y, dx, d topk_w, dW1, dW2)``."""
        gx, row0 = gather_rows(x)
        refs = [
            t.detach().float().requires_grad_()
            for t in (gx, gather_rows(topk_w)[0], gather_rows(w1)[0], gather_rows(w2)[0])
        ]
        yr = _reference(refs[0], gather_rows(topk_idx)[0], *refs[1:], activation)
        yr.backward(gather_rows(dy)[0].float())
        rows = slice(row0, row0 + x.shape[0])
        experts = slice(self.rank * w1.shape[0], (self.rank + 1) * w1.shape[0])
        return [
            yr.detach()[rows],
            refs[0].grad[rows],
            refs[1].grad[rows],
            refs[2].grad[experts],
            refs[3].grad[experts],
        ]

    def _check(
        self,
        num_tokens,
        hidden,
        inter,
        experts_per_rank,
        num_topk,
        *,
        activation=SWIGLU,
        token_step=0,
        skew=0.0,
        fuse=False,
        calls=2,
        delay_rank=None,
        snr_of=("y", "dx", "d topk_w", "dW1", "dW2"),
    ):
        torch.cuda.set_device(self.device)
        num_tokens += token_step * self.rank
        for call in range(calls):
            poison_mega_heap()
            inputs = self._inputs(num_tokens, hidden, inter, experts_per_rank, num_topk, skew, call)
            got = self._step(*inputs, activation, fuse, delay_rank)
            want = self._want(*inputs, activation)
            names = ("y", "dx", "d topk_w", "dW1", "dW2")
            results = list(zip(names, got, want))
            assert_snr_all_ranks(self, [r for r in results if r[0] in snr_of], 20, _snr)
            assert_rows_close_all_ranks(self, results[:2], 0.5)

    @skip_if_lt_x_gpu(2)
    def test_deepseek_v3(self):
        # DeepSeek-V3's expert shape with its EP8 expert count per rank; the wgrads into main_grad too
        self._check(1024, 7168, 2048, 32, 8, calls=3)
        self._check(1024, 7168, 2048, 32, 8, fuse=True, calls=1)

    @skip_if_lt_x_gpu(2)
    def test_ragged_skewed_late_rank(self):
        # different token counts per rank, most routes crowding rank 0's experts (empty experts elsewhere), and the
        # last rank held back before each launch
        self._check(333, 7168, 2048, 32, 8, token_step=7, skew=3.0, calls=3, delay_rank=-1)

    @skip_if_lt_x_gpu(2)
    def test_one_token_clamped(self):
        # one token per rank, a small expert shape, Megatron's clamp form of the activation (some ranks' experts get
        # no route: their wgrads must be exactly zero). d topk_w is 4 values per rank here, too few for an SNR
        # gate at the recipe's ~23 dB (test_stages_bitwise checks the gate gradient bit for bit on every rank).
        self._check(
            1,
            2048,
            512,
            4,
            4,
            activation=GLUActivation.swiglu(7.0),
            calls=3,
            snr_of=("y", "dx", "dW1", "dW2"),
        )

    @skip_if_lt_x_gpu(2)
    def test_stages_bitwise(self):
        """The per-row stages bit for bit on every rank (EP-invariant), fresh inputs and a poisoned heap per call,
        rank 0 held back on the second."""
        torch.cuda.set_device(self.device)
        for call, delay_rank in enumerate((None, 0, None)):
            poison_mega_heap()
            late_rank(delay_rank)
            check_stages(dist.group.WORLD, 1024, 7168, 2048, 32 * self.world_size, 8, seed=call, skew=True)
            check_stages(dist.group.WORLD, 333, 1024, 512, 8 * self.world_size, 4, seed=10 + call)

    @skip_if_lt_x_gpu(2)
    def test_repeat_bitwise(self):
        """Every output bitwise between two calls on the same inputs, with a call on other inputs between them, a
        poisoned heap before each and a different rank held back: the fold and reduce orders are fixed, and so is
        the pool order the wgrads contract over (token order within each segment part)."""
        torch.cuda.set_device(self.device)
        first_inputs = self._inputs(1024, 7168, 2048, 32, 8, skew=1.0, seed=0)
        other_inputs = self._inputs(1024, 7168, 2048, 32, 8, skew=1.0, seed=1)
        poison_mega_heap()
        first = self._step(*first_inputs, SWIGLU)
        poison_mega_heap()
        self._step(*other_inputs, SWIGLU, delay_rank=0)
        poison_mega_heap()
        again = self._step(*first_inputs, SWIGLU, delay_rank=-1)
        bad = torch.zeros(1, device=self.device)
        for a, b in zip(first, again):
            bad += int(not torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        dist.all_reduce(bad)
        self.assertEqual(int(bad.item()), 0, "an output changed between calls")

    @skip_if_lt_x_gpu(2)
    def test_interleaved_layers(self):
        """Two layers on one heap, forward A, forward B, backward B, backward A (A's backward after B's prologue and
        dispatches): each layer bitwise as it is alone."""
        torch.cuda.set_device(self.device)
        a = self._inputs(1024, 7168, 2048, 32, 8, skew=1.0, seed=3)
        b = self._inputs(1024, 7168, 2048, 32, 8, seed=4)
        poison_mega_heap()
        alone_a = self._step(*a, SWIGLU)
        poison_mega_heap()
        alone_b = self._step(*b, SWIGLU)
        poison_mega_heap()
        ya, leaves_a = self._forward(*a[:5], SWIGLU)
        yb, leaves_b = self._forward(*b[:5], SWIGLU)
        got_b = self._grads(yb, leaves_b, b[5])
        got_a = self._grads(ya, leaves_a, a[5])
        bad = torch.zeros(1, device=self.device)
        for got, alone in ((got_a, alone_a), (got_b, alone_b)):
            for x, y in zip(got, alone):
                bad += int(not torch.equal(x.view(torch.uint8), y.view(torch.uint8)))
        dist.all_reduce(bad)
        self.assertEqual(int(bad.item()), 0, "interleaved layers differ from the layers alone")


if __name__ == "__main__":
    run_tests()
