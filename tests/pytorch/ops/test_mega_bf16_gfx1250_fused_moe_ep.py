###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused Mega MoE op on gfx1250 across every visible GPU (EP2+), against torch.

Each rank holds its slice of the experts and routes its tokens over all of them. The reference
gathers every rank's tokens, routing and expert weights and runs the fp32 MoE of the single-GPU
suite; each rank compares its own tokens and its own experts' gradients.
"""

from datetime import timedelta

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

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.core.utils import is_gfx1250

if not is_gfx1250():
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from primus_turbo.pytorch.ops.moe.fused_mega_moe import (  # noqa: E402
    fused_mega_moe,
    fused_mega_moe_stage1,
    fused_mega_moe_stage2,
)
from tests.pytorch.ops.test_mega_bf16_gfx1250_fused_moe import (  # noqa: E402
    _reference,
    _snr,
)


@instantiate_parametrized_tests
class FusedMegaMoEGfx1250EPTest(MultiProcContinuousTest):
    timeout = timedelta(seconds=600)

    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _gather_rows(self, t):
        """Every rank's rows of ``t`` rank-major, and where this rank's start; counts may differ."""
        n = torch.tensor([t.shape[0]], device=self.device)
        counts = [torch.empty_like(n) for _ in range(self.world_size)]
        dist.all_gather(counts, n)
        counts = [int(c) for c in counts]
        buf = t.new_zeros((max(counts),) + t.shape[1:])
        buf[: t.shape[0]] = t
        out = [torch.empty_like(buf) for _ in counts]
        dist.all_gather(out, buf)
        return torch.cat([o[:c] for o, c in zip(out, counts)]), sum(counts[: self.rank])

    def _assert_snr(self, results):
        """Every ``(name, got, want)`` above 40 dB on the weakest rank, so all ranks agree."""
        snr = torch.tensor([_snr(got, want) for _, got, want in results], device=self.device)
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)
        for (name, _, _), s in zip(results, snr.tolist()):
            self.assertGreater(s, 40, f"{name} SNR {s:.1f} dB on the weakest rank")

    def _run(
        self,
        num_tokens,
        hidden,
        inter,
        experts_per_rank,
        num_topk,
        *,
        activation=None,
        staged=False,
        bias=False,
        token_step=0,
        skew=0.0,
        enable_torch_compile=False,
        l1_gain=2.0,
        calls=2,
    ):
        torch.cuda.set_device(self.device)
        torch._dynamo.reset()
        group, rank = dist.group.WORLD, self.rank
        num_experts = experts_per_rank * self.world_size
        num_tokens += token_step * rank
        activation = GLUActivation.swigluoai() if activation == "swigluoai" else None
        gen = torch.Generator(device=self.device).manual_seed(1234 + rank)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, device=self.device, generator=gen) * scale).to(torch.bfloat16)

        w1 = randn(experts_per_rank, 2 * inter, hidden, scale=l1_gain / hidden**0.5)
        w2 = randn(experts_per_rank, hidden, inter, scale=2 / inter**0.5)
        b1, b2 = (
            (randn(experts_per_rank, 2 * inter), randn(experts_per_rank, hidden)) if bias else (None, None)
        )

        def run(x, topk_idx, topk_w, w1, w2, b1, b2):
            if staged:
                l1_out, dispatch_weights, handle = fused_mega_moe_stage1(x, topk_idx, topk_w, w1, group)
                return fused_mega_moe_stage2(
                    l1_out, dispatch_weights, handle, topk_idx, topk_w, w2, group, activation, b1, b2
                )
            return fused_mega_moe(group, x, topk_idx, topk_w, w1, w2, activation=activation)

        if enable_torch_compile:
            run = torch.compile(run)
        weights = [w for w in (w1, w2, b1, b2) if w is not None]
        full_weights = [self._gather_rows(w)[0].float() for w in weights]
        experts = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)
        for _ in range(calls):
            x = randn(num_tokens, hidden)
            logits = torch.randn(num_tokens, num_experts, device=self.device, generator=gen)
            logits[:, :experts_per_rank] += skew  # crowd rank 0's experts
            topk_w, topk_idx = logits.softmax(-1).topk(num_topk, dim=-1)
            dy = randn(num_tokens, hidden)
            leaves = [t.clone().requires_grad_() for t in [x, topk_w] + weights]
            lb1, lb2 = (leaves[4], leaves[5]) if bias else (None, None)
            y = run(leaves[0], topk_idx, leaves[1], leaves[2], leaves[3], lb1, lb2)
            y.backward(dy)

            gx, row0 = self._gather_rows(x)
            refs = [
                t.detach().float().requires_grad_() for t in [gx, self._gather_rows(topk_w)[0]] + full_weights
            ]
            rb1, rb2 = (refs[4], refs[5]) if bias else (None, None)
            yr = _reference(refs[0], self._gather_rows(topk_idx)[0], *refs[1:4], activation, rb1, rb2)
            yr.backward(self._gather_rows(dy)[0].float())
            rows = slice(row0, row0 + num_tokens)
            names = ["dx", "d topk_w", "dW1", "dW2", "d fc1_bias", "d fc2_bias"]
            results = [("y", y, yr.detach()[rows])]
            for name, got, want, own in zip(names, leaves, refs, [rows, rows] + [experts] * 4):
                results.append((name, got.grad, want.grad[own]))
            self._assert_snr(results)

    @skip_if_lt_x_gpu(2)
    @parametrize("enable_torch_compile", [False, True])
    def test_deepseek_v3(self, enable_torch_compile):
        # DeepSeek-V3's expert shape with its EP8 expert count per rank
        self._run(1024, 7168, 2048, 32, 8, enable_torch_compile=enable_torch_compile)

    @skip_if_lt_x_gpu(2)
    def test_ragged_skewed(self):
        # ragged, different token counts per rank (one 128-row symmetric-buffer capacity), and
        # most routes crowding rank 0's experts
        self._run(333, 7168, 2048, 32, 8, token_step=7, skew=3.0)

    @skip_if_lt_x_gpu(2)
    def test_one_token(self):
        self._run(1, 2048, 1408, 4, 4)

    @skip_if_lt_x_gpu(2)
    def test_staged_bias_gpt_oss(self):
        # GPT-OSS-20B: 64-wide k-tiles, ragged N; wide L1 weights put the clamp in play
        self._run(1024, 2880, 2880, 8, 4, activation="swigluoai", staged=True, bias=True, l1_gain=5.0)


if __name__ == "__main__":
    run_tests()
