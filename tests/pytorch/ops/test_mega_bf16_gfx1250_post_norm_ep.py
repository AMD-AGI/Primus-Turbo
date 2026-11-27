###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The bf16 Mega MoE with the post-norm stage2 on gfx1250 across every visible GPU (EP2+), against torch.

Per routed row the experts compute ``n = RMSNorm((act @ w2^T) * (1 + w))`` and the token takes
``sum_k p_k * n_k`` (Muse Spark's latent MoE). The reference gathers every rank's tokens, routing and expert
weights and runs it in fp32; each rank compares its own tokens, its own experts' gradients and the norm
weight's gradient (summed over ranks, as data parallelism would). The Mega heap is filled with NaN bytes
between calls, so a missing wait reads NaN rather than a value an earlier call left.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
    fused_mega_moe_stage1,
    fused_mega_moe_stage2_post_norm,
)
from tests.pytorch.ops.mega_gfx1250_ep_utils import (  # noqa: E402
    assert_snr_all_ranks,
    gather_rows,
    poison_mega_heap,
)

EPS = 1e-8


def _reference(x, topk_idx, topk_w, w1, w2, norm_weight, eps):
    """fp32, differentiable in every input but ``topk_idx``; plain SiLU-SwiGLU on the gate||up layout."""
    inter = w2.shape[2]
    y = torch.zeros(x.shape[0], w2.shape[1], device=x.device)
    for e in range(w1.shape[0]):
        t, k = (topk_idx == e).nonzero(as_tuple=True)
        if not t.numel():
            continue
        l1 = x[t] @ w1[e].T
        act = F.silu(l1[:, :inter]) * l1[:, inter:]
        s = (act @ w2[e].T) * (1.0 + norm_weight)
        n = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
        y = y.index_add(0, t, topk_w[t, k, None] * n)
    return y


def _snr(got, want):
    return -20 * torch.log10((got.float() - want).norm() / want.norm()).item()


@instantiate_parametrized_tests
class MegaMoEPostNormGfx1250EPTest(MultiProcContinuousTest):
    timeout = timedelta(seconds=600)

    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _run(
        self,
        num_tokens,
        hidden,
        inter,
        experts_per_rank,
        num_topk,
        *,
        token_step=0,
        calls=2,
        fp32_accum=False
    ):
        torch.cuda.set_device(self.device)
        group, rank = dist.group.WORLD, self.rank
        num_experts = experts_per_rank * self.world_size
        num_tokens += token_step * rank
        activation = GLUActivation.swiglu()
        gen = torch.Generator(device=self.device).manual_seed(1234 + rank)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, device=self.device, generator=gen) * scale).to(torch.bfloat16)

        w1 = randn(experts_per_rank, 2 * inter, hidden, scale=2 / hidden**0.5)
        w2 = randn(experts_per_rank, hidden, inter, scale=2 / inter**0.5)
        # every rank holds the same (replicated) norm weight
        norm_weight = (
            torch.randn(hidden, device=self.device, generator=torch.Generator(self.device).manual_seed(7))
            * 0.1
        ).to(torch.bfloat16)
        full_w1, full_w2 = (gather_rows(w)[0].float() for w in (w1, w2))
        experts = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)
        for _ in range(calls):
            poison_mega_heap()
            x = randn(num_tokens, hidden)
            logits = torch.randn(num_tokens, num_experts, device=self.device, generator=gen)
            topk_w, topk_idx = logits.sigmoid().topk(num_topk, dim=-1)
            dy = randn(num_tokens, hidden)
            lx, ltw, lw1, lw2, lnw = (t.clone().requires_grad_() for t in (x, topk_w, w1, w2, norm_weight))
            accum = {}
            if fp32_accum:
                # Megatron's fused wgrad accumulation: dW is added into an fp32 main_grad holding a prior value
                for w in (lw1, lw2):
                    w.main_grad = torch.full(w.shape, 0.5, device=self.device, dtype=torch.float32)
                    w.grad_added_to_main_grad = False
                accum = {"fuse_wgrad_accum_pattern": "megatron", "wgrad_accum_fp32": True}
            l1_out, dispatch_weights, handle = fused_mega_moe_stage1(lx, topk_idx, ltw, lw1, group, **accum)
            y = fused_mega_moe_stage2_post_norm(
                l1_out, dispatch_weights, handle, topk_idx, lw2, lnw, EPS, group, activation, **accum
            )
            y.backward(dy)
            if fp32_accum:
                dw1, dw2 = (w.main_grad - 0.5 for w in (lw1, lw2))
            else:
                dw1, dw2 = lw1.grad, lw2.grad
            dnw = lnw.grad.float()
            dist.all_reduce(dnw, group=group)

            gx, row0 = gather_rows(x)
            refs = [
                t.detach().float().requires_grad_()
                for t in (gx, gather_rows(topk_w)[0], full_w1, full_w2, norm_weight)
            ]
            yr = _reference(refs[0], gather_rows(topk_idx)[0], *refs[1:], EPS)
            yr.backward(gather_rows(dy)[0].float())
            rows = slice(row0, row0 + num_tokens)
            results = [
                ("y", y, yr.detach()[rows]),
                ("dx", lx.grad, refs[0].grad[rows]),
                ("d topk_w", ltw.grad, refs[1].grad[rows]),
                ("dW1", dw1, refs[2].grad[experts]),
                ("dW2", dw2, refs[3].grad[experts]),
                ("d norm_weight", dnw, refs[4].grad),
            ]
            assert_snr_all_ranks(self, results, 30, _snr)

    @skip_if_lt_x_gpu(2)
    def test_muse_spark_shape(self):
        # Muse Spark's latent MoE: hidden (latent) 4096, expert FFN 4096, top-8
        self._run(1024, 4096, 4096, 16, 8)

    @skip_if_lt_x_gpu(2)
    def test_ragged(self):
        # different token counts per rank
        self._run(333, 4096, 4096, 8, 8, token_step=7)

    @skip_if_lt_x_gpu(2)
    def test_fp32_wgrad_accum(self):
        # dW1 / dW2 added into fp32 main_grad (Megatron's fused wgrad accumulation)
        self._run(1024, 4096, 4096, 16, 8, calls=1, fp32_accum=True)

    @skip_if_lt_x_gpu(2)
    def test_small_hidden(self):
        self._run(512, 1024, 512, 4, 4)

    @skip_if_lt_x_gpu(2)
    @parametrize("calls", [3])
    def test_repeated_calls(self, calls):
        self._run(256, 2048, 1024, 8, 8, calls=calls)


if __name__ == "__main__":
    run_tests()
