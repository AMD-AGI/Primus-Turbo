###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Accuracy tests for the fused ``mega_moe_fused`` op against the turbo DeepEP MoE."""

from __future__ import annotations

import math
import unittest

import pytest  # noqa: E402
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

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.utils import is_gfx950, is_gfx1250  # noqa: E402

# mega_moe_fused (the flydsl mega path) is not supported on gfx1250. Skip the whole
# module before importing the flydsl-backed modules below
if is_gfx1250():
    pytest.skip("mega_moe_fused is not supported on gfx1250", allow_module_level=True)

from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.pytorch.ops import grouped_gemm as _turbo_gg  # noqa: E402
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused  # noqa: E402
from tests.pytorch.test_utils import compute_snr  # noqa: E402

# bf16 fused vs bf16 turbo ref; comm + split-role reduce add noise -> use MegaMoE-family SNR floor.
_SNR_THRESHOLD_DB = 25.0


# Fused kernel only supports gfx950; skip everywhere else.
skip_unless_gfx950 = unittest.skipUnless(
    torch.cuda.is_available() and is_gfx950(), "mega_moe_fused only supports gfx950"
)


def _weighted_swiglu(fc1_out, weights):
    """SwiGLU with a per-token routing weight (matches Megatron weighted_bias_swiglu)."""
    gate, up = fc1_out.chunk(2, dim=-1)
    return F.silu(gate) * up * weights


def generate_inputs(rank, world, *, num_tokens, hidden, inter, num_experts, num_topk, device="cuda"):
    """One rank's local MoE inputs: x, this rank's L1/L2 expert shard, random top-k routing."""
    epr = num_experts // world
    g = torch.Generator(device=device).manual_seed(1234 + rank)
    x = torch.randn((num_tokens, hidden), generator=g, device=device, dtype=torch.float32).bfloat16()
    l1_weight = torch.randn((epr, 2 * inter, hidden), generator=g, device=device, dtype=torch.bfloat16)
    l1_weight *= 2.0 / math.sqrt(hidden)
    l2_weight = torch.randn((epr, hidden, inter), generator=g, device=device, dtype=torch.bfloat16)
    l2_weight *= 2.0 / math.sqrt(inter)

    logits = torch.randn(num_tokens, num_experts, generator=g, device=device, dtype=torch.float32)
    topk_weight, topk_idx = torch.topk(logits.softmax(-1), num_topk, dim=-1)
    return x, l1_weight, l2_weight, topk_idx.to(torch.int64), topk_weight.to(torch.float32)


def baseline_reference(group, x, topk_idx, topk_weight, l1_weight, l2_weight, *, num_experts, num_topk):
    """Turbo DeepEP MoE forward; differentiable in x/l1/l2/topk_weight -> serves as the backward ref."""
    # scatter (differentiable) routes topk_weight grad straight back through the reference
    gate_logits = torch.zeros(x.shape[0], num_experts, device=x.device, dtype=torch.float32).scatter(
        1, topk_idx, topk_weight
    )
    dispatcher = turbo.modules.DeepEPTokenDispatcher(
        num_experts=num_experts,
        router_topk=num_topk,
        ep_group=group,
        permute_fusion=True,
        deepep_num_use_cu=80,
    )
    permuted_hidden, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
        x, gate_logits, indices=topk_idx
    )
    group_lens = tokens_per_expert.to(device=x.device, dtype=torch.int64)
    fc1_out = _turbo_gg(permuted_hidden, l1_weight, group_lens, trans_b=True)
    inter = _weighted_swiglu(fc1_out, permuted_probs.unsqueeze(-1)).to(x.dtype)
    fc2_out = _turbo_gg(inter, l2_weight, group_lens, trans_b=True)
    return dispatcher.token_combine(fc2_out)


def _test_forward_backward_impl(
    group,
    symm,
    x,
    l1_weight,
    l2_weight,
    topk_idx,
    topk_weight,
    *,
    num_experts,
    num_topk,
    enable_cudagraph=False,
    enable_torch_compile=False,
):
    """tc-free fwd+bwd of mega_moe_fused vs turbo; returns (tag, actual, ref) triples, frees symm."""
    try:
        grad_y = torch.randn(x.shape, device=x.device, dtype=torch.bfloat16)

        # fused runner over grad-carrying inputs; topk_idx stays a constant closure
        def _fused(x, topk_weight, l1_weight, l2_weight):
            return mega_moe_fused(group, x, topk_idx, topk_weight, l1_weight, l2_weight)

        runner = _fused
        if enable_torch_compile:
            runner = torch.compile(runner)

        x_m = x.detach().requires_grad_(True)
        l1_m = l1_weight.detach().requires_grad_(True)
        l2_m = l2_weight.detach().requires_grad_(True)
        tw_m = topk_weight.detach().requires_grad_(True)

        # make_graphed_callables warms up then captures both forward and backward graphs
        if enable_cudagraph:
            runner = torch.cuda.make_graphed_callables(runner, (x_m, tw_m, l1_m, l2_m))

        # device-side epoch flag advances per replay, so replay #2+ stays correct
        num_iters = 4
        for _ in range(num_iters):
            y_m = runner(x_m, tw_m, l1_m, l2_m)
            dx_m, dl1_m, dl2_m, dtw_m = torch.autograd.grad(y_m, [x_m, l1_m, l2_m, tw_m], grad_y)

        # turbo reference: topk_weight flows through scatter, so dtw is compared directly
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

        # dx asserted: stale-L2 read in combine gate path fixed in ep_intranode (glc|slc).
        return [
            ("forward", y_m, y_t),
            ("dx", dx_m, dx_t),
            ("dl1_weight", dl1_m, dl1_t),
            ("dl2_weight", dl2_m, dl2_t),
            ("dtw", dtw_m, dtw_t),
        ]
    finally:
        # release symm buffer even on failure; PG is class-scoped (kept for next test)
        if symm is not None:
            symm.destroy()


@instantiate_parametrized_tests
class MegaMoEFusedTestBase(MultiProcContinuousTest):
    """EP8 accuracy tests vs turbo DeepEP; base spawns workers + inits PG once per class."""

    # PG backend picked by the base class at spawn time.
    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _setup_device(self):
        # PG already up (base class); just bind the CUDA device + seed for this test.
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)

    def _inputs(self, num_tokens, hidden, inter, num_experts, num_topk):
        return generate_inputs(
            self.rank,
            self.world_size,
            num_tokens=num_tokens,
            hidden=hidden,
            inter=inter,
            num_experts=num_experts,
            num_topk=num_topk,
            device=self.device,
        )

    def _symm(self, group, num_tokens, hidden, inter, num_experts, num_topk):
        return get_symm_buffer_for_mega_moe(
            group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=inter,
        )

    def _snr(self, actual, ref, *, tag):
        """SNR (dB) of actual vs reference; weakest rank governs the EP group."""
        snr = torch.tensor([compute_snr(ref, actual)], device=self.device)  # ref is the signal
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)
        snr = float(snr.item())
        if self.rank == 0:
            print(f"[{tag}] min SNR = {snr:.2f} dB")
        return snr

    def _assert_snr(self, actual, ref, *, tag):
        snr = self._snr(actual, ref, tag=tag)
        self.assertGreaterEqual(snr, _SNR_THRESHOLD_DB, f"[{tag}] SNR {snr:.2f} dB < {_SNR_THRESHOLD_DB}")

    # ── forward + backward vs turbo DeepEP; optional cudagraph / torch.compile ──
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize(
        "hidden, inter, num_experts, num_topk, num_tokens",
        [
            (7168, 2048, 256, 8, 8192),
        ],
    )
    @parametrize(
        "enable_cudagraph, enable_torch_compile",
        [
            (False, False),
            (True, False),
            (False, True),
        ],
    )
    def test_forward_backward(
        self,
        hidden,
        inter,
        num_experts,
        num_topk,
        num_tokens,
        enable_cudagraph,
        enable_torch_compile,
    ):
        self._setup_device()
        group = dist.group.WORLD
        x, l1_weight, l2_weight, topk_idx, topk_weight = self._inputs(
            num_tokens, hidden, inter, num_experts, num_topk
        )
        symm = self._symm(group, num_tokens, hidden, inter, num_experts, num_topk)
        results = _test_forward_backward_impl(
            group,
            symm,
            x,
            l1_weight,
            l2_weight,
            topk_idx,
            topk_weight,
            num_experts=num_experts,
            num_topk=num_topk,
            enable_cudagraph=enable_cudagraph,
            enable_torch_compile=enable_torch_compile,
        )
        for tag, actual, ref in results:
            self._assert_snr(actual, ref, tag=tag)


if __name__ == "__main__":
    run_tests()
