###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    set_buffer_global_config,
)

NUM_TOKENS = 4096
HIDDEN_SIZE = 4096
NUM_EXPERTS = 256
ROUTER_TOPK = 8


def _get_backends():
    """Return available backend names."""
    try:
        import deep_ep  # noqa: F401

        return ["TURBO", "DEEP_EP"]
    except ImportError:
        return ["TURBO"]


def _run_dispatch_combine(
    rank,
    ep_group,
    num_tokens=NUM_TOKENS,
    hidden_size=HIDDEN_SIZE,
    num_experts=NUM_EXPERTS,
    router_topk=ROUTER_TOPK,
    dtype=torch.bfloat16,
    permute_fusion=True,
    deepep_use_cuda_num_tokens_per_expert=False,
    deepep_num_worst_tokens=0,
    permute_max_token_num=0,
    expert_capacity_factor=None,
):
    """Core dispatch-combine logic shared by all test variants."""
    dispatcher = turbo.modules.DeepEPTokenDispatcher(
        num_experts,
        router_topk,
        ep_group,
        permute_fusion=permute_fusion,
        deepep_use_cuda_num_tokens_per_expert=deepep_use_cuda_num_tokens_per_expert,
        deepep_num_worst_tokens=deepep_num_worst_tokens,
        permute_max_token_num=permute_max_token_num,
        expert_capacity_factor=expert_capacity_factor,
    )

    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    ans = hidden_states.clone()
    hidden_states.requires_grad = True

    probs = torch.ones((num_tokens, num_experts), dtype=torch.float32, device="cuda") / router_topk

    permuted_local_hidden_states, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
        hidden_states, probs
    )

    permuted_local_hidden_states = permuted_local_hidden_states * permuted_probs.unsqueeze(-1)
    permuted_local_hidden_states = permuted_local_hidden_states.to(ans.dtype)

    restored_hidden_states = dispatcher.token_combine(permuted_local_hidden_states)

    assert torch.allclose(
        restored_hidden_states, ans
    ), "Restored hidden states do not match original hidden states"

    torch.autograd.backward(restored_hidden_states, hidden_states)
    assert torch.allclose(hidden_states.grad, ans), "Gradient does not match original hidden states"

    expected_device = "cuda" if deepep_use_cuda_num_tokens_per_expert else "cpu"
    assert (
        tokens_per_expert.device.type == expected_device
    ), f"Expected tokens_per_expert on {expected_device}, got {tokens_per_expert.device.type}"


@instantiate_parametrized_tests
class TestTokenDispatcher(MultiProcContinuousTest):
    # -2 tells MultiProcContinuousTest to use torch.cuda.device_count()
    world_size = -2

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    # ------------------------------------------------------------------
    # Basic dispatch/combine correctness
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    @parametrize("deepep_use_cuda_num_tokens_per_expert", [False, True])
    @parametrize("expert_capacity_factor", [None, 0.5])
    def test_basic(self, backend, deepep_use_cuda_num_tokens_per_expert, expert_capacity_factor):
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            _run_dispatch_combine(
                self.rank,
                dist.group.WORLD,
                deepep_use_cuda_num_tokens_per_expert=deepep_use_cuda_num_tokens_per_expert,
                expert_capacity_factor=expert_capacity_factor,
            )

    # ------------------------------------------------------------------
    # num_worst_tokens > 0 (requires deepep_use_cuda_num_tokens_per_expert)
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    @parametrize("permute_max_token_num", [0, NUM_TOKENS * 8 * ROUTER_TOPK])
    def test_worst_tokens(self, backend, permute_max_token_num):
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            _run_dispatch_combine(
                self.rank,
                dist.group.WORLD,
                deepep_use_cuda_num_tokens_per_expert=True,
                deepep_num_worst_tokens=NUM_TOKENS * 8,
                permute_max_token_num=permute_max_token_num,
            )

    # ------------------------------------------------------------------
    # CUDA graph compatibility (requires num_worst_tokens > 0 and
    # permute_max_token_num > 0 to avoid host syncs)
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    def test_cuda_graph(self, backend):
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            num_worst_tokens = NUM_TOKENS * 8
            permute_max_token_num = NUM_TOKENS * 8 * ROUTER_TOPK

            set_buffer_global_config(num_use_cu=32)

            dispatcher = turbo.modules.DeepEPTokenDispatcher(
                NUM_EXPERTS,
                ROUTER_TOPK,
                dist.group.WORLD,
                permute_fusion=True,
                deepep_use_cuda_num_tokens_per_expert=True,
                deepep_num_worst_tokens=num_worst_tokens,
                permute_max_token_num=permute_max_token_num,
            )

            hidden_states = torch.randn((NUM_TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
            probs = torch.ones((NUM_TOKENS, NUM_EXPERTS), dtype=torch.float32, device="cuda") / ROUTER_TOPK

            # Warmup (eager)
            permuted, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(hidden_states, probs)
            permuted = permuted * permuted_probs.unsqueeze(-1)
            permuted = permuted.to(hidden_states.dtype)
            restored = dispatcher.token_combine(permuted)

            # Capture CUDA graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                permuted, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(hidden_states, probs)
                permuted = permuted * permuted_probs.unsqueeze(-1)
                permuted = permuted.to(hidden_states.dtype)
                restored = dispatcher.token_combine(permuted)

            # Replay and verify
            g.replay()
            torch.cuda.synchronize()
            assert restored is not None, "CUDA graph replay should produce output"

    # ------------------------------------------------------------------
    # Autotune env var (PRIMUS_TURBO_AUTO_TUNE=1)
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    def test_autotune(self, backend):
        with patch.dict(
            os.environ,
            {
                "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend,
                "PRIMUS_TURBO_AUTO_TUNE": "1",
            },
        ):
            _run_dispatch_combine(
                self.rank,
                dist.group.WORLD,
            )


if __name__ == "__main__":
    run_tests()
