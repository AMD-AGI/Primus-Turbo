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

        # TODO: add backend deepep, temporal disable it since rocm7.2 deepep bug.
        # return ["TURBO", "DEEP_EP"]
        return ["TURBO"]
    except ImportError:
        return ["TURBO"]


def _build_tp_groups(tp_size):
    """Build (ep_group, tp_group, tp_ep_group) for the calling rank.

    Layout for ``world_size = W`` and ``tp_size = T`` (with ``ep_size = W // T``):
      - ``tp_ep_group``: ``dist.group.WORLD`` (size W).
      - ``ep_group``:    world split into ``T`` contiguous slabs of size ``ep_size``.
                         Slab ``s`` holds ranks ``[s*ep_size, (s+1)*ep_size)``.
      - ``tp_group``:    world split into ``ep_size`` slabs of size ``T``;
                         slab ``i`` holds ``[i, i+ep_size, i+2*ep_size, ...]``.

    Returns the groups the calling rank belongs to.
    """
    world_size = dist.get_world_size()
    assert world_size % tp_size == 0, f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    ep_size = world_size // tp_size
    my_rank = dist.get_rank()

    ep_group = None
    for s in range(tp_size):
        ranks = list(range(s * ep_size, (s + 1) * ep_size))
        group = dist.new_group(ranks)
        if my_rank in ranks:
            ep_group = group

    tp_group = None
    for i in range(ep_size):
        ranks = [i + s * ep_size for s in range(tp_size)]
        group = dist.new_group(ranks)
        if my_rank in ranks:
            tp_group = group

    tp_ep_group = dist.group.WORLD
    return ep_group, tp_group, tp_ep_group


def _run_dispatch_combine(
    rank,
    ep_group,
    num_tokens=NUM_TOKENS,
    hidden_size=HIDDEN_SIZE,
    num_experts=NUM_EXPERTS,
    router_topk=ROUTER_TOPK,
    dtype=torch.bfloat16,
    permute_fusion=None,
    deepep_use_cuda_num_tokens_per_expert=False,
    deepep_num_worst_tokens=0,
    permute_max_token_num=0,
    pad_multiple=0,
    expert_capacity_factor=None,
    deepep_use_comm_stream=False,
    tp_size=1,
    routing_map=None,
    token_indices=None,
):
    """Core dispatch-combine logic shared by all test variants."""
    if tp_size > 1:
        ep_group, tp_group, tp_ep_group = _build_tp_groups(tp_size)
    else:
        tp_group = None
        tp_ep_group = None

    dispatcher = turbo.modules.DeepEPTokenDispatcher(
        num_experts,
        router_topk,
        ep_group,
        tp_group=tp_group,
        tp_ep_group=tp_ep_group,
        permute_fusion=permute_fusion,
        deepep_use_cuda_num_tokens_per_expert=deepep_use_cuda_num_tokens_per_expert,
        deepep_num_worst_tokens=deepep_num_worst_tokens,
        permute_max_token_num=permute_max_token_num,
        pad_multiple=pad_multiple,
        expert_capacity_factor=expert_capacity_factor,
        deepep_use_comm_stream=deepep_use_comm_stream,
    )

    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    ans = hidden_states.clone()
    hidden_states.requires_grad = True

    # Per-token probs must sum to 1 across routes for the roundtrip identity.
    # With TP, probs are replicated to tp_size slots, so divide by tp_size too.
    probs = torch.ones((num_tokens, num_experts), dtype=torch.float32, device="cuda") / (
        router_topk * tp_size
    )

    permuted_local_hidden_states, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
        hidden_states, probs, routing_map=routing_map, indices=token_indices
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

    def _bind_device(self) -> None:
        """Bind the current worker process to its own GPU.

        ``MultiProcContinuousTest._init_pg`` only sets ``LOCAL_RANK`` but does
        not call ``torch.cuda.set_device``.  Without this call every rank would
        default to ``cuda:0``, causing NCCL to raise
        ``Duplicate GPU detected: rank 0 and rank 1 both on CUDA device ...``
        inside ``Buffer.__init__``'s ``all_gather_object`` and ultimately a
        GPU memory-access fault in the ``intranode::barrier`` kernel during
        buffer teardown.
        """
        torch.cuda.set_device(self.device)

    # ------------------------------------------------------------------
    # Basic dispatch/combine correctness
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    @parametrize("deepep_use_cuda_num_tokens_per_expert", [False, True])
    @parametrize("expert_capacity_factor", [None, 0.5])
    def test_basic(self, backend, deepep_use_cuda_num_tokens_per_expert, expert_capacity_factor):
        self._bind_device()
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
        self._bind_device()
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            _run_dispatch_combine(
                self.rank,
                dist.group.WORLD,
                deepep_use_cuda_num_tokens_per_expert=True,
                deepep_num_worst_tokens=NUM_TOKENS * 8,
                permute_max_token_num=permute_max_token_num,
            )

    # ------------------------------------------------------------------
    # pad_multiple > 0 (requires deepep_use_cuda_num_tokens_per_expert)
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    @parametrize("pad_multiple", [128, 256])
    def test_pad_multiple(self, backend, pad_multiple):
        self._bind_device()
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            _run_dispatch_combine(
                self.rank,
                dist.group.WORLD,
                deepep_use_cuda_num_tokens_per_expert=True,
                pad_multiple=pad_multiple,
            )

    # ------------------------------------------------------------------
    # Autotune env var (PRIMUS_TURBO_AUTO_TUNE=1)
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    def test_autotune(self, backend):
        self._bind_device()
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

    # ------------------------------------------------------------------
    # tp_size > 1 coverage
    # ------------------------------------------------------------------

    @parametrize("backend", _get_backends())
    def test_tp_size_2(self, backend):
        if dist.get_world_size() < 2:
            self.skipTest("requires world_size >= 2")
        self._bind_device()
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            _run_dispatch_combine(
                self.rank,
                None,
                num_tokens=512,
                hidden_size=512,
                num_experts=32,
                router_topk=2,
                tp_size=2,
            )

    @parametrize("backend", _get_backends())
    def test_tp_size_2_with_routing_map(self, backend):
        if dist.get_world_size() < 2:
            self.skipTest("requires world_size >= 2")
        self._bind_device()
        num_tokens = 512
        num_experts = 32
        router_topk = 2
        # Deterministic routing_map: mark the first `router_topk` experts True
        # for every token (pre-TP-expansion expert id space).
        routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device="cuda")
        routing_map[:, :router_topk] = True
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            _run_dispatch_combine(
                self.rank,
                None,
                num_tokens=num_tokens,
                hidden_size=512,
                num_experts=num_experts,
                router_topk=router_topk,
                tp_size=2,
                routing_map=routing_map,
            )

    @parametrize("backend", _get_backends())
    def test_tp_size_2_with_token_indices(self, backend):
        if dist.get_world_size() < 2:
            self.skipTest("requires world_size >= 2")
        self._bind_device()
        num_tokens = 512
        num_experts = 32
        router_topk = 2
        # token_indices in the pre-expansion expert id space; the dispatcher
        # is expected to TP-expand them internally.
        probs = torch.ones((num_tokens, num_experts), dtype=torch.float32, device="cuda") / router_topk
        _, token_indices = torch.topk(probs, router_topk, dim=-1)
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend}):
            _run_dispatch_combine(
                self.rank,
                None,
                num_tokens=num_tokens,
                hidden_size=512,
                num_experts=num_experts,
                router_topk=router_topk,
                tp_size=2,
                token_indices=token_indices,
            )


# ----------------------------------------------------------------------
# CUDA graph compatibility tests
#
# The C++ helper ``is_ep_force_current_stream()`` caches the value of
# ``PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM`` in a ``static`` local on first
# call, so toggling the env var inside a worker process after the first
# ``Buffer`` has been constructed is a no-op.  Because
# ``MultiProcContinuousTest`` reuses the same worker processes across every
# test method in a class, we cannot rely on ``patch.dict(os.environ, ...)``
# inside a test body to enable current-stream mode.
#
# Instead we put the CUDA-graph tests in a dedicated ``MultiProcContinuousTest``
# subclass.  Each subclass spawns its own worker pool via
# ``torch.multiprocessing.Process`` with the ``spawn`` start method, which
# copies the parent's ``os.environ`` at ``process.start()`` time.  By setting
# the env var in ``setUpClass`` and eagerly spawning the workers before
# restoring ``os.environ``, the workers for this class inherit the flag and
# their static cache is initialized to ``1`` on the very first Buffer
# construction, while other test classes' worker pools remain unaffected.
# ----------------------------------------------------------------------


@instantiate_parametrized_tests
class TestTokenDispatcherCudaGraph(MultiProcContinuousTest):
    world_size = -2

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _bind_device(self) -> None:
        torch.cuda.set_device(self.device)

    @classmethod
    def setUpClass(cls):
        os.environ["PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM"] = "1"
        super().setUpClass()

    @parametrize("backend", _get_backends())
    def test_cuda_graph(self, backend):
        from primus_turbo.pytorch.kernels.moe import moe_dispatch_combine_impl

        self._bind_device()
        with patch.dict(
            os.environ,
            {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend},
        ):
            num_worst_tokens = NUM_TOKENS * 8
            permute_max_token_num = NUM_TOKENS * 8 * ROUTER_TOPK

            # clear buffer for reset
            # a trick to reset the backend instance
            moe_dispatch_combine_impl._backend_instances.clear()

            set_buffer_global_config(num_use_cu=32)

            dispatcher = turbo.modules.DeepEPTokenDispatcher(
                NUM_EXPERTS,
                ROUTER_TOPK,
                dist.group.WORLD,
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


if __name__ == "__main__":
    run_tests()
