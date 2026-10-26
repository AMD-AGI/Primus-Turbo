###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# The reference flow this test drives is derived from DeepEP
# (https://github.com/deepseek-ai/DeepEP); see tests/pytorch/ref/deep_ep_ref.py.
#
# See LICENSE for license information.
###############################################################################

import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.deep_ep._compat import deep_ep_cpp
from tests.pytorch.ref.deep_ep_ref import (
    get_dispatch_layout_ref,
    per_token_cast_to_fp8,
    tune_and_verify_intranode,
)

# builds without the C++ DeepEP (gfx1250) have the FlyDSL backend only
HAS_CPP = deep_ep_cpp is not None
BACKENDS = ["cpp", "flydsl"] if HAS_CPP else ["flydsl"]


@instantiate_parametrized_tests
class DeepEPIntranodeTestCase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self, backend: str = "cpp"):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        buffer = pt.deep_ep.Buffer(dist.group.WORLD, int(1e9), intranode_backend=backend)
        assert buffer.intranode_backend == backend
        return buffer

    @skip_if_lt_x_gpu(2)
    @parametrize("num_tokens", [4096])
    @parametrize("hidden", [4096])
    @parametrize("num_topk", [8])
    @parametrize("num_experts", [128])
    @parametrize("num_sms", [24])
    @parametrize("backend", BACKENDS)
    def test_intranode(
        self, num_tokens: int, hidden: int, num_topk: int, num_experts: int, num_sms: int, backend: str
    ):
        # Random data
        buffer = self._init_process(backend)
        group = dist.group.WORLD
        rank = self.rank
        num_ranks = group.size()
        torch.manual_seed(42 + rank)

        tune_and_verify_intranode(
            num_sms, num_tokens, hidden, num_topk, num_experts, rank, num_ranks, rank, buffer, group
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("backend", BACKENDS)
    def test_dispatch_layout(self, backend: str):
        buffer = self._init_process(backend)
        num_ranks = dist.group.WORLD.size()
        torch.manual_seed(42 + self.rank)
        # (num_tokens, num_topk, num_experts, routing); top-k 20 exercises the per-expert fallback.
        cases = [
            (4099, 8, 128, "random"),
            (4099, 8, 128, "masked"),
            (32768, 8, 128, "even"),
            (1, 8, 128, "random"),
            (0, 8, 128, "random"),
            (777, 1, 64, "random"),
            (777, 9, 256, "masked"),
            (777, 20, 256, "random"),
        ]
        for num_tokens, num_topk, num_experts, routing in cases:
            if routing == "even":
                topk_idx = torch.arange(num_tokens * num_topk, device="cuda").view(num_tokens, num_topk)
                topk_idx = topk_idx % num_experts
            else:
                scores = torch.randn((num_tokens, num_experts), device="cuda").abs() + 1
                topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False)[1]
                if routing == "masked":
                    topk_idx.masked_fill_(torch.rand(topk_idx.shape, device="cuda") < 0.2, -1)

            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
                topk_idx, num_experts
            )
            ref_per_rank, _, ref_per_expert, ref_in_rank = get_dispatch_layout_ref(
                topk_idx, num_experts, num_ranks=num_ranks
            )
            case = (num_tokens, num_topk, num_experts, routing)
            assert torch.equal(num_tokens_per_rank, ref_per_rank), case
            assert torch.equal(num_tokens_per_expert, ref_per_expert), case
            assert torch.equal(is_token_in_rank, ref_in_rank), case

    @unittest.skipUnless(HAS_CPP, "needs the C++ DeepEP")
    @skip_if_lt_x_gpu(2)
    @parametrize("num_sms", [8, 24])
    def test_flydsl_matches_cpp(self, num_sms: int):
        """The FlyDSL backend returns bit-identical tensors and handles, and takes the C++ backend's handles."""
        cpp = self._init_process("cpp")
        fly = pt.deep_ep.Buffer(dist.group.WORLD, int(1e9), intranode_backend="flydsl")
        num_ranks = dist.group.WORLD.size()
        num_tokens, hidden, num_topk, num_experts = 1027, 2880, 8, 128
        torch.manual_seed(7 + self.rank)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        scores = torch.randn((num_tokens, num_experts), device="cuda").abs() + 1
        topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False)[1]
        topk_idx.masked_fill_(torch.rand(topk_idx.shape, device="cuda") < 0.1, -1)
        topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda")
        bias = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        config = pt.deep_ep.Config(num_sms, 8, 256)

        def assert_same(a, b, name):
            if isinstance(a, tuple):
                for i, (u, v) in enumerate(zip(a, b)):
                    assert_same(u, v, f"{name}[{i}]")
            elif isinstance(a, torch.Tensor):
                assert a.dtype == b.dtype and a.shape == b.shape, name
                assert torch.equal(
                    a.view(torch.uint8) if a.element_size() == 1 else a,
                    b.view(torch.uint8) if b.element_size() == 1 else b,
                ), name
            else:
                assert a == b, name

        layout = cpp.get_dispatch_layout(topk_idx, num_experts)
        assert_same(layout[:4], fly.get_dispatch_layout(topk_idx, num_experts)[:4], "layout")
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = layout
        # hidden 2880 (GPT-OSS) leaves a ragged row tail; the FP8 cast needs a multiple of 128
        for current_x in (x, per_token_cast_to_fp8(x[:, :2816].contiguous())):
            for with_topk in (False, True):
                args = dict(
                    x=current_x,
                    num_tokens_per_rank=num_tokens_per_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    config=config,
                )
                if with_topk:
                    args.update(topk_idx=topk_idx, topk_weights=topk_weights)
                ref, out = cpp.dispatch(**args), fly.dispatch(**args)
                assert_same(ref[:4], out[:4], f"dispatch(topk={with_topk})")
                # notify fills only this rank's column of the rank prefix matrix; the rest is heap residue
                assert_same(ref[4][0][:, self.rank], out[4][0][:, self.rank], "rank_prefix_matrix")
                assert_same(ref[4][1:], out[4][1:], f"handle(topk={with_topk})")
                handle = ref[4]
                if with_topk:
                    worst = num_tokens * num_ranks
                    ref_worst, out_worst = (
                        cpp.dispatch(**args, num_worst_tokens=worst),
                        fly.dispatch(**args, num_worst_tokens=worst),
                    )
                    # rows past the received ones are left unwritten, except top-k indices (-1)
                    n = ref[1].size(0)
                    x_ref, x_out = ref_worst[0], out_worst[0]
                    if isinstance(x_ref, tuple):
                        x_ref, x_out = tuple(t[:n] for t in x_ref), tuple(t[:n] for t in x_out)
                    else:
                        x_ref, x_out = x_ref[:n], x_out[:n]
                    assert_same(x_ref, x_out, "dispatch(num_worst_tokens) x")
                    assert_same(ref_worst[1], out_worst[1], "dispatch(num_worst_tokens) topk_idx")
                    assert_same(ref_worst[2][:n], out_worst[2][:n], "dispatch(num_worst_tokens) topk_weights")
                else:
                    # cached dispatch on the other backend's handle
                    assert_same(
                        ref[0], fly.dispatch(current_x, handle=handle, config=config)[0], "cached dispatch"
                    )
                if isinstance(current_x, tuple):
                    continue
                for combine_bias in (None, bias, (bias, bias)):
                    combine_args = dict(x=ref[0], handle=handle, config=config, bias=combine_bias)
                    if with_topk:
                        combine_args.update(topk_weights=ref[2])
                    assert_same(cpp.combine(**combine_args)[:2], fly.combine(**combine_args)[:2], "combine")
        torch.cuda.synchronize()
        dist.barrier()


if __name__ == "__main__":
    run_tests()
