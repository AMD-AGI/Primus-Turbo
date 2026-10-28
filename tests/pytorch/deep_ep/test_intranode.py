###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# The reference flow this test drives is derived from DeepEP
# (https://github.com/deepseek-ai/DeepEP); see tests/pytorch/ref/deep_ep_ref.py.
#
# See LICENSE for license information.
###############################################################################

import os
import unittest
from typing import Optional
from unittest import mock

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import primus_turbo.pytorch as pt
import primus_turbo.pytorch.deep_ep.flydsl_intranode as flydsl_intranode
from primus_turbo.flydsl.deep_ep.common import IS_GFX1250
from primus_turbo.flydsl.deep_ep.intranode_v2_kernel import combine_v2_layout, dispatch_v2_layout
from primus_turbo.pytorch.deep_ep._compat import deep_ep_cpp
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    _get_backend_instance,
    _resolve_backend_name,
    clear_backend_instances,
    set_buffer_global_config,
)
from tests.pytorch.ref.deep_ep_ref import (
    get_dispatch_layout_ref,
    per_token_cast_to_fp8,
    tune_and_verify_intranode,
)

# builds without the C++ DeepEP (gfx1250) have the FlyDSL backends only; flydsl_v2 runs on gfx1250 only
HAS_CPP = deep_ep_cpp is not None
FLYDSL_BACKENDS = ["flydsl"] + (["flydsl_v2"] if IS_GFX1250 else [])
BACKENDS = (["cpp"] if HAS_CPP else []) + FLYDSL_BACKENDS


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

    def _init_process(self, backend: Optional[str] = "cpp", **buffer_kwargs):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        buffer = pt.deep_ep.Buffer(dist.group.WORLD, int(1e9), intranode_backend=backend, **buffer_kwargs)
        assert backend is None or buffer.intranode_backend == backend
        return buffer

    @staticmethod
    def _check_round_trip(buffer, num_tokens=512, hidden=4096, num_topk=4, num_experts=64):
        """Dispatch random tokens and combine them back; returns the dispatch's x, outputs and handle."""
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        topk_idx = torch.topk(torch.rand((num_tokens, num_experts), device="cuda"), num_topk, dim=-1)[1]
        topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device="cuda")
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, num_experts
        )
        recv_x, _, recv_topk_weights, _, handle, _ = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        combined_x, combined_topk_weights, _ = buffer.combine(recv_x, handle, topk_weights=recv_topk_weights)
        torch.cuda.synchronize()
        ref_x = x.float() * is_token_in_rank.sum(dim=1, keepdim=True)
        assert torch.allclose(combined_x.float(), ref_x, rtol=2e-2, atol=2e-2)
        assert torch.allclose(combined_topk_weights, topk_weights)
        return x, recv_x, recv_topk_weights, handle

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

    @skip_if_lt_x_gpu(2)
    @parametrize("backend", BACKENDS)
    def test_rank_receives_nothing(self, backend: str):
        """No token goes to rank 0: its dispatch receives nothing and its combine pushes nothing."""
        buffer = self._init_process(backend)
        num_ranks = dist.group.WORLD.size()
        num_tokens, hidden, num_topk, num_experts = 64, 4096, 4, 64
        torch.manual_seed(42 + self.rank)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        scores = torch.rand((num_tokens, num_experts), device="cuda")
        scores[:, : num_experts // num_ranks] = -1
        topk_idx = torch.topk(scores, num_topk, dim=-1)[1]
        topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device="cuda")

        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, num_experts
        )
        recv_x, _, recv_topk_weights, _, handle, _ = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        if self.rank == 0:
            assert recv_x.size(0) == 0, recv_x.shape
        combined_x, combined_topk_weights, _ = buffer.combine(recv_x, handle, topk_weights=recv_topk_weights)
        torch.cuda.synchronize()
        ref_x = x.float() * is_token_in_rank.sum(dim=1, keepdim=True)
        assert torch.allclose(combined_x.float(), ref_x, rtol=2e-2, atol=2e-2)
        assert torch.allclose(combined_topk_weights, topk_weights)

    @skip_if_lt_x_gpu(2)
    @parametrize("backend", FLYDSL_BACKENDS)
    def test_destroy_and_recreate(self, backend: str):
        """``destroy()`` (collective) frees the heap, and a new buffer on the same group works."""
        buffer = self._init_process(backend, explicitly_destroy=True)
        torch.manual_seed(42 + self.rank)
        for _ in range(2):
            self._check_round_trip(buffer)
            buffer.destroy()
            buffer = pt.deep_ep.Buffer(
                dist.group.WORLD, int(1e9), intranode_backend=backend, explicitly_destroy=True
            )
        self._check_round_trip(buffer)
        buffer.destroy()

    @skip_if_lt_x_gpu(2)
    def test_default_backend(self):
        """A Buffer with no backend and no PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND takes this GPU's default
        and runs a dispatch + combine."""
        os.environ.pop("PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND", None)
        expected = "flydsl_v2" if IS_GFX1250 else ("cpp" if HAS_CPP else "flydsl")
        buffer = self._init_process(None)
        assert buffer.intranode_backend == expected, (buffer.intranode_backend, expected)
        torch.manual_seed(42 + self.rank)
        self._check_round_trip(buffer)
        dist.barrier()

    @unittest.skipUnless(IS_GFX1250, "flydsl_v2 runs on gfx1250 only")
    @skip_if_lt_x_gpu(2)
    def test_flydsl_v2_refuses_graph_capture(self):
        """A replay would rerun a call's host-side epochs: flydsl_v2 dispatch and combine raise under
        capture before touching any state, and the buffer keeps working."""
        buffer = self._init_process("flydsl_v2")
        torch.manual_seed(42 + self.rank)
        x, recv_x, recv_topk_weights, handle = self._check_round_trip(buffer)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with self.assertRaisesRegex(RuntimeError, "CUDA graph"):
                buffer.dispatch(x, handle=handle)
            with self.assertRaisesRegex(RuntimeError, "CUDA graph"):
                buffer.combine(recv_x, handle, topk_weights=recv_topk_weights)
        self._check_round_trip(buffer)
        dist.barrier()

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

    @unittest.skipUnless(IS_GFX1250, "flydsl_v2 runs on gfx1250 only")
    @skip_if_lt_x_gpu(2)
    def test_moe_ops_size_the_v2_buffer(self):
        """moe_dispatch / moe_combine on flydsl_v2 through init_buffer: the buffer is sized from the
        dispatched tokens and grows for a larger dispatch."""
        os.environ["PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND"] = "flydsl_v2"
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank, store=store)
        group = dist.group.WORLD
        num_ranks = group.size()
        clear_backend_instances()
        set_buffer_global_config(num_use_cu=24)
        backend = _get_backend_instance(_resolve_backend_name())
        hidden, num_topk, num_experts = 4096, 4, 64
        torch.manual_seed(42 + self.rank)
        sizes = []
        for num_tokens in (4096, 12288):
            x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            topk_idx = torch.topk(torch.rand((num_tokens, num_experts), device="cuda"), num_topk, dim=-1)[1]
            topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device="cuda")
            recv_x, _, _, _, handle = pt.ops.moe_dispatch(x, topk_idx, topk_weights, num_experts, group)
            combined_x = pt.ops.moe_combine(recv_x, group, handle)
            torch.cuda.synchronize()
            in_rank = torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device="cuda")
            in_rank.scatter_(1, topk_idx // (num_experts // num_ranks), True)
            ref_x = x.float() * in_rank.sum(dim=1, keepdim=True)
            assert torch.allclose(combined_x.float(), ref_x, rtol=2e-2, atol=2e-2)
            buffer = backend._buffer
            assert buffer.intranode_backend == "flydsl_v2"
            hint = pt.deep_ep.Buffer.get_token_buffer_size_hint(num_tokens, 2 * hidden, num_ranks, num_topk)
            assert buffer.num_nvl_bytes >= hint, (buffer.num_nvl_bytes, hint)
            sizes.append(buffer.num_nvl_bytes)
        assert sizes[1] > sizes[0], sizes
        dist.barrier()


class IntranodeBackendSelectionTestCase(TestCase):
    """``get_intranode_backend`` per architecture, host only (the architecture and the C++ DeepEP are
    patched)."""

    ENV = "PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND"

    def _patched(self, gfx1250: bool, has_cpp: bool, env: Optional[str] = None):
        environ = {k: v for k, v in os.environ.items() if k != self.ENV}
        if env is not None:
            environ[self.ENV] = env
        return (
            mock.patch.object(flydsl_intranode, "IS_GFX1250", gfx1250),
            mock.patch.object(flydsl_intranode, "deep_ep_cpp", object() if has_cpp else None),
            mock.patch.dict(os.environ, environ, clear=True),
        )

    def _resolve(self, gfx1250: bool, has_cpp: bool, env: Optional[str] = None, backend=None) -> str:
        arch, cpp, environ = self._patched(gfx1250, has_cpp, env)
        with arch, cpp, environ:
            return flydsl_intranode.get_intranode_backend(backend)

    def test_default_per_architecture(self):
        # gfx1250 never loads the C++ DeepEP; has_cpp=True covers a build that would
        assert self._resolve(gfx1250=True, has_cpp=False) == "flydsl_v2"
        assert self._resolve(gfx1250=True, has_cpp=True) == "flydsl_v2"
        assert self._resolve(gfx1250=False, has_cpp=True) == "cpp"
        assert self._resolve(gfx1250=False, has_cpp=False) == "flydsl"

    def test_env_and_argument_override_the_default(self):
        for gfx1250, has_cpp in ((True, False), (False, True)):
            for env in ("cpp", "flydsl", "flydsl_v2", " FlyDSL "):
                assert self._resolve(gfx1250, has_cpp, env=env) == env.strip().lower()
            assert self._resolve(gfx1250, has_cpp, env="cpp", backend="flydsl") == "flydsl"
        with self.assertRaisesRegex(ValueError, "Unknown DeepEP intranode backend"):
            self._resolve(True, False, env="v2")

    def test_flydsl_v2_off_gfx1250_names_the_alternatives(self):
        arch, cpp, environ = self._patched(gfx1250=False, has_cpp=True)
        with (
            arch,
            cpp,
            environ,
            self.assertRaisesRegex(RuntimeError, "gfx1250 only.*'cpp' \\(the default\\)"),
        ):
            flydsl_intranode.FlyDSLIntranodeRuntime(None, 0, None, version=2)


class TokenBufferSizeHintTestCase(TestCase):
    """``Buffer.get_token_buffer_size_hint``, host only."""

    def test_flydsl_v2_heap_holds_the_batch(self):
        for num_tokens, num_ranks, hidden, num_topk in (
            (4096, 2, 7168, 8),
            (32768, 4, 2880, 4),
            (1, 8, 256, 1),
        ):
            hidden_bytes, rows = 2 * hidden, num_tokens * num_ranks
            num_bytes = pt.deep_ep.Buffer.get_token_buffer_size_hint(
                num_tokens, hidden_bytes, num_ranks, num_topk, backend="flydsl_v2"
            )
            # bf16 rows, fp8 rows with per-32 scales, and the combine slots
            assert dispatch_v2_layout(num_bytes, hidden_bytes, num_topk, 8, 0)[1] >= rows
            assert dispatch_v2_layout(num_bytes, hidden, num_topk, 8, hidden // 32)[1] >= rows
            assert combine_v2_layout(num_bytes, hidden_bytes, num_topk)[1] >= rows

    def test_ring_backends_take_no_token_hint(self):
        for backend in ("cpp", "flydsl"):
            assert pt.deep_ep.Buffer.get_token_buffer_size_hint(4096, 8192, 2, 8, backend=backend) == 0
        with self.assertRaises(ValueError):
            pt.deep_ep.Buffer.get_token_buffer_size_hint(0, 8192, 2, 8, backend="flydsl_v2")


if __name__ == "__main__":
    run_tests()
