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

import primus_turbo.flydsl.deep_ep.arch as flydsl_arch
import primus_turbo.pytorch as pt
import primus_turbo.pytorch.deep_ep._compat as deep_ep_compat
import primus_turbo.pytorch.deep_ep.buffer as deep_ep_buffer
import primus_turbo.pytorch.deep_ep.flydsl_intranode as flydsl_intranode
from primus_turbo.flydsl.deep_ep.common import IS_GFX1250
from primus_turbo.flydsl.deep_ep.intranode_v2_kernel import combine_v2_layout, dispatch_v2_layout
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    _get_backend_instance,
    _resolve_backend_name,
    clear_backend_instances,
    set_buffer_global_config,
)
from tests.pytorch.ref.deep_ep_ref import (
    get_dispatch_layout_ref,
    tune_and_verify_intranode,
)

# gfx1250 runs the FlyDSL backends (the C++ DeepEP is not loaded there), every other GPU the C++ one
FLYDSL_BACKENDS = ["flydsl", "flydsl_v2"]
BACKENDS = FLYDSL_BACKENDS if IS_GFX1250 else ["cpp"]


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
    @parametrize("backend", BACKENDS)
    def test_combine_ignores_the_worst_case_tail(self, backend: str):
        """After a ``num_worst_tokens`` dispatch, the combine's x has the worst-case row count and
        only the received rows are meaningful: NaN in the rest must not reach any token."""
        buffer = self._init_process(backend)
        num_ranks = dist.group.WORLD.size()
        num_tokens, hidden, num_topk, num_experts = 1024, 4096, 8, 128
        torch.manual_seed(42 + self.rank)
        for _ in range(8):
            x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            topk_idx = torch.topk(torch.rand((num_tokens, num_experts), device="cuda"), num_topk, dim=-1)[1]
            topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device="cuda")
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
                topk_idx, num_experts
            )
            recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = buffer.dispatch(
                x,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_worst_tokens=num_tokens * num_ranks,
            )
            assert recv_x.size(0) == num_tokens * num_ranks, recv_x.shape
            received = int((recv_topk_idx >= 0).any(dim=1).sum())
            recv_x[received:] = float("nan")
            recv_topk_weights[received:] = float("nan")
            combined_x, combined_topk_weights, _ = buffer.combine(
                recv_x, handle, topk_weights=recv_topk_weights
            )
            torch.cuda.synchronize()
            ref_x = x.float() * is_token_in_rank.sum(dim=1, keepdim=True)
            assert torch.isfinite(combined_x).all(), "the worst-case tail reached the combine"
            assert torch.allclose(combined_x.float(), ref_x, rtol=2e-2, atol=2e-2)
            assert torch.allclose(combined_topk_weights, topk_weights)
        dist.barrier()

    @unittest.skipUnless(IS_GFX1250, "the FlyDSL backends run on gfx1250")
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
        """A Buffer with no backend takes this GPU's default and runs a dispatch + combine."""
        expected = "flydsl_v2" if IS_GFX1250 else "cpp"
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

    @unittest.skipUnless(IS_GFX1250, "flydsl_v2 runs on gfx1250 only")
    @skip_if_lt_x_gpu(2)
    def test_moe_ops_size_the_v2_buffer(self):
        """moe_dispatch / moe_combine on flydsl_v2 through init_buffer: the buffer is sized from the
        dispatched tokens and grows for a larger dispatch."""
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

    @unittest.skipUnless(IS_GFX1250, "flydsl_v2 runs on gfx1250 only")
    @skip_if_lt_x_gpu(2)
    def test_moe_ops_v2_buffer_with_uneven_tokens(self):
        """moe_dispatch / moe_combine on flydsl_v2 with a different number of tokens on every rank: the
        buffers are sized from the group's largest dispatch, so later dispatches up to it do not grow them."""
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
        max_tokens = 2048 * num_ranks
        sizes = []
        # the last rank sends the most first, then rank 0 does
        for num_tokens in (2048 * (self.rank + 1), 2048 * (num_ranks - self.rank)):
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
            hint = pt.deep_ep.Buffer.get_token_buffer_size_hint(max_tokens, 2 * hidden, num_ranks, num_topk)
            assert backend._buffer.num_nvl_bytes >= hint, (backend._buffer.num_nvl_bytes, hint)
            sizes.append(backend._buffer.num_nvl_bytes)
        assert sizes[0] == sizes[1], sizes
        dist.barrier()


class IntranodeBackendSelectionTestCase(TestCase):
    """``get_intranode_backend`` per architecture, host only (``get_rocm_arch`` is patched)."""

    def _resolve(self, arch: str, backend: Optional[str] = None) -> str:
        with mock.patch.object(flydsl_arch, "get_rocm_arch", return_value=arch):
            return flydsl_intranode.get_intranode_backend(backend)

    def test_default_per_architecture(self):
        expected = {"gfx942": "cpp", "gfx950": "cpp", "gfx942:sramecc+:xnack-": "cpp", "gfx1250": "flydsl_v2"}
        for arch, backend in expected.items():
            assert self._resolve(arch) == backend, arch

    def test_other_architectures_raise(self):
        for arch in ("gfx1260", "gfx1100", "gfx90a"):
            for backend in (None, "cpp", "flydsl", "flydsl_v2"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"not supported on {arch} \(supported: gfx942, gfx950, gfx1250\); use the all-to-all \(RCCL\)",
                ):
                    self._resolve(arch, backend)

    def test_other_architectures_log_the_rccl_fallback(self):
        with mock.patch.object(flydsl_arch.logger, "error") as log_error:
            with self.assertRaises(RuntimeError):
                self._resolve("gfx1260")
        log_error.assert_called_once()
        assert "not supported on gfx1260" in log_error.call_args.args[0]
        assert "all-to-all (RCCL) token dispatcher" in log_error.call_args.args[0]

    def test_no_environment_variable_picks_the_backend(self):
        with mock.patch.dict(os.environ, {"PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND": "flydsl"}):
            assert self._resolve("gfx1250") == "flydsl_v2"
            assert self._resolve("gfx950") == "cpp"

    def test_argument_overrides_the_default(self):
        for arch in ("gfx1250", "gfx950"):
            for backend in ("cpp", "flydsl", "flydsl_v2", " FlyDSL "):
                assert self._resolve(arch, backend) == backend.strip().lower()
        with self.assertRaisesRegex(ValueError, "Unknown DeepEP intranode backend"):
            self._resolve("gfx1250", "v2")

    def test_flydsl_v2_on_gfx9_names_the_default(self):
        with (
            mock.patch.object(flydsl_arch, "get_rocm_arch", return_value="gfx950"),
            self.assertRaisesRegex(
                RuntimeError, "does not run on gfx942 / gfx950; leave intranode_backend unset.*'cpp'"
            ),
        ):
            flydsl_intranode.FlyDSLIntranodeV2Runtime(None, 0, None)


class ArchDependentHostTestCase(TestCase):
    """The host decisions that follow ``get_rocm_arch`` (patched), host only."""

    def _arch(self, arch: str):
        return mock.patch.object(flydsl_arch, "get_rocm_arch", return_value=arch)

    def test_cpp_deepep_loads_on_its_architectures_only(self):
        try:
            import primus_turbo.pytorch._C.deep_ep  # noqa: F401

            built = True
        except ImportError:
            built = False
        for arch in ("gfx942", "gfx950", "gfx1250", "gfx1260"):
            with self._arch(arch):
                module = deep_ep_compat._load_deep_ep_cpp()
            loads = built and arch in flydsl_arch.CPP_DEEPEP_ARCHS
            assert (module is not None) == loads, (arch, built)

    def test_flydsl_v1_on_gfx1250_takes_larger_combine_chunks(self):
        expected = {
            ("gfx1250", "flydsl", 2): 48,
            ("gfx1250", "flydsl", 4): 24,
            ("gfx1250", "flydsl", 8): 4,
            ("gfx1250", "flydsl_v2", 2): 10,
            ("gfx1250", None, 2): 10,
            ("gfx950", "flydsl", 2): 10,
            ("gfx950", None, 4): 9,
        }
        for (arch, backend, num_ranks), send_tokens in expected.items():
            with self._arch(arch):
                config = pt.deep_ep.Buffer.get_combine_config(num_ranks, backend)
            assert config.num_max_nvl_chunked_send_tokens == send_tokens, (arch, backend, num_ranks)

    def test_device_arch_must_match_get_rocm_arch(self):
        props = mock.Mock(gcnArchName="gfx942:sramecc+:xnack-")
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_properties", return_value=props),
        ):
            with self._arch("gfx942"):
                deep_ep_buffer._check_device_arch()
            with (
                self._arch("gfx1250"),
                self.assertRaisesRegex(
                    RuntimeError, "this GPU is gfx942, but get_rocm_arch\\(\\) returns gfx1250"
                ),
            ):
                deep_ep_buffer._check_device_arch()

    def test_missing_cpp_deepep_names_the_reason(self):
        with mock.patch.object(deep_ep_buffer, "deep_ep_cpp", None):
            with (
                self._arch("gfx1250"),
                self.assertRaisesRegex(
                    RuntimeError,
                    "not loaded on gfx1250: only gfx942 / gfx950 run it. On gfx1250 DeepEP runs intranode only: "
                    "2 / 4 / 8 ranks.*all-to-all \\(RCCL\\) token dispatcher",
                ),
            ):
                deep_ep_buffer._require_deep_ep_cpp("the low-latency mode")
            with (
                self._arch("gfx950"),
                self.assertRaisesRegex(
                    RuntimeError, "this build of primus_turbo does not include.*needs a build with it"
                ),
            ):
                deep_ep_buffer._require_deep_ep_cpp("the low-latency mode")
            with self._arch("gfx1260"), self.assertRaisesRegex(RuntimeError, "not supported on gfx1260"):
                deep_ep_buffer._require_deep_ep_cpp("the low-latency mode")


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
