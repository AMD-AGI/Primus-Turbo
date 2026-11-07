###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Small two-GPU numerical regressions for dense/ragged context parallel attention."""

import os
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from primus_turbo.pytorch.ops import flash_attn_usp_func, flash_attn_varlen_usp_func
from tests.pytorch.ops.gradient_contract_helpers import attention_reference
from tests.pytorch.test_utils import compute_snr


def check(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.025, rtol=0.025)
    assert compute_snr(expected.float(), actual.float()) > 35


@instantiate_parametrized_tests
class AttentionCPGradientTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def initialize(self):
        os.environ["TRITON_CACHE_DIR"] = os.path.join(
            tempfile.gettempdir(), f"triton_cp_gradient_rank{self.rank}"
        )
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            "nccl",
            store=dist.FileStore(self.file_name, self.world_size),
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.manual_seed(718)

    @skip_if_lt_x_gpu(2)
    @parametrize("ulysses_degree", [1, 2])
    @parametrize("layout", ["bshd", "sbhd", "bhsd"])
    @parametrize("causal", [True, False])
    def test_dense(self, ulysses_degree, layout, causal):
        self.initialize()
        try:
            mesh = init_device_mesh(
                "cuda", (2 // ulysses_degree, ulysses_degree), mesh_dim_names=("ring", "ulysses")
            )
            full = [torch.randn(2, 256, h, 64, device="cuda", dtype=torch.bfloat16) for h in (4, 2, 2)]
            refs = [x.float().requires_grad_() for x in full]
            local = []
            for x in full:
                x = x.chunk(2, dim=1)[self.rank].contiguous()
                if layout == "sbhd":
                    x = x.transpose(0, 1).contiguous().transpose(0, 1)
                elif layout == "bhsd":
                    x = x.transpose(1, 2).contiguous().transpose(1, 2)
                local.append(x.detach().requires_grad_())
            with torch.autograd.set_multithreading_enabled(False):
                expected = attention_reference(*refs, causal=causal)
                out = flash_attn_usp_func(
                    *local,
                    causal=causal,
                    qkv_format=layout,
                    ulysses_group=mesh["ulysses"].get_group(),
                    ring_group=mesh["ring"].get_group(),
                )
                grad = torch.randn_like(full[0])
                expected.backward(grad.float())
                out.backward(grad.chunk(2, dim=1)[self.rank].contiguous())
            check(out, expected.chunk(2, dim=1)[self.rank])
            for x, ref in zip(local, refs):
                check(x.grad, ref.grad.chunk(2, dim=1)[self.rank])
        finally:
            dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @parametrize("causal", [True, False])
    def test_varlen_cross_rank_boundary(self, causal):
        self.initialize()
        try:
            # The second document crosses the rank boundary at token 128.
            lengths = [65, 191]
            cu = torch.tensor([0, 65, 256], device="cuda", dtype=torch.int32)
            full = [torch.randn(256, h, 64, device="cuda", dtype=torch.bfloat16) for h in (4, 2, 2)]
            refs = [x.float().requires_grad_() for x in full]
            local = [x.chunk(2)[self.rank].contiguous().detach().requires_grad_() for x in full]
            with torch.autograd.set_multithreading_enabled(False):
                expected = torch.cat(
                    [
                        attention_reference(*(x[s : s + n][None] for x in refs), causal=causal)[0]
                        for s, n in zip([0, 65], lengths)
                    ]
                )
                out = flash_attn_varlen_usp_func(
                    *local, cu, cu, 191, 191, causal=causal, ulysses_group=dist.group.WORLD
                )
                grad = torch.randn_like(full[0])
                expected.backward(grad.float())
                out.backward(grad.chunk(2)[self.rank].contiguous())
            check(out, expected.chunk(2)[self.rank])
            for x, ref in zip(local, refs):
                check(x.grad, ref.grad.chunk(2)[self.rank])
        finally:
            dist.destroy_process_group()
