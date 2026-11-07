###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-GPU Ulysses precision, head-padding and optional-output regressions."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScalingGranularity
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.ops import flash_attn_fp8_usp_func, flash_attn_usp_func, flash_attn_varlen_usp_func
from tests.pytorch.ops.gradient_contract_helpers import attention_reference
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950")


@instantiate_parametrized_tests
class AttentionCPPrecisionTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    @parametrize("mode", ["fp8", "odd_dense", "unequal_dense", "odd_varlen"])
    @parametrize("causal", [False, True])
    def test_precision_and_optional_outputs(self, mode, causal):
        torch.cuda.set_device(self.rank)
        dist.init_process_group("nccl", store=dist.FileStore(self.file_name, 2), rank=self.rank, world_size=2)
        try:
            mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("ring", "ulysses"))
            ring = mesh["ring"].get_group()
            torch.manual_seed(810)
            dim = 63 if mode.startswith("odd") else 64
            vdim = 56 if mode == "unequal_dense" else dim
            is_varlen = mode == "odd_varlen"
            full = [
                torch.randn(1, 256, h, d, device="cuda", dtype=torch.bfloat16)
                for h, d in [(4, dim), (2, dim), (2, vdim)]
            ]
            refs = [x.float().requires_grad_() for x in full]
            local = [x.chunk(2, dim=1)[self.rank].contiguous().detach().requires_grad_() for x in full]
            if is_varlen:
                local = [x.flatten(0, 1).detach().requires_grad_() for x in local]
                cu = torch.tensor([0, 65, 256], device="cuda", dtype=torch.int32)
            with torch.autograd.set_multithreading_enabled(False):
                if mode == "fp8":
                    out, lse = flash_attn_fp8_usp_func(
                        *local,
                        causal=causal,
                        return_lse=True,
                        ulysses_group=dist.group.WORLD,
                        ring_group=ring,
                    )
                    for config in [
                        Float8QuantConfig(),
                        Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128),
                    ]:
                        with pytest.raises(ValueError, match="only supports"):
                            flash_attn_fp8_usp_func(*local, fp8_config=config, ulysses_group=dist.group.WORLD)
                elif is_varlen:
                    with pytest.raises(NotImplementedError, match="ring is not supported"):
                        flash_attn_varlen_usp_func(
                            *local,
                            cu,
                            cu,
                            191,
                            191,
                            ulysses_group=dist.group.WORLD,
                            ring_group=dist.group.WORLD,
                        )
                    out, lse, _ = flash_attn_varlen_usp_func(
                        *local,
                        cu,
                        cu,
                        191,
                        191,
                        causal=causal,
                        return_lse=True,
                        return_attn_probs=True,
                        ulysses_group=dist.group.WORLD,
                    )
                else:
                    out, lse = flash_attn_usp_func(
                        *local,
                        causal=causal,
                        return_lse=True,
                        ring_group=ring,
                        ulysses_group=dist.group.WORLD,
                    )
                if is_varlen:
                    ref = torch.cat(
                        [
                            attention_reference(*(x[:, s : s + n] for x in refs), causal=causal)
                            for s, n in [(0, 65), (65, 191)]
                        ],
                        dim=1,
                    )
                else:
                    ref = attention_reference(*refs, causal=causal)
                grad = torch.randn_like(ref).bfloat16()
                ref.backward(grad.float())
                out.backward(grad.chunk(2, dim=1)[self.rank].reshape(out.shape).contiguous())
            assert torch.isfinite(lse).all()
            want = ref.chunk(2, dim=1)[self.rank].reshape(out.shape)
            assert compute_snr(want, out.float()) > (20 if mode == "fp8" else 35)
            for x, r in zip(local, refs):
                expected = r.grad.chunk(2, dim=1)[self.rank].reshape(x.shape)
                assert compute_snr(expected, x.grad.float()) > (15 if mode == "fp8" else 35)
        finally:
            dist.destroy_process_group()
