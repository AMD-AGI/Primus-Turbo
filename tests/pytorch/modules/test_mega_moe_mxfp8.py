###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 correctness test for the mega MoE forward with mxfp8 compute (bf16 comm).

Runs the milestone-1 forward path ``mega_moe_fused_mxfp8_forward`` (cross-rank
dispatch + combine in bf16, both FFN GEMMs in per-1x32 E8M0 block-scaled mxfp8) on
an EP8 world and gates SNR vs an fp32 dense MoE reference assembled from the
all-gathered global expert weights. The mxfp8 gate (15 dB) is looser than the bf16
kernel's 20 dB because two chained fp8 GEMMs + SwiGLU add quantization noise
(measured ~23 dB on this shape).

Run inside the FlyDSL container (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/modules/test_mega_moe_mxfp8.py
  # or: pytest tests/pytorch/modules/test_mega_moe_mxfp8.py
"""

import os

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

from primus_turbo.pytorch.core.low_precision import check_mxfp8_support
from tests.pytorch.modules.test_mega_moe import _dense_moe_reference
from tests.pytorch.test_utils import compute_snr

_WORLD = 8
_SNR_THRESHOLD_DB = 15.0  # mxfp8 compute vs fp32 dense; measured ~23 dB, broken << 10 dB
_H, _I, _E = 2048, 1024, 32
_T = 512


@instantiate_parametrized_tests
class TestMegaMoEMxfp8(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()

    @property
    def world_size(self) -> int:
        return _WORLD

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(123 + self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank, store=store)

    def _ep_group(self):
        return dist.new_group(list(range(self.world_size)))

    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("comm", ["fp8_fused", "fp8", "bf16"])
    @parametrize("top_k", [2, 4])
    def test_forward_mxfp8(self, top_k, comm):
        if not check_mxfp8_support():
            self.skipTest("MXFP8 requires gfx950")
        self._init_process()
        group = self._ep_group()
        from primus_turbo.flydsl.mega.fp8.mega_moe_fused_mxfp8 import (
            mega_moe_fused_mxfp8_forward,
        )

        world, rank, dev = self.world_size, self.rank, self.device
        epr = _E // world
        x = torch.randn(_T, _H, device=dev, dtype=torch.bfloat16)
        w1 = torch.randn(epr, 2 * _I, _H, device=dev, dtype=torch.bfloat16) * 0.05
        w2 = torch.randn(epr, _H, _I, device=dev, dtype=torch.bfloat16) * 0.05
        gate = torch.randn(_T, _E, device=dev)
        topk_w, topk_idx = torch.sigmoid(gate).topk(top_k, dim=-1)
        topk_w = (topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)

        with torch.no_grad():
            y = mega_moe_fused_mxfp8_forward(x, topk_idx, topk_w, w1, w2, group, comm=comm)
            # fp32 dense reference from the all-gathered global expert weights
            w1g = [torch.empty_like(w1) for _ in range(world)]
            w2g = [torch.empty_like(w2) for _ in range(world)]
            dist.all_gather(w1g, w1.contiguous(), group=group)
            dist.all_gather(w2g, w2.contiguous(), group=group)
            ref = _dense_moe_reference(
                x, topk_idx.to(torch.int64), topk_w, torch.cat(w1g, 0), torch.cat(w2g, 0)
            )

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        snr = torch.tensor([compute_snr(ref, y)], device=dev)
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)
        snr = float(snr.item())
        if rank == 0:
            print(f"[mxfp8 forward comm={comm} top_k={top_k}] min SNR = {snr:.2f} dB")
        self.assertGreaterEqual(snr, _SNR_THRESHOLD_DB, f"mxfp8 SNR {snr:.2f} dB < {_SNR_THRESHOLD_DB}")
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
