###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 correctness test for the all-fp8 mega MoE forward.

Runs the fused all-fp8 forward ``mega_moe_fused_mxfp8`` (L1 = fused mxfp8 dispatch+fc1,
L2 = fp8 combine; both FFN GEMMs in per-1x32 E8M0 block-scaled mxfp8) on an EP8 world and
gates SNR vs an fp32 dense MoE reference assembled from the all-gathered global expert
weights. The mxfp8 gate (15 dB) is looser than the bf16
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
    @parametrize("top_k", [2, 4])
    def test_forward_mxfp8(self, top_k):
        if not check_mxfp8_support():
            self.skipTest("MXFP8 requires gfx950")
        self._init_process()
        group = self._ep_group()
        from primus_turbo.pytorch.ops.moe.mega_moe_fused_mxfp8 import mega_moe_fused_mxfp8

        world, rank, dev = self.world_size, self.rank, self.device
        epr = _E // world
        x = torch.randn(_T, _H, device=dev, dtype=torch.bfloat16)
        w1 = torch.randn(epr, 2 * _I, _H, device=dev, dtype=torch.bfloat16) * 0.05
        w2 = torch.randn(epr, _H, _I, device=dev, dtype=torch.bfloat16) * 0.05
        gate = torch.randn(_T, _E, device=dev)
        topk_w, topk_idx = torch.sigmoid(gate).topk(top_k, dim=-1)
        topk_w = (topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)

        with torch.no_grad():
            y = mega_moe_fused_mxfp8(group, x, topk_idx, topk_w, w1, w2)
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
            print(f"[mxfp8 forward top_k={top_k}] min SNR = {snr:.2f} dB")
        self.assertGreaterEqual(snr, _SNR_THRESHOLD_DB, f"mxfp8 SNR {snr:.2f} dB < {_SNR_THRESHOLD_DB}")
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("top_k", [2])
    def test_backward_mxfp8_smoke(self, top_k):
        """MXFP8 forward + backward autograd Function (mxfp8 fwd + bf16 STEP1/STEP3/dW1 +
        fp8 dW2). Smoke test: fwd+bwd runs end-to-end and all grads (dx / dW1 / dW2 /
        grad_topk_weights) are finite + correctly shaped.

        NOTE: a numerical gradcheck vs the fp32 dense reference (mirroring the bf16
        `test_mega_moe.py::test_backward_gradcheck`) was attempted but is currently blocked:
        that bf16 reference gradcheck itself GPU-faults in this working tree (pre-existing
        WIP changes to gemm_bf16_kernel/gemm_helper it depends on) on a clean GPU, so it
        can't serve as a numerical reference. The fp8 dW2 wgrad is separately gated at
        22.51 dB vs bf16 on real mega-pool data (test_dw2_fp8_vs_bf16); the mxfp8 forward at
        ~23 dB (test_forward_mxfp8)."""
        if not check_mxfp8_support():
            self.skipTest("MXFP8 requires gfx950")
        self._init_process()
        group = self._ep_group()
        from primus_turbo.pytorch.ops.moe.mega_moe_fused_mxfp8 import mega_moe_fused_mxfp8

        rank, dev = self.rank, self.device
        epr = _E // self.world_size
        x = torch.randn(_T, _H, device=dev, dtype=torch.bfloat16).requires_grad_(True)
        w1 = (torch.randn(epr, 2 * _I, _H, device=dev, dtype=torch.bfloat16) * 0.05).requires_grad_(True)
        w2 = (torch.randn(epr, _H, _I, device=dev, dtype=torch.bfloat16) * 0.05).requires_grad_(True)
        gate = torch.randn(_T, _E, device=dev)
        topk_w0, topk_idx = torch.sigmoid(gate).topk(top_k, dim=-1)
        tw = (topk_w0 / (topk_w0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32).requires_grad_(True)
        g_seed = torch.randn(_T, _H, device=dev, dtype=torch.bfloat16)

        y = mega_moe_fused_mxfp8(group, x, topk_idx, tw, w1, w2)
        self.assertEqual(y.shape, (_T, _H))
        y.backward(g_seed)

        self.assertEqual(x.grad.shape, (_T, _H))
        self.assertEqual(w1.grad.shape, (epr, 2 * _I, _H))
        self.assertEqual(w2.grad.shape, (epr, _H, _I))
        self.assertEqual(tw.grad.shape, (_T, top_k))
        for name, t in [("dx", x.grad), ("dW1", w1.grad), ("dW2", w2.grad), ("grad_topk", tw.grad)]:
            self.assertTrue(torch.isfinite(t.float()).all().item(), f"{name} grad non-finite")
        if rank == 0:
            print(f"[mxfp8 backward smoke top_k={top_k}] fwd+bwd OK, all grads finite + shaped")
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("top_k", [2])
    def test_dw2_fp8_vs_bf16(self, top_k):
        """Validate fp8 dW2 on REAL mega-pool data (same-tensor comparison): replicate the
        backward's STEP1 (dispatch dy) + STEP2 (swiglu^T) once to get dispatch_l2_grad +
        act_weighted, then compute dW2 both ways (bf16 grouped_gemm_variable_k vs the fp8
        mxfp8 variable-K wgrad) on those SAME tensors and gate by SNR. This is the correct
        comparison (the earlier two-pass toggle diffed different pool data)."""
        if not check_mxfp8_support():
            self.skipTest("MXFP8 requires gfx950")
        self._init_process()
        group = self._ep_group()
        import primus_turbo.pytorch.ops.moe.mega_moe_fused_mxfp8 as mxmod
        from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
        from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
            dispatch_grouped_gemm_mxfp8,
        )
        from primus_turbo.flydsl.mega.fp8.quant import (
            quantize_grouped_weight_mxfp8,
            quantize_rowwise_mxfp8,
        )
        from primus_turbo.flydsl.mega.swiglu_kernel import swiglu_backward
        from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
            grouped_gemm_variable_k_impl,
        )
        from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (
            dispatch_grouped_gemm_impl,
        )

        rank, dev = self.rank, self.device
        world, epr = self.world_size, _E // self.world_size
        x = torch.randn(_T, _H, device=dev, dtype=torch.bfloat16)
        w1 = torch.randn(epr, 2 * _I, _H, device=dev, dtype=torch.bfloat16) * 0.05
        w2 = torch.randn(epr, _H, _I, device=dev, dtype=torch.bfloat16) * 0.05
        gate = torch.randn(_T, _E, device=dev)
        topk_w0, topk_idx = torch.sigmoid(gate).topk(top_k, dim=-1)
        topk_w = (topk_w0 / (topk_w0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)

        with torch.no_grad():
            # replicate the forward's L1 (fused mxfp8 dispatch+fc1) to get handle / l1 /
            # dispatch_weights (the intermediates the backward's STEP1+STEP2 need)
            symm = get_symm_buffer_for_mega_moe(
                group, num_experts=epr * world, num_max_tokens_per_rank=_T, num_topk=top_k,
                hidden=_H, intermediate_hidden=_I, block_m=256, block_n=256, use_mxfp8=True,
            )
            sym_layout = symm.make_sym_layout()
            handle = tuple(
                dispatch_prologue(
                    topk_idx.to(torch.int64), topk_w, sym_layout=sym_layout, num_tokens=_T,
                    num_topk=top_k, num_experts=epr * world, world_size=world, rank=symm.rank,
                    experts_per_rank=epr, block_m=256, num_max_pool_tokens=symm.num_max_pool_tokens,
                )
            )
            w1q, w1s = quantize_grouped_weight_mxfp8(w1)
            xq, xs = quantize_rowwise_mxfp8(x)
            symm.scoreboard.zero_()
            torch.cuda.synchronize(); group.barrier()
            l1 = dispatch_grouped_gemm_mxfp8(xq, xs, w1q, w1s, handle, sym_layout, symm, BM=256, BN=256)
            dispatch_weights = symm.weight_recv_buf.clone()

            dy = torch.randn(_T, _H, device=dev, dtype=torch.bfloat16)
            grad_swiglu, dispatch_l2_grad, _, _ = dispatch_grouped_gemm_impl(
                dy, w2, group, BackendType.FLYDSL.value, handle=handle, layout="nn", num_dispatch_cu=16,
            )
            _, _, act_weighted = swiglu_backward(
                grad_swiglu, l1, scale=dispatch_weights, return_gate=True, return_act_w=True,
            )
            group_lens, group_offs = handle[9], handle[10]
            dW2_bf = grouped_gemm_variable_k_impl(
                dispatch_l2_grad, act_weighted, group_lens, group_offs,
                trans_a=True, trans_b=False, trans_c=False, num_cu=None,
                default_backend=BackendType.TRITON.value,
            )
            dW2_fp8 = mxmod._mxfp8_variable_k_wgrad(dispatch_l2_grad, act_weighted, group_lens, group_offs)

        self.assertEqual(dW2_fp8.shape, dW2_bf.shape)
        self.assertTrue(torch.isfinite(dW2_fp8.float()).all().item(), "fp8 dW2 non-finite")
        snr = torch.tensor([compute_snr(dW2_bf, dW2_fp8)], device=dev)
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)
        snr = float(snr.item())
        if rank == 0:
            print(f"[mxfp8 dW2 top_k={top_k}] fp8 vs bf16 min SNR = {snr:.2f} dB (fmt={mxmod._DW2_FP8_FORMAT})")
        self.assertGreaterEqual(snr, 15.0, f"fp8 dW2 SNR {snr:.2f} dB < 15.0")
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
