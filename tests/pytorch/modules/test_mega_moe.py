###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the ``MegaMoE`` expert-parallel MoE module.

Style mirrors Megatron-LM / Primus MoE-layer unit tests: an EP8 world is brought
up with ``MultiProcessTestCase``; a small ``MegaMoETestContainer`` builds the
module and runs forward / forward+backward; correctness is checked against a
single-GPU dense reference assembled from the module's gathered global weights.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/modules/test_mega_moe.py
  # or: pytest tests/pytorch/modules/test_mega_moe.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from primus_turbo.pytorch.modules.moe.mega_moe import MegaMoE
from tests.pytorch.test_utils import compute_snr

_WORLD = 8
# bf16 fused kernel vs fp32 dense reference; worst rank observed ~26 dB, broken < 10 dB
_SNR_THRESHOLD_DB = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Reference helpers.
# ─────────────────────────────────────────────────────────────────────────────
def _dense_moe_reference(x, topk_idx, topk_w, w1g, w2g):
    """fp32 dense MoE given the routing decision (weight applied at the W2 output)."""
    xf = x.float()
    out = torch.zeros_like(xf)
    for e in range(w1g.shape[0]):
        we = (topk_w * (topk_idx == e)).sum(dim=-1)  # [T] weight of expert e per token
        sel = we > 0
        if not sel.any():
            continue
        xe = xf[sel]
        gate, up = (xe @ w1g[e].float().t()).chunk(2, dim=-1)
        o = (F.silu(gate) * up) @ w2g[e].float().t()
        out[sel] += we[sel].unsqueeze(-1) * o
    return out


def _shared_expert_reference(x, sw1, sw2, sgate):
    """fp32 replicated SwiGLU shared expert (matches ``MegaMoE._shared_expert``)."""
    xf = x.float()
    gate, up = (xf @ sw1.float().t()).chunk(2, dim=-1)
    out = (F.silu(gate) * up) @ sw2.float().t()
    if sgate is not None:
        out = torch.sigmoid(xf @ sgate.float().t()) * out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Test container: builds the module on the EP group + reference utilities.
# ─────────────────────────────────────────────────────────────────────────────
class MegaMoETestContainer:
    def __init__(
        self,
        ep_group,
        *,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        shared_expert=False,
        test_dtype=torch.bfloat16,
    ):
        self.ep_group = ep_group
        self.world = ep_group.size()
        self.rank = ep_group.rank()
        self.hidden_size = hidden_size
        device = torch.device("cuda", torch.cuda.current_device())
        self.moe = MegaMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            ep_group=ep_group,
            shared_expert_intermediate_size=intermediate_size if shared_expert else None,
            shared_expert_gate=shared_expert,
            device=device,
            dtype=test_dtype,
        )

    def gather_global_weights(self):
        """All-gather per-rank expert shards into full [E, ...] tensors."""
        w1 = [torch.empty_like(self.moe.w1) for _ in range(self.world)]
        w2 = [torch.empty_like(self.moe.w2) for _ in range(self.world)]
        dist.all_gather(w1, self.moe.w1.contiguous(), group=self.ep_group)
        dist.all_gather(w2, self.moe.w2.contiguous(), group=self.ep_group)
        return torch.cat(w1, dim=0), torch.cat(w2, dim=0)

    def reference(self, x, topk_idx, topk_w):
        w1g, w2g = self.gather_global_weights()
        ref = _dense_moe_reference(x, topk_idx, topk_w, w1g, w2g)
        if self.moe.shared_w1 is not None:
            ref = ref + _shared_expert_reference(
                x, self.moe.shared_w1, self.moe.shared_w2, self.moe.shared_gate_weight
            )
        return ref


# ─────────────────────────────────────────────────────────────────────────────
# EP8 multi-process test case.
# ─────────────────────────────────────────────────────────────────────────────
@instantiate_parametrized_tests
class TestMegaMoE(MultiProcessTestCase):
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

    def _assert_snr(self, actual, ref, *, tag):
        """SNR (dB) of fused output vs dense reference; worst rank governs the EP group."""
        snr = torch.tensor([compute_snr(ref, actual)], device=self.device)  # ref is the signal
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)  # conservative: weakest rank
        snr = float(snr.item())
        if self.rank == 0:
            print(f"[{tag}] min SNR = {snr:.2f} dB")
        self.assertGreaterEqual(snr, _SNR_THRESHOLD_DB, f"[{tag}] SNR {snr:.2f} dB < {_SNR_THRESHOLD_DB}")

    # ── correctness: fused forward vs dense reference ─────────────────────────
    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("top_k", [2, 4])
    @parametrize("shared_expert", [False, True])
    def test_forward(self, top_k, shared_expert):
        self._init_process()
        group = self._ep_group()
        c = MegaMoETestContainer(
            group,
            hidden_size=2048,
            intermediate_size=1024,
            num_experts=32,
            top_k=top_k,
            shared_expert=shared_expert,
        )
        x = torch.randn((512, c.hidden_size), device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            y, bias = c.moe(x)
            topk_idx, topk_w = c.moe.route(x)  # router is separately tested; reuse its decision
            ref = c.reference(x, topk_idx, topk_w)

        self.assertIsNone(bias, "MegaMoE must return (output, None) like Megatron MoELayer")
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        self._assert_snr(y, ref, tag=f"forward top_k={top_k} shared={shared_expert}")
        dist.destroy_process_group()

    # ── forward + backward: output contract + every param receives a grad ─────
    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("shared_expert", [False, True])
    def test_forward_backward(self, shared_expert):
        self._init_process()
        group = self._ep_group()
        c = MegaMoETestContainer(
            group,
            hidden_size=2048,
            intermediate_size=1024,
            num_experts=32,
            top_k=4,
            shared_expert=shared_expert,
        )
        x = torch.randn((512, c.hidden_size), device=self.device, dtype=torch.bfloat16, requires_grad=True)
        y, _ = c.moe(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, torch.bfloat16)

        y.sum().backward()
        self.assertIsNotNone(x.grad, "input gradient should exist")
        self.assertEqual(x.grad.dtype, torch.bfloat16)
        self.assertEqual(x.grad.shape, x.shape)
        # Megatron parity: every trainable parameter must receive a gradient
        for name, param in c.moe.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"gradient for {name} should exist")
        dist.destroy_process_group()

    # ── padding_mask: padded tokens contribute a zero MoE row ─────────────────
    @skip_if_lt_x_gpu(_WORLD)
    def test_padding_mask(self):
        self._init_process()
        group = self._ep_group()
        c = MegaMoETestContainer(group, hidden_size=2048, intermediate_size=1024, num_experts=32, top_k=4)
        seq, batch = 512, 1
        # Megatron layout: hidden [seq, batch, H], padding_mask [seq, batch], True = PADDING
        x = torch.randn((seq, batch, c.hidden_size), device=self.device, dtype=torch.bfloat16)
        padding_mask = torch.zeros((seq, batch), dtype=torch.bool, device=self.device)
        padding_mask[::3] = True

        with torch.no_grad():
            y, _ = c.moe(x, padding_mask=padding_mask)
        y = y.reshape(seq, c.hidden_size)
        pad = padding_mask.reshape(seq)
        self.assertEqual(float(y[pad].abs().max()), 0.0, "padded tokens must yield a zero row")
        self.assertGreater(float(y[~pad].abs().max()), 0.0, "real tokens must be non-zero")
        dist.destroy_process_group()

    # ── route(): index/weight shapes, dtypes, and valid_mask zeroing ──────────
    @skip_if_lt_x_gpu(_WORLD)
    def test_route(self):
        self._init_process()
        group = self._ep_group()
        c = MegaMoETestContainer(group, hidden_size=2048, intermediate_size=1024, num_experts=32, top_k=4)
        T, K, E = 256, c.moe.top_k, c.moe.num_experts
        x = torch.randn((T, c.hidden_size), device=self.device, dtype=torch.bfloat16)
        valid_mask = torch.ones(T, dtype=torch.bool, device=self.device)
        valid_mask[: T // 4] = False
        with torch.no_grad():
            idx, w = c.moe.route(x, valid_mask)
        self.assertEqual(idx.shape, (T, K))
        self.assertEqual(w.shape, (T, K))
        self.assertEqual(idx.dtype, torch.int32)
        self.assertEqual(w.dtype, torch.float32)
        self.assertLess(int(idx.max()), E)
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertEqual(float(w[~valid_mask].abs().max()), 0.0, "padded tokens get zero weight")
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
