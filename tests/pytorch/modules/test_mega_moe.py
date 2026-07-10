###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the ``MegaMoE`` expert-parallel MoE module."""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
    skip_if_rocm_arch_multiprocess,
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
# backward grads: bf16-fused vs fp32 ref cos should be ~1; a correct path clears this
_COS_THRESHOLD = 0.95


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
        self.num_experts = num_experts
        self.top_k = top_k
        device = torch.device("cuda", torch.cuda.current_device())
        self.moe = MegaMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            ep_group=ep_group,
            shared_expert_intermediate_size=intermediate_size if shared_expert else None,
            shared_expert_gate=shared_expert,
            device=device,
            dtype=test_dtype,
        )
        # MegaMoE is routing-free; the test owns a local gate to produce the routing
        # decision (sigmoid score + top-k + weight-normalize), same as a native router.
        self.gate_weight = torch.randn((num_experts, hidden_size), device=device, dtype=torch.float32)

    def route(self, x, valid_mask=None):
        """Local top-k routing -> (topk_idx [T,K] i32, topk_weights [T,K] f32)."""
        flat = x.reshape(-1, self.hidden_size)
        scores = torch.sigmoid(F.linear(flat.float(), self.gate_weight))  # [T, E]
        topk_w, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        if valid_mask is not None:
            topk_w = topk_w * valid_mask.to(topk_w.dtype).reshape(-1).unsqueeze(-1)
        return topk_idx.to(torch.int32), topk_w.to(torch.float32)

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
    @skip_if_rocm_arch_multiprocess(("gfx942", "gfx90a"))
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
            topk_idx, topk_w = c.route(x)  # local routing decision reused by the reference
            y = c.moe(x, topk_idx, topk_w)
            ref = c.reference(x, topk_idx, topk_w)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        self._assert_snr(y, ref, tag=f"forward top_k={top_k} shared={shared_expert}")
        dist.destroy_process_group()

    def _cos(self, a, b):
        """Cosine similarity of two tensors (flattened, fp32); min over the EP group."""
        a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
        cos = torch.tensor([float(F.cosine_similarity(a, b, dim=0))], device=self.device)
        dist.all_reduce(cos, op=dist.ReduceOp.MIN)  # weakest rank governs
        return float(cos.item())

    def _diff_global_weights(self, c, group):
        """Gathered global w1/w2 where ONLY this rank's shard is a differentiable leaf."""
        w1_local = c.moe.w1.detach().clone().requires_grad_(True)
        w2_local = c.moe.w2.detach().clone().requires_grad_(True)
        g1 = [torch.empty_like(c.moe.w1) for _ in range(self.world_size)]
        g2 = [torch.empty_like(c.moe.w2) for _ in range(self.world_size)]
        dist.all_gather(g1, c.moe.w1.detach().contiguous(), group=group)
        dist.all_gather(g2, c.moe.w2.detach().contiguous(), group=group)
        r = group.rank()
        w1g = torch.cat([w1_local if i == r else g1[i] for i in range(self.world_size)], dim=0)
        w2g = torch.cat([w2_local if i == r else g2[i] for i in range(self.world_size)], dim=0)
        return w1_local, w2_local, w1g, w2g

    # ── forward + backward: output contract + every param receives a grad ─────
    @skip_if_rocm_arch_multiprocess(("gfx942", "gfx90a"))
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
        with torch.no_grad():
            topk_idx, topk_w = c.route(x)
        y = c.moe(x, topk_idx, topk_w)
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

    # ── backward VALUE gradcheck: dx / dW1 / dW2 / d_topk_w vs dense reference ──
    @skip_if_rocm_arch_multiprocess(("gfx942", "gfx90a"))
    @skip_if_lt_x_gpu(_WORLD)
    def test_backward_gradcheck(self):
        self._init_process()
        group = self._ep_group()
        c = MegaMoETestContainer(group, hidden_size=2048, intermediate_size=1024, num_experts=32, top_k=4)
        # routed-expert path only (expert_compute == mega_moe_fused); fixed routing so
        # dx/dW isolate the dispatch<->combine duality and match the dense reference.
        x = torch.randn((512, c.hidden_size), device=self.device, dtype=torch.bfloat16, requires_grad=True)
        with torch.no_grad():
            topk_idx, topk_w = c.route(x)
        tw = topk_w.detach().clone().requires_grad_(True)
        # shared upstream gradient (same values for fused + reference)
        g = torch.randn((512, c.hidden_size), device=self.device, dtype=torch.bfloat16)

        # fused op grads
        y = c.moe.expert_compute(x, topk_idx, tw)
        dx_m, dW1_m, dW2_m, dtw_m = torch.autograd.grad(
            y, [x, c.moe.w1, c.moe.w2, tw], grad_outputs=g, allow_unused=True
        )

        # dense fp32 reference grads (local shard differentiable inside the global weight)
        xr = x.detach().clone().requires_grad_(True)
        twr = topk_w.detach().clone().requires_grad_(True)
        w1_local, w2_local, w1g, w2g = self._diff_global_weights(c, group)
        ref = _dense_moe_reference(xr, topk_idx, twr, w1g, w2g)
        dx_r, dW1_r, dW2_r, dtw_r = torch.autograd.grad(
            ref, [xr, w1_local, w2_local, twr], grad_outputs=g.float(), allow_unused=True
        )

        cos = {
            "dx": self._cos(dx_m, dx_r),
            "dW1": self._cos(dW1_m, dW1_r),
            "dW2": self._cos(dW2_m, dW2_r),
            "d_topk_w": self._cos(dtw_m, dtw_r),
        }
        if self.rank == 0:
            print("[gradcheck] min cos vs dense ref: " + ", ".join(f"{k}={v:.4f}" for k, v in cos.items()))
        for k, v in cos.items():
            self.assertGreaterEqual(v, _COS_THRESHOLD, f"[gradcheck] {k} cos {v:.4f} < {_COS_THRESHOLD}")
        dist.destroy_process_group()

    # ── padding_mask: padded tokens contribute a zero MoE row ─────────────────
    @skip_if_rocm_arch_multiprocess(("gfx942", "gfx90a"))
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
            topk_idx, topk_w = c.route(x, ~padding_mask)  # zero padded tokens' weights
            y = c.moe(x, topk_idx, topk_w)
        y = y.reshape(seq, c.hidden_size)
        pad = padding_mask.reshape(seq)
        self.assertEqual(float(y[pad].abs().max()), 0.0, "padded tokens must yield a zero row")
        self.assertGreater(float(y[~pad].abs().max()), 0.0, "real tokens must be non-zero")
        dist.destroy_process_group()

    # ── route(): index/weight shapes, dtypes, and valid_mask zeroing ──────────
    @skip_if_rocm_arch_multiprocess(("gfx942", "gfx90a"))
    @skip_if_lt_x_gpu(_WORLD)
    def test_route(self):
        self._init_process()
        group = self._ep_group()
        c = MegaMoETestContainer(group, hidden_size=2048, intermediate_size=1024, num_experts=32, top_k=4)
        T, K, E = 256, c.top_k, c.num_experts
        x = torch.randn((T, c.hidden_size), device=self.device, dtype=torch.bfloat16)
        valid_mask = torch.ones(T, dtype=torch.bool, device=self.device)
        valid_mask[: T // 4] = False
        with torch.no_grad():
            idx, w = c.route(x, valid_mask)
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
