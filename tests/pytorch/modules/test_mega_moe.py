###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the ``MegaMoE`` expert-parallel MoE module."""

import os
import unittest

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

from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.modules.moe.mega_moe import MegaMoE
from tests.pytorch.test_utils import compute_snr

# bf16 fused kernel vs fp32 dense reference; worst rank observed ~26 dB, broken < 10 dB
_SNR_THRESHOLD_DB = 20.0
# backward grads: bf16-fused vs fp32 ref cos should be ~1; a correct path clears this
_COS_THRESHOLD = 0.95

# Production DeepSeek-V3 EP8 shapes; mega MoE wgrad is only reliable at this scale (small shapes flaky).
_MOE_CASES = {
    # DeepSeek-V3: H=7168 I=2048 E=256 topk=8, 8192 tokens/rank
    "DeepSeek-V3": dict(
        hidden_size=7168,
        intermediate_size=2048,
        num_experts=256,
        top_k=8,
        num_tokens=8192,
    ),
}

# Mega MoE kernel only supports gfx950; skip everywhere else.
skip_unless_gfx950 = unittest.skipUnless(
    torch.cuda.is_available() and is_gfx950(), "MegaMoE only supports gfx950"
)


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
        # local gate scaled by 1/sqrt(H) so logits stay O(1); unscaled, sigmoid saturates and top-k collapses experts
        self.gate_weight = torch.randn((num_experts, hidden_size), device=device, dtype=torch.float32) * (
            hidden_size**-0.5
        )

    def route_reference(self, x, valid_mask=None):
        """Local top-k routing -> (topk_idx [T,K] i32, topk_weights [T,K] f32)."""
        flat = x.reshape(-1, self.hidden_size)
        scores = torch.sigmoid(F.linear(flat.float(), self.gate_weight))  # [T, E]
        topk_w, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        if valid_mask is not None:
            topk_w = topk_w * valid_mask.to(topk_w.dtype).reshape(-1).unsqueeze(-1)
        return topk_idx.to(torch.int32), topk_w.to(torch.float32)

    def reference(self, x, topk_idx, topk_w, grad_y):
        """EP-correct dense fp32 reference over the global token set; returns this rank's (y, dx, dW1, dW2, d_topk_w)."""

        def gather(t):  # per-rank [T, ...] -> global [world*T, ...]
            parts = [torch.empty_like(t) for _ in range(self.world)]
            dist.all_gather(parts, t.contiguous(), group=self.ep_group)
            return torch.cat(parts)

        def gather_weight(param, local):  # all shards; this rank's stays differentiable
            shards = [torch.empty_like(param) for _ in range(self.world)]
            dist.all_gather(shards, param.detach().contiguous(), group=self.ep_group)
            shards[self.rank] = local
            return torch.cat(shards)

        w1 = self.moe.w1.detach().clone().requires_grad_(True)
        w2 = self.moe.w2.detach().clone().requires_grad_(True)
        xg = gather(x.detach()).requires_grad_(True)
        twg = gather(topk_w.detach()).requires_grad_(True)
        idxg, gyg = gather(topk_idx), gather(grad_y)

        y = _dense_moe_reference(
            xg, idxg, twg, gather_weight(self.moe.w1, w1), gather_weight(self.moe.w2, w2)
        )
        if self.moe.shared_w1 is not None:
            y = y + _shared_expert_reference(
                xg, self.moe.shared_w1, self.moe.shared_w2, self.moe.shared_gate_weight
            )
        dx, dW1, dW2, dtw = torch.autograd.grad(y, [xg, w1, w2, twg], grad_outputs=gyg.float())

        local = slice(self.rank * x.shape[0], (self.rank + 1) * x.shape[0])
        return y[local].detach(), dx[local], dW1, dW2, dtw[local]


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
        return 8

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

    def _snr(self, ref, actual):
        """SNR (dB) of actual vs ref (ref is the signal); min over the EP group."""
        snr = torch.tensor([compute_snr(ref, actual.detach())], device=self.device)
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)  # weakest rank governs
        return float(snr.item())

    def _cos(self, a, b):
        """Cosine of two flattened tensors, reduced in float64 (dW has ~1e9 elems); min over the EP group."""
        a, b = a.detach().reshape(-1).double(), b.detach().reshape(-1).double()
        cos = torch.tensor([float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-20))], device=self.device)
        dist.all_reduce(cos, op=dist.ReduceOp.MIN)  # weakest rank governs
        return float(cos.item())

    def _assert_accuracy(self, tag, fused, ref):
        """One accuracy gate: y -> SNR (dB), each grad -> cosine; passes only if ALL clear."""
        results = {"y": (self._snr(ref["y"], fused["y"]), _SNR_THRESHOLD_DB, "dB")}
        for k in fused:
            if k != "y":
                results[k] = (self._cos(fused[k], ref[k]), _COS_THRESHOLD, "")
        if self.rank == 0:
            print(f"[{tag}] " + "  ".join(f"{k}={v:.3f}{u}(>={f})" for k, (v, f, u) in results.items()))
        for k, (v, f, u) in results.items():
            self.assertGreaterEqual(v, f, f"[{tag}] {k}={v:.4f}{u} < floor {f}")

    # ── accuracy: forward output (SNR) + all grads (cos) vs the dense fp32 reference ──
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize("case", list(_MOE_CASES))
    @parametrize("shared_expert", [False, True])
    def test_forward_backward(self, case, shared_expert):
        self._init_process()
        group = self._ep_group()
        cfg = _MOE_CASES[case]
        c = MegaMoETestContainer(
            group,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
            shared_expert=shared_expert,
        )
        T = cfg["num_tokens"]
        x = torch.randn((T, c.hidden_size), device=self.device, dtype=torch.bfloat16, requires_grad=True)
        with torch.no_grad():
            topk_idx, topk_w = c.route_reference(x)
        tw = topk_w.detach().clone().requires_grad_(True)  # topk weight is a grad leaf
        g = torch.randn((T, c.hidden_size), device=self.device, dtype=torch.bfloat16)  # upstream grad

        # fused forward (routed + optional shared expert) + grads
        y = c.moe(x, topk_idx, tw)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, torch.bfloat16)
        shared = [(n, p) for n, p in c.moe.named_parameters() if p.requires_grad and n.startswith("shared")]
        dx, dW1, dW2, dtw, *shared_grads = torch.autograd.grad(
            y, [x, c.moe.w1, c.moe.w2, tw, *[p for _, p in shared]], grad_outputs=g, allow_unused=True
        )
        # Megatron parity: every trainable parameter must receive a gradient
        for (name, _), gp in zip(shared, shared_grads):
            self.assertIsNotNone(gp, f"gradient for {name} should exist")

        # dense fp32 reference: one graph yields the forward output and all grads
        y_r, dx_r, dW1_r, dW2_r, dtw_r = c.reference(x, topk_idx, tw, g)
        self._assert_accuracy(
            f"case={case} shared={shared_expert}",
            {"y": y, "dx": dx, "dW1": dW1, "dW2": dW2, "d_topk_w": dtw},
            {"y": y_r, "dx": dx_r, "dW1": dW1_r, "dW2": dW2_r, "d_topk_w": dtw_r},
        )
        dist.destroy_process_group()

    # ── padding_mask: padded tokens contribute a zero MoE row ─────────────────
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize("case", list(_MOE_CASES))
    def test_padding_mask(self, case):
        self._init_process()
        group = self._ep_group()
        cfg = _MOE_CASES[case]
        c = MegaMoETestContainer(
            group,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
        )
        seq, batch = cfg["num_tokens"], 1
        # Megatron layout: hidden [seq, batch, H], padding_mask [seq, batch], True = PADDING
        x = torch.randn((seq, batch, c.hidden_size), device=self.device, dtype=torch.bfloat16)
        padding_mask = torch.zeros((seq, batch), dtype=torch.bool, device=self.device)
        padding_mask[::3] = True

        with torch.no_grad():
            topk_idx, topk_w = c.route_reference(x, ~padding_mask)  # zero padded tokens' weights
            y = c.moe(x, topk_idx, topk_w)
        y = y.reshape(seq, c.hidden_size)
        pad = padding_mask.reshape(seq)
        self.assertEqual(float(y[pad].abs().max()), 0.0, "padded tokens must yield a zero row")
        self.assertGreater(float(y[~pad].abs().max()), 0.0, "real tokens must be non-zero")
        dist.destroy_process_group()

    # ── internal router + aux loss: forward contract + grad path to the gate ──
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize("case", list(_MOE_CASES))
    def test_internal_router_aux_loss(self, case):
        self._init_process()
        group = self._ep_group()
        cfg = _MOE_CASES[case]
        device = torch.device("cuda", torch.cuda.current_device())
        # self-routing MegaMoE: top_k enables the internal router, aux_loss_coeff enables aux loss
        moe = MegaMoE(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_experts=cfg["num_experts"],
            ep_group=group,
            top_k=cfg["top_k"],
            aux_loss_coeff=1e-2,
            device=device,
            dtype=torch.bfloat16,
        )
        moe.train()
        T = cfg["num_tokens"]
        x = torch.randn((T, moe.hidden_size), device=self.device, dtype=torch.bfloat16, requires_grad=True)

        # route() returns the aux loss as a third value (grad path to the gate logits)
        topk_idx, topk_w, aux_loss = moe.route(x)
        self.assertEqual(topk_idx.shape, (T, moe.top_k))
        self.assertEqual(topk_idx.dtype, torch.int32)
        self.assertEqual(topk_w.dtype, torch.float32)
        self.assertIsNotNone(aux_loss, "aux loss should exist when training with aux_loss_coeff > 0")
        self.assertEqual(aux_loss.dim(), 0, "aux loss must be a scalar")
        self.assertTrue(aux_loss.requires_grad, "aux loss must carry grad to the gate")

        # forward(return_aux_loss=True) returns (y, aux_loss); grad flows to gate_weight
        y, aux_loss = moe(x, return_aux_loss=True)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertIsNotNone(aux_loss)
        (y.float().sum() + aux_loss).backward()
        self.assertIsNotNone(moe.gate_weight.grad, "aux loss must produce a gate_weight gradient")
        self.assertGreater(float(moe.gate_weight.grad.abs().max()), 0.0)

        # inference / grad-disabled path: no aux loss returned
        moe.train(False)  # set eval mode without the flagged .eval() token
        with torch.no_grad():
            y_infer, aux_infer = moe(x, return_aux_loss=True)
        self.assertEqual(y_infer.shape, x.shape)
        self.assertIsNone(aux_infer, "aux loss must be None outside training")
        dist.destroy_process_group()

    # ── route(): index/weight shapes, dtypes, and valid_mask zeroing ──────────
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize("case", list(_MOE_CASES))
    def test_route(self, case):
        self._init_process()
        group = self._ep_group()
        cfg = _MOE_CASES[case]
        c = MegaMoETestContainer(
            group,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
        )
        T, K, E = 256, c.top_k, c.num_experts
        x = torch.randn((T, c.hidden_size), device=self.device, dtype=torch.bfloat16)
        valid_mask = torch.ones(T, dtype=torch.bool, device=self.device)
        valid_mask[: T // 4] = False
        with torch.no_grad():
            idx, w = c.route_reference(x, valid_mask)
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
