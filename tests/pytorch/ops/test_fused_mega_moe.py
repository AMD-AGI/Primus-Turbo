###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Accuracy tests for the ``fused_mega_moe`` op against the turbo DeepEP MoE."""

from __future__ import annotations

import math
import unittest

import pytest  # noqa: E402
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.utils import is_gfx950, is_gfx1250  # noqa: E402

# mega_moe_fused (the flydsl mega path) is not supported on gfx1250. Skip the whole
# module before importing the flydsl-backed modules below
if is_gfx1250():
    pytest.skip("mega_moe_fused is not supported on gfx1250", allow_module_level=True)

from primus_turbo.flydsl.mega.bf16.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.pytorch.ops import grouped_gemm as _turbo_gg  # noqa: E402
from primus_turbo.pytorch.ops.moe.fused_mega_moe import (  # noqa: E402
    GLUActivation,
    fused_mega_moe,
    fused_mega_moe_stage1,
    fused_mega_moe_stage2,
)
from tests.pytorch.test_utils import compute_snr  # noqa: E402

# bf16 fused vs bf16 turbo ref; comm + split-role reduce add noise -> use MegaMoE-family SNR floor.
_SNR_THRESHOLD_DB = 40.0
# Use cosine to catch directional gradient errors that SNR alone can miss.
_COSINE_THRESHOLD = 0.99

# Match the small tensor-level grad norm seen under real training loss scaling.
_GRAD_OUT_NORM = 1e-3


# Fused kernel only supports gfx950; skip everywhere else.
skip_unless_gfx950 = unittest.skipUnless(
    torch.cuda.is_available() and is_gfx950(), "fused_mega_moe only supports gfx950"
)


def _weighted_glu(fc1_out, weights, activation=None):
    """The gated activation with a per-token routing weight.

    ``None`` is SiLU-SwiGLU (matches Megatron weighted_bias_swiglu). A ``GLUActivation`` is evaluated
    in fp32 with ``torch.clamp``, whose autograd -- gradient passes where the input is inside the
    range -- is the reference backward, as in Megatron's ``weighted_bias_quick_geglu_impl``."""
    if activation is None:
        gate, up = fc1_out.chunk(2, dim=-1)
        return F.silu(gate) * up * weights
    a = activation
    gate, up = fc1_out.float().chunk(2, dim=-1)
    if a.gate_clamp_lo is not None or a.gate_clamp_hi is not None:
        gate = gate.clamp(min=a.gate_clamp_lo, max=a.gate_clamp_hi)
    if a.up_clamp is not None:
        up = up.clamp(-a.up_clamp, a.up_clamp)
    return gate * torch.sigmoid(a.alpha * gate) * (up + a.glu_offset) * weights


def generate_inputs(
    rank, world, *, num_tokens, hidden, inter, num_experts, num_topk, device="cuda", seed=1234, l1_gain=1.0
):
    """One rank's local MoE inputs: x, this rank's L1/L2 expert shard, random top-k routing.

    The generator is seeded here and does NOT read the ambient RNG, so two calls with the same
    ``seed`` return identical tensors however the caller seeded torch in between. Callers that want
    genuinely different data -- a second micro-batch, a second layer -- must pass a different
    ``seed``; ``torch.manual_seed`` in front of the call does nothing.

    ``l1_gain`` scales the L1 weights; the default leaves the L1 output at std ~2, and a clamped
    activation needs it wider to land beyond its limit often enough to matter."""
    epr = num_experts // world
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn((num_tokens, hidden), generator=g, device=device, dtype=torch.float32).bfloat16()
    l1_weight = torch.randn((epr, 2 * inter, hidden), generator=g, device=device, dtype=torch.bfloat16)
    l1_weight *= 2.0 / math.sqrt(hidden)
    if l1_gain != 1.0:
        l1_weight *= l1_gain
    l2_weight = torch.randn((epr, hidden, inter), generator=g, device=device, dtype=torch.bfloat16)
    l2_weight *= 2.0 / math.sqrt(inter)

    logits = torch.randn(num_tokens, num_experts, generator=g, device=device, dtype=torch.float32)
    topk_weight, topk_idx = torch.topk(logits.softmax(-1), num_topk, dim=-1)
    return x, l1_weight, l2_weight, topk_idx.to(torch.int64), topk_weight.to(torch.float32)


def baseline_reference(
    group,
    x,
    topk_idx,
    topk_weight,
    l1_weight,
    l2_weight,
    *,
    num_experts,
    num_topk,
    activation=None,
    fc1_bias=None,
    fc2_bias=None,
):
    """Turbo DeepEP MoE forward; differentiable in x/l1/l2/topk_weight -> serves as the backward ref.

    The optional expert biases follow Megatron's TEGroupedMLP: ``fc1_bias`` [E_local, 2I] on the fc1
    output before the activation, ``fc2_bias`` [E_local, H] on the fc2 output scaled by the routing
    weight. Pass them as f32 leaves: a bf16 one would sum its per-row gradients in bf16."""
    # scatter (differentiable) routes topk_weight grad straight back through the reference
    gate_logits = torch.zeros(x.shape[0], num_experts, device=x.device, dtype=torch.float32).scatter(
        1, topk_idx, topk_weight
    )
    dispatcher = turbo.modules.DeepEPTokenDispatcher(
        num_experts=num_experts,
        router_topk=num_topk,
        ep_group=group,
        permute_fusion=True,
        deepep_num_use_cu=80,
    )
    permuted_hidden, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
        x, gate_logits, indices=topk_idx
    )
    group_lens = tokens_per_expert.to(device=x.device, dtype=torch.int64)
    fc1_out = _turbo_gg(permuted_hidden, l1_weight, group_lens, trans_b=True)
    if fc1_bias is not None or fc2_bias is not None:
        row_expert = torch.repeat_interleave(torch.arange(l1_weight.shape[0], device=x.device), group_lens)
    if fc1_bias is not None:
        fc1_out = fc1_out.float() + fc1_bias[row_expert]
    inter = _weighted_glu(fc1_out, permuted_probs.unsqueeze(-1), activation).to(x.dtype)
    fc2_out = _turbo_gg(inter, l2_weight, group_lens, trans_b=True)
    if fc2_bias is not None:
        fc2_out = (fc2_out.float() + fc2_bias[row_expert] * permuted_probs.unsqueeze(-1)).to(x.dtype)
    return dispatcher.token_combine(fc2_out)


def _test_forward_backward_impl(
    group,
    symm,
    x,
    l1_weight,
    l2_weight,
    topk_idx,
    topk_weight,
    *,
    num_experts,
    num_topk,
    enable_torch_compile=False,
    activation=None,
    staged=False,
    fc1_bias=None,
    fc2_bias=None,
):
    """tc-free fwd+bwd of fused_mega_moe (or, with ``staged``, the stage1+stage2 pair training
    drives) vs turbo; returns (tag, actual, ref) triples, frees symm.

    The expert biases (bf16 [E_local, 2I] / [E_local, H]) ride the staged op, which owns them."""
    assert staged or (fc1_bias is None and fc2_bias is None), "the expert biases are a stage2 input"
    bias_tags = [tag for tag, b in (("dfc1_bias", fc1_bias), ("dfc2_bias", fc2_bias)) if b is not None]
    try:
        # Normalize grad_out by tensor norm so gradient-independent floors remain visible.
        _gy = torch.randn(x.shape, device=x.device, dtype=torch.float32)
        grad_y = (_gy / (_gy.norm() + 1e-12) * _GRAD_OUT_NORM).bfloat16()

        # fused runner over grad-carrying inputs; topk_idx stays a constant closure
        def _fused(x, topk_weight, l1_weight, l2_weight, b1, b2):
            if staged:
                l1_out, dwib, handle = fused_mega_moe_stage1(x, topk_idx, topk_weight, l1_weight, group)
                return fused_mega_moe_stage2(
                    l1_out, dwib, handle, topk_idx, topk_weight, l2_weight, group, activation, b1, b2
                )
            return fused_mega_moe(group, x, topk_idx, topk_weight, l1_weight, l2_weight, activation)

        runner = _fused
        if enable_torch_compile:
            runner = torch.compile(runner)

        x_m = x.detach().requires_grad_(True)
        l1_m = l1_weight.detach().requires_grad_(True)
        l2_m = l2_weight.detach().requires_grad_(True)
        tw_m = topk_weight.detach().requires_grad_(True)
        b1_m, b2_m = (None if b is None else b.detach().requires_grad_(True) for b in (fc1_bias, fc2_bias))
        bias_m = [b for b in (b1_m, b2_m) if b is not None]

        # device-side epoch flags advance per call, so call #2+ must stay correct
        num_iters = 4
        for _ in range(num_iters):
            y_m = runner(x_m, tw_m, l1_m, l2_m, b1_m, b2_m)
            dx_m, dl1_m, dl2_m, dtw_m, *dbias_m = torch.autograd.grad(
                y_m, [x_m, l1_m, l2_m, tw_m, *bias_m], grad_y
            )

        # turbo reference: topk_weight flows through scatter, so dtw is compared directly
        x_t = x.detach().requires_grad_(True)
        l1_t = l1_weight.detach().requires_grad_(True)
        l2_t = l2_weight.detach().requires_grad_(True)
        tw_t = topk_weight.detach().requires_grad_(True)
        b1_t, b2_t = (
            None if b is None else b.detach().float().requires_grad_(True) for b in (fc1_bias, fc2_bias)
        )
        y_t = baseline_reference(
            group,
            x_t,
            topk_idx,
            tw_t,
            l1_t,
            l2_t,
            num_experts=num_experts,
            num_topk=num_topk,
            activation=activation,
            fc1_bias=b1_t,
            fc2_bias=b2_t,
        )
        bias_t = [b for b in (b1_t, b2_t) if b is not None]
        dx_t, dl1_t, dl2_t, dtw_t, *dbias_t = torch.autograd.grad(
            y_t, [x_t, l1_t, l2_t, tw_t, *bias_t], grad_y
        )

        torch.cuda.synchronize()
        group.barrier()

        # dx asserted: stale-L2 read in combine gate path fixed in ep_intranode (glc|slc).
        return [
            ("forward", y_m, y_t),
            ("dx", dx_m, dx_t),
            ("dl1_weight", dl1_m, dl1_t),
            ("dl2_weight", dl2_m, dl2_t),
            ("dtw", dtw_m, dtw_t),
            *zip(bias_tags, dbias_m, dbias_t),
        ]
    finally:
        # release symm buffer even on failure; PG is class-scoped (kept for next test)
        if symm is not None:
            symm.destroy()


@instantiate_parametrized_tests
class FusedMegaMoETestBase(MultiProcContinuousTest):
    """EP8 accuracy tests vs turbo DeepEP; base spawns workers + inits PG once per class."""

    # PG backend picked by the base class at spawn time.
    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _setup_device(self):
        # PG already up (base class); just bind the CUDA device + seed for this test.
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)

    def _inputs(self, num_tokens, hidden, inter, num_experts, num_topk, l1_gain=1.0):
        return generate_inputs(
            self.rank,
            self.world_size,
            num_tokens=num_tokens,
            hidden=hidden,
            inter=inter,
            num_experts=num_experts,
            num_topk=num_topk,
            device=self.device,
            l1_gain=l1_gain,
        )

    def _symm(self, group, num_tokens, hidden, inter, num_experts, num_topk):
        return get_symm_buffer_for_mega_moe(
            group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=inter,
        )

    def _metrics(self, actual, ref, *, tag):
        """SNR (dB, magnitude+direction) and cosine (direction only) vs ref; weakest rank governs."""
        cos = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), ref.float().flatten(), dim=0, eps=1e-12
        )
        m = torch.tensor([compute_snr(ref, actual), float(cos)], device=self.device)  # both higher=better
        dist.all_reduce(m, op=dist.ReduceOp.MIN)  # weakest rank governs the EP group
        snr, cos = float(m[0]), float(m[1])
        if self.rank == 0:
            print(f"[{tag}] min SNR = {snr:.2f} dB, min cos = {cos:.5f}")
        return snr, cos

    def _assert_snr(self, actual, ref, *, tag):
        # Require BOTH: SNR (catches floors + magnitude) AND cosine (catches direction errors).
        snr, cos = self._metrics(actual, ref, tag=tag)
        self.assertGreaterEqual(snr, _SNR_THRESHOLD_DB, f"[{tag}] SNR {snr:.2f} dB < {_SNR_THRESHOLD_DB}")
        self.assertGreaterEqual(cos, _COSINE_THRESHOLD, f"[{tag}] cosine {cos:.5f} < {_COSINE_THRESHOLD}")

    # ── forward + backward vs turbo DeepEP; optional torch.compile ──
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize(
        "hidden, inter, num_experts, num_topk, num_tokens",
        [
            (7168, 2048, 256, 8, 8192),
        ],
    )
    @parametrize("enable_torch_compile", [False, True])
    def test_forward_backward(
        self,
        hidden,
        inter,
        num_experts,
        num_topk,
        num_tokens,
        enable_torch_compile,
    ):
        self._setup_device()
        group = dist.group.WORLD
        x, l1_weight, l2_weight, topk_idx, topk_weight = self._inputs(
            num_tokens, hidden, inter, num_experts, num_topk
        )
        symm = self._symm(group, num_tokens, hidden, inter, num_experts, num_topk)
        results = _test_forward_backward_impl(
            group,
            symm,
            x,
            l1_weight,
            l2_weight,
            topk_idx,
            topk_weight,
            num_experts=num_experts,
            num_topk=num_topk,
            enable_torch_compile=enable_torch_compile,
        )
        for tag, actual, ref in results:
            self._assert_snr(actual, ref, tag=tag)

    # ── MiniMax-M3's swigluoai: the same comparison with the activation threaded through ──
    # l1_gain 2.5 widens the L1 output to std ~5, so ~8% of gates land above +7, ~8% below -7 and
    # ~16% of ups beyond +-7. SNR cannot see the gate's one-sided clamp (below -7 the gate term is
    # ~1e-5 either way); tests/pytorch/ops/test_mega_moe_activation.py checks that per regime.
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize(
        "hidden, inter, num_experts, num_topk, num_tokens",
        [
            # MiniMax-M3's expert shape, at this class's token count.
            (6144, 3072, 128, 4, 8192),
        ],
    )
    @parametrize(
        "staged, enable_torch_compile",
        [
            (True, False),
            (False, False),
            (False, True),
        ],
    )
    def test_forward_backward_swigluoai(
        self,
        hidden,
        inter,
        num_experts,
        num_topk,
        num_tokens,
        staged,
        enable_torch_compile,
    ):
        self._setup_device()
        group = dist.group.WORLD
        x, l1_weight, l2_weight, topk_idx, topk_weight = self._inputs(
            num_tokens, hidden, inter, num_experts, num_topk, l1_gain=2.5
        )
        symm = self._symm(group, num_tokens, hidden, inter, num_experts, num_topk)
        results = _test_forward_backward_impl(
            group,
            symm,
            x,
            l1_weight,
            l2_weight,
            topk_idx,
            topk_weight,
            num_experts=num_experts,
            num_topk=num_topk,
            enable_torch_compile=enable_torch_compile,
            activation=GLUActivation.swigluoai(),
            staged=staged,
        )
        for tag, actual, ref in results:
            self._assert_snr(actual, ref, tag=tag)

    # ── GPT-OSS: hidden = intermediate = 2880 and biased experts, swigluoai ──
    # 2880 is a 64-column (H, I) / 128-column (2I) tail of the 256-wide GEMM tile and a ragged row
    # for the 1 KiB token copy. The biases are drawn at std 1, where they move the activation's
    # clamp boundary and dominate the fc2 output (the bias-free reference scores ~10 dB).
    @skip_unless_gfx950
    @skip_if_lt_x_gpu(8)
    @parametrize(
        "hidden, inter, num_experts, num_topk, num_tokens",
        [
            (2880, 2880, 32, 4, 8192),  # GPT-OSS-20B
            (2880, 2880, 128, 4, 8192),  # GPT-OSS-120B
        ],
    )
    def test_forward_backward_gptoss(self, hidden, inter, num_experts, num_topk, num_tokens):
        self._setup_device()
        group = dist.group.WORLD
        x, l1_weight, l2_weight, topk_idx, topk_weight = self._inputs(
            num_tokens, hidden, inter, num_experts, num_topk, l1_gain=2.5
        )
        g = torch.Generator(device=self.device).manual_seed(4321 + self.rank)
        epr = num_experts // self.world_size
        fc1_bias = torch.randn((epr, 2 * inter), generator=g, device=self.device).bfloat16()
        fc2_bias = torch.randn((epr, hidden), generator=g, device=self.device).bfloat16()
        symm = self._symm(group, num_tokens, hidden, inter, num_experts, num_topk)
        results = _test_forward_backward_impl(
            group,
            symm,
            x,
            l1_weight,
            l2_weight,
            topk_idx,
            topk_weight,
            num_experts=num_experts,
            num_topk=num_topk,
            activation=GLUActivation.swigluoai(),
            staged=True,
            fc1_bias=fc1_bias,
            fc2_bias=fc2_bias,
        )
        for tag, actual, ref in results:
            if tag.startswith("dfc"):
                # A bias grad sums ~1k rows of a 1e-3-normed grad_out: small enough that
                # compute_snr's 1e-12 noise floor alone would cap it at ~40 dB, so measure it unit-normed.
                scale = ref.float().norm().clamp_min(1e-30)
                actual, ref = actual.float() / scale, ref.float() / scale
            self._assert_snr(actual, ref, tag=tag)


if __name__ == "__main__":
    run_tests()
