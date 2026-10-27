###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused Mega MoE op on gfx1250, one GPU (EP1), against torch.

``fused_mega_moe`` routes to the gfx1250 kernels there. Forward: dispatch + L1 GEMM, SwiGLU with
the routing weight, L2 GEMM + combine + reduce. Backward: dispatch dy + L2 dgrad, SwiGLU backward,
the two wgrads, L1 dgrad + combine + reduce with the routing-weight gradient. The staged pair
(``fused_mega_moe_stage1`` / ``_stage2``) also carries the expert biases.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.core.utils import is_gfx1250

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")

_CALLS = 3


@pytest.fixture(scope="module")
def ep1_group():
    owned = not dist.is_initialized()
    if owned:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)
        dev = torch.device("cuda", torch.cuda.current_device())
        dist.init_process_group("nccl", rank=0, world_size=1, device_id=dev)
    yield dist.group.WORLD
    if owned:
        dist.destroy_process_group()


def _clamp(v, lo, hi):
    """``v.clamp(lo, hi)`` whose gradient passes on the bounds too, as the kernels' does; torch's own
    stops there in recent versions, and bf16 lands many values exactly on a bound like 10 or 7."""
    inside = torch.ones_like(v, dtype=torch.bool)
    if lo is not None:
        inside &= v >= lo
    if hi is not None:
        inside &= v <= hi
    return torch.where(inside, v, v.detach().clamp(lo, hi))


def _reference(x, topk_idx, topk_w, w1, w2, activation, fc1_bias=None, fc2_bias=None):
    """fp32, differentiable in every input but ``topk_idx``; the biases as Megatron's TEGroupedMLP
    applies them, ``y = sum_k p_k (act_k @ w2^T + fc2_bias)`` with ``fc1_bias`` before the activation."""
    inter = w2.shape[2]
    y = torch.zeros(x.shape[0], w2.shape[1], device=x.device)
    for e in range(w1.shape[0]):
        t, k = (topk_idx == e).nonzero(as_tuple=True)
        if not t.numel():
            continue
        l1 = x[t] @ w1[e].T
        # the kernels keep fc1's output in bf16, and a clamped activation's mask is decided on
        # it: round it the same way (straight through), or rows at the clamp edge disagree
        l1 = l1 + (l1.bfloat16().float() - l1).detach()
        if fc1_bias is not None:
            l1 = l1 + fc1_bias[e]
        gate, up = l1[:, :inter], l1[:, inter:]
        a = GLUActivation() if activation is None else activation  # None is SiLU-SwiGLU clamped at +-10
        gate = _clamp(gate, a.gate_clamp_lo, a.gate_clamp_hi)
        if a.up_clamp is not None:
            up = _clamp(up, -a.up_clamp, a.up_clamp)
        act = gate * torch.sigmoid(a.alpha * gate) * (up + a.glu_offset)
        l2 = act @ w2[e].T
        if fc2_bias is not None:
            l2 = l2 + fc2_bias[e]
        y = y.index_add(0, t, topk_w[t, k, None] * l2)
    return y


def _snr(got, want):
    return -20 * torch.log10((got.float() - want).norm() / want.norm()).item()


@pytest.mark.parametrize(
    "num_tokens, hidden, inter, num_experts, num_topk, activation",
    [
        (1024, 7168, 2048, 32, 8, None),  # DeepSeek-V3, EP1 slice of the expert set
        (333, 7168, 2048, 32, 8, None),  # ragged token count
        (2048, 2048, 1408, 16, 4, "swigluoai"),  # a different hidden and K, clamped activation
    ],
)
def test_fused_mega_moe_forward_ep1(ep1_group, num_tokens, hidden, inter, num_experts, num_topk, activation):
    from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

    activation = GLUActivation.swigluoai() if activation == "swigluoai" else None
    torch.manual_seed(num_tokens)
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") * 2 / hidden**0.5).to(torch.bfloat16)
    w2 = (torch.randn(num_experts, hidden, inter, device="cuda") * 2 / inter**0.5).to(torch.bfloat16)
    for _ in range(_CALLS):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        topk_w, topk_idx = (
            torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
        )
        with torch.no_grad():
            y = fused_mega_moe(ep1_group, x, topk_idx, topk_w, w1, w2, activation=activation)
        ref = _reference(x.float(), topk_idx, topk_w, w1.float(), w2.float(), activation)
        assert (snr := _snr(y, ref)) > 40, f"SNR {snr:.1f} dB"


@pytest.mark.parametrize(
    "num_tokens, hidden, inter, num_experts, num_topk, activation",
    [
        (1024, 7168, 2048, 32, 8, None),
        (333, 2048, 1408, 16, 4, None),  # ragged tokens, a different hidden and K
        (512, 2048, 1408, 16, 4, "swigluoai"),
    ],
)
@pytest.mark.parametrize("enable_torch_compile", [False, True])
def test_fused_mega_moe_backward_ep1(
    ep1_group, num_tokens, hidden, inter, num_experts, num_topk, activation, enable_torch_compile
):
    from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

    activation = GLUActivation.swigluoai() if activation == "swigluoai" else None
    torch._dynamo.reset()

    def run(x, topk_idx, topk_w, w1, w2):
        return fused_mega_moe(ep1_group, x, topk_idx, topk_w, w1, w2, activation=activation)

    if enable_torch_compile:
        run = torch.compile(run)
    torch.manual_seed(num_tokens)
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") * 2 / hidden**0.5).to(torch.bfloat16)
    w2 = (torch.randn(num_experts, hidden, inter, device="cuda") * 2 / inter**0.5).to(torch.bfloat16)
    for _ in range(2):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        topk_w, topk_idx = (
            torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
        )
        dy = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        leaves = [t.clone().requires_grad_() for t in (x, topk_w, w1, w2)]
        run(leaves[0], topk_idx, *leaves[1:]).backward(dy)
        refs = [t.detach().float().requires_grad_() for t in (x, topk_w, w1, w2)]
        _reference(refs[0], topk_idx, *refs[1:], activation).backward(dy.float())
        for name, got, want in zip(("dx", "d topk_w", "dW1", "dW2"), leaves, refs):
            assert (snr := _snr(got.grad, want.grad)) > 40, f"{name} SNR {snr:.1f} dB"


@pytest.mark.parametrize(
    "num_tokens, hidden, inter, num_experts, num_topk",
    [
        (1024, 2880, 2880, 32, 4),  # GPT-OSS-20B: K = 2880 takes 64-wide k-tiles, N = 2880 / 5760 ragged
        (333, 2048, 1408, 16, 4),
    ],
)
def test_fused_mega_moe_staged_bias_ep1(ep1_group, num_tokens, hidden, inter, num_experts, num_topk):
    from primus_turbo.pytorch.ops.moe.fused_mega_moe import (
        fused_mega_moe_stage1,
        fused_mega_moe_stage2,
    )

    activation = GLUActivation.swigluoai()
    torch.manual_seed(num_tokens)
    # wide L1 weights put the clamp in play; std-1 biases move its boundary and dominate fc2
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") * 5 / hidden**0.5).to(torch.bfloat16)
    w2 = (torch.randn(num_experts, hidden, inter, device="cuda") * 2 / inter**0.5).to(torch.bfloat16)
    b1 = torch.randn(num_experts, 2 * inter, device="cuda").to(torch.bfloat16)
    b2 = torch.randn(num_experts, hidden, device="cuda").to(torch.bfloat16)
    for _ in range(2):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        topk_w, topk_idx = (
            torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
        )
        dy = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        leaves = [t.clone().requires_grad_() for t in (x, topk_w, w1, w2, b1, b2)]
        lx, ltw, lw1, lw2, lb1, lb2 = leaves
        l1_out, dispatch_weights, handle = fused_mega_moe_stage1(lx, topk_idx, ltw, lw1, ep1_group)
        y = fused_mega_moe_stage2(
            l1_out, dispatch_weights, handle, topk_idx, ltw, lw2, ep1_group, activation, lb1, lb2
        )
        y.backward(dy)
        refs = [t.detach().float().requires_grad_() for t in (x, topk_w, w1, w2, b1, b2)]
        rx, rtw, rw1, rw2, rb1, rb2 = refs
        yr = _reference(rx, topk_idx, rtw, rw1, rw2, activation, rb1, rb2)
        yr.backward(dy.float())
        assert (snr := _snr(y, yr.detach())) > 40, f"y SNR {snr:.1f} dB"
        for name, got, want in zip(
            ("dx", "d topk_w", "dW1", "dW2", "d fc1_bias", "d fc2_bias"), leaves, refs
        ):
            assert (snr := _snr(got.grad, want.grad)) > 40, f"{name} SNR {snr:.1f} dB"
