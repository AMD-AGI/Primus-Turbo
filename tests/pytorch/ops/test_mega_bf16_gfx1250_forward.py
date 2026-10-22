###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused Mega MoE forward op on gfx1250, one GPU (EP1), against torch.

``fused_mega_moe`` routes to the gfx1250 kernels there: dispatch + L1 GEMM, SwiGLU with the routing
weight, L2 GEMM + combine + reduce. Forward only -- the gfx1250 backward is not wired yet.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import primus_turbo.pytorch  # noqa: F401
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


def _reference(x, topk_idx, topk_w, w1, w2, activation):
    inter = w2.shape[2]
    y = torch.zeros(x.shape[0], w2.shape[1], device=x.device)
    for e in range(w1.shape[0]):
        t, k = (topk_idx == e).nonzero(as_tuple=True)
        if not t.numel():
            continue
        l1 = x[t].float() @ w1[e].float().T
        gate, up = l1[:, :inter], l1[:, inter:]
        if activation is None:
            act = F.silu(gate) * up
        else:
            a = activation
            gate = gate.clamp(min=a.gate_clamp_lo, max=a.gate_clamp_hi)
            if a.up_clamp is not None:
                up = up.clamp(-a.up_clamp, a.up_clamp)
            act = gate * torch.sigmoid(a.alpha * gate) * (up + a.glu_offset)
        y.index_add_(0, t, (act * topk_w[t, k, None]) @ w2[e].float().T)
    return y


@pytest.mark.parametrize(
    "num_tokens, hidden, inter, num_experts, num_topk, activation",
    [
        (1024, 7168, 2048, 32, 8, None),  # DeepSeek-V3, EP1 slice of the expert set
        (333, 7168, 2048, 32, 8, None),  # ragged token count
        (2048, 2048, 1408, 16, 4, "swigluoai"),  # a different hidden and K, clamped activation
    ],
)
def test_fused_mega_moe_forward_ep1(ep1_group, num_tokens, hidden, inter, num_experts, num_topk, activation):
    from primus_turbo.flydsl.utils.glu_activation import GLUActivation
    from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

    activation = GLUActivation.swigluoai() if activation == "swigluoai" else None
    torch.manual_seed(num_tokens)
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") * 2 / hidden**0.5).to(torch.bfloat16)
    w2 = (torch.randn(num_experts, hidden, inter, device="cuda") * 2 / inter**0.5).to(torch.bfloat16)
    for _ in range(_CALLS):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        topk_w, topk_idx = torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
        with torch.no_grad():
            y = fused_mega_moe(ep1_group, x, topk_idx, topk_w, w1, w2, activation=activation)
        ref = _reference(x, topk_idx, topk_w, w1, w2, activation)
        snr = -20 * torch.log10((y.float() - ref).norm() / ref.norm()).item()
        assert snr > 40, f"SNR {snr:.1f} dB"
