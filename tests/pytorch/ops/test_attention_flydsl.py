###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL gfx1250 varlen flash-attention (D_qk=192 / D_v=128 bf16) tests.

Forward + backward are compared against an fp32 PyTorch reference. The FlyDSL
backend is forced via PRIMUS_TURBO_ATTN_BACKEND=flydsl. All cases skip unless
running on gfx1250 (the only arch the WMMA kernel targets).
"""

from typing import List

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.ops import flash_attn_varlen_func
from tests.pytorch.test_utils import compute_snr

D_QK = 192
D_V = 128

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()),
    reason="FlyDSL FMHA kernels target gfx1250 (MI400/MI450) only",
)


def _cu_seqlens(seqlens: List[int], device: str):
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    return cu, max(seqlens), int(cu[-1].item())


def _ref_varlen(q, k, v, cu, scale, causal):
    """fp32 autograd reference (separate qk / v head dims, varlen, GQA)."""
    H = q.shape[1]
    gqa = H // k.shape[1]
    out = torch.zeros(q.shape[0], H, D_V, device=q.device, dtype=torch.float32)
    parts = []
    for b in range(cu.numel() - 1):
        s, e = int(cu[b]), int(cu[b + 1])
        S = e - s
        for h in range(H):
            hk = h // gqa
            sc = (q[s:e, h, :] @ k[s:e, hk, :].transpose(-1, -2)) * scale
            if causal:
                qi = torch.arange(S, device=q.device).view(-1, 1)
                ki = torch.arange(S, device=q.device).view(1, -1)
                sc = sc.masked_fill(ki > qi, float("-inf"))
            parts.append((s, e, h, torch.softmax(sc, dim=-1) @ v[s:e, hk, :]))
    for s, e, h, val in parts:
        out[s:e, h, :] = val
    return out


@pytest.mark.parametrize("seqlens", [[256], [128, 192], [512, 64]])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(2, 2), (4, 2)])  # MHA + GQA
def test_flydsl_flash_attn_varlen(monkeypatch, seqlens, causal, num_head_q, num_head_kv):
    monkeypatch.setenv("PRIMUS_TURBO_ATTN_BACKEND", "flydsl")
    device = "cuda"
    torch.manual_seed(0)

    cu, max_s, total = _cu_seqlens(seqlens, device)
    scale = 1.0 / (D_QK**0.5)

    q = (torch.randn(total, num_head_q, D_QK, device=device) * 0.5).to(torch.bfloat16).requires_grad_(True)
    k = (torch.randn(total, num_head_kv, D_QK, device=device) * 0.5).to(torch.bfloat16).requires_grad_(True)
    v = (torch.randn(total, num_head_kv, D_V, device=device) * 0.5).to(torch.bfloat16).requires_grad_(True)
    grad_out = (torch.randn(total, num_head_q, D_V, device=device) * 0.5).to(torch.bfloat16)

    out = flash_attn_varlen_func(q, k, v, cu, cu, max_s, max_s, softmax_scale=scale, causal=causal)
    out.backward(grad_out)

    # fp32 reference fwd + grads
    qr = q.detach().float().requires_grad_(True)
    kr = k.detach().float().requires_grad_(True)
    vr = v.detach().float().requires_grad_(True)
    out_ref = _ref_varlen(qr, kr, vr, cu, scale, causal)
    out_ref.backward(grad_out.float())

    assert torch.isfinite(out.float()).all()
    assert compute_snr(out_ref, out) > 40.0
    assert compute_snr(qr.grad, q.grad) > 25.0
    assert compute_snr(kr.grad, k.grad) > 25.0
    assert compute_snr(vr.grad, v.grad) > 25.0
