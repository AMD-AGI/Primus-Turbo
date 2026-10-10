###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""flash_attn_func(q_norm=...): RMSNorm and interleaved RoPE applied to q inside the attention forward, against the
same norm and RoPE in plain torch followed by flash_attn_func; the backward returns d(normalized q) as q's gradient and
q's rstd as the slot's."""
import pytest
import torch

import primus_turbo.pytorch as pt

D = 128


def _ok(b, s, h):
    try:
        from primus_turbo.pytorch.kernels.attention.attention_aiter_impl import attention_aiter_qnorm_ok

        return bool(attention_aiter_qnorm_ok(b, s, h, D, s // 256))
    except (ImportError, RuntimeError):
        return False


def _inputs(b, s, h, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rn = lambda *sh: torch.randn(*sh, device="cuda", dtype=torch.bfloat16, generator=g)  # noqa: E731
    # sbhd storage, bshd-shaped views (the Megatron caller's layout)
    q, k, v, dout = (rn(s, b, h, D).transpose(0, 1) for _ in range(4))
    w = (1.0 + 0.3 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    ang = torch.rand(s * b, D // 2, device="cuda", generator=g) * 6.3  # one angle per pair, rows s * b + b_i
    return q, k, v, dout, w, torch.cos(ang), torch.sin(ang)


def _norm_rope(q, w, cos, sin, eps):
    """fp32 RMSNorm (weight w) then interleaved RoPE on q [b, s, h, D]; the table row of (b_i, s_i) is s_i * b + b_i."""
    b, s, h, _ = q.shape
    x = q.float()
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    n = x * rstd * w.float()
    rows = (torch.arange(s, device="cuda")[None, :] * b + torch.arange(b, device="cuda")[:, None]).reshape(-1)
    c = cos.float()[rows].view(b, s, 1, D // 2)
    sn = sin.float()[rows].view(b, s, 1, D // 2)
    lo, hi = n[..., 0::2], n[..., 1::2]
    out = torch.stack([lo * c - hi * sn, hi * c + lo * sn], -1).reshape(b, s, h, D)
    return out.to(torch.bfloat16), rstd.squeeze(-1)


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape", [(4, 512, 24), (2, 512, 8)])
def test_q_norm_matches_unfused(shape):
    b, s, h = shape
    if not _ok(b, s, h):
        pytest.skip("no fused q-norm attention kernel for this shape")
    eps = 1e-6
    q, k, v, dout, w, cos, sin = _inputs(b, s, h, seed=sum(shape))
    tab = torch.cat([cos, sin], -1).to(torch.bfloat16).contiguous()

    # unfused reference: norm + RoPE in torch, then the plain attention
    q_ref, rstd_ref = _norm_rope(q, w, cos.to(torch.bfloat16), sin.to(torch.bfloat16), eps)
    q_ref = q_ref.transpose(0, 1).contiguous().transpose(0, 1).detach().requires_grad_(True)  # sbhd storage, as k, v
    kr, vr = k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    o_ref = pt.ops.flash_attn_func(q_ref, kr, vr)
    o_ref.backward(dout)

    # fused: q is the raw projection
    qf, kf, vf = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    slot = torch.zeros(s * b * h, device="cuda", dtype=torch.float32, requires_grad=True)
    o = pt.ops.flash_attn_func(qf, kf, vf, q_norm=(w, tab, w, tab, s // 256, eps, slot))
    o.backward(dout)

    assert _rel(o, o_ref) < 2e-2
    assert _rel(kf.grad, kr.grad) < 2e-2 and _rel(vf.grad, vr.grad) < 2e-2
    assert _rel(qf.grad, q_ref.grad) < 2e-2  # d(normalized q), what q's producer chains through the norm
    # rows s * b * h ordered (s, b, h)
    assert _rel(slot.grad, rstd_ref.permute(1, 0, 2).reshape(-1)) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_q_norm_rejects_sliding_window():
    b, s, h = 2, 512, 8
    if not _ok(b, s, h):
        pytest.skip("no fused q-norm attention kernel for this shape")
    q, k, v, _, w, cos, sin = _inputs(b, s, h, seed=1)
    tab = torch.cat([cos, sin], -1).to(torch.bfloat16).contiguous()
    slot = torch.zeros(s * b * h, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="sliding window"):
        pt.ops.flash_attn_func(q, k, v, window_size=(128, 0), q_norm=(w, tab, w, tab, s // 256, 1e-6, slot))
