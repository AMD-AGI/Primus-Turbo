###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""flash_attn_func(return_softmax_d_slot=True): the placeholder's gradient, when the output's consumer supplies one,
is the softmax_d the backward uses instead of computing its own."""
import pytest
import torch

import primus_turbo.pytorch as pt


class _Consumer(torch.autograd.Function):
    """y = o; its backward also returns softmax_d = rowsum(dO * O) as the placeholder's gradient (or nothing)."""

    @staticmethod
    def forward(ctx, o, slot, supply):
        ctx.save_for_backward(o)
        ctx.supply = supply
        return o.clone()

    @staticmethod
    def backward(ctx, g):
        (o,) = ctx.saved_tensors
        if not ctx.supply:
            return g, None, None
        d = (g.float() * o.float()).sum(-1).permute(0, 2, 1).contiguous()  # [b, s, h] -> [b, h, s]
        return g, d, None


def _qkv(b, s, h, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # sbhd storage, bshd-shaped views (the Megatron caller's layout)
    t = [torch.randn(s, b, h, d, device="cuda", dtype=torch.bfloat16, generator=g).transpose(0, 1) for _ in range(4)]
    for x in t[:3]:
        x.requires_grad_(True)
    return t


def _run(q, k, v, dout, slot, supply):
    for x in (q, k, v):
        x.grad = None
    if slot:
        o, ph = pt.ops.flash_attn_func(q, k, v, return_softmax_d_slot=True)
        y = _Consumer.apply(o, ph, supply)
    else:
        y = pt.ops.flash_attn_func(q, k, v)
    y.backward(dout)
    return [x.grad.clone() for x in (q, k, v)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape", [(4, 512, 24, 128), (2, 512, 8, 128)])  # s 512: dq deterministic (1024 is not)
def test_softmax_d_slot(shape, monkeypatch):
    # the a16 backward (no fp32 atomics) is deterministic, so "as before" can be bitwise
    monkeypatch.setenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "0")
    q, k, v, dout = _qkv(*shape, seed=sum(shape))
    base = _run(q, k, v, dout, False, False)
    # no gradient for the placeholder: the backward computes softmax_d itself, bitwise as before
    for a, b in zip(base, _run(q, k, v, dout, True, False)):
        assert torch.equal(a, b)
    # supplied: used in place of the backward's own (a different fp32 summation order: close, not bitwise)
    sup = _run(q, k, v, dout, True, True)
    for a, b in zip(base, sup):
        assert (a.float() - b.float()).abs().max() <= 2e-2 * a.float().abs().max()
    # and it is really used: a wrong softmax_d changes the gradients
    class _Wrong(_Consumer):
        @staticmethod
        def backward(ctx, g):
            gg, d, _ = _Consumer.backward(ctx, g)
            return gg, d + 1.0, None

    for x in (q, k, v):
        x.grad = None
    o, ph = pt.ops.flash_attn_func(q, k, v, return_softmax_d_slot=True)
    _Wrong.apply(o, ph, True).backward(dout)
    assert not torch.equal(q.grad, sup[0])
