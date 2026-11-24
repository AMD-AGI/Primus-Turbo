###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Expert-pool glue kernels of the FP8 block MegaMoE (gfx1250): combine, route dW partials, LPT expert order.

Checks against torch references: the fused combine + route-dW kernel equals ``combine_rows`` and the per-route sum
of the partial columns, the LPT expert order equals the sorted-by-size torch order (stable on ties), and the combine
writes every column when H is not a multiple of 1024. The Gluon routed gather quantizer of the dispatch
equals the Triton routed quantizer on the live rows.
"""

import pytest
import torch

from tests.pytorch.ops.gfx1250_helper import bits_equal, requires_gluon

pytestmark = requires_gluon
E, K, ALIGN = 8, 6, 256


def _tables(t, seed=0):
    from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import build_routing_blk

    g = torch.Generator(device="cuda").manual_seed(seed)
    idx = torch.randn(t, E, device="cuda", generator=g).topk(K, dim=1).indices.to(torch.int64)
    tw = torch.rand(t, K, device="cuda", generator=g)
    p = t * K + E * ALIGN
    rowmap, rw, inv, poffs, preal = build_routing_blk(idx, tw, E, p, align=ALIGN)
    return p, inv


@pytest.mark.parametrize("nc", [16, 8])
@pytest.mark.parametrize("t, h", [(64, 1024), (96, 1280), (128, 768)])
def test_combine_route_dw_fused_equals_separate(t, h, nc):
    from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import (
        combine_rows,
        combine_rows_route_dw,
    )

    p, inv = _tables(t)
    g = torch.Generator(device="cuda").manual_seed(1)
    y = torch.randn(p, h, device="cuda", generator=g).to(torch.bfloat16)
    gw = torch.randn(p, nc, device="cuda", generator=g)
    comb = combine_rows(y, inv, t, K)
    dw = gw.sum(1)[inv.long()]  # dw[r] = sum_c gw[inv[r], c]
    comb2, dw2 = combine_rows_route_dw(y, inv, t, K, gw, torch.float32)
    assert bits_equal(comb, comb2)
    torch.testing.assert_close(dw2, dw, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("h", [256, 768, 1280, 1536])
def test_combine_rows_writes_all_columns(h):
    """combine_rows = sum over the K routes of each token in route order, for H not a multiple of 1024 too."""
    from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import combine_rows

    t = 64
    p, inv = _tables(t, seed=2)
    y = torch.randn(p, h, device="cuda").to(torch.bfloat16)
    out = combine_rows(y, inv, t, K)
    ref = y.float()[inv.long()].view(t, K, h).sum(1)
    assert out.shape == (t, h)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("g_num", [4, 32, 33])
def test_expert_order_lpt_fused_equals_reference(g_num):
    from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import expert_order_lpt_fused

    def expert_order_lpt(poffs_i32):
        return torch.argsort(poffs_i32[1:] - poffs_i32[:-1], descending=True, stable=True).to(torch.int32)

    torch.manual_seed(0)
    for _ in range(5):
        lens = torch.randint(0, 4, (g_num,), device="cuda") * ALIGN  # many ties
        poffs = torch.zeros(g_num + 1, dtype=torch.int32, device="cuda")
        poffs[1:] = torch.cumsum(lens, 0).to(torch.int32)
        assert torch.equal(expert_order_lpt(poffs), expert_order_lpt_fused(poffs))


@pytest.mark.parametrize("compact_t", [True, False])
@pytest.mark.parametrize("layout", ["x4", "packed"])
@pytest.mark.parametrize("kw", [{}, {"nprog": 5, "ns": 3}], ids=str)
def test_q2_gather_bits_equal_triton_on_live_rows(compact_t, layout, kw):
    from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import q2_gather
    from primus_turbo.triton.quantization import quantization_fp8blk_moe_kernel as qb

    t, k, e, c = 256, 6, 4, 256
    p = t * k + e * 256
    g = torch.Generator(device="cuda").manual_seed(0)
    idx = (
        torch.randint(0, e, (t, k), device="cuda", generator=g)
        * torch.randint(1, 3, (t, k), device="cuda", generator=g)
    ) % e
    tw = torch.rand(t, k, device="cuda", generator=g)
    rowmap, _, _, _, preal = qb.build_routing_blk(idx.to(torch.int64), tw, e, p, align=256)
    pr = int(preal.item())
    rowmap, preal = rowmap.to(torch.int32).contiguous(), preal.to(torch.int32).contiguous()
    # x is a view of a NaN-filled buffer: an unchecked out-of-bounds gathered row would read NaN
    big = torch.full((t + 64, c), float("nan"), device="cuda", dtype=torch.bfloat16)
    x = big[:t]
    x.copy_(torch.randn(t, c, device="cuda", generator=g).to(torch.bfloat16))
    x[5, :] = 0
    x[7, 3] = 1e-38
    x[9, :] = 448.0
    x[11, 200] = -3.0e38
    x[:, 17] = 0
    ref = qb.quant_dual_blk(x, p, rowmap, preal=preal, compact_t=compact_t, scale_layout=layout)
    new = q2_gather(x, p, rowmap, preal, compact_t=compact_t, scale_layout=layout, **kw)
    packed = layout == "packed"
    live = [
        lambda a: a.view(torch.uint8)[:pr],
        (lambda a: a.view(torch.uint8)[: pr // 128]) if packed else (lambda a: a.view(torch.uint8)[:pr]),
        lambda a: a.view(torch.uint8)[:, :pr],
        (lambda a: a.view(torch.uint8)[:, : pr // 128])
        if compact_t
        else (
            (lambda a: a.view(torch.uint8)[:, : (pr // 128) * 128])
            if packed
            else (lambda a: a.view(torch.uint8)[:, : pr // 32])
        ),
    ]
    for f, a, b in zip(live, ref, new):
        assert torch.equal(f(a), f(b))
