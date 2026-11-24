###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Block-wise FP8 (1x128 activation / 128x128 weight, power-of-two scales) dense and grouped GEMM ops.

The op output (forward, dgrad, wgrad) is compared with a float32 torch emulation of the same block
quantization (quantize, dequantize, float32 matmul); the relative L2 error against it must stay below
5e-3. The scale tensors produced by the block quantizers must be exact powers of two.
"""

import pytest
import torch

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
)
from tests.pytorch.ops.gfx1250_helper import rel_l2, requires_gfx1250

FMAX = torch.finfo(float8_e4m3).max
BLK = 128
TOL = 5e-3
CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=BLK)

pytestmark = requires_gfx1250


@pytest.fixture(autouse=True)
def _block_env(monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    monkeypatch.delenv("PRIMUS_TURBO_FP8_BLOCK_KERNEL", raising=False)
    monkeypatch.delenv("PRIMUS_TURBO_FP8_WCACHE", raising=False)


def _pow2_scale(amax):
    s = FMAX / amax.clamp_min(1e-4)
    _, e = torch.frexp(s)
    return torch.ldexp(torch.ones_like(s), e - 1)


def _qdq_1x128(x, axis):
    """Quantize-dequantize x (2D) in 1x128 tiles along ``axis`` (1: along K, 0: along rows)."""
    x = x.float()
    if x.numel() == 0:
        return x.clone()
    if axis == 0:
        return _qdq_1x128(x.t(), 1).t()
    m, k = x.shape
    xp = torch.nn.functional.pad(x, (0, (-k) % BLK)).view(m, -1, BLK)
    s = _pow2_scale(xp.abs().amax(-1, keepdim=True))
    return ((xp * s).clamp(-FMAX, FMAX).to(float8_e4m3).float() / s).view(m, -1)[:, :k]


def _qdq_128x128(w):
    w = w.float()
    n, k = w.shape
    wp = torch.nn.functional.pad(w, (0, (-k) % BLK, 0, (-n) % BLK))
    npad, kpad = wp.shape
    t = wp.view(npad // BLK, BLK, kpad // BLK, BLK)
    s = _pow2_scale(t.abs().amax(dim=(1, 3), keepdim=True))
    return ((t * s).clamp(-FMAX, FMAX).to(float8_e4m3).float() / s).view(npad, kpad)[:n, :k]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("m, n, k", [(512, 1024, 1024), (512, 576, 1024), (256, 384, 224)])
def test_dense_block_fp8(m, n, k, seed):
    torch.manual_seed(seed)
    x = (
        (torch.randn(m, k, device="cuda") * (1 + 3 * torch.rand(m, 1, device="cuda")))
        .bfloat16()
        .requires_grad_()
    )
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16().requires_grad_()
    g = torch.randn(m, n, device="cuda").bfloat16()
    y = pt.ops.gemm_fp8(x, w, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=CFG)
    y.backward(g)

    xd, wd = _qdq_1x128(x.detach(), 1), _qdq_128x128(w.detach())
    assert rel_l2(y, xd @ wd.t()) < TOL
    assert rel_l2(x.grad, _qdq_1x128(g, 1) @ wd) < TOL
    assert rel_l2(w.grad, _qdq_1x128(g, 0).t() @ _qdq_1x128(x.detach(), 0)) < TOL


@pytest.mark.parametrize(
    "lens, n, k",
    [([128] * 4, 256, 256), ([0, 5, 130, 257, 128, 1], 512, 384)],
    ids=["balanced", "uneven_with_empty"],
)
def test_grouped_block_fp8(lens, n, k):
    torch.manual_seed(2)
    g_num, m = len(lens), sum(lens)
    group_lens = torch.tensor(lens, dtype=torch.int64, device="cuda")
    x = torch.randn(m, k, device="cuda").bfloat16().requires_grad_()
    w = (torch.randn(g_num, n, k, device="cuda") * 0.02).bfloat16().requires_grad_()
    g = torch.randn(m, n, device="cuda").bfloat16()
    y = pt.ops.grouped_gemm_fp8(x, w, group_lens, trans_b=True, config=CFG)
    y.backward(g)

    ys, dxs, dws = [], [], []
    off = 0
    for i, ln in enumerate(lens):
        xs, gs = x.detach()[off : off + ln], g[off : off + ln]
        wd = _qdq_128x128(w.detach()[i])
        ys.append(_qdq_1x128(xs, 1) @ wd.t())
        dxs.append(_qdq_1x128(gs, 1) @ wd)
        if ln:
            dws.append(_qdq_1x128(gs, 0).t() @ _qdq_1x128(xs, 0))
        else:
            dws.append(torch.zeros(n, k, device="cuda"))
        off += ln
    assert rel_l2(y, torch.cat(ys)) < TOL
    assert rel_l2(x.grad, torch.cat(dxs)) < TOL
    assert rel_l2(w.grad, torch.stack(dws)) < TOL


def test_block_quant_scales_are_powers_of_two():
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        quant_fp8_blockwise_dual_impl,
        quant_fp8_blockwise_for_weight_impl,
        quant_fp8_blockwise_segment_m_row_col_impl,
    )

    torch.manual_seed(0)
    x = torch.randn(1000, 704, device="cuda").bfloat16() * 3
    scales = list(quant_fp8_blockwise_dual_impl(x, float8_e4m3, BLK))[1::2]
    scales.append(quant_fp8_blockwise_for_weight_impl(x, float8_e4m3, BLK)[1])
    group_lens = torch.tensor([300, 0, 700], dtype=torch.int64, device="cuda")
    group_offs = torch.tensor([0, 300, 300, 1000], dtype=torch.int64, device="cuda")
    r = quant_fp8_blockwise_segment_m_row_col_impl(
        x, float8_e4m3, BLK, group_lens, group_offs, gemm_other_dim=4096
    )
    scales += [r[2], r[3][: (300 + 127) // 128 + (700 + 127) // 128]]
    for s in scales:
        mant, _ = torch.frexp(s[s > 0])
        assert torch.all(mant == 0.5), "scale is not a power of two"
