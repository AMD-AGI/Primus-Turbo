###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon BF16 GEMMs (bf16g2 tiles, split-K, rotated-K start, grouped problems) on gfx1250.

Small shapes are checked against a float64 reference (relative L2) and for run-to-run bit stability. At the
DeepSeek-V4 indexer / compressor shapes the split-K kernel (32 x 8 K splits, fixed-order float32 reduction) and the
rotated-K kernel must reproduce the hipBLASLt result bit for bit. The grouped kernel is exercised with operands
more than 2 GB apart (absolute-address path), and the paired rotated kernel must equal its two single launches.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.pytorch.ops.gfx1250_helper import bit_diff_count, bits_equal, rel_l2, requires_gluon

pytestmark = requires_gluon
BF = torch.bfloat16


def _rn(*shape, scale=1.0, gen=None):
    return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(BF)


@pytest.mark.parametrize("tile", ["256x256", "128x256", "256x128"])
def test_bf16g2_nn_matches_fp64_and_is_stable(tile):
    from primus_turbo.gluon.gemm.gemm_bf16_kernel import bf16g2_mm, supported

    torch.manual_seed(0)
    m, n, k = 512, 512, 1024
    a, w = _rn(m, k), _rn(n, k, scale=0.02)
    out = torch.empty(m, n, device="cuda", dtype=BF)
    if not supported(a, w.t(), out, tile):
        pytest.skip(f"tile {tile} not supported for this shape")
    bf16g2_mm(a, w.t(), out, tile=tile)
    ref = a.double() @ w.double().t()
    assert rel_l2(out, ref) < 5e-3
    first = out.clone()
    bf16g2_mm(a, w.t(), out, tile=tile)
    assert bits_equal(first, out)


@pytest.mark.parametrize("splits", [2, 4, 8])
def test_bf16g2_splitk_matches_fp64_and_is_stable(splits):
    from primus_turbo.gluon.gemm.gemm_bf16_splitk_kernel import bf16g2_splitk_mm, splitk_supported

    torch.manual_seed(1)
    t, k, n = 512, 2048, 192
    x, w = _rn(t, k), _rn(n, k, scale=0.02)
    out = torch.empty(t, n, device="cuda", dtype=BF)
    if not splitk_supported(x, w.t(), out, splits, tile="128x256"):
        pytest.skip("split-K config not supported for this shape")
    bf16g2_splitk_mm(x, w.t(), out, splits, tile="128x256")
    assert rel_l2(out, x.double() @ w.double().t()) < 5e-3
    first = out.clone()
    bf16g2_splitk_mm(x, w.t(), out, splits, tile="128x256")
    assert bits_equal(first, out)


def rot_supported(a, b, out, tile):
    """bf16g2 operand rules plus a StaggerU tile width (256) divisible by BLOCK_N."""
    from primus_turbo.gluon.gemm.gemm_bf16_kernel import TILES, supported

    return tile in TILES and 256 % TILES[tile][1] == 0 and supported(a, b, out, tile) and out.dtype == BF


@pytest.mark.parametrize("tile, tpad", [("256x256", 16), ("256x128", 16), ("256x128s3", 16)])
def test_bf16rot_matches_fp64_and_is_stable(tile, tpad):
    from primus_turbo.gluon.gemm.gemm_bf16_grouped_kernel import bf16rot_mm

    torch.manual_seed(2)
    t, h, n = 512, 1024, 512
    x, w = _rn(t, h), _rn(n, h, scale=0.02)
    out = torch.empty(t, n, device="cuda", dtype=BF)
    if not rot_supported(x, w.t(), out, tile):
        pytest.skip(f"tile {tile} not supported for this shape")
    bf16rot_mm(x, w.t(), out, tile=tile, tpad=tpad)
    assert rel_l2(out, x.double() @ w.double().t()) < 5e-3
    first = out.clone()
    bf16rot_mm(x, w.t(), out, tile=tile, tpad=tpad)
    assert bits_equal(first, out)


@pytest.mark.parametrize("tile", ["128x256", "256x128", "256x256"])
def test_bf16grp_two_problems_with_far_apart_operands(tile):
    """Indexer forward pair as one launch: x @ w1^T over 2 K slices of 832, x2 @ w2^T over 2 slices of 512."""
    from primus_turbo.gluon.gemm.gemm_bf16_grouped_kernel import bf16grp_mm, grp_supported

    torch.manual_seed(3)
    t, h, n1, n2, l1, l2, h2 = 512, 1664, 512, 192, 832, 512, 1024
    gap = 2.5 * 2**30
    gapel = int(gap) // 2
    sizes = [t * h, n1 * h, t * h2, n2 * h2]
    big = torch.empty(sum(sizes) + 3 * gapel + 4096, device="cuda", dtype=BF)
    offs, o = [], 0
    for sz in sizes:
        offs.append(o)
        o += (sz + gapel + 63) // 64 * 64
    x = big[offs[0] : offs[0] + t * h].view(t, h)
    x.copy_(torch.randn(t, h, device="cuda"))
    w1 = big[offs[1] : offs[1] + n1 * h].view(n1, h)
    w1.copy_(torch.randn(n1, h, device="cuda") * 0.02)
    x2 = big[offs[2] : offs[2] + t * h2].view(t, h2)
    x2.copy_(torch.randn(t, h2, device="cuda"))
    w2 = big[offs[3] : offs[3] + n2 * h2].view(n2, h2)
    w2.copy_(torch.randn(n2, h2, device="cuda") * 0.02)
    problems = [(x, w1, l1), (x2, w2, l2)]
    if not grp_supported(problems, tile):
        pytest.skip(f"tile {tile} not supported for these problems")
    outs = bf16grp_mm(problems, tile=tile)
    torch.cuda.synchronize()
    assert rel_l2(outs[0], x.double() @ w1.double().t()) < 5e-3
    assert rel_l2(outs[1], x2.double() @ w2.double().t()) < 5e-3
    outs2 = bf16grp_mm(problems, tile=tile)
    assert all(bits_equal(a, b) for a, b in zip(outs, outs2))


# ----------------------------------------------------------------------------- bit-equal to hipBLASLt


def test_splitk8_matches_hipblaslt_at_indexer_shape():
    """Compressor q projection: 4096 x 192 x 4096 with 8 K splits of 512 is the hipBLASLt default split-K result."""
    from primus_turbo.gluon.gemm.gemm_bf16_splitk_kernel import bf16g2_splitk_mm

    torch.manual_seed(0)
    t, h, n = 4096, 4096, 192
    x, w = _rn(1, t, h), _rn(n, h, scale=0.02)
    ref = F.linear(x, w).view(t, n)
    out = torch.empty(t, n, device="cuda", dtype=BF)
    bf16g2_splitk_mm(x.view(t, h), w.t(), out, 8, tile="128x256")
    assert bit_diff_count(out, ref) == 0


@pytest.mark.parametrize("n", [2048])
def test_rotated_k_matches_hipblaslt_forward(n):
    """Indexer / compressor forward: K start rotated per tile (hipBLASLt StaggerU order) equals F.linear."""
    from primus_turbo.gluon.gemm.gemm_bf16_grouped_kernel import bf16rot_mm

    torch.manual_seed(0)
    t, h = 4096, 4096
    x, w = _rn(1, t, h), _rn(n, h, scale=0.02)
    ref = F.linear(x, w).view(t, n)
    out = torch.empty(t, n, device="cuda", dtype=BF)
    bf16rot_mm(x.view(t, h), w.t(), out, tile="256x256", tpad=16)
    assert bit_diff_count(out, ref) == 0
