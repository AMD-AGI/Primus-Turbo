###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused dual (row + transposed) 1x128 E8M0 FP8 quantization kernels.

The Gluon dense quantizer (all launch variants), the packed-scale variant and the Triton kernel variants
must produce bit-identical FP8 data and E8M0 scales to the reference Triton kernel, including zero rows /
columns, denormals, inf and NaN.
"""

import pytest
import torch

from tests.pytorch.ops.gfx1250_helper import as_bits, requires_gfx1250, requires_gluon

F8 = torch.float8_e4m3fn
LAYOUTS = [("g90", "g90"), ("packed", "packed"), ("g90", "packed"), ("packed", "g90")]


def _make(m, n, seed, edge):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, n, device="cuda", generator=g) * (1.0 + 3 * seed)).to(torch.bfloat16)
    if edge:
        x[::97, :] = 0
        x[:, 5::131] = 0
        x[3, :] = 1e-38
        x[4, :128] = torch.tensor(9.0e-41, dtype=torch.float32).to(torch.bfloat16)
        x[6, 7] = -0.0
        x[7, :] = 448.0
        x[8, :] = -448.0 * 2**20
        x[9, 129] = 3.0e38
        x[10, :] = 1e-4
        x[11, ::3] = 1.0001e-4
        x[12, 17] = float("inf")
        x[13, 255] = float("-inf")
        x[14, 33] = float("nan")
        x[15, :128] = (torch.arange(128, device="cuda").float() * 448.0 / 127.0).to(torch.bfloat16)
    return x


def _same(ref, new):
    return all((a is None and b is None) or torch.equal(as_bits(a), as_bits(b)) for a, b in zip(ref, new))


@requires_gfx1250
@pytest.mark.parametrize("variant", ["t2w4", "t2w8"])
@pytest.mark.parametrize("rl, tl", LAYOUTS)
@pytest.mark.parametrize("seed, edge", [(0, False), (0, True), (1, True)])
def test_triton_dual_variants_bits_equal_default(monkeypatch, variant, rl, tl, seed, edge):
    from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel as qg

    x = _make(512, 384, seed, edge)
    monkeypatch.setattr(qg, "_MB_QDUAL", {})
    ref = qg.quant_dual_t_e8m0_v2(x, F8, row_layout=rl, t_layout=tl)
    monkeypatch.setattr(qg, "_MB_QDUAL", {"*": variant})
    out = qg.quant_dual_t_e8m0_v2(x, F8, row_layout=rl, t_layout=tl)
    assert _same(ref, out)


DENSE_VARIANTS = [{}, {"ns": 3, "nprog": 5}, {"ns": 2, "nprog": 3}, {"nprog": 5}]


@requires_gluon
@pytest.mark.parametrize("kw", DENSE_VARIANTS, ids=["default", "ns3_np5", "ns2_np3", "np5"])
@pytest.mark.parametrize("rl, tl", LAYOUTS)
@pytest.mark.parametrize(
    "m, n, seed, edge", [(512, 384, 0, False), (512, 384, 0, True), (256, 1024, 1, True)]
)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_q2_dense_bits_equal_triton(monkeypatch, kw, rl, tl, m, n, seed, edge, dtype):
    from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import q2_dense
    from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel as qg

    if dtype != F8 and not edge:
        pytest.skip("e5m2 is only covered with the edge-case data")
    monkeypatch.setattr(qg, "_q2_use", lambda x: False)  # the Triton kernels
    x = _make(m, n, seed, edge)
    ref = qg.quant_dual_t_e8m0_v2(x, dtype, row_layout=rl, t_layout=tl, num_warps=2)
    out = q2_dense(x, dtype, row_layout=rl, t_layout=tl, **kw)
    assert _same(ref, out)


@requires_gluon
@pytest.mark.parametrize("rl, tl", [("g90", "packed"), ("packed", "g90")])
def test_quant_dual_entry_routes_to_q2_with_same_bytes(monkeypatch, rl, tl):
    from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel as qg

    x = _make(1024, 512, 0, False)
    q2_use = qg._q2_use
    monkeypatch.setattr(qg, "_q2_use", lambda x: False)
    ref = qg.quant_dual_t_e8m0_v2(x, F8, row_layout=rl, t_layout=tl, site="b")
    monkeypatch.setattr(qg, "_q2_use", q2_use)
    out = qg.quant_dual_t_e8m0_v2(x, F8, row_layout=rl, t_layout=tl, site="b")
    assert _same(ref, out)


@requires_gluon
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_dual_pack_bits_equal_separate_quant_and_pack(monkeypatch, n_jobs):
    """quant + weight-scale packing in one launch (Triton and Gluon) equals the separate kernels."""
    from primus_turbo.gluon.gemm.gemm_fp8blk_utils import pack_scales
    from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import q2_dense_pack
    from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel as qg

    m, n = 512, 1024
    x = _make(m, n, 0, False)
    g = torch.Generator(device="cuda").manual_seed(5)
    e_w = torch.randint(
        100, 140, (n // 128, 8), dtype=torch.uint8, device="cuda", generator=g
    )  # [N/128, K/128], K=1024
    wjobs = [(e_w, 128, n, 1024), (e_w.t(), 128, 1024, n)][:n_jobs]
    monkeypatch.setattr(qg, "_q2_use", lambda x: False)
    ref = qg.quant_dual_t_e8m0_v2(x, F8, row_layout="packed", t_layout="packed")
    ref_packed = pack_scales(wjobs)
    for fn in (
        lambda: qg.quant_dual_pack(x, F8, "packed", "packed", wjobs, site="f"),
        lambda: q2_dense_pack(x, F8, "packed", "packed", wjobs),
    ):
        out, packed = fn()
        assert _same(ref, out)
        assert all(torch.equal(a, b) for a, b in zip(ref_packed, packed))


# ----------------------------------------------------------------------------- one program per tile (NS = 1)


@requires_gluon
def test_q2_dense_plan_one_program_per_tile_for_small_grids():
    from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import _dense_plan, _ncu

    dev = torch.device("cuda")
    ncu = _ncu(dev)
    assert _dense_plan(32 * 32, dev, None, 3, None) == (1, 32 * 32)  # 4096 x 4096
    assert _dense_plan(6 * ncu, dev, None, 3, None) == (1, 6 * ncu)
    assert _dense_plan(6 * ncu + 1, dev, None, 3, None) == (2, min(3 * ncu, 6 * ncu + 1))
    assert _dense_plan(64, dev, 2, 3, None) == (2, 64)  # an explicit NS keeps the ring
    assert _dense_plan(64, dev, None, 3, 5) == (2, 5)  # an explicit program count keeps the ring


@requires_gluon
@pytest.mark.parametrize("rl, tl", LAYOUTS[:3])
@pytest.mark.parametrize("m, n", [(4096, 1024), (4096, 2048), (4096, 4096), (4096, 512)])
def test_q2_dense_one_program_per_tile_bits_equal_ring_and_triton(monkeypatch, rl, tl, m, n):
    """DeepSeek-V4 activation shapes: the default launch (one program per tile, no ring) writes the bytes of the
    2-slot ring launch and of the Triton kernel."""
    from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import _ncu, q2_dense
    from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel as qg

    x = _make(m, n, 2, True)
    monkeypatch.setattr(qg, "_q2_use", lambda x: False)
    ref = qg.quant_dual_t_e8m0_v2(x, F8, row_layout=rl, t_layout=tl, num_warps=2)
    one = q2_dense(x, F8, row_layout=rl, t_layout=tl)
    ring = q2_dense(x, F8, row_layout=rl, t_layout=tl, ns=2, nprog=3 * _ncu(x.device))
    assert _same(ref, one)
    assert _same(ref, ring)
