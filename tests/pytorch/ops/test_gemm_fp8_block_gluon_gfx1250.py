###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon block-FP8 GEMMs (256x256 and small-tile variants) against the FlyDSL block GEMM and a float32 reference.

Operands are FP8 E4M3 with E8M0 scales: 1x128 for A, 128x128 blocks for B. Checked: scale packing against a
torch reference; the 256x256 kernel and every small-tile kernel bit-identical to the FlyDSL kernel for bf16
output; beta=1 / beta=2 accumulation epilogues; the split-output epilogue; the Linear op with the Gluon
kernel vs the FlyDSL kernel (forward, dgrad, wgrad into main_grad with beta 0 and 1); the concat-N op; and the
fallback to the FlyDSL kernel when Gluon is unavailable.
"""

import pytest
import torch

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.ops import utils as ops_utils
from tests.pytorch.ops.gfx1250_helper import (
    bit_diff_count,
    bits_equal,
    rel_l2,
    requires_flydsl,
    requires_gluon,
)

CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128)
FP8 = torch.float8_e4m3fn


def _operands(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", generator=g).to(FP8)
    b = torch.randn(n, k, device="cuda", generator=g).to(FP8)
    ea = torch.randint(120, 130, (m, k // 128), dtype=torch.uint8, device="cuda", generator=g)
    eb = torch.randint(120, 130, (n // 128, k // 128), dtype=torch.uint8, device="cuda", generator=g)
    return a, b, ea, eb


def _dequant(q, e, rows_per_scale):
    s = (
        torch.exp2(e.float() - 127)
        .repeat_interleave(rows_per_scale, 0)[: q.shape[0]]
        .repeat_interleave(128, 1)
    )
    return q.float() * s


def _packed(ea, eb, m, n, k):
    from primus_turbo.gluon.gemm.gemm_fp8blk_utils import pack_scales

    return pack_scales([(ea, 1, m, k), (eb, 128, n, k)])


def pack_scales_ref(src, rps, R, K):
    """Pure-torch reference of the packed MX block-32 layout (rows_per_scale = rps)."""
    KB = K // 128
    dev = src.device
    c = torch.arange(R // 128, device=dev).view(-1, 1, 1, 1, 1)
    kc = torch.arange(KB, device=dev).view(1, -1, 1, 1, 1)
    j = torch.arange(32, device=dev).view(1, 1, -1, 1, 1)
    i = torch.arange(4, device=dev).view(1, 1, 1, -1, 1)
    row = (c * 128 + i * 32 + j).expand(-1, KB, -1, -1, 4)
    g = src[row // rps, kc.expand_as(row)]
    return g.reshape(R // 128, K * 4).contiguous()


@requires_gluon
def test_pack_scales_matches_reference():
    from primus_turbo.gluon.gemm.gemm_fp8blk_utils import pack_scales

    torch.manual_seed(0)
    e_row = torch.randint(100, 140, (512, 8), dtype=torch.uint8, device="cuda")
    e_blk = torch.randint(100, 140, (8, 8), dtype=torch.uint8, device="cuda")
    e_blk2 = torch.randint(100, 140, (4, 16), dtype=torch.uint8, device="cuda")
    e_rowt = torch.randint(100, 140, (8, 512), dtype=torch.uint8, device="cuda").t()  # strided rows
    jobs = [
        (e_row, 1, 512, 1024),
        (e_blk, 128, 1024, 1024),
        (e_blk2.t(), 128, 2048, 512),
        None,
        (e_rowt, 1, 512, 1024),
        (e_blk, 128, 1024, 1024),
    ]  # five live jobs -> two launches
    outs = pack_scales(jobs)
    for job, out in zip(jobs, outs):
        if job is None:
            assert out is None
        else:
            assert torch.equal(out, pack_scales_ref(job[0].contiguous(), *job[1:]))


@requires_gluon
def test_dsv4up_vs_fp32_reference_and_accumulate():
    from primus_turbo.gluon.gemm.gemm_fp8blk_kernel import dsv4up_gemm

    m, n, k = 256, 512, 1024
    a, b, ea, eb = _operands(m, n, k)
    pa, pb = _packed(ea, eb, m, n, k)
    ref = _dequant(a, ea, 1) @ _dequant(b, eb, 128).t()
    c = dsv4up_gemm(a, b, pa, pb, out_dtype=torch.bfloat16)
    assert rel_l2(c, ref) < 7e-3

    cf = torch.empty(m, n, device="cuda")
    dsv4up_gemm(a, b, pa, pb, out=cf)
    assert rel_l2(cf, ref) < 2e-4
    c0 = torch.randn(m, n, device="cuda")
    c32 = c0.clone()
    dsv4up_gemm(a, b, pa, pb, out=c32, accumulate=True)
    assert rel_l2(c32, c0 + cf) < 1e-6
    c16 = c0.bfloat16()
    c16_0 = c16.clone()
    dsv4up_gemm(a, b, pa, pb, out=c16, accumulate=True)
    assert rel_l2(c16, c16_0.float() + ref) < 7e-3


@requires_gluon
@requires_flydsl
@pytest.mark.parametrize("m, n, k", [(512, 512, 1024), (256, 768, 1536)])
def test_dsv4up_bits_equal_flydsl(m, n, k):
    from primus_turbo.gluon.gemm.gemm_fp8blk_kernel import dsv4up_gemm
    from primus_turbo.pytorch.kernels.gemm import gemm_fp8_flydsl_gfx1250_impl

    a, b, ea, eb = _operands(m, n, k, seed=1)
    pa, pb = _packed(ea, eb, m, n, k)
    ref = gemm_fp8_flydsl_gfx1250_impl.gemm_nt(a, b, ea, eb)
    assert bits_equal(dsv4up_gemm(a, b, pa, pb), ref)


@requires_gluon
@requires_flydsl
@pytest.mark.parametrize("tile", ["256x128s3", "128x256s3"])  # gemm_fp8blk_smalltile_kernel.TILES
def test_dsv4s_small_tiles_bits_equal_flydsl(tile):
    from primus_turbo.gluon.gemm.gemm_fp8blk_smalltile_kernel import dsv4s_gemm, shape_supported
    from primus_turbo.pytorch.kernels.gemm import gemm_fp8_flydsl_gfx1250_impl

    m, n, k = 512, 512, 1536
    if not shape_supported(m, n, k, tile):
        pytest.skip(f"tile {tile} does not support {m}x{n}x{k}")
    a, b, ea, eb = _operands(m, n, k, seed=2)
    pa, pb = _packed(ea, eb, m, n, k)
    ref = gemm_fp8_flydsl_gfx1250_impl.gemm_nt(a, b, ea, eb)
    out = dsv4s_gemm(a, b, pa, pb, tile=tile)
    assert bit_diff_count(out, ref) == 0


@requires_gluon
def test_accumulate2_is_gemm_then_bf16_add():
    """accumulate=2: C = bf16(A.B^T) + C with one fp32 add and one rounding (torch's bf16 add)."""
    from primus_turbo.gluon.gemm.gemm_fp8blk_kernel import dsv4up_gemm
    from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl

    m = 512
    for n, k in ((1024, 1536), (1024, 1024)):
        for seed, scale in ((0, 0.1), (1, 30.0)):
            a, b, ea, eb = _operands(m, n, k, seed=seed)
            pa, pb = _packed(ea, eb, m, n, k)
            g = torch.Generator(device="cuda").manual_seed(3 + seed)
            c0 = (torch.randn(m, n, device="cuda", generator=g) * scale).to(torch.bfloat16)
            ref = c0 + gemm_fp8_gluon_gfx1250_impl.gemm_nt(a, b, pa, pb, torch.bfloat16)
            out = c0.clone()
            dsv4up_gemm(a, b, pa, pb, out=out, accumulate=2)
            assert bits_equal(out, ref)


@requires_gluon
@pytest.mark.parametrize("axis, split", [("n", 256), ("m", 256)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_split_output_epilogue_matches_plain(axis, split, accumulate):
    from primus_turbo.gluon.gemm.gemm_fp8blk_kernel import dsv4up_gemm, dsv4up_gemm_split

    m, n, k = 512, 768, 1024
    a, b, ea, eb = _operands(m, n, k, seed=4)
    pa, pb = _packed(ea, eb, m, n, k)
    dt = torch.float32 if accumulate else torch.bfloat16
    c0 = torch.randn(m, n, device="cuda").to(dt)
    ref = c0.clone()
    dsv4up_gemm(a, b, pa, pb, out=ref, accumulate=accumulate) if accumulate else ref.copy_(
        dsv4up_gemm(a, b, pa, pb, out_dtype=dt)
    )
    if axis == "n":
        o0, o1 = c0[:, :split].contiguous(), c0[:, split:].contiguous()
    else:
        o0, o1 = c0[:split].contiguous(), c0[split:].contiguous()
    dsv4up_gemm_split(a, b, pa, pb, o0, o1, axis, accumulate=accumulate)
    out = torch.cat([o0, o1], dim=1 if axis == "n" else 0)
    assert bits_equal(out, ref)


# ----------------------------------------------------------------------------- Linear / concat-N ops


def _use_kernel(monkeypatch, kernel):
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_KERNEL", kernel)
    monkeypatch.setenv("PRIMUS_TURBO_DSV4UP_PREFER_G90", "")  # every supported pass on dsv4up
    monkeypatch.delenv("PRIMUS_TURBO_FP8_WCACHE", raising=False)


def _linear_step(x, w, gy, mg0):
    w.main_grad.copy_(mg0)
    w.grad = None
    w.grad_added_to_main_grad = False
    xx = x.detach().requires_grad_(True)
    y = pt.ops.gemm_fp8(
        xx, w, trans_a=False, trans_b=True, out_dtype=None, config=CFG, fuse_bgrad_accum_pattern="megatron"
    )
    y.backward(gy)
    torch.cuda.synchronize()
    return y.detach().clone(), xx.grad.clone(), w.main_grad.clone()


@requires_gluon
@requires_flydsl
@pytest.mark.parametrize("beta0", [False, True], ids=["wgrad_beta1", "wgrad_beta0"])
@pytest.mark.parametrize("t, n, k", [(1024, 1024, 1024), (512, 1024, 2048)])
def test_linear_dsv4up_vs_flydsl(monkeypatch, t, n, k, beta0):
    torch.manual_seed(0)
    x = (torch.randn(t, k, device="cuda") * 0.5).bfloat16()
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16().requires_grad_(True)
    gy = (torch.randn(t, n, device="cuda") * 0.1).bfloat16()
    mg0 = torch.randn(n, k, device="cuda") * 0.01
    w.main_grad = mg0.clone()
    ops_utils.set_main_grad_first_write_policy((lambda p: True) if beta0 else None)
    try:
        res = {}
        for kernel in ("g90", "dsv4up"):
            _use_kernel(monkeypatch, kernel)
            res[kernel] = _linear_step(x, w, gy, mg0)
    finally:
        ops_utils.set_main_grad_first_write_policy(None)
    base = 0 if beta0 else mg0
    (yg, dxg, mgg), (yd, dxd, mgd) = res["g90"], res["dsv4up"]
    assert rel_l2(yd, yg) < 7e-3
    assert rel_l2(dxd, dxg) < 7e-3
    assert rel_l2(mgd - base, mgg - base) < 7e-3


@requires_gluon
def test_concat_n_matches_separate_linears(monkeypatch):
    from primus_turbo.pytorch.ops.gemm_fp8_concat import (
        gemm_fp8_block_concat_n,
        gemm_fp8_block_concat_n_supported,
    )

    _use_kernel(monkeypatch, "dsv4up")
    t, k, n0, n1 = 1024, 1024, 512, 512
    torch.manual_seed(0)
    x = (torch.randn(t, k, device="cuda") * 0.5).bfloat16()
    ws = [(torch.randn(n, k, device="cuda") * 0.02).bfloat16().requires_grad_(True) for n in (n0, n1)]
    gys = [(torch.randn(t, n, device="cuda") * 0.1).bfloat16() for n in (n0, n1)]
    mg0 = [torch.randn(n, k, device="cuda") * 0.01 for n in (n0, n1)]
    if not gemm_fp8_block_concat_n_supported(x, ws, CFG):
        pytest.skip("concat-N op not enabled for this configuration")

    def prepare():
        for w, m0 in zip(ws, mg0):
            w.main_grad = m0.clone()
            w.grad = None
            w.grad_added_to_main_grad = False

    prepare()
    xc = x.clone().requires_grad_(True)
    ys = gemm_fp8_block_concat_n(xc, ws, CFG, ["megatron", "megatron"])
    torch.autograd.backward(list(ys), gys)
    cat_y = [y.detach().clone() for y in ys]
    cat_mg = [w.main_grad.clone() for w in ws]
    cat_dx = xc.grad.clone()

    prepare()
    sep_y, sep_dx = [], torch.zeros_like(x, dtype=torch.float32)
    for w, gy in zip(ws, gys):
        xi = x.clone().requires_grad_(True)
        y = pt.ops.gemm_fp8(
            xi,
            w,
            trans_a=False,
            trans_b=True,
            out_dtype=None,
            config=CFG,
            fuse_bgrad_accum_pattern="megatron",
        )
        y.backward(gy)
        sep_y.append(y.detach().clone())
        sep_dx += xi.grad.float()
    torch.cuda.synchronize()
    for yc, ys_ in zip(cat_y, sep_y):
        assert bits_equal(yc, ys_)
    for w, mg in zip(ws, cat_mg):
        assert bits_equal(mg, w.main_grad)
    # dgrad runs one GEMM over the joint K (one rounding): equal to the separate sum only to bf16 precision
    assert rel_l2(cat_dx, sep_dx) < 1e-2


# ----------------------------------------------------------------------------- availability gate


@requires_gluon
@requires_flydsl
def test_fallback_to_flydsl_when_gluon_unavailable(monkeypatch):
    """With the Gluon kernels reported unavailable the block-FP8 Linear still runs (FlyDSL kernel)."""
    from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl

    monkeypatch.setenv("PRIMUS_GLUON", "0")
    monkeypatch.setattr(gemm_fp8_gluon_gfx1250_impl, "_AVAILABLE", False)
    assert not gemm_fp8_gluon_gfx1250_impl.available()
    assert not gemm_fp8_gluon_gfx1250_impl.enabled()

    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_KERNEL", "dsv4up")
    monkeypatch.delenv("PRIMUS_TURBO_FP8_WCACHE", raising=False)
    torch.manual_seed(0)
    t, n, k = 512, 1024, 1024
    x = (torch.randn(t, k, device="cuda") * 0.5).bfloat16().requires_grad_(True)
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16().requires_grad_(True)
    y = pt.ops.gemm_fp8(x, w, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=CFG)
    y.backward(torch.randn(t, n, device="cuda").bfloat16())
    ref = x.detach().float() @ w.detach().float().t()
    assert rel_l2(y, ref) < 5e-2
    assert torch.isfinite(x.grad.float()).all() and torch.isfinite(w.grad.float()).all()


def test_joint_main_grad_needs_one_storage():
    """The joint [sum N, K] main_grad view is built only over slices of one buffer (DDP's layout): back-to-back
    separate allocations, other orders, other dtypes and non-contiguous slices get None."""
    from torch.utils.dlpack import from_dlpack, to_dlpack

    from primus_turbo.pytorch.ops.gemm_fp8_concat import _joint_main_grad

    n0, n1, k = 256, 128, 64
    buf = torch.zeros((n0 + n1) * k, device="cuda")
    a, b = buf[: n0 * k].view(n0, k), buf[n0 * k :].view(n1, k)
    joint = _joint_main_grad(None, [a, b])
    assert joint is not None and joint.shape == (n0 + n1, k) and joint.data_ptr() == a.data_ptr()
    assert _joint_main_grad(None, [b, a]) is None
    # the same addresses, but b in a storage of its own (as two adjacent allocations would be)
    b_own = from_dlpack(to_dlpack(b))
    assert b_own.data_ptr() == b.data_ptr()
    assert b_own.untyped_storage().data_ptr() != a.untyped_storage().data_ptr()
    assert _joint_main_grad(None, [a, b_own]) is None
    assert _joint_main_grad(None, [a, b.bfloat16()]) is None
    assert _joint_main_grad(None, [a, buf[n0 * k :].view(k, n1).t()]) is None
