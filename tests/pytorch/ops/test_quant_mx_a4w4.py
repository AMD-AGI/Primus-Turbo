###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""quantize_mx_* with the A4W4 output formats (MXFP4 operands for AITER's f4gemm).

What is held to what:
  * The FP4 quantization must be AITER's MXFP4, byte for byte: the plain dual pack, decoded
    from the A4W4 layout, equals aiter.quant_mxfp4_gemm (decoded from its blob) of the tensor,
    in both directions. (Not the A6W4 weight packer, which can differ from AITER on a small
    fraction of codes, while the fused kernel's FP4 emit is exact.)
  * fmt=2's row direction is the FP6 blob, byte for byte.
  * A fused prologue changes what is packed, not how: the fused FP4 pack must equal the plain
    FP4 pack of the prologue's eager output -- byte for byte where the FP6 tests demand it
    (identity, AdaLN modulate, gate multiply), within the GELU modes' tolerance otherwise.
  * The layout is what AITER's A4W4 kernels read: gemm_a4w4 on a gradient (fmt=1) against a
    weight (fmt=2 column direction) matches an fp32 dequantise-and-multiply of the same codes.
Packed blobs are compared only through the logical decoders, never raw: padding is unwritten.
"""

import os

import pytest
import torch

from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
    MX_FMT_A4W4_ACT,
    MX_FMT_A4W4_GRAD,
    MX_FMT_FP6,
    a4w4_logical,
    a4w4_operand,
    a6w4_blob_logical,
    quantize_mx,
    quantize_mx_dual,
    quantize_mx_fused_dual,
    quantize_mx_gate_mul,
    quantize_mx_ln_modulate,
)
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
    mxfp6_apply_prologue,
    mxfp6_data_region,
    mxfp6_gate_mul_reference,
    mxfp6_ln_modulate_reference,
    quantize_mxfp6_dual,
)

SHAPES = [(256, 256), (512, 768), (1024, 3072), (768, 1280)]


def _skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not hasattr(torch.ops.primus_turbo_cpp_extension, "quantize_mx_dual"):
        pytest.skip("Primus-Turbo built without the quantize_mx_* ops")


def _rand(rows, cols, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16, generator=g)


def _ref_dirs(x):
    """AITER's own MXFP4 quantization of x (row direction) and x.T (column), logical form."""
    aiter = pytest.importorskip("aiter")
    rows, cols = x.shape
    rc, rs = aiter.quant_mxfp4_gemm(x)
    cc, cs = aiter.quant_mxfp4_gemm(x.t().contiguous())
    return a6w4_blob_logical(rc, rs, rows, cols), a6w4_blob_logical(cc, cs, cols, rows)


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_gradient_dual_is_aiter_mxfp4_bit_exact(rows, cols):
    _skip()
    x = _rand(rows, cols)
    rc, rs, cc, cs = quantize_mx_dual(x, MX_FMT_A4W4_GRAD)
    (ref_rc, ref_rs), (ref_cc, ref_cs) = _ref_dirs(x)
    got_rc, got_rs = a4w4_logical(rc, rs, rows, cols, is_b=False)
    got_cc, got_cs = a4w4_logical(cc, cs, cols, rows, is_b=False)
    assert torch.equal(got_rc, ref_rc) and torch.equal(got_rs, ref_rs), "row direction"
    assert torch.equal(got_cc, ref_cc) and torch.equal(got_cs, ref_cs), "column direction"


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_activation_dual_fp6_rows_and_b_shuffled_columns(rows, cols):
    _skip()
    x = _rand(rows, cols, seed=1)
    rc, rs, cc, cs = quantize_mx_dual(x, MX_FMT_A4W4_ACT)
    f6 = quantize_mxfp6_dual(x)
    assert torch.equal(mxfp6_data_region(rc, rows, cols), mxfp6_data_region(f6[0], rows, cols))
    assert torch.equal(
        mxfp6_data_region(rs, rows, cols, is_scale=True),
        mxfp6_data_region(f6[1], rows, cols, is_scale=True),
    )
    _, (ref_cc, ref_cs) = _ref_dirs(x)
    got_cc, got_cs = a4w4_logical(cc, cs, cols, rows, is_b=True)
    assert torch.equal(got_cc, ref_cc) and torch.equal(got_cs, ref_cs)


def test_fmt0_is_the_mxfp6_op():
    _skip()
    rows, cols = 512, 768
    x = _rand(rows, cols, seed=2)
    a, b = quantize_mx_dual(x, MX_FMT_FP6), quantize_mxfp6_dual(x)
    for got, ref, r, k, sc in (
        (a[0], b[0], rows, cols, False),
        (a[1], b[1], rows, cols, True),
        (a[2], b[2], cols, rows, False),
        (a[3], b[3], cols, rows, True),
    ):
        assert torch.equal(
            mxfp6_data_region(got, r, k, is_scale=sc), mxfp6_data_region(ref, r, k, is_scale=sc)
        )


@pytest.mark.parametrize("axis", [1, 0])
def test_single_direction_matches_dual(axis):
    _skip()
    rows, cols = 512, 1280
    x = _rand(rows, cols, seed=3)
    one = quantize_mx(x, axis, MX_FMT_A4W4_GRAD)
    both = quantize_mx_dual(x, MX_FMT_A4W4_GRAD)
    r, k = (rows, cols) if axis == 1 else (cols, rows)
    pair = both[:2] if axis == 1 else both[2:]
    assert all(
        torch.equal(u, v) for u, v in zip(a4w4_logical(*one, r, k, False), a4w4_logical(*pair, r, k, False))
    )


def _same_logical(got, ref, rows, cols, fmt):
    is_b = fmt == MX_FMT_A4W4_ACT
    out = []
    if fmt == MX_FMT_A4W4_GRAD:
        out.append(
            a4w4_logical(got[0], got[1], rows, cols, False) + a4w4_logical(ref[0], ref[1], rows, cols, False)
        )
    out.append(
        a4w4_logical(got[2], got[3], cols, rows, is_b) + a4w4_logical(ref[2], ref[3], cols, rows, is_b)
    )
    return out


@pytest.mark.parametrize("fmt", [MX_FMT_A4W4_GRAD, MX_FMT_A4W4_ACT])
@pytest.mark.parametrize("rows,cols,batch", [(256, 256, 1), (512, 768, 2), (1024, 512, 32)])
def test_ln_modulate_fp4_is_bit_exact(fmt, rows, cols, batch):
    _skip()
    x = _rand(rows, cols, seed=4)
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-6)
    scale = _rand(batch, cols, seed=5)
    shift = _rand(batch, cols, seed=6)
    got = quantize_mx_ln_modulate(x, mean, rstd, scale, shift, False, fmt)
    ref = quantize_mx_dual(mxfp6_ln_modulate_reference(x, mean, rstd, scale, shift), fmt)
    for gc, gs, rc, rs in _same_logical(got, ref, rows, cols, fmt):
        assert torch.equal(gc, rc) and torch.equal(gs, rs)


@pytest.mark.parametrize("rows,cols,batch", [(256, 256, 1), (512, 768, 2), (1024, 3072, 32)])
def test_gate_mul_fp4_is_bit_exact(rows, cols, batch):
    _skip()
    x = _rand(rows, cols, seed=7)
    gate = _rand(batch, cols, seed=8)
    got = quantize_mx_gate_mul(x, gate, False, MX_FMT_A4W4_GRAD)
    ref = quantize_mx_dual(mxfp6_gate_mul_reference(x, gate), MX_FMT_A4W4_GRAD)
    for gc, gs, rc, rs in _same_logical(got, ref, rows, cols, MX_FMT_A4W4_GRAD):
        assert torch.equal(gc, rc) and torch.equal(gs, rs)


@pytest.mark.parametrize(
    "mode,fmt", [(0, MX_FMT_A4W4_GRAD), (0, MX_FMT_A4W4_ACT), (1, MX_FMT_A4W4_ACT), (2, MX_FMT_A4W4_GRAD)]
)
@pytest.mark.parametrize("rows,cols", [(256, 256), (512, 1024), (256, 3072)])
def test_fused_prologue_fp4_matches_eager(mode, fmt, rows, cols):
    """Identity is bit-exact; the GELU modes allow the FP6 tests' tolerance for the fused tanh."""
    _skip()
    x = _rand(rows, cols, seed=9)
    aux = _rand(rows, cols, seed=10) if mode == 2 else None
    bias = _rand(1, cols, seed=11).view(-1) if mode != 0 else None
    got = quantize_mx_fused_dual(x, aux, bias, mode, False, fmt)
    ref = quantize_mx_dual(mxfp6_apply_prologue(x, aux, bias, mode), fmt)
    for gc, gs, rc, rs in _same_logical(got, ref, rows, cols, fmt):
        if mode == 0:
            assert torch.equal(gc, rc) and torch.equal(gs, rs)
        else:
            assert (gc != rc).float().mean().item() <= 2e-3
            assert (gs != rs).float().mean().item() <= 2e-3


@pytest.mark.parametrize("m,n,k", [(256, 256, 256), (1024, 768, 512), (8192, 3072, 3072)])
def test_a4w4_gemm_on_packed_operands(m, n, k):
    """dgrad-style product: gradient [m, k] (fmt 1 rows) x weight [k, n] (fmt 2 columns)."""
    _skip()
    aiter = pytest.importorskip("aiter")
    from aiter.utility import fp4_utils

    dy = _rand(m, k, seed=12)
    w = _rand(k, n, seed=13)  # weight as stored for dgrad: [out = k, in = n]
    a_c, a_s, _, _ = quantize_mx_dual(dy, MX_FMT_A4W4_GRAD)
    _, _, b_c, b_s = quantize_mx_dual(w, MX_FMT_A4W4_ACT)  # columns of w = rows of w.T
    A, As = a4w4_operand(a_c, a_s, m, k)
    B, Bs = a4w4_operand(b_c, b_s, n, k)
    out = aiter.gemm_a4w4(A, B, As, Bs, bpreshuffle=True)[:m, :n].float()

    def dequant(codes, scales, rows, kk, is_b):
        c, s = a4w4_logical(codes, scales, rows, kk, is_b)
        return fp4_utils.mxfp4_to_f32(c) * fp4_utils.e8m0_to_f32(s.repeat_interleave(32, 1))

    ref = dequant(a_c, a_s, m, k, False) @ dequant(b_c, b_s, n, k, True).T
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 4e-3, rel
    # and the rotated, quantized product still approximates the bf16 one (Hadamard cancels)
    true = dy.float() @ w.float()
    assert ((out - true).norm() / true.norm()).item() < 0.2


@pytest.mark.parametrize("m,heads", [(256, 2), (1024, 8)])
def test_qk_norm_rope_bwd_fp4_tracks_its_fp6_twin(m, heads):
    """The QK-norm+RoPE prologue reduces in its own order, so even its FP6 pack is not byte-equal
    to packing the eager reference. Calibrate on FP6 with the same operands and require the FP4
    pack to stay as close to its reference (the format must not add disagreement)."""
    _skip()
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import quantize_mx_qk_norm_rope_bwd
    from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
        mxfp6_qk_norm_rope_backward_reference,
    )

    d = 128
    n = 3 * heads * d
    qkv = _rand(m, n, seed=20)
    dq, dk, dv = (_rand(m, heads * d, seed=s) for s in (21, 22, 23))
    ang = torch.rand(m, d // 2, device="cuda") * 6.28
    cos = torch.cos(ang).repeat_interleave(2, 1).to(torch.bfloat16)
    sin = torch.sin(ang).repeat_interleave(2, 1).to(torch.bfloat16)
    wq, wk = (1 + 0.1 * _rand(1, d, seed=s).view(-1) for s in (24, 25))
    q = qkv.float().view(m, heads, 3, d)
    rstd_q = torch.rsqrt(q[:, :, 0].pow(2).mean(-1) + 1e-6).reshape(-1)
    rstd_k = torch.rsqrt(q[:, :, 1].pow(2).mean(-1) + 1e-6).reshape(-1)
    args = (qkv, dq, dk, dv, cos, sin, wq, wk, rstd_q, rstd_k)
    ref_x = mxfp6_qk_norm_rope_backward_reference(*args)[0]

    f6 = quantize_mx_qk_norm_rope_bwd(*args, False, MX_FMT_FP6)
    r6 = quantize_mx_dual(ref_x, MX_FMT_FP6)
    fp6_diff = max(
        (mxfp6_data_region(f6[i], r, k) != mxfp6_data_region(r6[i], r, k)).float().mean().item()
        for i, (r, k) in ((0, (m, n)), (2, (n, m)))
    )
    f4 = quantize_mx_qk_norm_rope_bwd(*args, False, MX_FMT_A4W4_GRAD)
    r4 = quantize_mx_dual(ref_x, MX_FMT_A4W4_GRAD)
    for gc, gs, rc, rs in _same_logical(f4, r4, m, n, MX_FMT_A4W4_GRAD):
        assert (gc != rc).float().mean().item() <= max(2 * fp6_diff, 1e-3)
        assert (gs != rs).float().mean().item() <= max(2 * fp6_diff, 1e-3)
    # the side outputs are format-independent
    assert torch.equal(f4[5], f6[5]) and torch.equal(f4[6], f6[6])


@pytest.mark.parametrize("fmt,rows_fp4,cols_fp4", [(3, False, True), (4, True, False)])
def test_single_gate_gradient_formats(fmt, rows_fp4, cols_fp4):
    """fmt 3 / 4: one direction is fmt 1's A4W4 operand, the other is fmt 0's FP6 blob, both exact."""
    _skip()
    rows, cols = 512, 1280
    x = _rand(rows, cols, seed=30)
    got = quantize_mx_dual(x, fmt)
    f4, f6 = quantize_mx_dual(x, MX_FMT_A4W4_GRAD), quantize_mx_dual(x, MX_FMT_FP6)
    for i, (r, k), fp4 in ((0, (rows, cols), rows_fp4), (2, (cols, rows), cols_fp4)):
        if fp4:
            g, ref = (
                a4w4_logical(got[i], got[i + 1], r, k, False),
                a4w4_logical(f4[i], f4[i + 1], r, k, False),
            )
            assert torch.equal(g[0], ref[0]) and torch.equal(g[1], ref[1])
        else:
            assert torch.equal(mxfp6_data_region(got[i], r, k), mxfp6_data_region(f6[i], r, k))
            assert torch.equal(
                mxfp6_data_region(got[i + 1], r, k, is_scale=True),
                mxfp6_data_region(f6[i + 1], r, k, is_scale=True),
            )


# dgrad-like and wgrad-like (token contraction, the split-K candidates) Flux shapes
@pytest.mark.parametrize(
    "m,n,k", [(1024, 768, 512), (8192, 3072, 3072), (3072, 3072, 8192), (12288, 3072, 8192)]
)
def test_gemm_fp6_a4w4_modes(m, n, k):
    """gemm_fp6_impl(a4w4=True) is aiter.gemm_a4w4 on the packed operands, and the out
    variant overwrites (beta 0, split-K included) a caller buffer with the same values."""
    _skip()
    aiter = pytest.importorskip("aiter")
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl, gemm_fp6_out_impl

    gran = ScalingGranularity.MX_BLOCKWISE.value
    a_c, a_s, _, _ = quantize_mx_dual(_rand(m, k, seed=31), MX_FMT_A4W4_GRAD)
    _, _, b_c, b_s = quantize_mx_dual(_rand(k, n, seed=32), MX_FMT_A4W4_ACT)
    A, As = a4w4_operand(a_c, a_s, m, k)
    B, Bs = a4w4_operand(b_c, b_s, n, k)
    ref = aiter.gemm_a4w4(A, B, As, Bs, bpreshuffle=True)[:m, :n]

    out = gemm_fp6_impl(a_c, a_s, b_c, b_s, m, n, k, torch.bfloat16, gran, None, a4w4=True)
    assert out.shape == (m, n) and torch.equal(out, ref)

    buf = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    gemm_fp6_out_impl(a_c, a_s, b_c, b_s, buf, m, n, k, gran, a4w4=True)
    assert torch.isfinite(buf).all()
    rel = ((buf.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 1e-2, rel  # the out variant may pick a different (asm) kernel than gemm_a4w4


def test_wrappers_accept_non_contiguous_operands():
    """The ops assert contiguity; the wrappers must make operands contiguous like the FP6 ones
    (a DiT gate arrives as a strided slice of the AdaLN output)."""
    _skip()
    x = _rand(512, 768, seed=40)
    gate = _rand(768, 4, seed=41).t()  # [4, 768], non-contiguous
    assert not gate.is_contiguous()
    got = quantize_mx_gate_mul(x, gate, False, MX_FMT_A4W4_GRAD)
    ref = quantize_mx_gate_mul(x, gate.contiguous(), False, MX_FMT_A4W4_GRAD)
    for gc, gs, rc, rs in _same_logical(got, ref, 512, 768, MX_FMT_A4W4_GRAD):
        assert torch.equal(gc, rc) and torch.equal(gs, rs)


# ---- stochastic rounding (fmt 5 / 6 / 7) ----


def _dequant_logical(codes, scales, rows, k, is_b=False):
    from aiter.utility import fp4_utils

    c, s = a4w4_logical(codes, scales, rows, k, is_b)
    return fp4_utils.mxfp4_to_f32(c) * fp4_utils.e8m0_to_f32(s.repeat_interleave(32, 1))


def _rotated(x):
    """What the packer quantizes: each 32-group Hadamard-rotated (normalised) and bf16-rounded."""
    import math

    h = torch.ones(1, 1, device=x.device)
    while h.shape[0] < 32:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    r = (x.float().reshape(x.shape[0], -1, 32) @ (h / math.sqrt(32))).reshape(x.shape)
    return r.to(torch.bfloat16).float()


@pytest.mark.parametrize("rows,cols", [(512, 768), (1024, 3072)])
def test_sr_keeps_scales_and_rounds_to_a_neighbour(rows, cols):
    """SR changes the code, never the block scale, and lands on one of the two grid points
    around the exact (rotated) value: within the widest E2M1 gap, 2 x scale."""
    _skip()
    pytest.importorskip("aiter")
    from aiter.utility import fp4_utils

    x = _rand(rows, cols, seed=50)
    rtn = quantize_mx_dual(x, MX_FMT_A4W4_GRAD)
    sr = quantize_mx_dual(x, 5)
    _, s_rtn = a4w4_logical(rtn[0], rtn[1], rows, cols, False)
    _, s_sr = a4w4_logical(sr[0], sr[1], rows, cols, False)
    assert torch.equal(s_rtn, s_sr), "SR must not change the block scale"
    _, s_c_rtn = a4w4_logical(rtn[2], rtn[3], cols, rows, False)
    _, s_c_sr = a4w4_logical(sr[2], sr[3], cols, rows, False)
    assert torch.equal(s_c_rtn, s_c_sr)
    ref = _rotated(x)
    scale = fp4_utils.e8m0_to_f32(s_sr.repeat_interleave(32, 1)).float()
    d_sr, d_rtn = _dequant_logical(sr[0], sr[1], rows, cols), _dequant_logical(rtn[0], rtn[1], rows, cols)
    assert ((d_sr - ref).abs() <= 2 * scale + 1e-6).all()
    same = (d_sr == d_rtn).float().mean().item()
    assert 0.3 < same < 0.98, same  # SR agrees with RTN often, not always


def test_sr_is_unbiased_and_varies_per_launch():
    """Mean of repeated SR packs converges to the exact rotated values; RTN's error does not shrink."""
    _skip()
    pytest.importorskip("aiter")
    rows, cols, n = 256, 1024, 64
    x = _rand(rows, cols, seed=51)
    ref = _rotated(x)
    rtn = quantize_mx_dual(x, MX_FMT_A4W4_GRAD)
    e_rtn = ((_dequant_logical(rtn[0], rtn[1], rows, cols) - ref).norm() / ref.norm()).item()
    acc = torch.zeros_like(ref)
    first = None
    for _ in range(n):
        p = quantize_mx_dual(x, 5)
        d = _dequant_logical(p[0], p[1], rows, cols)
        first = d if first is None else first
        acc += d
    assert not torch.equal(first, d), "consecutive SR launches must draw different rounding"
    e_mean = ((acc / n - ref).norm() / ref.norm()).item()
    assert e_mean < 0.35 * e_rtn, (e_mean, e_rtn)


@pytest.mark.parametrize("fmt,rows_fp4,cols_fp4", [(6, False, True), (7, True, False)])
def test_sr_single_gate_formats_keep_their_fp6_direction(fmt, rows_fp4, cols_fp4):
    _skip()
    rows, cols = 512, 1280
    x = _rand(rows, cols, seed=52)
    got, f6 = quantize_mx_dual(x, fmt), quantize_mx_dual(x, MX_FMT_FP6)
    for i, (r, k), fp4 in ((0, (rows, cols), rows_fp4), (2, (cols, rows), cols_fp4)):
        if not fp4:
            assert torch.equal(mxfp6_data_region(got[i], r, k), mxfp6_data_region(f6[i], r, k))


def test_sr_gradient_feeds_gemm_a4w4():
    _skip()
    aiter = pytest.importorskip("aiter")
    m, n, k = 1024, 768, 512
    dy, w = _rand(m, k, seed=53), _rand(k, n, seed=54)
    a_c, a_s, _, _ = quantize_mx_dual(dy, 5)
    _, _, b_c, b_s = quantize_mx_dual(w, MX_FMT_A4W4_ACT)
    A, As = a4w4_operand(a_c, a_s, m, k)
    B, Bs = a4w4_operand(b_c, b_s, n, k)
    out = aiter.gemm_a4w4(A, B, As, Bs, bpreshuffle=True)[:m, :n].float()
    true = dy.float() @ w.float()
    assert ((out - true).norm() / true.norm()).item() < 0.3


# ---- caller-buffer variants (e.g. a grouped MLP's packs under A4W4) ----


@pytest.mark.parametrize("fmt", [0, 1, 2, 3, 4])
def test_dual_out_matches_allocating(fmt):
    _skip()
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import mx_dir_sizes, quantize_mx_dual_out

    rows, cols = 512, 768
    x = _rand(rows, cols, seed=60)
    ref = quantize_mx_dual(x, fmt)
    rp, rs = mx_dir_sizes(rows, cols, fmt, False)
    cp, cs = mx_dir_sizes(cols, rows, fmt, True)
    bufs = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in (rp, rs, cp, cs)]
    quantize_mx_dual_out(x, *bufs, fmt)
    row_fp4, col_fp4 = fmt in (1, 4), fmt in (1, 2, 3)
    for i, (r, k), fp4 in ((0, (rows, cols), row_fp4), (2, (cols, rows), col_fp4)):
        if fp4:
            is_b = fmt == 2  # fmt 2's columns are the B operand
            g = a4w4_logical(bufs[i], bufs[i + 1], r, k, is_b)
            w = a4w4_logical(ref[i], ref[i + 1], r, k, is_b)
            assert torch.equal(g[0], w[0]) and torch.equal(g[1], w[1])
        else:
            assert torch.equal(mxfp6_data_region(bufs[i], r, k), mxfp6_data_region(ref[i], r, k))


@pytest.mark.parametrize("fmt", [1, 2])
def test_fused_dual_out_matches_allocating(fmt):
    _skip()
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
        mx_dir_sizes,
        quantize_mx_fused_dual_out,
    )
    from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_col_sum_rows

    rows, cols = 512, 768
    y, aux = _rand(rows, cols, seed=61), _rand(rows, cols, seed=62)
    b = _rand(1, cols, seed=63).view(-1)
    mode = 2 if fmt == 1 else 1  # GELU backward for a gradient, GELU forward for an activation
    ref = quantize_mx_fused_dual(y, aux if mode == 2 else None, b, mode, mode == 2, fmt)
    rp, rs = mx_dir_sizes(rows, cols, fmt, False)
    cp, cs = mx_dir_sizes(cols, rows, fmt, True)
    bufs = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in (rp, rs, cp, cs)]
    part = (
        torch.empty(mxfp6_col_sum_rows(rows), cols, dtype=torch.float32, device="cuda") if mode == 2 else None
    )
    quantize_mx_fused_dual_out(y, aux if mode == 2 else None, b, mode, *bufs, part, fmt)
    is_b = fmt == 2
    g = a4w4_logical(bufs[2], bufs[3], cols, rows, is_b)
    w = a4w4_logical(ref[2], ref[3], cols, rows, is_b)
    assert torch.equal(g[0], w[0]) and torch.equal(g[1], w[1])
    if mode == 2:
        assert torch.equal(part, ref[4])


# ---- FlyDSL operand formats (fmt 8 / 9 / 12) and the a4w4=2 GEMM ----


@pytest.mark.parametrize("rows,cols", [(512, 768), (1024, 3072)])
def test_plain_formats_hold_the_same_codes_as_aiter_formats(rows, cols):
    _skip()
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import plain_operand

    x = _rand(rows, cols, seed=70)
    g_a, g_p = quantize_mx_dual(x, MX_FMT_A4W4_GRAD), quantize_mx_dual(x, 8)
    for i, (r, k) in ((0, (rows, cols)), (2, (cols, rows))):
        ac, as_ = a4w4_logical(g_a[i], g_a[i + 1], r, k, False)
        pc, ps = plain_operand(g_p[i], g_p[i + 1], r, k)
        assert torch.equal(ac, pc) and torch.equal(as_, ps)
    a_a, a_p = quantize_mx_dual(x, MX_FMT_A4W4_ACT), quantize_mx_dual(x, 9)
    assert torch.equal(mxfp6_data_region(a_a[0], rows, cols), mxfp6_data_region(a_p[0], rows, cols))
    ac, as_ = a4w4_logical(a_a[2], a_a[3], cols, rows, True)
    pc, ps = plain_operand(a_p[2], a_p[3], cols, rows)
    assert torch.equal(ac, pc) and torch.equal(as_, ps)
    s1, s2 = quantize_mx_dual(x, 12), quantize_mx_dual(x, 12)
    c1, sc1 = plain_operand(s1[0], s1[1], rows, cols)
    c2, sc2 = plain_operand(s2[0], s2[1], rows, cols)
    assert torch.equal(sc1, sc2) and not torch.equal(c1, c2)  # SR: same scales, different draws


@pytest.mark.parametrize("m,n,k", [(1024, 768, 512), (8192, 3072, 3072), (3072, 12288, 8192)])
def test_flydsl_a4w4_matches_aiter_a4w4(m, n, k):
    _skip()
    pytest.importorskip("flydsl")
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl, gemm_fp6_out_impl

    gran = ScalingGranularity.MX_BLOCKWISE.value
    dy, w = _rand(m, k, seed=71), _rand(k, n, seed=72)
    ga, _, _, _ = quantize_mx_dual(dy, MX_FMT_A4W4_GRAD)
    gas = quantize_mx_dual(dy, MX_FMT_A4W4_GRAD)[1]
    _, _, wb, wbs = quantize_mx_dual(w, MX_FMT_A4W4_ACT)
    ref = gemm_fp6_impl(ga, gas, wb, wbs, m, n, k, torch.bfloat16, gran, None, a4w4=1).float()
    fa, fas, _, _ = quantize_mx_dual(dy, 8)
    _, _, fb, fbs = quantize_mx_dual(w, 9)
    got = gemm_fp6_impl(fa, fas, fb, fbs, m, n, k, torch.bfloat16, gran, None, a4w4=2).float()
    assert ((got - ref).norm() / ref.norm()).item() < 1e-2
    buf = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    gemm_fp6_out_impl(fa, fas, fb, fbs, buf, m, n, k, gran, a4w4=2)
    assert torch.isfinite(buf).all() and ((buf.float() - ref).norm() / ref.norm()).item() < 1e-2


# ---- FlyDSL packed scales written by the packers (ts_fmt, a4w4=3) ----


@pytest.mark.parametrize(
    "m,n,k",
    [
        (8192, 3072, 3072),
        (8192, 3072, 12288),  # the shape FlyDSL's split-K autotune picked a (non-exact) split for, now off
        (3072, 3072, 16384),
        (16384, 12288, 3072),
        (9216, 3072, 8192),
    ],
)
def test_fly_packed_scales_match_flydsl_repack(m, n, k):
    """GEMM [m, k] x [n, k]^T with the scales stored packed by the packers (a4w4=3) is bit-identical
    to the exact fp32-sum result on the same values (the A4W4 tile-blob kernel). Covers B tiles nt = 3 and 4,
    interleave 0 and 4. Operands as Primus packs them: a gradient's rows (A) and a weight's columns (B)."""
    _skip()
    pytest.importorskip("flydsl")
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl, gemm_fp6_out_impl
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import TS_A, ts_b_params, ts_fmt

    gemm_a4w4_blob_asm = pytest.importorskip("aiter.ops.gemm_op_a4w4_blob").gemm_a4w4_blob_asm

    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import quantize_mx

    gran = ScalingGranularity.MX_BLOCKWISE.value
    dy, w = _rand(m, k, seed=80), _rand(k, n, seed=81)  # w as stored: [out = k, in = n]
    # Reference: the exact A4W4 tile-blob kernel on the same values (one fp32 sum of the FP4 products).
    a16, a16s = quantize_mx(dy, 1, 16)
    b16, b16s = quantize_mx(w.t().contiguous(), 1, 16)
    ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    gemm_a4w4_blob_asm(a16, b16, a16s, b16s, ref, k)
    fa, fas, _, _ = quantize_mx_dual(dy, ts_fmt(row=TS_A, col=TS_A))
    _, _, fb, fbs = quantize_mx_dual(w, ts_fmt(col=ts_b_params(m, n, k)))
    got = gemm_fp6_impl(fa, fas, fb, fbs, m, n, k, torch.bfloat16, gran, None, a4w4=3)
    assert torch.equal(got, ref)
    buf = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    gemm_fp6_out_impl(fa, fas, fb, fbs, buf, m, n, k, gran, a4w4=3)
    assert torch.equal(buf, ref)


_FLY_SHAPES = [
    (r, i, o)
    for rows in (8192, 16384)
    for o, i in ((9216, 3072), (3072, 3072), (12288, 3072), (3072, 12288))
    for r, i, o in ((rows, i, o), (o, i, rows))
]


@pytest.mark.parametrize("m,n,k", _FLY_SHAPES)
def test_aiter_fly_matches_blob(m, n, k):
    """a4w4=4 (aiter `gemm_a4w4_fly_asm`, assembly ports of FlyDSL's 256-wide kernel) on operands the packers
    wrote in the fly formats is bit-identical to the tile-blob kernel on the same values (both are one exact fp32
    sum of the FP4 products), allocating and out variants, over repeated calls (the race the 256 tile avoids shows only
    on repeats). The Flux training shapes."""
    _skip()
    pytest.importorskip("aiter.ops.gemm_op_a4w4_fly")
    gemm_a4w4_blob_asm = pytest.importorskip("aiter.ops.gemm_op_a4w4_blob").gemm_a4w4_blob_asm

    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl, gemm_fp6_out_impl
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
        TS_A,
        ts_b_params,
        ts_fmt,
        quantize_mx,
    )

    gran = ScalingGranularity.MX_BLOCKWISE.value
    dy, w = _rand(m, k, seed=82), _rand(k, n, seed=83)
    a16, a16s = quantize_mx(dy, 1, 16)
    b16, b16s = quantize_mx(w.t().contiguous(), 1, 16)
    ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    gemm_a4w4_blob_asm(a16, b16, a16s, b16s, ref, k)
    fa, fas, _, _ = quantize_mx_dual(dy, ts_fmt(row=TS_A, col=TS_A))
    _, _, fb, fbs = quantize_mx_dual(w, ts_fmt(col=ts_b_params(m, n, k)))
    for _ in range(10):
        assert torch.equal(gemm_fp6_impl(fa, fas, fb, fbs, m, n, k, torch.bfloat16, gran, None, a4w4=4), ref)
    buf = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    gemm_fp6_out_impl(fa, fas, fb, fbs, buf, m, n, k, gran, a4w4=4)
    assert torch.equal(buf, ref)


@pytest.mark.parametrize("k", [3072, 8192, 9216, 12288, 16384, 512, 768, 1024, 1280])
def test_fly_b_params_matches_flydsl(k):
    """`ts_b_params` computes the B layout without importing FlyDSL; it must equal FlyDSL's own rule at the 256 tile,
    and FlyDSL must never pick the racing 192 tile."""
    pytest.importorskip("flydsl")
    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as FK
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import ts_b_params

    kw = (k + 255) // 256 * 256
    assert ts_b_params(8192, 3072, k) == (True, 4, FK.mxfp4_packed_scale_ilv(kw, block_n=256))
    for m, n, kk in _FLY_SHAPES:
        assert FK._mxfp4_pick_block_n(m, n, kk) == 256


def test_fly_packed_sr_gradient():
    _skip()
    pytest.importorskip("flydsl")
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import TS_A, ts_fmt

    x = _rand(1024, 3072, seed=82)
    f = ts_fmt(row=TS_A, col=TS_A, sr=True)
    a, b = quantize_mx_dual(x, f), quantize_mx_dual(x, f)
    assert torch.equal(a[1], b[1]) and not torch.equal(a[0], b[0])  # same packed scales, new draws


def test_plain_row_only_pack():
    """Row-only fmt 8 (a forward-only pack) equals the dual pack's row direction."""
    _skip()
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import plain_operand

    x = _rand(512, 768, seed=90)
    r = quantize_mx(x, 1, 8)
    d = quantize_mx_dual(x, 8)
    a, b = plain_operand(r[0], r[1], 512, 768), plain_operand(d[0], d[1], 512, 768)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("m,n,k", _FLY_SHAPES)
def test_flydsl_exact_on_fly_shapes(m, n, k):
    """FlyDSL's own GEMM (a4w4=2 plain scales, a4w4=3 packed) is exact on every shape in _FLY_SHAPES with the 192-wide
    tile and the split-K launches off: bit-identical to the exact tile-blob kernel over repeated calls."""
    _skip()
    pytest.importorskip("flydsl")
    gemm_a4w4_blob_asm = pytest.importorskip("aiter.ops.gemm_op_a4w4_blob").gemm_a4w4_blob_asm

    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
        TS_A,
        ts_b_params,
        ts_fmt,
        quantize_mx,
    )

    gran = ScalingGranularity.MX_BLOCKWISE.value
    dy, w = _rand(m, k, seed=84), _rand(k, n, seed=85)
    a16, a16s = quantize_mx(dy, 1, 16)
    b16, b16s = quantize_mx(w.t().contiguous(), 1, 16)
    ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    gemm_a4w4_blob_asm(a16, b16, a16s, b16s, ref, k)
    pa, pas, _, _ = quantize_mx_dual(dy, 8)
    _, _, pb, pbs = quantize_mx_dual(w, 9)
    fa, fas, _, _ = quantize_mx_dual(dy, ts_fmt(row=TS_A, col=TS_A))
    _, _, fb, fbs = quantize_mx_dual(w, ts_fmt(col=ts_b_params(m, n, k)))
    for _ in range(10):
        assert torch.equal(gemm_fp6_impl(pa, pas, pb, pbs, m, n, k, torch.bfloat16, gran, None, a4w4=2), ref)
        assert torch.equal(gemm_fp6_impl(fa, fas, fb, fbs, m, n, k, torch.bfloat16, gran, None, a4w4=3), ref)


# ---- MXFP6 in the FlyDSL A6W6 GEMM's layout (ts6_fmt) ----


def _fp6_blob_unpack(blob, sblob, rows, k):
    """Inverse of the MXFP6 tile blob (MX_FMT_FP6 rows, AITER mxfp6_c0c1_256_padk2): -> packed 24-byte groups
    [rows, k/32, 24] and E8M0 scales [rows, k/32]."""
    g_n = k // 32
    nk_pad = k // 128 + 2
    r = torch.arange(rows, device=blob.device).view(rows, 1)
    g = torch.arange(g_n, device=blob.device).view(1, g_n)
    tile_row, rem, step, kg = r // 256, r % 256, g // 4, g % 4
    block = (rem // 16) * 64 + kg * 16 + rem % 16
    tbase = (tile_row * nk_pad + step) * 24576
    c0 = (tbase + block * 16).unsqueeze(-1) + torch.arange(16, device=blob.device)
    c1 = (tbase + 16384 + block * 8).unsqueeze(-1) + torch.arange(8, device=blob.device)
    flat = blob.view(torch.uint8).reshape(-1)
    p = torch.cat([flat[c0.reshape(-1)].view(rows, g_n, 16), flat[c1.reshape(-1)].view(rows, g_n, 8)], -1)
    sa = (
        (tile_row * nk_pad + step) * 1024 + (rem // 128) * 512 + kg * 128 + (rem % 16) * 8 + (rem % 128) // 16
    )
    return p, sblob.view(torch.uint8).reshape(-1)[sa.reshape(-1)].view(rows, g_n)


@pytest.mark.parametrize(
    "m,n,k", [(16384, 12288, 3072), (16384, 9216, 3072), (8192, 3072, 12288), (512, 768, 1024)]
)
def test_fly6_kblk_matches_fp6_blob(m, n, k):
    """ts6_fmt rows (activation as A, weight as B) carry exactly the codes and scales of the MXFP6 tile blob,
    laid out as the FlyDSL A6W6 GEMM reads them: the K128-blocked C0 / C1 planes (kblk_planes of the plain-row planes)
    and FlyDSL's packed scale slab (preshuffle_mxfp6_scales, b_ilv 0). The column direction it is paired with is
    unchanged. The GEMM on them is bit-identical to AITER's A6W6 on the blobs."""
    _skip()
    pytest.importorskip("flydsl")
    import aiter

    from primus_turbo.flydsl.gemm.gemm_mxfp6_kernel import (
        gemm_mxfp6_persistent,
        kblk_planes,
        preshuffle_mxfp6_scales,
    )
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
        TS_A,
        ts6_fmt,
        ts_b_params,
        ts_fmt,
        ts6_operand,
    )

    x, w = _rand(m, k, seed=90), _rand(n, k, seed=91)  # activation [m, k]; weight as stored [out = n, in = k]
    col_x, col_w = TS_A, ts_b_params(m, k, n)  # column directions: any fly operand (here A, and dgrad B)
    ra, ras, ca, cas = quantize_mx_dual(x, ts_fmt(col=col_x))
    rb, rbs, cb, cbs = quantize_mx_dual(w, ts_fmt(col=col_w))
    ga, gas, gca, gcas = quantize_mx_dual(x, ts6_fmt(False, col=col_x))
    gb, gbs, gcb, gcbs = quantize_mx_dual(w, ts6_fmt(True, col=col_w))
    for got, ref in ((gca, ca), (gcas, cas), (gcb, cb), (gcbs, cbs)):
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
    pa, sa = _fp6_blob_unpack(ra, ras, m, k)
    pb, sb = _fp6_blob_unpack(rb, rbs, n, k)
    ea0, ea1 = kblk_planes(
        pa[..., :16].reshape(m, k // 2).contiguous(), pa[..., 16:].reshape(m, k // 4).contiguous()
    )
    eb0, eb1 = kblk_planes(
        pb[..., :16].reshape(n, k // 2).contiguous(), pb[..., 16:].reshape(n, k // 4).contiguous()
    )
    esa, esb = preshuffle_mxfp6_scales(sa, sb, m, n, k)
    a0, a1, asp = ts6_operand(ga, gas, m, k)
    b0, b1, bsp = ts6_operand(gb, gbs, n, k)
    assert (
        ga.numel() * ga.element_size() == m * k * 3 // 4 and gb.numel() * gb.element_size() == n * k * 3 // 4
    )
    assert torch.equal(a0, ea0) and torch.equal(a1, ea1) and torch.equal(b0, eb0) and torch.equal(b1, eb1)
    assert torch.equal(asp, esa) and torch.equal(bsp, esb)
    ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    aiter.gemm_a6w6_asm(ra, rb, ras, rbs, ref, k, "f6gemm_stnt_allk_nobias_kernel_func")
    for _ in range(5):
        assert torch.equal(gemm_mxfp6_persistent(a0, a1, b0, b1, asp, bsp), ref)


_A6W6_FLY_SHAPES = [
    (16384, 12288, 3072, False),
    (8192, 12288, 3072, False),
    (8192, 3072, 3072, False),
    (8192, 3072, 12288, False),
    (16384, 9216, 3072, True),
    (8192, 9216, 3072, True),
]


@pytest.mark.parametrize("m,n,k,has_bias", _A6W6_FLY_SHAPES)
def test_a6w6_fly_matches_a6w6(m, n, k, has_bias):
    """a6w6_ts (aiter `gemm_a6w6_fly_asm`, assembly ports of the FlyDSL MXFP6 GEMM) on operands packed in
    ts6_fmt rows -- with the fly FP4 column directions the backward reads -- is bit-identical to the A6W6
    tile-blob GEMM on the same tensors, allocating and out variants, bias in the epilogue, over repeated calls. The
    Flux forward shapes."""
    _skip()
    pytest.importorskip("aiter.ops.gemm_op_a6w6_fly")
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import (
        a6w6_ts_available,
        gemm_fp6_impl,
        gemm_fp6_out_impl,
    )
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
        ts_b_params,
        ts_fmt,
        with_ts6_row,
    )

    if not a6w6_ts_available(m, n, k, has_bias):
        pytest.skip("aiter has no f6flygemm kernel for this shape")
    gran = ScalingGranularity.MX_BLOCKWISE.value
    x, w = _rand(m, k, seed=92), _rand(n, k, seed=93)
    bias = (torch.randn(n, device="cuda") * 30).to(torch.bfloat16) if has_bias else None
    ra, ras, ca, cas = quantize_mx_dual(x, MX_FMT_FP6)
    rb, rbs, _, _ = quantize_mx_dual(w, MX_FMT_FP6)
    ref = gemm_fp6_impl(ra, ras, rb, rbs, m, n, k, torch.bfloat16, gran, bias)
    col_x, col_w = ts_fmt(col=ts_b_params(n, k, m)), ts_fmt(col=ts_b_params(m, k, n))
    fa, fas, fca, fcas = quantize_mx_dual(x, with_ts6_row(col_x, False))
    fb, fbs, fcb, fcbs = quantize_mx_dual(w, with_ts6_row(col_w, True))
    _, _, eca, ecas = quantize_mx_dual(x, col_x)
    assert torch.equal(fca.view(torch.uint8), eca.view(torch.uint8)) and torch.equal(
        fcas.view(torch.uint8), ecas.view(torch.uint8)
    )
    for _ in range(10):
        assert torch.equal(
            gemm_fp6_impl(fa, fas, fb, fbs, m, n, k, torch.bfloat16, gran, bias, a6w6_ts=True), ref
        )
    buf = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    gemm_fp6_out_impl(fa, fas, fb, fbs, buf, m, n, k, gran, False, bias, a6w6_ts=True)
    assert torch.equal(buf, ref)
    assert not a6w6_ts_available(m, n, k, not has_bias) or (m, n, k) in (
        (16384, 9216, 3072),
        (8192, 9216, 3072),
    )


@pytest.mark.parametrize("row_is_b", [False, True])
def test_row_only_fly_packs_match_dual(row_is_b):
    """Row-only packs (eval forwards: no column direction) in the fly layouts write the same row bytes as the dual
    packs: FP4 fly rows (forward MXFP4) and fly6 rows."""
    _skip()
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import TS_A, ts6_fmt, ts_b_params, ts_fmt

    x = _rand(512, 1024, seed=94)
    for fmt in (ts_fmt(row=ts_b_params(256, 512, 1024) if row_is_b else TS_A), ts6_fmt(row_is_b)):
        rc, rs = quantize_mx(x, 1, fmt)
        dc, ds, _, _ = quantize_mx_dual(x, fmt)
        assert torch.equal(rc.view(torch.uint8), dc.view(torch.uint8)) and torch.equal(
            rs.view(torch.uint8), ds.view(torch.uint8)
        )


@pytest.mark.parametrize(
    "m,n,k",
    [(16384, 3072, 3072), (16384, 3072, 12288), (4096, 3072, 3072), (2304, 3072, 12288), (640, 3072, 3072)],
)
def test_a4w4_blob_fallback_matches_fly(m, n, k):
    """a4w4=5 (aiter's tile-blob kernel on MX_FMT_BLOB rows; the fallback for a forward-FP4 GEMM with no fly code
    object, e.g. an eval batch) is bit-identical to the fly path on the same values: a4w4=4 where aiter has the shape,
    FlyDSL's exact 256-wide kernel (a4w4=3) where it does not; M padded to the tile."""
    _skip()
    pytest.importorskip("aiter.ops.gemm_op_a4w4_fly")
    pytest.importorskip("aiter.ops.gemm_op_a4w4_blob")
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import a4w4_ts_shapes, gemm_fp6_impl
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
        TS_A,
        MX_FMT_BLOB_GRAD,
        ts_b_params,
        ts_fmt,
    )

    gran = ScalingGranularity.MX_BLOCKWISE.value
    x, w = _rand(m, k, seed=95), _rand(n, k, seed=96)
    xa, xas = quantize_mx(x, 1, MX_FMT_BLOB_GRAD)
    wb, wbs = quantize_mx(w, 1, MX_FMT_BLOB_GRAD)
    got = gemm_fp6_impl(xa, xas, wb, wbs, m, n, k, torch.bfloat16, gran, None, a4w4=5)
    fa, fas = quantize_mx(x, 1, ts_fmt(row=TS_A))
    fb, fbs = quantize_mx(w, 1, ts_fmt(row=ts_b_params(m, n, k)))
    if (m, n, k) in a4w4_ts_shapes():
        ref = gemm_fp6_impl(fa, fas, fb, fbs, m, n, k, torch.bfloat16, gran, None, a4w4=4)
    else:
        pytest.importorskip("flydsl")
        mp = (
            -(-m // 256) * 256
        )  # FlyDSL's packed path wants M on the tile: zero rows give the same leading rows
        xp = torch.cat([x, x.new_zeros(mp - m, k)]) if mp != m else x
        fa, fas = quantize_mx(xp, 1, ts_fmt(row=TS_A))
        ref = gemm_fp6_impl(fa, fas, fb, fbs, mp, n, k, torch.bfloat16, gran, None, a4w4=3)[:m]
    assert got.shape == (m, n) and torch.equal(got, ref)


def test_flydsl_mxfp4_pinned_configs_skip_autotune():
    """A pinned shape (flydsl.gemm.mxfp4_pinned) runs its fixed config: no timed swizzle sweep (nothing lands in the
    autotune cache) and the config cache keeps the pinned tuple; one tile per WG by default."""
    _skip()
    pytest.importorskip("flydsl")
    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as FK
    from primus_turbo.flydsl.gemm.mxfp4_pinned import PINNED
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import TS_A, ts_b_params, ts_fmt

    if os.environ.get("PRIMUS_TURBO_MXFP4_TPW_MAX") is None:
        assert FK._MXFP4_TPW_MAX == 1
    m, n, k = 8192, 3072, 3072
    assert (m, n, k) in PINNED
    at_before = {key for key in FK._MXFP4_AT_CACHE if key[:3] == (m, n, k)}
    dy, w = _rand(m, k, seed=97), _rand(k, n, seed=98)
    fa, fas, _, _ = quantize_mx_dual(dy, ts_fmt(row=TS_A, col=TS_A))
    _, _, fb, fbs = quantize_mx_dual(w, ts_fmt(col=ts_b_params(m, n, k)))
    gemm_fp6_impl(
        fa, fas, fb, fbs, m, n, k, torch.bfloat16, ScalingGranularity.MX_BLOCKWISE.value, None, a4w4=3
    )
    assert FK._MXFP4_CFG_CACHE[(m, n, k, None, False)] == tuple(PINNED[(m, n, k)][0])
    assert {key for key in FK._MXFP4_AT_CACHE if key[:3] == (m, n, k)} == at_before


@pytest.mark.parametrize(
    "m,n,k,has_bias",
    [
        (16384, 9216, 3072, True),
        (8192, 12288, 3072, False),
        (16384, 3072, 12288, False),
        (3072, 3072, 16384, False),
        (512, 768, 1536, True),
        (1024, 768, 1408, True),
    ],  # the last: K % 512 != 0 -> stays on AITER
)
def test_a6w6_flydsl_backend_matches_aiter(m, n, k, has_bias):
    """set_a6w6_backend("flydsl"): A6W6 GEMMs on Turbo's FlyDSL MXFP6 kernel compiled at runtime, on the
    standard MXFP6 blobs, are bit-identical to the AITER backend -- allocating and out variants, bias in the epilogue,
    repeated calls; a shape the kernel does not take falls back to AITER."""
    _skip()
    pytest.importorskip("flydsl")
    import primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl as GI
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity

    gran = ScalingGranularity.MX_BLOCKWISE.value
    x, w = _rand(m, k, seed=99), _rand(n, k, seed=100)
    bias = (torch.randn(n, device="cuda") * 30).to(torch.bfloat16) if has_bias else None
    a, a_s, _, _ = quantize_mx_dual(x, MX_FMT_FP6)
    b, b_s, _, _ = quantize_mx_dual(w, MX_FMT_FP6)
    import primus_turbo.flydsl.gemm.gemm_mxfp6_kernel as F6

    prev = GI._A6W6_BACKEND
    try:
        GI.set_a6w6_backend("aiter")
        ref = GI.gemm_fp6_impl(a, a_s, b, b_s, m, n, k, torch.bfloat16, gran, bias)
        GI.set_a6w6_backend("flydsl")
        for _ in range(6):
            assert torch.equal(GI.gemm_fp6_impl(a, a_s, b, b_s, m, n, k, torch.bfloat16, gran, bias), ref)
        # The FlyDSL kernel really ran for an eligible shape (and not for the fallback one).
        ran = any(key[1:4] == (m, n, k) and "aiter" in key for key in F6._LAUNCH_CACHE)
        assert ran == (k % 512 == 0), (ran, m, n, k)
        buf = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        GI.gemm_fp6_out_impl(a, a_s, b, b_s, buf, m, n, k, gran, False, bias)
        assert torch.equal(buf, ref)
    finally:
        GI.set_a6w6_backend(prev)
