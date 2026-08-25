###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from contextlib import contextmanager

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP6_PROLOGUE_BIAS_GELU,
    MXFP6_PROLOGUE_BIAS_GELU_BACKWARD,
    MXFP6_PROLOGUE_IDENTITY,
    Float6QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
    check_mxfp6_support,
    mxfp6_apply_prologue,
    mxfp6_col_sum_rows,
    mxfp6_data_region,
    mxfp6_pack_sizes,
    quantize_mxfp6_col,
    quantize_mxfp6_dual,
    quantize_mxfp6_fused_dual,
    quantize_mxfp6_row,
)
from primus_turbo.pytorch.ops.gemm_fp6 import gemm_fp6
from tests.pytorch.test_utils import compute_snr

torch.manual_seed(42)

# MXFP6 (E2M3) is a much finer format than MXFP4, so the bar is correspondingly higher
# than the 10 dB used in the FP4 tests. Measured SNR sits around 30 dB for both the
# forward and the gradients; 24 dB leaves headroom for shape-to-shape variation while
# still catching a regression to FP4-class error.
SNR_THRESHOLD_DB = 24


def _skip_if_unsupported():
    supported, reason = check_mxfp6_support()
    if not supported:
        pytest.skip(reason)


@pytest.mark.parametrize("m", [256, 512])
@pytest.mark.parametrize("n", [256, 1024])
@pytest.mark.parametrize("k", [256, 512, 2048])
def test_gemm_fp6_mx_blockwise(m, n, k):
    """Forward and both gradients against a high-precision reference."""
    _skip_if_unsupported()
    device = "cuda:0"
    dtype = torch.bfloat16

    a = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((n, k), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()

    c_ref = a_ref @ b_ref.T
    c_ref.backward(torch.ones_like(c_ref))

    config = Float6QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M3,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )
    c = gemm_fp6(a, b, trans_a=False, trans_b=True, out_dtype=dtype, config=config)
    c.backward(torch.ones_like(c))

    assert c.shape == c_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    for name, ref, got in (
        ("C", c_ref, c),
        ("AGrad", a_ref.grad, a.grad),
        ("BGrad", b_ref.grad, b.grad),
    ):
        snr = compute_snr(ref, got)
        print(f"{name}-SNR: {snr:.2f} dB")
        assert snr > SNR_THRESHOLD_DB, f"{name} snr too low: {snr:.2f} dB"


@pytest.mark.parametrize("rows,cols", [(256, 512), (512, 2048), (1024, 128)])
def test_mxfp6_pack_sizes_match_aiter(rows, cols):
    """Our size arithmetic must agree with AITER's exactly.

    Since the size check in the A6W6 entry point is exact, a disagreement here is not a
    silent inefficiency -- it makes every GEMM fail.
    """
    _skip_if_unsupported()
    from primus_turbo.common.aiter_utils import get_aiter

    assert mxfp6_pack_sizes(rows, cols) == tuple(get_aiter().mxfp6_gemm_pack_size(rows, cols))


@pytest.mark.parametrize("rows,cols", [(256, 512), (512, 2048)])
def test_mxfp6_dual_matches_single_direction_packs(rows, cols):
    """The dual packer must produce exactly what the two single-direction packers do.

    Compared on the data region only: guard tiles are never read and are left
    uninitialised, so they differ between two calls on identical input.
    """
    _skip_if_unsupported()
    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")

    row_p, row_s = quantize_mxfp6_row(x)
    col_p, col_s = quantize_mxfp6_col(x)
    d_row_p, d_row_s, d_col_p, d_col_s = quantize_mxfp6_dual(x)

    def same(lhs, rhs, r, c, is_scale=False):
        return torch.equal(
            mxfp6_data_region(lhs, r, c, is_scale=is_scale),
            mxfp6_data_region(rhs, r, c, is_scale=is_scale),
        )

    assert same(d_row_p, row_p, rows, cols)
    assert same(d_row_s, row_s, rows, cols, is_scale=True)
    assert same(d_col_p, col_p, cols, rows)
    assert same(d_col_s, col_s, cols, rows, is_scale=True)


@pytest.mark.parametrize("rows,cols", [(256, 512), (512, 1024)])
def test_mxfp6_col_pack_is_row_pack_of_transpose(rows, cols):
    _skip_if_unsupported()
    from primus_turbo.common.aiter_utils import get_aiter

    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")
    col_p, col_s = quantize_mxfp6_col(x)
    ref_p, ref_s = get_aiter().quant_mxfp6_gemm(x.t().contiguous())

    assert torch.equal(mxfp6_data_region(col_p, cols, rows), mxfp6_data_region(ref_p, cols, rows))
    assert torch.equal(
        mxfp6_data_region(col_s, cols, rows, is_scale=True),
        mxfp6_data_region(ref_s, cols, rows, is_scale=True),
    )


def _fused_packer_available() -> bool:
    return hasattr(torch.ops.primus_turbo_cpp_extension, "quantize_mxfp6_dual")


def _skip_without_fused_packer():
    if not _fused_packer_available():
        pytest.skip("build has no fused MXFP6 packer (gfx950-only)")


@contextmanager
def _packer(choice: str):
    """Force the packer backend for the duration of a block."""
    previous = os.environ.get("PRIMUS_TURBO_MXFP6_PACKER")
    os.environ["PRIMUS_TURBO_MXFP6_PACKER"] = choice
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PRIMUS_TURBO_MXFP6_PACKER", None)
        else:
            os.environ["PRIMUS_TURBO_MXFP6_PACKER"] = previous


@pytest.mark.parametrize(
    "rows,cols", [(256, 256), (256, 512), (512, 256), (1024, 1024), (256, 3072), (4096, 3072)]
)
@pytest.mark.parametrize("direction", ["row", "col"])
def test_fused_packer_is_bit_exact_with_aiter(rows, cols, direction):
    """The fused packer must reproduce AITER's packer byte for byte.

    This is the property the whole substitution rests on. AITER's packer is the oracle the
    A6W6 assembly was validated against, so anything short of bit-exactness would mean the
    GEMM is consuming operands nothing has validated -- and MXFP6's Hadamard only cancels
    if both operands were rotated identically, which a near-miss would silently break.

    It is also easy to miss by a hair: the rotation's 1/sqrt(32) has to be the bf16-rounded
    constant, because the reference applies the Hadamard as a bf16 dot. Using the exact
    fp32 value shifts values by ~1e-4 relative, which changes ~0.1% of codes by one level
    -- invisible to an SNR check, caught here.

    Every shape here is 256-aligned on the packed row count, which is the only regime
    where AITER can be compared against at all; see test_fused_packer_pads_with_zeros.
    """
    _skip_if_unsupported()
    _skip_without_fused_packer()

    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")
    pack = quantize_mxfp6_row if direction == "row" else quantize_mxfp6_col
    # Packed rows/K are transposed for the column direction.
    r, k = (rows, cols) if direction == "row" else (cols, rows)

    with _packer("fused"):
        got_p, got_s = pack(x)
    with _packer("aiter"):
        ref_p, ref_s = pack(x)

    assert torch.equal(mxfp6_data_region(got_p, r, k), mxfp6_data_region(ref_p, r, k))
    assert torch.equal(
        mxfp6_data_region(got_s, r, k, is_scale=True),
        mxfp6_data_region(ref_s, r, k, is_scale=True),
    )


@pytest.mark.parametrize("rows,cols", [(288, 256), (256, 320), (288, 320)])
def test_fused_packer_pads_with_zeros(rows, cols):
    """A shape that does not fill whole 256-row tiles must pack as if zero-extended.

    Note that AITER is deliberately not the oracle here, unlike the bit-exactness test:
    its packer reads past the end of the tensor when the packed row count is not a
    multiple of 256, so its padded region reflects whatever memory follows the operand.
    That is invisible in practice because the A6W6 path gates on 256-alignment
    everywhere, but it does mean the only trustworthy reference for these shapes is an
    explicitly zero-extended input.
    """
    _skip_if_unsupported()
    _skip_without_fused_packer()

    def ceil256(v):
        return -(-v // 256) * 256

    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")
    extended = torch.zeros((ceil256(rows), ceil256(cols)), dtype=torch.bfloat16, device="cuda:0")
    extended[:rows, :cols] = x

    with _packer("fused"):
        got_rp, got_rs, got_cp, got_cs = quantize_mxfp6_dual(x)
        ref_rp, ref_rs, ref_cp, ref_cs = quantize_mxfp6_dual(extended)

    er, ec = extended.shape

    def same(got, ref, got_rows, got_k, ref_rows, ref_k, is_scale=False):
        # Zero-extending K gives the reference extra K-tiles the operand has no room for.
        # Packing a row is per-32-block, so the tiles they share must agree exactly; the
        # reference's surplus tiles have no counterpart to compare against.
        g = mxfp6_data_region(got, got_rows, got_k, is_scale=is_scale)
        r = mxfp6_data_region(ref, ref_rows, ref_k, is_scale=is_scale)
        return torch.equal(g, r[:, : g.shape[1], :])

    assert same(got_rp, ref_rp, rows, cols, er, ec)
    assert same(got_rs, ref_rs, rows, cols, er, ec, is_scale=True)
    assert same(got_cp, ref_cp, cols, rows, ec, er)
    assert same(got_cs, ref_cs, cols, rows, ec, er, is_scale=True)


def test_fused_packer_ignores_memory_past_the_operand():
    """Packing must not depend on whatever happens to follow the tensor in memory.

    The kernel tiles to 256 rows and reads a whole tile per block, so the rows past the
    logical extent have to be masked rather than merely landing on quiet memory. Backing
    the same values onto buffers with different tails is what distinguishes the two.
    """
    _skip_if_unsupported()
    _skip_without_fused_packer()

    rows, cols = 288, 256
    values = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")

    packs = []
    for filler in (0.0, 3.5, -100.0):
        backing = torch.full((512, cols), filler, dtype=torch.bfloat16, device="cuda:0")
        backing[:rows] = values
        with _packer("fused"):
            packed, scale = quantize_mxfp6_row(backing[:rows])
        packs.append((packed, scale))

    for packed, scale in packs[1:]:
        assert torch.equal(mxfp6_data_region(packed, rows, cols), mxfp6_data_region(packs[0][0], rows, cols))
        assert torch.equal(
            mxfp6_data_region(scale, rows, cols, is_scale=True),
            mxfp6_data_region(packs[0][1], rows, cols, is_scale=True),
        )


# ---------------------------------------------------------------------------
# Fused epilogue prologue modes.
#
# The reference throughout is "apply the epilogue eagerly, then pack the result", because
# that is literally the path being replaced: an inductor-generated pointwise kernel writing
# bf16 to HBM followed by the packer reading it back.
#
# Everything about the fusion is exact against that reference except the tanh. The packer is
# VALU bound the moment a prologue is switched on, and a libm tanh costs 54 instructions per
# element against the 24 of the whole rest of the epilogue, so the prologue evaluates
# 1 + tanh(u) as 2/(1 + e^-2u) from one hardware exp2 instead. That is a different rounding
# of the same quantity, not a worse one -- ``test_fused_prologue_is_no_less_accurate_
# than_aten`` is what holds it to that -- but it does mean the codes are not bit-identical,
# so the comparisons below bound the divergence instead.
#
# The bounds are set orders of magnitude below the signature of a real bug. Applying the
# wrong epilogue shows up as ~5% of bytes differing and the fp32 bias-add bug this file
# caught showed as 4.6%, against the ~0.0003% two roundings of tanh produce. The E8M0
# scales, being a per-32-block exponent, are unaffected and are still compared exactly.
# ---------------------------------------------------------------------------

# Fraction of packed bytes allowed to differ from the eager epilogue. Measured at 3e-6 over
# the shapes here; the margin is for the tail of a different random draw, not for slop.
_PROLOGUE_CODE_TOLERANCE = 5e-5

_PROLOGUE_MODES = [MXFP6_PROLOGUE_BIAS_GELU, MXFP6_PROLOGUE_BIAS_GELU_BACKWARD]


_GELU_BETA = 0.7978845608028654
_GELU_KAPPA = 0.044715
# The kernel's own constants: the change of base for E = e^-2u, and the u below which tanhf
# returns exactly -1 and the kernel short-circuits to zero to match it.
_GELU_NEG_TWO_LOG2E = -2.0 * 1.4426950408889634
_GELU_SATURATE = -9.02


def _skip_without_fused_prologue():
    if not hasattr(torch.ops.primus_turbo_cpp_extension, "quantize_mxfp6_fused_dual"):
        pytest.skip("build has no fused MXFP6 prologue packer (gfx950-only)")


def _closed_form_prologue(x, aux, bias, mode):
    """The prologue's own arithmetic in torch, rounded to bf16 as the kernel stages it.

    Deliberately not the kernel's output: this is the *formula* the kernel uses, so that the
    accuracy of the substitution can be judged without the quantizer in the way.
    """
    xa = (x.float() + bias.float()).to(x.dtype).float()
    inner = _GELU_BETA * (xa + _GELU_KAPPA * (xa * xa * xa))
    e = torch.exp2(inner * _GELU_NEG_TWO_LOG2E)
    d = 1.0 / (1.0 + e)
    if mode == MXFP6_PROLOGUE_BIAS_GELU:
        out = xa * d
    else:
        inner_d = 2.0 * _GELU_BETA * (1.0 + 3.0 * _GELU_KAPPA * xa * xa)
        out = aux.float() * (d + xa * e * d * d * inner_d)
    return torch.where(inner < _GELU_SATURATE, torch.zeros_like(out), out).to(x.dtype)


def _prologue_operands(rows, cols, mode, dtype, *, with_bias=True, device="cuda:0"):
    x = torch.randn((rows, cols), dtype=dtype, device=device)
    aux = (
        torch.randn((rows, cols), dtype=dtype, device=device)
        if mode == MXFP6_PROLOGUE_BIAS_GELU_BACKWARD
        else None
    )
    bias = torch.randn((cols,), dtype=dtype, device=device) if with_bias else None
    return x, aux, bias


def _blobs_equal(got, ref, rows, cols):
    """Compare the four packed blobs on their data regions, ignoring any fifth output."""
    got_rp, got_rs, got_cp, got_cs = got[:4]
    ref_rp, ref_rs, ref_cp, ref_cs = ref[:4]

    def same(lhs, rhs, r, k, is_scale=False):
        return torch.equal(
            mxfp6_data_region(lhs, r, k, is_scale=is_scale),
            mxfp6_data_region(rhs, r, k, is_scale=is_scale),
        )

    return (
        same(got_rp, ref_rp, rows, cols)
        and same(got_rs, ref_rs, rows, cols, is_scale=True)
        and same(got_cp, ref_cp, cols, rows)
        and same(got_cs, ref_cs, cols, rows, is_scale=True)
    )


def _scale_blobs_equal(got, ref, rows, cols):
    """Compare only the two E8M0 scale blobs."""

    def same(lhs, rhs, r, k):
        return torch.equal(
            mxfp6_data_region(lhs, r, k, is_scale=True),
            mxfp6_data_region(rhs, r, k, is_scale=True),
        )

    return same(got[1], ref[1], rows, cols) and same(got[3], ref[3], cols, rows)


def _assert_blobs_agree(got, ref, rows, cols, tolerance=_PROLOGUE_CODE_TOLERANCE):
    """Both directions' codes agree to within ``tolerance``, and the scales exactly.

    ``ref`` may be packed from a larger, zero-extended operand than ``got``, in which case
    it has trailing K-tiles ``got`` has no room for and only the shared ones are compared.
    """
    for name, g_blob, r_blob, r, k in (
        ("row", got[0], ref[0], rows, cols),
        ("col", got[2], ref[2], cols, rows),
    ):
        g = mxfp6_data_region(g_blob, r, k)
        r_full = mxfp6_data_region(r_blob, r, k)
        differing = (g != r_full).sum().item() / g.numel()
        assert differing <= tolerance, f"{name} direction: {differing:.6%} of bytes differ"

    assert _scale_blobs_equal(got, ref, rows, cols), "E8M0 scales differ"


@pytest.mark.parametrize("rows,cols", [(256, 256), (512, 1024), (256, 3072)])
@pytest.mark.parametrize("mode", _PROLOGUE_MODES)
def test_fused_prologue_matches_eager_epilogue(rows, cols, mode):
    """Fusing the epilogue must not change the operand the GEMM sees, beyond the tanh.

    bf16 is the production dtype and the one the whole optimisation is justified on. Only the
    tanh is inexact here; every other part of the epilogue is reproduced bit for bit, and two
    details are what make that achievable -- both found by this test failing. The epilogue has
    to be evaluated in fp32 and rounded back to the input dtype before staging, and the
    bias-add has to be rounded *separately*, because in the graph being replaced it is a bf16
    tensor op whose result the activation then reads. Carrying the bias-add on in fp32 instead
    is a perfectly reasonable-looking choice that changes 21% of the resulting bf16 codes,
    four orders of magnitude above the tolerance here.
    """
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    x, aux, bias = _prologue_operands(rows, cols, mode, torch.bfloat16)

    with _packer("fused"):
        got = quantize_mxfp6_fused_dual(x, aux, bias, mode)
        ref = quantize_mxfp6_dual(mxfp6_apply_prologue(x, aux, bias, mode))

    _assert_blobs_agree(got, ref, rows, cols)


@pytest.mark.parametrize("mode", _PROLOGUE_MODES)
def test_fused_prologue_is_no_less_accurate_than_aten(mode):
    """The closed-form tanh must not be a worse tanh, only a different rounding of one.

    This is the test that licenses not being bit-exact with ATen. Both candidates are the
    epilogue rounded to bf16, so both are compared against the activation evaluated in fp64,
    and the claim is that neither is measurably closer to it. Were the closed form actually
    losing precision -- which is the real risk, since 1 + tanh(u) cancels to nothing in the
    left tail if it is formed via tanh -- it would show up here as a lower SNR, and the
    divergence bounds elsewhere in this file would silently be absorbing an error rather
    than a rounding difference.
    """
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    rows, cols = 1024, 2048
    x, aux, bias = _prologue_operands(rows, cols, mode, torch.bfloat16)

    xa = (x.double() + bias.double()).to(x.dtype).double()
    inner = _GELU_BETA * (xa + _GELU_KAPPA * xa**3)
    tanh = torch.tanh(inner)
    if mode == MXFP6_PROLOGUE_BIAS_GELU:
        truth = 0.5 * xa * (tanh + 1.0)
    else:
        inner_d = _GELU_BETA * (1.0 + 3.0 * _GELU_KAPPA * xa * xa)
        truth = aux.double() * (0.5 * (1.0 + tanh) + 0.5 * xa * (1.0 - tanh * tanh) * inner_d)

    def snr_db(candidate):
        noise = ((candidate.double() - truth) ** 2).mean()
        return float(10 * torch.log10((truth**2).mean() / noise))

    aten = snr_db(mxfp6_apply_prologue(x, aux, bias, mode))
    closed = snr_db(_closed_form_prologue(x, aux, bias, mode))

    # Both are bf16 roundings of the same value, so both sit at bf16's own noise floor.
    assert closed > aten - 0.5, f"closed form is less accurate: {closed:.2f} vs {aten:.2f} dB"


@pytest.mark.parametrize("rows,cols", [(256, 256), (512, 1024)])
@pytest.mark.parametrize("mode", _PROLOGUE_MODES)
def test_fused_prologue_fp16_matches_eager_epilogue_to_a_few_codes(rows, cols, mode):
    """fp16 gets a near-exactness bound rather than the bit-exact one bf16 gets.

    Not a weaker property by choice. ATen's own tanh-GELU kernels are only reproducible to
    1-2 fp32 ULP by any reimplementation, because the compiler is free to contract
    ``a + b * c`` into an FMA and does so differently in different translation units. bf16
    absorbs that entirely -- it has 7 mantissa bits, so a 1-ULP fp32 difference is invisible
    -- while fp16's 10 bits occasionally resolve it, and the packer's own bf16 rounding of
    the staged operand then lets a handful of those through to the codes.

    The bound is set two orders of magnitude below the signature of a real bug: applying the
    wrong epilogue shows up as ~5% of bytes differing, and the fp32 bias-add bug this file
    caught showed as 4.6%, against the ~0.002% seen here.
    """
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    x, aux, bias = _prologue_operands(rows, cols, mode, torch.float16)

    with _packer("fused"):
        got = quantize_mxfp6_fused_dual(x, aux, bias, mode)
        ref = quantize_mxfp6_dual(mxfp6_apply_prologue(x, aux, bias, mode))

    for name, g, r, r_rows, r_k in (
        ("row", got[0], ref[0], rows, cols),
        ("col", got[2], ref[2], cols, rows),
    ):
        gd = mxfp6_data_region(g, r_rows, r_k)
        rd = mxfp6_data_region(r, r_rows, r_k)
        differing = (gd != rd).sum().item() / gd.numel()
        assert differing < 5e-4, f"{name} direction: {differing:.5%} of bytes differ"

    # The scales are a per-32-block exponent, coarse enough that nothing should reach them.
    assert _scale_blobs_equal(got, ref, rows, cols)


@pytest.mark.parametrize("mode", _PROLOGUE_MODES)
def test_fused_prologue_without_bias(mode):
    """A null bias must skip the add rather than reading a zero vector that is not there."""
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    rows, cols = 256, 512
    x, aux, _ = _prologue_operands(rows, cols, mode, torch.bfloat16, with_bias=False)

    with _packer("fused"):
        got = quantize_mxfp6_fused_dual(x, aux, None, mode)
        ref = quantize_mxfp6_dual(mxfp6_apply_prologue(x, aux, None, mode))

    _assert_blobs_agree(got, ref, rows, cols)


@pytest.mark.parametrize("rows,cols", [(288, 256), (256, 320), (288, 320)])
@pytest.mark.parametrize("mode", _PROLOGUE_MODES)
def test_fused_prologue_pads_with_zeros(rows, cols, mode):
    """The padded region must still encode zero once a prologue is active.

    This is the sharpest test in the file. The grid covers the operand padded to 256 on
    both axes, and the dual pack contracts N one way and M the other, so padding on either
    axis lands on a contraction axis where a nonzero code adds a spurious term to the dot
    product. The staged value there is zero, and ``gelu(0 + bias)`` is *not* zero -- so a
    prologue applied to the zero-fill instead of being suppressed outside the logical
    extent corrupts the blob. Every shape the model actually uses is 256-aligned, which
    means production traffic cannot catch this and only these shapes can.

    Each axis is misaligned independently, because suppressing the row guard while leaving
    the column guard broken (or vice versa) would still pass a both-axes-misaligned case
    on one of the two directions.

    The tanh's rounding difference is bounded rather than excluded here, as everywhere else,
    but it cannot hide the bug this test is for: a prologue leaking into the padding turns
    whole rows or columns of a zero-encoded region into ``gelu(bias)``, which moves the
    padded blocks' E8M0 exponents off zero. Those are still compared exactly.
    """
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    def ceil256(v):
        return -(-v // 256) * 256

    dtype = torch.bfloat16
    x, aux, bias = _prologue_operands(rows, cols, mode, dtype)

    # Reference: materialise the epilogue over the logical extent only, then zero-extend.
    # Zero past the extent is the property under test, so it has to come from the
    # reference's construction rather than from re-running the prologue on padding.
    epilogue = mxfp6_apply_prologue(x, aux, bias, mode)
    extended = torch.zeros((ceil256(rows), ceil256(cols)), dtype=dtype, device=x.device)
    extended[:rows, :cols] = epilogue

    with _packer("fused"):
        got = quantize_mxfp6_fused_dual(x, aux, bias, mode)
        ref = quantize_mxfp6_dual(extended)

    er, ec = extended.shape

    # Zero-extending K gives the reference K-tiles the operand has no room for; the tiles
    # they share must agree and the surplus has no counterpart.
    for gi, ri, r, k, rr, rk in ((0, 0, rows, cols, er, ec), (2, 2, cols, rows, ec, er)):
        g = mxfp6_data_region(got[gi], r, k)
        ref_region = mxfp6_data_region(ref[ri], rr, rk)[:, : g.shape[1], :]
        differing = (g != ref_region).sum().item() / g.numel()
        assert differing <= _PROLOGUE_CODE_TOLERANCE, f"{differing:.6%} of bytes differ"

    for gi, ri, r, k, rr, rk in ((1, 1, rows, cols, er, ec), (3, 3, cols, rows, ec, er)):
        g = mxfp6_data_region(got[gi], r, k, is_scale=True)
        ref_region = mxfp6_data_region(ref[ri], rr, rk, is_scale=True)[:, : g.shape[1], :]
        assert torch.equal(g, ref_region), "padded region's E8M0 scales are not zero"


@pytest.mark.parametrize("rows,cols", [(256, 256), (512, 1024), (288, 320), (32768, 512)])
@pytest.mark.parametrize("mode", _PROLOGUE_MODES + [MXFP6_PROLOGUE_IDENTITY])
def test_fused_packer_col_sum_matches_eager_reduction(rows, cols, mode):
    """The bias-gradient side output must reduce to the same vector as the eager sum.

    Not bit-exact, unlike the blobs: this is a different summation of the same values, and
    deliberately a more accurate one (fp32 accumulation, tree-ordered over M-tiles) than a
    single bf16-input reduction. The tolerance is on the *reduction*, so it scales with the
    number of terms rather than being an elementwise epsilon; the 32768-row case is here
    because that is the token count Flux 12B actually uses and the one where a naive
    accumulation would visibly drift.

    Misaligned shapes matter here too: rows past M are staged as zero, so they must not
    contribute, and columns past N must not be written at all.
    """
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    x, aux, bias = _prologue_operands(rows, cols, mode, torch.bfloat16)

    with _packer("fused"):
        *_, partial = quantize_mxfp6_fused_dual(x, aux, bias, mode, want_col_sum=True)

    assert partial.shape == (mxfp6_col_sum_rows(rows), cols)
    assert partial.dtype == torch.float32

    got = partial.sum(0)
    ref = mxfp6_apply_prologue(x, aux, bias, mode).float().sum(0)
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3 * max(1.0, ref.abs().max().item()))


def test_fused_packer_col_sum_is_absent_unless_requested():
    """The fifth output is degenerate by default, so the common path allocates nothing."""
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    x, aux, bias = _prologue_operands(256, 512, MXFP6_PROLOGUE_BIAS_GELU, torch.bfloat16)
    with _packer("fused"):
        *_, partial = quantize_mxfp6_fused_dual(x, aux, bias, MXFP6_PROLOGUE_BIAS_GELU)
    assert partial.numel() == 0


def test_fused_prologue_identity_matches_plain_dual():
    """Identity mode must be the plain dual pack, so the fused entry point is a superset."""
    _skip_if_unsupported()
    _skip_without_fused_packer()

    rows, cols = 256, 512
    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")

    with _packer("fused"):
        got = quantize_mxfp6_fused_dual(x, None, None, MXFP6_PROLOGUE_IDENTITY)
        ref = quantize_mxfp6_dual(x)

    assert _blobs_equal(got, ref, rows, cols)


@pytest.mark.parametrize("mode", _PROLOGUE_MODES)
@pytest.mark.parametrize("want_col_sum", [False, True])
def test_fused_prologue_custom_op_fake_matches_real(mode, want_col_sum):
    """FakeTensor shapes must match the real op, or torch.compile traces wrong sizes.

    The partial buffer makes this more than a formality: its row count is a function of M
    that the header, the wrapper and the Python fake each compute separately, so a
    disagreement is a wrong-sized allocation rather than a compile error.
    """
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    rows, cols = 288, 512
    x, aux, bias = _prologue_operands(rows, cols, mode, torch.bfloat16)
    op = torch.ops.primus_turbo.quantize_mxfp6_fused_dual_impl

    real = op(x, aux, bias, mode, want_col_sum)
    with torch._subclasses.FakeTensorMode() as fake_mode:
        fake_x = fake_mode.from_tensor(x)
        fake_aux = None if aux is None else fake_mode.from_tensor(aux)
        fake_bias = fake_mode.from_tensor(bias)
        fake = op(fake_x, fake_aux, fake_bias, mode, want_col_sum)

    assert [t.shape for t in fake] == [t.shape for t in real]
    assert [t.dtype for t in fake] == [t.dtype for t in real]


def test_fused_prologue_rejects_inconsistent_arguments():
    """aux is required by exactly one mode, and the bias has to match the column count."""
    _skip_if_unsupported()
    _skip_without_fused_prologue()

    rows, cols = 256, 512
    dtype = torch.bfloat16
    x = torch.randn((rows, cols), dtype=dtype, device="cuda:0")
    aux = torch.randn((rows, cols), dtype=dtype, device="cuda:0")
    bias = torch.randn((cols,), dtype=dtype, device="cuda:0")

    # Matching the message as well as the type, so that a different failure reaching the
    # same line cannot pass for the check under test.
    with _packer("fused"):
        # Backward mode without the incoming gradient.
        with pytest.raises(RuntimeError, match="aux is required"):
            quantize_mxfp6_fused_dual(x, None, bias, MXFP6_PROLOGUE_BIAS_GELU_BACKWARD)
        # Forward mode handed a gradient it has no use for.
        with pytest.raises(RuntimeError, match="aux is required"):
            quantize_mxfp6_fused_dual(x, aux, bias, MXFP6_PROLOGUE_BIAS_GELU)
        # Bias sized for the wrong axis.
        with pytest.raises(RuntimeError, match="one element per column"):
            wrong = torch.randn((rows,), dtype=dtype, device="cuda:0")
            quantize_mxfp6_fused_dual(x, None, wrong, MXFP6_PROLOGUE_BIAS_GELU)


@pytest.mark.parametrize("axis,expect_row", [(1, True), (-1, True), (0, False), (-2, False)])
def test_quantize_mxfp6_custom_op_axis(axis, expect_row):
    """The registered op's axis argument must agree with the direct helpers."""
    _skip_if_unsupported()
    rows, cols = 512, 1024
    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")

    op_p, op_s = torch.ops.primus_turbo.quantize_mxfp6_impl(x, axis)
    ref_p, ref_s = (quantize_mxfp6_row if expect_row else quantize_mxfp6_col)(x)
    r, c = (rows, cols) if expect_row else (cols, rows)

    assert torch.equal(mxfp6_data_region(op_p, r, c), mxfp6_data_region(ref_p, r, c))
    assert torch.equal(
        mxfp6_data_region(op_s, r, c, is_scale=True),
        mxfp6_data_region(ref_s, r, c, is_scale=True),
    )


def test_quantize_mxfp6_custom_op_rejects_bad_axis():
    _skip_if_unsupported()
    x = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda:0")
    with pytest.raises(ValueError, match="axis must be one of"):
        torch.ops.primus_turbo.quantize_mxfp6_impl(x, 5)


def test_quantize_mxfp6_dual_fake_matches_real():
    """A wrong fake silently mis-allocates under torch.compile, so pin it."""
    _skip_if_unsupported()
    from torch._subclasses.fake_tensor import FakeTensorMode

    x = torch.randn((512, 1024), dtype=torch.bfloat16, device="cuda:0")
    real = torch.ops.primus_turbo.quantize_mxfp6_dual_impl(x)
    with FakeTensorMode():
        fake = torch.ops.primus_turbo.quantize_mxfp6_dual_impl(
            torch.empty((512, 1024), dtype=torch.bfloat16, device="cuda:0")
        )
    for f, r in zip(fake, real):
        assert f.shape == r.shape and f.dtype == r.dtype


def test_mxfp6_guard_tiles_are_never_read():
    """The blob's trailing guard tiles must not affect the result.

    This is what lets the packer leave them uninitialised. If a future packer or kernel
    revision starts reading them, this test is the tripwire.
    """
    _skip_if_unsupported()
    from primus_turbo.common.aiter_utils import get_aiter
    from primus_turbo.pytorch.core.low_precision import (
        MXFP6_GUARD_K_TILES,
        MXFP6_K_TILE_SIZE,
        MXFP6_PACKED_TILE_BYTES,
        MXFP6_SCALE_TILE_BYTES,
        MXFP6_TILE_SIZE,
    )

    m = n = 512
    k = 1024
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    a_p, a_s = quantize_mxfp6_row(a)
    b_p, b_s = quantize_mxfp6_row(b)

    gemm = get_aiter().gemm_a6w6
    baseline = gemm(a_p, b_p, a_s, b_s, m, n, k).clone()

    n_k_tiles = k // MXFP6_K_TILE_SIZE
    for blob, rows, tile_bytes in (
        (a_p, m, MXFP6_PACKED_TILE_BYTES),
        (b_p, n, MXFP6_PACKED_TILE_BYTES),
        (a_s, m, MXFP6_SCALE_TILE_BYTES),
        (b_s, n, MXFP6_SCALE_TILE_BYTES),
    ):
        view = blob.view(rows // MXFP6_TILE_SIZE, n_k_tiles + MXFP6_GUARD_K_TILES, tile_bytes)
        view[:, n_k_tiles:, :] = 0xFF

    assert torch.equal(gemm(a_p, b_p, a_s, b_s, m, n, k), baseline)


def test_gemm_fp6_rejects_unsupported():
    _skip_if_unsupported()
    a = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda:0")

    # The packed layout fixes the contraction axis, so no layout but NT exists.
    with pytest.raises(AssertionError):
        gemm_fp6(a, b, trans_a=True)
    with pytest.raises(AssertionError):
        gemm_fp6(a, b, trans_b=False)
    # The A6W6 asm only writes bf16.
    with pytest.raises(AssertionError):
        gemm_fp6(a, b, out_dtype=torch.float16)
    # No beta=1 epilogue, so wgrad cannot accumulate into main_grad.
    with pytest.raises(AssertionError):
        gemm_fp6(a, b, fuse_bgrad_accum_pattern="megatron")
    with pytest.raises(AssertionError):
        gemm_fp6(a, torch.randn((256, 256), dtype=torch.bfloat16, device="cuda:0"))


@pytest.mark.parametrize(
    "m,n,k",
    [
        (100, 256, 256),  # M
        (256, 100, 256),  # N
        (256, 256, 128),  # K: legal for one GEMM, illegal once backward runs
    ],
)
def test_gemm_fp6_requires_256_alignment_on_all_dims(m, n, k):
    """K must be a multiple of 256, not just 128.

    A single forward GEMM is happy with K % 128, but the backward GEMMs use K as an
    output dimension (dgrad produces M x K, wgrad produces N x K), which subjects it to
    the 256 rule. The check belongs at the autograd entry point so a bad shape fails at
    the call site instead of part-way through backward.
    """
    _skip_if_unsupported()
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    with pytest.raises(AssertionError, match="multiples of 256"):
        gemm_fp6(a, b)


def test_float6_quant_config_rejects_unsupported():
    # MXFP6 has no un-rotated / 2D-block / SR variants; each must fail loudly rather
    # than being silently ignored.
    with pytest.raises(AssertionError):
        Float6QuantConfig(use_gradient_sr=True)
    with pytest.raises(AssertionError):
        Float6QuantConfig(format=Format.E2M1_X2)
    with pytest.raises(AssertionError):
        Float6QuantConfig(block_size=16)
    with pytest.raises(AssertionError):
        Float6QuantConfig(granularity=ScalingGranularity.ROWWISE)


def test_gemm_fp6_alignment_gate():
    """can_handle rejects unaligned M/N/K.

    gemm_a6w6 itself pads and is correct on unaligned shapes, so this is a
    padding-waste guard rather than a correctness one -- but it should stay explicit.
    """
    _skip_if_unsupported()
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import GEMMFP6AITERBackend

    x = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda:0")
    p, s = quantize_mxfp6_row(x)
    g = ScalingGranularity.MX_BLOCKWISE
    args = (p, s, p, s)

    assert GEMMFP6AITERBackend.can_handle(*args, 256, 256, 512, torch.bfloat16, g)
    assert not GEMMFP6AITERBackend.can_handle(*args, 100, 256, 512, torch.bfloat16, g)
    assert not GEMMFP6AITERBackend.can_handle(*args, 256, 100, 512, torch.bfloat16, g)
    assert not GEMMFP6AITERBackend.can_handle(*args, 256, 256, 100, torch.bfloat16, g)
    assert not GEMMFP6AITERBackend.can_handle(*args, 256, 256, 512, torch.float16, g)
