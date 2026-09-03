###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP6_GUARD_K_TILES,
    MXFP6_K_TILE_SIZE,
    MXFP6_PACKED_TILE_BYTES,
    MXFP6_PROLOGUE_BIAS_GELU,
    MXFP6_PROLOGUE_BIAS_GELU_BACKWARD,
    MXFP6_PROLOGUE_IDENTITY,
    MXFP6_SCALE_TILE_BYTES,
    MXFP6_TILE_SIZE,
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
    """Skip a test that needs the A6W6 kernels, which exist only on gfx950.

    Call this only when the test actually runs a kernel. CI has no gfx950 runner -- the
    PyTorch lane is gfx942, as it is for the MXFP4 and MXFP8 suites -- so a test that
    calls this is a test CI never executes, and every one of them that did not need to
    is coverage given away for nothing.

    A good deal of this file's surface is decided before any kernel is reached: the
    argument validation in ``gemm_fp6``, the blob validation in ``_validate_blobs``, the
    pack-size arithmetic, the eager prologue reference, and the aiter half of the
    capability check. Those are grouped at the end of this file and must stay ungated.
    """
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
    from primus_turbo.common.aiter_utils import get_aiter

    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")
    pack = quantize_mxfp6_row if direction == "row" else quantize_mxfp6_col
    # Packed rows/K are transposed for the column direction.
    r, k = (rows, cols) if direction == "row" else (cols, rows)

    got_p, got_s = pack(x)
    # AITER packs the row direction only, so the column reference has to go through the
    # materialised transpose that the fused packer exists to avoid.
    ref_p, ref_s = get_aiter().quant_mxfp6_gemm(x if direction == "row" else x.t().contiguous())

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

    def ceil256(v):
        return -(-v // 256) * 256

    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")
    extended = torch.zeros((ceil256(rows), ceil256(cols)), dtype=torch.bfloat16, device="cuda:0")
    extended[:rows, :cols] = x

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

    rows, cols = 288, 256
    values = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")

    packs = []
    for filler in (0.0, 3.5, -100.0):
        backing = torch.full((512, cols), filler, dtype=torch.bfloat16, device="cuda:0")
        backing[:rows] = values
        packs.append(quantize_mxfp6_row(backing[:rows]))

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


def _solve_x_for_gelu_inner(target):
    """Invert the monotonic negative branch of beta * (x + kappa*x^3)."""
    low, high = -16.0, 0.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        inner = _GELU_BETA * (midpoint + _GELU_KAPPA * midpoint**3)
        if inner < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _adjacent_bf16_values_around_gelu_cutoff(count=256):
    """The consecutive BF16 inputs straddling the kernel's cutoff."""
    root = torch.tensor(_solve_x_for_gelu_inner(_GELU_SATURATE), dtype=torch.bfloat16)
    center = int(root.view(torch.uint16))
    half = count // 2
    bits = torch.arange(center - half, center + count - half, dtype=torch.int32).to(torch.uint16)
    return bits.view(torch.bfloat16).float().sort().values.to(torch.bfloat16)


@pytest.mark.parametrize("mode", _PROLOGUE_MODES)
def test_fused_prologue_cutoff_matches_eager_for_adjacent_bf16_inputs(mode):
    """Exercise the compiled gfx950 cutoff on every adjacent BF16 input around it.

    The broad random prologue tests establish the ordinary rounding bound. This test
    isolates the explicit ``inner < -9.02`` branch: every representable BF16 input below
    it must pack exactly like eager ATen's zero, while the nearest input on the other side
    must remain non-zero. The latter prevents an accidentally inclusive or shifted branch
    from passing merely because the left tail is tiny.
    """
    _skip_if_unsupported()

    values = _adjacent_bf16_values_around_gelu_cutoff()
    inner = _GELU_BETA * (values.float() + _GELU_KAPPA * values.float() ** 3)
    below_values = values[inner < _GELU_SATURATE]
    at_or_above_values = values[inner >= _GELU_SATURATE]
    assert below_values.numel() and at_or_above_values.numel()

    # The packer needs whole 32-element blocks, and the fill value has to stay below the
    # cutoff too, so that a block's E8M0 scale cannot be set by a value under test.
    cols = 256
    rows = -(-below_values.numel() // 32) * 32
    x = torch.full((rows, cols), below_values[-1].item(), dtype=torch.bfloat16, device="cuda")
    x[: below_values.numel()] = below_values[:, None].to("cuda")
    bias = torch.zeros(cols, dtype=torch.bfloat16, device="cuda")
    aux = torch.ones_like(x) if mode == MXFP6_PROLOGUE_BIAS_GELU_BACKWARD else None
    eager = mxfp6_apply_prologue(x, aux, bias, mode)
    assert not torch.count_nonzero(eager)

    got = quantize_mxfp6_fused_dual(x, aux, bias, mode)
    ref = quantize_mxfp6_dual(eager)

    assert _blobs_equal(got, ref, rows, cols)
    assert not torch.count_nonzero(mxfp6_data_region(got[0], rows, cols))

    # The closest representable BF16 point on the non-saturating side, separately.
    nearest = at_or_above_values[:1, None].expand(32, cols).contiguous().to("cuda")
    nearest_aux = torch.ones_like(nearest) if mode == MXFP6_PROLOGUE_BIAS_GELU_BACKWARD else None
    nearest_eager = mxfp6_apply_prologue(nearest, nearest_aux, bias, mode)
    assert torch.count_nonzero(nearest_eager)
    nearest_got = quantize_mxfp6_fused_dual(nearest, nearest_aux, bias, mode)
    assert torch.count_nonzero(mxfp6_data_region(nearest_got[0], 32, cols))


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

    x, aux, bias = _prologue_operands(rows, cols, mode, torch.bfloat16)

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

    x, aux, bias = _prologue_operands(rows, cols, mode, torch.float16)

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

    rows, cols = 256, 512
    x, aux, _ = _prologue_operands(rows, cols, mode, torch.bfloat16, with_bias=False)

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

    x, aux, bias = _prologue_operands(rows, cols, mode, torch.bfloat16)

    *_, partial = quantize_mxfp6_fused_dual(x, aux, bias, mode, want_col_sum=True)

    assert partial.shape == (mxfp6_col_sum_rows(rows), cols)
    assert partial.dtype == torch.float32

    got = partial.sum(0)
    ref = mxfp6_apply_prologue(x, aux, bias, mode).float().sum(0)
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3 * max(1.0, ref.abs().max().item()))


def test_fused_packer_col_sum_is_absent_unless_requested():
    """The fifth output is degenerate by default, so the common path allocates nothing."""
    _skip_if_unsupported()

    x, aux, bias = _prologue_operands(256, 512, MXFP6_PROLOGUE_BIAS_GELU, torch.bfloat16)
    *_, partial = quantize_mxfp6_fused_dual(x, aux, bias, MXFP6_PROLOGUE_BIAS_GELU)
    assert partial.numel() == 0


def test_fused_prologue_identity_matches_plain_dual():
    """Identity mode must be the plain dual pack, so the fused entry point is a superset."""
    _skip_if_unsupported()

    rows, cols = 256, 512
    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda:0")

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

    rows, cols = 256, 512
    dtype = torch.bfloat16
    x = torch.randn((rows, cols), dtype=dtype, device="cuda:0")
    aux = torch.randn((rows, cols), dtype=dtype, device="cuda:0")
    bias = torch.randn((cols,), dtype=dtype, device="cuda:0")

    # Matching the message as well as the type, so that a different failure reaching the
    # same line cannot pass for the check under test.
    #
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


@pytest.mark.parametrize("m,n,k", [(256, 512, 256), (512, 256, 512)])
def test_gemm_fp6_torch_compile_backward(m, n, k):
    """Forward and backward must trace and run under torch.compile.

    The packers and the GEMM are opaque custom ops with hand-written fakes, so a wrong
    fake shows up here rather than in eager: inductor allocates from the fake and the
    kernel then writes somewhere else. Checked against a high-precision reference so a
    graph that traces but computes the wrong thing does not pass either.
    """
    _skip_if_unsupported()
    device = "cuda:0"
    dtype = torch.bfloat16

    a = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((n, k), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()

    c_ref = a_ref @ b_ref.T
    c_ref.backward(torch.ones_like(c_ref))

    compiled = torch.compile(gemm_fp6, backend="inductor")
    c = compiled(a, b, trans_a=False, trans_b=True, out_dtype=dtype)
    c.backward(torch.ones_like(c))

    for name, ref, got in (("C", c_ref, c), ("AGrad", a_ref.grad, a.grad), ("BGrad", b_ref.grad, b.grad)):
        assert got.shape == ref.shape, f"{name} shape {tuple(got.shape)} != {tuple(ref.shape)}"
        assert torch.isfinite(got).all(), f"{name} has non-finite entries"
        snr = compute_snr(ref, got)
        print(f"compiled {name}-SNR: {snr:.2f} dB")
        assert snr > SNR_THRESHOLD_DB, f"{name} snr too low: {snr:.2f} dB"


def test_mxfp6_packers_reject_fp32_input():
    """fp32 must be refused at the Python boundary.

    Only the bf16 and fp16 templates are instantiated, so an fp32 tensor previously got
    as far as the binding's own dtype check and failed there with a less specific error.
    """
    _skip_if_unsupported()
    x = torch.randn((256, 512), dtype=torch.float32, device="cuda:0")
    for pack in (quantize_mxfp6_row, quantize_mxfp6_col, quantize_mxfp6_dual, quantize_mxfp6_fused_dual):
        with pytest.raises(TypeError, match="bf16 or fp16"):
            pack(x)


def test_mxfp6_rejects_operands_on_different_devices():
    """Every operand of one call has to live on one device."""
    _skip_if_unsupported()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs at least two GPUs")

    x = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda:0")
    bias = torch.randn((512,), dtype=torch.bfloat16, device="cuda:1")
    with pytest.raises(ValueError, match="one device"):
        quantize_mxfp6_fused_dual(x, None, bias, MXFP6_PROLOGUE_BIAS_GELU)

    a = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda:1")
    with pytest.raises(ValueError, match="one device"):
        gemm_fp6(a, b)


def test_check_mxfp6_support_accepts_a_gfx950_device():
    """The check must answer about the operand's device, not the ambient one.

    The rejecting half of this contract needs no gfx950 and lives with the other
    hardware-independent tests, in ``test_check_mxfp6_support_rejects_a_cpu_device``.
    """
    _skip_if_unsupported()
    supported, _ = check_mxfp6_support(torch.device("cuda", 0))
    assert supported


def test_gemm_fp6_impl_wires_in_blob_validation():
    """The validator has to be reached through the real op, not merely exist.

    The enumeration of malformed blobs is in ``test_validate_blobs_rejects_malformed_operands``,
    which needs no hardware. What can only be checked here is that a genuine packed pair
    still passes and that a wrong K is caught on the way to the kernel rather than
    becoming an out-of-bounds read: the blobs carry no shape, so past this boundary the
    kernel derives strides from whatever dimensions it was handed.
    """
    _skip_if_unsupported()
    m = n = 256
    k = 512
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    a_p, a_s = quantize_mxfp6_row(a)
    b_p, b_s = quantize_mxfp6_row(b)
    g = ScalingGranularity.MX_BLOCKWISE.value

    def run(k=k):
        return torch.ops.primus_turbo.gemm_fp6_impl(a_p, a_s, b_p, b_s, m, n, k, torch.bfloat16, g)

    run()  # the well-formed call has to keep working

    with pytest.raises(ValueError, match="does not match"):
        run(k=256)


def test_low_level_gemm_accepts_the_k128_that_training_rejects():
    """The two alignment contracts differ on purpose.

    One forward GEMM is correct with K a multiple of 128, but ``gemm_fp6`` demands 256
    because its backward GEMMs use K as an output dimension. Pinning both halves keeps
    a future "cleanup" from unifying them.
    """
    _skip_if_unsupported()
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import GEMMFP6AITERBackend

    m = n = 256
    k = 128
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    a_p, a_s = quantize_mxfp6_row(a)
    b_p, b_s = quantize_mxfp6_row(b)
    g = ScalingGranularity.MX_BLOCKWISE

    assert GEMMFP6AITERBackend.can_handle(a_p, a_s, b_p, b_s, m, n, k, torch.bfloat16, g)
    out = torch.ops.primus_turbo.gemm_fp6_impl(a_p, a_s, b_p, b_s, m, n, k, torch.bfloat16, g.value)
    assert out.shape == (m, n) and torch.isfinite(out).all()

    with pytest.raises(ValueError, match="multiples of 256"):
        gemm_fp6(a, b)


def test_bias_falls_back_to_a_separate_add_on_an_older_aiter():
    """A bias must be honoured whether or not the installed aiter can fold it.

    This is what lets callers pass a bias unconditionally, and lets Primus carry no aiter
    version knowledge and no env var: the choice is made here, and both branches compute
    the same thing. They are not bitwise equal, and that difference is the point -- the
    fallback rounds to bf16 and then adds in bf16, where the epilogue adds into the fp32
    accumulator and rounds once -- so the check is that each branch matches *its own*
    reference exactly, rather than that the branches match each other.
    """
    _skip_if_unsupported()
    from primus_turbo.pytorch.kernels.gemm import gemm_fp6_impl as impl_mod
    from primus_turbo.pytorch.kernels.quantization import mxfp6_pack

    if not mxfp6_pack.aiter_has_bias_epilogue():
        pytest.skip("installed aiter has no bias epilogue, so there are not two paths to compare")

    m = n = k = 256
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    a_p, a_s = quantize_mxfp6_row(a)
    b_p, b_s = quantize_mxfp6_row(b)
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda:0")
    g = ScalingGranularity.MX_BLOCKWISE.value

    def run():
        return torch.ops.primus_turbo.gemm_fp6_impl(a_p, a_s, b_p, b_s, m, n, k, torch.bfloat16, g, bias)

    unbiased = torch.ops.primus_turbo.gemm_fp6_impl(a_p, a_s, b_p, b_s, m, n, k, torch.bfloat16, g, None)

    fused = run()
    assert not torch.equal(fused, unbiased), "the bias never reached the kernel"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(impl_mod, "aiter_has_bias_epilogue", lambda: False)
    try:
        fallback = run()
    finally:
        monkeypatch.undo()

    # The fallback is the pre-epilogue behaviour, so it must reproduce it exactly.
    assert torch.equal(fallback, unbiased + bias)

    # And the two branches must agree to within that one rounding step.
    def ulp(t):
        return torch.exp2(torch.floor(torch.log2(t.float().abs().clamp_min(1e-30))) - 7.0)

    diff = (fused.float() - fallback.float()).abs()
    assert bool((diff <= ulp(unbiased) + ulp(fallback)).all())


# 1/sqrt(32) rounded to bf16, which is what the packer's Hadamard multiplies by. Exact
# in binary (181 * 2**-10), so the model below stays exact too.
_HADAMARD32_NORM = 0.1767578125


def _e2m3_levels():
    """Every non-negative E2M3 value paired with its encoding, ascending.

    Written from the format rather than from the kernel: 1 sign, 2 exponent bits with
    bias 1, 3 mantissa bits, no inf and no NaN. Exponent field 0 is subnormal with an
    implicit leading zero, so the levels run 0, 1/8, ... 7/8 and then 1.0 to 7.5.
    """
    levels = [
        (m / 8.0 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 1), (e << 3) | m)
        for e in range(4)
        for m in range(8)
    ]
    return sorted(levels)


def _round_to_e2m3(x):
    """Round-to-nearest-even against the E2M3 level set, saturating at 7.5."""
    levels = _e2m3_levels()
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x)
    if ax >= levels[-1][0]:
        return sign * levels[-1][0], levels[-1][1]
    for (lo_v, lo_c), (hi_v, hi_c) in zip(levels, levels[1:]):
        if lo_v <= ax <= hi_v:
            below, above = ax - lo_v, hi_v - ax
            if below < above:
                value, code = lo_v, lo_c
            elif above < below:
                value, code = hi_v, hi_c
            else:
                value, code = (lo_v, lo_c) if lo_c % 2 == 0 else (hi_v, hi_c)
            return sign * value, code
    raise AssertionError(f"{x} is not bracketed by the E2M3 level set")


def _pack_constant_group(a):
    """Model the packer for one 32-group whose inputs are all ``a``.

    The butterfly collapses a constant group to ``[32a, 0 x 31]``, so the group has a
    single nonzero which is necessarily its own amax. The MX scale is
    ``2 ** (floor(log2(amax)) - 2)``, so ``value / scale`` always lands in ``[4, 8)``.

    Returns the dequantized value, its code, the ratio that was rounded, and the scale.
    """
    import math

    value = 32.0 * a * _HADAMARD32_NORM
    if value == 0.0:
        return 0.0, 0, 0.0, 1.0
    _, exponent = math.frexp(abs(value))  # abs(value) = mantissa * 2**exponent
    scale = 2.0 ** (exponent - 1 - 2)
    ratio = value / scale
    quantized, code = _round_to_e2m3(ratio)
    return quantized * scale, code, ratio, scale


def test_production_packer_rounds_e2m3_to_nearest_even():
    """Pin the packer's E2M3 rounding against an independently written model.

    Reads the packed codes back through the GEMM rather than by decoding the blob.
    A row of a constant ``a`` collapses to one nonzero per 32-group at index 0, so
    against a weight whose rows are all ones -- which collapses the same way -- the
    dot product over one group is just the two dequantized values multiplied. With
    K/32 groups the whole output is ``(K / 32) * q_a * q_b``, and every factor is an
    E2M3 value times a power of two, so the bf16 result is exact rather than rounded.

    Two honest limits. The single nonzero is always its own amax, so ``value / scale``
    only ever lands in ``[4, 8)``: this covers the top octave, at step 0.5, not the
    subnormals. And an exact tie is unreachable through this path at all, because every
    post-Hadamard value carries the factor ``181 * 2**-10`` from the bf16-rounded
    ``1/sqrt(32)`` while every midpoint is dyadic. The test therefore drives the closest
    approach to each midpoint that a bf16 input can produce, and records the distance.
    """
    _skip_if_unsupported()
    m = n = k = 256
    groups = k // 32
    device = "cuda:0"

    midpoints = [4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25]
    candidates = torch.arange(1.0, 2.0, 1.0 / 128, dtype=torch.float32).to(torch.bfloat16).float().tolist()

    # For each midpoint, the bf16 input whose ratio comes closest to it from either side.
    probes = []
    for mid in midpoints:
        for side in (-1, 1):
            best = min(
                (c for c in candidates if side * (_pack_constant_group(c)[2] - mid) > 0),
                key=lambda c: abs(_pack_constant_group(c)[2] - mid),
            )
            probes.append((best, mid))
    # Plus saturation. The scale tracks the amax, so the ratio stays in [4, 8) and the
    # sliver above 7.5 is the only way to overflow E2M3: it has to clamp, not wrap.
    saturating = [c for c in candidates if _pack_constant_group(c)[2] > 7.5]
    assert saturating, "no bf16 input in [1, 2) drives the ratio past the E2M3 maximum"
    probes.append((saturating[0], None))

    a = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
    for row, (value, _) in enumerate(probes):
        a[row].fill_(value)
    b = torch.ones((n, k), dtype=torch.bfloat16, device=device)

    a_p, a_s = quantize_mxfp6_row(a)
    b_p, b_s = quantize_mxfp6_row(b)
    out = torch.ops.primus_turbo.gemm_fp6_impl(
        a_p, a_s, b_p, b_s, m, n, k, torch.bfloat16, ScalingGranularity.MX_BLOCKWISE.value
    )

    q_b, _, _, _ = _pack_constant_group(1.0)
    for row, (value, mid) in enumerate(probes):
        q_a, code, ratio, scale = _pack_constant_group(value)
        expected = groups * q_a * q_b
        got = out[row, 0].item()
        distance = "n/a (saturating)" if mid is None else f"{abs(ratio - mid):.6g} from {mid}"
        print(
            f"input={value!r} scale={scale} ratio={ratio:.9g} -> code={code} value={q_a / scale} ({distance})"
        )
        assert got == expected, (
            f"input {value!r}: packer produced {got}, model expects {expected} "
            f"(ratio {ratio:.9g}, code {code}, scale {scale})"
        )

    # The saturating probe must land on the top code, not wrap to a small one.
    _, sat_code, sat_ratio, _ = _pack_constant_group(saturating[0])
    assert sat_ratio > 7.5 and sat_code == _e2m3_levels()[-1][1]


def test_gemm_fp6_impl_fake_matches_real():
    """A wrong GEMM fake mis-allocates the output under torch.compile."""
    _skip_if_unsupported()
    from torch._subclasses.fake_tensor import FakeTensorMode

    m, n, k = 256, 512, 256
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    a_p, a_s = quantize_mxfp6_row(a)
    b_p, b_s = quantize_mxfp6_row(b)
    g = ScalingGranularity.MX_BLOCKWISE.value

    real = torch.ops.primus_turbo.gemm_fp6_impl(a_p, a_s, b_p, b_s, m, n, k, torch.bfloat16, g)
    with FakeTensorMode():
        blob = lambda t: torch.empty(t.shape, dtype=t.dtype, device=t.device)  # noqa: E731
        fake = torch.ops.primus_turbo.gemm_fp6_impl(
            blob(a_p), blob(a_s), blob(b_p), blob(b_s), m, n, k, torch.bfloat16, g
        )
    assert fake.shape == real.shape and fake.dtype == real.dtype and fake.device == real.device


###############################################################################
# Hardware-independent contracts.
#
# Everything below runs wherever torch imports: no gfx950, no aiter, no GPU. That makes
# this the only part of the file CI executes today, since the PyTorch lane is gfx942.
# Nothing here may call _skip_if_unsupported() -- see its docstring -- and nothing here
# may reach a kernel, which is what keeps that true.
###############################################################################


def test_gemm_fp6_rejects_unsupported():
    """Every argument contract of the wrapper, on CPU.

    All of these are decided before ``FP6GemmMXFunction.apply``, so the hardware never
    comes into it. Should someone move one of these checks into forward, past the
    capability check, this stops raising ValueError and starts raising RuntimeError,
    which is the failure worth having.
    """
    a = torch.randn((256, 512), dtype=torch.bfloat16)
    b = torch.randn((256, 512), dtype=torch.bfloat16)

    # The packed layout fixes the contraction axis, so no layout but NT exists.
    with pytest.raises(ValueError, match="NT layout"):
        gemm_fp6(a, b, trans_a=True)
    with pytest.raises(ValueError, match="NT layout"):
        gemm_fp6(a, b, trans_b=False)
    # The A6W6 asm only writes bf16.
    with pytest.raises(ValueError, match="bf16 output"):
        gemm_fp6(a, b, out_dtype=torch.float16)
    # No beta=1 epilogue, so wgrad cannot accumulate into main_grad.
    with pytest.raises(NotImplementedError, match="fuse_bgrad_accum_pattern"):
        gemm_fp6(a, b, fuse_bgrad_accum_pattern="megatron")
    with pytest.raises(ValueError, match="K mismatch"):
        gemm_fp6(a, torch.randn((256, 256), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="2D"):
        gemm_fp6(a.unsqueeze(0), b)
    with pytest.raises(TypeError, match="one dtype"):
        gemm_fp6(a, b.to(torch.float16))


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
    a = torch.randn((m, k), dtype=torch.bfloat16)
    b = torch.randn((n, k), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiples of 256"):
        gemm_fp6(a, b)


def test_float6_quant_config_rejects_unsupported():
    # MXFP6 has no un-rotated / 2D-block / SR variants; each must fail loudly rather
    # than being silently ignored.
    with pytest.raises(NotImplementedError, match="use_gradient_sr"):
        Float6QuantConfig(use_gradient_sr=True)
    with pytest.raises(ValueError, match="E2M3"):
        Float6QuantConfig(format=Format.E2M1_X2)
    with pytest.raises(ValueError, match="block_size"):
        Float6QuantConfig(block_size=16)
    with pytest.raises(ValueError, match="MX_BLOCKWISE"):
        Float6QuantConfig(granularity=ScalingGranularity.ROWWISE)


def test_float6_quant_config_is_hashable():
    """Hashable, like Float4QuantConfig and Float8QuantConfig.

    Nothing in the MXFP6 path hashes it -- the ops take blobs, not a config -- but both
    siblings are hashable, and one that silently is not becomes a puzzle the first time a
    config lands in a set or a cache key.
    """
    assert hash(Float6QuantConfig()) == hash(Float6QuantConfig())
    assert len({Float6QuantConfig(), Float6QuantConfig()}) == 1


def test_check_mxfp6_support_rejects_a_cpu_device():
    """A CPU operand is refused for being on the wrong device, whatever the host is.

    The accepting half needs gfx950 and lives in
    ``test_check_mxfp6_support_accepts_a_gfx950_device``.
    """
    supported, reason = check_mxfp6_support(torch.device("cpu"))
    assert not supported and "ROCm device" in reason


def test_check_aiter_a6w6_survives_a_missing_aiter(monkeypatch):
    """A capability predicate that raises cannot be used to decide anything.

    ``get_aiter`` raises ImportError when aiter is absent, which used to escape from
    here and turn a graceful "MXFP6 unavailable" into a crash at import-adjacent time.
    """
    from primus_turbo.pytorch.kernels.quantization import mxfp6_pack

    def no_aiter():
        raise ImportError("aiter is not installed")

    monkeypatch.setattr(mxfp6_pack, "get_aiter", no_aiter)
    supported, reason = mxfp6_pack._check_aiter_a6w6()
    assert not supported
    assert "aiter is not installed" in reason


def test_check_aiter_a6w6_names_the_missing_symbols_and_the_minimum_commit(monkeypatch):
    """An aiter too old to have A6W6 has to say so, and say what would fix it.

    Naming the symbols is what distinguishes "your aiter predates A6W6" from the several
    other ways MXFP6 can be unavailable, and the commit is the only actionable form of
    the requirement: the pinned tag does not contain it, so no version tells the user
    what to install.
    """
    from primus_turbo.pytorch.kernels.quantization import mxfp6_pack

    monkeypatch.setattr(mxfp6_pack, "get_aiter", lambda: object())
    supported, reason = mxfp6_pack._check_aiter_a6w6()
    assert not supported
    for attr in ("quant_mxfp6_gemm", "gemm_a6w6", "mxfp6_gemm_pack_size"):
        assert attr in reason
    assert mxfp6_pack.MXFP6_MIN_AITER_COMMIT in reason


def test_check_aiter_a6w6_accepts_an_aiter_with_every_symbol(monkeypatch):
    """The probe must pass on a module that has the three names and nothing else.

    Without this the two rejection tests above are satisfied by a predicate that never
    returns True at all.
    """
    from primus_turbo.pytorch.kernels.quantization import mxfp6_pack

    class FakeAiter:
        quant_mxfp6_gemm = gemm_a6w6 = mxfp6_gemm_pack_size = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(mxfp6_pack, "get_aiter", lambda: FakeAiter())
    assert mxfp6_pack._check_aiter_a6w6() == (True, "")


@pytest.fixture
def bias_epilogue_probe():
    """Hand back the bias-epilogue probe with its cache cleared on both sides.

    The probe is cached because the forward path calls it once per GEMM, so a test that
    fakes an aiter would otherwise read a real answer cached by an earlier test -- or,
    worse, leave a fake answer cached for a later one.
    """
    from primus_turbo.pytorch.kernels.quantization import mxfp6_pack

    mxfp6_pack.aiter_has_bias_epilogue.cache_clear()
    yield mxfp6_pack
    mxfp6_pack.aiter_has_bias_epilogue.cache_clear()


def test_bias_epilogue_probe_reads_the_signature_not_the_symbol(bias_epilogue_probe):
    """The capability is a *parameter*, so the probe cannot be a ``hasattr`` check.

    ``gemm_a6w6`` exists either way -- what an older aiter lacks is the ``bias`` argument
    on it -- which is the whole reason this probe differs in kind from the symbol probe
    above. A fake with the symbol but not the parameter is exactly the case that a
    ``hasattr`` implementation would get wrong, and it is the case that actually ships.
    """
    mxfp6_pack = bias_epilogue_probe

    class OldAiter:
        @staticmethod
        def gemm_a6w6(a, b, a_scale, b_scale, m, n, k):
            return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mxfp6_pack, "get_aiter", lambda: OldAiter())
    assert mxfp6_pack.aiter_has_bias_epilogue() is False
    monkeypatch.undo()

    class NewAiter:
        @staticmethod
        def gemm_a6w6(a, b, a_scale, b_scale, m, n, k, bias=None):
            return None

    mxfp6_pack.aiter_has_bias_epilogue.cache_clear()
    monkeypatch.setattr(mxfp6_pack, "get_aiter", lambda: NewAiter())
    assert mxfp6_pack.aiter_has_bias_epilogue() is True
    monkeypatch.undo()


@pytest.mark.parametrize(
    "fake_aiter",
    [
        pytest.param(ImportError, id="aiter-absent"),
        pytest.param(object, id="no-gemm-a6w6"),
        pytest.param(print, id="signature-not-introspectable"),
    ],
)
def test_bias_epilogue_probe_answers_no_rather_than_raising(bias_epilogue_probe, fake_aiter, monkeypatch):
    """Every way of failing to answer must come back False.

    False is the safe answer: it costs one extra pass over the output and computes the
    same thing. Raising, or answering True on a module the probe could not read, turns a
    missing optimisation into a broken training run.
    """
    mxfp6_pack = bias_epilogue_probe

    if fake_aiter is ImportError:

        def get_aiter():
            raise ImportError("aiter is not installed")

    elif fake_aiter is object:
        get_aiter = object
    else:
        # A builtin has no introspectable signature, which stands in for aiter exposing
        # the entry point from a C extension.
        get_aiter = lambda: type("CAiter", (), {"gemm_a6w6": print})  # noqa: E731

    monkeypatch.setattr(mxfp6_pack, "get_aiter", get_aiter)
    assert mxfp6_pack.aiter_has_bias_epilogue() is False


def test_bias_epilogue_min_commit_is_a_full_sha():
    """The constant is only useful if it can be fed to pip, which needs a full sha.

    Same reason ``MXFP6_MIN_AITER_COMMIT`` is full-length: an abbreviated sha is not
    resolvable by ``pip install git+...@`` and this is the only actionable form of the
    requirement, since no version string contains it.
    """
    from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
        MXFP6_BIAS_EPILOGUE_MIN_AITER_COMMIT as sha,
    )

    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


@pytest.mark.parametrize(
    "rows,k,row_tiles,k_tiles",
    [
        (256, 128, 1, 1),
        (256, 512, 1, 4),
        (512, 512, 2, 4),
        (100, 100, 1, 1),  # both padded up to one whole tile
        (257, 129, 2, 2),  # one element past a tile in each direction
    ],
)
def test_mxfp6_pack_sizes_include_the_guard_tiles(rows, k, row_tiles, k_tiles):
    """Blob sizes are row-tiles x (k-tiles + guard) x tile-bytes, with both dims padded.

    The guard tiles are the part worth pinning. Their contents are never read, but the
    assembly derives its row-tile stride from ``k/128 + 2``, so a blob sized without them
    leaves every stride wrong -- a silent wrong-answer bug rather than a crash. The
    equivalent test against aiter's own sizer needs aiter; this one pins the arithmetic
    itself, which is what the validator in ``_validate_blobs`` computes from.
    """
    tiles = row_tiles * (k_tiles + MXFP6_GUARD_K_TILES)
    assert mxfp6_pack_sizes(rows, k) == (
        tiles * MXFP6_PACKED_TILE_BYTES,
        tiles * MXFP6_SCALE_TILE_BYTES,
    )


def test_mxfp6_data_region_drops_the_guard_tiles():
    """The comparison view must exclude exactly the uninitialised trailing tiles.

    Bit-exactness assertions go through this. If it returned the whole blob they would
    compare the guard tiles too, which the packers never write, so two calls on identical
    input would disagree at random.
    """
    rows, k = 512, 512
    operand_bytes, scale_bytes = mxfp6_pack_sizes(rows, k)
    n_row_tiles, n_k_tiles = rows // MXFP6_TILE_SIZE, k // MXFP6_K_TILE_SIZE

    operand = torch.zeros(operand_bytes, dtype=torch.uint8)
    region = mxfp6_data_region(operand, rows, k)
    assert region.shape == (n_row_tiles, n_k_tiles, MXFP6_PACKED_TILE_BYTES)

    scale = torch.zeros(scale_bytes, dtype=torch.uint8)
    scale_region = mxfp6_data_region(scale, rows, k, is_scale=True)
    assert scale_region.shape == (n_row_tiles, n_k_tiles, MXFP6_SCALE_TILE_BYTES)

    # Writing every guard byte must leave the data region untouched.
    operand.view(n_row_tiles, n_k_tiles + MXFP6_GUARD_K_TILES, MXFP6_PACKED_TILE_BYTES)[:, n_k_tiles:, :] = (
        0xFF
    )
    assert not mxfp6_data_region(operand, rows, k).any()


@pytest.mark.parametrize(
    "m,expected",
    [
        (1, 4),  # padded up to one 256-row pack tile, which is four col-sum tiles
        (256, 4),
        (257, 8),  # one row past the pack tile costs a whole further tile
        (16384, 256),
        (16385, 260),
    ],
)
def test_mxfp6_col_sum_rows_counts_one_partial_per_64_row_tile(m, expected):
    """Rows of the bias-gradient partial buffer: m padded to 256-row pack tiles, over 64.

    Must agree with ``mxfp6_col_sum_rows`` in quantization.h. It is the padding that
    makes this worth pinning -- the count follows the launch grid, not the logical M, so
    the buffer for an unaligned M is larger than the shape alone suggests and a version
    that reasoned from m directly would under-allocate it.
    """
    assert mxfp6_col_sum_rows(m) == expected


def test_mxfp6_apply_prologue_identity_returns_the_input_untouched():
    """Identity must not copy, or the fused path is being compared against a different tensor."""
    x = torch.randn((8, 16))
    assert mxfp6_apply_prologue(x, None, None, MXFP6_PROLOGUE_IDENTITY) is x


def test_mxfp6_apply_prologue_broadcasts_bias_along_the_last_axis():
    x = torch.randn((8, 16))
    bias = torch.randn((16,))
    got = mxfp6_apply_prologue(x, None, bias, MXFP6_PROLOGUE_BIAS_GELU)
    torch.testing.assert_close(got, torch.nn.functional.gelu(x + bias, approximate="tanh"))


def test_mxfp6_apply_prologue_bias_is_optional():
    x = torch.randn((8, 16))
    got = mxfp6_apply_prologue(x, None, None, MXFP6_PROLOGUE_BIAS_GELU)
    torch.testing.assert_close(got, torch.nn.functional.gelu(x, approximate="tanh"))


def test_mxfp6_apply_prologue_backward_differentiates_the_forward():
    """The backward mode must be the derivative of the forward one it is paired with.

    Checked against autograd rather than against ``aten.gelu_backward``, which is what
    the reference itself calls: comparing it to its own implementation would pass however
    wrong the pairing of modes was.
    """
    x = torch.randn((8, 16), dtype=torch.float64, requires_grad=True)
    bias = torch.randn((16,), dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn((8, 16), dtype=torch.float64)

    mxfp6_apply_prologue(x, None, bias, MXFP6_PROLOGUE_BIAS_GELU).backward(grad_out)
    got = mxfp6_apply_prologue(x.detach(), grad_out, bias.detach(), MXFP6_PROLOGUE_BIAS_GELU_BACKWARD)
    torch.testing.assert_close(got, x.grad)


def test_mxfp6_apply_prologue_rejects_a_backward_without_its_incoming_gradient():
    x = torch.randn((8, 16))
    with pytest.raises(ValueError, match="aux"):
        mxfp6_apply_prologue(x, None, None, MXFP6_PROLOGUE_BIAS_GELU_BACKWARD)


def test_mxfp6_apply_prologue_rejects_an_unknown_mode():
    """An unrecognised mode must not fall through to the identity.

    The modes are plain ints crossing into the kernel, so a caller that passes a stale
    or mistyped one has to hear about it rather than silently get no epilogue.
    """
    x = torch.randn((8, 16))
    with pytest.raises(ValueError, match="unknown MXFP6 prologue mode"):
        mxfp6_apply_prologue(x, None, None, 99)


def test_validate_blobs_rejects_malformed_operands():
    """Every malformed-blob case, against fabricated blobs on CPU.

    The blobs are opaque byte streams carrying neither shape nor dtype, so this is the
    last point at which a wrong M/N/K is detectable at all: past it the kernel derives
    strides from whatever dimensions it was handed and reads off the end of a blob that
    is too short. None of that reasoning involves the hardware, and the sizes come from
    ``mxfp6_pack_sizes``, so the cases can be built without a packer.
    """
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import _validate_blobs

    m = n = 256
    k = 512
    (a_bytes, a_scale_bytes) = mxfp6_pack_sizes(m, k)
    (b_bytes, b_scale_bytes) = mxfp6_pack_sizes(n, k)
    a = torch.zeros(a_bytes, dtype=torch.uint8)
    a_scale = torch.zeros(a_scale_bytes, dtype=torch.uint8)
    b = torch.zeros(b_bytes, dtype=torch.uint8)
    b_scale = torch.zeros(b_scale_bytes, dtype=torch.uint8)

    def run(a=a, a_scale=a_scale, b=b, b_scale=b_scale, m=m, n=n, k=k):
        _validate_blobs(a, a_scale, b, b_scale, m, n, k)

    run()  # the well-formed set has to keep passing

    with pytest.raises(ValueError, match="does not match"):
        run(k=256)
    with pytest.raises(ValueError, match="positive dimensions"):
        run(m=0)
    with pytest.raises(ValueError, match="positive dimensions"):
        run(n=-1)
    with pytest.raises(TypeError, match="uint8"):
        run(a=a.view(torch.int8))
    with pytest.raises(ValueError, match="1-D"):
        run(a=a.view(2, -1))
    with pytest.raises(ValueError, match="contiguous"):
        run(a=a[::2])
    # The scale blobs are checked as strictly as the operands.
    with pytest.raises(TypeError, match="uint8"):
        run(b_scale=b_scale.view(torch.int8))
    with pytest.raises(ValueError, match="does not match"):
        run(b_scale=b_scale[:-1].contiguous())
