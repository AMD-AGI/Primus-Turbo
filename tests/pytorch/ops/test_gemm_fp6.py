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
    Float6QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
    check_mxfp6_support,
    mxfp6_data_region,
    mxfp6_pack_sizes,
    quantize_mxfp6_col,
    quantize_mxfp6_dual,
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
        assert torch.equal(
            mxfp6_data_region(packed, rows, cols), mxfp6_data_region(packs[0][0], rows, cols)
        )
        assert torch.equal(
            mxfp6_data_region(scale, rows, cols, is_scale=True),
            mxfp6_data_region(packs[0][1], rows, cols, is_scale=True),
        )


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
