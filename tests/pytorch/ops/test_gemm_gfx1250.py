###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Correctness tests for the gfx1250 FlyDSL GEMM (bf16 / fp8 / mxfp8)."""

import pytest
import torch

gemm_mod = pytest.importorskip(
    "primus_turbo.flydsl.gemm.gemm_gfx1250_kernel",
    reason="FlyDSL is not installed",
)
gemm_gfx1250 = gemm_mod.gemm_gfx1250
MX_BLOCK = gemm_mod.MX_BLOCK
supported_dtypes = gemm_mod.supported_dtypes

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2

# The 8-bit kinds and the torch dtype each operand takes. gfx1250 accepts any
# e4m3/e5m2 pairing, and the mixed ones are the reason this table is explicit:
# the operand types are declared to the MMA atom in slot order (N side first),
# which is invisible while A and B share a dtype and silently wrong when they
# do not.
FP8_KIND_DTYPES = {
    "fp8": (E4M3, E4M3),
    "fp8_e5m2": (E5M2, E5M2),
    "fp8_e4m3_e5m2": (E4M3, E5M2),
    "fp8_e5m2_e4m3": (E5M2, E4M3),
    "mxfp8": (E4M3, E4M3),
    "mxfp8_e5m2": (E5M2, E5M2),
    "mxfp8_e4m3_e5m2": (E4M3, E5M2),
    "mxfp8_e5m2_e4m3": (E5M2, E4M3),
}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="requires a gfx1250 device",
)

# bf16 output rounding puts the ceiling near 55.6 dB for every operand type, so
# a threshold just under it catches a broken lane mapping without being flaky.
MIN_SNR_DB = 50.0

SHAPES = [
    (128, 128, 512),
    (256, 256, 1024),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (255, 128, 1024),  # ragged M: the TDM descriptor has to clamp it
    (1000, 256, 1024),
]


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float()
    got = got.float()
    noise = (ref - got).pow(2).mean().clamp_min(1e-30)
    return float(10 * torch.log10(ref.pow(2).mean() / noise))


def mx_quantize(x: torch.Tensor, dtype=E4M3):
    """Per-32-element-block E8M0 MX quantisation -> (fp8, scale bytes, dequant)."""
    rows, k = x.shape
    emax, fmax = (8, 448.0) if dtype == E4M3 else (15, 57344.0)
    xb = x.reshape(rows, k // MX_BLOCK, MX_BLOCK)
    amax = xb.abs().amax(-1).clamp_min(1e-30)
    exp = (torch.floor(torch.log2(amax)) - emax).clamp(-127, 127)
    scale = torch.pow(2.0, exp)
    q = (xb / scale[..., None]).clamp(-fmax, fmax).to(dtype)
    deq = (q.float() * scale[..., None]).reshape(rows, k)
    return q.reshape(rows, k).contiguous(), (exp + 127).to(torch.uint8).contiguous(), deq


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_16bit(shape, dtype):
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(n, k, device="cuda", dtype=dtype)
    out = gemm_gfx1250(a, b, out_dtype=dtype)
    assert snr_db(a.float() @ b.float().T, out) > MIN_SNR_DB


@pytest.mark.parametrize("shape", SHAPES)
def test_gemm_fp8(shape):
    m, n, k = shape
    a = (torch.randn(m, k, device="cuda") / 3).to(E4M3)
    b = (torch.randn(n, k, device="cuda") / 3).to(E4M3)
    out = gemm_gfx1250(a, b)
    assert snr_db(a.float() @ b.float().T, out) > MIN_SNR_DB


@pytest.mark.parametrize("kind", [k for k in FP8_KIND_DTYPES if not k.startswith("mx")])
def test_gemm_fp8_operand_pairings(kind):
    """Every e4m3/e5m2 pairing, including the mixed ones.

    A wrong operand-slot type order still scores ~55 dB when A and B share a
    dtype and collapses to ~5 dB when they differ, so the mixed cases are the
    only ones that catch it.
    """
    m, n, k = 512, 512, 1024
    da, db = FP8_KIND_DTYPES[kind]
    a = (torch.randn(m, k, device="cuda") / 3).to(da)
    b = (torch.randn(n, k, device="cuda") / 3).to(db)
    out = gemm_gfx1250(a, b, kind=kind)
    assert snr_db(a.float() @ b.float().T, out) > MIN_SNR_DB


@pytest.mark.parametrize("kind", [k for k in FP8_KIND_DTYPES if k.startswith("mx")])
def test_gemm_mxfp8_operand_pairings(kind):
    """As above for the scaled forms, where the same mistake produces NaN."""
    m, n, k = 512, 512, 1024
    da, db = FP8_KIND_DTYPES[kind]
    qa, sa, fa = mx_quantize(torch.randn(m, k, device="cuda"), da)
    qb, sb, fb = mx_quantize(torch.randn(n, k, device="cuda"), db)
    out = gemm_gfx1250(qa, qb, scale_a=sa, scale_b=sb, kind=kind)
    assert torch.isfinite(out.float()).all(), f"{kind} produced non-finite output"
    assert snr_db(fa @ fb.T, out) > MIN_SNR_DB


def test_every_supported_kind_is_reachable():
    """`_KINDS` and the constexpr id table must not drift apart.

    They were maintained by hand once, and six operand pairings shipped
    unreachable (KeyError on dispatch) because only one of the two was updated.
    """
    for kind in supported_dtypes():
        assert kind in gemm_mod._KIND_ID, f"{kind} has no constexpr id"
    assert len(gemm_mod._KIND_BY_ID) == len(supported_dtypes())


@pytest.mark.parametrize("shape", SHAPES)
def test_gemm_mxfp8(shape):
    m, n, k = shape
    qa, sa, da = mx_quantize(torch.randn(m, k, device="cuda"))
    qb, sb, db = mx_quantize(torch.randn(n, k, device="cuda"))
    out = gemm_gfx1250(qa, qb, scale_a=sa, scale_b=sb, kind="mxfp8")
    assert snr_db(da @ db.T, out) > MIN_SNR_DB


@pytest.mark.parametrize(
    "tile,warps,nb",
    [
        ((128, 128, 64), (2, 2), 2),
        ((128, 128, 64), (2, 2), 4),
        ((256, 256, 64), (4, 2), 2),
        ((128, 256, 64), (2, 4), 2),
        ((128, 128, 128), (2, 2), 2),
    ],
)
def test_gemm_bf16_tiles(tile, warps, nb):
    """The K pipeline has to stay correct across buffer depths and wave grids."""
    m, n, k = 512, 512, 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = gemm_gfx1250(a, b, tile=tile, m_warp=warps[0], n_warp=warps[1], num_buffers=nb)
    assert snr_db(a.float() @ b.float().T, out) > MIN_SNR_DB


def test_k_reduction_is_complete():
    """A missed or double-counted K tile shows up here as an exact miscount."""
    m = n = 128
    k = 1024
    a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(n, k, device="cuda", dtype=torch.bfloat16)
    for nb in (2, 3, 4):
        out = gemm_gfx1250(a, b, tile=(128, 128, 64), num_buffers=nb)
        assert torch.equal(out.float(), torch.full((m, n), float(k), device="cuda")), (
            f"num_buffers={nb} reduced {out[0, 0].item()} of {k}"
        )


# Real projection shapes (K, N) from the two models this kernel targets. M is
# the token count and is ragged by nature, so it is varied separately.
DSV3_SHAPES = [
    ("q_a_proj", 7168, 1536),
    ("q_b_proj", 1536, 24576),
    ("kv_a_proj", 7168, 576),  # N is not a multiple of any useful tile_n
    ("kv_b_proj", 512, 32768),
    ("o_proj", 16384, 7168),
    ("router", 7168, 48),  # N far narrower than a tile
    ("moe_gate_up", 7168, 2048),
    ("moe_down", 2048, 7168),
]
DSV4_FLASH_SHAPES = [
    ("kv_a_proj", 4096, 576),
    ("router", 4096, 48),
    ("dense_gate_up", 4096, 10944),
    ("dense_down", 10944, 4096),  # K is a multiple of 64 but not of 128
]


@pytest.mark.parametrize("proj,k,n", DSV3_SHAPES + DSV4_FLASH_SHAPES)
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_model_projection_shapes(proj, k, n, dtype):
    """The kernel has to cover the shapes the target models actually issue.

    Several of these are the reason ragged N and the 16x16x64 fp8 atom exist:
    kv_a_proj's N=576 divides no useful tile, router's N=48 is narrower than
    one, and DSV4 Flash's dense_down has K=10944, a multiple of 64 but not 128.
    """
    m = 512
    if dtype == "bf16":
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    else:
        a = (torch.randn(m, k, device="cuda") / 3).to(E4M3)
        b = (torch.randn(n, k, device="cuda") / 3).to(E4M3)
    out = gemm_gfx1250(a, b)
    assert out.shape == (m, n)
    assert snr_db(a.float() @ b.float().T, out) > MIN_SNR_DB


@pytest.mark.parametrize("n", [48, 129, 255, 576, 1000])
def test_ragged_n(n):
    """N need not divide the tile: the B load zero-fills and the C store drops."""
    m, k = 256, 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = gemm_gfx1250(a, b, tile=(128, 128, 64), num_buffers=2)
    assert out.shape == (m, n)
    assert snr_db(a.float() @ b.float().T, out) > MIN_SNR_DB


@pytest.mark.parametrize("k", [576, 1216, 10944])
def test_fp8_k64_atom(k):
    """K a multiple of 64 but not 128 has to route to the 16x16x64 fp8 atom."""
    assert k % 64 == 0 and k % 128 != 0, "test shape must force the narrow atom"
    m, n = 256, 256
    a = (torch.randn(m, k, device="cuda") / 3).to(E4M3)
    b = (torch.randn(n, k, device="cuda") / 3).to(E4M3)
    out = gemm_gfx1250(a, b)
    assert snr_db(a.float() @ b.float().T, out) > MIN_SNR_DB
