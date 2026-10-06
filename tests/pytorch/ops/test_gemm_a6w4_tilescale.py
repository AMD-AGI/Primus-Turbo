"""A6W4 on the tilescale layout: the K128-blocked MXFP4 weight pack (``with_ts4_row``) and ``gemm_fp6_impl(...,
weight_is_fp4=True, a6w4_ts=True)``.

The pack is checked against the plain-row FlyDSL FP4 pack of the same weight (same codes, re-blocked; same scales at
the 256 tile without interleave) and, for its backward column, against the weight pack the A4W4 backward uses. The
GEMM is checked against the bf16 product and, bitwise, against A6W6 on the FP6 re-encoding of the FP4 weight (every
E2M1 value is an E2M3 value and the MFMA treats them alike).
"""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.core.utils import is_gfx950_device

aiter = pytest.importorskip("aiter")
TS = pytest.importorskip("aiter.ops.tilescale")

from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import (  # noqa: E402
    a6w4_ts_available,
    a6w6_fly_available,
    gemm_fp6_impl,
)
from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx950_device(torch.device("cuda")), reason="A6W4 tilescale needs gfx950"
)
G = ScalingGranularity.MX_BLOCKWISE.value


def _k128(codes, rows, k):
    return codes.view(torch.uint8).reshape(rows // 16, 16, k // 128, 64).permute(0, 2, 1, 3).reshape(rows, k // 2)


@pytest.mark.parametrize("n,k", [(512, 1024), (768, 3072), (3072, 12288)])
def test_ts4_row_pack_matches_fly_rows(n, k):
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    c_ts, s_ts = MX.quantize_mx(w, 1, MX.with_ts4_row(0))[:2]
    c_fly, s_fly = MX.quantize_mx(w, 1, MX.fly_fmt(row=(True, 4, 0)))[:2]
    assert torch.equal(c_ts.view(torch.uint8).reshape(n, k // 2), _k128(c_fly, n, k))
    assert torch.equal(s_ts.view(torch.uint8).reshape(-1), s_fly.view(torch.uint8).reshape(-1))


@pytest.mark.parametrize("col_sr", [False, True])
def test_ts4_dual_keeps_backward_column(col_sr):
    n, k, m = 768, 3072, 1024
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    base = MX.fp4_options(MX.fly_fmt(col=MX.fly_b_params(m, k, n)), col_sr=col_sr)
    torch.manual_seed(0)
    _, _, cc_ref, cs_ref = MX.quantize_mx_dual(w, base)
    torch.manual_seed(0)
    rc, rs, cc, cs = MX.quantize_mx_dual(w, MX.with_ts4_row(base))
    if not col_sr:  # SR draws a fresh seed per launch: compare the column only when deterministic
        assert torch.equal(cc, cc_ref) and torch.equal(cs, cs_ref)
    c_fly, s_fly = MX.quantize_mx(w, 1, MX.fly_fmt(row=(True, 4, 0)))[:2]
    assert torch.equal(rc.view(torch.uint8).reshape(n, k // 2), _k128(c_fly, n, k))


SHAPES = [(512, 512, 1024, False), (512, 512, 1024, True), (1024, 768, 1536, True), (16384, 9216, 3072, True),
          (8192, 3072, 12288, False)]


@pytest.mark.parametrize("m,n,k,has_bias", SHAPES)
def test_gemm_a6w4_ts(m, n, k, has_bias):
    if not a6w4_ts_available(m, n, k, has_bias):
        pytest.skip(f"no A6W4 tilescale kernel for {m}x{n}x{k} bias={has_bias}")
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g) if has_bias else None
    a, a_s = MX.quantize_mx(x, 1, MX.fly6_fmt(False))[:2]
    b, b_s = MX.quantize_mx(w, 1, MX.with_ts4_row(0))[:2]
    out = gemm_fp6_impl(a, a_s, b, b_s, m, n, k, torch.bfloat16, G, bias, True, False, 0, False, True)
    ref = x.float() @ w.float().t() + (bias.float() if bias is not None else 0)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.99, cos
    # bitwise against A6W6 on fp6(W4): FP4 codes -> FP6 codes, same scales, the A6W6 tilescale kernel
    if a6w6_fly_available(m, n, k, has_bias):
        codes = TS.unpack_fp4_codes_ref(b.view(torch.uint8), n, k, "k128")
        b6 = TS.pack_fp6_codes_ref(((codes & 8) << 2) | ((codes & 6) << 2) | ((codes & 1) << 2))
        ref6 = gemm_fp6_impl(a, a_s, b6, b_s, m, n, k, torch.bfloat16, G, bias, False, False, 0, True)
        assert torch.equal(out.view(torch.int16), ref6.view(torch.int16))
    # the out-variant writes the same bits
    o2 = torch.empty_like(out)
    torch.ops.primus_turbo.gemm_fp6_out_impl(a, a_s, b, b_s, o2, m, n, k, G, True, bias, 0, False, True)
    assert torch.equal(o2.view(torch.int16), out.view(torch.int16))
