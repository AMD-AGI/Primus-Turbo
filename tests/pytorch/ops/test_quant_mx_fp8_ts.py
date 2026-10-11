###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tensorwise FP8 rows of the MX packers (``with_ts8_row``, the ``quantize_mx_*_f8`` ops) and the FP8 tilescale GEMM
(``gemm_fp8_ts_impl`` / ``gemm_fp8_ts_out_impl``, aiter's FP8 tilescale kernels).

* rows: E4M3 of ``x * qscale`` (saturating at 448, round to nearest even) in the tilescale FP8 planes, bitwise;
* columns: the same bytes as the MX pack with the same column format, bitwise;
* amax: max |x| (the prologue's output for the fused pack), maxed into the caller's slot;
* the fused BIAS_GELU pack equals the plain pack of the materialised activation, bitwise;
* the GEMM: the dequantized product within bf16 rounding, into a fresh tensor or a row slice of a buffer.
"""

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.utils import is_gfx950_device

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and is_gfx950_device(torch.device("cuda"))), reason="FP8 tilescale packs and GEMMs need gfx950")

M, K = 512, 1024


def _mx():
    from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX

    return MX


def _fmt(col=True):
    MX = _mx()
    base = MX.ts_fmt(col=(True, 4, 0)) if col else 0
    return MX.with_ts8_row(base)


def _ref_planes(x, q):
    from aiter.ops.tilescale import pack_fp8_codes_ref

    codes = (x.float() * q).clamp(-448, 448).to(torch.float8_e4m3fn)
    return pack_fp8_codes_ref(codes.view(torch.uint8))


def _data(seed, rows=M, cols=K, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g) * scale


@pytest.mark.parametrize("q", [1.0, 37.5, 2.0**-3])
def test_dual_rows_cols_amax(q):
    MX = _mx()
    x = _data(1)
    qs = torch.tensor([q], device="cuda")
    amax = torch.zeros(1, device="cuda")
    rp, rs, cp, cs = MX.quantize_mx_dual_f8(x, _fmt(), qs, amax)
    assert rs.numel() == 0
    assert torch.equal(rp, _ref_planes(x, q))
    _, _, cp6, cs6 = MX.quantize_mx_dual(x, _mx().with_ts6_row(MX.ts_fmt(col=(True, 4, 0)), False))
    assert torch.equal(cp, cp6) and torch.equal(cs, cs6)
    assert amax.item() == x.float().abs().max().item()


def test_amax_accumulates():
    MX = _mx()
    a, b = _data(2), _data(3, scale=3.0)
    qs, amax = torch.ones(1, device="cuda"), torch.zeros(1, device="cuda")
    MX.quantize_mx_f8(a, _fmt(False), qs, amax)
    MX.quantize_mx_f8(b, _fmt(False), qs, amax)
    assert amax.item() == max(a.float().abs().max().item(), b.float().abs().max().item())


def test_row_only_matches_dual():
    MX = _mx()
    x = _data(4)
    qs = torch.tensor([11.0], device="cuda")
    rp1 = MX.quantize_mx_f8(x, _fmt(False), qs, torch.zeros(1, device="cuda"))[0]
    rp2 = MX.quantize_mx_dual_f8(x, _fmt(), qs, torch.zeros(1, device="cuda"))[0]
    assert torch.equal(rp1, rp2)


def test_fused_bias_gelu_equals_materialised():
    MX = _mx()
    y = _data(5)
    b = _data(6, 1, K).view(-1)
    qs = torch.tensor([20.0], device="cuda")
    a1, a2 = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    fused = MX.quantize_mx_fused_dual_f8(y, None, b, 1, False, _fmt(), qs, a1)
    act = F.gelu((y + b).float(), approximate="tanh").to(torch.bfloat16)
    plain = MX.quantize_mx_dual_f8(act, _fmt(), qs, a2)
    # rows: the same values (the fused prologue may round a tiny negative to -0 where the materialised one gives +0)
    dec = lambda t: t.view(torch.float8_e4m3fn).float()  # noqa: E731
    assert torch.equal(dec(fused[0]), dec(plain[0]))
    for i in (1, 2, 3):
        assert torch.equal(fused[i], plain[i]), i
    assert a1.item() == a2.item()


def test_rejects_fp8_fmt_on_plain_ops():
    MX = _mx()
    with pytest.raises(RuntimeError):
        MX.quantize_mx_dual(_data(7), _fmt())


@pytest.mark.parametrize("bias", [False, True])
def test_gemm_fp8_ts(bias):
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import (
        fp8_ts_table,
        gemm_fp8_ts_impl,
        gemm_fp8_ts_out_impl,
        tilescale_table_has,
    )

    MX = _mx()
    m, n, k = 512, 512, 1024
    if not tilescale_table_has(fp8_ts_table(), m, n, k, bias):
        pytest.skip("no FP8 tilescale kernel for the test shape")
    x, w = _data(8, m, k), _data(9, n, k, 0.02)
    bv = _data(10, 1, n).view(-1) if bias else None
    qx = 448.0 / x.float().abs().max()
    qw = 448.0 / w.float().abs().max()
    ax, aw = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    xp = MX.quantize_mx_f8(x, _fmt(False), qx.reshape(1), ax)[0]
    wp = MX.quantize_mx_f8(w, _fmt(False), qw.reshape(1), aw)[0]
    sx, sw = (1.0 / qx).reshape(1), (1.0 / qw).reshape(1)
    out = gemm_fp8_ts_impl(xp, sx, wp, sw, m, n, k, bv)
    xd = (x.float() * qx).clamp(-448, 448).to(torch.float8_e4m3fn).float() / qx
    wd = (w.float() * qw).clamp(-448, 448).to(torch.float8_e4m3fn).float() / qw
    ref = xd.double() @ wd.double().t() + (bv.double() if bias else 0)
    assert (out.double() - ref).abs().max().item() <= ref.abs().max().item() * 2**-7
    big = torch.zeros(3 * m, n, dtype=torch.bfloat16, device="cuda")
    gemm_fp8_ts_out_impl(xp, sx, wp, sw, big[m : 2 * m], m, n, k, bv)
    assert torch.equal(big[m : 2 * m], out)
