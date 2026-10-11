###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gemm_fp8_ts_cat_gres_out: the two-segment tensorwise-FP8 tilescale GEMM with the gated residual, on operands from
the ``*_f8`` packers, against the product of the dequantized FP8 operands (within bf16 rounding), eager and under
torch.compile."""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx950_device


def _shapes():
    try:
        from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import fp8_ts_cat_table

        return sorted(fp8_ts_cat_table()) if torch.cuda.is_available() else []
    except ImportError:
        return []


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950_device(torch.device("cuda"))), reason="needs gfx950"
)


@pytest.mark.parametrize("shape", [s for s in _shapes() if s[0] * s[1] <= 1024 * 1024] or [None])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
def test_fp8_cat_vs_separate(shape, compiled):
    if shape is None:
        pytest.skip("no small FP8 cat kernel")
    from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp8_ts_cat_gres_out
    from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX

    M, N, K, K2, Bt = shape
    g = torch.Generator(device="cuda").manual_seed(3)
    rn = lambda *s, scale=1.0: torch.randn(*s, device="cuda", generator=g).to(torch.bfloat16) * scale  # noqa: E731
    o, wp, act, w2 = rn(M, K), rn(N, K, scale=0.02), rn(M, K2, scale=0.3), rn(N, K2, scale=0.01)
    x, bias = rn(M, N), rn(N, scale=0.1)
    table = rn(Bt, 4 * N, scale=0.5)
    gate = table[:, 2 * N : 3 * N]
    fmt = MX.with_ts8_row(0)

    def pack(t):
        q = (448.0 / t.float().abs().amax()).reshape(1)
        dq = (t.float() * q).clamp(-448, 448).to(torch.float8_e4m3fn).float() / q
        return MX.quantize_mx_f8(t, fmt, q, torch.zeros(1, device="cuda"))[0], (1.0 / q), dq

    (pa, sa, da), (pb, sb, db), (pa2, sa2, da2), (pb2, sb2, db2) = (pack(t) for t in (o, wp, act, w2))
    out, h = torch.empty_like(x), torch.empty_like(x)

    def fn(out, h):
        gemm_fp8_ts_cat_gres_out(pa, sa, pb, sb, pa2, sa2, pb2, sb2, bias, x, gate, out, h, K, K2)
        return out, h

    (torch.compile(fn, fullgraph=True) if compiled else fn)(out, h)
    h_ref = (da.double() @ db.double().t() + da2.double() @ db2.double().t()).float() + bias.float()
    out_ref = x.float() + gate.float().repeat(M // Bt, 1) * h_ref
    rel = lambda u, v: ((u.float() - v).abs().max() / v.abs().max()).item()  # noqa: E731
    assert rel(h, h_ref) < 2**-6 and rel(out, out_ref) < 2**-6
