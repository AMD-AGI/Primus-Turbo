###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""gemm_a6w6_ts_cat_gres_out: the custom op passes its operands to aiter's two-segment tilescale GEMM unchanged
(argument order, flattening, the optional separate C1 planes), eagerly and under torch.compile. The kernel's own
numerics are aiter's test (op_tests/test_gemm_tilescale_cat.py)."""
import pytest
import torch

from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import a6w6_ts_cat_table, gemm_a6w6_ts_cat_gres_out


def _table():
    try:
        return sorted(a6w6_ts_cat_table()) if torch.cuda.is_available() else []
    except Exception:
        return []


SHAPES = _table()


def _segment(M, N, K, g):
    import aiter.ops.tilescale as TS

    a6 = torch.randint(0, 64, (M, K), dtype=torch.uint8, device="cuda", generator=g)
    b6 = torch.randint(0, 64, (N, K), dtype=torch.uint8, device="cuda", generator=g)
    sa = torch.randint(122, 130, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    sb = torch.randint(122, 130, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    return (TS.pack_fp6_codes_ref(a6), TS.pack_scales_ref(sa, is_b=False), TS.pack_fp6_codes_ref(b6),
            TS.pack_scales_ref(sb, is_b=True))


@pytest.mark.skipif(not SHAPES, reason="no two-segment tilescale A6W6 kernels (needs gfx950 and aiter)")
@pytest.mark.parametrize("split_c1", [False, True], ids=["joined_b", "split_c1"])
def test_cat_op_matches_aiter(split_c1):
    from aiter.ops.gemm_op_tilescale import gemm_a6w6_tilescale_cat

    M, N, K, K2, Bt = SHAPES[0]
    g = torch.Generator(device="cuda").manual_seed(M + N + K + K2)
    a, sa, b, sb = _segment(M, N, K, g)
    a2, sa2, b2, sb2 = _segment(M, N, K2, g)
    bf = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device="cuda", generator=g)  # noqa: E731
    x, bias = bf(M, N), bf(N)
    gate = (bf(Bt, 6 * N) * 0.1)[:, 4 * N:5 * N]  # a chunk of a wider modulation table: row stride > N
    b_c1 = b2_c1 = None
    if split_c1:  # B's C1 plane (the last N*K/4 bytes) in its own buffer
        b, b_c1 = b[: N * K // 2].clone(), b[N * K // 2:].clone()
        b2, b2_c1 = b2[: N * K2 // 2].clone(), b2[N * K2 // 2:].clone()

    want_out, want_h = torch.empty(M, N, dtype=torch.bfloat16, device="cuda"), torch.empty(
        M, N, dtype=torch.bfloat16, device="cuda")
    gemm_a6w6_tilescale_cat(a, b, sa, sb, a2, b2, sa2, sb2, want_out, want_h, K, K2, bias, x, gate, b_c1, b2_c1)

    def run(out, h):
        gemm_a6w6_ts_cat_gres_out(a, sa, b, sb, b_c1, a2, sa2, b2, sb2, b2_c1, bias, x, gate, out, h, K, K2)
        return out, h

    for fn in (run, torch.compile(run, fullgraph=True)):
        out, h = torch.empty_like(want_out), torch.empty_like(want_h)
        fn(out, h)
        assert torch.equal(out, want_out) and torch.equal(h, want_h)
