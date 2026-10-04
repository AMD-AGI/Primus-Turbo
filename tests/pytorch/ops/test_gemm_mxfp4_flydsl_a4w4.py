###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL MXFP4 GEMM reading aiter gemm_a4w4's own operand layout (layout="a4w4").

A plain fp4x2 rows, B shuffle_weight(layout=(16, 16)), both scales e8m0_shuffle'd [rows, K/32] -- exactly
what aiter.gemm_a4w4(bpreshuffle=True) takes. Held to: bit-identity with FlyDSL's own layout and with
the exact A4W4 tile-blob kernel (aiter_a4w4_blob_stnt_allk, one fp32 sum) on the same values, over
repeated calls; the torch packers here equal aiter's shuffle_weight / e8m0_shuffle byte for byte.
"""

import pytest
import torch


def operands(M, N, K, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    asc = torch.randint(118, 134, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    bsc = torch.randint(118, 134, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    return a, b, asc, bsc


def shuffle_b(codes):
    """aiter shuffle_weight(layout=(16, 16)) of fp4x2 rows [R, K/2]: [R/16][K/64][2][16][16 B]."""
    R, KB = codes.shape
    return codes.view(R // 16, 16, KB // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(R, KB)


def shuffle_scale(scales):
    """aiter e8m0_shuffle of [R, K/32]: per 32 rows x 256 K, byte kg*64 + row16*4 + s*2 + half16."""
    R, sp = scales.shape
    return scales.view(R // 32, 2, 16, sp // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous().view(R, sp)


def blob(codes, scales):
    """The A4W4 tile blob (mxfp4_mxfp4_c0_256_padk2) of the same values, for the exact reference."""
    R, KB = codes.shape
    K = KB * 2
    G, nk, dev = K // 32, K // 128 + 2, codes.device
    r = torch.arange(R, device=dev).view(-1, 1)
    g = torch.arange(G, device=dev).view(1, -1)
    tile, rem, kg = (r // 256) * nk + g // 4, r % 256, g % 4
    base = tile * 16384 + ((rem // 16) * 64 + kg * 16 + rem % 16) * 16
    out = torch.zeros(((R + 255) // 256) * nk * 16384, dtype=torch.uint8, device=dev)
    out[(base.unsqueeze(-1) + torch.arange(16, device=dev)).reshape(-1)] = codes.view(R, G, 16).reshape(-1)
    s = torch.zeros(((R + 255) // 256) * nk * 1024, dtype=torch.uint8, device=dev)
    s[(tile * 1024 + (rem // 128) * 512 + kg * 128 + (rem % 16) * 8 + (rem % 128) // 16).reshape(-1)] = (
        scales.reshape(-1)
    )
    return out, s


def variants(M, N, K, seed=1):
    import aiter
    from aiter import dtypes

    gemm_a4w4_blob_asm = pytest.importorskip("aiter.ops.gemm_op_a4w4_blob").gemm_a4w4_blob_asm

    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as FK

    a, b, asc, bsc = operands(M, N, K, seed)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    sa, sb = FK.preshuffle_mxfp4_scales(asc, bsc, M, N, K)
    bs, AS, BS = shuffle_b(b), shuffle_scale(asc), shuffle_scale(bsc)
    Ab, Asb = blob(a, asc)
    Bb, Bsb = blob(b, bsc)
    ob = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    return {
        "fly": lambda: FK.gemm_mxfp4_flydsl_kernel(a, sa, b, sb, out=out, scales_prepacked=True, k=K),
        "fly a4w4-layout": lambda: FK.gemm_mxfp4_flydsl_kernel(
            a, AS, bs, BS, out=out, scales_prepacked=True, k=K, layout="a4w4"
        ),
        "aiter gemm_a4w4 (tuned)": lambda: aiter.gemm_a4w4(
            a.view(dtypes.fp4x2),
            bs.view(dtypes.fp4x2),
            AS.view(dtypes.fp8_e8m0),
            BS.view(dtypes.fp8_e8m0),
            bpreshuffle=True,
        ),
        "aiter blob": lambda: gemm_a4w4_blob_asm(Ab, Bb, Asb, Bsb, ob, K),
    }


def test_packers_are_aiters():
    pytest.importorskip("aiter")
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    a, b, asc, bsc = operands(512, 768, 1024)
    assert torch.equal(shuffle_b(b), shuffle_weight(b, layout=(16, 16)).view(torch.uint8))
    assert torch.equal(shuffle_scale(bsc), e8m0_shuffle(bsc).view(torch.uint8))


@pytest.mark.parametrize(
    "m,n,k", [(1024, 1024, 1024), (16384, 3072, 12288), (3072, 12288, 16384), (8192, 3072, 12288)]
)
def test_mxfp4_flydsl_a4w4_layout_exact(m, n, k):
    pytest.importorskip("flydsl")
    f = variants(m, n, k)
    ref = f["aiter blob"]().clone()
    assert torch.equal(f["fly"]()[:m, :n], ref)
    for _ in range(6):
        assert torch.equal(f["fly a4w4-layout"](), ref)
