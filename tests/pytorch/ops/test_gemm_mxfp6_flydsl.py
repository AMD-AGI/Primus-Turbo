###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL MXFP6 (A6W6, E2M3 x E2M3) GEMM: bit-exactness vs AITER's A6W6 asm and a K128-sequential fp32 model.

Operands: random bf16 -> MX quantised to E2M3 codes with per-32 E8M0 scales (pure torch, no
Hadamard). The SAME codes are packed twice: into the FlyDSL kernel's plain-row C0/C1 planes
(+ the MXFP4 packed scale layout, b_ilv=0), and into AITER's mxfp6_c0c1_256_padk2 blobs.

Held to: bit-identity with AITER's A6W6 kernel on the same values, and with a K128-sequential
fp32 accumulation model (each K128 block's dot product exact, added to the fp32 accumulator
with one rounding), against which the fp64 dequantise-and-multiply is also reported.

"""

import sys

import pytest
import torch

# E2M3 magnitudes for codes 0..31 (sign is bit 5).
_E2M3 = [
    ((c & 7) / 8.0) if (c >> 3) == 0 else (2.0 ** ((c >> 3) - 1)) * (1.0 + (c & 7) / 8.0) for c in range(32)
]


def e2m3_table(device):
    pos = torch.tensor(_E2M3, dtype=torch.float64, device=device)
    return torch.cat([pos, -pos])  # code -> value


def mx_quant_e2m3(x):
    """[R, K] float -> (codes uint8 [R, K], scale E8M0 uint8 [R, K/32]); scale = floor(log2 amax) - 2."""
    R, K = x.shape
    xg = x.float().reshape(R, K // 32, 32)
    amax = xg.abs().amax(-1)
    _, e = torch.frexp(amax)  # amax = m 2^e, m in [0.5, 1)
    unb = torch.where(amax > 0, e - 1 - 2, torch.full_like(e, -127)).clamp(-127, 127)
    xs = xg / torch.pow(2.0, unb.float()).unsqueeze(-1)
    pos = torch.tensor(_E2M3, dtype=torch.float32, device=x.device)
    mid = (pos[1:] + pos[:-1]) / 2
    mag = torch.bucketize(xs.abs().clamp(max=7.5), mid)  # nearest magnitude code
    codes = (mag + 32 * (xs < 0).long()).to(torch.uint8)
    return codes.reshape(R, K), (unb + 127).to(torch.uint8)


def pack24(codes):
    """[R, K] 6-bit codes -> [R, K/32, 24] bytes: 32 values little-endian contiguous (value i at bits 6i..)."""
    R, K = codes.shape
    c = codes.reshape(R, K // 4, 4).to(torch.int32)
    w = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    b = torch.stack([w & 255, (w >> 8) & 255, (w >> 16) & 255], -1).to(torch.uint8)
    return b.reshape(R, K // 32, 24)


def fly_planes(codes):
    p = pack24(codes)
    R, G, _ = p.shape
    return p[..., :16].reshape(R, G * 16).contiguous(), p[..., 16:].reshape(R, G * 8).contiguous()


def aiter_blobs(codes, scale):
    """AITER mxfp6_c0c1_256_padk2 operand + scale blobs (addressing of csrc quantization_mxfp6_gfx950.cu)."""
    import aiter

    R, K = codes.shape
    G = K // 32
    nk_pad = K // 128 + 2
    p = pack24(codes)
    sz_p, sz_s = aiter.mxfp6_gemm_pack_size(R, K)
    blob = torch.zeros(sz_p, dtype=torch.uint8, device=codes.device)
    sblob = torch.zeros(sz_s, dtype=torch.uint8, device=codes.device)
    r = torch.arange(R, device=codes.device).view(R, 1)
    g = torch.arange(G, device=codes.device).view(1, G)
    tile_row, rem = r // 256, r % 256
    step, kg = g // 4, g % 4
    block = (rem // 16) * 64 + kg * 16 + rem % 16
    tbase = (tile_row * nk_pad + step) * 24576
    c0 = (tbase + block * 16).unsqueeze(-1) + torch.arange(16, device=codes.device)
    c1 = (tbase + 16384 + block * 8).unsqueeze(-1) + torch.arange(8, device=codes.device)
    blob[c0.reshape(-1)] = p[..., :16].reshape(-1)
    blob[c1.reshape(-1)] = p[..., 16:].reshape(-1)
    sa = (
        (tile_row * nk_pad + step) * 1024 + (rem // 128) * 512 + kg * 128 + (rem % 16) * 8 + (rem % 128) // 16
    )
    sblob[sa.reshape(-1)] = scale.reshape(-1)
    assert blob.numel() == (R + 255) // 256 * nk_pad * 24576
    return blob, sblob


def dequant(codes, scale):
    tab = e2m3_table(codes.device)
    R, K = codes.shape
    v = tab[codes.long()].reshape(R, K // 32, 32) * torch.pow(2.0, scale.double() - 127).unsqueeze(-1)
    return v.reshape(R, K)


def reference_k128_fp32(a_deq, b_deq, chunk=128):
    """fp32 accumulator, K128 blocks in order, each block's exact sum added with one rounding."""
    M, K = a_deq.shape
    acc = torch.zeros(M, b_deq.shape[0], dtype=torch.float32, device=a_deq.device)
    for k0 in range(0, K, chunk):
        blk = a_deq[:, k0 : k0 + chunk] @ b_deq[:, k0 : k0 + chunk].t()  # fp64, exact for these values
        acc = (acc.double() + blk).float()
    return acc


def make_operands(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16, generator=g)
    a_codes, a_sc = mx_quant_e2m3(a)
    b_codes, b_sc = mx_quant_e2m3(b)
    return a_codes, a_sc, b_codes, b_sc


def run_fly(a_codes, a_sc, b_codes, b_sc, **kw):
    from primus_turbo.flydsl.gemm.gemm_mxfp6_kernel import (
        gemm_mxfp6_flydsl_kernel,
        gemm_mxfp6_persistent,
        kblk_planes,
        preshuffle_mxfp6_scales,
    )

    if kw.pop("aiter_layout", False):  # the persistent kernel on AITER's own blobs
        (M, K), N = a_codes.shape, b_codes.shape[0]
        A, As = aiter_blobs(a_codes, a_sc)
        B, Bs = aiter_blobs(b_codes, b_sc)
        kw.pop("persist", None)
        return lambda out=None: gemm_mxfp6_persistent(  # noqa: E731
            A, None, B, None, As, Bs, out=out, layout="aiter", m=M, n=N, k=K, **kw
        )
    persist = kw.pop("persist", False)
    if persist:
        kw["kblk"] = True

    M, K = a_codes.shape
    N = b_codes.shape[0]
    a0, a1 = fly_planes(a_codes)
    b0, b1 = fly_planes(b_codes)
    if kw.get("kblk"):
        (a0, a1), (b0, b1) = kblk_planes(a0, a1), kblk_planes(b0, b1)
    a_sp, b_sp = preshuffle_mxfp6_scales(a_sc, b_sc, M, N, K)
    if persist:
        kw.pop("kblk")
        fn = lambda out=None: gemm_mxfp6_persistent(a0, a1, b0, b1, a_sp, b_sp, out=out, **kw)  # noqa: E731
    else:
        fn = lambda out=None: gemm_mxfp6_flydsl_kernel(a0, a1, b0, b1, a_sp, b_sp, out=out, **kw)  # noqa: E731
    return fn


def run_aiter(a_codes, a_sc, b_codes, b_sc, kernel="f6gemm_stnt_allk_nobias_kernel_func", bias=None):
    import aiter

    M, K = a_codes.shape
    N = b_codes.shape[0]
    A, As = aiter_blobs(a_codes, a_sc)
    B, Bs = aiter_blobs(b_codes, b_sc)
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    def fn():
        if bias is None:
            aiter.gemm_a6w6_asm(A, B, As, Bs, out, K, kernel)
        else:
            aiter.gemm_a6w6_asm(A, B, As, Bs, out, K, kernel, 1.0, bias)
        return out

    return fn


@pytest.mark.parametrize("m,n,k", [(256, 256, 512), (512, 768, 1024), (16384, 12288, 3072)])
def test_mxfp6_flydsl_matches_aiter_and_exact(m, n, k):
    pytest.importorskip("flydsl")
    a_codes, a_sc, b_codes, b_sc = make_operands(m, n, k, seed=1)
    fly = run_fly(a_codes, a_sc, b_codes, b_sc)
    ref = reference_k128_fp32(dequant(a_codes, a_sc), dequant(b_codes, b_sc)).to(torch.bfloat16)
    got = fly()
    assert torch.equal(got, ref)
    for _ in range(5):
        assert torch.equal(fly(), ref)
    assert torch.equal(run_aiter(a_codes, a_sc, b_codes, b_sc)(), ref)


@pytest.mark.parametrize(
    "m,n,k,tpw",
    [
        (512, 512, 1536, 1),
        (512, 512, 1536, 4),
        (512, 768, 8192, 2),
        (1024, 1024, 3072, 2),
        (16384, 12288, 3072, None),
    ],
)
def test_mxfp6_flydsl_persistent_matches_aiter(m, n, k, tpw):
    """The persistent kernel (kblk planes, tiles per WG in one asm, hardware K loop, folded C store) is
    bit-identical to AITER's A6W6 over repeated calls."""
    pytest.importorskip("flydsl")
    a_codes, a_sc, b_codes, b_sc = make_operands(m, n, k, seed=4)
    ref = run_aiter(a_codes, a_sc, b_codes, b_sc)().clone()
    fly = run_fly(a_codes, a_sc, b_codes, b_sc, persist=True, tpw=tpw)
    for _ in range(5):
        assert torch.equal(fly(), ref)


def make_bias(N, acc_scale=1.0, seed=5):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(N, device="cuda", generator=g) * acc_scale).to(torch.bfloat16)


def test_aiter_bias_semantics():
    """AITER A6W6's bias epilogue is fp32(acc) + fp32(bias), rounded once (RNE) to bf16 -- checked
    against the exact K128-sequential fp32 accumulation model, not assumed (a bf16-rounded
    accumulator plus bias disagrees on a large fraction of outputs)."""
    a_codes, a_sc, b_codes, b_sc = make_operands(512, 768, 1024, seed=7)
    acc = reference_k128_fp32(dequant(a_codes, a_sc), dequant(b_codes, b_sc))
    for scale in (1e-3, 1.0, 30.0):
        bias = make_bias(768, scale * acc.abs().mean().item())
        got = run_aiter(a_codes, a_sc, b_codes, b_sc, "f6gemm_stnt_kernel_func", bias=bias)()
        assert torch.equal(got, (acc + bias.float()).to(torch.bfloat16))
        assert not torch.equal(got, (acc.to(torch.bfloat16).float() + bias.float()).to(torch.bfloat16))


@pytest.mark.parametrize(
    "m,n,k,tpw",
    [(512, 768, 1536, 1), (512, 768, 1536, 2), (16384, 9216, 3072, None), (8192, 9216, 3072, None)],
)
def test_mxfp6_flydsl_persistent_bias_matches_aiter(m, n, k, tpw):
    """bias epilogue bit-identical to AITER f6gemm_stnt with bias, over repeated calls."""
    pytest.importorskip("flydsl")
    a_codes, a_sc, b_codes, b_sc = make_operands(m, n, k, seed=6)
    bias = make_bias(n, 20.0)
    ref = run_aiter(a_codes, a_sc, b_codes, b_sc, "f6gemm_stnt_kernel_func", bias=bias)().clone()
    fly = run_fly(a_codes, a_sc, b_codes, b_sc, persist=True, tpw=tpw, bias=bias)
    for _ in range(6):
        assert torch.equal(fly(), ref)


@pytest.mark.parametrize(
    "m,n,k,tpw,with_bias",
    [
        (512, 768, 1536, 1, False),
        (512, 768, 1536, 2, True),
        (16384, 12288, 3072, 1, False),
        (16384, 12288, 3072, 1, True),
        (8192, 9216, 3072, 1, True),
        (8192, 9216, 3072, 1, False),
    ],
)
def test_mxfp6_flydsl_persistent_aiter_layout(m, n, k, tpw, with_bias):
    """layout="aiter": the persistent kernel reading AITER's own mxfp6_c0c1_256_padk2 operand and
    scale blobs is bit-identical to aiter.gemm_a6w6_asm on those blobs (f6gemm_stnt, with or
    without its bias epilogue) over repeated calls."""
    pytest.importorskip("flydsl")
    a_codes, a_sc, b_codes, b_sc = make_operands(m, n, k, seed=8)
    bias = make_bias(n, 20.0) if with_bias else None
    ref = run_aiter(a_codes, a_sc, b_codes, b_sc, "f6gemm_stnt_kernel_func", bias=bias)().clone()
    fly = run_fly(a_codes, a_sc, b_codes, b_sc, aiter_layout=True, tpw=tpw, bias=bias)
    for _ in range(6):
        assert torch.equal(fly(), ref)


@pytest.mark.parametrize(
    "m,n,k,with_bias",
    [(512, 768, 1536, False), (512, 768, 1536, True), (16384, 12288, 3072, False), (8192, 9216, 3072, True)],
)
def test_mxfp6_flydsl_aiter_abi(m, n, k, with_bias):
    """abi="aiter": the layout="aiter", tpw=1 kernel compiled for aiter's gemm_a6w6_asm ABI (packed 0x180-byte
    KernelArgs, 2D grid, bias via ptr_C) and launched with the field values that launcher passes is
    bit-identical to aiter.gemm_a6w6_asm(f6gemm_stnt) over repeated calls."""
    pytest.importorskip("flydsl")
    from primus_turbo.flydsl.gemm.gemm_mxfp6_kernel import gemm_mxfp6_aiterabi

    a_codes, a_sc, b_codes, b_sc = make_operands(m, n, k, seed=12)
    bias = make_bias(n, 20.0) if with_bias else None
    ref = run_aiter(a_codes, a_sc, b_codes, b_sc, "f6gemm_stnt_kernel_func", bias=bias)().clone()
    A, As = aiter_blobs(a_codes, a_sc)
    B, Bs = aiter_blobs(b_codes, b_sc)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    for _ in range(6):
        out.fill_(0)
        gemm_mxfp6_aiterabi(A, B, As, Bs, out, k, bias=bias)
        assert torch.equal(out, ref)


@pytest.mark.parametrize("wrong", ["M", "K"])
def test_mxfp6_flydsl_aiter_abi_shape_guard(wrong):
    """A launch whose kernarg M/N/K/stride_D0 differs from the shape the code object was compiled for
    traps (s_trap 2) before touching memory: the process dies with a GPU error. Run in a subprocess."""
    import subprocess
    import textwrap

    code = textwrap.dedent(
        f"""
        import sys, torch
        sys.path.insert(0, {__file__.rsplit("/", 1)[0]!r})
        from test_gemm_mxfp6_flydsl import make_operands, aiter_blobs
        from primus_turbo.flydsl.gemm.gemm_mxfp6_kernel import gemm_mxfp6_aiterabi
        a, asc, b, bsc = make_operands(512, 768, 1536, seed=1)
        A, As = aiter_blobs(a, asc); B, Bs = aiter_blobs(b, bsc)
        out = torch.empty(512, 768, dtype=torch.bfloat16, device="cuda")
        kw = dict(mn=(768, 768)) if {wrong!r} == "M" else dict(kmn=1024)
        gemm_mxfp6_aiterabi(A, B, As, Bs, out, 1536, **kw)
        torch.cuda.synchronize()
        print("NO TRAP")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "NO TRAP" not in r.stdout, r.stdout + r.stderr
