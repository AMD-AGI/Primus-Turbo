###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gfx1250 MXFP8 GEMMs (``primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250``) against the recipe reference,
bitwise on integer-valued operands (exact in fp32) in the GEMM layout (packed scales, b in 16x16 blocks): every
kernel the dense DeepSeek-V3 linears launch (forward, dgrad, wgrad plain and into a bf16 / fp32 main_grad), the
routed experts' grouped-M and variable-K kernels with the entry points' default configs (ragged, one-row and empty
experts), M / N / K tails, operands without storage slack, SNR on random data, determinism and the refused calls.
Outputs sit between NaN guard rows: a missed store leaves NaN, a stray one past ``out`` hits a guard."""

import math

import pytest
import torch

from primus_turbo.flydsl.gemm.gemm_mxfp8_gfx1250_configs import DENSE, pick_config
from primus_turbo.flydsl.gemm.gemm_mxfp8_gfx1250_configs import MXGemmConfig as C
from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.ref.mxfp8_ref import (
    PAD_SCALE,
    gemm_ref,
    int_valued_operands,
    pack_scales,
    preshuffle_b16,
    quant_rows,
    snr_db,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 GPU"
)

if is_gfx1250():
    import primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 as KM
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import (
        gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_variable_k_gfx1250,
    )

DEV = "cuda"
BF16, F32 = torch.bfloat16, torch.float32
DEFAULT = C(256, 256, 2, 2, 2, 2)
KP1_C2X4 = C(256, 256, 4, 1, 2, 4)  # dense o_proj fwd; the routed experts' grouped-M


def _slack(t, extra=4096):
    """``t`` in a storage with ``extra`` bytes after it (the MX quantizers allocate that slack after their outputs)."""
    n = t.numel() * t.element_size()
    v = torch.empty(n + extra, dtype=torch.uint8, device=t.device)[:n].view(t.dtype).view(t.shape)
    return v.copy_(t)


def _exact(t):
    """``t`` in a fresh allocation of exactly its size. At 10 MiB or more that is a segment of its own, so the tensor
    likely ends where the mapping does (best effort: the address range after it may still be mapped)."""
    torch.cuda.empty_cache()
    x = torch.empty(t.numel() * t.element_size(), dtype=torch.uint8, device=t.device)
    return x.view(t.dtype).view(t.shape).copy_(t)


def _guarded(shape, dtype, init=None):
    """An output of ``shape`` between two NaN guard rows (of its last dim), NaN itself unless ``init`` is given:
    (out, check), where check() asserts the guards are untouched."""
    buf = torch.full((math.prod(shape[:-1]) + 2, shape[-1]), float("nan"), dtype=dtype, device=DEV)
    out = buf[1:-1].view(shape)
    if init is not None:
        out.copy_(init)

    def check():
        assert torch.isnan(buf[0]).all() and torch.isnan(buf[-1]).all(), "a store past out"

    return out, check


def _eq(out, ref):
    torch.testing.assert_close(out, ref.to(out.dtype), rtol=0, atol=0)


def _old_c(shape, dtype, seed):
    """The contents of an accumulation target before the call."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(shape, device=DEV, generator=g) * 64).to(dtype)


def _run_dense(cfg, M, N, K, dt, acc):
    """One dense call into a guarded out: == the exact result, or round(fp32 result + C) (beta = 1, rounded once)."""
    a, sa, b, sb = int_valued_operands(M, N, K, seed=M + N + K)
    ref = gemm_ref(a, sa, b, sb).float()
    c0 = _old_c((M, N), dt, seed=M + N)
    out, check = _guarded((M, N), dt, c0 if acc else None)
    a, sa, b, sb = _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), pack_scales(sb)
    gemm_mxfp8_gfx1250(a, sa, b, sb, out_dtype=dt, out=out, config=cfg, accumulate=acc)
    _eq(out, (ref + c0.float()) if acc else ref)
    check()


# DeepSeek-V3's MXFP8 linears (N_out, K_in): MLA q_down, q_up, kv_down, kv_up, o_proj, the shared expert's fc1, fc2.
# On T tokens each runs fwd (T, N, K), dgrad (T, K, N) and wgrad (N, K, T): plain, and into a bf16 / fp32 main_grad.
# T = 8192 is the token count of the config table; kv_down at 4096 tokens adds the three 128x128 kernels that the
# grid-fill rule gives a GEMM of few tiles at any other token count.
LINEARS = [(1536, 7168), (24576, 1536), (576, 7168), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
SHAPES = [(8192, n_out, k_in) for n_out, k_in in LINEARS] + [(4096, 576, 7168)]


def _production_kernels():
    """{constexprs: (config, (m, n, k), out dtype, accumulate)}: the distinct kernels of those GEMMs (K padded to
    128, as the quantizers pad it), each with a reduced shape that compiles to the same constexprs: the cluster the
    entry point shrinks to, an M tail, the N and K tails of the production shape, K past one pipeline revolution; a
    persistent kernel gets more tiles than workgroups, so that workgroups run several."""
    kernels = {}
    for T, n_out, k_in in SHAPES:
        calls = [(T, n_out, k_in, BF16, False), (T, k_in, n_out, BF16, False)]
        calls += [(n_out, k_in, T, dt, acc) for dt, acc in ((BF16, False), (BF16, True), (F32, True))]
        for M, N, K, dt, acc in calls:
            K = -(-K // 128) * 128
            cfg = pick_config(M, N, K, beta1=acc and dt == BF16)
            step = 128 * cfg.kpair
            const = KM.constexprs(DENSE, cfg, M, N, 1, K % step != 0, dt == F32, acc)
            cm, cn, persist = const[4], const[5], const[-2]
            nt = 3 * cn  # odd multiples of the cluster: the entry point keeps it
            mt = cm * (2 * (150 // (cm * nt)) + 1 if persist else 3)
            m, n = mt * cfg.tile_m - 100, nt * cfg.tile_n - (48 if N % cfg.tile_n else 0)
            k = (cfg.num_buffers + 1) * step + (128 if K % step else 0)
            assert KM.constexprs(DENSE, cfg, m, n, 1, k % step != 0, dt == F32, acc) == const
            kernels.setdefault(const, (cfg, (m, n, k), dt, acc))
    return kernels


def _kernel_id(c):
    flags = zip(("_kt", "", "", "_f32", "_acc", "_ce", "_p"), c[6:13])
    return f"t{c[0]}x{c[1]}_nb{c[2]}_kp{c[3]}_c{c[4]}x{c[5]}" + "".join(f for f, on in flags if on is True)


PRODUCTION = _production_kernels() if is_gfx1250() else {}


@pytest.mark.parametrize("const", PRODUCTION, ids=_kernel_id)
def test_dense_production_kernels(const):
    """Every kernel the DeepSeek-V3 linears launch, bitwise."""
    cfg, (m, n, k), dt, acc = PRODUCTION[const]
    _run_dense(cfg, m, n, k, dt, acc)


def test_dense_cluster_on_grid_z():
    """A clustered kernel on a grid with z > 1 (40 M tiles: grid (8, 2, 5)), as the wide MLA projections launch
    it: plain, and beta = 1 into bf16 with the early old-C fetch (kv_up's wgrad into main_grad)."""
    _run_dense(C(256, 256, 4, 1, 2, 2), 40 * 256 - 100, 512, 1280, BF16, False)
    _run_dense(DEFAULT, 40 * 256 - 100, 512, 1280, BF16, True)


def test_dense_accumulate_partial_n_tile():
    """beta = 1 into bf16 with a partial N tile: no early old-C fetch, the old C is read at the stores (and a
    persistent config runs non-persistent)."""
    _run_dense(DEFAULT, 300, 720, 1024, BF16, True)
    _run_dense(C(256, 256, 2, 2, 2, 2, persist=True), 300, 720, 1024, BF16, True)


def test_gemm_mxfp8_random_snr():
    M, N, K = 1024, 2048, 4224
    g = torch.Generator(device=DEV).manual_seed(5)
    a, sa = quant_rows(torch.randn(M, K, device=DEV, generator=g) * torch.logspace(-2, 2, K, device=DEV))
    b, sb = quant_rows(torch.randn(N, K, device=DEV, generator=g))
    ref = gemm_ref(a, sa, b, sb)
    floor = snr_db(ref.to(BF16), ref)  # the bf16 output rounding alone: ~55.6 dB
    a, sa, b, sb = _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), pack_scales(sb)
    # the WMMA accumulation error is invisible next to the bf16 output rounding
    assert snr_db(gemm_mxfp8_gfx1250(a, sa, b, sb, config=KP1_C2X4), ref) > floor - 0.05
    assert snr_db(gemm_mxfp8_gfx1250(a, sa, b, sb, out_dtype=F32, config=DEFAULT), ref) > 60


def _grouped(lens, N, K, seed):
    """A row copy padded to 32 rows per expert (zero rows, scale 127) and b [G, N, K], int-valued, as the kernel
    takes them; the tight and the padded offsets; the exact result."""
    pads = [-(-n // 32) * 32 for n in lens]
    a, sa, _, _ = int_valued_operands(max(sum(pads), 1), 16, K, seed=seed)
    a, sa = a[: sum(pads)].clone(), sa[: sum(pads)].clone()
    offs = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    offs_p = torch.tensor([0] + torch.tensor(pads).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    for g, n in enumerate(lens):
        a[int(offs_p[g]) + n : int(offs_p[g + 1])] = 0
        sa[int(offs_p[g]) + n : int(offs_p[g + 1])] = 127
    bb = [int_valued_operands(N, 16, K, seed=seed + 1 + g)[:2] for g in range(len(lens))]
    b, sb = torch.stack([x[0] for x in bb]), torch.stack([x[1] for x in bb])
    ref = torch.zeros(sum(lens), N, dtype=torch.float64, device=DEV)
    for g, n in enumerate(lens):
        r0, rp = int(offs[g]), int(offs_p[g])
        ref[r0 : r0 + n] = gemm_ref(a[rp : rp + n], sa[rp : rp + n], b[g], sb[g])
    return _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), pack_scales(sb), offs, offs_p, ref.float()


# 32 experts, as production. m tiles of 256 rows per expert: 1 3 0 0 1 1 3, four times, then 2 0 1 0 (39). Of the
# 2x4 clusters (two consecutive m tiles share B) those that straddle two experts load B per workgroup, those inside
# an expert multicast it, and the last one has a dead second tile.
RAGGED = [1, 513, 0, 0, 64, 255, 700] * 4 + [300, 0, 2, 0]


def test_grouped_gemm_mxfp8():
    """Routed experts fwd / dgrad with ragged, one-row and empty experts, the default config (K >= 2048: c2x4); and
    no expert with rows (out untouched)."""
    N, K = 1024, 2048
    a, sa, b, sb, offs, offs_p, ref = _grouped(RAGGED, N, K, seed=7)
    out, check = _guarded((sum(RAGGED), N), BF16)
    grouped_gemm_mxfp8_gfx1250(a, sa, b, sb, offs, offs_p, sum(RAGGED), out=out)
    _eq(out, ref)
    check()
    zero = torch.zeros_like(offs)
    out, check = _guarded((64, N), BF16)
    grouped_gemm_mxfp8_gfx1250(a, sa, b, sb, zero, zero, 64, out=out)
    assert torch.isnan(out).all()
    check()


@pytest.mark.parametrize("K", [640, 2048], ids=["persistent", "c2x4"])
def test_grouped_gemm_mxfp8_many_experts(K):
    """48 experts (two 32-lane lookup chunks), ~2 m tiles each, with the default configs: K below 2048 runs
    persistent kpair 2 (a half last K step), K = 2048 the 2x4 clusters."""
    lens = [(53 * i) % 900 for i in range(48)]
    N = 1024
    a, sa, b, sb, offs, offs_p, ref = _grouped(lens, N, K, seed=11)
    out, check = _guarded((sum(lens), N), BF16)
    grouped_gemm_mxfp8_gfx1250(a, sa, b, sb, offs, offs_p, sum(lens), out=out)
    _eq(out, ref)
    check()


def _vk(lens, M, N, seed):
    """Variable-K operands over each expert's 128-padded tokens (padding: zero codes, PAD_SCALE for the 32-blocks
    with no token), as the kernel takes them; the offsets; the exact per-expert results [G, M, N]."""
    pads = [-(-n // 128) * 128 for n in lens]
    a, sa, b, sb = int_valued_operands(M, N, max(sum(pads), 128), seed=seed)
    offs = torch.tensor([0] + torch.tensor(pads).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    o = [int(x) for x in offs]
    for g, n in enumerate(lens):
        t0, t1 = o[g] + n, o[g + 1]
        a[:, t0:t1], b[:, t0:t1] = 0, 0
        sa[:, -(-t0 // 32) : t1 // 32], sb[:, -(-t0 // 32) : t1 // 32] = PAD_SCALE, PAD_SCALE
    ref = [
        gemm_ref(a[:, t0:t1], sa[:, t0 // 32 : t1 // 32], b[:, t0:t1], sb[:, t0 // 32 : t1 // 32])
        for t0, t1 in zip(o, o[1:])
    ]
    return (
        _slack(a),
        pack_scales(sa),
        _slack(preshuffle_b16(b)),
        pack_scales(sb),
        offs,
        torch.stack(ref).float(),
    )


def _vk_lens(G):
    """Empty experts and odd counts of 128 tokens (a half last K step)."""
    return [128 * (i % 5) + (0 if i % 7 == 3 else 37 * (i % 3)) for i in range(G)]


@pytest.mark.parametrize(
    "G, N, kinds",
    [
        (32, 2560, ((BF16, True), (BF16, False), (F32, True))),  # 2x2 clusters; fp32: not persistent
        (32, 2048, ((BF16, True), (BF16, False))),  # a narrow N into bf16: no cluster, 2 tiles per workgroup
        (
            40,
            2560,
            ((BF16, True),),
        ),  # more than 32 experts: the offsets are read per tile, not from a lane table
    ],
    ids=["G32", "G32-narrow", "G40"],
)
def test_grouped_gemm_mxfp8_variable_k(G, N, kinds):
    """Routed experts wgrad with an M tail and the default configs, as production runs them: beta = 1 into a bf16
    main_grad (persistent, the old C fetched early), bf16 out (persistent) and beta = 1 into fp32 (the old C read
    at the stores); an empty expert leaves its slice unchanged (or zero)."""
    M = 320
    a, sa, b, sb, offs, ref = _vk(_vk_lens(G), M, N, seed=13)
    for dt, acc in kinds:
        c0 = _old_c((G, M, N), dt, seed=G + N)
        out, check = _guarded((G, M, N), dt, c0 if acc else None)
        grouped_gemm_mxfp8_variable_k_gfx1250(a, sa, b, sb, offs, out_dtype=dt, out=out, accumulate=acc)
        _eq(out, (ref + c0.float()) if acc else ref)
        check()


def test_gemm_mxfp8_deterministic():
    """The production kernels repeat bit for bit (operands prepared once): dense c2x4, grouped-M c2x4 and the
    persistent variable-K wgrad into bf16."""
    g = torch.Generator(device=DEV).manual_seed(7)
    a, sa = quant_rows(torch.randn(1000, 7168, device=DEV, generator=g))
    b, sb = quant_rows(torch.randn(2048, 7168, device=DEV, generator=g))
    a, sa, b, sb = _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), pack_scales(sb)
    ga, gsa, gb, gsb, offs, offs_p, _ = _grouped(RAGGED, 1024, 2048, seed=7)
    va, vsa, vb, vsb, voffs, _ = _vk(_vk_lens(32), 320, 2560, seed=13)
    c0 = _old_c((32, 320, 2560), BF16, seed=7)

    def dense(out):
        gemm_mxfp8_gfx1250(a, sa, b, sb, out=out, config=KP1_C2X4)

    def grouped(out):
        grouped_gemm_mxfp8_gfx1250(ga, gsa, gb, gsb, offs, offs_p, sum(RAGGED), out=out)

    def vk(out):
        out.copy_(c0)
        grouped_gemm_mxfp8_variable_k_gfx1250(va, vsa, vb, vsb, voffs, out=out, accumulate=True)

    for call, shape in ((dense, (1000, 2048)), (grouped, (sum(RAGGED), 1024)), (vk, (32, 320, 2560))):
        first = torch.empty(shape, dtype=BF16, device=DEV)
        call(first)
        out = torch.empty_like(first)
        for _ in range(100):
            call(out)
            assert torch.equal(out, first)


def test_gemm_mxfp8_tail_at_allocation_end():
    """Operands that end exactly where their (10 MiB or more) allocation does, so a read past them could fault
    (best effort). With the default configs: a half last K step at kpair 2 reads half rows of a, nothing past it; a
    b without its slack makes the entry fall back to kpair 1 (never persistent); the variable-K wgrad's empty last
    expert reads inside the operands, and without any token nothing is launched."""
    M, N, K = 16384, 256, 640  # dense default: 128x128, kpair 2 (a half last K step)
    a, sa, b, sb = int_valued_operands(M, N, K, seed=1)
    ref = gemm_ref(a, sa, b, sb).float()
    a = _exact(a)
    assert a.numel() == 10 << 20 and KM._slack(a) == 0
    sa, sb, bs = pack_scales(sa), pack_scales(sb), preshuffle_b16(b)
    _eq(gemm_mxfp8_gfx1250(a, sa, _slack(bs), sb), ref)
    _eq(gemm_mxfp8_gfx1250(a, sa, _exact(bs), sb), ref)  # no slack: kpair 1
    # beta = 1 into bf16, K % 256 == 128: the persistent c2x2 default needs b's slack; without it kpair 1
    M, N, K = 4096, 3072, 4224
    a, sa, b, sb = int_valued_operands(M, N, K, seed=2)
    ref = gemm_ref(a, sa, b, sb).float()
    out = _old_c((M, N), BF16, seed=2)
    ref = ref + out.float()
    a, sa, b, sb = _slack(a), pack_scales(sa), _exact(preshuffle_b16(b)), pack_scales(sb)
    gemm_mxfp8_gfx1250(a, sa, b, sb, out=out, accumulate=True)
    _eq(out, ref)
    # grouped-M, K < 2048 (the persistent kpair-2 default), K % 256 == 128, b without its slack: kpair 1
    a, sa, b, sb, offs, offs_p, ref = _grouped([5000, 0, 6000, 4000], 512, 640, seed=3)
    _eq(grouped_gemm_mxfp8_gfx1250(a, sa, _exact(b), sb, offs, offs_p, 15000), ref)
    # variable-K, odd counts of 128 tokens and an empty last expert (its K step starts at T, the end of a and b)
    a, sa, b, sb, offs, ref = _vk([384, 128, 1408, 2048, 1152, 0], 2048, 2048, seed=4)
    a = _exact(a)
    for bs in (b, _exact(b)):  # kpair 2 with half steps; no slack: kpair 1
        _eq(grouped_gemm_mxfp8_variable_k_gfx1250(a, sa, bs, sb, offs), ref)
    out = grouped_gemm_mxfp8_variable_k_gfx1250(a[:, :0], sa[:, :0], b[:, :0], sb[:, :0], offs * 0)
    assert out.shape == ref.shape and not out.any()


def test_gemm_mxfp8_rejects():
    a, sa, b, sb = int_valued_operands(256, 256, 256)
    sa4, sb4, bs = pack_scales(sa), pack_scales(sb), preshuffle_b16(b)
    with pytest.raises(ValueError, match="16x16 blocks"):  # N % 16 != 0
        gemm_mxfp8_gfx1250(a, sa4, bs[:248], sb4)
    with pytest.raises(ValueError, match="multiple of 128"):  # (the quantizers pad K)
        gemm_mxfp8_gfx1250(a[:, :192], sa4, bs[:, :192].contiguous(), sb4)
    with pytest.raises(ValueError, match="packed"):  # raw scales [rows, K / 32]
        gemm_mxfp8_gfx1250(a, sa, bs, sb)
    with pytest.raises(ValueError, match="E8M0"):
        gemm_mxfp8_gfx1250(a, sa4.float(), bs, sb4)
    with pytest.raises(ValueError, match="persist=True needs"):
        gemm_mxfp8_gfx1250(a, sa4, bs, sb4, config=C(256, 256, 4, 1, 1, 1, persist=True))
    out = torch.empty(256 * 128 + 128, dtype=BF16, device=DEV).as_strided((256, 256), (128, 1))
    with pytest.raises(ValueError, match="row stride"):  # overlapping output rows
        gemm_mxfp8_gfx1250(a, sa4, bs, sb4, out=out)
    offs = torch.tensor([0, 128, 256, 256], dtype=torch.int64, device=DEV)
    with pytest.raises(ValueError, match="contiguous int64"):
        grouped_gemm_mxfp8_gfx1250(
            a, sa4, bs.view(1, 256, 256), sb4.view(1, 8, 2, 32, 4), offs[::2], offs[::2], 256
        )
