###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FP4 directions' options of the quantize_mx_* ops (fmt bits 16-24, ``fp4_options``): scale rule,
Hadamard along the contraction axis, 2-D 32x32 block scaling.

Every option combination is held, byte for byte, to a pure-PyTorch MXFP4 reference: the 32-group optionally
Hadamard-rotated (H32 or two H16, normalisation first, the packer's butterfly order), rounded to bf16, an E8M0
scale from the group amax (or the 32x32 tile amax) by the selected rule, and E2M1 round-to-nearest-even with
saturation at 6. Options must not touch an FP6 direction, and the defaults must not move a byte.
"""

import itertools

import pytest
import torch

from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as P
from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
    TS_A,
    MX_FMT_A4W4_GRAD,
    MX_FMT_PLAIN_ACT,
    MX_FMT_PLAIN_GRAD,
    MX_FMT_PLAIN_GRAD_SR,
    a4w4_logical,
    ts_fmt,
    ts_operand,
    fp4_options,
    plain_operand,
    quantize_mx,
    quantize_mx_dual,
    quantize_mx_ln_modulate,
)
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_data_region

SHAPES = [(256, 256), (512, 3072), (768, 1280)]
ROUNDS = ["rceil", "m0", "m1", "m2"]
HADS = ["h32", "none", "h16"]
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not hasattr(torch.ops.primus_turbo_cpp_extension, "quantize_mx_dual"):
        pytest.skip("Primus-Turbo built without the quantize_mx_* ops")


def _rand(rows, cols, seed=0):
    """Gaussian with a per-column magnitude sweep over five decades, so groups cover many exponents and
    every mantissa position of their amax."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, cols), device="cuda", generator=g)
    return (x * torch.logspace(-3, 2, cols, device="cuda")).to(torch.bfloat16)


def _butterfly(v, stages, first=0):
    """The packer's in-place network on [.., 32] fp32: stage h pairs (i, i + h) in blocks of 2h."""
    for s in range(first, stages):
        h = 1 << s
        b = v.reshape(*v.shape[:-1], 32 // (2 * h), 2, h)
        x0, x1 = b[..., 0, :], b[..., 1, :]
        v = torch.stack([x0 + x1, x0 - x1], -2).reshape(v.shape)
    return v


def _fma_f32(a, n, b):
    """fp32 fma(a, n, b) exactly (one rounding), for fp32 tensors a, b and an fp32 scalar n.

    a n is exact in float64 (24 + 24 significant bits), so a n + b has one float64 rounding; rounding that sum to
    fp32 directly could round twice. The float64 sum is therefore taken round-to-odd (TwoSum gives its exact error:
    if inexact, truncate toward zero and set the last bit), and round-to-odd at 53 bits >= 2 * 24 + 2 makes the
    following rounding to fp32 the correctly rounded one."""
    an, bd = a.double() * n.double(), b.double()
    s = an + bd
    bb = s - an
    e = (an - (s - bb)) + (bd - bb)
    bits = s.view(torch.int64)
    away = (e != 0) & ((e > 0) != (s > 0))  # s is above the exact sum in magnitude
    bits = (bits - away.long()) | (e != 0).long()
    return bits.view(torch.float64).float()


def _h32(v):
    """The FP4 emit's H32 on fp32 [.., 32], bit for bit: the normalisation fused into the first stage,
    fma(a, n, b n) and fma(a, n, -(b n)) (b n rounded to fp32), then the fp32 butterflies h = 2 .. 16."""
    n = torch.tensor(0.17677669529663687, dtype=torch.float32)
    a, b = v[..., 0::2], v[..., 1::2] * n.to(v.device)
    v = torch.stack([_fma_f32(a, n, b), _fma_f32(a, n, -b)], -1).reshape(v.shape)
    return _butterfly(v, 5, first=1)


def _ref_fp4(x, rnd="rceil", had="h32", tile2d=False, device="cpu"):
    """MXFP4 of x [rows, k] along k: (codes [rows, k] uint8, one per element, sign in bit 3; scales
    [rows, k / 32] uint8)."""
    rows, k = x.shape
    v = x.float().to(device).reshape(rows, k // 32, 32)
    if had == "h32":
        v = _h32(v)
    elif had == "h16":
        v = _butterfly(v * 0.25, 4)
    v = v.to(torch.bfloat16).float()
    if tile2d:
        t = v.abs().reshape(rows // 32, 32, k // 32, 32).amax(dim=(1, 3))
        amax = t.repeat_interleave(32, 0)
    else:
        amax = v.abs().amax(-1)
    amax = amax.clamp_min(1e-10)
    if rnd == "rceil":
        bits = (
            (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))
            .view(torch.int32)
            .long()
        )
        e = (bits >> 23) & 0xFF
        e = e + ((e < 0xFF) & ((bits & 0x7FFFFF) != 0)).long()
    else:
        bias = {"m0": 1 << 21, "m1": 1 << 22, "m2": 3 << 19}[rnd]
        e = (((amax.view(torch.int32).long() + bias) >> 23) & 0x1FF) - 129
        e = e.clamp(-127, 128) + 127
    scale = torch.where(
        e == 0, torch.tensor(2.0**-127), torch.pow(2.0, (e - 127).double()).float()
    )
    q = (v / scale[..., None]).abs()
    d = (q[..., None] - _E2M1.to(q.device)).abs()
    best = d.min(-1, keepdim=True).values
    # Ties to the even code; above 6 everything saturates (6 is then the unique nearest).
    tie_even = (d == best) & (torch.arange(8, device=q.device) % 2 == 0)
    code = torch.where(
        tie_even.any(-1), tie_even.float().argmax(-1), (d == best).float().argmax(-1)
    )
    code = (
        code | torch.signbit(v).long() << 3
    )  # the convert keeps the sign of a value that rounds to 0
    return code.reshape(rows, k).to(torch.uint8), e.to(torch.uint8)


def _unpack(codes):
    """[rows, k / 2] packed nibbles (first element in the low nibble) -> [rows, k]."""
    c = codes.cpu()
    return torch.stack([c & 0xF, c >> 4], -1).reshape(c.shape[0], -1)


def _plain(pack, rows, cols, which):
    if which == "row":
        c, s = plain_operand(pack[0], pack[1], rows, cols)
    else:
        c, s = plain_operand(pack[2], pack[3], cols, rows)
    return _unpack(c), s.cpu()


def _check(got, ref, what):
    (gc, gs), (rc, rs) = got, ref
    assert torch.equal(
        gs, rs
    ), f"{what}: scales differ at {(gs != rs).sum().item()} of {rs.numel()}"
    assert torch.equal(
        gc, rc
    ), f"{what}: codes differ at {(gc != rc).sum().item()} of {rc.numel()}"


def test_reference_is_the_default_packer():
    """Anchors the reference: with no options it reproduces the plain gradient pack (H32, RCEIL)."""
    _skip()
    rows, cols = 512, 3072
    x = _rand(rows, cols, seed=1)
    p = quantize_mx_dual(x, MX_FMT_PLAIN_GRAD)
    _check(_plain(p, rows, cols, "row"), _ref_fp4(x), "row")
    _check(_plain(p, rows, cols, "col"), _ref_fp4(x.t()), "col")


def test_all_zero_options_are_the_default():
    for fmt in (
        MX_FMT_A4W4_GRAD,
        MX_FMT_PLAIN_GRAD,
        MX_FMT_PLAIN_ACT,
        ts_fmt(TS_A, TS_A),
    ):
        assert fp4_options(fmt) == fmt, hex(fmt)


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("rnd,had", list(itertools.product(ROUNDS, HADS)))
def test_gradient_options_bit_exact(rows, cols, rnd, had):
    """Both FP4 directions of a gradient pack, each with its own options (the column direction gets the
    next rule / Hadamard in the lists, so a row / column mix-up cannot pass)."""
    _skip()
    x = _rand(rows, cols, seed=3)
    col_rnd, col_had = (
        ROUNDS[(ROUNDS.index(rnd) + 1) % 4],
        HADS[(HADS.index(had) + 1) % 3],
    )
    fmt = fp4_options(MX_FMT_PLAIN_GRAD, rnd, col_rnd, had, col_had)
    p = quantize_mx_dual(x, fmt)
    _check(_plain(p, rows, cols, "row"), _ref_fp4(x, rnd, had), "row")
    _check(_plain(p, rows, cols, "col"), _ref_fp4(x.t(), col_rnd, col_had), "col")


@pytest.mark.parametrize("rnd,had", [("m0", "none"), ("m2", "h16"), ("rceil", "none")])
def test_options_reach_every_fp4_layout(rnd, had):
    """The emit is shared, so the A4W4 and packed-FlyDSL layouts carry the same codes and scales as plain."""
    _skip()
    rows, cols = 512, 3072
    x = _rand(rows, cols, seed=4)
    ref = _ref_fp4(x, rnd, had)
    a = quantize_mx_dual(x, fp4_options(MX_FMT_A4W4_GRAD, rnd, rnd, had, had))
    c, s = a4w4_logical(a[0], a[1], rows, cols, is_b=False)
    _check((_unpack(c[:rows, : cols // 2]), s[:rows, : cols // 32].cpu()), ref, "a4w4")
    f = quantize_mx_dual(x, fp4_options(ts_fmt(TS_A, TS_A), rnd, rnd, had, had))
    fc, fs = ts_operand(f[0], f[1], rows, cols)
    assert torch.equal(_unpack(fc), ref[0]), "tilescale codes"
    # The packed slab is a permutation of the scales (its layout is tested elsewhere).
    got = fs.view(torch.uint8)[: rows * cols // 32].cpu().sort().values
    assert torch.equal(got, ref[1].reshape(-1).sort().values), "tilescale scales"


@pytest.mark.parametrize("rnd", ROUNDS)
def test_weight_column_options_leave_fp6_rows(rnd):
    """An activation / weight pack: options on its FP4 column direction; the FP6 forward rows unchanged."""
    _skip()
    rows, cols = 768, 1280
    x = _rand(rows, cols, seed=5)
    base = quantize_mx_dual(x, MX_FMT_PLAIN_ACT)
    p = quantize_mx_dual(
        x, fp4_options(MX_FMT_PLAIN_ACT, col_round=rnd, col_hadamard="h16")
    )
    assert torch.equal(
        mxfp6_data_region(p[0], rows, cols), mxfp6_data_region(base[0], rows, cols)
    )
    assert torch.equal(
        mxfp6_data_region(p[1], rows, cols, is_scale=True),
        mxfp6_data_region(base[1], rows, cols, is_scale=True),
    )
    _check(_plain(p, rows, cols, "col"), _ref_fp4(x.t(), rnd, "h16"), "col")


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("rnd", ["rceil", "m0"])
def test_tile2d(rows, cols, rnd):
    """2-D: one scale per 32x32 tile in both directions, and -- with no Hadamard -- the column codes are the
    row codes transposed, so one quantized weight serves the forward and the backward.
    """
    _skip()
    x = _rand(rows, cols, seed=6)
    fmt = fp4_options(MX_FMT_PLAIN_GRAD, rnd, rnd, "none", "none", tile2d=True)
    p = quantize_mx_dual(x, fmt)
    row, col = _plain(p, rows, cols, "row"), _plain(p, rows, cols, "col")
    _check(row, _ref_fp4(x, rnd, "none", tile2d=True), "row")
    _check(col, _ref_fp4(x.t(), rnd, "none", tile2d=True), "col")
    assert torch.equal(col[0], row[0].t())
    assert torch.equal(row[1][::32], col[1][::32].t())


def test_tile2d_single_direction():
    """The single-direction op stages the same tiles, so it gets the same 2-D scales."""
    _skip()
    rows, cols = 512, 1280
    x = _rand(rows, cols, seed=7)
    fmt = fp4_options(MX_FMT_PLAIN_GRAD, "m0", "m0", "none", "none", tile2d=True)
    dual = quantize_mx_dual(x, fmt)
    for axis, which in ((1, "row"), (0, "col")):
        c, s = quantize_mx(x, axis, fmt)
        got = _plain((c, s, c, s), rows, cols, which)
        assert all(
            torch.equal(u, v) for u, v in zip(got, _plain(dual, rows, cols, which))
        ), which


def test_sr_keeps_the_option_scale():
    """SR changes codes, not scales: an SR pack's scales are the RTN pack's under the same options."""
    _skip()
    rows, cols = 512, 3072
    x = _rand(rows, cols, seed=8)
    opts = dict(row_round="m0", col_round="m2", row_hadamard="none", col_hadamard="h16")
    rtn = quantize_mx_dual(x, fp4_options(MX_FMT_PLAIN_GRAD, **opts))
    sr = quantize_mx_dual(x, fp4_options(MX_FMT_PLAIN_GRAD_SR, **opts))
    for which in ("row", "col"):
        assert torch.equal(
            _plain(rtn, rows, cols, which)[1], _plain(sr, rows, cols, which)[1]
        ), which


def test_saturating_rules_clip_and_rceil_does_not():
    """A group whose amax mantissa is 1.7: m0 / m2 keep the scale low and clip it to 6, RCEIL / m1 do not."""
    _skip()
    x = torch.full((256, 256), 0.25, device="cuda")
    x[:, ::32] = 1.7
    x = x.to(torch.bfloat16)
    for rnd, clips in (("rceil", False), ("m1", False), ("m0", True), ("m2", True)):
        c, s = _plain(
            quantize_mx_dual(x, fp4_options(MX_FMT_PLAIN_GRAD, rnd, rnd, "none", "none")),
            256,
            256,
            "row",
        )
        assert (c[:, 0] == 7).all().item() == clips, rnd  # 7 = 6.0, the saturated code
        _check((c, s), _ref_fp4(x, rnd, "none"), rnd)


def test_bad_option_combinations_are_rejected():
    _skip()
    x = _rand(256, 256)
    with pytest.raises(RuntimeError, match="FP6"):
        quantize_mx_dual(x, fp4_options(MX_FMT_PLAIN_ACT, row_round="m0"))
    with pytest.raises(RuntimeError, match="FP6"):
        quantize_mx_dual(x, fp4_options(MX_FMT_PLAIN_ACT, row_hadamard="none"))
    with pytest.raises(RuntimeError, match="no Hadamard"):
        quantize_mx_dual(x, fp4_options(MX_FMT_PLAIN_GRAD, tile2d=True))
    with pytest.raises(RuntimeError, match="prologue"):
        mean, mod = torch.zeros(256, device="cuda"), _rand(1, 256)
        tile2d = fp4_options(
            MX_FMT_PLAIN_GRAD, "rceil", "rceil", "none", "none", tile2d=True
        )
        quantize_mx_ln_modulate(x, mean, mean + 1, mod, mod, False, tile2d)
    with pytest.raises(RuntimeError, match="unknown fmt bits"):
        quantize_mx_dual(x, 1 << 27)  # bit 26 is MX_FMT_COL_KOUTER


@pytest.mark.parametrize("row", [None, "fp4"])
def test_column_only_sr(row):
    """col_sr: the column direction (a backward copy) rounds stochastically, the row direction (an FP6 or FP4
    forward operand) stays round-to-nearest, byte for byte; column scales are the RTN pack's.
    """
    _skip()
    rows, cols = 512, 3072
    x = _rand(rows, cols, seed=9)
    col = P.ts_b_params(rows, cols, rows)
    base = P.ts_fmt(row=P.TS_A if row else None, col=col)
    rtn = quantize_mx_dual(x, base)
    sr1 = quantize_mx_dual(x, fp4_options(base, col_sr=True))
    sr2 = quantize_mx_dual(x, fp4_options(base, col_sr=True))
    if row:
        assert torch.equal(sr1[0], rtn[0]) and torch.equal(
            sr1[1], rtn[1]
        ), "row direction must stay RTN"
    else:
        assert torch.equal(
            mxfp6_data_region(sr1[0], rows, cols), mxfp6_data_region(rtn[0], rows, cols)
        )
    c_rtn, s_rtn = ts_operand(rtn[2], rtn[3], cols, rows)
    c1, s1 = ts_operand(sr1[2], sr1[3], cols, rows)
    c2, _ = ts_operand(sr2[2], sr2[3], cols, rows)
    assert torch.equal(s1, s_rtn), "SR keeps the scales"
    assert not torch.equal(c1, c2), "two launches draw independently"
    assert not torch.equal(c1, c_rtn)
    # Every SR code is the RTN code or a grid neighbour of the same sign (codes 0-7 per sign, magnitude ordered).
    u1, ur = _unpack(c1).long(), _unpack(c_rtn).long()
    mag1, magr = u1 & 7, ur & 7
    assert ((mag1 - magr).abs() <= 1).all()


def test_column_only_sr_rejected_off_tilescale():
    _skip()
    with pytest.raises(RuntimeError, match="column-only stochastic"):
        quantize_mx_dual(_rand(256, 256), fp4_options(MX_FMT_PLAIN_ACT, col_sr=True))


@pytest.mark.parametrize("bits", [4, 2])
@pytest.mark.parametrize("rows,cols", [(512, 3072), (256, 12288)])
def test_column_prob4_deferred_sr(rows, cols, bits):
    """col_prob: the FP4 tile column (K256-outer) emitted rounded down, with each code's 4-bit round-up probability
    in a second buffer of the same layout; fp4_prob_round finishes the stochastic rounding in place. The row
    direction and the column scales equal the round-to-nearest pack's; a floor code is the RTN code or one below
    it, as its probability says; a receiver rounds a code up with frequency p / 2^bits, deterministically per seed.
    """
    _skip()
    from primus_turbo.triton.quantization.fp4_prob_round import fp4_prob_round

    x = _rand(rows, cols, seed=11)
    fmt = P.ts_fmt(row=P.TS_A, col=P.ts_b_params(rows, cols, rows)) | P.MX_FMT_COL_KOUTER
    rp, rs = P.mx_dir_sizes(rows, cols, fmt, False)
    cp, cs = P.mx_dir_sizes(rows, cols, fmt, True)
    nb = lambda n: torch.empty(n, dtype=torch.uint8, device="cuda")  # noqa: E731
    rn, fl, prob = [nb(rp), nb(rs), nb(cp), nb(cs)], [nb(rp), nb(rs), nb(cp), nb(cs)], nb(cp if bits == 4 else cp // 2)
    P.quantize_mx_dual_out(x, *rn, fmt)
    P.quantize_mx_dual_out(x, *fl, fmt, col_prob=prob)
    assert torch.equal(fl[0], rn[0]) and torch.equal(fl[1], rn[1]) and torch.equal(fl[3], rn[3])

    def nib(t):
        return torch.stack([t & 15, t >> 4], -1).reshape(-1).long()

    def probs(t):  # per code, in code order
        if bits == 4:
            return nib(t)
        return torch.stack([(t >> (2 * i)) & 3 for i in range(4)], -1).reshape(-1).long()

    c_rn, c_fl, p = nib(rn[2]), nib(fl[2]), probs(prob)
    half = 1 << (bits - 1)
    m_rn, m_fl = c_rn & 7, c_fl & 7
    assert ((m_fl == m_rn) | (p >= half)).all() and ((m_fl + 1 == m_rn) | (p <= half)).all()
    assert (p[m_fl == 7] == 0).all()
    nz = (m_rn > 0) & (m_fl > 0)
    assert ((c_rn & 8) == (c_fl & 8))[nz].all()
    ups, S = torch.zeros_like(p, dtype=torch.float32), 128
    for seed in range(S):
        c = fl[2].clone()
        fp4_prob_round(c, prob, seed)
        d = (nib(c) & 7) - m_fl
        assert ((d == 0) | (d == 1)).all() and ((nib(c) & 8) == (c_fl & 8)).all()
        ups += d.float()
    assert abs((ups / S - p.float() / (1 << bits)).mean().item()) < 3e-3 and (ups[p == 0] == 0).all()
    a, b, c = fl[2].clone(), fl[2].clone(), fl[2].clone()
    fp4_prob_round(a, prob, 5), fp4_prob_round(b, prob, 5), fp4_prob_round(c, prob, 6)
    assert torch.equal(a, b) and not torch.equal(a, c)


def _ts_scale_idx(rows, groups, *, k128, is_b, ilv, kouter=False):
    """Byte offsets [rows, groups] of the tilescale scale slab (nt 4) -- a vectorised port of the kernels'
    ts_scale_byte, including the K256-outer column slab."""
    row = torch.arange(rows, device="cuda").view(-1, 1)
    kblk = torch.arange(groups, device="cuda").view(1, -1)
    kk = k128 >> 1
    ku_shift = 1 if kk % 2 == 0 else 0
    nw_shift = ku_shift + 1
    kdw, g = kblk >> 2, kblk & 3
    kh, rem = kdw >> nw_shift, kdw & ((1 << nw_shift) - 1)
    u, lo = rem >> 1, rem & 1
    grp, loc = row >> 6, row & 63
    if is_b:
        r_region, wi = (grp & 3) >> 1, (grp >> 2) * 2 + (grp & 1)
    else:
        wi, r_region = grp >> 1, grp & 1
    r, t = (loc >> 2, loc & 3) if ilv else (loc & 15, loc >> 4)
    last = r_region * 2 + lo
    if kouter:
        nwi = (rows + 255) // 256 * 2
        return ((kblk >> 3) * nwi + wi) * 1024 + g * 256 + r * 16 + last * 4 + t
    base = ((wi * kk + (kh << ku_shift)) * 64 + r) * 4
    return (base + u * 256 + g * 64 + last) * 4 + t


def _decode_ts6_rows(c0, c1, rsc, R, K):
    """K128-blocked MXFP6 rows (role B) -> (values [R, K] float64, exact; scale bytes [R, K / 32]). C0
    [R/16, K/128, 16, 64], C1 [R/32, K/128, 32, 32]; a group's 24 bytes are 32 little-endian 6-bit E2M3 codes."""
    nk = K // 128
    b0 = c0.view(R // 16, nk, 16, 64).permute(0, 2, 1, 3).reshape(R, nk, 4, 16)
    b1 = c1.view(R // 32, nk, 32, 32).permute(0, 2, 1, 3).reshape(R, nk, 4, 8)
    b = torch.cat([b0, b1], -1).reshape(R, K // 32, 8, 3).long()
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    codes = torch.stack([(word >> (6 * i)) & 63 for i in range(4)], -1).reshape(R, K)
    m, e = codes & 7, (codes >> 3) & 3
    mag = torch.where(e == 0, m.double() / 8, torch.exp2(e.double() - 1) * (1 + m.double() / 8))
    val = torch.where(codes & 32 > 0, -mag, mag)
    sbyte = rsc[_ts_scale_idx(R, K // 32, k128=nk, is_b=True, ilv=0)].long()
    return (val.view(R, K // 32, 32) * torch.exp2(sbyte.double() - 127).unsqueeze(-1)).reshape(R, K), sbyte


def _decode_fp4_col(cp, cs, R, K, ilv):
    """The K256-outer FP4 tile column of an [R, K] weight (the [K, R] operand contracting R) -> (codes [K, R],
    sign in bit 3; scale bytes [K, R / 32]; values [K, R] float64)."""
    codes = _unpack(cp.view(R // 256, K, 128).permute(1, 0, 2).reshape(K, R // 2)).cuda()
    s = cs[_ts_scale_idx(K, R // 32, k128=R // 128, is_b=True, ilv=ilv, kouter=True)]
    mag = _E2M1.double().cuda()[(codes & 7).long()]
    val = torch.where(codes & 8 > 0, -mag, mag).view(K, R // 32, 32) * torch.exp2(s.double() - 127).unsqueeze(-1)
    return codes, s, val.reshape(K, R)


@pytest.mark.parametrize("R,K", [(3072, 12288), (12288, 3072)])
@pytest.mark.parametrize("sr", [False, True], ids=["rn", "sr"])
def test_mxfp6_rows_to_fp4_col_unrot(R, K, sr):
    """mxfp6_tile_to_fp4_col on a weight's forward rows (K128-blocked MXFP6, role-B scales, H32 along K -- the
    dual weight pack's rows) with the dual pack's dgrad column format (RCEIL, H32 along the rows, K256-outer): the
    receiver rotates each decoded K group back, so its column is the column pack of H32_K(dequant(rows)).

    Reference: the rows decoded exactly, the emit's H32 along K in its fp32 operation order (the fused first stage
    emulated as an exactly rounded fma, see _fma_f32), then the column emit (H32 along R, bf16, group amax, RCEIL,
    E2M1 round to nearest even with saturation). RN: codes, scale bytes and decoded values bitwise. SR: the RN
    scales, codes the RN code or a grid neighbour, one draw per seed. Both: close to the column pack of the
    original bf16 weight, at that pack's error plus the FP6 rows'."""
    _skip()
    x = _rand(R, K, seed=12)
    # the weight pack of a dgrad GEMM [m, R] x [K, R]^T: ts6 rows (B), the FP4 column with the default options (RCEIL, H32)
    fmt = P.fp4_options(
        P.with_ts6_row(P.ts_fmt(col=P.ts_b_params(256, K, R)), True), col_round="rceil", col_hadamard="h32"
    ) | P.MX_FMT_COL_KOUTER
    ilv = P.ts_b_params(256, K, R)[2]
    rp, rs = P.mx_dir_sizes(R, K, fmt, False)
    cp, cs = P.mx_dir_sizes(R, K, fmt, True)
    nb = lambda n: torch.empty(n, dtype=torch.uint8, device="cuda")  # noqa: E731
    c0, c1, rsc = nb(rp * 2 // 3), nb(rp // 3), nb(rs)
    full = [nb(cp), nb(cs)]
    P.quantize_mx_dual_out(x, c0, rsc, *full, fmt, row_c1=c1)  # the owner's dual pack: rows + the direct column

    w_deq, _ = _decode_ts6_rows(c0, c1, rsc, R, K)
    v = w_deq.float()
    assert torch.equal(v.double(), w_deq), "FP6 values are exact in fp32"
    unrot = _h32(v.view(R, K // 32, 32)).reshape(R, K)  # the receiver's staged values
    ref_c, ref_s = _ref_fp4(unrot.t().contiguous(), device="cuda")

    got_c, got_s = nb(cp), nb(cs)
    seed = 0x2545F491
    P.mxfp6_tile_to_fp4_col(c0, c1, rsc, R, K, got_c, got_s, fmt | (P.MX_FMT_FP4_COL_SR if sr else 0), sr, seed)
    gc, gs, gv = _decode_fp4_col(got_c, got_s, R, K, ilv)
    assert torch.equal(gs.long(), ref_s.long()), f"scales differ at {(gs.long() != ref_s.long()).sum().item()}"
    mag_r = _E2M1.double().cuda()[(ref_c & 7).long()]
    ref_v = torch.where(ref_c & 8 > 0, -mag_r, mag_r).view(K, R // 32, 32) * torch.exp2(
        ref_s.double() - 127
    ).unsqueeze(-1)
    if not sr:
        assert torch.equal(gc, ref_c), f"codes differ at {(gc != ref_c).sum().item()} of {ref_c.numel()}"
        assert torch.equal(gv.view(-1).view(torch.int64), ref_v.view(-1).view(torch.int64)), "decoded values"
    else:
        mg, mr = (gc & 7).long(), (ref_c & 7).long()
        assert ((mg - mr).abs() <= 1).all()
        nz = (mg > 0) & (mr > 0)
        assert ((gc & 8) == (ref_c & 8))[nz].all()
        assert not torch.equal(gc, ref_c)
        again = [nb(cp), nb(cs)]
        P.mxfp6_tile_to_fp4_col(c0, c1, rsc, R, K, *again, fmt | P.MX_FMT_FP4_COL_SR, sr, seed)
        assert torch.equal(again[0], got_c) and torch.equal(again[1], got_s), "one draw per seed"

    # close to the direct column pack of the bf16 weight: both rotated back along R, against the weight
    hr = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(32)] for i in range(32)], device="cuda")
    hr = hr.double() / 32**0.5

    def err(vals, rows, cols, ref):
        back = (vals.view(rows, cols // 32, 32) @ hr).reshape(rows, cols)
        return ((back - ref).norm() / ref.norm()).item()

    xd = x.double()
    e_got, e_direct = err(gv, K, R, xd.t()), err(_decode_fp4_col(*full, R, K, ilv)[2], K, R, xd.t())
    e_fp6 = err(w_deq, R, K, xd)
    assert e_fp6 < 0.05
    # RN: the direct column's error and the rows' FP6 error, independent; SR's error is about sqrt(2) RN's
    assert e_got < (e_direct**2 + e_fp6**2) ** 0.5 * (1.6 if sr else 1.02)


def test_mxfp6_rows_to_fp4_col_rejects_unsupported():
    """The receiver writes only the K256-outer role-B column and only whole 256-row blocks: any other column format,
    or R / K not multiples of 256, raises instead of emitting wrong bytes."""
    _skip()
    nb = lambda n: torch.empty(n, dtype=torch.uint8, device="cuda")  # noqa: E731

    def call(R, K, kouter):
        fmt = P.fp4_options(
            P.with_ts6_row(P.ts_fmt(col=P.ts_b_params(256, K, R)), True), col_round="rceil", col_hadamard="h32"
        ) | (P.MX_FMT_COL_KOUTER if kouter else 0)
        rp, rs = P.mx_dir_sizes(R, K, fmt, False)
        cp, cs = P.mx_dir_sizes(R, K, fmt, True)
        P.mxfp6_tile_to_fp4_col(nb(rp * 2 // 3), nb(rp // 3), nb(rs), R, K, nb(cp), nb(cs), fmt, False, 1)

    call(512, 512, True)  # supported
    with pytest.raises(RuntimeError, match="K256-outer"):
        call(512, 512, False)
    with pytest.raises(RuntimeError, match="multiples of 256"):
        call(288, 512, True)


@pytest.mark.parametrize("sr", [False, True])
def test_mxfp6_tile_to_fp4_col(sr):
    """mxfp6_tile_to_fp4_col: the FP4 column (dgrad copy) made from a weight's K128-blocked MXFP6 rows -- here an
    unrotated plane with 2-D 32x32 tile scales -- is bitwise the column direction a dual pack of the dequantized
    weight emits (same format, the dual pack's column seed = its launch seed ^ 0x5bd1e995), stochastic or RN."""
    _skip()
    from primus_turbo.flydsl.utils.gemm_helper import mxfp4_packed_scale_byte
    from primus_turbo.pytorch.ops.quantization import set_sr_seed_next_pack

    R, K = 512, 3072
    x = _rand(R, K, seed=5)
    base = P.with_ts6_row(P.ts_fmt(col=P.ts_b_params(16384, K, R)), True)
    fmt = P.fp4_options(base, row_hadamard="none", col_hadamard="none", tile2d=True) | P.MX_FMT_COL_KOUTER
    if sr:
        fmt |= P.MX_FMT_FP4_COL_SR
    rp, rs = P.mx_dir_sizes(R, K, fmt, False)
    cp, cs = P.mx_dir_sizes(R, K, fmt, True)
    nb = lambda n: torch.empty(n, dtype=torch.uint8, device="cuda")  # noqa: E731
    c0, c1, rsc = nb(rp * 2 // 3), nb(rp // 3), nb(rs)
    P.quantize_mx_dual_out(x, c0, rsc, nb(0), nb(0), fmt, row_c1=c1)  # the plane: rows only

    nk = K // 128
    sidx = torch.tensor([[mxfp4_packed_scale_byte(r, g, k128=nk, b_ilv=0, is_b=True) for g in range(K // 32)]
                         for r in range(R)], device="cuda")
    assert torch.equal(sidx, _ts_scale_idx(R, K // 32, k128=nk, is_b=True, ilv=0)), "scale index port"
    w_deq, sbyte = _decode_ts6_rows(c0, c1, rsc, R, K)
    assert (sbyte.view(R // 32, 32, K // 32) == sbyte.view(R // 32, 32, K // 32)[:, :1]).all(), "tile scales"
    assert ((w_deq - x.double()).norm() / x.double().norm()).item() < 0.05
    wb = w_deq.to(torch.bfloat16)
    assert torch.equal(wb.double(), w_deq), "FP6 values are exact in bf16"

    seed = 1234567
    ref_c, ref_s = nb(cp), nb(cs)
    set_sr_seed_next_pack(seed)
    # the dequantized weight's column direction with per-32 scales (the plane's tile2d bit would give the column its
    # 32x32 tile scales; the receiver re-derives each column block's own, which the tile scale bounds)
    P.quantize_mx_dual_out(wb, nb(0), nb(0), ref_c, ref_s, fmt & ~P.MX_FMT_FP4_TILE2D)
    got_c, got_s = nb(cp), nb(cs)
    P.mxfp6_tile_to_fp4_col(c0, c1, rsc, R, K, got_c, got_s, fmt, sr, seed ^ 0x5BD1E995)
    assert torch.equal(got_s, ref_s) and torch.equal(got_c, ref_c)
