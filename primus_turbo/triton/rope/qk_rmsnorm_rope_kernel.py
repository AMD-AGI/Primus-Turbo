###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton backend of ``fused_qkv_rmsnorm_rope``: per-head Q / K RMSNorm and rotate-half RoPE on a packed QKV
``[S, B, NG, W]``, per query group ``q: NPG x D | (gate: NPG x D) | k: D | v: D``.

Same numerics as the FlyDSL backend: ``n = round(x * rstd * gamma)`` (``round(x * rstd)`` without weights), then
``round(n_lo * cos - n_hi * sin)`` / ``round(n_hi * cos + n_lo * sin)`` from fp32 cos / sin; the backward rounds
the inverse RoPE to bf16 before the RMSNorm backward. Each program takes ``GP`` query groups of one token; a head
is handled as its two halves, so the rotation is a swap of halves. The weight gradients are fp32 partial sums,
one row per program, reduced by the caller in a fixed order.
"""

import triton
import triton.language as tl


@triton.jit
def _r(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rope_tables(COS, SIN, s, j, D: tl.constexpr, HALF: tl.constexpr, ROT: tl.constexpr):
    if ROT:
        cos_lo = tl.load(COS + s * D + j)
        cos_hi = tl.load(COS + s * D + HALF + j)
        sin_lo = tl.load(SIN + s * D + j)
        sin_hi = tl.load(SIN + s * D + HALF + j)
    else:
        cos_lo = tl.zeros([HALF], tl.float32)
        cos_hi = cos_lo
        sin_lo = cos_lo
        sin_hi = cos_lo
    return cos_lo, cos_hi, sin_lo, sin_hi


@triton.jit
def _gamma(G, j, HALF: tl.constexpr, HAS_W: tl.constexpr):
    if HAS_W:
        g_lo = tl.load(G + j).to(tl.float32)
        g_hi = tl.load(G + HALF + j).to(tl.float32)
    else:
        g_lo = tl.full([HALF], 1.0, tl.float32)
        g_hi = g_lo
    return g_lo, g_hi


@triton.jit
def _norm_rope_rows(src, dst, rows, valid, j, g_lo, g_hi, cos_lo, cos_hi, sin_lo, sin_hi, eps,
                    D: tl.constexpr, HALF: tl.constexpr, HAS_W: tl.constexpr, ROT: tl.constexpr):  # fmt: skip
    m = valid[:, None]
    lo = tl.load(src + rows[:, None] * D + j[None, :], mask=m, other=0.0).to(tl.float32)
    hi = tl.load(src + rows[:, None] * D + HALF + j[None, :], mask=m, other=0.0).to(tl.float32)
    r = tl.math.rsqrt((tl.sum(lo * lo, axis=1) + tl.sum(hi * hi, axis=1)) / D + eps)
    if HAS_W:
        nlo = _r(lo * r[:, None] * g_lo[None, :])
        nhi = _r(hi * r[:, None] * g_hi[None, :])
    else:
        nlo = _r(lo * r[:, None])
        nhi = _r(hi * r[:, None])
    if ROT:
        olo = nlo * cos_lo[None, :] - nhi * sin_lo[None, :]
        ohi = nhi * cos_hi[None, :] + nlo * sin_hi[None, :]
    else:
        olo = nlo
        ohi = nhi
    tl.store(dst + rows[:, None] * D + j[None, :], olo.to(tl.bfloat16), mask=m)
    tl.store(dst + rows[:, None] * D + HALF + j[None, :], ohi.to(tl.bfloat16), mask=m)


@triton.jit
def _norm_rope_rows_bwd(gsrc, xsrc, dst, rows, valid, j, g_lo, g_hi, cos_lo, cos_hi, sin_lo, sin_hi, eps,
                        D: tl.constexpr, HALF: tl.constexpr, HAS_W: tl.constexpr, ROT: tl.constexpr):  # fmt: skip
    m = valid[:, None]
    glo = tl.load(gsrc + rows[:, None] * D + j[None, :], mask=m, other=0.0).to(tl.float32)
    ghi = tl.load(gsrc + rows[:, None] * D + HALF + j[None, :], mask=m, other=0.0).to(tl.float32)
    if ROT:
        dlo = _r(glo * cos_lo[None, :] + ghi * sin_hi[None, :])
        dhi = _r(ghi * cos_hi[None, :] - glo * sin_lo[None, :])
    else:
        dlo = glo
        dhi = ghi
    lo = tl.load(xsrc + rows[:, None] * D + j[None, :], mask=m, other=0.0).to(tl.float32)
    hi = tl.load(xsrc + rows[:, None] * D + HALF + j[None, :], mask=m, other=0.0).to(tl.float32)
    r = tl.math.rsqrt((tl.sum(lo * lo, axis=1) + tl.sum(hi * hi, axis=1)) / D + eps)
    ulo = lo * r[:, None]
    uhi = hi * r[:, None]
    dulo = dlo * g_lo[None, :]
    duhi = dhi * g_hi[None, :]
    c = (tl.sum(ulo * dulo, axis=1) + tl.sum(uhi * duhi, axis=1)) / D
    tl.store(
        dst + rows[:, None] * D + j[None, :], ((dulo - ulo * c[:, None]) * r[:, None]).to(tl.bfloat16), mask=m
    )
    tl.store(
        dst + rows[:, None] * D + HALF + j[None, :],
        ((duhi - uhi * c[:, None]) * r[:, None]).to(tl.bfloat16),
        mask=m,
    )
    # weight-gradient contributions of these rows (masked rows are zero)
    return tl.sum(dlo * ulo, axis=0), tl.sum(dhi * uhi, axis=0)


@triton.jit
def qk_rmsnorm_rope_fwd_kernel(
    QKV, QG, KG, COS, SIN, QO, KO, B, eps,
    NG: tl.constexpr, NPG: tl.constexpr, NPG_P2: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
    GATED: tl.constexpr, HAS_W: tl.constexpr, ROT: tl.constexpr, GP: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0).to(tl.int64)  # (s, b, block of GP groups) flattened
    g0 = (pid % (NG // GP)) * GP
    sb = pid // (NG // GP)
    s = sb // B
    W: tl.constexpr = (NPG * (1 + GATED) + 2) * D
    KOFF: tl.constexpr = NPG * (1 + GATED) * D
    j = tl.arange(0, HALF)
    h = tl.arange(0, NPG_P2)
    hv = h < NPG
    one = tl.arange(0, 1)
    ov = one < 1
    cos_lo, cos_hi, sin_lo, sin_hi = _rope_tables(COS, SIN, s, j, D, HALF, ROT)
    qg_lo, qg_hi = _gamma(QG, j, HALF, HAS_W)
    kg_lo, kg_hi = _gamma(KG, j, HALF, HAS_W)
    for gi in tl.static_range(GP):
        g = g0 + gi
        base = QKV + (sb * NG + g) * W
        _norm_rope_rows(base, QO + (sb * NG * NPG + g * NPG) * D, h, hv, j, qg_lo, qg_hi,
                        cos_lo, cos_hi, sin_lo, sin_hi, eps, D, HALF, HAS_W, ROT)  # fmt: skip
        _norm_rope_rows(base + KOFF, KO + (sb * NG + g) * D, one, ov, j, kg_lo, kg_hi,
                        cos_lo, cos_hi, sin_lo, sin_hi, eps, D, HALF, HAS_W, ROT)  # fmt: skip


@triton.jit
def qk_rmsnorm_rope_bwd_kernel(
    QKV, DQ, DK, DGATE, DV, QG, KG, COS, SIN, DQKV, DQG, DKG, B, eps,
    sq_s, sq_b, sk_s, sk_b, sg_s, sg_b, sv_s, sv_b,
    NG: tl.constexpr, NPG: tl.constexpr, NPG_P2: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
    GATED: tl.constexpr, HAS_W: tl.constexpr, ROT: tl.constexpr, GP: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0).to(tl.int64)
    g0 = (pid % (NG // GP)) * GP
    sb = pid // (NG // GP)
    s = sb // B
    bb = sb % B
    W: tl.constexpr = (NPG * (1 + GATED) + 2) * D
    KOFF: tl.constexpr = NPG * (1 + GATED) * D
    j = tl.arange(0, HALF)
    h = tl.arange(0, NPG_P2)
    hv = h < NPG
    one = tl.arange(0, 1)
    ov = one < 1
    d = tl.arange(0, D)
    hd = tl.arange(0, NPG_P2 * D)
    hdv = hd < NPG * D
    cos_lo, cos_hi, sin_lo, sin_hi = _rope_tables(COS, SIN, s, j, D, HALF, ROT)
    qg_lo, qg_hi = _gamma(QG, j, HALF, HAS_W)
    kg_lo, kg_hi = _gamma(KG, j, HALF, HAS_W)
    dqg_lo = tl.zeros([HALF], tl.float32)
    dqg_hi = tl.zeros([HALF], tl.float32)
    dkg_lo = tl.zeros([HALF], tl.float32)
    dkg_hi = tl.zeros([HALF], tl.float32)
    for gi in tl.static_range(GP):
        g = g0 + gi
        base = QKV + (sb * NG + g) * W
        out = DQKV + (sb * NG + g) * W
        a_lo, a_hi = _norm_rope_rows_bwd(DQ + s * sq_s + bb * sq_b + g * NPG * D, base, out, h, hv, j, qg_lo, qg_hi,
                                         cos_lo, cos_hi, sin_lo, sin_hi, eps, D, HALF, HAS_W, ROT)  # fmt: skip
        b_lo, b_hi = _norm_rope_rows_bwd(DK + s * sk_s + bb * sk_b + g * D, base + KOFF, out + KOFF, one, ov, j,
                                         kg_lo, kg_hi, cos_lo, cos_hi, sin_lo, sin_hi, eps, D, HALF, HAS_W, ROT)  # fmt: skip
        dqg_lo += a_lo
        dqg_hi += a_hi
        dkg_lo += b_lo
        dkg_hi += b_hi
        # the gate and v gradients pass through into the packed gradient
        if GATED:
            tl.store(
                out + NPG * D + hd,
                tl.load(DGATE + s * sg_s + bb * sg_b + g * NPG * D + hd, mask=hdv),
                mask=hdv,
            )
        tl.store(out + KOFF + D + d, tl.load(DV + s * sv_s + bb * sv_b + g * D + d))
    if HAS_W:
        tl.store(DQG + pid * D + j, dqg_lo)
        tl.store(DQG + pid * D + HALF + j, dqg_hi)
        tl.store(DKG + pid * D + j, dkg_lo)
        tl.store(DKG + pid * D + HALF + j, dkg_hi)
