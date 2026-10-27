###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ref_softmax.py -- numpy reference of the shipped FlyDSL gfx1250 MLA forward (worktree 86a5f691), one head.

Op order copied from primus_turbo/flydsl/attention/gfx1250_mla/fwd_kernel.py and from its gfx1250 ISA
(ref/dump/mla_fwd_kernel_0/22_final_isa.s, dumped on the CPU by dump_fly_fwd.py), where LLVM contracted ops:

  per 16-row q tile (= FlyDSL's per-wave, per-q-tile ballot), per 64-kv tile t (kv rows 64t..64t+63):
    S  = K Q^T : 6 chained v_wmma_f32_16x16x32_bf16 (k = 0..191 in 32-steps), acc starts at +0
    s  = -inf where kv > q (bottom-right causal, Sq == Skv)
    lm = max over the 64 kv (exact; tree shape irrelevant)
    need  = fma(lm, c, -m_prev) > fp32(LOG2E)       [ISA: v_fma_f32 + v_cmp_lt_f32 0x3fb8aa3b]
    resc  = any(need) over the 16 rows (ballot)
    m     = resc ? max(m_prev, fp32(c * lm)) : m_prev
    corr  = exp2(m_prev - m)                          [v_sub_f32, v_exp_f32]
    p     = exp2(fma(s, c, -m))                       [v_pk_fma_f32 with neg modifier, v_exp_f32 (afn: bare)]
    P     = bf16_rne(p)                               [v_cvt_pk_bf16_f32]
    ls    = 3-ary tree ((a+b)+c, last pair a+b) over each lane's 32 p (kv order: kvt-major, then the lane's 8
            rows: lane half h holds kv 16 kvt + 8 h + [0, 8)), fp32 adds
    tot   = ls(h0) + ls(h1)
    d     = fma(corr, d_prev, tot)                    [v_add_f32 then v_fmac_f32]
    O     = resc ? O * corr : O                       [v_pk_mul_f32]
    O    += P V : 2 chained WMMAs per 64-kv tile (kv 0..31, 32..63)
  seed: m = -1e30 (finite), d = 0, O = 0
  epilogue: inv = d > 0 ? 1/d : 0 (IEEE division sequence: correctly rounded), out = bf16_rne(O * inv),
            lse = (m + log2(d)) * fp32(1/LOG2E), log2(d) = v_log_f32(ldexp(d, 32)) - 32 if d < 2^-126 else
            v_log_f32(d) - 0.
Fully masked tiles past the diagonal (FlyDSL runs some as masked / sync-only tiles) are exact no-ops for finite
inputs (corr = exp2(0) = 1, d = fma(1, d, 0), P = 0), so they are skipped here.

All fp32 roundings are exact emulations (products of fp32 are exact in fp64; sums / fmas use TwoSum + round-to-odd
in fp64, then one rounding to fp32).  Hardware-specific primitives are pluggable:
  --wmma exact  : D = RN_f32(C + sum of the 32 exact bf16 products)  (one rounding)          [default]
         seq    : D = fp32 sequential: acc = RN(acc + a_k b_k), k = 0..31
  --exp2 cr     : correctly rounded exp2  (v_exp_f32 is a ~1 ulp approximation: use --oracle npz from the GPU)
  --log2 cr     : correctly rounded log2
  --oracle F.npz: arrays exp2_in/exp2_out, log2_in/log2_out (fp32 bit patterns) measured on the GPU with the
                  bare v_exp_f32 / v_log_f32 (jobs/j7_ref.sh); inputs missing from the table are an error.
usage (library): ref_head(q[S,192] bf16-as-f32, k[S,192], v[S,128], scale) -> (out bf16 bits u16 [S,128], lse f32 [S])
usage (CLI self-test): ref_softmax.py --selftest [--s 512]"""

import argparse
import sys

import numpy as np

LOG2E = 1.4426950408889634
F32 = np.float32


# ------------------------------------------------------------------ exact fp32 emulation helpers
def _ro_to_f32(s, e):
    """s + e exactly (s = RN64(true), e = true - s), rounded to fp32 via round-to-odd in fp64."""
    s = np.asarray(s, np.float64)
    e = np.asarray(e, np.float64)
    bits = s.view(np.int64)
    odd = (bits & 1) == 1
    toward = np.nextafter(s, np.where(e > 0, np.inf, -np.inf))
    ro = np.where((e == 0) | odd | ~np.isfinite(s), s, toward)
    return ro.astype(np.float32)


def _two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    e = np.where(np.isfinite(s), e, 0.0)
    return s, e


def add_f32(a, b):
    """RN_f32(a + b) for fp32 a, b."""
    with np.errstate(invalid="ignore", over="ignore"):
        s, e = _two_sum(np.asarray(a, np.float64), np.asarray(b, np.float64))
        return _ro_to_f32(s, e)


def mul_f32(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        return (np.asarray(a, np.float64) * np.asarray(b, np.float64)).astype(
            np.float32
        )  # exact product, 1 rounding


def fma_f32(a, b, c):
    """RN_f32(a*b + c) for fp32 a, b, c (a*b exact in fp64)."""
    with np.errstate(invalid="ignore", over="ignore"):
        p = np.asarray(a, np.float64) * np.asarray(b, np.float64)
        s, e = _two_sum(p, np.asarray(c, np.float64))
        return _ro_to_f32(s, e)


def sum_exact_to_f32(c, terms_sum, terms_err=None):
    """RN_f32(c + T) where T = terms_sum (+ terms_err) is exact in fp64."""
    with np.errstate(invalid="ignore", over="ignore"):
        s, e = _two_sum(np.asarray(c, np.float64), terms_sum)
        if terms_err is not None:
            e = e + terms_err
        return _ro_to_f32(s, e)


def bf16_rne_bits(x):
    """fp32 -> bf16 round-to-nearest-even, returned as uint16 bit patterns (finite inputs)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def bf16_to_f32(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_bf16_f32(x):
    return bf16_to_f32(bf16_rne_bits(x))


# ------------------------------------------------------------------ hardware primitives (pluggable)
class Prims:
    def __init__(self, wmma="exact", exp2="cr", log2="cr", oracle=None):
        self.wmma_mode, self.exp2_mode, self.log2_mode = wmma, exp2, log2
        self.tab = {}
        if oracle is not None:
            z = np.load(oracle)
            for f in ("exp2", "log2"):
                if f + "_in" in z:
                    self.tab[f] = dict(
                        zip(z[f + "_in"].view(np.uint32).tolist(), z[f + "_out"].view(np.uint32).tolist())
                    )
        self.need = {"exp2": set(), "log2": set()}  # inputs seen (to build the oracle request)

    def _cr(self, fn, x):
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            return fn(np.asarray(x, np.float64)).astype(np.float32)

    def _lookup(self, f, x):
        x = np.asarray(x, np.float32)
        u = x.view(np.uint32)
        self.need[f].update(np.unique(u).tolist())
        if f not in self.tab:
            raise KeyError(f"no oracle table for {f}")
        t = self.tab[f]
        out = np.fromiter((t[v] for v in u.ravel().tolist()), np.uint32, count=u.size).reshape(u.shape)
        return out.view(np.float32)

    def exp2(self, x):
        """bare v_exp_f32 (FlyDSL: math.exp2 with afn)."""
        x = np.asarray(x, np.float32)
        if self.exp2_mode == "oracle":
            return self._lookup("exp2", x)
        self.need["exp2"].update(
            np.unique(x.view(np.uint32)).tolist()
        ) if self.exp2_mode == "collect" else None
        return self._cr(np.exp2, x)

    def vlog(self, x):
        """bare v_log_f32."""
        x = np.asarray(x, np.float32)
        if self.log2_mode == "oracle":
            return self._lookup("log2", x)
        self.need["log2"].update(
            np.unique(x.view(np.uint32)).tolist()
        ) if self.log2_mode == "collect" else None
        return self._cr(np.log2, x)

    def wmma(self, acc, a, b):
        """acc[M, N] f32 + a[M, 32] @ b[32, N] (bf16 values held in f32) -> f32 [M, N] (one WMMA per 16x16 block;
        the chunk of 32 along k is the WMMA's k)."""
        if self.wmma_mode == "exact":
            t = np.asarray(a, np.float64) @ np.asarray(b, np.float64)  # 32 exact products, exact fp64 sum*
            return sum_exact_to_f32(acc, t)
        if self.wmma_mode == "seq":
            out = np.asarray(acc, np.float32)
            a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
            for kk in range(a.shape[1]):
                out = add_f32(out, (a64[:, kk : kk + 1] * b64[kk : kk + 1, :]).astype(np.float32))
            return out
        raise ValueError(self.wmma_mode)


# ------------------------------------------------------------------ FlyDSL _tree_reduce on 32 leaves
def tree_sum(leaves):
    """leaves: list of arrays (same shape). FlyDSL _tree_reduce with op3 = (a+b)+c, op2 = a+b, fp32."""
    cur = list(leaves)
    while len(cur) > 1:
        nxt, i = [], 0
        while i < len(cur):
            n = len(cur) - i
            if n >= 3:
                nxt.append(add_f32(add_f32(cur[i], cur[i + 1]), cur[i + 2]))
                i += 3
            elif n == 2:
                nxt.append(add_f32(cur[i], cur[i + 1]))
                i += 2
            else:
                nxt.append(cur[i])
                i += 1
        cur = nxt
    return cur[0]


# ------------------------------------------------------------------ one head
def ref_head(q, k, v, scale, P=None, trace=None):
    """q [S,192], k [S,192], v [S,128]: bf16 values as float32 arrays (Sq == Skv == S, S % 64 == 0).
    Returns out bf16 bits [S,128] (uint16), lse fp32 [S]; trace (dict) collects per-tile state if given."""
    P = P or Prims()
    S = q.shape[0]
    assert q.shape == (S, 192) and k.shape == (S, 192) and v.shape == (S, 128) and S % 64 == 0
    c = F32(F32(scale) * F32(LOG2E))  # s_mul_f32 s6, s8, 0x3fb8aa3b
    thr = F32(LOG2E)
    m = np.full(S, -1.0e30, np.float32)
    d = np.zeros(S, np.float32)
    O = np.zeros((S, 128), np.float32)
    rows_all = np.arange(S)
    for t in range(S // 64):
        r0 = 64 * t  # rows that see kv tile t: r >= 64 t
        rows = rows_all[r0:]
        n = rows.size
        kt = k[r0 : r0 + 64]  # [64, 192]
        acc = np.zeros((n, 64), np.float32)  # S[q, kv] (FlyDSL holds S^T; same elementwise chain)
        for ks in range(6):
            acc = P.wmma(acc, q[r0:, 32 * ks : 32 * ks + 32], kt[:, 32 * ks : 32 * ks + 32].T)
        kvpos = 64 * t + np.arange(64)
        s = np.where(kvpos[None, :] > rows[:, None], np.float32(-np.inf), acc)
        lm = s.max(axis=1)
        mp = m[r0:]
        need = fma_f32(lm, c, -mp) > thr
        resc = need.reshape(-1, 16).any(axis=1).repeat(16)
        m_full = np.maximum(mp, mul_f32(c, lm))
        mn = np.where(resc, m_full, mp)
        corr = P.exp2(add_f32(mp, -mn))
        p = P.exp2(fma_f32(s, c, -mn[:, None]))
        Pb = to_bf16_f32(p)
        # lane-local sum trees: half h holds kv 16 kvt + 8 h + [0, 8), leaves kvt-major
        ls = []
        for h in (0, 1):
            leaves = [p[:, 16 * kvt + 8 * h + i] for kvt in range(4) for i in range(8)]
            ls.append(tree_sum(leaves))
        tot = add_f32(ls[0], ls[1])
        dn = fma_f32(corr, d[r0:], tot)
        On = np.where(resc[:, None], mul_f32(O[r0:], corr[:, None]), O[r0:])
        for kt2 in range(2):
            On = P.wmma(On, Pb[:, 32 * kt2 : 32 * kt2 + 32], v[64 * t + 32 * kt2 : 64 * t + 32 * kt2 + 32])
        m[r0:], d[r0:], O[r0:] = mn, dn, On
        if trace is not None:
            trace.setdefault("resc_tiles", 0)
            trace["resc_tiles"] += int(resc.reshape(-1, 16)[:, 0].sum())
            trace.setdefault("tiles", 0)
            trace["tiles"] += n // 16
    if trace is not None:  # the final fp32 state before the epilogue (m in log2 units, d, O)
        trace["m"], trace["d"], trace["O"] = m.copy(), d.copy(), O.copy()
    with np.errstate(divide="ignore"):
        inv = np.where(
            d > 0, (np.float32(1.0) / np.where(d > 0, d, np.float32(1.0))).astype(np.float32), np.float32(0)
        )
    out = bf16_rne_bits(mul_f32(O, inv[:, None]))
    small = d < np.float32(2.0**-126)
    lg = add_f32(
        P.vlog(np.where(small, mul_f32(d, np.float32(2.0**32)), d)),
        -np.where(small, np.float32(32), np.float32(0)),
    )
    lse = mul_f32(add_f32(m, lg), F32(1.0 / LOG2E))
    return out, lse


# ------------------------------------------------------------------ self tests (CPU only)
def _tie_ok(ex, g):
    """g is the correctly rounded fp32 of the rational ex (round to nearest, ties to even)."""
    from fractions import Fraction

    gf = Fraction(g)
    lo = Fraction(float(np.nextafter(np.float32(g), np.float32(-np.inf))))
    hi = Fraction(float(np.nextafter(np.float32(g), np.float32(np.inf))))
    dg, dl, dh = abs(ex - gf), abs(ex - lo), abs(ex - hi)
    if dg > dl or dg > dh:
        return False
    if dg == dl or dg == dh:
        return (np.float32(g).view(np.uint32) & 1) == 0
    return True


def _selftest(S, seed):
    from fractions import Fraction

    rng = np.random.default_rng(seed)
    # 1) fma / add emulation vs exact rationals
    a = (rng.standard_normal(4000) * np.exp2(rng.integers(-30, 30, 4000))).astype(np.float32)
    b = (rng.standard_normal(4000) * np.exp2(rng.integers(-30, 30, 4000))).astype(np.float32)
    cc = (rng.standard_normal(4000) * np.exp2(rng.integers(-60, 60, 4000))).astype(np.float32)
    cc[:500] = -(a[:500].astype(np.float64) * b[:500]).astype(np.float32)  # near-cancellation
    got = fma_f32(a, b, cc)
    bad = 0
    for x, y, z, g in zip(a.tolist(), b.tolist(), cc.tolist(), got.tolist()):
        ex = Fraction(x) * Fraction(y) + Fraction(z)
        # correctly rounded fp32 of ex: compare against neighbours
        gf = Fraction(g)
        lo = Fraction(float(np.nextafter(np.float32(g), np.float32(-np.inf))))
        hi = Fraction(float(np.nextafter(np.float32(g), np.float32(np.inf))))
        if abs(ex - gf) > abs(ex - lo) or abs(ex - gf) > abs(ex - hi):
            bad += 1
        elif abs(ex - gf) == abs(ex - lo) or abs(ex - gf) == abs(ex - hi):  # tie: must be even
            if np.float32(g).view(np.uint32) & 1:
                bad += 1
    print(f"SELFTEST fma_f32 vs exact rationals: {bad} / 4000 wrong")
    got = add_f32(a, cc)
    bad2 = sum(
        1
        for x, z, g in zip(a.tolist(), cc.tolist(), got.tolist())
        if float(np.float32(Fraction(x) + Fraction(z))) != g and not _tie_ok(Fraction(x) + Fraction(z), g)
    )
    print(f"SELFTEST add_f32 vs exact rationals: {bad2} / 4000 wrong; bf16 RNE:", end=" ")
    x = rng.standard_normal(100000).astype(np.float32)
    try:
        import torch

        tb = torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        print("mismatches", int(np.sum(tb != bf16_rne_bits(x))), "/ 100000 (torch CPU)")
    except Exception as e:  # noqa
        print("torch unavailable:", e)
    # 2) attention reference vs fp64 softmax attention
    scale = 0.1352337788608801
    qb = to_bf16_f32(rng.standard_normal((S, 192)).astype(np.float32))
    kb = to_bf16_f32(rng.standard_normal((S, 192)).astype(np.float32))
    vb = to_bf16_f32(rng.standard_normal((S, 128)).astype(np.float32))
    tr = {}
    out, lse = ref_head(qb, kb, vb, scale, Prims(), trace=tr)
    s64 = (qb.astype(np.float64) @ kb.astype(np.float64).T) * scale
    s64[np.triu_indices(S, 1)] = -np.inf
    mx = s64.max(1, keepdims=True)
    e = np.exp(s64 - mx)
    l64 = e.sum(1)
    o64 = (e @ vb.astype(np.float64)) / l64[:, None]
    lse64 = mx[:, 0] + np.log(l64)
    snr = lambda ref, x: 10 * np.log10(np.sum(ref**2) / np.sum((ref - x) ** 2))
    o = bf16_to_f32(out).astype(np.float64)
    print(
        f"SELFTEST ref_head S={S}: out SNR {snr(o64, o):.2f} dB, lse SNR {snr(lse64, lse.astype(np.float64)):.2f} dB, "
        f"rescaled q-tile x kv-tile steps {tr['resc_tiles']} / {tr['tiles']}"
    )
    out2, lse2 = ref_head(qb, kb, vb, scale, Prims(wmma="seq"))
    print(
        f"SELFTEST wmma=exact vs wmma=seq: out bits differ {int(np.sum(out != out2))} / {out.size}, "
        f"lse bits differ {int(np.sum(lse.view(np.uint32) != lse2.view(np.uint32)))} / {lse.size}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--s", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()
    if A.selftest:
        _selftest(A.s, A.seed)
    else:
        print(__doc__)
        sys.exit(0)
