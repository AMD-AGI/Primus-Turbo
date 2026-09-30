#!/usr/bin/env python3
"""dkdv_divfree bounds + equivalence proof (CPU only, no torch).

For every workgroup of k_dkdv / k_dkdv_sp at each shape, causal on and off, this
replays the loop-index arithmetic of r29 (qi = ii // G, clamped jj // G) and of the
division-free arm (carried (qc, gc) counters with a wrap at G, clamped jj reusing the
wrap) and asserts:
  1. every body gets the IDENTICAL (qt, gh, qt_n, gh_n) tuple in both versions
     (=> identical addresses, identical accumulation order => bitwise identical);
  2. every Q/dO/LSE/delta address _ldqd issues (all 32 lanes, hh, dt, both halves)
     lies inside its tensor's true extent, including the clamped prefetch at n-1.
Covers qloop_mask, qloop_full (KV_U2 = False, shipped) and qloop_full2 + qloop_tail
(KV_U2 = True).  Python // == floor division; all operands are >= 0 here, so it equals
the kernel's signed truncating division (asserted).
"""
import sys

D, DV8, NDT, BLOCK_KV = 128, 16, 4, 32
SHAPES = {  # name: (B, S, Hq, Hkv)
    "prod": (4, 8192, 32, 8),
    "fast": (1, 1024, 8, 2),
    "toy": (1, 128, 2, 1),
}


def sdiv(a, b):
    assert a >= 0 and b > 0
    return a // b


def wrap(qc, gc, G):
    g1 = gc + 1
    return (qc, g1) if g1 < G else (qc + 1, 0)


# ---------------- r29 loops ----------------
def r29_mask(n, qt0, G):
    return [(qt0 + sdiv(ii, G), ii - sdiv(ii, G) * G, None, None) for ii in range(n)]


def r29_full(n, qt0, G):
    out = []
    for ii in range(n):
        qi = sdiv(ii, G)
        jj = ii + 1
        jj = jj if jj < n else n - 1
        qj = sdiv(jj, G)
        out.append((qt0 + qi, ii - qi * G, qt0 + qj, jj - qj * G))
    return out


def r29_full2(n2, qt0, G):
    out = []
    for it in range(n2):
        ii = it * 2
        i1 = ii + 1
        jj = ii + 2
        jj = jj if jj < 2 * n2 else 2 * n2 - 1
        qi, q1, qj = sdiv(ii, G), sdiv(i1, G), sdiv(jj, G)
        out.append((qt0 + qi, ii - qi * G, qt0 + q1, i1 - q1 * G))
        out.append((qt0 + q1, i1 - q1 * G, qt0 + qj, jj - qj * G))
    return out


def tail(n, i0, qt0, G):  # unchanged in both versions
    return [(qt0 + sdiv(ii, G), ii - sdiv(ii, G) * G, None, None)
            for ii in (i0 + k for k in range(n))]


# ---------------- divfree loops ----------------
def df_mask(n, qt0, G):
    out, qc, gc = [], 0, 0
    for ii in range(n):
        qn, gn = wrap(qc, gc, G)
        out.append((qt0 + qc, gc, None, None))
        qc, gc = qn, gn
    return out


def df_full(n, qt0, G):
    out, qc, gc = [], 0, 0
    for ii in range(n):
        qn, gn = wrap(qc, gc, G)
        live = ii + 1 < n
        qj, gj = (qn, gn) if live else (qc, gc)
        out.append((qt0 + qc, gc, qt0 + qj, gj))
        qc, gc = qn, gn
    return out


def df_full2(n2, qt0, G):
    out, qi, gi = [], 0, 0
    for it in range(n2):
        ii = it * 2
        q1, g1 = wrap(qi, gi, G)
        q2, g2 = wrap(q1, g1, G)
        live = ii + 2 < 2 * n2
        qj, gj = (q2, g2) if live else (q1, g1)
        out.append((qt0 + qi, gi, qt0 + q1, g1))
        out.append((qt0 + q1, g1, qt0 + qj, gj))
        qi, gi = q2, g2
    return out


def clampqt(t, nqt2):
    t = t if t < nqt2 else nqt2 - 1
    return t if t >= 0 else 0


def ldqd_check(qt, gh, B, Sq, Hq, Hkv, G, bat, hkv):
    """Every address _ldqd(qt, gh) issues, all lanes; returns (min, max) per buffer."""
    assert 0 <= gh < G
    qh = hkv * G + gh
    rs_q = Hq * DV8
    base_q = bat * Sq * rs_q + qh * DV8
    base_l = (bat * Hq + qh) * Sq
    q0 = qt * 32
    nq_v8 = B * Sq * Hq * DV8          # vec8 records in Q / dO
    nl = B * Hq * Sq                   # fp32 records in LSE / delta
    for lane in range(32):
        row, half = lane % 16, lane // 16
        for hh in range(2):
            qg = q0 + hh * 16 + row
            a = base_l + qg
            assert 0 <= a < nl, ("lse/delta OOB", qt, gh, lane, a, nl)
            qh0 = q0 + hh * 16
            for dt in range(NDT):
                t = base_q + (qh0 + row) * rs_q + half + dt * 4
                for tt in (t, t + 2):
                    assert 0 <= tt < nq_v8, ("q/do OOB", qt, gh, lane, tt, nq_v8)


def run_shape(name, B, S, Hq, Hkv):
    Sq = Skv = S
    G = Hq // Hkv
    nqt = Sq // 16
    nqt2 = nqt // 2
    cshift = Skv - Sq
    wgs = (Skv // BLOCK_KV) * Hkv * B
    nsp = 1
    while wgs * nsp < 2048 and nsp < 16:
        nsp *= 2
    iters = 0
    pf_checked = set()
    for causal in (0, 1):
        for bid in range(Skv // BLOCK_KV):
            kv0 = bid * BLOCK_KV
            _c = kv0 - cshift
            qp_start = (0 if _c < 0 else _c) // 32
            qp_start = qp_start if causal else 0
            nqp_eff = nqt2 - qp_start
            _u = kv0 + BLOCK_KV - 1 - cshift
            _qsf = 0 if _u < 0 else (_u + 31) // 32
            _qsf = min(_qsf, nqt2)
            _nm = max(0, min(_qsf - qp_start, nqp_eff))
            nmaskp = _nm if causal else 0
            PARTIAL = nsp > 1
            for sp in range(nsp if PARTIAL else 1):
                if PARTIAL:
                    _fn = max(0, nqp_eff - nmaskp)
                    _ch = (_fn + nsp - 1) // nsp
                    _off = sp * _ch
                    _cnt = max(0, min(_fn - _off, _ch))
                    _mk = 0 if sp != 0 else nmaskp
                    qt0f = qp_start + nmaskp + _off
                    nmask, nfull = G * _mk, G * _cnt
                else:
                    qt0f = qp_start + nmaskp
                    nmask, nfull = G * nmaskp, G * (nqp_eff - nmaskp)
                a, b = r29_mask(nmask, qp_start, G), df_mask(nmask, qp_start, G)
                assert a == b, ("mask mismatch", name, causal, bid, sp)
                a, b = r29_full(nfull, qt0f, G), df_full(nfull, qt0f, G)
                assert a == b, ("full mismatch", name, causal, bid, sp)
                # KV_U2 path
                n2 = nfull // 2
                a2 = r29_full2(n2, qt0f, G) + tail(nfull - 2 * n2, 2 * n2, qt0f, G)
                b2 = df_full2(n2, qt0f, G) + tail(nfull - 2 * n2, 2 * n2, qt0f, G)
                assert a2 == b2, ("full2 mismatch", name, causal, bid, sp)
                assert [x[:2] for x in a2] == [x[:2] for x in a], "U2 order differs"
                # every issued address in range: bodies (mask, tail) load (qt, gh);
                # full bodies prefetch (qt_n, gh_n); prologue loads clampqt(qt0f), gh 0.
                loads = {(clampqt(qt0f, nqt2), 0)}
                loads |= {(x[0], x[1]) for x in r29_mask(nmask, qp_start, G)}
                loads |= {(x[2], x[3]) for x in b}
                loads |= {(x[2], x[3]) for x in b2 if x[2] is not None}
                loads |= {(x[0], x[1]) for x in b2 if x[2] is None}
                for (qt, gh) in loads:
                    assert 0 <= qt < nqt2, ("qt OOB", name, causal, bid, sp, qt)
                    assert 0 <= gh < G
                    pf_checked.add((qt, gh))
                iters += nmask + nfull
    # address enumeration over every (qt, gh) that any workgroup issues, every
    # (bat, hkv) corner and every lane.
    for (qt, gh) in sorted(pf_checked):
        for bat in sorted({0, B - 1}):
            for hkv in sorted({0, Hkv - 1}):
                ldqd_check(qt, gh, B, Sq, Hq, Hkv, G, bat, hkv)
    print(f"{name:5s} B{B} S{S} Hq{Hq} Hkv{Hkv} G{G} nsp{nsp}: {iters} body iterations "
          f"(x2 causal modes) identical to r29; {len(pf_checked)} distinct (qt,gh) "
          f"loads, all lanes in bounds")


if __name__ == "__main__":
    for k, v in SHAPES.items():
        run_shape(k, *v)
    # the wrap itself, exhaustively against divmod for small G / long runs
    for G in range(1, 17):
        qc, gc = 0, 0
        for ii in range(5000):
            assert (qc, gc) == divmod(ii, G)
            qc, gc = wrap(qc, gc, G)
    print("wrap == divmod for G in 1..16, ii < 5000")
    print("PASS")
    sys.exit(0)
