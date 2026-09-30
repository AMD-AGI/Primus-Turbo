"""CPU bounds + coverage proof for the dkdv_flip arm (k_dkdv / k_dkdv_sp, DKDV_FLIP=True).

Host-only (python3 + numpy, no torch, no GPU). It replays _dkdv_impl's control flow
exactly (qp_start / nmaskp / split-K chunking / prologue _clampqt / qloop_full's jj clamp,
KV_U2=False) and enumerates EVERY workgroup, iteration, lane and vector element of each
index expression the arm changed:

  (A) LSE/delta vec4 loads in _ldqd (every call: masked loop, prologue, full-loop prefetch)
        t4 = (bat*Hq + qh)*(Sq/4) + qt*8 + half*2 + hh*4 + u        (vec4 units)
      must satisfy 0 <= 4*t4 and 4*t4+3 < B*Hq*Sq, and element e of vec (hh,u) must be
      exactly LSE[bat, qh, q] for the C-layout q = qt*32 + hh*16 + half*8 + (4u+e).
      The Q/dO b128 loads keep r29's index but are re-checked on the same qt set.
  (B) causal predicate with swapped roles: lane l / element si of C(hh, kh) is
      (q = q0 + hh*16 + (l//16)*8 + si, kv = kv0 + kh*16 + l%16); the 32x32 tile must be
      covered exactly once and the mask `kv > q + cshift` must equal the reference.
      Also checks that the B operand of dV^T/dK^T (concat of hh0, hh1 C registers) has
      the gfx1250 16x16x32 16-bit fragment layout: lane l holds N = l%16 and element e
      holds K = (e//8)*16 + (l//16)*8 + e%8.
  (C) bf16 epilogue (PARTIAL=False): vec8 index
        t = base_kv + (kv0 + kh*16 + row)*rs_kv + half + dtile*2
      in-bounds, and every dK/dV element written exactly once over the whole grid.
  (D) fp32 split-K epilogue (PARTIAL=True): vec4 index
        t4 = base_o/4 + (kv0 + kh*16 + row)*Hkv*(D/4) + half*2 + dtile*4 + u
      in-bounds of [nsp, B, Skv, Hkv, D] and every element written exactly once.
Every intermediate is also checked to fit int32.
"""
import sys
import numpy as np

D = 128
DV8 = D // 8
NKV = 2
NDO = D // 16
NDT = D // 32
BLOCK_KV = 32
I32 = 2 ** 31


def nsp_rule(B, Skv, Hkv):
    wgs = (Skv // BLOCK_KV) * Hkv * B
    nsp = 1
    while wgs * nsp < 2048 and nsp < 16:
        nsp *= 2
    return nsp


def ldqd_calls(Sq, Skv, G, causal, nsp):
    """{(bid, sp): [(qt, gh, kind)]} of every _ldqd call, replaying _dkdv_impl."""
    nqt = Sq // 16
    nqt2 = nqt // 2
    cshift = Skv - Sq
    clampqt = lambda t: max(0, min(t, nqt2 - 1))
    out = {}
    for bid in range(Skv // BLOCK_KV):
        kv0 = bid * BLOCK_KV
        c = kv0 - cshift
        qp_start = (0 if c < 0 else c) // 32          # c >= 0 here, // == trunc
        qp_start = qp_start if causal else 0
        nqp_eff = nqt2 - qp_start
        u = kv0 + BLOCK_KV - 1 - cshift
        qsf = 0 if u < 0 else (u + 31) // 32
        qsf = min(qsf, nqt2)
        nm = min(max(qsf - qp_start, 0), nqp_eff)
        nmaskp = nm if causal else 0
        for sp in range(nsp):
            calls = []
            if nsp > 1:
                fn = max(nqp_eff - nmaskp, 0)
                ch = (fn + nsp - 1) // nsp
                off = sp * ch
                cnt = min(max(fn - off, 0), ch)
                mk = 0 if sp != 0 else nmaskp
                qt0 = qp_start + nmaskp + off
                nmask_it, qtm0 = G * mk, qp_start
                nfull, qtf0 = G * cnt, qt0
                pc0 = clampqt(qt0)
            else:
                nmask_it, qtm0 = G * nmaskp, qp_start
                nfull, qtf0 = G * (nqp_eff - nmaskp), qp_start + nmaskp
                pc0 = clampqt(qp_start + nmaskp)
            for ii in range(nmask_it):                 # qloop_mask: own load
                qi = ii // G
                calls.append((qtm0 + qi, ii - qi * G, "mask"))
            calls.append((pc0, 0, "prologue"))
            for ii in range(nfull):                    # qloop_full: prefetch of jj
                jj = ii + 1
                jj = jj if jj < nfull else nfull - 1
                qj = jj // G
                calls.append((qtf0 + qj, jj - qj * G, "prefetch"))
                qi = ii // G
                calls.append((qtf0 + qi, ii - qi * G, "consume"))  # not a load; checked below
            out[(bid, sp)] = (calls, nmaskp, qp_start)
    return out


def check_fragment_layout():
    lanes = np.arange(32)
    row, half = lanes % 16, lanes // 16
    # (B) C layout -> P/dS as B operand of dV^T/dK^T.
    for kh in range(NKV):
        cov = np.zeros((32, 32), np.int32)             # [q in pair, kv in tile]
        for hh in range(2):
            for si in range(8):
                q = hh * 16 + half * 8 + si
                kv = kh * 16 + row
                np.add.at(cov, (q, kv), 1)
        assert (cov[:, kh * 16:(kh + 1) * 16] == 1).all()
        assert (cov[:, (1 - kh) * 16:(2 - kh) * 16] == 0).all()
    for e in range(16):
        hh, si = e // 8, e % 8                         # concat order: hh0 then hh1
        q_from_c = hh * 16 + half * 8 + si
        k_frag = (e // 8) * 16 + half * 8 + e % 8
        assert (q_from_c == k_frag).all()
    # dV^T accumulator: lane l element si = C[M = half*8+si (d), N = l%16 (kv)] --
    # 8 consecutive d of one kv row, which is what (C)/(D) store.
    print("layout: C(hh,kh) covers each (q,kv) of the 32x32 tile once; "
          "concat(hh0,hh1) == 16x32 B fragment K order: OK")


def check_mask(Sq, Skv, causal, calls):
    cshift = Skv - Sq
    lanes = np.arange(32)
    row, half = lanes % 16, lanes // 16
    n = 0
    for (bid, sp), (cl, nmaskp, qp_start) in calls.items():
        kv0 = bid * BLOCK_KV
        for qt, gh, kind in cl:
            if kind != "mask" or gh != 0:
                continue
            q0 = qt * 32
            for hh in range(2):
                qcs = q0 + hh * 16 + half * 8 + cshift
                for kh in range(NKV):
                    kvl = kv0 + kh * 16 + row
                    for si in range(8):
                        pred = (kvl > qcs + si) & (causal != 0)
                        q = q0 + hh * 16 + half * 8 + si
                        ref = (kvl > q + cshift) & bool(causal)
                        assert (pred == ref).all()
                        n += 1
        # the full loop's pairs must be entirely unmasked (mask-split invariant)
        for qt, gh, kind in cl:
            if kind == "consume" and causal:
                assert kv0 + BLOCK_KV - 1 <= qt * 32 + cshift, (bid, qt)
    return n


def run(B, Sq, Skv, Hq, Hkv, causal, nsp):
    G = Hq // Hkv
    assert Sq % 64 == 0 and Skv % 32 == 0 and Sq % 8 == 0
    nqt2 = Sq // 32
    Sq4 = Sq // 4
    nl = B * Hq * Sq                                   # fp32 elements of LSE/delta
    nq8 = B * Sq * Hq * DV8                            # vec8 records of Q/dO
    nkv_el = B * Skv * Hkv * D
    rs_q, rs_kv = Hq * DV8, Hkv * DV8
    for v in (nl * 4, nq8 * 16, nkv_el * 2, nsp * nkv_el * 4):
        assert v < I32, "byte extent overflows int32"
    calls = ldqd_calls(Sq, Skv, G, causal, nsp)

    # ---- (A) LSE/delta vec4 + Q/dO b128, every call ----
    qg = set()
    for cl, _, _ in calls.values():
        for qt, gh, kind in cl:
            if kind == "consume":
                assert 0 <= qt < nqt2, ("consumed pair out of range", qt)
                continue
            assert 0 <= qt < nqt2, ("_ldqd pair out of range", kind, qt)
            assert 0 <= gh < G
            qg.add((qt, gh))
    qg = np.array(sorted(qg), np.int64)
    lanes = np.arange(32, dtype=np.int64)
    row, half = lanes % 16, lanes // 16
    bat = np.arange(B, dtype=np.int64)
    hkv = np.arange(Hkv, dtype=np.int64)
    # shape: [bat, hkv, call, lane]
    qt = qg[:, 0][None, None, :, None]
    gh = qg[:, 1][None, None, :, None]
    qh = hkv[None, :, None, None] * G + gh
    bb = bat[:, None, None, None]
    base_l4 = (bb * Hq + qh) * Sq4 + qt * 8 + half * 2
    lmax, lmin = -1, 1 << 62
    for hh in range(2):
        for u in range(2):
            t4 = base_l4 + hh * 4 + u
            assert t4.max() < I32
            for e in range(4):
                el = 4 * t4 + e
                q = qt * 32 + hh * 16 + half * 8 + 4 * u + e
                assert (el == (bb * Hq + qh) * Sq + q).all(), "vec4 element != LSE[b,qh,q]"
                assert (q < Sq).all()
            lmax = max(lmax, int((4 * t4 + 3).max()))
            lmin = min(lmin, int((4 * t4).min()))
    assert lmin >= 0 and lmax < nl, (lmin, lmax, nl)
    # Q/dO (index unchanged from r29, re-checked on the same call set)
    base_q = bb * Sq * rs_q + qh * DV8
    qmax = -1
    for hh in range(2):
        for dt in range(NDT):
            t = base_q + (qt * 32 + hh * 16 + row) * rs_q + half + dt * 4
            for off in (0, 2):
                assert (t + off).min() >= 0
                qmax = max(qmax, int((t + off).max()))
    assert qmax < nq8

    # ---- (B) mask ----
    nm = check_mask(Sq, Skv, causal, calls)

    # ---- (C)/(D) epilogue: every element written exactly once ----
    nbid = Skv // BLOCK_KV
    bidv = np.arange(nbid, dtype=np.int64)
    kv0 = (bidv * BLOCK_KV)[None, None, :, None]
    bb3 = bat[:, None, None, None]
    hk3 = hkv[None, :, None, None]
    rw = row[None, None, None, :]
    hf = half[None, None, None, :]
    if nsp == 1:
        cov = np.zeros(nkv_el, np.int8)
        base_kv = bb3 * Skv * rs_kv + hk3 * DV8
        for kh in range(NKV):
            gt = base_kv + (kv0 + kh * 16 + rw) * rs_kv + hf
            for dtile in range(NDO):
                t = gt + dtile * 2
                assert t.min() >= 0 and t.max() < I32
                assert (8 * t + 7).max() < nkv_el, "bf16 epilogue OOB"
                # element semantics: kv row, d = dtile*16 + half*8 + si
                kv = kv0 + kh * 16 + rw
                d0 = dtile * 16 + hf * 8
                assert (8 * t == ((bb3 * Skv + kv) * Hkv + hk3) * D + d0).all()
                for si in range(8):
                    np.add.at(cov, (8 * t + si).ravel(), 1)
        assert (cov == 1).all(), "bf16 epilogue: not every dK/dV element written once"
        emax = nkv_el
        kind = "bf16 vec8"
    else:
        tot = nsp * nkv_el
        cov = np.zeros(tot, np.int8)
        for sp in range(nsp):
            base_o4 = ((sp * B + bb3) * Skv * Hkv) * (D // 4) + hk3 * (D // 4)
            for kh in range(NKV):
                gt4 = base_o4 + (kv0 + kh * 16 + rw) * Hkv * (D // 4) + hf * 2
                for dtile in range(NDO):
                    for u in range(2):
                        t4 = gt4 + dtile * 4 + u
                        assert t4.min() >= 0 and t4.max() < I32
                        assert (4 * t4 + 3).max() < tot, "fp32 split epilogue OOB"
                        kv = kv0 + kh * 16 + rw
                        d0 = dtile * 16 + hf * 8 + 4 * u
                        assert (4 * t4 == (((sp * B + bb3) * Skv + kv) * Hkv + hk3) * D
                                + d0).all()
                        for e in range(4):
                            np.add.at(cov, (4 * t4 + e).ravel(), 1)
        assert (cov == 1).all(), "fp32 epilogue: not every workspace element written once"
        emax = tot
        kind = "fp32 vec4"
    print(f"B{B} Sq{Sq} Skv{Skv} Hq{Hq} Hkv{Hkv} causal{causal} nsp{nsp}: OK  "
          f"ldqd calls(qt,gh)={len(qg)} lse max_el={lmax}<{nl}  qdo max_vec8={qmax}<{nq8}  "
          f"mask checks={nm}  epilogue {kind} covers {emax} el exactly once")


if __name__ == "__main__":
    check_fragment_layout()
    shapes = [
        (4, 8192, 8192, 32, 8),     # prod
        (1, 1024, 1024, 8, 2),      # fast
        (1, 128, 128, 2, 1),        # toy
        (1, 512, 1024, 8, 2),       # Sq != Skv (UT style)
        (1, 1024, 512, 8, 2),
    ]
    for B, Sq, Skv, Hq, Hkv in shapes:
        rule = nsp_rule(B, Skv, Hkv)
        for causal in (1, 0):
            for nsp in sorted({1, rule, 2}):
                run(B, Sq, Skv, Hq, Hkv, causal, nsp)
    print("ALL OK")
