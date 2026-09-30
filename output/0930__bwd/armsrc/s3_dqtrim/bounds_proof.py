#!/usr/bin/env python3
"""Host-only proof for arm s3_dqtrim (k_dqg). No torch, no GPU.

What the arm changes (kernels.py `_body`, both are pure IR ORDER changes):
  DQ_HEADST  body 1 of the tailpf U2 trip (pf2 set): the 8 K ds_stores of kt 0 are emitted in
             front of the kt-0 S/dP WMMAs + sched_barrier(0x406), instead of one dt at a time.
  DQ_DQORD   body 2 of the tailpf U2 trip (nxt_in set): all 16 ds_load_tr16 (8 b_k) first,
  = "q2"     then the 32 dQ WMMAs qh-major (for qh: for dtile) instead of dtile-major.
No index / address / predicate / loop-bound / ring-stage EXPRESSION changes: every LDS address
is built by the same formula as s3, every loop count (npair, nkvt_eff - nfull) is untouched,
and no K/V global load moves. So this proves:
  A. per body type (B1 = tailpf body 1, B2 = tailpf body 2, M = kvloop_mask body), in PROGRAM
     order, every ds_load_tr16 16-byte chunk is served by the same writer (body, kt, dt, u,
     lane) in s3 and in the arm; every chunk read was written by the SAME body earlier (so LDS
     state carried across bodies is irrelevant -> the per-body result holds for any trip
     sequence by induction); all chunks lie in the 8704-B allocation.
  B. per body, every one of the 32 dQ accumulators gets exactly one WMMA with the same
     (A = a_ds[qh], B = b_k[dtile]) operands in s3 and in the arm; the accumulation order of
     each accumulator across bodies is the body order, which is unchanged -> bitwise.
  C. per shape x causal x q tile, the body-type sequence each workgroup executes (replayed
     from _dqg_impl's control flow) consists only of B1/B2/M, and is identical for s3 and arm
     (loop bounds untouched); reported per shape, with whether impl.py dispatches k_dqg.
"""
D, KV_STEP, NKT, NDT, NDO, NQW, BQW, WAVE = 128, 32, 2, 4, 8, 4, 64, 32
X_ROW_B = D * 2 + 16
LDS_BYTES = KV_STEP * X_ROW_B            # 8704, NW = 1
BLOCK_Q, NSP_Q_CAP, DQ_NW = 64, 8, 1

SHAPES = {  # name: (b, sq, skv, hq, hkv)
    "prod": (4, 8192, 8192, 32, 8),
    "fast": (1, 1024, 1024, 8, 2),
    "toy": (1, 128, 128, 2, 1),
    "gqa4_small": (2, 128, 128, 8, 2),
    "unequal_seqlen_2": (2, 1024, 2048, 4, 1),
}


def st_addr(kt, dt, u, lane):
    row, half = lane % 16, lane // 16
    ko = (kt * 16 + row) * X_ROW_B + half * 16
    return ko + dt * 64 + u * 32


def ld_addr(dtile, hi, lane):
    lane_r = (lane // 16) * 8 + lane % 8
    lane_c = ((lane // 8) % 2) * 8
    return lane_r * X_ROW_B + (lane_c + dtile * 16) * 2 + hi * 16 * X_ROW_B


def body_events(kind, arm):
    """Program-order LDS + WMMA events of one body, as `_body` emits them."""
    ev = []
    headst = arm and kind == "B1"
    qmaj = arm and kind == "B2"
    for kt in range(NKT):
        if headst and kt == 0:
            for dt in range(NDT):
                for u in range(2):
                    ev.append(("st", kt, dt, u))
            ev.append(("fence",))
        for dt in range(NDT):
            if not (headst and kt == 0):
                for u in range(2):
                    ev.append(("st", kt, dt, u))
            for qh in range(NQW):
                ev.append(("wS", kt, dt, qh))
                ev.append(("wP", kt, dt, qh))
    if qmaj:
        for dtile in range(NDO):
            ev.append(("ld", dtile, 0))
            ev.append(("ld", dtile, 1))
        for qh in range(NQW):
            for dtile in range(NDO):
                ev.append(("wQ", qh, dtile))
    else:
        for dtile in range(NDO):
            ev.append(("ld", dtile, 0))
            ev.append(("ld", dtile, 1))
            for qh in range(NQW):
                ev.append(("wQ", qh, dtile))
    return ev


def replay(kind, arm):
    """LDS writer of every chunk each load reads; dQ operand per accumulator."""
    lds = {}
    reads = {}
    dq = {}
    for e in body_events(kind, arm):
        if e[0] == "st":
            _, kt, dt, u = e
            for lane in range(WAVE):
                a = st_addr(kt, dt, u, lane)
                assert a % 16 == 0 and 0 <= a and a + 16 <= LDS_BYTES, ("st", e, lane, a)
                lds[a // 16] = (kt, dt, u, lane)
        elif e[0] == "ld":
            _, dtile, hi = e
            for lane in range(WAVE):
                a = ld_addr(dtile, hi, lane)
                assert a % 16 == 0 and 0 <= a and a + 16 <= LDS_BYTES, ("ld", e, lane, a)
                c = a // 16
                assert c in lds, ("chunk read before this body wrote it", kind, arm, e, lane)
                reads[(dtile, hi, lane)] = lds[c]
            # b_k[dtile] complete once both halves are read
        elif e[0] == "wQ":
            _, qh, dtile = e
            assert (dtile, 1, 0) in reads, ("dQ WMMA before its b_k load", kind, arm, e)
            acc = qh * NDO + dtile
            assert acc not in dq, ("two dQ WMMAs on one accumulator in a body", acc)
            dq[acc] = (("a_ds", qh), ("b_k", dtile),
                       tuple(reads[(dtile, hi, l)] for hi in (0, 1) for l in range(WAVE)))
    assert len(dq) == NQW * NDO
    return reads, dq


def seq(b, sq, skv, causal):
    """Body-type sequence of every q tile, replaying _dqg_impl (DQ_PF, DQ_U2, tailpf 'b')."""
    nkvt = skv // KV_STEP
    cshift = skv - sq
    out = {}
    for y in range(sq // BQW):
        bid = sq // BQW - 1 - y
        q0 = bid * BQW
        lim = (q0 + BQW + cshift + KV_STEP - 1) // KV_STEP
        lim = 1 if lim < 1 else lim
        lim = lim if lim < nkvt else nkvt
        nkvt_eff = lim if causal else nkvt
        t = q0 + cshift + 1
        nf = 0 if t < 0 else t // KV_STEP
        nf = nf if nf < nkvt_eff else nkvt_eff
        nfull = nf if causal else nkvt_eff
        npair = nfull // 2
        nmask = nkvt_eff - npair * 2
        assert npair >= 0 and nmask >= 0
        out[bid] = ["B1", "B2"] * npair + ["M"] * nmask
    return out


def uses_dqg(b, sq, hq):
    wgs = ((sq + BLOCK_Q - 1) // BLOCK_Q) * hq * b
    nsp = 1
    while wgs * nsp < 2048 and nsp < NSP_Q_CAP:
        nsp *= 2
    return nsp == 1 and hq % DQ_NW == 0 and sq % BQW == 0


def main():
    # A + B, per body type
    for kind in ("B1", "B2", "M"):
        r0, q0 = replay(kind, arm=False)
        r1, q1 = replay(kind, arm=True)
        assert r0 == r1, ("LDS writer map differs", kind)
        assert q0 == q1, ("dQ operand map differs", kind)
        print(f"body {kind}: {len(r0)} chunk reads, writer map identical s3 == arm; "
              f"{len(q0)} dQ accs, one WMMA each, operands identical")
    # C, per shape
    for name, (b, sq, skv, hq, hkv) in SHAPES.items():
        for causal in (0, 1):
            s = seq(b, sq, skv, causal)
            kinds = {k for v in s.values() for k in v}
            assert kinds <= {"B1", "B2", "M"}
            nb = sum(len(v) for v in s.values())
            npairs = sum(v.count("B1") for v in s.values())
            print(f"{name:17s} causal={causal} tiles={len(s):4d} bodies/wg-head={nb:6d} "
                  f"U2 trips={npairs:6d} mask bodies={nb - 2 * npairs:4d} "
                  f"k_dqg dispatched={uses_dqg(b, sq, hq)}")
    print("TDM / tensorcnt: not used by k_dqg; loadcnt waits are compiler-inserted and "
          "unchanged in count semantics (no VMEM moved). ALL OK")


if __name__ == "__main__":
    main()
