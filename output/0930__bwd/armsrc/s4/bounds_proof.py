#!/usr/bin/env python3
"""CPU bounds proof for the s4 arm = s3_trim (k_dkdv addresses) + s3_df (k_dkdv counters)
+ s3_dqtrim (k_dqg order only). Host python3, no torch, no GPU.

Composition argument (checked below, not assumed):
  s3_df only changes the loop DRIVERS (qloop_mask / qloop_full / _tdm_prologue) that compute
  _body's positional arguments (qt, gh, qt_n, gh_n, cur_off, nxt_off, pf_qt, pf_gh, rb_off).
  s3_trim only changes _body / _ldl / _rdqd, whose addresses are pure functions of those
  arguments plus loop invariants (_lds0, lb_tr, lb_rd, voff_l, rsrc, bat, hkv):
     TDM     : _tdm_qdo(pf_qt, pf_gh, nxt_off)
     LSE/del : _ldl(qt_n, gh_n)  -> voffset 4*row, soffset 4*((bat*Hq+hkv*G+gh_n)*Sq + 32*qt_n) + 64*hh
     tr16    : tr_base = _lds0 + cur_off + lb_tr, + imm (A1)
     readback: rb_base = _lds0 + rb_off + lb_rd, + imm (A2)
     mask    : _ldl(qt, gh, trim=False), _tdm_qdo(qt, gh, 0), _rdqd(0)
  So s4 == s3 iff df's tuple == s3's tuple on every iteration (s3_df E1/E2) and trim's
  address maps are value-identical and in range on THOSE tuples (s3_trim A1/A2/A3, here
  evaluated on the df-produced values). This file replays the ring model driven by the df
  counters and evaluates trim's address formulas on them, then runs both ISA gate sets.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", name, "bounds_proof.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DF = _load("s3_df")
TR = _load("s3_trim")
D, BLOCK_KV, X_ROW_B, QDO_B, PART, LDS_SEG, ALLOC = (TR.D, TR.BLOCK_KV, TR.X_ROW_B, TR.QDO_B,
                                                    TR.PART, TR.LDS_SEG, TR.ALLOC)
Fail, check = TR.Fail, TR.check
assert (DF.D, DF.X_ROW_B, DF.QDO_B, DF.PART) == (D, X_ROW_B, QDO_B, PART)


def trim_tr_addrs(cur):
    """A1 evaluated at stage offset `cur`: every tr16 address (tr_base + imm) vs s3's."""
    out = []
    for lane in range(32):
        lane_r = (lane // 16) * 8 + lane % 8
        lane_c = ((lane // 8) % 2) * 8
        base = cur + lane_r * X_ROW_B + lane_c * 2
        for dtile in range(8):
            for part in (0, PART):
                for r2 in (0, 16):
                    imm = part + dtile * 32 + r2 * X_ROW_B
                    old = cur + part + lane_r * X_ROW_B + (lane_c + dtile * 16) * 2 + r2 * X_ROW_B
                    check(base + imm == old and 0 <= imm < 65536, "A1 on df cur")
                    out.append(base + imm)
    return out


def trim_rb_addrs(rbo):
    out = []
    for lane in range(32):
        row, half = lane % 16, lane // 16
        base = rbo + row * X_ROW_B + half * 16
        for hh in range(2):
            for part in (PART, 0):
                for dt in range(4):
                    for u in range(2):
                        imm = hh * 16 * X_ROW_B + part + dt * 64 + u * 32
                        old = rbo + part + (hh * 16 + row) * X_ROW_B + half * 16 + dt * 64 + u * 32
                        check(base + imm == old and 0 <= imm < 65536, "A2 on df rbo")
                        out.append(base + imm)
    return out


STAGES = (0, QDO_B, 2 * QDO_B)
A12 = {s: (trim_tr_addrs(s), trim_rb_addrs(s)) for s in STAGES}
for s, (t, r) in A12.items():
    for a in t + r:
        assert s <= a and a + 16 <= s + QDO_B and a + 16 <= 3 * QDO_B <= LDS_SEG


def run_shape(B, Sq, Skv, Hq, Hkv, causal):
    G = Hq // Hkv
    nqt2 = Sq // 32
    cshift = Skv - Sq
    nsp = TR.nsp_rule(B, Skv, Hkv)
    PARTIAL = nsp > 1
    n_elem = B * Sq * Hq * D
    nl_elem = B * Hq * Sq
    st = dict(wgs=0, e1=0, ldl=0, tdm=0, full=0, mask=0, maxcnt=0)

    def clampqt(t):
        t = t if t < nqt2 else nqt2 - 1
        return 0 if t < 0 else t

    for bat in range(B):
        for bid in range(Skv // BLOCK_KV):
            for xw in range(Hkv * nsp):
                hkv, sp = (xw // nsp, xw % nsp) if PARTIAL else (xw, 0)
                st["wgs"] += 1
                kv0 = bid * BLOCK_KV
                c = kv0 - cshift
                qp_start = ((0 if c < 0 else c) // 32) if causal else 0
                nqp_eff = nqt2 - qp_start
                u = kv0 + BLOCK_KV - 1 - cshift
                qsf = 0 if u < 0 else (u + 31) // 32
                qsf = min(qsf, nqt2)
                nm = min(max(qsf - qp_start, 0), nqp_eff)
                nmaskp = nm if causal else 0
                if PARTIAL:
                    fn = max(nqp_eff - nmaskp, 0)
                    ch = (fn + nsp - 1) // nsp
                    cnt = min(max(fn - sp * ch, 0), ch)
                    qt_mask0, n_mask = qp_start, G * (0 if sp else nmaskp)
                    qt_full0, n_full = qp_start + nmaskp + sp * ch, G * cnt
                else:
                    qt_mask0, n_mask = qp_start, G * nmaskp
                    qt_full0, n_full = qp_start + nmaskp, G * (nqp_eff - nmaskp)

                fifo, pend, stage_tile = [], [], {}

                def tdm(qt, gh, stage):
                    check(0 <= qt < nqt2 and 0 <= gh < G and Sq - 32 * qt >= 32, f"TDM tile {qt, gh}")
                    qh = hkv * G + gh
                    base = ((bat * Sq + 32 * qt) * Hq + qh) * D
                    check(base >= 0 and base + 31 * Hq * D + D - 1 < n_elem, "TDM global OOB")
                    check(stage in STAGES, f"stage {stage}")
                    check(all(ps != stage for ps, _ in pend), "TDM into stage with unretired read")
                    fifo.extend([stage, stage])
                    st["maxcnt"] = max(st["maxcnt"], len(fifo))
                    stage_tile[stage] = (qh, qt)
                    st["tdm"] += 1

                def wait(n):
                    del fifo[:max(len(fifo) - n, 0)]

                def read(stage, key, kind):
                    check(stage not in fifo, f"{kind} read of stage {stage} with TDM in flight")
                    if key is not None:
                        check(stage_tile.get(stage) == key, f"{kind}: stage holds wrong tile")
                    pend.append((stage, kind))

                def retire(kind):
                    pend[:] = [x for x in pend if x[1] != kind]

                def ldl(qt, gh, trim):
                    check(0 <= gh < G and 0 <= qt < nqt2, "ldl tile")
                    base_l = (bat * Hq + hkv * G + gh) * Sq
                    lo, hi = base_l + 32 * qt, base_l + 32 * qt + 31
                    check(0 <= lo and hi < nl_elem and 32 * qt + 31 < Sq, "LSE/delta OOB")
                    if trim:
                        so = 4 * (base_l + 32 * qt)
                        check(0 <= so and so + 64 + 4 * 15 < 1 << 31, "soffset range")
                        for hh in range(2):
                            for row in (0, 15):
                                check(4 * row + so + 64 * hh == 4 * (base_l + 32 * qt + 16 * hh + row), "A3")
                    st["ldl"] += 1

                # qloop_mask: df counters -> trim's masked body (s3 loads)
                qc, gc = 0, 0
                for ii in range(n_mask):
                    check((qc, gc) == divmod(ii, G), "E1 mask")
                    qt, gh = qt_mask0 + qc, gc
                    ldl(qt, gh, False)
                    tdm(qt, gh, 0)
                    wait(0)
                    read(0, (hkv * G + gh, qt), "m")
                    retire("m")
                    qc, gc = DF.wrap(qc, gc, G)
                    st["mask"] += 1
                # prologue: _ldl(_pc0, 0) (trim form) + df stage-1 tile
                ldl(clampqt(qt_full0), 0, True)
                tdm(clampqt(qt_full0), 0, 0)
                q1, g1 = DF.prologue_df(n_full, G)
                tdm(clampqt(qt_full0 + q1), g1, QDO_B)
                wait(2)
                read(0, (hkv * G, qt_full0) if n_full > 0 else None, "rb")
                cst = (0, 0, 0)
                for ii in range(n_full):
                    tup, cst = DF.full_df(ii, n_full, G, qt_full0, cst)
                    check(tup == DF.full_ref(ii, n_full, G, qt_full0), "E1 full")
                    st["e1"] += 1
                    qt, gh, qt_n, gh_n, cur, nxo, pf_qt, pf_gh, rbo = tup
                    # _body(carry=True), s3_trim order: TDM, then LSE/delta prefetch
                    tdm(pf_qt, pf_gh, nxo)
                    ldl(qt_n, gh_n, True)
                    retire("rb")
                    check(cur in A12, "cur not a stage")      # tr16 addresses = A12[cur][0]
                    read(cur, (hkv * G + gh, qt), "tr")
                    wait(2)
                    check(rbo in A12, "rbo not a stage")      # readback addresses = A12[rbo][1]
                    read(rbo, (hkv * G + gh_n, qt_n) if ii + 1 < n_full else None, "rb")
                    retire("tr")
                    st["full"] += 1
                wait(0)
                retire("rb")
                check(not fifo and not pend, "not drained at exit")
    return st, nsp


def main():
    shapes = [("prod", 4, 8192, 8192, 32, 8), ("fast", 1, 1024, 1024, 8, 2),
              ("toy", 1, 128, 128, 2, 1), ("gqa4_small", 2, 128, 128, 8, 2),
              ("unequal_seqlen_2", 2, 1024, 2048, 4, 1)]
    ok = True
    for name, *shp in shapes:
        for causal in (1, 0):
            try:
                s, nsp = run_shape(*shp, causal)
                print(f"PASS {name:16s} causal={causal} nsp={nsp:2d} {s}")
            except Fail as e:
                ok = False
                print(f"FAIL {name} causal={causal}: {e}")
    good, msg = DF.e2_sweep()
    print(("PASS" if good else "FAIL") + " df E2 sweep:", msg)
    ok &= good
    for k in ("dkdv/k_dkdv_0", "dkdv_sp/k_dkdv_sp_0"):
        p = os.path.join(HERE, ".dump", k, "21_final_isa.s")
        for tag, fn in (("ring", TR.isa_check), ("trim", TR.isa_trim_check)):
            try:
                print(f"ISA-{tag} {k}:", fn(p))
            except Fail as e:
                ok = False
                print(f"FAIL ISA-{tag} {k}:", e)
    print("RESULT:", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
