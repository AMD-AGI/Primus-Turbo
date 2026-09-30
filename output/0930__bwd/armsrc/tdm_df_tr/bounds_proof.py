#!/usr/bin/env python3
"""CPU bounds proof for the tdm_df_tr arm = dkdv_tdm + dkdv_divfree + dkdv_trorder
(host python3, no torch, no GPU). Adapted from arms/dkdv_tdm/bounds_proof.py; the added
parts (marked tdm_df_tr) are:
  DF  the division-free loop state: (qi, gh) wrap counters in qloop_mask / qloop_full,
      the carried ring-stage byte offset cur (wrap at DEPTH*QDO_B), the prefetch stage
      nxo = cur == 0 ? (DEPTH-1)*QDO_B : cur - QDO_B, the prefetch tile via
      wrap(wrap(qi, gh)) / clamped jj, and the prologue's j1 via wrap(0, 0). Every value is
      asserted EQUAL to the division formula of dkdv_tdm for every iteration of every WG,
      so all TDM descriptors / LSE / delta indices are the ones proven below.
  LD  LSE/delta element index of every _ldl (current + carried prefetch) < B*Hq*Sq.
  TR  trorder's final-phase tr16 address multiset per lane == the dkdv_tdm (r29) order's,
      for every ring stage; each access stays inside its stage part / P/dS ring.
  ISA the WAR / tensor-wait order checks run on BOTH k_dkdv and k_dkdv_sp dumps.

--- original dkdv_tdm description ---

Replays, per workgroup, the exact integer control flow of kernels.py:_dkdv_impl
(both PARTIAL=False -> k_dkdv and PARTIAL=True -> k_dkdv_sp, with impl.py's nsp rule)
and enumerates EVERY TDM descriptor the kernel issues:
  - qloop_mask body   : own tile (qt, gh) -> stage 0, then tensor_wait(0)
  - prologue          : clamp(qt_first), gh 0 -> stage 0; for TDM_DEPTH 3 also
                        iteration max(min(1,n-1),0)'s tile (clamped pair) -> stage 1
  - qloop_full iter ii: tensor_wait(DEPTH-2 pairs = 2 ops for depth 3, 0 for depth 2);
                        tile(min(ii+DEPTH-1, n-1)) -> stage (ii+DEPTH-1)%DEPTH
  - after the loop    : tensor_wait(0)
TDM ops are modelled as retiring IN ORDER (s_wait_tensorcnt N = all but the newest N
retired), the premise aiter's gemm_a16w16_kernel_gfx1250 ring also relies on.
For each descriptor it checks
  G1 every global element read (dO and Q, 32 rows x 128 cols, outer stride Hq*D)
     lies in [0, B*Sq*Hq*D) and the TDM outer extent (Sq-q0) >= 32 (no HW clamp live,
     so the LDS image is exactly c1's image),
  L1 every LDS byte written lies inside the 70656 B allocation and inside its own stage,
  L2 no TDM write lands in the stage that the SAME iteration reads, and every LDS read of
     a stage happens after all writes to that stage were retired by a tensor_wait,
  C1 tensorcnt bookkeeping: every wait threshold (always 0) is reachable, the counter is
     0 at kernel exit and never exceeds 2,
  V1 the stage read by iteration ii holds tile (qt(ii), gh(ii)) -- i.e. the ring delivers
     the same data c1 delivered (value-level equivalence of the schedule).
Also checks, from the compiled ISA if present, that every ds_load in each TDM loop body
is followed by an s_wait_dscnt 0x0 before the back edge (the WAR premise of L2).
"""
import os
import re
import sys

D = 128
BLOCK_KV = 32
X_ROW_B = D * 2 + 16            # 272
S_ROW_B = BLOCK_KV * 2 + 16     # 80
LDS_SEG = 65536
ALLOC = LDS_SEG + 2 * 32 * S_ROW_B      # 70656, unchanged from c1
QDO_B = 2 * 32 * X_ROW_B                # 17408 per stage (dO + Q)
DEPTHS = (3, 2)                         # 3 = shipped default (kernels.TDM_DEPTH)
PART = 32 * X_ROW_B                     # 8704: dO at +0, Q at +8704
TILE_ROWS, TILE_COLS = 32, D
EPI_END = 4 * 128 * 48                  # epilogue LDS image [0, 24576)


def tdm_lds_range(stage_off, part):
    """Bytes the TDM writes: row r -> [lb + r*272, lb + r*272 + 256); the 16 B pad after
    each row may or may not be written, so count it (conservative): [lb, lb + 32*272)."""
    lb = stage_off + part
    return lb, lb + TILE_ROWS * X_ROW_B


def read_ranges(stage_off):
    """Every LDS byte the body reads from a stage (ds_load_b128 + ds_load_tr16_b128)."""
    lo, hi = 1 << 60, -1
    for part in (0, PART):
        base = stage_off + part
        # S/dP operands: lane (row, half): (hh*16+row)*272 + half*16 + dt*64 + u*32, 16 B
        for hh in range(2):
            for row in range(16):
                for half in range(2):
                    for dt in range(4):
                        for u in range(2):
                            a = base + (hh * 16 + row) * X_ROW_B + half * 16 + dt * 64 + u * 32
                            lo, hi = min(lo, a), max(hi, a + 16)
                            assert (a - base) % X_ROW_B + 16 <= 2 * D, "read touches pad"
        # tr16: lane_r*272 + (lane_c + dtile*16)*2, and +16 rows; 16 B each
        for lane in range(32):
            lane_r = (lane // 16) * 8 + lane % 8
            lane_c = ((lane // 8) % 2) * 8
            for dtile in range(8):
                for r2 in (0, 16):
                    a = base + (lane_r + r2) * X_ROW_B + (lane_c + dtile * 16) * 2
                    lo, hi = min(lo, a), max(hi, a + 16)
                    assert (a - base) % X_ROW_B + 16 <= 2 * D, "tr16 read touches pad"
    return lo, hi


RR = [read_ranges(k * QDO_B) for k in range(3)]
for k, (lo, hi) in enumerate(RR):
    assert k * QDO_B <= lo and hi <= (k + 1) * QDO_B, (k, lo, hi)
assert 3 * QDO_B <= LDS_SEG, "all stages must stay in LDS segment 0"
assert EPI_END <= LDS_SEG


class Fail(Exception):
    pass


# ---------------- tdm_df_tr: divfree helpers ----------------
def wrap(qc, gc, G):
    g1 = gc + 1
    return (qc, g1) if g1 < G else (qc + 1, 0)


# ---------------- tdm_df_tr: trorder LDS multiset (per stage) ----------------
NDO, NKV = 8, 2
LDS_P = LDS_SEG
LDS_DS = LDS_P + 32 * S_ROW_B


def _lane(lane):
    return (lane // 16) * 8 + lane % 8, ((lane // 8) % 2) * 8


def _tr(base, rowb):
    return [(base, 16), (base + 16 * rowb, 16)]


def loads_tdm_order(lane, stage):          # dkdv_tdm / r29 emission order
    lane_r, lane_c = _lane(lane)
    do, q = stage, stage + PART
    out = []
    for dt in range(NDO):
        c = (lane_c + dt * 16) * 2
        out += [("do",) + x for x in _tr(do + lane_r * X_ROW_B + c, X_ROW_B)]
        out += [("q",) + x for x in _tr(q + lane_r * X_ROW_B + c, X_ROW_B)]
    for kh in range(NKV):
        col = lane_c * 2 + kh * 32
        out += [("p",) + x for x in _tr(LDS_P + lane_r * S_ROW_B + col, S_ROW_B)]
        out += [("ds",) + x for x in _tr(LDS_DS + lane_r * S_ROW_B + col, S_ROW_B)]
    return out


def loads_trorder(lane, stage):           # R1 a_p0,b_do ; R2 a_ds0,b_q ; R3 a_p1,a_ds1
    lane_r, lane_c = _lane(lane)
    do, q = stage, stage + PART
    _a = lambda base, kh: _tr(base + lane_r * S_ROW_B + lane_c * 2 + kh * 32, S_ROW_B)
    _b = lambda base, dt: _tr(base + lane_r * X_ROW_B + (lane_c + dt * 16) * 2, X_ROW_B)
    out = [("p",) + x for x in _a(LDS_P, 0)]
    for dt in range(NDO):
        out += [("do",) + x for x in _b(do, dt)]
    out += [("ds",) + x for x in _a(LDS_DS, 0)]
    for dt in range(NDO):
        out += [("q",) + x for x in _b(q, dt)]
    out += [("p",) + x for x in _a(LDS_P, 1)] + [("ds",) + x for x in _a(LDS_DS, 1)]
    return out


def trorder_check():
    n = 0
    for stage in (k * QDO_B for k in range(3)):
        ring = {"do": (stage, stage + PART), "q": (stage + PART, stage + 2 * PART),
                "p": (LDS_P, LDS_DS), "ds": (LDS_DS, ALLOC)}
        for lane in range(32):
            a, b = loads_trorder(lane, stage), loads_tdm_order(lane, stage)
            assert len(a) == len(b) == 40 and sorted(a) == sorted(b), (stage, lane)
            for r, addr, w in a:
                lo, hi = ring[r]
                assert lo <= addr and addr + w <= hi, (stage, lane, r, addr)
                if r in ("do", "q"):
                    assert (addr - lo) % X_ROW_B + w <= 2 * D, "tr16 read touches pad"
                n += 1
    return n


TR_N = trorder_check()


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def nsp_rule(B, Skv, Hkv):
    wgs = (Skv // BLOCK_KV) * Hkv * B
    nsp = 1
    while wgs * nsp < 2048 and nsp < 16:
        nsp *= 2
    return nsp


def run_shape(name, B, Sq, Skv, Hq, Hkv, causal, depth):
    assert Sq % 64 == 0 and Skv % 32 == 0 and Hq % Hkv == 0   # impl.py asserts
    G = Hq // Hkv
    nqt = Sq // 16
    cshift = Skv - Sq
    nsp = nsp_rule(B, Skv, Hkv)
    PARTIAL = nsp > 1
    n_elem = B * Sq * Hq * D
    row_stride = Hq * D
    stats = dict(df_checked=0, ldl=0, desc=0, waits=0, wgs=0, iters_full=0, iters_mask=0, prefetch_clamped=0,
                 prologue_clamped=0, empty_full=0, gmax=-1, gmin=1 << 62, maxcnt=0)

    def tdm(bat, hkv, qt, gh, stage_off):
        """Validate one TDM issue (dO and Q descriptors) and return its tile key."""
        check(0 <= qt < nqt // 2, f"qt {qt} outside [0,{nqt//2})")
        check(0 <= gh < G, f"gh {gh}")
        qh = hkv * G + gh
        q0 = qt * 32
        valid = Sq - q0
        check(valid >= TILE_ROWS, f"outer extent {valid} < 32: HW zero-fill would be live")
        base = ((bat * Sq + q0) * Hq + qh) * D
        first = base
        last = base + (TILE_ROWS - 1) * row_stride + (TILE_COLS - 1)
        check(first >= 0 and last < n_elem, f"global OOB [{first},{last}] vs {n_elem}")
        stats["gmax"] = max(stats["gmax"], last)
        stats["gmin"] = min(stats["gmin"], first)
        for part in (0, PART):
            lo, hi = tdm_lds_range(stage_off, part)
            check(0 <= lo and hi <= ALLOC, "LDS write outside allocation")
            check(stage_off <= lo and hi <= stage_off + QDO_B, "LDS write crosses stage")
            check(hi <= LDS_SEG, "LDS write reaches the P/dS segment")
            check(hi <= depth * QDO_B, "LDS write outside the ring")
        stats["desc"] += 2
        return (bat, qh, qt)

    nqt2 = nqt // 2

    def ldl(bat, hkv, qt, gh):
        """tdm_df_tr LD: the 4 LSE/delta b32 element indices of one _ldl (all 16 rows)."""
        qh = hkv * G + gh
        base_l = (bat * Hq + qh) * Sq
        for hh in range(2):
            for row in range(16):
                idx = base_l + qt * 32 + hh * 16 + row
                check(0 <= idx < B * Hq * Sq, f"LSE/delta OOB {idx}")
        stats["ldl"] += 1

    def eq(a, b, what):
        check(a == b, f"divfree mismatch {what}: {a} != {b}")
        stats["df_checked"] += 1

    def clampqt(t):
        t = t if t < nqt2 else nqt2 - 1
        return 0 if t < 0 else t

    for bat in range(B):
        for bid in range(Skv // BLOCK_KV):
            for xw in range(Hkv * nsp):
                hkv, sp = (xw // nsp, xw % nsp) if PARTIAL else (xw, 0)
                stats["wgs"] += 1
                kv0 = bid * BLOCK_KV
                c = kv0 - cshift
                qp_start = (0 if c < 0 else c) // 32
                qp_start = qp_start if causal else 0
                nqp_eff = nqt2 - qp_start
                u = kv0 + BLOCK_KV - 1 - cshift
                qsf = 0 if u < 0 else (u + 31) // 32
                qsf = qsf if qsf < nqt2 else nqt2
                nm = max(qsf - qp_start, 0)
                nm = nm if nm < nqp_eff else nqp_eff
                nmaskp = nm if causal else 0

                stage_tile = {}
                fifo = []            # outstanding TDM ops, oldest first: stage offsets

                def wait(n):
                    stats["waits"] += 1
                    check(0 <= n, "negative wait")
                    # reachable: the counter can always drain to n (n >= 0); in-order
                    # retirement leaves exactly the newest n.
                    del fifo[:max(len(fifo) - n, 0)]

                def issue(qt, gh, stage):
                    check(stage not in reading, "TDM write into the stage being read")
                    key = tdm(bat, hkv, qt, gh, stage)
                    fifo.extend([stage, stage])
                    stats["maxcnt"] = max(stats["maxcnt"], len(fifo))
                    check(len(fifo) <= 63, "tensorcnt overflow")
                    stage_tile[stage] = key

                def read(stage, qt, gh):
                    check(stage not in fifo, "LDS read of a stage with a TDM write in flight")
                    check(stage_tile.get(stage) == (bat, hkv * G + gh, qt),
                          f"stage holds {stage_tile.get(stage)}, iteration needs {(qt, gh)}")

                reading = set()
                if PARTIAL:
                    fn = max(nqp_eff - nmaskp, 0)
                    ch = (fn + nsp - 1) // nsp
                    off = sp * ch
                    cnt = max(fn - off, 0)
                    cnt = cnt if cnt < ch else ch
                    mk = 0 if sp != 0 else nmaskp
                    qt_mask0, n_mask = qp_start, G * mk
                    qt_full0, n_full = qp_start + nmaskp + off, G * cnt
                else:
                    qt_mask0, n_mask = qp_start, G * nmaskp
                    qt_full0, n_full = qp_start + nmaskp, G * (nqp_eff - nmaskp)

                # qloop_mask: carry=False -> own tile into stage 0, wait(0), read stage 0.
                # WAR: the previous iteration's reads of stage 0 retired (ISA check).
                qc, gc = 0, 0                        # tdm_df_tr: divfree mask counters
                for ii in range(n_mask):
                    qt, gh = qt_mask0 + ii // G, ii % G
                    eq((qt_mask0 + qc, gc), (qt, gh), "mask (qt, gh)")
                    qc, gc = wrap(qc, gc, G)
                    ldl(bat, hkv, qt, gh)
                    issue(qt, gh, 0)
                    wait(0)
                    reading = {0}
                    read(0, qt, gh)
                    reading = set()
                    stats["iters_mask"] += 1
                # prologue
                if clampqt(qt_full0) != qt_full0:
                    stats["prologue_clamped"] += 1
                issue(clampqt(qt_full0), 0, 0)
                if depth == 3:
                    j1 = 1 if 1 < n_full else n_full - 1
                    j1 = 0 if j1 < 0 else j1
                    # tdm_df_tr: divfree prologue decomposition of j1 in {0, 1}
                    w1 = wrap(0, 0, G)
                    q1d, g1d = w1 if 1 < n_full else (0, 0)
                    eq((q1d, g1d), (j1 // G, j1 - (j1 // G) * G), "prologue j1")
                    issue(clampqt(qt_full0 + q1d), g1d, QDO_B)
                ldl(bat, hkv, clampqt(qt_full0), 0)   # prologue _ldl(_pc0, 0)
                if n_full == 0:
                    stats["empty_full"] += 1
                # qloop_full
                qc, gc, cc = 0, 0, 0                 # tdm_df_tr: carried (qi, gh, cur)
                for ii in range(n_full):
                    cur = (ii % depth) * QDO_B
                    qt, gh = qt_full0 + ii // G, ii % G
                    kk = ii + depth - 1
                    kk = kk if kk < n_full else n_full - 1
                    if kk != ii + depth - 1:
                        stats["prefetch_clamped"] += 1
                    nxo = ((ii + depth - 1) % depth) * QDO_B
                    # ---- tdm_df_tr: the kernel's DIVFREE arithmetic, verbatim ----
                    qn, gn = wrap(qc, gc, G)
                    live = ii + 1 < n_full
                    qj, gj = (qn, gn) if live else (qc, gc)
                    cn = cc + QDO_B
                    cn = cn if cn < depth * QDO_B else 0
                    nxo_df = (depth - 1) * QDO_B if cc < QDO_B else cc - QDO_B
                    if depth == 3:
                        q2, g2 = wrap(qn, gn, G)
                        pf = (q2, g2) if ii + 2 < n_full else (qj, gj)
                    else:
                        pf = (qj, gj)
                    jj = ii + 1 if ii + 1 < n_full else n_full - 1
                    eq((qt_full0 + qc, gc), (qt, gh), "full (qt, gh)")
                    eq((qt_full0 + qj, gj), (qt_full0 + jj // G, jj % G), "full (qt_n, gh_n)")
                    eq(cc, cur, "ring cur")
                    eq(nxo_df, nxo, "ring nxo")
                    eq((qt_full0 + pf[0], pf[1]), (qt_full0 + kk // G, kk % G), "prefetch tile")
                    ldl(bat, hkv, qt_full0 + qj, gj)    # carried LSE/delta prefetch
                    qc, gc, cc = qn, gn, cn
                    reading = {cur}
                    wait(2 if depth == 3 else 0)
                    check(nxo != cur, "stage aliasing")
                    issue(qt_full0 + kk // G, kk % G, nxo)
                    read(cur, qt, gh)       # tile must be there (V1) and retired (L2)
                    reading = set()
                    stats["iters_full"] += 1
                wait(0)                     # after-loop wait; epilogue may reuse stages
                check(not fifo, "tensorcnt not drained at exit")
    return stats, nsp


def isa_war_check(isa_path):
    """In every loop body containing tensor_load_to_lds: (a) no ds_* op precedes the
    s_wait_tensorcnt (reads of the current stage are ordered after its retirement), and
    (b) after the last ds_load there is an s_wait_dscnt 0x0 (or s_wait_loadcnt_dscnt with
    dscnt 0) before the back-edge branch (WAR premise for the next TDM into that stage)."""
    if not os.path.exists(isa_path):
        return "ISA not present, skipped"
    L = open(isa_path).read().split("\n")
    labels = {l[:-1]: i for i, l in enumerate(L) if re.match(r"^\.LBB\d+_\d+:$", l)}
    out = []
    for lab, s in labels.items():
        ends = [i for i in range(s + 1, len(L)) if re.search(r"s_cbranch_\w+ " + re.escape(lab) + "$", L[i])]
        if not ends:
            continue
        e = ends[0]
        body = [x.strip() for x in L[s + 1:e]]
        if not any(x.startswith("tensor_load_to_lds") for x in body):
            continue
        last_ld = max(i for i, x in enumerate(body) if x.startswith("ds_load"))
        ok = any(re.match(r"s_wait_dscnt 0x0$", x) or re.match(r"s_wait_loadcnt_dscnt 0x[0-9a-f]*00$", x)
                 for x in body[last_ld + 1:])
        tl = [i for i, x in enumerate(body) if x.startswith("tensor_load_to_lds")]
        tw = [i for i, x in enumerate(body) if x.startswith("s_wait_tensorcnt")]
        pre_ds = [x for x in body[:tw[0]] if x.startswith("ds_")] if tw else ["<no tensor wait>"]
        if pre_ds:
            raise Fail(f"{lab}: LDS op {pre_ds[0]} precedes the s_wait_tensorcnt")
        out.append(f"{lab}: ds_load retired before back edge={ok}; no ds op before the tensor wait; "
                   f"tensor_load at body idx {tl}, "
                   f"s_wait_tensorcnt at {tw} ({[body[i] for i in tw]})")
        if not ok:
            raise Fail(f"{lab}: a ds_load may still be in flight at the back edge")
    return "; ".join(out)


def main():
    shapes = [
        ("prod", 4, 8192, 8192, 32, 8),
        ("fast", 1, 1024, 1024, 8, 2),
        ("toy", 1, 128, 128, 2, 1),
    ]
    ok = True
    for depth in DEPTHS:
        for name, B, Sq, Skv, Hq, Hkv in shapes:
            for causal in (1, 0):
                try:
                    st, nsp = run_shape(name, B, Sq, Skv, Hq, Hkv, causal, depth)
                    n_elem = B * Sq * Hq * D
                    print(f"PASS depth={depth} {name:4s} causal={causal} nsp={nsp:2d} "
                          f"({'k_dkdv_sp' if nsp > 1 else 'k_dkdv'}): wgs={st['wgs']} "
                          f"tdm_desc={st['desc']} waits={st['waits']} max_tensorcnt={st['maxcnt']} "
                          f"mask_iters={st['iters_mask']} full_iters={st['iters_full']} "
                          f"clamped_prefetch={st['prefetch_clamped']} clamped_prologue={st['prologue_clamped']} "
                          f"empty_full_loops={st['empty_full']} divfree_eq={st['df_checked']} ldl={st['ldl']} global_elem=[{st['gmin']},{st['gmax']}] < {n_elem}")
                except Fail as e:
                    ok = False
                    print(f"FAIL depth={depth} {name} causal={causal}: {e}")
    print(f"LDS: alloc={ALLOC} stages=[k*{QDO_B},(k+1)*{QDO_B}) k<3 (ring end {3*QDO_B} < {LDS_SEG}) "
          f"P/dS=[{LDS_SEG},{ALLOC}) read ranges per stage={RR}")
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"TR: trorder tr16 multiset == dkdv_tdm order for 3 stages x 32 lanes, {TR_N} accesses in-bounds")
    for k in ("dkdv/k_dkdv_0", "dkdv_sp/k_dkdv_sp_0"):
        try:
            print(f"ISA {k}:", isa_war_check(os.path.join(here, f".dump/{k}/21_final_isa.s")))
        except Fail as e:
            ok = False
            print(f"FAIL ISA {k}:", e)
    print("RESULT:", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
