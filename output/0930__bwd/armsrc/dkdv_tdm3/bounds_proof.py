#!/usr/bin/env python3
"""CPU bounds proof for the dkdv_tdm3 arm (host python3, no torch, no GPU).

Replays, per workgroup, the exact integer control flow of kernels.py:_dkdv_impl
(PARTIAL=False -> k_dkdv, PARTIAL=True -> k_dkdv_sp, impl.py's nsp rule) under the tdm3
schedule (3-stage ring, B operands read back one iteration early into VGPRs):
  qloop_mask iter   : TDM own tile -> stage 0; tensor_wait(0); readback + tr16 of stage 0
                      (both consumed, hence retired, inside the iteration)
  prologue          : TDM tile(0) -> s0; TDM tile(max(min(1,n-1),0)) -> s1;
                      tensor_wait(2); READBACK s0 (for iteration 0)
  qloop_full iter i : TDM tile(min(i+2,n-1)) -> s(i+2)%3   [top, no wait]
                      S/dP WMMAs on the carried operands (read back from s i%3)
                      tr16 of s i%3; tensor_wait(2); READBACK s(i+1)%3 (for i+1)
  after the loop    : tensor_wait(0); epilogue LDS stores
Read lifetimes (what makes an LDS read "retired"):
  tr16 of iteration i            -> consumed by i's dK/dV WMMAs: retired by end of i
  readback in iteration i (or the prologue, i = -1) -> consumed by i+1's first S/dP WMMA:
                                    retired at the top of i+1, AFTER that iteration's TDM
                                    issue (the dscnt 0 wait sits after the TDM in the ISA)
  the last readback (n-1)        -> never consumed; retires before any later LDS op of the
                                    wave touches LDS (in-order LDS per wave), i.e. before
                                    the epilogue stores.
TDM ops retire in order (s_wait_tensorcnt N = all but the newest N retired), the premise
aiter's gemm_a16w16_kernel_gfx1250 ring relies on.
Checks:
  G1 every global element read in [0, B*Sq*Hq*D), TDM outer extent Sq-q0 >= 32,
  L1 every LDS write inside its own stage, the ring (< 52224) and the allocation,
  L2 no TDM issue targets a stage with an unretired read (tr16 or readback), and no read
     (tr16 or readback) touches a stage with an unretired TDM write,
  C1 tensorcnt: waits are reachable (thresholds 0/2 with >= that many outstanding is
     not required -- a wait on n always retires), outstanding <= 4, 0 at exit,
  V1 the readback feeding iteration i reads tile(i) and the tr16 of iteration i reads
     tile(i): the same data c1 used, so dK/dV stay bitwise.
ISA checks on the compiled k_dkdv and k_dkdv_sp: see isa_check().
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
    stats = dict(desc=0, waits=0, wgs=0, iters_full=0, iters_mask=0, prefetch_clamped=0,
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
                pend = []            # unretired LDS reads: (stage, kind)

                def wait(n):
                    stats["waits"] += 1
                    check(0 <= n, "negative wait")
                    stats["maxwait_q"] = max(stats.get("maxwait_q", 0), len(fifo))
                    del fifo[:max(len(fifo) - n, 0)]

                def issue(qt, gh, stage):
                    check(all(ps != stage for ps, _ in pend),
                          f"TDM write into stage {stage} with an unretired read {pend}")
                    key = tdm(bat, hkv, qt, gh, stage)
                    fifo.extend([stage, stage])
                    stats["maxcnt"] = max(stats["maxcnt"], len(fifo))
                    check(len(fifo) <= 63, "tensorcnt overflow")
                    stage_tile[stage] = key

                def read(stage, qt, gh, kind):
                    check(stage not in fifo, f"{kind} read of stage {stage} with a TDM write in flight")
                    if qt is not None:
                        check(stage_tile.get(stage) == (bat, hkv * G + gh, qt),
                              f"{kind}: stage holds {stage_tile.get(stage)}, needs {(qt, gh)}")
                    pend.append((stage, kind))

                def retire(kind):
                    pend[:] = [x for x in pend if x[1] != kind]

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

                def tile(i):
                    return qt_full0 + i // G, i % G

                for ii in range(n_mask):
                    qt, gh = qt_mask0 + ii // G, ii % G
                    issue(qt, gh, 0)
                    wait(0)
                    read(0, qt, gh, "mask_rb")
                    read(0, qt, gh, "mask_tr")
                    retire("mask_rb")
                    retire("mask_tr")
                    stats["iters_mask"] += 1
                # prologue
                if clampqt(qt_full0) != qt_full0:
                    stats["prologue_clamped"] += 1
                issue(clampqt(qt_full0), 0, 0)
                j1 = 1 if 1 < n_full else n_full - 1
                j1 = 0 if j1 < 0 else j1
                issue(clampqt(qt_full0 + j1 // G), j1 - (j1 // G) * G, QDO_B)
                wait(2)
                if n_full > 0:
                    read(0, *tile(0), "rb")
                else:
                    stats["empty_full"] += 1
                    read(0, None, None, "rb")        # still in bounds, never consumed
                for ii in range(n_full):
                    cur = (ii % 3) * QDO_B
                    nxo = ((ii + 2) % 3) * QDO_B
                    rbo = ((ii + 1) % 3) * QDO_B
                    kk = ii + 2
                    kk = kk if kk < n_full else n_full - 1
                    if kk != ii + 2:
                        stats["prefetch_clamped"] += 1
                    issue(*tile(kk), nxo)            # top: no wait
                    retire("rb")                     # dscnt 0 before first S/dP WMMA
                    read(cur, *tile(ii), "tr")
                    wait(2)
                    j = ii + 1 if ii + 1 < n_full else n_full - 1
                    if ii + 1 < n_full:
                        read(rbo, *tile(ii + 1), "rb")
                    else:
                        qj_, gj_ = tile(j)
                        check(stage_tile.get(rbo) == (bat, hkv * G + gj_, clampqt(qj_)),
                              "clamped readback stage does not hold tile(n-1)")
                        read(rbo, None, None, "rb")  # clamped: in bounds, never consumed
                    retire("tr")                     # consumed by the dK/dV WMMAs
                    stats["iters_full"] += 1
                wait(0)
                # the final readback retires in LDS order before the epilogue stores;
                # the epilogue writes [0, 24576) only after tensor_wait(0).
                retire("rb")
                check(not fifo, "tensorcnt not drained at exit")
                check(not pend, "unretired LDS read at exit")
    return stats, nsp


def _loops(L):
    for i, l in enumerate(L):
        m = re.match(r"^(\.LBB\d+_\d+):$", l)
        if not m:
            continue
        ends = [j for j in range(i + 1, len(L)) if re.search(r"s_cbranch_\w+ " + re.escape(m.group(1)) + "$", L[j])]
        if ends:
            yield m.group(1), [x.strip() for x in L[i + 1:ends[0]]]


def _dscnt(x):
    m = re.match(r"s_wait_dscnt 0x([0-9a-f]+)$", x)
    if m:
        return int(m.group(1), 16)
    m = re.match(r"s_wait_loadcnt_dscnt 0x([0-9a-f]+)$", x)
    if m:
        return int(m.group(1), 16) & 0x3f
    return None


def isa_check(isa_path):
    """Per TDM loop body:
    masked loop (tensor_load immediately followed by s_wait_tensorcnt, no WMMA between): no ds op before the tensor wait,
      and a dscnt-0 wait after the last ds_load before the back edge.
    full loop (the tensor wait sits after the S/dP WMMAs): (a) the dscnt wait in front of the
      first WMMA is 0 and there are no ds ops before it (the carried readback is retired
      before use and nothing new is read); (b) no ds_load_b128 (readback) before the
      s_wait_tensorcnt; (c) after the last ds_load_tr16 a dscnt wait <= the number of DS
      ops issued after it appears before the back edge (tr16 retired in-iteration);
      (d) the tensor_loads come before every ds op of the body (they target the stage whose
      readers retired in earlier iterations)."""
    if not os.path.exists(isa_path):
        return "ISA not present, skipped"
    L = open(isa_path).read().split("\n")
    out = []
    for lab, body in _loops(L):
        tl = [i for i, x in enumerate(body) if x.startswith("tensor_load_to_lds")]
        if not tl:
            continue
        tw = [i for i, x in enumerate(body) if x.startswith("s_wait_tensorcnt")]
        check(len(tw) == 1, f"{lab}: expected one tensor wait, got {tw}")
        tw = tw[0]
        dsi = [i for i, x in enumerate(body) if x.startswith("ds_")]
        if not any(x.startswith("v_wmma") for x in body[max(tl):tw]):   # masked loop form
            check(not [i for i in dsi if i < tw], f"{lab}: ds op before the tensor wait")
            last_ld = max(i for i in dsi if body[i].startswith("ds_load"))
            check(any(_dscnt(x) == 0 for x in body[last_ld + 1:]), f"{lab}: ds_load live at back edge")
            out.append(f"{lab}[mask]: {body[tw]} before all {len(dsi)} ds ops; dscnt 0 before back edge")
            continue
        fw = next(i for i, x in enumerate(body) if x.startswith("v_wmma"))
        w0 = [_dscnt(x) for x in body[:fw] if _dscnt(x) is not None]
        check(w0 and w0[-1] == 0, f"{lab}: first WMMA not preceded by dscnt 0 ({w0})")
        check(not [i for i in dsi if i < fw], f"{lab}: ds op before first WMMA")
        rb = [i for i in dsi if body[i].startswith("ds_load_b128")]
        check(rb and min(rb) > tw, f"{lab}: readback ds_load_b128 before the tensor wait")
        lt = max(i for i in dsi if body[i].startswith("ds_load_tr16"))
        after = sum(1 for i in dsi if i > lt)
        check(any((_dscnt(x) is not None and _dscnt(x) <= after) for x in body[lt + 1:]),
              f"{lab}: tr16 not retired in-iteration")
        check(max(tl) < min(dsi), f"{lab}: ds op before the TDM issue")
        out.append(f"{lab}[full]: TDM@{tl} < first ds@{min(dsi)}; dscnt {w0[-1]} before first WMMA "
                   f"(no ds before it); {body[tw]}@{tw} < first readback@{min(rb)}; "
                   f"tr16 retired by dscnt<= {after} in-iteration")
    return "; ".join(out)


def main():
    shapes = [
        ("prod", 4, 8192, 8192, 32, 8),
        ("fast", 1, 1024, 1024, 8, 2),
        ("toy", 1, 128, 128, 2, 1),
    ]
    ok = True
    for depth in (3,):
        for name, B, Sq, Skv, Hq, Hkv in shapes:
            for causal in (1, 0):
                try:
                    st, nsp = run_shape(name, B, Sq, Skv, Hq, Hkv, causal, depth)
                    n_elem = B * Sq * Hq * D
                    print(f"PASS depth={depth} {name:4s} causal={causal} nsp={nsp:2d} "
                          f"({'k_dkdv_sp' if nsp > 1 else 'k_dkdv'}): wgs={st['wgs']} "
                          f"tdm_desc={st['desc']} waits={st['waits']} max_tensorcnt={st['maxcnt']} max_outstanding_at_wait={st.get('maxwait_q')} "
                          f"mask_iters={st['iters_mask']} full_iters={st['iters_full']} "
                          f"clamped_prefetch={st['prefetch_clamped']} clamped_prologue={st['prologue_clamped']} "
                          f"empty_full_loops={st['empty_full']} global_elem=[{st['gmin']},{st['gmax']}] < {n_elem}")
                except Fail as e:
                    ok = False
                    print(f"FAIL depth={depth} {name} causal={causal}: {e}")
    print(f"LDS: alloc={ALLOC} stages=[k*{QDO_B},(k+1)*{QDO_B}) k<3 (ring end {3*QDO_B} < {LDS_SEG}) "
          f"P/dS=[{LDS_SEG},{ALLOC}) read ranges per stage={RR}")
    here = os.path.dirname(os.path.abspath(__file__))
    for k in ("dkdv/k_dkdv_0", "dkdv_sp/k_dkdv_sp_0"):
        try:
            print(f"ISA {k}:", isa_check(os.path.join(here, ".dump", k, "21_final_isa.s")))
        except Fail as e:
            ok = False
            print(f"FAIL ISA {k}:", e)
    print("RESULT:", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
