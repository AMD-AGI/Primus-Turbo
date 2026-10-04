#!/usr/bin/env python3
"""CPU bounds proof for kernels.py (stdlib only: no torch, no flydsl, no GPU).

    python3 bounds_proof.py [shape ...]      default: toy fast proxy prod rect edge64 gqa

The layout constants and helpers (_pow2_segments, _RD_ORDER, ...) are exec'd from
kernels.py itself, so the proof cannot drift from the kernel's constants; the per-thread
index formulas are re-derived here line by line from kernels.py and replayed for every
lane, every loop iteration and every kv/query tile of the extreme (head, batch) workgroups.

Checks (each prints its count; any violation raises):
  K1  k_dkdv TDM tiles: q pair in [0, Sq/32), outer extent Sq - q0 >= 32 (TDM ignores
      bounds), every segment's global elements inside q / do; LDS data writes inside their
      stage image, segments tile each row's [0, 2D) exactly, rings inside segment 0.
  K2  k_dkdv ring schedule with TDM_OPS_QDO ops per stage and tensorcnt waits TW_QDO / 0
      (in-order retirement): every LDS read hits a retired stage holding the right tile,
      no TDM targets a stage with an unconsumed read; tensorcnt 0 at exit.
  K3  k_dkdv LDS reads (readback, tr16, P/dS) inside written bytes; DS immediates < 64 KiB;
      epilogue images inside the dead ring; K/V fragment and LSE/delta loads in bounds.
  K4  causal split: query pairs < qp_start fully masked, pairs >= qp_start + nmaskp fully
      unmasked; division-free (qi, gh) counters == (ii // G, ii % G) for every ii.
  K5  dk/dv: every output element written exactly once (all workgroups, all lanes).
  Q1  k_dqg TDM tiles (kv block <= nkvt-1, extent >= 32, K/V elements in bounds), LDS
      writes; Q2 ring schedule with TDM_OPS_KV ops per stage, waits DQT_TW / 0;
      Q3 LDS reads (readback, tr16) inside written bytes, DS immediates; Q/dO fragment
      and LSE/delta loads in bounds, inside k_dqg's fake 1 GiB / 256 MiB extents;
      Q4 causal split (nfull iterations fully unmasked, nkvt_eff covers every attended
      kv); Q5 dq written exactly once; the XCD q-head remap is a bijection.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE / "kernels.py").read_text()


def _block(start, end):
    """Source from the line that starts with `start` to the next line starting with `end`."""
    i = SRC.index("\n" + start) + 1
    j = SRC.index("\n" + end, i) + 1
    return SRC[i:j]


C = {}
exec(_block("D_QK = ", "def _bv("), C)                 # noqa: S102  head dims, LDS layout
exec(_block("KV_STEP = 32", "def _dqg_tdm_impl("), C)  # noqa: S102  k_dqg constants
g = C.get
D_QK, D_V, BLOCK_KV, KV_STEP, TDM_DEPTH = g("D_QK"), g("D_V"), g("BLOCK_KV"), g("KV_STEP"), g("TDM_DEPTH")
XK, XV, S_ROW_B, LDS_SEG = g("XK_ROW_B"), g("XV_ROW_B"), g("S_ROW_B"), g("LDS_SEG")
QOFF, QDO_B, TDM_OPS_QDO, TW_QDO = g("QOFF"), g("QDO_B"), g("TDM_OPS_QDO"), g("TW_QDO")
NDT_QK, NDT_V, NDO_QK, NDO_V, NKV, EPI_CB = (g("NDT_QK"), g("NDT_V"), g("NDO_QK"), g("NDO_V"),
                                             g("NKV"), g("EPI_CB"))
DQ_BQW, VOFF, KV_B, TDM_OPS_KV, DQT_TW = g("DQ_BQW"), g("VOFF"), g("KV_B"), g("TDM_OPS_KV"), g("DQT_TW")
RD_ORDER, NKT, NQW = g("_RD_ORDER"), g("NKT"), g("NQW")
segs = g("_pow2_segments")
ALLOC_KV = LDS_SEG + 2 * 32 * S_ROW_B
LANES = range(32)


def lane_rc(lane):
    return lane % 16, lane // 16                                     # row, half


def lane_tr(lane):
    return (lane // 16) * 8 + lane % 8, ((lane // 8) % 2) * 8         # lane_r, lane_c


SHAPES = {  # name: (B, Sq, Skv, Hq, Hkv, causal)
    "toy": (1, 256, 256, 2, 2, 1), "fast": (1, 1024, 1024, 8, 8, 1),
    "proxy": (1, 4096, 4096, 64, 64, 1), "prod": (2, 4096, 4096, 128, 128, 1),
    "rect": (1, 256, 384, 2, 2, 1), "edge64": (2, 64, 64, 2, 2, 1),
    "gqa": (2, 512, 512, 8, 2, 1), "noncausal": (1, 256, 256, 2, 2, 0),
    "rect_short_kv": (1, 384, 256, 2, 2, 1),
}
COUNT = {}


def ok(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"{tag} violated: {msg}")
    COUNT[tag] = COUNT.get(tag, 0) + 1


# --------------------------------------------------------------- static layout facts
def tdm_image_writes(d, xrow):
    """Bytes the TDM ops of one [32][d] image write (data only: the pad is skipped, not
    written), relative to the image base; and the per-row coverage check."""
    spans = []
    for c0, w in segs(d):
        ok(w & (w - 1) == 0 and 2 <= w // 2 <= 256, "K1", f"pad interval {w}")
        pad_dw = (d + 8 - w) // 2
        ok(1 <= pad_dw <= 128 and (d + 8 - w) % 2 == 0, "K1", f"pad amount {pad_dw}")
        ok(w + (d + 8 - w) == xrow // 2, "K1", "segment stride == image row stride")
        for r in range(32):
            spans.append((r * xrow + 2 * c0, r * xrow + 2 * c0 + 2 * w))
    # segments tile each row's data [0, 2d) exactly, no overlap
    for r in range(32):
        row = sorted((a - r * xrow, b - r * xrow) for a, b in spans
                     if r * xrow <= a < (r + 1) * xrow)
        ok(row[0][0] == 0 and row[-1][1] == 2 * d
           and all(row[i][1] == row[i + 1][0] for i in range(len(row) - 1)), "K1", f"row tiling {row}")
    return spans


SPAN_K = tdm_image_writes(D_QK, XK)
SPAN_V = tdm_image_writes(D_V, XV)
ok(QOFF == 32 * XV and QDO_B == QOFF + 32 * XK, "K1", "k_dkdv stage = dO image + Q image")
ok(TDM_DEPTH * QDO_B <= LDS_SEG, "K1", "Q/dO ring inside segment 0")
ok(VOFF == 32 * XK and KV_B == VOFF + 32 * XV, "Q1", "k_dqg stage = K image + V image")
ok(TDM_DEPTH * KV_B <= LDS_SEG, "Q1", "K/V ring inside segment 0")
ok(TDM_OPS_QDO == len(segs(D_QK)) + len(segs(D_V)) == TDM_OPS_KV, "K2", "ops per stage")
# the wait immediates themselves are validated by the ring replays (K2, Q2)

# bank groups of the tr16 / readback row starts (perf, not safety): 16 distinct groups
for xrow in (XK, XV):
    ok(len({(r * xrow // 4) % 64 // 4 for r in range(16)}) == 16, "K3", f"bank groups {xrow}")


def written(img_spans, img_base, lo, hi):
    """[lo, hi) lies inside one data span of the image at img_base."""
    return any(img_base + a <= lo and hi <= img_base + b for a, b in img_spans)


# k_dkdv LDS reads relative to the stage base, and their DS immediates
DS_IMM = []
RD_KV = []      # (lo, hi) of every readback/tr16 byte range, stage-relative, per lane
for lane in LANES:
    row, half = lane_rc(lane)
    lr, lc = lane_tr(lane)
    for hh in range(2):
        for dt in range(NDT_QK):
            for u in range(2):
                imm = hh * 16 * XK + QOFF + dt * 64 + u * 32
                a = row * XK + half * 16 + imm
                DS_IMM.append(imm)
                ok(written(SPAN_K, QOFF, a, a + 16), "K3", f"Q readback {a}")
                RD_KV.append((a, a + 16))
        for dt in range(NDT_V):
            for u in range(2):
                imm = hh * 16 * XV + dt * 64 + u * 32
                a = row * XV + half * 16 + imm
                DS_IMM.append(imm)
                ok(written(SPAN_V, 0, a, a + 16), "K3", f"dO readback {a}")
                RD_KV.append((a, a + 16))
    for dtile in range(NDO_V):
        for r2 in (0, 16):
            imm = dtile * 32 + r2 * XV
            a = lr * XV + lc * 2 + imm
            DS_IMM.append(imm)
            ok(written(SPAN_V, 0, a, a + 16), "K3", f"b_do tr16 {a}")
    for dtile in range(NDO_QK):
        for r2 in (0, 16):
            imm = QOFF + dtile * 32 + r2 * XK
            a = lr * XK + lc * 2 + imm
            DS_IMM.append(imm)
            ok(written(SPAN_K, QOFF, a, a + 16), "K3", f"b_q tr16 {a}")
    # P/dS tiles [32 q][BLOCK_KV kv] bf16 at LDS_SEG (+32*S_ROW_B for dS)
    for hh in range(2):
        for kh in range(NKV):
            off = (hh * 16 + row) * S_ROW_B + kh * 32 + half * 16
            ok(off + 16 <= 32 * S_ROW_B and (off % S_ROW_B) + 16 <= BLOCK_KV * 2, "K3", "P/dS store")
    for kh in range(NKV):
        for r2 in (0, 16):
            a = lr * S_ROW_B + lc * 2 + kh * 32 + r2 * S_ROW_B
            ok(a + 16 <= 32 * S_ROW_B and (a % S_ROW_B) + 16 <= BLOCK_KV * 2, "K3", "P/dS tr16")
    # epilogue images: per kh dV [D_V rows] then dK [D_QK rows], EPI_CB bytes per d row
    for kh in range(NKV):
        ev = kh * (D_V + D_QK) * EPI_CB
        ek = ev + D_V * EPI_CB
        for dtile in range(max(NDO_V, NDO_QK)):
            o = (dtile * 16 + row) * EPI_CB + half * 16
            if dtile < NDO_V:
                ok(ev + o + 16 <= ek, "K3", "dV image store")
            if dtile < NDO_QK:
                ok(ek + o + 16 <= ek + D_QK * EPI_CB, "K3", "dK image store")
            ok(o % EPI_CB + 16 <= 32, "K3", "image store stays in the 32 B kv row")
        for sub in range(max(NDO_V, NDO_QK)):
            a = (sub * 16 + lr) * EPI_CB + lc * 2
            if sub < NDO_V:
                ok(ev + a + 16 + 0 <= ek and (a % EPI_CB) + 16 <= 32, "K3", "dV image tr16")
            if sub < NDO_QK:
                ok(ek + a + 16 <= ek + D_QK * EPI_CB, "K3", "dK image tr16")
ok(NKV * (D_V + D_QK) * EPI_CB <= TDM_DEPTH * QDO_B <= LDS_SEG, "K3", "epilogue inside dead ring")
ok(max(DS_IMM) < 65536 and 2 * QDO_B + max(DS_IMM) + 16 <= LDS_SEG + 2 * 32 * S_ROW_B, "K3",
   f"DS immediate {max(DS_IMM)}")

# k_dqg LDS reads relative to the stage base
DSQ_IMM = []
RD_Q = []
for lane in LANES:
    row, half = lane_rc(lane)
    lr, lc = lane_tr(lane)
    for kt, dt, w, u in RD_ORDER:
        if w == "k":
            imm = kt * 16 * XK + dt * 64 + u * 32
            a = row * XK + half * 16 + imm
            ok(written(SPAN_K, 0, a, a + 16), "Q3", f"K readback {a}")
        else:
            imm = VOFF + kt * 16 * XV + dt * 64 + u * 32
            a = row * XV + half * 16 + imm
            ok(written(SPAN_V, VOFF, a, a + 16), "Q3", f"V readback {a}")
        DSQ_IMM.append(imm)
        RD_Q.append((a, a + 16))
    for dtile in range(NDO_QK):
        for r2 in (0, 16):
            imm = dtile * 32 + r2 * XK
            a = lr * XK + lc * 2 + imm
            DSQ_IMM.append(imm)
            ok(written(SPAN_K, 0, a, a + 16), "Q3", f"K tr16 {a}")
ok(len(RD_ORDER) == 2 * NKT * (NDT_QK + NDT_V), "Q3", "readback count")
ok(max(DSQ_IMM) < 65536, "Q3", f"DS immediate {max(DSQ_IMM)}")


class Ring:
    """TDM tensorcnt model: ops retire in order; wait(n) retires all but the newest n."""

    def __init__(self, tag, ops_per_stage, nst):
        self.tag, self.k, self.q = tag, ops_per_stage, []
        self.stage_tile = [None] * nst       # tile whose TDM ops last targeted the stage
        self.pending = [0] * nst             # unretired ops per stage
        self.reads = [0] * nst               # reads of the stage not yet consumed

    def issue(self, stage, tile):
        ok(self.reads[stage] == 0, self.tag, f"TDM into stage {stage} with an unconsumed read")
        self.stage_tile[stage] = tile
        for _ in range(self.k):
            self.q.append(stage)
            self.pending[stage] += 1
        ok(len(self.q) <= 2 * self.k, self.tag, "more than two stages in flight")

    def wait(self, n):
        while len(self.q) > n:
            self.pending[self.q.pop(0)] -= 1

    def read(self, stage, tile):
        ok(self.pending[stage] == 0, self.tag, f"read of stage {stage} with TDM in flight")
        ok(self.stage_tile[stage] == tile, self.tag, f"stage {stage} holds {self.stage_tile[stage]} not {tile}")
        self.reads[stage] += 1

    def consume(self, stage):
        self.reads[stage] = 0


def wrap(qc, gc, G):
    g1 = gc + 1
    return (qc, g1) if g1 < G else (qc + 1, 0)


# ------------------------------------------------------------------- k_dkdv replay
def dkdv_shape(B, Sq, Skv, Hq, Hkv, causal, written_kv):
    G = Hq // Hkv
    nqt = Sq // 16
    nqt2 = nqt // 2
    cshift = Skv - Sq
    n_q = B * Sq * Hq
    ldl_max = B * Hq * Sq

    def clampqt(t):
        t = t if t < nqt2 else nqt2 - 1
        return 0 if t < 0 else t

    for bat in sorted({0, B - 1}):
        for hkv in sorted({0, Hkv - 1}):
            for bid in range(Skv // BLOCK_KV):
                kv0 = bid * BLOCK_KV
                _c = kv0 - cshift
                qp_start = (0 if _c < 0 else _c) // 32 if causal else 0
                nqp_eff = nqt2 - qp_start
                _u = kv0 + BLOCK_KV - 1 - cshift
                qsf = 0 if _u < 0 else (_u + 31) // 32
                qsf = min(qsf, nqt2)
                nm = min(max(qsf - qp_start, 0), nqp_eff)
                nmaskp = nm if causal else 0
                # K4: masked/unmasked classification by brute force over the 32x32 pair
                for qt in range(nqt2):
                    full = (not causal) or (kv0 + BLOCK_KV - 1 <= 32 * qt + cshift)
                    none = causal and (kv0 > 32 * qt + 31 + cshift)
                    if qt < qp_start:
                        ok(none, "K4", f"pair {qt} < qp_start {qp_start} not fully masked")
                    elif qt >= qp_start + nmaskp:
                        ok(full, "K4", f"pair {qt} in the full loop has masked entries")

                def tdm(qt, gh, stage):
                    ok(0 <= qt < nqt2 and 0 <= gh < G, "K1", f"tile ({qt},{gh})")
                    q0 = qt * 32
                    ok(Sq - q0 >= 32, "K1", "TDM outer extent < 32")
                    qh = hkv * G + gh
                    row0 = (bat * Sq + q0) * Hq + qh
                    for d in (D_V, D_QK):
                        for c0, w in segs(d):
                            first = row0 * d + c0
                            last = first + 31 * Hq * d + w - 1
                            ok(0 <= first and last < n_q * d, "K1", f"TDM global [{first},{last}]")
                    return (qt, gh)

                def ldl(qt, gh):
                    qh = hkv * G + gh
                    base_l = (bat * Hq + qh) * Sq
                    for hh in range(2):
                        for row in range(16):
                            idx = base_l + qt * 32 + hh * 16 + row
                            ok(0 <= idx < ldl_max, "K3", "LSE/delta index")

                ring = Ring("K2", TDM_OPS_QDO, TDM_DEPTH)
                # qloop_mask: own tile -> stage 0, wait 0, read stage 0 (consumed in-iteration)
                qi = gh = 0
                for ii in range(G * nmaskp):
                    ok((qi, gh) == divmod(ii, G), "K4", "mask-loop counters")
                    t = tdm(qp_start + qi, gh, 0)
                    ldl(qp_start + qi, gh)
                    ring.issue(0, t)
                    ring.wait(0)
                    ring.read(0, t)
                    ring.consume(0)
                    qi, gh = wrap(qi, gh, G)
                n = G * (nqp_eff - nmaskp)
                qt0 = qp_start + nmaskp
                ldl(clampqt(qt0), 0)
                # prologue: tile(0) -> s0, tile(min(1, n-1)) -> s1, wait, read back s0
                ring.issue(0, tdm(clampqt(qt0), 0, 0))
                q1w, g1w = wrap(0, 0, G)
                q1, g1 = (q1w, g1w) if 1 < n else (0, 0)
                ring.issue(1, tdm(clampqt(qt0 + q1), g1, 1))
                ring.wait(TW_QDO)
                ring.read(0, (clampqt(qt0), 0))
                cur, qi, gh = 0, 0, 0
                for ii in range(n):
                    ok((qi, gh) == divmod(ii, G), "K4", "full-loop counters")
                    st = cur
                    qn, gn = wrap(qi, gh, G)
                    qj, gj = (qn, gn) if ii + 1 < n else (qi, gh)
                    q2, g2 = wrap(qn, gn, G)
                    pf = (q2, g2) if ii + 2 < n else (qj, gj)
                    kk = min(ii + 2, n - 1)
                    ok(pf == divmod(kk, G), "K4", "prefetch counters == min(ii+2, n-1)")
                    nxo = 2 if cur == 0 else cur - 1
                    ncur = 0 if cur == 2 else cur + 1
                    ok(nxo == (ii + 2) % 3 and ncur == (ii + 1) % 3 and st == ii % 3, "K2", "stage rotation")
                    ring.issue(nxo, tdm(qt0 + pf[0], pf[1], nxo))   # top of the body, no wait
                    ldl(qt0 + qj, gj)
                    ring.consume(st)                     # carried readback -> S/dP WMMAs
                    ring.read(st, (qt0 + qi, gh))       # tr16 of this iteration's stage
                    ring.consume(st)                     # consumed by the dK/dV WMMAs
                    ring.wait(TW_QDO)
                    ring.read(ncur, (qt0 + qj, gj))      # readback for ii+1 (min(ii+1, n-1))
                    cur, qi, gh = ncur, qn, gn
                ring.wait(0)
                ok(not ring.q, "K2", "tensorcnt 0 at exit")
                # K/V fragments of this tile, dk/dv epilogue stores (vec8 index)
                rs_k, rs_v = Hkv * D_QK // 8, Hkv * D_V // 8
                base_k = bat * Skv * rs_k + hkv * D_QK // 8
                base_v = bat * Skv * rs_v + hkv * D_V // 8
                for kh in range(NKV):
                    for lane in LANES:
                        row, half = lane_rc(lane)
                        for base, rs, ndt, d in ((base_k, rs_k, NDT_QK, D_QK), (base_v, rs_v, NDT_V, D_V)):
                            for dt in range(ndt):
                                for t in (base + (kv0 + kh * 16 + row) * rs + half + dt * 4,):
                                    for tt in (t, t + 2):
                                        ok(0 <= tt * 8 and tt * 8 + 8 <= B * Skv * Hkv * d, "K3", "K/V frag")
                                        ok((tt * 8) % d + 8 <= d, "K3", "K/V frag inside its row")
                        for sub in range(max(NDO_V, NDO_QK)):
                            for base, rs, nd, d, key in ((base_v, rs_v, NDO_V, D_V, "v"),
                                                         (base_k, rs_k, NDO_QK, D_QK, "k")):
                                if sub < nd:
                                    t = base + (kv0 + kh * 16 + row) * rs + sub * 2 + half
                                    for e in range(8):
                                        el = t * 8 + e
                                        ok(0 <= el < B * Skv * Hkv * d, "K5", "dk/dv store")
                                        written_kv[key][el] = written_kv[key].get(el, 0) + 1


def dkdv_cover(B, Sq, Skv, Hq, Hkv):
    """K5 exactly-once over ALL workgroups, by closed form (cheap): each (bat, hkv, bid, kh,
    row, sub, half) writes 8 consecutive elements of one row; the map is injective."""
    for key, d, nd in (("v", D_V, NDO_V), ("k", D_QK, NDO_QK)):
        seen = set()
        for kvr in range(BLOCK_KV):                 # kh*16 + row
            for sub in range(nd):
                for half in range(2):
                    c = (sub * 2 + half) * 8        # column of the 8-element run
                    ok(c + 8 <= d, "K5", "run inside the row")
                    seen.add((kvr, c))
        ok(len(seen) == BLOCK_KV * d // 8, "K5", f"d{key} tile coverage {len(seen)}")


# -------------------------------------------------------------------- k_dqg replay
def dqg_shape(B, Sq, Skv, Hq, Hkv, causal, written_q):
    G = Hq // Hkv
    nkvt = Skv // KV_STEP
    cshift = Skv - Sq
    BQW = DQ_BQW
    # XCD remap bijection
    if Hq % 8 == 0:
        m = sorted((x % 8) * (Hq // 8) + x // 8 for x in range(Hq))
        ok(m == list(range(Hq)), "Q5", "XCD remap is a bijection")
    nblk = Sq // BQW
    ok(sorted(nblk - 1 - y for y in range(nblk)) == list(range(nblk)), "Q5", "descending tile walk")
    for bat in sorted({0, B - 1}):
        for qh in sorted({0, Hq - 1}):
            hkv = qh // G
            for bid in range(nblk):
                q0 = bid * BQW
                lim = (q0 + BQW + cshift + KV_STEP - 1) // KV_STEP
                lim = min(max(lim, 1), nkvt)
                nkvt_eff = lim if causal else nkvt
                t_ = q0 + cshift + 1
                nf = 0 if t_ < 0 else t_ // KV_STEP
                nf = min(nf, nkvt_eff)
                nfull = nf if causal else nkvt_eff
                nlast = nkvt_eff - 1
                ok(nlast >= 0 and nfull <= nkvt_eff, "Q4", "loop bounds")
                # Q4: every attended kv of every query of the tile is inside [0, 32*nkvt_eff);
                # iterations < nfull are fully unmasked for every query of the tile
                for q in (q0, q0 + BQW - 1):
                    hi = min(q + cshift, Skv - 1) if causal else Skv - 1
                    ok(hi < KV_STEP * nkvt_eff, "Q4", "kv coverage")
                for i in range(nfull):
                    ok((not causal) or KV_STEP * i + KV_STEP - 1 <= q0 + cshift, "Q4", "full iteration masked")

                def tdm(kb):
                    ok(0 <= kb <= nkvt - 1, "Q1", f"kv block {kb}")
                    kv0p = kb * KV_STEP
                    ok(Skv - kv0p >= KV_STEP, "Q1", "TDM outer extent")
                    row0 = (bat * Skv + kv0p) * Hkv + hkv
                    for d in (D_QK, D_V):
                        for c0, w in segs(d):
                            first = row0 * d + c0
                            last = first + (KV_STEP - 1) * Hkv * d + w - 1
                            ok(0 <= first and last < B * Skv * Hkv * d, "Q1", "TDM global")
                    return kb

                ring = Ring("Q2", TDM_OPS_KV, TDM_DEPTH)
                ring.issue(0, tdm(0))
                for s_ in range(1, TDM_DEPTH - 1):
                    ring.issue(s_, tdm(min(s_, nlast)))
                ring.wait(DQT_TW)
                ring.read(0, 0)
                cur = 0
                for ii in range(nkvt_eff):
                    kk = min(ii + TDM_DEPTH - 1, nlast)
                    nxo = (TDM_DEPTH - 1) if cur == 0 else cur - 1
                    ncur = 0 if cur == TDM_DEPTH - 1 else cur + 1
                    ok(cur == ii % 3 and nxo == (ii + 2) % 3 and ncur == (ii + 1) % 3, "Q2", "rotation")
                    ring.issue(nxo, tdm(kk))              # top of the body, no wait
                    ring.consume(cur)                     # carried K/V A operands -> S/dP WMMAs
                    ring.read(cur, ii)                    # dQ B operand tr16 of K(ii)
                    ring.wait(DQT_TW)
                    ring.read(ncur, min(ii + 1, nlast))   # readback for ii+1
                    ring.consume(cur)                     # tr16 operands -> dQ WMMAs
                    cur = ncur
                ring.wait(0)
                ok(not ring.q, "Q2", "tensorcnt 0 at exit")
                # Q/dO fragments, LSE/delta, dQ stores
                base_l = (bat * Hq + qh) * Sq
                for lane in LANES:
                    row, half = lane_rc(lane)
                    for qh_ in range(NQW):
                        qg = q0 + qh_ * 16 + row
                        ok(0 <= base_l + qg < B * Hq * Sq and (base_l + qg) * 4 < (1 << 28), "Q3", "lse")
                        for d, ndt in ((D_QK, NDT_QK), (D_V, NDT_V)):
                            rs = Hq * d // 8
                            base = bat * Sq * rs + qh * d // 8
                            for dt in range(ndt):
                                t = base + qg * rs + half + dt * 4
                                for tt in (t, t + 2):
                                    ok(tt * 8 + 8 <= B * Sq * Hq * d and tt * 16 < (1 << 30), "Q3", "Q/dO frag")
                                    ok((tt * 8) % d + 8 <= d, "Q3", "frag inside its row")
                        base_dq = bat * Sq * Hq * D_QK + qh * D_QK
                        for dtile in range(NDO_QK):
                            for si in range(8):
                                q_i = q0 + qh_ * 16 + half * 8 + si
                                el = base_dq + q_i * Hq * D_QK + dtile * 16 + row
                                ok(0 <= el < B * Sq * Hq * D_QK and el * 2 < (1 << 30), "Q5", "dq store")
                                written_q[el] = written_q.get(el, 0) + 1


def main(names):
    for name in names:
        B, Sq, Skv, Hq, Hkv, causal = SHAPES[name]
        assert Sq % 64 == 0 and Sq % DQ_BQW == 0 and Skv % 32 == 0 and Hq % Hkv == 0  # impl.py
        wkv = {"k": {}, "v": {}}
        wq = {}
        dkdv_shape(B, Sq, Skv, Hq, Hkv, causal, wkv)
        dkdv_cover(B, Sq, Skv, Hq, Hkv)
        dqg_shape(B, Sq, Skv, Hq, Hkv, causal, wq)
        # exactly-once on the replayed (extreme) workgroups: every replayed element once
        ok(all(v == 1 for v in wkv["k"].values()) and all(v == 1 for v in wkv["v"].values()),
           "K5", "dk/dv element written twice")
        ok(all(v == 1 for v in wq.values()), "Q5", "dq element written twice")
        nb, nh = len({0, B - 1}), len({0, Hkv - 1})
        ok(len(wkv["k"]) == nb * nh * Skv * D_QK and len(wkv["v"]) == nb * nh * Skv * D_V,
           "K5", f"dk/dv coverage {len(wkv['k'])}")
        ok(len(wq) == len({0, B - 1}) * len({0, Hq - 1}) * Sq * D_QK, "Q5", "dq coverage")
        print(f"[{name}] B{B} Sq{Sq} Skv{Skv} Hq{Hq} Hkv{Hkv} causal{causal}: ok", flush=True)
    print("checks:", " ".join(f"{k}={v}" for k, v in sorted(COUNT.items())))
    print(f"layout: D_QK {D_QK} D_V {D_V} rows {XK}/{XV} B, segments qk {segs(D_QK)} v {segs(D_V)}; "
          f"k_dkdv stage {QDO_B} B x {TDM_DEPTH} = {TDM_DEPTH * QDO_B} B, {TDM_OPS_QDO} TDM ops/stage, "
          f"wait {TW_QDO}; k_dqg stage {KV_B} B x {TDM_DEPTH} = {TDM_DEPTH * KV_B} B, {TDM_OPS_KV} ops, "
          f"wait {DQT_TW}; DS imm max {max(DS_IMM)} / {max(DSQ_IMM)}")
    print("BOUNDS_PROOF PASS")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SHAPES))
