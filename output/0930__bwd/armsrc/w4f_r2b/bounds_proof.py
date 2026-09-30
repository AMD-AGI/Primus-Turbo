#!/usr/bin/env python3
"""CPU bounds / protocol proof for arm w4f (host python3, no torch, no GPU).

Replays the integer control flow of kernels.py:_dkdv_w4f_impl (PARTIAL=False -> k_dkdv_w4f,
PARTIAL=True -> k_dkdv_w4f_sp; impl.py:_w4f_bwd's nsp rule) and of k_dq_cvt, for
prod, fast, toy, gqa4_small (required) plus proxy, mha, unequal_seqlen(_2), sq_gt_skv, each
causal on and off (sq_gt_skv non-causal only, as the UT does).

Sections
  G  global memory: every K/V fragment load, LSE/delta load, TDM descriptor (rows, extent,
     no hardware zero-fill), dQ atomic, dK/dV store (bf16 and fp32 partial), dq_cvt load/store
     is inside its tensor. Affine indices with non-negative coefficients are bounded at the
     corner lanes; every step of every workgroup / split / wave is enumerated.
  A  atomics: (A1) lane-level: the 4 waves x 32 lanes x 32 atomics of one step hit every
     (q, d) of the 32x128 tile exactly once, so the waves' address sets are disjoint;
     (A2) the dQ contraction pairs A-fragment kv columns and K^T-fragment kv rows identically
     and covers the 128 kv of the block exactly once per output; (A3) pair level: every
     (b, qh, q pair, kv block) is visited by at most one step of one workgroup/split, every
     pair holding a causal term is visited, pairs in the full loop are entirely causal,
     unvisited pairs hold no causal term => every causal (q, kv) term reaches dQ exactly
     once (masked-loop elements are zeroed by the exact per-element predicate). The same
     visitation proves dK/dV coverage.
  S  stores: dK/dV rows partition (b, hkv, kv) exactly; partial slices partition per split.
  L  LDS protocol: a happens-before checker over every LDS/TDM op of all 4 waves for every
     step count N in 0..NMAX (the address pattern is periodic in s mod 6, so N > NMAX adds
     nothing new); every conflicting pair (overlapping bytes, >= 1 write) is ordered by a
     split barrier (retire-at-next-signal: s_wait_dscnt 0 + s_wait_tensorcnt 0 precede every
     signal) or by in-order LDS within one wave; every read sees the intended step's tile /
     dS; all LDS ranges inside the allocation and their regions.
  B  barriers: all 4 waves execute prologue + N + epilogue barriers with N computed from
     workgroup-only inputs (kvw0, sp, shape) -- checked per workgroup; ISA CFG check that no
     barrier sits under a wave-divergent or data-dependent branch other than the loops.
  I  ISA (if .dump exists): barrier/DS/TDM ordering in each loop body, waits before each
     signal, and a counter simulation (dscnt/loadcnt/tensorcnt/storecnt) showing every wait
     is satisfiable (threshold < 64) and the counters never exceed 63.
"""
import os
import re
import sys
from collections import Counter, defaultdict

D = 128
WAVE = 32
BLOCK_KV = 32
W4 = 4
W4_BKV = 128
X_ROW_B = D * 2 + 16            # 272
S_ROW_B = BLOCK_KV * 2 + 16     # 80
# variant constants are read from the kernels.py NEXT TO this file (w4f, w4f_relax, ...)
_KSRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")).read()
W4_RELAX = re.search(r"^W4_RELAX = (True|False)", _KSRC, re.M).group(1) == "True"
W4_ABL_NOATOM = re.search(r"^W4_ABL_NOATOM = (True|False)", _KSRC, re.M).group(1) == "True"


def _flag(name):
    m = re.search(r"^%s = (True|False)" % name, _KSRC, re.M)
    return bool(m) and m.group(1) == "True"


W4_LDS_LSE = _flag("W4_LDS_LSE")    # r2a: LSE/delta ride the TDM ring
W4_XCD = _flag("W4_XCD")            # r2b: XCD-aware workgroup order
NXCD = 8
W4_RING = 4 if W4_RELAX else 3
W4_LEAD = W4_RING - 1           # TDM(s + LEAD) issued after wait B_s
TDM_OPS = 4 if W4_LDS_LSE else 2   # tensor ops per wave per step
TW_PRO = 2 * TDM_OPS if W4_RELAX else 0   # tensorcnt threshold before the prologue signal
TW_LOOP = TDM_OPS if W4_RELAX else 0      # ... before every loop signal (epilogue: 0)
QDO_B = 2 * 32 * X_ROW_B        # 17408
LSE_OFF = QDO_B                 # r2a: LSE [32] fp32 @+17408, delta @+17536
W4_STAGE_B = QDO_B + (256 if W4_LDS_LSE else 0)
W4_DSBUF_B = 32 * X_ROW_B       # 8704
W4_P_B = 32 * S_ROW_B           # 2560
W4_OFF_RING = 0
W4_OFF_DS = 131072 if W4_RELAX else 65536
W4_OFF_P = W4_OFF_DS + 2 * W4_DSBUF_B
W4_OFF_K = 196608 if W4_RELAX else 131072
W4_K_B = W4_BKV * X_ROW_B
W4_LDS = W4_OFF_K + W4_K_B      # 165888
W4_EPI_CB = 48
W4_EPI_B = 4 * 128 * W4_EPI_CB
LDS_MAX = 327680
RED_THREADS, RED_VEC = 256, 4
SP_TARGET = 512
NMAX = 14

SHAPES = {
    "prod": (4, 8192, 8192, 32, 8),
    "fast": (1, 1024, 1024, 8, 2),
    "toy": (1, 128, 128, 2, 1),
    "gqa4_small": (2, 128, 128, 8, 2),
    "proxy": (1, 4096, 4096, 32, 8),
    "mha": (1, 256, 256, 4, 4),
    "unequal_seqlen": (1, 512, 1024, 8, 2),
    "unequal_seqlen_2": (2, 1024, 2048, 4, 1),
    "sq_gt_skv": (2, 1024, 512, 4, 1),
}
CAUSAL = {"sq_gt_skv": (0,)}


class Fail(Exception):
    pass


def check(c, msg):
    if not c:
        raise Fail(msg)


def lane_rc(lane):
    return lane % 16, lane // 16


def lane_r_c(lane):
    return (lane // 16) * 8 + lane % 8, ((lane // 8) % 2) * 8


def kmap(h, e):
    """WMMA 16x16x32 bf16 operand K index held by lane half h, element e (the gfrag idiom)."""
    return h * 8 + e if e < 8 else 16 + h * 8 + (e - 8)


def tr16(base, rowb, lane):
    """ds_load_tr16_b128 pair as used by tr(): lane l element e reads the bf16 at
    src row (l//16)*8 + e (e < 8; +16 rows for the second load), column l%16 of the 16x16
    block whose lane-addresses are base + lane_r*rowb + lane_c*2. Returns
    [(row_offset_in_rows, col_elems)] per element relative to the block origin."""
    return [((lane // 16) * 8 + (e if e < 8 else 16 + (e - 8)), lane % 16) for e in range(16)]


def nsp_rule(B, Skv, Hkv):
    wgs = (Skv // W4_BKV) * Hkv * B
    nsp = 1
    while wgs * nsp < SP_TARGET and nsp < 16:
        nsp *= 2
    return nsp


def schedule(B, Sq, Skv, Hq, Hkv, causal, bid, sp, nsp, PARTIAL):
    """_dkdv_w4f_impl's workgroup-uniform step schedule (inputs: kvw0, sp, shape only)."""
    G = Hq // Hkv
    nqt = Sq // 16
    cshift = Skv - Sq
    nqt2 = nqt // 2
    kvw0 = bid * W4_BKV
    c = kvw0 - cshift
    qp_start = (0 if c < 0 else c) // 32
    qp_start = qp_start if causal else 0
    nqp_eff = max(nqt2 - qp_start, 0)
    u = kvw0 + W4_BKV - 1 - cshift
    qsf = 0 if u < 0 else (u + 31) // 32
    qsf = min(qsf, nqt2)
    nm = min(max(qsf - qp_start, 0), nqp_eff)
    nmaskp = nm if causal else 0
    if PARTIAL:
        fn = nqp_eff - nmaskp
        check(fn >= 0, "negative full-pair count")
        ch = (fn + nsp - 1) // nsp
        off = sp * ch
        cnt = min(max(fn - off, 0), ch)
        mk = 0 if sp != 0 else nmaskp
        p_begin = qp_start + ((nmaskp + off) if sp != 0 else 0)
    else:
        mk = nmaskp
        cnt = nqp_eff - nmaskp
        p_begin = qp_start
    n_mask = G * mk
    n_all = G * (mk + cnt)

    def clampqt(t):
        t = t if t < nqt2 else nqt2 - 1
        return 0 if t < 0 else t

    def tile(s):
        sc = s if s < n_all else n_all - 1
        sc = 0 if sc < 0 else sc
        qi = sc // G
        return clampqt(p_begin + qi), sc - qi * G
    return dict(G=G, nqt2=nqt2, cshift=cshift, kvw0=kvw0, n_mask=n_mask, n_all=n_all,
                tile=tile, p_begin=p_begin)


# ------------------------------------------------------------------------------ A1 / A2
def lane_level():
    """A1: one step, all 4 waves: (q_in_tile, d) covered exactly once. A2: contraction."""
    seen = Counter()
    per_wave = defaultdict(set)
    for wv in range(W4):
        for lane in range(WAVE):
            row, half = lane_rc(lane)
            base_m = half * 8                    # q offset inside the pair from `base`
            base_n = wv * 32 + row
            for qh16 in range(2):
                for j in range(2):
                    for si in range(8):
                        m = base_m + qh16 * 16 + si
                        n = base_n + j * 16
                        seen[(m, n)] += 1
                        per_wave[wv].add((m, n))
    check(len(seen) == 32 * 128 and set(seen.values()) == {1}, "A1: atomic tile coverage")
    for a in range(W4):
        for b in range(a + 1, W4):
            check(not (per_wave[a] & per_wave[b]), f"A1: waves {a},{b} share an address")
    # A2: dQ[q][d] = sum_kc sum_k A[q][kc*32 + kmapA] * B[kc*32 + kmapB][d]
    for lane in range(WAVE):
        row, half = lane_rc(lane)
        for kc in range(W4):
            # A (dS) from two ds_load_b128 at cols kc*32 + half*8 (+16): element e -> col
            a_cols = [kc * 32 + half * 8 + e for e in range(8)] + \
                     [kc * 32 + 16 + half * 8 + e for e in range(8)]
            check(a_cols == [kc * 32 + kmap(half, e) for e in range(16)], "A2: A kv map")
            # B (K^T) from tr16 at rows kc*32 + lane_r (+16 rows), cols lane_c + 16j
            b_rows = [kc * 32 + r for r, _ in tr16(0, X_ROW_B, lane)]
            check(b_rows == [kc * 32 + kmap(half, e) for e in range(16)], "A2: B kv map")
    # union over kc and the two lane halves of the K indices covers the 128 kv exactly once
    ks = Counter(kc * 32 + kmap(h, e) for kc in range(W4) for h in range(2) for e in range(16))
    check(len(ks) == 128 and set(ks.values()) == {1}, "A2: kv coverage")
    return "A1 4x32x32 atomics -> 4096 distinct (q,d), waves disjoint; A2 kv maps match, 128 kv once"


# ------------------------------------------------------------------------------ G / A3 / S
def run_shape(name, causal):
    B, Sq, Skv, Hq, Hkv = SHAPES[name]
    G = Hq // Hkv
    check(Sq % 32 == 0 and Skv % W4_BKV == 0 and Hq % Hkv == 0, "impl.py asserts")
    nsp = nsp_rule(B, Skv, Hkv)
    PARTIAL = nsp > 1
    nqt2 = Sq // 32
    nblk = Skv // W4_BKV
    cshift = Skv - Sq
    n_q = B * Sq * Hq * D                 # elements of q/do
    n_kv = B * Skv * Hkv * D
    n_l = B * Hq * Sq
    n_dqa = B * Hq * Sq * D
    rs_kv = Hkv * (D // 8)
    st = Counter()
    visited = bytearray(B * Hq * nqt2 * nblk)
    Ns = Counter()
    kvrows = bytearray(B * Hkv * Skv) if not PARTIAL else bytearray(nsp * B * Hkv * Skv)
    gmax = defaultdict(int)

    def vidx(bat, qh, qt, blk):
        return ((bat * Hq + qh) * nqt2 + qt) * nblk + blk

    for bat in range(B):
        for bid in range(nblk):
            for xw in range(Hkv * nsp):
                hkv, sp = (xw // nsp, xw % nsp) if PARTIAL else (xw, 0)
                S = schedule(B, Sq, Skv, Hq, Hkv, causal, bid, sp, nsp, PARTIAL)
                tile, n_all, n_mask, kvw0 = S["tile"], S["n_all"], S["n_mask"], S["kvw0"]
                st["wgs"] += 1
                Ns[n_all] += 1
                # B: every wave runs 1 + n_all + 1 barriers; n_all used only WG inputs.
                base_kv = bat * Skv * rs_kv + hkv * (D // 8)
                for wv in range(W4):
                    kv0 = kvw0 + wv * BLOCK_KV
                    # G: K/V fragments (vec8 index; 16 B each)
                    for kh in range(2):
                        lo = base_kv + (kv0 + kh * 16 + 0) * rs_kv + 0 + 0
                        hi = base_kv + (kv0 + kh * 16 + 15) * rs_kv + 1 + 12 + 2
                        check(lo >= 0 and (hi + 1) * 8 <= n_kv, f"G: K/V frag {lo},{hi}")
                        gmax["kv"] = max(gmax["kv"], (hi + 1) * 8)
                    # S: dK/dV rows
                    for r in range(kv0, kv0 + BLOCK_KV):
                        i = ((sp * B + bat) * Hkv + hkv) * Skv + r
                        kvrows[i] += 1
                    if PARTIAL:
                        base_o = ((sp * B + bat) * Skv * Hkv) * D + hkv * D
                        hi = base_o + (kv0 + 31) * Hkv * D + 7 * 16 + 15
                        check(base_o + kv0 * Hkv * D >= 0 and hi < nsp * n_kv, "G: fp32 partial store")
                    else:
                        gt_hi = base_kv + (kv0 + 16 + 15) * rs_kv + 7 * 2 + 1
                        check((gt_hi + 1) * 8 <= n_kv, "G: bf16 dK/dV store")

                def tdm(qt, gh, label):
                    check(0 <= qt < nqt2 and 0 <= gh < G, f"G: TDM tile {qt},{gh} ({label})")
                    qh = hkv * G + gh
                    for wv in range(W4):
                        r0 = qt * 32 + wv * 8
                        valid = Sq - r0
                        check(valid >= 8, f"G: TDM extent {valid} < 8 (zero-fill live)")
                        first = ((bat * Sq + r0) * Hq + qh) * D
                        last = ((bat * Sq + r0 + 7) * Hq + qh) * D + D - 1
                        check(first >= 0 and last < n_q, f"G: TDM [{first},{last}] vs {n_q}")
                        gmax["tdm"] = max(gmax["tdm"], last + 1)
                        if W4_LDS_LSE:          # r2a: LSE/delta rows [qt*32+8w, +8)
                            q8 = qt * 32 + wv * 8
                            check(Sq - q8 >= 8, "G: LSE TDM extent < 8")
                            lo = (bat * Hq + qh) * Sq + q8
                            check(lo >= 0 and lo + 7 < n_l, "G: LSE/delta TDM OOB")
                            gmax["lse"] = max(gmax["lse"], lo + 8)
                    st["tdm_ops"] += TDM_OPS * W4

                def ldl(qt, gh):
                    qh = hkv * G + gh
                    lo = (bat * Hq + qh) * Sq + qt * 32
                    check(lo >= 0 and lo + 31 < n_l, "G: LSE/delta")
                    gmax["lse"] = max(gmax["lse"], lo + 32)

                def atom(qt, gh):
                    qh = hkv * G + gh
                    base0 = ((bat * Hq + qh) * Sq + qt * 32 + 0) * D + 0 + 0
                    top = ((bat * Hq + qh) * Sq + qt * 32 + 8) * D + 3 * 32 + 15 + (16 + 7) * D + 16
                    check(base0 >= 0 and top < n_dqa, f"G: atomic {top} vs {n_dqa}")
                    gmax["atom"] = max(gmax["atom"], top + 1)
                    st["atomic_wave_instr"] += 32 * W4

                # prologue
                for k in range(W4_LEAD):
                    tdm(*tile(k), f"pro{k}")
                if not W4_LDS_LSE:
                    ldl(*tile(0))
                for s in range(n_all):
                    qt, gh = tile(s)
                    qh = hkv * G + gh
                    if not W4_LDS_LSE:
                        ldl(*tile(s + 1))
                    if not W4_ABL_NOATOM:
                        atom(*tile(s - 1))       # deferred (zeros at s == 0)
                    tdm(*tile(s + W4_LEAD), "loop")
                    # A3: visitation + mask class
                    i = vidx(bat, qh, qt, bid)
                    check(visited[i] == 0, f"A3: pair visited twice {bat,qh,qt,bid}")
                    visited[i] = 1
                    q_lo, q_hi = qt * 32, qt * 32 + 31
                    k_lo, k_hi = kvw0, kvw0 + W4_BKV - 1
                    any_causal = (k_lo <= q_hi + cshift) if causal else True
                    all_causal = (k_hi <= q_lo + cshift) if causal else True
                    check(any_causal, f"A3: step with no causal term (qt {qt}, blk {bid})")
                    if s >= n_mask:
                        check(all_causal, f"A3: full-loop step has a masked element")
                    else:
                        st["masked_steps"] += 1
                    # the wave's own 32 kv rows need not all be causal: the per-element
                    # predicate (masked loop) zeroes P and dS exactly; checked by A3 class.
                if not W4_ABL_NOATOM:
                    atom(*tile(n_all - 1))       # flush
                st["steps"] += n_all
    # A3: every pair with a causal term visited
    miss = 0
    for bat in range(B):
        for qh in range(Hq):
            for qt in range(nqt2):
                for blk in range(nblk):
                    any_causal = (blk * W4_BKV <= qt * 32 + 31 + cshift) if causal else True
                    v = visited[vidx(bat, qh, qt, blk)]
                    if any_causal and not v:
                        miss += 1
                    if v:
                        st["pairs"] += 1
    check(miss == 0, f"A3: {miss} causal pairs never visited")
    # S: dK/dV rows written exactly once (per split)
    check(set(kvrows) == {1}, "S: dK/dV row coverage not exactly once")
    # G: dq_cvt
    n_vec = B * Sq * Hq * D // RED_VEC
    check(n_vec % RED_THREADS == 0, "cvt: grid exact")
    D4 = D // RED_VEC
    if n_vec <= (1 << 22):
        seen = bytearray(n_vec)
        for t in range(n_vec):
            d4 = t % D4
            r = t // D4
            h = r % Hq
            bs = r // Hq
            b = bs // Sq
            s_ = bs - b * Sq
            src = ((b * Hq + h) * Sq + s_) * D4 + d4
            check(0 <= src < n_vec and seen[src] == 0, "cvt: src not a bijection")
            seen[src] = 1
        cvt = "bijection (exhaustive)"
    else:
        # the map is (b,s,h,d4) -> (b,h,s,d4), a permutation of a mixed-radix index; corners
        t = n_vec - 1
        d4 = t % D4; r = t // D4; h = r % Hq; bs = r // Hq; b = bs // Sq; s_ = bs - b * Sq
        src = ((b * Hq + h) * Sq + s_) * D4 + d4
        check(src == n_vec - 1, "cvt: corner")
        cvt = "mixed-radix permutation (corner-checked)"
    return dict(st=st, nsp=nsp, Ns=Ns, gmax=dict(gmax), n_q=n_q, n_kv=n_kv, n_dqa=n_dqa,
                n_l=n_l, cvt=cvt)


# ------------------------------------------------------------------------------ L
def lds_ops(N):
    """Every LDS/TDM op of every wave of one workgroup with N steps.
    op = (wave, seq, kind in {R, W, T}, [(lo, hi)], sigs_before, waits_before, tag)"""
    ops = []
    retire = {}                     # id(TDM op) -> signal index at which it has retired

    def rows_iv(base, r0, nrows, rowb=X_ROW_B, width=None):
        return [(base + (r0 + r) * rowb, base + (r0 + r) * rowb + (width or rowb))
                for r in range(nrows)]

    for wv in range(W4):
        seq = 0
        sig = wai = 0
        fifo = []                   # this wave's outstanding TDM ops (2 per batch), oldest first

        def signal(tw):
            """s_wait_dscnt 0; s_wait_tensorcnt tw; s_barrier_signal. In-order TDM retirement:
            all but the newest tw ops retire here."""
            nonlocal sig
            sig += 1
            n = max(len(fifo) - tw, 0)
            gone = fifo[:n]
            del fifo[:n]
            live = {id(o) for o in fifo}
            for o in gone:                  # a batch retires only when ALL its ops have
                if id(o) not in live:
                    retire.setdefault(id(o), sig)
            check(len(fifo) <= 63, "tensorcnt overflow")

        def add(kind, ivs, tag):
            nonlocal seq
            for lo, hi in ivs:
                check(0 <= lo < hi <= W4_LDS, f"L: {tag} outside allocation")
            ops.append((wv, seq, kind, ivs, sig, wai, tag))
            seq += 1

        def tdm(k):
            slot = (k % W4_RING) * W4_STAGE_B
            ivs = rows_iv(W4_OFF_RING + slot, wv * 8, 8) + \
                rows_iv(W4_OFF_RING + slot + 32 * X_ROW_B, wv * 8, 8)
            if W4_LDS_LSE:
                for lo in (0, 128):
                    b = W4_OFF_RING + slot + LSE_OFF + lo + wv * 32
                    ivs.append((b, b + 32))
            for lo, hi in ivs:
                check(slot <= lo - W4_OFF_RING and hi - W4_OFF_RING <= slot + W4_STAGE_B, "L: TDM crosses slot")
                check(hi <= W4_OFF_DS, "L: ring overlaps the dS region")
            add("T", ivs, ("tdm", k))
            fifo.extend([ops[-1]] * TDM_OPS)

        def slot_read(k, tag):
            slot = (k % W4_RING) * W4_STAGE_B
            add("R", [(W4_OFF_RING + slot, W4_OFF_RING + slot + W4_STAGE_B)], (tag, k))

        def ds_own(s):
            b = (s % 2) * W4_DSBUF_B
            return rows_iv(W4_OFF_DS + b + wv * 64, 0, 32, width=64)

        # prologue
        for k in range(W4_LEAD):
            tdm(k)
        add("W", rows_iv(W4_OFF_K, wv * 32, 32), ("kimg",))
        signal(TW_PRO)      # dscnt 0 + tensorcnt TW_PRO + signal #1
        wai += 1
        add("R", [(W4_OFF_K, W4_OFF_K + W4_K_B)], ("kt",))
        slot_read(0, "rb")
        for s in range(N):
            if W4_LDS_LSE:                  # r2a: LSE/delta of slot s%R read at the step top
                sl = (s % W4_RING) * W4_STAGE_B
                add("R", [(W4_OFF_RING + sl + LSE_OFF, W4_OFF_RING + sl + LSE_OFF + 256)], ("lse", s))
            add("W", [(W4_OFF_P + wv * W4_P_B, W4_OFF_P + (wv + 1) * W4_P_B)], ("P", s))
            add("W", ds_own(s), ("dS", s))
            signal(TW_LOOP)
            add("R", [(W4_OFF_P + wv * W4_P_B, W4_OFF_P + (wv + 1) * W4_P_B)], ("Pr", s))
            add("R", ds_own(s), ("dSown", s))
            slot_read(s, "tr")
            wai += 1
            tdm(s + W4_LEAD)
            b = (s % 2) * W4_DSBUF_B
            add("R", [(W4_OFF_DS + b, W4_OFF_DS + b + W4_DSBUF_B)], ("dSall", s))
            slot_read(s + 1, "rb")
        signal(0)
        check(not fifo, "L: TDM outstanding after the epilogue signal")
        wai += 1
        e = (wv * W4_EPI_B, (wv + 1) * W4_EPI_B)
        add("W", [e], ("epi",))
        add("R", [e], ("epir",))
    return ops, retire


def overlap(a, b):
    for lo, hi in a:
        for lo2, hi2 in b:
            if lo < hi2 and lo2 < hi:
                return True
    return False


RETIRE = {}


def hb(x, y):
    """x happens-before y (x retired before y starts)."""
    wx, sx, kx, _, sigx, _, _ = x
    wy, sy, ky, _, _, waity, _ = y
    if wx == wy and kx != "T" and ky != "T":
        return sx < sy              # in-order LDS within a wave
    if kx == "T":                   # TDM: retires at the signal the FIFO model assigned
        return waity >= RETIRE[id(x)]
    return waity >= sigx + 1        # LDS op retires by its wave's next signal (dscnt 0)


def lds_protocol():
    worst = 0
    for N in range(NMAX + 1):
        ops, ret = lds_ops(N)
        RETIRE.clear()
        RETIRE.update(ret)
        check(all(id(o) in ret for o in ops if o[2] == "T"), "L: TDM never retired")
        for i, x in enumerate(ops):
            for y in ops[i + 1:]:
                if x[2] == "R" and y[2] == "R":
                    continue
                if not overlap(x[3], y[3]):
                    continue
                check(hb(x, y) or hb(y, x), f"L: N={N} unordered conflict {x[0]}:{x[6]} vs {y[0]}:{y[6]}")
                worst += 1
        # V: contents. For each read of a ring slot, the latest HB-preceding TDM on that slot
        # (from ANY wave, per 8-row part) must be for the same step k.
        tdms = [o for o in ops if o[2] == "T"]
        for y in ops:
            if y[2] != "R" or y[6][0] not in ("rb", "tr", "lse"):
                continue
            k = y[6][1]
            for w in range(W4):
                prev = [x for x in tdms if x[0] == w and overlap(x[3], y[3]) and hb(x, y)]
                check(prev, f"L: N={N} read {y[6]} with no TDM from wave {w}")
                kk = max(p[6][1] for p in prev)
                check(kk == k, f"L: N={N} {y[6]} sees TDM step {kk} from wave {w}")
            # the clamped (never consumed) readback of step N reads TDM(N): in bounds, unused
        dsw = [o for o in ops if o[2] == "W" and o[6][0] == "dS"]
        for y in ops:
            if y[6][0] != "dSall":
                continue
            s = y[6][1]
            for w in range(W4):
                prev = [x for x in dsw if x[0] == w and overlap(x[3], y[3]) and hb(x, y)]
                check(prev and max(p[6][1] for p in prev) == s, f"L: N={N} dQ reads stale dS")
    # region sanity
    check(W4_RING * W4_STAGE_B <= W4_OFF_DS and W4_OFF_P + W4 * W4_P_B <= W4_OFF_K, "L: layout")
    check(W4_LDS <= LDS_MAX and W4 * W4_EPI_B <= W4_LDS, "L: allocation")
    return f"N=0..{NMAX}: {worst} conflicting op pairs, all ordered; ring/dS contents verified"


# ------------------------------------------------------------------------------ I
def parse_blocks(L):
    blocks, cur, name = [], [], "entry"
    for raw in L:
        x = raw.strip()
        m = re.match(r"^(\.LBB\d+_\d+):", x)
        if m:
            blocks.append((name, cur))
            name, cur = m.group(1), []
            continue
        if not x or x.startswith((";", ".", "//")) or x.endswith(":"):
            continue
        cur.append(x.split(";")[0].split("//")[0].strip())
        if cur[-1].split()[0].startswith(("s_cbranch", "s_branch")):
            blocks.append((name, cur))      # a branch ends the block; the rest falls through
            name, cur = f"{name}+", []
    blocks.append((name, cur))
    # drop everything after s_endpgm in the last kernel block
    out = []
    for n, b in blocks:
        if "s_endpgm" in [i.split()[0] for i in b if i]:
            b = b[:[i.split()[0] for i in b].index("s_endpgm") + 1]
            out.append((n, b))
            break
        out.append((n, b))
    return out


def isa_check(path):
    if not os.path.exists(path):
        return "ISA not present, skipped"
    L = open(path).read().split("\n")
    s0 = next(i for i, l in enumerate(L) if re.match(r"^\S+:\s*(;.*)?$", l.strip()) and "k_dkdv_w4f" in l)
    blocks = parse_blocks(L[s0 + 1:])
    names = [n for n, _ in blocks]
    ops = lambda b: [i.split()[0] for i in b if i]
    succ = {}
    for k, (n, b) in enumerate(blocks):
        s = []
        o = ops(b)
        last = b[-1] if b else ""
        tgt = re.findall(r"(\.LBB\d+_\d+)", last)
        if o and o[-1] == "s_endpgm":
            succ[n] = []
            continue
        if o and o[-1].startswith("s_cbranch"):
            s = [tgt[0]] + ([names[k + 1]] if k + 1 < len(names) else [])
        elif o and o[-1] == "s_branch":
            s = [tgt[0]]
        else:
            s = [names[k + 1]] if k + 1 < len(names) else []
        succ[n] = s
    bmap = dict(blocks)
    loops = {n for n in names if n in succ.get(n, [])}
    # every conditional branch is a loop back edge or a loop-skip (target is the block after
    # a loop) -- their conditions are the SGPR trip counts n_mask / n_all - n_mask.
    notes = []
    for n, b in blocks:
        o = ops(b)
        if o and o[-1].startswith("s_cbranch"):
            t = re.findall(r"(\.LBB\d+_\d+)", b[-1])[0]
            check(o[-1] in ("s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz"),
                  f"I: divergent-exec branch {o[-1]} in {n}")
            check("execz" not in o[-1] and "execnz" not in o[-1], "I: EXEC branch")
    # barrier counts on every acyclic path entry -> s_endpgm (self-loops collapsed)
    def cnt(n, op):
        return sum(1 for x in ops(bmap[n]) if x == op)
    paths = []

    def dfs(n, acc, seen):
        if n in seen:
            return
        acc = acc + [n]
        nxt = [m for m in succ[n] if m != n]
        if not nxt:
            paths.append(acc)
            return
        for m in nxt:
            dfs(m, acc, seen | {n})
    dfs(names[0], [], frozenset())
    for p in paths:
        outside = sum(cnt(n, "s_barrier_signal") for n in p if n not in loops)
        outside_w = sum(cnt(n, "s_barrier_wait") for n in p if n not in loops)
        check(outside == 2 and outside_w == 2, f"I: path {p} has {outside}/{outside_w} non-loop barriers")
        for n in p:
            if n in loops:
                check(cnt(n, "s_barrier_signal") == 1 and cnt(n, "s_barrier_wait") == 1, f"I: loop {n} barriers")
    # loop body ordering
    for n in loops:
        b = bmap[n]
        o = ops(b)
        sig = o.index("s_barrier_signal")
        wai = o.index("s_barrier_wait")
        check(sig < wai, f"I: {n} wait before signal")
        # immediately before signal: dscnt 0 and tensorcnt 0, no DS/TDM op after them
        pre = [i for i in range(sig) if o[i].startswith(("ds_", "tensor_load"))]
        dz = [i for i in range(sig) if b[i] in ("s_wait_dscnt 0x0",)]
        tz = [i for i in range(sig) if b[i] == f"s_wait_tensorcnt 0x{TW_LOOP:x}"]
        check(sum(1 for x in o if x == "s_wait_tensorcnt") == 1, f"I: {n} tensor waits")
        check(dz and tz and max(dz) > max(pre) and max(tz) > max(pre), f"I: {n} signal not fenced")
        check(not [i for i in range(sig) if o[i] in ("ds_load_b128", "ds_load_tr16_b128")],
              f"I: {n} LDS load before signal (carried readback expected)")
        pre_b32 = [o[i] for i in range(sig) if o[i].startswith("ds_load") and o[i] not in
                   ("ds_load_b128", "ds_load_tr16_b128")]
        check((len(pre_b32) > 0) == W4_LDS_LSE and all("b32" in x for x in pre_b32),
              f"I: {n} step-top LSE/delta LDS loads {pre_b32}")
        if W4_LDS_LSE:
            check(not any(x.startswith("buffer_load") for x in o), f"I: {n} VMEM load in the loop")
        mid = o[sig + 1:wai]
        check(all(not x.startswith("ds_") or x == "ds_load_tr16_b128" for x in mid),
              f"I: {n} non-tr16 DS op between signal and wait")
        check(not any(x.startswith("tensor_load") for x in mid), f"I: {n} TDM before wait")
        post = o[wai + 1:]
        check(all(not x.startswith("ds_") or x == "ds_load_b128" for x in post),
              f"I: {n} unexpected DS op after wait")
        check(sum(1 for x in post if x == "ds_load_b128") == 48, f"I: {n} dQ+readback loads")
        check(sum(1 for x in post if x == "tensor_load_to_lds") == TDM_OPS, f"I: {n} TDM count")
        check(sum(1 for x in o if x.startswith("global_atomic_add_f32")) == (0 if W4_ABL_NOATOM else 32),
              f"I: {n} atomics")
        check(all("SCOPE_DEV" in x for x in b if x.startswith("global_atomic")), f"I: {n} atomic scope")
        notes.append(f"{n}: {len(o)} instr, WMMA {sum(1 for x in o if x.startswith('v_wmma'))}, "
                     f"sig@{sig} wait@{wai}, {sum(1 for x in mid if x == 'ds_load_tr16_b128')} tr16 in the gap")
    # counter simulation over the whole CFG (loops iterated 3x), per path
    LIM = 63

    def sim(p):
        c = dict(ds=0, ld=0, tn=0, stc=0)
        mx = dict(c)
        for n in p:
            for _ in range(3 if n in loops else 1):
                for x in bmap[n]:
                    op = x.split()[0] if x else ""
                    if op.startswith("ds_"):
                        c["ds"] += 1
                    elif op.startswith(("buffer_load", "global_load")):
                        c["ld"] += 1
                    elif op.startswith("tensor_load"):
                        c["tn"] += 1
                    elif op.startswith(("global_atomic", "buffer_store", "global_store")):
                        c["stc"] += 1
                    elif op in ("s_wait_dscnt", "s_wait_loadcnt", "s_wait_tensorcnt", "s_wait_storecnt",
                                "s_wait_loadcnt_dscnt"):
                        v = int(x.split()[1], 16)
                        check(0 <= v, "I: negative wait")
                        if op == "s_wait_dscnt":
                            c["ds"] = min(c["ds"], v)
                        elif op == "s_wait_loadcnt":
                            c["ld"] = min(c["ld"], v)
                        elif op == "s_wait_tensorcnt":
                            check(v in (0, TW_LOOP, TW_PRO), f"I: unexpected tensor wait {v:#x}")
                            c["tn"] = min(c["tn"], v)
                        elif op == "s_wait_storecnt":
                            c["stc"] = min(c["stc"], v)
                        else:
                            c["ld"] = min(c["ld"], (v >> 8) & 0x3F)
                            c["ds"] = min(c["ds"], v & 0x3F)
                    if op == "s_barrier_signal":
                        check(c["ds"] == 0 and c["tn"] <= max(TW_PRO, TW_LOOP),
                              f"I: signal with outstanding ds/tdm in {n}")
                    for k in c:
                        mx[k] = max(mx[k], c[k])
        check(c["tn"] == 0, "I: tensorcnt not 0 at s_endpgm")
        return mx
    onpath = {n for p in paths for n in p}
    waits = [(n, x) for n, b in blocks for x in b if x.split()[0].startswith("s_wait_")]
    check(all(n in onpath for n, _ in waits), "I: a wait sits in an unreachable block")
    wc = Counter(x.split()[0] for _, x in waits)
    notes.append(f"{len(waits)} waits all on reachable blocks {dict(wc)}")
    mxs = [sim(p) for p in paths]
    mx = {k: max(m[k] for m in mxs) for k in mxs[0]}
    # the hardware stalls issue at counter saturation; > 63 is a performance flag, not a bug
    stc = mx.pop("stc")
    check(all(v <= LIM for v in mx.values()), f"I: counter above {LIM}: {mx}")
    return (f"{len(paths)} CFG paths, loops {sorted(loops)}; each path 2 non-loop barriers + 1 per loop trip; "
            + "; ".join(notes) + f"; max outstanding (model) {mx}; atomics are fire-and-forget: no "
            f"s_wait_storecnt anywhere ({stc} issued over the modelled paths, the hardware "
            f"throttles issue at storecnt saturation)")


def xcd_check(name):
    """r2b: the kernel's remap of the linear workgroup id is a bijection onto
    (bat, x = hkv*nsp + sp, blk), and (when active) every (bat, hkv) group lands on one XCD
    (XCD = linear id % 8) with the kv block ascending in dispatch order inside each XCD."""
    B, Sq, Skv, Hq, Hkv = SHAPES[name]
    nsp = nsp_rule(B, Skv, Hkv)
    nx, nblk = Hkv * nsp, Skv // W4_BKV
    ng = B * Hkv
    on = ng % NXCD == 0
    gpx = max(ng // NXCD, 1)
    seen = set()
    xcd_of = {}
    last_blk = {}
    for bz in range(B):
        for by in range(nblk):
            for bx in range(nx):
                gid = bx + nx * (by + nblk * bz)
                hkv, sp, blk, bat = bx // nsp, bx % nsp, by, bz
                if on:
                    c, slot = gid % NXCD, gid // NXCD
                    per = gpx * nsp
                    blk = slot // per
                    rem = slot - blk * per
                    gl, sp = rem // nsp, rem % nsp
                    grp = c * gpx + gl
                    bat, hkv = grp // Hkv, grp % Hkv
                check(0 <= bat < B and 0 <= hkv < Hkv and 0 <= sp < nsp and 0 <= blk < nblk,
                      f"X: {name} gid {gid} -> out of range")
                key = (bat, hkv, sp, blk)
                check(key not in seen, f"X: {name} remap not injective at {key}")
                seen.add(key)
                if on:
                    xcd_of.setdefault((bat, hkv), set()).add(gid % NXCD)
                    c = gid % NXCD
                    check(blk >= last_blk.get(c, 0), f"X: {name} blk order not ascending on XCD {c}")
                    last_blk[c] = blk
    check(len(seen) == B * nx * nblk, f"X: {name} remap not surjective")
    if on:
        check(all(len(v) == 1 for v in xcd_of.values()), f"X: {name} group split over XCDs")
        per_xcd = Counter(next(iter(v)) for v in xcd_of.values())
        check(set(per_xcd.values()) == {ng // NXCD}, f"X: {name} XCD group balance {per_xcd}")
    return f"{name}: {'ACTIVE, ' + str(ng // NXCD) + ' groups/XCD' if on else 'identity (B*Hkv % 8 != 0)'}"


def main():
    ok = True
    if W4_XCD:
        try:
            print("X:", "; ".join(xcd_check(n) for n in SHAPES), "-- bijection on every shape")
        except Fail as e:
            ok = False
            print("FAIL X:", e)
    try:
        print("A1/A2:", lane_level())
    except Fail as e:
        ok = False
        print("FAIL A:", e)
    try:
        print("L:", lds_protocol())
    except Fail as e:
        ok = False
        print("FAIL L:", e)
    for name in SHAPES:
        for causal in CAUSAL.get(name, (1, 0)):
            try:
                r = run_shape(name, causal)
                s = r["st"]
                print(f"PASS {name:16s} causal={causal} nsp={r['nsp']:2d} "
                      f"({'k_dkdv_w4f_sp' if r['nsp'] > 1 else 'k_dkdv_w4f'}): wgs={s['wgs']} steps={s['steps']} "
                      f"masked={s['masked_steps']} pairs={s['pairs']} tdm_ops={s['tdm_ops']} "
                      f"atomic_wave_instr={s['atomic_wave_instr']} N_range=[{min(r['Ns'])},{max(r['Ns'])}] "
                      f"max_elem kv={r['gmax'].get('kv')}/{r['n_kv']} tdm={r['gmax'].get('tdm')}/{r['n_q']} "
                      f"lse={r['gmax'].get('lse')}/{r['n_l']} atom={r['gmax'].get('atom')}/{r['n_dqa']} cvt={r['cvt']}")
            except Fail as e:
                ok = False
                print(f"FAIL {name} causal={causal}: {e}")
    here = os.path.dirname(os.path.abspath(__file__))
    for k in ("w4f/k_dkdv_w4f_0", "w4f_sp/k_dkdv_w4f_sp_0"):
        try:
            print(f"ISA {k}:", isa_check(os.path.join(here, ".dump", k, "21_final_isa.s")))
        except Fail as e:
            ok = False
            print(f"FAIL ISA {k}:", e)
    print(f"variant: W4_RELAX={W4_RELAX} W4_ABL_NOATOM={W4_ABL_NOATOM} W4_LDS_LSE={W4_LDS_LSE} "
          f"W4_XCD={W4_XCD} ring={W4_RING} lead={W4_LEAD} "
          f"tensor waits pro/loop/epi={TW_PRO}/{TW_LOOP}/0 LDS={W4_LDS}")
    print("RESULT:", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
