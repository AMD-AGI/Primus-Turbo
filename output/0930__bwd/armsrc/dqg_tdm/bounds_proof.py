#!/usr/bin/env python3
"""CPU bounds proof for arm dqg_tdm (k_dqg K/V through a TDM LDS ring). Host python3 only.

What changed vs s4 (k_dqg only; k_dkdv / k_dq_sp / impl.py are byte-identical copies):
  * K/V tile of loop index i (kv0 = 32*i) reaches LDS by TDM (tensor_load_to_lds), not by
    buffer_load_b128 + ds_store_b128.  New index/address/predicate/ring expressions:
      T1 TDM origin  ((bat*Skv + kv0p)*Hkv + hkv)*D, row stride Hkv*D, 32 x 128 box,
         outer extent Skv - kv0p, kv0p = 32*min(i + DEPTH-1, N-1)  (prologue: 32*min(s, N-1))
      T2 ring stage offsets cur = (i%DEPTH)*KV_B, nxo = (i-1)%DEPTH, ncur = (i+1)%DEPTH, carried
         as a select-wrapped counter across BOTH loops (full then mask)
      T3 LDS addresses: readback  _lds0 + ncur + row*X_ROW_B + half*16 + imm
                        dQ tr16   _lds0 + cur  + lane_r*X_ROW_B + lane_c*2 + imm
      T4 loop split: full loop i in [0, nfull) UNMASKED, mask loop i in [nfull, N) masked.
         (s4 ran full blocks [0, 2*floor(nfull/2)) and the odd leftover in the MASK loop.)
      T5 tensorcnt: tensor_wait(2*(DEPTH-2)) before each readback, wait(0) at the end.
  * Bitwise vs s4: (B1) every readback chunk is the 16 B chunk s4's _ldkv_kt loaded from
    global for the same (kt, dt, K/V, u, lane); (B2) every dQ tr16 address, relative to the
    K image base, equals s4's lds_k tr16 address and the K image is the same bytes s4's
    ds_stores built; (B3) the S/dP/softmax/dQ arithmetic and the per-accumulator WMMA order
    over kv blocks are unchanged; the only predicate change is T4, proven identically false
    on the blocks it moves (so s4's masked select returned sv*scale there, as now).
Shapes: prod, fast, toy, gqa4_small, unequal_seqlen_2, each causal on/off, DEPTH 3 and 4.
(fast/toy/gqa4/unequal take k_dq_sp in impl.py, not k_dqg; they are proven anyway.)
"""
import sys

D = 128
KV_STEP = 32
NKT = 2
NDT = 4
NDO = 8
BQW = 64
X_ROW_B = D * 2 + 16
KV_B = 2 * KV_STEP * X_ROW_B
V_OFF = KV_STEP * X_ROW_B
NEG_MASK = True


class Fail(Exception):
    pass


def check(c, msg):
    if not c:
        raise Fail(msg)


# ---------------------------------------------------------------- lane-level maps
def tdm_image(r, byte):
    """TDM box row r, LDS byte offset within the row -> tensor column (element) or None (pad).
    pad_interval = D elements, pad_amount = 8 elements: row r starts at r*(D+8)*2 = r*X_ROW_B."""
    check(byte % 2 == 0, "odd byte")
    c = byte // 2
    return c if c < D else None


def s4_chunk(lane, kt, dt, kv, u):
    """s4 _ldkv_kt / gfrag2: lane (row, half) loads row kt*16+row, element cols
    half*8 + dt*32 + u*16 .. +8 of K (kv=0) or V (kv=1)."""
    row, half = lane % 16, lane // 16
    return (kv, kt * 16 + row, half * 8 + dt * 32 + u * 16)


def rb_chunk(lane, kt, dt, kv, u):
    """dqg_tdm _rdkv: address (relative to the stage) = lb_rd + imm."""
    row, half = lane % 16, lane // 16
    lb_rd = row * X_ROW_B + half * 16
    imm = kt * 16 * X_ROW_B + (V_OFF if kv else 0) + dt * 64 + u * 32
    check(0 <= imm < 65536, "readback imm")
    a = lb_rd + imm
    reg = 1 if a >= V_OFF else 0
    r, byte = divmod(a - reg * V_OFF, X_ROW_B)
    c = tdm_image(r, byte)
    check(c is not None and c + 8 <= D and tdm_image(r, byte + 14) == c + 7, "readback hits pad")
    check(0 <= a and a + 16 <= KV_B, "readback outside stage")
    return (reg, r, c), a


def tr_addr_new(lane, dtile, r2):
    lane_r = (lane // 16) * 8 + lane % 8
    lane_c = ((lane // 8) % 2) * 8
    lb_tr = lane_r * X_ROW_B + lane_c * 2
    imm = dtile * 32 + r2 * 16 * X_ROW_B
    check(0 <= imm < 65536, "tr16 imm")
    return lb_tr + imm


def tr_addr_s4(lane, dtile, r2):
    lane_r = (lane // 16) * 8 + lane % 8
    lane_c = ((lane // 8) % 2) * 8
    return lane_r * X_ROW_B + (lane_c + dtile * 16) * 2 + r2 * 16 * X_ROW_B


def s4_kstore(lane, kt, dt, u):
    """s4 ds_store of K chunk (kt, dt, u) into lds_k: ko + dt*64 + u*32 -> global chunk."""
    row, half = lane % 16, lane // 16
    a = (kt * 16 + row) * X_ROW_B + half * 16 + dt * 64 + u * 32
    return a, s4_chunk(lane, kt, dt, 0, u)


def lane_proofs():
    n = 0
    # B1: readback chunk == s4 global chunk
    for lane in range(32):
        for kt in range(NKT):
            for dt in range(NDT):
                for kv in range(2):
                    for u in range(2):
                        got, _ = rb_chunk(lane, kt, dt, kv, u)
                        check(got == s4_chunk(lane, kt, dt, kv, u), "B1 readback != s4 chunk")
                        n += 1
    # B2: the TDM K image and s4's ds_store K image agree on every byte s4 wrote, and every
    # tr16 read (s4 == new address) lands on a byte that image defines (not pad).
    s4_img = {}
    for lane in range(32):
        for kt in range(NKT):
            for dt in range(NDT):
                for u in range(2):
                    a, (kv, r, c) = s4_kstore(lane, kt, dt, u)
                    for e in range(8):
                        s4_img[a + 2 * e] = (r, c + e)
    for a, (r, c) in s4_img.items():
        rr, byte = divmod(a, X_ROW_B)
        check(rr == r and tdm_image(rr, byte) == c, "B2 K image differs")
    for lane in range(32):
        for dtile in range(NDO):
            for r2 in range(2):
                a = tr_addr_new(lane, dtile, r2)
                check(a == tr_addr_s4(lane, dtile, r2), "B2 tr16 address != s4")
                for e in range(8):
                    check(a + 2 * e in s4_img, "B2 tr16 reads a byte s4 never stored")
                check(0 <= a and a + 16 <= V_OFF, "tr16 outside K region")
                n += 1
    return n


# ---------------------------------------------------------------- per-WG replay
def run_shape(B, Sq, Skv, Hq, Hkv, causal, depth):
    G = Hq // Hkv
    nkvt = Skv // KV_STEP
    cshift = Skv - Sq
    stages = [s * KV_B for s in range(depth)]
    ALLOC = depth * KV_B
    check(ALLOC <= 81920, "LDS > 4-WG/CU budget (327680/4)")
    TW = 2 * (depth - 2)
    nkv_el = B * Skv * Hkv * D
    st = dict(wgs=0, iters=0, tdm=0, moved=0, maxcnt=0)
    # T1 for every (bat, hkv, tile): only kv0p and (bat, hkv) enter, so check all of them.
    for bat in range(B):
        for hkv in range(Hkv):
            for t in range(nkvt):
                kv0p = 32 * t
                check(Skv - kv0p >= 32, "T1 outer extent < box")
                org = ((bat * Skv + kv0p) * Hkv + hkv) * D
                last = org + 31 * Hkv * D + D - 1
                check(0 <= org and last < nkv_el, "T1 TDM global OOB")
                # rows of the box = kv rows kv0p..kv0p+31 of (bat, hkv), cols 0..127
                check(org + 5 * Hkv * D + 7 == ((bat * Skv + kv0p + 5) * Hkv + hkv) * D + 7, "T1 map")
    for bid_y in range(Sq // BQW):
        bid = Sq // BQW - 1 - bid_y
        q0 = bid * BQW
        lim = (q0 + BQW + cshift + KV_STEP - 1) // KV_STEP   # python // == floor; cshift>=0
        check(q0 + BQW + cshift + KV_STEP - 1 >= 0, "C-division sign")
        lim = max(lim, 1)
        lim = min(lim, nkvt)
        N = lim if causal else nkvt
        t_ = q0 + cshift + 1
        nf = 0 if t_ < 0 else t_ // KV_STEP
        nf = min(nf, N)
        nfull = nf if causal else N
        nlast = N - 1
        check(N >= 1 and 0 <= nfull <= N, "loop split")
        # T4: blocks [0, nfull) unmasked: the causal predicate is false for all elements.
        # s4 ran [0, 2*(nfull//2)) unmasked and the rest masked; the moved block (if any)
        # must also have an identically-false predicate.
        for i in range(nfull):
            kvmax = 32 * i + 31
            qmin = q0
            check(not (causal and kvmax > qmin + cshift), "T4 masked element in full loop")
        if nfull % 2:
            st["moved"] += 1
        # every element that the causal mask keeps is inside [0, N) blocks (N covers all)
        if causal:
            check(N == nkvt or 32 * N > q0 + BQW - 1 + cshift, "N misses an attended kv")
        # ring replay: TDM fifo (in-order), stage contents, pending DS reads
        fifo, pend, content = [], [], {}

        def tdm(tile, stage):
            check(0 <= tile <= nlast and tile < nkvt, "TDM tile range")
            check(stage in stages, "stage")
            check(all(p != stage for p in pend), "TDM into a stage with an unconsumed read")
            fifo.extend([(stage, tile), (stage, tile)])        # K op, V op
            st["maxcnt"] = max(st["maxcnt"], len(fifo))
            st["tdm"] += 1

        def wait(n):
            k = max(len(fifo) - n, 0)
            for stg, tile in fifo[:k]:
                content[stg] = tile
            del fifo[:k]

        def read(stage, tile, kind):
            check(all(s != stage for s, _ in fifo), f"{kind}: stage has TDM in flight")
            check(content.get(stage) == tile, f"{kind}: stage holds tile {content.get(stage)} != {tile}")
            for a in ([rb_chunk(l, kt, dt, kv, u)[1] for l in (0, 31) for kt in range(2)
                       for dt in (0, 3) for kv in (0, 1) for u in (0, 1)] if kind == "rb" else
                      [tr_addr_new(l, dtile, r2) for l in (0, 31) for dtile in (0, 7) for r2 in (0, 1)]):
                check(0 <= stage + a and stage + a + 16 <= ALLOC, "LDS OOB")
            pend.append(stage)

        # prologue
        tdm(0, 0)
        for s in range(1, depth - 1):
            tdm(min(s, nlast), s * KV_B)
        wait(TW)
        read(0, 0, "rb")
        carried = 0                     # tile whose A operands are in the carried VGPRs
        cur = 0
        for i in range(N):
            check(cur == (i % depth) * KV_B, "T2 cur")
            nxo = (depth - 1) * KV_B if cur == 0 else cur - KV_B
            ncur = 0 if cur == (depth - 1) * KV_B else cur + KV_B
            check(nxo == ((i + depth - 1) % depth) * KV_B and nxo == ((i - 1) % depth) * KV_B, "T2 nxo")
            check(ncur == ((i + 1) % depth) * KV_B, "T2 ncur")
            kk = min(i + depth - 1, nlast)
            tdm(kk, nxo)                # top of body
            # S/dP WMMAs consume the carried operands (readback of stage i%depth in i-1)
            check(carried == i, "S/dP operands are not block i")
            pend.remove(cur)            # the readback that fed them is consumed here
            read(cur, i, "tr")          # dQ B operands from this stage's K
            wait(TW)
            read(ncur, min(i + 1, nlast), "rb")
            carried = min(i + 1, nlast)
            pend.remove(cur)            # tr16 consumed by this body's dQ WMMAs
            cur = ncur
            st["iters"] += 1
        wait(0)
        check(not fifo, "T5 final wait leaves TDM in flight")
        check(len(pend) == 1, "exactly the last (never consumed) readback is pending")
        st["wgs"] += B * Hq
    return st


SHAPES = [
    ("prod", 4, 8192, 8192, 32, 8),
    ("fast", 1, 1024, 1024, 8, 2),
    ("toy", 1, 128, 128, 2, 1),
    ("gqa4_small", 2, 128, 128, 8, 2),
    ("unequal_seqlen_2", 2, 1024, 2048, 4, 1),
]


def main():
    print("lane proofs B1/B2:", lane_proofs(), "cases OK")
    ok = True
    for depth in (3, 4):
        for name, B, Sq, Skv, Hq, Hkv in SHAPES:
            assert Sq % BQW == 0 and Skv % KV_STEP == 0 and Hq % Hkv == 0
            for causal in (0, 1):
                try:
                    st = run_shape(B, Sq, Skv, Hq, Hkv, causal, depth)
                    print(f"DEPTH={depth} {name:17s} causal={causal} OK {st}")
                except Fail as e:
                    ok = False
                    print(f"DEPTH={depth} {name:17s} causal={causal} FAIL {e}")
    # immediates actually emitted (ISA) must equal the model's maxima
    rb_max = max(rb_chunk(l, kt, dt, kv, u)[1] - (l % 16) * X_ROW_B - (l // 16) * 16
                 for l in range(32) for kt in range(2) for dt in range(4) for kv in (0, 1) for u in (0, 1))
    tr_max = max(dtile * 32 + r2 * 16 * X_ROW_B for dtile in range(8) for r2 in (0, 1))
    print("max DS immediates: readback", rb_max, "tr16", tr_max)
    check(rb_max == 13280 and tr_max == 4576, "immediates")
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
