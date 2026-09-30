#!/usr/bin/env python3
"""CPU bounds proof for arm mem_soffset_q (k_dqg K/V loop loads). No torch, no GPU.

r29 (gfrag2):  tile t = base_kv + (kv0 + kt*16 + row)*rs_kv + half + dt*4 + 2u
               byte  = 16*t, one i32 voffset, soffset = 0, imm = 0
arm:           voffset = 16*(base_kv + (kt*16 + row)*rs_kv + half)   (i32, loop-invariant)
                         + (dt*64 + u*32)                             (add nsw nuw -> imm)
               soffset = readfirstlane(kv0 * (rs_kv*16))              (i32, wave-uniform)
               byte    = voffset + soffset

For every shape x causal x launched (bat, q head) x q tile (bid) x every kv0 that the
prologue / kvloop_full (U2, clamped jj) / kvloop_mask actually load x lane x kt x dt x u,
this checks:
  1. arm byte == r29 byte                      (same bytes -> same registers -> bitwise)
  2. 0 <= byte and byte + 16 <= numel(K)*2     (the 16-byte load stays inside K and V;
                                                 K and V have the same shape)
  3. every i32 intermediate of both formulas lies in [0, 2^31)   (no signed wrap; the
     `add nsw nuw` flag is truthful; hardware sum voffset+soffset+imm < 2^31)
  4. voffset+imm < 2^30 and voffset+imm+soffset < 2^30 (the fake 1<<30 num_records never
     clamps under either range-check convention).
kv0 depends on the q tile only through the trip lists, and the address depends on kv0,
not on bid, so per (shape, causal, bat, hkv) we check the union of all per-bid kv0 lists
(each list is also asserted to be a subset of [0, Skv) in steps of 32).
"""
import sys

D, DV8, KV_STEP, NKT, NDT, BQW, NW, WAVE = 128, 16, 32, 2, 4, 64, 1, 32
I32 = 1 << 31
NREC = 1 << 30

SHAPES = {  # name: (b, s, hq, hkv)
    "prod": (4, 8192, 32, 8),
    "fast": (1, 1024, 8, 2),
    "toy": (1, 128, 2, 1),
}


def i32(x, what):
    assert 0 <= x < I32, (what, x)
    return x


def kv0_lists(sq, skv, causal):
    """kv0 values loaded by one workgroup per q tile, replaying _dqg_impl's control flow."""
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
        loads = [0]                                   # prologue _ldkv(0)
        for it in range(npair):                       # kvloop_full, U2
            ii = it * 2
            jj = ii + 2
            jj = jj if jj < npair * 2 else npair * 2 - 1
            loads.append((ii + 1) * KV_STEP)          # body 1 prefetch
            loads.append(jj * KV_STEP)                # body 2 prefetch (clamped)
        nfull = npair * 2
        for it in range(nkvt_eff - nfull):            # kvloop_mask, in-iteration load
            loads.append((it + nfull) * KV_STEP)
        for kv0 in loads:
            assert 0 <= kv0 <= skv - KV_STEP and kv0 % KV_STEP == 0, (bid, kv0)
        out[bid] = loads
    return out


def launched_bh(b, hq, hkv):
    g = hq // hkv
    ngrp = hq // NW
    pairs = set()
    for gx in range(ngrp):
        grp = (gx % 8) * (ngrp // 8) + gx // 8 if ngrp % 8 == 0 else gx
        assert 0 <= grp < ngrp
        for wave in range(NW):
            qh = grp * NW + wave
            for bat in range(b):
                pairs.add((bat, qh // g))
    return sorted(pairs)


def check(name, b, s, hq, hkv):
    sq = skv = s
    rs_kv = hkv * DV8
    kbytes = b * skv * hkv * D * 2
    i32(kbytes, "numel(K)*2")
    rs_kv_b = i32(rs_kv * 16, "rs_kv*16")
    n = 0
    lo, hi = 1 << 62, -1
    for causal in (0, 1):
        lists = kv0_lists(sq, skv, causal)
        kv0s = sorted({k for v in lists.values() for k in v})
        trips = sum(len(v) for v in lists.values())
        for bat, hk in launched_bh(b, hq, hkv):
            base_kv = i32(i32(bat * skv, "bat*Skv") * rs_kv + hk * DV8, "base_kv")
            for lane in range(WAVE):
                row, half = lane % 16, lane // 16
                for kt in range(NKT):
                    rr = i32((kt * 16 + row) * rs_kv, "(kt*16+row)*rs_kv")
                    vt = i32(base_kv + rr + half, "voff tile")
                    voff = i32(vt * 16, "voff bytes")
                    for dt in range(NDT):
                        for u in range(2):
                            c = dt * 64 + u * 32
                            vo = i32(voff + c, "voff+imm")        # add nsw nuw is truthful
                            assert vo < NREC, ("voff+imm >= num_records", vo)
                            for kv0 in kv0s:
                                soff = i32(kv0 * rs_kv_b, "soffset")
                                new = i32(vo + soff, "hw byte")
                                assert new < NREC
                                # r29, exactly as gfrag2 + BufferCopy form it
                                t = i32(base_kv + i32(i32(kv0 + kt * 16 + row, "r") * rs_kv,
                                                      "r*rs") + half + dt * 4 + 2 * u, "t")
                                old = i32(t * 16, "r29 byte")
                                assert new == old, (name, causal, bat, hk, lane, kt, dt, u,
                                                    kv0, new, old)
                                assert 0 <= new and new + 16 <= kbytes, (name, new, kbytes)
                                lo, hi = min(lo, new), max(hi, new + 16)
                                n += 1
        print(f"  {name:4s} causal={causal}: q tiles={len(lists)} kv0 loads/tile total={trips}"
              f" distinct kv0={len(kv0s)} max kv0={kv0s[-1]}")
    print(f"  {name:4s} OK: {n} (bat,hkv,lane,kt,dt,u,kv0) addresses, byte range [{lo}, {hi})"
          f" within K/V of {kbytes} B; max soffset {(skv - KV_STEP) * rs_kv_b}")
    return n


def main():
    tot = 0
    for name, (b, s, hq, hkv) in SHAPES.items():
        tot += check(name, b, s, hq, hkv)
    print(f"ALL OK ({tot} addresses): arm byte == r29 byte, in bounds, no i32 wrap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
