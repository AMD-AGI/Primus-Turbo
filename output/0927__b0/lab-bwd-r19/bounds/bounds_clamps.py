#!/usr/bin/env python3
"""CPU-only bounds check of the h33 defect-c clamps as ported to r19h (k_dkdv / k_dkdv_sp).

Mirrors r19h/kernels.py _dkdv_impl index arithmetic in exact Python ints:
  nqt2, qp_start, nmaskp (:319-335, :530-540), PARTIAL chunking (:544-556),
  qloop_full prefetch jj = ii+1 (clamped to n-1 in r19h), prologue _ldqd(pair, 0)
  (clamped by _clampqt in r19h). Every prefetched query pair index is checked against
  [0, nqt2-1] (each pair = 32 rows; Q/dO rows qt*32 .. qt*32+31 < Sq).
Also: value-neutrality -- for every iteration that CONSUMES a prefetch (ii+1 < n; the
prologue when n > 0) the clamped index must equal the unclamped one.
Also defect a: Sq % BLOCK_Q (64) == 0 on every shape; defect b: Int64 dq extent vs Int32 wrap.
"""
import sys
sys.path.insert(0, sys.argv[1])  # ut/ dir for SHAPES
from common import SHAPES, causal_modes

BLOCK_KV, BLOCK_Q, D = 32, 64, 128
tot = dict(pref=0, oob19=0, oob19h=0, nonneutral=0, groups=0)
for name, (b, sq, skv, hq, hkv, d) in SHAPES.items():
    for causal in causal_modes(name):
        g = hq // hkv
        nqt = sq // 16
        nqt2 = nqt // 2
        cshift = skv - sq
        wgs = (skv // BLOCK_KV) * hkv * b
        nsp = 1
        while wgs * nsp < 2048 and nsp < 16:
            nsp *= 2
        PARTIAL = nsp > 1
        s = dict(pref=0, oob19=0, oob19h=0, nonneutral=0)
        clampqt = lambda t: max(0, min(t, nqt2 - 1))
        for bid in range(skv // BLOCK_KV):
            kv0 = bid * BLOCK_KV
            c = kv0 - cshift
            qp_start = (max(c, 0) // 32) if causal else 0
            nqp_eff = nqt2 - qp_start
            u = kv0 + BLOCK_KV - 1 - cshift
            qsf = 0 if u < 0 else (u + 31) // 32
            qsf = min(qsf, nqt2)
            nm = min(max(qsf - qp_start, 0), nqp_eff)
            nmaskp = nm if causal else 0
            for sp in range(nsp):
                if PARTIAL:
                    fn = max(nqp_eff - nmaskp, 0)
                    ch = (fn + nsp - 1) // nsp
                    off = sp * ch
                    cnt = min(max(fn - off, 0), ch)
                    qt0 = qp_start + nmaskp + off
                    n = g * cnt
                else:
                    qt0 = qp_start + nmaskp
                    n = g * (nqp_eff - nmaskp)
                # prologue
                issued = [(qt0, clampqt(qt0), n > 0)]
                for ii in range(n):
                    jj = ii + 1
                    jjh = jj if jj < n else n - 1
                    issued.append((qt0 + jj // g, qt0 + jjh // g, jj < n))
                for raw, clamped, consumed in issued:
                    s["pref"] += 1
                    if not (0 <= raw < nqt2):
                        s["oob19"] += 1
                    if not (0 <= clamped < nqt2):
                        s["oob19h"] += 1
                    if consumed and raw != clamped:
                        s["nonneutral"] += 1
        # defect b
        nsp_q = 1
        wq = ((sq + BLOCK_Q - 1) // BLOCK_Q) * hq * b
        while wq * nsp_q < 2048 and nsp_q < 8:
            nsp_q *= 2
        ext = nsp_q * b * sq * hq * D * 4
        wrap = ((ext + 2**31) % 2**32) - 2**31
        print(f"{name:17s} causal={int(causal)} nsp={nsp:2d} nsp_q={nsp_q} sq%64={sq % 64} "
              f"prefetches={s['pref']:7d} OOB_r19={s['oob19']:5d} OOB_r19h={s['oob19h']} "
              f"nonneutral={s['nonneutral']}  dq_ext={ext} int32={wrap}"
              f"{'' if nsp_q > 1 else ' (dq_sp not used)'}")
        for k in ("pref", "oob19", "oob19h", "nonneutral"):
            tot[k] += s[k]
print("TOTAL", tot)
ok = tot["oob19h"] == 0 and tot["nonneutral"] == 0
print("BOUNDS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
