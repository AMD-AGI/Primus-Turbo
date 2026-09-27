"""CPU bounds/coverage proof for FUSED5 dq_acc atomics (mirrors _dkdv_impl control flow).

For every workgroup (kv tile bid, split sp) it replays qloop_mask + qloop_full, emits the
(q pair, q head) of every iteration that issues atomics, and checks:
  (1) every atomic index < B*Sq*Hq*D and >= 0 (int32-safe)
  (2) each (b, qh, q pair, kv tile) that is causally live is visited EXACTLY once
      (so dq_acc receives each dS@K partial once -> correct sum, no double count)
  (3) no causally dead pair (all 32x32 masked) is visited except the one straddle pair.
"""
import sys
BLOCK_KV = 32

def run(B, Sq, Skv, Hq, Hkv, nsp):
    G = Hq // Hkv; nqt = Sq // 16; nqt2 = nqt // 2; cshift = Skv - Sq
    size = B * Sq * Hq * 128
    assert size < 2**31
    seen = {}
    maxidx = 0
    for bid in range(Skv // BLOCK_KV):
        kv0 = bid * BLOCK_KV
        c = kv0 - cshift
        qp_start = max(c, 0) // 32
        nqp_eff = nqt2 - qp_start
        u = kv0 + BLOCK_KV - 1 - cshift
        qsf = 0 if u < 0 else (u + 31) // 32
        qsf = min(qsf, nqt2)
        nm = max(qsf - qp_start, 0); nm = min(nm, nqp_eff); nmaskp = nm
        for sp in range(nsp):
            its = []
            if nsp == 1:
                its += [(qp_start + ii // G, ii % G) for ii in range(G * nmaskp)]
                q0f = qp_start + nmaskp
                its += [(q0f + ii // G, ii % G) for ii in range(G * (nqp_eff - nmaskp))]
            else:
                fn = max(nqp_eff - nmaskp, 0); ch = (fn + nsp - 1) // nsp
                off = sp * ch; cnt = min(max(fn - off, 0), ch)
                mk = 0 if sp != 0 else nmaskp
                its += [(qp_start + ii // G, ii % G) for ii in range(G * mk)]
                qt0 = qp_start + nmaskp + off
                its += [(qt0 + ii // G, ii % G) for ii in range(G * cnt)]
            for qt, gh in its:
                assert 0 <= qt < nqt2, (bid, sp, qt)
                # causally live: some (q, kv) with kv <= q + cshift
                live = kv0 <= qt * 32 + 31 + cshift
                assert live, ("dead pair visited", bid, qt)
                for hkv in sorted({0, Hkv - 1}):
                    for bat in sorted({0, B - 1}):
                        qh = hkv * G + gh
                        key = (bat, qh, qt, bid)
                        seen[key] = seen.get(key, 0) + 1
                        for hh in (0, 1):
                            for half in (0, 1):
                                for si in (0, 7):
                                    for dtile in (0, 7):
                                        for row in (0, 15):
                                            q = qt * 32 + hh * 16 + half * 8 + si
                                            idx = ((bat * Sq + q) * Hq + qh) * 128 + dtile * 16 + row
                                            assert 0 <= idx < size
                                            maxidx = max(maxidx, idx)
    # coverage: every live (qt, bid) visited exactly once per (bat, qh) sampled
    for bat in sorted({0, B - 1}):
        for hkv in sorted({0, Hkv - 1}):
            for gh in range(G):
                qh = hkv * G + gh
                for bid in range(Skv // BLOCK_KV):
                    for qt in range(nqt2):
                        live = bid * BLOCK_KV <= qt * 32 + 31 + (Skv - Sq)
                        n = seen.get((bat, qh, qt, bid), 0)
                        assert n == (1 if live else 0), (bat, qh, qt, bid, n, live)
    ntiles = sum(v for v in seen.values())
    print(f"B{B} Sq{Sq} Skv{Skv} Hq{Hq} Hkv{Hkv} nsp{nsp}: OK  max_idx={maxidx} size={size} "
          f"(sampled heads/batches) visits={ntiles}")

for args in [(1, 1024, 1024, 8, 2, 16), (1, 1024, 1024, 8, 2, 1), (1, 4096, 4096, 32, 8, 2),
             (1, 4096, 4096, 32, 8, 1), (4, 8192, 8192, 32, 8, 1),
             (1, 512, 1024, 8, 2, 1), (1, 1024, 512, 8, 2, 1), (1, 1024, 512, 8, 2, 4)]:
    run(*args)
