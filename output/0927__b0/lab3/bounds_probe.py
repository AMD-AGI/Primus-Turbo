"""CPU bounds proof for atomprobe/k_atom.py itself (its walk differs from _dkdv_impl: no mask/full
split, grid.y = Skv/(32*KVG)). Replays EVERY (bat, hkv, bid, it) at prod with numpy; per iteration the
index is monotone in (hh, half, si, dtile, row), so min/max lane indices are the all-0 / all-max lanes.
Also checks the host lane_ops count equals the kernel's trip count, and int32 safety of every
intermediate (base, idx, the store-mode num_records_bytes product)."""
import numpy as np
D = 128
def run(b, sq, hq, hkv, kvg):
    G = hq // hkv; nqt2 = sq // 32; cshift = 0; rs = hq * D; size = b * sq * hq * D
    assert b * sq * rs * 4 < 2**31, "num_records_bytes int32 overflow"
    tot = 0; mx = -1; mn = 1 << 62
    for bid in range(sq // (32 * kvg)):
        kv0 = bid * 32 * kvg; c = kv0 - cshift
        qps = max(c, 0) // 32; n = G * (nqt2 - qps)
        assert n >= 0
        ii = np.arange(n, dtype=np.int64); qi = ii // G; gh = ii - qi * G
        for bat in range(b):
            for h in range(hkv):
                qh = h * G + gh
                q0 = (qps + qi) * 32
                lo = (bat * sq + q0 + 0) * rs + qh * D + 0
                hi = (bat * sq + q0 + 16 + 8) * rs + qh * D + 15 + 7 * rs + 7 * 16
                mn = min(mn, int(lo.min()) if n else mn); mx = max(mx, int(hi.max()) if n else mx)
        tot += n * hkv * b
    host = sum(G * (sq // 32 - (bid * 32 * kvg) // 32) for bid in range(sq // (32 * kvg))) * hkv * b
    assert tot == host, (tot, host)
    assert 0 <= mn and mx < size and mx < 2**31, (mn, mx, size)
    print(f"KVG={kvg}: iters={tot} lane_ops={tot*4096} payload_GB={tot*4096*4/1e9:.2f} min_idx={mn} max_idx={mx} size={size} OK")
for k in (1, 2, 4):
    run(4, 8192, 32, 8, k)
