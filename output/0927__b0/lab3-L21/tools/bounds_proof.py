"""CPU proof that the L21 peeled loop partition visits exactly the champion's tiles, in order,
each tile under masks at least as general as the champion's (so every K/V address, LDS
parity and prefetch predicate is the champion's). Mirrors fmha_fwd_prefill_a16w16_m32x8.py
_core_attention tile-range math (champion lines ~832-870 and ~1245-1300). No GPU.
"""
import itertools

BLOCK_M, NB = 256, 64


def ceildiv(a, b):
    return -(-a // b)


def champ_ranges(q_len, kv_len, gqa, bx, mask_left, mask_right, wl, wr):
    causal_off = kv_len - q_len
    if mask_right:
        wg_max_seq = min((bx * BLOCK_M + BLOCK_M - 1) // gqa, q_len - 1)
        kv_len_wg = max(min(wg_max_seq + causal_off + wr + 1, kv_len), 1)
    else:
        kv_len_wg = kv_len
    n_tiles = ceildiv(kv_len_wg, NB)
    if mask_left:
        wg_min_seq = (bx * BLOCK_M) // gqa
        kv_lo = max(wg_min_seq + causal_off - wl, 0)
        start = min(kv_lo // NB, n_tiles - 1)
    else:
        start = 0
    n_last = n_tiles - 1
    if mask_right:
        wg_min_seq = (bx * BLOCK_M) // gqa
        qmax_min = max(wg_min_seq + causal_off + wr, 0)
        clean_hi = (qmax_min + 1) // NB
    else:
        clean_hi = n_tiles
    clean_hi = max(min(clean_hi, n_last), start)
    if mask_left:
        wg_max_seq = min((bx * BLOCK_M + BLOCK_M - 1) // gqa, q_len - 1)
        qmin_max = max(wg_max_seq + causal_off - wl, 0)
        clean_lo = (qmin_max + NB - 1) // NB
    else:
        clean_lo = start
    clean_lo = min(max(clean_lo, start), clean_hi)
    return start, clean_lo, clean_hi, n_tiles


def visits(ranges):
    out = []
    for lo, hi, tag in ranges:
        out += [(t, tag) for t in range(lo, hi)]   # scf.for lo>=hi -> zero trips
    return out


# mask "generality": which masks a loop applies. A tile may move to a loop that applies a
# SUPERSET of masks (masks are exact per element, so an extra mask is a no-op), never fewer.
def check(q_len, kv_len, gqa, bx, ml, mr, wl, wr):
    s, clo, chi, nt = champ_ranges(q_len, kv_len, gqa, bx, ml, mr, wl, wr)
    L, C, Rt = {"l": ml, "r": mr, "kv": False}, {"l": False, "r": False, "kv": False}, {"l": ml, "r": mr, "kv": True}
    FULL = {"l": ml, "r": mr, "kv": True}
    champ = visits(([(s, clo, "L")] if ml else []) + [(clo, chi, "C"), (chi, nt, "R")])
    cmask = {"L": L, "C": C, "R": Rt, "P": FULL}
    for peel in (False, True):
        s_lo = s + 1 if peel else s
        c_lo, c_hi = max(clo, s_lo), max(chi, s_lo)
        new = visits(([(s, s + 1, "P")] if peel else []) + ([(s_lo, c_lo, "L")] if ml else [])
                     + [(c_lo, c_hi, "C"), (c_hi, nt, "R")])
        assert [t for t, _ in new] == [t for t, _ in champ], (q_len, kv_len, gqa, bx, ml, mr, wl, wr, peel, champ, new)
        assert nt >= s + 1, "peeled tile must exist"
        for (t, tn), (_, tc) in zip(new, champ):
            a, b = cmask[tn], cmask[tc]
            assert all(a[k] or not b[k] for k in a), ("fewer masks", t, tn, tc)
    return nt - s


n = 0
for q_len, kv_len in itertools.product([1, 63, 64, 65, 128, 255, 256, 257, 511, 1024, 2048, 8192],
                                       [1, 63, 64, 65, 128, 255, 256, 511, 1024, 2048, 8192]):
    for gqa in (1, 2, 4, 8):
        n_bx = ceildiv(q_len * gqa, BLOCK_M)
        for bx in range(n_bx):
            for ml, mr in ((False, True), (False, False), (True, True), (True, False)):
                if mr and not ml and q_len > kv_len:
                    pass  # empty rows are allowed; bounds math must still hold
                for wl, wr in ((0, 0), (37, 0), (128, 0), (1000, 0), (0, 64)) if ml or mr else ((0, 0),):
                    check(q_len, kv_len, gqa, bx, ml, mr, wl, wr)
                    n += 1
print(f"BOUNDS_OK {n} (shape, gqa, block_x, mask, window) configs: peeled partition == champion tile "
      f"sequence, every tile under a superset of the champion's masks, n_tiles >= start_tile+1")
