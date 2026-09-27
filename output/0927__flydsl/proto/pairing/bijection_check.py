"""Offline (CPU, pure python) coverage + balance check for L3 in-WG q-tile pairing.

Mirrors, integer for integer, the prototype's launcher grid (``_ensure_bshd_kernel._launch``)
and kernel mapping (``_lpt_rank`` + the pass loop in ``kn_..._bshd``) in
op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py. Checks, per shape:
  1. every (batch, kv_head, q-tile) is run exactly once, every x in [0, n_qt), lo <= hi;
  2. every (batch, seq, q_head) row is produced by exactly one (WG, pass) -- determinism: one
     writer per o/lse element, no atomics, no split-k;
  3. per-WG causal KV-tile work (kernel's own n_tiles formula) and a 256-CU in-order
     dispatch simulation, paired vs the round-2 champion (longest-first, unpaired).
Exit code != 0 on any coverage failure.
"""
import sys

BLOCK_M, N_BLOCK, NUM_CU, PAIR_MIN_WGS = 256, 64, 256, 256


def ceildiv(a, b):
    return -(-a // b)


POLICY = "exact"


def grid(b, sq, g, hkv, paired_build=True):
    n_qt = ceildiv(sq * g, BLOCK_M)
    hb = hkv * b
    p_wgs = (n_qt // 2) * hb
    pair_exact = (1 - n_qt % 2) * (1 - min(p_wgs % PAIR_MIN_WGS, 1)) * min(p_wgs // PAIR_MIN_WGS, 1)
    pair_over = min((n_qt * hb) // (PAIR_MIN_WGS + 1), 1)
    pair = (pair_exact if POLICY == "exact" else pair_over) * int(paired_build)
    return n_qt, (n_qt - pair * (n_qt // 2), hkv, b)


def wg_tiles_paired(gx, gy, gz, n_qt, bx, by, bz):
    lin = bx + gx * (by + gy * bz)
    rank = lin // (gy * gz)
    rem = lin - rank * gy * gz
    y, z = rem % gy, rem // gy
    x_hi = n_qt - 1 - rank
    x_lo = rank if gx < n_qt else x_hi
    assert x_lo <= x_hi, (x_lo, x_hi)
    step = max(x_hi - x_lo, 1)
    return z, y, list(range(x_lo, x_hi + 1, step)), lin


def wg_tiles_lpt(gx, gy, gz, n_qt, bx, by, bz):  # round-2 champion (_lpt_block_id)
    lin = bx + gx * (by + gy * bz)
    rank = lin // (gy * gz)
    rem = lin - rank * gy * gz
    return rem // gy, rem % gy, [gx - 1 - rank], lin


def kv_tiles(x, g, sq, skv):  # _core_attention mask_right branch, window_right = 0
    wg_max_seq = min((x * BLOCK_M + BLOCK_M - 1) // g, sq - 1)
    kv_len_wg = max(min(wg_max_seq + (skv - sq) + 1, skv), 1)
    return ceildiv(kv_len_wg, N_BLOCK)


def check(name, b, sq, skv, hq, hkv, fixed=2.0):
    g = hq // hkv
    n_qt, (gx, gy, gz) = grid(b, sq, g, hkv)
    paired = gx < n_qt
    seen_tile, seen_row, wgs = {}, {}, []
    for bz in range(gz):
        for by in range(gy):
            for bx in range(gx):
                z, y, xs, lin = wg_tiles_paired(gx, gy, gz, n_qt, bx, by, bz)
                assert paired or len(xs) == 1
                for x in xs:
                    assert 0 <= x < n_qt, x
                    k = (z, y, x)
                    seen_tile[k] = seen_tile.get(k, 0) + 1
                    for r in range(x * BLOCK_M, x * BLOCK_M + BLOCK_M):
                        s, h = r // g, y * g + r % g
                        if s < sq:
                            seen_row[(z, s, h)] = seen_row.get((z, s, h), 0) + 1
                wgs.append((lin, sum(kv_tiles(x, g, sq, skv) for x in xs) + fixed * len(xs)))
    want = {(z, y, x) for z in range(b) for y in range(hkv) for x in range(n_qt)}
    ok_tiles = set(seen_tile) == want and all(v == 1 for v in seen_tile.values())
    ok_rows = len(seen_row) == b * sq * hq and all(v == 1 for v in seen_row.values())
    # champion (unpaired LPT) for the same shape
    ch = []
    for bz in range(b):
        for by in range(hkv):
            for bx in range(n_qt):
                z, y, xs, lin = wg_tiles_lpt(n_qt, hkv, b, n_qt, bx, by, bz)
                ch.append((lin, kv_tiles(xs[0], g, sq, skv) + fixed))

    def makespan(jobs):  # in-order dispatch to the first free CU (1 WG / CU)
        cus = [0.0] * NUM_CU
        for _, c in sorted(jobs):
            i = min(range(NUM_CU), key=cus.__getitem__)
            cus[i] += c
        return max(cus), sum(c for _, c in jobs) / NUM_CU

    mp, lb = makespan(wgs)
    mc, lbc = makespan(ch)
    w = [c for _, c in wgs]
    print(f"{name:16s} b{b} sq{sq} skv{skv} hq{hq} hkv{hkv}: n_qt={n_qt:3d} ({'odd' if n_qt % 2 else 'even'}) "
          f"grid=({gx},{gy},{gz}) {'PAIRED' if paired else 'unpaired'} WGs={len(wgs):5d} "
          f"tiles={'OK' if ok_tiles else 'FAIL'} rows={'OK' if ok_rows else 'FAIL'} | "
          f"WG work min/max {min(w):.0f}/{max(w):.0f} | makespan paired {mp:.0f} vs champion {mc:.0f} "
          f"(lower bound {lb:.0f}/{lbc:.0f}) -> {mc / mp:.3f}x")
    return ok_tiles and ok_rows


SHAPES = [
    ("prod", 4, 8192, 8192, 32, 8),
    ("proxy", 1, 4096, 4096, 32, 8),
    ("fast", 1, 1024, 1024, 8, 2),
    ("toy", 1, 256, 256, 2, 1),
    ("short_q", 1, 128, 512, 4, 1),
    ("gqa4_batch2", 2, 256, 256, 8, 2),
    ("mha", 1, 256, 256, 4, 4),
    ("unequal_seqlen", 1, 512, 1024, 8, 2),
    ("unequal_seqlen2", 2, 1024, 2048, 4, 1),
    # odd tile counts / paired with a middle solo rank
    ("odd129", 4, 8256, 8256, 32, 8),     # n_qt = 129
    ("odd21", 2, 1344, 1344, 32, 8),      # n_qt = 21, 336 WGs unpaired -> paired
    ("odd33_b3", 3, 2112, 2112, 32, 8),   # n_qt = 33
    ("odd_tailrow", 2, 1300, 1300, 32, 8),  # n_qt = 21, last tile partly past sq
    ("odd_sq_lt_skv", 2, 1344, 2048, 32, 8),
    ("mha_odd", 8, 768, 768, 16, 16),     # g=1, n_qt = 3, 384 WGs -> paired
    ("just_over", 1, 2112, 2112, 8, 2),   # n_qt = 33, 66 WGs (stays unpaired)
    ("even_partial", 3, 4096, 4096, 32, 8),  # n_qt = 64, 768 paired WGs = 3 rounds
    ("even_frac", 3, 2048, 2048, 32, 8),  # n_qt = 32, 384 paired WGs = 1.5 rounds
]

if __name__ == "__main__":
    fixed = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    ok = True
    for POLICY in (sys.argv[2:] or ["exact", "overfill"]):
        print(f"\n## launcher policy {POLICY!r}, per-pass fixed cost (prologue+epilogue) = {fixed} KV-tile units")
        ok &= all(check(*s, fixed=fixed) for s in SHAPES)
    print("ALL OK" if ok else "COVERAGE FAILURE")
    sys.exit(0 if ok else 1)
