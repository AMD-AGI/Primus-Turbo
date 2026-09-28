"""Offline CPU gate for L4 (XCD-major (b, kv_head) remap on top of longest-first).

Mirrors integer-for-integer `_lpt_block_id` + `_xcd_group` in <arm>/flydsl_fwd/
fmha_fwd_prefill_a16w16_m32x8.py (int32, all operands >= 0 so C-style // and % equal
Python's). For every grid it enumerates every hardware block id (bx, by, bz) and asserts:
  1. the remapped (x, y, z) is in range and hit exactly once (bijection) -> every
     (batch, kv_head, q-tile) is computed by exactly one WG; one writer per o/lse row;
  2. x (hence the WG's causal KV-tile work) equals the champion's for every lin, so the
     dispatch order of work -- the longest-first balance -- is unchanged by construction;
  3. int32 headroom: every intermediate < 2^31.
Also reports, under the lin % 8 XCD model, how many distinct (b, kvh) groups each XCD
touches (the L2-locality quantity L4 changes) and the 256-CU list-scheduled makespan.
Also re-parses the arm source to assert the mode constant matches.
"""
import itertools, re, sys, pathlib

BLOCK_M, N_BLOCK, NUM_XCD, NUM_CU = 256, 64, 8, 256
I32 = 2**31


def cdiv(a, b):
    return -(-a // b)


def xcd_group(mode, rem, rank, gyz):
    if mode == "off":
        return rem
    if mode == "spread":
        return (rem + rank) % gyz
    q = gyz // NUM_XCD
    ok = 1 - min(gyz % NUM_XCD, 1)
    mapped = (rem % NUM_XCD) * q + rem // NUM_XCD
    return rem + ok * (mapped - rem)


def block(mode, gx, gy, gz, bx, by, bz):
    lin = bx + gx * (by + gy * bz)
    gyz = gy * gz
    rank = lin // gyz
    rem = lin - rank * gyz
    grp = xcd_group(mode, rem, rank, gyz)
    for v in (lin, gyz, rem + rank, (rem % NUM_XCD) * max(gyz // NUM_XCD, 1) + rem // NUM_XCD):
        assert 0 <= v < I32
    return lin, gx - 1 - rank, grp % gy, grp // gy


def kv_tiles(x, g, sq, skv, causal):
    if not causal:
        return cdiv(skv, N_BLOCK)
    wg_max_seq = min((x * BLOCK_M + BLOCK_M - 1) // g, sq - 1)
    return cdiv(max(min(wg_max_seq + (skv - sq) + 1, skv), 1), N_BLOCK)


def makespan(work):  # in-order dispatch to the first free CU (1 WG/CU: LDS-full)
    import heapq
    cus = [0] * NUM_CU
    heapq.heapify(cus)
    for w in work:
        t = heapq.heappop(cus)
        heapq.heappush(cus, t + w + 2)
    return max(cus)


SHAPES = {  # b, sq, skv, hq, hkv  (job ut/common.py + gqa=1 + odd grids)
    "prod": (4, 8192, 8192, 32, 8), "proxy": (1, 4096, 4096, 32, 8), "fast": (1, 1024, 1024, 8, 2),
    "prod_g1": (4, 8192, 8192, 32, 32), "proxy_g1": (1, 4096, 4096, 32, 32), "fast_g1": (1, 1024, 1024, 8, 8),
    "toy": (1, 256, 256, 2, 1), "short_q": (1, 128, 512, 4, 1), "gqa4_batch2": (2, 256, 256, 8, 2),
    "mha": (1, 256, 256, 4, 4), "unequal_seqlen": (1, 512, 1024, 8, 2), "unequal2": (2, 1024, 2048, 4, 1),
    "sq_gt_skv": (1, 1024, 512, 8, 2), "odd129": (4, 8256, 8256, 32, 8), "odd_tail": (2, 1300, 1300, 32, 8),
    "gyz12": (3, 2048, 2048, 16, 4), "gyz24": (3, 4096, 4096, 64, 8), "gyz40": (5, 1024, 1024, 64, 8),
    "b3_mha": (3, 768, 768, 16, 16), "gyz6": (2, 512, 512, 12, 3),
}


def check(mode, name, b, sq, skv, hq, hkv, causal):
    g = hq // hkv
    gx, gy, gz = cdiv(sq * g, BLOCK_M), hkv, b
    seen, work, work_ref, xcd_groups = set(), [], [], [set() for _ in range(NUM_XCD)]
    for bz, by, bx in itertools.product(range(gz), range(gy), range(gx)):
        lin, x, y, z = block(mode, gx, gy, gz, bx, by, bz)
        _, xr, _, _ = block("off", gx, gy, gz, bx, by, bz)
        assert 0 <= x < gx and 0 <= y < gy and 0 <= z < gz, (name, x, y, z)
        assert (x, y, z) not in seen, (name, "dup", x, y, z)
        assert x == xr
        seen.add((x, y, z))
        work.append((lin, kv_tiles(x, g, sq, skv, causal)))
        work_ref.append((lin, kv_tiles(xr, g, sq, skv, causal)))
        xcd_groups[lin % NUM_XCD].add((y, z))
    assert len(seen) == gx * gy * gz
    # rows: every (batch, seq, q_head) produced once (packed row r -> seq r//g, head kvh*g + r%g)
    rows = set()
    for (x, y, z) in seen:
        for r in range(x * BLOCK_M, min((x + 1) * BLOCK_M, sq * g)):
            rows.add((z, r // g, y * g + r % g))
    assert len(rows) == b * sq * hq
    w = [t for _, t in sorted(work)]
    wr = [t for _, t in sorted(work_ref)]
    assert w == wr
    per = [len(s) for s in xcd_groups]
    return gx * gy * gz, gy * gz, per, makespan(w)


def main():
    here = pathlib.Path(__file__).resolve().parent.parent
    for arm, mode in (("ctrl", "off"), ("bmajor", "bmajor"), ("spread", "spread")):
        src = (here / arm / "flydsl_fwd" / "fmha_fwd_prefill_a16w16_m32x8.py").read_text()
        got = re.search(r'^XCD_REMAP = "(\w+)"', src, re.M).group(1)
        assert got == mode, (arm, got)
    print(f"## XCD model: xcd = lin % {NUM_XCD}; per-XCD = distinct (b,kvh) groups on each XCD")
    fails = 0
    for name, (b, sq, skv, hq, hkv) in SHAPES.items():
        for causal in (True, False):
            if causal and sq > skv:
                continue
            row = []
            for mode in ("off", "bmajor", "spread"):
                try:
                    n, gyz, per, ms = check(mode, name, b, sq, skv, hq, hkv, causal)
                    row.append(f"{mode}: groups/XCD max {max(per):3d} span {ms}")
                except AssertionError as e:
                    fails += 1
                    row.append(f"{mode}: FAIL {e}")
            print(f"{name:14s} {'causal' if causal else 'nc    '} WGs={n:5d} gyz={gyz:3d} | " + " | ".join(row))
    # exhaustive small-grid sweep
    cnt = 0
    for gx in range(1, 20):
        for gy in range(1, 18):
            for gz in range(1, 7):
                for mode in ("bmajor", "spread"):
                    seen = {block(mode, gx, gy, gz, bx, by, bz)[1:]
                            for bz in range(gz) for by in range(gy) for bx in range(gx)}
                    ok = len(seen) == gx * gy * gz and all(
                        0 <= x < gx and 0 <= y < gy and 0 <= z < gz for x, y, z in seen)
                    fails += not ok
                    cnt += 1
    print(f"exhaustive sweep gx<20 gy<18 gz<7: {cnt} grid x mode bijections, fails={fails}")
    print("ALL OK" if fails == 0 else f"FAILURES {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
