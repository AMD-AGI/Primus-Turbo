"""Round-19 host replica: same (h, q) as round-18 hierarchical map for groups (32, 16, 8)."""

TAIL = 8
GROUPS = (32, 16, 8)


def map_r18(slot, nq, hpx):
    nq_head = nq - TAIL if nq > TAIL else 0
    lo = 0
    h = q = None
    for i, g in enumerate(GROUPS):
        u = slot - lo * hpx
        grp = u // (g * hpx)
        r = u - grp * (g * hpx)
        hs, qs = r // g, lo + grp * g + r % g
        if i == 0 or slot >= lo * hpx:
            h, q = hs, qs
        lo = lo + (nq_head - lo) // g * g
    if slot >= lo * hpx:
        t = slot - lo * hpx
        h, q = t % hpx, lo + t // hpx
    return h, q


def map_r19(slot, nq, hpx):
    nq_head = nq - TAIL if nq > TAIL else 0
    g0, g1, g2 = GROUPS
    lo0 = 0
    lo1 = (nq_head // g0) * g0
    lo2 = lo1 + (nq_head - lo1) // g1 * g1
    lo3 = lo2 + (nq_head - lo2) // g2 * g2

    def sec(lo, g):
        u = slot - lo * hpx
        grp = u // (g * hpx)
        r = u - grp * (g * hpx)
        return r // g, lo + grp * g + r % g

    h0, q0 = sec(lo0, g0)
    h1, q1 = sec(lo1, g1)
    h2, q2 = sec(lo2, g2)
    t = slot - lo3 * hpx
    ht, qt = t % hpx, lo3 + t // hpx
    if slot >= lo3 * hpx:
        return ht, qt
    if slot >= lo2 * hpx:
        return h2, q2
    if slot >= lo1 * hpx:
        return h1, q1
    return h0, q0


mism = 0
for hpx in (1, 2, 8, 16):
    for nq in range(1, 401):
        for s in range(nq * hpx):
            a, b = map_r18(s, nq, hpx), map_r19(s, nq, hpx)
            if a != b:
                mism += 1
                if mism <= 5:
                    print("mismatch", hpx, nq, s, a, b)
        seen = {map_r19(s, nq, hpx) for s in range(nq * hpx)}
        want = {(h, r) for h in range(hpx) for r in range(nq)}
        assert seen == want, (hpx, nq)
print(f"identical to r18 for nq 1..400, hpx 1/2/8/16; mismatches={mism}")
