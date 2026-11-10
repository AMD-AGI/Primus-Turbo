"""Host replica of the round-18 XCD CTA map (attn_helper_mla.py): bijection check over every slot and
the per-launch section layout. Usage: python xcd_map_check.py"""

TAIL = 8


def xcd_map(slot, nq, hpx, groups):
    nq_head = nq - TAIL if nq > TAIL else 0
    lo = 0
    h = q = None
    for i, g in enumerate(groups):
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


def sections(nq, groups):
    nq_head = nq - TAIL if nq > TAIL else 0
    lo, out = 0, []
    for g in groups:
        hi = lo + (nq_head - lo) // g * g
        if hi > lo:
            out.append(f"G{g}x{(hi - lo) // g}")
        lo = hi
    out.append(f"tail{nq - lo}")
    return " ".join(out)


for groups in ((32, 16, 8), (16, 8), (16,)):
    for hpx in (1, 2, 8, 16):
        for nq in range(1, 401):
            seen = {xcd_map(s, nq, hpx, groups) for s in range(nq * hpx)}
            want = {(h, r) for h in range(hpx) for r in range(nq)}
            assert seen == want, (groups, hpx, nq)
    print(f"{groups}: bijective for nq 1..400, heads/XCD 1, 2, 8, 16")

print("sections per launch (BLOCK_M 256):")
for s in (4096, 8192, 16384, 32768, 65536):
    nq = s // 256
    print(f"  S{s // 1024}K nq {nq}: (32,16,8) {sections(nq, (32, 16, 8))} | (16,8) {sections(nq, (16, 8))}")
