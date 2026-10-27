"""aid_model.py: check the AID half-split remaps are bijections and count, per AID, the distinct (expert, B block)
and A row blocks its tiles read, for the Mega GEMM tile orders (dispatch_gemm: 16-row super-tiles after the comm
workgroups; gemm_combine: row major with the lead combine workgroups in between), XCC = (bid + 7) % 8."""
import random

XCC0, NX = 7, 8


def aid_half(local, group, bid):
    xcc = (bid + XCC0) % NX
    whole = group - group % NX
    if local >= whole:
        return local
    return (xcc // 4) * (whole // 2) + (local // NX) * 4 + xcc % 4


def dispatch_gemm_tiles(real_tiles, nbn, group_m, ncomm, split, num_experts=32):
    """split: 0 none, 1 N halves, 2 m halves (contiguous), 3 N halves if A blocks < B blocks else none,
    4 N halves if A < B else m halves."""
    out = {}
    for pid in range(real_tiles * nbn):
        bid = ncomm + pid
        npg = group_m * nbn
        first_m = pid // npg * group_m
        size_m = max(1, min(group_m, real_tiles - first_m))
        l = pid % npg
        g = size_m * nbn
        a_small = real_tiles < num_experts * nbn
        mode = {0: 0, 1: 1, 2: 2, 3: 1 if a_small else 0, 4: 1 if a_small else 2}[split]
        if mode == 2 and size_m % 2:
            mode = 1
        if mode:
            l = aid_half(l, g, bid)
        if mode == 2:
            hm, hg = size_m // 2, g // 2
            out[bid] = (first_m + (l // hg) * hm + (l % hg) % hm, (l % hg) // hm)
        else:
            out[bid] = (first_m + l % size_m, l // size_m)
    return out


def combine_tiles(real_tiles, nbn, lead_n, lead_pct, split, group_m=1):
    out = {}
    start = real_tiles * nbn * lead_pct // 100
    for idx in range(real_tiles * nbn):
        bid = idx if idx < start else idx + lead_n
        if split:
            npg = group_m * nbn
            first_m = idx // npg * group_m
            size_m = max(1, min(group_m, real_tiles - first_m))
            l = aid_half(idx % npg, size_m * nbn, bid)
            out[bid] = (first_m + l % size_m, l // size_m)
        else:
            out[bid] = (idx // nbn, idx % nbn)
    return out


def check(name, tiles, real_tiles, nbn, tile_expert):
    got = sorted(tiles.values())
    assert got == sorted((m, n) for m in range(real_tiles) for n in range(nbn)), name + ": not a bijection"
    b = [set(), set()]
    a = [set(), set()]
    for bid, (m, n) in tiles.items():
        aid = ((bid + XCC0) % NX) // 4
        b[aid].add((tile_expert[m], n))
        a[aid].add(m)
    return [len(x) for x in b], [len(x) for x in a]


def experts(counts):
    t = []
    for e, c in enumerate(counts):
        t += [e] * ((c + 255) // 256)
    return t


rng = random.Random(0)
cases = {
    "random T=8192": [rng.randint(1900, 2200) for _ in range(32)],
    "skew-ish": [rng.choice([600, 9000]) for _ in range(32)],
    "ragged small": [rng.randint(0, 700) for _ in range(32)],
}
for cname, counts in cases.items():
    te = experts(counts)
    rt = len(te)
    print(f"== {cname}: {rt} tiles_m")
    for kname, nbn in (("L1 fwd dispatch_gemm N=4096", 16), ("L2 dgrad dispatch_gemm N=2048", 8)):
        for split in (0, 1, 2, 3, 4):
            bb, aa = check(kname, dispatch_gemm_tiles(rt, nbn, 16, 64, split), rt, nbn, te)
            print(f"  {kname:32s} split={split}: B per AID {bb}, A per AID {aa}, total blocks {sum(bb) + sum(aa)}")
    for kname, nbn in (("gemm_combine N=7168", 28),):
        for split, gm in ((False, 1), (True, 2), (True, 4)):
            bb, aa = check(kname, combine_tiles(rt, nbn, 16, 25, split, gm), rt, nbn, te)
            print(f"  {kname:32s} split={split:d} gm={gm}: B blocks per AID {bb} (sum {sum(bb)}), A row blocks per AID {aa}")
print("all bijections OK")
