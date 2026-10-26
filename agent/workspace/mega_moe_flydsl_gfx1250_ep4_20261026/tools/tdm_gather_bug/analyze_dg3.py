"""Per 16-row gather slot of a wrong tile: whose data each slot held in the wrong k-tile.

python analyze_dg3.py DUMP.pt [...]
For every bad row: recover the wrong k-tile's A data y (least squares, as analyze_dg2) and match it against
this row at kt-2 (slot left stale) and rows r + 16*d (d = -3..3) at kt (an op written d slots away).
"""
import sys, torch

dev = torch.device("cuda", 0)
T, H, I, EPR, K, BM, TN, TK, WORLD = 8192, 7168, 2048, 32, 8, 256, 256, 128, 4
E = EPR * WORLD
KT = H // TK


def regen(r, V):
    gen = torch.Generator(device=dev).manual_seed(1000 + r)
    randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5)
    randn(EPR, H, I, scale=2 / I**0.5)
    torch.randn(T, E, device=dev, generator=gen)
    return w1, [randn(T, H) for _ in range(V)]


cache = {}
for path in sys.argv[1:]:
    d = torch.load(path, weights_only=False)
    V, v, rk = d["vars"], d["v"], d["rank"]
    for r in range(WORLD):
        if r not in cache:
            cache[r] = regen(r, V)
    w1 = cache[rk][0].float()
    out, ref, valid, h = d["out"], d["ref"], d["valid"], d["handle"]
    p2t = h["pool_row_to_recv_token"].long()
    t2e = h["tile_to_expert"]
    bad = ((out != ref).any(1) & valid).nonzero().flatten()
    print(f"== {path.split('/')[-1]}: receiver r{rk} v={v} bad rows {bad.numel()}")

    def xrow(p, var):
        if not bool(valid[p]):
            return None
        t = int(p2t[p])
        return cache[t // T][1][var % V][t % T].float()

    for tile in (bad // BM).unique().tolist():
        e = int(t2e[tile])
        rows = bad[(bad // BM) == tile]
        nb = int((out[rows[0]] != ref[rows[0]]).view(-1, TN).any(1).nonzero()[0])
        sl = slice(nb * TN, (nb + 1) * TN)
        Wn = w1[e][sl]
        # one k-tile for the whole tile: the most common worst k-tile
        kts = []
        info = {}
        for p in rows.tolist():
            xa = xrow(p, v)
            o = out[p][sl].to(dev).float()
            Ln = Wn @ xa
            pa = (Wn.view(TN, KT, TK) * xa.view(1, KT, TK)).sum(2)
            kts.append(int((o[:, None] - (Ln[:, None] - pa)).norm(dim=0).argmin()))
            info[p] = (o, Ln, pa)
        kt = max(set(kts), key=kts.count)
        Wk = Wn[:, kt * TK:(kt + 1) * TK]
        line = []
        for slot in range(BM // 16):
            members = [p for p in rows.tolist() if (p % BM) // 16 == slot]
            if not members:
                line.append(f"{slot:2d}:ok")
                continue
            votes = {}
            for p in members:
                o, Ln, pa = info[p]
                c = o - (Ln - pa[:, kt])
                y = torch.linalg.lstsq(Wk, c[:, None]).solution.flatten()
                cands = {"stale": xrow(p, v)[(kt - 2) * TK:(kt - 1) * TK]}
                for dd in range(-3, 4):
                    q = p + 16 * dd
                    if dd != 0 and tile * BM <= q < tile * BM + BM and xrow(q, v) is not None:
                        cands[f"{dd:+d}"] = xrow(q, v)[kt * TK:(kt + 1) * TK]
                best, bc = "?", 0.9
                for k_, vec in cands.items():
                    cs = float(vec @ y / (vec.norm() * y.norm()).clamp_min(1e-30))
                    if cs > bc:
                        best, bc = k_, cs
                votes[best] = votes.get(best, 0) + 1
            line.append(f"{slot:2d}:" + "/".join(f"{k_}x{n}" for k_, n in sorted(votes.items(), key=lambda t: -t[1])))
        print(f"  tile {tile} e{e} N-block {nb} kt {kt}: " + "  ".join(line))
