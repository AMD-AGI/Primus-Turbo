"""Where did the data of unexplained bad rows come from? (stress_dg dumps)

python analyze_dg5.py DUMP.pt [...]
For each bad row: for every k, solve the A data y_k assuming only k-tile k is wrong (least squares); keep the best k.
Then match y in 16-element chunks against every row of the tile (and all variants) at every 8-element position;
print per chunk (row offset, element offset from k*128, variant) of the best exact-ish match.
"""
import sys, torch

dev = torch.device("cuda", 0)
T, H, I, EPR, K, BM, TN, TK, WORLD = 8192, 7168, 2048, 32, 8, 256, 256, 128, 4
E = EPR * WORLD
KT = H // TK
CH = 16


def regen(r, V):
    gen = torch.Generator(device=dev).manual_seed(1000 + r)
    randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5)
    randn(EPR, H, I, scale=2 / I**0.5)
    torch.randn(T, E, device=dev, generator=gen)
    return w1, [randn(T, H) for _ in range(V)]


cache = {}
maxrows = int(sys.argv[1]) if sys.argv[1].isdigit() else 6
paths = [a for a in sys.argv[1:] if not a.isdigit()]
for path in paths:
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
        if not (0 <= p < valid.numel()) or not bool(valid[p]):
            return None
        t = int(p2t[p])
        return cache[t // T][1][var % V][t % T].float()

    for tile in (bad // BM).unique().tolist():
        e = int(t2e[tile])
        rows = bad[(bad // BM) == tile].tolist()
        nb = int((out[rows[0]] != ref[rows[0]]).view(-1, TN).any(1).nonzero()[0])
        Wn = w1[e][nb * TN:(nb + 1) * TN]
        Wk3 = Wn.view(TN, KT, TK)
        qs = [q for q in range(tile * BM, tile * BM + BM) if bool(valid[q])]
        # candidate bank: [n_var * n_rows, H]
        bank = torch.stack([xrow(q, var) for var in range(V) for q in qs])
        tags = [(var, q) for var in range(V) for q in qs]
        # all CH-element windows at 8-element steps: [N, P, CH]
        win = bank.unfold(1, CH, 8)
        print(f"  tile {tile} e{e} N-block {nb}")
        shown = 0
        for p in rows:
            slot = (p % BM) // 16
            x = xrow(p, v)
            o = out[p][nb * TN:(nb + 1) * TN].to(dev).float()
            L = Wn @ x
            pa = (Wk3 * x.view(1, KT, TK)).sum(2)  # [TN, KT]
            best = None
            for k in range(KT):
                c = o - (L - pa[:, k])
                A = Wn[:, k * TK:(k + 1) * TK]
                y = torch.linalg.lstsq(A, c[:, None]).solution.flatten()
                res = float((A @ y - c).norm() / c.norm().clamp_min(1e-30))
                if best is None or res < best[0]:
                    best = (res, k, y)
            res, k, y = best
            parts = []
            for ci in range(TK // CH):
                yc = y[ci * CH:(ci + 1) * CH]
                err = (win - yc).abs().amax(2)  # [N, P]
                n, pp = divmod(int(err.argmin()), err.shape[1])
                e_ = float(err.view(-1)[n * err.shape[1] + pp])
                var, q = tags[n]
                if e_ > 0.05 * float(yc.abs().max()):
                    parts.append("?")
                else:
                    off = pp * 8 - (k * TK + ci * CH)
                    parts.append(f"r{q - p:+d}/e{off:+d}" + (f"/v{var - v:+d}" if var != v else ""))
            print(f"    row {p} slot {slot} k {k} res {res:.3f}: " + " ".join(parts))
            shown += 1
            if shown >= maxrows:
                break
