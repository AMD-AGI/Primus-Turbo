"""Recover the A data a wrong (row, N-block) actually used, assuming one k-tile was wrong.

python analyze_dg2.py DUMP.pt [...]
For each bad row: obs - L_new = W[:, kt-block] @ (y - x[kt-block]) for the worst k-tile kt; solve y by least
squares (256 equations, 128 unknowns), report the fit residual and y's best match among candidates:
this row / other rows of the tile, new / old variant, every k-block.
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
        t = int(p2t[p])
        return cache[t // T][1][var % V][t % T].float()

    for j in list(range(0, bad.numel(), 16)) + [bad.numel() - 1]:
        p = int(bad[j]); tile = p // BM; e = int(t2e[tile])
        nb = int((out[p] != ref[p]).view(-1, TN).any(1).nonzero()[0])
        sl = slice(nb * TN, (nb + 1) * TN)
        Wn = w1[e][sl]  # [256, H]
        xa = xrow(p, v)
        Ln = Wn @ xa
        o = out[p][sl].to(dev).float()
        pa = (Wn.view(TN, KT, TK) * xa.view(1, KT, TK)).sum(2)
        zero_err = (o[:, None] - (Ln[:, None] - pa)).norm(dim=0)
        kt = int(zero_err.argmin())
        Wk = Wn[:, kt * TK:(kt + 1) * TK]
        c = o - (Ln - pa[:, kt])
        y = torch.linalg.lstsq(Wk, c[:, None]).solution.flatten()
        fit = float((Wk @ y - c).norm() / c.norm())
        # candidates
        cands = {}
        rows_in_tile = [q for q in range(tile * BM, tile * BM + BM) if bool(valid[q])]
        for var, tag in ((v, "new"), (v - 1, "old")):
            for q in rows_in_tile:
                xq = xrow(q, var)
                for k2 in range(KT):
                    cands[(tag, q - tile * BM, k2)] = xq[k2 * TK:(k2 + 1) * TK]
        keys = list(cands)
        M = torch.stack([cands[k] for k in keys])
        cos = (M @ y) / (M.norm(dim=1) * y.norm()).clamp_min(1e-30)
        best = torch.topk(cos, 3).indices.tolist()
        print(f"  row {p % BM} (e{e}, N-block {nb}): worst kt {kt}, one-ktile fit residual {fit:.3f}, |y| {float(y.norm()):.2f}"
              f" vs |x_kt| {float(xa[kt*TK:(kt+1)*TK].norm()):.2f}; best matches " +
              ", ".join(f"{keys[b][0]} row{keys[b][1]} kt{keys[b][2]} cos {float(cos[b]):.3f}" for b in best))
