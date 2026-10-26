"""Rows no single k-tile explains: test 'k-tiles [k0, k1) came from another source'.

python analyze_dg6.py [N_ROWS] DUMP.pt [...]
Sources for the window: any received token t (all ranks' x rows, current variant), zeros, or "stuck" (the slot
never written again: a_k = x[k0-2 + (k-k0) % 2]). Windows [k0, k1) over all pairs. Prints the best per row.
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
nrows = int(sys.argv[1]) if sys.argv[1].isdigit() else 4
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
    X = torch.cat([cache[r][1][v % V] for r in range(WORLD)]).float()  # all senders' tokens [WORLD*T, H]

    for tile in (bad // BM).unique().tolist():
        e = int(t2e[tile])
        rows = bad[(bad // BM) == tile].tolist()
        nb = int((out[rows[0]] != ref[rows[0]]).view(-1, TN).any(1).nonzero()[0])
        Wn = w1[e][nb * TN:(nb + 1) * TN]
        Wk = Wn.view(TN, KT, TK).permute(1, 0, 2).contiguous()  # [KT, TN, TK]
        print(f"  tile {tile} e{e} N-block {nb}")
        # pick rows spread over the bad slots
        slots = sorted({(p % BM) // 16 for p in rows})
        pick = []
        for s in slots:
            pick += [p for p in rows if (p % BM) // 16 == s][:nrows]
        for p in pick:
            t = int(p2t[p])
            x = X[t].view(KT, TK)
            o = out[p][nb * TN:(nb + 1) * TN].to(dev).float()
            pa = torch.einsum("knj,kj->kn", Wk, x)  # [KT, TN]
            cs = torch.cat([torch.zeros(1, TN, device=dev), pa.cumsum(0)])  # cs[k] = sum_{<k}
            tot = cs[-1]
            best = (float((o - tot).norm()), "none", -1, -1)
            # base[k0,k1] = o - (tot - (cs[k1]-cs[k0])): what the window must produce
            ks = [(k0, k1) for k0 in range(KT) for k1 in range(k0 + 1, KT + 1)]
            k0s = torch.tensor([a for a, _ in ks], device=dev)
            k1s = torch.tensor([b for _, b in ks], device=dev)
            need = o[None] - (tot[None] - (cs[k1s] - cs[k0s]))  # [W, TN]
            # zeros
            rn = need.norm(dim=1)
            i = int(rn.argmin())
            if float(rn[i]) < best[0]:
                best = (float(rn[i]), "zero", ks[i][0], ks[i][1])
            # stuck: a_k = x[k0-2 + (k-k0)%2] for k in window
            for wi, (k0, k1) in enumerate(ks):
                if k0 < 2:
                    continue
                src = torch.stack([x[k0 - 2 + (k - k0) % 2] for k in range(k0, k1)])
                val = torch.einsum("knj,kj->n", Wk[k0:k1], src)
                r_ = float((need[wi] - val).norm())
                if r_ < best[0]:
                    best = (r_, "stuck", k0, k1)
            # another token: chunk over tokens
            for c0 in range(0, X.shape[0], 256):
                Xc = X[c0:c0 + 256].view(-1, KT, TK)
                pc = torch.einsum("knj,tkj->tkn", Wk, Xc)  # [t, KT, TN]
                cc = torch.cat([torch.zeros(pc.shape[0], 1, TN, device=dev), pc.cumsum(1)], 1)  # [t, KT+1, TN]
                val = cc[:, k1s] - cc[:, k0s]  # [t, W, TN]
                rn = (need[None] - val).norm(dim=2)  # [t, W]
                j = int(rn.argmin())
                tj, wj = divmod(j, len(ks))
                if float(rn[tj, wj]) < best[0]:
                    tt = c0 + tj
                    rows_tt = (p2t == tt).nonzero().flatten().tolist()
                    best = (float(rn[tj, wj]), f"token {tt} (r{tt // T}, pool rows {rows_tt[:3]}, d={[q - p for q in rows_tt[:3]]})", ks[wj][0], ks[wj][1])
            rel = best[0] / float(o.norm())
            print(f"    row {p} slot {(p % BM) // 16} token {t}: res {rel:.3f} {best[1]} k [{best[2]},{best[3]})")
