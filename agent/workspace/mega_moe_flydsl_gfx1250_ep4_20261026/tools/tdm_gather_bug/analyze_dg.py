"""Analyze stress_dg dumps: where the wrong rows sit and which A data they were computed from.

python analyze_dg.py DUMP.pt [...]   (one GPU; regenerates every rank's x variants and the receiver's w)
Per bad (row, 256-col block), rel err ||obs - L|| / ||L_new|| against
  new   : the dispatched row x_v (the reference)
  old   : the previous launch's row x_{v-1} (what the slot held before this launch)
  split : x_{v-1} for k-tiles < k*, x_v after (best k*): the row landed while the tile was reading it
"""
import sys, torch

dev = torch.device("cuda", 0)
T, H, I, EPR, K, BM, TN, TK, WORLD = 8192, 7168, 2048, 32, 8, 256, 256, 128, 4
E = EPR * WORLD


def regen(r, V):
    gen = torch.Generator(device=dev).manual_seed(1000 + r)
    randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5)
    w2 = randn(EPR, H, I, scale=2 / I**0.5)
    torch.randn(T, E, device=dev, generator=gen)
    xs = [randn(T, H) for _ in range(V)]
    return w1, w2, xs


cache = {}
for path in sys.argv[1:]:
    d = torch.load(path, weights_only=False)
    V, v, rk = d["vars"], d["v"], d["rank"]
    assert not d["b_nn"], "fwd dumps only"
    for r in range(WORLD):
        if r not in cache:
            cache[r] = regen(r, V)
    w1 = cache[rk][0].float()
    out, ref, valid, h = d["out"], d["ref"], d["valid"], d["handle"]
    bad = ((out != ref).any(1) & valid).nonzero().flatten()
    tok = h["pool_row_to_recv_token"][bad].long()
    t2e = h["tile_to_expert"]
    print(f"== {path.split('/')[-1]}: launch {d['it']} v={v} receiver r{rk}: {bad.numel()} bad rows")
    tiles = (bad // BM).unique().tolist()
    for t in tiles:
        m = (bad // BM) == t
        rows = bad[m] - t * BM
        snd = (tok[m] // T).unique().tolist()
        colbad = (out[bad[m]] != ref[bad[m]]).view(-1, 4096 // TN, TN).any(2).any(0).nonzero().flatten().tolist()
        print(f"  tile {t} expert {int(t2e[t])}: rows-in-tile {rows.tolist()}")
        print(f"    senders {snd}, bad N-blocks {colbad}")
    # hypotheses on up to 12 rows
    for j in range(min(12, bad.numel())):
        p = int(bad[j]); s, tk = int(tok[j]) // T, int(tok[j]) % T
        e = int(t2e[p // BM])
        xa = cache[s][2][v][tk].float()
        xb = cache[s][2][(v - 1) % V][tk].float()
        Wj = w1[e]
        pa = (Wj.view(2 * I, H // TK, TK) * xa.view(1, H // TK, TK)).sum(2)
        pb = (Wj.view(2 * I, H // TK, TK) * xb.view(1, H // TK, TK)).sum(2)
        Ln, Lo = pa.sum(1), pb.sum(1)
        ca, cb = pa.cumsum(1), pb.cumsum(1)
        Ls = cb + (ca[:, -1:] - ca)
        o = out[p].to(dev).float()
        refrow_ok = float((ref[p].to(dev).float() - Ln).norm() / Ln.norm())
        cells = []
        for b in range(4096 // TN):
            sl = slice(b * TN, (b + 1) * TN)
            nrm = Ln[sl].norm()
            en = float((o[sl] - Ln[sl]).norm() / nrm)
            if en < 0.02:
                cells.append("ok")
                continue
            eo = float((o[sl] - Lo[sl]).norm() / nrm)
            es = (o[sl][:, None] - Ls[sl]).norm(dim=0) / nrm
            ks = int(es.argmin())
            # one k-tile j wrong: old data, zero, or another k-tile j2 of the new row (stale LDS stage)
            pas, pbs = pa[sl], pb[sl]
            one_old = (o[sl][:, None] - (Ln[sl][:, None] - pas + pbs)).norm(dim=0) / nrm
            one_zero = (o[sl][:, None] - (Ln[sl][:, None] - pas)).norm(dim=0) / nrm
            jo, jz = int(one_old.argmin()), int(one_zero.argmin())
            cells.append(f"n{en:.2f}/o{eo:.2f}/s{float(es[ks]):.2f}@{(ks + 1) * TK} "
                         f"1old{float(one_old[jo]):.3f}@kt{jo} 1zero{float(one_zero[jz]):.3f}@kt{jz}")
        print(f"  row {p} (tile row {p % BM}) sender {s} tok {tk} e{e} ref~new {refrow_ok:.3f}: " + " ".join(cells))
