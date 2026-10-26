"""Greedy multi-k-tile explanation of every bad row of a stress_dg dump.

python analyze_dg4.py DUMP.pt [...]
For each bad row: out = sum_k W_k a_k with a_k = x_k except a few k-tiles. Greedily pick (k, source) replacements
that best reduce the residual; sources = same row at k+-1/2/4 (stale / future), rows r+16d (op shift) and r+-1..3
(row shift) at k, zeros. Prints per row the replacement list, then per tile a slot x k-tile map.
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

    def xrow(p, dv=0):
        if not (0 <= p < valid.numel()) or not bool(valid[p]):
            return None
        t = int(p2t[p])
        return cache[t // T][1][(v + dv) % V][t % T].float()

    for tile in (bad // BM).unique().tolist():
        e = int(t2e[tile])
        rows = bad[(bad // BM) == tile].tolist()
        nb = int((out[rows[0]] != ref[rows[0]]).view(-1, TN).any(1).nonzero()[0])
        Wn = w1[e][nb * TN:(nb + 1) * TN].view(TN, KT, TK).permute(1, 0, 2).contiguous()  # [KT, TN, TK]
        grid = {}
        print(f"  tile {tile} e{e} N-block {nb}  (slot = row//16 within tile)")
        for p in rows:
            x = xrow(p).view(KT, TK)
            o = out[p][nb * TN:(nb + 1) * TN].to(dev).float()
            names, srcs = [], []
            for dk in (-4, -2, -1, 1, 2, 4):
                names.append(f"k{dk:+d}")
                srcs.append(torch.roll(x, -dk, 0))  # srcs[k] = x[k+dk]
            for dd in (-3, -2, -1, 1, 2, 3):
                q = xrow(p + 16 * dd) if tile * BM <= p + 16 * dd < tile * BM + BM else None
                if q is not None:
                    names.append(f"op{dd:+d}")
                    srcs.append(q.view(KT, TK))
            for dr in (-3, -2, -1, 1, 2, 3):
                q = xrow(p + dr) if (p // 16) == ((p + dr) // 16) else None
                if q is not None:
                    names.append(f"row{dr:+d}")
                    srcs.append(q.view(KT, TK))
            for dv in range(1, V):
                names.append(f"var{-dv:+d}")
                srcs.append(xrow(p, -dv).view(KT, TK))
                for dk in (-2, 2):
                    names.append(f"var{-dv:+d}k{dk:+d}")
                    srcs.append(torch.roll(xrow(p, -dv).view(KT, TK), -dk, 0))
            names.append("zero")
            srcs.append(torch.zeros_like(x))
            S = torch.stack(srcs)  # [C, KT, TK]
            cur = x.clone()
            chosen = []
            res = o - torch.einsum("knj,kj->n", Wn, cur)
            r0 = float(res.norm())
            for _ in range(4):
                delta = torch.einsum("knj,ckj->ckn", Wn, S - cur[None])  # [C, KT, TN]
                rn = (res[None, None] - delta).norm(dim=2)
                c, k = divmod(int(rn.argmin()), KT)
                if float(rn[c, k]) > 0.5 * float(res.norm()):
                    break
                cur[k] = S[c, k]
                res = res - delta[c, k]
                chosen.append((k, names[c]))
            rel = float(res.norm()) / max(float(o.norm()), 1e-30)
            for k, nm in chosen:
                grid.setdefault(((p % BM) // 16, k), {}).setdefault(nm, 0)
                grid[((p % BM) // 16, k)][nm] += 1
            if not chosen:
                grid.setdefault(((p % BM) // 16, -1), {}).setdefault("?", 0)
                grid[((p % BM) // 16, -1)]["?"] += 1
        for slot in sorted({s for s, _ in grid}):
            items = sorted((k, kv) for (s, k), kv in grid.items() if s == slot)
            desc = "  ".join(f"kt{k}:" + "/".join(f"{n}x{c}" for n, c in sorted(kv.items(), key=lambda t: -t[1])) for k, kv in items)
            print(f"    slot {slot:2d} (wave {slot // 2} op {slot % 2}): {desc}")
