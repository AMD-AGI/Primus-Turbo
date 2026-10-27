"""Single-GPU check of the gather GEMM against torch per (K, tile_k): which rows of a 256-row tile are off.

python3 probe_tilek.py [--k 2880 2048] [--tile_k 64 128] [--pad 70]
Prints, per config, the worst per-row relative error and the rows (position within the tile) above 1e-2.
"""
import argparse, os, sys, torch

sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_nt_tile import mega_grouped_gemm_nt_gfx1250

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, nargs="+", default=[2880, 2048])
ap.add_argument("--tile_k", type=int, nargs="+", default=[64, 128])
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--rows", type=int, default=4096)
ap.add_argument("--tiles", type=int, default=16)
ap.add_argument("--pad", type=int, default=0)
args = ap.parse_args()
dev = torch.device("cuda")
gen = torch.Generator(device=dev).manual_seed(5)
G, BM = 4, 256
for K in args.k:
    for tk in args.tile_k:
        if K % tk:
            continue
        a = torch.randn(args.rows, K, device=dev, generator=gen).to(torch.bfloat16)
        b = (torch.randn(G, args.n, K, device=dev, generator=gen) / K**0.5).to(torch.bfloat16)
        tte = (torch.arange(args.tiles, device=dev) % G).to(torch.int32)
        ntb = torch.tensor([args.tiles], dtype=torch.int32, device=dev)
        idx = torch.randint(0, args.rows, (args.tiles * BM,), device=dev, generator=gen, dtype=torch.int32)
        if args.pad:
            idx.view(args.tiles, BM)[:, BM - args.pad:] = -1
        out = torch.zeros(args.tiles * BM, args.n, dtype=torch.bfloat16, device=dev)
        mega_grouped_gemm_nt_gfx1250(a, b, out, tte, ntb, idx, tile_k=tk, m_warp=4, n_warp=2)
        torch.cuda.synchronize()
        e = tte.repeat_interleave(BM).long()
        valid = idx >= 0
        ref = torch.einsum("pk,pnk->pn", a[idx.clamp_min(0).long()].float(), b[e].float())
        err = (out.float() - ref).norm(dim=1) / ref.norm(dim=1)
        err[~valid] = 0
        bad = (err > 1e-2).view(args.tiles, BM)
        pos = bad.any(0).nonzero().flatten().tolist()
        print(f"K={K} tile_k={tk}: worst row rel err {err.max().item():.2e}, bad rows {int(bad.sum())}"
              f"/{int(valid.sum())}, positions in tile {pos[:40]}{' ...' if len(pos) > 40 else ''}", flush=True)
