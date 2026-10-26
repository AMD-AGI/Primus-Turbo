"""GEMM-only TDM-gather stress, no comm: does the gather corruption need the dispatch traffic?

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 stress_gemm.py [--iters N] [--b_nn]
Each rank (own GPU, no process group): Mega's EP4 fwd shape -- A = 32768 x 7168 recv-token rows,
~273 pool tiles of 256 rows over 32 experts with random token indices, w1 [32, 4096, 7168] --
through mega_grouped_gemm_nt_gfx1250 with the 8-wave 256x256 gather tile; every launch is compared
bitwise on GPU against the first launch (itself checked against torch). Prints bad launches.
"""
import argparse, os, sys, time, torch

sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_nt_tile import mega_grouped_gemm_nt_gfx1250

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=5000)
ap.add_argument("--b_nn", action="store_true")
ap.add_argument("--rows", type=int, default=32768)
ap.add_argument("--tiles", type=int, default=273)
ap.add_argument("--pad", type=int, default=0, help="padding rows (index -1) at the end of every expert's last tile")
ap.add_argument("--pad_idx", type=int, default=-1, help="index the padding rows gather (-1: out of bounds)")
ap.add_argument("--sorted", action="store_true", help="sort the row indices within every tile (Mega's pools are ascending)")
ap.add_argument("--zero_rows", type=int, default=0, help="append Z zero rows to A; padding rows gather row rows + p %% Z")
args = ap.parse_args()
rank = int(os.environ.get("LOCAL_RANK", 0))
dev = torch.device("cuda", rank)
torch.cuda.set_device(dev)
gen = torch.Generator(device=dev).manual_seed(77 + rank)
H, I, EPR, BM = 7168, 2048, 32, 256
randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
if args.b_nn:
    a = randn(args.rows, 2 * I)  # dy-like [tokens, 2I]... b_nn reads b [G, K, N]
    b = randn(EPR, 2 * I, H, scale=2 / (2 * I) ** 0.5)
    N = H
else:
    a = randn(args.rows, H)
    b = randn(EPR, 2 * I, H, scale=2 / H**0.5)
    N = 2 * I
tiles = args.tiles
pool = tiles * BM
counts = torch.full((EPR,), tiles // EPR, dtype=torch.int64)
counts[: tiles % EPR] += 1
tile_to_expert = torch.repeat_interleave(torch.arange(EPR), counts).to(torch.int32).to(dev)
num_tile_blocks = torch.tensor([tiles], dtype=torch.int32, device=dev)
row_idx = torch.randint(0, args.rows, (pool,), device=dev, generator=gen, dtype=torch.int64).to(torch.int32)
if args.sorted:
    row_idx = row_idx.view(-1, BM).sort(1).values.flatten().contiguous()
if args.pad:
    last = torch.cumsum(counts, 0) - 1  # last tile of each expert
    for t in last.tolist():
        row_idx[(t + 1) * BM - args.pad:(t + 1) * BM] = -1
valid = (row_idx >= 0)
if args.pad_idx >= 0:
    row_idx[~valid] = args.pad_idx
if args.zero_rows:
    a = torch.cat([a, torch.zeros(args.zero_rows, a.shape[1], dtype=a.dtype, device=dev)])
    p_ = torch.arange(pool, device=dev, dtype=torch.int32)
    row_idx = torch.where(valid, row_idx, args.rows + p_ % args.zero_rows)
out = torch.empty(pool, N, dtype=torch.bfloat16, device=dev)


def run():
    return mega_grouped_gemm_nt_gfx1250(a, b, out, tile_to_expert, num_tile_blocks, row_idx, m_warp=4, n_warp=2,
                                        group_m=16, b_nn=args.b_nn)


run()
torch.cuda.synchronize()
ref = out.clone()
# spot-check the reference against torch on a few tiles
for t in (0, tiles // 2, tiles - 1):
    e = int(tile_to_expert[t])
    rows = row_idx[t * BM:(t + 1) * BM].long()
    keep = valid[t * BM:(t + 1) * BM]
    bt = b[e] if args.b_nn else b[e].t()
    want = (a[rows[keep]].float() @ bt.float()).to(torch.bfloat16)
    err = (ref[t * BM:(t + 1) * BM][keep].float() - want.float()).abs().max().item()
    assert err < 0.05 * want.float().abs().max().item(), f"rank {rank}: reference tile {t} off by {err}"
bad = torch.zeros(args.iters, dtype=torch.int32, device=dev)
slot_hits = torch.zeros(tiles, BM // 16, dtype=torch.int32, device=dev)
t0 = time.time()
for j in range(args.iters):
    run()
    m = (out != ref).any(1) & valid
    bad[j] = m.sum()
    slot_hits += m.view(tiles, BM // 16, 16).any(2).int()
torch.cuda.synchronize()
dt = time.time() - t0
b_ = bad.cpu()
nb = int((b_ > 0).sum())
print(f"rank {rank}: {args.iters} launches in {dt:.1f} s ({dt / args.iters * 1e3:.2f} ms each); bad launches {nb}", flush=True)
for j in (b_ > 0).nonzero().flatten().tolist()[:6]:
    print(f"  it {j}: r{rank} rows {int(b_[j])}", flush=True)
nvalid = valid.view(tiles, BM).sum(1).cpu()
sh = slot_hits.cpu()
for t in (sh.sum(1) > 0).nonzero().flatten().tolist():
    print(f"  tile {t} (expert {int(tile_to_expert[t])}, valid {int(nvalid[t])}): slots " +
          " ".join(f"{s_}x{int(sh[t, s_])}" for s_ in range(BM // 16) if sh[t, s_] > 0), flush=True)
