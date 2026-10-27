"""Variable-K wgrad TDM-gather stress, no comm: is Mega's gathered wgrad hit by the same corruption?

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 stress_wgrad.py [--iters N] [--dw2]
Each rank (own GPU, no process group) runs grouped_gemm_bf16_variable_k_flydsl_kernel the way Mega's EP4
backward does: dW1 = grad_l1[pool].T @ x_recv[b_index] ([E, 2I, H], the default) or with --dw2
dW2 = dy_recv[a_index].T @ act_w[pool] ([E, H, I]). The pool is 256-row padded per expert, the per-expert
real counts go in masked_k, so every expert's last k-tile re-reads its last row (duplicate gather
indices). Every launch is compared bitwise on GPU against the first launch (itself checked against
torch on a few experts). Prints bad launches and the (expert, 256-row output block) they hit.
"""
import argparse, os, sys, time, torch

sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    grouped_gemm_bf16_variable_k_flydsl_kernel as variable_k,
)

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=5000)
ap.add_argument("--dw2", action="store_true")
ap.add_argument("--rows", type=int, default=32768, help="rows of the gathered (dispatch-buffer) operand")
ap.add_argument("--tokens", type=int, default=2048, help="mean real tokens per expert")
ap.add_argument("--spread", type=int, default=512, help="per-expert count is tokens +- spread")
ap.add_argument("--sorted", action="store_true", help="ascending gather indices within every expert")
ap.add_argument("--bench", type=int, default=0, help="only time N launches (no compare)")
args = ap.parse_args()
rank = int(os.environ.get("LOCAL_RANK", 0))
dev = torch.device("cuda", rank)
torch.cuda.set_device(dev)
gen = torch.Generator(device=dev).manual_seed(91 + rank)
H, I, EPR, BM = 7168, 2048, 32, 256
randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)

counts = torch.randint(args.tokens - args.spread, args.tokens + args.spread + 1, (EPR,), device=dev, generator=gen)
padded = (counts + BM - 1) // BM * BM
prefix = torch.zeros(EPR + 1, dtype=torch.int64, device=dev)
prefix[1:] = torch.cumsum(padded, 0)
pool = int(prefix[-1])
row_index = torch.full((pool,), -1, dtype=torch.int32, device=dev)
for e in range(EPR):
    c, p0 = int(counts[e]), int(prefix[e])
    r = torch.randperm(args.rows, device=dev, generator=gen)[:c].to(torch.int32)
    row_index[p0:p0 + c] = r.sort().values if args.sorted else r
recv = randn(args.rows, H)
if args.dw2:
    a, b, kw = recv, randn(pool, I), dict(a_index=row_index)
else:
    a, b, kw = randn(pool, 2 * I), recv, dict(b_index=row_index)
counts = counts.to(torch.int64)


def run():
    return variable_k(a, b, prefix, masked_k=counts, **kw)


ref = run()
torch.cuda.synchronize()
for e in (0, EPR // 2, EPR - 1):
    rows = prefix[e] + torch.arange(int(counts[e]), device=dev)
    src = row_index[rows].long()
    ae = a[src] if args.dw2 else a[rows]
    be = b[rows] if args.dw2 else b[src]
    want = ae.float().t() @ be.float()
    err = (ref[e].float() - want).abs().max().item()
    assert err < 0.02 * want.abs().max().item(), f"rank {rank}: reference expert {e} off by {err}"
if args.bench:
    for _ in range(10):
        run()
    ts = []
    for rep in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(args.bench):
            run()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / args.bench)
    ts.sort()
    print(f"rank {rank}: bench median {ts[2] * 1e3:.1f} us  min {ts[0] * 1e3:.1f} us", flush=True)
    sys.exit(0)
G, OM, ON = ref.shape
bm, bn = OM // BM, ON // BM
bad = torch.zeros(args.iters, dtype=torch.int32, device=dev)
blk_hits = torch.zeros(G, bm, bn, dtype=torch.int32, device=dev)
t0 = time.time()
for j in range(args.iters):
    out = run()
    m = (out != ref).view(G, bm, BM, bn, BM).any(4).any(2)
    bad[j] = m.sum()
    blk_hits += m.int()
    del out
torch.cuda.synchronize()
dt = time.time() - t0
b_ = bad.cpu()
nb = int((b_ > 0).sum())
print(f"rank {rank}: {args.iters} launches in {dt:.1f} s ({dt / args.iters * 1e3:.2f} ms each); bad launches {nb}",
      flush=True)
for j in (b_ > 0).nonzero().flatten().tolist()[:6]:
    print(f"  it {j}: r{rank} output blocks {int(b_[j])}", flush=True)
bh = blk_hits.cpu()
for g, i, k in bh.nonzero().tolist()[:12]:
    print(f"  expert {g} (count {int(counts[g])}) block ({i}, {k}): x{int(bh[g, i, k])}", flush=True)
