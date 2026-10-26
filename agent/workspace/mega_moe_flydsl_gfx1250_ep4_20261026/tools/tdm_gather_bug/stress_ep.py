"""EP stress for intermittent corruption in fused_mega_moe fwd+bwd (gfx1250).

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 stress_ep.py [--iters N] [--sets S] [--skew X]
S fixed input sets; the first fwd+bwd of each set is its reference. Every later iteration cycles the sets and
records, per tensor (y, dx, dtw, dW1, dW2), non-finite elements and max|got-ref| / max|ref| (atomic-order
nondeterminism is ~1e-3; a race reading stale or foreign rows is far larger). Prints every bad iteration.
"""
import argparse, os, sys, time, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--sets", type=int, default=4)
ap.add_argument("--T", type=int, default=8192)
ap.add_argument("--skew", type=float, default=0.0)
ap.add_argument("--tol", type=float, default=2e-2)
args = ap.parse_args()
dev = torch.device("cuda", rank)
g = dist.group.WORLD
T, H, I, EPR, K = args.T, 7168, 2048, 32, 8
E = EPR * world
NAMES = ("y", "dx", "dtw", "dW1", "dW2")


def make_set(seed):
    gen = torch.Generator(device=dev).manual_seed(seed * 100 + rank)
    randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5).requires_grad_()
    w2 = randn(EPR, H, I, scale=2 / I**0.5).requires_grad_()
    x = randn(T, H).requires_grad_()
    logits = torch.randn(T, E, device=dev, generator=gen)
    logits[:, :EPR] += args.skew
    tw, idx = logits.softmax(-1).topk(K, dim=-1)
    tw = tw.detach().requires_grad_()
    dy = randn(T, H)
    return x, idx, tw, w1, w2, dy


def run(s):
    x, idx, tw, w1, w2, dy = s
    for t in (x, tw, w1, w2):
        t.grad = None
    y = fused_mega_moe(g, x, idx, tw, w1, w2)
    y.backward(dy)
    return (y.detach(), x.grad, tw.grad, w1.grad, w2.grad)


sets = [make_set(i + 1) for i in range(args.sets)]
refs = []
for s in sets:
    out = run(s)
    refs.append(([o.clone() for o in out], [o.float().abs().max().clamp_min(1e-30) for o in out]))
torch.cuda.synchronize(); dist.barrier()
# per iteration, per tensor: [nonfinite count, rel max diff]
stats = torch.zeros(args.iters, len(NAMES), 2, device=dev)
t0 = time.time()
for it in range(args.iters):
    si = it % args.sets
    out = run(sets[si])
    ref, scale = refs[si]
    for j, (o, r) in enumerate(zip(out, ref)):
        stats[it, j, 0] = (~torch.isfinite(o)).sum()
        stats[it, j, 1] = (o.float() - r.float()).abs().max() / scale[j]
torch.cuda.synchronize()
elapsed = time.time() - t0
allst = [torch.empty_like(stats) for _ in range(world)]
dist.all_gather(allst, stats)
if rank == 0:
    st = torch.stack(allst)  # world, iters, tensors, 2
    bad = (st[..., 0] > 0) | (st[..., 1] > args.tol)
    n_bad_it = int(bad.any(0).any(-1).sum())
    print(f"T={T} skew={args.skew} sets={args.sets}: {args.iters} iters in {elapsed:.1f} s; "
          f"bad iterations {n_bad_it}", flush=True)
    worst = st[..., 1].amax(dim=(0, 1))
    print("max rel diff per tensor over all iters/ranks: "
          + ", ".join(f"{n} {float(w):.2e}" for n, w in zip(NAMES, worst)), flush=True)
    nonzero = (st[..., 1] > 0).any(0).sum(0)
    print("iterations with any nonzero diff per tensor: "
          + ", ".join(f"{n} {int(c)}" for n, c in zip(NAMES, nonzero)), flush=True)
    shown = 0
    for it in range(args.iters):
        for r in range(world):
            for j, n in enumerate(NAMES):
                if bad[r, it, j] and shown < 40:
                    shown += 1
                    print(f"  BAD it {it} (set {it % args.sets}) rank {r} {n}: nonfinite "
                          f"{int(st[r, it, j, 0])}, rel diff {float(st[r, it, j, 1]):.3e}", flush=True)
dist.barrier()
dist.destroy_process_group()
