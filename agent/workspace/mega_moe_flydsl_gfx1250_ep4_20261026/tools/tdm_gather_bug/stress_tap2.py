"""stress_tap.py, plus a crime-scene dump of the first iteration whose L1 output differs.

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 stress_tap2.py [--iters N] [--dump DIR]
Every iteration clones (no host sync): the token buffer before the fwd dispatch (= last iteration's
dispatch_only), after it, the L1 output and the handle tables, into a 4-deep ring. Each iteration's
stats row goes to pinned host memory asynchronously; two iterations later (its kernels are done by then:
the prologue's .item() of the following iteration drained them) the host reads it, and the first bad L1
on this rank dumps that iteration's ring entry to DIR/rank<r>_it<i>.pt. Also counts per-iteration
handle-table differences vs the set's reference.
"""
import argparse, os, sys, time, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe
import primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_gfx1250_impl as _impl
from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--sets", type=int, default=4)
ap.add_argument("--T", type=int, default=8192)
ap.add_argument("--dump", default="/home/xiaompen/mega_scratch/ep4c/dumps")
args = ap.parse_args()
dev = torch.device("cuda", rank)
g = dist.group.WORLD
T, H, I, EPR, K = args.T, 7168, 2048, 32, 8
E = EPR * world
TAPS = ("xdisp", "l1", "y", "dx", "dtw", "dW1", "dW2")
HFIELDS = ("expert_send_count", "expert_send_offset", "tile_to_expert", "num_tokens_per_expert",
           "num_tokens_per_expert_prefix", "num_tile_blocks", "combine_recv_dst_rank", "combine_recv_start_row",
           "combine_recv_fold_count")
RING = 4
ring = [dict() for _ in range(RING)]
cur = {}


def route_order(handle, num_rows, num_tokens):
    n = min(handle.pool_row_to_route.numel(), handle.pool_row_to_recv_token.numel(), num_rows)
    route = handle.pool_row_to_route[:n].long()
    tok = handle.pool_row_to_recv_token[:n].long()
    valid = (route >= 0) & (tok >= 0) & (tok < num_tokens)
    order = torch.argsort(torch.where(valid, route, torch.iinfo(torch.int64).max), stable=True)
    keep = valid[order]
    return order, keep, torch.where(keep, tok[order], 0)


def masked(t, keep):
    return torch.where(keep.view(-1, *([1] * (t.dim() - 1))), t, torch.zeros((), dtype=t.dtype, device=t.device))


_orig_pro = _impl.run_dispatch_prologue


def _pro(*a, **kw):
    # the token buffer before this step's fwd dispatch overwrites it (queued before the prologue)
    try:
        cur["tok_before"] = get_symm_buffer_for_mega_moe().dispatch_token_buffer.clone()
    except RuntimeError:  # first step: the buffer is created later in this call
        pass
    return _orig_pro(*a, **kw)


_impl.run_dispatch_prologue = _pro
_orig_dg = _impl.dispatch_gemm_nt_gfx1250


def _dg(x, w, handle, *a, b_nn=False, **kw):
    out = _orig_dg(x, w, handle, *a, b_nn=b_nn, **kw)
    if not b_nn:
        cur["tok_after"] = get_symm_buffer_for_mega_moe().dispatch_token_buffer.clone()
        cur["l1"] = out.clone()
        cur["handle"] = {f: getattr(handle, f).clone() for f in handle._fields}
    return out


_impl.dispatch_gemm_nt_gfx1250 = _dg


def make_set(seed):
    gen = torch.Generator(device=dev).manual_seed(seed * 100 + rank)
    randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5).requires_grad_()
    w2 = randn(EPR, H, I, scale=2 / I**0.5).requires_grad_()
    x = randn(T, H).requires_grad_()
    tw, idx = torch.randn(T, E, device=dev, generator=gen).softmax(-1).topk(K, dim=-1)
    tw = tw.detach().requires_grad_()
    return x, idx, tw, w1, w2, randn(T, H)


def run(s):
    x, idx, tw, w1, w2, dy = s
    for t in (x, tw, w1, w2):
        t.grad = None
    cur.clear()
    y = fused_mega_moe(g, x, idx, tw, w1, w2)
    y.backward(dy)
    h = cur["handle"]

    class _H:
        pass

    hh = _H()
    for f, v in h.items():
        setattr(hh, f, v)
    ntok = cur["tok_after"].shape[0]
    order, keep, tok = route_order(hh, cur["l1"].shape[0], ntok)
    out = dict(xdisp=masked(cur["tok_after"][tok], keep), l1=masked(cur["l1"][order], keep), y=y.detach(),
               dx=x.grad, dtw=tw.grad, dW1=w1.grad, dW2=w2.grad)
    return out, dict(cur, order=order, keep=keep)


sets = [make_set(i + 1) for i in range(args.sets)]
refs = []
for s in sets:
    out, scene = run(s)
    refs.append(({k: (v.clone(), v.float().abs().max().clamp_min(1e-30)) for k, v in out.items()},
                 {f: scene["handle"][f].clone() for f in HFIELDS}))
torch.cuda.synchronize(); dist.barrier()
NCOL = len(TAPS) + len(HFIELDS)
stats = torch.zeros(args.iters, NCOL, device=dev)
host = torch.zeros(args.iters, NCOL).pin_memory()
dumped = False
os.makedirs(args.dump, exist_ok=True)


def check(it):
    global dumped
    row = host[it]
    forced = it == int(os.environ.get("FORCE_DUMP_IT", "-1")) and rank == 0
    if dumped or (float(row[TAPS.index("l1")]) == 0.0 and not forced):
        return
    sc = ring[it % RING]
    assert sc["it"] == it
    prev = ring[(it - 1) % RING]
    path = f"{args.dump}/rank{rank}_it{it}.pt"
    l1_ref = refs[it % args.sets][0]["l1"][0]
    torch.save(dict(it=it, set=it % args.sets, rank=rank, stats=row.clone(), taps=TAPS, hfields=HFIELDS,
                    l1=sc["l1"].cpu(), l1_ref_routed=l1_ref.cpu(), order=sc["order"].cpu(), keep=sc["keep"].cpu(),
                    handle={f: v.cpu() for f, v in sc["handle"].items()},
                    tok_before=sc["tok_before"].cpu(), tok_after=sc["tok_after"].cpu(),
                    prev_it=prev.get("it")), path)
    print(f"[rank {rank}] dumped it {it} -> {path}", flush=True)
    dumped = True


t0 = time.time()
for it in range(args.iters):
    if it >= 2:
        check(it - 2)
    si = it % args.sets
    out, scene = run(sets[si])
    ref, href = refs[si]
    for j, k in enumerate(TAPS):
        r, scale = ref[k]
        o = out[k]
        stats[it, j] = -1.0 if o.shape != r.shape else (o.float() - r.float()).abs().max() / scale
    for j, f in enumerate(HFIELDS):
        a, b = scene["handle"][f], href[f]
        stats[it, len(TAPS) + j] = -1.0 if a.shape != b.shape else (a != b).sum().float()
    host[it].copy_(stats[it], non_blocking=True)
    scene["it"] = it
    ring[it % RING] = scene
torch.cuda.synchronize()
for it in range(max(0, args.iters - 2), args.iters):
    check(it)
elapsed = time.time() - t0
allst = [torch.empty_like(stats) for _ in range(world)]
dist.all_gather(allst, stats)
if rank == 0:
    st = torch.stack(allst)
    names = TAPS + HFIELDS
    nz = (st != 0).any(0).sum(0)
    print(f"T={T} sets={args.sets}: {args.iters} iters in {elapsed:.1f} s", flush=True)
    print("iterations with a nonzero diff: " + ", ".join(f"{k} {int(c)}" for k, c in zip(names, nz)), flush=True)
    stable = [j for j, c in enumerate(nz) if int(c) < args.iters // 10]
    shown = 0
    for it in range(args.iters):
        bad = [(r, j) for r in range(world) for j in stable if st[r, it, j] != 0]
        if bad and shown < 20:
            shown += 1
            print(f"  it {it} (set {it % args.sets}): " + "; ".join(
                f"{names[j]}@r{r} {float(st[r, it, j]):.2e}" for j in stable for r in range(world)
                if st[r, it, j] != 0), flush=True)
dist.barrier()
dist.destroy_process_group()
