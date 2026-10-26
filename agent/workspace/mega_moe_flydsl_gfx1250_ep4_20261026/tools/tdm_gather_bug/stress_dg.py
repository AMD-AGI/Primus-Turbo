"""Focused reproducer: the fused fwd dispatch_gemm alone, fixed routing, rotating x, bitwise vs reference.

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 stress_dg.py [--iters N] [--vars V] [--skew_us S]
  [--churn] [--b_nn]
One prologue (routing fixed), V x variants. Each launch: random per-rank spin (0..S us) to skew the ranks,
optional churn (gather-read the whole token buffer so its lines sit in the cache, like the wgrads),
dispatch_gemm(x[v]), compare the live pool rows bitwise with x[v]'s reference, then a tiny all_reduce
(GPU-side cross-rank barrier, no host sync) so no rank overwrites a peer's token buffer early.
"""
import argparse, os, random, sys, time, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
import primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_gfx1250_impl as _impl
from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=5000)
ap.add_argument("--vars", type=int, default=4)
ap.add_argument("--T", type=int, default=8192)
ap.add_argument("--skew_us", type=float, default=100.0)
ap.add_argument("--churn", action="store_true")
ap.add_argument("--b_nn", action="store_true")
ap.add_argument("--no_barrier", action="store_true")
ap.add_argument("--dump", default="")
ap.add_argument("--standalone", action="store_true", help="after the refs: standalone gather GEMM on the symmetric token buffer, no comm")
ap.add_argument("--copy_a", action="store_true", help="standalone: gather from a plain-device-memory copy of the token buffer")
ap.add_argument("--no_dead", action="store_true", help="standalone: grid = the live tiles only (no masked tiles)")
ap.add_argument("--remap_pad", action="store_true", help="standalone: padding rows gather scattered valid rows instead of -1")
args = ap.parse_args()
dev = torch.device("cuda", rank)
g = dist.group.WORLD
T, H, I, EPR, K = args.T, 7168, 2048, 32, 8
E = EPR * world
gen = torch.Generator(device=dev).manual_seed(1000 + rank)
randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5)
w2 = randn(EPR, H, I, scale=2 / I**0.5)
tw, idx = torch.randn(T, E, device=dev, generator=gen).softmax(-1).topk(K, dim=-1)
xs = [randn(T, H) for _ in range(args.vars)]
handle, _ = _impl.run_dispatch_prologue(xs[0], w1, g, idx, tw)
w = w2 if args.b_nn else w1
tok_buf = get_symm_buffer_for_mega_moe().dispatch_token_buffer
route = handle.pool_row_to_route.long()
tokidx = handle.pool_row_to_recv_token[: route.numel()].long()
valid = (route >= 0) & (tokidx >= 0) & (tokidx < tok_buf.shape[0])
gidx = torch.where(valid, tokidx, 0)
one = torch.zeros(1, device=dev)
clk = 2.0e3  # spin cycles per us (approx)


def launch(v):
    out = _impl.dispatch_gemm_nt_gfx1250(xs[v], w, handle, b_nn=args.b_nn)
    return out


refs = []
for v in range(args.vars):
    out = launch(v)
    torch.cuda.synchronize(); dist.barrier()
    refs.append(out[: route.numel()].clone())
# reference stability: rerun each once with host syncs
for v in range(args.vars):
    out = launch(v)
    torch.cuda.synchronize(); dist.barrier()
    assert torch.equal(out[: route.numel()][valid], refs[v][valid]), "reference not reproducible"
if args.standalone:
    from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_nt_tile import mega_grouped_gemm_nt_gfx1250
    out = launch(0)  # token buffer now holds x[0]'s tokens
    torch.cuda.synchronize(); dist.barrier()
    pool_rows = (route.numel() + 255) // 256 * 256
    if args.no_dead:
        pool_rows = int(handle.num_tile_blocks[0]) * 256
    print(f"[rank {rank}] standalone pool tiles {pool_rows // 256}, live {int(handle.num_tile_blocks[0])}", flush=True)
    out_sa = torch.empty(pool_rows, w.shape[2] if args.b_nn else w.shape[1], dtype=torch.bfloat16, device=dev)
    ridx = handle.pool_row_to_recv_token[:pool_rows].contiguous()
    a_sa = tok_buf.clone() if args.copy_a else tok_buf
    if args.remap_pad:
        pr = torch.arange(ridx.numel(), device=dev, dtype=torch.int64)
        ridx = torch.where(ridx >= 0, ridx.long(), (pr * 2654435761) % tok_buf.shape[0]).to(torch.int32).contiguous()

    def launch(v):
        return mega_grouped_gemm_nt_gfx1250(a_sa, w, out_sa, handle.tile_to_expert, handle.num_tile_blocks, ridx,
                                            m_warp=4, n_warp=2, group_m=16, b_nn=args.b_nn)

    launch(0)
    torch.cuda.synchronize()
    same = torch.equal(out_sa[: route.numel()][valid], refs[0][valid])
    print(f"[rank {rank}] standalone vs fused ref bitwise equal: {same}", flush=True)
    refs = [out_sa[: route.numel()].clone()] * args.vars
stats = torch.zeros(args.iters, device=dev)
badrows = torch.zeros(args.iters, device=dev)
host = torch.zeros(args.iters).pin_memory()
RING = 4
ring, events = [None] * RING, [None] * args.iters
dumped = 0


def check(j):
    global dumped
    events[j].synchronize()
    if float(host[j]) == 0.0 or dumped >= 2 or not args.dump:
        return
    vj = 0 if args.standalone else j % args.vars
    os.makedirs(args.dump, exist_ok=True)
    path = f"{args.dump}/dg_rank{rank}_it{j}.pt"
    torch.save(dict(it=j, v=vj, vars=args.vars, rank=rank, b_nn=args.b_nn, out=ring[j % RING].cpu(),
                    ref=refs[vj].cpu(), valid=valid.cpu(),
                    handle={f: getattr(handle, f).cpu() for f in handle._fields}), path)
    print(f"[rank {rank}] dumped launch {j} -> {path}", flush=True)
    dumped += 1
rng = random.Random(rank * 7 + 1)
torch.cuda.synchronize(); dist.barrier()
t0 = time.time()
for it in range(args.iters):
    v = it % args.vars
    if args.skew_us > 0:
        torch.cuda._sleep(int(rng.random() * args.skew_us * clk))
    if args.churn:
        one += tok_buf[gidx].float().sum(dtype=torch.float32) * 0
    out = launch(v)
    d = (out[: route.numel()] != refs[v]).any(1) & valid
    stats[it] = d.any().float()
    badrows[it] = d.sum().float()
    if args.dump:
        ring[it % RING] = out[: route.numel()].clone() if args.standalone else out[: route.numel()]
        host[it].copy_(stats[it], non_blocking=True)
        events[it] = torch.cuda.Event()
        events[it].record()
        if it >= RING - 1:
            check(it - (RING - 1))
    if not args.no_barrier:
        dist.all_reduce(one)
torch.cuda.synchronize()
elapsed = time.time() - t0
if args.dump:
    for j in range(max(0, args.iters - (RING - 1)), args.iters):
        check(j)
allst = [torch.empty_like(stats) for _ in range(world)]
allrows = [torch.empty_like(badrows) for _ in range(world)]
dist.all_gather(allst, stats)
dist.all_gather(allrows, badrows)
if rank == 0:
    st = torch.stack(allst)
    br = torch.stack(allrows)
    nb = int((st.sum(0) > 0).sum())
    print(f"T={T} vars={args.vars} skew={args.skew_us}us churn={args.churn} b_nn={args.b_nn} "
          f"barrier={not args.no_barrier}: {args.iters} launches in {elapsed:.1f} s; bad launches {nb}", flush=True)
    shown = 0
    for it in range(args.iters):
        if st[:, it].sum() > 0 and shown < 20:
            shown += 1
            print(f"  it {it}: " + ", ".join(f"r{r} rows {int(br[r, it])}" for r in range(world) if st[r, it] > 0),
                  flush=True)
dist.barrier()
dist.destroy_process_group()
