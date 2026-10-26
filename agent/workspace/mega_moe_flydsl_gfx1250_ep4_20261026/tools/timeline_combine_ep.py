"""EP=world timeline of one gemm_combine launch (gc_timed2 stamps), per role, on each rank.

torchrun --nproc-per-node 2 timeline_combine_ep.py T H EXPERTS_PER_RANK K I [nn]
nn: the L1 dgrad form (b_nn with grad_gate), N=H, K=2I; else the L2 forward (K=I).
"""
import os, sys, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
sys.path.insert(0, "/home/xiaompen/mega_scratch")
import primus_turbo.pytorch  # noqa
import gc_timed2 as gc
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import run_dispatch_prologue

T, H, EPR, K, I = (int(v) for v in sys.argv[1:6])
nn = len(sys.argv) > 6 and sys.argv[6] == "nn"
E, g = EPR * world, dist.group.WORLD
gen = torch.Generator(device="cuda").manual_seed(7 + rank)
w1 = torch.empty(EPR, 2 * I, H, device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
tw, idx = torch.randn(T, E, device="cuda", generator=gen).softmax(-1).topk(K, dim=-1)
h, _ = run_dispatch_prologue(x, w1, g, idx, tw)
rows = h.pool_row_to_route.numel()
kdim = 2 * I if nn else I
act = torch.randn(rows, kdim, device="cuda", generator=gen).to(torch.bfloat16)
w = (torch.randn(EPR, kdim, H, device="cuda", generator=gen) / kdim**0.5).to(torch.bfloat16)
if not nn:
    w = w.transpose(1, 2).contiguous()
gate = torch.randn(rows, device="cuda", generator=gen) if nn else None


def call():
    return gc.gemm_combine_nt_gfx1250(act, w, h, idx, grad_gate=gate, b_nn=nn)


for _ in range(5):
    call()
torch.cuda.synchronize(); dist.barrier()
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
a.record()
for _ in range(20):
    call()
b.record(); torch.cuda.synchronize()
dist.barrier()
gc.DBG.zero_()
torch.cuda.synchronize(); dist.barrier()
call()
torch.cuda.synchronize()

symm = gc.get_symm_buffer_for_mega_moe()
NB = (H + 255) // 256
max_tiles = int(symm.num_max_pool_tokens) // 256
ntm = int(h.num_tile_blocks)
ccu = 16
num_cu = torch.cuda.get_device_properties(0).multi_processor_count
tcu = num_cu - ccu
cstart = ntm * (NB * 25 // 100)
ngemm = ntm * NB
grid = ngemm + ccu + tcu
d = gc.DBG[: grid * 4].view(grid, 4).cpu()
t0 = d[:, 0][d[:, 0] > 0].min()
us = lambda v: (v - t0).double() / 100.0
bid = torch.arange(grid)
is_lead = (bid >= cstart) & (bid < cstart + ccu)
is_tail = bid >= ngemm + ccu
gidx = torch.where(bid < cstart, bid, bid - ccu)
live = ~is_lead & ~is_tail & (gidx < ntm * NB)
q = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
fmt = lambda v: "[" + " ".join(f"{float(x):7.1f}" for x in torch.quantile(v, q)) + "]"
lines = [f"rank {rank} {'L1 dgrad nn' if nn else 'L2 fwd'}: launch {a.elapsed_time(b) / 20 * 1e3:.1f} us "
         f"(avg of 20); tiles_m {ntm}, gemm live {int(live.sum())}; quantiles 0/10/50/90/100%"]
lines.append(f"  gemm live end        {fmt(us(d[live, 1]))}")
lines.append(f"  lead push  start     {fmt(us(d[is_lead, 1]))}  end {fmt(us(d[is_lead, 2]))}")
lines.append(f"  tail start           {fmt(us(d[is_tail, 0]))}")
lines.append(f"  tail push  end       {fmt(us(d[is_tail, 2]))}")
lines.append(f"  reduce end (all)     {fmt(us(d[is_lead | is_tail, 3]))}")
gemm_end = float(us(d[live, 1]).max())
late = is_tail & (us(d[:, 0]) > gemm_end + 100)
lines.append(f"  late tail (start > gemm end + 100us): {int(late.sum())} of {int(is_tail.sum())}; "
             f"bid%8 {torch.bincount(bid[late] % 8, minlength=8).tolist()}; lead bid%8 "
             f"{torch.bincount(bid[is_lead] % 8, minlength=8).tolist()}; tail bid%8 "
             f"{torch.bincount(bid[is_tail] % 8, minlength=8).tolist()}")
lines.append(f"  late tail bids {bid[late][:8].tolist()} .. {bid[late][-4:].tolist()}; tail range "
             f"{int(bid[is_tail].min())}..{int(bid[is_tail].max())}; lead {bid[is_lead].tolist()}")
ch = gc.DBG[gc.CH0:].view(-1, 4).cpu()
ch = ch[ch[:, 0] > 0]
taker = ch[:, 0]
c_lead = (taker >= cstart) & (taker < cstart + ccu)
cw0, cw1, cp1 = us(ch[:, 1]), us(ch[:, 2]), us(ch[:, 3])
lines.append(f"  chunks {len(ch)}: lead {int(c_lead.sum())} (done before gemm end "
             f"{int((c_lead & (cp1 < gemm_end)).sum())}), tail {int((~c_lead).sum())}")
lines.append(f"  lead: wait sum/WG {float((cw1 - cw0)[c_lead].sum()) / ccu:7.1f} us, push sum/WG "
             f"{float((cp1 - cw1)[c_lead].sum()) / ccu:7.1f} us; push per chunk {fmt((cp1 - cw1)[c_lead])}")
lines.append(f"  tail: wait per chunk {fmt((cw1 - cw0)[~c_lead])}; push per chunk {fmt((cp1 - cw1)[~c_lead])}")
lines.append(f"  chunk push end after gemm end: {fmt(cp1[cp1 > gemm_end])}")
# chunk id -> segment -> dst rank, as the kernel numbers them (segments in order, 16 fold rows a chunk)
fc = h.combine_recv_fold_count.cpu().long()
nch = (fc + 15) // 16
seg_of_chunk = torch.repeat_interleave(torch.arange(len(fc)), nch)
dst_of_chunk = h.combine_recv_dst_rank.cpu().long()[seg_of_chunk]
allch = gc.DBG[gc.CH0:gc.CH0 + 4 * len(seg_of_chunk)].view(-1, 4).cpu()
push = us(allch[:, 3]) - us(allch[:, 2])
tail_ch = (allch[:, 0] >= ngemm + ccu)
for name, m in (("local", dst_of_chunk == rank), ("remote", dst_of_chunk != rank)):
    lines.append(f"  tail {name:6s} chunks {int((m & tail_ch).sum()):4d}: push {fmt(push[m & tail_ch])}; "
                 f"lead {name} {int((m & ~tail_ch).sum())}: push {fmt(push[m & ~tail_ch])}")
try:
    import ep_timed
    rw = gc.DBG[ep_timed.RW0:ep_timed.RW0 + 8 * len(seg_of_chunk)].view(-1, 2, 4).cpu()
    for name, m in (("local", dst_of_chunk == rank), ("remote", dst_of_chunk != rank)):
        for u in range(2):
            r = rw[m & tail_ch & (rw[:, u, 3] > 0), u]
            if len(r):
                lines.append(f"  tail {name:6s} row {u}: index {fmt(us(r[:, 1]) - us(r[:, 0]))} sum+store issue "
                             f"{fmt(us(r[:, 2]) - us(r[:, 1]))} retire {fmt(us(r[:, 3]) - us(r[:, 2]))}")
except ImportError:
    pass
msg = "\n".join(lines)
if os.environ.get("TL_DUMP"):
    torch.save(dict(d=d, t0=t0, ngemm=ngemm, cstart=cstart, ccu=ccu, live=live, is_lead=is_lead, is_tail=is_tail),
               f"{os.environ['TL_DUMP']}_r{rank}.pt")
out = [None] * world
dist.all_gather_object(out, msg)
if rank == 0:
    print("\n".join(out), flush=True)
dist.barrier()
dist.destroy_process_group()
