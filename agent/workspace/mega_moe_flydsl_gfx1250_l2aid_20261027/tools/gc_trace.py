"""gc_trace.py T [SKEW]: timeline of the two gemm_combine launches (L2 fwd, L1 dgrad) of one fused_mega_moe step.

Needs the tracing patch (tools/gc_trace.patch) applied and MEGA_GC_TRACE=1. The kernel stamps the realtime clock
per workgroup (entry, GEMM tile / combine done, reduce done) and per combine chunk (ready, pushed, rows, dst rank);
the clock rate is calibrated against the host over a 0.5 s gap. Prints per rank, in us from the launch's first
workgroup entry: GEMM tile completion quantiles, combine workgroup starts, pushed bytes before / after the last
GEMM tile, the tail push rate, the reduce, and the chunk ready-wait vs push split. Raw traces: OUT/trace_r<rank>.pt
"""
import os, sys, time, torch, torch.distributed as dist
import numpy as np

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
g = dist.group.WORLD
import primus_turbo.pytorch  # noqa
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe
import primus_turbo.flydsl.mega.bf16_gfx1250.gemm_combine_kernel as GC

T = int(sys.argv[1])
SKEW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
OUT = os.environ.get("GC_TRACE_OUT", "/tmp/gc_trace")
H, I, EPR, K = 7168, 2048, 32, 8
E = EPR * world
gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
randn = lambda *s, scale=1.0: (torch.randn(*s, device="cuda", generator=gen) * scale).to(torch.bfloat16)
w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5).requires_grad_()
w2 = randn(EPR, H, I, scale=2 / I**0.5).requires_grad_()
x = randn(T, H).requires_grad_()
logits = torch.randn(T, E, device="cuda", generator=gen)
if SKEW:
    logits[:, :EPR] += SKEW
tw, idx = logits.softmax(-1).topk(K, dim=-1)
tw = tw.detach().requires_grad_()
dy = randn(T, H)


def step():
    for t in (x, tw, w1, w2):
        t.grad = None
    fused_mega_moe(g, x, idx, tw, w1, w2).backward(dy)


for _ in range(4):
    step()
torch.cuda.synchronize(); dist.barrier()
h0 = time.perf_counter(); step(); torch.cuda.synchronize()
first = [t.clone() for _, t in GC.TRACES[-2:]]
time.sleep(0.5)
h1 = time.perf_counter(); step(); torch.cuda.synchronize()
metas = [m for m, _ in GC.TRACES[-2:]]
last = [t.cpu().numpy() for _, t in GC.TRACES[-2:]]


def entry0(tr, m):
    wg = tr[: m["chunk_base"]].reshape(-1, 4)[: m["grid"]]
    return wg[wg[:, 3] > 0, 0].min()


ticks_per_us = (entry0(last[0], metas[0]) - entry0(first[0].cpu().numpy(), metas[0])) / ((h1 - h0) * 1e6)
os.makedirs(OUT, exist_ok=True)
torch.save(dict(metas=metas, traces=last, ticks_per_us=ticks_per_us), f"{OUT}/trace_r{rank}.pt")


def analyze(tr, m):
    us = lambda t: (t - t0) / ticks_per_us
    wg = tr[: m["chunk_base"]].reshape(-1, 4)[: m["grid"]]
    role, ent, t1, t2 = wg[:, 3], wg[:, 0], wg[:, 1], wg[:, 2]
    t0 = ent[role > 0].min()
    gm, ld, tl = role == 1, role == 2, role == 3
    q = lambda a, p: us(np.quantile(a, p))
    ch = tr[m["chunk_base"] :].reshape(-1, 4)
    ch = ch[ch[:, 1] > 0]
    rows = ch[:, 3] & 0xFFFFFFFF
    dst = ch[:, 3] >> 32
    nbytes = rows * m["hidden"] * 2
    remote = dst != m["rank"]
    gemm_end = t1[gm].max()
    done = ch[:, 2]
    pre = done <= gemm_end
    last_push = done.max()
    by_lead = np.isin(ch[:, 0], np.nonzero(ld)[0])
    # ready-wait: from the workgroup's previous chunk done (or its entry) to this chunk's ready
    order = np.lexsort((ch[:, 1], ch[:, 0]))
    wait = np.zeros(len(ch))
    prev_bid, prev_done = -1, 0
    for i in order:
        b = ch[i, 0]
        start = prev_done if b == prev_bid else ent[b]
        wait[i] = (ch[i, 1] - start) / ticks_per_us
        prev_bid, prev_done = b, ch[i, 2]
    push = (ch[:, 2] - ch[:, 1]) / ticks_per_us
    tail_dur = (last_push - gemm_end) / ticks_per_us
    lines = [
        f"  GEMM tiles {gm.sum()}: done 10/50/90/100% at {q(t1[gm], .1):.0f} / {q(t1[gm], .5):.0f} / "
        f"{q(t1[gm], .9):.0f} / {us(gemm_end):.0f} us",
        f"  lead combine {ld.sum()}: enter {us(ent[ld].min()):.0f}-{us(ent[ld].max()):.0f}, combine done "
        f"{us(t1[ld].max()):.0f}; tail combine {tl.sum()}: enter 10/50/90% {q(ent[tl], .1):.0f} / "
        f"{q(ent[tl], .5):.0f} / {q(ent[tl], .9):.0f}",
        f"  pushed before last GEMM tile: {nbytes[pre & remote].sum() / 1e6:.0f} of {nbytes[remote].sum() / 1e6:.0f} "
        f"MB remote, {nbytes[pre & ~remote].sum() / 1e6:.0f} of {nbytes[~remote].sum() / 1e6:.0f} MB local "
        f"(lead WGs {nbytes[by_lead].sum() / 1e6:.0f} MB in {len(ch[by_lead])} chunks)",
        f"  tail push: last chunk at {us(last_push):.0f} us, {tail_dur:.0f} us after the GEMM, "
        f"{nbytes[~pre & remote].sum() / 1e6:.0f} MB remote -> {nbytes[~pre & remote].sum() / 1e3 / max(tail_dur, 1e-9):.0f} GB/s",
        f"  reduce: done {us(t2[ld | tl].max()):.0f} us ({(t2[ld | tl].max() - last_push) / ticks_per_us:.0f} after last push)",
        f"  chunk push us median/p90: lead {np.median(push[by_lead]) if by_lead.any() else 0:.1f}/"
        f"{np.quantile(push[by_lead], .9) if by_lead.any() else 0:.1f}, tail {np.median(push[~by_lead]):.1f}/"
        f"{np.quantile(push[~by_lead], .9):.1f}; ready-wait median lead {np.median(wait[by_lead]) if by_lead.any() else 0:.1f}"
        f", tail {np.median(wait[~by_lead]):.1f}",
    ]
    return "\n".join(lines)


text = f"rank {rank} (clock {ticks_per_us:.2f} ticks/us)\n"
for m, tr in zip(metas, last):
    text += f" {'L1 dgrad' if m['grad_gate'] else 'L2 fwd'} gemm_combine:\n" + analyze(tr, m) + "\n"
texts = [None] * world
dist.all_gather_object(texts, text)
if rank == 0:
    print("\n".join(texts), flush=True)
dist.destroy_process_group()
