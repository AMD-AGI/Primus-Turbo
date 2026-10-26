"""Mega MoE BF16 gfx1250 EP4 campaign: per-shape correctness (SNR vs fp32 torch) + fwd / fwd+bwd timing.

HIP_VISIBLE_DEVICES=0,1,2,3 ONLY_SHAPE=<label> torchrun --nproc-per-node 4 quick_test_bench.py [--csv OUT.csv]
One shape per process at EP4 (run_round.sh): a second symmetric-buffer key inside one EP4 process can hang
its rendezvous. Without ONLY_SHAPE every shape runs (fine at EP2).

Per shape: tokens/rank, hidden, inter, experts/rank, topk, skew (logit bias on rank 0's experts).
Each rank checks its own tokens and experts' grads (y, dx, d topk_w, dW1, dW2 > 40 dB on the weakest
rank), then times fused_mega_moe fwd and fwd+bwd with fresh grads; times are the max over ranks,
median of --reps repeats. Combined Step TFLOPS = 3 * fwd_flops / (fwd+bwd ms).
"""
import argparse, csv, os, statistics, sys, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
sys.path.insert(0, "/home/xiaompen/Turbo")
import primus_turbo.pytorch  # noqa
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe
from tests.pytorch.ops.test_mega_bf16_gfx1250_fused_moe import _reference, _snr

SHAPES = [  # label, tokens/rank, hidden, inter, experts/rank, topk, skew
    ("dsv3_t8192_random", 8192, 7168, 2048, 32, 8, 0.0),
    ("dsv3_t8192_skew1", 8192, 7168, 2048, 32, 8, 1.0),
    ("dsv3_t4096_random", 4096, 7168, 2048, 32, 8, 0.0),
]
SHAPES = [s for s in SHAPES if os.environ.get("ONLY_SHAPE", s[0]) == s[0]]
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--iters", type=int, default=10)
args = ap.parse_args()
dev = torch.device("cuda", rank)
g = dist.group.WORLD


def p0(*a):
    if rank == 0:
        print(*a, flush=True)


def gather_rows(t):
    n = torch.tensor([t.shape[0]], device=dev)
    counts = [torch.empty_like(n) for _ in range(world)]
    dist.all_gather(counts, n)
    counts = [int(c) for c in counts]
    buf = t.new_zeros((max(counts),) + t.shape[1:])
    buf[: t.shape[0]] = t
    out = [torch.empty_like(buf) for _ in counts]
    dist.all_gather(out, buf.contiguous())
    return torch.cat([o[:c] for o, c in zip(out, counts)]), sum(counts[:rank])


def max_over_ranks(v):
    t = torch.tensor([v], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def timeit(fn, n):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record(); torch.cuda.synchronize()
    return max_over_ranks(a.elapsed_time(b) / n)


rows = []
for label, T, H, I, EPR, K, skew in SHAPES:
    E = EPR * world
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5).requires_grad_()
    w2 = randn(EPR, H, I, scale=2 / I**0.5).requires_grad_()
    x = randn(T, H).requires_grad_()
    logits = torch.randn(T, E, device=dev, generator=gen)
    logits[:, :EPR] += skew
    tw, idx = logits.softmax(-1).topk(K, dim=-1)
    tw = tw.detach().requires_grad_()
    dy = randn(T, H)
    leaves = (x, tw, w1, w2)

    # ---- correctness
    for t in leaves:
        t.grad = None
    y = fused_mega_moe(g, x, idx, tw, w1, w2)
    y.backward(dy)
    gx, row0 = gather_rows(x.detach())
    refs = [t.float().requires_grad_() for t in (gx, gather_rows(tw.detach())[0],
                                                  gather_rows(w1.detach())[0], gather_rows(w2.detach())[0])]
    yr = _reference(refs[0], gather_rows(idx)[0], *refs[1:], None)
    yr.backward(gather_rows(dy)[0].float())
    own_t = slice(row0, row0 + T)
    own_e = slice(rank * EPR, (rank + 1) * EPR)
    checks = [("y", y, yr.detach()[own_t]), ("dx", x.grad, refs[0].grad[own_t]),
              ("dtopk_w", tw.grad, refs[1].grad[own_t]), ("dW1", w1.grad, refs[2].grad[own_e]),
              ("dW2", w2.grad, refs[3].grad[own_e])]
    snr = torch.tensor([_snr(got, want) for _, got, want in checks], device=dev)
    dist.all_reduce(snr, op=dist.ReduceOp.MIN)
    snr = snr.tolist()
    ok = all(s > 40 for s in snr)
    del refs, yr, gx, y
    torch.cuda.empty_cache()

    # ---- timing
    def fwd():
        with torch.no_grad():
            fused_mega_moe(g, x, idx, tw, w1, w2)

    def fwd_bwd():
        for t in leaves:
            t.grad = None
        fused_mega_moe(g, x, idx, tw, w1, w2).backward(dy)

    tf = statistics.median(timeit(fwd, args.iters) for _ in range(args.reps))
    tfb = statistics.median(timeit(fwd_bwd, args.iters) for _ in range(args.reps))
    flops = 2 * T * K * H * 3 * I
    fwd_tf = flops / tf / 1e9
    bwd_tf = 2 * flops / (tfb - tf) / 1e9
    step_tf = 3 * flops / tfb / 1e9
    row = dict(label=label, T=T, H=H, I=I, EPR=EPR, K=K, skew=skew, check="PASS" if ok else "FAIL",
               snr_min=min(snr), fwd_ms=tf, fwd_bwd_ms=tfb, fwd_tflops=fwd_tf, bwd_tflops=bwd_tf,
               step_tflops=step_tf)
    rows.append(row)
    p0(f"{label:20s} {row['check']} snr(y,dx,dtw,dW1,dW2)=" + ",".join(f"{s:.1f}" for s in snr)
       + f"  fwd {tf:.3f} ms ({fwd_tf:.0f} TF)  fwd+bwd {tfb:.3f} ms  bwd {bwd_tf:.0f} TF  step {step_tf:.1f} TF")
    del x, w1, w2, tw, leaves
    torch.cuda.empty_cache()

if rank == 0:
    passed = [r for r in rows if r["check"] == "PASS"]
    gm = lambda k: statistics.geometric_mean(r[k] for r in passed) if len(passed) == len(rows) else 0.0
    p0(f"AGGREGATE step {gm('step_tflops'):.3f} TF  fwd {gm('fwd_tflops'):.3f} TF  bwd {gm('bwd_tflops'):.3f} TF"
       f"  ({len(passed)}/{len(rows)} PASS)")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
dist.barrier()
dist.destroy_process_group()
