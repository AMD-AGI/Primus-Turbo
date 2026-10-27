"""EP MoE layer, same inputs: baseline (all-to-all + FlyDSL grouped GEMM), eager and tuned glue, vs fused_mega_moe.

HIP_VISIBLE_DEVICES=0,1,2,3 ONLY_SHAPE=<label> torchrun --nproc-per-node 4 bench_ep_baseline.py [--reps R]
One shape per process at EP4 (a second symmetric-buffer key inside one EP4 process can hang its rendezvous).

Baseline per rank, Megatron's all-to-all token dispatcher: permute the routed (token, k) rows by global expert,
all-to-all the per-expert row counts (host sync for the split sizes), all-to-all the rows and their probs, sort the
received chunks by local expert, grouped GEMM with w1, SwiGLU times the prob, grouped GEMM with w2, then back the
same way and sum each token's k rows. Backward is autograd. Eager glue is torch (argsort, gather, index_add, SwiGLU);
tuned glue is Turbo's moe_permute / moe_unpermute (TURBO backend) and swiglu_with_probs. Grouped GEMM is Turbo's
grouped_gemm on PRIMUS_TURBO_GROUPED_GEMM_BACKEND (default FLYDSL).
Each baseline's y, dx, d topk_w, dW1, dW2 are checked against fused_mega_moe's (SNR, weakest rank); times are the
max over ranks, median of --reps; Step TFLOPS = 3 * fwd_flops / (fwd+bwd ms).
"""
import argparse, os, statistics, sys, torch, torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_to_all_single as a2a

os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "FLYDSL")
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.ops import grouped_gemm, swiglu_with_probs
from primus_turbo.pytorch.ops.moe import moe_permute, moe_unpermute
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

SHAPES = [  # label, tokens/rank, hidden, inter, experts/rank, topk, skew
    ("dsv3_t8192_random", 8192, 7168, 2048, 32, 8, 0.0),
    ("dsv3_t8192_skew1", 8192, 7168, 2048, 32, 8, 1.0),
    ("dsv3_t4096_random", 4096, 7168, 2048, 32, 8, 0.0),
]
SHAPES = [s for s in SHAPES if os.environ.get("ONLY_SHAPE", s[0]) == s[0]]
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--iters", type=int, default=10)
args = ap.parse_args()
dev = torch.device("cuda", rank)
g = dist.group.WORLD


def p0(*a):
    if rank == 0:
        print(*a, flush=True)


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


def chunk_sort_index(counts):
    """counts [S, G] of rows arriving source-major; the gather index that makes them local-expert-major."""
    S, G = counts.shape
    flat = counts.flatten()
    start = torch.cumsum(flat, 0) - flat
    by_e = torch.arange(S * G, device=dev).view(S, G).t().flatten()
    lens, starts = flat[by_e], start[by_e]
    dst = torch.cumsum(lens, 0) - lens
    return torch.repeat_interleave(starts - dst, lens) + torch.arange(int(flat.sum()), device=dev)


def baseline(x, idx, tw, w1, w2, tuned):
    T, H = x.shape
    K, EPR = idx.shape[1], w1.shape[0]
    E = EPR * world
    if tuned:
        xp, row_id_map, tpe, _, _, pp = moe_permute(
            x, topk_indices=idx, num_local_experts=E, num_topk=K, probs=tw, backend=BackendType.TURBO)
        tpe = tpe.long()
    else:
        order = idx.flatten().argsort(stable=True)
        src = order // K
        xp, pp = x[src], tw.flatten()[order]
        tpe = torch.bincount(idx.flatten(), minlength=E)
    recv = torch.empty_like(tpe)
    dist.all_to_all_single(recv, tpe)
    recv = recv.view(world, EPR)
    send_split = tpe.view(world, EPR).sum(1).tolist()
    recv_split = recv.sum(1).tolist()
    n = sum(recv_split)
    xr = a2a(xp.new_empty(n, H), xp, recv_split, send_split)
    pr = a2a(pp.new_empty(n), pp.contiguous(), recv_split, send_split)
    perm = chunk_sort_index(recv)
    lens = recv.sum(0)
    l1 = grouped_gemm(xr.index_select(0, perm), w1, lens, trans_b=True)
    ps = pr.index_select(0, perm)
    if tuned:
        act = swiglu_with_probs(l1, ps.float(), None)
    else:
        gate, up = l1.float().chunk(2, dim=-1)
        act = (F.silu(gate) * up * ps[:, None]).to(l1.dtype)
    l2 = grouped_gemm(act, w2, lens, trans_b=True)
    back = l2.new_zeros(l2.shape).index_copy(0, perm, l2)
    yb = a2a(back.new_empty(sum(send_split), H), back, send_split, recv_split)
    if tuned:
        return moe_unpermute(yb, row_id_map, restore_shape=x.shape, num_local_experts=E,
                             backend=BackendType.TURBO)[0]
    return x.new_zeros(T, H).index_add(0, src, yb)


def snr(got, want):
    return -20 * torch.log10((got.float() - want.float()).norm() / want.float().norm()).item()


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
    impls = [("base_eager", lambda: baseline(x, idx, tw, w1, w2, False)),
             ("base_tuned", lambda: baseline(x, idx, tw, w1, w2, True)),
             ("mega", lambda: fused_mega_moe(g, x, idx, tw, w1, w2))]

    def run_grads(fn):
        for t in leaves:
            t.grad = None
        y = fn()
        y.backward(dy)
        return [y.detach().clone()] + [t.grad.clone() for t in leaves]

    ref = run_grads(impls[-1][1])
    flops = 2 * T * K * H * 3 * I
    res = {}
    for name, fn in impls:
        s = "SNR vs mega (y,dx,dtw,dW1,dW2) = "
        if name != "mega":
            got = run_grads(fn)
            v = torch.tensor([snr(a, b) for a, b in zip(got, ref)], device=dev)
            dist.all_reduce(v, op=dist.ReduceOp.MIN)
            s += ",".join(f"{t:.1f}" for t in v.tolist())
            del got
        else:
            s = ""
        torch.cuda.empty_cache()

        def fwd():
            with torch.no_grad():
                fn()

        def fwd_bwd():
            for t in leaves:
                t.grad = None
            fn().backward(dy)

        tf = statistics.median(timeit(fwd, args.iters) for _ in range(args.reps))
        tfb = statistics.median(timeit(fwd_bwd, args.iters) for _ in range(args.reps))
        res[name] = (tf, tfb)
        p0(f"{label:20s} {name:10s} fwd {tf:.3f} ms ({flops / tf / 1e9:.0f} TF)  fwd+bwd {tfb:.3f} ms "
           f"({3 * flops / tfb / 1e9:.0f} TF)  {s}")
    m = res["mega"]
    for name in ("base_eager", "base_tuned"):
        b = res[name]
        p0(f"{label:20s} mega vs {name}: fwd {b[0] / m[0]:.2f}x  fwd+bwd {b[1] / m[1]:.2f}x")
    del x, w1, w2, tw, leaves, ref
    torch.cuda.empty_cache()

dist.barrier()
dist.destroy_process_group()
