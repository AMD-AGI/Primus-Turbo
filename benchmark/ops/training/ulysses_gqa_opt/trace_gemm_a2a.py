"""s_memrealtime timeline of the backward o_proj GemmA2A (dgrad + in-kernel wgrad), rank 0.

    torchrun --nproc_per_node 2 trace_gemm_a2a.py --model deepseek-v3 --tokens 65536 [--BM 256]
GEMM tiles: [start, end]; pull units: [claim, ready, end]. wgrad tiles are not traced.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO = os.environ.get(
    "TURBO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
sys.path.insert(0, os.path.join(REPO, "benchmark", "ops", "training"))
from bench_ulysses_overlap import MODELS
from primus_turbo.flydsl.ulysses.gemm_a2a_kernel import gemm_a2a_bank_bytes, ulysses_gemm_a2a
from primus_turbo.flydsl.ulysses.workspace import REGION_O_GEMM_A2A, get_ulysses_workspace


def q(x, qs=(0.1, 0.5, 0.9)):
    return [round(x.quantile(v).item(), 1) for v in qs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-v3")
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--BM", type=int, default=128)
    p.add_argument("--num_pull_cu", type=int, default=16)
    p.add_argument("--num_tail_pull", type=int, default=16)
    p.add_argument("--wgrad_at", type=int, default=None)
    p.add_argument("--no-wgrad", action="store_true")
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    cfg = MODELS[args.model]
    hidden, hq, d = cfg["hidden"], cfg["hq"], cfg["d"]
    M, N, K = args.tokens, hq * d, hidden
    heads = hq // n
    dy = torch.randn(M, K, device="cuda").bfloat16()
    w = (torch.randn(K, N, device="cuda") * 0.02).bfloat16()
    og = torch.randn(M, N, device="cuda").bfloat16()
    dw = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
    do = torch.empty(M * n, heads, d, device="cuda", dtype=torch.bfloat16)
    ws = get_ulysses_workspace(group, gemm_a2a_bank_bytes(M, N, 256))
    BM = args.BM
    m_tiles, n_tiles = -(-M // BM), N // 256
    nt = m_tiles * n_tiles
    trace = torch.zeros(2 * nt + 3 * n * m_tiles + 16, dtype=torch.int64, device="cuda")
    kw = dict(BM=BM, num_pull_cu=args.num_pull_cu, num_tail_pull=args.num_tail_pull)
    if not args.no_wgrad:
        kw["wgrad"] = (dy, og, dw)
        if args.wgrad_at is not None:
            kw["wgrad_at"] = args.wgrad_at
    for _ in range(4):
        trace.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        ulysses_gemm_a2a(
            dy,
            w,
            (do, do, do),
            ws,
            layout="nn",
            head_dim=d,
            groups=1,
            nq=heads,
            nk=0,
            nv=0,
            seq_local=M,
            qkv_format="sbhd",
            flag_region=REGION_O_GEMM_A2A,
            trace=trace,
            **kw,
        )
        e.record()
        torch.cuda.synchronize()
    if rank == 0:
        t = trace.cpu().double()
        tiles = t[: 2 * nt].view(nt, 2)
        pulls = t[2 * nt : 2 * nt + 3 * n * m_tiles].view(-1, 3)
        t0 = tiles[tiles[:, 0] > 0, 0].min()
        us = lambda x: (x - t0) / 100.0
        ts, te = us(tiles[:, 0]), us(tiles[:, 1])
        pc, pr, pe = us(pulls[:, 0]), us(pulls[:, 1]), us(pulls[:, 2])
        end = max(te.max().item(), pe.max().item())
        print(
            f"{args.model} M={M} n={n} BM={BM} tiles={nt} pull units={pulls.shape[0]} "
            f"event {s.elapsed_time(e) * 1e3:.0f} us, traced span {end:.0f} us"
        )
        dur = te - ts
        print(f"  GEMM tile dur p10/50/90 {q(dur)} us; tile ends p10/50/90/100 {q(te, (0.1, 0.5, 0.9, 1.0))}")
        remote = torch.tensor([(rank + u % n) % n != rank for u in range(pulls.shape[0])])
        unit_bytes = BM * (N // n) * 2
        wait, copy = pr - pc, pe - pr
        for name, m in (("local", ~remote), ("remote", remote)):
            if m.any():
                print(
                    f"  pull {name}: n={int(m.sum())} wait p50/90 {q(wait[m], (0.5, 0.9))} copy p10/50/90 "
                    f"{q(copy[m])} us -> {unit_bytes / copy[m].median().item() / 1e3:.1f} GB/s per unit"
                )
        # remote bytes landed over time
        rem_end = pe[remote].sort().values
        for frac in (0.25, 0.5, 0.75, 1.0):
            i = max(0, int(frac * len(rem_end)) - 1)
            print(
                f"  remote pulls {int(frac * 100)}% done at {rem_end[i].item():.0f} us "
                f"({(i + 1) * unit_bytes / rem_end[i].item() / 1e3:.1f} GB/s avg)"
            )
        # concurrency of pull copies over time
        bins = torch.linspace(0, end, 21)
        act = [int(((pr < b) & (pe > b)).sum()) for b in bins[1:-1]]
        ready = [int((te <= b).sum()) for b in bins[1:-1]]
        print("  copies in flight per 5% of the span:", act)
        print("  GEMM tiles done   per 5% of the span:", ready)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
