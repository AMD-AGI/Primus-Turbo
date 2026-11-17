"""s_memrealtime timeline of a GemmA2A: the backward o_proj (dgrad + in-kernel wgrad) or the forward QKV.

    torchrun --nproc_per_node 4 trace_gemm_a2a.py --model glm-4.7 --tokens 8192 --op fwd_qkv [--BM 256]
    torchrun --nproc_per_node 2 trace_gemm_a2a.py --model deepseek-v3 --tokens 65536 [--BM 256]
``--tokens`` is the per-rank token count (seq / n). GEMM tiles: [start, end]; pull units: [claim, ready,
end]. wgrad tiles are not traced. Defaults of --BM / --num_pull_cu / --num_tail_pull follow the op.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = os.environ.get(
    "TURBO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmark", "ops", "training"))
from bench_ulysses_overlap import MODELS, timeit

from primus_turbo.flydsl.ulysses.gemm_a2a_kernel import gemm_a2a_bank_bytes, ulysses_gemm_a2a
from primus_turbo.flydsl.ulysses.workspace import (
    REGION_O_GEMM_A2A,
    REGION_QKV_GEMM_A2A,
    get_ulysses_workspace,
)


def q(x, qs=(0.1, 0.5, 0.9)):
    return [round(x.quantile(v).item(), 1) for v in qs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-v3")
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--op", default="bwd_o", choices=["bwd_o", "fwd_qkv"])
    p.add_argument("--BM", type=int, default=None)
    p.add_argument("--num_pull_cu", type=int, default=None)
    p.add_argument("--num_tail_pull", type=int, default=None)
    p.add_argument("--num_lead", type=int, default=None)
    p.add_argument("--wgrad_at", type=int, default=None)
    p.add_argument("--no-wgrad", action="store_true")
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    cfg = MODELS[args.model]
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    M = args.tokens
    if args.op == "bwd_o":
        N, K = hq * d, hidden
        BM = args.BM or 256
        a = torch.randn(M, K, device="cuda").bfloat16()
        w = (torch.randn(K, N, device="cuda") * 0.02).bfloat16()
        og = torch.randn(M, N, device="cuda").bfloat16()
        dw = torch.empty(K, N, device="cuda", dtype=torch.bfloat16)
        do = torch.empty(M * n, hq // n, d, device="cuda", dtype=torch.bfloat16)
        outs, layout, region = (do, do, do), "nn", REGION_O_GEMM_A2A
        heads = dict(groups=1, nq=hq // n, nk=0, nv=0)
        kw = dict(num_pull_cu=8 if n == 2 else 16, num_tail_pull=16)
        hip = [lambda: a @ w] + ([] if args.no_wgrad else [lambda: a.t() @ og])
        if not args.no_wgrad:
            kw["wgrad"] = (a, og, dw)
    else:
        N, K = (hq + 2 * hkv) * d, hidden
        BM = args.BM or 128
        a = torch.randn(M, K, device="cuda").bfloat16()
        w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        outs = tuple(
            torch.empty(M * n, h // n, d, device="cuda", dtype=torch.bfloat16) for h in (hq, hkv, hkv)
        )
        layout, region = "nt", REGION_QKV_GEMM_A2A
        heads = dict(groups=hkv // n, nq=hq // hkv, nk=1, nv=1)
        kw = dict(num_pull_cu=24, num_tail_pull=8)
        hip = [lambda: F.linear(a, w)]
    for k_, v_ in (("num_pull_cu", args.num_pull_cu), ("num_tail_pull", args.num_tail_pull)):
        if v_ is not None:
            kw[k_] = v_
    if args.num_lead is not None:
        kw["num_lead"] = args.num_lead
    if args.wgrad_at is not None and "wgrad" in kw:
        kw["wgrad_at"] = args.wgrad_at
    ws = get_ulysses_workspace(group, gemm_a2a_bank_bytes(M, N, 256))
    m_tiles, n_tiles = -(-M // BM), -(-N // 256)
    nt = m_tiles * n_tiles
    trace = torch.zeros(2 * nt + 3 * n * m_tiles + 16, dtype=torch.int64, device="cuda")

    def launch(tr):
        ulysses_gemm_a2a(
            a,
            w,
            outs,
            ws,
            layout=layout,
            head_dim=d,
            seq_local=M,
            qkv_format="sbhd",
            flag_region=region,
            trace=tr,
            BM=BM,
            **heads,
            **kw,
        )

    untraced_us = timeit(lambda: launch(None), group)
    for _ in range(4):
        trace.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        launch(trace)
        e.record()
        torch.cuda.synchronize()
    hip_us = [timeit(f, group) for f in hip]
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
            f"{args.model} {args.op} M={M} N={N} K={K} n={n} BM={BM} tiles={nt} pull units={pulls.shape[0]} "
            f"{kw.get('num_pull_cu')}+{kw.get('num_tail_pull')} pull blocks; untraced {untraced_us:.0f} us, "
            f"event {s.elapsed_time(e) * 1e3:.0f} us, traced span {end:.0f} us"
        )
        dur = te - ts
        hip_tile = hip_us[0] * num_cus / nt
        print(
            f"  hipBLASLt gemm {hip_us[0]:.0f} us"
            + (f" wgrad {hip_us[1]:.0f} us" if len(hip_us) > 1 else "")
            + f" -> {hip_tile:.1f} CU-us per {BM}x256 tile"
        )
        print(
            f"  GEMM tile dur p10/50/90 {q(dur)} us (median / hipBLASLt tile {dur.median().item() / hip_tile:.2f});"
            f" tile ends p10/50/90/100 {q(te, (0.1, 0.5, 0.9, 1.0))}"
        )
        gemm_cu = dur.sum().item()
        pull_cu = (pe - pc).sum().item()
        last_gemm = te.max().item()
        print(
            f"  CU-us: GEMM tiles {gemm_cu:.0f} (hipBLASLt {hip_us[0] * num_cus:.0f}, cfg "
            f"{(dur.median().item() - hip_tile) * nt:.0f}, spread {gemm_cu - dur.median().item() * nt:.0f}),"
            f" pull units {pull_cu:.0f}; last GEMM tile ends {last_gemm:.0f}, pull tail {end - last_gemm:.0f} us"
        )
        remote = torch.tensor([(rank + u % n) % n != rank for u in range(pulls.shape[0])])
        unit_bytes = BM * (N // n) * 2
        wait, copy = pr - pc, pe - pr
        for name, m in (("local", ~remote), ("remote", remote)):
            if m.any():
                print(
                    f"  pull {name}: n={int(m.sum())} wait p50/90 {q(wait[m], (0.5, 0.9))} copy p10/50/90 "
                    f"{q(copy[m])} us -> {unit_bytes / copy[m].median().item() / 1e3:.1f} GB/s per unit"
                )
        rem_end = pe[remote].sort().values
        for frac in (0.25, 0.5, 0.75, 1.0):
            i = max(0, int(frac * len(rem_end)) - 1)
            print(
                f"  remote pulls {int(frac * 100)}% done at {rem_end[i].item():.0f} us "
                f"({(i + 1) * unit_bytes / rem_end[i].item() / 1e3:.1f} GB/s avg)"
            )
        bins = torch.linspace(0, end, 21)
        act = [int(((pr < b) & (pe > b)).sum()) for b in bins[1:-1]]
        ready = [int((te <= b).sum()) for b in bins[1:-1]]
        running = [int(((ts < b) & (te > b)).sum()) for b in bins[1:-1]]
        print("  copies in flight per 5% of the span:", act)
        print("  GEMM tiles running per 5% of the span:", running)
        print("  GEMM tiles done    per 5% of the span:", ready)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
