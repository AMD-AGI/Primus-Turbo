###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 forward benchmark: mega MoE mxfp8-compute forward (fp8_fused / fp8 / bf16 comm).

Times ``mega_moe_fused_mxfp8_forward`` (both FFN GEMMs in per-1x32 E8M0 block-scaled
mxfp8) and reports per-rank worst-case latency + SNR vs an fp32 dense reference. Comm
modes:
  * ``fp8_fused`` — ONE kernel pushes fp8 tokens + E8M0 scales cross-rank AND computes
    the L1 grouped mxfp8 GEMM (scoreboard-gated, comm hidden under the MFMA GEMM);
  * ``fp8`` — decoupled: push fp8 tokens + E8M0 scales, then a standalone L1 GEMM;
  * ``bf16`` — push bf16 tokens (2x the dispatch bytes) then quantize the pool.

``--comm <mode>`` times a single mode; ``--compare`` times several modes back-to-back
(default ``bf16,fp8_fused``) and prints a speedup table (the symm buffer reallocates on
a mode switch, so each mode is timed with its own fresh workspace).

Run inside the FlyDSL container (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_mega_moe_mxfp8.py \
      --num-processes 8 --compare --iters 50
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_mega_moe_mxfp8.py \
      --num-processes 8 --comm fp8_fused --iters 50
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _compute_snr(ref, act):
    ref = ref.float()
    act = act.float()
    noise = ref - act
    return 10.0 * torch.log10((ref * ref).mean() / (noise * noise).mean()).item()


def _dense_moe_reference(x, topk_idx, topk_w, w1g, w2g):
    """fp32 dense MoE (routing weight applied at the W2 output)."""
    xf = x.float()
    out = torch.zeros_like(xf)
    for e in range(w1g.shape[0]):
        we = (topk_w * (topk_idx == e)).sum(dim=-1)
        sel = we > 0
        if not sel.any():
            continue
        xe = xf[sel]
        gate, up = (xe @ w1g[e].float().t()).chunk(2, dim=-1)
        o = (F.silu(gate) * up) @ w2g[e].float().t()
        out[sel] += we[sel].unsqueeze(-1) * o
    return out


def _worker(rank, world, args):
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    port = int(os.environ.get("MASTER_PORT", "8531"))
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank)
    group = dist.new_group(list(range(world)))
    torch.manual_seed(123 + rank)

    from primus_turbo.flydsl.mega.fp8.mega_moe_fused_mxfp8 import mega_moe_fused_mxfp8_forward

    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.topk, args.num_tokens
    epr = E // world
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(epr, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(epr, H, I, device=dev, dtype=torch.bfloat16) * 0.05
    gate = torch.randn(T, E, device=dev)
    topk_w, topk_idx = torch.sigmoid(gate).topk(K, dim=-1)
    topk_w = (topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)

    w1g = [torch.empty_like(w1) for _ in range(world)]
    w2g = [torch.empty_like(w2) for _ in range(world)]
    dist.all_gather(w1g, w1.contiguous(), group=group)
    dist.all_gather(w2g, w2.contiguous(), group=group)
    ref = _dense_moe_reference(x, topk_idx.to(torch.int64), topk_w, torch.cat(w1g, 0), torch.cat(w2g, 0))

    def timed(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        group.barrier()
        ts = []
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            torch.cuda.synchronize()
            group.barrier()
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    def bench_mode(comm):
        def fwd():  # mxfp8 compute; dispatch comm precision = comm
            return mega_moe_fused_mxfp8_forward(x, topk_idx, topk_w, w1, w2, group, comm=comm)

        with torch.no_grad():
            snr = _compute_snr(ref, fwd())
            torch.cuda.synchronize(); group.barrier()
            ms = timed(fwd)
        stats = torch.tensor([ms, -snr], device=dev)  # worst rank: max latency, min SNR
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        return float(stats[0]), -float(stats[1])

    # One mode per spawn: switching the symm IPC workspace (use_mxfp8 differs across
    # bf16 vs fp8*) mid-process reallocates the IPC handles and races (HIP "invalid
    # argument"). main() re-spawns per mode; each writes its result to args.result_file.
    ms, snr = bench_mode(args.comm)
    if rank == 0:
        print(f"  comm={args.comm:<10s} forward: {ms:8.3f} ms   SNR={snr:5.2f} dB", flush=True)
        if args.result_file:
            with open(args.result_file, "a") as f:
                f.write(json.dumps({"comm": args.comm, "ms": ms, "snr": snr}) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="EP8 mega MoE forward benchmark: bf16 vs mxfp8")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=1024)
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--num-tokens", type=int, default=512)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--comm", choices=["fp8_fused", "fp8", "bf16"], default="fp8_fused",
                    help="dispatch comm precision for a single-mode run")
    ap.add_argument("--compare", nargs="?", const="bf16,fp8_fused", default=None,
                    help="comma-separated comm modes to time (default bf16,fp8_fused); each mode "
                         "runs in its own spawn (avoids the mid-process symm IPC realloc race); "
                         "the first mode is the speedup baseline")
    ap.add_argument("--result-file", default=None, help=argparse.SUPPRESS)  # internal: per-mode result sink
    args = ap.parse_args()

    if not args.compare:
        torch.multiprocessing.spawn(_worker, args=(args.num_processes, args), nprocs=args.num_processes)
        return

    modes = [m.strip() for m in args.compare.split(",")]
    result_file = f"/tmp/bench_mega_moe_mxfp8_{os.getpid()}.jsonl"
    if os.path.exists(result_file):
        os.remove(result_file)
    args.result_file = result_file
    for m in modes:  # one fresh 8-proc spawn per mode
        args.comm = m
        torch.multiprocessing.spawn(_worker, args=(args.num_processes, args), nprocs=args.num_processes)

    rows = {}
    with open(result_file) as f:
        for line in f:
            r = json.loads(line)
            rows[r["comm"]] = (r["ms"], r["snr"])
    os.remove(result_file)
    print("\n==== mega MoE EP8 forward (mxfp8 compute) — comm comparison ====", flush=True)
    print(f"  shape: T={args.num_tokens} H={args.hidden} I={args.inter} "
          f"E={args.num_experts} topk={args.topk} world={args.num_processes}", flush=True)
    base_ms = rows[modes[0]][0]
    for m in modes:
        ms, snr = rows[m]
        print(f"  comm={m:<10s} forward: {ms:8.3f} ms   SNR={snr:5.2f} dB   "
              f"({base_ms / ms:5.2f}x vs {modes[0]})", flush=True)


if __name__ == "__main__":
    main()
