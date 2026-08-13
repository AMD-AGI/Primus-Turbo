#!/usr/bin/env python3
"""Repro: the mxfp8 mega-MoE combine's GEMM-done gate releases the PUSH before the GEMM
has written L2Y, so the PUSH ships bytes left over from the previous call.

The defect is pre-existing (reproduces well before the two-stage split) and is the same one
recorded as P1 in mxfp8_bwd_perf_note.md 3.1. Every fp8 correctness test to date runs a single
fixed routing, where the leftover bytes happen to equal the correct bytes, so it measured clean.

Two modes:

  --poison  fills the local L2Y scratch with the fp8 NaN byte 0xFF before each call. Any byte the
            PUSH ships that the GEMM did not write this call becomes a NaN in the output. Turns the
            defect into an unambiguous count instead of a data-dependent one. No kernel change
            needed -- the fill is host side.

  (default) no fill. Fresh routing every iteration, which is what a real training step does; the
            leftover bytes are then the previous call's, and where a row used to be group padding
            its L2Y is fp8 NaN, so NaN reaches the output on its own.

Run (8 GPUs):
  PYTHONPATH=<repo> python3 repro_fp8_combine_gate.py --poison
  PYTHONPATH=<repo> python3 repro_fp8_combine_gate.py
  PYTHONPATH=<repo> python3 repro_fp8_combine_gate.py --fixed-routing   # clean: how it stayed hidden

Measured on 8 x gfx950, T=8192 K=8, NaN output tokens per rank out of 8192:

  --poison         iter 0    0 (nothing stale to leak yet)
                   iter 1+   850-1000 on every rank, every iteration
  default          iter 0    0
                   iter 1    39-86 on every rank
                   iter 2+   0-23, only some ranks
  --fixed-routing  all 0 -- this is the condition every fp8 test has run under

--poison also reproduces under --fixed-routing, which is the point: the defect does not depend on
routing at all. Routing only decides whether the leftover bytes happen to equal the correct ones.

WORKED AROUND, NOT FIXED: the GEMM's release kept only one `s_waitcnt(0)` + `s_barrier` before
storing the done flag, and a second rendezvous (which the bf16 combine has always carried) makes the
symptom go away. Delete that second pair to get the old behaviour back and watch every mode above
light up.

But the second pair cannot be what closes the hole. Dumping the ISA both ways shows the two builds
identical apart from those two instructions, and after the first `s_waitcnt(0)` vmcnt is already 0
with no memory op in between -- nothing to wait for, no wave left to sync. What it buys is time, so
the race is still there and a different compiler or schedule could reopen it. The `s_waitcnt` between
the barriers is load-bearing even so: LLVM folds adjacent `s_barrier`s.

Shape of the corruption, for whoever picks this up: with the pair removed it is ~800 of 8192 output
tokens per rank, torn at 32-column granularity INSIDE tiles rather than whole tiles missing, spread
over 10-12 of the 28 tiles per bad token with no fixed hot tile. That is a propagation race, not a
gate that fails to cover something. `.ab/diag_combine_gate.py` prints that breakdown.

What was ruled out on the way, so nobody repeats it:
  - stale origin_rank in the pool: the live-row count matches num_tokens_per_expert.sum() exactly
  - the pushed data itself: L2Y live rows are never NaN once the kernel has finished, i.e. the GEMM
    covers every row it should -- the PUSH just read them too early
  - the reduce's comb reads: every slot is pushed, and the E8M0 scale is never the 0xFF NaN encoding
  - the whole memory-visibility family: `buffer_wbl2` sc1 and sc0|sc1 on the release (worse, ~1100),
    dropping sc1 from the C stores and releasing with wbl2 (worse still, ~1400), `gpu.barrier` for
    the bare `s_barrier` (no change), sc0|sc1 on the C stores, sc1 on the PUSH's L2Y loads, dropping
    the PUSH's `buffer_inv`, sys-scope flag traffic, an `s_waitcnt` before every flag read, and
    coherent epoch parity/expected traffic. Every fence-shaped fix leaves it at ~870 or worse, and
    both sides already read correct at the ISA level: the writer releases with sc1 stores +
    `s_waitcnt` + barrier, the reader acquires with an sc0|sc1 flag load + `s_waitcnt` +
    `buffer_inv`. Untried: making the flag store a release atomic rather than a plain sc1 store.

Measurement trap: flyc's on-disk runtime cache can serve a stale binary across runs that differ only
by an env-driven kernel constant, so an A/B can silently compare a config against itself. Give every
arm its own FLYDSL_RUNTIME_CACHE_DIR and repeat each arm at least twice.
"""

import argparse
import datetime
import math
import os

import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.mega.fp8.grouped_gemm_combine_fp8_kernel import _L2Y_FP8_SCRATCH
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import fused_mega_moe_fp8

H, I, E = 7168, 2048, 256
_FP8_NAN = 0xFF


def worker(rank, world, args):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '8503')}",
        world_size=world,
        rank=rank,
        timeout=datetime.timedelta(seconds=900),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    T, K = args.num_tokens, args.num_topk
    experts_per_rank = E // world

    torch.manual_seed(123 + rank)
    w1 = torch.randn(experts_per_rank, 2 * I, H, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    w2 = torch.randn(experts_per_rank, H, I, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    torch.manual_seed(1000 + rank)
    x = torch.randn(T, H, dtype=torch.bfloat16)

    for it in range(args.iters):
        seed = 2000 + rank if args.fixed_routing else 2000 + rank + it * 97
        torch.manual_seed(seed)
        gate = torch.randn(T, E)
        w0, topk_idx = torch.sigmoid(gate).topk(K, dim=-1)
        topk_w = (w0 / (w0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)

        with torch.no_grad():
            if args.poison and _L2Y_FP8_SCRATCH:
                # every L2Y byte the GEMM does not rewrite this call is now an fp8 NaN
                for l2y_fp8, _ in _L2Y_FP8_SCRATCH.values():
                    l2y_fp8.fill_(_FP8_NAN)
                torch.cuda.synchronize()
                group.barrier()

            y = fused_mega_moe_fp8(group, x, topk_idx.to(torch.int64), topk_w, w1, w2)
            torch.cuda.synchronize()

        nan_tokens = torch.tensor([float(torch.isnan(y.float()).any(dim=1).sum())], device="cuda")
        gathered = [torch.zeros_like(nan_tokens) for _ in range(world)]
        dist.all_gather(gathered, nan_tokens, group=group)
        if rank == 0:
            counts = " ".join(f"{int(g):5d}" for g in gathered)
            print(f"  iter {it}  NaN tokens per rank (of {T}): {counts}", flush=True)
        torch.cuda.synchronize()
        group.barrier()

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--poison", action="store_true", help="prefill L2Y with the fp8 NaN byte")
    ap.add_argument(
        "--fixed-routing", action="store_true",
        help="reuse one routing across iterations, as every fp8 test does",
    )
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
