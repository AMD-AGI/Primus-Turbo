"""A2AGemm launch-knob sweep (fwd o_proj, bwd QKV dgrad+wgrad) against the shape heuristic.

torchrun --nproc_per_node 4 sweep_a2a_gemm.py --model glm-4.7 --seq 32768
"""

import os
import sys

_REPO = os.environ.get(
    "TURBO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "benchmark", "ops", "training"))

import argparse
import itertools
import os

import torch
import torch.distributed as dist
from bench_ulysses_overlap import MODELS, build_ops, timeit

from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import A2AGemmTuning, _o_gemm_forward, _qkv_backward

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--seq", type=int, default=32768)
p.add_argument("--only", choices=("fwd", "bwd"), default=None)
p.add_argument("--bm", type=int, nargs="+", default=[128, 256])
p.add_argument("--push", type=int, nargs="+", default=[8, 16, 32, 48])
p.add_argument("--split_m", type=int, nargs="+", default=[0], help="fwd o_proj: last m-tiles split in K")
p.add_argument("--split_k", type=int, nargs="+", default=[2])
p.add_argument(
    "--wat", type=float, nargs="+", default=None, help="bwd wgrad_at as fractions of the dgrad tiles"
)
a = p.parse_args()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group("nccl")
g = dist.group.WORLD
n, rank = g.size(), g.rank()
ops = build_ops(MODELS[a.model], a.seq, 1, g)
cl = ops["fwd o_proj (A2AGemm)"][5].__closure__
fwd_default = ops["fwd o_proj (A2AGemm)"][5]
bwd_default = ops["bwd QKV (A2AGemm+wgrad)"][5]
# pull the tensors the bench closures captured
cells = {c: v.cell_contents for c, v in zip(fwd_default.__code__.co_freevars, fwd_default.__closure__)}
o, w_o, res = cells["o"], cells["w_o"], cells["res"]
bcells = {c: v.cell_contents for c, v in zip(bwd_default.__code__.co_freevars, bwd_default.__closure__)}
ctx, dq, dk, dv = bcells["ctx"], bcells["dq"], bcells["dk"], bcells["dv"]
cfg = MODELS[a.model]
s_local = a.seq // n
N = cfg["hidden"]
dgrad_tiles = lambda bm: -(-s_local // bm) * (N // 256)


def out(tag, us):
    if rank == 0:
        print(f"{a.model} n{n} {a.seq // 1024}K {tag:<44}{us:>8.0f}", flush=True)


if a.only != "bwd":
    out("fwd o_proj default", timeit(fwd_default, g))
    y_ref, og_ref = (t.clone() for t in _o_gemm_forward(o, w_o, res, g, "bshd", True))


def fwd_diff(t):
    y, og = _o_gemm_forward(o, w_o, res, g, "bshd", True, tuning=t)
    torch.cuda.synchronize()
    d = torch.tensor([(y.float() - y_ref.float()).abs().max().item(), float(not torch.equal(og, og_ref))])
    d = d.cuda()
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    return f" maxdiff={d[0].item():.3g}{' OG-MISMATCH' if d[1].item() else ''} (|y| {y_ref.abs().max().item():.3g})"


fwd_grid = itertools.product(a.bm, a.push, a.split_m, a.split_k) if a.only != "bwd" else ()
for bm, npush, sm, sk in fwd_grid:
    if sm == 0 and sk != a.split_k[0]:
        continue
    t = A2AGemmTuning(BM=bm, num_push=npush, split_m=sm, split_k=sk)
    tag = f"fwd o_proj BM={bm} push={npush} split_m={sm} split_k={sk if sm else '-'}"
    try:
        us = timeit(lambda t=t: _o_gemm_forward(o, w_o, res, g, "bshd", True, tuning=t), g)
        out(tag + fwd_diff(t), us)
    except Exception as e:
        out(f"{tag} ERR {type(e).__name__}: {e}"[:200], -1)


def bwd_with(t):
    ctx.bwd_tuning = t
    return _qkv_backward(ctx, dq, dk, dv)


ctx.bwd_tuning = None
if a.only != "fwd":
    out("bwd QKV default", timeit(bwd_default, g))
fracs = a.wat if a.wat is not None else (0.0, 1.0)
for bm, npush, frac in itertools.product(a.bm, a.push, fracs) if a.only != "fwd" else ():
    t = A2AGemmTuning(BM=bm, num_push=npush, wgrad_at=int(dgrad_tiles(bm) * frac))
    last = f"{frac:.2f}"
    try:
        out(
            f"bwd QKV BM={bm} push={npush} wgrad_at={last}",
            timeit(lambda t=t: bwd_with(t), g),
        )
    except Exception as e:
        out(f"bwd QKV BM={bm} push={npush} wgrad_at={last} ERR {type(e).__name__}", -1)
dist.barrier()
dist.destroy_process_group()
