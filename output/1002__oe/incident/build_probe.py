#!/usr/bin/env python3
"""g79 Gate-2-lite: what does w4f actually allocate per wave?

Gate 1 (host arithmetic) left one question open: the F=4 decomposition (16 waves,
4 waves/SIMD) only fits if w4f's real per-wave VGPR allocation is <= 512. Nothing in
the pool knows that number. Dump the ISA of the shipped w4f arm and read .vgpr_count.
One prod call, no timing, no ranking.
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--impl", required=True)
ap.add_argument("--dump-dir", required=True)
ap.add_argument("--shape", default="prod")
a = ap.parse_args()
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = a.dump_dir
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
os.makedirs(a.dump_dir, exist_ok=True)
HERE = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op"
sys.path.insert(0, HERE + "/ut"); sys.path.insert(0, HERE)
import torch
from common import load_impl, make_inputs
from refcache_util import cached_forward
fn = load_impl(a.impl)
q, k, v, do = make_inputs(a.shape, seed=0)
o, lse = cached_forward(a.shape, q, k, v, causal=True)
fn(do, q, k, v, o, lse, causal=True)
torch.cuda.synchronize()
print("DUMP_OK")
