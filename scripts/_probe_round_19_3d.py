#!/usr/bin/env python3
"""R19 — confirm 3D positional torch.empty also wins."""
import os, sys, time
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch
import primus_turbo.pytorch as turbo  # noqa: F401

def _bench_py(fn, iters=20000):
    fn()
    deltas = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        deltas.append(time.perf_counter_ns() - t0)
    deltas.sort()
    return deltas[len(deltas) // 5] / 1000.0

# Var-K output: (group_num, n, k) bf16 cuda — Down-B4 wgrad: (4, 2880, 2880)
G, N, K = 4, 2880, 2880
out_dtype = torch.bfloat16
dev = torch.device("cuda:0")

def a_tuple_3d():
    return torch.empty((G, N, K), dtype=out_dtype, device=dev)
print(f"  [A] torch.empty((G,N,K), ...):           {_bench_py(a_tuple_3d):.3f} µs")

def b_pos_3d():
    return torch.empty(G, N, K, dtype=out_dtype, device=dev)
print(f"  [B] torch.empty(G, N, K, ...):           {_bench_py(b_pos_3d):.3f} µs")

# Sanity
t1 = a_tuple_3d()
t2 = b_pos_3d()
print(f"  shapes: tuple={tuple(t1.shape)} pos={tuple(t2.shape)}  match={t1.shape == t2.shape}")
