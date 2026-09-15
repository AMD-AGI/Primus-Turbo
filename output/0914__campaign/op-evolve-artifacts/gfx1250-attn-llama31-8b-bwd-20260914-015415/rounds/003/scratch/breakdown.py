"""Where do the 8.69 ms of the backward actually go, dispatch by dispatch?

Nobody in this job has taken this. facts.md talks only about bwd_kernel_causal.
The backward call also runs _bwd_preprocess, an LSE gather+contiguous+float, and
three empty_like allocations. If any of those is a material slice of the 8.69 ms
it is a cheaper 10% than anything inside the loop body.

rocprofv3 records zero dispatches on this benchmark (dead_ends.md); torch.profiler
works (round 1 used it).
"""
import os, sys, statistics
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0, os.path.join(JC, "op")); sys.path.insert(0, os.path.join(JC, "op", "ut"))
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
import torch, benchmark
from shapes import ALL_SHAPES
from torch.profiler import profile, ProfilerActivity

shape = [s for s in ALL_SHAPES if s["name"] == "b4_s8192_hq32_hkv8_d128"][0]
mod = benchmark.load_impl(os.environ.get("OPDIR", os.path.join(JC, "op", "current")))
q, k, v, do = benchmark.make_inputs(shape)
out = mod.attention(q, k, v, causal=True)
def bwd(): torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
benchmark._warm(bwd, 3.0)
t = benchmark._time_continuous(bwd, 30)
print(f"WALL bwd median {statistics.median(t):.4f} ms  min {min(t):.4f}", flush=True)

N = 20
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(N):
        bwd()
    torch.cuda.synchronize()
ev = {}
for e in prof.key_averages():
    d = getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0)
    if d > 0:
        ev[e.key] = ev.get(e.key, 0.0) + d
tot = sum(ev.values())
print(f"GPU total over {N} iters: {tot/1000:.3f} ms  -> {tot/1000/N:.4f} ms/iter", flush=True)
for k_, v_ in sorted(ev.items(), key=lambda x: -x[1]):
    print(f"  {v_/1000/N:9.4f} ms/iter  {100*v_/tot:6.2f}%  {k_[:100]}", flush=True)
