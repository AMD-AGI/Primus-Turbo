"""Run bwd_kernel_causal in a tight loop for N seconds so a power/clock sampler
has something to look at. No timing claim is made here."""
import os, sys, time
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0, os.path.join(JC, "op")); sys.path.insert(0, os.path.join(JC, "op", "ut"))
import torch, benchmark, statistics
from shapes import ALL_SHAPES
shape = [s for s in ALL_SHAPES if s["name"] == "b4_s8192_hq32_hkv8_d128"][0]
mod = benchmark.load_impl(os.path.join(JC, "op", "current"))
q, k, v, do = benchmark.make_inputs(shape)
out = mod.attention(q, k, v, causal=True)
def bwd(): torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
bwd(); torch.cuda.synchronize()
print("LOADSTART", flush=True)
t0 = time.time(); n = 0
while time.time() - t0 < float(os.environ.get("SECS", "60")):
    for _ in range(20): bwd()
    torch.cuda.synchronize(); n += 20
dt = time.time() - t0
print(f"LOADDONE iters={n} mean_ms={dt/n*1e3:.4f}", flush=True)
