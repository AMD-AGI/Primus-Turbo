"""M8 PMC driver: per arm, a marker fill whose grid encodes the arm index, then LEAD untimed + N calls of that arm,
one sync per arm. Kernels are identified in the rocprofv3 csv by dispatch order after their marker.
Usage: m8drive.py <shape> <N> <label=dir[:nc]>...   (':nc' = non-causal call)"""
import os, sys
H = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec/harness"
sys.path.insert(0, H + "/ut"); sys.path.insert(0, H)
import torch  # noqa: E402
from common import load_impl, make_inputs  # noqa: E402
shape, N = sys.argv[1], int(sys.argv[2])
arms = []
for spec in sys.argv[3:]:
    lab, path = spec.split("=", 1)
    nc = path.endswith(":nc"); path = path[:-3] if nc else path
    arms.append((lab, load_impl(path), not nc))
q, k, v = make_inputs(shape, seed=0)
LEAD = 3
for i, (lab, fn, causal) in enumerate(arms):
    fn(q, k, v, causal=causal)               # build/JIT outside the counted block
torch.cuda.synchronize()
for i, (lab, fn, causal) in enumerate(arms):
    torch.full(((i + 1) * 65536,), float(i), device="cuda")   # marker: grid = (i+1)*65536/elements-per-thread
    torch.cuda.synchronize()
    for _ in range(LEAD + N):
        fn(q, k, v, causal=causal)
    torch.cuda.synchronize()
    print(f"ARM {i} {lab} causal={causal} calls={LEAD + N}", flush=True)
print("M8DONE", flush=True)
