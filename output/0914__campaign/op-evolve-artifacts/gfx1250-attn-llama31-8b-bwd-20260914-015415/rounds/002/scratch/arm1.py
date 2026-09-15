# ONE arm, one process. Three vendored primus_turbo trees in a single process
# segfault (each registers the same torch.library custom op for the forward),
# so arms cannot share a process; they share a session instead, run
# palindromically back to back.  Speed comes from benchmark.measure, the only
# timing loop in this job.
import os, sys
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0, os.path.join(JC, "op")); sys.path.insert(0, os.path.join(JC, "op", "ut"))
import benchmark
from shapes import ALL_SHAPES
lab, d = sys.argv[1], sys.argv[2]
shape = [s for s in ALL_SHAPES if s["name"] == "b4_s8192_hq32_hkv8_d128"][0]
r = benchmark.measure(d, shape)
print(f"TIME {lab:<6} bwd_ms={r['bwd_ms']:.4f} fwd_ms={r['fwd_ms']:.4f} "
      f"tf={r['bwd_tflops']:.1f} spread={r['bwd_spread_pct']:.2f}%", flush=True)
