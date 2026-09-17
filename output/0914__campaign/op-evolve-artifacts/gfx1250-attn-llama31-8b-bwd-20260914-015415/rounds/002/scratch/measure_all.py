import os, sys
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0, os.path.join(JC, "op")); sys.path.insert(0, os.path.join(JC, "op", "ut"))
import benchmark
from shapes import ALL_SHAPES
lab, d = sys.argv[1], sys.argv[2]
for s in ALL_SHAPES:
    r = benchmark.measure(d, s)
    print(f"{lab:<10} {s['name']:<26} fwd={r['fwd_ms']:9.4f} bwd={r['bwd_ms']:9.4f} "
          f"tf={r['bwd_tflops']:8.2f} gbs={r['bwd_gbps']:7.1f} spread={r['bwd_spread_pct']:5.2f}%", flush=True)
