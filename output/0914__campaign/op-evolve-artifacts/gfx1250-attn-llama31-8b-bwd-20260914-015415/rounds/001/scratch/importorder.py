import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "job_context", "op"))
import benchmark
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context/op"
for d in [sys.argv[1], JC + "/baseline"]:
    try:
        m = benchmark.load_impl(d)
        print("OK   ", d, "->", getattr(m, "__file__", "?"))
    except Exception as e:
        print("FAIL ", d, "->", type(e).__name__, str(e)[:160])
        break
