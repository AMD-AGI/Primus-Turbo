import sys, os
OP = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context/op"
sys.path.insert(0, OP); sys.path.insert(0, OP + "/ut")
import benchmark, correctness
from shapes import GATE_SHAPES, DIAGNOSTIC_SHAPES, SPEC_SHAPES
ARM = sys.argv[1]
mode = sys.argv[2]
if mode == "full":
    for sh in GATE_SHAPES + DIAGNOSTIC_SHAPES:
        ok, s_ = correctness.check_shape(ARM, sh)
        print("corr", sh["name"], ok, flush=True)
if mode in ("corr", "both"):
    for sh in GATE_SHAPES:
        ok, s = correctness.check_shape(ARM, sh)
        print("corr", sh["name"], ok, flush=True)
if mode in ("bench", "both", "full"):
    try:
        r = benchmark.measure(ARM, SPEC_SHAPES[0], iters=5, warmup_s=0.5)
        print("bench OK bwd_ms=%.4f" % r["bwd_ms"], flush=True)
    except Exception as e:
        print("bench FAIL", type(e).__name__, e, flush=True)
