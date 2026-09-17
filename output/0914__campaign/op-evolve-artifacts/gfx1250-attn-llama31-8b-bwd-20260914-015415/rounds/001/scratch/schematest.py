import sys, gc, os
OP = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context/op"
sys.path.insert(0, OP); sys.path.insert(0, OP + "/ut")
import torch, benchmark, correctness
from shapes import GATE_SHAPES, DIAGNOSTIC_SHAPES
ARM = sys.argv[1]

def probe(tag):
    try:
        m = sys.modules.get("_probe_mod")
        ok = torch._C._dispatch_has_kernel("primus_turbo::attention_triton_forward_impl")
    except Exception as e:
        ok = "ERR " + str(e)[:60]
    print(f"[probe] {tag}: has_kernel={ok}", flush=True)

probe("start")
m = benchmark.load_impl(ARM); probe("after candidate load")
gc.collect(); probe("after gc")
correctness._install_eager(); gc.collect(); probe("after eager+gc")
for sh in GATE_SHAPES + DIAGNOSTIC_SHAPES:
    correctness.check_shape(ARM, sh)
gc.collect(); probe("after correctness+gc")
for i in range(5):
    gc.collect()
probe("after 5x gc")
