"""Job-style gate for r13ns (and r13 alongside) on the job harness COPY (../harness).
usage: gate.py SHAPES(comma) [det]
Per shape and allowed causal mode: o AND lse vs the fp32 reference (refcache for spec shapes, eager otherwise),
full isfinite coverage, SQNR >= 49 dB (gates.check_correctness), for each arm; plus r13ns vs r13 bitwise / max|diff|.
With 'det': 200x bitwise determinism at fast for r13ns (gates.check_determinism)."""
import os, sys, json
from pathlib import Path
L = Path(__file__).resolve().parents[1]; H = L / "harness"
sys.path.insert(0, str(H / "ut")); sys.path.insert(0, str(H))
import torch  # noqa
from common import SHAPES, causal_modes, load_impl, make_inputs  # noqa
from gates import check_correctness, check_determinism, fmt_correctness, GATE_DB  # noqa
ARMS = {"r13": L / "arms" / "fwd_r13", "r13ns": L / "arms" / "fwd_r13ns"}
fns = {a: load_impl(p) for a, p in ARMS.items()}
for a in fns:
    m = sys.modules[fns[a].__module__]
    print("arm", a, m.__file__, "SPEC m32x8", m._kern.SPEC_STALE_MAX, "m32x2", m._kern_m32x2.SPEC_STALE_MAX, flush=True)
ok = True
out = []
for shape in sys.argv[1].split(","):
    for causal in causal_modes(shape):
        for a in ("r13ns", "r13"):
            r = check_correctness(fns[a], shape, causal, GATE_DB)
            print(f"GATE {a:6s}", fmt_correctness(r), flush=True)
            if a == "r13ns":
                ok &= r["ok"]
            out.append(dict(r, arm=a))
        q, k, v = make_inputs(shape, seed=0)
        o1, l1 = fns["r13ns"](q, k, v, causal=causal); o1, l1 = o1.clone(), l1.clone()
        o0, l0 = fns["r13"](q, k, v, causal=causal); torch.cuda.synchronize()
        bit = torch.equal(o1.view(torch.int16), o0.view(torch.int16)) and torch.equal(l1.view(torch.int32), l0.view(torch.int32))
        fin = torch.isfinite(l0) & torch.isfinite(l1)
        print(f"VS13 {shape} causal={causal} bitwise={bit} max|do|={(o1.float()-o0.float()).abs().max().item():.3g} "
              f"max|dlse|={(l1[fin]-l0[fin]).abs().max().item():.3g}", flush=True)
        del q, k, v, o0, o1, l0, l1; torch.cuda.empty_cache()
if len(sys.argv) > 2 and sys.argv[2] == "det":
    dok, detail = check_determinism(fns["r13ns"], "fast", 200)
    print("DET r13ns", detail, "PASS" if dok else "FAIL", flush=True); ok &= dok
print("GATE_RESULT", "PASS" if ok else "FAIL")
