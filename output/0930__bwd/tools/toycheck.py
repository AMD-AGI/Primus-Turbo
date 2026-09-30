"""First-launch check for risky arms at a small shape, one shape per process.
usage: toycheck.py SHAPE LABEL=/abs/arm ...   (SHAPE from ut/common.py SHAPES, e.g. toy / gqa4_small / fast)
o and lse come from a CPU fp32 forward (tiny shapes only; no fp32 GEMM on the card).
Checks: outputs finite and fully written (NaN-poisoned via empty + fill), dq/dk/dv bitwise vs the FIRST arm,
and SQNR vs a CPU fp32 reference backward."""
import sys, math
from pathlib import Path
HERE = Path.cwd(); sys.path.insert(0, str(HERE / "ut")); sys.path.insert(0, str(HERE))
import torch
from common import SHAPES, load_impl, make_inputs
shape = sys.argv[1]
arms = [(a.split("=", 1)[0], Path(a.split("=", 1)[1]).resolve()) for a in sys.argv[2:]]
b, sq, skv, hq, hkv, d = SHAPES[shape]
q, k, v, do = make_inputs(shape, seed=0)
g = hq // hkv; scale = d ** -0.5
qc, kc, vc, doc = (t.float().cpu().requires_grad_(True) for t in (q, k, v, do))
kr = kc.repeat_interleave(g, dim=2); vr = vc.repeat_interleave(g, dim=2)
s = torch.einsum("bqhd,bkhd->bhqk", qc, kr) * scale
i = torch.arange(sq)[:, None]; j = torch.arange(skv)[None, :]
s = s.masked_fill(j > i + (skv - sq), float("-inf"))
lse_c = torch.logsumexp(s, dim=-1)
o_c = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vr)
o_c.backward(doc.float())
ref = {"dq": qc.grad, "dk": kc.grad, "dv": vc.grad}
o = o_c.detach().to(torch.bfloat16).contiguous().cuda(); lse = lse_c.detach().float().contiguous().cuda()
def sq_db(r, x):
    r = r.float().cpu(); x = x.float().cpu(); return 10 * math.log10((r * r).sum() / ((r - x) ** 2).sum().clamp_min(1e-30))
first = None; ok = True
for label, path in arms:
    fn = load_impl(path)
    torch.cuda.synchronize()
    outs = fn(do, q, k, v, o, lse, causal=True); torch.cuda.synchronize()
    outs = [x.detach().clone() for x in outs]
    row = []
    for tag, x in zip(("dq", "dk", "dv"), outs):
        fin = bool(torch.isfinite(x).all())
        row.append(f"{tag} {sq_db(ref[tag], x):6.2f}dB fin={fin}")
        ok &= fin
    if first is None: first = outs; bit = "ref"
    else: bit = " ".join(f"{t}={'bitwise' if torch.equal(a, c) else 'DIFF'}" for t, a, c in zip(("dq", "dk", "dv"), first, outs))
    print(f"TOY shape={shape} arm={label} {' '.join(row)} vs_first: {bit}", flush=True)
print("TOYCHECK", "PASS" if ok else "FAIL")
