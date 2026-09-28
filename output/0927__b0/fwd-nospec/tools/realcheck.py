"""Real training inputs (6 dumps, prod shape): r13ns vs r13 (bitwise / SQNR), both vs ASM, and all three vs an
fp32 reference on sampled heads (batch 0, q heads 0,13,31; full causal rows). One process, no timing."""
import sys, glob
from pathlib import Path
L = Path(__file__).resolve().parents[1]; H = L / "harness"
sys.path.insert(0, str(H / "ut")); sys.path.insert(0, str(H))
import torch  # noqa
from common import load_impl, sqnr_db  # noqa
fns = {"r13": load_impl(L / "arms/fwd_r13"), "r13ns": load_impl(L / "arms/fwd_r13ns"), "asm": load_impl(H / "beat")}
HS = (0, 13, 31)
ok = True
for f in sorted(glob.glob("/home/lihuzhan/_prof_dump/qkv_call*.pt")):
    t = torch.load(f); name = f"L{t['call'] % 32:02d}"
    q, k, v = (t[n].cuda().contiguous() for n in ("q", "k", "v")); del t
    b, s, hq, d = q.shape; hkv = k.shape[2]; g = hq // hkv; sc = d ** -0.5
    outs = {}
    for a, fn in fns.items():
        o, l = fn(q, k, v, softmax_scale=sc, causal=True); torch.cuda.synchronize(); outs[a] = (o.clone(), l.clone())
    (o1, l1), (o0, l0) = outs["r13ns"], outs["r13"]
    bit = torch.equal(o1.view(torch.int16), o0.view(torch.int16)) and torch.equal(l1.view(torch.int32), l0.view(torch.int32))
    fin = all(bool(torch.isfinite(x).all()) for x in (o1, l1))
    line = (f"REAL {name} finite={fin} ns_vs_r13 bitwise={bit} sqnr={sqnr_db(o0, o1):.1f}dB max|do|={(o1.float()-o0.float()).abs().max().item():.3g} "
            f"max|dlse|={(l1-l0).abs().max().item():.3g} | vs_asm o: r13 {sqnr_db(outs['asm'][0], o0):.1f} r13ns {sqnr_db(outs['asm'][0], o1):.1f} dB")
    # fp32 reference, sampled heads
    ref = []
    for h in HS:
        qh = q[0, :, h].float(); kh = k[0, :, h // g].float(); vh = v[0, :, h // g].float()
        S = (qh @ kh.t()) * sc
        S.masked_fill_(torch.ones(s, s, device=q.device, dtype=torch.bool).triu(1), float("-inf"))
        lse = torch.logsumexp(S, -1); P = torch.exp(S - lse[:, None]); del S
        ref.append((P @ vh, lse)); del P
    for a in fns:
        o, l = outs[a]
        dbo = sqnr_db(torch.stack([r[0] for r in ref]), torch.stack([o[0, :, h].float() for h in HS]))
        dbl = sqnr_db(torch.stack([r[1] for r in ref]), torch.stack([l[0, h] for h in HS]))
        line += f" | ref[{a}] o {dbo:.1f} lse {dbl:.1f}"
        if a == "r13ns":
            ok &= fin and dbo >= 49
    print(line, flush=True)
    del q, k, v, outs, ref; torch.cuda.empty_cache()
print("REAL_RESULT", "PASS" if ok else "FAIL")
