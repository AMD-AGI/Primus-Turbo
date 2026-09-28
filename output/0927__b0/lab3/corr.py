"""FUSED5 correctness vs op/eager (refcache) and vs the champion, ONE shape per process.

usage: corr.py <shape> <det_runs>
  shape: toy256 (CPU fp32 autograd reference, no GPU GEMM) | fast | proxy | prod
Same thresholds as the job's validation.py: coverage first, then dq/dk/dv >= 50 dB vs eager;
determinism: dk/dv bitwise across runs, dq run-to-run >= 70 dB. Extra (h58 prediction):
dk/dv bitwise equal to the champion; dq SQNR vs champion.
Each impl is loaded ONCE per process (benchmark.py r15: a dropped module's GpuJitModule is
unloaded at a later GC pass, possibly mid-launch).
"""
import math, sys, time, hashlib
from pathlib import Path

OP = Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/"
          "gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op")
LAB = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab3")
sys.path.insert(0, str(OP / "ut")); sys.path.insert(0, str(OP))
import torch
import common
from common import load_impl, make_inputs, sqnr_db
from poison_util import poison_allocator

GATE_DB, DQ_STAB_DB = 50.0, 70.0
shape, det_runs = sys.argv[1], int(sys.argv[2])
common.SHAPES["toy256"] = (1, 256, 256, 2, 1, 128)

champ = load_impl(OP / "current")
f5 = load_impl(LAB / "f5")
print(f"shape {shape} {common.SHAPES[shape]}  det_runs {det_runs}", flush=True)

q, k, v, do = make_inputs(shape, seed=0)
if shape == "toy256":
    def ref_fwd_bwd(q, k, v, do):
        qf, kf, vf, dof = (t.detach().cpu().float() for t in (q, k, v, do))
        b, sq, hq, d = qf.shape; skv, hkv = kf.shape[1], kf.shape[2]; g = hq // hkv
        qf.requires_grad_(); kf.requires_grad_(); vf.requires_grad_()
        K = kf.repeat_interleave(g, dim=2); V = vf.repeat_interleave(g, dim=2)
        s = torch.einsum("bqhd,bkhd->bhqk", qf, K) * d ** -0.5
        i = torch.arange(sq)[:, None]; j = torch.arange(skv)[None, :]
        s = s.masked_fill(j > i + (skv - sq), float("-inf"))
        lse = torch.logsumexp(s, dim=-1)
        o = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), V)
        o.backward(dof)
        return (o.detach().to(torch.bfloat16), lse.detach(), qf.grad, kf.grad, vf.grad)
    o, lse, rq, rk, rv = ref_fwd_bwd(q, k, v, do)
    o, lse = o.contiguous().cuda(), lse.contiguous().cuda()
else:
    blob = torch.load(OP / "refcache" / f"{shape}.pt", map_location="cpu")
    prov = blob["provenance"]
    assert prov["shape"] == shape and tuple(prov["dims"]) == tuple(common.SHAPES[shape]) \
        and prov["seed"] == 0, prov
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    assert prov["eager_sha"] == sha(OP / "eager" / "impl.py"), "eager sha mismatch"
    if prov["common_sha"] != sha(OP / "ut" / "common.py"):
        # ut/common.py was edited 2026-09-25 after the cache (09-22); dims/seed/eager match.
        # Control: the champion arm must reproduce its known 52.5-52.8 dB against this cache.
        print("  NOTE refcache common_sha differs (common.py edited 09-25); champion arm is the control")
    o, lse = blob["o"].to(q.device), blob["lse"].to(q.device)
    rq, rk, rv = blob["dq"], blob["dk"], blob["dv"]
    del blob

ok = True
outs = {}
for name, fn in (("champ", champ), ("f5", f5)):
    poison_allocator()
    torch.cuda.synchronize(); t0 = time.time()
    got = fn(do, q, k, v, o, lse, causal=True)
    torch.cuda.synchronize()
    row = []
    for tag, g_, r_ in zip(("dq", "dk", "dv"), got, (rq, rk, rv)):
        fin = int(torch.isfinite(g_).sum())
        if fin != g_.numel():
            row.append(f"{tag} UNCOVERED {fin}/{g_.numel()}")
            if name == "f5": ok = False
            continue
        rd = r_.to(g_.device)
        db = sqnr_db(rd, g_); del rd
        row.append(f"{tag} {db:6.2f} dB")
        if name == "f5" and not db >= GATE_DB: ok = False
    print(f"  correctness {name:5s} vs eager: " + "  ".join(row) + f"   ({time.time()-t0:.2f}s)",
          flush=True)
    outs[name] = [t.clone() for t in got]
    del got
del rq, rk, rv

c, f = outs["champ"], outs["f5"]
for tag, a, b in zip(("dk", "dv"), c[1:], f[1:]):
    eq = torch.equal(a, b)
    print(f"  f5 vs champ {tag}: {'BITWISE EQUAL' if eq else 'DIFFERS max|d|=%.3e sqnr=%.2f dB' % (float((a.float()-b.float()).abs().max()), sqnr_db(a, b))}")
print(f"  f5 vs champ dq: sqnr {sqnr_db(c[0], f[0]):.2f} dB  max|d|={float((c[0].float()-f[0].float()).abs().max()):.3e}")
del outs, c

def _sq(ref, got):
    ref = ref.float(); den = float(((ref - got.float()) ** 2).sum())
    return float("inf") if den == 0.0 else 10.0 * math.log10(float((ref ** 2).sum()) / den)

det_ok = True
if det_runs > 1:
    ref = f
    worst = float("inf")
    for i in range(1, det_runs):
        got = f5(do, q, k, v, o, lse, causal=True)
        for tag, a, b in zip(("dk", "dv"), ref[1:], got[1:]):
            if not torch.equal(a, b):
                print(f"  determinism: {tag} differs on run {i+1}"); det_ok = False
        db = _sq(ref[0], got[0]); worst = min(worst, db)
        if db < DQ_STAB_DB:
            print(f"  determinism: dq run-to-run {db:.1f} dB < {DQ_STAB_DB} on run {i+1}"); det_ok = False
        del got
        if not det_ok: break
    torch.cuda.synchronize()
    print(f"  determinism {shape}: {det_runs} runs, dk/dv {'bitwise' if det_ok else 'FAIL'}; "
          f"dq worst run-to-run {worst:.1f} dB (floor {DQ_STAB_DB})")
print(f"VERDICT correctness={'pass' if ok else 'FAIL'} determinism={'pass' if det_ok else 'FAIL'}")
