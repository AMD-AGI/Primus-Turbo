#!/usr/bin/env python3
"""ONE shape per process: cand correctness vs job reference (NaN-poisoned outputs, gate 49 dB),
cand vs champion (bitwise / max abs / SQNR), causal and non-causal, determinism (N runs bitwise).
Reads the job dir only.     corr_one.py <cand_dir> <shape> <out.json> [det_runs]
"""
import json, os, sys, time
from pathlib import Path

OP = Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/"
          "gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op")
CHAMP = OP / "current"
sys.path.insert(0, str(OP)); sys.path.insert(0, str(OP / "ut"))
import benchmark as B                                  # noqa: E402  (sets BLAS env, torch)
import torch                                           # noqa: E402
from common import load_impl, make_inputs, sqnr_db     # noqa: E402
from gates import check_correctness, check_determinism  # noqa: E402

cand_dir, shape, out = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
det_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 50
fns = {"cand": load_impl(cand_dir), "champ": load_impl(CHAMP)}
wit = {l: B.witness(f) for l, f in fns.items()}
for l in fns:
    print(f"# arm {l} {wit[l]}", flush=True)
print(f"# device {torch.cuda.get_device_properties(0).gcnArchName} OE_PHYS_GPU={os.environ.get('OE_PHYS_GPU')} "
      f"AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL')}", flush=True)
res = {"shape": shape, "cand": str(cand_dir), "witness": wit}
t0 = time.time()
# 1) cand alone first (a fault is attributable to the new kernel)
r = check_correctness(fns["cand"], shape, True, gate_db=49.0)
res["corr_cand_causal"] = r
print(f"# corr cand causal: o {r['o']['db']:.2f} dB lse {r['lse']['db']:.2f} dB finite o "
      f"{r['o']['finite']}/{r['o']['numel']} lse {r['lse']['finite']}/{r['lse']['numel']} ref={r['ref']} ok={r['ok']}",
      flush=True)
torch.cuda.empty_cache()
# 2) cand vs champ, causal and non-causal
q, k, v = make_inputs(shape, seed=0)
for causal in (True, False):
    oc, lc = fns["champ"](q, k, v, causal=causal); oc, lc = oc.clone(), lc.clone()
    ox, lx = fns["cand"](q, k, v, causal=causal)
    torch.cuda.synchronize()
    cmp = {
        "o_bitwise_equal": bool(torch.equal(ox.view(torch.int16), oc.view(torch.int16))),
        "lse_bitwise_equal": bool(torch.equal(lx.view(torch.int32), lc.view(torch.int32))),
        "o_max_abs_diff": float((ox.float() - oc.float()).abs().max()),
        "lse_max_abs_diff": float((lx.float() - lc.float()).abs().max()),
        "o_sqnr_vs_champ_db": float(sqnr_db(oc, ox)),
        "o_finite": int(torch.isfinite(ox).sum()), "o_numel": ox.numel(),
    }
    res[f"vs_champ_{'causal' if causal else 'noncausal'}"] = cmp
    print(f"# cand-vs-champ causal={causal} {cmp}", flush=True)
    del oc, lc, ox, lx
del q, k, v
torch.cuda.empty_cache()
# 3) determinism (cand)
ok, detail = check_determinism(fns["cand"], shape, runs=det_runs, causal=True)
res["determinism"] = {"ok": ok, "detail": detail}
print(f"# determinism {ok} {detail}", flush=True)
res["pass"] = bool(res["corr_cand_causal"]["ok"] and ok)
res["secs"] = time.time() - t0
print(f"CORR {shape} pass={res['pass']} secs={res['secs']:.0f}", flush=True)
out.write_text(json.dumps(res, indent=2, default=str))
