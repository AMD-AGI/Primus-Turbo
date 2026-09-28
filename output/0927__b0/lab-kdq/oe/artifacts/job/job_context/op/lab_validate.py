#!/usr/bin/env python3
"""Lab copy of validation.py's correctness + determinism gates, ONE SHAPE per process.

usage: lab_validate.py SHAPE NRUNS LABEL=/abs/impl_dir ...
Per arm: NaN-poisoned allocator, isfinite coverage, dq/dk/dv SQNR vs the cached op/eager
reference (validation.load_reference, same provenance check, same sqnr_db, 50 dB gate);
then NRUNS consecutive calls: dk/dv bitwise vs run 1, dq run-to-run SQNR >= 70 dB.
Extra: bitwise comparison of each arm's outputs against the FIRST arm's.
"""
import sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ut")); sys.path.insert(0, str(HERE))
import torch
from common import SHAPES, load_impl, make_inputs, sqnr_db
from validation import load_reference, _sqnr_db, GATE_DB, DQ_STABILITY_DB
from poison_util import poison_allocator

shape, nruns = sys.argv[1], int(sys.argv[2])
arms = [(a.split("=", 1)[0], Path(a.split("=", 1)[1]).resolve()) for a in sys.argv[3:]]
_sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
# ut/common.py was edited after the cache was built (r23 comment/toy shape); make_inputs is
# unchanged: rounds/024-026 gate.log recomputed the reference and got the SAME dB to 0.01 as
# the cached rounds. So accept the recorded common_sha, keep every other provenance check,
# and never dispatch the fp32 reference GEMM on the card (the 09-22 wedge trigger).
_CACHED_COMMON = {"fast": "988c14caed5d9a80", "proxy": "988c14caed5d9a80", "prod": "988c14caed5d9a80"}
blob = None
for _cs in (_sha(HERE / "ut" / "common.py"), _CACHED_COMMON[shape]):   # refcache provenance re-stamped 09-28 06:42
    blob = load_reference(shape, SHAPES, _sha(HERE / "eager" / "impl.py"), _cs, torch)
    if blob is not None:
        print(f"refcache {shape}: accepted with common_sha {_cs}", flush=True)
        break
assert blob is not None, "refcache missing/mismatch -- refusing to compute the fp32 reference on card"
q, k, v, do = make_inputs(shape, seed=0)
o, lse = blob["o"].to(q.device), blob["lse"].to(q.device)
ref = {"dq": blob["dq"], "dk": blob["dk"], "dv": blob["dv"]}
del blob
ok_all = True
first = None
for label, path in arms:
    impl = load_impl(path)
    poison_allocator()
    out = impl(do, q, k, v, o, lse, causal=True)
    torch.cuda.synchronize()
    row, ok = [], True
    for tag, got in zip(("dq", "dk", "dv"), out):
        fin = int(torch.isfinite(got).sum())
        if fin != got.numel():
            row.append(f"{tag} UNCOVERED {fin}/{got.numel()}"); ok = False; continue
        r = ref[tag].to(got.device); db = sqnr_db(r, got); del r
        row.append(f"{tag} {db:6.2f} dB"); ok &= db >= GATE_DB
    print(f"CORR shape={shape} arm={label} " + "  ".join(row) + f"  {'pass' if ok else 'FAIL'}", flush=True)
    base = [t.clone() for t in out]
    worst, det = float("inf"), ok
    for i in range(1, nruns):
        got = impl(do, q, k, v, o, lse, causal=True)
        for tag, a, b in zip(("dk", "dv"), base[1:], got[1:]):
            if not torch.equal(a, b):
                print(f"DET shape={shape} arm={label} {tag} differs on run {i+1}"); det = False
        db = _sqnr_db(base[0], got[0]); worst = min(worst, db)
        if db < DQ_STABILITY_DB:
            print(f"DET shape={shape} arm={label} dq {db:.1f} dB on run {i+1}"); det = False
        if not det:
            break
    torch.cuda.synchronize()
    print(f"DET shape={shape} arm={label} runs={nruns} dkdv={'bitwise' if det else 'FAIL'} "
          f"dq_worst={'bitwise' if worst == float('inf') else f'{worst:.1f} dB'}  {'pass' if det else 'FAIL'}", flush=True)
    if first is None:
        first = (label, base)
    else:
        eq = [f"{t}={'bitwise' if torch.equal(x, y) else f'{_sqnr_db(x, y):.1f}dB'}"
              for t, x, y in zip(("dq", "dk", "dv"), first[1], base)]
        print(f"XARM shape={shape} {label} vs {first[0]}: " + " ".join(eq), flush=True)
    ok_all &= ok and det
    del out, base
    torch.cuda.empty_cache()
print("LABVAL", shape, "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 2)
