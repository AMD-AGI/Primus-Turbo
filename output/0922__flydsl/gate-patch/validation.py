#!/usr/bin/env python3
"""The success criterion for this job. Later modules read its EXIT CODE and nothing else.

    python3 validation.py                      # tests op/current/
    python3 validation.py ../rounds/007/op     # tests that round's code

Exit 0 only when every gate below passes. Exit 2 otherwise. There is no partial credit and
no flag that relaxes anything.

    correctness   dq, dk and dv checked SEPARATELY against op/eager/ (the fp32 reference,
                  and the ONLY precision reference), each >= 50 dB, on every shape in
                  op.shape. Output buffers are NaN-prefilled and full `isfinite` coverage
                  is asserted BEFORE SQNR, because a kernel that drops half its lanes
                  reports a fine SQNR over the half it did write.
    determinism   op.config.determinism_gate: dq/dk/dv bitwise identical across 200
                  consecutive runs at the fast-iteration shape. This is the operational
                  form of "atomic-free": a split-k or atomic reduction fails it even when
                  it is faster.
    speed         op.target.beat_margin_pct = 0, op.shape.mode = sweep: the GEOMETRIC MEAN
                  of (candidate TFLOP/s / beat TFLOP/s) across all three shapes must be
                  >= 1.0. beat is measured in the SAME run, never read from a file.

EVERY timing number comes from benchmark.py, invoked as a subprocess. This file contains no
timing loop of its own, and adding one would be a bug: two loops drift apart and the gate
stops describing what the benchmark reports.

It measures. It does not assert what it was told. Editing the candidate cannot satisfy it;
only making the candidate correct, deterministic and fast can.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ut"))

GATE_DB = 50.0          # op.precision_sqnr_db
REFCACHE = HERE / "refcache"  # see load_reference() -- precomputed op/eager/ outputs
DETERMINISM_RUNS = 200  # op.config.determinism_gate
BEAT_MARGIN = 0.0       # op.target.beat_margin_pct, as a ratio floor of 1.0 + 0/100


def measure(impl_dir: Path, shapes):
    """Every timing number in this file comes from here.

    ⚠ impl_dir is handed to benchmark.py as a PATH (`--arm-path LABEL=/abs/dir`), never as
    a name. `rounds/<n>/op/` has basename "op"; turning the directory into a name and
    looking the name up under op/ measures whatever happens to sit at op/op -- on a
    previous job that silently graded the wrong round's code for nine rounds and the gate
    reported nothing. The label below is cosmetic; the path is the identity.

    Because benchmark.py accepts a path directly there is no staging step at all -- and so
    no chance of staging a symlink, whose .resolve() would walk back out of op/ and kill
    every import.
    """
    out = Path(tempfile.mkdtemp(prefix="validation-")) / "bench.json"
    cmd = [sys.executable, str(HERE / "benchmark.py"),
           "--arm-path", f"candidate={impl_dir}",
           "--arms", "beat",
           "--shapes", ",".join(shapes),
           "--json", str(out)]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(HERE))
    if proc.returncode != 0:
        print(f"benchmark.py exited {proc.returncode}")
        return None
    rows = json.loads(out.read_text())
    return {(r["shape"], r["arm"]): r for r in rows}


def load_reference(name, shapes_table, eager_sha, common_sha, torch_mod):
    """The cached op/eager/ reference for `name`, or None to compute it as before.

    WHY THIS EXISTS. The precision gate's own reference is the most reliable way this job has
    found to fault the card. op/eager/impl.py and common.forward_reference are fp32 GEMMs, and
    on gfx1250 in this image they reach a Tensile kernel that intermittently raises
    HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (dispatch workgroup=[256,1,1],
    group_seg_size=6144) or a GCVM_L2 no-retry page fault. It aborted round 3's gate, aborted
    round 8's re-validation, and on 2026-09-22 escalated into an unrecoverable MES state that
    cost a power cycle. None of the kernels under test is implicated: k_dkdv is 32 threads /
    22528 B of LDS, k_dq 32 / 8704, k_delta 256 / 0.

    The reference is deterministic -- make_inputs seeds a torch.Generator and both reference
    functions are fixed fp32 arithmetic -- so it is computed once by
    tools/gfx1250/build_refcache.py and read here. THIS DOES NOT CHANGE THE CRITERION. Same
    reference values, same sqnr_db, same 50 dB gate; only the recomputation is gone.

    A cache whose provenance does not match is REFUSED rather than used. A stale reference
    would weaken the gate silently, which is worse than a gate that fails loudly: it would
    keep reporting dB figures against inputs or an eager implementation that no longer exist.
    """
    path = REFCACHE / f"{name}.pt"
    if not path.exists():
        return None
    blob = torch_mod.load(path, map_location="cpu")
    prov = blob.get("provenance", {})
    want = {"shape": name, "dims": tuple(shapes_table[name]), "seed": 0,
            "eager_sha": eager_sha, "common_sha": common_sha}
    bad = [k for k, v in want.items() if prov.get(k) != v]
    if bad:
        print(f"  refcache {name}: IGNORED, provenance differs on {bad} -- recomputing",
              flush=True)
        return None
    return blob


def check_correctness(impl_dir: Path, shapes):
    import torch
    from common import (forward_reference, load_impl, make_inputs, poison_allocator,
                        sqnr_db)

    impl = load_impl(impl_dir)
    # op/eager/ is the precision reference and the only one. Loaded by path like every
    # other implementation, so it can never collide with the candidate's own module names.
    eager_attn_bwd = load_impl(HERE / "eager")
    ok = True
    import hashlib
    from common import SHAPES as _SHAPES
    _sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    eager_sha, common_sha = _sha(HERE / "eager" / "impl.py"), _sha(HERE / "ut" / "common.py")

    for name in shapes:
        q, k, v, do = make_inputs(name, seed=0)
        blob = load_reference(name, _SHAPES, eager_sha, common_sha, torch)
        if blob is None:
            o, lse = forward_reference(q, k, v, causal=True)
            rq, rk, rv = eager_attn_bwd(do, q, k, v, o, lse, causal=True)
        else:
            # Move the reference to the device one tensor at a time rather than all five at
            # once: at prod dq alone is 537 MB in fp32, and the candidate's own outputs plus
            # the poisoned allocator are already resident.
            o = blob["o"].to(q.device, non_blocking=False)
            lse = blob["lse"].to(q.device, non_blocking=False)
            rq = blob["dq"].to(q.device, non_blocking=False)
            rk = blob["dk"].to(q.device, non_blocking=False)
            rv = blob["dv"].to(q.device, non_blocking=False)
            del blob
        poison_allocator()
        gq, gk, gv = impl(do, q, k, v, o, lse, causal=True)
        row = []
        for tag, got, ref in (("dq", gq, rq), ("dk", gk, rk), ("dv", gv, rv)):
            fin = int(torch.isfinite(got).sum())
            # Coverage BEFORE SQNR. An uncovered buffer makes the dB figure meaningless.
            if fin != got.numel():
                row.append(f"{tag} UNCOVERED {fin}/{got.numel()}")
                ok = False
                continue
            db = sqnr_db(ref, got)
            row.append(f"{tag} {db:6.2f} dB")
            if not (db >= GATE_DB):
                ok = False
        print(f"  correctness {name:8s} " + "  ".join(row), flush=True)
        del q, k, v, do, o, lse, rq, rk, rv, gq, gk, gv
        torch.cuda.empty_cache()
    return ok


def check_determinism(impl_dir: Path, shape="fast"):
    """op.config.determinism_gate, run as stated: 200 consecutive runs, bitwise.

    Bitwise identity is the OBSERVABLE form of "no atomics on any output". A float atomic
    reduction lands in a different order each launch and breaks this within a handful of
    runs; nothing else this job would plausibly introduce does.
    """
    import torch
    from common import forward_reference, load_impl, make_inputs

    impl = load_impl(impl_dir)
    q, k, v, do = make_inputs(shape, seed=0)
    o, lse = forward_reference(q, k, v, causal=True)
    ref = [t.clone() for t in impl(do, q, k, v, o, lse, causal=True)]
    for i in range(1, DETERMINISM_RUNS):
        got = impl(do, q, k, v, o, lse, causal=True)
        for tag, a, b in zip(("dq", "dk", "dv"), ref, got):
            if not torch.equal(a, b):
                print(f"  determinism {shape}: {tag} differs on run {i + 1} of "
                      f"{DETERMINISM_RUNS}  max|d|={float((a - b).abs().max()):.3e}")
                return False
    print(f"  determinism {shape}: dq/dk/dv bitwise identical across "
          f"{DETERMINISM_RUNS} runs", flush=True)
    return True


def main() -> int:
    from common import SPEC_SHAPES

    impl_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (HERE / "current")
    shapes = list(SPEC_SHAPES)
    print(f"validation.py  impl={impl_dir}")
    print(f"  gate {GATE_DB} dB vs op/eager/ | bitwise x{DETERMINISM_RUNS} | "
          f"geomean speedup vs beat >= {1.0 + BEAT_MARGIN / 100:.2f} | shapes {shapes}")
    if not impl_dir.is_dir():
        print(f"FAILED: no such implementation directory: {impl_dir}")
        return 2
    print()

    correct = check_correctness(impl_dir, shapes)
    deterministic = check_determinism(impl_dir) if correct else False
    print()

    rows = measure(impl_dir, shapes) if correct else None
    print()
    fast_enough = False
    ratios = []
    if rows:
        print(f"  {'shape':8s} {'candidate ms':>13s} {'TF/s':>8s} {'beat ms':>10s} "
              f"{'TF/s':>8s} {'x beat':>8s}")
        for name in shapes:
            c, b = rows.get((name, "candidate")), rows.get((name, "beat"))
            if not c or not b:
                print(f"  {name:8s} MISSING ROW")
                ratios = []
                break
            r = c["tflops"] / b["tflops"]
            ratios.append(r)
            print(f"  {name:8s} {c['latency_ms']:13.4f} {c['tflops']:8.1f} "
                  f"{b['latency_ms']:10.4f} {b['tflops']:8.1f} {r:8.3f}")
        if ratios:
            geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            fast_enough = geo >= 1.0 + BEAT_MARGIN / 100
            print(f"  geomean over {len(ratios)} shapes (op.shape.mode=sweep): {geo:.3f}x beat")

    print()
    print(f"  correctness  {'pass' if correct else 'FAIL'}")
    print(f"  determinism  {'pass' if deterministic else 'FAIL'}")
    print(f"  speed        {'pass' if fast_enough else 'FAIL'}")
    ok = correct and deterministic and fast_enough
    print("RESULT:", "SUCCESS" if ok else "FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
