"""The success criterion for this job. Exit 0 iff the candidate is correct AND fast enough.

    python op/validation.py                 # judges op/current/
    python op/validation.py op/baseline     # judges any directory holding an impl.py

Later modules read NOTHING from this file but its exit code. Everything printed
is for a human.

WHY IT IS BUILT THE WAY IT IS
-----------------------------
This file must not be satisfiable by editing it. So it holds no expected
numbers and makes no assertions about what it was told:

  * Correctness is SQNR against `op/eager/`, recomputed here every run, in
    fp64, over the whole tensor, for out AND dq AND dk AND dv separately. The
    reference is the eager implementation and only the eager implementation --
    never a library, never `op/beat/`, and never the candidate against itself.
  * Speed is a RATIO between two things measured in THIS run, on THIS machine,
    minutes apart in the same process. The bar (`op/beat/`) is re-measured
    every time rather than recalled from a constant, so the file cannot be made
    to pass by changing a number in it: there is no number in it to change. The
    only literals here are the gate threshold and the margin, both of which
    come from the spec, and lowering either is a visible edit to the criterion
    rather than a tweak to a measurement.
  * Every timing comes from `op/benchmark.py`, imported and called. There is no
    second timing loop in this job. If the two disagreed, the report would be
    measuring something the loop is not optimising.

The FLOP and byte counts are `tools/op_flops.py`'s, reached through
`benchmark.py`, which imports it. That file is NOT copied into this job. One
file, one place: a copy is a second source of truth that drifts silently and
makes every TFLOP/s figure in the campaign unfalsifiable.

WHAT IS GATED, AND WHAT IS ONLY REPORTED
----------------------------------------
Gated:
  1. SQNR >= 50 dB (op.precision_sqnr_db) on all four tensors, on the spec
     shape and on every edge shape.
  2. The BACKWARD is at least 1.50x faster than `op/beat/`'s backward on the
     spec shape (op.target.beat, "must be beaten by 50%").

Reported but not gated: the forward, the fwd+bwd total, achieved bandwidth,
the ratio to `op/baseline/`, measurement spread, and the diagnostic shapes.

The speed gate is on the BACKWARD because that is the half this job exists to
move: the op is bf16 attention fwd+bwd, the backward is ~70% of it, and the
job's own name and reference logic are the backward. Gating the fwd+bwd total
instead would quietly change the target -- `op/beat/`'s forward is FASTER than
the baseline's on the spec shape, so a total-based gate would demand the
candidate also win a forward race the spec never asked for. That choice is
recorded here rather than buried: a later round that believes the forward
should be gated too must edit this paragraph, not discover the omission.
"""

import os
import sys

_OP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _OP)
sys.path.insert(0, os.path.join(_OP, "ut"))

import benchmark  # noqa: E402  -- the only timing path in this job
import correctness  # noqa: E402
from shapes import DIAGNOSTIC_SHAPES, GATE_SHAPES, SPEC_SHAPES  # noqa: E402

# From the spec. op.precision_sqnr_db.
SQNR_DB = correctness.SQNR_DB
# From the spec. op.target.beat: "must be beaten by 50%".
MARGIN = 1.50

BEAT = "beat"
BASELINE = "baseline"


def _rule(ch="-"):
    print(ch * 78)


def check_correctness(impl_dir):
    """SQNR vs op/eager/ on every gated shape. Returns (ok, rows)."""
    _rule("=")
    print(f"CORRECTNESS  {impl_dir}  vs op/eager/  (gate: all four tensors >= {SQNR_DB:.0f} dB)")
    _rule()
    print(f"{'shape':<26} {'out':>8} {'dq':>8} {'dk':>8} {'dv':>8}   verdict")
    rows, ok_all = [], True
    for shape in GATE_SHAPES:
        ok, s = correctness.check_shape(impl_dir, shape, threshold=SQNR_DB)
        ok_all &= ok
        rows.append((shape["name"], s, ok, True))
        print(f"{shape['name']:<26} {s['out']:8.2f} {s['dq']:8.2f} {s['dk']:8.2f} {s['dv']:8.2f}   "
              f"{'pass' if ok else 'FAIL'}")
    for shape in DIAGNOSTIC_SHAPES:
        ok, s = correctness.check_shape(impl_dir, shape, threshold=SQNR_DB)
        rows.append((shape["name"], s, ok, False))
        print(f"{shape['name']:<26} {s['out']:8.2f} {s['dq']:8.2f} {s['dk']:8.2f} {s['dv']:8.2f}   "
              f"{'pass' if ok else 'fail'}  (diagnostic, not gated)")
    return ok_all, rows


def measure_speed(impl_dir, iters, warmup_s):
    """Candidate, beat and baseline on the spec shapes, in ONE session, palindromically.

    Palindromic order (cand, beat, base, base, beat, cand) gives all three arms
    the same mean position in the session, so thermal or contention drift over
    the run cannot systematically favour whichever one happened to go first.
    Every arm is measured with the same statistic, the same iteration count and
    the same warmup, because a comparison between two different measurement
    definitions is not a comparison.
    """
    order = [impl_dir, BEAT, BASELINE, BASELINE, BEAT, impl_dir]
    out = {}
    _rule("=")
    print(f"SPEED  statistic={benchmark.STATISTIC}  iters={iters}  warmup_s={warmup_s}  "
          f"order={'>'.join(os.path.basename(o.rstrip('/')) for o in order)}")
    for shape in SPEC_SHAPES:
        per = {}
        for arm in order:
            r = benchmark.measure(arm, shape, iters=iters, warmup_s=warmup_s)
            per.setdefault(arm, []).append(r)
        out[shape["name"]] = per
    return out


def _avg(rs, key):
    return sum(r[key] for r in rs) / len(rs)


def main(argv):
    impl_dir = argv[1] if len(argv) > 1 else os.path.join(_OP, "current")
    if not os.path.isfile(os.path.join(impl_dir, "impl.py")):
        print(f"VALIDATION FAILED: no impl.py under {impl_dir}")
        return 1
    iters = int(os.environ.get("VALIDATION_ITERS", benchmark.DEFAULT_ITERS))
    warmup_s = float(os.environ.get("VALIDATION_WARMUP_S", benchmark.DEFAULT_WARMUP_S))

    correct, _ = check_correctness(impl_dir)
    if not correct:
        # No numbers are printed past this point. A speed figure for an
        # implementation that computes the wrong answer is not a slower or
        # faster result, it is a meaningless one, and printing it invites it to
        # be quoted.
        _rule("=")
        print("VALIDATION FAILED: correctness")
        print(f"  at least one tensor is below {SQNR_DB:.0f} dB against op/eager/. "
              "Speed was not measured.")
        return 1

    speed = measure_speed(impl_dir, iters, warmup_s)

    _rule()
    print(f"{'shape':<26} {'arm':<10} {'fwd ms':>9} {'bwd ms':>9} {'bwd TF/s':>9} "
          f"{'bwd GB/s':>9} {'spread%':>8}")
    fast_all, target_rows = True, []
    for shape_name, per in speed.items():
        cand, beat, base = per[impl_dir], per[BEAT], per[BASELINE]
        for label, rs in (("candidate", cand), ("beat", beat), ("baseline", base)):
            print(f"{shape_name:<26} {label:<10} {_avg(rs,'fwd_ms'):9.4f} {_avg(rs,'bwd_ms'):9.4f} "
                  f"{_avg(rs,'bwd_tflops'):9.2f} {_avg(rs,'bwd_gbps'):9.1f} "
                  f"{max(r['bwd_spread_pct'] for r in rs):8.2f}")
        ratio_beat = _avg(beat, "bwd_ms") / _avg(cand, "bwd_ms")
        ratio_base = _avg(base, "bwd_ms") / _avg(cand, "bwd_ms")
        ok = ratio_beat >= MARGIN
        fast_all &= ok
        target_rows.append((shape_name, ratio_beat, ratio_base, ok))

    _rule()
    print(f"{'shape':<26} {'vs beat':>9} {'required':>9} {'vs baseline':>12}   verdict")
    for name, rb, rbase, ok in target_rows:
        print(f"{name:<26} {rb:8.3f}x {MARGIN:8.2f}x {rbase:11.3f}x   {'pass' if ok else 'FAIL'}")

    _rule("=")
    if correct and fast_all:
        print("VALIDATION PASSED")
        return 0
    print("VALIDATION FAILED: speed")
    for name, rb, _rbase, ok in target_rows:
        if not ok:
            short = (MARGIN - rb) / MARGIN * 100.0
            print(f"  {name}: backward is {rb:.3f}x op/beat/, needs {MARGIN:.2f}x "
                  f"-- short by {short:.1f}% of the target")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
