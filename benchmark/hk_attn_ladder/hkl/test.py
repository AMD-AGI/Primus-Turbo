###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Correctness gate of a ladder: every rung (and baseline) against the fp32 reference, on a GPU.

    python3 -m hkl.test llama/ladder_fwd.json                        # every rung + baseline, every shape
    python3 -m hkl.test llama/ladder_bwd.json --rung v0 p1 --shapes toy rect
    python3 -m hkl.test llama/ladder_fwd.json --rung f00 --shapes toy --json out.json

Shapes run in order, smallest first, and every arm finishes a shape before any arm starts the next:
a broken new rung shows up on the toy shape, before it ever runs at the production size. With
``--stop-on-fail`` (default on) the first failing shape ends the run.

Per (arm, shape): SNR in dB of every output against the fp32 reference (fwd: o, lse; bwd: dq, dk, dv),
all finite (the allocator is NaN-poisoned before every call, so an unwritten output element is NaN),
determinism (a second call: the ladder's ``exact_outputs`` bitwise -- default fwd o / lse, bwd dk / dv --
the others to >= 90 dB, e.g. a dQ summed with atomics), and, where the ladder file says
``"bitwise_with": <arm>``, bitwise equality of the exact outputs with that arm's on the same inputs. PASS = finite, every SNR >= --min-snr, determinism, bitwise.
"""

import argparse
import json
import sys
import time

import torch

from . import ref as R
from . import runner

SHAPES = {
    "gqa": [  # name, overrides of the ladder's problem
        ("toy", dict(b=1, sq=256, skv=256, hq=8, hkv=2)),
        ("toy_full", dict(b=1, sq=256, skv=256, hq=8, hkv=2, causal=False)),
        ("rect", dict(b=1, sq=512, skv=1024, hq=8, hkv=2)),
        ("small", dict(b=2, sq=1024, skv=1024, hq=32, hkv=8)),
        ("prod", dict()),
    ],
    "mla": [
        ("toy", dict(b=1, sq=384, skv=384, hq=16, hkv=16)),
        ("toy_full", dict(b=1, sq=384, skv=384, hq=16, hkv=16, causal=False)),  # not causal: every tile full
        ("s640", dict(b=1, sq=640, skv=640, hq=16, hkv=16)),
        ("b1_s4096", dict(b=1, sq=4096, skv=4096, hq=16, hkv=16)),
        # causal with (Skv - Sq) % 64 == 32: the steady-loop region-C read of the dQ kernel from 5c20de7e (r2d) on
        # is suspected to take the wrong half-block for row 2 nf here (CPU replay of its idx.h); FlyDSL skips it
        ("rect32", dict(b=1, sq=4096, skv=4128, hq=4, hkv=4)),
        # the smallest persistent shapes of the forward's ticket schedule (512 blocks = 2 per CU at 256 CUs)
        ("b1_h32", dict(b=1, sq=4096, skv=4096, hq=32, hkv=32)),
        ("prod", dict()),
        ("prod_b4", dict(b=4)),
    ],
}


def load_arms(ladder, rungs, baselines):
    """Baselines first (a rung may be required to equal one bitwise), then the rungs in ladder order."""
    arms = [runner.load_baseline(ladder, bid) for bid in baselines]
    for rid in rungs:
        arms.append(runner.load(ladder, rid))
    return arms


def call(arm, direction, inp, sh):
    if direction == "fwd":
        o, lse = arm.fwd(inp["q"], inp["k"], inp["v"], causal=sh.causal)
        return {"o": o, "lse": lse}
    dq, dk, dv = arm.bwd(inp["do"], inp["q"], inp["k"], inp["v"], inp["o"], inp["lse"], causal=sh.causal)
    return {"dq": dq, "dk": dk, "dv": dv}


def run_shape(arms, direction, sh, name, a, results, exact):
    dev = torch.device("cuda")
    inp = R.make_inputs(sh, dev, seed=a.seed)
    t0 = time.time()
    if direction == "fwd":
        o_ref, lse_ref = R.ref_fwd(sh, inp["q"], inp["k"], inp["v"])
        refs = {"o": o_ref, "lse": lse_ref}
    else:
        o_ref, lse_ref = R.ref_fwd(sh, inp["q"], inp["k"], inp["v"])
        inp["o"], inp["lse"] = o_ref.to(torch.bfloat16), lse_ref
        dq, dk, dv = R.ref_bwd(sh, inp["q"], inp["k"], inp["v"], inp["do"])
        refs = {"dq": dq, "dk": dk, "dv": dv}
    print(f"[{name}] {sh}  (reference {time.time() - t0:.1f} s)", flush=True)
    outs, ok_all = {}, True
    for arm in arms:
        rec = {"arm": arm.id, "shape": name, "desc": repr(sh)}
        try:
            R.poison_allocator(dev)
            out1 = call(arm, direction, inp, sh)
            torch.cuda.synchronize()
            R.poison_allocator(dev)
            out2 = call(arm, direction, inp, sh)
            torch.cuda.synchronize()
        except (NotImplementedError, ValueError) as e:  # the arm's host gate: not this arm's problem range
            rec.update(status="skip", reason=f"{type(e).__name__}: {e}")
            print(f"  {arm.id:<16} skip  {rec['reason'][:110]}")
            results.append(rec)
            continue
        snr = {k: R.snr_db(v, refs[k]) for k, v in out1.items()}
        finite = all(torch.isfinite(v.float()).all().item() for v in out1.values())
        det = {}  # exact outputs bitwise across calls; the others (e.g. an atomically summed dQ) to >= 90 dB
        for k in out1:
            det[k] = torch.equal(out1[k], out2[k]) if k in exact else R.snr_db(out2[k], out1[k]) >= 90.0
        bw = None
        want = getattr(arm, "rung", {}).get("bitwise_with") if hasattr(arm, "rung") else None
        if want is not None and want in outs:
            bw = all(torch.equal(out1[k], outs[want][k]) for k in out1 if k in exact)
        ok = finite and all(v >= a.min_snr for v in snr.values()) and all(det.values()) and bw is not False
        outs[arm.id] = out1
        rec.update(status="PASS" if ok else "FAIL", snr={k: round(v, 2) for k, v in snr.items()}, finite=finite,
                   det=det, bitwise_with=want, bitwise=bw)  # fmt: skip
        results.append(rec)
        s = " ".join(f"{k} {v:6.2f}" for k, v in snr.items())
        d = "det" if all(det.values()) else "NONDET " + ",".join(k for k, v in det.items() if not v)
        b = "" if want is None else f" bitwise({want})={'yes' if bw else ('no' if bw is False else 'n/a')}"
        print(f"  {arm.id:<16} {rec['status']}  {s}  {d}{'' if finite else ' NONFINITE'}{b}", flush=True)
        ok_all &= ok
    return ok_all


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladder")
    ap.add_argument("--rung", nargs="*", help="rung ids (default: every rung)")
    ap.add_argument("--baseline", nargs="*", help="baseline ids (default: every baseline; pass none for none)")
    ap.add_argument("--shapes", nargs="*", help="shape names in order (default: all of the problem kind)")
    ap.add_argument("--min-snr", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-going", action="store_true", help="continue with larger shapes after a failure")
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args(argv)

    spec = runner.load_ladder(a.ladder)
    direction, prob = spec["direction"], spec["problem"]
    rungs = a.rung if a.rung is not None else runner.rung_ids(a.ladder)
    baselines = a.baseline if a.baseline is not None else runner.baseline_ids(a.ladder)
    table = SHAPES[prob["kind"]]
    names = a.shapes or [n for n, _ in table]
    over = dict(table)
    if not torch.cuda.is_available():
        sys.exit("no GPU visible to torch")
    print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  torch {torch.__version__}  ladder {spec['name']}")
    arms = load_arms(a.ladder, rungs, baselines)
    results, ok = [], True
    for name in names:
        sh = R.Shape.from_problem(prob, **over[name])
        exact = spec.get("exact_outputs") or (["o", "lse"] if direction == "fwd" else ["dk", "dv"])
        ok &= run_shape(arms, direction, sh, name, a, results, exact)
        if not ok and not a.keep_going:
            print(f"stopping after shape {name}: a rung failed (use --keep-going to continue)")
            break
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"ladder": spec["name"], "results": results}, f, indent=1)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
