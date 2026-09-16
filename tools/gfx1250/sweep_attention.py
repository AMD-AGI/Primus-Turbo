###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Drive tune_attention.py over a candidate list, one subprocess per candidate.

Appends one JSON object per candidate to a JSONL ledger and is resumable: re-running
skips candidates already in the ledger, so a wedged card costs you the current candidate
and nothing else.

    python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 \
        --axis fwd:num_stages=1,2,3 --ledger out/round1.jsonl

    # cross product of two axes
    python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 \
        --axis bwd:num_warps=1,2,4,8 --axis bwd:num_stages=1,2 --ledger out/round4.jsonl

    python3 tools/gfx1250/sweep_attention.py --report out/round4.jsonl

Three behaviours are load-bearing:

* **rc=139 is a retry, not a verdict.** A segfault or GPU page fault kills the process.
  Scoring that as "this candidate is slow/invalid" silently discards good candidates, and
  on this card the fault is not deterministic -- flex's backward with a B>1 BlockMask was
  measured faulting 2 runs in 12 with the same inputs. Retries are counted and reported.
* **A winner at the end of a swept range means the range was too small.** The sweep says
  so explicitly and names the value to extend to, rather than leaving it to whoever reads
  the table. An earlier campaign on this codebase left +18% on the floor exactly here.
* **Correctness gates acceptance, not ranking.** A candidate that fails SQNR is recorded
  with its timing -- knowing that the wrong answer was also the fast one is the whole
  signal behind the known BLOCK_N1 trap -- but it can never win.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "tune_attention.py"

# Signals that mean "the process died", not "this candidate is slow". Both observed
# non-deterministically on this card at configs whose neighbours run clean:
#   139 = SIGSEGV, a GPU page fault surfacing as a process kill
#   134 = SIGABRT, seen as an HSA abort ("Falling back to file-based dump")
# subprocess reports a signal death as -N, so accept both spellings.
RC_FAULT = (139, -11, 134, -6)

# Hard minimums, so an edge-winner there is not "extend the range".
_KNOB_FLOOR = {"num_stages": 1, "num_warps": 1, "waves_per_eu": 0}
MAX_RETRIES = 3

# A candidate that hangs is worse than one that crashes: the crash is a retry, the hang
# pins the card until someone notices, and on this machine recovering a pinned card means
# an AC cycle. num_warps=16 is a known hang -- it does not fail to compile, it never
# returns -- and it was found by pinning the card, not by reading the code. So every
# candidate gets a wall clock, and a candidate that exceeds it is recorded as a timeout
# rather than being allowed to run.
CANDIDATE_TIMEOUT_S = int(os.environ.get("SWEEP_TIMEOUT_S", 600))


def parse_axis(spec: str) -> tuple[str, str, list[str]]:
    """'bwd:num_warps=1,2,4' -> ('bwd', 'num_warps', ['1','2','4'])."""
    half, rest = spec.split(":", 1) if ":" in spec else ("both", spec)
    if half not in ("fwd", "bwd", "both"):
        raise SystemExit(f"--axis: half must be fwd, bwd or both, got {half!r}")
    key, values = rest.split("=", 1)
    return half, key.strip(), [v.strip() for v in values.split(",") if v.strip()]


def build_spec(points: list[tuple[str, str, str]]) -> str:
    """[(half, key, value), ...] -> the PRIMUS_TURBO_ATTN_TRITON_TUNE string."""
    halves: dict[str, list[str]] = {}
    for half, key, value in points:
        halves.setdefault(half, []).append(f"{key}={value}")
    if set(halves) == {"both"}:
        return ",".join(halves["both"])
    parts = []
    for half in ("fwd", "bwd"):
        items = halves.get(half, []) + halves.get("both", [])
        if items:
            parts.append(f"{half}:" + ",".join(items))
    return ";".join(parts)


def _reap(cmd) -> None:
    """Make sure a timed-out candidate leaves no process holding the GPU.

    subprocess.run's own kill on timeout reaps the direct child, but the child owns a GPU
    context and a half-torn-down context is exactly what leaves a KFD holder behind --
    which is indistinguishable, from the outside, from the card being wedged.
    """
    import time as _time

    pat = str(HARNESS)
    subprocess.run(["pkill", "-f", pat], capture_output=True)
    _time.sleep(2)
    if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        _time.sleep(3)


def run_one(shape: str, spec: str, extra: list[str]) -> dict:
    """One candidate, one process. Retries a fault; records how many it took."""
    cmd = [sys.executable, str(HARNESS), "--shape", shape, *extra]
    if spec:
        cmd += ["--tune", spec]

    retries = 0
    while True:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=CANDIDATE_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as exc:
            # Killing the parent is not enough: the child holds a GPU context, and a
            # half-dead context is exactly what leaves KFD holders behind. Kill the
            # process group and record the timeout as a terminal verdict for this
            # candidate -- unlike a fault, a hang is reproducible and retrying it just
            # spends the timeout again.
            _reap(cmd)
            return {
                "shape": shape,
                "tune": spec,
                "ok": False,
                "retries": retries,
                "returncode": "timeout",
                "timeout_s": CANDIDATE_TIMEOUT_S,
                "stderr_tail": (exc.stderr or b"").decode(errors="replace").strip().splitlines()[-25:]
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "").strip().splitlines()[-25:],
            }
        line = next(
            (ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), ""
        )
        if proc.returncode in RC_FAULT and retries < MAX_RETRIES:
            retries += 1
            print(
                f"  rc={proc.returncode} (process killed, not a result), "
                f"retry {retries}/{MAX_RETRIES}",
                file=sys.stderr,
            )
            continue

        if line:
            rec = json.loads(line)
        else:
            rec = {"shape": shape, "tune": spec, "ok": False}
        rec["retries"] = retries
        rec["returncode"] = proc.returncode
        if not rec.get("ok"):
            # Keep the tail so a compile error or an OOM is diagnosable from the ledger
            # alone, without re-running.
            rec["stderr_tail"] = proc.stderr.strip().splitlines()[-25:]
        return rec


def report(path: Path) -> int:
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    if not records:
        print("ledger is empty")
        return 1

    good = [r for r in records if r.get("ok") and r.get("correct", True)]
    wrong = [r for r in records if r.get("ok") and not r.get("correct", True)]
    broke = [r for r in records if not r.get("ok")]

    print(f"{len(records)} candidates: {len(good)} ok, {len(wrong)} WRONG, {len(broke)} failed\n")
    header = f"{'config':<46} {'fwd ms':>8} {'bwd ms':>8} {'total':>8} {'TFLOP/s':>8} {'step ms':>8} {'rtry':>4}"
    print(header)
    print("-" * len(header))
    for r in sorted(good, key=lambda r: r["total_ms"]):
        print(
            f"{(r['tune'] or '<default>'):<46} {r['fwd_ms']:>8.3f} {r['bwd_ms']:>8.3f} "
            f"{r['total_ms']:>8.3f} {r['total_tflops']:>8.1f} {r['per_step_ms']:>8.1f} {r.get('retries', 0):>4}"
        )

    if wrong:
        # total_ms may be absent: tune_attention returns BEFORE timing when SQNR fails, so a
        # wrong candidate carries its dB values and no time. Reading it unguarded crashed the
        # report -- and the same read in the run loop below aborted the sweep -- on exactly
        # the candidate this tool exists to surface.
        print("\nWRONG -- fast but incorrect, never eligible to win:")
        for r in sorted(wrong, key=lambda r: r.get("total_ms", float("inf"))):
            db = r.get("sqnr_db", {})
            failed = ",".join(r.get("failed_tensors", []))
            _ms = f"{r['total_ms']:>8.3f} ms" if "total_ms" in r else "  (untimed)"
            print(
                f"  {(r['tune'] or '<default>'):<44} {_ms}  "
                f"failed={failed}  "
                + " ".join(f"{t}={db.get(t, float('nan')):.1f}dB" for t in ("out", "dq", "dk", "dv"))
            )

    if broke:
        print("\nfailed to run:")
        for r in broke:
            tail = (r.get("stderr_tail") or ["(no stderr)"])[-1]
            print(f"  {(r['tune'] or '<default>'):<44} rc={r.get('returncode')}  {tail[:90]}")

    # Endpoint check: a winner sitting at the edge of a swept range means the range was
    # the constraint, not the hardware.
    if good:
        best = min(good, key=lambda r: r["total_ms"])
        print(f"\nbest: {best['tune'] or '<default>'}  {best['total_ms']:.3f} ms  "
              f"({best['total_tflops']:.1f} TFLOP/s, {best['per_step_ms']:.1f} ms/step)")
        for key, value in dict(
            kv.split("=", 1) for part in best["tune"].replace(";", ",").split(",")
            if "=" in part for kv in [part.split(":")[-1]]
        ).items():
            swept = sorted(
                {
                    v
                    for r in good
                    for p in r["tune"].replace(";", ",").split(",")
                    if "=" in p and p.split(":")[-1].split("=")[0] == key
                    for v in [p.split("=", 1)[1]]
                },
                key=lambda x: float(x) if x.replace(".", "").isdigit() else 0,
            )
            # A winner at a range edge usually means the range was the constraint. The
            # exception is an edge that is the knob's own hard minimum -- num_stages and
            # num_warps cannot go below 1 -- where there is nothing to extend into and
            # reporting it as unconverged just trains the reader to ignore the warning.
            at_floor = value == swept[0] and key in _KNOB_FLOOR and float(value) <= _KNOB_FLOOR[key]
            if len(swept) > 1 and value in (swept[0], swept[-1]) and not at_floor:
                print(
                    f"  NOT CONVERGED: {key}={value} is at the {'low' if value == swept[0] else 'high'} "
                    f"end of the swept range {swept}. Extend the range and re-run."
                )
            elif at_floor:
                print(f"  converged: {key}={value} is this knob's hard minimum.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="llama31-8b-s4096")
    ap.add_argument("--axis", action="append", default=[],
                    help="e.g. 'bwd:num_warps=1,2,4,8'. Repeat for a cross product.")
    ap.add_argument("--ledger", default="", help="JSONL ledger; resumable")
    ap.add_argument("--report", default="", help="print a report for an existing ledger and exit")
    ap.add_argument("--baseline", action="store_true", help="also run the shipped default")
    ap.add_argument("--harness-arg", action="append", default=[],
                    help="passed through to tune_attention.py")
    args = ap.parse_args()

    if args.report:
        return report(Path(args.report))
    if not args.ledger:
        raise SystemExit("--ledger is required (or use --report)")
    if not args.axis:
        raise SystemExit("at least one --axis is required")

    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if ledger.exists():
        done = {
            json.loads(ln)["tune"]
            for ln in ledger.read_text().splitlines()
            if ln.strip()
        }
        print(f"resuming: {len(done)} candidates already in {ledger}")

    axes = [parse_axis(a) for a in args.axis]
    specs = []
    if args.baseline:
        specs.append("")
    for combo in itertools.product(*[[(h, k, v) for v in vals] for h, k, vals in axes]):
        specs.append(build_spec(list(combo)))

    todo = [s for s in specs if s not in done]
    print(f"{len(specs)} candidates, {len(todo)} to run on shape {args.shape}\n")

    with ledger.open("a") as fh:
        for i, spec in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {spec or '<default>'}", flush=True)
            rec = run_one(args.shape, spec, args.harness_arg)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # survive a wedge that takes the box with it

            if rec.get("new_dmesg_faults"):
                print("  !! new GPU faults in dmesg -- stopping the sweep", file=sys.stderr)
                for ln in rec["new_dmesg_faults"][:5]:
                    print("     " + ln, file=sys.stderr)
                return 3
            status = (
                "ok" if rec.get("ok") and rec.get("correct", True)
                else "WRONG" if rec.get("ok") else f"rc={rec.get('returncode')}"
            )
            if rec.get("ok") and "total_ms" in rec:
                print(f"  {status}  {rec['total_ms']:.3f} ms  {rec['total_tflops']:.1f} TFLOP/s")
            else:
                print(f"  {status}")

    return report(ledger)


if __name__ == "__main__":
    sys.exit(main())
