#!/usr/bin/env python3
"""CPU-only self-test of attn_backends/e2e_attn/attn_timer.py (no torch, no GPU).

Simulates the training loop's call pattern with fake events: per step, layer 0..L-1 forward
(drain at layer 0 when step > 0), then backward in reverse layer order; some events complete
late. Checks every step is written exactly once with the right sums and arms, that incomplete
pairs are deferred (never dropped), and that an exception disables the timer without raising.
usage: python3 selftest_timer.py   -> prints SELFTEST_OK or raises AssertionError
"""
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "attn_timer", HERE.parent / "attn_backends" / "e2e_attn" / "attn_timer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CLOCK = [0.0]
DRAINS = []


class FakeEvent:
    late = set()                 # ids of events that report incomplete on the first query

    def __init__(self):
        self.t = None
        self.queried = 0
        self.id = id(self)

    def record(self):
        CLOCK[0] += 1.0
        self.t = CLOCK[0]

    def query(self):
        self.queried += 1
        return not (self.id in FakeEvent.late and self.queried == 1)

    def elapsed_time(self, other):
        return other.t - self.t


def run(L=4, steps=5, sched=("asm", "fly", "flyr29")):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)
    logs = []
    t = mod.AttnTimer(path, event_factory=FakeEvent, log=logs.append)
    expect = {}
    for s in range(steps):
        arm = sched[s % len(sched)]
        fwd_sum = bwd_sum = 0.0
        for layer in range(L):
            if layer == 0 and s > 0:
                t.drain(s)
                done = [json.loads(x)["step"] for x in open(path)] if os.path.exists(path) else []
                DRAINS.append((s, done))
            e0 = t.begin()
            CLOCK[0] += 0.5 + layer          # "kernel time" inside the pair
            t.end(e0, s, layer, "fwd", arm)
            fwd_sum += 1.0 + 0.5 + layer
        for layer in reversed(range(L)):
            e0 = t.begin()
            CLOCK[0] += 2.0 + layer
            t.end(e0, s, layer, "bwd", arm)
            bwd_sum += 1.0 + 2.0 + layer
            if s == 2 and layer == 1:        # this pair completes late
                FakeEvent.late.add(t.pending[-1][5].id)
        expect[s + 1] = (arm, fwd_sum, bwd_sum)
    t.close()
    lines = [json.loads(x) for x in open(path)]
    os.unlink(path)
    return lines, expect, logs, t


lines, expect, logs, t = run()
steps = [r["step"] for r in lines]
assert sorted(steps) == list(expect), (steps, list(expect))
assert len(steps) == len(set(steps)), "a step was written twice"
for r in lines:
    arm, f, b = expect[r["step"]]
    assert r["arm_fwd"] == arm and r["arm_bwd"] == arm, r
    assert abs(r["fwd_ms"] - f) < 1e-9 and abs(r["bwd_ms"] - b) < 1e-9, (r, f, b)
    assert r["n_fwd"] == 4 and r["n_bwd"] == 4 and None not in r["fwd"] + r["bwd"], r
# the late pair of step 3 (s0=2) keeps it unwritten at the drain that opens step 4 (s=3),
# and it is written at the next drain (s=4), once
d = dict(DRAINS)
assert 3 not in d[3] and 2 in d[3], DRAINS
assert 3 in d[4], DRAINS
assert t.ok and any("4" in m or "steps written" in m for m in logs), logs


class Boom(FakeEvent):
    def record(self):
        raise RuntimeError("device gone")


fd, p = tempfile.mkstemp()
os.close(fd)
logs = []
t = mod.AttnTimer(p, event_factory=Boom, log=logs.append)
tok = t.begin()                      # must not raise
t.end(tok, 0, 0, "fwd", "asm")
t.drain(1)
t.close()
assert not t.ok and tok is None and any("disabled" in m for m in logs), logs
os.unlink(p)
print("SELFTEST_OK steps", steps)
