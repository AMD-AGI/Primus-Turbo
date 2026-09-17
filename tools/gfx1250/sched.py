#!/usr/bin/env python3
"""Serial single-GPU experiment scheduler.

One worker, one card, one ledger. Written for a ~16 h unattended window on a box whose GPU
has repeatedly needed an AC cycle, so the two things it must never do are block and stop.

WHY NOT bin/queue.sh + bin/watchdog.sh. Those were written for the 4-GPU box and carry four
defects that are structural rather than incidental:

  - queue.sh:40 and watchdog.sh:66 redirect into output/0914__campaign/logs/, which does not
    exist in this checkout. Every candidate records "r":null while the queue looks healthy.
    That is the failure mode this file is most careful about: a scheduler that appears to
    work and produces nothing is worse than one that crashes.
  - watchdog.sh hardcodes a 4-GPU stream map, so on one card it declares two nonexistent
    streams dead and restarts them every 60 s forever. With one serial worker the whole
    fleet-liveness problem disappears, so there is no separate watchdog here at all -- the
    heartbeat is this process's own status write.
  - the phase grid is a bash `case`, so work cannot be added without a restart, and
    forever_queue.sh's round-counter tags make an exhausted queue look busy forever.
  - sweep_attention.py:93 has no subprocess timeout, so one wedged candidate blocks the day.

RESUME KEY IS THE TAG, and only the tag. sweep_attention.py:214 keys on the `tune` spec
string; today's most important rows differ only in --impl and --asm-fwd at an identical
empty tune, so that key would collapse them and silently skip the entire A/B. It also cannot
express repeats, and n>=3 is where this session's credibility comes from.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "output" / "0915__opt"
QUEUE = OUT / "queue.jsonl"
LEDGER = OUT / "ledger.jsonl"
EVENTS = OUT / "events.jsonl"
BUDGETS = OUT / "budgets.json"
STATUS_MD = OUT / "STATUS.md"
STATUS_JSON = OUT / "status.json"
STOP = OUT / "STOP"
PIDFILE = OUT / "sched.pid"
RUNDIR = OUT / "run"
LOGDIR = OUT / "logs"

CTR = os.environ.get("CTR", "fa-repro")
CACHE = os.environ.get("TRITON_CACHE_DIR", "/tmp/triton_cache_g0")
AITER = "/home/lihuzhan/code/aiter-src"

# Static budgets, seconds. Derived from measured wall on THIS box: 9.5-12.0 s in-process
# across 13 rows on 0915, ~20 s end-to-end cold. HANDOFF.md's ">= 900 s floor" is retired
# here -- it was dominated by a 58-172 s torch.cuda.init() stall that scales with concurrent
# GPU processes and does not occur on one card. Keeping 1800 s would mean one wedged kernel
# eats 3% of the window, and a repeatedly-wedging config eats the night.
STATIC = {"measure": 120, "measure_cold": 300, "bringup": 300, "sweep": 300, "e2e": 3600, "cpu": 600}
FLOOR = {"measure": 60, "measure_cold": 180, "bringup": 120, "sweep": 90, "e2e": 1800, "cpu": 120}

DEGRADED_RE = "failed to respond to msg|GPU Hang|Memory access fault"
WEDGED_RE = "wait for reset ack|ring gfx timeout|GPU reset begin"
# Terminal outcomes count as done. `timeout` and `fault` deliberately do NOT: a transient
# failure gets another pass rather than being silently burned.
TERMINAL = {"ok", "wrong", "quarantined", "asserted_fail"}
# How many non-terminal attempts (timeout / fault) a tag gets before it is quarantined and
# the queue moves past it for good. 2 means one retry: enough to ride out a transient fault,
# not enough for one bad spec to own the card.
MAX_ATTEMPTS = 2


def now() -> float:
    return time.time()


def iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t if t is not None else now()))


def event(kind: str, **kw) -> None:
    rec = {"t": now(), "iso": iso(), "kind": kind, **kw}
    with EVENTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def sh(cmd: str, timeout: int = 20) -> str:
    """Bounded shell read. Only ever used for dmesg and sysfs -- see gpu_health.sh for why
    rocm-smi, ps, pgrep and torch.cuda.device_count() may not appear anywhere in this file."""
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def dmesg_counts() -> tuple[int, int]:
    txt = sh("timeout 15 dmesg 2>/dev/null | tail -n 4000", timeout=25)
    d = sum(1 for ln in txt.splitlines() if any(p in ln for p in DEGRADED_RE.split("|")))
    w = sum(1 for ln in txt.splitlines() if any(p in ln for p in WEDGED_RE.split("|")))
    return d, w


def sclk() -> str:
    return sh("cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | awk '/\\*/{print $2; exit}'")


# ---------------------------------------------------------------- queue / ledger

def read_specs() -> tuple[list[dict], int]:
    """A malformed line is counted and skipped, never fatal. Crash-on-bad-JSON is the same
    class of failure as the missing logs dir: the scheduler stops producing and it is not
    obvious why."""
    specs, bad = [], 0
    if not QUEUE.exists():
        return specs, bad
    for i, ln in enumerate(QUEUE.read_text().splitlines()):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            s = json.loads(ln)
            s.setdefault("prio", 5)
            s.setdefault("class", "measure")
            s.setdefault("track", "gpu")
            s.setdefault("n", 1)
            s.setdefault("needs", [])
            s.setdefault("script", "tools/gfx1250/tune_attention.py")
            s["_order"] = i
            specs.append(s)
        except Exception:
            bad += 1
    return specs, bad


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for ln in LEDGER.read_text().splitlines():
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def expand(spec: dict) -> list[tuple[str, dict]]:
    n = int(spec.get("n", 1))
    if n <= 1:
        return [(spec["tag"], spec)]
    return [(f"{spec['tag']}|{i}", spec) for i in range(1, n + 1)]


def pick(specs: list[dict], rows: list[dict]) -> tuple[tuple[str, dict] | None, dict]:
    done = {r["tag"] for r in rows if r.get("outcome") in TERMINAL}
    passed = {r["tag"] for r in rows if r.get("outcome") == "ok"}

    # Retries are bounded, or "timeout is not terminal" turns into its own way of hanging:
    # the spec stays ready, its prio keeps winning, and the queue re-runs one wedged
    # candidate for the rest of the window while looking busy. Caught by the self-test.
    attempts: dict[str, int] = {}
    for r in rows:
        if r.get("outcome") not in TERMINAL:
            attempts[r["tag"]] = attempts.get(r["tag"], 0) + 1
    done |= {t for t, n in attempts.items() if n >= MAX_ATTEMPTS}
    # A spec's `needs` is satisfied when every named tag has at least one passing row, either
    # its bare form or any repeat of it.
    def needs_ok(s: dict) -> bool:
        for nd in s.get("needs", []):
            if nd not in passed and not any(p.split("|")[0] == nd for p in passed):
                return False
        return True

    ready_gpu, ready_cpu, blocked = [], [], 0
    for s in sorted(specs, key=lambda x: (x["prio"], x["_order"])):
        pend = [(t, s) for t, s in expand(s) if t not in done]
        if not pend:
            continue
        if not needs_ok(s):
            blocked += len(pend)
            continue
        (ready_cpu if s["track"] == "cpu" else ready_gpu).append(pend[0])
    stats = {"queued": len(ready_gpu) + len(ready_cpu), "blocked": blocked,
             "ready_gpu": len(ready_gpu), "ready_cpu": len(ready_cpu),
             "next": [t for t, _ in (ready_gpu + ready_cpu)[:4]]}
    return (ready_gpu, ready_cpu), stats


# ---------------------------------------------------------------- budgets

def load_budgets() -> dict:
    if BUDGETS.exists():
        try:
            return json.loads(BUDGETS.read_text())
        except Exception:
            pass
    return {"walls": {}, "budget": dict(STATIC)}


def save_budgets(b: dict) -> None:
    BUDGETS.write_text(json.dumps(b, indent=1))


def budget_for(b: dict, cls: str) -> int:
    return int(b["budget"].get(cls, STATIC.get(cls, 300)))


def update_budget(b: dict, cls: str, wall: float) -> None:
    """4*p90 + 30, floored, and capped at 2x static. The cap is the important half: without
    it the budget is self-referential and a slow-but-alive card stretches the whole day
    without anything looking wrong."""
    ring = b["walls"].setdefault(cls, [])
    ring.append(round(wall, 2))
    del ring[:-20]
    if len(ring) < 3:
        return
    p90 = sorted(ring)[max(0, int(len(ring) * 0.9) - 1)]
    b["budget"][cls] = int(max(FLOOR.get(cls, 60),
                               min(4 * p90 + 30, 2 * STATIC.get(cls, 300))))


# ---------------------------------------------------------------- launching

def build_cmd(tag: str, spec: dict, inner: int) -> tuple[list[str], Path]:
    """The nested-timeout construct from HANDOFF.md:236-237, which is the only thing in this
    tree that actually reaps a container-side process.

    A host-side `timeout` kills the `docker exec` CLIENT; the process inside the container
    keeps running and keeps holding the card, so the next candidate measures a contended GPU
    -- and contention produces 42-48 dB SQNR against 53.7-53.9 clean, i.e. it manufactures
    correctness failures. Putting a `timeout` INSIDE the container, as the parent of python,
    reaps regardless of what happens to the host client.

    HOST = inner + 30 so the inner one always fires first: that yields a real rc=124 and
    stderr rather than a decapitated client and an orphan.

    argv elements are passed through "$@" as separate execve arguments. The naive
    `bash -lc "cd $R && python3 ... $*"` form is re-word-split by the inner shell and
    truncated a --tune spec at its semicolon, recording 25.695 ms for a row that was actually
    18.916 -- a 1.36x error that survived into a published table.
    """
    cpid = RUNDIR / (tag.replace("|", "_").replace("/", "_") + ".cpid")
    argv = [str(a) for a in spec.get("argv", [])]
    return ([
        "timeout", str(inner + 30),
        "docker", "exec",
        "-e", "GPU=0", "-e", f"PYTHONPATH={AITER}", "-e", f"TRITON_CACHE_DIR={CACHE}",
        CTR, "bash", "-lc",
        'cd "$0" && echo $$ > "$1" && exec timeout -k 15 -s TERM "$2" python3 "$3" "${@:4}"',
        str(REPO), str(cpid), str(inner), spec["script"], *argv,
    ], cpid)


def reap(cpid: Path) -> bool:
    """Second layer, used only when the inner timeout was itself wedged. Children first, then
    the timeout parent. NEVER `docker kill` the container: fa-repro is shared and the user may
    be attached -- that converts one stuck job into a dead session."""
    if not cpid.exists():
        return False
    pid = cpid.read_text().strip()
    if not pid.isdigit():
        return False
    sh(f"timeout 20 docker exec {CTR} bash -c 'pkill -9 -P {pid}; kill -9 {pid}' 2>/dev/null", timeout=25)
    return True


def check_asserts(spec: dict, r: dict | None) -> dict:
    """Annotation only. An assert NEVER discards a row; the raw number always prints beside
    the criterion. A wrong assert should look wrong, not manufacture a confident conclusion."""
    outs = {}
    for key, cond in (spec.get("assert") or {}).items():
        got = r
        for part in key.split("."):
            got = (got or {}).get(part) if isinstance(got, dict) else None
        op, v = cond.get("op", ">="), cond.get("v")
        met = None
        if isinstance(got, (int, float)):
            met = {">=": got >= v, "<=": got <= v, ">": got > v, "<": got < v}.get(op)
        outs[key] = {"want": f"{op}{v}", "got": got, "met": met}
    return outs


# ---------------------------------------------------------------- status

def write_status(st: dict) -> None:
    tmp = STATUS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATUS_JSON)

    L = []
    L.append(f"# 0915 opt — {st['state']} — {st['iso']}   uptime {st['uptime']}   idle {st['idle_pct']:.1f}%")
    L.append("")
    if st.get("now_running"):
        L.append(f"NOW     {st['now_running']:<28} started {st['now_started']} ({st['now_elapsed']}s)  budget {st['now_budget']}s")
    else:
        L.append(f"NOW     (idle) {st.get('idle_reason','')}")
    L.append(f"NEXT    {' · '.join(st.get('next', [])) or '(nothing ready)'}")
    L.append(f"        {st['queued']} queued · {st['blocked']} blocked on needs · {st['malformed']} malformed")
    L.append(f"GPU     {st['state']}  sclk {st['sclk']}  new faults deg={st['deg_new']} wedge={st['wedge_new']}  sigkills {st['sigkills']}")
    L.append(f"        budgets: " + "  ".join(f"{k} {v}s" for k, v in st["budgets"].items()))
    d = st["counts"]
    L.append(f"DONE    {d.get('ok',0)} ok · {d.get('wrong',0)} wrong · {d.get('timeout',0)} timeout · {d.get('fault',0)} fault")
    if st.get("best"):
        L.append(f"BEST    {st['best']}")
    L.append("")
    L.append("LEARNED")
    for ln in st.get("learned", [])[-12:]:
        L.append(f"  {ln}")
    if not st.get("learned"):
        L.append("  (nothing asserted yet)")
    L.append("")
    L.append("RECENT")
    for ln in st.get("recent", [])[-8:]:
        L.append(f"  {ln}")
    L.append("")
    L.append("ACTION")
    for ln in st.get("action", []):
        L.append(f"  !! {ln}")
    tmp = STATUS_MD.with_suffix(".tmp")
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, STATUS_MD)


# ---------------------------------------------------------------- main

def main() -> int:
    for d in (OUT, RUNDIR, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
    QUEUE.touch()
    LEDGER.touch()

    if PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text().strip()), 0)
            print("already running", file=sys.stderr)
            return 3
        except Exception:
            pass
    PIDFILE.write_text(str(os.getpid()))

    base_d, base_w = dmesg_counts()
    event("start", base_degraded=base_d, base_wedged=base_w, pid=os.getpid())
    budgets = load_budgets()
    t0 = now()
    sigkills = 0
    idle_s = 0.0
    recent: list[str] = []
    learned: list[str] = []
    action: list[str] = []
    last_wedge_probe = 0.0
    state = "HEALTHY"

    while True:
        loop_top = now()
        specs, bad = read_specs()
        rows = read_ledger()
        (ready_gpu, ready_cpu), qstats = pick(specs, rows)

        d, w = dmesg_counts()
        if w > base_w:
            state = "WEDGED"
        elif d > base_d and state != "WEDGED":
            state = "DEGRADED"

        counts: dict[str, int] = {}
        for r in rows:
            counts[r.get("outcome", "?")] = counts.get(r.get("outcome", "?"), 0) + 1
        best = ""
        bw = [r for r in rows if r.get("outcome") == "ok" and (r.get("r") or {}).get("bwd_ms")]
        if bw:
            b = min(bw, key=lambda r: r["r"]["bwd_ms"])
            best = f"bwd {b['r']['bwd_ms']:.3f} ms ({b['tag']})   must-beat 17.692"

        job = None
        if state != "WEDGED" and ready_gpu:
            job = ready_gpu[0]
        elif ready_cpu:
            job = ready_cpu[0]

        st = {
            "iso": iso(), "state": state, "uptime": f"{int((now()-t0)//3600)}h{int(((now()-t0)%3600)//60)}m",
            "idle_pct": 100.0 * idle_s / max(1.0, now() - t0),
            "sclk": sclk(), "deg_new": d - base_d, "wedge_new": w - base_w, "sigkills": sigkills,
            "budgets": budgets["budget"], "counts": counts, "best": best,
            "queued": qstats["queued"], "blocked": qstats["blocked"], "malformed": bad,
            "next": qstats["next"], "recent": recent, "learned": learned, "action": action,
            "now_running": None, "idle_reason": "",
        }

        if STOP.exists():
            st["idle_reason"] = "STOP sentinel present"
            write_status(st)
            idle_s += now() - loop_top
            time.sleep(10)
            continue

        if job is None:
            st["idle_reason"] = ("WEDGED and no CPU work left" if state == "WEDGED" else "queue exhausted")
            if state == "WEDGED" and now() - last_wedge_probe > 600:
                # Self-recovery probe. If a human AC-cycles the box, the queue resumes with no
                # further intervention -- which is the whole point over a 16 h window.
                last_wedge_probe = now()
                rc = subprocess.run(["bash", "-c",
                    f"timeout 210 docker exec -e GPU=0 -e TRITON_CACHE_DIR={CACHE} {CTR} bash -lc "
                    f"'cd \"$0\" && exec timeout -k 15 -s TERM 180 python3 tools/gfx1250/tune_attention.py --shape smoke --impl turbo' {REPO}"],
                    capture_output=True, text=True).returncode
                if rc == 0:
                    base_d, base_w = dmesg_counts()
                    state = "HEALTHY"
                    event("recovered", note="smoke probe passed after wedge")
            write_status(st)
            idle_s += now() - loop_top
            time.sleep(15)
            continue

        tag, spec = job
        cls = spec["class"]
        # A spec may pin its own budget. Used for the deliberate-timeout self-test, and for
        # anything whose wall is known to be unlike its class.
        bud = int(spec["budget_s"]) if spec.get("budget_s") else budget_for(budgets, cls)
        cmd, cpid = build_cmd(tag, spec, bud)
        errf = LOGDIR / (tag.replace("|", "_").replace("/", "_") + ".err")
        errf.parent.mkdir(parents=True, exist_ok=True)

        st["now_running"] = tag
        st["now_started"] = iso()
        st["now_elapsed"] = 0
        st["now_budget"] = bud
        # NEXT is computed before the job is picked, so its head is the job now running.
        # Showing the same tag on both lines reads as a stuck queue.
        st["next"] = [t for t in st["next"] if t != tag]
        write_status(st)
        event("launch", tag=tag, cls=cls, budget=bud, cmd=" ".join(shlex.quote(c) for c in cmd))

        started = now()
        with errf.open("w") as ef:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=ef, text=True)
            out_lines: list[str] = []
            while proc.poll() is None:
                st["now_elapsed"] = int(now() - started)
                write_status(st)
                time.sleep(5)
            out_lines = (proc.stdout.read() or "").splitlines() if proc.stdout else []
        rc = proc.returncode
        wall = now() - started

        outcome, r = "fault", None
        jline = next((l for l in reversed(out_lines) if l.strip().startswith("{")), None)
        if jline:
            try:
                r = json.loads(jline)
            except Exception:
                r = None
        if rc == 124 or wall >= bud + 25:
            outcome = "timeout"
            if reap(cpid):
                sigkills += 1
        elif r is not None:
            outcome = "ok" if r.get("correct") else "wrong"
        elif rc != 0:
            outcome = "fault"

        asserts = check_asserts(spec, r)
        if outcome == "ok" and any(a["met"] is False for a in asserts.values()):
            outcome = "asserted_fail"

        row = {"tag": tag, "t": started, "iso": iso(started), "wall_s": round(wall, 2),
               "outcome": outcome, "rc": rc, "class": cls, "degraded": state == "DEGRADED",
               "asserts": asserts, "r": r,
               "stderr_tail": errf.read_text().splitlines()[-8:] if errf.exists() else []}
        with LEDGER.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        event("done", tag=tag, outcome=outcome, wall_s=round(wall, 2), rc=rc)

        if outcome == "ok":
            update_budget(budgets, cls, wall)
            save_budgets(budgets)

        if r:
            s = r.get("sqnr_db") or {}
            recent.append(f"{iso(started)[11:]}  {tag:<26} {outcome:<6} {wall:6.1f}s  "
                          f"bwd {r.get('bwd_ms', 0):8.4f}  sqnr {s.get('out', 0):.2f}")
        else:
            recent.append(f"{iso(started)[11:]}  {tag:<26} {outcome:<6} {wall:6.1f}s  rc={rc}")
        del recent[:-40]
        for k, a in asserts.items():
            mark = "OK" if a["met"] else ("--" if a["met"] is None else "NO")
            learned.append(f"{mark} {tag:<24} {k} {a['want']}  got {a['got']}")
        del learned[:-40]
        att = sum(1 for r in read_ledger() if r["tag"] == tag and r.get("outcome") not in TERMINAL)
        if outcome not in TERMINAL and att >= MAX_ATTEMPTS:
            msg = f"quarantined after {att} attempts: {tag} (outcome={outcome})"
            if msg not in action:
                action.append(msg)
            event("quarantined", tag=tag, attempts=att, outcome=outcome)
        if state == "WEDGED" and "AC cycle required -- driver reset does not complete" not in action:
            action.append("AC cycle required -- driver reset does not complete")

    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main())
