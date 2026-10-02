#!/usr/bin/env python3
"""Hand edits of an op-evolve job's state.yaml for the gb-ruler rollout (PT/output/1002__oe/RULER.md).

    state_edit.py rebase-champions --job JOBDIR --round N --reason TEXT [--write]
    state_edit.py close-round      --job JOBDIR --round N --reason TEXT [--write]
    state_edit.py realign-best     --job JOBDIR --round N --reason TEXT [--write]

Without --write it prints the unified diff and changes nothing. With --write it first copies state.yaml to
state.yaml.bak.pre-gb-<stamp>, then writes. Host python3 + PyYAML only; never touches the card.

rebase-champions  every per-shape champion -> round N (the promoted best round), and a `hand_edit` lifecycle
                  event. Refused unless N == best_round AND rounds/N/op is byte-identical to op/current (a
                  refactor round whose own candidate was rejected leaves rounds/N/op != op/current -- the fwd
                  job's rounds 17-19 then re-measured the wrong tree as "champion_round 16").
close-round       the LAST round, unfinished, is closed as `outcome: failed` (terminal for core/loop.py
                  _round_to_run), `running` removed, closed_by_hand/closed_at/note set; nothing is promoted.
realign-best      N == best_round but rounds/N/op != op/current (a refactor round whose own candidate was then
                  rejected): rounds/N/op is moved to rounds/N/op.rejected-cand-<stamp> and replaced by a copy of
                  op/current, so every later prompt that names rounds/N/op as "the incumbent" measures the
                  promoted code. state.yaml only gets a `hand_edit` event. Run rebase-champions after it.

Both refuse when the loop is running (JOBDIR/.pid names a live process) or when the file does not round-trip
byte-identically through yaml.safe_dump(sort_keys=False, width=100) -- the framework's own dump settings
(core/state.py save()), so the diff shows only the edit.
"""
import argparse
import difflib
import filecmp
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SKIP = {"__pycache__", ".runs", ".build", "build"}


def dump(d) -> str:
    return yaml.safe_dump(d, sort_keys=False, width=100)


def loop_alive(job: Path):
    pid_file = job / ".pid"
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError):
        return None
    except PermissionError:
        return pid


def same_tree(a: Path, b: Path) -> list:
    """Differences between two source trees, ignoring caches and core dumps."""
    filecmp.clear_cache()      # its cache keys on (size, mtime): a same-size copy made within one clock tick is stale
    diffs = []

    def walk(x: Path, y: Path, rel=""):
        cmp = filecmp.dircmp(x, y, ignore=list(SKIP))
        diffs.extend(f"{rel}{n} only in {x}" for n in cmp.left_only if not n.startswith("core"))
        diffs.extend(f"{rel}{n} only in {y}" for n in cmp.right_only if not n.startswith("core"))
        for n in cmp.common_files:
            if n.startswith("core") or n.endswith(".pyc"):
                continue
            if not filecmp.cmp(x / n, y / n, shallow=False):
                diffs.append(f"{rel}{n} differs")
        for n in cmp.common_dirs:
            walk(x / n, y / n, f"{rel}{n}/")

    walk(a, b)
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("rebase-champions", "close-round", "realign-best"))
    ap.add_argument("--job", required=True, type=Path, help="artifacts/<job> directory")
    ap.add_argument("--round", required=True, type=int)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    job = a.job.resolve()
    sp = job / "job_context" / "state.yaml"
    text = sp.read_text()
    d = yaml.safe_load(text)
    if dump(d) != text:
        raise SystemExit(f"{sp} does not round-trip through safe_dump(width=100): refusing (the diff would "
                         "reformat the file)")
    pid = loop_alive(job)
    if pid:
        raise SystemExit(f"loop pid {pid} from {job / '.pid'} is alive: stop the job first (op-evolve stop)")
    stamp = datetime.now().replace(microsecond=0).isoformat()

    if a.action == "realign-best":
        n = a.round
        if d.get("best_round") != n:
            raise SystemExit(f"best_round is {d.get('best_round')}, not {n}")
        rop, cur = job / "rounds" / f"{n:03d}" / "op", job / "job_context" / "op" / "current"
        diffs = same_tree(rop, cur)
        if not diffs:
            print(f"rounds/{n:03d}/op == op/current already: nothing to do")
            return 0
        aside = rop.with_name(f"op.rejected-cand-{time.strftime('%m%d-%H%M%S')}")
        print(f"rounds/{n:03d}/op != op/current ({'; '.join(diffs[:5])})\nwould move {rop} -> {aside} and copy "
              f"{cur} -> {rop}")
        if not a.write:
            print("\n(dry run: nothing written; add --write)")
            return 0
        os.rename(rop, aside)
        shutil.copytree(cur, rop, symlinks=True, ignore=shutil.ignore_patterns(*SKIP, "*.pyc", "core", "core.*"))
        left = same_tree(rop, cur)
        if left:
            raise SystemExit(f"copy differs from op/current after realign: {left[:5]}")
        d.setdefault("lifecycle", []).append({"at": stamp, "event": "hand_edit", "reason":
            f"rounds/{n:03d}/op realigned to op/current (the rejected candidate kept as {aside.name}). {a.reason}"})
        bak = sp.with_name(f"state.yaml.bak.pre-gb-{time.strftime('%m%d-%H%M%S')}")
        shutil.copy2(sp, bak)
        sp.write_text(dump(d))
        print(f"realigned; state.yaml event added; backup {bak}")
        return 0

    if a.action == "rebase-champions":
        n = a.round
        if d.get("best_round") != n:
            raise SystemExit(f"best_round is {d.get('best_round')}, not {n}: champions may only be rebased onto "
                             "the promoted round")
        diffs = same_tree(job / "rounds" / f"{n:03d}" / "op", job / "job_context" / "op" / "current")
        if diffs:
            raise SystemExit(f"rounds/{n:03d}/op != op/current ({'; '.join(diffs[:5])}): the prompt would name "
                             "the wrong tree as the incumbent -- refusing")
        old = dict(d.get("champions") or {})
        d["champions"] = {name: n for name in (old or {"fast": n, "proxy": n, "prod": n})}
        event = {"at": stamp, "event": "hand_edit",
                 "reason": f"gb ruler rollout: champions {old} -> all {n} (rounds/{n:03d}/op == op/current, "
                           f"verified). {a.reason}"}
    else:
        rows = d.get("rounds") or []
        if not rows or rows[-1].get("round") != a.round:
            raise SystemExit(f"round {a.round} is not the last row (last: {rows[-1].get('round') if rows else None})")
        row = rows[-1]
        if row.get("outcome") == "failed":
            raise SystemExit(f"round {a.round} is already closed (outcome failed)")
        row.pop("running", None)
        row["closed_by_hand"] = True
        row["closed_at"] = stamp
        row["outcome"] = "failed"
        row["passed"] = False
        row["accepted"] = False
        row["note"] = f"CLOSED BY HAND {stamp}: {a.reason}"
        event = {"at": stamp, "event": "hand_edit", "reason": f"round {a.round} closed by hand. {a.reason}"}

    d.setdefault("lifecycle", []).append(event)
    new = dump(d)
    sys.stdout.writelines(difflib.unified_diff(text.splitlines(True), new.splitlines(True),
                                               "state.yaml", "state.yaml (edited)"))
    if not a.write:
        print("\n(dry run: nothing written; add --write)")
        return 0
    bak = sp.with_name(f"state.yaml.bak.pre-gb-{time.strftime('%m%d-%H%M%S')}")
    shutil.copy2(sp, bak)
    tmp = sp.with_name("state.yaml.tmp-gb")
    tmp.write_text(new)
    os.replace(tmp, sp)
    print(f"\nwritten {sp}; backup {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
