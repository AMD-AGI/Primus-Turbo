#!/usr/bin/env python3
"""Add operator hints to a job's hint.md the way core/hints.py reads them (PT/output/1002__oe/RULER.md).

    hintadd.py --hint JOBDIR/job_context/hint.md --add ruler/bwd/hint_add.md [--retire 'h29=superseded (...)'] [--write]

--add FILE   rows `| hN | type | title | status |` are inserted at the end of hint.md's index table (the first
             block of `| hN |` rows); `## hN -- ...` sections are appended at the end of the file. A row whose
             section already exists in hint.md is fine (e.g. bwd h83, whose row was missing). Refused if a row
             id, or a section id being added, is already present.
--retire     hN=STATUS: that row's status cell becomes STATUS and `standing` is dropped from its type cell, so
             it stops rendering as a constraint in route.md (core/hints.py renders every `standing` row
             whatever its status; this is how bwd h62/h65 were retired on 2026-09-30).
The result is parsed with op-evolve's own core/hints.py (loaded by path) and checked before anything is
written. Without --write: prints the diff. With --write: backup hint.md.bak.pre-gb-<stamp>, then write.
"""
import argparse
import difflib
import importlib.util
import re
import shutil
import sys
import time
from pathlib import Path

OE = Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve")
ROW = re.compile(r"^\s*\|\s*(h\d+)\s*\|")
SEC = re.compile(r"^\s{0,3}#{1,6}\s+(h\d+)\b")


def oe_hints():
    spec = importlib.util.spec_from_file_location("oe_core_hints", OE / "op_evolve" / "core" / "hints.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                       # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def sections(lines):
    out, cur = {}, None
    for ln in lines:
        m = SEC.match(ln)
        if m:
            cur = m.group(1).lower()
            out[cur] = [ln]
        elif cur:
            out[cur].append(ln)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint", required=True, type=Path)
    ap.add_argument("--add", required=True, type=Path)
    ap.add_argument("--retire", action="append", default=[], help="hN=STATUS")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    text = a.hint.read_text()
    lines = text.splitlines(True)
    add = a.add.read_text().splitlines(True)
    add_rows = [ln for ln in add if ROW.match(ln)]
    add_secs = sections([ln for ln in add if not ROW.match(ln)])
    have_rows = {ROW.match(ln).group(1).lower() for ln in lines if ROW.match(ln)}
    have_secs = {SEC.match(ln).group(1).lower() for ln in lines if SEC.match(ln)}
    clash = [ROW.match(r).group(1) for r in add_rows if ROW.match(r).group(1).lower() in have_rows]
    clash += [s for s in add_secs if s in have_secs]
    if clash:
        raise SystemExit(f"already in {a.hint}: {sorted(set(clash))} -- renumber the new hints")

    # end of the index table = the first contiguous block of table lines that contains an `| hN |` row
    first = next((i for i, ln in enumerate(lines) if ROW.match(ln)), None)
    if first is None:
        raise SystemExit(f"{a.hint}: no `| hN |` index table found")
    end = first
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    new = lines[:end] + [r if r.endswith("\n") else r + "\n" for r in add_rows] + lines[end:]

    retire = dict(x.split("=", 1) for x in a.retire)
    for i, ln in enumerate(new):
        m = ROW.match(ln)
        if m and m.group(1).lower() in retire:
            cells = ln.rstrip("\n").strip().strip("|").split("|")
            cells = [c.strip() for c in cells]
            cells[1] = " ".join(t for t in cells[1].split() if t.lower() != "standing")
            cells[3] = retire[m.group(1).lower()]
            new[i] = "| " + " | ".join(cells) + " |\n"
    if not new[-1].endswith("\n"):
        new[-1] += "\n"
    for sid, body in add_secs.items():
        new += ["\n"] + [b if b.endswith("\n") else b + "\n" for b in body]
    out = "".join(new)

    parsed = {h.id: h for h in oe_hints().parse(out)}
    for r in add_rows:
        hid = ROW.match(r).group(1).lower()
        h = parsed.get(hid)
        if h is None:
            raise SystemExit(f"{hid} does not parse after the edit")
        print(f"# {hid}: level={h.level} kind={h.kind} standing={h.standing} status={h.status!r} "
              f"body={len(h.body)} chars")
        if not h.body:
            raise SystemExit(f"{hid} has no section")
    for hid, st in retire.items():
        h = parsed.get(hid)
        if h is None or h.standing or h.status != st:
            raise SystemExit(f"retire {hid} did not take: {h}")
        print(f"# {hid}: retired, level={h.level} kind={h.kind} standing={h.standing} status={h.status!r}")
    sys.stdout.writelines(difflib.unified_diff(lines, out.splitlines(True), "hint.md", "hint.md (edited)", n=1))
    if not a.write:
        print("\n(dry run: nothing written; add --write)")
        return 0
    bak = a.hint.with_name(f"hint.md.bak.pre-gb-{time.strftime('%m%d-%H%M%S')}")
    shutil.copy2(a.hint, bak)
    a.hint.write_text(out)
    print(f"\nwritten {a.hint}; backup {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
