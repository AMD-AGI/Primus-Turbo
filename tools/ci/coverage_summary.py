###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render coverage.py JSON as a compact Markdown table for the CI run summary.

Usage:
    coverage_summary.py REPORT.json [REPORT.json ...]
"""

import json
import sys
from collections import defaultdict

PKG = "primus_turbo/"
# Kernel layers excluded from the headline number (JIT-compiled; Python line
# coverage is not meaningful). Kept in sync with [tool.coverage.report] omit.
OMIT_MODULES = {"triton", "flydsl"}
# Top-level groups whose sub-packages are shown as indented detail rows; every
# other group (common, ...) is a single bold row.
DETAILED_GROUPS = ("pytorch", "jax")


def classify(path: str):
    """Return (group, detail) for a covered file, or None to skip it.

    group is the top-level row key (e.g. pytorch); detail is the sub-row key for
    DETAILED_GROUPS (e.g. pytorch/ops), else None.
    """
    seg = (path[path.find(PKG) :] if PKG in path else path).split("/")
    if seg[-1] == "__init__.py":
        return None
    if len(seg) < 2:
        return "(top-level)", None
    if len(seg) == 2:  # primus_turbo/<file>.py
        return seg[1], None
    if seg[1] in DETAILED_GROUPS:
        return seg[1], seg[1] + "/" + seg[2]
    return seg[1], None


def _pct(covered: int, total: int) -> float:
    return (100.0 * covered / total) if total else 0.0


def _merge_files(reports: list) -> dict:
    """Union of files across reports; per file take the max covered lines.

    Approximate fallback for the multi-report case only. num_statements is
    stable across reports for the same source file, so the union denominator is
    well-defined. Taking the max covered lines avoids double counting a module
    both jobs import, but undercounts when two reports cover disjoint lines of
    the same file. CI feeds a single already-combined report (line-level union),
    so this path is not exercised there.
    """
    merged = {}
    for rep in reports:
        for fpath, info in rep.get("files", {}).items():
            s = info["summary"]
            cur = merged.get(fpath)
            if cur is None:
                merged[fpath] = [s["covered_lines"], s["num_statements"]]
            else:
                cur[0] = max(cur[0], s["covered_lines"])
                cur[1] = max(cur[1], s["num_statements"])
    return merged


def _aggregate(merged: dict):
    """Return {group: {detail|group: [covered, statements]}} for kept modules."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for fpath, (cov, stmts) in merged.items():
        result = classify(fpath)
        if result is None:
            continue
        group, detail = result
        if group in OMIT_MODULES:
            continue
        a = agg[group][detail or group]
        a[0] += cov
        a[1] += stmts
    return agg


def render(reports: list) -> str:
    agg = _aggregate(_merge_files(reports))

    def row(label, cov, stmts, bold=False):
        vals = [format(cov, ","), format(stmts, ","), "%.1f%%" % _pct(cov, stmts)]
        w = "**" if bold else ""
        return "| " + " | ".join("%s%s%s" % (w, x, w) for x in [label] + vals) + " |"

    def group_totals(group):
        cov = sum(v[0] for v in agg[group].values())
        stmts = sum(v[1] for v in agg[group].values())
        return cov, stmts

    groups = [g for g in agg if group_totals(g)[1] > 0]
    tc = sum(group_totals(g)[0] for g in groups)
    tn = sum(group_totals(g)[1] for g in groups)
    excl = ", ".join(sorted(OMIT_MODULES))

    out = ["## Primus-Turbo coverage\n"]
    out.append(
        "**Total line coverage: %.1f%%** (%s / %s statements; excludes %s)\n"
        % (_pct(tc, tn), format(tc, ","), format(tn, ","), excl)
    )
    out += ["| Module | Covered | Stmts | Coverage |", "|---|--:|--:|--:|"]

    # Top-level groups, sorted by coverage (desc).
    for group in sorted(groups, key=lambda g: -_pct(*group_totals(g))):
        cov, stmts = group_totals(group)
        out.append(row("`%s`" % group, cov, stmts, bold=True))
        if group in DETAILED_GROUPS:
            details = ((k, v) for k, v in agg[group].items() if v[1] > 0)
            for k, v in sorted(details, key=lambda kv: -_pct(kv[1][0], kv[1][1])):
                out.append(row("&emsp;`%s`" % k, v[0], v[1]))

    out.append(row("TOTAL", tc, tn, bold=True))
    return "\n".join(out)


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: coverage_summary.py REPORT.json [REPORT.json ...]",
            file=sys.stderr,
        )
        return 2
    reports = [_load(p) for p in sys.argv[1:]]
    print(render(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
