#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Make an MLA backward rung whose dK/dV source changed: compile it, regenerate its asm patch, write the rung.

    python3 tools/mkrung.py --rung r8c --parent r8a --tree <edited tree> [-D NAME=V ...] [--no-asm] [--side]
        [--feature TEXT] [--bitwise-with ID] [--build]

<edited tree> is a copy of build/mla_bwd/trees/<parent> with the change applied. The tool
  1. compiles its k_dkdv96ds with the parent's flags plus -D ... in lhz-hk-cpu (the exact hkl.build command), keeping
     the compiler's .s;
  2. unless --no-asm: runs tools/warv_regen.py --mode llvm --msrc on that .s (Q^T re-target incl. memory-source
     hazards, va_vdst / vm_vsrc recomputed, every final check against the compiler's .s) and writes the patch into the
     tree as asm/k_dkdv96ds_<rung>.s.patch;
  3. writes mla/patches/<rung>.patch (the tree against the parent's) and the rung into mla/ladder_bwd.json (the
     parent's kernels, dK/dV flags + -D, asm patch + asm_input_sha256, no sha256 pin; k_dqds_c / k_delta keep the
     parent's flags and pins);
  4. with --build: hkl.build --rung <rung> and hkl.kdcheck against the parent.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TURBO = Path(os.environ.get("HKL_TURBO_TOOLS", "/home/lihuzhan/projects/2026_1023__v2/Primus-Turbo-hk-b1"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("-D", dest="defs", action="append", default=[])
    ap.add_argument("--no-asm", action="store_true")
    ap.add_argument("--no-msrc", action="store_true", help="asm patch v1 (va_vdst near-drain re-targets only)")
    ap.add_argument("--side", action="store_true")
    ap.add_argument("--feature", default="")
    ap.add_argument("--bitwise-with")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    ladder = HERE / "mla/ladder_bwd.json"
    spec = json.loads(ladder.read_text())
    by = {r["id"]: r for r in spec["rungs"]}
    parent = by[a.parent]
    tree = Path(a.tree).resolve()
    ptree = HERE / "build/mla_bwd/trees" / a.parent
    if not ptree.is_dir():
        sys.exit(f"{ptree}: build the parent first")
    k = copy.deepcopy(parent["kernels"]["k_dkdv96ds"])
    flags = [f for f in k["extra_flags"] if not f.startswith("-D")] + [f"-D{d}" for d in a.defs]
    asm_name = f"asm/k_dkdv96ds_{a.rung}.s.patch"
    (tree / asm_name).unlink(missing_ok=True)
    work = HERE / "build/mkrung" / a.rung
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cmd = (["amdclang++"] + spec["flags"] + [f"-cuid=k_dkdv96ds"] + flags + ["-c", k["src"], "-o", str(work / "k.hsaco")])
    cmd = [c.replace("-save-temps=obj", "-save-temps=obj") for c in cmd]
    r = subprocess.run(["docker", "exec", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-w", str(tree),
                        "lhz-hk-cpu"] + cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"compile failed:\n{r.stderr[-3000:]}")
    s_file = next(work.glob("*.s"))
    sha = hashlib.sha256(s_file.read_bytes()).hexdigest()
    print(f"[mkrung] compiled {k['src']} {' '.join(a.defs)}: .s sha {sha[:16]}")
    for f in ("sha256", "asm_patch", "asm_input_sha256"):
        k.pop(f, None)
    k["extra_flags"] = flags
    if not a.no_asm:
        r = subprocess.run([sys.executable, str(HERE / "tools/warv_regen.py"), str(s_file), "--out", str(tree / asm_name),
                            "--mode", "llvm", "--turbo", str(TURBO)] + ([] if a.no_msrc else ["--msrc"]),
                           capture_output=True, text=True)
        tail = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "warv_regen]" in l or "warning" in l
                         or "Error" in l or "failed" in l)
        print(tail)
        if r.returncode:
            sys.exit("asm patch regeneration failed")
        k["asm_patch"] = asm_name
        k["asm_input_sha256"] = sha
    d = subprocess.run(["diff", "-ruN", "a", "b"], cwd=work, capture_output=True)  # placeholder cwd for labels
    shutil.rmtree(work / "a", ignore_errors=True)
    shutil.rmtree(work / "b", ignore_errors=True)
    shutil.copytree(ptree, work / "a")
    shutil.copytree(tree, work / "b", ignore=shutil.ignore_patterns("*.o", "*.s", "*.bc", "*.hipi", "*.hsaco", "*.txt"))
    d = subprocess.run(["diff", "-ruN", "a", "b"], cwd=work, capture_output=True, text=True)
    if d.returncode not in (0, 1):
        sys.exit(d.stderr)
    pfile = HERE / f"mla/patches/{a.rung}.patch"
    pfile.write_text(d.stdout)
    rung = {"id": a.rung, "parent": a.parent, "patch": f"patches/{a.rung}.patch",
            "kernels": dict(copy.deepcopy(parent["kernels"]), k_dkdv96ds=k), "launch": copy.deepcopy(parent["launch"]),
            "feature": a.feature}
    if a.side:
        rung["side"] = True
    if a.bitwise_with:
        rung["bitwise_with"] = a.bitwise_with
    rungs = [x for x in spec["rungs"] if x["id"] != a.rung]
    at = [x["id"] for x in rungs].index(a.parent) + 1
    while at < len(rungs) and rungs[at].get("parent") == a.parent and rungs[at]["id"] < a.rung:
        at += 1
    rungs.insert(at, rung)
    spec["rungs"] = rungs
    ladder.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"[mkrung] wrote {pfile.name} ({len(d.stdout.splitlines())} lines) and rung {a.rung}")
    if a.build:
        for c in (["-m", "hkl.build", str(ladder), "--rung", a.rung, "--check"],
                  ["-m", "hkl.kdcheck", str(ladder), "--ref", a.parent, "--rung", a.rung]):
            r = subprocess.run([sys.executable] + c, cwd=HERE, capture_output=True, text=True)
            print("\n".join(l for l in r.stdout.splitlines() if "k_dkdv" in l or "FAIL" in l))
            if r.returncode:
                sys.exit("build / kdcheck failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
