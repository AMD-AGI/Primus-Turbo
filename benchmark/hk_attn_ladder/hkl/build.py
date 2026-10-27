###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build the rungs of an HK attention ladder into gfx1250 code objects, device-only, with no GPU.

    python3 -m hkl.build llama/ladder_fwd.json                 # every rung
    python3 -m hkl.build llama/ladder_fwd.json --rung F1 F3    # some rungs
    python3 -m hkl.build mla/ladder_bwd.json --check           # exit 1 unless every pinned sha matches

A ladder file names a source tree (``tree``: a directory next to it, e.g. ``hip``), the compile flags
shared by every rung (``flags``, run inside the tree so the relative ``-I`` paths resolve) and the
rungs in ladder order. A rung is ``{"id", "kernels": {name: {"src", "extra_flags"[, "sha256"]}}}``
plus optional ``"parent"`` and ``"patch"``: the rung's source tree is its parent's tree with the
rung's patch applied (``git apply``), so a rung differs from the one below it by exactly that patch
and its flags. Rungs without ``patch`` build from their parent's tree (the ladder's ``tree`` at the
root).

Each kernel is compiled once per rung into ``<out>/<ladder>/<rung>/``:
``<kernel>.hsaco``, ``.s`` (the compiler's own assembly, from -save-temps), ``.dis`` (llvm-objdump),
``.flags`` (the exact command), ``.meta.json`` (sha256, toolchain, per-kernel resources, ok) and, when the
rung's ``launch`` has a ``launch_json``, ``<kernel>.launch.json`` (fwd_v1's schedule mode). The
lab launchers (``llama/lab``, ``mla/lab``) refuse a code object whose meta.json is not ok or whose
bytes are not the recorded ones. ``-cuid=<kernel>`` pins the ``__hip_cuid_*`` symbol, so the bytes do
not depend on the build directory, and a ``sha256`` in the ladder file proves a rung rebuilt the
exact code object that was measured when the rung was first written.

A kernel with ``"gen": [args]`` is not compiled from ``src``: the rung's tree holds a generator package (the MLA
forward's ``gfx1250_mla_fwd/`` next to ``asmkit/``, or ``gen_pkg``), and ``gen/build.py --out <tmp> --asmkit-root
<tree> <args>`` builds it (marker shell compiled by LLVM, generated hot-loop regions spliced in, assembled, verified);
the rung takes its ``k_mla_fwd.hsaco``, or the HIP reference variant's ``ref/k_mla_fwd.hsaco`` with ``"gen_ref":
true``. The generator's own build.json is kept as ``<kernel>.gen.json``.

The compiler runs in a container that maps no device (default ``lhz-hk-cpu``, image
amdprimus/amdprimus:tas-20261019, AMD clang 862306352beb, the toolchain every pin was made with);
the build refuses a container that maps any device. ``--local`` runs ``amdclang++`` from PATH instead.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import codeobj

HERE = Path(__file__).resolve().parent.parent  # benchmark/hk_attn_ladder
DEFAULT_OUT = HERE / "build"
DEFAULT_CONTAINER = "lhz-hk-cpu"
TOOLCHAIN = "862306352beb9946c78b3e355d0d5201bbccfa3a"
LLVM_BIN = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/llvm/bin"
# Warnings that point at undefined behaviour fail a build (the flags below do not change the code).
WARN_FLAGS = ["-Wuninitialized", "-Wsometimes-uninitialized"]
UB_WARNINGS = (
    "-Wuninitialized", "-Wsometimes-uninitialized", "-Wconditional-uninitialized", "-Warray-bounds",
    "-Wshift-count-overflow", "-Wshift-count-negative", "-Wshift-negative-value", "-Wshift-overflow",
    "-Winteger-overflow", "-Wreturn-type", "-Wnull-dereference", "-Wdivision-by-zero", "-Wunsequenced",
    "-Wdangling", "-Wreturn-stack-address", "-Wvla",
)  # fmt: skip


class BuildError(RuntimeError):
    pass


def load_ladder(path):
    path = Path(path).resolve()
    with open(path) as f:
        spec = json.load(f)
    spec["_path"] = path
    spec["_dir"] = path.parent
    ids = [r["id"] for r in spec["rungs"]]
    if len(set(ids)) != len(ids):
        raise BuildError(f"{path}: duplicate rung ids")
    by_id = {r["id"]: r for r in spec["rungs"]}
    for r in spec["rungs"]:
        p = r.get("parent")
        if p is not None and p not in by_id:
            raise BuildError(f"{path}: rung {r['id']} has unknown parent {p}")
    spec["_by_id"] = by_id
    return spec


def ladder_name(spec):
    return spec.get("name") or spec["_path"].stem


def patch_chain(spec, rung):
    """Patches from the ladder's base tree up to ``rung``, oldest first."""
    chain, r = [], rung
    while r is not None and not r.get("tree"):
        if r.get("patch"):
            chain.append(spec["_dir"] / r["patch"])
        r = spec["_by_id"].get(r.get("parent")) if r.get("parent") else None
    return chain[::-1]


def materialize(spec, rung, out_root):
    """The rung's source tree: the ladder's tree itself when no patch applies, else a patched copy. A rung with
    its own ``tree`` (relative to the ladder file, e.g. the MLA ladder's new bottom rungs built from the Llama
    tree's hk_attn primitives) builds from that directory as is."""
    if rung.get("tree"):
        if rung.get("patch"):
            raise BuildError(f"rung {rung['id']}: a rung with its own tree takes no patch")
        return (spec["_dir"] / rung["tree"]).resolve()
    base = spec["_dir"] / spec["tree"]
    chain = patch_chain(spec, rung)
    if not chain:
        return base
    dst = out_root / "trees" / rung["id"]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base, dst)
    for p in chain:
        # GNU patch, not git apply: inside a git work tree (build/ is one) git apply silently skips every path
        # outside the current directory. Exact application only: no fuzz, rejects fail the build.
        r = subprocess.run(["patch", "-p1", "-s", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch",
                            "-r", "-", "-i", str(p)], cwd=dst, capture_output=True, text=True)  # fmt: skip
        if r.returncode:
            raise BuildError(f"rung {rung['id']}: patch {p.name} does not apply: {(r.stdout + r.stderr).strip()[:2000]}")
    return dst


def _runner(args):
    if args.local:
        return []
    ct = args.container
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}|{{range .HostConfig.Devices}}"
                        "{{.PathOnHost}} {{end}}|{{.HostConfig.Privileged}}", ct],
                       capture_output=True, text=True)  # fmt: skip
    if r.returncode:
        raise BuildError(f"container {ct} not found: start it with tools/start_cpu_container.sh")
    running, devs, priv = r.stdout.strip().split("|")
    if running != "true":
        raise BuildError(f"container {ct} is not running (docker start {ct})")
    if devs.strip() or priv == "true":
        raise BuildError(f"container {ct} maps devices ({devs.strip() or 'privileged'}); refusing to build in it")
    return ["docker", "exec", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", ct]


def _run(runner, cmd, cwd, timeout=900):
    full = runner + (["sh", "-c", 'cd "$0" && exec "$@"', str(cwd), *cmd] if runner else cmd)
    return subprocess.run(full, cwd=None if runner else cwd, capture_output=True, text=True, timeout=timeout)


def toolchain(runner):
    r = _run(runner, ["amdclang++", "--version"], "/")
    if r.returncode:
        raise BuildError(f"amdclang++ --version failed: {r.stderr.strip()}")
    return r.stdout.splitlines()[0].strip()


def apply_unified_diff(text, patch):
    """Apply a one-file unified diff with exact context (no fuzz); BuildError on any mismatch.
    (The applier of primus_turbo/hipkittens/gfx1250_mla_bwd/hsaco.py.)"""
    src, out, pos = text.split("\n"), [], 0
    lines, i = patch.split("\n"), 0
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1
    while i < len(lines):
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
        i += 1
        if not m:
            continue
        start = int(m.group(1)) - 1 if int(m.group(2) or 1) > 0 else int(m.group(1))
        out.extend(src[pos:start])
        pos = start
        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            if line[:1] in (" ", "-"):
                if pos >= len(src) or src[pos] != line[1:]:
                    raise BuildError(f"asm patch does not apply at line {pos + 1}")
                if line[0] == " ":
                    out.append(src[pos])
                pos += 1
            elif line[:1] == "+":
                out.append(line[1:])
            i += 1
    out.extend(src[pos:])
    return "\n".join(out)


def _patch_and_reassemble(name, kspec, cmd, tree, tmp, runner):
    """A kernel with ``asm_patch``: the compile left the compiler's own .s in ``tmp`` (-save-temps=obj). It must
    have ``asm_input_sha256``; the patched .s is assembled and linked with the driver's own commands (-###)."""
    s_files = sorted(tmp.glob(f"{name}-hip-*.s"))
    if len(s_files) != 1:
        raise BuildError(f"{name}: expected one saved .s, found {[p.name for p in s_files]}")
    got = hashlib.sha256(s_files[0].read_bytes()).hexdigest()
    if got != kspec["asm_input_sha256"]:
        raise BuildError(f"{name}: compiler .s sha256 {got[:16]} != asm_input_sha256 {kspec['asm_input_sha256'][:16]}")
    s_files[0].write_text(apply_unified_diff(s_files[0].read_text(), (Path(tree) / kspec["asm_patch"]).read_text()))
    dry = _run(runner, cmd + ["-###"], tree)
    steps = [shlex.split(line) for line in dry.stderr.splitlines() if line.startswith(' "')]
    for step in (next(s for s in steps if "-cc1as" in s), next(s for s in steps if s[0].endswith("lld"))):
        r = _run(runner, step, tree)
        if r.returncode:
            raise BuildError(f"{name}: {step[0]} failed ({r.returncode}): {r.stderr[-2000:]}")


def _gen_build(kspec, tree, tmp, runner):
    """A generator kernel (module docstring): gen/build.py in the tree's package; returns (cmd, process, hsaco, .s)."""
    pkg = Path(tree) / kspec.get("gen_pkg", "gfx1250_mla_fwd")
    gen_out = tmp / "gen"
    cmd = ["env", "PYTHONDONTWRITEBYTECODE=1", "python3", "gen/build.py", "--out", str(gen_out), "--asmkit-root",
           str(tree), *kspec["gen"]]  # fmt: skip
    r = _run(runner, cmd, pkg)
    sub = gen_out / "ref" if kspec.get("gen_ref") else gen_out
    return cmd, r, sub / "k_mla_fwd.hsaco", sub / "k_mla_fwd.s", gen_out / "build.json"


def build_kernel(spec, rung, name, kspec, tree, out_dir, runner, tc):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".tmp_{name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    hsaco = out_dir / f"{name}.hsaco"
    for p in (hsaco, out_dir / f"{name}.meta.json"):
        if p.exists():
            p.unlink()
    gen = kspec.get("gen") is not None
    if not gen:
        cmd = ["amdclang++", *spec["flags"], f"-cuid={name}", *WARN_FLAGS, *kspec.get("extra_flags", []),
               "-c", kspec["src"], "-o", str(tmp / f"{name}.hsaco")]  # fmt: skip
        (out_dir / f"{name}.flags").write_text(" ".join(cmd) + "\n")
    t0 = time.time()
    if gen:
        cmd, r, gen_hsaco, gen_s, gen_json = _gen_build(kspec, tree, tmp, runner)
        (out_dir / f"{name}.flags").write_text(" ".join(cmd) + "\n")
        if not r.returncode:
            if gen_hsaco.is_file():
                shutil.copy2(gen_hsaco, tmp / f"{name}.hsaco")
                s_src = gen_s if gen_s.is_file() else next(iter(sorted(gen_s.parent.rglob("*-hip-*.s"))), None)
                if s_src is not None:  # the REF variant keeps only the compiler's -save-temps .s
                    shutil.copy2(s_src, tmp / f"{name}.gen.s")
                shutil.copy2(gen_json, out_dir / f"{name}.gen.json")
            else:
                r = subprocess.CompletedProcess(cmd, 1, r.stdout, r.stderr + f"\nno {gen_hsaco}")
    else:
        r = _run(runner, cmd, tree)
    secs = time.time() - t0
    (out_dir / f"{name}.log").write_text(r.stdout + r.stderr)
    errors = []
    if r.returncode:
        errors.append(f"compile rc={r.returncode}")
    ub = [ln for ln in r.stderr.splitlines() if "warning:" in ln and any(f"[{w}" in ln for w in UB_WARNINGS)]
    if ub:
        errors.append(f"{len(ub)} UB-class warning(s): {ub[0][:200]}")
    meta = {"name": name, "rung": rung["id"], "ladder": ladder_name(spec), "toolchain": tc, "flags": cmd,
            "tree": str(tree), "ok": False, "errors": errors, "kernels": {}}  # fmt: skip
    if not r.returncode and kspec.get("asm_patch") and not gen:
        try:
            meta["asm_patch"] = kspec["asm_patch"]
            _patch_and_reassemble(name, kspec, cmd, tree, tmp, runner)
        except BuildError as e:
            errors.append(str(e))
            r = subprocess.CompletedProcess(cmd, 1, r.stdout, r.stderr)
    if not r.returncode:
        shutil.copy2(tmp / f"{name}.hsaco", hsaco)
        s = sorted(tmp.glob("*.s"))
        if s:
            shutil.copy2(s[0], out_dir / f"{name}.s")
        else:
            errors.append("no -save-temps assembly")
        dis = _run(runner, [f"{LLVM_BIN}/llvm-objdump", "-d", "--mcpu=gfx1250", str(hsaco)], "/")
        (out_dir / f"{name}.dis").write_text(dis.stdout)
        meta["hsaco_sha256"] = hashlib.sha256(hsaco.read_bytes()).hexdigest()
        meta["kernels"] = codeobj.kernels(str(hsaco))
        for kn, k in meta["kernels"].items():
            # SGPR spills go to VGPR lanes (no memory traffic) and are tolerated, as in the shipped MLA dK/dV
            # kernel; anything that reaches scratch memory fails the build.
            if k["scratch"] or k["vgpr_spill"] or k["dynamic_stack"]:
                errors.append(f"{kn}: scratch {k['scratch']} vgpr_spill {k['vgpr_spill']} "
                              f"dynamic_stack {k['dynamic_stack']}")  # fmt: skip
    if TOOLCHAIN not in tc:
        errors.append(f"toolchain is not {TOOLCHAIN[:12]}: {tc}")
    want = kspec.get("sha256")  # full sha256, or a prefix of >= 12 hex digits where only that was recorded
    got = meta.get("hsaco_sha256")
    if want and len(want) < 12:
        errors.append(f"sha256 pin {want!r} is shorter than 12 hex digits")
    meta["expected_sha256"] = want
    meta["pin"] = "none" if not want else ("match" if got and got.startswith(want) else "MISMATCH")
    meta["ok"] = not errors
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return meta, secs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladder", help="ladder json (e.g. llama/ladder_fwd.json)")
    ap.add_argument("--rung", nargs="*", help="rung ids to build (default: all)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"output root (default {DEFAULT_OUT})")
    ap.add_argument("--container", default=DEFAULT_CONTAINER)
    ap.add_argument("--local", action="store_true", help="run amdclang++ from PATH, not in the container")
    ap.add_argument("--check", action="store_true", help="exit 1 unless every build is ok and every pin matches")
    a = ap.parse_args(argv)

    spec = load_ladder(a.ladder)
    rungs = spec["rungs"] if not a.rung else [spec["_by_id"][i] for i in a.rung]
    out_root = Path(a.out).resolve() / ladder_name(spec)
    runner = _runner(a)
    tc = toolchain(runner)
    print(f"ladder {ladder_name(spec)}: {len(rungs)} rung(s), {tc}")
    bad = 0
    for rung in rungs:
        try:
            tree = materialize(spec, rung, out_root)
        except BuildError as e:
            print(f"  {rung['id']:<6} FAIL {e}")
            bad += 1
            continue
        lj = rung.get("launch", {}).get("launch_json")
        for name, kspec in rung["kernels"].items():
            meta, secs = build_kernel(spec, rung, name, kspec, tree, out_root / rung["id"], runner, tc)
            if lj is not None:  # launch configuration the rung's lab module reads next to the code object
                (out_root / rung["id"] / f"{name}.launch.json").write_text(json.dumps(lj) + "\n")
            res = " ".join(f"{kn}:v{k['vgpr']}/s{k['sgpr']}/lds{k['lds']}" for kn, k in meta["kernels"].items())
            sha = meta.get("hsaco_sha256", "")[:8]
            state = "ok" if meta["ok"] else "FAIL " + "; ".join(meta["errors"])
            print(f"  {rung['id']:<6} {name:<16} {state:<4} sha {sha or '-':<8} pin {meta['pin']:<8} "
                  f"{secs:4.1f}s  {res}")  # fmt: skip
            bad += (not meta["ok"]) or meta["pin"] == "MISMATCH"
    if bad:
        print(f"{bad} rung kernel(s) failed or mismatched")
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
