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
``.flags`` (the exact command) and ``.meta.json`` (sha256, toolchain, per-kernel resources, ok). The
lab launchers (``llama/lab``, ``mla/lab``) refuse a code object whose meta.json is not ok or whose
bytes are not the recorded ones. ``-cuid=<kernel>`` pins the ``__hip_cuid_*`` symbol, so the bytes do
not depend on the build directory, and a ``sha256`` in the ladder file proves a rung rebuilt the
exact code object that was measured when the rung was first written.

The compiler runs in a container that maps no device (default ``lhz-hk-cpu``, image
amdprimus/amdprimus:tas-20261019, AMD clang 862306352beb, the toolchain every pin was made with);
the build refuses a container that maps any device. ``--local`` runs ``amdclang++`` from PATH instead.
"""

import argparse
import hashlib
import json
import os
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
    while r is not None:
        if r.get("patch"):
            chain.append(spec["_dir"] / r["patch"])
        r = spec["_by_id"].get(r.get("parent")) if r.get("parent") else None
    return chain[::-1]


def materialize(spec, rung, out_root):
    """The rung's source tree: the ladder's tree itself when no patch applies, else a patched copy."""
    base = spec["_dir"] / spec["tree"]
    chain = patch_chain(spec, rung)
    if not chain:
        return base
    dst = out_root / "trees" / rung["id"]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base, dst)
    for p in chain:
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-p1", str(p)], cwd=dst,
                           capture_output=True, text=True)  # fmt: skip
        if r.returncode:
            raise BuildError(f"rung {rung['id']}: patch {p.name} does not apply: {r.stderr.strip()[:2000]}")
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
    cmd = ["amdclang++", *spec["flags"], f"-cuid={name}", *WARN_FLAGS, *kspec.get("extra_flags", []),
           "-c", kspec["src"], "-o", str(tmp / f"{name}.hsaco")]  # fmt: skip
    (out_dir / f"{name}.flags").write_text(" ".join(cmd) + "\n")
    t0 = time.time()
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
            if k["scratch"] or k["vgpr_spill"] or k["sgpr_spill"] or k["dynamic_stack"]:
                errors.append(f"{kn}: scratch {k['scratch']} vgpr_spill {k['vgpr_spill']} "
                              f"sgpr_spill {k['sgpr_spill']} dynamic_stack {k['dynamic_stack']}")  # fmt: skip
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
        for name, kspec in rung["kernels"].items():
            meta, secs = build_kernel(spec, rung, name, kspec, tree, out_root / rung["id"], runner, tc)
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
