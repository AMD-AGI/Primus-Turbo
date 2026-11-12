###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build the gfx1250 MLA attention backward code objects into ``primus_turbo/lib/hk_gfx1250_mla/<kernel>.hsaco``.

Each kernel is one device-only code object compiled from ``src/`` with exactly ``manifest.json``'s flags (not the
csrc extension's). The manifest pins every code object's sha256 for its ``toolchain``: only a code object with the
pinned bytes is installed, and one that is not (an older build) is removed. ``-cuid=<kernel>`` makes the build
byte-identical in any directory.

    python csrc/kernels/attention/hipkittens/gfx1250_mla/hsaco.py [--check] [--force] [--strict] [--cxx PATH]

setup.py runs ``build()`` when GPU_ARCHS contains gfx1250. Standard library only: setup.py loads it by path.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

HIP_DIR = Path(__file__).resolve().parent  # compile cwd: src/, include/
MANIFEST_PATH = HIP_DIR / "manifest.json"
BIN_DIR = HIP_DIR.parents[4] / "primus_turbo" / "lib" / "hk_gfx1250_mla"
CXX_ENV = "PRIMUS_TURBO_HK_CXX"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def binary_path(name: str, bin_dir: Path = BIN_DIR) -> Path:
    return Path(bin_dir) / f"{name}.hsaco"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unavailable_reason(bin_dir: Path = BIN_DIR, names: Optional[list] = None) -> Optional[str]:
    """None when every kernel in ``names`` (default: all) is in ``bin_dir`` with its pinned sha256, else why not."""
    kernels = load_manifest()["kernels"]
    for name in kernels if names is None else names:
        path = binary_path(name, bin_dir)
        if not path.is_file():
            return f"{path.name} is not built"
        if _sha256(path) != kernels[name]["hsaco_sha256"]:
            return f"{path.name} is not the pinned code object"
    return None


def find_compiler(cxx: Optional[str] = None) -> Optional[str]:
    """amdclang++ as an absolute path: ``cxx`` or $PRIMUS_TURBO_HK_CXX when given, else the first of PATH, $ROCM_PATH,
    $ROCM_HOME and /opt/rocm that is the manifest's toolchain (the first one found when none is)."""
    for c in (cxx, os.environ.get(CXX_ENV)):
        if c:
            found = shutil.which(c)
            return os.path.abspath(found) if found else None
    roots = (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"), "/opt/rocm")
    candidates = [shutil.which("amdclang++")] + [
        shutil.which(str(Path(r) / "llvm" / "bin" / "amdclang++")) for r in roots if r
    ]
    found = list(dict.fromkeys(os.path.abspath(c) for c in candidates if c))
    toolchain = load_manifest()["toolchain"]
    return next((c for c in found if compiler_version(c) == toolchain), found[0] if found else None)


def compiler_version(cxx: str) -> str:
    try:
        r = subprocess.run([cxx, "--version"], capture_output=True, text=True, errors="replace", check=True)
    except (OSError, subprocess.SubprocessError):
        return "?"
    return r.stdout.splitlines()[0].strip() if r.stdout else "?"


def compile_command(cxx: str, name: str, obj: Path) -> list:
    """The compiler command of one kernel; run it with cwd = HIP_DIR."""
    m = load_manifest()
    k = m["kernels"][name]
    return [cxx, *m["flags"], f"-cuid={name}", *k["extra_flags"], "-c", k["src"], "-o", str(obj)]


def compile_kernel(cxx: str, name: str, out_path: Path) -> None:
    """One kernel -> one device-only code object at ``out_path``."""
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / f"{name}.hsaco"
        cmd = compile_command(cxx, name, obj)
        r = subprocess.run(cmd, cwd=HIP_DIR, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            raise RuntimeError(
                f"{name}: {' '.join(cmd)} failed ({r.returncode}):\n{(r.stdout + r.stderr)[-4000:]}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(obj, out_path)


def build(cxx: Optional[str] = None, bin_dir: Path = BIN_DIR, force: bool = False, log=print) -> bool:
    """Build every kernel into ``bin_dir``; True when every code object there has its pinned sha256.

    Code objects already pinned are kept (unless ``force``). A new one is installed only when it has the pinned
    sha256, and an unpinned one left in ``bin_dir`` is removed. Without the manifest's ``toolchain`` (it cannot
    reproduce the pins) nothing is compiled. Never raises for a kernel, so it cannot fail an install."""
    bin_dir = Path(bin_dir)
    manifest = load_manifest()
    kernels = manifest["kernels"]
    todo = [n for n in kernels if force or unavailable_reason(bin_dir, [n]) is not None]
    if todo:
        cxx = find_compiler(cxx)
        version = compiler_version(cxx) if cxx else None
        if cxx is None:
            log(f"[gfx1250_mla] WARNING: amdclang++ not found (set {CXX_ENV}); kernels not built")
        elif version != manifest["toolchain"]:
            log(f"[gfx1250_mla] WARNING: {cxx} is {version!r}, not the pinned toolchain; kernels not built")
        else:
            for name in todo:
                out = binary_path(name, bin_dir)
                tmp = out.with_name(out.name + ".tmp")
                try:
                    compile_kernel(cxx, name, tmp)
                    got, pin = _sha256(tmp), kernels[name]["hsaco_sha256"]
                    if got != pin:
                        raise RuntimeError(f"sha256 {got} is not the pinned {pin}")
                    os.replace(tmp, out)
                    log(f"[gfx1250_mla] built {out.name}")
                except Exception as e:  # noqa: BLE001 -- a kernel that does not build must not fail the install
                    log(f"[gfx1250_mla] WARNING: {name} not built: {type(e).__name__}: {e}")
                finally:
                    tmp.unlink(missing_ok=True)
    ok = True
    for name in kernels:
        if unavailable_reason(bin_dir, [name]) is not None:
            ok = False
            if binary_path(name, bin_dir).is_file():
                binary_path(name, bin_dir).unlink()
                log(f"[gfx1250_mla] removed {name}.hsaco: not the pinned code object")
    if ok and not todo:
        log(f"[gfx1250_mla] code objects up to date in {bin_dir}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only verify the installed code objects")
    ap.add_argument("--force", action="store_true", help="rebuild code objects that are already pinned")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every code object is pinned")
    ap.add_argument("--cxx", default=None, help=f"compiler (default: ${CXX_ENV}, PATH, $ROCM_PATH)")
    ap.add_argument("--bin-dir", default=str(BIN_DIR))
    a = ap.parse_args(argv)
    if a.check:
        reason = unavailable_reason(Path(a.bin_dir))
        print(reason or "ok")
        return 1 if reason else 0
    ok = build(a.cxx, Path(a.bin_dir), a.force)
    return 0 if ok or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
