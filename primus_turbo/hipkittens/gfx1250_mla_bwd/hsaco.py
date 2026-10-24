###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build, verify and locate the gfx1250 MLA backward code objects (``bin/<kernel>.hsaco``).

The kernels are HIP sources (``hip/src``, headers in ``hip/include``) built one kernel per code
object, device code only, with exactly ``manifest.json``'s flags (never Turbo's global csrc flags,
which define ``KITTENS_CDNA4``). ``manifest.json`` pins the sha256 of every code object as built
and validated with its ``toolchain``; a code object with another sha256 (another compiler, edited
sources) is never loaded, so the backward falls back to the FlyDSL kernels. ``-cuid=<kernel>``
pins the ``__hip_cuid_*`` symbol, so a rebuild in any directory is byte-identical; ``-save-temps=obj``
is part of that identity (the code object is assembled from the saved ``.s``, which differs in bytes
from the integrated path), so its temporaries go to a scratch directory. A kernel with an ``asm_patch``
(k_dkdv96ds) is assembled from that ``.s`` with the patch applied, and only from the ``.s`` whose
sha256 is ``asm_input_sha256``; ``tools/regen_asm_patch.py`` regenerates the patch (see README.md).
A kernel marked ``optional`` (k_delta, which replaces the FlyDSL k_delta) may be missing or foreign without taking
the others down: ``unavailable_reason`` covers the required kernels only, and the backward runs the FlyDSL kernel
in its place (``kernel_unavailable_reason``).

    python primus_turbo/hipkittens/gfx1250_mla_bwd/hsaco.py [--check] [--force] [--strict] [--cxx PATH]

setup.py runs ``build()`` when GPU_ARCHS contains gfx1250. Standard library only: setup.py loads
this file by path, and the backward's gate imports it without torch or flydsl.
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
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
HIP_DIR = PACKAGE_DIR / "hip"
BIN_DIR = PACKAGE_DIR / "bin"
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
CXX_ENV = "PRIMUS_TURBO_HK_CXX"

_MANIFEST = None
# (binary dir, kernel) -> None (the code object is present with the manifest's sha256) or the reason it is not.
# A plain dict, not functools.lru_cache: the gate runs inside torch.compile'd regions.
_REASONS: Dict[Tuple[str, str], Optional[str]] = {}


def load_manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads(MANIFEST_PATH.read_text())
    return _MANIFEST


def binary_path(name: str, bin_dir: Path = BIN_DIR) -> Path:
    return Path(bin_dir) / f"{name}.hsaco"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_unavailable_reason(name: str, bin_dir: Path = BIN_DIR) -> Optional[str]:
    """None when kernel ``name``'s code object is in ``bin_dir`` with the manifest's sha256, else why not.
    Reads the file once per directory, kernel and process."""
    key = (str(bin_dir), name)
    if key in _REASONS:
        return _REASONS[key]
    k = load_manifest()["kernels"][name]
    path = binary_path(name, bin_dir)
    reason = None
    if not path.is_file():
        reason = f"{path.name} is not built (build Primus-Turbo with GPU_ARCHS=gfx1250, or run hsaco.py)"
    else:
        got = _sha256(path)
        if got != k["hsaco_sha256"]:
            reason = (
                f"{path.name} sha256 {got[:16]} != {k['hsaco_sha256'][:16]}: built by another toolchain or "
                f"from edited sources (validated with: {load_manifest()['toolchain']})"
            )
    _REASONS[key] = reason
    return reason


def unavailable_reason(bin_dir: Path = BIN_DIR) -> Optional[str]:
    """None when every required (not ``optional``) code object is in ``bin_dir`` with the manifest's sha256,
    else why not."""
    for name, k in load_manifest()["kernels"].items():
        if not k.get("optional"):
            reason = kernel_unavailable_reason(name, bin_dir)
            if reason is not None:
                return reason
    return None


def find_compiler(cxx: Optional[str] = None) -> Optional[str]:
    """amdclang++: ``cxx``, $PRIMUS_TURBO_HK_CXX, PATH, then $ROCM_PATH / $ROCM_HOME / /opt/rocm."""
    for c in (cxx, os.environ.get(CXX_ENV)):
        if c:
            return c
    found = shutil.which("amdclang++")
    if found:
        return found
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"), "/opt/rocm"):
        if root:
            p = Path(root) / "llvm" / "bin" / "amdclang++"
            if p.is_file():
                return str(p)
    return None


def compiler_version(cxx: str) -> str:
    out = subprocess.run([cxx, "--version"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0].strip() if out else "?"


def compile_command(cxx: str, name: str, obj: Path) -> list:
    """The compiler command of one kernel; run it with cwd = hip/ (the manifest's -I paths are relative)."""
    m = load_manifest()
    k = m["kernels"][name]
    return [cxx, *m["flags"], f"-cuid={name}", *k["extra_flags"], "-c", k["src"], "-o", str(obj)]


def compile_kernel(cxx: str, name: str, out_path: Path) -> None:
    """One kernel -> one device-only code object."""
    k = load_manifest()["kernels"][name]
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / f"{name}.hsaco"
        cmd = compile_command(cxx, name, obj)
        r = subprocess.run(cmd, cwd=HIP_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{name}: {' '.join(cmd)} failed ({r.returncode}):\n{r.stderr[-4000:]}")
        if "asm_patch" in k:
            _patch_and_reassemble(name, k, cmd, Path(tmp))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(obj, out_path)


def _apply_unified_diff(text: str, patch: str) -> str:
    """Apply a one-file unified diff with exact context (no fuzz); raises ValueError on any mismatch."""
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
                    raise ValueError(f"asm patch does not apply at line {pos + 1}")
                if line[0] == " ":
                    out.append(src[pos])
                pos += 1
            elif line[:1] == "+":
                out.append(line[1:])
            i += 1
    out.extend(src[pos:])
    return "\n".join(out)


def _patch_and_reassemble(name: str, k: dict, cmd: list, tmp: Path) -> None:
    """``asm_patch`` kernels: the compile above left the compiler's own .s in ``tmp`` (-save-temps=obj). It must have
    ``asm_input_sha256`` (else the toolchain or the sources changed: RuntimeError, nothing is built, the backward
    falls back to FlyDSL); the patched .s is assembled and linked with the driver's own commands (from -###)."""
    s_files = list(tmp.glob(f"{name}-hip-*.s"))
    if len(s_files) != 1:
        raise RuntimeError(f"{name}: expected one saved .s, found {s_files}")
    got = _sha256(s_files[0])
    if got != k["asm_input_sha256"]:
        raise RuntimeError(
            f"{name}: compiler .s sha256 {got[:16]} != asm_input_sha256 {k['asm_input_sha256'][:16]}"
        )
    try:
        patched = _apply_unified_diff(s_files[0].read_text(), (HIP_DIR / k["asm_patch"]).read_text())
    except ValueError as e:
        raise RuntimeError(f"{name}: {e}") from e
    s_files[0].write_text(patched)
    dry = subprocess.run(cmd + ["-###"], cwd=HIP_DIR, capture_output=True, text=True).stderr
    steps = [shlex.split(line) for line in dry.splitlines() if line.startswith(' "')]
    for step in (next(s for s in steps if "-cc1as" in s), next(s for s in steps if s[0].endswith("lld"))):
        r = subprocess.run(step, cwd=HIP_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{name}: {' '.join(step)} failed ({r.returncode}):\n{r.stderr[-4000:]}")


def build(cxx: Optional[str] = None, bin_dir: Path = BIN_DIR, force: bool = False, log=print) -> bool:
    """Build every kernel the manifest lists into ``bin_dir``; True when all match the manifest.

    A code object already present with the manifest's sha256 is kept (unless ``force``). One that
    comes out with another sha256 is deleted, so it can never be loaded; the backward then falls
    back to FlyDSL (for an optional kernel: that kernel alone). No compiler, or one that cannot
    compile them: nothing is built (the same fallback)."""
    bin_dir = Path(bin_dir)
    kernels = load_manifest()["kernels"]
    todo = [
        n
        for n, k in kernels.items()
        if force
        or not binary_path(n, bin_dir).is_file()
        or _sha256(binary_path(n, bin_dir)) != k["hsaco_sha256"]
    ]
    for key in [key for key in _REASONS if key[0] == str(bin_dir)]:
        del _REASONS[key]
    if not todo:
        log(f"[gfx1250_mla_bwd] code objects up to date in {bin_dir}")
        return True
    cxx = find_compiler(cxx)
    if cxx is None:
        log(f"[gfx1250_mla_bwd] WARNING: amdclang++ not found (set {CXX_ENV}); HIP MLA backward not built")
        return False
    ok = True
    for name in todo:
        out = binary_path(name, bin_dir)
        try:
            compile_kernel(cxx, name, out)
        except RuntimeError as e:  # e.g. a compiler without the gfx1250 builtins: never fail the install
            ok = False
            log(f"[gfx1250_mla_bwd] WARNING: {e}")
            continue
        got = _sha256(out)
        if got != kernels[name]["hsaco_sha256"]:
            out.unlink()
            ok = False
            log(
                f"[gfx1250_mla_bwd] WARNING: {name} sha256 {got} != manifest "
                f"{kernels[name]['hsaco_sha256'][:16]} with {compiler_version(cxx)}; removed "
                f"(validated with: {load_manifest()['toolchain']})"
            )
        else:
            log(f"[gfx1250_mla_bwd] built {out.name} ({got[:16]})")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only verify bin/ against the manifest")
    ap.add_argument("--force", action="store_true", help="rebuild code objects that already match")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every code object matches")
    ap.add_argument("--cxx", default=None, help=f"compiler (default: ${CXX_ENV}, PATH, $ROCM_PATH)")
    ap.add_argument("--bin-dir", default=str(BIN_DIR))
    a = ap.parse_args(argv)
    if a.check:
        reason = unavailable_reason(Path(a.bin_dir))
        print(reason or "ok")
        for name, k in load_manifest()["kernels"].items():
            note = kernel_unavailable_reason(name, Path(a.bin_dir)) if k.get("optional") else None
            if note is not None:
                print(f"optional {name}: {note} (the FlyDSL kernel runs in its place)")
        return 1 if reason else 0
    ok = build(a.cxx, Path(a.bin_dir), a.force)
    return 0 if ok or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
