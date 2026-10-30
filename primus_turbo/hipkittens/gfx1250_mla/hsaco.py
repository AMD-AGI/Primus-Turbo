###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build, verify and locate the gfx1250 MLA attention code objects (``bin/<kernel>.hsaco``).

Every kernel is one device-only code object built from ``hip/src`` with exactly ``manifest.json``'s flags (never
Turbo's csrc flags or headers). The manifest pins the sha256 of every code object as validated with its
``toolchain``; a code object with other bytes (another compiler, edited sources) is never loaded, and the pass it
belongs to runs on FlyDSL. ``-cuid=<kernel>`` pins the ``__hip_cuid_*`` symbol, so a rebuild in any directory is
byte-identical; ``-save-temps=obj`` is part of that identity (the code object is assembled from the saved ``.s``),
so its temporaries go to a scratch directory. A kernel with an ``asm_patch`` is assembled from the compiler's ``.s``
with the patch applied, and only from the ``.s`` whose sha256 is ``asm_input_sha256``
(``tools/regen_asm_patch.py`` regenerates a patch, see README.md).

    python primus_turbo/hipkittens/gfx1250_mla/hsaco.py [--check] [--force] [--strict] [--cxx PATH]

setup.py runs ``build()`` when GPU_ARCHS contains gfx1250. Standard library only: setup.py loads this file by path,
and the attention gate imports it without torch or flydsl.
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
from typing import Dict, Iterable, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
HIP_DIR = PACKAGE_DIR / "hip"
BIN_DIR = PACKAGE_DIR / "bin"
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
CXX_ENV = "PRIMUS_TURBO_HK_CXX"
FWD_KERNELS = ("k_mla_fwd",)
BWD_KERNELS = ("k_delta", "k_dkdv96ds", "k_dqds_c")

_MANIFEST = None
# (binary dir, kernel) -> None (the code object is present with the manifest's sha256) or why it is not.
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
    if key not in _REASONS:
        path = binary_path(name, bin_dir)
        want = load_manifest()["kernels"][name]["hsaco_sha256"]
        reason = None
        if not path.is_file():
            reason = f"{path.name} is not built (build Primus-Turbo with GPU_ARCHS=gfx1250, or run hsaco.py)"
        elif _sha256(path) != want:
            reason = (
                f"{path.name} sha256 {_sha256(path)[:16]} != {want[:16]}: built by another toolchain or from "
                f"edited sources (validated with: {load_manifest()['toolchain']})"
            )
        _REASONS[key] = reason
    return _REASONS[key]


def unavailable_reason(names: Iterable[str], bin_dir: Path = BIN_DIR) -> Optional[str]:
    """None when every kernel in ``names`` is available (``kernel_unavailable_reason``), else the first reason."""
    for name in names:
        reason = kernel_unavailable_reason(name, bin_dir)
        if reason is not None:
            return reason
    return None


def find_compiler(cxx: Optional[str] = None) -> Optional[str]:
    """amdclang++: ``cxx``, $PRIMUS_TURBO_HK_CXX, PATH, then $ROCM_PATH / $ROCM_HOME / /opt/rocm; None when the
    first one given is not an executable."""
    for c in (cxx, os.environ.get(CXX_ENV)):
        if c:
            return shutil.which(c)
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
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return "?"
    return out.splitlines()[0].strip() if out else "?"


def compile_command(cxx: str, name: str, obj: Path, extra_flags: Optional[Sequence[str]] = None) -> list:
    """The compiler command of one kernel (``extra_flags``: instead of the manifest's); run it with cwd = hip/."""
    m = load_manifest()
    k = m["kernels"][name]
    extra = k["extra_flags"] if extra_flags is None else extra_flags
    return [cxx, *m["flags"], f"-cuid={name}", *extra, "-c", k["src"], "-o", str(obj)]


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
            patch_and_reassemble(name, k, cmd, Path(tmp))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(obj, out_path)


def apply_unified_diff(text: str, patch: str) -> str:
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


def patch_and_reassemble(name: str, k: dict, cmd: list, tmp: Path) -> None:
    """``asm_patch`` kernels: ``cmd`` left the compiler's own .s in ``tmp`` (-save-temps=obj). It must have
    ``asm_input_sha256`` (else the toolchain or the sources changed: RuntimeError, nothing is built); the patched
    .s is assembled and linked with the driver's own commands (from -###) into ``cmd``'s output."""
    s_files = list(tmp.glob(f"{name}-hip-*.s"))
    if len(s_files) != 1:
        raise RuntimeError(f"{name}: expected one saved .s, found {s_files}")
    got = _sha256(s_files[0])
    if got != k["asm_input_sha256"]:
        raise RuntimeError(
            f"{name}: compiler .s sha256 {got[:16]} != asm_input_sha256 {k['asm_input_sha256'][:16]}"
        )
    try:
        patched = apply_unified_diff(s_files[0].read_text(), (HIP_DIR / k["asm_patch"]).read_text())
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
    """Build every kernel of the manifest into ``bin_dir``; True when all match the manifest.

    A code object already present with the manifest's sha256 is kept (unless ``force``). A new one replaces it only
    when it has the manifest's sha256; one with other bytes is discarded, and that pass runs on FlyDSL. No compiler,
    or one that cannot compile the kernels: nothing is built (the same fallback). Never raises for a kernel, so it
    cannot fail an install."""
    bin_dir = Path(bin_dir)
    kernels = load_manifest()["kernels"]
    todo = [n for n, k in kernels.items() if force or kernel_unavailable_reason(n, bin_dir) is not None]
    for key in [key for key in _REASONS if key[0] == str(bin_dir)]:
        del _REASONS[key]
    if not todo:
        log(f"[gfx1250_mla] code objects up to date in {bin_dir}")
        return True
    cxx = find_compiler(cxx)
    if cxx is None:
        log(f"[gfx1250_mla] WARNING: amdclang++ not found (set {CXX_ENV}); HIP MLA kernels not built")
        return False
    ok = True
    for name in todo:
        out = binary_path(name, bin_dir)
        tmp = out.with_name(out.name + ".tmp")
        try:
            compile_kernel(cxx, name, tmp)
            got, want = _sha256(tmp), kernels[name]["hsaco_sha256"]
            if got != want:
                raise RuntimeError(
                    f"sha256 {got[:16]} != manifest {want[:16]} with {compiler_version(cxx)} (validated with: "
                    f"{load_manifest()['toolchain']})"
                )
            os.replace(tmp, out)
            log(f"[gfx1250_mla] built {out.name} ({got[:16]})")
        except Exception as e:  # noqa: BLE001 -- a kernel that does not build must not fail the install
            ok = False
            log(f"[gfx1250_mla] WARNING: {name} not built: {type(e).__name__}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only verify bin/ against the manifest")
    ap.add_argument("--force", action="store_true", help="rebuild code objects that already match")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every code object matches")
    ap.add_argument("--cxx", default=None, help=f"compiler (default: ${CXX_ENV}, PATH, $ROCM_PATH)")
    ap.add_argument("--bin-dir", default=str(BIN_DIR))
    a = ap.parse_args(argv)
    bin_dir = Path(a.bin_dir)
    if a.check:
        reasons = {
            p: unavailable_reason(ks, bin_dir)
            for p, ks in (("forward", FWD_KERNELS), ("backward", BWD_KERNELS))
        }
        for p, reason in reasons.items():
            print(f"{p}: {reason or 'ok'}")
        return 1 if any(reasons.values()) else 0
    ok = build(a.cxx, bin_dir, a.force)
    return 0 if ok or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
