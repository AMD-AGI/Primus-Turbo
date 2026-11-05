###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build the gfx1250 MLA attention code objects into ``primus_turbo/lib/hk_gfx1250_mla/<kernel>.hsaco``.

Each kernel is one device-only code object compiled from ``src/`` with exactly ``manifest.json``'s flags (not the
csrc extension's). The manifest pins every code object's sha256 for its ``toolchain``: only a code object with the
pinned bytes is installed, and one that is not (an older build) is removed, so the runtime loads whatever is present
and a pass whose code objects are missing runs on FlyDSL. ``-cuid=<kernel>`` makes the build byte-identical in any
directory. A kernel with an ``asm_patch`` is assembled from the compiler's ``.s`` (``-save-temps=obj``) with the
patch applied, and only from the ``.s`` whose sha256 is ``asm_input_sha256`` (``tools/regen_asm_patch.py``).

    python csrc/kernels/attention/hipkittens/gfx1250_mla/hsaco.py [--check] [--force] [--strict] [--cxx PATH]

setup.py runs ``build()`` when GPU_ARCHS contains gfx1250. Standard library only: setup.py loads it by path.
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
from typing import Iterable, Optional, Sequence

HIP_DIR = Path(__file__).resolve().parent  # compile cwd: src/, include/, asm/
MANIFEST_PATH = HIP_DIR / "manifest.json"
BIN_DIR = HIP_DIR.parents[4] / "primus_turbo" / "lib" / "hk_gfx1250_mla"
CXX_ENV = "PRIMUS_TURBO_HK_CXX"
FWD_KERNELS = ("k_mla_fwd",)
BWD_KERNELS = ("k_delta", "k_dkdv96ds", "k_dqds_c")


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def binary_path(name: str, bin_dir: Path = BIN_DIR) -> Path:
    return Path(bin_dir) / f"{name}.hsaco"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unavailable_reason(names: Iterable[str], bin_dir: Path = BIN_DIR) -> Optional[str]:
    """None when every kernel in ``names`` is in ``bin_dir`` with its pinned sha256, else why not."""
    kernels = load_manifest()["kernels"]
    for name in names:
        path = binary_path(name, bin_dir)
        if not path.is_file():
            return f"{path.name} is not built"
        if _sha256(path) != kernels[name]["hsaco_sha256"]:
            return f"{path.name} is not the pinned code object"
    return None


def find_compiler(cxx: Optional[str] = None) -> Optional[str]:
    """amdclang++: ``cxx``, $PRIMUS_TURBO_HK_CXX, PATH, then $ROCM_PATH / $ROCM_HOME / /opt/rocm."""
    for c in (cxx, os.environ.get(CXX_ENV)):
        if c:
            return shutil.which(c)
    found = shutil.which("amdclang++")
    if found:
        return found
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"), "/opt/rocm"):
        if root and (Path(root) / "llvm" / "bin" / "amdclang++").is_file():
            return str(Path(root) / "llvm" / "bin" / "amdclang++")
    return None


def compiler_version(cxx: str) -> str:
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return "?"
    return out.splitlines()[0].strip() if out else "?"


def compile_command(cxx: str, name: str, obj: Path, extra_flags: Optional[Sequence[str]] = None) -> list:
    """The compiler command of one kernel (``extra_flags``: instead of the manifest's); run it with cwd = HIP_DIR."""
    m = load_manifest()
    k = m["kernels"][name]
    extra = k["extra_flags"] if extra_flags is None else extra_flags
    return [cxx, *m["flags"], f"-cuid={name}", *extra, "-c", k["src"], "-o", str(obj)]


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
    """``cmd`` (-save-temps=obj) left the compiler's .s in ``tmp``; it must have ``asm_input_sha256``. Applies the
    patch and reruns the driver's own assembler and linker (from ``cmd -###``) into ``cmd``'s output."""
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


def compile_kernel(cxx: str, name: str, out_path: Path) -> None:
    """One kernel -> one device-only code object at ``out_path``."""
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


def build(cxx: Optional[str] = None, bin_dir: Path = BIN_DIR, force: bool = False, log=print) -> bool:
    """Build every kernel into ``bin_dir``; True when every code object there has its pinned sha256.

    Code objects already pinned are kept (unless ``force``). A new one is installed only when it has the pinned
    sha256, and an unpinned one left in ``bin_dir`` is removed. Without the manifest's ``toolchain`` (it cannot
    reproduce the pins) nothing is compiled. Never raises for a kernel, so it cannot fail an install."""
    bin_dir = Path(bin_dir)
    kernels = load_manifest()["kernels"]
    todo = [n for n in kernels if force or unavailable_reason((n,), bin_dir) is not None]
    if todo:
        cxx = find_compiler(cxx)
        version = compiler_version(cxx) if cxx else None
        if cxx is None:
            log(f"[gfx1250_mla] WARNING: amdclang++ not found (set {CXX_ENV}); kernels not built")
        elif version != load_manifest()["toolchain"]:
            log(f"[gfx1250_mla] WARNING: {cxx} is {version!r}, not the pinned toolchain; kernels not built")
        else:
            for name in todo:
                out = binary_path(name, bin_dir)
                tmp = out.with_name(out.name + ".tmp")
                try:
                    compile_kernel(cxx, name, tmp)
                    got, pin = _sha256(tmp), kernels[name]["hsaco_sha256"]
                    if got != pin:
                        raise RuntimeError(f"sha256 {got[:16]} is not the pinned {pin[:16]}")
                    os.replace(tmp, out)
                    log(f"[gfx1250_mla] built {out.name}")
                except Exception as e:  # noqa: BLE001 -- a kernel that does not build must not fail the install
                    log(f"[gfx1250_mla] WARNING: {name} not built: {type(e).__name__}: {e}")
                finally:
                    tmp.unlink(missing_ok=True)
    ok = True
    for name in kernels:
        reason = unavailable_reason((name,), bin_dir)
        if reason is not None:
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
        reasons = {
            p: unavailable_reason(ks, a.bin_dir)
            for p, ks in (("forward", FWD_KERNELS), ("backward", BWD_KERNELS))
        }
        for p, reason in reasons.items():
            print(f"{p}: {reason or 'ok'}")
        return 1 if any(reasons.values()) else 0
    ok = build(a.cxx, Path(a.bin_dir), a.force)
    return 0 if ok or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
