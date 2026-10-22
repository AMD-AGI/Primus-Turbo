###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build, verify and locate the gfx1250 HK attention code objects (``bin/<kernel>.hsaco``).

The kernels are HIP sources (``hip/src``, headers in ``hip/include``) built one code object per
``manifest.json`` kernel entry, device code only, with exactly the manifest's flags (never Turbo's
global csrc flags, which define ``KITTENS_CDNA4``). The manifest pins the sha256 of every code object
as built and validated with its ``toolchain``; a code object with another sha256 (another compiler,
edited sources) is never loaded, and neither is a kernel whose entry has no pin yet
(``hsaco_sha256: null``, a kernel still in development): the HIPKITTENS backend then runs the FlyDSL
kernels for that direction. ``-cuid=<kernel>`` pins the ``__hip_cuid_*`` symbol, so a rebuild in any
directory is byte-identical; ``-save-temps=obj`` is part of that identity (the code object is
assembled from the saved ``.s``, which differs in bytes from the integrated path), so its temporaries
go to a scratch directory.

Manifest (schema 2; schema 1 is the dsv3 MLA package's, read the same way): ``flags``, ``toolchain``
and per kernel entry ``src``, ``extra_flags``, ``module`` (the Python kernel module that launches it:
``fwd_pp``, ``fwd_v0``, ``bwd_v0``), ``hsaco_sha256`` and the launch metadata the pin tool reads from
the code object (``block``, ``kernarg_size``, ``lds``, ``vgpr``, ``sgpr``, ``args``). Pinning is one
command (``tools/hk_gfx1250_attn_pin.py``, which compiles with ``compile_kernel`` below).

    python primus_turbo/hipkittens/gfx1250_attn/hsaco.py [--check] [--force] [--strict] [--cxx PATH]
                                                          [--kernels a,b]

setup.py runs ``build()`` when GPU_ARCHS contains gfx1250. Standard library only: setup.py loads
this file by path, and the gate imports it without torch or flydsl.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
HIP_DIR = PACKAGE_DIR / "hip"
BIN_DIR = PACKAGE_DIR / "bin"
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
CXX_ENV = "PRIMUS_TURBO_HK_CXX"
# Seconds one kernel may take to compile before the build gives up on it (a warning, never a failed
# install): the largest one takes about a minute.
COMPILE_TIMEOUT_ENV = "PRIMUS_TURBO_HK_COMPILE_TIMEOUT"
DEFAULT_COMPILE_TIMEOUT = 1800
SCHEMAS = (1, 2)
PIN_TOOL = "tools/hk_gfx1250_attn_pin.py"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_MANIFEST = None
# (binary dir, kernels or None) -> None (every code object present with the manifest's sha256) or
# the reason it is not. A plain dict, not functools.lru_cache: the gate runs inside torch.compile'd
# regions.
_REASONS: Dict[Tuple[str, Optional[Tuple[str, ...]]], Optional[str]] = {}


def load_manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads(MANIFEST_PATH.read_text())
    return _MANIFEST


def reload_manifest() -> None:
    """Forget the cached manifest and every cached verdict (after the pin tool rewrote it)."""
    global _MANIFEST
    _MANIFEST = None
    _REASONS.clear()


def binary_path(name: str, bin_dir: Optional[Path] = None) -> Path:
    """``<bin_dir>/<name>.hsaco``; ``bin_dir`` None: this package's ``bin/``."""
    return Path(BIN_DIR if bin_dir is None else bin_dir) / f"{name}.hsaco"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_pinned(entry: dict) -> bool:
    """Whether a manifest kernel entry carries a sha256 pin (null: not validated yet)."""
    sha = entry.get("hsaco_sha256")
    return isinstance(sha, str) and _SHA256.match(sha) is not None


def kernel_names(module: Optional[str] = None) -> List[str]:
    """Manifest kernel names, in manifest order; only those launched by ``module`` when given.
    Empty when there is no manifest."""
    if not MANIFEST_PATH.is_file():
        return []
    kernels = load_manifest()["kernels"]
    return [n for n, k in kernels.items() if module is None or k.get("module") == module]


def unavailable_reason(
    bin_dir: Optional[Path] = None, kernels: Optional[Iterable[str]] = None
) -> Optional[str]:
    """None when every code object (or those named in ``kernels``) is in ``bin_dir`` (None: ``bin/``)
    with the manifest's sha256, else why not. Reads the files once per directory, kernel set and process."""
    bin_dir = BIN_DIR if bin_dir is None else bin_dir
    names = None if kernels is None else tuple(kernels)
    key = (str(bin_dir), names)
    if key in _REASONS:
        return _REASONS[key]
    _REASONS[key] = reason = _unavailable_reason(Path(bin_dir), names)
    return reason


def check_pins(bin_dir: Optional[Path] = None, kernels: Optional[Iterable[str]] = None) -> Optional[str]:
    """``unavailable_reason`` read from disk now (not cached): for the check right before a code object
    is loaded."""
    return _unavailable_reason(
        Path(BIN_DIR if bin_dir is None else bin_dir), None if kernels is None else tuple(kernels)
    )


def _unavailable_reason(bin_dir: Path, names: Optional[Tuple[str, ...]]) -> Optional[str]:
    if not MANIFEST_PATH.is_file():
        return f"{MANIFEST_PATH.name} is missing (no pinned kernels yet)"
    m = load_manifest()
    if m.get("schema") not in SCHEMAS:
        return f"{MANIFEST_PATH.name} schema {m.get('schema')!r} is not one of {SCHEMAS}"
    entries = m["kernels"]
    if names is not None and not names:
        return "no kernels to check"
    for name in entries if names is None else names:
        k = entries.get(name)
        if k is None:
            return f"{name} is not listed in {MANIFEST_PATH.name}"
        if not is_pinned(k):
            return f"{name} is not pinned in {MANIFEST_PATH.name} (validate it, then run {PIN_TOOL})"
        path = binary_path(name, bin_dir)
        if not path.is_file():
            return f"{path.name} is not built (build Primus-Turbo with GPU_ARCHS=gfx1250, or run hsaco.py)"
        got = _sha256(path)
        if got != k["hsaco_sha256"]:
            return (
                f"{path.name} sha256 {got[:16]} != {k['hsaco_sha256'][:16]}: built by another toolchain or "
                f"from edited sources (validated with: {m.get('toolchain')})"
            )
    return None


def code_object(name: str, hsaco_dir=None) -> Tuple[Path, Optional[str]]:
    """(path, sha256 to enforce) of one kernel's code object: ``<hsaco_dir>/<name>.hsaco`` unpinned
    (None) when a directory is given (lab), else this package's ``bin/<name>.hsaco`` with the
    manifest's pin (ValueError when the kernel is unknown or unpinned). For kernel modules that load
    with ``hiprt.load_hsaco(path, specs, sha256)``."""
    if hsaco_dir is not None:
        return binary_path(name, Path(hsaco_dir)), None
    k = load_manifest()["kernels"].get(name) if MANIFEST_PATH.is_file() else None
    if k is None or not is_pinned(k):
        raise ValueError(f"{name} has no sha256 pin in {MANIFEST_PATH} (run {PIN_TOOL} after validating it)")
    return binary_path(name, BIN_DIR), k["hsaco_sha256"]


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


def compile_command(cxx: str, name: str, out_path: Path) -> List[str]:
    """The exact compiler command of one kernel (run in hip/: the manifest's -I paths are relative)."""
    m = load_manifest()
    k = m["kernels"][name]
    return [cxx, *m["flags"], f"-cuid={name}", *k.get("extra_flags", []), "-c", k["src"], "-o", str(out_path)]


def compile_timeout() -> float:
    """$PRIMUS_TURBO_HK_COMPILE_TIMEOUT seconds, else ``DEFAULT_COMPILE_TIMEOUT``."""
    try:
        return float(os.environ.get(COMPILE_TIMEOUT_ENV) or DEFAULT_COMPILE_TIMEOUT)
    except ValueError:
        return float(DEFAULT_COMPILE_TIMEOUT)


def compile_kernel(cxx: str, name: str, out_path: Path, timeout: Optional[float] = None) -> None:
    """One kernel entry -> one device-only code object, cwd = hip/. RuntimeError on a compile error or
    after ``timeout`` seconds (default ``compile_timeout()``)."""
    timeout = compile_timeout() if timeout is None else timeout
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / f"{name}.hsaco"
        cmd = compile_command(cxx, name, obj)
        # its own process group: on timeout the compiler's children (which hold the pipes) die with it
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=HIP_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as e:
            raise RuntimeError(f"{name}: cannot run {cmd[0]}: {e}") from e
        with proc:
            try:
                _, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise RuntimeError(f"{name}: {' '.join(cmd)} did not finish in {timeout:g} s") from None
        if proc.returncode != 0:
            raise RuntimeError(f"{name}: {' '.join(cmd)} failed ({proc.returncode}):\n{err[-4000:]}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(obj, out_path)


def build(
    cxx: Optional[str] = None,
    bin_dir: Optional[Path] = None,
    force: bool = False,
    log=print,
    kernels: Optional[Iterable[str]] = None,
) -> bool:
    """Build every pinned kernel the manifest lists (or those in ``kernels``) into ``bin_dir``; True
    when all of them match their pins.

    A code object already present with the manifest's sha256 is kept (unless ``force``). One that
    comes out with another sha256 is deleted, so it can never be loaded; that direction then falls
    back to FlyDSL. Kernels without a pin are not built (nothing would load them). No compiler, or one
    that cannot compile them: nothing is built (the same fallback). ``bin_dir`` None: ``bin/``."""
    bin_dir = Path(BIN_DIR if bin_dir is None else bin_dir)
    if not MANIFEST_PATH.is_file():
        log(f"[gfx1250_attn] no {MANIFEST_PATH.name}: nothing to build")
        return True
    entries = load_manifest()["kernels"]
    names = list(entries) if kernels is None else list(kernels)
    unknown = [n for n in names if n not in entries]
    if unknown:
        log(f"[gfx1250_attn] WARNING: not in {MANIFEST_PATH.name}: {unknown}")
    unpinned = [n for n in names if n in entries and not is_pinned(entries[n])]
    if unpinned:
        log(f"[gfx1250_attn] not pinned yet, not built: {', '.join(unpinned)}")
    pinned = [n for n in names if n in entries and is_pinned(entries[n])]
    todo = [
        n
        for n in pinned
        if force
        or not binary_path(n, bin_dir).is_file()
        or _sha256(binary_path(n, bin_dir)) != entries[n]["hsaco_sha256"]
    ]
    for key in [k for k in _REASONS if k[0] == str(bin_dir)]:
        del _REASONS[key]
    if not todo:
        log(f"[gfx1250_attn] code objects up to date in {bin_dir} ({len(pinned)} pinned)")
        return not unknown
    cxx = find_compiler(cxx)
    if cxx is None:
        log(f"[gfx1250_attn] WARNING: amdclang++ not found (set {CXX_ENV}); HIP attention kernels not built")
        return False
    ok = not unknown
    for name in todo:
        out = binary_path(name, bin_dir)
        if not (HIP_DIR / entries[name]["src"]).is_file():
            ok = False
            log(f"[gfx1250_attn] WARNING: {name}: source {entries[name]['src']} is missing")
            continue
        try:
            compile_kernel(cxx, name, out)
        except RuntimeError as e:  # e.g. a compiler without the gfx1250 builtins: never fail the install
            ok = False
            log(f"[gfx1250_attn] WARNING: {e}")
            continue
        got = _sha256(out)
        if got != entries[name]["hsaco_sha256"]:
            out.unlink()
            ok = False
            log(
                f"[gfx1250_attn] WARNING: {name} sha256 {got[:16]} != manifest "
                f"{entries[name]['hsaco_sha256'][:16]} with {compiler_version(cxx)}; removed "
                f"(validated with: {load_manifest().get('toolchain')})"
            )
        else:
            log(f"[gfx1250_attn] built {out.name} ({got[:16]})")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only verify bin/ against the manifest")
    ap.add_argument("--force", action="store_true", help="rebuild code objects that already match")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every code object matches")
    ap.add_argument("--cxx", default=None, help=f"compiler (default: ${CXX_ENV}, PATH, $ROCM_PATH)")
    ap.add_argument("--bin-dir", default=None, help="default: this package's bin/")
    ap.add_argument("--kernels", default=None, help="comma-separated kernel names (default: all)")
    a = ap.parse_args(argv)
    kernels = [k for k in a.kernels.split(",") if k] if a.kernels else None
    if a.check:
        reason = unavailable_reason(a.bin_dir and Path(a.bin_dir), kernels)
        print(reason or "ok")
        return 1 if reason else 0
    ok = build(a.cxx, a.bin_dir and Path(a.bin_dir), a.force, kernels=kernels)
    return 0 if ok or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
