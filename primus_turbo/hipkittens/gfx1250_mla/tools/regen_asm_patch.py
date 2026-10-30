###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regenerate the assembly patches hsaco.py applies (``hip/asm/<kernel>.s.patch``) and their manifest sha256s.

Two kernels are generated: ``k_mla_fwd`` (``tools/fwdgen``) and ``k_dkdv96ds`` (``tools/bwdgen``). Compiled with the
manifest's flags, their HIP source is a marker shell: the workgroup set-up, the TDM descriptors and the uniform
control flow, with inline-asm markers (``; HKGEN <region> ...``) where the generated code goes. A generator compiles
the shell (``hsaco.compile_command``), emits the regions (``tools/asmkit`` derives ``s_set_vgpr_msb``, the
expert-mode waits and the hazard ``v_nop``s from the instruction stream), splices and assembles them and checks the
result; the patch is the difference between the compiler's ``.s`` and the spliced ``.s``.

Per kernel:
  1. compile exactly as hsaco.py does and keep the compiler's ``.s``: its sha256 is ``asm_input_sha256``;
  2. run the generator (``<generator>/build.py --out DIR --cxx CXX``); its ``shell.s`` must be that very ``.s``;
  3. patch = unified diff (compiler ``.s`` -> generated ``.s``); hsaco.py must reproduce the generated ``.s`` from
     it and assemble the generator's code object byte for byte (``hsaco_sha256``);
  4. write the patch and both sha256 values.

    python primus_turbo/hipkittens/gfx1250_mla/tools/regen_asm_patch.py [KERNEL ...] [--check] [--cxx PATH]

``--check`` writes nothing and fails unless every regenerated patch and sha256 equals the committed one. CPU only;
needs the pinned toolchain (manifest ``toolchain``).
"""

import argparse
import difflib
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
GENERATORS = {"k_mla_fwd": TOOLS_DIR / "fwdgen" / "build.py", "k_dkdv96ds": TOOLS_DIR / "bwdgen" / "build.py"}


def _load_hsaco():
    spec = importlib.util.spec_from_file_location("_gfx1250_mla_hsaco", TOOLS_DIR.parent / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compile(hsaco, cxx, name, out_dir: Path):
    """hsaco.py's compile of ``name`` into ``out_dir``: (command, the compiler's .s)."""
    out_dir.mkdir(parents=True)
    cmd = hsaco.compile_command(cxx, name, out_dir / f"{name}.hsaco")
    r = subprocess.run(cmd, cwd=hsaco.HIP_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: compile failed:\n{r.stderr[-4000:]}")
    s_files = list(out_dir.glob(f"{name}-hip-*.s"))
    if len(s_files) != 1:
        raise RuntimeError(f"{name}: expected one saved .s, found {s_files}")
    return cmd, s_files[0]


def regenerate(hsaco, cxx, name, tmp: Path, keep=None):
    """(patch text, asm_input_sha256, hsaco_sha256) of kernel ``name``; raises RuntimeError on a failed step."""
    k = hsaco.load_manifest()["kernels"][name]
    _, s_in = _compile(hsaco, cxx, name, tmp / "cc")
    sha_in = _sha256(s_in)
    out = tmp / "gen"
    r = subprocess.run(
        [sys.executable, str(GENERATORS[name]), "--out", str(out), "--cxx", cxx],
        capture_output=True,
        text=True,
    )
    print(r.stdout[-2000:], end="")
    if r.returncode != 0:
        raise RuntimeError(f"{name}: generator failed:\n{r.stderr[-4000:]}")
    if _sha256(out / "shell.s") != sha_in:
        raise RuntimeError(f"{name}: the generator's shell .s differs from hsaco.py's compile")
    sha_out = _sha256(out / f"{name}.hsaco")
    text_in, text_out = s_in.read_text(), (out / f"{name}.s").read_text()
    diff = difflib.unified_diff(
        text_in.split("\n"),
        text_out.split("\n"),
        f"{name}.s\tsha256 {sha_in}",
        f"{name}.s\tgenerated",
        lineterm="",
    )
    patch = "\n".join(diff) + "\n"
    if hsaco.apply_unified_diff(text_in, patch) != text_out:
        raise RuntimeError(f"{name}: hsaco.py's patcher does not reproduce the generated .s")
    (tmp / "p.patch").write_text(patch)
    cmd, _ = _compile(hsaco, cxx, name, tmp / "rb")
    hsaco.patch_and_reassemble(
        name, dict(k, asm_input_sha256=sha_in, asm_patch=str(tmp / "p.patch")), cmd, tmp / "rb"
    )
    if _sha256(tmp / "rb" / f"{name}.hsaco") != sha_out:
        raise RuntimeError(f"{name}: hsaco.py's build path does not reproduce the generator's code object")
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        for f in (f"{name}.s", f"{name}.hsaco", "build.json"):
            shutil.copyfile(out / f, keep / f)
    return patch, sha_in, sha_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernels", nargs="*", help=f"some of {', '.join(GENERATORS)} (default: all)")
    ap.add_argument(
        "--check", action="store_true", help="write nothing; fail unless patches and sha256s match"
    )
    ap.add_argument("--cxx", default=None, help="compiler (default: $PRIMUS_TURBO_HK_CXX, PATH, $ROCM_PATH)")
    ap.add_argument("--keep", default=None, help="save the generated .s, code object and build.json here")
    a = ap.parse_args(argv)
    unknown = sorted(set(a.kernels) - set(GENERATORS))
    if unknown:
        ap.error(f"not a generated kernel: {', '.join(unknown)}")
    hsaco = _load_hsaco()
    m = hsaco.load_manifest()
    cxx = hsaco.find_compiler(a.cxx)
    if cxx is None:
        print(f"no compiler (set ${hsaco.CXX_ENV} or --cxx)", file=sys.stderr)
        return 2
    version = hsaco.compiler_version(cxx)
    if version != m["toolchain"]:
        print(f"toolchain {version!r} != manifest {m['toolchain']!r}", file=sys.stderr)
        return 2
    stale = []
    for name in a.kernels or list(GENERATORS):
        k = m["kernels"][name]
        with tempfile.TemporaryDirectory() as tmp:
            keep = Path(a.keep) / name if a.keep else None
            try:
                patch, sha_in, sha_out = regenerate(hsaco, cxx, name, Path(tmp), keep)
            except RuntimeError as e:
                print(e, file=sys.stderr)
                return 1
        dest = hsaco.HIP_DIR / k["asm_patch"]
        current = dest.is_file() and dest.read_text() == patch
        current = current and (k["asm_input_sha256"], k["hsaco_sha256"]) == (sha_in, sha_out)
        print(
            f"{name}: {'current' if current else 'STALE'} (asm_input_sha256 {sha_in}, hsaco_sha256 {sha_out})"
        )
        if not current:
            stale.append(name)
            if not a.check:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(patch)
                k["asm_input_sha256"], k["hsaco_sha256"] = sha_in, sha_out
    if stale and not a.check:
        hsaco.MANIFEST_PATH.write_text(json.dumps(m, indent=2) + "\n")
        print(f"updated {', '.join(stale)}: patches and manifest.json")
    return 1 if stale and a.check else 0


if __name__ == "__main__":
    sys.exit(main())
