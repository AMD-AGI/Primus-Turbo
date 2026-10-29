###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regenerate ``hip/asm/k_dkdv96ds.s.patch``, the assembly patch hsaco.py applies to k_dkdv96ds.

k_dkdv96ds is a generated kernel: ``hip/src/k_dkdv96ds.hip`` built with ``-DHKGEN=1`` (manifest extra_flags) is a
marker shell -- the workgroup set-up, the TDM descriptors and the uniform control flow in HIP, with inline-asm
markers (``; HKGEN INIT / LOOP / FINAL nkv=3 | nkv=2``) where the dK / dV loop goes. ``tools/r8gen`` (gen/r8gen.py:
the regions; gen/bwdgen.py: dependence graph and scheduler; gen/build.py: compile, splice, assemble, check;
asmkit/: s_set_vgpr_msb, expert-mode waits and hazard v_nops derived from the instruction stream) writes the
regions; the patch is the difference between the compiler's own ``.s`` of the shell and the spliced ``.s``.
Built without ``-DHKGEN`` the same source is the LLVM-compiled kernel (the reference the generator learns the
VGPR-MSB operand fields from).

Steps:
  1. compile k_dkdv96ds exactly as hsaco.py does (``hsaco.compile_command``: manifest flags + extra_flags,
     ``-save-temps=obj``) and keep the compiler's ``.s``: its sha256 is the manifest's ``asm_input_sha256``;
  2. run the generator (``tools/r8gen/gen/build.py``: it compiles the same shell and the ``-DHKGEN=0`` reference,
     emits and splices the regions, assembles, re-walks every block for open hazards, ``verify_obj``, contract
     checks: generated VGPR defs only above the shell's registers, SGPR defs only in the clobbered set) and require
     its ``shell.s`` to be that very ``.s`` (its flags are the manifest's);
  3. the patch = a unified diff (compiler ``.s`` -> generated ``.s``); hsaco.py's own ``_apply_unified_diff`` must
     reproduce the generated ``.s`` from it, and hsaco.py's build path (``_patch_and_reassemble``: sha check,
     patch, the driver's own -cc1as + lld) must assemble the generator's code object byte for byte;
  4. write the patch and print the sha256 values for manifest.json (``asm_input_sha256``, ``hsaco_sha256``).

    python primus_turbo/hipkittens/gfx1250_mla_bwd/tools/regen_asm_patch.py [--check] [--cxx PATH] [--keep DIR]
        [--gen-arg ARG ...]

``--check`` writes nothing and fails unless the regenerated patch equals the committed one byte for byte and both
sha256 values equal the manifest's. The generator's options are the manifest's ``gen_args`` of k_dkdv96ds, and the
``-D`` flags of its ``extra_flags`` beyond ``-DHKGEN=1`` go to the shell compile too; ``--gen-arg`` adds options for
experiments. CPU only; needs the pinned toolchain (``manifest.json`` ``toolchain``).
"""

import argparse
import difflib
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TOOLS_DIR.parent
GEN = TOOLS_DIR / "r8gen" / "gen" / "build.py"
NAME = "k_dkdv96ds"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="write nothing; fail unless patch and shas match")
    ap.add_argument("--cxx", default=None, help="compiler (default: $PRIMUS_TURBO_HK_CXX, PATH, $ROCM_PATH)")
    ap.add_argument("--keep", default=None, help="save the compiler's .s, the generated .s and the hsaco here")
    ap.add_argument("--gen-arg", action="append", default=[], help="an option for tools/r8gen/gen/build.py")
    a = ap.parse_args(argv)
    hsaco = _load("_hk_mla_bwd_hsaco", PACKAGE_DIR / "hsaco.py")
    gen = _load("_hk_mla_bwd_r8gen_build", GEN)
    m = hsaco.load_manifest()
    k = m["kernels"][NAME]
    cxx = hsaco.find_compiler(a.cxx)
    if cxx is None:
        print("no compiler (set $PRIMUS_TURBO_HK_CXX or --cxx)", file=sys.stderr)
        return 2
    version = hsaco.compiler_version(cxx)
    if version != m["toolchain"]:
        print(f"toolchain {version!r} != manifest {m['toolchain']!r}", file=sys.stderr)
        return 2
    base = list(gen.EXTRA) + ["-DHKGEN=1"]
    extra = list(k["extra_flags"])
    defs = extra[len(base):]
    if list(m["flags"]) != list(gen.FLAGS) or extra[: len(base)] != base or not all(f.startswith("-D") for f in defs):
        print(f"manifest flags {m['flags']} + {extra} != the generator's {gen.FLAGS} + {base} + -D...", file=sys.stderr)
        return 2
    gen_args = list(k.get("gen_args", [])) + [x for d in defs for x in ("-D", d[2:])] + list(a.gen_arg)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1. the compiler's .s, as hsaco.py makes it
        obj = tmp / "cc" / f"{NAME}.hsaco"
        obj.parent.mkdir()
        cmd = hsaco.compile_command(cxx, NAME, obj)
        r = subprocess.run(cmd, cwd=hsaco.HIP_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-4000:], file=sys.stderr)
            return 1
        s_in = list(obj.parent.glob(f"{NAME}-hip-*.s"))
        if len(s_in) != 1:
            print(f"expected one saved .s, found {s_in}", file=sys.stderr)
            return 1
        s_in = s_in[0]
        sha_in = _sha256(s_in)
        # 2. the generator
        out = tmp / "gen"
        r = subprocess.run([sys.executable, str(GEN), "--out", str(out), "--cxx", cxx, *gen_args],
                           capture_output=True, text=True)  # fmt: skip
        print(r.stdout[-3000:], end="")
        if r.returncode != 0:
            print(r.stderr[-4000:], file=sys.stderr)
            return 1
        if _sha256(out / "shell.s") != sha_in:
            print("the generator's shell .s differs from hsaco.py's compile", file=sys.stderr)
            return 1
        sha_out = _sha256(out / f"{NAME}.hsaco")
        # 3. the patch, and hsaco.py's own path from it
        text_in, text_out = s_in.read_text(), (out / f"{NAME}.s").read_text()
        patch = "\n".join(
            difflib.unified_diff(text_in.split("\n"), text_out.split("\n"), fromfile=f"{NAME}.s\tsha256 {sha_in}",
                                 tofile=f"{NAME}.s\tregen_asm_patch.py (tools/r8gen)", n=3, lineterm="")
        ) + "\n"  # fmt: skip
        if hsaco._apply_unified_diff(text_in, patch) != text_out:
            print("hsaco.py's patcher does not reproduce the generated .s", file=sys.stderr)
            return 1
        ptmp = tmp / "k.s.patch"
        ptmp.write_text(patch)
        rb = tmp / "rb"
        rb.mkdir()
        obj2 = rb / f"{NAME}.hsaco"
        cmd2 = hsaco.compile_command(cxx, NAME, obj2)
        r = subprocess.run(cmd2, cwd=hsaco.HIP_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-4000:], file=sys.stderr)
            return 1
        hsaco._patch_and_reassemble(NAME, dict(k, asm_input_sha256=sha_in, asm_patch=str(ptmp)), cmd2, rb)
        if _sha256(obj2) != sha_out:
            print(f"hsaco.py's build path gives {_sha256(obj2)[:16]}, the generator {sha_out[:16]}", file=sys.stderr)
            return 1
        if a.keep:
            keep = Path(a.keep)
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s_in, keep / f"{NAME}.compiler.s")
            shutil.copyfile(out / f"{NAME}.s", keep / f"{NAME}.s")
            shutil.copyfile(out / f"{NAME}.hsaco", keep / f"{NAME}.hsaco")
            shutil.copyfile(out / "build.json", keep / "build.json")
        dest = hsaco.HIP_DIR / k["asm_patch"]
        if a.check:
            ok = dest.is_file() and dest.read_text() == patch
            ok_sha = k.get("asm_input_sha256") == sha_in and k.get("hsaco_sha256") == sha_out
            print(f"patch {'current' if ok else 'STALE'}; shas {'match' if ok_sha else 'DIFFER'}: "
                  f"asm_input_sha256 {sha_in} hsaco_sha256 {sha_out}")  # fmt: skip
            return 0 if ok and ok_sha else 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(patch)
        print(f"wrote {dest} ({patch.count(chr(10))} lines)")
        print(f'manifest.json kernels.{NAME}: "asm_input_sha256": "{sha_in}", "hsaco_sha256": "{sha_out}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
