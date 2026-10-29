###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit command line.

    PYTHONPATH=primus_turbo/hipkittens python -m asmkit loops    K.s
    PYTHONPATH=primus_turbo/hipkittens python -m asmkit map      K.s [--label L | --nwmma N] [--out map.json]
    PYTHONPATH=primus_turbo/hipkittens python -m asmkit identity K.s OUT.s [--label L | --nwmma N] [--no-expert]
                                                                 [--entry both|cfg|preheader]
    PYTHONPATH=primus_turbo/hipkittens python -m asmkit generate K.s GEN.s OUT.s [--label L | --nwmma N] [...]
    PYTHONPATH=primus_turbo/hipkittens python -m asmkit build    IN.s OUT.hsaco --main-file-name F.hip [--mllvm X ...]
    PYTHONPATH=primus_turbo/hipkittens python -m asmkit verify-obj INTENDED.s OBJDUMP.s

identity / generate exit 1 when a static check fails (the text is written anyway, for inspection).
--entry: how the state at the loop header is modelled: 'cfg' = exact dataflow from the kernel entry (LLVM's view),
'preheader' = the preheader block from an unknown state (the original AO model, blind to older VMEM loads),
'both' (default) = every wait satisfies both.
"""

import argparse
import json
import shlex
import sys

from . import build, splice, verify_obj
from .isa import read_lines


def _region_args(ap):
    ap.add_argument("--label", default=None, help="header label of the shell loop")
    ap.add_argument("--nwmma", type=int, default=None, help="WMMA count of the shell loop")
    ap.add_argument("--func", default=None, help="kernel function name (default: the first)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="asmkit", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loops")
    p.add_argument("s")
    p.add_argument("--func", default=None)
    p = sub.add_parser("map")
    p.add_argument("s")
    _region_args(p)
    p.add_argument("--out", default=None)
    for name in ("identity", "generate"):
        p = sub.add_parser(name)
        p.add_argument("s")
        if name == "generate":
            p.add_argument("gen", help="the generated region text (labels + instructions)")
        p.add_argument("out")
        _region_args(p)
        p.add_argument("--no-expert", action="store_true", help="the kernel is not in expert scheduling mode")
        p.add_argument("--entry", default="both", choices=("both", "cfg", "preheader"))
        p.add_argument("--map", default=None, help="also write the entry register map (JSON)")
    p = sub.add_parser("build")
    p.add_argument("s")
    p.add_argument("out")
    p.add_argument("--main-file-name", required=True)
    p.add_argument("--mllvm", action="append", default=[], help="an -mllvm option of the kernel's compile")
    p.add_argument("--llvm-bin", default=None, help="directory of clang / lld (default: next to amdclang++)")
    p.add_argument(
        "--compile-cmd", default=None, help="the kernel's compile command (shell words); its -### steps"
    )
    p.add_argument("--cwd", default=".")
    p = sub.add_parser("verify-obj")
    p.add_argument("intended")
    p.add_argument("objdump")
    a = ap.parse_args(argv)

    if a.cmd == "loops":
        for lab, h, mem, lat, nw in splice.loops_summary(a.s, a.func):
            print(
                f"{lab}: header block {h}, blocks {mem[0]}..{mem[-1]} ({len(mem)}), latches {lat}, {nw} WMMA"
            )
        return 0
    if a.cmd == "map":
        r = splice.find_shell(a.s, label=a.label, nwmma=a.nwmma, func=a.func)
        m = splice.entry_map(r)
        text = json.dumps(m, indent=1)
        if a.out:
            open(a.out, "w").write(text + "\n")
            print(
                f"{r.describe()}\n  live-in VGPRs {m['vgpr_live_in_count']}, free {m['vgpr_free_count']}, "
                f"scalar live-in {len(m['scalar_live_in'])} -> {a.out}"
            )
        else:
            print(text)
        return 0
    if a.cmd in ("identity", "generate"):
        expert = not a.no_expert
        shell = splice.find_shell(a.s, label=a.label, nwmma=a.nwmma, func=a.func)
        if a.cmd == "identity":
            new = splice.rebuild(shell, expert=expert, entry=a.entry)
        else:
            new = splice.generate(shell, read_lines(a.gen), expert=expert, entry=a.entry)
        open(a.out, "w").write("\n".join(new) + "\n")
        fail, st = splice.check(shell, new, func=a.func, expert=expert, entry="both")
        print(f"{shell.describe()}\n  {a.cmd} -> {a.out}: {'PASS' if not fail else 'FAIL'} {st}")
        for f in fail:
            print("   ", f)
        if a.map:
            open(a.map, "w").write(json.dumps(splice.entry_map(shell), indent=1) + "\n")
        return 1 if fail else 0
    if a.cmd == "build":
        if a.compile_cmd:
            cc1as, lld = build.driver_steps(shlex.split(a.compile_cmd), a.cwd)
        else:
            import shutil
            from pathlib import Path

            llvm_bin = a.llvm_bin
            if llvm_bin is None:
                cxx = shutil.which("amdclang++")
                if cxx is None:
                    print("amdclang++ not found: pass --llvm-bin", file=sys.stderr)
                    return 2
                llvm_bin = str(Path(cxx).resolve().parent.parent / "lib" / "llvm" / "bin")
                if not Path(llvm_bin, "lld").exists():
                    llvm_bin = str(Path(cxx).resolve().parent)
            cc1as, lld = build.default_steps(llvm_bin, a.main_file_name, a.mllvm, a.cwd)
        meta = build.assemble(a.s, a.out, cc1as, lld, cwd=a.cwd)
        print(json.dumps(meta))
        return 0
    if a.cmd == "verify-obj":
        bad = verify_obj.compare(a.intended, a.objdump)
        for t in bad:
            print("  ", t)
        return 1 if bad else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
