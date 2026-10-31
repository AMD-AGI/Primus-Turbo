###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hkl.build's generator entry for the fwd2 tree (the Primus-Turbo gfx1250_mla package layout).

    python3 gen/build.py --out DIR --asmkit-root TREE [generator args...]

Runs tools/fwdgen/build.py (which imports the package's own tools/asmkit, so --asmkit-root is ignored) with
--out DIR and the generator args, then puts the HIP reference variant where hkl.build looks for it
(DIR/ref/k_mla_fwd.hsaco, from the reference compile's k_mla_fwd.cc.hsaco)."""

import runpy
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    out, rest, i = None, [], 0
    while i < len(args):
        if args[i] == "--asmkit-root":
            i += 2
            continue
        if args[i] == "--out":
            out = args[i + 1]
            i += 2
            continue
        rest.append(args[i])
        i += 1
    if out is None:
        sys.exit("gen/build.py: --out DIR is required")
    script = PKG / "tools" / "fwdgen" / "build.py"
    sys.argv = [str(script), "--out", out, *rest]
    rc = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    ref = Path(out) / "ref"
    cc = ref / "k_mla_fwd.cc.hsaco"
    if cc.is_file():
        shutil.copy2(cc, ref / "k_mla_fwd.hsaco")
        s = sorted(ref.glob("*-gfx1250.s"))
        if s:
            shutil.copy2(s[0], ref / "k_mla_fwd.s")
    sys.exit(rc)


if __name__ == "__main__":
    main()
