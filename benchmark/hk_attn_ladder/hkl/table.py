###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Print a ladder file as a Markdown table (README) -- rung, parent, how it is built, a08-1 reference, feature.

    python3 -m hkl.table llama/ladder_fwd.json [--build]    # --build: add VGPRs / LDS / sha from build/
"""

import argparse
import json
import sys

from .build import DEFAULT_OUT, ladder_name, load_ladder


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    spec = load_ladder(a.ladder)
    cols = ["Rung", "Parent", "Source / flags", "Pin", "a08-1 ms", "Feature"] + (["VGPR / LDS"] if a.build else [])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in spec["rungs"]:
        ks = r["kernels"]
        def kdesc(k):
            if k.get("gen") is not None:  # a generator kernel (hkl.build): its gen/build.py arguments
                return f"gen `{' '.join(k['gen']) or '(defaults)'}`" + (" ref" if k.get("gen_ref") else "")
            fl = [f for f in k["extra_flags"] if f.startswith("-D") or f.startswith("-amdgpu")]
            return f"`{k['src'].split('/')[-1]}`" + (f" `{' '.join(fl)}`" if fl else "")

        src = "; ".join(kdesc(k) for k in ks.values())
        if r.get("patch"):
            src = f"`{r['patch'].split('/')[-1]}`; " + src
        pins = [("yes" if k.get("sha256") else "-") for k in ks.values()]
        pin = "yes" if all(p == "yes" for p in pins) else ("part" if "yes" in pins else "new")
        ref = r.get("ref_ms")
        refs = f"{ref:.3f}" if isinstance(ref, (int, float)) else ""
        feat = r.get("feature", "") + (" *(side)*" if r.get("side") else "")
        row = [r["id"], r.get("parent", ""), src, pin, refs, feat]
        if a.build:
            res = []
            for k in ks:
                try:
                    m = json.load(open(DEFAULT_OUT / ladder_name(spec) / r["id"] / f"{k}.meta.json"))
                    res += [f"{v['vgpr']}/{v['lds'] // 1024}K" for n, v in m["kernels"].items()
                            if not n.endswith(("_prep", "_prep_z", "_cvt", "_dqcvt"))]  # fmt: skip
                except OSError:
                    res.append("?")
            row.append(", ".join(res))
        print("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
