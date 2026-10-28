###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-descriptor guard: compare a rung's built kernels with a reference rung before anything runs on a GPU.

    python3 -m hkl.kdcheck mla/ladder_bwd.json --rung r8a r8b --ref r7 [--allow amdhsa_group_segment_fixed_size]

For every kernel of the rung (build/<ladder>/<rung>/<kernel>.s) the .amdhsa_* kernel-descriptor directives and the
kernel's metadata (argument count, kernarg segment size, hidden arguments) must equal the reference rung's, except
the register counts (next_free_vgpr / sgpr, accum_offset, reserve flags), which move with any code change. A
difference in user SGPRs (dispatch / queue pointer, kernarg preload), hidden arguments, scratch, dynamic stack,
LDS or wave setup fails the check: such a kernel needs a launcher change and a toy run first (the 2026-10-27 GPU1
hang came from a variant that read the dispatch packet). Pure stdlib; CPU only.
"""

import argparse
import re
import sys
from pathlib import Path

from .build import DEFAULT_OUT, ladder_name, load_ladder

FREE = {  # change with any code change; never a launch-interface difference
    ".amdhsa_next_free_vgpr", ".amdhsa_next_free_sgpr", ".amdhsa_accum_offset", ".amdhsa_reserve_vcc",
    ".amdhsa_reserve_xnack_mask", ".amdhsa_named_barrier_count",
}


def descriptor(s_text, kernel):
    """.amdhsa_* directives of `kernel` and its metadata summary from a compiler .s."""
    m = re.search(r"\.amdhsa_kernel\s+" + re.escape(kernel) + r"\s*\n(.*?)\.end_amdhsa_kernel", s_text, re.S)
    if not m:
        raise SystemExit(f"{kernel}: no .amdhsa_kernel block")
    d = {}
    for line in m.group(1).splitlines():
        t = line.split()
        if len(t) >= 2 and t[0].startswith(".amdhsa_"):
            d[t[0]] = t[1]
    md = re.search(r"\.name:\s+" + re.escape(kernel) + r"\b", s_text)
    meta = {}
    if md:
        # the metadata entry of this kernel: from its '.args:' list (which precedes .name) to the next entry
        start = s_text.rfind("  - .args:", 0, md.start())
        end = s_text.find("\n  - ", md.end())
        blk = s_text[start:end if end > 0 else len(s_text)]
        kinds = re.findall(r"\.value_kind:\s+(\S+)", blk)
        meta["args"] = str(len(kinds))
        meta["hidden_args"] = ",".join(k for k in kinds if k.startswith("hidden")) or "-"
        for key in (".kernarg_segment_size", ".group_segment_fixed_size", ".private_segment_fixed_size",
                    ".uses_dynamic_stack", ".wavefront_size", ".max_flat_workgroup_size"):
            mm = re.search(re.escape(key) + r":\s+(\S+)", blk)
            if mm:
                meta[key] = mm.group(1)
    return d, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladder")
    ap.add_argument("--rung", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--allow", nargs="*", default=[], help="directives / metadata keys allowed to differ")
    a = ap.parse_args(argv)
    spec = load_ladder(a.ladder)
    out = DEFAULT_OUT / ladder_name(spec)
    bad = 0
    for rid in a.rung:
        rung = spec["_by_id"][rid]
        for kname in rung["kernels"]:
            ref_s, s = out / a.ref / f"{kname}.s", out / rid / f"{kname}.s"
            if not ref_s.exists():
                print(f"  {rid:<12} {kname:<14} (reference {a.ref} has no {kname}: skipped)")
                continue
            if not s.exists():
                print(f"  {rid:<12} {kname:<14} FAIL not built")
                bad += 1
                continue
            rd, rm = descriptor(ref_s.read_text(), kname)
            dd, dm = descriptor(s.read_text(), kname)
            diffs = [f"{k} {rd.get(k)} -> {dd.get(k)}" for k in sorted(set(rd) | set(dd))
                     if k not in FREE and k not in a.allow and rd.get(k) != dd.get(k)]
            diffs += [f"meta {k} {rm.get(k)} -> {dm.get(k)}" for k in sorted(set(rm) | set(dm))
                      if k not in a.allow and rm.get(k) != dm.get(k)]
            regs = f"vgpr {rd.get('.amdhsa_next_free_vgpr')}->{dd.get('.amdhsa_next_free_vgpr')}"
            if diffs:
                bad += 1
                print(f"  {rid:<12} {kname:<14} FAIL ({regs}): " + "; ".join(diffs))
            else:
                print(f"  {rid:<12} {kname:<14} ok   ({regs}, args {dm.get('args')}, hidden {dm.get('hidden_args')})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
