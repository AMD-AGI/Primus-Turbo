#!/usr/bin/env python3
"""Read the ISA reg-note out of a FlyDSL stage dump, per kernel.

Registers come from the ISA only -- the rocprofv3 counter CSV reports a granule count and is
AGPR-blind (see pitfalls/09). The stage dump concatenates every kernel of a module with no
separator, so split on .amdhsa_kernel before counting anything.

usage: _probe_isa.py <dump_dir> [name_filter]
"""
import glob
import json
import os
import re
import sys

KEYS = ("vgpr_count", "agpr_count", "vgpr_spill_count", "sgpr_count", "private_segment_fixed_size")
PATS = {
    "lds": r"^\s*\.amdhsa_group_segment_fixed_size\s+(\d+)",
    "next_free_vgpr": r"^\s*\.amdhsa_next_free_vgpr\s+(\d+)",
    "accum_offset": r"^\s*\.amdhsa_accum_offset\s+(\d+)",
}


def main():
    dump_dir = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    out = {}
    for path in sorted(glob.glob(os.path.join(dump_dir, "**", "*isa*.s"), recursive=True)):
        name, cur = os.path.basename(path), {}
        for line in open(path):
            m = re.match(r"^\s*\.amdhsa_kernel\s+(\S+)", line)
            if m:
                if cur:
                    out[name] = cur
                name, cur = m.group(1), {}
                continue
            m = re.match(r"^\s*;\s*(\w+):\s*(\d+)", line)
            if m and m.group(1) in KEYS:
                cur[m.group(1)] = int(m.group(2))
            for k, p in PATS.items():
                m = re.match(p, line)
                if m:
                    cur[k] = int(m.group(1))
        if cur:
            out[name] = cur
    for name, st in sorted(out.items()):
        if filt and filt not in name:
            continue
        v, a = st.get("vgpr_count", 0), st.get("agpr_count", 0)
        st["unified"] = v + a
        st["granule8"] = -(-st["unified"] // 8) * 8
        print(name, json.dumps(st, sort_keys=True))


if __name__ == "__main__":
    main()
