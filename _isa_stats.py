"""Per-kernel ISA instruction histogram + register/LDS metadata for a FLYDSL_DUMP_IR dump."""
import re
import sys

p = sys.argv[1]
raw = open(p).read()
lines = raw.splitlines()
cur, secs = None, {}
for ln in lines:
    m = re.match(r"^([A-Za-z_][\w$.]*):\s*(;.*)?$", ln)
    if m and not m.group(1).startswith(".L"):
        cur = m.group(1)
        secs.setdefault(cur, [])
    elif cur is not None:
        secs[cur].append(ln)

PATS = (
    "buffer_load_dword ",
    "buffer_load_dwordx2",
    "buffer_load_dwordx4",
    "buffer_store_short",
    "buffer_store_dword",
    "v_mfma",
    "s_barrier",
    "ds_read_b128",
    "ds_read_b32",
    "ds_write",
    "s_waitcnt vmcnt",
    "s_waitcnt lgkmcnt",
    "v_cmp",
    "v_cndmask",
    "s_setprio",
    "scratch_",
)
for k, v in secs.items():
    if len(v) < 60:
        continue
    body = "\n".join(v)
    print(f"== {k}  ({len(v)} lines)")
    for pat in PATS:
        n = body.count(pat)
        if n:
            print(f"   {pat:24s} {n}")

for key in (
    "amdhsa_next_free_vgpr",
    "amdhsa_next_free_sgpr",
    "amdhsa_accum_offset",
    "amdhsa_private_segment_fixed_size",
    "amdhsa_group_segment_fixed_size",
):
    print(key, re.findall(key + r"\s+(\S+)", raw))
for key in (
    r"\.vgpr_count",
    r"\.agpr_count",
    r"\.sgpr_count",
    r"\.private_segment_fixed_size",
    r"\.group_segment_fixed_size",
    r"\.vgpr_spill_count",
    r"\.name",
):
    print(key, re.findall(key + r":\s*(\S+)", raw))

# where the narrow dword loads live: print the enclosing 1 line of context, deduped
narrow = [(i, l.strip()) for i, l in enumerate(lines) if "buffer_load_dword " in l]
print(f"\nnarrow buffer_load_dword: {len(narrow)}")
srcs = {}
for i, l in narrow:
    m = re.search(r"buffer_load_dword\s+\S+,\s*(\S+),\s*(\S+)", l)
    srcs[m.group(2) if m else "?"] = srcs.get(m.group(2) if m else "?", 0) + 1
print("  by srd:", sorted(srcs.items(), key=lambda x: -x[1])[:10])
