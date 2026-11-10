"""Per-loop counts of the dkdv kernel ISA: python isa_loop_counts.py <22_final_isa.s> ..."""
import re
import sys

for path in sys.argv[1:]:
    L = open(path).read().split("\n")
    meta = {k: re.search(r"\.%s:\s+(\d+)" % k, "\n".join(L)) for k in ("vgpr_spill_count", "private_segment_fixed_size")}
    lab = {m.group(1): i for i, l in enumerate(L) if (m := re.match(r"^(\.LBB0_\d+):", l))}
    loops = {}
    for i, l in enumerate(L):
        m = re.search(r"s_cbranch_\w+\s+(\.LBB0_\d+)|s_branch\s+(\.LBB0_\d+)", l)
        t = m and (m.group(1) or m.group(2))
        if t in lab and lab[t] < i:
            loops[lab[t]] = max(loops.get(lab[t], 0), i)
    out = []
    for a, b in sorted(loops.items()):
        seg = L[a : b + 1]
        if sum("v_mfma" in s for s in seg) < 50:
            continue
        out.append("scratch_load %d vmcnt(0) %d" % (sum("scratch_load" in s for s in seg), sum("vmcnt(0)" in s for s in seg)))
    print("%-60s spill %s scratch %s B | masked loop: %s | hot loop: %s" % (
        path.split("/dev/")[-1].split("/flash_attn")[0], meta["vgpr_spill_count"].group(1),
        meta["private_segment_fixed_size"].group(1), out[0], out[-1]))
