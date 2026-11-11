"""MFMA cover of every scratch reload in the dkdv hot loop: python isa_reload_cover.py <22_final_isa.s> ...

gfx950's vmcnt counts loads, stores, atomics and scratch in issue order, so a reload is complete at the
first later `s_waitcnt vmcnt(k)` with k <= (vector-memory ops issued after it). Cover = MFMAs issued
between the reload and that wait. Prints spill count, then per reload `offset:cover`, and the minimum."""
import re
import sys

VM = ("scratch_load", "scratch_store", "buffer_load", "buffer_store", "buffer_atomic", "global_load",
      "global_store", "global_atomic")


def hot_loop(lines):
    lab = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"^(\.LBB0_\d+):", l))}
    loops = {}
    for i, l in enumerate(lines):
        m = re.search(r"s_cbranch_\w+\s+(\.LBB0_\d+)|s_branch\s+(\.LBB0_\d+)", l)
        t = m and (m.group(1) or m.group(2))
        if t in lab and lab[t] < i:
            loops[lab[t]] = max(loops.get(lab[t], 0), i)
    segs = [(a, b) for a, b in sorted(loops.items()) if sum("v_mfma" in s for s in lines[a : b + 1]) >= 50]
    a, b = segs[-1]
    return [l.strip().split(";")[0].strip() for l in lines[a : b + 1] if l.strip() and not l.strip().startswith((";", "."))]


def covers(ins):
    out = []
    for i, x in enumerate(ins):
        if not x.startswith("scratch_load"):
            continue
        m = re.search(r"offset:(\d+)", x)
        later, mf = 0, 0
        for y in ins[i + 1 :]:
            if y.startswith("v_mfma"):
                mf += 1
            elif y.startswith(VM):
                later += 1
            elif y.startswith("s_waitcnt") and (w := re.search(r"vmcnt\((\d+)\)", y)) and int(w.group(1)) <= later:
                break
        out.append((int(m.group(1)) if m else 0, mf))
    return out


for path in sys.argv[1:]:
    text = open(path).read()
    spill = re.search(r"\.vgpr_spill_count:\s+(\d+)", text).group(1)
    c = covers(hot_loop(text.split("\n")))
    print("spill %s | hot reloads %s | min cover %d" % (spill, " ".join("%d:%d" % t for t in c), min(m for _, m in c)))
