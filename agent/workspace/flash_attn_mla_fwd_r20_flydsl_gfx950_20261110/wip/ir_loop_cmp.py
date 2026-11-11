import re, sys, difflib
def bbs(path):
    cur, out, order = None, {}, []
    for ln in open(path):
        m = re.match(r"^(\d+|[\w.]+):", ln)
        if m:
            cur = m.group(1); out[cur] = []; order.append(cur); continue
        if cur is not None and ln.strip() and not ln.startswith("}"):
            out[cur].append(ln.strip())
    return out, order
def loops(path):
    out, order = bbs(path)
    return [b for b in order if sum("mfma" in x for x in out[b]) >= 100], out
def norm(lines):
    # rename SSA values by first appearance within the block; keep external refs as EXT
    names = {}
    res = []
    for ln in lines:
        defs = re.match(r"^(%\d+) = ", ln)
        def rep(m):
            v = m.group(0)
            if v in names: return names[v]
            return "%EXT"
        if defs:
            body = ln[len(defs.group(0)):]
            body = re.sub(r"%\d+", rep, body)
            names[defs.group(1)] = "%v" + str(len(names))
            res.append(names[defs.group(1)] + " = " + body)
        else:
            res.append(re.sub(r"%\d+", rep, ln))
    return res
a_loops, a = loops(sys.argv[1]); b_loops, b = loops(sys.argv[2])
for la, lb in zip(a_loops, b_loops):
    x, y = norm(a[la]), norm(b[lb])
    d = [l for l in difflib.unified_diff(x, y, lineterm="", n=1)]
    print(f"== {la} vs {lb}: {len(x)} vs {len(y)} lines, diff lines {len(d)}")
    print("\n".join(d[:int(sys.argv[3]) if len(sys.argv) > 3 else 60]))
