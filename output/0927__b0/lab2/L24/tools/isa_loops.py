"""Resources + per-main-loop instruction histogram of a FlyDSL 22_final_isa.s.

Loops = label X with a backward branch from a later line; region = X .. last such branch
(covers rotated latches and out-of-line blocks). Outermost WMMA loops only. The clean loop
is the one with the fewest v_cmp/v_cndmask (no per-element mask).
usage: isa_loops.py <isa.s> [--dump-clean N]
"""
import collections, re, sys

KEYS = ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
        "group_segment_fixed_size", "private_segment_fixed_size")
CLASSES = [
    ("wmma", lambda o: o.startswith("v_wmma")),
    ("v_exp", lambda o: o.startswith("v_exp")),
    ("v_nop", lambda o: o == "v_nop"),
    ("ds_load", lambda o: o.startswith("ds_load")),
    ("ds_store", lambda o: o.startswith("ds_store")),
    ("tdm", lambda o: o.startswith("tensor_")),
    ("s_barrier_signal", lambda o: o == "s_barrier_signal"),
    ("s_barrier_wait", lambda o: o == "s_barrier_wait"),
    ("s_wait_tensorcnt", lambda o: o == "s_wait_tensorcnt"),
    ("s_wait_dscnt", lambda o: o == "s_wait_dscnt"),
    ("s_wait_other", lambda o: o.startswith("s_wait") and o not in ("s_wait_tensorcnt", "s_wait_dscnt")),
    ("branch", lambda o: o.startswith("s_cbranch") or o == "s_branch"),
    ("sched_meta", lambda o: o in ("s_set_vgpr_msb", "s_delay_alu")),
    ("salu", lambda o: o.startswith("s_")),
    ("valu", lambda o: o.startswith("v_")),
]


def cls(op):
    for n, f in CLASSES:
        if f(op):
            return n
    return "other"


def parse(path):
    lines = open(path).read().split("\n")
    res = {}
    for ln in lines:
        m = re.match(r"\s+\.(\w+):\s+(\d+)", ln)
        if m and m.group(1) in KEYS:
            res[m.group(1)] = int(m.group(2))
    return lines, res


def ops(lines, idxs):
    out = []
    for i in idxs:
        s = lines[i].strip()
        if not s or s.startswith((".", ";")) or s.endswith(":"):
            continue
        out.append((i, s.split()[0]))
    return out


def loops(lines):
    """Main loops = strongly connected components of the basic-block CFG holding >=64 WMMA.
    Returns a list of sorted line-index lists (a loop body may be laid out non-contiguously)."""
    code = [i for i, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith((".", ";"))
            or re.match(r"^\.LBB\w+:", ln)]
    blocks, cur = [], []
    for i in code:
        if re.match(r"^\.LBB\w+:", lines[i]) and cur:
            blocks.append(cur); cur = []
        cur.append(i)
        op = lines[i].strip().split()[0]
        if op.startswith("s_cbranch") or op in ("s_branch", "s_endpgm", "s_setpc_b64"):
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    lab = {}
    for b, blk in enumerate(blocks):
        m = re.match(r"^(\.LBB\w+):", lines[blk[0]])
        if m:
            lab[m.group(1)] = b
    succ = collections.defaultdict(set)
    for b, blk in enumerate(blocks):
        last = lines[blk[-1]].strip().split()
        op = last[0]
        if op.startswith("s_cbranch") or op == "s_branch":
            if last[1] in lab:
                succ[b].add(lab[last[1]])
        if op not in ("s_branch", "s_endpgm", "s_setpc_b64") and b + 1 < len(blocks):
            succ[b].add(b + 1)
    # Tarjan (iterative)
    idx, low, onst, st, sccs, n = {}, {}, set(), [], [], [0]
    for root in range(len(blocks)):
        if root in idx:
            continue
        work = [(root, iter(sorted(succ[root])))]
        idx[root] = low[root] = n[0]; n[0] += 1; st.append(root); onst.add(root)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is None:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == idx[v]:
                    comp = []
                    while True:
                        x = st.pop(); onst.discard(x); comp.append(x)
                        if x == v:
                            break
                    sccs.append(comp)
            elif w not in idx:
                idx[w] = low[w] = n[0]; n[0] += 1; st.append(w); onst.add(w)
                work.append((w, iter(sorted(succ[w]))))
            elif w in onst:
                low[v] = min(low[v], idx[w])
    out = []
    for comp in sccs:
        ls = sorted(i for b in comp for i in blocks[b])
        if sum(lines[i].strip().startswith("v_wmma") for i in ls) >= 64:
            out.append(ls)
    return sorted(out)


def hist(lines, r):
    o = ops(lines, r)
    h = collections.Counter(cls(op) for _, op in o)
    h["masking"] = sum(1 for _, op in o if op.startswith(("v_cmp", "v_cndmask")))
    h["total"] = len(o)
    return h


if __name__ == "__main__":
    lines, res = parse(sys.argv[1])
    print("res", " ".join(f"{k}={res.get(k)}" for k in KEYS))
    ls = loops(lines)
    for r in ls:
        h = hist(lines, r)
        print(f"loop L{r[0]+1}-{r[-1]+1} " + " ".join(f"{n}={h[n]}" for n, _ in CLASSES) +
              f" masking={h['masking']} total={h['total']}")


def dyn(lines, loop_idxs, G):
    """Steady-state executed instructions per 256 KV (4 iterations of 64 KV), clean loop.
    Block weights: group-gated blocks (barrier wait/signal, TDM issue, tensorcnt) 1/G;
    the tail-only 's_wait_tensorcnt 0x0' alternative 0 when a 0x4/0x2 twin exists;
    the deferred-rescale block (>=16 v_pk_mul_f32, rarely taken) 0. Others 1."""
    s = set(loop_idxs)
    blocks, cur = [], []
    for i in loop_idxs:
        if re.match(r"^\.LBB\w+:", lines[i]) and cur:
            blocks.append(cur); cur = []
        cur.append(i)
        op = lines[i].strip().split()[0]
        if op.startswith("s_cbranch") or op == "s_branch":
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    has_deep = any("s_wait_tensorcnt" in lines[i] and "0x0" not in lines[i] for i in loop_idxs)
    tot = collections.Counter()
    for blk in blocks:
        txt = [lines[i].strip() for i in blk]
        o = [t.split()[0] for t in txt if not t.endswith(":")]
        w = 1.0
        if sum(x == "v_pk_mul_f32" for x in o) >= 16:
            w = 0.0
        elif any(x in ("s_barrier_wait", "s_barrier_signal") or x.startswith("tensor_load") for x in o):
            w = 1.0 / G
        elif any(t.startswith("s_wait_tensorcnt") for t in txt):
            w = 0.0 if (has_deep and any(t == "s_wait_tensorcnt 0x0" for t in txt)) else 1.0 / G
        for x in o:
            tot[cls(x)] += 4 * w
        tot["total"] += 4 * w * len(o)
    return tot


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--dyn":
    G = int(sys.argv[3])
    lines, _ = parse(sys.argv[1])
    ls = loops(lines)
    clean = [r for r in ls if sum(lines[i].strip().startswith(("v_cmp", "v_cndmask")) for i in r) < 20]
    for r in clean[:1]:
        d = dyn(lines, r, G)
        print("dyn/256KV " + " ".join(f"{k}={d[k]:g}" for k in
              ("wmma", "v_exp", "v_nop", "ds_load", "tdm", "s_barrier_signal", "s_barrier_wait",
               "s_wait_tensorcnt", "s_wait_dscnt", "branch", "salu", "valu", "total")))
