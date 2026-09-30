#!/usr/bin/env python3
"""N-arm steady-state step times from one e2e log (no torch; run in the container, no card).

  e2e_arms3.py <main log> <E2E_ATTN spec> <profile_freq|0> [warmup_steps=7]

Same window as ../../0927__b0/e2e/tools/steady_arms.py (drop steps <= warmup and F-1, F, F+1 around every
profiled step F; step ms = tokens / tps). Generalised to any number of arms:
  cycle ratios   for every schedule cycle fully inside the window, mean(arm)/mean(ref) per cycle, median over cycles
  adjacent pairs every two consecutive in-window steps with different arms, ms(a)/ms(b), median per arm pair
"""
import sys, statistics as st
sys.path.insert(0, "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e/tools")
from steady_arms import parse, sched, arm, TOK  # noqa: E402


def main():
    path, spec, pf = sys.argv[1], sys.argv[2], int(sys.argv[3])
    warm = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    rows = parse(path)
    last = rows[-1]["step"] if rows else 0
    excl = set(range(1, warm + 1))
    if pf:
        for f in range(pf, last + 2, pf):
            excl |= {f - 1, f, f + 1}
    ms = {r["step"]: TOK / r["tps"] * 1000 for r in rows if r["tps"]}
    ok = lambda s: s not in excl and s in ms
    print(f"log {path}\nsteps {len(rows)} last {last}; window {sum(ok(s) for s in range(1, last + 1))} steps")
    bad = [r for r in rows if "nan" in r["loss"] or "inf" in r["loss"]]
    print(f"non-finite loss steps: {len(bad)}; loss first/last: {rows[0]['loss']} / {rows[-1]['loss']}" if rows else "no steps")
    w, c = sched(spec)
    arms = list(dict.fromkeys(c))
    med = {}
    for a in arms:
        t = sorted(ms[s] for s in range(1, last + 1) if ok(s) and arm(spec, s) == a)
        if not t:
            continue
        med[a] = st.median(t)
        pk = max((r["mem"] or 0) for r in rows if arm(spec, r["step"]) == a)
        print(f"arm {a:8s} n={len(t):3d} step ms median {med[a]:8.1f}  range {t[0]:.1f}-{t[-1]:.1f}  "
              f"tps {TOK / med[a] * 1000:8.0f}  peak mem {pk} GiB")
    cyc = {}
    s0 = len(w) + 1
    while s0 + len(c) - 1 <= last:
        blk = range(s0, s0 + len(c))
        if all(ok(s) for s in blk):
            m = {a: st.mean(ms[s] for s in blk if arm(spec, s) == a) for a in arms}
            for i, a in enumerate(arms):
                for b in arms[i + 1:]:
                    cyc.setdefault((b, a), []).append(m[b] / m[a])
        s0 += len(c)
    adj = {}
    for s in range(1, last):
        if ok(s) and ok(s + 1):
            a, b = arm(spec, s), arm(spec, s + 1)
            if a != b:
                k = tuple(sorted((a, b), key=arms.index))[::-1]
                num, den = (ms[s], ms[s + 1]) if a == k[0] else (ms[s + 1], ms[s])
                adj.setdefault(k, []).append(num / den)
    for k in sorted(set(cyc) | set(adj), key=lambda k: (arms.index(k[1]), arms.index(k[0]))):
        cr, ar = cyc.get(k, []), adj.get(k, [])
        cs = f"cycles n={len(cr)} median {st.median(cr):.4f} range {min(cr):.4f}-{max(cr):.4f}" if cr else "cycles n=0"
        as_ = f"adjacent n={len(ar)} median {st.median(ar):.4f}" if ar else "adjacent n=0"
        print(f"RATIO {k[0]}/{k[1]}: {cs}; {as_}; per-arm medians {med[k[0]] / med[k[1]]:.4f}")


if __name__ == "__main__":
    main()
