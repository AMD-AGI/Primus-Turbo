#!/usr/bin/env python3
"""N-arm steady-state step times from ONE e2e training log (no torch, no GPU; host python is fine).

  steady_arms3.py <main log> <E2E_ATTN spec> <profile_freq|0> [--warmup 7] [--ref asm] [--json out.json]

Merges output/0927__b0/e2e/tools/steady_arms.py (2 arms, ABBA cycles) and
output/0928__a0_repro/tools/e2e_arms3.py (N arms), same rules, so numbers compare with B0's
1.0323 / 1.0328 and A0 09-28's p3a:
  * arm of training step s (1-based) = parse_schedule(spec): W tokens once, then C cycled
  * steady window: drop steps <= warmup and F-1, F, F+1 around every profiled step F
    (multiples of profile_freq); profiled steps carry profiler warmup / trace export
  * step ms = local_batch * seq / tps * 1000 (log timestamps are 1 s; tps is per step at log_freq 1)
  * cycle ratio: for every schedule cycle fully inside the window, mean(arm b)/mean(arm a); median over cycles
  * adjacent pairs: every two consecutive in-window steps with different arms -> ratio and
    difference (ms); median per arm pair (both orders pooled; the per-order medians are printed too)
Also: non-finite loss / grad_norm steps (any -> the run is void), per-arm peak reserved memory.
"""
import argparse
import json
import re
import statistics as st

TOK = 4 * 8192


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step:\s*(\d+)\s+loss:\s*([-\d.naninfNaNInf]+)", line)
        if not m:
            continue

        def g(k):
            x = re.search(k + r":\s*([\d,]+\.?\d*)", line)
            return float(x.group(1).replace(",", "")) if x else None
        gn = re.search(r"grad_norm:\s*([-\w.]+)", line)
        mem = re.search(r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)", line)
        rows.append(dict(step=int(m.group(1)), loss=m.group(2), tps=g("tps"),
                         gn=gn.group(1) if gn else None,
                         mem_gib=float(mem.group(1)) if mem else None,
                         mem_pct=float(mem.group(2)) if mem else None))
    return rows


def sched(spec):
    w, c = spec.split(";") if ";" in spec else ("", spec)
    t = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    return t(w), t(c)


def arm(spec, s):
    w, c = sched(spec)
    i = s - 1
    return w[i] if i < len(w) else c[(i - len(w)) % len(c)]


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def q(xs):
    xs = sorted(xs)
    if len(xs) >= 4:
        a, _, b = st.quantiles(xs, n=4)
        return [a, b]
    return [xs[0], xs[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("spec")
    ap.add_argument("pfreq", type=int)
    ap.add_argument("--warmup", type=int, default=7)
    ap.add_argument("--ref", default="asm")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = parse(a.log)
    last = rows[-1]["step"] if rows else 0
    excl = set(range(1, a.warmup + 1))
    if a.pfreq:
        for f in range(a.pfreq, last + 2, a.pfreq):
            excl |= {f - 1, f, f + 1}
    ms = {r["step"]: TOK / r["tps"] * 1000 for r in rows if r["tps"]}
    ok = lambda s: s not in excl and s in ms  # noqa: E731
    bad = [r["step"] for r in rows if not finite(r["loss"]) or (r["gn"] is not None and not finite(r["gn"]))]
    out = {"log": a.log, "spec": a.spec, "pfreq": a.pfreq, "warmup": a.warmup, "steps": len(rows),
           "last": last, "window": [s for s in range(1, last + 1) if ok(s)],
           "nonfinite_steps": bad, "loss_first": rows[0]["loss"] if rows else None,
           "loss_last": rows[-1]["loss"] if rows else None, "arms": {}, "pairs": {}}
    print(f"log {a.log}\nsteps {len(rows)} last {last}; window {len(out['window'])} steps; "
          f"excluded {sorted(x for x in excl if x <= last)}")
    print(f"non-finite loss/grad_norm steps: {bad if bad else 'none'}; loss first/last "
          f"{out['loss_first']} / {out['loss_last']}")
    if bad:
        print("!! NON-FINITE STEPS: this run is void (card-safety section 8: corrupted runs look fastest)")
    w, c = sched(a.spec)
    arms = list(dict.fromkeys(c))
    if a.ref in arms:
        arms.remove(a.ref)
        arms.insert(0, a.ref)
    for x in arms:
        t = [ms[s] for s in range(1, last + 1) if ok(s) and arm(a.spec, s) == x]
        if not t:
            continue
        mem = [r for r in rows if arm(a.spec, r["step"]) == x and r["mem_pct"] is not None]
        pk = max(mem, key=lambda r: r["mem_pct"]) if mem else None
        out["arms"][x] = {"n": len(t), "ms_median": st.median(t), "ms_iqr": q(t),
                          "ms_range": [min(t), max(t)], "tps_median": TOK / st.median(t) * 1000,
                          "peak_mem_gib": pk["mem_gib"] if pk else None,
                          "peak_mem_pct": pk["mem_pct"] if pk else None}
        r_ = out["arms"][x]
        print(f"arm {x:8s} n={len(t):3d} step ms median {r_['ms_median']:8.1f}  IQR {r_['ms_iqr'][0]:.1f}-"
              f"{r_['ms_iqr'][1]:.1f}  range {min(t):.1f}-{max(t):.1f}  tps {r_['tps_median']:8.0f}  "
              f"peak mem {r_['peak_mem_gib']} GiB ({r_['peak_mem_pct']}%)")
    # cycles fully inside the window
    cyc = {}
    s0 = len(w) + 1
    while s0 + len(c) - 1 <= last:
        blk = range(s0, s0 + len(c))
        if all(ok(s) for s in blk):
            m = {x: st.mean(ms[s] for s in blk if arm(a.spec, s) == x) for x in arms}
            for i, x in enumerate(arms):
                for y in arms[i + 1:]:
                    cyc.setdefault((y, x), []).append(m[y] / m[x])
        s0 += len(c)
    # adjacent pairs
    adj, adjd, adjo = {}, {}, {}
    for s in range(1, last):
        if ok(s) and ok(s + 1):
            x, y = arm(a.spec, s), arm(a.spec, s + 1)
            if x == y:
                continue
            k = tuple(sorted((x, y), key=arms.index))[::-1]      # (later-listed arm, earlier-listed arm)
            num, den = (ms[s], ms[s + 1]) if x == k[0] else (ms[s + 1], ms[s])
            adj.setdefault(k, []).append(num / den)
            adjd.setdefault(k, []).append(num - den)
            adjo.setdefault((k, "first" if x == k[0] else "second"), []).append(num / den)
    for k in sorted(set(cyc) | set(adj), key=lambda k: (arms.index(k[1]), arms.index(k[0]))):
        cr, ar, dr = cyc.get(k, []), adj.get(k, []), adjd.get(k, [])
        rec = {"cycles_n": len(cr), "cycles_median": st.median(cr) if cr else None,
               "cycles_range": [min(cr), max(cr)] if cr else None,
               "adjacent_n": len(ar), "adjacent_median": st.median(ar) if ar else None,
               "adjacent_diff_ms_median": st.median(dr) if dr else None,
               "adjacent_first_median": st.median(adjo[(k, "first")]) if (k, "first") in adjo else None,
               "adjacent_second_median": st.median(adjo[(k, "second")]) if (k, "second") in adjo else None,
               "arm_medians_ratio": (out["arms"][k[0]]["ms_median"] / out["arms"][k[1]]["ms_median"]
                                     if k[0] in out["arms"] and k[1] in out["arms"] else None)}
        out["pairs"][f"{k[0]}/{k[1]}"] = rec
        cs = (f"cycles n={len(cr)} median {rec['cycles_median']:.4f} range {min(cr):.4f}-{max(cr):.4f}"
              if cr else "cycles n=0")
        f4 = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
        as_ = (f"adjacent n={len(ar)} median {rec['adjacent_median']:.4f} (diff {rec['adjacent_diff_ms_median']:+.1f} ms; "
               f"{k[0]}-first {f4(rec['adjacent_first_median'])}, {k[0]}-second {f4(rec['adjacent_second_median'])})"
               if ar else "adjacent n=0")
        am = f"{rec['arm_medians_ratio']:.4f}" if rec["arm_medians_ratio"] else "-"
        print(f"RATIO {k[0]}/{k[1]}: {cs}; {as_}; per-arm medians {am}  (<1 = {k[0]} faster)")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
