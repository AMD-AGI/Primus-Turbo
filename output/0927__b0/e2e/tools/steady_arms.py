#!/usr/bin/env python3
"""Per-arm steady-state tps / step time from one e2e log (no torch; run in the container, no card).

  steady_arms.py <main log> <E2E_ATTN spec> <profile_freq|0> [warmup_steps=7]

Arm of training step s (1-based) follows e2e_attn.parse_schedule: the W tokens once, then C cycling
(step_idx = s-1). Excluded from the steady window: steps <= warmup, and around every profiled step F
(multiples of profile_freq): F-1 (profiler warmup), F (active), F+1 (trace export lands in its interval).
Step ms is taken from the log timestamps is too coarse (1 s), so it is derived from tps:
ms = local_batch * seq / tps * 1000 (torchtitan's tps = tokens since the last log / wall time).
Paired ratio: for every ABBA cycle of 4 steps fully inside the window, mean(fly)/mean(asm) per cycle.
"""
import re, sys, statistics as st

TOK = 4 * 8192


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step:\s*(\d+)\s+loss:\s*([\d.naninf]+)", line)
        if not m:
            continue
        g = lambda k: (lambda x: float(x.group(1).replace(",", "")) if x else None)(
            re.search(k + r":\s*([\d,]+\.?\d*)", line))
        rows.append(dict(step=int(m.group(1)), loss=m.group(2), tps=g("tps"), mem=g("memory"),
                         gn=g("grad_norm")))
    return rows


def sched(spec):
    if spec in ("turbo", "asm", "fly"):
        return [], [spec]
    w, c = spec.split(";") if ";" in spec else ("", spec)
    t = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    return t(w), t(c)


def arm(spec, s):
    w, c = sched(spec)
    i = s - 1
    return w[i] if i < len(w) else c[(i - len(w)) % len(c)]


def main():
    path, spec, pf = sys.argv[1], sys.argv[2], int(sys.argv[3])
    warm = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    rows = parse(path)
    last = rows[-1]["step"] if rows else 0
    excl = set(range(1, warm + 1))
    if pf:
        for f in range(pf, last + 2, pf):
            excl |= {f - 1, f, f + 1}
    print(f"log {path}\nsteps {len(rows)} last {last}; excluded {sorted(x for x in excl if x <= last)}")
    print("loss per step: " + " ".join(f"{r['step']}:{r['loss']}[{arm(spec, r['step'])}]" for r in rows))
    by = {}
    for r in rows:
        if r["step"] in excl or not r["tps"]:
            continue
        by.setdefault(arm(spec, r["step"]), []).append(r)
    out = {}
    for a, rs in by.items():
        t = sorted(r["tps"] for r in rs)
        med = st.median(t)
        q = st.quantiles(t, n=4) if len(t) >= 4 else [t[0], med, t[-1]]
        out[a] = med
        print(f"arm {a:6s} n={len(t):3d} tps median {med:8.0f}  IQR {q[0]:.0f}-{q[2]:.0f}  "
              f"range {t[0]:.0f}-{t[-1]:.0f}  step ms median {TOK / med * 1000:8.1f}  "
              f"peak mem {max(r['mem'] or 0 for r in rs)} GiB")
    w, c = sched(spec)
    if "asm" in out and "fly" in out and len(c) == 4:
        tps = {r["step"]: r["tps"] for r in rows}
        rat = []
        s0 = len(w) + 1
        while s0 + 3 <= last:
            blk = list(range(s0, s0 + 4))
            if not any(s in excl for s in blk) and all(s in tps for s in blk):
                ms = lambda a: st.mean(TOK / tps[s] for s in blk if arm(spec, s) == a)
                rat.append(ms("fly") / ms("asm"))
            s0 += 4
        if rat:
            print(f"paired ABBA cycles n={len(rat)}: step-time fly/asm median {st.median(rat):.4f} "
                  f"range {min(rat):.4f}-{max(rat):.4f}  (<1 = fly faster)")
        print(f"median tps fly/asm {out['fly'] / out['asm']:.4f}")


if __name__ == "__main__":
    main()
