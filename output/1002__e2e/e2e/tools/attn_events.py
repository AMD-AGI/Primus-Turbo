#!/usr/bin/env python3
"""Per-step attention GPU time from the adapter's CUDA-event JSONL (no torch, no GPU).

  attn_events.py <attn_ev.jsonl> <E2E_ATTN spec> <profile_freq|0> --log <main log> [--warmup 7]
                 [--ref asm] [--order asm,flyr29,fly] [--layers 32] [--json out.json]

Input lines come from attn_backends/e2e_attn/attn_timer.py (one per training step):
  fwd_ms / bwd_ms = sum over layers of the GPU time between an event pair recorded around each
  attention call -- the same scope as the trace's FA path (adapter copies, the arm's kernels, the
  ASM GQA sum; for bwd s6 the side-stream k_dqg is inside because the pair closes after the join).
Same steady window as steady_arms3.py (warmup and F-1, F, F+1 around profiled steps dropped).
Prints, per arm: n, median fwd / bwd / FA ms per step, IQR, the FA share of the step (step ms from
the log's tps), per-layer medians; per arm pair: median of adjacent-step differences (b - a) of
fwd / bwd / FA ms, i.e. the attention part of the step-time difference that steady_arms3.py measures.
Pairs use steady_arms3's FIXED arm order (--order; default asm,flyr29,fly), so a pair key means the same
difference in every process (fly/flyr29 = fly - flyr29 in p1 and in p2).
Checks: every window step has n_fwd == n_bwd == --layers and its arms match the schedule.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steady_arms3 import ORDER, TOK, arm, arm_order, parse  # noqa: E402


def med(xs):
    return st.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("spec")
    ap.add_argument("pfreq", type=int)
    ap.add_argument("--log", required=True)
    ap.add_argument("--warmup", type=int, default=7)
    ap.add_argument("--ref", default="asm")
    ap.add_argument("--order", default=ORDER, help="fixed arm order (pair direction later/earlier)")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--json")
    a = ap.parse_args()
    ev = {}
    for line in open(a.events):
        line = line.strip()
        if line:
            r = json.loads(line)
            ev[r["step"]] = r
    rows = parse(a.log)
    ms = {r["step"]: TOK / r["tps"] * 1000 for r in rows if r["tps"]}
    last = max(list(ev) + [r["step"] for r in rows] + [0])
    excl = set(range(1, a.warmup + 1))
    if a.pfreq:
        for f in range(a.pfreq, last + 2, a.pfreq):
            excl |= {f - 1, f, f + 1}
    problems = []
    for s, r in sorted(ev.items()):
        lab = r["arm_fwd"] if r["arm_fwd"] == r["arm_bwd"] else f"{r['arm_fwd']}/{r['arm_bwd']}"
        if lab != arm(a.spec, s):
            problems.append(f"step {s}: events say {lab}, schedule says {arm(a.spec, s)}")
        if r["n_fwd"] != a.layers or r["n_bwd"] != a.layers:
            problems.append(f"step {s}: n_fwd {r['n_fwd']} n_bwd {r['n_bwd']} != {a.layers}")
    ok = lambda s: s not in excl and s in ev and ev[s]["n_fwd"] == a.layers and ev[s]["n_bwd"] == a.layers  # noqa
    arms = arm_order(a.spec, a.order, a.ref)
    missing = [s for s in range(a.warmup + 1, last + 1) if s not in excl and s not in ev]
    print(f"events {a.events}: {len(ev)} steps (last {last}); window {sum(ok(s) for s in range(1, last + 1))} "
          f"steps; window steps without events: {missing if missing else 'none'}")
    for p in problems[:10]:
        print("!!", p)
    out = {"events": a.events, "spec": a.spec, "pfreq": a.pfreq, "arm_order": arms, "problems": problems,
           "missing_window_steps": missing, "arms": {}, "pairs": {}}
    for x in arms:
        ss = [s for s in range(1, last + 1) if ok(s) and arm(a.spec, s) == x]
        if not ss:
            continue
        f = [ev[s]["fwd_ms"] for s in ss]
        b = [ev[s]["bwd_ms"] for s in ss]
        fa = [ev[s]["fwd_ms"] + ev[s]["bwd_ms"] for s in ss]
        share = [(ev[s]["fwd_ms"] + ev[s]["bwd_ms"]) / ms[s] for s in ss if s in ms]
        lay_f = [med([ev[s]["fwd"][i] for s in ss if ev[s]["fwd"][i] is not None]) for i in range(a.layers)]
        lay_b = [med([ev[s]["bwd"][i] for s in ss if ev[s]["bwd"][i] is not None]) for i in range(a.layers)]
        rec = {"n": len(ss), "fwd_ms": med(f), "bwd_ms": med(b), "fa_ms": med(fa),
               "fwd_range": [min(f), max(f)], "bwd_range": [min(b), max(b)],
               "fa_share_of_step": med(share), "fwd_per_layer_ms": lay_f, "bwd_per_layer_ms": lay_b,
               "fwd_first_vs_last_third": [med(f[: max(1, len(f) // 3)]), med(f[-max(1, len(f) // 3):])]}
        out["arms"][x] = rec
        print(f"arm {x:8s} n={len(ss):3d}  attn fwd {rec['fwd_ms']:7.2f} ms/step ({min(f):.2f}-{max(f):.2f})  "
              f"bwd {rec['bwd_ms']:7.2f} ({min(b):.2f}-{max(b):.2f})  FA {rec['fa_ms']:7.2f}  "
              f"share of step {100 * (rec['fa_share_of_step'] or 0):5.2f}%  "
              f"per layer fwd {min(v for v in lay_f if v):.3f}-{max(v for v in lay_f if v):.3f} "
              f"bwd {min(v for v in lay_b if v):.3f}-{max(v for v in lay_b if v):.3f} ms; "
              f"fwd early/late window {rec['fwd_first_vs_last_third'][0]:.2f}/{rec['fwd_first_vs_last_third'][1]:.2f}")
    # adjacent in-window steps with different arms: attention differences (b - a), a = earlier-listed arm
    diffs = {}
    for s in range(1, last):
        if ok(s) and ok(s + 1):
            x, y = arm(a.spec, s), arm(a.spec, s + 1)
            if x == y or x not in arms or y not in arms:
                continue
            k = tuple(sorted((x, y), key=arms.index))[::-1]
            sb, sa = (s, s + 1) if x == k[0] else (s + 1, s)
            d = diffs.setdefault(k, {"fwd": [], "bwd": [], "fa": [], "step": []})
            d["fwd"].append(ev[sb]["fwd_ms"] - ev[sa]["fwd_ms"])
            d["bwd"].append(ev[sb]["bwd_ms"] - ev[sa]["bwd_ms"])
            d["fa"].append(ev[sb]["fwd_ms"] + ev[sb]["bwd_ms"] - ev[sa]["fwd_ms"] - ev[sa]["bwd_ms"])
            if sb in ms and sa in ms:
                d["step"].append(ms[sb] - ms[sa])
    for k in sorted(diffs, key=lambda k: (arms.index(k[1]), arms.index(k[0]))):
        d = diffs[k]
        rec = {kk: med(v) for kk, v in d.items()}
        rec["n"] = len(d["fa"])
        rec["unexplained_ms"] = (rec["step"] - rec["fa"]) if rec["step"] is not None and rec["fa"] is not None else None
        out["pairs"][f"{k[0]}/{k[1]}"] = rec
        print(f"DIFF {k[0]}-{k[1]} (adjacent n={rec['n']}): attn fwd {rec['fwd']:+7.2f}  bwd {rec['bwd']:+7.2f}  "
              f"FA {rec['fa']:+7.2f} ms/step; step {rec['step']:+7.2f} ms; step - FA = {rec['unexplained_ms']:+6.2f} ms")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
