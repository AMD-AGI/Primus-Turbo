"""Trend-table numbers from a round's benchmark_full.csv (geomeans over PASS shapes).
Usage: python trend_row.py <csv> [baseline_csv]"""

import csv
import math
import sys


def load(path):
    return list(csv.DictReader(open(path)))


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def stats(rows):
    ok = [r for r in rows if r["Check"] == "PASS"]
    f = [float(r["Forward TFLOPS"]) for r in ok]
    b = [float(r["Backward TFLOPS"]) for r in ok]
    s = [float(r["Step TFLOPS"]) for r in ok]
    rs = [float(r["step_ratio_vs_aiter"]) for r in ok if r.get("step_ratio_vs_aiter")]
    rb = [float(r["bwd_ratio_vs_aiter"]) for r in ok if r.get("bwd_ratio_vs_aiter")]
    return {
        "pass": [r["label"] for r in ok],
        "fail": [r["label"] for r in rows if r["Check"] != "PASS"],
        "fwd_avg": gm(f), "fwd_peak": max(f), "bwd_avg": gm(b), "bwd_peak": max(b), "step": gm(s),
        "step_ratio": gm(rs) if rs else float("nan"), "bwd_ratio": gm(rb) if rb else float("nan"),
    }


rows = load(sys.argv[1])
st = stats(rows)
print(f"PASS {st['pass']}  FAIL {st['fail']}")
print(
    f"fwd_avg {st['fwd_avg']:.3f} fwd_peak {st['fwd_peak']:.3f} bwd_avg {st['bwd_avg']:.3f} bwd_peak {st['bwd_peak']:.3f}"
    f" step {st['step']:.3f} step_ratio {st['step_ratio']:.4f} bwd_ratio {st['bwd_ratio']:.4f}"
)
for r in rows:
    print(
        f"  {r['label']:<8} {r['Check']:<4} step {float(r.get('Step TFLOPS') or 'nan'):8.2f} bwd {float(r.get('Backward TFLOPS') or 'nan'):8.2f}"
        f" ({float(r.get('Backward Time (ms)') or 'nan'):9.3f} ms) snr {float(r.get('dq_snr') or 'nan'):.2f}/{float(r.get('dk_snr') or 'nan'):.2f}/{float(r.get('dv_snr') or 'nan'):.2f}"
    )
if len(sys.argv) > 2:
    base = {r["label"]: r for r in load(sys.argv[2])}
    bst = stats(list(base.values()))
    common = [r for r in rows if r["Check"] == "PASS" and base.get(r["label"], {}).get("Check") == "PASS"]
    print(
        f"vs baseline: step {100 * (st['step'] / bst['step'] - 1):+.3f}%, fwd {100 * (st['fwd_avg'] / bst['fwd_avg'] - 1):+.3f}%,"
        f" bwd {100 * (st['bwd_avg'] / bst['bwd_avg'] - 1):+.3f}%  (each over its own PASS set)"
    )
    if common:
        d = [float(r["Step TFLOPS"]) / float(base[r["label"]]["Step TFLOPS"]) for r in common]
        print(f"common-PASS step geomean delta {100 * (gm(d) - 1):+.3f}% over {[r['label'] for r in common]}")
        for r in common:
            b0 = base[r["label"]]
            print(
                f"  {r['label']:<8} step {100 * (float(r['Step TFLOPS']) / float(b0['Step TFLOPS']) - 1):+.3f}%"
                f" bwd {100 * (float(r['Backward TFLOPS']) / float(b0['Backward TFLOPS']) - 1):+.3f}%"
                f" dq {float(r['dq_snr']) - float(b0['dq_snr']):+.2f} dB"
            )
