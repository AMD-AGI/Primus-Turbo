"""score_gqa.py RESULTS.jsonl [BASE.jsonl]: per shape and op fused / floor, and the geomean of step_us
(sum of the four fused ops; the primary metric). Floors are measured in the same run (gqa_quick_bench.py).
"""

import json
import math
import sys

KEYS = ("fwd_qkv", "fwd_o", "bwd_o", "bwd_qkv")


def load(path):
    return {(r["shape"], r["n"]): r for r in map(json.loads, open(path))}


rows = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
head = "".join(f"{k:>9}" for k in KEYS)
print(f"{'shape':<24}{'n':>3}{'step':>8}{'floor':>8}{'st/fl':>7}{'vs base':>9} | fused/floor:{head} | check")
for key, r in rows.items():
    delta = f"{(r['step_us'] / base[key]['step_us'] - 1) * 100:+.1f}%" if key in base else ""
    ratios = "".join(f"{r[k] / r[k + '_floor']:>9.2f}" for k in KEYS)
    unf = ""
    if "fwd_qkv_unfused" in r:
        u = sum(r[k + "_unfused"] for k in KEYS)
        unf = f"  unfused {u:.0f} ({u / r['step_us']:.2f}x)"
    print(
        f"{key[0]:<24}{key[1]:>3}{r['step_us']:>8.0f}{r['floor_us']:>8.0f}{r['step_us'] / r['floor_us']:>7.2f}"
        f"{delta:>9} |            {ratios} | {r['check'][:40]}{unf}"
    )
steps = [r["step_us"] for r in rows.values()]
ok = all(r["check"] in ("PASS", "SKIPPED") for r in rows.values())
geo = math.exp(sum(map(math.log, steps)) / len(steps)) if steps else float("nan")
line = f"geomean step_us {geo:.1f} over {len(rows)} shapes, all PASS: {ok}"
common = [k for k in rows if k in base]
if common:
    bgeo = math.exp(sum(math.log(base[k]["step_us"]) for k in common) / len(common))
    cgeo = math.exp(sum(math.log(rows[k]["step_us"]) for k in common) / len(common))
    line += f", vs base on {len(common)} shared shapes: {(cgeo / bgeo - 1) * 100:+.2f}%"
print(line)
