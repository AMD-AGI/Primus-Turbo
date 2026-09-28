"""Pass/fail table for the adversarial suite. Criteria (fixed before the card A/B):
  finite o and lse everywhere the fp64 reference is finite; o and lse bitwise deterministic run to run;
  o SQNR vs fp64 ref >= 49 dB, or (where the champion itself is below 49 dB) >= champion - 1.0 dB;
  lse max abs error vs ref <= max(1.5 * champion's, champion's + 1e-5)."""
import json, sys
from pathlib import Path
W = Path(__file__).resolve().parent.parent
tot = {}
lines = []
for s in sys.argv[1:]:
    r = json.loads((W / "run" / f"adv_{s}.json").read_text())
    for cn, c in r["cases"].items():
        ch = c["arms"]["champ"]
        row = [f"{s}:{cn:16s} champ o={ch['o_db']:6.2f} lse={ch['lse_maxabs']:.1e}"]
        for a, m in c["arms"].items():
            fin = m["o_finite"] == m["o_numel"] and m["lse_finite"] == m["lse_numel"]
            odb_ok = m["o_db"] >= 49.0 or m["o_db"] >= ch["o_db"] - 1.0
            lse_ok = m["lse_maxabs"] <= max(1.5 * ch["lse_maxabs"], ch["lse_maxabs"] + 1e-5)
            ok = fin and m["deterministic"] and odb_ok and lse_ok
            t = tot.setdefault(a, [0, 0, []]); t[0] += ok; t[1] += 1
            if not ok: t[2].append(f"{s}:{cn}")
            if a != "champ":
                row.append(f"{a}={'PASS' if ok else 'FAIL'}({m['o_db']:.2f}{'' if fin else ',nonfinite'})")
        lines.append(" ".join(row))
print("\n".join(lines))
print()
for a, (p, n, f) in tot.items():
    print(f"{a:8s} pass {p}/{n}  fails: {', '.join(f) if f else '-'}")
