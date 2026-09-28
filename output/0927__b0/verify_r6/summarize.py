"""Aggregate data/*/eval.json into a PASS/FAIL table (results.md + results.json)."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAGS = ["toy_causal", "toy_full", "short_q_causal", "short_q_full", "gqa4_causal", "gqa4_full",
        "proxy_causal", "prod_causal"]
GATE = 49.0
rows, allrec = [], []
cnt = {"PASS": 0, "PASS(shared<49)": 0, "FAIL": 0, "INFO": 0}
for tag in TAGS:
    for r in json.loads((HERE / "data" / tag / "eval.json").read_text()):
        r4, r6 = r["r4_db_o"], r["r6_db_o"]
        l4, l6 = r["r4_db_lse"], r["r6_db_lse"]
        fin4 = r["r4_finite_o"] == r["numel_o"] and r["r4_finite_lse"] == r["numel_lse"]
        fin6 = r["r6_finite_o"] == r["numel_o"] and r["r6_finite_lse"] == r["numel_lse"]
        why = []
        if fin4 and not fin6:
            why.append("r6 non-finite")
        if not (r6 >= r4 - 1.0):
            why.append("o >1 dB worse")
        if r4 >= GATE and not r6 >= GATE:
            why.append("o < 49 dB")
        if isinstance(l4, float) and math.isfinite(l4) and not (l6 >= l4 - 1.0):
            why.append("lse >1 dB worse")
        if r["informational"]:
            st = "INFO"
        elif why:
            st = "FAIL"
        elif r4 < GATE:
            st = "PASS(shared<49)"
        else:
            st = "PASS"
        cnt[st] += 1
        r["status"], r["why"] = st, why
        allrec.append(r)
        trig = f"{r['trig_fired']}/{r['trig_visited']}" if r["trig_visited"] else "-"
        rows.append(f"| {tag} | {r['kind']} | {r6:.2f} | {r4:.2f} | {l6:.2f} | {l4:.2f} | "
                    f"{'Y' if fin6 else 'N'}/{'Y' if fin4 else 'N'} | {'Y' if r['bit_r6_r6_as'] else 'N'} | "
                    f"{'Y' if r['bit_r6_r4'] else 'N'} | {r['maxdlse_r6_r4']:.2g} | {r['rowsdiff_r6_r6_nvs']}/{r['rows_total']} | "
                    f"{r['r6_nvs_db_o']:.2f} | {trig} | {st}{(' ' + ','.join(why)) if why else ''} |")
hdr = ("| shape | kind | r6 o dB | r4 o dB | r6 lse dB | r4 lse dB | finite r6/r4 | r6==r6_as bitwise | r6==r4 bitwise "
       "| max abs dlse r6-r4 | rows r6!=inf-only-trigger | inf-only-trigger o dB | est. trigger (row,tile>=1) | status |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
bit_as = sum(r["bit_r6_r6_as"] for r in allrec)
txt = (f"# r6 adversarial verification -- per-case table\n\n{len(allrec)} cases; status counts {cnt}; "
       f"r6 bitwise == r6_as (always-exact) in {bit_as}/{len(allrec)} cases.\n\n" + hdr + "\n" + "\n".join(rows) + "\n")
(HERE / "results.md").write_text(txt)
(HERE / "results.json").write_text(json.dumps(allrec, indent=1))
print(cnt, "bit r6==as", bit_as, "/", len(allrec))
for r in allrec:
    if r["status"] == "FAIL":
        print("FAIL", r["shape"], r["causal"], r["kind"], r["why"])
print("max |dlse r6-r4| over scored cases:", max(r["maxdlse_r6_r4"] for r in allrec if not r["informational"] and math.isfinite(r["maxdlse_r6_r4"])))
print("max |do r6-r4| over scored cases:", max(r["maxdo_r6_r4"] for r in allrec if not r["informational"]))
