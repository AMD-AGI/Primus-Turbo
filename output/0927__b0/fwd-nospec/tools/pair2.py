"""Paired 4-step-cycle step-time ratio A/B for any two arm names (e2e/tools/steady_arms.py logic, same exclusions).
usage: pair2.py <log> <schedule> <pfreq> <armA> <armB> [warm=7]"""
import sys, statistics as st
sys.path.insert(0, "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e/tools")
from steady_arms import parse, sched, arm, TOK
path, spec, pf, A, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
warm = int(sys.argv[6]) if len(sys.argv) > 6 else 7
rows = parse(path); last = rows[-1]["step"]
excl = set(range(1, warm + 1))
for f in range(pf, last + 2, pf):
    excl |= {f - 1, f, f + 1}
tps = {r["step"]: r["tps"] for r in rows}
w, c = sched(spec); s0 = len(w) + 1; rat = []
while s0 + 3 <= last:
    blk = list(range(s0, s0 + 4))
    if not any(s in excl for s in blk) and all(s in tps for s in blk):
        ms = lambda a: st.mean(TOK / tps[s] for s in blk if arm(spec, s) == a)
        rat.append(ms(A) / ms(B))
    s0 += 4
print(f"paired cycles n={len(rat)}: step-time {A}/{B} median {st.median(rat):.4f} range {min(rat):.4f}-{max(rat):.4f}  all={[round(x, 4) for x in rat]}")
