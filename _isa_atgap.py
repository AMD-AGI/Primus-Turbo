#!/usr/bin/env python3
"""How the dQ atomics sit against the MFMA run, which is what decides whether their
latency is hidden. aiter's hd128 backward issues one atomic every eight MFMA; a burst
of atomics with no MFMA between them has nothing to hide behind.
usage: _isa_atgap.py <path to 21_final_isa.s>"""
import bisect, re, statistics as st, sys

L = [l for l in open(sys.argv[1]) if re.match(r"\s+(s_|v_|ds_|buffer_|global_)", l)]
at = [i for i, l in enumerate(L) if "buffer_atomic" in l]
mf = [i for i, l in enumerate(L) if "v_mfma" in l]
print(f"instructions {len(L)}  atomic {len(at)}  MFMA {len(mf)}")
if not at:
    sys.exit()
gaps = [bisect.bisect_left(mf, at[k + 1]) - bisect.bisect_left(mf, at[k]) for k in range(len(at) - 1)]
runs, c = [], 1
for k in range(1, len(at)):
    if at[k] - at[k - 1] <= 2:
        c += 1
    else:
        runs.append(c)
        c = 1
runs.append(c)
print(f"MFMA between adjacent atomics: median {st.median(gaps):.0f}  mean {sum(gaps)/len(gaps):.1f}"
      f"  zero {sum(1 for g in gaps if g == 0)/len(gaps):.0%}")
print(f"longest back-to-back atomic run {max(runs)}  median run {st.median(runs):.0f}")
