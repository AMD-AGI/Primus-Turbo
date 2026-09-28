# Round 19 opt (fast round)

## Starting state (read)
- op/current = r13ns (round 16). rounds/019/op is byte-identical to it (diff -rq: pycache only).
- Round 18 NOT accepted: g64 (ones d-tile inside PV) randn prod x1.009 vs r17 / x1.018 vs current, but Python
  throughput 1.0027x best ever < 0.70% bar. On the REAL dumps after a GEMM burst (h47 ruler) g64 read x1.0019 = inside
  A/A, where randn under the same burst read x1.0136.
- facts.md's open question (current-bottleneck line): "Measure the rescale-branch rate on real dumps before trusting any
  chain edit's randn number". g65's pool condition (a) asks the same thing (vote rate on real dumps).
- r15 profile read (profiling_summary.md): prod power/clock-limited (cycles fixed +-1.2%, wall steps +13%), proxy tail;
  no stall/VALU/byte counters on this chip, ATT captures no FlyDSL kernels. Nothing in it prices the rescale branch.
- Route (r18): h47 is the only open must; g65 row 6 conditional on the vote-rate probe; g57/g04 blocked.
- Card: fa-g2 (physical GPU 2) 0% use at start. Node neighbours: an e2e opcheck (seconds) and then a Llama training run
  started 10:40Z -- both in container **fa-g0** (cgroup c87432 = fa-g0, rocm-smi in fa-g0 100%, fa-g2 0%). Not our
  card, but same node; noted in case the node power budget couples them.
- dmesg: the VM_PAGE_FAULT / "MES might be in unrecoverable state" lines seen at monitor start are from boot (1461 s)
  and from uptime 69803 s (the known MES hang); uptime now 102096 s. Nothing new during this round's runs (dmesg -W).

## Survey
rocprofv3 --stats / kernel-trace is closed for FlyDSL on this box (r9.i3.g22), and the kernel list is known (one
dispatch: m32x8 at proxy/prod, m32x2 at fast). This round's survey is the question facts.md left open, taken two ways:
(1) a torch emulation of the branch predicate on the real dumps (no GPU kernel code), (2) wall time of threshold arms
on the real dumps after a GEMM burst (h47 ruler).

### I1 = r19.i1 instrument: deferred-rescale branch rate, real dumps vs randn (raw/rescale_rate/)
Emulates m32x8 exactly: `need = row_max - m_prev > T` (natural logits, s = q.k/sqrt(128)), ballot over the 16 packed
rows of a q-tile (4 seqs x 4 GQA heads, head fast), m updated only when the q-tile fires, m seeded BIG_NEG (first valid
tile always fires), bottom-right causal, 64-key tiles. All 32 (b,kvh) groups x 6 layers; 4.23 M steps per set.

Prediction before running: real 10-25% at T=8 (h43 measured 13-25% for the speculation trigger), randn ~0.

Fraction of (q-tile, KV-tile) steps that take the branch, **first tile excluded**:

| set | T=0 | T=4 | **T=8 (shipped)** | T=12 | T=16 | T=24 | T=32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L00 | 0.439 | 0.268 | **0.210** | 0.177 | 0.155 | 0.126 | 0.106 |
| L01 | 0.356 | 0.264 | **0.199** | 0.153 | 0.121 | 0.079 | 0.054 |
| L02 | 0.399 | 0.217 | **0.129** | 0.083 | 0.057 | 0.030 | 0.018 |
| L08 | 0.389 | 0.300 | **0.239** | 0.193 | 0.158 | 0.110 | 0.080 |
| L16 | 0.321 | 0.169 | **0.107** | 0.074 | 0.055 | 0.035 | 0.021 |
| L31 | 0.349 | 0.253 | **0.188** | 0.143 | 0.111 | 0.070 | 0.045 |
| randn | 0.426 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 |

(First tile included, randn is 1.55% = the first tiles.) Reading: prediction held. On real data the branch runs on
~11-24% of steps (mean ~18%); on randn never. The tail is heavy: T=32 still fires 2-11%. So every randn ranking of a
change that touches the rescale branch (g64 adds a d-tile rescale there; g65 moves the max peer into it) measures a
path real data takes 1 step in 5 and randn never.

### I2 = r19.i1 (part b): threshold arms timed on the real dumps after a GEMM burst (raw/real/, h47 ruler)
Arms (raw/arms/, each differs from its base ONLY at m32x8 L166 `RESCALE_THRESHOLD`): champ2 = byte copy of cur (A/A);
tinf = 1e4 (the branch runs on the first tile only: a price ceiling, NOT a correct kernel on real data); t24, t16;
g64 = rounds/018/op; g64t24 = g64 with T=24. ab19.py = r18's ab18.py with these PATHS, condition gb (10-GEMM burst
before each timed call), prod, 6 dump layers + randn, 3 sessions in rotated orders, fresh FLYDSL cache per process, no
beat. rc 0 x3; GPU 0-4% before/after (the only other KFD pid is fa-g0's training); no new dmesg lines.

| arm | real (18 readings) geo vs cur | min..max | randn (3) geo | min..max |
| --- | --- | --- | --- | --- |
| champ2 (A/A) | 0.9995 | 0.9933..1.0047 | 1.0025 | 1.0018..1.0030 |
| tinf (ceiling) | 1.0096 | 1.0033..1.0161 | 1.0028 | 1.0014..1.0051 |
| t24 | 1.0062 | 0.9975..1.0099 | 1.0033 | 0.9993..1.0055 |
| t16 | 1.0041 | 0.9999..1.0082 | 1.0035 | 1.0022..1.0056 |
| g64 | 1.0035 | 0.9961..1.0123 | 1.0157 | 1.0135..1.0185 |
| **g64t24** | **1.0156** | **1.0064..1.0220** | **1.0140** | 1.0105..1.0161 |

Reading:
- The whole deferred-rescale branch is worth <= ~1% of prod wall on real data (tinf ceiling x1.0096). t24 recovers
  ~2/3 of it, t16 ~0.4. On randn the threshold arms equal A/A, as I1 predicts (the branch never runs there).
- **g64 + T24 is super-additive**: g64 alone x1.0035 real (r18 act read x1.0019), t24 alone x1.0062, together
  x1.0156, with the min above the A/A max. That fits the mechanism: g64 adds a 16-row d-tile rescale (one more
  accumulator tile to scale) to every branch visit, a cost randn never pays. That is why g64 read +1.6% randn and ~0 real.
  Raising T cuts visits 18% -> ~8% (I1), which gives back most of g64's real-data win. This is my
  reading of 18 readings, not a counter measurement.
- g64t24 is still below h47 item 3's ">= 2% real-dump" bar, and on randn it reads the same as g64 (T is inert there).
  Under the acceptance rule (mean over 3 shapes on randn vs best ever) it therefore scores what g64 scores: randn prod
  ~+1.4..1.8% vs current = ~+0.5% mean, under the 0.70% bar unless proxy moves too. Threshold is a real-data lever
  the scoring ruler cannot see; it rides on g64, it does not ship alone.
- Precision of T=24 (the reason HK keeps T=8 in log2 units): the O accumulator can hold up to e^24 ~ 2.6e10 times the
  final scale before the deferred fix. That is far from fp32 overflow (e^88), and the relative bf16->fp32 PV error is scale-free, so
  no loss is expected. Checked below on real dumps vs an fp32 reference and by validation.py.

### Correctness of the threshold arms (raw/val/)
- validation.py on g64t24 (fresh cache): **16/16 PASS, min o 49.82 dB** (short_q full, as every round), prod 50.91 dB,
  det 200/200 at fast. rc 2 only from the speed-vs-beat bar that op/current also fails (in-process-with-beat ratios,
  not a ranking, h31: fast 1.052 / proxy 0.902 / prod 0.950). On randn this says little, because T is inert there (I1).
- realchk.py (real dumps, prod, no timing; op/ut/common.sqnr_db vs an fp32 reference on batch 0 heads 0/13/31, full
  causal rows; all finite):

  | layer | ref o dB cur | g64 | **g64t24** | t24 |
  | --- | --- | --- | --- | --- |
  | L00 | 54.91 | 55.17 | 55.16 | 54.60 |
  | L01 | 54.20 | 54.75 | 54.74 | 53.35 |
  | L02 | 53.74 | 54.33 | 54.33 | 53.40 |
  | L08 | 52.32 | 52.55 | 52.56 | 51.98 |
  | L16 | 54.68 | 55.25 | 55.24 | 54.11 |
  | L31 | 54.84 | 55.10 | 55.09 | 54.60 |

  lse vs ref is identical across arms (55-69 dB, per layer). vs cur: g64t24 o 55.8-62.3 dB, same as g64.
  Reading: **T=24 costs nothing on top of g64** (within 0.01 dB of g64, +0.3..+0.6 dB above cur). **T=24 alone (on the fp32
  row sum) loses 0.3..0.9 dB** on every layer. That fits: g64 sums the same bf16 P that feeds the PV numerator, so a
  larger unrescaled range cancels in o = O/d, while cur's fp32 row sum does not share P's rounding. That is one more
  reason T rides on g64 only. Margin to the 49 dB gate on real data: >= 3.5 dB.

## Corpus consulted (explored.consulted)
- knowledge/ops/attention/online-softmax.md (deferred rescale / threshold; exactness argument)
- knowledge/backends/hipkittens/attention/recipes/gqa_d128.md §3 (conditional rescale, threshold 8 in log2 units),
  §6-7 (scale exp only in the rescale branch)
- knowledge/backends/flydsl/attention/techniques.md (deferred rescale knob, ballot form)
- knowledge/optimization/routes/1-metrics-to-techniques.md (Table B: v_exp = 2 issue slots; issue-bound rows)
- job_context/findings/{facts,pool,route,dead_ends}.md; rounds/015/1-profiling/profiling_summary.md;
  rounds/018/1-opt/{opt.md,raw/real/ab18.py}
- Cross-backend: the HK recipe is the only one with a tuned threshold. Its T=8 is in log2 units = 5.5 natural, lower
  than ours, and it was tuned on randn-like inputs where the branch never runs.

## Ids
- r19.i1.g66 = instrument (I1 emulated branch rate + I2 threshold arms on the real-dump ruler + realchk). Facts, not pool.
- r19.i2.g67 = idea: RESCALE_THRESHOLD 8 -> 24 on m32x8 as a rider on r17.i2.g64 (pool).
- r18.i1.g65 re-priced by I1: its own condition (a) said drop it if > ~10% of tiles vote. At T=8 real data votes
  11-24% (mean ~18%), so it fails. At T=24 it is 3-13% (mean ~8%), borderline. It stays conditional on g67.

## Expectation
g67 (= rounds/018/op with one constant changed, the arm raw/arms/g64t24) is the real-data best of anything built
since r13ns: +1.56% real after a GEMM burst (range 1.0064..1.0220, min above the A/A max), where g64 alone read
+0.2..0.35%. Precision is free with g64 (>= 52.5 dB real, 16/16 randn). I expect act to re-read g64t24 at real
x1.012..1.020 vs op/current. On randn, benchmark.py cannot see the threshold at all (bitwise equal to g64 there), so it
scores what g64 scored: prod ~+1.5..1.8% vs current, proxy ~+0.5%, fast 0, which is ~+0.3% on the mean vs best-ever
per shape, below the 0.70% bar. So I expect the scoring rule NOT to accept it, and h47's ">= 2% real" not quite met
either. It is still the right thing to deliver: the only ruler it misses is the one that never runs the branch. A
T=32 arm on g64 (fires 2-11%) is the only cheap extra, worth <~0.3% by the tinf ceiling. Beyond that, the rescale
family is exhausted.

# Act (route rows 1-5)

## Build (route row 5, r19.i2.g67), attempt 1, no failures
- rounds/019/op = rsync of rounds/018/op (g64), __pycache__ removed, then m32x8 L166 `RESCALE_THRESHOLD = 24.0`.
  `diff -r rounds/018/op rounds/019/op` = that one line; equal to the opt arm raw/arms/g64t24 except a trailing comment.
  Compile cache cleared first: /tmp/flycache removed in fa-g2, and every process runs with a fresh FLYDSL_RUNTIME_CACHE_DIR.
- Found while building: **rounds/016/op is NOT byte-identical to op/current**. rounds/016/op carries g61 (per-row
  `m_new = need.select(...)`, m32x8 L565), and op/current is r13ns (`do_rescale.select`). Both are measured below.
- raw/build/: ut/test_correctness.py --impl rounds/019/op **RESULT PASS**. validation.py: every correctness row PASS,
  **min o 49.82 dB** (short_q full), prod 50.91, det 200/200 (o=9eb9b58556bf, same hash as r18). rc 2 only from the
  speed-vs-beat bar (in-process min 0.8966), which op/current also fails.
- h16 adversarial suite (raw/build/adv_m32x8.py = r16's copy; arm=rounds/019/op, ref=rounds/018/op): finite on all
  15. Bitwise = r18 on randn (proxy/prod), late_high, full big_ramp. Not bitwise where the branch fires differently
  (big, ramp, outlier, causal big_ramp), as designed. o SQNR vs fp32 equal to r18 on big/outlier/big_ramp (22-53 dB,
  low for every arm on big), and -0.17 / -0.48 dB on proxy causal/full ramp (51.73 / 51.67 vs 51.90 / 52.15). lse
  equal. The suite's RESULT line reads FAIL only because its criterion is bitwise-vs-ref, which a threshold change
  cannot meet by construction. The ramp case (row max +17 every tile, so the branch fires every tile at T=8 and
  about every other tile at T=24) is the case built to stress this; it costs <= 0.5 dB and stays >= 51.6.

## Measure (raw/measure/, benchmark.py blocked ruler, 11:00:17-11:13:12Z, fa-g2)
- 3 rotated sessions, one process each, fast+proxy+prod, no beat in process (h31), fresh JIT cache per process.
  Arms: cand = rounds/019/op, r16/r17/r18 = rounds/01x/op, cur = op/current (r13ns), champ2 = byte copy of cur (A/A).
  Orders `cand r16 r17 r18 cur champ2 | champ2 cur r18 r17 r16 cand | r18 cand cur champ2 r16 r17`. No round before 16
  measured. beat measured in its own process right after (cur beside it). GPU 0-4% before/after every session; the
  only KFD pids listed belong to fa-g3 (another card). rc 0 everywhere; no new dmesg lines.
- Mean of the 3 session medians, TFLOP/s:

  | shape | cand | r16 | r17 | r18 | cur | champ2 | beat | best ever (>= r16) | cand / best |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | fast | **165.46** | 165.13 | 164.53 | 164.46 | 164.94 | 165.03 | 154.65 | r16 165.13 | 1.0020 |
  | proxy | **1580.41** | 1561.75 | 1566.62 | 1583.29 | 1566.28 | 1569.04 | 1667.47 | r18 1583.29 | 0.9982 |
  | prod | **1466.61** | 1448.82 | 1454.29 | 1467.79 | 1441.80 | 1446.21 | 1553.19 | r18 1467.79 | 0.9992 |

- Speed vs cur, per session: prod cand 1.0193/1.0151/1.0172, r18 1.0165/1.0191/1.0185, champ2 1.0021/1.0035/1.0036.
  proxy cand 1.0108/1.0039/1.0123, r18 1.0125/1.0075/1.0125, champ2 1.0085/1.0046/0.9923. fast m32x2 is untouched,
  so it is all noise (champ2 1.000..1.0015). **cand vs r18: prod 0.9992, proxy 0.9982**, both inside the A/A spread.
- Acceptance: throughput does not improve on the best ever (proxy and prod are r18's within noise; fast +0.2% is m32x2
  noise, since that body is byte-identical in every arm). Expected, because the threshold never runs on randn (I1).
  Score (capped, target = beat x 1.0): fast 1.0 (1.0699 uncapped), proxy 0.9478, prod 0.9443 -> 0.9640. This
  session's proxy beat read 1667 vs 1602 in r18's sweep; beat moves between processes (r18 caveat).

## h47 real-dump ruler (route row 1; raw/real/ab19act.py + agg_act.py, 3 rotated orders, 11:13-11:17Z)
| arm | real geo vs cur (18) | min..max | randn (3) |
| --- | --- | --- | --- |
| **cand (rounds/019/op)** | **1.0150** | **1.0078..1.0259** | 1.0167 |
| r18 | 1.0036 | 0.9958..1.0115 | 1.0160 |
| champ2 (A/A) | 1.0001 | 0.9961..1.0070 | 1.0040 |

Reproduces opt (x1.0156) within 0.1%: cand's min is above the A/A max, and it is +1.1% over r18 on real data. That is
still short of h47's >= 2% line. The rescale family has no big lever left (tinf ceiling), so row 1 is discharged at
this number.

## Verdict
Delivered, and left in rounds/019/op. Correct (16/16, 49.82 dB, det 200/200). On the scoring ruler it ties r18, the
best ever (prod 0.9992, proxy 0.9982): no acceptance. On real training inputs after a GEMM burst it is the fastest
body measured on the current structure (+1.5% vs op/current, +1.1% vs r18). The randn benchmark cannot see it:
this is exactly the scoring ruler's blind spot that I1 identified. Bound: power (prod, r15/r17/r18 pmc), medium confidence.
