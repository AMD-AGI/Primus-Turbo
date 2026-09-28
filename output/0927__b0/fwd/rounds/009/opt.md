# Round 9 (fast), attempt 2 (day 2, GPU 2 / fa-g2) -- opt / planning step

## State found at the start
- `rounds/009/1-opt.stale-20260927T154032/` holds attempt 1 (day 1, fa-g0). It read the findings, ran a pmc
  survey (r9.i2.g21, already in pool.md), wrote the current Route (h37 row 1, r7.i2.g18 as arm 2), and built and gated
  B (h37), G (g18) and BG in ~/.cache/op-evolve-r009-scratch. Correctness passed on all three (ut determinism 200, the
  verify_r6 192-case suite on BG). **It never measured speed.** The machine handover interrupted it.
- **The working copy `rounds/009/op` is NOT a clean copy of op/current.** It holds the BG merge: the m16x8 file is
  added, and impl.py and m32x8 differ. That is attempt 1's merged tree. The build step must build B and G from
  op/current regardless, and must know that this tree is BG.
- Card: h38. Physical GPU 2, container fa-g2, which was restarted ~4 min before this round. Idle check: rocm-smi
  inside fa-g2 reports 0% busy and no KFD pids, and pgrep shows no python. The host dmesg shows MES failures on
  0002:04:00.0, which is GPU 1 (wedged per h38) and not ours.
- The h35 ruler investigation (PT/output/0927__b0/ruler) left no report: its runs stop at a04 (day 1). The A/A protocol stays.

## Read
- findings: facts.md, dead_ends.md, pool.md, route.md (operator tables incl. h37, h38, and attempt 1's Route).
- rounds/005/1-profiling/profiling_summary.md (round 5, before rounds 6-8's changes): proxy/prod = **power**
  (data-dependent time: K zeroed -22.7%, all zeros -38.2%; sclk 1.0-1.27 GHz). fast = **latency/underfill**
  (32 WGs on 256 CUs, top clock, data-insensitive). Its c3 names "split-KV at small sq" as the alternative. The c1
  "energy per tile" direction is what g14 (r6) cashed. This still frames the plan: fast is a grid lever, and
  prod/proxy need fewer dynamic ops per WMMA.
- rounds/008/2-reflect/reflect.yaml: B (m16x8) held at x1.294 fast. nodelay failed in sign. Rejected on proxy noise.

## Survey 1: `rocprofv3 --kernel-trace --stats` on benchmark.py (current+beat, all shapes, 51 iters, 4 s warmup)
Script raw/survey.sh, rows raw/survey_results.txt. Bulk goes to ~/.cache/op-evolve-r009b-scratch/survey, because
**/tmp inside fa-g2 is not the host's /tmp**. The first launch wrote to a /tmp path that did not exist in the
container, so nothing ran (checked with pgrep, then relaunched).
- Expected: ratios near r8's no-beat ones (fast ~0.70, proxy ~0.95, prod ~0.84). A prod ratio >= 0.95 or <= 0.75
  would mean the card move changed the regime.
- Measured (gate form: beat in process, under kernel-trace):
  | shape | current ms | beat ms | current/beat | sclk MHz |
  |---|---|---|---|---|
  | fast | 0.02283 | 0.01450 | **0.635** | 2307-2311 |
  | proxy | 0.09418 | 0.08437 | **0.896** | 1107-1159 |
  | prod | 1.64988 | 1.25224 | **0.759** | 995-1118 |
- Reading: these ratios match round 5 on GPU 0 (0.634 / 0.903 / 0.734) and round 6's gate form (0.632 / 0.936 / 0.775).
  They sit below my no-beat prediction by the known beat-presence penalty (h28). **GPU 2 is in the same regime as
  GPU 0**: fast at top clock, proxy/prod at 1.0-1.16 GHz (power). Nothing here re-orders the plan.
- Instrument finding: benchmark.py under rocprofv3 finishes every shape and then dies at exit with
  `malloc(): unsorted double linked list corrupted`. The process then **hangs** (0% GPU, 6 min) and rocprofv3 never
  writes its CSVs, so the kernel list is lost. I killed it myself (pid 387 in fa-g2, rc 137). The RESULT lines survive.
  So use a minimal driver for traces (below).
- rocm-smi --showpids inside fa-g2 lists **host-wide** KFD pids. The two pids seen after the run were fa-g3's bwd
  benchmark and an fa-g0 benchmark that started after our run, each on its own card. GPU 2 use was 0% before and after.
- dmesg: the monitor's backlog shows GPU 2 page faults at uptime 55 888 s. Uptime now is 70 077 s, so those are
  ~4 h old, before the container restart. No new line during this round. GPU 1 (0002:04:00.0) keeps logging MES
  failures (wedged, h38).

## Corpus (c, d)
- optimization/routes/1-metrics-to-techniques.md, row A6 (grid fill): below the crossover only tile size or dispatch
  helps, and no inner-loop change recovers it. fast is 32 WGs on 256 CUs, so this supports h37 (BLOCK_M=128) and its
  successor g20 (BLOCK_M=64).
- backends/flydsl/attention/README.md and recipes/hd128.md section 6: gfx950, and the ranked mechanisms are
  backward band width. A poor match for this config, so I took nothing.
- backends/hipkittens/attention/recipes/gqa_d128.md section 6: gfx950, backward, hand-pinned registers (1.72x vs
  compiler-allocated at 1 wave/SIMD). The mechanism (compiler RA at 1 wave/SIMD spills to AGPR moves) is relevant only if
  L12 (4-wave) ever reaches us via g20. Then the 4-wave body's v_accvgpr/v_mov census is the first thing to check.

## Survey 2: kernel-trace on the minimal driver (current vs working copy, fast+proxy, no beat) -- no data
raw/survey_kt.sh. All four runs gave rc 0 and DONE, but rocprofv3 recorded **zero kernels**: output generation 0.6 ms,
no files anywhere in the container. It is the same as round 8 in fa-g0. Recorded as **r9.i3.g22** (instrument,
closed). Attempt 1's pmc rows stand for the dispatch structure: 1 fmha dispatch per call, current 32 WGs vs B 64 WGs
at fast, GUI cycles x1.20.
I did not take the BG-vs-current fast wall number here, because it is exactly the build step's first ranked
measurement (h35 A/A protocol, 3 sessions). A one-session copy would only be a number to argue with.

## Arms on disk (checked this round)
~/.cache/op-evolve-r009-scratch/{B,G,BG,champ2} from attempt 1. `diff -rq` against op/current: B = +m16x8 file
+impl.py, G = m32x8 only, BG = both, champ2 = identical. rounds/009/op == BG. Correctness (ut determinism 200,
the verify_r6 192-case suite on BG) was run on GPU 0 in attempt 1. **validation.py must be re-run on GPU 2.**

## Decision
- The pool's open entries were weighed as follows. h37 is a must with the job's largest measured lever (fast
  x1.294). r7.i2.g18 is disjoint in lines and holds the best-ever proxy. r8.i1.g19 needs h37 landed. r9.i1.g20 is
  blocked on L12. r1.i4.g04 / r5.i3.g15 (split-KV, deep-born) is outranked by h37 for the same underfill, with no
  combine pass and no h16 ruling needed. r1.i2.g02 / h3 lost proxy in round 1. r6.i1.g16 and r7.i1.g17 are
  code-size levers, worth +1.6..2.9% at most, next to the x1.29 grid term.
- Nothing measured on GPU 2 re-orders this. The ratios match GPU 0's regime.
- **Build: arm 1 = h37 (B), arm 2 = r7.i2.g18 (G), merge BG only if neither lost beyond the in-process A/A spread.**
  This is the same decision as attempt 1, re-confirmed on the new card.

## Expectation (a number to be wrong about)
No beat in process, vs current, mean of 3 rotated sessions with champ/champ2 A/A:
- B: fast x1.22..1.30, prod and proxy x1.000 +-0.5% (byte-identical kernels there).
- G: proxy +1.0..1.6%, prod +0..0.3%, fast +1..2%.
- BG: fast ~x1.25, proxy +1..1.6%, prod +0..0.3%. Gate-form score up ~8% vs current's 0.635/0.896/0.759 survey
  ratios, i.e. fast to ~0.80 of beat.
If B's fast gain on GPU 2 is below x1.15, the underfill model is card-sensitive and g20 should drop in rank.

## After
GPU 2 at 0% busy, no process of ours alive in fa-g2, and no new dmesg line for 0003:04:00.0 since the round began.
Route rewritten (findings/route.md, operator tables byte-unchanged). Pool: +r9.i3.g22.

## Consulted
knowledge/optimization/routes/1-metrics-to-techniques.md; knowledge/backends/flydsl/attention/README.md;
knowledge/backends/flydsl/attention/recipes/hd128.md (section 6); knowledge/backends/flydsl/attention/techniques.md (grep);
knowledge/backends/hipkittens/attention/recipes/gqa_d128.md (section 6); rounds/005/1-profiling/profiling_summary.md and
benchmark-results.md; rounds/008/2-reflect/reflect.yaml.

---

# Build step (attempt 2, GPU 2 / fa-g2)

## Build
No new build. The route's arms already exist from attempt 1, and each was checked this round as op/current plus
exactly its own change: B (h37) = ~/.cache/op-evolve-r009-scratch/B, G (r7.i2.g18) = .../G, BG = rounds/009/op
(the working copy, == .../BG), champ2 = byte-identical current (the h35 A/A). The compile cache (/tmp/flycache,
~/.flydsl/cache in fa-g2) was cleared at the head of raw/correct.sh, so every arm recompiles from source on this card.
No build failures.

## Correctness (raw/correct.sh; out in ~/.cache/op-evolve-r009b-scratch/correct)
ut test_correctness.py --determinism 200 on B, G and the working copy, then validation.py on the working copy.
The 192-case verify_r6 adversarial suite ran on BG in attempt 1 (GPU 0, same arch, same code, 192/192 bitwise ==
r4 on the m16x8 path). I did not repeat it.
Results (raw/correct/):
- ut: B, G and the working copy all PASS (rc 0). Determinism 200/200 bitwise. The hashes are B = wc = o=26a89a2db0cd
  lse=d6ac8da1e101 (r4's, because at fast m16x8 runs the r4 body) and G = o=9eb9b58556bf lse=bbf23654e600 (current's).
  Identical to attempt 1 on GPU 0.
- validation.py on the working copy: **rc 2**. Precision and determinism pass, with worst o SQNR 49.82 dB against the 49 dB gate
  (unchanged since r4). The only failure is the speed bar vs beat: gate-form geomean 0.8471 (fast 0.8250, proxy 0.9395,
  prod 0.7842). For comparison, the survey's current/beat ratios in gate form were 0.635 / 0.896 / 0.759 (51 iters,
  under kernel-trace, so this is not a same-session comparison).

## Measure, batch 1: no beat in process (h31), 7 arms, 3 rotated sessions, PYTHONHASHSEED=0 (raw/bench_r9.sh, raw/bench_r9_agg_batch1.txt)
Arms: B (h37), G (g18), BG (= working copy), r6 (the incumbent = op/current), champ2 (byte-identical current, h35 A/A),
r7, r8. The references r6/r7/r8 were re-run in these same processes. Mean TF/s (ratio vs r6):
| shape | B | G | BG | r6 | champ2 | r7 | r8 |
|---|---|---|---|---|---|---|---|
| prod | 1495.65 (1.0009) | 1500.38 (1.0041) | **1500.58 (1.0042)** | 1494.28 | 1496.14 (1.0012) | 1501.69 (1.0050) | 1496.62 (1.0016) |
| proxy | 1555.73 (0.9904) | 1601.11 (1.0193) | **1587.21 (1.0104)** | 1570.80 | 1564.16 (0.9958) | 1596.26 (1.0162) | 1560.11 (0.9932) |
| fast | 137.69 (1.2604) | 116.57 (1.0671) | **139.30 (1.2751)** | 109.24 | 109.07 (0.9984) | 116.66 (1.0679) | 139.09 (1.2733) |
- A/A (champ2 vs r6): prod per session 1.0038 / 0.9990 / 1.0010. That is within 0.5%, so **prod is valid**. Proxy is 1.029 /
  0.982 / 0.978, beyond h35's 0.5%, so proxy resolves only ~3%+ effects this session. Fast is 1.012 / 0.992 / 0.992.
- sclk: prod 1251-1278, proxy 1307-1577, fast 2306-2319 MHz.
- **Arms vs expectation**: B fast x1.260, inside the x1.22-1.30 prediction. prod 1.0009 and proxy 0.990 are inside the A/A spread
  (byte-identical kernels there). G proxy x1.019, at the top of the +1.0-1.6% prediction. prod x1.004 and fast x1.067 are above the
  +1-2% predicted for fast, which agrees with r7's own fast figure (r7 = G's tree). Neither arm lost beyond the spread, so the
  merge was measured.
- **Merge BG**: fast x1.275, proxy x1.010, prod x1.004. It holds B's fast and G's prod. Proxy sits between B and G, all inside
  proxy's A/A noise. The split B-vs-G-vs-BG at proxy **cannot be resolved at this effect size**.
- BG against the best of each re-measured reference: prod vs r7 0.9993, proxy vs r7 0.9943, fast vs r8 1.0015. BG is the union of
  r7 (proxy/prod) and r8 (fast) in one tree. That is its whole value: it reaches each shape's best in one tree, not beyond it.
  All three are within noise of their per-shape best.
- **Shipped: BG** (rounds/009/op). It is the only arm at the best level on all three shapes. B loses G's proxy/prod, and G loses
  B's fast.

## Measure, batch 2: gate form (beat in process, as validation.py runs it), 1 session (raw/bench/wb_*.json)
X / beat, TF/s:
| shape | r6 | BG | B | G | r7 | r8 |
|---|---|---|---|---|---|---|
| prod | .7854 | .7800 | .7894 | .7918 | .7892 | .7892 |
| proxy | .9316 | .9441 | .9305 | .9384 | .9400 | .9280 |
| fast | .6209 | .8285 | .8278 | .6308 | .6239 | .8314 |
- Beat, mean over the 6 processes: prod 1751.01, proxy 1623.25, fast 152.74 TF/s.
- Gate-form means: BG 0.8509, r6 0.7793. Fast gains 0.21 in gate form as well, so h28's beat penalty does not cancel B.
  prod's gate-form spread (.780-.792) is the one-session noise. BG's prod .7800 is not a loss: batch 1 has BG = r7 at prod
  with no beat.

## Score (r8 convention: no-beat means of batch 1 / beat means of batch 2, target = beat)
| shape | BG | beat | ratio | champion (re-measured) | vs champion |
|---|---|---|---|---|---|
| fast | 139.30 | 152.74 | 0.9120 | r8 139.09 | 1.0015 |
| proxy | 1587.21 | 1623.25 | 0.9778 | r7 1596.26 | 0.9943 |
| prod | 1500.58 | 1751.01 | 0.8570 | r7 1501.69 | 0.9993 |
**Score 0.9156.** On the same beat, same sweep: r8 0.9088, r7 0.8683, r6 0.8454.

## Acceptance reading
- Throughput beats the best ever: 0.9156 > r8's 0.9088, both re-measured in this sweep. Yes.
- Every shape is below target, and each is >= 95% of its own best: 1.0015 / 0.9943 / 0.9993. Yes.
- No shape had reached its target. Nothing to hold.
- Honest size: the gain over r8 is proxy alone (+1.7%, g18's term), and it sits inside this session's proxy A/A spread
  (+-3%). Otherwise BG = r8 at fast and BG = r7 at prod. What the round delivers is the union of r7 and r8 in one tree,
  not a new per-shape best.

## Expectation check
- B fast x1.260 is inside x1.22-1.30. prod/proxy are inside the spread, as predicted.
- G proxy x1.019 is at the top edge of +1.0-1.6%. prod x1.004 is slightly over +0-0.3%. fast x1.067 is over +1-2%: the
  prediction missed the m32 fast path, which also runs clean tiles.
- BG gate-form fast 0.8285 against the ~0.80 predicted. The gate-form mean moved 0.7793 -> 0.8509 (+9%) against the ~+8% predicted.
- The underfill model holds on GPU 2 (x1.26 > x1.15), so g20 keeps its rank.

## Left in place
rounds/009/op = BG (impl.py m16x8 gate + fmha_fwd_prefill_a16w16_m16x8.py + g18's one-condition change in m32x8). Not reverted.
GPU 2: 0% busy after the runs, no live process in fa-g2, no new dmesg line for 0003:04:00.0.
