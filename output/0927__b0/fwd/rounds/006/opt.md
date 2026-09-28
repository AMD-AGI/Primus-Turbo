# Round 6 -- fast round, design act (read, look, choose, route)

Base: op/current = round 4 champion (byte-identical to rounds/006/op, `diff -rq` clean). Nothing built
in this act; every probe below touches no kernel code. Container fa-g0, physical GPU 0; the card was idle
(`rocm-smi -d 0` 0%, the two KFD PIDs on the node are on gpu_ids 51359/57865 = other cards) before and after
every batch. A dmesg monitor was armed throughout. It replayed three old page faults on 0001:04:00.0 from
10:05 (uptime 13467 s, hours before this act); no new events came in during the act.

## Read
findings/{facts,dead_ends,pool,route}.md; rounds/005/1-profiling (summary, benchmark-results.md,
5-power-wall-analysis/regime.yaml); rounds/005/2-plan/planner_step0.yaml. Round 5's plan died at its
reviewer (401), so its three candidates r5.i1.g13 / r5.i2.g14 / r5.i3.g15 never reached pool.md. They
keep their ids here. Round-4 opt.md for the ISA census method.

## Survey
`rocprofv3 --stats --kernel-trace` on benchmark.py (current, prod, 21 iters). The benchmark completed and
printed its RESULT (1.505 ms). The python process then died at exit with `corrupted double-linked list`
and hung; I killed it by PID and wrote no stats file (raw/survey_rocprofv3_crash.txt). Round 4 recorded
the same failure, so this container still has no usable survey. The kernel list is round 5's: one
dispatch per call. I did not spend more time on it.

## Probes (no code change) -- what the measurement basis actually is

### r5.i1.g13 -- predecessor-labelled per-iteration series (deep-born instrument, executed)
Prediction (planner's): the prod spread is an order effect; current after beat is fast (~1.21-1.35 ms)
and current after current is slow (>=1.55 ms). Driver: raw/g13_driver.py, which is benchmark.measure's
loop recording (arm, predecessor, ms).

| shape | process | current after beat | current after current | beat | first timed call of current |
|---|---|---|---|---|---|
| prod | current+beat | 1.752 | 1.626 | 1.21 (either side) | 1.217 |
| prod | current+copy (A/A) | -- | 1.522 (either side) | -- | 1.13-1.21, decays over ~8 calls |
| proxy | current+beat | 0.0929 | 0.0886 | 0.0819 | 0.170 (cold) |
| proxy | A/A | -- | 0.0868 | -- | |
| fast | current+beat | 0.02279 | 0.02261 | 0.01410 | |
| fast | A/A | -- | 0.01847 | -- | |

**The planner's hypothesis is refuted in direction.** Current after beat is the SLOWER population
(+8% at prod), not the fast one. The prod min (1.21 ms) is the first timed call only. Separately, beat's
presence in the process costs current +23% at fast (22.7 vs 18.5 us) whatever the immediate predecessor
was, and +2..7% at proxy. hwmon freq1_input/power1_input read after every call (raw/g13b_driver.py): freq
updates only about once a second and power reads -1, so there is no per-call telemetry on this card.

### Idle-gap probe at prod (raw/gap_probe.py, gap_prod.txt, precond_prod_run{1,2}.txt)
current / beat medians, 31 iters per condition, flush as in benchmark:
- rest >= 1-2 ms before each call: current **1.10-1.13 ms**, beat 1.21-1.24 ms. Current is 9-11% FASTER.
- no rest (benchmark's loop): current 1.63-1.66, beat 1.21-1.23.
- rest 0.2 ms: 1.47; 0.5 ms: 1.25; 1 ms: 1.13. Recovery time constant is under a millisecond.
- 2 ms rest then a 256 MB memset right before the call: 1.108, so the flush is not the trigger.
  2 ms rest then one untimed current call, then the timed one: 1.130, so one heavy predecessor is not
  enough either. Only sustained back-to-back load slows current.

### Data probe at prod (raw/data_probe.py, data_prod.txt) -- this corrects my first reading of the gap probe
| inputs | current no-rest | current 2 ms rest | beat no-rest | beat 2 ms rest |
|---|---|---|---|---|
| randn | 1.692 | 1.111 | 1.214 | 1.240 |
| K=0 | 1.307 | 1.029 | 0.979 | 0.944 |
| V=0 | 1.429 | 1.051 | 1.066 | 1.034 |
| Q=0 | 1.439 | 1.035 | 1.073 | 1.034 |
| all zero | 0.977 | 0.940 | 0.731 | 0.721 |

Beat is data-limited even from rest (1.24 randn vs 0.72 zeros), so its "flat" behaviour means it is
throttled from its first microseconds on, not that it is unthrottled. With all-zero inputs the rest and
no-rest times agree for both kernels, which says neither is throttled there, and the ratio is
**beat/current = 0.75**. Caveat, from pitfalls/measurement-traps.md "A cycle count on a power-limited
part" and "Fix one realistic input and compare arms, never data": 0.75 is a cycle ratio only if both
kernels reach the same clock on zeros. That is likely, since both are flat across rest there, but the
clock was not read per call. That is the cycle gap, the same as the 0.77
round 4 measured at the pinned 1100 MHz clock. Under sustained load the throttle multiplies time by about
1.73x for current and 1.66x for beat.
**Conclusion: the prod gap is cycles (≈0.75), plus about 4% more power throttle for current. The "current
is 10% faster after rest" number is a burst transient** (current runs unthrottled for its first ~1 ms;
beat does not) and is not headroom the gate can see. Round 5's c1 ("cut energy per tile, raise the clock")
therefore does not open a lever separate from cycles. In the sustained regime time scales with energy,
and removing instructions removes both, which is why g09 and g10 paid.

### Fast-shape sequence probe (raw/fast_seq_probe.py, fast_seq.txt)
current alone 18.55 us; beat then current 22.63; beat, current, current 22.19; beat, current x4,
current 20.75. **beat, then a full-grid current (proxy grid), then current: 18.51.** beat on the proxy
grid, then current: 22.87. current, then beat: 14.06 (beat is unaffected). beat, 20 ms sleep, current:
83 us, which is a separate idle clock-down effect, noted and not pursued.
Reading: beat leaves per-CU state that costs current about 4 us at fast. Only the CUs current actually
runs on get re-warmed (32 WGs, placement varies), and a full-grid call clears it at once. That matches
round 2's I$ reading and A0's (I$ misses 0 -> 808). r3.i1.g07 killed explicit prefetch because the
descriptor already prefetches the whole 27.6 KB, so the lever left is **how much code there is**.
Static census, round-4 descriptors: 256 v_wmma in the ISA against 64 per 64-KV tile body (32 QK + 32
PV per wave), so there are ~4 body copies: the clean loop plus the causal masked loop, each unrolled x2
for compile-time ping-pong buffer selection (kernel :882). This penalty is IN the gate: validation runs
current and beat in one process (fast 0.62 of beat in the gate against 0.76 in isolation).

## Decision
Pool (a): r5.i2.g14 is the deep round's code candidate. It removes the per-tile max machinery (30
v_max3, the max permlanes, compare, ballot, corr exp) from the serial softmax block, which is h30 item 1
and this fork's own family. The probes above say cycles and energy both pay in the sustained regime, so
its premise stands. Survey (b): the fast beat-presence penalty is the largest single term in the score
that nobody is working on, and code size is its only surviving lever. That is new idea r6.i1.g16.
r5.i3.g15 (split-KV) stays behind both: its fast premise was "0.63 is underfill", but 0.62 -> 0.76 of
that is the beat-presence penalty, and h16 "no split-k" needs a human ruling first.

Corpus: see explored.consulted in the YAML. I opened INDEX.md,
optimization/routes/1-metrics-to-techniques.md, backends/flydsl/attention/README.md and
pitfalls/measurement-traps.md (the corpus already covers the burst-vs-sustained trap; see the corpus check below).

Route written to findings/route.md. Rows 2 (g14, arm A) and 3 (g16, arm B) are the two to build; row 1
(h28) binds the build step to the h31 no-beat ranking protocol.

## Expectation (reflect checks this)
- g14 (arm A): output bitwise equal to op/current at every gate shape (the slow path is today's path).
  prod +2..+4%, proxy +2..+4%, fast +2..+5% same-session vs champion, without beat (h31). VGPR <= 512,
  0 spill. If it comes in at <= 0 with bitwise output, the max chain was not on the critical path.
- g16 (arm B): A/A-style (no beat) within ±0.5% of the champion; that is the no-regression check. In the
  gate's configuration (with beat) fast +5..+12% (a 1-2 us share of the 4 us penalty, if the penalty
  scales with code size) and proxy +1..+2%. If no-beat is neutral and with-beat is also neutral, the
  penalty does not scale with code size, and the fast lever moves to g15.

## Corpus check on the probes
pitfalls/measurement-traps.md "A warmup that never reached the steady clock" already covers the trap
the gap probe walks into ("do not sleep to cool the device", and "clock sensitivity is not uniform across
candidates ... enough to invert a comparison"). This act adds a number for this card: at prod a >=1 ms
rest turns current 0.742 of beat into 1.09 of beat, an inversion, and 0.2 ms of rest already moves it
halfway. Any probe that sleeps, syncs with host work, or times a single call measures the burst regime,
not the gate's. arch/gfx1250/profiling-surface.md: SQC_ICACHE_REQ/MISSES are among the few live
counters on this part, so g16's mechanism can be confirmed by a counter, not only by time.
Cross-backend (d): backends/flydsl/attention/recipes/hd128.md is gfx950 and bwd-centred. Its b5 (a fixed
zero softmax reference plus an MFMA row sum, confirmed present in the gfx950 FlyDSL forward) is the same
family as g14/L21/L18, but unpriced there, so it does not change the order. No hipkittens attention recipe
exists for gfx1250.

## Step 5 -- build (turn 2: "build and measure what the route says")
Compile cache cleared first (`rm -rf /tmp/flycache` in fa-g0 = FLYDSL_RUNTIME_CACHE_DIR). All GPU work via
`docker exec fa-g0`, detached with rc sentinels, card 0 idle (rocm-smi -d 0 0%, no KFD holder on gpuid 34992)
before each launch, dmesg monitor re-armed with a boot-time stamp (reports only lines newer than the arm).

### Route row 1 (h28, must): nothing to build. Its protocol binds below: arms vs champion ranked in
processes WITHOUT beat; beat only in its own process.

### Route row 2 = arm A = r5.i2.g14 (working copy rounds/006/op; raw/armA_g14.diff)
- `_softmax(speculative=True)`: pass 1 masking as today, no row-max tree, no max permlane, no corr exp;
  p = exp2(fma(s, log2e, -m_prev*log2e)) (the deferred path's exact expression); sum tree as today;
  trigger = ballot(any row's per-lane half-sum > e^7) (one logit unit of margin under today's e^8
  threshold for fma/exp2 rounding; both halves of a q row are in the same wave, so the ballot sees every
  lane today's row_max test sees).
- `main_loop`: the fast pass runs first; a wave-uniform `scf.if` on the trigger reloads K(j) from its
  still-resident LDS slot (k_curr -- the t+1 prefetch writes the other slot), re-runs `_qk_gemm`, runs
  TODAY's `_softmax`, and reloads V(j) (so the slow path never holds K, S and V at once -- the pre-softmax V
  burst is dead on that path). The if yields p, m, d, corr, rescale flag (i32) and the V fragments.
- Compile-only (fast shape, raw/descriptors_A.txt): VGPR 448 (champion 456), SGPR 105, scratch 0, spill 0.
  ISA 6064 lines vs 4091 (+48%: the slow path is a full QK+softmax+V copy in each of the 4 bodies):
  v_wmma 256 -> 384, v_exp 264 -> 520, ds_load_b128 160 -> 288, ds_load_tr16 128 -> 256.
  The code growth works AGAINST row 3's I$ mechanism; noted, measured below.
- Build failures: none (compiled first time).

### Precision -- a defect found and fixed inside the mechanism
- First gate run (raw/adv_A_v1_vs_current.txt, raw/diffloc_A_v1.txt): bitwise vs op/current on every input
  that always takes the slow path (big, big_ramp), NOT bitwise where the fast path runs (randn, ramp,
  late_high): lse 1 ULP (9.5e-7) on 554/8192 rows at fast, o 3/1M elements, SQNR equal to 2 decimals.
- Mechanism: today's d update `fadd(fmul(corr,d), fadd(own, peer))` carries `fast` (reassoc) flags and
  LLVM re-associates it (champion ISA: a dual fmac with the peer sum, then the own sum added last). So the
  stale path is NOT `d + (own + peer)` in ULPs even though corr == 1 exactly. Fix (same mechanism, not a
  new one): both d updates use `fadd_t` (fast minus reassoc), so the order is as written and
  fma(1, d, x) == d + x makes the paths equal by construction.
- Exactness proof: reference build `curdfix` = op/current + ONLY that d-order change (raw/curdfix.diff).
  raw/adv_A_vs_curdfix.txt: arm A == curdfix **bitwise on all 52 cases**: toy/short_q/unequal_seqlen
  causal+full, sq_gt_skv, fast, proxy, prod x {randn, big (+-1e4 logits), ramp (max rises ~17 logits per
  tile -> rescale every tile), outlier (one huge key per 256), late_high (tile 0 far below the rest),
  big_ramp}. All finite. SQNR vs the fp32 eager reference identical to op/current's at 2 decimals in every
  case. vs op/current itself: lse max |diff| 1.9e-6 at prod randn (3e-5..6e-5 on the ramp inputs, where
  d is large), from the add order only.
- (The `big` rows' o SQNR of ~21-25 dB is op/current's too, identical: bf16 P of a near-one-hot softmax.)
- ut (raw/ut_A.txt): rc 2, the SAME 7 edge cases at 49.82-49.99 dB under ut's hard-coded 50 dB, with dB
  lines IDENTICAL to round 4's champion (diffed); spec shapes PASS; determinism 200/200 bitwise, new hash
  o=9eb9b58556bf lse=bbf23654e600 (champion's 26a89a2db0cd/d6ac8da1e101 -- differs by the add order).
- validation.py on the working copy (raw/val_A.txt): **exit 2**. Correctness 16/16 PASS at the spec's
  49 dB (min o 49.82 dB short_q full, same as the champion), determinism PASS (200/200), speed FAIL
  geomean 0.7597 vs beat (fast 0.6105, proxy 0.9258, prod 0.7757). The failure is the speed gate only,
  as in round 4.
- Final census (post-fix, raw/descriptors_AB_final.txt): VGPR 448, SGPR 105, 0 scratch, 6054 ISA lines.

### Route row 3 = arm B = r6.i1.g16 (scratch ~/.cache/op-evolve-r006-scratch/g16 = op/current + raw/armB_g16.diff)
Premise corrected while reading: the ~4 body copies are 2 warp types (LO/HI traces) x 2 loops (clean +
right-masked). They are NOT an x2 unroll; the comment near the loop is about compile-time buffer parity.
So "one loop body" halves the copies (4 -> 2). There was no unroll to throw away.
- Build: `_softmax`'s pass 1 (masking) is factored verbatim into `_softmax_mask_only`. When there is no
  left edge (`ONE_BODY and not mask_left`: causal AND non-causal), one `_run_tiles` covers
  [clean_lo, n_tiles). Inside it, the right loop's masking (q_max, kv_len clamp/tail) runs under a
  wave-uniform `scf.if` on `t >= clean_hi`, which yields the 64xR masked S scalars. Clean tiles take the
  else branch with S unmasked. `_softmax` receives the result as `_pre_masked`. Built first try.
- Census (raw/descriptors_AB_final.txt): **2809 ISA lines vs 4091 (-31%)**. v_wmma 256 -> 128,
  v_exp 264 -> 132, ds_load_b128 160 -> 96, VGPR 444 (champion 456), SGPR 91, 0 scratch/spill.
- Correctness: raw/adv_B_vs_current.txt shows **bitwise == op/current on all 52 adversarial cases**
  (causal and full, including the non-causal kv_len tail that now also runs merged). raw/ut_B.txt shows
  the same 7 edge cases at 49.82-49.99 dB under ut's hard-coded 50 dB as the champion. Determinism
  200/200, hash o=26a89a2db0cd lse=d6ac8da1e101 = **the champion's hash**.
- SQC_ICACHE counters: not taken. rocprofv3 crashes this container at exit (see Survey). The code-size
  claim rests on the census plus time.

## Step 6 -- measure (raw/bench_r6.sh, raw/bench_r6/, raw/bench_r6_agg.txt)
Protocol (h28/h31): benchmark.py unprofiled, median of 101, 8 s continuous warmup, palindromic order,
one process per shape, fa-g0 / GPU 0, idle before (0% use, no KFD holder on gpuid 34992). The dmesg
monitor showed no new events.
- Batch 1: arms A, B, inc = rounds/004/op (the best-ever champion of every shape), cur =
  job_context/op/current (a copy of r4 = A/A floor). NO beat in the process. 3 sessions with cyclically
  rotated order.
- Batch 2: gate form, X + beat, one process per (X, shape) for X in inc, A, B.

The clock regime has moved since round 4: sclk 2306-2319 MHz at fast, 1245-1531 at proxy/prod (round 4:
1019-1100 VR-throttled). Absolute TF/s and score are not comparable with rounds 1-4. Only same-session
ratios are.

Batch 1, TF/s mean of 3 sessions (per-session ratio vs inc):
| shape | A (g14) | B (g16) | inc (r4) | cur (A/A) |
|---|---|---|---|---|
| prod | **1524.08** x1.0572 (1.0585/1.0545/1.0586) | 1416.70 x0.9827 (0.981/0.985/0.982) | 1441.65 | 1443.67 x1.0014 |
| proxy | **1583.64** x1.0234 (1.025/1.020/1.025) | 1544.68 x0.9982 (1.001/0.992/1.001) | 1547.47 | 1544.64 x0.9982 |
| fast | 97.40 x0.9638 (0.962/0.971/0.959) | **104.75** x1.0366 (1.049/1.010/1.051) | 101.06 | 99.08 x0.9804 |
A/A floor: prod +-0.4%, proxy +-0.6%, fast +-2%.

Batch 2 (gate form, ratio to the beat in the same process):
| shape | inc | A | B | beat (mean of the 3 processes) |
|---|---|---|---|---|
| prod | 0.7372 | **0.7752** | 0.7298 | 1796.73 |
| proxy | 0.9092 | **0.9358** | 0.8964 | 1663.53 |
| fast | 0.6357 | 0.6322 | **0.6566** | 151.19 |
Gate-form mean: A 0.7811, inc 0.7607, B 0.7609.

### Verdicts
- **Row 2, g14 (A): delivered, and it wins.** Mean vs inc 1.0148 (geomean 1.0140), above the 0.70% bar.
  prod +5.7% and proxy +2.3% in every session, both above the prediction's +2..4% band.
  **fast -3.6% in every session**, outside the +-2% floor and against the prediction (+2..+5%). Likely
  mechanism: +48% code (a full slow-path QK+softmax+V copy in each of 4 bodies) costs I$ at fast, the
  shape with short kv loops. That is B's term, and B shows its sign. It is not proven by a counter.
  Acceptance, from my numbers: throughput x1.0148 > 1.0070 PASS; below-target shapes >= 95% of their best
  (fast 0.964, proxy 1.023, prod 1.057) PASS; no shape reached its target, so no target loss. PASS.
  Score (this_round / target, target = beat measured this session, margin 0%): prod 0.8483, proxy 0.9520,
  fast 0.6442, **mean 0.8148** (inc on the same basis 0.8004). Gate-form score 0.7811.
- **Row 3, g16 (B): delivered, and it lost at prod.** It is not shipped.
  - No-beat check: required to stay within the floor. It did not. prod -1.7% in all 3 sessions, beyond
    the 0.4% floor.
  - Mechanism, where the route predicted a gain: fast +3.7% without beat (2 of 3 sessions above the
    floor) and +3.3% in gate form (0.6566 vs 0.6357), inside the low end of the predicted +5..12%.
  - Beat-presence penalty at fast (no-beat -> gate-form TF/s, one gate sample each): inc 101.06 -> 95.82
    (-5.2%), B 104.75 -> 101.24 (-3.4%). Code -31% moved it by about a third. So the penalty scales with
    code size in part only.
  - Where the prod loss comes from: the clean tiles now pay the uniform branch and the 128-scalar yield
    around pass 1. v_cndmask 147 -> 153 is not the cost; it is the loop-carried/yield copies around the
    if. This is an inference, not a counter.
- **Merge A+B: not built.** Route row 3 says merge only if neither lost beyond the floor; A lost fast and
  B lost prod. The two are complementary on paper (B cuts exactly the code A adds; A's prod gain is 3x
  B's prod loss), so the merge is the obvious next candidate. It is a new build for the next round, not
  this one.
- **Row 4 (g04 split-KV): its condition did not fire cleanly.** The code shrank >= 30%, but the with-beat
  penalty did move (-5.2% -> -3.4%). It stays conditional and still needs the h16 ruling.

Shipped in the working copy: arm A (r5.i2.g14), rounds/006/op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py,
exactly as measured. B stays in scratch plus raw/armB_g16.diff. No restore was needed.
