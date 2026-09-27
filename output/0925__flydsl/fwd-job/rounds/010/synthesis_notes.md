# round 10 step 1 -- planner synthesis notes

## s1. Id collision
Both agents minted `r10.i1.g27` blind. The reviewer's candidate keeps it (fixed zero reference). The planner's
post-flush candidate (step0_planner.yaml `r10.i1.g27`) is renumbered **r10.i4.g29**. Pairing keeps r10.i3.g28.

## s2. Fixed zero reference vs the job's own h16 gate (disagreement on r10.i1.g27)
- h16 (route.md:100-104) mandates, for any change to max tracking, an adversarial test with "scores ~+-1e4 after
  scale, one huge outlier per row, rows where the first tile's max is far below a later tile's" (route.md:102-103), o and lse vs fp32.
- The kernel's exp argument is exp2 with scale*log2e folded (r4.i1.g09, packed v_pk_fma). fp32 exp2 overflows for
  argument >= 128, i.e. natural scaled score > ~88.7. With reference 0, a +1e4 score gives p = inf -> o NaN/inf.
- Underflow side: a row whose every scaled score is < ~-87 (natural) gives p = 0 for every key -> l = 0 ->
  o = 0/0, lse = -inf, where the reference is finite. The -1e4 end of the h16 range hits this.
- A guard that detects either case needs the row max per tile -- the v_max3 + peer permlanex16 + ballot chain the
  candidate exists to delete. So a guarded fixed reference is the current deferred-rescale kernel (threshold 8.0)
  with extra work, and an unguarded one fails h16 by arithmetic, not by chance.
- Why it works in Primus-Turbo hd64 (knowledge/backends/flydsl/attention/recipes/hd64.md a1): that library ships
  it; the recipe records no large-logit test, and its +13.5% is "cited, not re-measured" (hd64.md:1106-1108).
  Neither point says it is safe under h16.
- Settle for free: a torch fp32 emulation of fixed-reference online softmax on the h16 case set, before any
  build. If it produces a non-finite o or lse, or lse off the reference by >1e-3, the arm is dead with no card time.

The reviewer's "cold start is outside scoring" is right about the *beat-preceded* state (h28). But this excess is
measured with no beat in the process, so it is in the score. Round 6's current-only flush row
(rounds/006/1-opt/opt.md s3c) put `_sleep(1e5)` between zero_ and the call; benchmark.py (job_context/op/benchmark.py
113-116) does not. The scored pattern has never been isolated without beat.

## s3. The scored fast excess (disagreement on r10.i4.g29 / r9.i3.g26 / r1.i4.g04)
Planner's data, no beat in the process (rounds/009/1-opt/raw/meas/*.out):
| shape | current (arm=inc) median/min us, s1 s2 s3 | med/min | beat median/min us (beat_*.out) | med/min |
| fast  | 52.20/38.74, 51.24/39.14, 55.68/39.30 | 1.35, 1.31, 1.42 | 32.53/31.25 | 1.04 |
| proxy | 171.62/145.78, 155.55/148.58, 153.87/146.26 | 1.18, 1.05, 1.05 | 136.56/134.72 | 1.01 |
| prod  | 2015.3/1990.3, 2015.9/1988.1, 2023.6/1989.9 | 1.01, 1.01, 1.02 | 1557.7/1545.3 | 1.01 |
(val_merged/val_lock beat fast 33.13/31.77, 33.29/31.73: also 1.04.) Excess = ~12-16 us at fast, current only.
The reviewer's "cold start is outside scoring" is right about the *beat-preceded* state (h28). But this excess is
measured with no beat in the process, so it is in the score. Round 6's current-only flush row
(rounds/006/1-opt/opt.md s3c) put `_sleep(1e5)` between zero_ and the call; benchmark.py (job_context/op/benchmark.py
113-116) does not. The scored pattern has never been isolated without beat.

## s4. Discriminator for g29 (current only, fast, 101 reps, no beat in process, no build)
- A `zero_; e0; f; e1` (the scored pattern) -- expect ~52 us median
- B `zero_; _sleep(1e5); e0; f; e1` (round-6 s3c pattern) -- expect <= 41 us
- C `_sleep(1e5); zero_; e0; f; e1` -- >= 48 us: the flush drain overlaps the kernel (the g29 premise holds);
  <= 42 us: a host-gap/launch term instead, so g29 dies and the lever is host-side
- A ~= B ~= 39-41 us: there is no flush term at all; the excess is timing noise, g29 and g26 both die
Decides the premise of g29 and g26, and whether scored fast/proxy readings of any other arm are honest.
