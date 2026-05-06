# Round 78 — BF16 grouped: H4-elimination via RRR N-tail MFMA path REVIVED from R31 backlog (no code change, +20 score multi-round target)

## Round / date / GPU / sha

- Round 78 (auto-loop run 1/100, fresh chat).
- 2026-05-05, HIP_VISIBLE_DEVICES pinned to 3 by auto_optimize.py
  (MI355X / gfx950 / NUM_CUS=256).
- Primus-Turbo coming in: `5969104` (R77 dispatch CLOSURE on
  tiles_m=11 CRR + tiles_m=8 RCR).
- HK SHA: `9a860d59` (last touched at R65 docs); no kernel change
  this round.

## Coming-in state

```
score          = 874  (3-sample paired: 874 / 875 / 874 → ±1 noise)
correct_fail   = 0/24
above_target   = 0/24    (all 24 shapes < 1.25 ratio)
below_target   = 24/24
gpt_oss family geomean = 1.0763  (target 1.25, weight 3×, gap +0.17)
DSV3 family geomean    = 1.1190  (target 1.25, weight 1×, gap +0.13)
Qwen3 family geomean   = 1.1140  (target 1.25, weight 1×, gap +0.14)
```

Lowest-progress shape this round (sorted ascending):

| rank | shape | ratio | progress | weight |
|---|---|---|---|---|
| 1 | gpt_oss-GateUP-B32-M2048 | 1.049 | 0.839 | 3 |
| 2 | gpt_oss-Down-B32-M2048   | 1.053 | 0.842 | 3 |
| 3 | gpt_oss-Down-B4-M2048    | 1.057 | 0.846 | 3 |
| 4 | gpt_oss-GateUP-B4-M2048  | 1.078 | 0.862 | 3 |

The lowest-progress shape rotates from R77's `gpt_oss-Down-B4-M2048`
(1.054, +1.0pp lift over recent baseline) to `gpt_oss-GateUP-B32-M2048`.
R26 fully audited the dispatch rule for this shape (40 cells coarse +
7-trial × 200-iter tight verify on top-8) and committed
`(gm=4, num_xcds=4)` at flat optimum. R77 closed two more dispatch
rules. The dispatch surface is **fully audited** now.

## The lever that's been sitting on the table since R31

**R31 finding** (round-31-bf16-grouped-h4-gate-tighten-FALSIFIED):

> "H4 is NOT a 393 µs overhead — it is the **fast-path switch** that
> reroutes the bwd dA from a broken+slow native RRR path onto the
> working+fast RCR fuse path. The 393 µs transpose pays for itself
> ~10× over by avoiding the native RRR N-tail slow path."
>
> "GateUP-B32-M2048 hk: 1141 TF (post-transpose RCR) → 718 TF
>  (native RRR+N-tail) … H4 'cost' (393 µs) is dwarfed by the +3370 µs
>  native-RRR-N-tail penalty."

R31 explicitly recommended:

> "**R32 recommendation**: do NOT touch the H4 gate. Pivot to the
>  kernel C++ side. Start with (B) — native RRR N-tail handler is
>  the 80 % lever. `kernel_bf16_dynamic.cpp` uses an external
>  `grouped_ntail_kernel_lds_rrr<64>` for N%256 != 0; profile what
>  that launcher does and look for why it's 0.6× on K=2880 shapes."

**This recommendation was never followed up.** R32 went on to BF16
grouped-fwd dispatch tuning (`gpt-oss-gateup-bwd-da-rcr-rule`,
FALSIFIED noise). R33 same pattern. R34+ pivoted to FP8 entirely.
The N-tail-handler opportunity (R30's +20-22 score structural lever)
has been uninvestigated for 47 rounds.

## R30 wall-breakdown re-confirmed (no probe re-run needed)

R30 measured H4 transpose wall on idle MI355X with DSV3 warm-up:

| shape (B=32, weight 3×)  | H4 transpose | full wall | H4 share |
|--------------------------|-------------:|----------:|---------:|
| gpt_oss-GateUP-B32-M2048 | 393 µs       | 5710 µs   | 6.9 %    |
| gpt_oss-Down-B32-M2048   | 215 µs       | 3020 µs   | 7.1 %    |

Per-shape headroom if H4 fully eliminated:

| shape | current | post-elim | progress Δ | score Δ |
|---|---|---|---|---|
| GateUP-B32-M2048 | 1.049 | 1.127 | +0.062 | +4.6 |
| Down-B32-M2048   | 1.053 | 1.134 | +0.065 | +4.9 |
| GateUP-B32-M4096 | 1.085 | ~1.15 | +0.05  | +3.7 |
| Down-B32-M4096   | 1.076 | ~1.14 | +0.05  | +3.7 |
| (+ 4 B=4 gpt_oss shapes, smaller per-shape, ~3-5 score sum total) |

**Total potential: +20 to +22 score points.**

This is **larger than every individual lever that's landed since R7**.

## Why is `grouped_ntail_kernel_lds_rrr<64>` 0.6× of RCR?

Reading the kernel header (`kernel_bf16_dynamic.cpp:3497-3531`):

> "The legacy `grouped_tail_kernel<RRR>` recovers correctness with a
>  per-cell scalar full-K loop … This kernel mirrors the K-tail
>  variant's coop-load + transposed LDS layout … inner loop runs
>  K_CHUNK / 4 ``v_dot2c_f32_bf16`` packed dot products per cell per
>  chunk."

The N-tail kernel uses **`v_dot2c_f32_bf16`** (packed scalar dot
product, 2 muls/cycle/lane), NOT `mfma_f32_32x32x16_bf16` (1024
muls/cycle/lane). That's a per-cell ~32× compute throughput gap
(modulo MFMA pipeline fill / load bandwidth ceilings).

The reason the kernel uses `v_dot2c` instead of MFMA: per the kernel
comment at line 1700-1710, RRR layout has B with N on the column
axis, where the SRD-clamp trick that lets RCR handle partial N-tile
output via masked MFMA store doesn't trigger — OOB N reads "wrap into
the next K row's valid columns inside the per-group SRD". So the
existing kernel can't use MFMA on the partial N-tile because the
B-side reads would be wrong for OOB N cols.

The fix would have to either:

1. **Add per-load N-mask to the RRR main kernel** (kernel C++ change):
   the main kernel reads B with `buffer_load_b128` along stride-N in
   K. Add a per-load mask that zeros out lanes whose `(col_block_base
   + col_offset_in_warp) >= g.n`, so OOB N reads contribute 0 to MFMA
   accumulation. Then make the RRR persistent grid use
   `bpc = ceil_div(n, BLOCK_SIZE)` instead of floor (= fast_n / 256),
   covering the partial N-tile via MFMA. This eliminates the need
   for `grouped_ntail_kernel_lds_rrr` for the metric's gpt_oss
   shapes. Risk: the per-load mask adds VGPR / SGPR cost on the hot
   path of RRR main — could regress all 8 RRR-main metric shapes.
   Multi-round investigation.

2. **Optimize `grouped_ntail_kernel_lds_rrr` to use MFMA directly**
   on a 32×16 or 16×32 register tile via cooperative LDS staging
   (similar to how the K-tail RRR variant works for K%128 == 64).
   The N-tail region is `(N % 256)` cols wide — for gpt_oss = 64
   cols, which is exactly `HALF_BLOCK_SIZE` and matches the rt_16x32
   sub-tile width. Could plausibly tile a 256×64 output region with
   1×1 MFMA tile per CU, padded to 256×128 with mask. Risk: the
   N-tail kernel rewrite is a several-day project; the per-tile
   layout is intricate (transposed LDS scatter, K-coop load along N
   not K, etc).

Both paths require non-trivial HK kernel surgery. Neither fits a
single round.

## Why R31 + R30 missed this for 47 rounds (failure mode)

R31's note explicitly recommended the path. R32-R47 (BF16 grouped) +
R1-R52 (FP8) all worked on:

* Dispatch rule sweeps (R32-R47, R57, R66, R77) — exhausted, all
  FALSIFIED in noise band or CLOSED at flat optimum.
* MFMA scheduling discipline (R51, R54, R55, R76) — landed on MAIN
  + EPILOG, falsified on FUSE.
* Compile-time KI specialization (R52, R53) — landed.
* Work-stealing for fwd grouped (R61-R65) — landed for B=4
  imbalanced; per-XCD attempt + extension FALSIFIED.
* Var-K LDS swizzle (R74, R75) — both FALSIFIED on shape-swap or
  work-stealing variants.

**No round picked up R31's "fix RRR N-tail" thread** — the
recommendation was tagged "kernel-side, multi-round" and
de-prioritized for the smaller mechanical levers. After 47 rounds
those mechanical levers have all been exhausted (R77 closed the last
two dispatch rules), and the only structural lever left is exactly
R31's.

## R79 / R79+ recommendation — pick one path to commit to

**Path A — N-mask the RRR main kernel (preferred, 1-2 rounds):**

* Step 1 (R79): standalone correctness probe — write a Python
  script that calls the existing RRR main kernel with `bpc =
  ceil_div(n, BLOCK_SIZE)` and per-load N-mask injected via
  template arg (i.e., compile-time predicate the kernel checks at
  `buffer_load_b128` site). Verify allclose vs Triton on 4 gpt_oss-
  GateUP H4 shapes. If PASS, the lever is technically clean — risk
  reduces to VGPR pressure cost.
* Step 2 (R80): integrate by widening RRR's `bpc` ceil_div in the
  layout globals + dispatcher; adjust `grouped_tail_kernel<RRR>`
  skip predicates so the (now-MFMA-served) cells aren't double-
  counted. Run metric. Expected score: +6-8 if the per-load mask
  doesn't regress the main loop (~256 cols → 11.25 → 12 N-tiles per
  group, +6.7% extra MFMA work per launch but eliminates the
  separate N-tail launch entirely).
* Step 3 (R81): if step 2 lands +5+, narrow the H4 gate from R19's
  `K_RRR%64 != 0 OR N_RRR%256 != 0` to only `K_RRR%64 != 0`,
  avoiding the bf16_transpose_3d call on the 4 gpt_oss-GateUP
  shapes (N_RRR misaligned-only path). Expected: +10-15 incremental.

**Path B — Optimize the existing N-tail kernel to MFMA (3-5 rounds):**

* Larger rewrite of `grouped_ntail_kernel_lds_rrr<64>` to use
  `mfma_f32_32x32x16_bf16` instead of `v_dot2c_f32_bf16`. Higher
  ceiling (closer to RCR throughput) but the persistent grid
  topology, LDS scatter pattern, and K-coop load pattern all need
  re-derivation. Defer until Path A lands — Path A's 60-70% of the
  available headroom comes first.

**Path C (anti-recommendation): more dispatch sweeps.** R26 / R66 /
R77 audited every rule for the lowest-progress shape's tiles_m /
tiles_n / k / m_total bracket. Net Δ on all paired 4-run verifies in
the last 7 rounds: 0 ± 1 (noise band, see git log). Path C cannot
move the score beyond 875 ± 2.

## Things I confirmed are NOT the lever this round

To not waste subsequent rounds re-deriving:

### bf16_transpose_3d tile shape (HBM-bound, exhausted)

Sweep on /tmp/r78_bf16_transpose_sweep.py over 11 (BK, BN) cells × 4
metric H4 shapes:

| shape (BF16)        | current | best  | gain   |
|---------------------|--------:|------:|-------:|
| GateUP-B4 K>N       | 40.8 µs | 39.2  | -1.6 µs|
| GateUP-B32 K>N      | 429 µs  | 425   | -4 µs  |
| Down-B4 K==N        | 21.7 µs | 20.4  | -1.3 µs|
| Down-B32 K==N       | 214 µs  | 209   | -5 µs  |

Per-shape gains 1-6%. Per-iter total wall savings ~9 µs across all
4 calls. At 5710 µs per iter, that's 0.16% wall = +0.0016 ratio per
shape. Score impact: 8 shapes × 0.0016 × 3 weight / 40 / 1.25 =
+0.0008 weighted progress = **+0.8 score**. Below noise. R30
already characterized BF16 transpose as HBM-bound — confirmed today.

### FUSE pipeline prefetch (ceiling too low)

R76 falsified FUSE-RCR sched_barrier extension — closeout note
quantified the FUSE block's per-tile share as 4 / 348 MMAs = 1.15 %
of per-tile MMA budget. Even a perfect FUSE optimization saturates at
+1 % per-tile wall. R77's deferred B2 lever (slab-1 A-load
prefetch via 2nd register tile) carries +24-32 VGPR risk on the
already-tight 256-VGPR cap; R76 hit +6 VGPR and was at the cap. Net:
ROI < +5 score, risk medium-high. Not the right next lever.

### Var-K LDS pad (R75 closed, structural ceiling reached)

R75 added work-stealing infrastructure to var-K (+1 VGPR spill from
do_body refactor → -2 score), reverted. R74 swap-shape FALSIFIED at
+66 VGPR spill and 24/24 dB-allclose FAIL. The structural argument
in R75 ("var-K per-tile body is small, atomic contention at NUM_CUS=
256 dominates the imbalance gain") rules out re-attempt at the
current kernel scaffolding.

### Launch-overhead reduction (already gated)

R77 surface map flagged `hipMemsetAsync(counter, 0, 4)` per launch
as a candidate lever. Reading the kernel: `prime_grouped_tile_counter`
already gates on `should_use_work_stealing(M_total, bpc)` — only
fires for tiles ∈ (0, NUM_CUS*4) AND tiles % NUM_CUS != 0 (R61
gate). The lowest-progress shape (gpt_oss-GateUP-B32-M2048) has
total tiles = 32×8×22 = 5632 > NUM_CUS*4 = 1024, so the gate doesn't
fire — no memset overhead for B=32. Lever is already optimized for
the worst shape.

## Decision this round

**No code change.** Round-note documenting the analysis path so the
next agent (whether resume window or cold-start) can pick up Path A
without re-deriving R30/R31's findings or wasting rounds on
already-exhausted levers.

```
$ python3 scripts/_metric_grouped_bf16_weighted_wall.py
[metric_bf16_weighted] Goals: per-shape ratio >= 1.25
                             weighted_progress=0.8740 score=874/1000 PARTIAL
[metric_bf16_weighted] correct_fail=0/24  reject=0/24  below_target=24/24
```

## Compliance check

* No metric file modified.
* No `can_handle` tightening.
* No CPU sync / host-pad / per-group launch introduced.
* HIPKITTEN registered with `autotune=False`.
* All 24 shapes correctness PASS this round.
* HK + Primus working trees clean except for this round note.

## Files

* `analysis/_notes/round-78-bf16-grouped-h4-elim-rrr-ntail-direction-revived-from-R31.md`
  — this note.
* `/tmp/r78_bf16_transpose_sweep.py` — BF16 transpose tile-shape
  sweep confirming HBM-bound (not committed).

## Suggested R79 next step

Path A step 1: standalone correctness probe of the proposed
N-mask path. Specifically (in HipKittens repo):

1. Add a `template <bool MASK_N_TAIL>` switch on the RRR branch of
   `device_gemm_tile_body`'s `buffer_load_b128` for B reads. When
   `MASK_N_TAIL=true`, gate each lane's load result by `(col_global +
   per_lane_col_offset) < g.n`; OOB lanes write 0 to the register.
2. Add an instantiation `grouped_kernel<RRR, KI=0, MASK_N_TAIL=true>`
   alongside the existing `grouped_kernel<RRR, KI=0>`.
3. Add a `dispatch_grouped<RRR>` branch that routes the
   `(g.fast_n < g.n)` case to the masked instantiation (with bpc =
   ceil_div(n, 256)) AND skips the `grouped_ntail_kernel_lds_rrr<64>`
   launch. Keep the legacy path as fallback.
4. Run the metric's downsized correctness probe on the 4 gpt_oss
   H4-reroute RRR shapes (Down-B{4,32}-M{2048,4096} hit
   N_RRR=2880 misaligned). Demand allclose PASS on dA.
5. If correctness passes, **then** run the metric — the per-load
   mask's VGPR cost will decide the score impact. Target: +6-8
   from this round alone (eliminates the N-tail kernel's separate
   launch + the `v_dot2c` slow inner loop). Subsequent round can
   widen the H4 gate to capture the GateUP family for the rest.
