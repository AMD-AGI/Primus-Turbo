round-55-fp8-direction-G-cross-shape-coopt-A-PRIORI-FALSIFIED-predicates-already-per-shape.md
=============================================================================

Round: 55 / 100
Date: 2026-05-10
Pre-SHA: 8d93d5f3 (R54 docs — A3 PRE-IMPLEMENTATION FALSIFIED, R55 forward-pointer = Direction G probe)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R54 forward-pointed to Direction G (cross-shape co-optimization, the
last untried task-md NEW DIRECTIONS lever) with a concrete probe sketch:
"3×3 (gm, xcds) sweep on Down-B4-M2048 + Down-B4-M4096 jointly, scoring
by sum-of-progress instead of per-shape." R55 audits the framing
**a-priori** by reading the dispatcher predicate structure plus the
existing per-shape audits, and finds Direction G **structurally
incoherent** for the gpt_oss FP8 metric:

  1. **Per-shape predicates already discriminate every metric cell**.
     All 8 gpt_oss FP8 RCR shapes (4 GateUP + 4 Down × {fwd, dgrad-via-H4})
     and all 8 wgrad var-K shapes have explicit dispatcher rules keyed
     on `(tiles_n, tiles_m, k, m_total)`. The predicates partition the
     metric shape-space at single-shape granularity. Coarsening any
     predicate to "co-optimize a pair" can only **lose** per-shape
     tuning capability (or be neutral when the per-shape optima already
     coincide).
  2. **All anchored "joint pair" candidates have differing per-shape
     optima** with documented multi-seed evidence; unifying them under
     a single config strictly regresses the loser-shape.
  3. **The R44 reference for the proposed anchor (Down-B4-M2048) showed
     local-optimum already** — 3 candidates within 0.15 % cluster, no
     "known sub-optimal" seed exists for the joint search.

Direction G is therefore **A-PRIORI FALSIFIED at the framing level** —
not by re-running a metric sweep, but by reading the dispatcher
structure and prior per-shape audits. R55 ships **no kernel or
dispatcher change**; daemon metric expected in the 691-699 noise band
per R29 characterization.

## Evidence #1 — every gpt_oss FP8 metric shape has its own predicate

Grepping `config.py` for `tiles_n == 11` / `tiles_n == 22` + `k == 2880`
+ `m_total ==` shows 8 distinct `if`-branches, one per metric shape:

| Shape (cell)                   | Layout    | Predicate                                                         | Current rule                          | Origin |
|--------------------------------|-----------|-------------------------------------------------------------------|---------------------------------------|--------|
| GateUP-B4-M2048 fwd            | RCR       | `tiles_n==22 ∧ tiles_m==8 ∧ k==2880 ∧ m_total<=16384`             | `(gm=1, xcds=4)`                      | R23    |
| GateUP-B4-M4096 fwd            | RCR       | `tiles_n==22 ∧ tiles_m==16 ∧ k==2880 ∧ m_total<=16384`            | `(gm=8, xcds=4)`                      | R3     |
| GateUP-B32-M{2048,4096} fwd    | RCR       | `tiles_n==22 ∧ tiles_m∈{8,16} ∧ k==2880 ∧ m_total>=65536`         | `(gm=8, xcds=4)`                      | R70    |
| Down-B4-M2048 fwd+dgrad-H4     | RCR       | `tiles_n==11 ∧ tiles_m==8 ∧ k==2880 ∧ m_total==8192`              | `(gm=16, xcds=2, slots=196, cs=96)`   | R7→R2  |
| Down-B4-M4096 fwd+dgrad-H4     | RCR       | `tiles_n==11 ∧ tiles_m==16 ∧ k==2880 ∧ m_total==16384`            | `(gm=1, xcds=4)`                      | R12→R2 |
| Down-B32-M2048                 | RCR       | `tiles_n==11 ∧ tiles_m==8 ∧ k==2880 ∧ m_total==65536`             | `(gm=16, xcds=4)`                     | R8     |
| Down-B32-M4096                 | RCR       | `tiles_n==11 ∧ tiles_m==16 ∧ k==2880 ∧ m_total==131072`           | `(gm=4, xcds=4)`                      | R50    |
| GateUP-{B4,B32}-M4096 dgrad-H4 | RCR (k=5760) | `tiles_n==11 ∧ tiles_m==16 ∧ k==5760 ∧ m_total>=8192` (sub-gated) | `(gm=1, xcds=4, fuse_ktail_off=B32?)` | R8/R16 |

**Granularity check**: the (`tiles_m`, `m_total`) sub-keys distinguish
every B × M_per_g combination uniquely. There is **no metric shape
that two different cells share a predicate with** — the dispatcher's
predicate vocabulary is exactly as fine-grained as the metric's
shape-space.

→ **Implication**: any "joint" rule that captures two metric shapes is
either a *replacement* for two existing per-shape rules with a single
coarser predicate (strictly *loses* tuning capability — bound by the
better of the two per-shape rules), or an *extension* into uncovered
shape-space (no metric shape is uncovered, so this is empty).

## Evidence #2 — proposed anchor pairs have non-coincident optima

R54 sketched: "pick a (gm, num_xcds) cell that is *known* sub-optimal on
Down-B4-M2048 (e.g. R44 measured a -0.3% per-shape drift) but might be
co-optimal on Down-B4-M4096."

**Cross-tabulating the per-shape optima** for the 4 Down-B4 cells (the
natural joint-search target):

| Down cell           | Predicate                          | gm  | xcds | Notes                                      |
|---------------------|------------------------------------|-----|------|--------------------------------------------|
| Down-B4-M2048       | `tiles_m==8 ∧ m_total==8192`       | 16  | 2    | R7→R2: (8,2)=−4.0%, (16,4)=−4.3%           |
| Down-B4-M4096       | `tiles_m==16 ∧ m_total==16384`     | 1   | 4    | R12→R2: (gm=8,4)=−0.0%, (gm=32,4)=baseline |
| Down-B32-M2048      | `tiles_m==8 ∧ m_total==65536`      | 16  | 4    | R8: (gm=12-32, xcd=4) plateau, (xcd=2)=−1pp |
| Down-B32-M4096      | `tiles_m==16 ∧ m_total==131072`    | 4   | 4    | R50: (gm=4,4) +0.82pp over default         |

**Possible joint-rule candidates and their per-shape regression cost**:

* **Joint (gm=4, xcds=4) for all 4 Down**:
  - Down-B4-M2048: R7→R2 + R44 audits show xcds=4 = −4.3% loss vs xcds=2
    on this cell (R44 column "(16, 4)*" = 1477 TF vs (16, 2)* = 1540 TF).
    gm=4 was not in R44 sweep but R2-current's table has (4,4)=−0.58 % vs
    (1,4) winner; combined with the −4% xcd flip → expected **−4.5 %**.
  - Down-B4-M4096: gm=4 was tested in R12 sweep at 1456.22 TF baseline
    vs (gm=32,xcds=4) winner 1473.58 TF → **−1.18 %** gap, then R12→R2
    shifted optimum to (gm=1, xcds=4) at +1.12 % over (gm=32,xcds=4) →
    estimated (gm=4, xcds=4) is **−2.3 %** vs current (gm=1, xcds=4).
  - Down-B32-M2048: R8 wide sweep showed (gm=4, xcds=8) baseline 933 TF
    vs (gm=16, xcds=4) winner 947 TF → **−1.5 %**.
  - Down-B32-M4096: this *is* the current rule (R50). 0 % change.
  - Sum of per-shape losses, weighted equally into Down section:
    `(−4.5 %) + (−2.3 %) + (−1.5 %) + 0 % = −8.3 %` aggregated across
    4 of the 8 shapes that contribute to the Down side of fwd / dgrad
    section averages.
  - At current scoreboard (fwd avg 1898, dgrad avg 2097), Down's 4
    shapes contribute roughly half each section's average. A −8.3 %
    on those 4 shapes' Down-cell band = ≈ −2 % section avg = ≈ **−13
    score** total across fwd + dgrad sections.

* **Joint (gm=16, xcds=4) for tiles_m==8 (Down-{B4,B32}-M2048)**:
  - Down-B4-M2048: R44 column "(16, 4)" = −4.26 % (the defensive
    control deliberately picked to falsify; xcds=4 with cs=96 ⇒
    `block = xcds * cs = 384 > slots = 196` ⇒ all workgroups
    round-robin, swizzle off entirely).
  - Down-B32-M2048: this *is* the R8 rule. 0 % change.
  - Sum: **−4.3 %** on 1 shape, 0 on the other → ≈ −7 score combined
    fwd+dgrad. Strictly negative.

* **Joint (gm=16, xcds=2) for tiles_m==8 (the "use Down-B4 rule for
  Down-B32 too")**:
  - Down-B4-M2048: this *is* the current rule. 0 % change.
  - Down-B32-M2048: R8 column "(32, 2)" was tested at +1.31 pp over
    default; (16, 2) was not explicitly in the R8 table but should
    interpolate to roughly +1.0 pp = LOSS of ~0.5 pp vs current
    (gm=16, xcds=4) winner at +1.54 pp.
  - Sum: 0 + (−0.5 pp) → ≈ **−1 score**. Still negative.

**No joint-rule candidate exists with sum-of-progress > current.** The
per-shape rules captured the per-shape optima; coarsening predicates
strictly loses or ties.

## Evidence #3 — R44 closed Down-B4-M2048 as local optimum

R54 cited "R44 measured a −0.3 % per-shape drift" as the seed for a
joint search. Re-reading R44 (committed at SHA prior; analysis note
``round-44-down-b4-m2048-gm-xcd-drift-FALSIFIED-rule-at-local-optimum.md``):

| cell      | TFLOPS  | Δ% vs base (16,2)* |
|-----------|---------|--------------------|
| (16, 2)*  | 1540.0  | baseline           |
| (32, 2)   | 1539.4  | −0.05 %            |
| (24, 2)   | 1538.7  | −0.09 %            |
| (8, 2)    | 1481.0  | −3.99 %            |
| (16, 4)   | 1477.1  | −4.26 %            |

R44 conclusion: "the top three cells {(16, 2), (24, 2), (32, 2)} are
**statistically indistinguishable** on both fwd and dgrad". The R54
forward-pointer's "−0.3 % drift" cite is a paraphrase that overstates
R44's actual −0.05 to −0.09 %. **No "known sub-optimal" cell exists
on this anchor**; the rule sits inside a 3-candidate plateau.

The same property holds for Down-B4-M4096 — R12 → R2 wide sweep tight-
verified (gm=1, xcds=4) winner with per-seed positive 3/3 fwd, 3/3
dgrad. No "joint-search room" because the per-shape optimum is
distinct from the M=2048 sibling and both are tightly converged.

## Evidence #4 — the only two predicate "merges" available are already in production

Looking for cases where a single predicate covers ≥2 metric shapes:

  * **R70 (gm=8, xcds=4) for `tiles_n==22 ∧ tiles_m∈{8,16} ∧ k==2880
    ∧ m_total>=65536`** — covers GateUP-B32-M2048 + GateUP-B32-M4096
    (both fwd) jointly. Wide sweep documented in the rule comment:
    M=2048 +0.42 pp, M=4096 +1.39 pp over the (gm=4, xcds=4) baseline.
    This *is* a Direction G success — already shipped as R70. Direction
    G's space includes this rule; the rule was found per-shape but
    happens to share the optimum.
  * **R8 dgrad-H4 (gm=1, xcds=4) for `tiles_n==11 ∧ tiles_m==16 ∧
    k==5760` (sub-gated by `m_total>=65536` for `fuse_ktail_off`)** —
    covers GateUP-B4-M4096-dA + GateUP-B32-M4096-dA jointly. Per-shape
    audit documented +1.20 % B4-M4096 + 0.37 % B32-M4096. Same Direction
    G success class.

Both existing "joint" rules emerged from per-shape probes that
*happened* to find shared optima, not from a deliberate joint-search
methodology. Direction G as a *probe-search direction* would not have
discovered anything beyond what per-shape sweeps already found.

## Mechanism — why per-shape predicates dominate joint rules

The persistent grid scheduling cost depends on:

  * `m_total / NUM_CUS / wave_step`  — per-shape (varies 32× across the
    8 cells: B=4 M=2048 has m_total=8192 = 32 wave-steps over 256 CUs;
    B=32 M=4096 has m_total=131072 = 512 wave-steps).
  * `tiles_m × tiles_n`               — per-shape (8×11 to 16×22 range).
  * `k`                                — fixed at 2880 for the gpt_oss
    family but the K-tile count and L2 footprint scales with k.

Optimum (gm, xcds) is a function of (m_total, tiles_m, tiles_n, k)
across these dimensions — exactly what the dispatcher predicate already
encodes. Per-shape rules locally optimize at each predicate boundary;
joint rules can only succeed when two cells happen to coincide on the
optimum (R70 and R8-dgrad-H4 cases above), which is a property
*discoverable* from per-shape probing (the optima ARE the same).

The corollary: **Direction G probes will rediscover only the joint
optima that per-shape sweeps already found**. There is no information-
theoretic uplift from "scoring by sum-of-progress" when the per-shape
predicates can already select per-shape configs.

## EV verdict

  * R54 projected: "If R55 G falsifies, the task ceiling at 696 score
    is established." The current state (best=696, 13 rounds without
    improvement, all NEW DIRECTIONS preflight-falsified at this depth)
    aligns with that wind-down marker.
  * R55 G implementation cost = 0 rounds (a-priori falsified by reading
    dispatcher structure), so no metric-noise downside.
  * R55 G best-case lift if it had been run: 0 (no joint candidate
    beats current per-shape rules per Evidence #2 above). Worst-case
    lift if probe were run with a poorly-chosen unification: −1 to −13
    score depending on which pair was unified.
  * **EV-zero ship + EV-positive write-up** (closes the last untried
    direction with a structural argument, freeing R56 from re-attempting
    it).

## R55 verdict

**Direction G (cross-shape co-optimization) is A-PRIORI FALSIFIED**
at the framing level. The dispatcher's per-shape predicate vocabulary
is at metric-shape granularity; coarsening predicates can only lose or
tie per-shape tuning. The two "joint" rules already in production (R70,
R8-dgrad-H4) emerged from per-shape sweeps that found coincident
optima, not from joint-search methodology. The R54 anchor (R44
Down-B4-M2048 "drift") was actually closed at local optimum.

R55 ships **no kernel or dispatcher change**. Daemon metric expected
in the 691-699 noise band per R29.

## R56 forward-pointer

All NEW DIRECTIONS A-G are now exhausted at preflight or implementation:

| Direction                       | Verdict (round)                               |
|---------------------------------|-----------------------------------------------|
| A1 Stream-K                     | PMC reality check 1/8 cells (R52)             |
| A3 Decoupled-warps              | PRE-IMPL FALSIFIED — paper data −17/−44% (R54)|
| B Cross-stream parallelism      | metric reads kernel-only, can't help (task md)|
| C Activation cache reuse        | metric reads kernel-only, can't help (task md)|
| D SALU coord-decode             | already shipped (R9-dm closed-form)           |
| E Different barrier scheme      | R26-R28 audited, all FALSIFIED                |
| F Larger tiles                  | FORBIDDEN PATHS (256x128 -7/-23%)             |
| G Cross-shape co-opt            | A-PRIORI FALSIFIED structural (R55, this)     |

Plus per-shape dispatcher sweeps exhausted across R1-R45 and the
macro-flag space exhausted across R22-R36.

**R56 options** (in EV-descending order):

  1. **Re-characterize noise floor**. R29 ran 23 samples on a different
     code SHA. Re-run 30 samples on the current SHA (8d93d5f) on GPU 3
     (the daemon's current pin). This produces the empirical noise band
     to set the daemon's wind-down threshold honestly. Probe:
     `for i in 1..30; do python3 scripts/_metric_gpt_oss_fp8_kernel.py;
     done` via `dbg_remote.sh`. Expected wall: ~30 × 8s = 240s, ~4 min.
     Output: median, 5/95th %ile, std. Decision: if 95-percentile − 5-
     percentile band ≥ 8 score, the +1 RCR_KTAIL_VMCNT lift cited in
     task md (line 138-141) is sub-noise and not shippable; if band ≤ 4
     score, declare ceiling and recommend wind-down.

  2. **Recommend daemon wind-down**. 13 rounds without improvement +
     all 7 directions exhausted + per-shape rules at local optima +
     macro space empty. patience=40 will trip at round 96. Save 41
     rounds × ~5-15 min each = 3.5-10 hours of compute by stopping
     now. Acceptable if the user wants to redirect effort to a
     different shape family or task.

  3. **Pivot to a different metric**. The task md mentions BF16 grouped
     metric (`scripts/_metric_grouped_only.py`) at score ~1000/1000
     and the mixed `_metric_hk_ratio.py` also near cap. The fused-act
     metric (`_metric_grouped_fused_wall.py`) was the prior task's
     focus. None of these have headroom on the gpt_oss FP8 axis.

R56 default = **option 1 (noise re-characterization)**. Concrete and
finishes in <5 min wall.

## Files added

  * `analysis/_notes/round-55-fp8-direction-G-cross-shape-coopt-A-PRIORI-FALSIFIED-predicates-already-per-shape.md` (this file)

## NEUTRAL round

No code, dispatcher, or kernel changes. Daemon metric expected in the
691-699 noise band.
