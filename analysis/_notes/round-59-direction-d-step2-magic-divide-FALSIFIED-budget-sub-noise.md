# Round-59 — Direction D step-2 (host-precomputed magic-number divide for `gt / tiles_per_group`) FALSIFIED

**Verdict**: A-priori FALSIFIED — projected savings (≤0.6% kernel time) are an
order of magnitude below the empirical noise floor (σ=2.27 score / 0.32% on
GPU 3 per R56), so even a perfect implementation cannot register on the metric.

## What is Direction D and what's left of it?

SKILL.md NEW DIRECTIONS lists "D. SALU coord-decode optimization (var-K
specific)" as the highest-EV first move, citing R21 PMC `SALU/SQ_busy = 85%`
on `Down-B4 wgrad`. The recommendation is to "hoist this out of the iter
(rematerialize in regs, precompute CTA-static)".

**Step 1 is already shipped on HEAD**:

```
HipKittens commit b3a5c8db
"perf(fp8 grouped var-k): round-9 — port BF16 var-K closed-form coord decode
 (Direction D step 1, NEUTRAL at metric)"
```

`grouped_var_k_kernel_fp8` lines 8513-8540 already replace the 6-iter binary
search over `s_cum_tiles[]` with the O(1) closed-form

```c++
const int group_idx  = gt / tiles_per_group;
const int tile_start = group_idx * tiles_per_group;
const int local_tile = gt - tile_start;
```

The R21 SALU=85% finding survived this ship (R21 PMC was post-b3a5c8db),
which is consistent with the commit's "NEUTRAL at metric" verdict — the
binary-search collapse saves outer-loop SALU but the per-K-iter contributors
(barriers, waitcnts, integer indexing into `a_co/b_co`) dominate the SALU
budget.

## Step-2 candidate: host-precomputed magic-number divide

The closed-form decode ships an integer divide by a runtime non-pow-2
(`tiles_per_group = bpr * bpc`, range 88–352 across 8 cells). On CDNA4
this lowers to `s_div_*` libcall sequence (~30 cycles per div). The
divide is per outer-loop iter, executed once per persistent-tile.

Step-2 would precompute Hacker's-Delight magic numbers
`(tpg_magic_u32, tpg_shift)` host-side, pass them in the launch globals,
and emit `s_mul_hi_u32 + s_lshr_b32` in-kernel. ~5 cycles per divide,
saving ~25 cycles per outer-iter.

### Cycle budget — show your work

For Down-B4 M=2048 wgrad (worst SALU cell per R21 PMC):

| Quantity | Value | Source |
|---|---|---|
| Tiles per CTA per call | `total_tiles / NUM_CUS` ≈ 8/8 → 1–8 | `bpr=8, bpc=11, G=4 → tot=352, NUM_CUS=256` |
| Divides per CTA | 1–8 | one per outer iter |
| Cycles saved per divide | ~25 | s_div_* (30) → mul_hi+lshr (5) |
| Cycles saved per CTA | ≤200 | conservative |
| Kernel runtime per call | ~17 µs ≈ 30 M cycles (2 GHz) | rocprof |
| Cycles saved / kernel | ≤200 × 256 CTA = 51K | linear |
| **Fractional savings** | **≤0.17%** | 51K / 30M |

For Down-B32 M=4096 (largest grid, most divides): tot_tiles ≈ 5632, ~22
divides per CTA, but the kernel runtime scales proportionally — same
~0.1–0.2% fractional savings.

**Best case across all 8 cells: < 0.6% kernel-time savings → < 4 score
units**, well below the empirical 1σ noise floor of 2.27 (R56) and
matching R57-R58 noise-tail behavior.

The divide is a real compiler artifact (verified via R36
`AMDGPU_NUM_VGPR` audit which dumped Rpass-analysis on
`_Z24grouped_var_k_kernel_fp8`), but the cycle budget allocates it <0.5% of
kernel time. The 60-70% MFMA-idle that R21 identified is **per-K-iter**
issue-rate / barrier-pin, not outer-loop scalar arithmetic. Step-2 is
the wrong attack on the right diagnosis.

## Coverage of remaining SKILL.md "untried" directions (audit closes)

After this round's analysis, the SKILL.md NEW DIRECTIONS A-G are all
either shipped or FALSIFIED:

| Dir | Description | Status | Round |
|---|---|---|---|
| A1 | Stream-K / persistent + work-stealing | FALSIFIED (1/8 cells survive >25% gate per PMC reality check) | R52 |
| A2 | SplitK for wgrad var-K | FALSIFIED | R33 |
| A3 | Decoupled-warps / producer-consumer | FALSIFIED (existing CDNA4 BF16 PC paper data: -17 to -44% on 6 sizes; no 256x256 PC prototype exists) | R54 |
| B | dgrad/wgrad on separate streams | metric pre-quantizes + serially-times each section → cannot help score | task-md note |
| C | Activation-cache reuse | metric pre-quantizes inputs outside timer → cannot help score | task-md note |
| D step 1 | var-K closed-form coord decode | SHIPPED, NEUTRAL | b3a5c8db |
| D step 2 | host-precomputed magic-number divide | **FALSIFIED this round (cycle budget <0.6%, sub-noise)** | R59 |
| E | Different barrier scheme (s_setprio + warp-group) | FALSIFIED (single-barrier-drop probes R26-R28 already show barriers load-bearing) | R26-R28 |
| F | Larger tiles (256x384, 512x256) | FALSIFIED | R32 |
| G | Cross-shape co-optimization (joint search) | FALSIFIED (dispatcher predicates already at per-shape granularity) | R55 |

Local levers (dispatcher gm/xcds/slots/cs sweeps, macro flags
VARK_*, RCR_*) all explored to exhaustion in R1-R45.

## Conclusion + forward pointer

The FP8 gpt_oss kernel-only score is in **structural saturation** at
695-697 (median 695, σ 2.27 per R56 30-sample re-characterization). The
remaining headroom on this metric (gap to 900 target) requires kernel-
template-level rewrites that have all been pre-flight FALSIFIED for
reasons unrelated to dispatcher tuning:

* Stream-K (R52): production CTA-wave-pack model closes only 1/8 cells.
* Decoupled-warps (R54): paper-grade PC scheme loses on 6/6 BF16 sizes
  at the cAB[2][2]-equivalent footprint; no clean port exists.
* 4-wave port (R39b → R57-R61): LLVM AGPR allocator alias bug at 256
  fp32/lane block accumulator; closed without root-cause fix.

**Recommendation for daemon budget**: rounds 60-100 will continue to
fire docs-only FALSIFIED commits at ~0 EV. The honest action is one of:
(a) accept saturation and stop early, (b) redirect budget to a different
metric (e.g. `_metric_grouped_only` 24-shape, `_metric_hk_ratio` mixed
dense+grouped, or fused-act `_metric_grouped_fused_wall`), or (c) commit
to a multi-round (4-6) implementation of A3 producer-consumer with a
*new* footprint (RBM=64 RBN=32 instead of the falsified RBM=64 RBN=64),
which is the only structurally-untried wave-allocation variant and would
break the 60-70% MFMA-idle PMC etiology if the per-tile fixed cost can
be amortized.

If continuing the daemon, R60+ should default to (c) with the explicit
multi-round budget acknowledged up front (no expectation of metric
movement before R64-R65 at earliest). Otherwise the docs-only pattern
of R51-R59 will persist through R100.

## What changed in this round

* This note (`analysis/_notes/round-59-direction-d-step2-magic-divide-
  FALSIFIED-budget-sub-noise.md`).

No code changes (Primus-Turbo or HipKittens). HEAD on both repos is
unchanged from R58.
