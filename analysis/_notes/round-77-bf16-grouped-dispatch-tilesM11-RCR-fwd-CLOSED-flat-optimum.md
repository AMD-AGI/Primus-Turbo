# Round 77 — BF16 grouped dispatch (CRR var-K tiles_m=11 + RCR fwd tiles_m=8/tiles_n=11/k=2880/m_total≤8192) CLOSED — flat optimum

## Round / date / GPU / sha

- Round 77 (chat-resume of R76); same metric `_metric_grouped_bf16_weighted_wall.py`.
- 2026-05-04, HIP_VISIBLE_DEVICES=3 (MI355X / gfx950 / NUM_CUS=256 / 8 XCDs).
- Primus-Turbo: `bc91e8f7` → this closure note.
- HK: unchanged at `9a860d59` (no kernel touch).

## Why this lever (continuing R76 thread)

R76 closed FUSE-RCR sched_barrier extension. R76 next-round candidate
was a dispatch-rule sweep for the `tiles_m == 11` sub-family — the
CRR var-K rule at `config.py` line 1203 lumps `tiles_m ∈ [8, 24]`
(gpt_oss-Down + GateUP) into a single `(gm=4, xcds=4)`. R66 audited
tiles_m=22 (GateUP); tiles_m=11 (gpt_oss-Down) was untested under
the weighted-wall metric.

R77 sweeps **both** dispatch rules that fire on the lowest-progress
shape `gpt_oss-Down-B4-M2048` (`ratio=1.054 ± 0.005`, weight 3×):

1. **CRR var-K** (line 1203): fires on dB var-K for the 4 gpt_oss-Down
   shapes (B∈{4,32} × M∈{2048,4096}). Predicate `tiles_m=11 + tiles_n=11
   + k≤4096 → (gm=4, xcds=4)`.
2. **RCR fwd/dA** (line 478): fires on fwd RCR + dA RCR (post-H4 reroute)
   for `gpt_oss-Down-B4-M2048` only. Predicate `tiles_m=8 + tiles_n=11
   + k=2880 + m_total≤8192 → (gm=2, xcds=2)`. Was set in R10 against
   the **pre-R54** sched_barrier kernel; never re-validated under the
   post-R55 kernel.

Both rules are pure scheduling knobs (group_m / num_xcds reorder the
persistent tile schedule, not the arithmetic — bit-equivalence proven
by every previous rule comment).

## Sweep methodology

Wrapper scripts (`/tmp/r77_run_with_patch{,_rcr}.py`) monkey-patch
`hipkitten.config.select_default_config` to substitute `(gm, xcds)`
into the target rule **only**; all other rules untouched, so non-target
shapes don't drift. Each cfg single-run, then top candidates paired
(3-4 runs each).

## Lever 1: CRR var-K (`tiles_m=11`) — 12 cells

| cfg     | score | gpt_oss geo | D-B4-M2048 | D-B4-M4096 | D-B32-M2048 | D-B32-M4096 |
|---------|-------|-------------|------------|------------|-------------|-------------|
| ( 4, 4)←rule | 873 | 1.0764 | 1.053 | 1.107 | 1.052 | 1.087 |
| ( 1, 4) | 875  | 1.0777 | 1.053 | 1.103 | 1.066 | 1.082 |
| ( 2, 4) | **883** | 1.0929 | **1.095** | **1.136** | 1.058 | 1.082 |
| ( 8, 4) | 875  | 1.0774 | 1.055 | 1.109 | 1.059 | 1.077 |
| (16, 4) | 874  | 1.0743 | 1.052 | 1.099 | 1.054 | 1.086 |
| ( 1, 8) | 874  | 1.0767 | 1.051 | 1.110 | 1.061 | 1.078 |
| ( 2, 8) | 875  | 1.0768 | 1.052 | 1.110 | 1.060 | 1.085 |
| ( 4, 8) | 874  | 1.0766 | 1.045 | 1.109 | 1.059 | 1.087 |
| ( 1, 2) | 874  | 1.0769 | 1.053 | 1.109 | 1.065 | 1.082 |
| ( 2, 2) | 876  | 1.0787 | 1.060 | 1.108 | 1.056 | 1.087 |
| ( 4, 2) | **883** | 1.0931 | **1.108** | 1.105 | 1.053 | 1.083 |
| ( 1, 1) | 873  | 1.0746 | 1.051 | 1.096 | 1.055 | 1.088 |

Top-2 single-run candidates `(2, 4)` and `(4, 2)` both scored **883**
(+10 over baseline). Triple-paired verification:

| cfg     | R1 | R2 | R3 | mean | D-B4-M2048 mean |
|---------|----|----|----|------|-----------------|
| (4, 4)←rule | 874 | 875 | 875 | 874.7 | 1.053 |
| (2, 4) | 874 | 875 | 875 | 874.7 | 1.059 (+0.006) |
| (4, 2) | 876 | 874 | 875 | 875.0 | 1.056 (+0.003) |

**Both single-run +10 wins evaporated under paired sampling**
(mean Δ ≤ +0.3, ≪ ±2 noise band). Initial sweep run captured outlier-
high samples for both cfgs simultaneously — flat optimum confirmed.

## Lever 2: RCR fwd/dA (`tiles_m=8/tiles_n=11/k=2880/m_total≤8192`) — 17 cells

| cfg     | score | gpt_oss geo | D-B4-M2048 |
|---------|-------|-------------|------------|
| ( 2, 2)←rule | 877 | 1.0816 | 1.050 |
| (32, 8) | 879  | 1.0857 | **1.076** |
| ( 1, 4) | 879  | 1.0853 | **1.076** |
| ( 2, 4) | 874  | 1.0775 | 1.045 |
| ( 4, 4) | 874  | 1.0770 | 1.058 |
| ( 8, 4) | 876  | 1.0795 | 1.055 |
| (16, 4) | 874  | 1.0769 | 1.055 |
| (32, 4) | 875  | 1.0777 | 1.055 |
| ( 1, 2) | 873  | 1.0728 | 1.045 |
| ( 4, 2) | 874  | 1.0763 | 1.054 |
| ( 8, 2) | 875  | 1.0783 | 1.056 |
| (16, 2) | **881** | 1.0893 | **1.093** |
| ( 2, 8) | 874  | 1.0759 | 1.056 |
| ( 4, 8) | 875  | 1.0765 | 1.051 |
| ( 8, 8) | 876  | 1.0775 | 1.049 |
| ( 1, 1) | 875  | 1.0782 | 1.057 |
| ( 2, 1) | 873  | 1.0745 | 1.050 |

3 candidates above (2, 2) by ≥ +2 single-run: `(16, 2) +4`, `(32, 8)
+2`, `(1, 4) +2`. 4-paired verification:

| cfg     | R1 | R2 | R3 | R4 | mean | D-B4-M2048 mean |
|---------|----|----|----|----|------|-----------------|
| (2, 2)←rule | 874 | 875 | 875 | 875 | 874.75 | 1.055 |
| (16, 2) | 875 | 876 | 875 | 876 | 875.5  | 1.057 (+0.002) |
| (32, 8) | 875 | 874 | 875 | 873 | 874.25 | 1.054 (-0.001) |
| ( 1, 4) | 874 | 875 | 875 | 875 | 874.75 | 1.054 (-0.001) |

`(16, 2)` mean Δ = +0.75 — strictly positive but ≪ +5 commit
threshold. The +4 single-run signal was a sample-size-1 outlier;
4-paired Δ does not clear the noise band.

## Both rules confirmed at flat optimum

| Rule | sample size | best paired-mean Δ |
|---|---|---|
| CRR var-K (line 1203) | 12 + 3 cells × 3 paired | +0.3 |
| RCR fwd/dA (line 478) | 17 + 4 cells × 4 paired | +0.75 |

Both well within ±2 noise band. The dispatch surface for the
lowest-progress shape is now **fully audited** (R66 closed tiles_m=22;
R77 closes tiles_m=11 CRR + tiles_m=8/tiles_n=11/k=2880/m_total≤8192 RCR).

## Why is the lowest-progress shape at the dispatch ceiling

For `gpt_oss-Down-B4-M2048` (M_total=8192, N=2880, K=2880):
- **fwd persistent grid**: M_total/256 × ceil(N/256) = 32 × 12 = 384 tiles
  → tiles_per_CU = 1.5 (16 CUs do 2 tiles, 240 do 1 tile, 0 idle).
- **dB var-K persistent grid**: B × ceil(N/256) × ceil(K/256) = 4 × 12 × 12
  = 576 tiles → tiles_per_CU = 2.25 (64 CUs do 3, 192 do 2).

Both grids are **wave-imbalanced by structural shape, not by schedule
cell choice**. Reordering tiles via `(group_m, num_xcds)` only re-permutes
the same imbalanced work onto the same grid: any cfg that keeps all 256
CUs busy hits the same critical-path tile (the slowest CU finishes 2 or 3
tiles regardless of order). This is the **imbalanced-grid ceiling** also
identified in R47 (CRR per-tile arithmetic hoist falsified at low
tiles_per_group) and R75 (var-K work-stealing falsified — atomic
contention matched imbalance gain).

Triton at `1.0/1.054 = 0.949` for this shape uses its own different
persistent kernel; HK is already 5.4% faster (ratio=1.054). Pushing
to 1.25 needs **structural** levers — not dispatcher tuning.

## Surface map (post-R77)

**Closed (no headroom):** R10/R21/R26/R44/R45/R47/R54/R55/R57/R58/R60/
R63/R64/R65/R66/R67/R71/R72/R73/R74/R75/R76 + R77 (this round).

**Remaining lever inventory (kernel-level, all higher-risk than dispatch):**

1. **B2: FUSE pipeline prefetch** (R75 deferred) — overlap slab-1 A-HBM-load
   with slab-0 MMAs in FUSE-RCR. Risk: VGPR spill on the already-tight
   256-VGPR FUSE path. Need SASS-level prefetch order audit.
   Expected payoff +5-10 if it lands.
2. **Var-K LDS bank-conflict v2** — R68 PMC found 217M conflicts.
   R74 swap-shape FALSIFIED at 66 VGPR spill. Single-row bf16 pad
   (2 bytes/row) less invasive. Risk: numerical off-by-one (R74 pattern).
3. **Launch-overhead reduction for B=4** — `hipMemsetAsync(counter, 0, 4)`
   per launch ≈ 5 µs × 3 = 15 µs / step on 500 µs total wall (~3%).
   Lever: kernel-side counter zero with sequence-number protocol.

R78 candidate: **option (2) — single-row bf16 LDS pad**. Cheapest,
addresses R68 PMC bottleneck at finer granularity than R74's
shape-swap.

## Reverted state

No code changes. Working tree clean (modulo .nfs / build artefacts).

## Conclusion

**Two dispatch rules CLOSED** at flat optimum on the lowest-progress
shape — `gpt_oss-Down-B4-M2048` is now fully audited (12+17 = 29 cells
× paired-verified). Score remains pinned to the 874-878 noise band.
Remaining headroom requires kernel-level structural changes (LDS bank-
conflict fix, launch-overhead reduction, or FUSE pipeline prefetch).
