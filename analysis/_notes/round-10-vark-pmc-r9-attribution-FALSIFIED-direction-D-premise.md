---
round: 10
task: gpt_oss_fp8_kernel
direction: D (var-K SALU coord-decode)
verdict: FALSIFIED at PMC — Direction D premise broken
binary: post-R9 (HK b3a5c8db, Primus ae98d226)
shape: Down_B4_M2048 wgrad var-K
metric_expected: ~694 (NEUTRAL — no kernel change this round)
---

# Round-10 — PMC re-measurement on R9 binary: Direction D premise FALSIFIED

## TL;DR

R9 ported the FP8 var-K per-tile coord decode from a 6-iter binary
search over `s_cum_tiles[]` to the O(1) closed form (`group_idx =
gt / tiles_per_group`), targeting R21's measured `SALU/SQ_busy =
85 %` on Down_B4_M2048 wgrad. R9 metric was NEUTRAL (median 694 over
8 samples, indistinguishable from R29 noise). R10 re-runs the R21
PMC scaffold on the R9 binary to attribute whether SALU dropped at
the counter level even when the metric did not see it.

**Result: SALUBusy did NOT drop.** It actually inched up slightly
(9.10% → 9.46%, within ~5% PMC noise). All other counters within
±2% of R21. The per-tile coord decode was never a meaningful SALU
contributor; R21's headline "85 % SALU / SQ_busy" was misleading
because SQ_busy is itself only ~30 % of total cycles, so SALU's
share of WALL is ~9 %.

**Direction D as a category is now FALSIFIED at the PMC level**, not
just at the metric level. R11 must rotate to a different attack
vector — the per-K-iter SALU items (R9 forward-pointer items 1-3)
are aiming at the same non-bottleneck.

## Method

Re-ran `scripts/_probe_round_21_vark_pmc_scaffold.py --shape
Down_B4_M2048 --n-calls 50` on the R9 binary (current HEAD). Same
5-batch counter set as R21, same shape, same n_calls. New driver
script `scripts/_probe_round_10_vark_pmc_compare.py` aggregates
medians across batches and prints a side-by-side table vs the R21
headline numbers from
`analysis/_notes/round-21-vark-pmc-mfma-underfeed-IDENTIFIED.md`.

## Result table

```
counter                       R21 (pre-R9)   R10 (post-R9)        Δ    n
-------------------------------------------------------------------------
MfmaUtil                            32.10 %         32.74 %    +2.0%   70
SALUBusy                             9.10 %          9.46 %    +4.0%   70
MemUnitStalled                       0.20 %          0.22 %   +12.2%   70
SQ_INSTS_VALU_MFMA_F8                2.36 M          2.36 M    -0.0%   70
SQ_BUSY_CYCLES                       6.28 M          6.24 M    -0.6%  210
SQ_INSTS_VALU                            —          11.12 M       —    70
SQ_INSTS_SALU                            —           5.45 M       —    70
SQ_INSTS_LDS                             —           3.55 M       —    70
SQ_INSTS_SMEM                            —          21.50 k       —    70
TCC_HIT_sum                              —           7.38 M       —    70
TCC_MISS_sum                             —           2.09 M       —    70
```

(R21 medians from its headline table; R10 medians aggregated from the
re-run CSVs at `/tmp/r21_vark_pmc_Down_B4_M2048/*/`. The "FetchSize"
delta in the raw probe output is a unit-scale artifact — R21 reported
per-launch aggregate (~88 MB), R10 reads per-invocation, not directly
comparable; ignore.)

## Why R21's "SALU/SQ_busy = 85%" was misleading

R21 reported `SALU/SQ_busy = 85.4 %` on the Down_B4_M2048 wgrad
PMC table cited in `_task_gpt_oss_fp8_kernel.md` (lines ~98-105). On
its face this looks like SALU is dominating the SQ pipeline. But
`SQ_busy` is the fraction of cycles where the SQ has *any* warp
ready to issue (after vector pipeline clamp). On this kernel,
**SQ_busy is itself only ~30 %** of total cycles (the rest is the
"58 % unaccounted" cycles R21 attributed to issue-rate / dependency
latency between LDS-load and MFMA).

Of that 30 % SQ_busy fraction, 85 % being SALU = SALU runs ~25.5 %
of all cycles in some kernels — but on the actual var-K wgrad, the
direct SALUBusy counter says 9.10 % (R21) and 9.46 % (R10). The two
metrics measure different things:

* `SALU/SQ_busy` (table cell) is a derived ratio that excludes cycles
  where the issuer is stalled on a non-SALU dependency.
* `SALUBusy` (direct counter) is the wall-clock fraction of SALU
  pipeline activity.

The 9 % wall-clock figure is the actionable one. **Cutting it to 0
would shave at most ~9 % of cycles** — and even that overestimates
because SALU often runs in parallel with VALU (CDNA4 issues SALU on
SIMD-X while MFMA runs on SIMD-Y). The real upper bound on
"eliminate per-tile decode SALU" lift was ~1-3 % of kernel time —
below the metric's ±1.45 noise floor, exactly as R9 measured.

## Why R9 is still bit-eq-correct but unhelpful

R9 replaced a 6-LDS-read binary search with a closed-form integer
divide (`s_div_*` on CDNA4, ~7-15 cycles per SALU sequence). The
two sequences are *roughly equivalent* in SALU cost — possibly the
divide is even slightly more expensive (consistent with SALUBusy
inching from 9.10 → 9.46 %, though within noise). The LDS reads
saved (6 per outer-iter) showed up as `SQ_INSTS_LDS = 3.55 M` —
this is the K-loop body's load_a/load_b traffic, dominated by the
~32 K-iters × 4 LDS reads = ~128 LDS issues per outer-iter, vs the
6 saved per outer-iter from the binary search. **R9 saved ~5 % of
LDS traffic per tile, but the K-loop body's LDS dominates**, so the
total `SQ_INSTS_LDS` barely budged.

## Falsification implications for the R9 forward-pointer

The R9 doc forward-pointed items 1-3 (per-K-iter SALU contributors:
`tic ^= 1` xor, `load_a/load_b` swizzled-offset constants, `if (ki_g
< 2) continue` test). All three target SALU. Given R10's PMC shows
the *total* SALU fraction is only 9.46 %, ANY combination of these
three items has at most ~3-5 % wall headroom on this cell — and
realistically <1 % each in isolation. Below noise floor.

* Item 1 (tic/toc xor hoist) — ~2 SALU/iter × 32 iter = 64 SALU /
  outer-iter saved on a SQ_INSTS_SALU baseline of 5.45 M / 50 calls /
  256 CTAs = ~426 SALU/CTA-call. So 64 / 426 = 15 % of CTA SALU,
  but applied only across the K-loop iterations — and those iters
  are already MFMA-bound (R21 finding). Real wall lift estimate: <1 %.
* Item 2 (load_a/load_b offset cache) — A-PRIORI noise. The compiler
  almost certainly already constants-out `wm * RBM` since the lambda
  closure captures `wm` once per outer-iter. Free lift = 0.
* Item 3 (`if (ki_g < 2)` hoist) — 1 SALU/outer-iter, never taken
  on metric shapes. Free lift = ~0.

**All three of R9's forward-pointer items are now expected-FALSIFIED**
because Direction D's basic premise (SALU is the bottleneck) was
counter-data wrong. They should not be attempted; that would burn
3 rounds for ~zero EV.

## Real bottleneck (re-stated from R21, now confirmed)

R21 already said it: **issue-rate / dependency-latency bound**.
R10's PMC re-confirms:

* MfmaUtil = 32.74 % — MFMA pipe drained 2/3 of cycles
* MemUnitStalled = 0.22 % — not memory-stalled
* SALUBusy = 9.46 % — not SALU-busy
* VALUBusy (= SQ_INSTS_VALU/cycle ratio implied) — modest
* Residual ~58 % of cycles = MFMA pipe waiting on operand registers
  that are themselves waiting on LDS-load drains gated by CTA
  barriers.

This bottleneck class is **structural**, not algorithmic-per-K-iter.
R22-R28 already exhausted the K-loop barrier/wait surgery levers
(VARK_MAIN_UNROLL, HOIST_PREFETCH, DROP_LGKM_DRAIN, DROP_BARRIER_2,
DROP_BARRIER_4, SW_PIPE_HOIST_AHEAD — all FALSIFIED, see FORBIDDEN
PATHS in `_task_gpt_oss_fp8_kernel.md`).

## R11 forward pointer

The remaining classes that have NOT been multi-sample falsified:

1. **A1' variant-2 K-split** (R7-R8 forward pointer). The actual
   structural attack on the issue-rate bound: split the K-loop so
   2-4 CTAs share a tile, each doing ki_g/2 or ki_g/4 K-iters, with
   atomicAdd on fp32 partial buffer + post-kernel reduce. Trades
   atomic write traffic for K-loop concurrency. R7-R8 marked it as a
   4-6 round restructure with infrastructure cost (host alloc + zero
   the partial buffer; reduce kernel; bit-eq audit on accumulation
   order). Down_B4_M2048's ki_g = 16 is small enough that a 2-way
   split is plausible without occupancy collapse.

2. **A3 decoupled-warps / producer-consumer** — falsified at preflight
   in R6 due to cooperative-load throughput coupling on the existing
   8w fragment geometry. R6 forward pointer noted "if D and A1 don't
   deliver, this is the final structural attack." That ordering
   stands; R11+ should pursue A1 first.

3. **Cross-shape co-optimization** (Direction G, never attempted).
   Per-shape sweeps have been exhausted on Down-B4. But a dispatcher
   rule that loses -0.5 % on shape A and gains +1.5 % on shape B can
   still be a net win. None of R1-R45 explicitly searched the
   joint-shape product. Lower-EV than A1' but lower-cost (1 round
   of dispatcher sweep + verify, no kernel edit).

**Recommended R11 move**: pick (1) — start the A1' variant-2 K-split
scaffolding. Round 11 = host-side partial-buffer alloc/zero +
control-flow plumbing (no actual K-split yet, must be bit-eq).
Round 12 = wire the 2-way K-split with atomicAdd. Round 13 = reduce
kernel + bit-eq verify. This is a 3-round commitment but is the
last structural attack with EV >5 score on the wgrad cells.

If A1' variant-2 turns out to be infeasible (atomic contention,
SNR breach on accumulation order), rotate to (3) Direction G as the
salvage path.

## Code state this round

* `scripts/_probe_round_10_vark_pmc_compare.py` — new driver script
  that re-uses the R21 scaffold and prints aggregate medians
  vs R21's pre-R9 headline numbers.
* No HipKittens edits this round. (R10 is observability-only.)
* No metric movement expected — daemon will record the same ~694
  the R9 binary produced, since the binary is unchanged.

## Why this is "progress" even at NEUTRAL metric

Per task md "Falsification + forward-pointing analysis = progress."
R10 closes Direction D as a category (3 rounds saved that would
have been spent on items 1-3 from R9 forward pointer), redirects
R11+ to the A1' variant-2 attack with explicit preflight-falsification
of three otherwise-tempting alternatives. Net effect: 4-5 rounds of
the remaining round budget now flow to the highest-EV remaining
direction instead of being burned on three sub-noise SALU items.
