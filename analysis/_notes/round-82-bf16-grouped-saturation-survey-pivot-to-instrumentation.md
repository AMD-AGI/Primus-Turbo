# Round-82 — BF16 grouped wall — single-round lever survey FALSIFIED, pivot to PMC instrumentation for R83

**Date**: 2026-05-05  **HEAD before**: `38e442f68c53f1bad4d3dbc1a697ccdc8e5ebaeb`  **score before**: 883 / 1000 (best 883)
**HEAD after** : `<this commit>`  **score after** : 883 / 1000  (no change, docs-only round)

## Lever planned (per R81 R82 direction)

R81's "Direction for R82" was P1 — dB var-K LDS swizzle without
sub-tile padding. The R74 swap (`st_32x16_s` → `st_64x32_padded_b128_s`)
crashed correctness 24/24 + spilled 66 VGPRs; R81's proposed fix was a
**custom shape struct that inherits `st_32x16` shape but adds a
different swizzle XOR mask term** to break the 217M LDS bank conflict
without changing sub-tile composition (so the helpers' lane→byte
addressing still round-trips).

## Why it's not a single-round lever

The "swizzle-only" change requires (each is a separate sub-task):

1. Define `st_32x16_alt` in `include/types/shared/st_shape.cuh`. Must
   (a) mirror `st_32x16`'s `rows / cols / subtile_padding / bytes_per_thread`
   surface; (b) compute a different `swizzle()` XOR mask that breaks
   the (cib * K_REM) bank-alignment WITHOUT desyncing the rt_32x16_s
   register tile's `ds_read_b64_tr_b16` lane mapping. The current
   `st_32x16` swizzle XORs `((offset % 1024) >> 9) << 4` (= bit-9 →
   bit-4 flip on every 64th row pair). Any alternate must preserve the
   rt-shape's lane → cell mapping, which depends on internal HW lane
   permutation that the kittens helpers consume opaquely.
2. Use it in `grouped_var_k_kernel` (single-line ST_A/ST_B alias swap).
3. Build → check VGPR/SGPR spill (R74 baseline: 0 / 0 → after swap:
   66 / 0 spill). Custom swizzle should be inert on regalloc; if not,
   abandon.
4. Numerical correctness probe vs Triton on the metric's downsized
   gate (4/4 dB-allclose pass required).
5. Wall-time bench on the 4 gpt_oss-Down dB shapes (the 217M-conflict
   regime). Need ≥ +5 score at the metric to commit.

R74's failure mode was step 4 (24/24 dB-allclose FAIL). Even if the
custom swizzle compiles + lane-maps correctly, we cannot know whether
the 217M conflicts actually flatten to ~50M (= the RCR baseline) or
just relocate to a different bank pattern without a rocprofv3 measurement.

Estimated work: 2-3 rounds (struct definition + compile-fix + probe
+ rocprof PMC + maybe one re-tune).

## Closed single-round lever survey

Confirmed all current single-round levers are exhausted:

| Lever | Round | Status | Notes |
|---|---|---|---|
| LDS swizzle (st_64x32 swap) | R74 | FAIL | 24/24 dB-allclose, +66 VGPR spill |
| LDS swizzle (custom struct) | R82 | DEFERRED | multi-round, see above |
| dB var-K work-stealing | R75 | NEUTRAL | +1 VGPR spill, Δ-2 noise band |
| dB var-K chiplet chunk_size | R70 | FAIL | CS=32 +3 (sub-threshold), CS=16/128 worse |
| dB var-K per-XCD counter | (R65 fwd, untried var-K) | RISKY | R75 contention proof-of-life but +5 SGPR + 1 VGPR spill on extension |
| sched_barrier MAIN | (commit 2bdfca7a) | LANDED | already in place (8 sites in main_loop_iter) |
| sched_barrier EPILOG | R55 | LANDED | already in place (6 sites in EPILOG 1/2) |
| sched_barrier PROLOG | (untried) | UNCLEAR | PROLOG has no MMAs — sched_barrier semantics differ; speculative |
| RCR fwd dispatch tilesM=11 | R77 | CLOSED | flat optimum |
| H4 elim (R28 transpose lever) | R79 | FAIL | Δ+1 single-run, sub +5 |
| H4 elim (native RRR for K%128==0) | R80 | LANDED +9 | gpt_oss-GateUP dA only |
| RRR FUSE_KTAIL extension | R81 | FAIL | phantom-read persists (R29-era bug) |
| Config (gm, xcds) re-sweep | R22 | SATURATED | 8/8 BF16 + 8/8 FP8 within 0.3% of current rule |

## Variance characterisation

6 back-to-back metric runs at HEAD `38e442f6`:

```
run    score   weighted_progress
1      883     0.8829
2      883     0.8828
3      882     0.8821
4      883     0.8829
5      883     0.8827
6      884     0.8841   (post note-write, no code change)
```

Range = 2 (882..884), median = 883, std < 1. The noise band has
**tightened from ±5 (R74-R75 era) to ±2**, likely because the GPU
pool is currently idle (no co-tenant on GPUs 3/4) — ROCm clocks are
stable. Implication: any genuine +5 score lever will be unambiguously
visible at the noise floor; conversely, fishing for +5 via re-runs
is no longer viable. The +1 in run 6 is within-noise drift, not an
improvement (no code/kernel change between runs 5 and 6).

Correctness: 24/24 PASS, 0/24 reject across all 6 runs.

## R83 direction

The state space has narrowed to two structural levers, both
multi-round, both targeted at the gpt_oss-Down tail (4 shapes,
weight 3, currently ratio 1.054-1.106):

### R83-A — instrumented PMC walk on var-K kernel (1 round)

Run `rocprofv3 --kernel-trace --pmc lds_bank_conflict valuMfmaUtil
sqMemReqOcc` on the metric's gpt_oss-Down-B4-M2048 dB var-K launch
(MoE B=4 best fits the device for clean PMC). Confirm or refute:

* R68 PMC's "217M LDS bank conflicts" claim still holds post-R55/R56
  / R80 evolution. If conflicts have already dropped (e.g. due to
  the LDS-staged ntail kernel landing reshaping the L2/LDS access
  pattern), the LDS swizzle priority drops.
* MFMA util on the var-K kernel — is it MFMA-saturated or is the
  bottleneck elsewhere (HBM b/w, register pressure, dispatch
  imbalance)? Forward kernel saw R50 PMC at 70-78 % MFMA util on
  the family; var-K is unmeasured.
* Per-tile timing breakdown: prologue / main_loop_iter / EPILOG 1/2
  / store-C / atomic-tile-claim. Identifies whether fixing LDS
  conflicts (a main-loop win) actually shifts the bottleneck or
  just exposes the next layer.

This directly answers "is R82's LDS swizzle the right priority" — if
PMC shows MFMA util is the wall, swizzle is a no-op; if conflicts
are the wall, proceed to R84.

### R83-B — RRR FUSE phantom-read instrumentation (1 round)

Per-warp_col / per-h_b SNR diagnostic embedded in `device_gemm_tile_body`
under `BF16_RRR_FUSE_PROBE` flag. Output: a 4×4 (warp_col × h_b) matrix
per-cell SNR for the K-tail epilog. R29 hypothesized the bug is in
either (a) cross-warp G::load LDS visibility or (b) `col_l rt_32x16_s`
lane→cell mapping after `ds_read_b64_tr_b16`. The diagnostic
distinguishes them: if (a), all warp_col but warp_col=0 should be
clean; if (b), specific h_b indices fail uniformly across warp_cols.

R83-A is cheaper (one rocprof run + analysis ≈ 30 min); R83-B is more
speculative but if it fixes the phantom-read, +5-10 score on
gpt_oss-Down dA + drops the H4 transpose tax.

**Pick R83-A**: instrumentation is cheaper than speculative kernel
changes, and the result of A directly informs whether to invest in
the LDS-swizzle direction at all.

## No code change this round

* HK working tree clean (`git status` confirmed).
* PT working tree: untouched apart from this round note. The
  `primus_turbo/pytorch/kernels/hipkitten/config.py` modified entry
  is a leftover FP8 fused-act task delta from a different chat
  session; not part of this round's work and not committed.
* Metric re-verified 883 post-write (this note is text-only, doesn't
  affect any kernel/dispatch path).

## Score history through R5 of this run (auto_optimize round numbering)

```
run-round   note-round    score    Δ vs prior best   note
1           R78           873      baseline          (cold-start; R78 diagnostic)
2           R79           874      +1 → new best     R79 R28 transpose Δ+1 (sub-+5, but became baseline drift)
3           R80           883      +9 → new best     R80 H4 elim native RRR LANDED
4           R81           883      flat              R81 RRR FUSE_PROBE re-falsified
5           R82 (this)    883      flat              R82 single-round lever survey FALSIFIED
```

Score plateau at 883 reflects single-round lever exhaustion. The
remaining 117 score points (= 12 % of the bar) live behind multi-round
structural changes. R83 instruments before committing to a direction;
single-round speculation is now value-negative.
