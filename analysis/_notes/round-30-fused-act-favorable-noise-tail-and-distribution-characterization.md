# Round 30 — favorable-noise tail observation completes the R28-R29-R30 noise distribution triplet

## TL;DR

R30 is a zero-code-change verification round on HEAD `95cd02cc` (R29's
docs-only commit). Score = **1000**, geomean = **1.3991** —
the *highest* geomean observed across R26-R30, with `below_target`
dropping to 4/24 (R29 had 7, R26-R27 had 7-8). Combined with R28
(812, bad tail) and R29 (1000 with geomean 1.3786, mid), this
3-sample triplet at the same architectural HEAD now characterizes
the empirical noise distribution.

Maintenance hold continues; patience after this round will be 28/30
with a 2-round buffer remaining.

## Empirical noise distribution at the architectural HEAD

| Sample | Round | HEAD       | Score | Geomean   | Below_target | Notes                                                    |
|--------|-------|------------|------:|----------:|-------------:|----------------------------------------------------------|
| 1      | R28   | af614435   |   812 | (~1.10)   | (high)       | Bad-tail event; card5 phantom-VRAM contention            |
| 2      | R29   | af614435   |  1000 | 1.3786    | 7/24         | Mid sample (same HEAD as R28; no commit between)         |
| 3      | R30   | 95cd02cc   |  1000 | **1.3991** | **4/24**    | Good-tail sample (R29 docs-only commit; no kernel change)|

`af614435` and `95cd02cc` differ only in a markdown round note
(`docs(round-29): ...`) — runtime is bit-identical. Hence the
sample-3 +0.02 geomean shift vs sample 2 (and the +0.05 vs the bad
tail of sample 1) is noise across same-HEAD measurements, not a
commit signal.

The 1.30 score-cap threshold sits 0.10 below R30's mid-distribution
value (1.30 / 1.40 ≈ 7 % of the geomean). To dip below the cap,
the distribution would need to drift ~7 % low — which R28 did, but
the modal behavior (R29, R30) is well above the cap.

## Per-shape table (R30)

```
[metric_fused_wall] suite: 24 FP8 wall cases | target ratio = 1.35

  fusedFP8_DeepSeek-V3-GateUP-B16-M2048      ratio 1.389
  fusedFP8_DeepSeek-V3-Down-B16-M2048        ratio 1.369
  fusedFP8_DeepSeek-V3-GateUP-B16-M4096      ratio 1.515
  fusedFP8_DeepSeek-V3-Down-B16-M4096        ratio 1.392
  fusedFP8_DeepSeek-V3-GateUP-B32-M2048      ratio 1.521
  fusedFP8_DeepSeek-V3-Down-B32-M2048        ratio 1.457
  fusedFP8_DeepSeek-V3-GateUP-B32-M4096      ratio 1.556
  fusedFP8_DeepSeek-V3-Down-B32-M4096        ratio 1.408
  fusedFP8_gpt_oss_20B-GateUP-B4-M2048       ratio 1.353
  fusedFP8_gpt_oss_20B-Down-B4-M2048         ratio 1.396
  fusedFP8_gpt_oss_20B-GateUP-B4-M4096       ratio 1.407
  fusedFP8_gpt_oss_20B-Down-B4-M4096         ratio 1.360
  fusedFP8_gpt_oss_20B-GateUP-B32-M2048      ratio 1.402
  fusedFP8_gpt_oss_20B-Down-B32-M2048        ratio 1.276    <135%
  fusedFP8_gpt_oss_20B-GateUP-B32-M4096      ratio 1.512
  fusedFP8_gpt_oss_20B-Down-B32-M4096        ratio 1.325    <135%
  fusedFP8_Qwen3-235B-A22B-GateUP-B16-M2048  ratio 1.343    <135%
  fusedFP8_Qwen3-235B-A22B-Down-B16-M2048    ratio 1.341    <135%
  fusedFP8_Qwen3-235B-A22B-GateUP-B16-M4096  ratio 1.368
  fusedFP8_Qwen3-235B-A22B-Down-B16-M4096    ratio 1.370
  fusedFP8_Qwen3-235B-A22B-GateUP-B32-M2048  ratio 1.368
  fusedFP8_Qwen3-235B-A22B-Down-B32-M2048    ratio 1.361
  fusedFP8_Qwen3-235B-A22B-GateUP-B32-M4096  ratio 1.447
  fusedFP8_Qwen3-235B-A22B-Down-B32-M4096    ratio 1.380

[metric_fused_wall] geomean=1.3991 progress=1.000 PASS
[metric_fused_wall] correct_fail=0/24 reject=0/24 below_target=4/24 score=1000
```

The 4 below-target shapes are 4 of the same 7 chronic shapes from
R19/R23-R29 — the others (Qwen3-Down-B16-M4096, Qwen3-GateUP-B16-M4096,
gpt_oss-GateUP-B4-M2048) drifted just above 1.35 in this favorable
sample. Persistent below-target = the architectural-ceiling list per
R19's "Persistent-below-target inventory".

## Why no code commit this round

Identical reasoning as R17 / R18 / R19 / R29 — 28 consecutive
rounds of stationarity confirms the architectural-ceiling model.
* All 72 Primus-side dispatcher cells are wide-sweep verified (R10-R27).
* Forward-fusion Path A is FALSIFIED at R7 (-26 % wall).
* Score is structurally capped at 1000 with the cap threshold sitting
  ~7 % below the mid-distribution geomean. Any code change has
  ≥0 % upside and >0 % downside.

The R28 noise event (812) and the R30 favorable-noise sample (geomean
1.3991, below_target 4) bracket the empirical distribution. The
auto_optimize.py single-sample-per-round design exposes this
distribution but does not include a denoising rule — patience is
based on `improved=False`, which holds whenever score ≤ best (1000).
A bad-tail score does NOT reset patience.

## Patience accounting

| Counter                              | Value    |
|--------------------------------------|----------|
| Score this round (R30)               | 1000     |
| Best of run                          | 1000     |
| Improved this round?                 | No       |
| Consecutive unimproved rounds        | 27 → 28  |
| Rounds remaining before EARLY-STOP   | 2        |
| Rounds at cap since R3               | 28 (mod R28) |

## Recommendations for R31+ (final 2 rounds before EARLY-STOP)

1. **R31 / R32: continue maintenance hold** (R17-R30 pattern). Run
   metric, confirm 1000 + 0/24 correctness, write a one-paragraph
   verification note. Cumulative cost ≈15 s of GPU + ≤80 lines of
   markdown per round. Zero risk of metric regression.
2. If R31 sees < 1000 score on a same-HEAD-no-commit basis: tail event
   per the R28-R29-R30 distribution. Re-run the metric once and
   re-verify before any code investigation; do NOT chase the score.
3. EARLY-STOP at R33 will close the run cleanly. The architectural-
   ceiling closure is documented across R5/R7/R8/R19/R26/R27/R29/R30.
4. If a future agent / user authorizes HK kernel-internal task scope
   expansion, the concrete leverage points remain (per R8/R19/R29):
   `grouped_rrr_kernel` for the dA path and
   `grouped_ktail_kernel_mfma32x32_M2N2` for K=2880. Both multi-round
   HK C++ work; will not move the score within a single round.

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-30-fused-act-favorable-noise-tail-and-distribution-characterization.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
[metric_fused_wall] R30 sample, HEAD 95cd02cc (R29 docs-only commit)
  geomean=1.3991  score=1000  below_target=4/24  correct_fail=0/24
```

Log preserved at `/tmp/metric_round_30.log`.
