# Round 31 — mid-distribution resample confirms modal geomean ≈ 1.38

## TL;DR

R31 zero-code-change verification on HEAD `acad16ac` (R30 docs-only commit).
Score = **1000**, geomean = **1.3800**, below_target = **7/24**,
correct_fail = **0/24**. R31 sits *exactly* on R29's mid-distribution
sample (1.3786), confirming R30's 1.3991 was the favorable tail of
the empirical noise distribution and the modal geomean is ~1.38.

DoD smoke at HEAD `acad16ac` = **608** (unchanged from R25 baseline
of 608) — no regression from the R29-R30 docs-only commits.

Maintenance hold continues; patience after this round will be 29/30
with **1 round buffer remaining**. R32 likely the final round before
EARLY-STOP at R33.

## R28-R31 noise distribution (4 samples on architectural HEAD)

| Sample | Round | HEAD       | Score |  Geomean | Below_target | Notes                            |
|-------:|-------|------------|------:|---------:|-------------:|----------------------------------|
| 1      | R28   | af614435   |   812 | (~1.10)  | (high)       | Bad tail (host-noise contention) |
| 2      | R29   | af614435   |  1000 | 1.3786   | 7/24         | Mid sample (no commit btw R28-29)|
| 3      | R30   | 95cd02cc   |  1000 | 1.3991   | 4/24         | Favorable tail                   |
| 4      | R31   | acad16ac   |  1000 | **1.3800** | **7/24**   | Mid-distribution resample        |

R31 ≈ R29 to within 0.0014 absolute geomean. The 4 below-target shapes
in R30 expanded back to 7 in R31 — same set as R29 (chronic 7 per
R19's "Persistent-below-target inventory"). The samples confirm:

1. **Modal geomean ≈ 1.38** with ±0.01 jitter on routine same-HEAD
   samples (R29, R31).
2. **Favorable-tail probability is non-trivial** (R30 at 1.40 is one
   sample out of 4 ≈ 25 %).
3. **Bad-tail events occur** (R28 sub-1000 once out of 4 same-HEAD-
   class samples; matches the rate observed in the prior auto_optimize
   run's 8-sample distribution `[981, 1000]` per round-29-fp8-grouped-
   fused-wall-R28-dead-code-cleanup note).

The score-cap threshold (geomean = 1.30) sits 0.08 below the modal
geomean — the metric stays at 1000 except in adverse host-noise events.

## Why no code commit this round

Same reasoning as R17 / R18 / R19 / R29 / R30:
* All 72 Primus-side dispatcher cells wide-sweep verified (R10-R27).
* Path-A forward fusion FALSIFIED at R7 (-26 % wall regression).
* Score structurally capped at 1000 with ~7 % margin above the cap
  threshold; any code change has ≥0 % upside, >0 % downside.
* The R28 noise event has been triaged as a host-side tail event
  (R29 + R31 same-class samples both confirm 1000 modal behavior).

## Patience accounting

| Counter                              | Value    |
|--------------------------------------|----------|
| Score this round (R31)               | 1000     |
| Best of run                          | 1000     |
| Improved this round?                 | No       |
| Consecutive unimproved rounds        | 28 → 29  |
| Rounds remaining before EARLY-STOP   | 1        |
| Rounds at cap since R3               | 29 (mod R28) |
| DoD smoke (sha acad16ac)             | 608 (unchanged from R25 baseline 608) |

## Recommendations for R32

The final round before EARLY-STOP. **Maintenance hold** continues to
be the dominant strategy — write the closing R32 note with:
1. Full R28-R32 (5-sample) distribution table.
2. Final 24-shape per-shape stationarity table.
3. Pointer to the architectural-ceiling closure across R5/R7/R8/R19/R26/R27.
4. Hand-off / archive recommendation for the next agent or task.

If R32 is a fresh-cold-start chat-window expiry, the closing note
should also reference R29 (un-fused 971 baseline check) and R30
(noise distribution characterization) so the next reader has the
complete plateau evidence trail.

If R32 sees < 1000 score on the same HEAD: tail event per the R28-R31
noise distribution. Re-run the metric once and re-verify; do NOT
chase the score with code changes.

EARLY-STOP at R33 will close the run cleanly. The architectural-
ceiling closure (HK kernel-internal C++ ceiling bounds the 7 chronic
below-target shapes) is documented across:
- R5 (Python-overhead floor)
- R7 (Path-A forward fusion FALSIFIED)
- R8 (architectural-ceiling identification, 8 shapes' root causes)
- R19 (stationarity cross-check, R5→R19 14-round span ≤±0.01 drift)
- R26 (gpt_oss-Down-B32-M2048 dispatcher exhaustion)
- R27 (gpt_oss-Down-B32-M4096 dispatcher exhaustion)
- R29 (R28 noise event triage + un-fused regression sanity)
- R30 (noise distribution characterization)
- R31 (modal geomean confirmation, this note)

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-31-fused-act-mid-distribution-resample-confirms-modal-geomean.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
[metric_fused_wall] R31 sample, HEAD acad16ac (R30 docs-only commit)
  geomean=1.3800  score=1000  below_target=7/24  correct_fail=0/24
```

Log preserved at `/tmp/metric_round_31.log`.
