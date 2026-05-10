# R45 — GateUP-B4-M2048 fwd RCR (gm, xcds) drift re-audit FALSIFIED

## Forward pointer source

R44 (probe scripts/_probe_round_44_down_b4_m2048_drift.py, evidence
documented in analysis/_notes/round-44-down-b4-m2048-gm-xcd-drift-FALSIFIED-rule-at-local-optimum.md;
the daemon round-44 commit d6ca891 was a no-op-by-design placeholder
that absorbed this falsification) closed the (gm, xcds) drift re-audit on
**Down-B4-M2048** RCR rule (config.py:2335-2342) as FALSIFIED — top
three cells {(16,2), (24,2), (32,2)} statistically indistinguishable
(median gap ≤ 0.14 pp vs per-seed spread 0.09-0.59%). R44 forward-
pointed to the next analogous candidate:

> "primary: GateUP-B4-M2048 fwd RCR rule (gm=1, xcd=4) at config.py:1418-1482,
>  also tuned multiple rebuilds ago. Probe: clone scripts/_probe_round_44...,
>  swap to (B=4, M=2048, N=5760, K=2880), sweep {(1,2), (1,4), (1,8), (3,4),
>  (4,4), (1,1)} holding slots=0, cs=0 fixed (R23 rule has no slots/cs)."
>
> "BACKUP: closure documentation (R43 option 2) — if GateUP also returns
>  NEUTRAL, R44+R45 establish two consecutive empirical FALSIFICATIONS,
>  justifying a 34-round (R10-R44) audit summary marking the task feature-
>  complete at the structural ~700 plateau and recommending daemon transition."

R45 executes the primary plan.

## Methodology

- Shape: GateUP B=4 M=2048 N=5760 K=2880 (gpt_oss family fwd RCR cell).
- Rule under test: config.py:1418-1482, currently
  `HipKittenConfig(group_m=1, num_xcds=4, kernel=None)` — slots and cs
  defaulted to 0.
- Cells (gm, xcd): {(1,4)\*, (1,2), (1,8), (3,4), (4,4), (1,1), (2,4)}
  (R23's original sweep + drift-flip candidates).
- 5 seeds × 2000-iter p20 each cell, kernel-only timing via direct
  `grouped_rcr_dscale` call with monkey-patched (gm, xcd, slots=0, cs=0).
- Probe at `scripts/_probe_round_45_gateup_b4_m2048_drift.py`.
- Run: `bash scripts/dbg_remote.sh 'python3 scripts/_probe_round_45_gateup_b4_m2048_drift.py'`
  on remote MI355X HIP_VISIBLE_DEVICES=3 (idle GPU pinned by daemon).
- SHIP gate: best != baseline AND lift ≥ 1.0% AND signal > spread on the fwd
  section. (dgrad-via-H4 column probed for completeness but is not a valid
  drift test of the GateUP fwd rule — see §"Instrumentation artifact" below.)

## Result

```
=== GateUP-B4-M2048 fwd (B=4, M=2048, N=5760, K=2880) ===
       cell    med ms    min ms    max ms  spread%    TFLOPS   delta%
     (1, 4)    0.1432    0.1430    0.1434    0.34%    1897.4   +0.00% *base
     (1, 2)    0.1446    0.1444    0.1447    0.22%    1879.1   -0.98%
     (1, 8)    0.1517    0.1516    0.1522    0.37%    1791.4   -5.92%
     (3, 4)    0.1460    0.1459    0.1460    0.11%    1862.1   -1.90%
     (4, 4)    0.1442    0.1441    0.1443    0.17%    1885.3   -0.64%
     (1, 1)    0.1519    0.1518    0.1522    0.32%    1789.5   -6.03%
     (2, 4)    0.1434    0.1434    0.1437    0.25%    1895.8   -0.08%
  BEST: cell=(1, 4)  (+0.00% over baseline (1, 4))
```

**FALSIFIED** — baseline (1, 4) is the unique top of the swept set.

- (2, 4) is the only competing cell within noise: -0.08% with 0.25% spread,
  i.e. statistically indistinguishable from baseline. Cannot dethrone (1, 4)
  on signal-vs-spread grounds.
- (4, 4) at -0.64% is also within the 0.34% × ~2 noise band but consistently
  slower across all 5 seeds.
- (1, 2) at -0.98%, (3, 4) at -1.90% are real losses (>2× spread).
- (1, 8) at -5.92% and (1, 1) at -6.03% are catastrophic — confirms xcd=4
  is the right chiplet partition and xcd ∈ {1, 8} are way off.

The R23 rule (gm=1, xcd=4) tuned ~22 rebuilds ago **remains at local
optimum on the current FP8 binding**. No drift detected; consistent with
R44's Down-B4-M2048 finding.

## Instrumentation artifact: dgrad-via-H4 column is uninterpretable

The probe also ran the same (gm, xcd, slots=0, cs=0) sweep through
the dgrad-via-H4 path:

```
=== GateUP-B4-M2048 dgrad-via-H4 (B=4, M=2048, N=5760, K=2880) ===
       cell    med ms    min ms    max ms  spread%    TFLOPS   delta%
     (1, 4)    0.1472    0.1472    0.1476    0.27%    1846.4   +0.00% *base
     (1, 2)    0.1351    0.1350    0.1353    0.24%    2011.5   +8.21%
     (1, 8)    0.1380    0.1379    0.1382    0.17%    1968.9   +6.22%
     (3, 4)    0.1502    0.1500    0.1505    0.29%    1809.0   -2.07%
     (4, 4)    0.1463    0.1460    0.1466    0.44%    1857.5   +0.60%
     (1, 1)    0.1382    0.1381    0.1385    0.29%    1966.6   +6.11%
     (2, 4)    0.1464    0.1462    0.1464    0.11%    1857.0   +0.57%
```

**This is NOT a valid drift test of the GateUP-B4-M2048 dgrad-via-H4
rule.** Why:

- The dgrad-via-H4 path traverses a **different dispatcher rule**: after
  the H4 transpose (grouped_gemm_fp8_impl.py:467-503), the dispatcher
  sees `(avg_m=2048, n=K_fwd=2880, k=N_fwd=5760)` → tiles_n=11, k=5760
  → hits config.py:2602-2986 (NOT config.py:1418-1482). The rule under
  test (R45 hypothesis) only governs the fwd RCR direction.
- The dgrad-via-H4 production rule at config.py:2602-2986 has compounded
  R10 (slots=200) + R15 (chunk_size=24) + R16 (gm=1, xcds=None=8) levers,
  yielding +8.4% kernel lift over the R8 baseline. My probe forced
  `slots=0, cs=0` — i.e. the **pessimal** baseline that has not been
  shipped since R8. The (1, 4) baseline at 1846.4 T is the
  R8-era pre-R10/R15/R16 cell, NOT the production rule.
- The "winners" at +6-8% over this artificial baseline (e.g. (1,2) 2011.5T,
  (1,8) 1968.9T, (1,1) 1966.6T) are merely cells that happen to outperform
  a deliberately broken baseline. They likely do NOT outperform the
  shipped (gm=1, xcds=None=8, slots=200, cs=24) rule which probably
  delivers ≥2080 T per R16's "+3.48% over 2012 T" estimate.

A proper dgrad-via-H4 drift audit would need to (a) hold (slots=200,
cs=24) FIXED and sweep (gm, xcd) — different probe, different
hypothesis. Doing it here would violate the SKILL "1 hypothesis per
round" rule. **Logged as the R46 forward pointer below.**

## Decision

**FALSIFIED** on the R45 hypothesis (GateUP-B4-M2048 fwd RCR rule
drift). Mirrors R44 (Down-B4-M2048 fwd RCR rule drift FALSIFIED).

Per R44's BACKUP plan: R44 + R45 now establish **two consecutive empirical
drift FALSIFICATIONS** on the two B=4 M=2048 fwd RCR rules in the
gpt_oss FP8 metric — the very cells that R10-R23-R34 documented as the
worst absolute-TFLOPS shapes in the suite. Both rules tuned multiple
rebuilds ago hold at local optima on the current binding. The structural
~700 plateau is robust to the dispatcher (gm, xcds) lever class on
this round's FP8 binding.

## Forward pointer to R46

Two viable next-round directions (single-hypothesis, ranked):

1. **PRIMARY — GateUP-B4-M2048 dgrad-via-H4 RCR rule (gm, xcds) drift
   audit, METHODOLOGY-CORRECTED.** The R45 dgrad column (uninterpretable
   above) hinted that the (1, 8) and (1, 2) cells at slots=0/cs=0 beat
   the (1, 4) cell at slots=0/cs=0. With the R10/R15/R16 levers held
   FIXED at production (slots=200, cs=24), the (gm, xcd) optimum may
   have shifted. R45 baseline data point: (1, 4) at slots=0/cs=0 =
   1846.4 T; production (1, xcds=None=8, slots=200, cs=24) per R16
   estimate ≈ 2082 T. Probe should clone _probe_round_45 → swap to
   `time_dgrad`-only, FIX slots=200 + cs=24, sweep cells {(1, 8)*current,
   (1, 4), (1, 2), (1, 1), (2, 8), (4, 8), (8, 8)} where (1,8) is the
   current production. Same SHIP gate: best ≥ 1.0% lift over (1,8) AND
   signal > spread.

   This audits the rule that has the LARGEST stack of compounded R10/
   R15/R16 levers in the gpt_oss FP8 dispatcher — the highest a priori
   probability of drift-induced rule mismatch on the current binding.
   Estimated probe wall: ~12 s (same methodology as R45). If FALSIFIED,
   the rule audit is functionally complete on the dispatcher (gm, xcds)
   surface for B=4 RCR shapes in the gpt_oss FP8 suite.

2. **BACKUP — closure documentation (R44 option 2 escalated).** Three
   consecutive empirical FALSIFICATIONS (R44 Down fwd + R45 GateUP fwd
   + R46 GateUP dgrad if it also returns NEUTRAL) would justify
   declaring the dispatcher (gm, xcds) lever class feature-complete on
   the structural ~700 plateau. Write a 34-round audit summary
   (R10-R46) and recommend transitioning the daemon to a different
   task or accepting the plateau. This option is held in reserve until
   R46's primary returns.

## Files touched (Primus-Turbo only, no HK edits, no config.py edits)

- `scripts/_probe_round_45_gateup_b4_m2048_drift.py` (new probe)
- `analysis/_notes/round-45-gateup-b4-m2048-fwd-gm-xcd-drift-FALSIFIED-rule-at-local-optimum.md` (this note)
