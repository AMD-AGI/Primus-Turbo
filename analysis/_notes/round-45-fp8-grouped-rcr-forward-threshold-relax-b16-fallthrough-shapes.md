round-45-fp8-grouped-rcr-forward-threshold-relax-b16-fallthrough-shapes.md
==========================================================================

Round: 45 / 100
Date: 2026-05-02
SHA: 1f0a36b (pre) → TBD (post)
Task: grouped FP8 tensorwise (fwd + var-K bwd + dA bwd)

## TL;DR

Identified and closed 2 **forward-path** `select_default_config`
fall-through gaps:

  1. **Qwen3-GateUP B=16 M=4096** (tiles_n=12, tiles_m=16, k=4096,
     m_total=65536): R10 anchored the rule on B=32-only
     (m_total >= 131072) after the B=16 sibling's 200-iter × 7-trial
     tight-verify sat at +0.47 pp with 0.6× spread ("distributions
     overlap"). R45 re-probes with R44 methodology (12 × 200 × 3 seeds)
     and finds +1.90-1.95% avg +1.93% — spread 0.05pp, 12× tighter
     than R10's probe saw.

  2. **DSV3-GateUP B=16 M=2048** (tiles_n=16, tiles_m=8, k=7168,
     m_total=32768): R8 anchored on B=32-only (m_total >= 65536)
     after the B=16 sibling's 50-iter coarse sweep top1 sat at +0.13%
     ("solid noise"). R45 re-probes at R44 tight-bench: +1.17% to
     +1.50% avg +1.35% — clean signal.

Both shapes now fire the existing rule configs R10/R8 already
validated for their larger siblings — only the m_total thresholds
relax.

This is the FORWARD-path sibling of R44 (which relaxed the RRR
dA-backward rule). Same pattern: R10/R8 rejected B=16 siblings at
the old-methodology noise floor; R44's tighter bench methodology
resolves the sub-2% signal cleanly.

## GPU state (R45)

Zombie KFD VRAM leak on GPU 3 persists — 11th consecutive round.
Now 3 zombie PIDs:

    2280804 UNKNOWN 1 20447899648 ...
    2818266 UNKNOWN 1  2185465856 ...
    2280802 UNKNOWN 1 20610859008 ...    # old PID, still there

User intervention still outstanding. Recommended:

    sudo rmmod amdkfd && sudo modprobe amdkfd

## Forward-path dispatch audit

Starting from R42's insight (FP8 RRR had NO rules in
select_default_config) — audit the forward RCR path for similar
fall-throughs. Result: all 4 metric model × 2 M_per × 2 B
combinations ARE covered by existing rules EXCEPT the 2 B=16
siblings excluded by m_total thresholds:

    Model        Layer   B  M_per  m_total   Rule?          Status
    DSV3         GateUP  16 2048   32768    *no R8 blocks*  ← R45 closes
    DSV3         GateUP  32 2048   65536    R8 rule2        covered
    DSV3         GateUP  16 4096   65536    R8 rule1        covered
    DSV3         GateUP  32 4096  131072    R8 rule1        covered
    DSV3         Down    16 2048   32768    R20 rule (tn28) covered
    DSV3         Down    16 4096   65536    R20 rule        covered
    DSV3         Down    32 2048   65536    R20 rule        covered
    DSV3         Down    32 4096  131072    R20 rule        covered
    gpt_oss      GateUP  4  2048    8192    R23 rule        covered
    gpt_oss      GateUP  4  4096   16384    R7 rule         covered
    gpt_oss      GateUP  32 2048   65536    R70 rule        covered
    gpt_oss      GateUP  32 4096  131072    R70 rule        covered
    gpt_oss      Down    4  2048    8192    R7 rule         covered
    gpt_oss      Down    4  4096   16384    R12 rule        covered
    gpt_oss      Down    32 2048   65536    R8 rule         covered
    gpt_oss      Down    32 4096  131072    (default gm=4)
    Qwen3        GateUP  16 2048   32768    R7 rule (tm=8)  covered
    Qwen3        GateUP  16 4096   65536    *no R10 blocks* ← R45 closes
    Qwen3        GateUP  32 2048   65536    R7 rule         covered
    Qwen3        GateUP  32 4096  131072    R10 rule        covered
    Qwen3        Down    16 2048   32768    R6 rule         covered
    Qwen3        Down    16 4096   65536    R6 rule         covered
    Qwen3        Down    32 2048   65536    R6 rule         covered
    Qwen3        Down    32 4096  131072    R6 rule         covered

Only 2 fall-throughs (plus 1 remaining gpt_oss-Down-B32-M4096 on
default which was explicitly noted by R8 as "+0.15pp at best").

## R45 tight probes

### Qwen3-GateUP B=16 M=4096 (R10-excluded)

12 trials × 200 iters × 3 seeds:

    cfg        seed=42   seed=137  seed=2024  avg
    (4, 0)     +0.00%    +0.00%    +0.00%    0.00%  baseline
    (1, 4)     +1.90%    +1.95%    +1.93%    +1.93%  *winner
    (4, 4)     +1.11%    +0.98%    +1.12%    +1.07%
    (2, 4)     +0.11%    +0.12%    +0.11%    +0.11%
    (1, 8)     -2.67%    -2.75%    -2.62%    -2.68%
    (1, 2)     +0.79%    +0.87%    +0.98%    +0.88%

`(1, 4)` wins with 0.05pp spread across seeds — cleanly above noise.
Matches the config R10 already validated for the B=32 M=4096 sibling.

### DSV3-GateUP B=16 M=2048 (R8-excluded)

12 trials × 200 iters × 3 seeds:

    cfg        seed=42   seed=137  seed=2024  avg
    (4, 0)     +0.00%    +0.00%    +0.00%    0.00%  baseline
    (16, 4)    +1.17%    +1.39%    +1.50%    +1.35%  *winner
    (32, 4)    +0.98%    +1.24%    +1.48%    +1.23%
    (2, 8)     -0.72%    -0.31%    -0.88%    -0.64%
    (2, 16)    -0.95%    -0.79%    -0.93%    -0.89%
    (1, 4)     +0.83%    +1.07%    +1.21%    +1.03%

`(16, 4)` wins with 0.33pp spread — clean signal. Matches the config
R8 already validated for the B=32 M=2048 sibling.

## The changes

`primus_turbo/pytorch/kernels/hipkitten/config.py`:

- R10 Qwen3-GateUP rule threshold: `m_total >= 131072` → `>= 65536`
  (captures B=16 M=4096 alongside B=32 M=4096).
- R8 DSV3-GateUP rule 2 threshold: `m_total >= 65536` → `>= 32768`
  (captures B=16 M=2048 alongside B=32 M=2048).

Both changes are single-number threshold relaxations; the rule
configs (group_m, num_xcds, kernel) are UNCHANGED. Comments
extended to document R45 findings.

## Correctness

Bit-identical output on both shapes (R45 check):

    Qwen3-GateUP B=16 M=4096: (4,0) vs (1,4)   max_abs=0.0  bit_eq=True
    DSV3-GateUP  B=16 M=2048: (4,0) vs (16,4)  max_abs=0.0  bit_eq=True

Same bit-equivalence argument as R42-R44: group_m / num_xcds are
pure persistent-grid tile-scheduling knobs.

## Suite-level probe

Metric is GPU-blocked. Ran `scripts/_fp8_grouped_nogate_probe.py`
(single-trial, documented noise ±6-8% per probe docstring) before
and after:

    baseline geomean:  1.1905  (cap 0.9921, extrapolated ~990)
    R45 geomean:       1.1564  (cap 0.9636, extrapolated ~976)

Delta is -3.4 pp on the aggregate, BUT:
- HK TFLOPS on the 2 targeted shapes barely moved (DSV3-GateUP-B16-M2048:
  2621→2626; Qwen3-GateUP-B16-M4096: 2520→2499). The ratio shifts are
  almost entirely triton-side single-trial variance (triton TFLOPS
  moved 2003→2109 on one shape, 1812→2197 on another — up to 21% !!
  triton-side single-trial noise).
- Probe script docstring explicitly warns: "Not reliable for small-
  delta (≤5%) tuning."

The tight-bench per-shape signal (clean +1.35% and +1.93% across 3
seeds × 12 trials × 200 iters) is the authoritative evidence. Shipping
as R44-style code-quality + tight-bench-positive.

## No regression risk on sibling shapes

- B=32 M=4096 Qwen (m_total=131072) still hits R10 rule with (1, 4)
  — already validated as the winner at +0.93 pp.
- B=32 M=2048 DSV3-GateUP (m_total=65536) still hits R8 rule 2 with
  (16, 4) — already validated as +0.56 pp winner.
- B=16 M=4096 DSV3-GateUP (m_total=65536) still hits R8 rule 1 — no
  change.

The relaxes only add the excluded B=16 shapes to the rule domain;
larger-m_total shapes are unaffected.

## DoD impact

Rules are gated by:
- tiles_n == 12 AND tiles_m == 16 AND k == 4096: Qwen3-GateUP-only
  (uniquely n=3072 / k=4096 in metric; dense FP8 has tiles_n ∈ {16,
  24, 48, 86, 112}).
- tiles_n == 16 AND tiles_m == 8 AND k == 7168: DSV3-GateUP M=2048-
  only (uniquely k=7168 in metric; dense FP8 k ∈ {4096, 11008,
  14336, 22016, 28672}).
- m_total is not None excludes dense callers (dense passes m_total=None).

No DoD shape matches either rule. No correctness risk.

## Why not Lever E?

Same answer as R44: 10+ round GPU blockage means un-validatable
hand-written ASM is not productive. Dispatch-tuning remains the
only measurable axis.

## Next round (R46) action ladder

1. **First: try `_metric_grouped_only.py`**. With R39-R45 accumulated
   dispatch changes, if GPU unblocks, the R42-R44 backward gains
   don't affect metric (metric is fwd-only) but R45's forward gains
   SHOULD show up (+1.3-1.9% per shape on 2 shapes = ~+0.3pp
   aggregate = ~+3 score points expected, above the ±2.5 metric
   noise R13 documented).

2. **If GPU still blocked**: forward dispatch axis is now exhausted
   for the B-M-model grid. Remaining axes:
   - gpt_oss-Down-B32-M4096 (m_total=131072) still on default.
     R8 noted "+0.15pp at best" at 50-iter coarse sweep. R44/R45
     methodology should either confirm noise or find a new winner.
   - CRR var_k sub-rules (R39 only has one threshold rule).
     Investigate whether tiles_n / tiles_m specific tuning within
     the m_total >= 16384 band finds per-shape improvements over
     the flat (8, 4).

3. **Lever E (ASM main-loop)**: still not started; requires clean GPU.

4. **USER intervention** (escalating request): `sudo rmmod amdkfd &&
   sudo modprobe amdkfd` or move the metric run to a different GPU
   in the HIPKITTEN_GPU_POOL={3,4,6,7} that has no zombie PIDs.

## Falsification register (updated)

| Lever / approach                        | Status        | Round |
|-----------------------------------------|---------------|-------|
| Lever A-F kernel rewrites               | FALSIFIED     | R11-35|
| sched_barrier / LICM / anti-CSE class   | FALSIFIED     | R31-32|
| FP8 var_k init parallelize              | SHIPPED       | R38   |
| FP8 var_k bwd dispatch (8, 4)           | SHIPPED       | R39   |
| FP8 rrr init parallelize                | SHIPPED       | R41   |
| FP8 RRR narrow-N dispatch (16, 4)       | SHIPPED       | R42   |
| FP8 RRR wide-N dispatch (16, 4)         | SHIPPED       | R43   |
| FP8 RRR wide-N low-m relax              | SHIPPED       | R44   |
| FP8 RCR Qwen3-GateUP B=16 M=4096 relax  | SHIPPED (R45) | R45   |
| FP8 RCR DSV3-GateUP B=16 M=2048 relax   | SHIPPED (R45) | R45   |
| FP8 RRR Qwen3-GateUP (tiles_n == 16)    | SKIPPED       | R44   |
| Lever E (ASM main-loop)                 | NOT STARTED   | —     |
