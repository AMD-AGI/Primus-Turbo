# Round-8 (gpt_oss FP8 kernel-only ceiling) — R7 xcds=8 audit extension FALSIFIED

**Goal**: continue the R7 (commit `1d526e2`) audit thread —
"R34 picked (gm=16, xcd=4) for GateUP-B4-M2048 dA RCR by sweeping
xcds=4-only; widening to xcds=8 found (gm=8, xcds=8) wins +2.11%
robustly. Audit other R34/R8/R9/R3-style narrow rules for the same
missing-column hole."

**Round-8 anchor**: 3 sibling rules where the original sweep was
xcds-limited (xcds=4-only or xcds∈{2,4}-only); each tight-verified
with the exact R7 methodology (1500-iter × 7-trial × 3-seed p20,
direct kernel call via `grouped_rcr_dscale` / `grouped_variable_k_crr_dscale`).

## Summary table

| Probe | Anchor shape | Current rule | xcds=8 column tested? | Verdict |
|---|---|---|---|---|
| 1 | GateUP_B32_M2048 dgrad RCR (m_total=65536) | R34 (gm=16, xcd=4) | NO (R7 + R34 both xcds=4-only) | **FALSIFIED** |
| 2 | GateUP_B4_M2048 wgrad var-K dB (m_total=8192) | R3 (gm=1, xcd=4) | NO (R3 + R35 both xcds∈{1,2,4}-only) | **FALSIFIED** |
| 3 | GateUP_B4_M4096 dgrad RCR (m_total=16384) | R8 (gm=1, xcd=4) | partial (only (8,8) cell explicitly logged in R8) | **FALSIFIED** |

Every cell in every probe LOST against the current rule, confirming
the existing carve-outs are at the local optimum on the current
binding. R7's win was a special small-grid case unique to
m_total=8192 + B=4 + dgrad geometry; the lever does NOT generalize
to the siblings audited here.

## Probe details

### Probe 1 — GateUP_B32_M2048 dgrad RCR (R34) xcds=8 column

Probe script: `scripts/_probe_round_8_gateup_b32_m2048_dgrad_xcds8.py`

```
baseline R34 (gm=16, xcd=4):
  seed 42  : 0.8590 ms / 2531 T  (spread 0.47pp)
  seed 137 : 0.8601 ms / 2528 T  (spread 0.32pp)
  seed 2024: 0.8613 ms / 2525 T  (spread 0.41pp)

candidates (xcds=8 column + R7 borderlines):
  cell      seed42 Δ%  seed137 Δ%  seed2024 Δ%   med Δ%   verdict
  ( 1, 8)   -9.07      -9.13       -9.18         -9.13    LOSS
  ( 4, 8)   -3.56      -3.29       -3.23         -3.29    LOSS  (= R34's pre-rule default)
  ( 8, 8)   -7.33      -7.23       -7.13         -7.23    LOSS  (the R7 winner cell on B=4 sibling)
  (16, 8)   -9.22      -9.28       -9.45         -9.28    LOSS
  (32, 8)   -9.43      -9.24       -9.43         -9.43    LOSS
  ( 4, 0)   -6.61      -6.58       -6.46         -6.58    LOSS  (default = xcds=8 via BSNX)
  ( 1, 4)   -0.55      -0.68       -0.41         -0.55    LOSS  (R4-borderline; R7 already TIE'd)
```

Verdict: **R34 (gm=16, xcd=4) is the unique optimum**. The xcds=8
column LOSS pattern matches the m_total=65536 grid's preference for
xcds=4 chiplet partition: 88 tile-steps per group × 32 groups =
2816 tile-steps over 256 CUs ≈ 11 wave-steps per slot. The deeper
grid (vs B=4's 1.4 wave-steps) saturates the chiplet-pair work
distribution, so the wider xcds=8 spread loses parallelism without
the small-grid amortisation benefit that wins on B=4.

Bit-eq: max_abs_diff = 0 across 7 candidate cells vs (16, 4) at
seed 42.

### Probe 2 — GateUP_B4_M2048 wgrad var-K dB (R3) xcds=8 column

Probe script: `scripts/_probe_round_8_gateup_b4_m2048_wgrad_xcds8.py`

```
baseline R3 (gm=1, xcd=4):
  seed 42  : 0.1648 ms / 1650 T  (spread 0.36pp)
  seed 137 : 0.1650 ms / 1647 T  (spread 0.19pp)
  seed 2024: 0.1646 ms / 1651 T  (spread 0.15pp)

candidates (xcds=8 column):
  cell      seed42 Δ%  seed137 Δ%  seed2024 Δ%   med Δ%   verdict
  ( 1, 8)   -4.34      -4.40       -4.59         -4.40    LOSS
  ( 2, 8)   -1.18      -0.67       -0.58         -0.67    LOSS
  ( 4, 8)   -0.10      +0.12       -0.10         -0.10    TIE
  ( 8, 8)   -2.00      -1.93       -2.07         -2.00    LOSS
  (16, 8)   -1.91      -1.67       -1.91         -1.91    LOSS
  (32, 8)   -3.96      -3.80       -4.10         -3.96    LOSS
  ( 1, 0)   -4.17      -4.18       -4.35         -4.18    LOSS  (default = xcds=8 via BSNX)
```

Verdict: **R3 (gm=1, xcd=4) is the unique optimum** on this
geometry. The (4, 8) cell ties baseline within noise but doesn't
beat it. The 968 tile-step / ~3.78 wave-step grid prefers xcds=4
even at the small B=4 m_total=8192 scale — the var-K kernel's
LDS/scale prologue per tile-step weights differently than the
RCR kernel's prologue, so the R7 RCR small-grid-prefers-xcds=8
finding does NOT transfer to var-K. R9's sibling B4-M4096 probe
already foreshadowed this with xcds=8 LOSS at -1.94..-3.19pp;
this round's M=2048 sibling shows the same pattern.

Bit-eq: max_abs_diff = 0 across 7 candidate cells vs (1, 4) at
seed 42.

### Probe 3 — GateUP_B4_M4096 dgrad RCR (R8) xcds=8 column

Probe script: `scripts/_probe_round_8_gateup_b4_m4096_dgrad_xcds8.py`

```
baseline R8 (gm=1, xcd=4):
  seed 42  : 0.2146 ms / 2533 T  (spread 0.41pp)
  seed 137 : 0.2144 ms / 2536 T  (spread 0.21pp)
  seed 2024: 0.2139 ms / 2542 T  (spread 0.22pp)

candidates (xcds=8 column):
  cell      seed42 Δ%  seed137 Δ%  seed2024 Δ%   med Δ%   verdict
  ( 1, 8)   -4.55      -4.56       -4.40         -4.55    LOSS
  ( 4, 8)   -0.98      -0.89       -1.09         -0.98    LOSS
  ( 8, 8)   -2.21      -2.37       -2.36         -2.36    LOSS  (R8 documented as -1.80%)
  (16, 8)   -0.68      -0.80       -1.05         -0.80    LOSS
  (32, 8)   -0.65      -0.72       -0.94         -0.72    LOSS
  ( 4, 0)   -0.24      -0.28       -0.45         -0.28    TIE   (default = xcds=8 via BSNX)
```

Verdict: **R8 (gm=1, xcd=4) is the unique optimum** at m_total=
16384. The closest xcds=8 cells (16, 8) and (32, 8) lose by
-0.68..-1.05% — clear of run-to-run spread (≤0.41pp). The
m_total=16384 grid sits between the B=4 M=2048 (8192) and B=4
M=4096 boundaries; the xcds=8 win at the M=2048 small grid does
NOT extend even one M-tier up.

## Conclusion

R7's `(gm=8, xcds=8)` win was specific to the tiny m_total=8192
RCR dgrad grid (≈1.4 wave-steps / slot) on the B=4 M=2048 GateUP
shape. The siblings tested here all sit on grids where the xcds=4
chiplet partition is already optimal, AND the var-K kernel's
prologue weighting flips the RCR small-grid xcds=8 lesson on its
head. Three more "R34/R8/R3-style narrow rule" audit threads
closed; no further metric improvement landed this round.

Score check: post-falsification metric = 685 (vs round-7 baseline
686, within ±5 noise band per the metric noise floor characterised
in R19 (`efb9bc1f`) and R36).

## Followups

* **No more "missing xcds=8 column" audits warranted** for gpt_oss
  RCR rules. The R34 / R8 / R3 family is exhausted on the current
  binding; the only un-audited cells are at xcds ∈ {16, 32} which
  the BF16 path uses but FP8 grouped's persistent grid (256 CUs)
  has no apparent benefit from since num_xcds > 8 on a 1.4 ..
  11 wave-step grid is meaningless (no extra parallelism beyond
  the chiplet topology).
* **Phase 3 "kernel template overrides" is NOT a lever for grouped
  GEMM**. Per `primus_turbo/pytorch/kernels/hipkitten/dispatch.py`
  lines 132-154 / 195-212, `force_rcr_kernel` only wraps `dense_run`
  (i.e., `gemm_rcr`). The grouped path (`grouped_run` and the
  grouped FP8 inline calls in `grouped_gemm_fp8_impl.py`) has no
  template-id parameter. Selecting `kernel="4"` / `"8"` in
  `HipKittenConfig` is **silently ignored** for the grouped path —
  any "Phase 3" template override would require a NEW
  `grouped_rcr_with_kernel(...)` binding entry, which falls under
  "kernel surgery" and is out of scope for the dispatcher tuning
  task.
* Next-round dispatcher audit candidates (lower probability of WIN
  but un-tested on the current binding):
  - **num_slots** lever for var-K wgrad on GateUP shapes (R2 wired
    `num_slots=192` only for Down `k=n=2880` shapes; GateUP
    `n=5760` was excluded). The 3.78 wave-step / slot grid for
    GateUP B=4 M=2048 var-K might benefit from `num_slots=192`
    for similar amortisation. R2 explicitly excluded GateUP from
    the rule — worth re-probing on the current binding to either
    confirm exclusion is needed or extend the rule.
  - The dgrad RCR rules at tiles_n=11 (Down family) also have
    R2/R7-shared dispatcher keys — these were tested in R17 but
    the metric still sits at 685 on the current build. Worth
    confirming R17's "all 5 fwd RCR rules at ceiling" still holds
    on today's HEAD via a re-run probe.
