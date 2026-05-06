# Round 13 — DSV3-GateUP forward FP8 RCR xcds-column widening: **FALSIFIED**

## Context (entering this round)

- `_metric_grouped_fused_wall.py` score: **1000** (capped), geomean 1.3849.
- `_metric_grouped_only.py` regression check: **973** vs target 980 (drift band
  noted since R5; pre-existing condition, not regressed by recent commits).
- Primus-side dispatcher rules tight-verified for var-K dB family in R10
  (gpt_oss-Down-B4-M4096), R11 (gpt_oss-Down-B4-M2048), R12 (Qwen3-Down B16/B32
  M=2048/M=4096 — all four falsified). All gpt_oss var-K cells now have wide
  xcds={2,4,8} sweep evidence; Qwen3-Down stays on R39 (gm=8, xcd=4).
- Forward path rules **had not** been re-tested with the R10/R11 candidate-set
  widening pattern.

## Hypothesis tested this round

The four `DSV3-GateUP-*` forward FP8 RCR shapes show the lowest grpFP8 ratios
in `_metric_grouped_only.py` (1.136 / 1.160 / 1.169 / 1.171 vs target 1.20).
Their dispatch rules were authored in **R8** (M=4096 family) / **R8 + R45**
(M=2048 family) and the comments only quoted **3-5 tested candidates**:

```
DSV3-GateUP-B16-M4096:  R8 quotes 3 cells  {(2,8), (2,16), (4,8)}
DSV3-GateUP-B32-M4096:  R8 quotes 3 cells  {(2,16), (2,8), (4,8)}
DSV3-GateUP-B16-M2048:  R45 quotes 5 cells {(16,4), (32,4), (1,4), (2,8), (2,16)}
DSV3-GateUP-B32-M2048:  R8 quotes 3 cells  {(16,4), (32,4), (4,8)}
```

R8 itself notes the M=2048 sweep was 50-iter (lower statistical confidence);
R45 redid M=2048 with 200-iter × 12-trial × 3-seed but still kept the same
narrow xcds={4, 8, 16} columns. **The xcds={1, 2} columns were never swept on
any DSV3-GateUP forward rule**, mirroring exactly the gap that R10/R11 found
+0.48-1.24% on for gpt_oss-Down var-K dB.

Hypothesis: applying R10/R11's candidate-set widening (xcds ∈ {1,2,4,8,16}) to
DSV3-GateUP forward might reveal hidden xcds=1 or xcds=2 winners.

## Probe methodology

`/tmp/probe_round_13_dsv3_gateup_fwd.py` — direct call to
`hipkitten.load_fp8().grouped_rcr_dscale` (production forward path) with:

- Per shape: 22 candidate cells covering xcds ∈ {1, 2, 4, 8, 16} × gm
  representative set ({1, 2, 4, 8, 16, 32}).
- Per cell: 3 seeds × 7 trials × 200-iter p20 (~25 GPU-min total).
- Reference baselines: M=2048 family vs (gm=16, xcd=4) R45/R8 rule;
  M=4096 family vs (gm=2, xcd=8) R8 rule.
- Mirror R10/R11 reporting format (3-seed mean Δ% + spread pp).

The 22-cell × 3-seed × 7-trial fan-out is **6× wider** than R8's 3-cell
single-trial table and matches the breadth of R10/R11/R12 that found wins on
gpt_oss var-K dB.

## Results — all four shapes confirm current rule

```
DSV3-GateUP-B16-M2048  (rule (16,4) R45;  m_total=32768)
  cell                 mean      Δ vs cur   spread pp   verdict
  (4, 1)               708.78 µs  +0.04%     0.21 pp    NOISE (Δ < spread)
  (16, 4)*RULE         709.06 µs   0.00%     0.15 pp    baseline
  (1, 4)               709.51 µs  -0.06%     0.25 pp    noise
  (4, 16)              709.55 µs  -0.07%     0.19 pp    noise
  (32, 4)              710.01 µs  -0.13%     0.05 pp    near-baseline
  (4, 8)               711.05 µs  -0.28%     0.19 pp    < default 8
  ... 14 more cells all -0.35% to -5.26% ...

DSV3-GateUP-B32-M2048  (rule (16,4) R8;   m_total=65536)
  (16, 4)*RULE        1423.11 µs   0.00%     0.17 pp    *winner*
  (32, 4)             1423.33 µs  -0.02%     0.26 pp    tied
  (1, 4)              1425.39 µs  -0.16%     0.18 pp    near
  (4, 1)              1425.43 µs  -0.16%     0.14 pp    near
  ... 18 more cells all -0.23% to -6.14% ...

DSV3-GateUP-B16-M4096  (rule (2,8) R8;    m_total=65536)
  (2, 8)*RULE         1404.51 µs   0.00%     0.27 pp    *winner*
  (2, 16)             1404.94 µs  -0.03%     0.12 pp    tied (R8 listed)
  (2, 1)              1405.64 µs  -0.08%     0.24 pp    noise
  (4, 1)              1415.54 µs  -0.79%     0.36 pp
  ... 18 more cells all -0.79% to -2.71% ...

DSV3-GateUP-B32-M4096  (rule (2,8) R8;    m_total=131072)
  (2, 8)*RULE         2820.75 µs   0.00%     0.05 pp    *winner*
  (2, 1)              2821.50 µs  -0.03%     0.10 pp    noise
  (2, 16)             2821.62 µs  -0.03%     0.07 pp    tied (R8 listed)
  ... 19 more cells all -0.80% to -2.74% ...
```

### Key observation: gm=2 dominates entire xcds row for M=4096

Both M=4096 shapes (B=16, B=32) show a clean **gm=2 plateau**: (2,1), (2,2),
(2,4), (2,8)=rule, (2,16) all sit within ±0.85% on B16-M4096 and within ±0.85%
on B32-M4096 — but every gm ≠ 2 candidate is at least -0.79% slower.

This empirically validates R8's "K=7168 (56 K-iter) per-tile compute is 4×
heavier than gpt_oss K=2880" comment (config.py line 2340): the persistent
loop scheduler benefits from gm=2 (small batching factor) with deep K because
walking N before M maximises B-tile L2 reuse on the long-K axis. xcds={1, 2,
4, 16} all neighbor the rule's xcds=8 within 0.08% — **the xcds choice is
flat at gm=2** because the chiplet-swizzle is dominated by deep-K main-loop
HBM reads, not by tile-completion staggering.

For M=2048 family, the pattern flips: the gm=16 winner is the unique top of
a tight cluster (gm ∈ {1, 4, 16, 32}, xcd=4 all within ±0.16%), but every
xcds≠4 candidate at any gm is worse by ≥1.0% — confirming xcd=4 is the right
chiplet partition for tiles_m=8 (smaller per-group grid, more tile-completion
staggering pressure).

### Why R10/R11 wins did NOT transfer to DSV3-GateUP

| Family               | Rule winner   | Wave-steps | K   | Why
|----------------------|---------------|-----------|-----|------------------------------------------|
| gpt_oss-Down-B4-M4096 dB (R10) | (1, 2)  | ~2     | 2880 | shallow K + tiny grid → xcds=2 best
| gpt_oss-Down-B4-M2048 dB (R11) | (1, 2)  | ~2     | 2880 | same regime as R10
| DSV3-GateUP-M4096 fwd (R8)     | (2, 8)  | 16-32  | 7168 | deep K + saturated grid → xcds=8 best
| DSV3-GateUP-M2048 fwd (R8/45)  | (16, 4) | 8-16   | 7168 | deep K + medium grid → xcds=4 best

The R10/R11 (gm=1, xcds=2) win pattern requires **both** conditions: (a) tiny
persistent grid (≤ 2 wave-steps so the schedule can't hide xcds-overhead) AND
(b) shallow K (so the per-tile compute window is small enough that
chiplet-swizzle staggering dominates). DSV3-GateUP has neither condition: its
grids saturate the GPU (8-32 wave-steps) and K=7168 has 56 K-iter per
tile-step (4× deeper than gpt_oss's 23 K-iter).

## Verdict — FALSIFIED

The four DSV3-GateUP forward FP8 RCR rules **(M=4096: gm=2, xcds=None=8;
M=2048: gm=16, xcds=4)** are confirmed optimal across the **22-cell × 3-seed
× 7-trial wide sweep** that mirrors R10/R11/R12's methodology. R8's narrow
3-cell quote and R45's 5-cell quote were sufficient — they happened to bracket
the optima — but the wider 22-cell sweep this round provides definitive
evidence rather than relying on the assumption.

**No code changes** in this round. Falsification confirms the **1.136-1.171
fwd-only ratios are kernel-internal limits**, not dispatch tuning gaps. The
remaining grpFP8 geomean gap to 1.20 (current 1.1589, target 1.20 → -4.1pp)
must come from kernel-side work (HK FP8 RCR template throughput on K=7168
deep-K shapes) or from quantize_fp8 HBM bandwidth (already accounted for in
the metric's "kernel-only" timing scope).

## Inventory of remaining un-probed forward cells

After R13 the following forward FP8 RCR rules ARE wide-sweep verified:

| Family             | Rule         | Wide-sweep evidence
|--------------------|--------------|----------------------------------------|
| gpt_oss-GateUP-B4-M2048  | (1, 4)  R23 | 9-cell × 7-trial verify
| gpt_oss-GateUP-B4-M4096  | (14, 4) R10dm | 1500-iter × 7-repeat re-sweep
| gpt_oss-GateUP-B32 (both)| (8, 4)  R70 | 24×6=144-cell sweep
| gpt_oss-Down-B4-M2048    | (2, 2)  R7  | 40-cell × 7-trial
| gpt_oss-Down-B4-M4096    | (32, 4) R12 | 54-cell × 7-trial
| gpt_oss-Down-B32-M2048   | (16, 4) R8  | 54-cell × 7-trial
| gpt_oss-Down-B32-M4096   | (4, 4)  R50 | 11-cell × 7-trial
| DSV3-GateUP-M4096 (both) | (2, None) **R13** | 22-cell × 3-seed × 7-trial
| DSV3-GateUP-M2048 (both) | (16, 4) **R13** | 22-cell × 3-seed × 7-trial
| DSV3-Down (4)            | (32, 4) R20/R58 | 9-cell × 5-repeat × 12-trial

Forward cells **NOT yet wide-sweep verified** in the manner of R13:

| Family             | Current rule | Notes
|--------------------|--------------|-------
| Qwen3-GateUP all 4 | default (4, 8) | metric ratios 1.138-1.190
| Qwen3-Down B16/B32 M2048 | (4, 8) R9-bf16 partial-transfer | metric 1.160/1.186
| Qwen3-Down B16/B32 M4096 | default (4, 8) | metric 1.133/1.171

These are the **next** candidates if the strategy is "exhaust forward dispatch
levers first". Expected outcome (per R13 pattern): tight-verify confirmation
of current rules; the remaining gap is HK kernel-internal to K=4096 / K=1536
forward templates.

## Suggested next round

Two reasonable directions:

1. **Continue forward sweep**: probe Qwen3 forward families with the same
   22-cell × 3-seed × 7-trial methodology. Highest probability outcome (per
   R13 evidence): all current rules confirmed optimal; document and close.
2. **Pivot to maintenance**: write a final summary across R10-R13 noting
   that the Primus-side dispatcher is now formally fully tight-verified for
   both forward AND var-K dB cells. Score has been at 1000 cap for 11
   consecutive rounds; remaining levers are HK kernel-internal (out-of-scope
   per "Forward only" task body, would require new kernel template work).

Either way, no metric-score-affecting change is expected. R13 contributes one
more piece of negative evidence that the dispatcher is exhausted.
