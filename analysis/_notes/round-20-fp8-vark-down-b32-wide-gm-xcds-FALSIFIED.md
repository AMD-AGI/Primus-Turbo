# Round-20 — gpt_oss FP8 var-K wgrad Down-B32 family wide (gm, xcds) sweep FALSIFIED

**Date**: 2026-05-08 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD `efb9bc1` → R20)
**Scope**: gpt_oss FP8 kernel-only metric, var-K wgrad section. Two metric cells:
`Down-B32-M2048-wgrad` (m_total=65536, ratio 0.597 — second-worst wgrad cell after Down-B4-M2048
wgrad at 0.467) and `Down-B32-M4096-wgrad` (m_total=131072, ratio 0.713).
**Goal**: Widen R1 / R30's narrow (gm, xcds) candidate set to test whether the kernel-rebuild
drift seen on the RCR rules (R17 falsified — drift tested, all RCR rules at ceiling) extends
to the var-K wgrad rules. R1 (this run) tested only `{(4, 4), (8, 4)}` for the m_total==65536
B32 cell; the xcds=2 column was never tested despite winning on the B=4 sibling (R11 (1, 2) for
Down-B4-M2048 wgrad with the SAME per-group [N=2880, K=2880] tile geometry).

## Bottom line

**Both shapes confirmed at dispatcher ceiling.** No alternative cell beats the current rule by
even 0.3 pp on either shape. The xcds=2 column is uniformly LOSS (-1.5 to -4.5 % on B32-M2048,
-2.0 to -4.0 % on B32-M4096). Following R17's RCR re-verify and R15's var-K fine-slot probe,
this closes the var-K wgrad dispatcher track for the B=32 Down sub-family — every shape × every
column × every row tested at this point is at or above its current rule.

| Shape | Current rule | Best alternative | Δ best alt | Verdict |
|---|---|---|---|---|
| Down-B32-M2048 wgrad | (gm=8, xcds=4) — R1 | (4, 4) | -0.60 % | UNIQUE TOP (R1 holds) |
| Down-B32-M4096 wgrad | (gm=4, xcds=4) — R30 | (8, 4) | -0.24 % | UNIQUE TOP (R30 holds) |

The 0.24 % "tie-loss" of (8, 4) on B32-M4096 confirms R1's earlier finding (R1 reported
-0.06..-0.13 pp 0/3 seeds) under the current binding — drift is ≤ 0.2 pp, well below the rule
margin.

## Probe protocol

- Script: `scripts/_probe_round_20_down_b32_wgrad.py` (single file, anchor variable
  swapped between M=2048 and M=4096 for the two cells; archived as
  `_probe_round_20_down_b32_m2048_wgrad.py` and `_probe_round_20_down_b32_m4096_wgrad.py`).
- Method: 30 warmup + 250 iters × 5 trials × 3 seeds, p20 per trial → median across trials,
  median across seeds. Direct kernel call to `hk.grouped_variable_k_crr_dscale` (bypassing the
  Python dispatcher).
- Candidate set (B32-M2048): 13 cells across (gm, xcds) ∈ {1, 2, 4, 8, 16, 32} × {2, 4, 8} —
  expanded over R1's 2-cell set to cover the xcds=2 column and the gm=1 / gm=32 extremes.
- Candidate set (B32-M4096): 12 cells, similar coverage plus (12, 4) (R25's DSV3 family
  winner).

## Down-B32-M2048 wgrad (m_total=65536) results

```
3-seed summary (median ms / TFLOPS / Δ vs current (8, 4) / cell spread)
  ★ ( 8, 4)   ms_med=0.6404  TFLOPS=1697.6   Δ=+0.00%  spread=1.37%   ← R1 winner
    ( 4, 4)   ms_med=0.6442  TFLOPS=1687.5   Δ=-0.60%  spread=0.57%   ← R30 universal
    ( 2, 4)   ms_med=0.6452  TFLOPS=1685.1   Δ=-0.74%  spread=0.42%
    ( 1, 4)   ms_med=0.6494  TFLOPS=1674.1   Δ=-1.41%  spread=1.36%
    ( 1, 2)   ms_med=0.6501  TFLOPS=1672.2   Δ=-1.52%  spread=0.88%   ← R11 B4 winner — LOSS on B32
    (32, 4)   ms_med=0.6504  TFLOPS=1671.4   Δ=-1.57%  spread=1.78%
    (16, 4)   ms_med=0.6510  TFLOPS=1670.1   Δ=-1.65%  spread=1.58%
    (32, 2)   ms_med=0.6519  TFLOPS=1667.7   Δ=-1.79%  spread=1.49%
    (16, 2)   ms_med=0.6530  TFLOPS=1665.0   Δ=-1.96%  spread=1.55%
    ( 8, 2)   ms_med=0.6550  TFLOPS=1659.7   Δ=-2.29%  spread=0.43%
    ( 2, 2)   ms_med=0.6579  TFLOPS=1652.5   Δ=-2.73%  spread=0.43%
    ( 8, 8)   ms_med=0.6654  TFLOPS=1633.9   Δ=-3.90%  spread=1.35%
    ( 4, 2)   ms_med=0.6692  TFLOPS=1624.7   Δ=-4.49%  spread=0.25%
```

(8, 4) is the unique top with 1697.6 TF; nearest contender (4, 4) is 0.60 pp slower (within R1's
+0.51..+2.89 pp R1 win margin — current binding maintains R1's relative ordering). The xcds=2
column is uniformly LOSS by 1.5-4.5 pp.

The R11 winner (1, 2) — which dominates the geometrically-similar B=4-M=2048 sibling by +0.65 pp
— LOSES by -1.52 pp on B=32. This is the cleanest B4-vs-B32 wave-step amortisation split in the
metric: B=4 has 484 tile-steps over 256 CUs ≈ 2 wave-steps (sparse, gm=1 + chiplet-pair locality
xcds=2 wins), B=32 has 3872 tile-steps ≈ 15 wave-steps (deep, gm-batching amortises and
cross-chiplet xcds=4 wins). Same R10-doc'd amortisation cutoff at ~3 wave-steps.

## Down-B32-M4096 wgrad (m_total=131072) results

```
3-seed summary (median ms / TFLOPS / Δ vs current (4, 4) / cell spread)
  ★ ( 4, 4)   ms_med=1.0870  TFLOPS=2000.4   Δ=+0.00%  spread=0.74%   ← R30 winner
    ( 8, 4)   ms_med=1.0895  TFLOPS=1995.7   Δ=-0.24%  spread=0.76%   (R1 tested as -0.06..-0.13 pp)
    ( 1, 4)   ms_med=1.0924  TFLOPS=1990.4   Δ=-0.50%  spread=0.65%
    ( 2, 4)   ms_med=1.0937  TFLOPS=1988.0   Δ=-0.62%  spread=1.10%
    (16, 4)   ms_med=1.0954  TFLOPS=1984.9   Δ=-0.78%  spread=0.58%
    (32, 4)   ms_med=1.0955  TFLOPS=1984.7   Δ=-0.79%  spread=0.54%
    (12, 4)   ms_med=1.0960  TFLOPS=1983.8   Δ=-0.84%  spread=0.57%   (R25 DSV3 winner — LOSS on gpt_oss-Down-B32-M4096)
    ( 1, 2)   ms_med=1.1093  TFLOPS=1960.0   Δ=-2.06%  spread=1.15%
    (16, 2)   ms_med=1.1098  TFLOPS=1959.3   Δ=-2.10%  spread=1.22%
    ( 8, 2)   ms_med=1.1155  TFLOPS=1949.1   Δ=-2.63%  spread=1.66%
    ( 4, 2)   ms_med=1.1266  TFLOPS=1929.9   Δ=-3.65%  spread=0.71%
    ( 2, 2)   ms_med=1.2300  TFLOPS=1924.2   Δ=-3.96%  spread=0.78%
```

(4, 4) is the unique top with 2000.4 TF; nearest contender (8, 4) is 0.24 pp slower (matches
R1's tie-loss finding). xcds=2 column LOSS by 2.0-4.0 pp. The R25 DSV3 winner (12, 4) LOSES by
0.84 pp — confirms the m_total > 65536 gate R25 ships under (R25 only fires on m_total==65536
and a∈{2048,7168} which excludes gpt_oss-Down by construction).

The deeper grid (~30 wave-steps) shifts the optimum from (8, 4) on M=2048 to (4, 4) on M=4096 —
matches R30's documented "wave-step amortisation bifurcates the optimum" pattern. (4, 4) wins
because the wider K-loop per tile-step amortises the smaller batch-factor more efficiently than
(8, 4) on the deeper grid.

## Sibling regression check

No production rule changed — the (gm, xcds) cells tested are NOT promoted. The R1 / R30 rules
remain as-is. No risk to:
- Down-B32-M2048 wgrad (R1 confirmed unique top)
- Down-B32-M4096 wgrad (R30 confirmed unique top)
- All other 22 cells (rule scope unchanged)
- DoD smoke FP8 grouped fwdbwd (R32 audit excludes by k != 2880)

## Implication for future rounds

R20 + R17 + R15 now establish the var-K + RCR dispatcher track is **fully exhausted** for the
gpt_oss B=4 + B=32 metric cells under the current FP8 binding. The remaining metric headroom
must come from kernel-source changes (R19 Track B):

1. **AGPR usage on var-K path** — re-verified untested in R47/R48 (those covered dense FP8).
   The var-K kernel uses 4 VGPR-class accumulators (cA, cB, cC, cD = `rt_fl<RBM, RBN, col_l,
   rt_16x16_s>`). On MI300 / MI355X the MFMA `_aft_a` variant accepts AGPR accumulators which
   wouldn't count against the 41-VGPR ceiling that R16 hit. If the compiler can be coaxed
   into using AGPRs, the freed-up VGPR space could absorb a deeper unroll or new prefetch.

2. **K-tail wave specialisation** for K%128==64 — currently every wave handles both the masked
   tail and the body. A two-class split could let body waves use a tighter inner loop.

3. **Cross-XCD affinity scheduling** for the small-grid wgrad (Down-B4-M2048 wgrad at 1.5
   wave-steps/CU). Currently all XCDs share the persistent grid; partitioning into
   XCD-affinity sub-grids might recover under-saturation tax.

R21 should pick (1) — lowest VGPR-pressure risk among the three.

## Files committed

- `analysis/_notes/round-20-fp8-vark-down-b32-wide-gm-xcds-FALSIFIED.md` (this note)
- `scripts/_probe_round_20_down_b32_m2048_wgrad.py`
- `scripts/_probe_round_20_down_b32_m4096_wgrad.py`

No production code changes.
