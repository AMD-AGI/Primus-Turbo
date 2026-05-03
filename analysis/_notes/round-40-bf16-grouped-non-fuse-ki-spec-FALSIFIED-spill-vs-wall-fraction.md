# Round-40 BF16 grouped — non-fuse KI specialization FALSIFIED

## Lever (R39 note proposal — revisited and tested)

Add KI specialization cases to the **non-fuse** dispatch switch in
`dispatch_grouped<L>` for ki values currently falling through to
`KI_HINT=0` (#pragma unroll 2 dynamic):

| ki  | metric path                               | layout | shapes |
|-----|-------------------------------------------|--------|--------|
| 24  | Qwen3-Down forward (K=1536)               | RCR    | 4 (w=1) |
| 32  | DSV3-Down forward (K=2048)                | RCR    | 4 (w=1) |
| 48  | Qwen3-GateUP dA RRR (K=N_fwd=3072)        | RRR    | 4 (w=1) |
| 88  | gpt_oss-GateUP dA H4 RCR (K=N_fwd=5760)   | RCR    | 4 (w=3) |

R39's falsification note explicitly recommended this lever as low-risk
("non-fuse template = no K-tail epilog block = no spill"). The R39
hypothesis was **partially wrong**: spill DOES appear on short-K cases.

## Build resource report (R40 v1 — all 4 added)

| KI×Layout         | VGPRs | VGPR Spill | Occupancy |
|-------------------|-------|------------|-----------|
| 24 RCR            | 256   | **20**     | 2         |
| 24 RRR            | 256   | **20**     | 2         |
| 24 CRR            | 256   | 16         | 2         |
| 32 RCR            | 256   | **20**     | 2         |
| 32 RRR            | 256   | **20**     | 2         |
| 32 CRR            | 256   | 16         | 2         |
| 48 RCR            | 256   | **0**      | 2         |
| 48 RRR            | 256   | **20**     | 2         |
| 48 CRR            | 256   | 16         | 2         |
| 88 RCR            | 256   | **0**      | 2         |
| 88 RRR            | 256   | **0**      | 2         |
| 88 CRR            | 256   | **0**      | 2         |
| (existing) 56 RCR | 256   | 14         | 2         |
| (existing) 64 RCR | 256   | 0          | 2         |
| (existing) 112 RCR| 256   | 19         | 2         |

The R39 note's claim "non-fuse compiles spill-clean" was based only on
KI=64 RCR (0 spill). Other existing cases (56/112/128/172) DO spill
12-29 VGPRs. The new short-K cases (24/32) spill ~20 — which is
already in the existing-case range. The unroll's win-vs-spill balance
depends on K-loop length:

* Short K (ki ≤ 48): few `main_loop_iter` calls → small unroll savings
  → spill cost dominates → NET REGRESSION.
* Long K (ki ≥ 56): many iters → large unroll savings → savings cover
  spill → NET WIN (proven by existing cases 56/112).
* Middle (ki=88): clean compile (0 spill all layouts) but the ki=88
  shapes are all on dA backward where the dA wall-fraction is too
  small for the unroll savings to be metric-detectable.

## R40 v1 — all 4 cases (24, 32, 48, 88): FALSIFIED

```
                              R40 baseline   R40 v1     Δ
score                              879           871      -8
gpt_oss family geomean             1.0838        1.0946   +0.011  (ki=88 helped)
DeepSeek-V3   family geomean       1.1259        1.0953   -0.031  (ki=32 RCR spill hurt)
Qwen3-235B-A22B family geomean     1.1142        1.0601   -0.054  (ki=24 RCR + ki=48 RRR hurt)
```

Per-shape damage on Qwen3 family:
```
  Qwen3-Down-B16-M2048   1.119 → 1.031   -8.8pp  (ki=24 RCR fwd spill)
  Qwen3-Down-B16-M4096   1.105 → 1.038   -6.7pp
  Qwen3-Down-B32-M2048   1.108 → 1.033   -7.5pp
  Qwen3-Down-B32-M4096   1.115 → 1.027   -8.8pp
```

DSV3-Down forward also regressed:
```
  DSV3-Down-B16-M2048    1.120 → 1.062   -5.8pp  (ki=32 RCR fwd spill)
  DSV3-Down-B16-M4096    1.113 → 1.054   -5.9pp
  DSV3-Down-B32-M2048    1.111 → 1.057   -5.4pp
  DSV3-Down-B32-M4096    1.110 → 1.042   -6.8pp
```

gpt_oss-GateUP DID improve marginally (the only positive piece):
```
  gpt_oss-GateUP-B4-M2048    1.101 → 1.114   +1.3pp  (ki=88 dA H4 RCR)
  gpt_oss-GateUP-B4-M4096    1.100 → 1.106   +0.6pp
  gpt_oss-GateUP-B32-M2048   1.044 → 1.054   +1.0pp
  gpt_oss-GateUP-B32-M4096   1.086 → 1.085   ~flat
```

## R40 v2 — keep only ki=88 (clean compile): SUB-NOISE

Reverted ki=24/32/48; kept ki=88 alone. 4-run mean of v2 vs 1-run
baseline (881.5 ± 1.7 spread vs 879):

```
v2 runs    882, 878, 879, 879   →   mean 879.5,  spread ±1.7
baseline                                879
                                       Δ = +0.5  (sub-noise)
```

ki=88 RCR is clean (0 spill) but the dA H4 RCR wall fraction is too
small for full-unroll savings on 21-iter loops to register. Dispatcher
math: 4 weight-3 shapes × ~5 % wall improvement (estimated upper bound
from a clean-compile unroll) × 12/40 weight share = +1.5 weighted
progress = +1.5 score expected; reality matches noise. Mechanism is
real but impact is below detection threshold.

## Implications for future rounds

1. **KI specialization for short-K (ki ≤ 48) BF16 grouped is exhausted.**
   Spill (16-20 VGPRs on RCR/RRR) costs more than unroll saves on
   short main loops. Don't retry without first re-shaping the
   `main_loop_iter` body to reduce live-state.
2. **KI=88 for long-K dA shapes is mechanism-correct but sub-noise.**
   The dA wall fraction in the metric (fwd + bwd combined) is too
   small to see a few-% kernel improvement. Likely also true for
   other unmapped long-K dA values — this surface is exhausted at the
   current metric resolution.
3. **The BF16 KI dispatch surface is closed.** R39 (fuse KI=44) +
   R40 (non-fuse KI=24/32/48/88) have systematically tested every
   ki value hit by the metric. Result: existing 56/64/112 stay,
   nothing new wins.
4. **Pivot for R41+**:
   - Option C from R38 — rocprofv3 marker-bracketed K-tail
     wall-fraction on gpt_oss B=32 (still pending, free diagnostic).
   - dB var-K kernel investigation — `grouped_var_k_kernel` (line 4511)
     is currently 256 VGPRs / 0 spill / 2 occ, untouched for many
     rounds. Different code path from grouped_kernel; might have
     under-exploited optimization surface.
   - Per-shape wall decomposition (fwd vs dA vs dB) using nsys/rocprof
     on a single shape to identify whether bwd or fwd contributes the
     residual gap to 1.25.

## Revert state

`launch_one_grouped` switch back to original 11 cases (56, 64, 112,
128, 172, 224, 256, 296, 448, 462, 832). `INSTANTIATE_K_GRP` macro
reverted to original 11 instantiations. `git diff` of `kernel_bf16_dynamic.cpp`
empty after revert. Rebuild verified score=883 (within ±5 of baseline 879;
not committed HipKittens-side).
