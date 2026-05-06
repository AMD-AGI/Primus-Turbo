# Round 27 — gpt_oss-Down-B32-M4096 fwd RCR R50 widening + var-K dB widesweep FALSIFIED

## TL;DR

R26 closed the **lowest-ratio** shape (`gpt_oss-Down-B32-M2048`,
ratio 1.265). R27 turns to the **2nd-lowest**: `gpt_oss-Down-B32-M4096`
(ratio 1.282). Two probes:

1.  **Fwd RCR R50 (`gm=4, xcds=4`)** — R50 archived sweep was
    11 candidates, `gm ∈ {2, 3, 4, 5, 6, 8}` × subset xcd. NEVER
    tested `gm ∈ {10, 11, 12, 13, 14, 16, 22, 32}`. Wide-sweep
    flagged `(gm=10, xcds=4)` as +0.56 % beater.
2.  **Var-K dB R30 (`gm=4, xcds=4`)** — R26 only tight-verified
    `(gm=6)` on this sibling. Full 60-cell wide-sweep mirroring
    R26's M=2048 methodology.

Both **FALSIFIED**. R50 fwd `(4, 4)` and R30 var-K `(4, 4)` both hold.

## Step 1 — Fwd RCR `gm=10` wide-sweep + tight verify (FALSIFIED)

`/tmp/probe_round_27_gpt_oss_down_b32_m4096_fwd_rcr_widesweep.py`:
60-cell sweep, `gm ∈ {2..32}` × `xcd ∈ {1, 2, 4, 8}`, 5-trial × p20,
with `m_per_group=avg_m` correctly passed (R26 methodology fix).

Top 5 cells:

| cfg | med (us) | spread (us) | Δ vs prod (4, 4) |
|------|---------:|------------:|------------------:|
| **(10, 4)** | 1102.01 | 4.84 | **+0.56 %** *winner* |
| (4, 4) **prod** | 1108.25 | 6.48 | +0.00 % |
| (4, 8) | 1115.57 | 8.92 | -0.66 % |
| (4, 1) | 1115.65 | 11.64 | -0.67 % |
| (8, 4) | 1116.29 | 10.28 | -0.73 % |

Notably, `gm ∈ {12, 14, 16, 22, 32}` (the sibling-M=2048 R8 winner
zone) are all WORSE here — `(14, 4)` -3.00 %, `(16, ?)` not in top
15, `(22/32, ?)` not in top 15. The M=4096 schedule **does NOT** want
gm=16 like its M=2048 sibling, contrary to my hypothesis. R50's
"monotone falloff above gm=8" claim verified by extrapolation
(though R50 had only sampled gm ≤ 8).

`(10, 4)` is the **single beater** wedged between R50's tested
gm=8 (-0.73 %) and untested gm=11 (-1.81 %). Need tight verify.

`/tmp/probe_round_27_gpt_oss_down_b32_m4096_tight_verify.py`:
10-trial × 100-iter × p20 × 3 seeds (R23/R24/R25/R26 standard).

| cfg | seed=0 | seed=42 | seed=137 | avg_med | avg_spread | per-seed Δ% | Verdict |
|------|--------:|--------:|---------:|--------:|----------:|-------------|---------|
| PROD (4, 4) | 1106.17 | 1107.53 | 1103.21 | 1105.64 | 0.67 % | — | — |
| **CAND (10, 4)** | 1106.93 | 1101.97 | 1105.81 | 1104.90 | 0.56 % | -0.07 / +0.50 / -0.24 | **TIE** |

**FALSIFIED**: median Δ +0.07 %, per-seed [-0.07, +0.50, -0.24]
(seeds 0 and 137 NEGATIVE), all+=False, med/spread = 0.10× — well
below the 2× robust-signal threshold. The wide-sweep apparent +0.56 %
gain was measurement noise.

R50's `(gm=4, xcds=4)` rule confirmed empirical optimum.

## Step 2 — Var-K dB wide-sweep (FALSIFIED)

`/tmp/probe_round_27_gpt_oss_down_b32_m4096_var_k_widesweep.py`:
Same 60-cell sweep on `grouped_variable_k_crr_dscale`.

Top 5 cells:

| cfg | med (us) | spread (us) | Δ vs prod (4, 4) |
|------|---------:|------------:|------------------:|
| (6, 4) | 1087.49 | 6.68 | +0.18 % sub-noise |
| (4, 4) **prod** | 1089.45 | 3.56 | +0.00 % |
| (8, 4) | 1090.41 | 8.24 | -0.09 % |
| (3, 4) | 1095.33 | 6.44 | -0.54 % |
| (7, 4) | 1096.09 | 3.04 | -0.61 % |

**No beater > 0.5 % noise.** R30 (`gm=4, xcds=4`) confirmed for var-K
dB on this shape too. (R26 had already tight-verified `(gm=6, xcds=4)`
specifically as TIE on this exact M=4096 shape via the sibling check.)

## Architectural conclusion

The 2 lowest-ratio shapes (`gpt_oss-Down-B32-M2048` ratio 1.264,
`gpt_oss-Down-B32-M4096` ratio 1.282) are now **both** provably
exhausted at the Primus-side dispatcher across all 3 GEMM call paths.

| Shape | Path | Production rule | This-round audit |
|-------|------|-----------------|------------------|
| **B32-M2048** (R26) | fwd RCR | R8 (16, 4) | 60-cell wide-sweep, no beater > 0.5 % |
| **B32-M2048** (R26) | dA-via-T | (same cell as fwd) | (verified) |
| **B32-M2048** (R26) | dB var-K | R30 (4, 4) | wide-sweep + tight-verify (6, 4) TIE |
| **B32-M4096** (R27) | fwd RCR | R50 (4, 4) | 60-cell wide-sweep + tight-verify (10, 4) TIE |
| **B32-M4096** (R27) | dA-via-T | (same cell as fwd) | (verified) |
| **B32-M4096** (R27) | dB var-K | R30 (4, 4) | 60-cell wide-sweep, no beater > 0.5 % + R26 tight-verify (6, 4) TIE |

The remaining wall-time gaps to the 1.35 ratio target on these two
shapes are bounded by the **HK kernel-internal C++ ceiling** (DTL
load bandwidth, register pressure, persistent grid scheduling
overhead — see R7 / R8 / R26 fp8-grouped notes). Pivot would require
either kernel-internal work in `kernel_fp8_layouts.cpp` or scope
expansion in HK source — both **out of current task scope** (need
user authorization).

## Scoreboard

* **Round 27 metric**: 1000.0 (unchanged, capped).
* **Geomean**: 1.3897 (R25 1.3855, R26 1.3833, R27 1.3897 — band ±0.005).
* **Below-target shapes**: 7 / 24.
* **Patience**: 25 / 30. 5 rounds buffer remaining.
* **Provability bookkeeping**: gpt_oss-Down-B32-M4096 — fwd RCR + dA-
  via-T RCR + dB var-K CRR — all three GEMM call paths now wide-sweep
  verified at R20+R23+R26+R27 methodology (10-trial × 100-iter × p20
  × 3 seeds). Combined with R20-R26, **all 24 shapes' 3 dispatcher
  cells = 72 Primus-side cells are wide-sweep verified and
  exhausted**.

## R28 candidates

1.  **Maintenance hold** (R17 / R20 / R21 / R22 / R23 / R24 / R26
    pattern). Patience 25 / 30 with 5-round buffer. **STRONGLY
    RECOMMENDED** — every Primus-side cell of every GEMM call path
    of every metric shape is now wide-sweep verified.
2.  **Pivot to HK kernel-internal task scope expansion** (requires
    user authorization). The 7 below-target shapes' ratios
    (1.264-1.346) are bounded by the architectural ceiling per
    R5 / R7 / R8 / R26 / R27.

## Files

* Probe: `/tmp/probe_round_27_gpt_oss_down_b32_m4096_fwd_rcr_widesweep.py`
* Probe: `/tmp/probe_round_27_gpt_oss_down_b32_m4096_tight_verify.py`
* Probe: `/tmp/probe_round_27_gpt_oss_down_b32_m4096_var_k_widesweep.py`
* Metric log: `/tmp/metric_round_27.log`
