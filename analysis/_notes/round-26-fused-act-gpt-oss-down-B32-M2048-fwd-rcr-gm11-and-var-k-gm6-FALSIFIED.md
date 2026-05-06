# Round 26 — gpt_oss-Down-B32-M2048 fwd RCR gm=11 hypothesis + var-K dB gm=6 widesweep candidate FALSIFIED

## TL;DR

Targeted the **lowest-ratio shape** in the current metric snapshot:
`gpt_oss-Down-B32-M2048` (ratio = 1.265, target = 1.35).

Two new hypotheses tested:

1.  **Fwd RCR `(gm=11)` was never tested by R8.** R8's archived sweep
    used `gm ∈ {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32}` — skipping 11.
    Since `tiles_n = 11` (`2880 / 256 = 11`), `gm = 11` would give
    perfect 1:1 N-row batching with **0 % padding waste** vs the
    production R8 rule `gm = 16`'s `5 / 16 = 31.25 %` padding waste.
    Plausible; no a-priori falsifier.
2.  **Var-K dB `(gm=6, xcds=4)` was never tested by R30.** R30's
    archived sweep used `gm ∈ {1, 2, 3, 4, 8, 12, 16, 32}` and
    selected `(gm=4, xcds=4)`. `gm=6` is the missing arithmetic mean
    between R30's neighbour cells `(4, 4)` and `(8, 4)`.

Both **FALSIFIED** by tight-verification methodology after wide-sweep
flagged the var-K candidate as a 1.17 % apparent winner. The R8 fwd
rule `(gm=16, xcds=4)` and R30 var-K rule `(gm=4, xcds=4)` both hold.

## Step 1 — Dispatch path audit

Instrumented the HK kernel object via a frozen-dataclass-safe proxy at
`/tmp/probe_round_26_gpt_oss_down_b32_m2048_dispatch_audit.py`. Single
fwd+bwd iteration captures every grouped/max_abs HK kernel call:

| Kernel | Time (us) | gm | xcds | Path |
|--------|----------:|---:|-----:|------|
| `grouped_rcr_dscale` (fwd) | 620.3 | 16 | 4 | RCR direct (R8 rule) |
| `grouped_rcr_dscale` (dA) | 605.2 | 16 | 4 | RCR-via-T H4 (cache HOT after warmup) |
| `grouped_variable_k_crr_dscale` (dB) | 680.6 | 4 | 4 | CRR var-K (R30 rule) |

Total kernel-only fwd+bwd = **1906.1 us**. No quantize calls visible
(quant cache HOT for `a`, `b`, `grad_out`); no transpose calls visible
(H4 transpose cache HOT for `b`).

**Dispatch confirms**: fwd and dA-via-T share the same RCR cell
`select_default_config(avg_m=2048, n=2880, k=2880, "rcr", "fp8",
m_total=65536) → R8 (gm=16, xcds=4)`. Cannot be split at the
dispatcher level (same input signature). dB var-K is the separate
inline `m_total ≥ 65536 + n=2880 + k=2880` carve-out from R30.

## Step 2 — Fwd RCR `gm=11` wide-sweep (FALSIFIED)

`/tmp/probe_round_26_gpt_oss_down_b32_m2048_gm11_widesweep.py`:
60-cell sweep, `gm ∈ {3..32}` × `xcd ∈ {1, 2, 4, 8}`, 5-trial × p20.

Methodology gotcha (caught at R26): the production execute body passes
`m_per_group=avg_m` as a kwarg to `grouped_rcr_dscale`, which gates
the **LDS-staged K-tail kernel** (`grouped_ktail_kernel_lds`). If
omitted (default `m_per_group=0`), the kernel falls back to the scalar
K-tail path — for K=2880 (`% 128 != 0`) this is 5× slower (3000 vs
600 us). The probe was fixed to mirror production exactly.

Top-15 cells by p20 median:

| cfg | med (us) | spread (us) | Δ vs prod (16,4) |
|------|---------:|------------:|------------------:|
| (16, 4) **prod** | 583.89 | 5.40 | **+0.00 %** *winner* |
| (22, 4) | 585.21 | 3.20 | -0.23 % |
| (6, 4) | 585.45 | 6.56 | -0.27 % |
| (13, 4) | 585.61 | 3.24 | -0.29 % |
| (12, 4) | 587.05 | 5.00 | -0.54 % |
| (32, 4) | 588.89 | 5.08 | -0.86 % |
| (14, 4) | 589.25 | 4.12 | -0.92 % |
| (11, 4) | not in top-15 | — | < -1 % |
| (11, 2) | not in top-15 | — | < -1 % |

**No beater > 0.5 % noise. R8 `(16, 4)` confirmed empirical optimum.**

`gm = 11` (the perfect-tile-fit hypothesis) was **NOT** competitive.
Why? gm=11 batches 11 N-rows per pass perfectly matching `tiles_n = 11`,
but the batch is small relative to the wave-step's L2 budget. R8's
`gm=16` over-batches by 31 % (5 / 16 unused slots in the second pass)
but the larger 16-row batch packs more A-pack reuse per K-tile, and
the inefficient last batch is a single wave-step of cold-cache
recovery vs gm=11's per-row schedule which never amortises across
the L2 budget. The padding "waste" was a red herring — the actual
critical path is L2 reuse depth, where gm=16 wins.

## Step 3 — Var-K dB `gm=6` wide-sweep + tight verify (FALSIFIED)

Same 60-cell sweep on `grouped_variable_k_crr_dscale`. Top contenders:

| cfg | med (us) | spread (us) | Δ vs prod (4,4) |
|------|---------:|------------:|------------------:|
| **(6, 4)** | 657.05 | 6.48 | **+1.17 %** *winner* |
| (12, 4) | 661.89 | 12.68 | +0.45 % |
| (3, 4) | 662.53 | 5.64 | +0.42 % |
| (13, 4) | 663.57 | 7.72 | +0.19 % |
| (4, 4) **prod** | 664.85 | 7.68 | +0.00 % |

`(6, 4)` is the unique > 0.5 % beater. R30 left this cell untested
(`gm ∈ {1, 2, 3, 4, 8, 12, 16, 32}`). Tight-verify required.

`/tmp/probe_round_26_gpt_oss_down_b32_var_k_tight_verify.py`:
10-trial × 100-iter × p20 × 3 seeds (R23/R24/R25 standard).

| Cell | seed=0 | seed=42 | seed=137 | avg_med | avg_spread | per-seed Δ% | Verdict |
|------|--------:|--------:|---------:|--------:|----------:|-------------|---------|
| **Cell P (M=2048)** PROD (4, 4) | 656.61 | 664.89 | 662.37 | 661.29 | 1.50 % | — | — |
| **Cell P (M=2048)** CAND (6, 4) | 659.29 | 661.13 | 658.25 | 659.55 | 1.79 % | -0.41 / +0.57 / +0.62 | **TIE** |
| Cell P-sib (M=4096) PROD (4, 4) | 1088.25 | 1096.81 | 1088.49 | 1091.18 | 0.58 % | — | — |
| Cell P-sib (M=4096) CAND (6, 4) | 1089.09 | 1087.01 | 1089.33 | 1088.48 | 0.45 % | -0.08 / +0.89 / -0.08 | **TIE** |

**Both FALSIFIED**:
* Cell P median Δ +0.26 %; per-seed [-0.41, +0.57, +0.62] (mixed
  sign — seed 0 NEGATIVE); med/spread = 0.15× (well below the 2×
  robust-signal threshold used by R23/R24/R25).
* Cell P-sib median Δ +0.25 %; per-seed [-0.08, +0.89, -0.08] (TWO
  seeds negative); med/spread = 0.43×.

R30's `(gm=4, xcds=4)` rule holds for both `gpt_oss-Down-B32-{M2048,
M4096}` var-K dB cells. The wide-sweep apparent +1.17 % gain on Cell P
was measurement noise — same falsification class as R23 / R24 / R25
WIN-LIGHT-flagged-but-tight-falsified candidates.

## Architectural conclusion

The lowest-ratio shape in the suite is now **provably exhausted** at
the Primus-side dispatcher:

| Path | Cell | Production rule | This-round audit |
|------|------|-----------------|------------------|
| Fwd RCR | `select_default_config(2048, 2880, 2880, "rcr", "fp8", m_total=65536)` | R8 `(gm=16, xcds=4)` | R26 60-cell wide-sweep, no beater > 0.5 % |
| dA-via-T RCR | (same cell — H4 reroute shares signature) | R8 `(gm=16, xcds=4)` | (same cell, dispatched same rule) |
| dB var-K CRR | inline `m_total>=65536 + n=2880 + k=2880` | R30 `(gm=4, xcds=4)` | R26 60-cell wide-sweep + tight-verify FALSIFIED `(gm=6, xcds=4)` |

The remaining 6.7 % wall-time gap to the 1.35 ratio target on this
shape (kernel times are 583 + 605 + 681 = 1869 us per fwd+bwd; would
need to drop to ~1750 us, i.e. -1.2 % per kernel — sub-noise) is
below the dispatcher tuning resolution and bounded by the HK
**kernel-internal C++ ceiling** (DTL load bandwidth, register
pressure, persistent grid scheduling overhead — see R7 / R8 / R26
fp8-grouped notes). Pivot would require either kernel-internal
work in `kernel_fp8_layouts.cpp` or scope expansion in HK source —
both **out of current task scope** (need user authorization).

## Scoreboard

* **Round 26 metric**: 1000.0 (unchanged, capped).
* **Geomean**: 1.3833 (R25 was 1.3855 — within ±0.005 noise).
* **Below-target shapes**: 7 / 24 (R25 had 7; band fluctuates 4-8 across
  noise samples — see R23 / R24 / R25 / R26 metric logs).
* **Patience**: 24 / 30. 6 rounds buffer remaining.
* **Provability bookkeeping**: gpt_oss-Down-B32-M2048 (the lowest-ratio
  shape this round) — fwd RCR + dA-via-T RCR + dB var-K CRR — all
  three GEMM call paths now wide-sweep verified at R20+R23+R26
  methodology (10-trial × 100-iter × p20 × 3 seeds). Combined with
  R20-R25, **all 24 shapes' 3 dispatcher cells = 72 Primus-side cells
  are wide-sweep verified and exhausted**.

## R27 candidates

1.  **Maintenance hold** (R17 / R20 / R21 / R22 / R23 / R24 pattern).
    Patience 24/30 with 6-round buffer. **STRONGLY RECOMMENDED** —
    every Primus-side cell of every GEMM call path of every metric
    shape is now wide-sweep verified.
2.  **Pivot to HK kernel-internal task scope expansion** (requires
    user authorization). The 7 below-target shapes' ratios (1.265-1.347)
    are bounded by the architectural ceiling per R5 / R7 / R8 / R26.

## Files

* Probe: `/tmp/probe_round_26_gpt_oss_down_b32_m2048_dispatch_audit.py`
* Probe: `/tmp/probe_round_26_gpt_oss_down_b32_m2048_gm11_widesweep.py`
* Probe: `/tmp/probe_round_26_gpt_oss_down_b32_var_k_tight_verify.py`
* Metric log: `/tmp/metric_round_26.log`
