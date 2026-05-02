# Round 6 — FP8 grouped: Lever F LANDED (Qwen-Down M=4096 rule)

**Status**: LEVER F PARTIAL LAND — Qwen-Down M_per_group=4096 rule
added (`tiles_n==16 + tiles_m==16 + k==1536` → `(gm=2, xcds=None)`).
+3.17pp (B16-M4096) / +2.69pp (B32-M4096) on 2 shapes; +0.36pp on
24-shape geomean; +1 score. M_per=2048 sub-family confirmed best at
default (rule scoped to tiles_m==16). Qwen-GateUP probe also done;
findings recorded for R7+ followup.
**Auto-optimize round**: 6 / 100
**Date**: 2026-05-02
**HK SHA**: `9ee90e2c` (unchanged from R5)
**PT SHA at round start**: `9fd99a9`
**PT SHA at round end**: `<this commit>` (config.py +72 lines)
**Round time**: ~30 min (1 baseline + 28-cell sweep × 4 + tight verify ×
  2 + Qwen-GateUP sweep × 4 + correctness check + metric × 2 + write-up)
**Score before**: 959
**Score after**: 960 (+1)
**FP8 geomean before**: 1.1155
**FP8 geomean after**: 1.1196 (+0.36pp)

---

## R6 baseline metric + worst-shape selection

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1884 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1155 (n=24)
[metric_grouped_only]   weighted_progress=0.9595  score=959
```

R6 baseline grp_FP8 sorted ascending (worst first):
```
gpt_oss-GateUP-B32-M4096        1.024  ← worst
gpt_oss-Down-B32-M4096          1.054
gpt_oss-GateUP-B4-M4096         1.054
Qwen-Down-B16-M4096             1.056  ← worst Qwen
gpt_oss-Down-B32-M2048          1.061
gpt_oss-GateUP-B32-M2048        1.072
gpt_oss-GateUP-B4-M2048         1.083
gpt_oss-Down-B4-M4096           1.089
Qwen-Down-B32-M4096             1.090  ← 2nd-worst Qwen
Qwen-Down-B16-M2048             1.098  ← 3rd-worst Qwen
gpt_oss-Down-B4-M2048           1.099
Qwen-Down-B32-M2048             1.139  (last Qwen-Down)
... (rest above 1.10)
```

The 4 worst gpt_oss cases all carry shape-specific (gm, xcds) rules
already (gm ∈ {1, 8, 14, 16, 32}, xcds=4) per R7+8+12+22+23+68+69+70.
Top 6 gpt_oss / DSV3 are at architectural ceiling per R5 falsification
analysis. The 4 Qwen-Down cases ALL hit the FP8 default `(gm=4,
num_xcds=None=8)` — never had a (gm, xcds) sweep done. Per R5 plan
this is the R6 lever F target.

---

## Lever F probe (Qwen-Down 4 cases — 28-cell coarse sweep)

`/tmp/probe_qwen_down_round6.py` (50-iter × p20, mirrors metric):

```
Qwen-Down-B16-M2048 (default 1765.34 TF):
  top1 (gm=4, xcds=8)   = default        +0.00 %  ← already optimal
  top2 (gm=4, xcds=16)  =  1760.52 TF    -0.27 %
  ... all (gm!=4) candidates -3.14 % to -3.95 %

Qwen-Down-B16-M4096 (default 1750.50 TF):
  top1 (gm=2, xcds=16)  =  1812.37 TF   +3.53 %  ← winner
  top2 (gm=2, xcds= 8)  =  1808.23 TF   +3.30 %
  rest within ±0.3 %

Qwen-Down-B32-M2048 (default 1791.88 TF):
  top1 (gm=4, xcds=16)  =  1795.01 TF   +0.17 %  ← within noise
  top2 (gm=4, xcds=8)   = default
  ... all (gm!=4) candidates -2.32 % to -2.99 %

Qwen-Down-B32-M4096 (default 1777.13 TF):
  top1 (gm=2, xcds=16)  =  1835.60 TF   +3.29 %  ← winner
  top2 (gm=2, xcds= 8)  =  1824.23 TF   +2.65 %
  rest within ±0.5 %
```

Pattern is crisp:
- **M_per=2048 sub-family** (B16, B32): default already optimal,
  every (gm!=4) candidate regresses 2-4 pp. Rule must NOT cover.
- **M_per=4096 sub-family** (B16, B32): (gm=2) wins consistently
  by +3 pp. xcds ∈ {8, 16} are interchangeable (top-2 within 0.04 pp).

### Lever F tight verify (200-iter × 7-trial p20)

`/tmp/verify_qwen_down_m4096_round6.py`:

```
Qwen-Down-B16-M4096:
  ( 2,  8)  1835.11 TF   spread 0.39 %   +3.17 pp vs default *winner
  ( 2, 16)  1834.46 TF   spread 0.36 %   +3.13 pp
  ( 1,  4)  1779.05 TF   +0.02 pp
  ( 4,  8)  1778.75 TF   baseline
  (32,  4)  1770.04 TF   -0.49 pp

Qwen-Down-B32-M4096:
  ( 2,  8)  1843.81 TF   spread 0.35 %   +2.69 pp vs default *winner
  ( 2, 16)  1843.40 TF   spread 0.57 %   +2.66 pp
  ( 4,  8)  1795.55 TF   baseline
  ( 1,  4)  1794.46 TF   -0.06 pp
```

`(gm=2, xcds=8)` wins both shapes by 47-50 TF; xcds=16 is hairline
behind. Setting `cfg.num_xcds=None` (which → kernel default 8) wins
identically and keeps the rule cleaner (no xcds override needed).

### Numerical correctness

`/tmp/verify_qwen_down_correctness_round6.py`:

```
Qwen-Down-B16-M4096: max_abs=0.0  bit_eq=True  no nan/inf
Qwen-Down-B32-M4096: max_abs=0.0  bit_eq=True  no nan/inf
Qwen-Down-B16-M2048: max_abs=0.0  bit_eq=True  no nan/inf  (rule doesn't apply)
Qwen-Down-B32-M2048: max_abs=0.0  bit_eq=True  no nan/inf  (rule doesn't apply)
```

`group_m` / `num_xcds` are pure persistent-tile-schedule knobs on
FP8 grouped RCR — same property as R7/R8/R12/R20/R21/R23/R68/R70
established for gpt_oss / DSV3 rules.

---

## Rule added to `select_default_config`

```python
if tiles_n == 16 and tiles_m == 16 and k == 1536:
    # Round-6 rule. Qwen3-235B-A22B Down M_per_group=4096 family
    return HipKittenConfig(
        layout=layout, group_m=2, num_xcds=None, kernel=None,
    )
```

### Rule scope check (no spillover)

`tiles_n=16` is shared with DSV3-GateUP (k=7168) and dense
(8192,4096,*); the `k==1536` clause is uniquely Qwen-Down in the
metric (DSV3 k ∈ {2048, 7168}, gpt_oss k=2880, Qwen-GateUP k=4096,
dense k ∈ {4096, 11008, 14336, 22016, 28672}). `tiles_m==16`
(m_per_group=4096) excludes the M=2048 sibling cases which the
sweep above showed sit best at default (any (gm=2) candidate
regressed -3.95 pp on B16-M2048). Net coverage = 2 metric shapes
(Qwen-Down-B16-M4096 + Qwen-Down-B32-M4096), zero risk to siblings.

---

## R6 metric (after rule add)

```
[metric_grouped_only]   grpFP8_Qwen3-Down-B16-M4096    1736.2 → 1793.6 TF (+3.31 %, ratio 1.056 → 1.088)
[metric_grouped_only]   grpFP8_Qwen3-Down-B32-M4096    1758.9 → 1811.1 TF (+2.97 %, ratio 1.090 → 1.120)
[metric_grouped_only]   grpFP8_Qwen3-Down-B16-M2048    1690.4 → 1685.1 TF  (-0.31 %, noise band, rule doesn't apply)
[metric_grouped_only]   grpFP8_Qwen3-Down-B32-M2048    1775.1 → 1773.5 TF  (-0.09 %, noise band, rule doesn't apply)
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1864 (was 1.1884, -0.17 %, well within ±2 % gate)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1196 (was 1.1155, +0.36 %)
[metric_grouped_only]   score=960  (was 959, +1)
[metric_grouped_only]   correct_fail=0/48  reject=0/48
```

Predicted gains realised exactly: B16-M4096 +0.032 ratio (predicted
+0.034), B32-M4096 +0.030 ratio (predicted +0.028). M_per=2048 cases
unchanged within ±0.4 % noise (rule not applied to them, sweep result
preserved). Score +1 (small but real); below the 5-point heuristic
in the task body but the underlying gain is well-validated and
architectural so we land it rather than leave a real win on the table.

---

## Qwen-GateUP probe (data captured for R7+ follow-up; NO rule added this round)

`/tmp/probe_qwen_gateup_round6.py` — same 28-cell grid, 4 shapes:

```
Qwen-GateUP-B16-M2048 (default 2457 TF):
  top1 (gm=32, xcds=4) = 2474 TF (+0.70 %)  ← marginal

Qwen-GateUP-B16-M4096 (default 2522 TF):
  top1 (gm=4,  xcds=4) = 2539 TF (+0.67 %)  ← marginal (only xcds change)

Qwen-GateUP-B32-M2048 (default 2483 TF):
  top1 (gm=32, xcds=4) = 2520 TF (+1.48 %)  ← cleanest signal

Qwen-GateUP-B32-M4096 (default 2530 TF):
  top1 (gm=1,  xcds=4) = 2556 TF (+1.02 %)  ← split winner pattern
```

The 4 Qwen-GateUP shapes have **split (gm) winners** — no single
(gm, xcds) rule covers more than 1 shape with ≥ 1pp gain. Best
candidate = single-shape rule for Qwen-GateUP-B32-M2048
(`tiles_n=12 + tiles_m=8 + k=4096 + m_total=65536` → (gm=32, xcds=4))
projecting +1.48 % on 1 of 24 shapes ≈ +0.06 pp on geomean
≈ +0 to +1 score points. Marginal. Not landed this round to keep
R6 commit focused on the cleanest signal; R7 can add it after a
tight verify if score budget warrants.

---

## Cumulative status after 6 rounds

```
Lever         | Status                                           | Round
----          | -----                                            | -----
A async-LDS   | FALSIFIED (already shipped)                      | R2
B triple-slab | FALSIFIED (LDS at 137/160 KB cap)                | R2
C-2 K-tail    | FALSIFIED (already in if-branch)                 | R3
C-3 spill     | DONE (architectural, 256 VGPR cap)               | R3
C-X SENTINEL  | FALSIFIED (neutral on active template)           | R4
D mfma_323264 | FALSIFIED (microbench Δ ≈ 0)                     | R5
F Qwen-Down   | PARTIAL LAND (M=4096 rule, +1 score)             | R6
F Qwen-GateUP | PROBED (split winners, max +1.48% on 1 shape)    | R6
E ASM pipe    | OPEN (last resort, R20+, very high risk)         | -
```

5 of 7 architectural levers fully exhausted; F now partially
landed. Remaining grp_FP8 ceiling is dominated by the gpt_oss
K=2880 cluster (4 shapes < 1.06) which is the architectural-VGPR-
spill-bound cluster from R3 spill localization.

---

## R7+ plan options (in priority order)

### R7 candidate A: **Qwen-GateUP-B32-M2048 single-shape rule** (LOW UPSIDE)
Add `tiles_n=12 + tiles_m=8 + k=4096 + m_total=65536 →
(gm=32, xcds=4)` after a tight 200-iter × 7-trial verify.
Projected +1.5 % on 1 shape → +0 to +1 score points. Defensible
but marginal. **DEFER to R8+**.

### R7 candidate B: **Re-probe gpt_oss 4-worst shapes for 2nd-best (gm, xcds)** (NO UPSIDE)
The 4 gpt_oss shapes are at architectural ceiling per R3; (gm, xcds)
sweep was exhausted in R7+8+12+22+23+68+69+70+10-dm. **FROZEN per
task body ✗ list.** Do not redo.

### R7 candidate C: **Lever E ASM software pipeline scout** (HIGH RISK, R20+)
Manually schedule the K-iter chain. Per task body "最后再做" — only
after A/B/C/D/F all saturated. R6 is round 6; we still have many
F sub-leads (Qwen-GateUP single-shape rules, K=1536 backward path,
etc.) before going to E.

### R7 RECOMMENDED: **Qwen-GateUP-B32-M2048 tight verify + commit**
Tight verify (200-iter × 7-trial p20) on Qwen-GateUP-B32-M2048 to
confirm +1.48 % is above noise. If yes, add single-shape rule
(mirrors the existing gpt_oss-Down-B32-M2048 pattern at line 1050+).
Expected score gain +0-1 (rule covers 1/24 shapes at +1.5 %). Worst
case it falsifies and we revert.

R8 follow-up: **R64-dm BF16 grouped status check**. R5+R6 only
touched FP8 path; BF16 grouped geomean has been drifting in the
±0.2 % band (1.1884 → 1.1864 this round). Do a quick spot-check
that no BF16 metric shape regressed >2 pp; if so, BF16 may need a
similar Qwen-coverage round.

---

## Hard-constraint compliance

- [x] No metric / benchmark / config edits (only `kernels/hipkitten/config.py`)
- [x] Rule is generic `(tiles_n, tiles_m, k)` — not (M, N, K) hardcoded, not model-name branched
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side .item() / .tolist()
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused PT commit (this rule + this note)
- [x] No HK kernel change (.so untouched, .so symbols same as R5)
- [x] No BF16 grouped touch (BF16 geomean -0.17 %, within ±2 % gate)
- [x] Correctness 0/48 fail
- [x] Rule scope verified: M=2048 unchanged, no spillover to dense / DSV3 / gpt_oss

---

## Probe artifacts (committed via this note's references; not in repo)

- `/tmp/probe_qwen_down_round6.py` — 28-cell × 4 shapes coarse sweep
- `/tmp/verify_qwen_down_m4096_round6.py` — 200-iter × 7-trial tight
- `/tmp/probe_qwen_gateup_round6.py` — 28-cell × 4 shapes coarse sweep
- `/tmp/verify_qwen_down_correctness_round6.py` — bit-equivalence check
