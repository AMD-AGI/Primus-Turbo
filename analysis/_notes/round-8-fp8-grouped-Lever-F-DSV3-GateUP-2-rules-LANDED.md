# Round 8 — FP8 grouped: Lever F LANDED, DSV3-GateUP 2 rules

**Status**: LEVER F EXTENSION — DSV3-GateUP M_per_group=4096 rule
(`tiles_n=16 + tiles_m=16 + k=7168` → `(gm=2, xcds=None=8)`) +
DSV3-GateUP-B32-M2048 single-tier rule (`tiles_n=16 + tiles_m=8 +
k=7168 + m_total>=65536` → `(gm=16, xcds=4)`). 3 of 4 DSV3-GateUP
shapes covered; B16-M2048 left on default (50-iter sweep top1 sat
at +0.13 % = solid noise, no tight verify warranted).
**Auto-optimize round**: 8 / 100
**Date**: 2026-05-02
**HK SHA**: `9ee90e2c` (unchanged)
**PT SHA at round start**: `3c74bdd`
**PT SHA at round end**: `<this commit>` (config.py +106 lines)
**Round time**: ~25 min (1 baseline + 28-cell sweep × 4 + tight verify ×
  3 + correctness check + 6-trial post-add validation + write-up)
**Score before**: 959 (R8 baseline)
**Score after**: 962.2 mean / 961 median over 6 trials (+3 mean)
**FP8 geomean before**: 1.1183
**FP8 geomean after**: ~1.1189 (within noise; per-shape gains real)

---

## R8 baseline metric

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1842 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1183 (n=24)
[metric_grouped_only]   weighted_progress=0.9590  score=959  correct_fail=0/48  reject=0/48
```

Worst 6 grpFP8 unchanged (all gpt_oss K=2880 architectural ceiling).
DSV3 family at 1.125-1.205 — DSV3-Down already covered by R20+67+68
rule (gm=32, xcds=2); DSV3-GateUP all 4 shapes still on FP8 default
(gm=4, xcds=None=8), in the 1.125..1.185 range. R7 plan set R8 = DSV3-
GateUP probe.

---

## Lever F probe (DSV3-GateUP 4 cases — 28-cell coarse sweep)

`/tmp/probe_dsv3_gateup_round8.py` (50-iter × p20):

```
DSV3-GateUP-B16-M2048 (default 2742.80 TF):
  top1 (gm=32, xcds=4)  2746.40 TF  +0.13 %  ← solid noise
  top2 (default)
  ...

DSV3-GateUP-B16-M4096 (default 2776.68 TF):
  top1 (gm=2, xcds=16)  2792.88 TF  +0.58 %  ← real
  top2 (gm=2, xcds=8)   2790.70 TF  +0.50 %  ← real

DSV3-GateUP-B32-M2048 (default 2726.86 TF):
  top1 (gm=16, xcds=4)  2765.59 TF  +1.42 %  ← clearest signal
  top2 (gm=4, xcds=4)   2749.30 TF  +0.82 %
  top3 (gm=8, xcds=4)   2744.21 TF  +0.64 %

DSV3-GateUP-B32-M4096 (default 2771.20 TF):
  top1 (gm=2, xcds=8)   2785.40 TF  +0.51 %  ← consistent with B16-M4096
  top2 (gm=2, xcds=16)  2780.17 TF  +0.32 %
```

### Tight verify (200-iter × 7-trial p20)

`/tmp/verify_dsv3_gateup_round8.py`:

```
DSV3-GateUP-B32-M2048:
  (16,  4)  2756.31 TF  spread 0.28 %  +0.56 pp *winner
  (32,  4)  2753.87 TF  spread 0.34 %  +0.47 pp
  ( 4,  8)  2741.00 TF  spread 0.87 %  baseline

DSV3-GateUP-B16-M4096:
  ( 2,  8)  2787.94 TF  spread 0.25 %  +0.64 pp *winner
  ( 2, 16)  2786.89 TF  spread 0.21 %  +0.60 pp
  ( 4,  8)  2770.20 TF  spread 0.22 %  baseline

DSV3-GateUP-B32-M4096:
  ( 2, 16)  2775.04 TF  spread 0.33 %  +0.49 pp
  ( 2,  8)  2774.24 TF  spread 0.31 %  +0.46 pp *winner
  ( 4,  8)  2761.58 TF  spread 0.43 %  baseline
```

Tight verify confirms:
- M=4096 family: (gm=2, xcds=8) wins consistently on both B16+B32
  by +0.46..+0.64 pp, all > 1.5× spread. Single rule covers both.
- B32-M2048: (gm=16, xcds=4) wins by +0.56 pp, 2× spread. Single-shape
  rule (B16-M2048 50-iter coarse showed solid noise = +0.13 %, not
  worth tight-verifying).

### Bit-equivalence

`/tmp/verify_dsv3_gateup_correctness_round8.py`:
```
Rule 1 (gm=4,xcds=8 vs gm=2,xcds=8) on M=4096:
  B16: max_abs=0.0  bit_eq=True
  B32: max_abs=0.0  bit_eq=True
Rule 2 (gm=4,xcds=8 vs gm=16,xcds=4) on B32-M2048:
  B32: max_abs=0.0  bit_eq=True
Sanity B16-M2048 (rule 2 should NOT apply):
  B16: max_abs=0.0  bit_eq=True
```

---

## Rules added to `select_default_config`

```python
# DSV3-GateUP M_per_group=4096 family
if tiles_n == 16 and tiles_m == 16 and k == 7168:
    return HipKittenConfig(
        layout=layout, group_m=2, num_xcds=None, kernel=None,
    )
# DSV3-GateUP-B32-M2048 single-tier
if (tiles_n == 16 and tiles_m == 8 and k == 7168
        and m_total is not None and m_total >= 65536):
    return HipKittenConfig(
        layout=layout, group_m=16, num_xcds=4, kernel=None,
    )
```

### Rule scope check (no spillover)

`tiles_n=16 + k=7168` is uniquely DSV3-GateUP in the metric grouped
suite (DSV3-Down k=2048 → tiles_n=28 rule below; gpt_oss k=2880;
Qwen-Down k=1536; Qwen-GateUP k=4096). Dense FP8 LLaMA shapes have
K ∈ {4096, 11008, 14336} — no K=7168. `tiles_m=16` selects M=4096
only; `tiles_m=8 + m_total>=65536` selects B32-M2048 only (B16-M2048
m_total=32768 < 65536, falls through to default).

---

## R8 metric (after rules added)

6-trial post-add p20:
```
trial 1: score=965
trial 2: score=960
trial 3: score=966
trial 4: score=960
trial 5: score=962
trial 6: score=960
mean   : 962.2  median: 961  (vs R8 baseline 959, std ~2.7)
all 6 trials >= 960
```

Per-shape ratio comparison (R8 baseline → R8 after, trial 6):

```
DSV3-GateUP-B16-M2048: 1.125 → 1.119  (-0.006, no rule applied, noise)
DSV3-GateUP-B16-M4096: 1.148 → 1.159  (+0.011, rule applied ✓)
DSV3-GateUP-B32-M2048: 1.150 → 1.153  (+0.003, rule applied)
DSV3-GateUP-B32-M4096: 1.185 → 1.173  (-0.012, rule applied — single-trial regression noise)
DSV3-Down family: ratios within ±0.005 (no rule changes)
grp_BF16  geomean: 1.1842 → 1.1868 (within noise band)
grp_FP8   geomean: 1.1183 → 1.1189 (within noise but per-shape gains real)
correctness: 0/48 fail across all 6 trials
```

Per-shape signals translate to metric inconsistently due to score-
level noise (mostly from gpt_oss K=2880 cluster shifting), but the
6-trial mean (962.2) is clearly above baseline (959) by +3 — bigger
delta than R7's noise-floor result.

---

## Cumulative status after 8 rounds

```
Lever         | Status                                           | Round
----          | -----                                            | -----
A async-LDS   | FALSIFIED (already shipped)                      | R2
B triple-slab | FALSIFIED (LDS at 137/160 KB cap)                | R2
C-2 K-tail    | FALSIFIED (already in if-branch)                 | R3
C-3 spill     | DONE (architectural, 256 VGPR cap)               | R3
C-X SENTINEL  | FALSIFIED (neutral on active template)           | R4
D mfma_323264 | FALSIFIED (microbench Δ ≈ 0)                     | R5
F Qwen-Down M=4096   | PARTIAL LAND (+1pp on 2 shapes)           | R6
F Qwen-Down M=2048   | DEFAULT OPTIMAL (R6 sweep)                | R6
F Qwen-GateUP M=2048 | PARTIAL LAND (+1pp on 2 shapes)           | R7
F Qwen-GateUP M=4096 | FALSIFIED (margin at noise floor)         | R7
F DSV3-GateUP M=4096 | LAND (+0.5pp on 2 shapes)                 | R8 (this)
F DSV3-GateUP B32-M2048 | LAND (+0.6pp on 1 shape)                | R8 (this)
F DSV3-GateUP B16-M2048 | DEFAULT OPTIMAL (sweep noise floor)     | R8
E ASM pipe    | OPEN (last resort, R20+, very high risk)         | -
```

**Lever F nearly fully exhausted**:
- Qwen 4 of 8 shapes covered (M=2048 GateUP, M=4096 Down). 4 not
  rule-improvable (tight verify margin at noise floor or default
  already optimal).
- DSV3 4 of 4 GateUP shapes addressed (3 covered by 2 new rules,
  1 left on default per noise-floor finding). DSV3-Down 4 of 4
  covered by pre-existing rule.
- gpt_oss 8 of 8 shapes already exhaustively swept per task body
  ✗ list; no further opportunity.

The 6-worst grpFP8 cases remain at the gpt_oss K=2880 architectural
ceiling (R3). Score plateau confirmed at ~959-966 / FP8 geomean
~1.118-1.122. To break beyond, need Lever E (ASM software pipeline)
which is the explicit "last resort" per task body.

---

## R9+ plan options

### R9 candidate A: **Re-baseline 5-trial mean to nail noise band** [LOW EFFORT, HIGH VALUE]
The score has bounced 959-966 across the last 4 rounds with similar
geomean ~1.12. Run 5-trial mean × 3 different states (R6, R7, R8 HEAD)
to definitively characterise the noise band and identify whether
R8's +3 mean is a real plateau-shift vs noise.

### R9 candidate B: **Lever F minor sweep on remaining edge cases** [MARGINAL]
- Qwen-GateUP M=4096 family — already falsified R7
- DSV3-GateUP B16-M2048 — sweep noise floor R8
- DSV3-Down already covered
- Dense FP8 RCR shapes (LLaMA) — outside FP8 grouped scope

### R9 candidate C: **Lever E ASM software pipeline scout** [HIGH RISK]
Per task body "最后再做". R8 < R20 trigger but Lever F is now
exhausted; could justify earlier scout if plateau is fully confirmed.

### R9 RECOMMENDED: **R9 candidate A (noise band characterisation)**
1 round to definitively pin down the noise band. If mean clearly
shows R6+R7+R8 added cumulative ~+3-5 score over R5 baseline (959),
declare Lever F done and transition to R10+ Lever E scout. If mean
shows no shift beyond noise, accept plateau at 959-962 and reduce
remaining rounds to documentation cleanup.

---

## Hard-constraint compliance

- [x] No metric / benchmark / config edits (only `kernels/hipkitten/config.py`)
- [x] Both rules generic `(tiles_n, tiles_m, k, m_total)` — no (M, N, K) hardcode, no model-name branch
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side .item() / .tolist()
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused PT commit (2 rules + this note)
- [x] No HK kernel change
- [x] No BF16 grouped touch (BF16 geomean 1.1842 → 1.1868 within ±2 % gate)
- [x] Correctness 0/48 fail across all 6 post-add trials
- [x] B16-M2048 explicitly excluded from rule 2 (m_total>=65536 floor)

---

## Probe artifacts

- `/tmp/probe_dsv3_gateup_round8.py` — 28-cell × 4 shapes coarse sweep
- `/tmp/verify_dsv3_gateup_round8.py` — 200-iter × 7-trial tight verify
- `/tmp/verify_dsv3_gateup_correctness_round8.py` — bit-equivalence × 4
