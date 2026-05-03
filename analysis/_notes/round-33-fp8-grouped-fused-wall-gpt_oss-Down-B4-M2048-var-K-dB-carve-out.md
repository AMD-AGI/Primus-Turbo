# Round 33 — FP8 grouped GEMM fused-wall — gpt_oss-Down-B4-M2048 var-K dB carve-out

## TL;DR

R33 first re-tight-verifies R7's `(gm=16, xcds=4)` rule for Qwen3-GateUP forward
RCR with R31 methodology (12-trial × 200-iter × 3-seed). **Result: R7 confirmed
saturated** — no candidate cell beats `(16, 4)` with a robust signal. R7 is
genuinely the optimum (positive falsification of R32 note's R33 candidate #1).

R33 then pivots to the next R32 note candidate: tight-verify the R30-deferred
gpt_oss-Down-B4-M4096 var-K dB cell. **Result: B4-M4096 itself is also
saturated** (max +1.12% TIE on (16, 4); spread 1.22pp). But the calibrated
sibling probe finds that **gpt_oss-Down-B4-M2048 var-K dB is on a significantly
suboptimal `(4, 0)` cell** (the universal m_total<16384 default branch), with
multiple candidate cells WIN-class.

R33 ships a narrow carve-out: when `m_total < 16384 AND a.shape[1]==2880 AND
b.shape[1]==2880`, switch the var-K dB rule to `(gm=16, xcds=4)`. This catches
gpt_oss-Down B4-M2048 only (the single metric shape with that var-K geometry
below the m_total threshold).

Commit:
- **Primus-Turbo**: `<R33 SHA>` (this round)
- **HipKittens**: no change

## Today's metric baseline

```
Pre-R33 (R32 HEAD = 1ba60a0), 5 runs: 1000 982 995 995 993  median=995  mean=993.0
Bottom non-exhausted shapes (single run):
  1.247  gpt_oss-Down-B32-M2048
  1.259  Qwen3-GateUP-B32-M2048      (R7 fwd + R32 dA carve-out, just landed)
  1.272  gpt_oss-Down-B32-M4096
  1.288  Qwen3-Down-B32-M2048        (R29 exhausted)
  1.293  Qwen3-GateUP-B16-M2048      (R7 fwd + R27 dA RRR)
  1.306  Qwen3-GateUP-B16-M4096      (R10/R45 fwd + R27 dA RRR)
  1.312  gpt_oss-Down-B4-M2048   <-- var-K dB target this round
  1.322  gpt_oss-Down-B4-M4096       (R30 deferred candidate falsified)
```

## R33 work item 1 — re-tight-verify R7 forward RCR (negative result)

### Probe: `/tmp/probe_r33_qwen3_gateup_fwd_rcr_cells.py`
- **Methodology**: 12 trials × 200 iters × 3 seeds, p20 + median.
- **Cells tested** (R7 sweep was 5 cells; R33 expands to 9):
  - R7 winner: `(16, 4)` — current rule
  - R7 close runners: `(1, 4)`, `(4, 4)`, `(32, 4)`, `default (4, 0)`
  - **NEW (not in R7 sweep)**: `(2, 4)`, `(8, 4)`, `(24, 4)`, `(16, 2)`, `(16, 8)`

### Verdict (median-of-medians vs R7 baseline, 3 seeds):
```
Qwen3-GateUP-B16-M2048
  (1, 4)        +0.36%  spread 2.48pp  TIE
  (2, 4)        -0.75%  spread 2.64pp  TIE
  (4, 4)        -0.75%  spread 0.87pp  TIE
  (8, 4)        -0.18%  spread 1.01pp  TIE
  (24, 4)       -0.40%  spread 1.86pp  TIE
  (32, 4)       -0.40%  spread 2.28pp  TIE
  default (4,0) -2.67%  spread 0.65pp  LOSS  (R7 still solidly beats default)
  (16, 2)       -1.18%  spread 2.34pp  TIE
  (16, 8)       -5.41%  spread 1.24pp  LOSS

Qwen3-GateUP-B32-M2048
  (1, 4)        -0.63%  spread 0.99pp  TIE
  (2, 4)        -0.91%  spread 0.55pp  LOSS
  (4, 4)        -0.68%  spread 0.18pp  LOSS  (clean -0.68% loss)
  (8, 4)        -0.98%  spread 0.30pp  LOSS  (clean -0.98% loss)
  (24, 4)       -0.58%  spread 0.92pp  TIE
  (32, 4)       +0.00%  spread 0.46pp  TIE  (R7 winner has identical sibling here)
  default (4,0) -2.64%  spread 0.90pp  LOSS
  (16, 2)       -1.65%  spread 0.22pp  LOSS  (clean)
  (16, 8)       -6.05%  spread 0.45pp  LOSS  (clean)
```

### Conclusion
R7 `(gm=16, xcds=4)` is **genuinely the optimum** for Qwen3-GateUP forward RCR
M=2048 cohort. Of 9 candidates tested with R31 methodology, none beats R7 with
a robust signal:
- B16-M2048: All "competitive" cells are TIE or LOSS; no WIN.
- B32-M2048: `(32, 4)` is exactly tied (Δ=0), all others LOSS or TIE.

This is the SAME finding R7 reached back when (200-iter × 7-trial p20). The
new methodology resolves slightly better signals (smaller spreads) but doesn't
change the verdict. **R7 is saturated** — the lever class for this rule is
exhausted, leave alone.

This is a **positive falsification**: R7's choice was correct; R32's
recommendation #1 yields no new metric points.

## R33 work item 2 — gpt_oss-Down var-K dB cell probe (mixed result)

### Probe: `/tmp/probe_r33_gpt_oss_down_b4_m4096_var_k.py` + calibrated re-probe.

**Initial probe** used the universal `(8, 4)` rule cell as baseline and tested
6 cells across 4 gpt_oss-Down shapes. Found B4-M2048 had multiple cells beating
`(8, 4)` by +2-5% — but B4-M2048's actual current rule is **`(4, 0)`** (the
`m_total < 16384` else branch) not `(8, 4)`.

**Calibrated re-probe** (`/tmp/probe_r33_gpt_oss_down_b4_m2048_var_k_calibrated.py`)
benchmarks each candidate vs **each shape's actual current rule**:

```
=== gpt_oss-Down-B4-M2048 (m_total=8192, current rule = (4, 0)) ===

  cell      Δ med vs (4,0)   spread (pp)   ratio   verdict
  --------------------------------------------------------------
  (8, 4)    +2.27%           2.45          0.93x   TIE
  (1, 4)    +5.15%           3.42          1.51x   WIN
  (2, 4)    +4.55%           0.34          13.4x   WIN  <-- super tight
  (4, 4)    +3.48%           1.91          1.82x   WIN
  (16, 4)   +5.43%           1.81          3.00x   WIN  <-- best median
```

Per-seed deltas for top candidates:
- `(16, 4)`: +5.42% / +6.60% / +4.78%   (every seed positive, lowest 4.78%)
- `(2, 4)`:  +4.55% / +4.41% / +4.74%   (super-tight, 0.33pp range)
- `(1, 4)`:  +5.14% / +6.55% / +3.12%   (noisier)

Winner-min (104.7 us at (16,4)) beats baseline-max (~110.5 us at (4,0)) in 3/3
seeds.

```
=== gpt_oss-Down-B4-M4096 (m_total=16384, current rule = (8, 4)) ===

  cell      Δ med vs (8,4)   spread (pp)   verdict
  --------------------------------------------------
  (4, 0)    -4.02%           1.47          LOSS  (universal rule's choice over else branch is correct)
  (1, 4)    +0.51%           0.85          TIE
  (2, 4)    +0.15%           1.80          TIE
  (4, 4)    +0.22%           0.87          TIE
  (16, 4)   +1.12%           1.22          TIE   <-- R30 deferred candidate
```

B4-M4096 itself: **R30's deferred (16, 4) candidate falsified** — +1.12%
median falls within the 1.22pp spread, so signal is at the noise floor. Keep
universal `(8, 4)` rule.

### Code change

`primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`:
subdivide the `m_total < 16384` else branch to add a gpt_oss-Down-B4-M2048
specific carve-out:

```python
else:
    # Round-33: subdivide m_total<16384 default branch.
    if a.shape[1] == 2880 and b.shape[1] == 2880:
        # gpt_oss-Down-B4-M2048 (only metric shape in this subspace)
        vk_group_m = 16
        vk_num_xcds = 4
    else:
        vk_group_m = 4
        vk_num_xcds = 0
```

#### Why (16, 4) over (2, 4)
Both cells WIN. (2, 4) has the cleanest signal (median/spread = 13.4×) but
+0.9pp less median improvement. (16, 4) has the best median (+5.43%) with
acceptable spread (3.0× ratio still robust). Picking (16, 4) prioritises lift;
(2, 4) is a known-safer alternative if a future round sees regression.

#### Why (gm=16) wins for B4-M2048 dB var-K (mechanistic rationale)
Per-group output is `[N_fwd, K_fwd] = [2880, 2880]` → `tiles_n=11, tiles_k=11`
→ 121 tile-steps per group × 4 groups = 484 tile-steps. Persistent NUM_CUS=256
→ ~2 wave-steps per slot. With so few wave-steps, `(group_m=4)` (the binding
default) only batches 4 M-tiles per pass before exhausting the K-axis traversal
under each persistent slot, losing L2-reuse opportunity on the per-K B-pack.
`(group_m=16)` batches 16 tiles per pass, fully pipelining K-tile traversal
under each batch group and saturating L2 reuse across the small persistent grid.

#### Why this doesn't generalise to B4-M4096 (per probe)
m_total=16384 → ~4 wave-steps per slot. Already enough wave-steps to amortise
default group_m=4 batching, so further batching with group_m=16 only yields
marginal +1.12% (within noise). Consistent with R30's coarse +1.01% reading
on this shape.

#### Rule scope check (m_total < 16384 with k=2880, n=2880)
In the 24-shape MoE metric, this matches **only gpt_oss-Down-B4-M2048**:
- gpt_oss-GateUP B4-M2048 (m_total=8192): k=2880, n=5760 → excluded by `n!=2880`.
- gpt_oss-Down B4-M4096 (m_total=16384): hits if branch above (universal (8,4)).
- All DSV3/Qwen3 metric shapes have B in {16, 32}, m_total ≥ 32768, hits if branch.

Verified at runtime by spy on `grouped_variable_k_crr_dscale`:
- gpt_oss-Down  B4-M2048   (m_total=8192)   → (16, 4)   [R33 NEW]
- gpt_oss-Down  B4-M4096   (m_total=16384)  → (8, 4)    [universal]
- gpt_oss-GateUP B4-M2048  (m_total=8192)   → (4, 0)    [default unchanged]
- gpt_oss-Down  B32-M2048  (m_total=65536)  → (4, 4)    [R30]
- gpt_oss-Down  B32-M4096  (m_total=131072) → (4, 4)    [R30]
- gpt_oss-GateUP B32-M2048 (m_total=65536)  → (1, 4)    [R31]
- Qwen3-GateUP B16-M2048   (m_total=32768)  → (8, 4)    [universal]

#### Bit-equivalence
`group_m` / `num_xcds` are pure persistent-grid scheduling knobs (same property
documented for R30/R31/R32/R39 and the dense fwd RCR rules). Arithmetic and
quantization rounding invariant.

## Self-bench `bench_grouped_gemm_turbo.py` (backward path required)

### FP8 (24 shapes, all PASS)
```
Average Forward TFLOPS: 2155.69
Average Backward TFLOPS: 1396.54
```

Spotlight on the affected shape (row 10):
```
gpt_oss_20B-Down  B=4  M=2048  N=2880  K=2880  fp8  tensorwise  PASS
  fwd: 1451.65 TFLOPS  bwd: 867.47 TFLOPS
```
Bwd 867 TFLOPS is consistent with +5% improvement over the (4, 0) baseline.

### BF16 (24 shapes, all PASS — unaffected since R33 only touches fp8 path)
```
Average Forward TFLOPS: 1246.88
Average Backward TFLOPS:  917.37
```

## Metric impact

```
Pre-R33  (5 runs): 1000 982 995 995 993   median=995   mean=993.0   range=18
Post-R33 (5 runs): 991 989 1000 987 1000  median=991   mean=993.4   range=13
```

- Median: 995 → 991 (Δ = −4, well within ±10 noise floor)
- Mean: 993.0 → 993.4 (Δ = +0.4)
- Range tightens slightly post (18 → 13).

Same situation as R32 (kernel-real signal, metric noise dominates):
expected wall impact = ~+0.6 score points (var-K dB ≈ 25% of bwd ≈ 12% of
fwd+bwd; +5.43% kernel → +0.66% wall on this shape → +0.66/24 ≈ +0.027% on
geomean → ~+0.6 score points). Sub-noise.

## Risk analysis

- **Correctness risk**: zero — `(group_m, num_xcds)` are pure scheduling knobs
  (same property documented for R30/R31/R32 dispatch tweaks). 24/24 fp8 PASS,
  24/24 bf16 PASS via `bench_grouped_gemm_turbo.py` SNR/allclose checks.
- **Regression risk on touched shape**: zero — kernel data shows +5.43% on the
  exact shape, with winner-min beating baseline-max in 3/3 seeds.
- **Regression risk on sibling shapes**: zero — rule scope verified to catch
  only gpt_oss-Down-B4-M2048 in the metric suite.
- **Downside if metric noise + bad seed cancels the gain**: −0.1 score points
  median in the worst case, well within ±10 noise floor.
- **Upside if signal materialises at metric level**: +0.6 score points median.

Same risk-benefit profile as R30/R32 ship decisions.

## R34 candidates

The plateau remains at metric=1000 best, mean ~993. Forward path (R7, R10, R45)
and dB var-K (R30, R31, R33) are largely saturated for the high-impact gpt_oss
+ Qwen3-GateUP shapes. Remaining low-ratio cohort:
- 1.288  Qwen3-Down-B32-M2048  (R29 exhausted dB; var-K dB universal default)
- 1.293-1.306  Qwen3-GateUP M=2048 (R7) and M=4096 (R10/R45)  — saturated
- 1.272-1.247  gpt_oss-Down B32 (R30 covers)  — saturated

Suggestions for R34+:
1. **Tight-verify Qwen3-Down-B32 var-K dB cells** with R31 methodology — last
   var-K dB family not yet probed at this resolution. R30 swept generic but
   may have missed a Qwen3-Down-specific cell.
2. **Tight-verify Qwen3-Down-B16-M4096 forward RCR** — current ratio 1.265,
   would need to check the R6 rule (Qwen-Down M=4096 family).
3. If R34 also exhausts, accept plateau (consecutive 22+ no-improve) and pivot
   to either:
   - Lever A (async global→LDS prologue overlap) — 10+ round commitment,
     +5-10pp geomean potential.
   - Lever D-Qwen 32x32x64 mfma microbench validation gate — would need new
     HK kernel.
   - Stop and accept current score as plateau-converged.
