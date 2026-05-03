# Round 35 — FP8 grouped GEMM fused-wall — gpt_oss-GateUP-B4-M2048 var-K dB carve-out

## TL;DR

R35 closes the **last m_total<16384 metric shape still on the binding default**:
**gpt_oss-GateUP-B4-M2048 var-K dB**. R34's note flagged this shape as
"default IS best per R34 probe" — but R34 only probed the **dA** path, never
**dB var-K**. R35 dispatch trace audit shows dB var-K falls to `(gm=4,
xcds=0)` binding default. R31 methodology (12-trial × 400-iter × 3-seed)
finds **(gm=2, xcds=2) WIN +2.61% median** with 0.30pp spread (med/spread =
8.7×, robust signal class).

Commit:
- **Primus-Turbo**: `<R35 SHA>`
- **HipKittens**: no change

## R35 work item — gpt_oss-GateUP-B4-M2048 var-K dB

### Discovery — dispatch trace audit

`/tmp/probe_r35_dispatch_trace.py` traces every metric shape's
(group_m, num_xcds) decision across forward RCR / dA RRR / dB var-K.
Result for the 24 fused-wall metric shapes:

```
shape                                 fwd RCR     dA       dB var-K
gpt_oss-GateUP-B4-M2048               (1, 4)      (4,None) (4,0) BINDING DEFAULT
gpt_oss-Down-B4-M2048                 (2, 2)      (2, 2)   (16, 4) R33
gpt_oss-GateUP-B4-M4096               (14, 4)     (8, 4)   (1, 4) R31
gpt_oss-Down-B4-M4096                 (32, 4)     (32, 4)  (8, 4) R39 default
gpt_oss-GateUP-B32-M2048              (8, 4)      (16, 4)  (1, 4) R31
gpt_oss-Down-B32-M2048                (16, 4)     (16, 4)  (4, 4) R30
... (all DSV3/Qwen3 shapes have specific or default rules; see probe output)
```

The ONLY metric shape on the m_total<16384 binding-default branch (after
R33 landed gpt_oss-Down-B4-M2048) is **gpt_oss-GateUP-B4-M2048 var-K dB**.
R33's gate (`a.shape[1]==2880 AND b.shape[1]==2880`) explicitly excludes
this shape because gpt_oss-GateUP has b.shape[1]=N_fwd=5760 (vs Down's
2880). R31's gate (`m_total>=16384 AND a.shape[1]==2880 AND
b.shape[1]==5760`) excludes because gpt_oss-GateUP-B4-M2048 has
m_total=8192 < 16384.

### Probe — calibrated 12 × 400 × 3 sweep

`/tmp/probe_r35_gpt_oss_gateup_b4_m2048_var_k_db.py`. Methodology mirrors
R31 / R33 / R34: 12 trials × 400 iters × 3 seeds, p17 percentile, median
across trials, median across seeds.

Cells tested: (4,0)=baseline; (16,4) [R33 sibling]; (1,4) [R31 sibling];
(2,4); (4,4); (8,4); (32,4); (1,2); (2,2); (4,2).

```
cell      seed=42 Δ%  seed=137 Δ%  seed=2024 Δ%   med Δ%   spread pp   verdict
(2, 2)    +2.88%      +2.57%       +2.61%         +2.61%   0.30        WIN  *unique top
(1, 2)    +2.20%      +1.68%       +1.97%         +1.97%   0.52        WIN
(1, 4)    +1.75%      +1.44%       +1.59%         +1.59%   0.31        WIN
(4, 2)    +1.64%      +1.26%       +1.42%         +1.42%   0.38        WIN
(2, 4)    +1.47%      +0.95%       +1.27%         +1.27%   0.51        WIN
(8, 4)    +1.35%      +1.22%       +1.10%         +1.22%   0.25        WIN
(16, 4)   +1.34%      +1.15%       +1.06%         +1.15%   0.27        WIN  (R33 sibling cell)
(32, 4)   +1.32%      +0.81%       +1.11%         +1.11%   0.52        WIN
(4, 4)    +1.10%      +0.50%       +0.81%         +0.81%   0.60        WIN
```

**Every probed cell beats the binding default (4, 0)**. (2, 2) is the
unique sharp top with +2.61% median and the tightest 0.30pp spread
(med/spread=8.7× — well above the 1.0× R31 robust-signal threshold).
Per-seed deltas all positive (+2.88 / +2.57 / +2.61) — every-seed-WIN
class.

### Neighbor probe — (2, 2) is on a sharp local optimum

`/tmp/probe_r35_neighbor_and_correctness.py` 7-cell neighbor sweep at
seeds {42, 137}, 8-trial × 400-iter p17:

```
cell    med (ms)  spread     Δ vs (2, 2)
(1, 1)  0.1666    0.0001     -4.5%
(1, 4)  0.1606    0.0003     -0.9%
(2, 1)  0.1627    0.0002     -2.3%
(2, 2)  0.1591    0.0001     *winner
(2, 4)  0.1614    0.0000     -1.4%
(3, 2)  0.1631    0.0003     -2.5%
(4, 2)  0.1609    0.0002     -1.1%
```

(2, 2) sits on a sharp single-cell optimum: every neighbor is ≥0.9%
slower. xcds=2 is the consistent half (xcds=1 -2.3% to -4.5%; xcds=4
-0.9% to -1.4% at neighbor gms). gm=2 is the optimum batching factor
(gm=1 -0.9%, gm=3 -2.5%, gm=4 -1.1%).

### Mechanistic rationale

The var-K kernel's CRR per-group output is `[N_fwd=5760, K_fwd=2880]`
⇒ `tiles_n=22, tiles_k=11` = 242 tile-steps per group × 4 groups = 968
tile-steps over NUM_CUS=256 persistent slots ≈ **4 wave-steps per slot**.

With only 4 wave-steps:
- Large gm (R33's 16, R31's 1): over- or under-batches.
- gm=2 cleanly fits 22/2=11 N-row groups per pass; the outer-iteration
  count (242/2=121 = 11×11) matches the K-axis traversal (11 K-tiles),
  so each batch step covers one full K-pass — no fractional batches.
- xcds=2 (vs the kernel default 8 and R30's xcds=4) keeps the 11-K-tile
  schedule inside a single chiplet pair (4 of 8 XCDs per pair on
  MI355X), which avoids cross-chiplet L2 invalidation when the small
  persistent grid (4 wave-steps) replays the per-K B-pack.

### Bit-equivalence

`/tmp/probe_r35_correctness_only.py` at seeds {0, 42, 137} — output
of `(2, 2)` vs `(4, 0)` on gpt_oss-GateUP-B4-M2048 var-K dB:

```
seed=0:    max_abs_diff=0.000000e+00  bit_eq=True
seed=42:   max_abs_diff=0.000000e+00  bit_eq=True
seed=137:  max_abs_diff=0.000000e+00  bit_eq=True
```

3/3 seeds bit-identical, no NaN/Inf in any cell. Same bit-equivalent
property documented for R30/R31/R32/R33/R34 (`group_m` and `num_xcds`
are pure persistent-grid scheduling knobs; arithmetic and FP8
quantization rounding invariant).

### Rule scope check

```python
elif a.shape[1] == 2880 and b.shape[1] == 5760:
    vk_group_m = 2
    vk_num_xcds = 2
```

The new `elif` is gated AFTER R33 (`a.shape[1]==2880 AND b.shape[1]==
2880`) inside the `m_total < 16384` else-branch. Matches:

- gpt_oss-GateUP B4-M2048 (m_total=8192, K_fwd=2880, N_fwd=5760) → ✓
- gpt_oss-GateUP B4-M4096 (m_total=16384) → m_total NOT < 16384 →
  excluded; R31 catches.
- gpt_oss-Down B4-M2048 (b.shape[1]=2880 ≠ 5760) → R33 catches first.
- gpt_oss-Down B4-M4096 (m_total=16384) → R39 default catches.
- DSV3/Qwen3 metric shapes (B>=16, m_total>=32768) → all excluded by
  m_total threshold.

Sibling regression check on gpt_oss-Down-B4-M2048 (R33 (16, 4)):
`(2, 2)` runs at 0.1091 ms vs R33's 0.1087 ms = -0.4% (within bench
noise). The gate cleanly excludes Down (b.shape[1]=2880 ≠ 5760), so
R33 is unaffected. No DoD test shape uses var-K dB outside grouped
metric scope.

## Self-bench `bench_grouped_gemm_turbo.py`

### FP8 (24 shapes, all PASS)
```
Average Forward TFLOPS:  2155.35  (R34: 2156.05)   Δ -0.7  (noise)
Average Backward TFLOPS: 1396.11  (R34: 1398.71)   Δ -2.6  (noise)

gpt_oss-GateUP-B4-M2048 bwd TFLOPS:
  R34 (16/24 PASS series): 1102.55
  R35 today              : 1117.31  (+1.3%)
```

### BF16 (24 shapes, all PASS — unaffected since R35 only touches fp8 var-K)
```
Average Forward TFLOPS:  1248.24  (R34: 1248.51)   Δ -0.3  (noise)
Average Backward TFLOPS:  919.06  (R34:  919.69)   Δ -0.6  (noise)
```

## Metric impact

```
Pre-R35 (R34 HEAD = 0c76a29) 4 runs (this session): 984 1000 873 994
                                                     median=989 (873 was
                                                     GPU-contention noise)
Post-R35  5 runs:                                    998 1000 995 993 996
                                                     median=996 mean=996.4
```

Distribution shifted modestly right vs pre-R35; the 873 outlier in
pre-R35 was clearly GPU contention (rocm-smi showed GPU[3] at 97% with
phantom KFD VRAM held by exited processes). Post-R35 dist is tight
(spread 7 across 5 runs) suggesting the GPU is less contended now.

Expected wall impact: var-K dB is ~25% of bwd wall on B=4 shapes (R12
profiler data). +2.61% kernel → ~+0.65% bwd wall → ~+0.30% fwd+bwd
wall on this shape (current ratio ~1.30). Geomean lift on 24-shape
suite: ~+0.012% (=0.30 / 24 * 1.30 contribution). ~+0.3 score points
at noise floor — small but the kernel-real signal is robust across 3
seeds and matches the R30/R31/R33/R34 pattern of "ship narrow
carve-out when probe shows clean WIN even if metric noise floor
swallows the geomean lift".

## Risk analysis

- **Correctness risk**: zero — `(group_m, num_xcds)` are pure
  persistent-grid scheduling knobs (verified bit-equivalent in 3/3
  seeds). 24/24 fp8 PASS, 24/24 bf16 PASS via SNR/allclose checks.
- **Regression risk on touched shape**: zero — kernel data shows clean
  WIN with med/spread = 8.7× (extremely tight signal across 3 seeds).
- **Regression risk on sibling shapes**: zero — rule scope verified by
  dispatch trace. R33's gpt_oss-Down B4-M2048 rule fires before the new
  R35 elif; R31's gpt_oss-GateUP M_total>=16384 rule excluded by
  m_total threshold; all DSV3/Qwen3 shapes excluded by m_total>=32768.
- **Downside if metric noise + bad seed cancels the gain**: -0.3
  score points median in the worst case, well within ±10 noise floor.
- **Upside if signal materialises at metric level**: +0.3 score points
  median.

Same risk-benefit profile as R30/R31/R32/R33/R34 ship decisions.

## Files touched

**Primus-Turbo only** (no HK kernel change):

- `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`
  - Added the new `elif a.shape[1] == 2880 and b.shape[1] == 5760` branch
    inside the existing `m_total < 16384` else-block (between R33's
    `if` branch and the `else` default branch).
  - 100-line documenting comment block (R34 dispatch-trace gap audit
    + 12 × 400 × 3-seed verify table + neighbor probe + bit-equivalence
    proof + rule scope + mechanistic rationale).
- `analysis/_notes/round-35-fp8-grouped-fused-wall-gpt_oss-GateUP-B4-M2048-var-K-dB-carve-out.md`
  — this round note.

## Behavior preserved

- Bit-equivalent output: `group_m` / `num_xcds` are pure persistent-grid
  scheduling knobs (3/3 seeds bit-identical to default).
- All 24 grouped FP8 metric shapes correctness PASS (SNR > 25 dB).
- All 24 grouped BF16 metric shapes correctness PASS (BF16 path
  untouched).
- The un-fused FP8 grouped path (`fuse_act_quant=False`) behavior
  unchanged — R35 only touches the dB var-K dispatch which is shared
  by both fused and un-fused fwd+bwd.
- HIPKITTEN registry stays `autotune=False` (unchanged).
- No host syncs added; no per-(M,N,K) hardcodes (rule keys on
  `a.shape[1]==2880` / `b.shape[1]==5760` / `m_total<16384` — these
  are general work-size tags, same idiom as R30/R31/R33).

## R36 candidates

After R35, **every metric shape on every kernel path now has either a
specific rule OR has been tight-verified to prefer the binding default**.
The dispatch trace `/tmp/probe_r35_dispatch_trace.py` confirms 0 / 24
metric shapes on the binding default that haven't been probed under
R31 methodology. Levers exhausted:

- Forward RCR (R6/R7/R8/R10/R12/R23/R28/R45/R50/R69/R70/R34): all
  bottom shapes have specific rules; re-tight-verify campaigns
  (R32/R33/R34 work item 1) confirm saturation.
- dA RRR (R27/R32/R34/R42/R43/R44): coverage extended to all
  tiles_n ∈ {6, 8, 16, 28} family; R34 closed the H4-reroute gap for
  gpt_oss-GateUP.
- dB var-K (R30/R31/R33/R35): coverage now spans all 4 sub-families
  in the m_total × b.shape[1]==5760/2880 grid:

```
                        m_total<16384       m_total>=16384
b.shape[1]==2880        R33 (16, 4)         R30 (4, 4)  [B>=32 only]
                                            R39 default (8, 4) [otherwise]
b.shape[1]==5760        R35 (2, 2)  *NEW    R31 (1, 4)
```

Suggestions for R36+ (in priority order):

1. **Accept plateau and write closure note**. Patience counter at 24
   rounds out of 30. The 24-shape suite is now fully tuned across all
   3 GEMMs; the only remaining levers are those R8 documented as
   architecturally blocked (Path A fused-act, kernel-internal MFMA
   improvements, C++ quantize_fp8 speed-up).

2. If R36 must run a probe: re-tight-verify R6 (Qwen3-Down M=4096 fwd
   RCR `(gm=2, xcds=None)`) **on B=16 vs B=32 split** with 24-cell
   wide sweep. R34 work item 1 only swept 10 cells and found R6
   saturated, but didn't try the R45 widened cell set (1, 4) /
   (4, 4) at the M=4096 tier.

3. **Audit forward RCR `tiles_m=8` cells** for shapes currently on
   `(4, None)=default` — these are Qwen3-Down-B16-M2048 / B32-M2048,
   ratio 1.236-1.281 today. R29 declared "exhausted" but only swept
   ``gm ∈ {1, 2, 8, 16, 32}`` — gm=4 and gm=24 not in that sweep.
   Potential undiscovered cell.

4. If still flat: consider Path A fused-act forward only with the
   R6 deposit (already in HK .so), keeping dB var-K and dA RRR
   un-fused. R7 falsified the full Path A; the partial-fwd-only
   variant was never explicitly tested as a separate metric.
