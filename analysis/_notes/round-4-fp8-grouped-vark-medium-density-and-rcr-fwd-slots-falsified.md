# Round-4 — gpt_oss FP8 kernel-only ceiling: GateUP-B4 wgrad + RCR fwd slots levers FALSIFIED

**Date**: 2026-05-07 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD `70608201` → R4)
**Scope**: gpt_oss_20B Balanced FP8 kernel-only, 8-shape suite (4 GateUP + 4 Down × {B=4, B=32}).
**Goal**: Extend the R3 short-grid `slots=192` lever from var-K wgrad to (a) GateUP-B4 wgrad (the
medium-density wave-step regime not covered by R2), and (b) the file-scope `grouped_rcr_kernel`
fwd persistent kernel (mirror lever applied to the largest-section-by-absolute-miss).

## Bottom line

Both extensions **falsified**. The R3 `slots=192` rule for var-K wgrad on `gpt_oss-Down-B4`
remains the only metric-moving change in this run. R4 ships **kernel-template parameterization
infrastructure for fwd RCR** (NUM_CUS → gridDim.x + `TK_RCR_NUM_CUS` env hook) but **no Python
dispatcher rule** because no shape benefits at metric magnitudes worth a rule.

| Direction | Anchor shape(s) | Best non-256 slots | Δ vs slots=256 | Verdict |
|---|---|---|---|---|
| GateUP-B4 wgrad slots | `Down-B4-M2048 wgrad` (R2 anchor)<br>`GateUP-B4-M2048 wgrad`<br>`GateUP-B4-M4096 wgrad` | slots=192<br>slots=224<br>slots=224 | (R3 rule, already shipped)<br>**−1.06 %** (within noise)<br>**−1.40 %** (within noise) | FALSIFIED |
| RCR fwd slots | `Down-B4-M2048 fwd` (sparsest)<br>`Down-B4-M4096 fwd` (medium)<br>`GateUP-B32-M4096 fwd` (counter) | slots=208<br>slots=208<br>slots=240 | **+1.47 %** (within ±2 % p20 spread)<br>**−17.58 %**<br>**−12.90 %** | FALSIFIED |

The +1.47 % on Down-B4-M2048 fwd was tested without a tight-verify pass because its magnitude is
inside the kernel-only p20 spread observed across earlier rounds (±2 %); even if real, the section
avg lift would be ≤ 0.1 % (1894 → ≤ 1897 T) — well below the noise of the 24-shape metric. Not worth
a per-shape rule.

## Probe protocol

### A. GateUP-B4 wgrad NUM_CUS sweep
- Script: `scripts/_probe_round_4_gateup_b4_wgrad_numcus.py`
- Driver: subprocess-per-slot with `TK_VARK_NUM_CUS=<v>` (R2 env hook still live).
- Anchors: `GateUP_B4_M2048 wgrad` (m_total=8192), `GateUP_B4_M4096 wgrad` (m_total=16384). Both
  have N=5760, K=2880 → `(tiles_n, tiles_k) = (22, 11)` tiles per CRR var-K, **968 tile-steps total**
  ≡ 3.78 wave-steps/CU. Density sits between R2's anchor (Down-B4 var-K wgrad: 1.89 wave-steps/CU,
  slots=192 wins +6.6 %) and R2's counter (GateUP-B32-M4096 wgrad: 30.25 wave-steps/CU, slots=192
  loses −17 %).
- Sweep grid: `slots ∈ {128, 160, 192, 208, 224, 240, 256}`, 250 iters × 7 trials × 3 seeds, p20 per
  seed → median across seeds.
- Saved JSON: `/tmp/round_4_gateup_b4_numcus_sweep.json`.

```text
GateUP_B4_M2048 wgrad  (3.78 wave-steps/CU)
    slots=128   1129.8   -34.05%
    slots=160   1411.5   -17.61%
    slots=192   1551.3    -9.45%
    slots=208   1525.9   -10.93%
    slots=224   1695.1    -1.06%   ← closest contender, within noise
    slots=240   1664.9    -2.82%
    slots=256   1713.2    +0.00%   ← BEST

GateUP_B4_M4096 wgrad  (3.78 wave-steps/CU, larger absolute size)
    slots=128   1378.7   -32.76%
    slots=160   1721.5   -16.04%
    slots=192   1871.8    -8.71%
    slots=208   1845.3   -10.00%
    slots=224   2021.7    -1.40%   ← closest contender, within noise
    slots=240   2018.5    -1.56%
    slots=256   2050.4    +0.00%   ← BEST
```

**Reading**: GateUP-B4 wgrad's tile-step density (3.78 wave-steps/CU) is high enough that the
per-tile prologue/epilogue is already well-amortized over the K-iter MFMAs. The `slots=192` lever
that nets +6.6 % on the much sparser Down-B4 wgrad anchor (1.89 wave-steps/CU) **does not generalize**
to the medium-density regime. Predicate ceiling for short-grid wgrad: roughly `wave-steps/CU < 2.5`.

### B. RCR fwd NUM_CUS sweep
- Script: `scripts/_probe_round_4_rcr_fwd_numcus.py`
- Required HK kernel mod (R4 infrastructure):
  - `kernel_fp8_layouts.cpp` file-scope `grouped_rcr_kernel`: replaced constexpr `NUM_CUS` with
    `gridDim.x` for both the chiplet swizzle range (line ~2696) and the persistent loop stride
    (line ~2766). Replicated in the `kernel_b128::grouped_rcr_kernel` body (env-quarantined per
    Round-F so dead code in production, kept consistent for cleanliness).
  - `dispatch_grouped_rcr`: cached process-static `rcr_slots` from `TK_RCR_NUM_CUS` (clamped to
    `(0, NUM_CUS]`) and propagated to all four production launch sites (4-wave / 8-wave × n_aligned ∈
    {true, false}). The b128 launch sites at lines ~7378/7381 keep `dim3(NUM_CUS)` because the
    b128 path is env-gated under `TURBO_FP8_B128=1` and stays at the default grid for that
    research path.
- Bit-equivalence: when `TK_RCR_NUM_CUS` is unset (`rcr_slots == NUM_CUS == 256`), the kernel body's
  use of `gridDim.x` resolves to the same constant the original code used; no math change.
  Confirmed by the post-rebuild metric: 682 → 684 (within ±2 noise; same shape-by-shape ratios).
- Anchors: `Down-B4-M2048 fwd` (1.5 wave-steps/CU, sparsest), `Down-B4-M4096 fwd` (medium),
  `GateUP-B32-M4096 fwd` (46 wave-steps/CU, saturated counter).
- Sweep grid + protocol: same as section A.

```text
Down_B4_M2048 fwd  (1.5 wave-steps/CU — sparsest)
    slots=128   1331.4   -16.84%
    slots=160   1298.4   -18.91%
    slots=192   1590.1    -0.69%
    slots=208   1624.6    +1.47%   ← tiny lift, within ±2% noise
    slots=224   1587.3    -0.86%
    slots=240   1503.0    -6.13%
    slots=256   1601.1    +0.00%

Down_B4_M4096 fwd
    slots=128   1353.5   -33.50%
    slots=160   1552.4   -23.73%
    slots=192   1627.9   -20.02%
    slots=208   1677.5   -17.58%   ← best non-256, still huge regression
    slots=224   1648.2   -19.02%
    slots=240   1521.6   -25.24%
    slots=256   2035.4    +0.00%   ← BEST

GateUP_B32_M4096 fwd  (46 wave-steps/CU — counter / regression guard)
    slots=128   1255.9   -39.80%
    slots=160   1517.5   -27.26%
    slots=192   1679.9   -19.47%
    slots=208   1732.8   -16.94%
    slots=224   1785.4   -14.41%
    slots=240   1817.0   -12.90%
    slots=256   2086.1    +0.00%   ← BEST (saturated, expected)
```

**Reading**: Unlike var-K wgrad, fwd RCR fundamentally does NOT respond to slots reduction. Even
the sparsest gpt_oss fwd shape (Down-B4-M2048 at 1.5 wave-steps/CU) shows only a +1.47 % flicker at
slots=208 — within the p20 noise floor. The denser fwd shapes (Down-B4-M4096 at 3 wave-steps/CU and
saturated B=32 shapes) regress severely at any slots < 256. The likely reason: RCR's per-tile body
(`rcr_8w_load_hoist` with persistent `cA / cB / cC / cD` accumulators across the K-iter loop) already
issues most of its HBM/MFMA work *inside* the per-tile inner K loop; the per-tile prologue/epilogue
amortizes through K-iters more efficiently than the var-K wgrad inner reduction loop. Reducing the
slot count just gives each CU more sequential tiles to chew through with the same per-tile cost,
which scales linearly until you exhaust the under-saturated regime — which fwd RCR doesn't really
have.

## Falsification implications for the run

1. The `slots=192` short-grid lever is a **var-K wgrad-specific** optimization, not a generalizable
   gpt_oss FP8 ceiling primitive. R3's rule (`m_total ≤ 16384` AND `K==N==2880`) remains the only
   shape family it touches.
2. **Fwd RCR ceiling for sparse shapes is bound by something OTHER than CU count.** R5+ should pivot
   off the launch-geometry hypothesis and instead PMC-profile the fwd RCR kernel directly on
   Down-B4-M2048 (the sparsest fwd shape) — mirror of R1's wgrad PMC pass. Likely candidates:
   per-tile epilog stall, K-iter overlap inefficiency on the K-tail (K%128==64), or VGPR spill-driven
   occupancy ceiling. PMC will tell.
3. **wgrad ceiling for medium-density shapes is also other-than-CU-bound.** Same pivot — PMC the
   GateUP-B4 wgrad to figure out why slots=256 is optimal.

## R4 deliverables

### HipKittens (`/workspace/code/HipKittens`)
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - File-scope `grouped_rcr_kernel`: NUM_CUS → `gridDim.x` (chiplet swizzle, persistent loop).
  - `kernel_b128::grouped_rcr_kernel`: same change, kept consistent across the env-quarantined spec.
  - `dispatch_grouped_rcr`: `TK_RCR_NUM_CUS` env hook, propagated to 4 production launch sites.
  - **Bit-equivalent at default NUM_CUS=256** (post-rebuild metric 684 ≈ pre-rebuild 682).

### Primus-Turbo (this repo)
- `scripts/_probe_round_4_gateup_b4_wgrad_numcus.py` — GateUP-B4 wgrad slots sweep (driver).
- `scripts/_probe_round_4_rcr_fwd_numcus.py` — RCR fwd slots sweep (driver).
- `analysis/_notes/round-4-fp8-grouped-vark-medium-density-and-rcr-fwd-slots-falsified.md` — this note.
- **No `select_default_config` change.** No `grouped_gemm_fp8_impl.py` change. Metric unchanged.

## R5 plan

Pivot to **PMC profiling of fwd RCR on Down-B4-M2048 fwd** (the worst fwd shape). Mirror the R1
methodology:
- `rocprofv3 --pmc GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_VALU_MFMA_BUSY_CYCLES
   SQ_VALU_MFMA_COEXEC_CYCLES SQ_INSTS_VALU_MFMA_F8 SQ_INST_CYCLES_VMEM_RD SQ_WAIT_INST_LDS`
- 250 iters of `grouped_gemm_fp8_impl` for that one shape, kernel-isolated (mirror
  `_probe_round_1_wgrad_pmc.py`).
- Compute MFMA-active fraction, LDS / VMEM stall fractions, and look for the dominant stall.
  R1 showed wgrad was at 16.6 % MFMA-active (under-saturated) with low LDS / VMEM stalls →
  pointed to launch geometry (correct for var-K, falsified for RCR fwd here).
- If fwd RCR is at ~50 % MFMA-active with high VMEM stalls: per-tile data-prefetch lever is open
  (e.g., `cp.async.cg` cluster size). If ~50 % MFMA-active with high SQ stalls: VGPR spill /
  occupancy lever. If 80%+ MFMA-active: ceiling is tile-shape, not amortization — different lever
  needed (4-wave / 16x16 spec swap).
