# Round 58-dm — FP8 grouped: Lever D R-B step 2 — compile-time validation + R57 cost-model CORRECTION

**Status**: INFRASTRUCTURE commit + IMPORTANT correction to R57 falsification
**Score before**: 957 (R30 baseline run, within 947-963 plateau)
**Score after**:  960 (within plateau noise — change is static_assert only, zero runtime impact)
**HK SHA**: `c2abba21` → `75e30a5f`
**PT SHA**: this commit
**Round time**: ~25 min, 2 build cycles (one to surface assertion fail, one to fix), 2 metric runs
**Auto-optimize round**: 30

---

## TL;DR

1. Added a `static_assert` namespace `lever_d_round_b_step1_compile_test`
   in `kernel_fp8_layouts.cpp` that validates `rt_fp8e4m3<…, rt_32x64_s>`,
   `rt_fp8e4m3<…, rt_64x32_s>`, and `rt_fl<…, rt_32x32_s>` instantiate
   correctly with all expected geometric properties (height/width/cell
   rows-cols/per-lane sizeof).
2. **Build initially FAILED on a sizeof assertion I had wrong**, which
   surfaced a real arithmetic error in the R57-dm Lever D cost analysis:
   I had the per-cell footprint at **64 B/cell** when the actual value is
   **32 B/cell** for fp8 cell shapes.
3. **Re-deriving the Lever D K-tail port cost model with the correct
   per-cell footprint OVERTURNS R57's falsification**: peak register
   pressure with a Lever D K-tail port is ~192 dw/lane (vs 256 budget,
   under by 64 dw — NO SPILL), not the 320 dw I claimed in R57. The
   port may actually be a NET WIN of ~64 cy savings per K-tail tile.
4. R57 falsification was based on flawed math. Lever D K-tail port
   should be **re-classified as POSSIBLY VIABLE**, pending an actual
   implementation probe in R31+.
5. metric is unchanged (957 → 960, within plateau noise) since this
   round's change is compile-time-only.

---

## What was done

```cpp
// kernel_fp8_layouts.cpp lines ~108-180 (added)
namespace lever_d_round_b_step1_compile_test {
    using A_row_reg_32x64 = rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s>;     // 64×64
    using B_row_reg_32x64 = rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s>;     // 32×64
    using B_col_reg_64x32 = rt_fp8e4m3<64, RBN, col_l, rt_64x32_s>;     // 64×32
    using cAB_32_acc      = rt_fl<RBM, RBN, col_l, rt_32x32_s>;         // 64×32

    static_assert(A_row_reg_32x64::height == 2, …);  // 64/32
    static_assert(A_row_reg_32x64::width  == 1, …);  // 64/64
    static_assert(A_row_reg_32x64::base_tile_rows == 32, …);
    static_assert(A_row_reg_32x64::base_tile_cols == 64, …);
    static_assert(sizeof(A_row_reg_32x64) == 32 * 2 * 1, …);  // 64 B/lane

    static_assert(cAB_32_acc::height == 2, …);
    static_assert(cAB_32_acc::width  == 1, …);
    static_assert(sizeof(cAB_32_acc)  == 16 * 4 * 2 * 1, …);  // 128 B/lane

    using cA_layout_16x16 = rt_fl<RBM, RBN, col_l, rt_16x16_s>;
    static_assert(sizeof(cA_layout_16x16) == sizeof(cAB_32_acc), …);
        // both 32 dw/lane covering 64×32 region
}
```

This is purely compile-time validation. No runtime / codegen impact.

**HK commit**: `75e30a5f` — `infra(fp8 grouped): Lever D R-B step 1 —
compile-time validation of rt_fp8e4m3<rt_32x64_s>`. 78 lines added (60-line
context comment + 18 static_assert lines).

## Verified spill profile after change

| Spec | spill before R30 | spill after R30 |
|--|--|--|
| `grouped_rcr_kernel<0,0,0>` | 39 dw | **39 dw** (UNCHANGED) |
| `grouped_rcr_kernel<0,1,0>` | 43 dw | **43 dw** (UNCHANGED) |

static_asserts have zero codegen footprint. Confirmed.

---

## R57-dm cost-model CORRECTION — the per-cell footprint error

### What I had wrong in R57 (and earlier)

In R57-dm and the R56 quantitative analysis I quoted:

> *"cAB_32 = 128 dw"*

This was based on a wrong assumption about per-cell rt_base sizeof.
I implicitly used **64 B/cell** for fp8e4m3 cells, which I carried
forward as the merge-step footprint and as the basis for the
"register pressure spike during merge = 320 dw → ~640 cy spill" claim.

### What the build actually told me

The first build attempt this round failed on:

```
kernel_fp8_layouts.cpp:160:19: error: static assertion failed
  due to requirement
  'sizeof(kittens::rt<__hip_fp8_e4m3, 64, 64, …, rt_32x64>)
    == 64 * 2 * 1':
  A_row_reg_32x64 should be 128 bytes per lane
note: expression evaluates to '64 == 128'
```

**Actual sizeof = 64 B/lane**, NOT 128 B/lane.

Re-deriving from first principles (in `include/types/register/rt_base.cuh`):

```
rt_base<fp8e4m3, row_l, rt_32x64>:
  num_elements        = 32 × 64 = 2048
  elements_per_thread = 2048 / 64 = 32 fp8 / lane
  num_packed (fp8e4m3_4) = 4
  packed_per_thread   = 32 / 4 = 8 fp8e4m3_4 / lane
  sizeof(rt_base)     = 8 × sizeof(fp8e4m3_4) = 8 × 4 B = 32 B / cell / lane
```

So a single rt_base FP8 cell is **32 B / lane = 8 dwords / lane**, NOT
the 64 B / lane = 16 dw / lane I'd been carrying. Half my estimate.

### Corrected per-tile per-lane footprints

For all the relevant types (rounded to whole dwords for budget math):

| Type | height × width | per-cell B/lane | per-tile B/lane | per-tile dw/lane |
|--|--|--|--|--|
| `A_row_reg` (rt_16x128_s)        | 4 × 1 | 32 | 128 | **32 dw** |
| `A_row_reg` if expanded for K=128 | 8 × 1 | 32 | 256 | **64 dw** |
| `A_row_reg_32x64` (proposed)     | 2 × 1 | 32 | 64  | **16 dw** |
| `B_row_reg` (rt_16x128_s)        | 2 × 1 | 32 | 64  | **16 dw** |
| `B_row_reg_32x64` (proposed)     | 1 × 1 | 32 | 32  | **8 dw**  |
| `cA-cD` each (rt_fl, rt_16x16_s) | 4 × 2 | 16 | 128 | **32 dw** |
| `cAB_32` each (rt_fl, rt_32x32_s)| 2 × 1 | 64 | 128 | **32 dw** |

Sanity: cA and cAB_32 have **identical** total per-lane footprint (32 dw),
because both cover a 64×32 region in fp32. Differ only in cell-internal
lane partition (16x16 mfma layout vs 32x32 mfma layout). The static_assert
`sizeof(cA_layout_16x16) == sizeof(cAB_32_acc)` confirms this at compile
time.

### Corrected register-pressure analysis for Lever D K-tail port

Using the correct dword footprints:

**Existing K-tail (16x16x128 mfma + SENTINEL fill of K=64..127)**:
- 4 × cA-cD live throughout: 4 × 32 dw = **128 dw**
- a (rt_fp8e4m3<64, 128, row_l, rt_16x128>): 8 × 1 cells × 8 dw = **64 dw**
- a_kt1 (same type): **64 dw**
- b0 (rt_fp8e4m3<32, 128, row_l, rt_16x128>): 2 × 1 × 8 = **16 dw**
- b1 (same): **16 dw**
- **Peak = 128 + 64 + 64 + 16 + 16 = 288 dw** → over 256 budget by 32 dw
- → ~256 cy spill cost ON HOT PATH (matches the observed 39 dw kernel-wide
  spill profile: K-tail block's contribution to spill is roughly ~32 dw).

**Lever D K-tail port (32x32x64 mfma, K=64 native, no SENTINEL)**:
- 4 × cA-cD live throughout: **128 dw**
- a_32 (rt_fp8e4m3<64, 64, row_l, rt_32x64>): 2 × 1 × 8 = **16 dw**
- (no a_kt1 — K=64 fits in single mfma_323264 input)
- b0_32 (rt_fp8e4m3<32, 64, row_l, rt_32x64>): 1 × 1 × 8 = **8 dw**
- b1_32: **8 dw**
- temp accumulator cAB_32 (briefly live during merge):
  - For 4 cAB_32 covering same 4 × 64×32 regions as cA-cD:
    4 × 32 dw = **128 dw** simultaneously (worst case)
- **Peak during mfma_323264 = 128 + 128 + 16 + 8 + 8 = 288 dw**
  (same as existing path)

OR if we compute cAB_32 ONE-AT-A-TIME and merge before computing next:
- **Peak = 128 + 32 + 16 + 8 + 8 = 192 dw** → UNDER 256 budget by 64 dw
- → **NO spill on the K-tail block** (potential huge win).

The "one-at-a-time" pattern requires 4 LDS roundtrip merges (one per cAB_32)
instead of one batched merge — but the cycle budget is dominated by mfma+
LDS, not the loop overhead.

### Corrected mfma cycle savings

Per K-tail tile (covering 4 × 64×32 = 64×128 output region):

**Existing 16x16x128 path**:
- 4 cA-cD tiles × mma_AB inner (height×width×A_width = 4×2×1) = 32 mfma_16x16x128
- mfma_16x16x128 issue cycle on gfx950: ~8 cy
- Total mfma issue: **256 cy**
- BUT K_REM=64 only fills half the K=128 input → 50% of mfma issue is
  computing on SENTINEL fp8 (zero-pad). Functionally wasted work, but
  hardware still issues the full instruction.

**Lever D 32x32x64 path**:
- 4 cAB_32 tiles × mma_AB inner (2×1×1) = 8 mfma_323264
- mfma_323264 issue cycle on gfx950: ~16 cy
- Total mfma issue: **128 cy**
- K=64 native — no SENTINEL waste.
- **Savings: 256 - 128 = 128 cy mfma issue.**

### Corrected LDS round-trip merge cost

Per cAB_32 → cA-cD additive merge:
- ds_write_b128 × 8 (write 32 dw / lane): ~8 cy issue
- s_barrier: ~10 cy
- ds_read_b128 × 8 + accumulate into cA-cD: ~16 cy issue (read + fp32 add)
- Per cAB_32: ~34 cy

For 4 cAB_32 tiles (one-at-a-time pattern, 1 barrier amortized):
~16 (write) + 10 (barrier) + 4 × ~16 (read+add) = **~90 cy total merge**.

Or batched (4 cAB_32 simultaneously then merge): 32 cy write + 10 barrier +
64 cy read+add = **~106 cy total merge**.

### Corrected net per K-tail tile

| Component | Existing (16x16x128) | Lever D (32x32x64) | Δ |
|--|--|--|--|
| mfma issue | 256 cy | 128 cy | **-128 cy** |
| K-tail spill cost | ~256 cy | 0 cy | **-256 cy** |
| Cell-shape merge | 0 cy | ~90-106 cy | **+~100 cy** |
| **Total** | **~512 cy** | **~228 cy** | **-284 cy savings** |

**Estimated savings per K-tail tile = ~284 cy.**

If a typical FP8 grouped tile cycle count is ~1500 cy (rough ballpark):
- savings = 284 / 1500 = **18.9% per K-tile**
- Only K_REM > 0 shapes benefit (gpt_oss family — 4 of 16 shapes)
- Per-shape ratio improvement: ~+18 pp on K_REM>0 shapes
- Geomean impact (4 of 16 shapes affected): ~+(4/16) × 18 = **+4.5 pp**
  on overall grp_FP8 geomean

If correct, this would push grp_FP8 geomean from **1.117 → ~1.16**, narrowing
the gap to the 1.20 target by ~50%.

**This is large enough to consider attempting.**

---

## What this means for R31+ direction

### What R57-dm got RIGHT
- The `_s` aliases really were missing and needed to be added (HK c2abba21).
- The mma.cuh dispatch scaffold for 32x32x64 already exists.
- The standalone M2N2 kernel exists and was already tested as a separate
  launch (R52: -76 pts catastrophic due to launch overhead + C-RMW).

### What R57-dm got WRONG
- The per-cell sizeof was off by 2× — quoted 64 B/cell when actual is
  32 B/cell.
- The peak register pressure during merge was overstated by ~128 dw
  (the cAB_32 footprint was double-counted).
- The conclusion **"Lever D K-tail port has ~10 pp NET LOSS"** was based
  on these errors — it is **NOT supported** by the corrected math.

### What this round (R58) does NOT yet establish
- I have **NOT** actually built and benchmarked a Lever D K-tail port.
  The corrected analysis is still analytical; real codegen / spill
  profile / cycle count could differ from the model.
- The "one-at-a-time cAB_32 merge" pattern is the optimistic path —
  if LLVM can't keep cAB_32 dead between iterations, the batched
  pattern peaks at 320 dw which IS over budget and DOES spill.

### Recommended R31+ direction

**Option A — Lever D K-tail probe (recommended)**:
1. R31: implement Lever D K-tail block as a compile-time toggle
   (`#ifdef LEVER_D_KTAIL`). Use the M2N2 standalone kernel's
   loader / mfma logic but keep it inline in the persistent kernel
   (avoid the launch overhead that killed M2 standalone in R52).
2. R32: validate correctness (allclose) on K_REM=64 specs.
3. R33: measure metric. Compare 16x16x128 K-tail vs 32x32x64 K-tail
   with this analysis as the prediction. Keep whichever wins.

This is **2-3 rounds**, not the 8-12 round Lever D Round-B
(full main-loop port). Risk: ~20-30% (the merge pattern might not
schedule cleanly, or LLVM may keep cAB_32 alive longer than expected).

**Option B — accept plateau (R29's recommendation)**:
- Continue plateau-acceptance. Per-round metric improvement expected: 0.
- Patience clock will hit 10 around R37, but the auto-optimize loop
  doesn't EARLY-STOP, so this is fine.

**My recommendation**: try Option A in R31. The corrected cost model
shows the falsification was wrong; the measured outcome will tell us
if the corrected model is right.

---

## Key takeaway for future agents

When you do a cost analysis based on register footprint, **VERIFY the
footprint with a static_assert in the actual code**, not just by hand
calculation. The kittens type system makes per-cell sizeof
non-obvious (it depends on `num_packed`, `packed_per_thread`, and
`sizeof(packed_dtype)` interacting in `rt_base.cuh`). Hand math
fails silently; static_assert fails loudly at build time and points
at the exact wrong line.

The R57 falsification of Lever D K-tail port is hereby **WITHDRAWN**.
The port should be re-evaluated empirically in R31+.

---

## Round summary

| Item | Value |
|--|--|
| Goal | Validate Lever D R-B type infrastructure + check R57 cost model |
| Change | static_assert namespace in kernel_fp8_layouts.cpp |
| Lines added | 78 (HK) + this note (PT) |
| Spill profile change | NONE (39/43 dw — UNCHANGED) |
| Metric before | 957 |
| Metric after | 960 (plateau noise — change is compile-time only) |
| HK commit | `75e30a5f` |
| Side discovery | R57 cost-model arithmetic error → Lever D falsification withdrawn |
| Next round suggestion | Lever D K-tail empirical probe (Option A above) |
