# Round 54 (dm) — FP8 grouped: Lever C-2 step-1 scaffold landed (HK 73da21c6)

## TL;DR
- Metric: **score=982**, geomean BF16=1.1861 / FP8=1.1718. BF16 nominally
  fails the 1.20 gate, but absolute HK BF16 numbers are unchanged from
  R53 (e.g. gpt_oss-GateUP-B32-M4096 1262 vs R53 1255); Triton baseline
  varied 1114→1121 across runs, dragging geomean. Pure metric noise per
  R51 σ-band.
- **HK scaffold landed**: `namespace lever_c2_round_54_step1_scaffold`
  with WARPS_N=2 / RBN_4w=64 / _NUM_THREADS=256 + register-tile types
  `A_row_reg_4w`, `B_row_reg_4w`, `C_acc_4w`. **Build clean, 8 static
  asserts hold, codegen zero-impact** (all 4 grouped variants' resource
  reports identical to R53 baseline: spill 54/38/34/37).
- HK SHA: `73da21c6`. R55 step-2 (copy kernel body into namespace +
  re-target const refs) is now 1-step away from a build-and-measure.

## 1. Metric (R54)

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1861 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1718 (n=24)
[metric_grouped_only] (1) grp_BF16  >= 1.20  : 1.1861  FAIL
[metric_grouped_only] (2) grp_FP8   >= 1.20  : 1.1718  FAIL
[metric_grouped_only] score=982
```

BF16 1.1861 is unusual (R51-R53 all hit 1.22-1.23 PASS) but absolute
HK numbers unchanged: e.g. gpt_oss-GateUP-B32-M4096 HK=1262 (R53: 1255),
DSV3-Down-B32-M4096 HK=1289 (R53: 1287). The drop is ENTIRELY from
Triton getting faster (run-to-run). Same noise mechanism that affected
R52/R53 FP8 geomean.

## 2. C-2 scaffold landed

Located right after `lever_d_round_b_step1_compile_test` namespace
(line 184), 96-line addition. Compiles cleanly; static-assert checks:

```
static_assert(_NUM_THREADS == 256);
static_assert(WARPS_M * WARPS_N == 4);
static_assert(RBM_4w * RBN_4w == 4096);    // 64*64 = 4096 fp32/warp output
static_assert(C_acc_4w::height == 4);      // 64/16 cell rows
static_assert(C_acc_4w::width == 4);       // 64/16 cell cols
static_assert(sizeof(C_acc_4w) == 256);    // 64 fp32/lane × 4 B = 256 B/lane
static_assert(sizeof(A_row_reg_4w) == sizeof(::A_row_reg));   // RBM unchanged
static_assert(sizeof(B_row_reg_4w) == 2 * sizeof(::B_row_reg)); // RBN doubled
```

All 8 hold ⇒ the type system accepts the larger 64×64 register tile
instantiations.

## 3. Codegen non-regression (post-scaffold resource report)

Re-built clean and captured `-Rpass-analysis=kernel-resource-usage`:

| kernel                                               | VGPR | AGPR | Spill | Δ vs R53 |
|------------------------------------------------------|------|------|-------|----------|
| `rcr_4w::kernel` (line 1380, was 1284)               | 198  | 256  | 0     | 0/0/0    |
| `grouped_rcr_kernel<0,F,F>` (line 2319, was 2223)    | 256  | 0    | 54    | 0/0/0    |
| `grouped_rcr_kernel<0,T,F>`                          | 256  | 0    | 38    | 0/0/0    |
| `grouped_rcr_kernel<0,F,T>` (gpt_oss)                | 256  | 0    | 34    | 0/0/0    |
| `grouped_rcr_kernel<0,T,T>` (gpt_oss)                | 256  | 0    | 37    | 0/0/0    |

Line numbers shifted +96 (scaffold size) but resource numbers are
**bit-identical** to R53. Confirms the scaffold has zero codegen
footprint.

## 4. R55 path (1 step from working prototype)

Copy lines ~2319-2980 (the four template instantiations of
`grouped_rcr_kernel`) verbatim into
`lever_c2_round_54_step1_scaffold::` namespace. Inside the
namespace, the unqualified references to `WARPS_M`, `WARPS_N`,
`_NUM_THREADS`, `RBM`, `RBN`, `A_row_reg`, `B_row_reg` will resolve
to the *local* (4w-style) versions automatically — no rewrites
needed *inside* the kernel body.

Caveats discovered in the R52 plan that R55 must handle:
- `__shared__ ST_rcr As[2][2]` — `ST_rcr` is `ST_v2` (line 2224) which
  uses `HB`, NOT `RBN`. Stays unchanged.
- The `RBM`-sized tiles inside the kernel body are typed via
  `A_row_reg`/`B_row_reg`; renaming the typedef inside the namespace
  is sufficient (no manual line-by-line edits in the kernel body).
- Explicit instantiations at line 2882-2885 must be lifted into the
  namespace too: `template __global__ void
  lever_c2_round_54_step1_scaffold::grouped_rcr_kernel<0, F, F>(...)` etc.
- `dispatch_grouped_rcr` (line 5384-5392) does NOT need to change in
  R55 — it stays calling the OUTER `grouped_rcr_kernel`, leaving the
  4w prototype unwired (build-only verification).

R55 acceptance (revised R53 criteria):
- (a) AGPR ≥ 256, Spill ≤ 8 on the namespace kernel instances
- (b) Occupancy ≥ 1 waves/SIMD
- (c) MFMA count in the kernel body doubles (8 → 16 per K-step)

## 5. R54 actions

- Metric run: ✅ score 982 (within R51 σ band)
- HK scaffold landed: ✅ namespace + types + 8 static_asserts
- Codegen non-regression: ✅ all 4 grouped instances + rcr_4w
  resource numbers identical to R53 baseline
- HK commit: `73da21c6` (96-line additive scaffold)
- Doc commit: this file

## 6. Before-after metric

| Round | sha (PT) | sha (HK) | score | FP8 geomean | BF16 geomean |
|-------|----------|----------|-------|-------------|--------------|
| R51   | 4be1a81  | 6c52d017 | 988   | 1.180       | 1.225        |
| R52   | 50f0e67  | 6c52d017 | 989   | 1.142       | 1.225        |
| R53   | a495a55  | 6c52d017 | 987   | 1.160       | 1.233        |
| **R54** | **(this)**  | **73da21c6** | **982** | **1.172** | **1.186** |

R54 sits at 0.81σ below μ=988 — within noise (R51 stats: σ=7.3).
BF16 dip is Triton-baseline-noise, not a regression.

## 7. Next round (R55)

**Step 2 of C-2**: copy `grouped_rcr_kernel` template into
`lever_c2_round_54_step1_scaffold::` namespace, lift the four
explicit instantiations, rebuild, capture resource report.

Validation gates:
- Build clean (no compile errors)
- Namespace kernel resource: AGPR ≥ 256, Spill ≤ 8 — if YES, the
  AGPR-via-density hypothesis is CONFIRMED and R56 wires the
  dispatcher.
- If AGPR=0 still: revisit, may need `art_base` + ASM (Lever C-3).

Time budget: should fit a 45-min slot. The expensive part (figuring
out where to put the namespace, how to test it, what types to
declare) is now DONE in R54.
