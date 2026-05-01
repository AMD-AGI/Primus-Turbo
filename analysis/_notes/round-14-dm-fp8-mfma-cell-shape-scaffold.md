# Round-14-dm — FP8 MFMA cell-shape `16x16x128` → `32x32x64` scaffolding (multi-round step 1/3)

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `564252e` (round-13-dm post-MFMA barrier + setprio falsified)
**HipKittens HEAD before**: `25386036` (round-12-dm K-tail split-vmcnt)
**HipKittens HEAD after**: this commit (scaffold-only — adds 2 rt_shape types + 1 mma_AB_base dispatcher branch; zero behaviour change)
**Primus-Turbo HEAD after**: this commit (docs only)

**Metric**: 825 baseline mean → 822/830/822 = 825 post-scaffold (identical, scaffold is dead code).

---

## Round target (per the round-14 prompt rule)

Lowest FP8 ratio shape after round-13-dm = **`grpFP8_DeepSeek-V3-Down-B16-M2048` at
0.952** (HK 1151.6 TF / Triton 1209.1 TF). All 4 DSV3-Down shapes
(0.952–1.009) still dominate — they're K=2048 N=7168, fully aligned
(no FUSED_KTAIL, no N-tail), pure main-loop bound. ki_dyn=16 means the
prologue/epilogue costs amortise over only 16 K-iters (vs 56 for
DSV3-GateUP K=7168 and 22 for gpt_oss K=2880), inflating per-tile
overhead's relative impact.

Round-13-dm closed out the last 2 untried single-knob main-loop levers
(post-MFMA `s_barrier` removal, `s_setprio(0)` removal). Both
falsified by inter-wave LDS-coherence and 2-wave co-occupancy
priority-handoff mechanisms. **All single-knob levers in
`grouped_rcr_kernel` are now exhausted.**

Round-13-dm's recommended round-14 work was **option 1: FP8 MFMA
cell-shape `mfma_f8f6f4_16x16x128` → `mfma_scale_f32_32x32x64_f8f6f4`**.
This is a 2-3 round structural project that directly attacks the
mechanism the round-13-dm probes failed to exploit via barrier removal:

- HK FP8 main-loop body has 8 `s_barrier`s per K-iter for 4 *short*
  ~32-cyc 16x16x128 MFMAs ⇒ MfmaUtil ≈ 33.6 %.
- Triton uses `mfma_scale_f32_32x32x64_f8f6f4` (~64 cyc/MFMA, 2× the
  latency per MFMA) — same ~8 barriers / K-iter amortise over 2× the
  MFMA cycles ⇒ MfmaUtil ≈ 41.9 % (round-16 PMC).
- The barriers cannot be removed (round-13-dm proved both inter-wave
  LDS-coherence and co-occupant scheduling are load-bearing); the
  only remaining lever is to make each MFMA **cover more useful
  work** so the same overhead amortises further.

Round 14 = **scaffold step 1**: shape-type + dispatcher infrastructure.

## What's in this commit (HipKittens)

Two minimal additions, **no behaviour change** to any existing kernel:

### 1. `include/types/register/rt_shape.cuh` — added `rt_32x64` and `rt_64x32`

```diff
+ using rt_32x64 = rt_shape<32, 64, 16>;
+ using rt_64x32 = rt_shape<64, 32, 16>;
```

Stride 16 mirrors the existing FP8 family (`rt_16x128 = rt_shape<16, 128, 16>`,
`rt_128x16 = rt_shape<128, 16, 16>`):

| shape       | rows × cols | num_elements | elements_per_thread (×4 lanes/SIMD32×2 SIMD/wave) | stride | num_packed (fp8e4m3_4) | packed_per_stride | num_strides |
|---          |---          |---           |---                                                 |---     |---                     |---                |---          |
| rt_16x128   | 16 × 128    | 2048         | 32                                                 | 16     | 4                      | 4                 | 2           |
| rt_128x16   | 128 × 16    | 2048         | 32                                                 | 16     | 4                      | 4                 | 2           |
| **rt_32x64**| 32 × 64     | 2048         | 32                                                 | 16     | 4                      | 4                 | 2           |
| **rt_64x32**| 64 × 32     | 2048         | 32                                                 | 16     | 4                      | 4                 | 2           |

Identical pack ratio and lane storage volume to the 16x128 / 128x16 path
(`fp8e4m3_4 data[8]` per lane → `intx8_t` per lane, exactly what
`mfma_scale_f32_32x32x64_f8f6f4` consumes per A and B operand). The
ducks::rt_shape::all concept is widened to admit them, plus `transpose<>`
specialisations are added so the ABt operand commutation works.

### 2. `include/ops/warp/register/tile/mma.cuh` — added FP8 32x32x64 case to `mma_AB_base`

```diff
+ } else if constexpr (std::is_same_v<MM_Operand_T, fp8e4m3> &&
+               std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> &&
+               A_rows == 32 && A_cols == 64 &&
+               B_rows == 64 && B_cols == 32 &&
+               std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
+     mfma323264(d.data, a.data, b.data, c.data);
```

The corresponding ABt branch (D=rt_32x32, A=32x64, B=32x64 row-major)
**already existed** at lines 220-224 of `mma.cuh` but was unreachable
because `rt_32x64` was missing from `concept all` — adding the
shape types in step (1) unblocks it. No new asm-level case is needed
in this commit; the underlying `mfma323264` builtin wrapper has been
in `mma.cuh:97-110` since round-… (pre-FP8-grouped, used by
`kernels/torch_scaled/scaled_matmul.cu`).

### Verification

- Clean rebuild of `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  occupancy = 2 waves/SIMD (unchanged), VGPR spill = 52 (unchanged for
  the FUSED_KTAIL × N_MASKED variant). The new dispatcher branch is
  unreachable from current FP8 grouped call sites (which still pass
  `A_row_reg = rt_fp8e4m3<RBM=64, BK=128, row_l, rt_16x128_s>` →
  matches the `rt_16x16` D-shape path), so resource usage is bit-
  identical.
- `_metric_grouped_only.py` 3-run re-test post-scaffold:
  `runs = [822, 830, 822], mean = 825`, matching the round-13-dm
  baseline mean of 826. **No metric movement** (expected — scaffold
  has no live caller yet).

## What's left for round 15+ (cell-shape migration plan)

The remaining work is contained in `kernel_fp8_layouts.cpp`'s
`grouped_rcr_kernel` (line 1973) and the helper subroutines feeding
its main-loop section. No core kittens type/op machinery needs
further extension after this round.

### Round 15 — type-flip + load-path re-derivation (isolated, single section)

Migrate **just the main-loop section** (not init, not epilogs, not
FUSED_KTAIL) to use the new shapes. Steps:

1. Add a separate `A_row_reg_32` / `B_row_reg_32` typedef in
   `kernel_fp8_layouts.cpp` lines 92-93:
   ```cpp
   using A_row_reg_32 = rt_fp8e4m3<RBM, BK, row_l, rt_32x64_s>;
   using B_row_reg_32 = rt_fp8e4m3<RBN, BK, row_l, rt_32x64_s>;
   ```
   Keep the existing 16x128 typedefs alive — coexistence makes the
   diff reviewable and lets a single section migrate at a time.

2. Add a `rcr_mma_32` overload that takes the new register tile types
   and dispatches to the 32x32x64 branch.

3. Re-derive the LDS → register lane mapping in `rcr_8w_load_hoist`
   for the 32x64 input. The 32x32x64 MFMA has a different lane-cell
   distribution than 16x16x128:
   - **16x16x128** A: 2 cols × 16 lanes contribute the row-strip; each
     lane holds 32 fp8 across 4 col-groups of 8 fp8 each (stride=16,
     num_strides=2).
   - **32x32x64** A: 4 cols × 16 lanes; each lane holds 32 fp8 across
     2 row-groups of 16 fp8 each (stride=16, num_strides=2).
   The current `rcr_8w_load_hoist` (lines 343+) uses the 16x16x128 lane
   mapping; round 15 must add a parallel `rcr_8w_load_hoist_32x64`
   that uses the 32x32x64 mapping, keyed by a constexpr template
   parameter.

4. Re-derive the `St_subtile` 4-lane HW transpose layout (used by
   `load_a` / `load_b` in main-loop body, lines 2124-2131): the LDS
   tile's `subtile_inplace` returns a sub-region of LDS that maps to
   the full register tile via the SQ's strided LDS read pattern. With
   `rt_32x64` instead of `rt_16x128`, the subtile slice changes from
   `<RBM=64, BK=128>` (a single 64x128 chunk) to `<RBM=64, BK=128>`
   *with cells of 32x64* — same total slice, different cell decomp.
   The `ds_read_b128` access pattern (currently 2 strides per lane,
   loading 32 fp8) is preserved at this step (stride parity).

5. Per-section migration: replace the 4 `rcr_mma(cA/cB/cC/cD, a, b)`
   calls in the main-loop body (lines 2181, 2188, 2195, 2200) with
   `rcr_mma_32(...)` calls operating on the new types. Verify SNR on
   `tests/pytorch/ops/test_grouped_gemm_fp8.py` — must pass before
   committing.

6. Deferred to round 16: migrate the init / Epilog 1 / Epilog 2 / FUSED_KTAIL
   blocks. Until they migrate, the kernel runs in **mixed mode** (init
   + epilog use the 16x16x128 cells, main loop uses 32x32x64). Round
   13-dm proved that LDS coherence is per-section-bounded (the
   pre-MFMA barriers gate cross-section LDS races), so the main-loop
   sub-tile shape can differ from the init/epilog shape as long as
   the LDS slot contents remain the same — which they do, since the
   cell-shape change is a register / lane mapping change, not an LDS
   write change.

7. Run `_metric_grouped_only.py` baseline. Estimated +5-15 score on
   DSV3-Down (the 4 most-affected shapes) if MfmaUtil rises from
   33.6 % toward 41.9 % (Triton's level). Other shapes (gpt_oss
   FUSED_KTAIL, GateUP, etc.) get only the main-loop benefit, no
   K-tail or epilog change ⇒ smaller delta.

### Round 16 — extend to init / Epilog 1 / Epilog 2 + FUSED_KTAIL block

- Migrate init section (lines 2142-2156) to use the 32x64 types if
  beneficial — these are pure HBM→LDS prefetches, no MFMA, so the
  cell-shape change is moot. Skip unless the load-hoist's lane
  mapping must align with the main-loop's new mapping for LDS
  consistency.
- Migrate Epilog 1 (lines 2204-2232) and Epilog 2 (lines 2234-2256).
  Same load_a / load_b / `rcr_mma` substitutions.
- Migrate FUSED_KTAIL block (lines 2425-2456). This block uses
  per-element `buffer_load_b128` + `rcr_mma` — straightforward to
  re-thread.

### Round 17 — clean up, drop the 16x128 types

Once the migration is complete and verified, remove `A_row_reg`,
`B_row_reg`, `rcr_mma` (the 16x16 versions), and the 16x128 lane
mappings. This is the same pattern as round-26 / 27 deprecation
clean-ups for the dense FP8 path.

## Risks & mitigations

1. **Stride mismatch on rt_32x64**: my best estimate is `stride=16`
   (matches FP8 16x128 / 128x16). If the AMD CDNA4 lane mapping for
   `mfma_f32_32x32x64_f8f6f4` actually expects a different stride (e.g.
   8 for tighter packing), `mfma323264` will read mis-aligned data and
   produce SNR garbage. Mitigation: round 15 step 3 derives the
   actual mapping from the AMD CDNA4 ISA reference; if stride=16 is
   wrong, change it.

2. **Per-section migration breaks something cross-section**: the
   round-13-dm probes proved that LDS slot contents survive
   cross-section provided pre-MFMA barriers stay. Mixed-mode
   (16x128 init / 32x64 main) is theoretically sound but untested.
   Mitigation: round 15 step 7 runs the full FP8 grouped test suite,
   gated on SNR ≥ 25 dB on every shape.

3. **The 32x32x64 MFMA might not actually beat 16x16x128 on MI355X**:
   the round-25 doc and the round-16 PMC both *conjecture* MfmaUtil
   improvement; nobody's measured it on this exact code path.
   Mitigation: round 15 includes a single-section perf measurement
   *before* committing to the full migration. If MfmaUtil doesn't
   move ≥ 2 pp on the worst shape, abandon the project (revert the
   scaffold + main-loop change).

## Files / commits

* HipKittens: this commit —
  - `include/types/register/rt_shape.cuh` (+15 lines: `rt_32x64`,
    `rt_64x32`, `concept all` widening, `transpose<>` specs).
  - `include/ops/warp/register/tile/mma.cuh` (+12 lines: FP8 32x32x64
    `mma_AB_base` branch).

* Primus-Turbo: this commit —
  `analysis/_notes/round-14-dm-fp8-mfma-cell-shape-scaffold.md` (this
  file).

Self-bench: not required (no backward path touched, no callable kernel
change — both new dispatcher branches are unreachable from current
`A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>` call sites; the
round-15 main-loop migration commit will be the first to exercise
the new branch).
