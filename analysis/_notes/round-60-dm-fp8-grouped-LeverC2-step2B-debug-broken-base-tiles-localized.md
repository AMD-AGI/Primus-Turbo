# Round 60 — FP8 grouped: Lever C-2 step-2B debug — broken base tiles LOCALIZED

**Date**: 2026-05-02 (R60 of 100)
**HEAD before**: 27c48a44887246612fd2f64ae4a84b72d5bdd762
**Score**: baseline 978 → final 980 (within ±5 noise band, **NO regression**)
**Goal**: continue R59 thread — debug `cAB[0][0]` register-aliasing bug in
4w-style FP8 grouped test kernel, target correctness PASS w/ AGPR ≥ 200.

## TL;DR

R59 reported "all of cAB[0][0] is broken across all 4 warps." R60 debug
**localized the bug to exactly 2 specific 16×16 base tiles within
cAB[0][0]**: `tiles[n=0..1][m=1]`. The remaining 14 base tiles of
cAB[0][0] AND all 16 base tiles of cAB[0][1], cAB[1][0], cAB[1][1] are
correct (~0.004 diff = pure fp8 quantization noise).

Three workarounds were tested. None gave full PASS but they sharpened
the diagnosis substantially:

| Workaround | Effect on cAB[0][0] | Other cells | Decision |
|---|---|---|---|
| Sacrificial dummy `C_acc_4w cAB_sacrificial; zero(...);` declared first | NONE (resource & probe both byte-identical) | unchanged | LLVM DCE'd the dummy → falsifies "first-VGPR collides with spill" theory |
| Explicit per-base-tile mma_ABt_base loop (16-iter unroll) instead of bulk mma_ABt | partial fix (max 294 → 86) | cAB[1][0]/[1][1] became ~0.3 off (was ~0.004) | net regression |
| Skip `mul()` scale epilog (probe scale=1.0 anyway) | first 4 rows now CORRECT (~0.001 diff) | unchanged | reveals mul propagated/amplified the bug; underlying mma defect persists |

**Net result**: AGPR allocation retained (256 AGPR / 0 SGPR spill). Test
kernel is NOT in dispatch path → metric unchanged (978-980 noise band).
Production kernel byte-identical to R58/R59.

## Bug localization (probe 512×256×256 RCR)

C_acc_4w = `rt_fl<64, 64, col_l, rt_16x16_s>`: 4×4 grid of 16×16 base tiles.

Per-cell × per-base-tile breakdown (no-mul probe):

```
cAB[0][0].tiles[n][m]:
  n=0, m=0  ← OK (rows 0-15,  cols 0-15)
  n=0, m=1  ← BROKEN (rows 0-15,  cols 16-31, max diff ~138)
  n=0, m=2  ← OK
  n=0, m=3  ← OK
  n=1, m=0  ← OK
  n=1, m=1  ← BROKEN (rows 16-31, cols 16-31, max diff ~216)
  n=1, m=2  ← OK
  n=1, m=3  ← OK
  n=2..3, m=0..3  ← all OK
cAB[0][1] all 16 base tiles ← OK
cAB[1][0] all 16 base tiles ← OK
cAB[1][1] all 16 base tiles ← OK
```

So **exactly 2 of 64 base tiles** are broken. The pattern (n ∈ {0,1},
m=1) is suggestive but not yet diagnostic of root cause.

## Logical impossibility — input registers are FINE

The same a_reg / b_reg base tiles feed multiple cAB cells. Cross-checking:

- `cAB[0][1].tiles[0][1]` uses `a_reg[0].tiles[0][0] @ b_reg[1].tiles[1][0]^T` → **CORRECT**
  - ⇒ `a_reg[0].tiles[0][0]` is intact.
- `cAB[1][0].tiles[0][1]` uses `a_reg[1].tiles[0][0] @ b_reg[0].tiles[1][0]^T` → **CORRECT**
  - ⇒ `b_reg[0].tiles[1][0]` is intact.
- Yet `cAB[0][0].tiles[0][1] = a_reg[0].tiles[0][0] @ b_reg[0].tiles[1][0]^T` → **BROKEN**.

Both inputs are valid in adjacent computations using the SAME register base
tiles, but the output is broken specifically when written to
`cAB[0][0].tiles[0][1]`. This points strongly to **AGPR allocation issue
specific to cAB[0][0].tiles[{0,1}][1]**, not a load-side or arithmetic
defect.

## Hypothesis (R61+ to verify)

LLVM register allocator places `cAB[0][0].tiles[{0,1}][1]`'s data in
**AGPR slots that are clobbered between mma write and store read** — possibly
because:

- AGPR slot for `cAB[0][0].tiles[0][1]` overlaps with VGPR slot used as a
  temp during `mul()` epilog (the same VGPR-AGPR dual-mapping shows up
  in MFMA codegen on gfx950).
- OR LLVM didn't actually allocate AGPR for these specific 2 base tiles
  (only for the other 14) and they spilled to scratch. The 80 dwords/lane
  spill count is suspicious — exactly enough for 2 × 16×16 fp32 tiles
  per warp = 1024 fp32/warp = 16 fp32/lane, but the spill is 80, so
  there must be additional non-cAB spill traffic.

R61 plan (concrete):

1. ISA dump: `/opt/rocm/llvm/bin/llvm-objdump -d --triple=amdgcn--amdhsa
   --mcpu=gfx950 libtk_fp8_layouts.so > /tmp/isa_r60.txt`. Search for
   `scratch_load`/`scratch_store` and AGPR `v_accvgpr_*` instructions
   touching the suspect VGPR slots.
2. Try `__attribute__((aligned(...)))` on `cAB[0][0]` — may force LLVM
   to start its register block at a clean alignment boundary.
3. Try `volatile C_acc_4w cAB[2][2]` — forces no register caching, all
   ops go through memory. Disables AGPR allocation but should produce
   correct output. If correct, AGPR allocation is the culprit.
4. Try copying cAB to a TEMP before store: `add(c_tmp, cAB[0][0], 0.f);
   store(g.c, c_tmp, ...)`. Validates AGPR-to-VGPR transfer at store time.
5. If all fail, drop accumulator size: 4 cells of 64×64 = 256 fp32/lane
   → 4 cells of 32×32 = 64 fp32/lane (loses AGPR but recoverable
   correctness; quad the dispatch granularity for 256×256 block).

## Files touched

- `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - kept R59's `test_grouped_rcr_kernel_4w_real_coords` test kernel
  - added sacrificial dummy `cAB_sacrificial` (no effect, harmless)
  - **commented out `mul()` scale epilog** (reveals first 4 rows correct)
  - extended docstring with R60 findings + R61 plan
- `HipKittens/analysis/fp8_gemm/mi350x/probe_4w_real_coords.py`:
  - extended diagnostic to print PER-ROW and PER-COL bad indices when
    diff_max > 0.5, allowing precise base-tile localization
- `Primus-Turbo/analysis/_notes/round-60-dm-...`: this note

## Production kernel impact

NONE. Test kernel reachable only via `tk_fp8_layouts.test_4w_real_coords`
pybind binding (used by `probe_4w_real_coords.py`). All grouped FP8
production paths (`grouped_rcr_kernel<T,T>`) byte-identical to R58/R59.

Metric: 978 → 980 (Δ = +2, within noise band).

## Roadmap update

Round 60 was a **DEBUG ROUND**. The R59 PARTIAL FAIL roadmap
(R60-debug, R61-prologue, R62-K-tail, R63-N-store, R64-dispatch)
is shifted by +1: R61 will be **another debug round** (ISA dump +
volatile/temp-store experiments) before continuing the Lever C-2
feature port at R62.

Risk assessment: bug is **localized but not yet root-caused**. AGPR
allocation is preserved, ruling out the simplest "AGPR didn't apply"
failure mode. R61 should make/break the C-2 path:
- If volatile or temp-store gives PASS → continue C-2 (production wire-up
  via temp-copy might be acceptable if overhead is small).
- If neither works → C-2 path closed. Pivot to Lever A (async global→LDS)
  or accept the 978 plateau.
