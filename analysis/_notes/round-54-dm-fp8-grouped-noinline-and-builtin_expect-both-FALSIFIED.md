# Round 54-dm — FP8 grouped: two register-allocator hints both FALSIFIED

**Status**: 2 PROBES FALSIFIED / no commit on HK kernel
**Score before**: 957-960 noise band (3-run sample 957/958/959, R50-dm winner ceiling)
**Score after**:  957-960 noise band (revert)
**HK SHA**: clean (no commit; both probes reverted in same round)
**Round time**: ~30 min, 3 build cycles, 3 metric runs
**Auto-optimize round**: 26

---

## Goal

After R51/R52/R53 conclusively exhausted micro-knobs (vmcnt, lgkmcnt, K-tail
load order, setprio, half-barriers), looked for **register-allocator-hint
levers** to reduce the `<0, true, true>` spec's 39-dword spill (gpt_oss
spec) toward the `<0, false, true>` spec's 32-dword spill (DSV3 spec).

The 7-dword spill gap maps to ~10% of per-tile main-loop time on gpt_oss
(7 spills × 16 cy × 22 K-iters = ~2500 cy/tile of the ~5500 cy/tile budget),
plausibly explaining the ratio gap (gpt_oss 1.07 avg vs DSV3 1.16 avg, ~9pp).

Worst shape (selection driver):
```
gpt_oss-GateUP-B32-M4096   1.018-1.022   ← worst
gpt_oss-GateUP-B4-M4096    1.051
gpt_oss-Down-B32-M4096     1.040-1.068
```

All worst cases on gpt_oss → all use `<0, true, true>` spec (FUSED + n_masked).

---

## ISA spill profile snapshot (R54-dm baseline, build A)

`-Rpass-analysis=kernel-resource-usage` reports for `grouped_rcr_kernel`:

| Spec                          | TotalSGPRs | VGPRs | ScratchSize | Spill (dw) |
|-------------------------------|-----------:|------:|------------:|-----------:|
| `<0, false, false>`           | 65         | 256   | 160 B/lane  | 39         |
| `<0, true , false>`           | 71         | 256   | 176 B/lane  | 43         |
| `<0, false, true >` (DSV3)    | 79         | 256   | 132 B/lane  | **32**     |
| `<0, true , true >` (gpt_oss) | 83         | 256   | 160 B/lane  | **39**     |

All occupancy = 2 waves/SIMD. LDS = 139796 B/block. Compare:
- Dense `gemm_kernel` (RCR): VGPRs 256, Spill 8-24 (3 specs).
- `grouped_rrr_kernel` (RRR, dA-bwd, not on metric path): Spill 76.
- `grouped_var_k_kernel_fp8` (CRR, dB-bwd, not on metric path): Spill 52.

Dense FP8 spills 8-24 vs grouped FP8 spills 32-43. The +20-dword grouped
overhead comes from the persistent-loop bookkeeping (group binary search,
`s_offs`/`s_cum_tiles` LDS caches, group-by-M swizzle, per-group SRD
construction); R6 (`round-6-fp8-grouped-2tile-port-fail.md`) already
falsified the obvious "port dense's 2-tile main-loop body" attempt
(-144 score: gpt_oss preload window too tight at ki=22 with grouped's
longer prologue).

---

## Probe 1: `__attribute__((__noinline__))` on `store_c_tile_n_masked`

### Hypothesis

R59 already hoisted the `(bc + 1) * BLOCK_SIZE <= g.n` runtime branch
out of the helper into the kernel epilog (see kernel_fp8_layouts.cpp
line 2523). For interior tiles (~22/23 on gpt_oss N=5760), the bare
`store(...)` runs; the masked-helper only fires on the 1 boundary tile.

The +7 spill on `<0, true, true>` vs `<0, false, true>` is the helper's
inlined body inflating the register-allocator's live-range graph:
even if interior tiles never call the helper at runtime, LLVM still
plans for both paths and pessimistically allocates VGPRs.

Hypothesis: marking the helper `__attribute__((__noinline__))` would
move it to a separate function, freeing the caller's register-allocator
from having to plan its body. Cost: 1 function-call overhead per
boundary tile (1/12 on N=2880, 1/23 on N=5760). Estimated upside:
+1-2 pp on grp_FP8 if spill drops from 39 → 32.

### Result: BUILD FAIL (LLVM SGPR/VGPR ABI conflict)

```
kernel_fp8_layouts.cpp:536:32: error: invalid operand for instruction
  536 |             "s_mov_b32 m0, %0\n\t"
      |                                ^
<inline asm>:2:27: note: instantiated into assembly here
    2 |         buffer_load_dwordx4 v34, v[20:23], s5 offen lds
```

The error fires on `rcr_8w_load_hoist`'s inline asm at line 536:
```cpp
asm volatile(
    "s_mov_b32 m0, %0\n\t"
    "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
    :
    : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
    : "memory"
);
```
Operand `%0 = lds_off` has constraint `"s"` (SGPR) but LLVM allocates
it to a VGPR (`v34`) instead. Root cause: marking
`store_c_tile_n_masked` as `__noinline__` changes the function-call
ABI inside `grouped_rcr_kernel` — LLVM must spill caller-saved SGPRs
across the call site, and the readfirstlane'd `lds_off` is no longer
guaranteed to remain in SGPR class throughout main loop. The inline
asm's hard SGPR constraint cannot be satisfied.

### Falsification

**FROZEN: cannot mark `store_c_tile_n_masked` `__noinline__`** without
also relaxing `rcr_8w_load_hoist`'s inline asm SGPR constraints —
which would defeat its purpose (the SGPR hoisting is what saves
~30 cy/iter in main loop). The two optimisations are mutually
exclusive at current code structure.

A future round could attempt to move `rcr_8w_load_hoist` to use
`__builtin_amdgcn_raw_buffer_load_lds` (the LLVM intrinsic) instead
of inline asm — that bypasses the SGPR constraint problem but
historically (per the comments at line 451) is what motivated the
inline-asm approach in the first place (the intrinsic let LLVM
recompute m0 every iteration via VGPR; inline asm forecloses it).
Untangling that requires a full main-loop rewrite — out of scope
for a single round.

---

## Probe 2: `__builtin_expect((bc + 1) * BLOCK_SIZE <= g.n, 1)`

### Hypothesis

If we cannot move the helper to a separate function (Probe 1 failed),
the next-best register-allocator hint is to mark the bare-store branch
as the hot path. `__builtin_expect(cond, 1)` biases:
- branch prediction (relayed to hardware via instruction layout)
- LLVM's register allocation (hot path's live-range graph weighted higher)

Hypothesis: the hot bare-store path would win the alloc tournament,
displacing cold-path state to scratch but keeping the hot path lean.
Net spill on the hot path may go DOWN.

### Result: SPILL WORSENED (39 → 41 on `<0, true, true>`)

Build comparison (only n_masked specs change; n_aligned specs are
constant since `if constexpr (N_MASKED_STORE)` keeps the branch
compile-time dead for `<0, false, *>`):

| Spec                          | Before | After | Δ      |
|-------------------------------|-------:|------:|-------:|
| `<0, false, false>`           | 39     | 39    | 0      |
| `<0, true , false>`           | 43     | 43    | 0      |
| `<0, false, true >`           | 32     | 32    | 0      |
| `<0, true , true >` (gpt_oss) | 39     | **41** | **+2** |

ScratchSize `<0, true, true>`: 160 B → **168 B** (+8 B).

The hint actually made it WORSE: LLVM took the cold-path-displacement
suggestion and spilled MORE state to scratch, but the displaced state
IS still alive on the hot path (across the if-else fallthrough into
the `s_waitcnt lgkmcnt(0)` + `s_barrier` at line 2569). So additional
scratch traffic happens on every tile (boundary AND interior), not
just boundary tiles.

### Root cause

The `if (cond) { hot } else { cold }` pattern with a compile-time
hint biases reg-alloc, but on FP8 grouped with already-saturated
VGPR pressure (256/256 lane regs), there's no slack: any state
forced toward "cold" must spill, and EVERY tile pays the spill cost
regardless of branch outcome (the spill happens at PHI nodes when
control flow rejoins after the branch).

`__builtin_expect` works for ICache layout (cold path placed further
away → fewer cache lines touched on hot path) but DOESN'T help when
register pressure is the limit, not instruction fetch.

### Falsification

**FROZEN: `__builtin_expect` hint on the hot bare-store branch
WORSENS spill** (+2 dw on the gpt_oss spec). Reverted immediately.

The reverse hint (`__builtin_expect(cond, 0)`) would only flip the
preferred branch — wouldn't reduce TOTAL spill since the cold path
is intrinsically larger (helper body inline). Not tested; symmetry
argument predicts +2 dw the other direction.

---

## Cumulative state after R54-dm

### Lever taxonomy (current understanding)

```
Lever A (async global→LDS copy + MFMA pipelining)
  STATUS: ALREADY SHIPPED via rcr_8w_load_hoist (line 464+) using
  inline asm `buffer_load_dwordx4 ... offen lds`. Direct HBM→LDS
  bypassing register staging.

Lever B (Dual / triple LDS buffer ping-pong)
  STATUS: ALREADY SHIPPED via As[2][2] / Bs[2][2]. Triple buffer
  blocked by LDS capacity (current 139.8 KB / WG; gfx950 LDS cap
  is 160 KB / WG; triple would need ~210 KB).

Lever C (register usage hints)
  STATUS: SATURATED — multiple sub-probes falsified:
    - `__noinline__` on n_mask helper: ABI conflict with main-loop asm
    - `__builtin_expect(hot)`: makes spill WORSE
    - `__forceinline__` (current): baseline 39 spill on gpt_oss
    - drop K-tail a_kt1 (R47-dm): -6 pts (lost overlap)
    - dummy register R34-pattern (R38-dm): falsified
    - cluster prologue reorder (R42-dm): half-barrier sync break

Lever D (rt_32x64 / rt_64x32 cell shape, 32x32x64 mfma)
  STATUS: UNFALSIFIED, only remaining structural option.
  Per R52/R53 recommendation: Round-A is K-tail-only port
  (4× 16x16 mfma → 2× 32x32 mfma, K=64 native = 100% util).
  Scaffold ready in HK commit 96a84c08 (rt_32x64 + FP8 32x32x64
  mma_AB_base branch, mma.cuh line 173-185).
  Estimated: 2-3 round commitment.

Lever E (manual ASM main loop scheduling)
  STATUS: HIGH RISK, last resort.
```

### Score plateau

```
R44 (won): rocprof-guided interleave mul/store          948 → 956 (+8)
R50 (won): drop end-of-tile vmcnt(0) drain              956 → 961 (+5)
R51 (lost): VMCNT main-loop INIT0/INIT1 sweep           961 → 959
R52 (lost): wm==0 / M2N2 / lgkmcnt drops                959 → 959
R53 (lost): K-tail load reorder + setprio                959 → 959
R54 (lost): noinline + __builtin_expect both falsified  959 → 959
```

Five consecutive rounds (R51, R52, R53, R54) at 957-960 noise band.

---

## Recommended next round

R55-dm (the auto-optimize round 27) should COMMIT to **Lever D
Round-A — K-tail-only 32x32x64 mfma port**:

### Plan

1. Define `cAB_32 = rt_fl<RBM=64, RBN=32, col_l, rt_32x32_s>` accumulator
   type (covers same warp output as 4× rt_16x16, but with 2 sub-tiles
   instead of 8).

2. Implement `rcr_mma_32(cAB_32, a, b)` → `mma_ABt(...)` using existing
   `mma_ABt_base<rt_32x32, ..., 32x32x64>` branch (already scaffolded
   at mma.cuh line 234-238).

3. In the K-tail block ONLY (lines 2285-2469), replace the current
   pattern:
   ```
   load_b_kt(b0, 0); load_b_kt(b1, 1);
   load_a_kt(a, 0); load_a_kt(a_kt1, 1);
   vmcnt(8); rcr_mma(cA, a, b0); rcr_mma(cB, a, b1);
   vmcnt(0); rcr_mma(cC, a_kt1, b0); rcr_mma(cD, a_kt1, b1);
   ```
   with:
   ```
   load_b_kt_32(b0_32, b1_32);  // 32x32 reg layout
   load_a_kt_32(a_32);           // single 32x64 reg, K=64 native
   vmcnt(0); rcr_mma_32(cAB_32_0, a_32, b0_32);
              rcr_mma_32(cAB_32_1, a_32, b1_32);
   ```
   (4 mfmas → 2 mfmas, K=64 native = 100% util, +1 mfma latency saved)

4. Fan out `cAB_32` results into existing `cA`, `cB`, `cC`, `cD` via
   register-level rebind (no LDS round-trip; `rt_32x32` and `rt_16x16`
   share the same lane-cell mapping for FP32 acc — verify this).

5. Verify correctness on gpt_oss-GateUP-B32-M4096 first (worst case,
   K-tail fires every tile).

### Expected upside

K-tail block is currently ~5-10% of per-tile time on gpt_oss
(R45-dm note: ~314 cy of ~2112 cy on Down-B32-M2048). Halving K-tail
mfma count (4 → 2) and going to 100% mfma util (50% → 100%) saves
~32 cy/tile. On 256 tiles/CU that's ~8K cy = 4 us = ~0.2% of kernel
runtime. Modest but consistent +1-2 pp on gpt_oss FP8 ratio.

### Risk

Per R6 §5: "the 2-tile body itself is sound (correctness PASSed; just
slow)". Lever D is a different change (mfma cell shape, not loop
unrolling). Risk is medium: register-tile layout rebind between
rt_32x32 and rt_16x16 must be exact to share data via register copy
(no LDS). If wrong, fwd-correctness fails immediately.

If Round-A succeeds, Round-B extends to main loop (more challenging:
LDS swizzle, ds_read patterns, vmcnt scheduling all need rework).

### Acceptance: only commit Round-A if:
- All 16 FP8 fwd correctness PASS (no NaN / Inf / SNR<22dB)
- gpt_oss geomean ≥ 1.07 (current; no regression)
- Total score ≥ 959 (preserve plateau)

If any of those fail, REVERT and file Lever D as falsified at this
fidelity. Then R56-dm explores Lever C sub-options (e.g., AGPR
migration per R8-dm note: scope = same as full kernel rewrite,
estimated 4-8 hr but unfalsified).

---

## Files touched (R54-dm)

- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (HK):
  modified then reverted (Probe 1 + Probe 2 both undone)
- `analysis/_notes/round-54-dm-fp8-grouped-noinline-and-builtin_expect-both-FALSIFIED.md` (PT, this file)

## Repo state at end of round

- HipKittens: HEAD `6a93fa32` (R50-dm winner; unchanged)
- Primus-Turbo: HEAD will advance after this doc-only commit
- Score: 957-960 noise band; no regression, no improvement
