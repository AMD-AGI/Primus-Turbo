# Round 73 — BF16 grouped bwd var-K occupancy audit FALSIFIED (already at max)

**Status:** FALSIFIED — `grouped_var_k_kernel` is already at max HW occupancy
(2 waves/SIMD on CDNA4). No room to tune without VGPR reduction.
**Score:** baseline 875 (unchanged, no code change committed).

## R73 Hypothesis (from R72 direction note P3 / B2 lever)

From R72 closeout note's R73 direction priority list:
- P1: var-K CRR LDS swizzle (unpadded -> padded) — expected +10-15 score
- P2: FUSE pipeline prefetch — expected +5-10 score (VGPR risk)
- P3: `__launch_bounds__` tuning for large-K kernels — expected +3-5 score

Given the chat-window budget (R73 opened at 81/90 min of resume window),
the attempt was P3 as the lowest-risk / fastest-verify lever.

## Audit

Built `kernel_bf16_dynamic.cpp` with `-Rpass-analysis=kernel-resource-usage`
and grepped the var-K instantiation report:

```
kernel_bf16_dynamic.cpp:4717:1: remark: Function Name: _Z20grouped_var_k_kernelILi0EEv28grouped_var_k_layout_globals
kernel_bf16_dynamic.cpp:4717:1: remark:     TotalSGPRs: 95
kernel_bf16_dynamic.cpp:4717:1: remark:     VGPRs: 256
kernel_bf16_dynamic.cpp:4717:1: remark:     AGPRs: 0
kernel_bf16_dynamic.cpp:4717:1: remark:     Occupancy [waves/SIMD]: 2
kernel_bf16_dynamic.cpp:4717:1: remark:     SGPRs Spill: 0
kernel_bf16_dynamic.cpp:4717:1: remark:     VGPRs Spill: 0
kernel_bf16_dynamic.cpp:4717:1: remark:     LDS Size [bytes/block]: 272
```

The kernel declares `__launch_bounds__(NUM_THREADS, 1)` at line 4716,
asking the compiler for AT LEAST 1 wave/SIMD. The compiler auto-derives
**2 waves/SIMD** based on the VGPR=256 and SGPR=95 budget. CDNA4 per-SIMD
VGPR pool is 512, so 256 VGPR × 2 waves = 512 = max. Any attempt to push
to 3 waves/SIMD would require **VGPR <= 170** per wave.

Current kernel VGPR=256 is at the per-wave cap (512 VGPR/SIMD / 2 waves).
Lowering to `__launch_bounds__(NUM_THREADS, 3)` would force register spill
(observed ~60-100 VGPR drop needed, guaranteed scratch reads).

## Why VGPR is so high

- 4× C accumulator tiles (`C_accum[2][2]` of `rt_fl<..., rt_16x16_s>`) —
  16 fp32 per lane × 4 tiles = 64 VGPRs
- 4× double-buffered A/B LDS slot tracking (a_lds_{00,01,10,11},
  b_lds_{00,01,10,11}) — 8 SGPRs (hoisted to uniform) but each tile
  load consumes working VGPRs for HBM→reg staging
- Per-tile MMA working set for RCR-style CRR load path
- Chiplet swizzle scalars, group-offset LDS stash cache, per-tile K-offset
  bookkeeping

Reducing VGPR would require either:
1. Dropping C_accum from 4 tiles to 2 (halves M-axis register tiles,
   would need 2-pass C write — large restructure).
2. Using AGPRs for C_accum (CDNA4 supports AGPR->VGPR swap on mfma
   output). Hipkittens core doesn't currently expose this knob.

Neither is a 1-round change.

## R73 outcome

**No commit attempted.** The P3 lever is closed:
- `__launch_bounds__(NUM_THREADS, 1) -> (NUM_THREADS, 2)` is already the
  compiler's derived occupancy (no change).
- `(NUM_THREADS, 3)` forces spill (regression).

## R74 direction (still from R72's P1/P2)

P1 and P2 remain the actionable levers:

**P1 (primary, +10-15 expected):** Swap `grouped_var_k_kernel`'s ST shape
from `st_32x16_s` (unpadded) to `st_64x32_padded_b128_s` (padded). At
`kernel_bf16_dynamic.cpp:4723-4724`:

```cpp
// Current:
using ST_A = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
using ST_B = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
// Target:
using ST_A = st_bf<K_STEP, HALF_BLOCK_SIZE, st_64x32_padded_b128_s>;
using ST_B = st_bf<K_STEP, HALF_BLOCK_SIZE, st_64x32_padded_b128_s>;
```

Risks:
- Shape dimension mismatch: `st_64x32_padded_b128_s` expects a 64x32
  sub-tile; current shape is `K_STEP=64 × HALF_BLOCK_SIZE=128` (row x col).
  The padded shape's bank mapping is for (64, 32) so the 128-wide dim
  needs to be addressed via 4 sub-tiles. Check that st_bf's internal tile
  subdivision supports this.
- `A_reg_t` / `B_reg_t` are `rt_32x16_s col_l`. The `shared_to_register.cuh`
  Path B for `st_64x32_padded_b128_s` -> `rt_32x16_s col_l` is listed in
  R69 note as "compiles only, not semantically verified". Numerical probe
  required before metric run.

**P2 (fallback, +5-10 expected):** In the FUSE K-tail epilog
(`kernel_bf16_dynamic.cpp:808-946`), reorder slab-0 vs slab-1 load/MMA
sequence to overlap slab-1 A-HBM-load with slab-0 MMAs. Needs a second
A-tile register — risk of pushing FUSE VGPR over the spill threshold
(currently at KI=44 which is close to the threshold per R67).

## No code change this round

HK + Primus working trees clean. Baseline 875 verified.

## Chat-window note

This chat session will likely wrap within 1-2 more rounds (currently at
85-86 min of 90-min resume window). R74 opens cold-start; the next
agent must re-derive context from this note and the R72 direction note
to continue P1/P2.
