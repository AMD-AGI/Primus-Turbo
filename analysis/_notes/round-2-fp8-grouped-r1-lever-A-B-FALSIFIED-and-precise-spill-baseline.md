# Round 2 — FP8 grouped: R1 Lever A/B FALSIFIED + precise spill baseline + R3 Lever C plan

**Status**: NO KERNEL CHANGE — R2 = data collection + falsify R1 lever
roadmap + redirect to evidence-based Lever C in R3
**Auto-optimize round**: 2 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `ecbead9a` (unchanged)
**PT SHA at round start**: `771d7d58`
**Round time**: ~25 min (2 metric runs + 1 build + ASM/spill probe)
**Score before**: 958 (R1 baseline)
**Score after**: 960 (within ±2 noise)

---

## R2 metric (sanity check on baseline persistence)

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1871 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1181 (n=24)
[metric_grouped_only]   weighted_progress=0.9601  correct_fail=0/48  reject=0/48
[metric_grouped_only]   below_target=38/48  goals=0/2  score=960
```

R1 said 958, R2 says 960 — within ±2 noise band. **Worst 5 cases unchanged**:
1. gpt_oss-GateUP-B32-M4096 1.024
2. gpt_oss-GateUP-B32-M2048 1.054
3. gpt_oss-GateUP-B4-M4096  1.059
4. Qwen3-Down-B16-M4096    1.056
5. gpt_oss-Down-B32-M2048  1.067

(Same shapes, same rank order, ±0.02 ratio noise.)

---

## R1 Lever A FALSIFIED — async global→LDS copy is already shipped

R1 round-note hypothesised Lever A ("Async global→LDS copy + MFMA pipelining,
+5..10pp expected") based on the empirical observation that
`objdump | grep global_load_lds` returned 0 hits in the compiled `.so`.

**That observation was wrong** because:
1. The `objdump` binary segfaulted partway through dumping `tk_fp8_layouts.so`
   (only 35K lines of ASM written before crash; full kernel ASM is far longer
   given 4 grouped_rcr_kernel template specs + 2 grouped_rrr + 2 grouped_var_k
   + 6 K-tail kernels). My grep operated on a TRUNCATED dump.
2. **Even if grep had completed**, the LLVM intrinsic translation for
   `llvm.amdgcn.raw.buffer.load.lds` on gfx950 emits the mnemonic
   `buffer_load_dwordx4 ... offen lds`, NOT `global_load_lds_*`. So a
   text grep for `global_load_lds` would never find the hits I expected.

**The actual evidence**: `kernel_fp8_layouts.cpp:714-819` defines
`rcr_8w_load_hoist`, the load helper used by `grouped_rcr_kernel` for both
prologue (lines 2392-2403) and main loop (line 2435 / 2442 / 2448). Its
core (line 785-790) is **inline ASM emitting exactly the async global→LDS
intrinsic Lever A wanted to introduce**:

```cpp
asm volatile(
    "s_mov_b32 m0, %0\n\t"
    "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
    :
    : "s"(lds_off), "v"(goff), "s"(srsrc), "s"(tile_byte_offset)
    : "memory");
```

The "P19 Dev A — m0-broadcast hoist for the 8-wave RCR DTL loads" comment
block (line 689-708) explicitly documents that this hand-rolled ASM was
chosen over the LLVM intrinsic precisely because the intrinsic let the
scheduler CSE m0 plumbing back into vector ops, defeating the SGPR
residency the kernel needs.

**Conclusion**: **Lever A is already 100% in production**. There is no
"+5..10pp" headroom on this axis. R1 wasted no rounds on this falsely (it
was a docs-only round) but R2's first job is to retire the lever.

---

## R1 Lever B FALSIFIED — dual LDS slab + N-strip is already shipped

R1 hypothesised Lever B ("Dual / triple LDS buffer ping-pong, +3..6pp,
LDS budget 64 KB cap is tight").

**That observation was wrong** in TWO ways:

### (a) Dual LDS slab already in use

`grouped_rcr_kernel` declares (line 2225-2226):
```cpp
__shared__ ST_rcr As[2][2];   // [tic/toc][which N-strip-of-2]
__shared__ ST_rcr Bs[2][2];   // 4 slots each → 4 × 16 KB = 64 KB per matrix
```

The main loop (line 2425) toggles `tic ^= 1, toc ^= 1` exactly once per
K-iter, with prologue priming `tic` slab and main loop loading `toc` slab
ahead. This is dual-slab ping-pong, plus per-slab N-strip duplication
(2 strips because RBN=32 × WARPS_N=4 covers half the BLK=256 column
extent). Total = 4-way LDS multiplexing for both A and B.

### (b) LDS budget is NOT 64 KB; it is 137 KB and nearly full

`-Rpass-analysis=kernel-resource-usage` on the rebuilt `.so` reports:

```
grouped_rcr_kernel<0, false, true>: LDS Size [bytes/block] = 139796
                                                            ≈ 137 KB
```

Budget breakdown:
- `As[2][2]` = 4 × ST_v2 (st_fp8e4m3<128,128>) = 4 × 16 KB = **64 KB**
- `Bs[2][2]` = 4 × ST_v2                       = 4 × 16 KB = **64 KB**
- `s_offs[65] + s_cum_tiles[65] + s_total_tiles` = ~530 bytes
- ST_v2 `subtile_padding` per tile: 8 sub-tiles × 64 B = ~4 KB
- Total ≈ 137 KB (matches the LLVM remark)

gfx950 (CDNA4) per-CU LDS cap is **160 KB**. With 137 KB used, headroom
is **23 KB**, which fits exactly *one* extra ST_v2 tile (16 KB) plus
some scratch overhead — but adding a third LDS slab to either A or B
(e.g. As[3][2]) would push to 153 KB which is closer to the cap and
would also force the LLVM compiler to demote LDS allocation onto VMEM
scratch (=> regression).

**Conclusion**: **Lever B (triple LDS slab) is infeasible — LDS is full**.
The R1 thesis mis-estimated the budget by 2× (assumed 64 KB cap, actual
160 KB cap, but kernel already uses 137 KB).

What IS feasible on the LDS-headroom axis: **redirect VMEM scratch spill
into LDS scratch** (1 KB or so of explicitly-managed `__shared__ uint32_t
scratch[N]`). This is part of Lever C, not Lever B.

---

## Precise spill / VGPR / occupancy baseline (the empirical truth)

`-Rpass-analysis=kernel-resource-usage` extracts per-template:

| Template `<KI_HINT, N_MASK, FUSED_KTAIL>` | TotalSGPRs | VGPRs | Spill (dw) | Scratch (B/lane) | LDS (B/blk) | Occupancy | Used by |
|---|--:|--:|--:|--:|--:|--:|---|
| `<0, false, false>` | 65 | **256** | **39** | 160 | 139796 | 2 w/SIMD | (dead, never dispatched) |
| `<0, true , false>` | 71 | **256** | **43** | 176 | 139796 | 2 w/SIMD | (dead) |
| `<0, false, true >` | 79 | **256** | **32** | 132 | 139796 | 2 w/SIMD | DSV3 (8c) + Qwen3 (8c) = 16/24 |
| `<0, true , true >` | 83 | **256** | **39** | 160 | 139796 | 2 w/SIMD | gpt_oss (8c) = 8/24 |
| `grouped_rrr_kernel<0>` | 71 | **256** | **76** | 308 | (?)    | 2 w/SIMD | RRR forward |
| `grouped_var_k_kernel_fp8<0>` | 79 | **256** | **52** | 212 | 139796 | 2 w/SIMD | dB backward |
| (BF16 dense fwd, comparison) | — | 48 | 0 | 0 | 2564 | 8 w/SIMD | dense BF16 RCR |

**Key observations**:

1. **VGPR cap reached** (256/256). LLVM cannot reduce VGPR count below
   256 without either (a) forcing more spill (already at 32-43 dw/lane!)
   or (b) changing `__launch_bounds__(_NUM_THREADS=512, MIN=1)`. Changing
   MIN was already falsified in R5/R7-dm.
2. **Spill 32-43 dw/lane × 64 lane × 8 waves/block = 16-22 KB scratch
   per CTA**. With kernel running ~hundreds of K-iters per output tile,
   spill→scratch_store_sbyte and reload→scratch_load_sbyte_d16 each cost
   ~60-100 cy, dominating the LDS ds_read latency.
3. **Occupancy = 2 waves/SIMD** (= 1 wavegroup-of-8/CU, since 1 CTA = 8
   waves and CU has 4 SIMD). Bound by *both* VGPR (256) AND LDS (137 KB).
   Reducing either alone won't increase occupancy (the other will still
   gate). Increasing occupancy is OUT OF SCOPE — the persistent kernel
   design (`for gt = pid; gt < total_tiles; gt += NUM_CUS`) explicitly
   wants 1 CTA/CU.
4. **gpt_oss spec (`<0,true,true>`) has +7 dw spill vs DSV3/Qwen spec
   (`<0,false,true>`)**: 39 vs 32. The `N_MASKED_STORE=true` template
   parameter adds 7 dw of live state (the per-lane masking branch
   bookkeeping in `store_c_tile_n_masked`). This is consistent with R59
   and R44-dm probes.
5. **K%128==0 cases** (DSV3+Qwen3, 16/24) **DO already go through the
   FUSED_KTAIL=true template** (see line 5266-5289 dispatch
   `fuse_ktail_eligible = (K_rem == 64) || (K_rem == 0)`). This was
   shipped in HK commit `823ef236` ("perf(grouped-fp8-dispatch): route
   K_REM=0 (DSV3) shapes through FUSED_KTAIL=true template spec for
   free-lunch register allocation"). So the R1 task-body suggestion to
   "use `if constexpr` to eliminate K-tail in K%128==0" is **already
   implemented** — and the empirically-better template for those cases
   is FUSED_KTAIL=true (32 spill) NOT false (39 spill), the OPPOSITE of
   what the naive thesis predicted.

---

## R3 plan — Lever C (VGPR live-range / spill reduction)

The only remaining axis with measurable headroom is **reducing the 32-43
dw/lane VGPR spill**. Three sub-levers, ranked by cost-of-investigation:

### Lever C-1: re-route VMEM scratch spill into LDS scratch (~1-2 rounds)

LLVM emits `scratch_store_sbyte_d16` / `scratch_load_sbyte_d16` for
spilled VGPR cells (R1's grep counted 27/31 of these in the truncated
dump). VMEM scratch round-trip is ~100-150 cy. LDS load is ~14 cy.

**Mechanism**: declare an explicit `__shared__ uint32_t scratch_pool[...]`
sized to absorb 16-22 KB of spill (need to fit in the LDS 23 KB headroom),
then at the spill-source code locations (epilog mul+store + K-tail load
issuance, the two LLVM-tagged hot spill spots in R44-dm probe) use
inline `ds_write_b32` / `ds_read_b32` to hand-spill cold VGPR slots
into LDS instead of letting LLVM choose VMEM scratch.

**Risk**: LLVM may not honour the manual LDS spill if it can't prove the
hand-spilled register is otherwise unused (might re-spill anyway). Need
to use `asm volatile("" : "=v"(reg))` clobber pattern to force the
register to be considered "free" after the LDS write.

**Validation**: after rebuild, the `-Rpass-analysis=kernel-resource-usage`
report should show ScratchSize bytes/lane DROP (target: spill < 16 dw =
half of current). If scratch goes UP instead → revert immediately.

**Expected gain**: each spill→load round-trip saves ~80 cy. Main loop
emits ~4 such pairs per K-iter (R44-dm rocprof breakdown). 22 K-iters
on gpt_oss × 4 pairs × 80 cy = ~7K cy/tile saved. With ~368 tiles/CU on
gpt_oss-GateUP-B32-M4096, ~2.6M cy = ~1.3 ms saved on a 5.4 ms kernel
= ~24 % speedup → projected gpt_oss-GateUP B32-M4096 ratio 1.022 → 1.27.
But this is an upper-bound estimate; LLVM may not free the slots and
the actual gain may be 1/3 of this (~+8 % on worst-case shapes).

### Lever C-2: shrink the K-tail-fused lambda capture set (~1 round)

The `FUSED_KTAIL=true` branch (line 2535-2719) wraps a 200-line
`if (g.fast_k < g.k)` block whose lambdas (`load_a_kt`, `load_b_kt`)
capture by reference: `m_subtile_A`, `br`, `bc`, `wm`, `wn`, `group_idx`,
`a_total_bytes`, `b_per_group_bytes`, `a_srsrc_kt`, `b_srsrc_kt`,
`a_row_stride_bytes`, `b_row_stride_bytes`, `K_tail_base_bytes`,
`b_group_byte_base`, plus the buffer SRDs and laneid/row_lane/k_lane_byte
intermediaries. **Even when `g.fast_k == g.k` (DSV3/Qwen3 K%128==0)**,
LLVM extends these captures' liveness across the entire `for (gt = pid;
gt < total_tiles; ...)` loop (the runtime branch on line 2540 is dead
code but the captures are alive).

**Mechanism**: refactor the K-tail block to compute `K_tail_base_bytes`,
`b_group_byte_base`, and the SRD constants from `g` and per-tile state
*inside* the `if (g.fast_k < g.k)` block, dropping the captures back to
the bare minimum needed by mfma issue. Or: factor the K-tail body into
a separate `__device__ __forceinline__ void k_tail_apply(...)` function
that LLVM cannot speculatively hoist captures out of.

**Risk**: refactor may regress codegen on the K=2880 (gpt_oss) shapes
where the captures genuinely need to be live, in exchange for K%128==0
shape gains. Need to verify both subset metrics independently.

**Expected gain**: 5-15 dw spill reduction on the FUSED_KTAIL=true
spec → +1-3 pp on K%128==0 cases (16/24 → +0.7-2 pp geomean impact).

### Lever C-3: profile-guided spill localization (~1 round, scout only)

Use `-mllvm -print-after-all` or `-Rpass=spill` with `-Rpass-analysis=spill`
on the LLVM/AMDGCN compile pipeline to extract per-line spill source
info. Would tell us EXACTLY which expression in the kernel forces the
spill (so we can rewrite that expression vs. blindly refactoring the
whole K-tail block). Pure data-collection round, no code change.

---

## R3 recommendation: start with **Lever C-2** (refactor K-tail captures)

Rationale:
- C-2 has the SHORTEST development time (1 round, ~50 lines of refactor).
- Risk of regression is bounded by the existing test correctness gate
  (metric runs allclose vs torch ref).
- It's the cleanest test of whether the FUSED_KTAIL=true code's
  captures dominate the spill (the R1 task-body's main hypothesis).
- If C-2 lands +1-3 pp → use that as evidence basis for C-1 (LDS
  scratch hand-spill) in R4.
- If C-2 falsified (no change) → R4 = C-3 spill localization probe to
  find actual spill source before committing to C-1's larger investment.

**Lever D microbench gate** (R64-dm scaffolded, not run): defer to R5+
unless C-2/C-3 falsify Lever C entirely. The cost-model from R64-dm
already showed identical mfma cycle count and identical accumulator
VGPR pressure between mfma_1616128 and mfma_323264, so D is a long
shot.

**Lever F (Qwen-Down K=1536 short-K specialization)**: deferred to
R10+ — even if it adds +5pp on 4/24 case, that's only +0.8 pp on the
geomean, not worth burning a round before C is exhausted.

---

## What this round changed

**Nothing in code.** Only:
- New round note at this path (this file)
- Rebuilt `tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so` (no source
  change — just to capture LLVM resource-usage remarks). The rebuilt
  `.so` is bit-identical to the pre-R2 baseline because R64-dm's
  st_32x64 type addition is force-instantiated only and DCE'd.

**HipKittens repo**: no change (HK SHA stays at `ecbead9a`).
**Primus-Turbo repo**: this `.md` is the only diff.

---

## What R3 should do (concrete first step)

1. Read `kernel_fp8_layouts.cpp` lines 2535-2719 (the FUSED_KTAIL=true
   block) to enumerate all by-reference captures in `load_a_kt` and
   `load_b_kt` lambdas.
2. Refactor those lambdas to be `__device__ __forceinline__` free
   functions taking only the strict minimum parameters needed by the
   mfma. Move `K_tail_base_bytes`, `b_group_byte_base`, the buffer SRDs,
   and `a_total_bytes`/`b_per_group_bytes` to be COMPUTED INSIDE the
   `if (g.fast_k < g.k)` block (so the false branch sees no live state).
3. Rebuild + check `-Rpass-analysis=kernel-resource-usage`:
   - target: VGPR Spill on `<0,false,true>` template DROPS from 32 to ≤24
   - target: ScratchSize/lane DROPS from 132 B to ≤96 B
4. If spill DROPPED → run metric. If score ≥ +5pts → commit. Else revert.
5. If spill UNCHANGED or INCREASED → revert immediately, jump to C-3.

Hard rules unchanged from R1: no metric/dispatch/can_handle edits, one
focused commit per repo, BF16 grouped is `[watch]`, FP8 grouped is the
only optimization target.

---

## Updated lever roadmap (replace R1 table)

| Lever | Status | Next action |
|---|---|---|
| **A** Async global→LDS copy | **FALSIFIED R2** — already shipped via inline-ASM `buffer_load_dwordx4 ... lds` | (none) |
| **B** Dual/triple LDS slab | **FALSIFIED R2** — dual already shipped, triple infeasible (137/160 KB used) | (none) |
| **C** VGPR spill / live-range | **OPEN** — 32-43 dw spill measured, real headroom | **R3 = C-2 refactor; R4 = C-1 LDS spill; R5 = C-3 profile** |
| **D** mfma_323264 cell-shape | DEFER — R64 scaffolded, microbench gate not run, cost-model shaky | R10+ if C exhausted |
| **E** ASM software pipelining | LAST RESORT | R20+ |
| **F** Qwen3-Down K=1536 short-K | DEFER — only +0.8pp geomean even if successful | R15+ |

---

## DoD smoke status

Not run this round (docs + rebuild only, no dispatch / can_handle /
shared-code change). Last run was at R64 SHA `94fc3121`. Will re-run
at R5 checkpoint per task-body cadence.
