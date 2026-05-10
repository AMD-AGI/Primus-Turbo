round-53-A3-decoupled-warps-preflight-inventory-PC-PROTOTYPE-EXISTS-A3-SURVIVES.md
=============================================================================

Round: 53 / 100
Date: 2026-05-10
Pre-SHA: 7963775b (R52 docs)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R52 forward-pointer asked R53 to **preflight Direction A3 (decoupled-warps
producer-consumer)** with a 1-round inventory of:
  1. existing producer-consumer prototypes in HK
  2. removable CTA-wide barriers in the var-K main loop
  3. register-budget headroom for 4 producer + 8 consumer waves (12 waves/CTA)

**Verdict: A3 SURVIVES preflight on all 3 inventory axes.** No code changes
this round; R54 is greenlit for kernel-template implementation.

## Inventory result #1 — PC prototype EXISTS in HK

`HipKittens/kernels/gemm/bf16fp32/micros/producer_consumer/{16x32,32x16}/`
contains a complete BF16/FP32 producer-consumer prototype family:

  | Variant                                | Out tile    | Layout            | Status |
  |----------------------------------------|-------------|-------------------|--------|
  | micro_02_2stage_8c4p.cpp               | (128)x(256) | 8 cons + 4 prod   | ✓      |
  | micro_03_3stage_8c4p.cpp               | (128)x(256) | 8 cons + 4 prod, 3-stage | ✓ |
  | micro_04_2stage_12c4p.cpp              | (192)x(256) | 12 cons + 4 prod  | ✓      |
  | micro_05_2stage_16c2p.cpp (above-SW)   | (128)x(256) | 16 cons + 2 prod  | ⚠ HW limit |
  | micro_06_*_64x96 / 96x64 (spills)      | (128)x(256) | 8c4p + asym       | ⚠ regs |
  | micro_05_async.cpp / micro_09_async.cpp| -           | async-load variants | ✓ |

  * Plus a per-CTA-paper plot at `analysis/paper_experiments/producer_consumer_micro/`.

Architecture (per `micro_02_2stage_8c4p.cpp`):
  * 12-warp CTA: warp_group 0 = producer (4 warps issue HBM→LDS via
    `G::load<2,false>(...)` — group-cooperative full-tile load); warp_groups
    1..M_BLOCK = consumers (8 warps run `mma_ABt(...)`).
  * 2-stage LDS double-buffer (`As[2][...]`, `Bs[2][...]`); `tic ^= 1` per
    iter.
  * Producer ends iter with `__builtin_amdgcn_s_waitcnt(0)`; consumer ends
    iter with `s_waitcnt lgkmcnt(0)` + `s_setprio(1)/mma/s_setprio(0)`.
  * **Single CTA-wide `__builtin_amdgcn_s_barrier()` per iter** (line 156),
    immediately followed by `__builtin_amdgcn_sched_barrier(0)` (scheduler
    hint, not a sync). The producer / consumer loops are otherwise
    independent within an iter.

This is the natural starting point for the FP8 var-K port — no
greenfield kernel-architecture work required.

## Inventory result #2 — barriers in var-K main-loop (4 → 1 reduction possible)

Source: `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` lines
8418-8727 (`grouped_var_k_kernel_fp8` main loop).

**Current var-K main loop has 4 CTA-wide `__builtin_amdgcn_s_barrier()`
per iter** (the FP8 path the metric exercises):

  | # | Source line | Position           | Load-bearing? | Etiology |
  |---|-------------|--------------------|---------------|----------|
  | 1 | 8671        | post-prefetch lgkm-drain | yes      | per-warp slice loads need CTA convergence |
  | 2 | 8698 (CRR_STEADY_MID_BARRIER) | mid-MMA | **R26 yes** | per-warp slice-skew on cooperative load (bit-exact analysis 8270/8271) |
  | 3 | 8706        | post-vmcnt drain   | yes           | symmetric to #1 (other half-tile slot) |
  | 4 | 8724        | end-of-iter        | **R28 yes**   | symmetric to #2 (next-iter cooperative load) |

R26-R28 single-drop falsifications (`VARK_DROP_BARRIER_2`, `VARK_DROP_BARRIER_4`)
both produced catastrophic SNR failures (≤12 dB across 8/8 shapes) because
the EXISTING architecture loads tile data **per-warp-slice** via
`rcr_8w_load_hoist<_NUM_THREADS>` — each warp owns 1/8 of a tile, and
without the CTA barrier fast warps overwrite slow warps' as-yet-unread
LDS slots. This is exactly the failure mode PC eliminates: in PC, the
**producer warps own the COMPLETE tile load** (4 warps cooperative via
`G::load<2,false>`), so consumer warps see fully-coherent LDS tiles
without needing CTA-wide convergence with the producers.

Per the BF16 prototype, **PC has 1 CTA barrier per iter** (line 156 of
`micro_02_2stage_8c4p.cpp`). The producer's `s_waitcnt(0)` and consumer's
`s_waitcnt lgkmcnt(0)` inside the warp-group are per-warp scalar-counter
syncs, not CTA-wide barriers.

→ **PC port = 4× barrier reduction in steady-state main loop.** This
directly attacks the R21 PMC etiology (32% MfmaUtil, 58% issue-rate idle,
0.2% MemStall) where MFMA pipe stalls are dominated by operand-not-ready
intervals — the most common cause of which is barrier sync delay on a
load-bearing per-iter `s_barrier`.

## Inventory result #3 — register & LDS budget headroom for 12 waves

### Register budget

MI355X CDNA4: **1536 VGPR/SIMD × 4 SIMD/CU = 6144 VGPR/CU**. Each wave
occupies one SIMD; max VGPR per wave depends on co-resident waves.

Production fp8 RCR (8 waves):
  * 8 waves / CU = 2 waves / SIMD
  * 256 VGPR/wave × 2 = 512 VGPR/SIMD (33% of 1536) — comfortable
  * Spill: 37 dw / wave (per task md)

PC port (4 producer + 8 consumer = 12 waves / CTA):
  * 12 waves / CU = 3 waves / SIMD (still 1 CTA / CU, single-CTA occupancy unchanged)
  * Producer wave: needs ~64-96 VGPR (load-address arithmetic + `G::load`
    swizzled-offset cache, no MMA accumulator). BF16 prototype confirms.
  * Consumer wave: needs full ~256 VGPR (4× rt_fl<RBM, RBN, col_l>
    accumulator tile = the same cA/cB/cC/cD as today + b0/b1/a register tiles).
  * Worst-case per-SIMD: 256 × 3 = 768 VGPR/SIMD (50% of 1536) — comfortable.
  * Producer waves don't allocate the accumulator tile, so per-wave reg
    pressure should DROP for them — net spill expected ≤ current 37 dw/wave.

→ **Register budget fits 12 waves / CTA with 50% SIMD headroom.**

### LDS budget

Production var-K LDS layout (lines 8421-8427):
  * `ST_crr_a As[2][2]`  = 4 × 16KB = 64KB  (st_fp8e4m3<128, 128>, 1B/elt)
  * `ST_crr_b Bs[2][2]`  = 4 × 16KB = 64KB
  * `s_offs[65]` + `s_cum_tiles[65]` + `s_total_tiles` ≈ 528B
  * **Total ≈ 128.5 KB / 160 KB CDNA4 LDS** → 31.5 KB headroom.

PC 2-stage port: **same `As[2][2]` / `Bs[2][2]` 2-stage layout** (the
production var-K is already 2-stage). The only LDS overhead PC adds is
optional tokens / barriers (e.g., per-stage producer-done flags), which
are at most ~64B. → No LDS-budget impact.

→ **LDS budget OK for 2-stage PC port.**

## Risk inventory

| Risk | Mitigation in R54+ kernel-implementation arc |
|------|------|
| LLVM AGPR allocator alias bug (R59-R61, RBN=64 4-wave death) | Consumer waves keep RBN=32 (= production layout) → accumulator-tile size unchanged → bug doesn't trigger. Verified per task md FORBIDDEN PATHS line 130-131. |
| Producer-side `rcr_8w_load_hoist` is hard-coded to `_NUM_THREADS = 512` (= 8w·64) | Need a 4w-cooperative analogue (`G::load<2,false>` from BF16 prototype, or new `crr_4w_load_full`). Implementation cost: 1 helper. |
| LDS bank-conflict on producer writes vs consumer reads of same buffer  | PC's 2-stage decouples write-set (`tic`) from read-set (`toc`) — no concurrent access on the same buffer. Standard double-buffer guarantee. |
| `__shared__ s_offs / s_cum_tiles` init runs cooperatively with all 12 warps | Only threads 0..g.G write; ≤64 entries; trivially scales to 768 threads. |
| Variable per-group K (ki_g varies 16-32 across groups) → consumers may finish iter before producer next-tile is staged | This already happens in production (any group's K is independent of CTA's persistent slot count). The PC's `s_barrier` at end-of-iter handles it identically. |
| BF16 PC prototype's `mma_ABt` is BF16 32-cycle; FP8 MFMA is 32-cycle but with different K-block semantics (BK=128 fp8 vs BK=16 bf16) | The number of MMA issues per LDS-tile-load is much higher in fp8 (8× more K reduction per tile-load) → PC's load-shadow-MMA pattern actually becomes MORE favorable in fp8, since one producer iter feeds many consumer MMAs. |
| Wave-count change (8 → 12) may interact with chiplet swizzle / num_xcds defaults | The XCD swizzle is `chiplet_transform_chunked(blockIdx.x, slots_eff, xcds_eff, chunk_size_eff)` — depends on grid_x and xcds, NOT on warps/CTA. PC port does not change these. ✓ |

## EV vs cost

  * **EV upper bound**: PMC R21 said 58% issue-rate idle on var-K wgrad
    Down-B4 (32% MfmaUtil active). If PC eliminates 75% of barrier-induced
    idle (i.e., recovers 75% of the 4→1 barrier traffic = 0.75 × 0.58 ×
    Down-B4 fraction of section-mean weight ≈ 0.43), per-cell wgrad
    Down-B4 lift ≈ 30-40% TFLOPS. Section-mean wgrad lift bounded by the
    weight of var-K cells: Down-B4 (M=2048,M=4096) = 2/8 cells, GateUP-B4
    same = 4/8 cells contribute. Other 4 cells are B=32, less SALU-bound.
    **Realistic section-mean wgrad lift: +5 to +10% on the wgrad average**
    (range corresponds to "PC removes 50% to 90% of issue-rate idle on B=4
    cells").
  * **EV on score**: wgrad section progress = mean(TFLOPS / 2800), so
    +5-10% TFLOPS → +0.65 to +1.3 progress / 3 sections → **+22 to +44 score**.
  * Plus indirect uplift on fwd / dgrad RCR / RRR — these have the same
    barrier-pin pattern but different fraction of issue-rate idle (per
    R21 PMC: fwd Down-B4-M2048 = 39.8% SALU/SQ_busy = lower than var-K's
    85%, but still 35.6% MfmaUtil = ~64% issue-idle). PC port to RCR fwd
    is a separate R55-R56 task once var-K port is validated.
  * **Implementation cost**: 4-6 rounds per task md A3 estimate, broken
    down as:
    - R54: write `grouped_var_k_kernel_fp8_pc` template variant under
      `kernel_fp8_layouts.cpp`, registered behind a new `kernel="vark_pc"`
      template id; bind in pybind table; build via `dbg_remote.sh`.
    - R55: bit-eq + SNR verification (7-seed × 2500-iter) on Down-B4 cells;
      tight-verify spread; run metric; compare baseline.
    - R56: dispatcher rule for PC variant (which cells should pick `vark_pc`
      vs default `vark_8w`); cross-shape sweep.
    - R57: port the same PC pattern to RCR fwd / RRR dgrad if R54-R56 wins.
    - R58: contingency / debug round.
  * **EV/round**: +4 to +11 score/round (over 4-6 rounds), competitive
    with the +1-3 score/round dispatcher tweaks lately, and with much
    higher upper-bound.

## R53 verdict

**A3 (decoupled-warps producer-consumer) SURVIVES preflight on all 3
gates** (PC prototype exists, 4×barrier reduction available, register
& LDS budget fits). R54 is greenlit for kernel-template implementation.

## R54 forward-pointer

R54 = **port `micro_02_2stage_8c4p` to var-K FP8** as a new template
variant `grouped_var_k_kernel_fp8_pc`:

  1. Add `#define VARK_4P8C_PC` build-flag macro guard to
     `kernel_fp8_layouts.cpp` (default = 0; flag enables a new template
     variant via `kernel="vark_pc"` dispatcher).
  2. Implement `grouped_var_k_kernel_fp8_pc<TPL_ID>` as a sibling of
     `grouped_var_k_kernel_fp8`. Body is the BF16 `micro_02_2stage_8c4p`
     skeleton, but:
     * `mma_ABt` → `crr_mma` (fp8 8x128x128 MFMA, exists today)
     * `G::load<2, false>` → 4w-cooperative analogue of `rcr_8w_load_hoist`,
       call it `vark_4w_load_full` (write inline if no helper exists)
     * `is_consumer` warp index → reuses existing `wm`/`wn` math
     * The persistent loop `for (int gt = pid; ...)` wraps the PC steady-state
     * Producer warps share the s_offs/s_cum_tiles SALU coord-decode — no
       per-warp duplication needed; the SALU coord-decode is per-CTA
  3. Register the template id in `pybind11` module bindings at the bottom
     of the .cpp.
  4. Add a dispatcher rule: when `(layout == CRR_VAR_K) and (B==4) and
     (M_per_g >= 2048) and (K == 2880)`, return `HipKittenConfig(...,
     kernel="vark_pc")`. Anchor on Down-B4-M2048 wgrad (worst SALU/issue-rate
     cell per R21 PMC).
  5. Build via `dbg_remote.sh`; run `_metric_gpt_oss_fp8_kernel.py` × 3
     samples for SNR-and-score sanity check.
  6. If R54 catastrophically fails SNR (e.g. <15 dB), document and pivot
     to R55 = simpler 1-stage PC variant (no double-buffer) as a debug
     stepping-stone.

If R54 ships +5 score with SNR > 25 dB on all 8 shapes, R55-R56 expands
to RCR/RRR PC port. If R54 falsified, R55 = pivot to Direction G
(cross-shape co-optimization) per task md NEW DIRECTIONS list (G is the
last untried direction).

## Files added

  * `analysis/_notes/round-53-A3-decoupled-warps-preflight-inventory-PC-PROTOTYPE-EXISTS-A3-SURVIVES.md` (this file)

## NEUTRAL round

No code, dispatcher, or kernel changes. Daemon metric expected in the
691-699 R29 noise band. Fallback if metric drifts up: that's R52
NEUTRAL-tail noise; A3 implementation begins R54 regardless.
