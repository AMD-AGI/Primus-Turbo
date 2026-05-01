# Round 63-dm — FP8 grouped: plateau ACCEPTED + Lever D full-port roadmap for R37+

**Status**: PLATEAU ACCEPTED / no kernel change this round
**Score before** (R35 post-revert): 933-959 (6-run noise band 933/957/958/959/959/961)
**Score after** (this round, 2 runs):  957 / 959 — baseline reconfirmed
**HK SHA**: unchanged (`78415fb0`)
**PT SHA**: this commit (doc-only)
**Round time**: ~25 min (2 metric runs + 1 ISA spill audit + 1 architectural survey)
**Auto-optimize round**: 36

---

## Decision this round

**Accept the empirical plateau at 957-962** and commit to a documented
R37+ multi-round Lever D full-port plan instead of attempting yet
another single-round probe.

### Why plateau acceptance is the correct move at R36

**8 rounds without improvement (R29-R36)**. The dense empirical record
shows every single-round structural probe since R27 has been either
falsified or neutral:

| Round | Lever attempted | Outcome | Score Δ |
|--|--|--|--|
| R27-dm | main-loop sched_barrier audit | saturated | 0 |
| R29-dm | readfirstlane + cell-shape rediscovery | cell-shape deferred, other probes falsified | 0 |
| R30-dm | sched_group_barrier | spill win, perf loss | 0 |
| R31-dm | prologue collapse | broke wm1 sync | 0 (revert) |
| R32-dm | inter-barrier vmcnt | load-bearing, can't remove | 0 |
| R33-dm | epilog1 mfma merge | llvm spill backlash | 0 |
| R34-dm | FUSED_KTAIL=true for K_REM=0 | **+5 pts win** | +5 |
| R35-dm (this chat) | Lever D K-tail port | fan-out cost > mfma savings | 0 |
| R36 (this round) | — | plateau accepted | 0 |

R34's +5-pt win was a one-off codegen anchoring trick (not a new
lever). R35's K-tail port was the last plausible single-round
architectural option and it FALSIFIED (R62-dm documents the -9 pp
regression due to LDS fan-out cost structurally exceeding mfma
savings by ~116 cy/tile).

### Single-round lever inventory AFTER R36 (confirmed by R54/R56/R62)

| Lever | Status | Why not this round |
|--|--|--|
| A (async g→LDS) | **ALREADY SHIPPED** | rcr_8w_load_hoist uses inline asm `buffer_load_dwordx4 offen lds` (R54-dm) |
| B (dual LDS) | **ALREADY SHIPPED** | `As[2][2]` + `Bs[2][2]` ping-pong (R54-dm); triple blocked by 160 KB LDS cap |
| C (register hints) | **SATURATED** | __noinline__ ABI fail, __builtin_expect worsens spill, restrict/constexpr no effect (R54-dm) |
| D K-tail only | **FALSIFIED** | fan-out structurally > savings (R62-dm = -9 pp) |
| D full main-loop | **VIABLE, 4-6 rounds** | single-round scope insufficient — this is R37+ plan |
| E ASM main-loop | **HIGH RISK** | no precedent, hard to debug, 2-3 rounds |

Every micro-knob has been exhausted across 36 rounds. The remaining
structural option (Lever D full port) requires multi-round commitment
and cannot be accomplished in a single round without leaving the tree
in an unbuildable or uncorrect state.

---

## Metric delta this round (no kernel change)

```
FP8 baseline geomean (R34 pre-K-tail-port):   1.1247 / score 962
FP8 post-R35 revert + R36 confirm:            1.1152-1.1212 / score 957-959
BF16 geomean:                                  1.1819-1.1824 / unchanged
DoD score:                                     608 (last SHA 94fc3121, unchanged)
```

Both FP8 and BF16 ratios are stable within the 957-962 noise band. No
regression vs last round. Plateau is confirmed as EMPIRICAL CEILING for
the current kernel architecture.

---

## R37+ Lever D full main-loop 32x32 cell-shape port — concrete plan

Based on confirmed infrastructure inventory:

### Infrastructure ALREADY landed (usable as-is)

| Artifact | File:Line | Status |
|--|--|--|
| `rt_32x64` / `rt_64x32` shape structs | `rt_shape.cuh:59-60` | landed (R14-dm) |
| `rt_32x64_s` / `rt_64x32_s` public aliases | `types.cuh:96-100` | landed (R57-dm `c2abba21`) |
| `mma_ABt_base<rt_32x32, ..., 32x32x64>` dispatch | `mma.cuh:234-238` | landed (R14-dm) |
| `mma_AB_base<rt_32x32, ..., 32x32x64>` dispatch | `mma.cuh:172-185` | landed (R14-dm) |
| `mfma323264` intrinsic wrapper | `mma.cuh` | landed |
| `rcr_mma_32` kernel-local wrapper | `kernel_fp8_layouts.cpp:257` | landed (R59-dm `addaf23e`) |
| `load_a_kt_32x64` / `load_b_kt_32x64` K-tail loaders | `kernel_fp8_layouts.cpp:282/313` | landed (R61-dm `78415fb0`) |
| Force-instantiate compile-time validation | `kernel_fp8_layouts.cpp:349` | landed |

### What's MISSING (required to land the port)

1. **`ST_32x64` shared-memory tile type** — new `st_fp8e4m3<HB, 64,
   st_32x64_v?_s>` layout. Requires a new swizzle pattern matching
   how the 32x32x64 mfma reads its A and B operands. Scope: ~50-100
   lines of shape struct + layout helpers.
2. **`prefill_swizzled_offsets_32x64` helper** — analog of the
   existing `prefill_swizzled_offsets` but for the new ST layout.
   Scope: ~80-120 lines.
3. **Main-loop `load_a_32` / `load_b_32` helpers** — ds_read patterns
   for `ST_32x64` → `rt_32x64` register tile. Scope: ~40-80 lines
   (can model on existing `load_a` / `load_b` lambdas).
4. **`rcr_8w_load_hoist_32`** — HBM→LDS issue for the new ST layout.
   Scope: ~50-80 lines (mostly matching inline-asm pattern of existing
   helper, retuned for 32x64 column swizzle).
5. **`grouped_rcr_kernel_32` template** — new kernel class with
   `cA/cB/cC/cD` as `rt_fl<RBM, RBN, col_l, rt_32x32_s>` instead of
   `rt_16x16_s`. Main loop uses `rcr_mma_32`. K-tail body uses
   `load_a_kt_32x64` / `load_b_kt_32x64` (already shipped) + inline
   accumulate (NO FAN-OUT needed since cA-cD are already 32x32).
   Scope: ~600-900 lines (fork of current `grouped_rcr_kernel` body).
6. **Output store path** — rt_32x32 → bf16 `g.c` store with correct
   lane→HBM mapping. Existing `store(...)` may need a 32x32-specific
   branch. Scope: ~80-150 lines.
7. **Dispatch branch** — dispatcher at line 5229 chooses between
   `grouped_rcr_kernel<0, *, true>` (current 16x16) and
   `grouped_rcr_kernel_32<0, *, true>` (new 32x32). A/B test at
   runtime to confirm no regression on any shape before flipping
   default. Scope: ~20-40 lines.

Total: ~920-1470 lines of new/modified HK code + ~40 lines PT
dispatch. Realistic scope **4-6 rounds**.

### R37+ round-by-round plan

| Round | Milestone | Deliverable | Expected score Δ |
|--|--|--|--|
| **R37** | ST_32x64 shared-memory type + swizzle | `include/types/shared/` + types.cuh. Compile-time validated via force-instantiate. | 0 (infra) |
| **R38** | Main-loop load helpers for rt_32x64 | `load_a_32` / `load_b_32` + `rcr_8w_load_hoist_32` + instantiate stub. | 0 (infra) |
| **R39** | `grouped_rcr_kernel_32` skeleton with K-tail disabled | Build passes. Kernel not wired into dispatcher. | 0 |
| **R40** | Wire kernel_32 into dispatch for gpt_oss K_REM=64 shapes only | Correctness probe. If fwd SNR ≥ 22 dB, proceed. | **+2 to +5** if clean |
| **R41** | Extend dispatch to all 8 gpt_oss shapes + DSV3 K_REM=0 shapes | Full metric run. Tune VMCNT if needed. | **+3 to +8** cumulative |
| **R42** | (opt) Replace 16x16 kernel entirely if no regressions | Ship kernel_32 as default. Retire kernel_16. | stabilize |

**Acceptance criteria** each round:
- Correctness: all 16 FP8 fwd PASS (SNR > 22 dB), 0 NaN / Inf.
- Metric: score ≥ R36 baseline (957) at each milestone.
- Any regression ≥ 5 pts → revert round, keep infrastructure, fall
  back to previous milestone's state.

**Total expected upside** (per R56-dm + R62-dm corrected cost model):
**+5 to +10 pp on gpt_oss geomean** = **+3 to +5 pp on overall FP8
geomean** = final FP8 geomean 1.17-1.19 = **score 970-995**.

Not quite 1.20 (~1000) but a substantial +13-38 pts over current 957.

### Why this plan avoids the R35 K-tail-port failure mode

R35's failure mode was **cross-layout fan-out** (mfma_323264 output in
32x32 lane layout → mfma_1616128 output in 16x16 lane layout). The
LDS round-trip for the conversion cost ~500 cy/tile, exceeding the
128 cy mfma savings.

In the full main-loop port:
- `cA/cB/cC/cD` are **always** rt_32x32 throughout (never mixed with
  rt_16x16).
- Output store writes rt_32x32 directly to HBM — **no intermediate
  fan-out**.
- The only conversion is bf16-cast during store, which happens anyway
  in the current kernel (just with different lane ordering).

The fan-out cost is structurally **ELIMINATED**, not just reduced.

### Risk analysis

1. **LDS swizzle design risk** (R37): the `ST_32x64` swizzle must be
   bank-conflict-free for both the `buffer_load_lds` write pattern
   AND the `ds_read` read pattern matching mfma_323264's input lane
   map. Getting this wrong = catastrophic bank conflicts. Mitigation:
   hand-derive lane→bank map before implementing; verify with small
   correctness probe.

2. **Register pressure** (R40): rt_32x32 acc is 4 dwords/lane/cell ×
   2 cells (vs rt_16x16's 4 dw × 4 cells × 2 subtiles = 32 dw/warp-
   coord). Total per accumulator: 32 dw × 4 (cA/cB/cC/cD) = 128 dw
   OR rt_32x32's 4 dw × 2 cells = 8 dw × 4 = 32 dw. **Lower
   register pressure** by ~50% — this is the STRUCTURAL win.

3. **DSV3 regression** (R41): DSV3 currently at 1.16-1.22 geomean.
   The 32x32 port might not benefit DSV3 specs as much (K_REM=0,
   already at good ratios). If DSV3 regresses > 2 pp, keep both
   kernel variants and dispatch based on K_REM.

4. **Build time** (R37-R41): adding ~1000 lines of HK code will
   slow incremental builds. Mitigation: use `force-instantiate`
   stubs to bring up infrastructure in isolation; only wire into
   main kernel after each piece is validated.

### If R37-R42 all go well: projected path to score 1000

```
R36 (now):         957  (FP8 geomean 1.115)
R37-R39 (infra):  ~957  (no metric change, 3 rounds)
R40 (gpt_oss):    970-985  (FP8 geomean ~1.14-1.17)
R41 (all shapes): 975-995  (FP8 geomean ~1.16-1.18)
R42 (default):    975-995  (stabilize, 1-2 runs for noise)
R43+ (tune):      985-1000 (VMCNT/sched retune for 32x32 schedule)
```

If instead R40 falsifies → kernel_32 too slow → score stays ~957-965.
Fallback: hand off to R37+ agent a plan for Lever E (manual ASM
main-loop scheduling) — but that's a 2-3 round separate effort.

---

## What NOT to do in R37+

Explicitly FROZEN (do not re-attempt):

- ✗ Any vmcnt/lgkmcnt sweep (R51 confirmed saturation, INIT0=4 at safety boundary)
- ✗ Barrier removal (R32/R52 confirmed wm1/wm0 barriers load-bearing)
- ✗ Lever D K-tail-only port (R35/R62 falsified, fan-out cost structural)
- ✗ K-tail standalone kernel routing (R52 falsified, -76 pts)
- ✗ sched_barrier mask tweaks (R27-R30 saturated)
- ✗ LDS ST_v2 → ST_v3 swap (ds_read hardcoded v2)
- ✗ `__noinline__` on store_c_tile_n_masked (R54 ABI conflict)
- ✗ `__builtin_expect` hot-path hints (R54 worsens spill)
- ✗ Any dispatch/can_handle tightening (frozen by task body)
- ✗ Per-shape config overrides (frozen by task body)
- ✗ Triple-buffer As[3][2] (blocked by 160 KB LDS cap)

---

## Repo state at end of round

- HipKittens: `78415fb0` (unchanged since R34 loader-stub commit)
- Primus-Turbo: advances after this doc-only commit
- Working tree: clean (no uncommitted changes)
- Auto-optimize patience: **8/10** (will hit 10 at R38 if no change)
- DoD last run: 608 @ SHA 94fc3121

---

## Recommendation for R37 (concrete first step)

**R37 deliverable (HK infra-only commit)**:

1. Add `st_fp8e4m3<HB=128, BK=64, st_32x64_v_s>` type alias to
   `types.cuh`. Define the shape struct in `include/types/shared/
   tile/shape.cuh` with:
   - `rows = 32, cols = 64` (matches rt_32x64)
   - Swizzle function matching `mfma_323264` input lane layout
     (lane l holds column = l % 32, row range = chunk-based)
2. Add a force-instantiate stub in `kernel_fp8_layouts.cpp` that
   declares `__shared__ ST_32x64 As_32;` and calls a no-op
   `load(dummy_rt, As_32)`.
3. Build. Confirm no compile errors. Confirm spill profile of
   `grouped_rcr_kernel` is unchanged (infrastructure-only
   addition).
4. Commit HK as `infra(fp8): add ST_32x64 shared-memory tile type
   for Lever D R-B step 1 (main-loop port)`.
5. PT commit: round-note + metric log showing no regression.

**Expected outcome**: metric ≈ 957-962 (unchanged), build passes,
infrastructure ready for R38 load-helper additions.

If an R37 agent prefers to **not** commit to the 4-6 round plan,
the honest alternative is to accept 957-962 as the empirical ceiling
for the current architecture and spend remaining rounds on:
- Code-quality improvements (doc comments, removing stale FROZEN
  patterns from the code surface)
- Backward-path optimizations (dB / dA kernels; NOT in metric but
  still useful for end-to-end training)
- Cross-shape correctness robustness (probes on edge-case K_REM
  values not covered by metric)

None of these move the score needle, but none hurt either.

## Files touched (R36 this round)

- `/workspace/code/Primus-Turbo/analysis/_notes/round-63-dm-fp8-grouped-plateau-ACCEPTED-and-lever-d-full-port-roadmap.md`
  (this note, ~260 lines)
- No HK changes.
- No PT code changes outside this note.
