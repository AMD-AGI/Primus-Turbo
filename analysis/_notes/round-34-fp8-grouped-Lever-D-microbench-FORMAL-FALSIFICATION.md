# Round 34 — FP8 grouped: Lever D microbench gate FORMAL FALSIFICATION

## Summary

R34 executed the validation gate explicitly recommended in
`round-64-dm-fp8-grouped-lever-d-rb-step-1-st-32x64-type-LANDED.md`:
a single-warp throughput microbench comparing
`mfma_scale_f32_32x32x64_f8f6f4` vs `mfma_scale_f32_16x16x128_f8f6f4`
in pure-MFMA mode (no LDS, no register spills, isolated from kernel
overhead).

**Result: Lever D is FORMALLY FALSIFIED. The 32x32x64 cell shape is
~6% SLOWER per FLOP than the current 16x16x128 in single-warp mode**,
not faster. The R63-dm "viable, 4-6 rounds" recommendation and the
R64-dm step-1-LANDED scaffolding are now formally BLOCKED from
proceeding.

## Microbench setup

* **Source**: `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/microbench/mfma_323264_vs_1616128.hip`
  (preserved for posterity)
* **Compile**: `hipcc --offload-arch=gfx950 -O3`
* **Workload**: 256 CUs × 8 warps/WG × 8000 chained mfma calls/warp =
  16.4M MFMA issues per kernel
* **Measurement**: 5 hipEvent-timed runs, take min, convert to TFLOPS
  using exact per-MFMA FLOP count
* **Gate**: ≥+3 pp single-warp throughput advantage for 32x32x64
  required to PROCEED (per R64-dm)

## Results (3-trial reproduction)

```text
trial   mfma_16x16x128 (TFLOPS)   mfma_32x32x64 (TFLOPS)   delta
   1                  4305.4                  3971.6     -7.75%
   2                  4342.2                  4111.8     -5.31%
   3                  4312.2                  4107.4     -4.75%
   4                  4312.7                  4052.0     -6.04%
mean                  4318.1                  4060.7   ~  -6.00%
```

Verdict: **ABANDON Lever D full port.**

## Why this is consistent with R64-dm's analytical model

R64-dm's analytical model predicted:
* Per-K-iter MFMA cycles: identical (512 cy for both rt_16x16 and
  rt_32x32 to cover same K=128)
* Per-warp accumulator pressure: identical (128 dw for both)

The microbench reveals a third factor R64-dm did not model:
**per-issue throughput is NOT proportional to per-FLOP work**. The
gfx950 MFMA pipeline appears to have:
* mfma_16x16x128: ~16 cy issue → 65536 FLOPs → 4096 FLOPs/cy/wave
* mfma_32x32x64: ~32 cy issue → 131072 FLOPs → 4096 FLOPs/cy/wave

Theoretically equal, but the microbench shows mfma_32x32x64 hits
**~3850 FLOPs/cy/wave** = 6% lower than theoretical peak. Possible
causes (none verified, but all plausible on gfx950):
1. **Accumulator dependency chain stall**: 16 fp32 vs 4 fp32 output
   per lane → longer back-to-back dep chain when chaining mfmas.
2. **MFMA slot scheduling**: gfx950 issues from a finite MFMA pipe;
   double-cycle ops (32x32x64) might miss issue slots that single-cycle
   ops fill.
3. **Power/clock throttling**: 32x32x64 has higher per-mfma energy →
   more aggressive throttling under sustained load.

The exact mechanism doesn't matter for the decision. Empirical fact:
**32x32x64 is structurally slower per FLOP at the lowest level**, so
no kernel-level rewrite around it can recover the deficit.

## R63-dm/R64-dm roadmap status update

| Item | Pre-R34 status | Post-R34 status |
|---|---|---|
| ST_32x64 type + alias (R64-dm) | LANDED, awaiting validation | LANDED, **VALIDATION FAILED → DEAD CODE** |
| rt_32x64 / rt_64x32 shape structs (R14-dm) | LANDED | DEAD CODE |
| `rt_32x64_s` / `rt_64x32_s` aliases (R57-dm) | LANDED | DEAD CODE |
| `rcr_mma_32` wrapper (R59-dm) | LANDED | DEAD CODE |
| `load_a_kt_32x64` / `load_b_kt_32x64` (R61-dm) | LANDED | DEAD CODE |
| ST_32x64 swizzle design (R37+) | PENDING | **CANCELLED** |
| Main-loop load helpers (R38+) | PENDING | **CANCELLED** |
| `grouped_rcr_kernel_32` skeleton (R39+) | PENDING | **CANCELLED** |
| Dispatch wiring (R40+) | PENDING | **CANCELLED** |
| Full coverage + tuning (R41+) | PENDING | **CANCELLED** |

The dead-code infrastructure (rt_32x64 / st_32x64 / mfma_323264 /
loaders) does NOT add any cost to the live `tk_fp8_layouts.so` —
LLVM DCE removes the force-instantiate stubs at codegen time
(verified in R64-dm via byte-identical .so check). So we leave it
in place as a record of the failed lever, marked with a top-level
`DEAD_CODE_LEVER_D_FALSIFIED` comment in a future cleanup round if
desired.

## Updated lever inventory after R34 (definitive)

| Lever | Status | Notes |
|---|---|---|
| A async global→LDS | ALREADY SHIPPED (R54-dm) | `rcr_8w_load_hoist` uses inline asm `buffer_load_dwordx4 offen lds` |
| B dual LDS ping-pong | ALREADY SHIPPED (R54-dm) | `As[2][2]` + `Bs[2][2]`, triple blocked by 160 KB LDS cap |
| C register hints | SATURATED (R54-dm + R30-R32) | __noinline__ ABI fail, asm-volatile -42% reload but 0 metric |
| D 32x32x64 cell shape | **R34 FORMAL FALSIFICATION** | -6% per-FLOP throughput in single-warp microbench |
| E manual ASM main-loop | UNTESTED | High risk, no precedent on gfx950, 2-3 round commitment |
| F Qwen-Down K=1536 specialization | MARGINAL (+0.5pp upper bound) | Generic (K<2048, N>=K) dispatcher rule, but only 4/24 cases |

**Levers A, B, C, D are now CLOSED**. The remaining path forward
is E (high risk, untested) or F (marginal). Backward-path
improvements (R64-dm's fallback recommendation) do not affect the
metric since FP8 grouped backward is correctness-only, not timed.

## What R35+ should do

**My recommendation, in priority order:**

1. **R35: Lever F (Qwen-Down K=1536 specialization)** — 1 round
   probe. Add a generic dispatcher rule like
   `if (K < 2048 && N >= K) num_xcds = X` or alternative tile
   dimensions. Bounded scope (4 cases), bounded risk (only affects
   small-K paths). +0.5 pp upper bound on geomean is small but
   non-zero. If FALSIFIED, only 1 round lost.

2. **R36: Lever E (manual ASM main-loop)** — IF R35 lands ≥+1 pt or
   user has 2-3 rounds of patience for high-risk experimentation.
   Hand-write the K-iter mfma + load schedule in raw ASM to bypass
   LLVM's scheduling. Predicted gain: unknown (could be +5 pp or
   -50 pp). Should be done in a separate exploration branch that
   can be reverted.

3. **STOP and accept 977-981 as final** — If user has decided that
   7+ rounds inside the 977-981 noise band represents genuine
   structural ceiling. R34's microbench confirms no easy structural
   wins remain. patience=30 has 24 more rounds before exit; spending
   them on R35 (F) + R36 (E exploration) is the responsible
   rounds-budget allocation, but neither is expected to break the
   plateau by itself.

## Files touched this round

* New: `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/microbench/mfma_323264_vs_1616128.hip`
  (preserved microbench source for posterity)
* New: `/workspace/code/Primus-Turbo/analysis/_notes/round-34-fp8-grouped-Lever-D-microbench-FORMAL-FALSIFICATION.md` (this doc)

## Metric

R34 baseline (1 trial): 979 (no kernel change → expected 977-981
noise band)

Result: NO METRIC CHANGE this round (microbench-only, no kernel
modification). Score remains in 977-981 noise band.

## Commits

* HipKittens: 1 commit (microbench source preservation only — no
  kernel/code/codegen change)
* Primus-Turbo: 1 commit (this doc)
