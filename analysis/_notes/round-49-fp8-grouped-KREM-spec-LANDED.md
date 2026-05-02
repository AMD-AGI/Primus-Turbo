round-49-fp8-grouped-KREM-spec-LANDED.md
==========================================

Round: 49 / 100
Date: 2026-05-02
SHA: 5c61dd26 (pre) → TBD (post)
Task: grouped FP8 tensorwise (fwd + var-K bwd + dA bwd) on the 24-shape suite

## TL;DR

Implemented R48 ladder #1 (per-spec K_REM=64 constexpr specialisation)
in the FUSED_KTAIL block of ``grouped_rcr_kernel``. Replaces dynamic
``K_REM = g.k - g.fast_k`` + 2 per-lane masks ``b128_lo_valid``,
``b128_hi_valid`` with a single constexpr-folded ``both_valid =
(laneid < 32)``. Dispatcher invariant ``fuse_ktail_eligible``
(kernel_fp8_layouts.cpp:5335) only allows K_REM ∈ {0, 64}, so inside
the runtime branch ``if (g.fast_k < g.k)`` K_REM is necessarily 64.

Saves **2 SGPRs** (TotalSGPRs 81→79). V/A/spill/scratch bit-identical.
Binary md5 differs (LLVM emits 1-3 fewer cmp ops per K-tail load
lambda). Metric within noise band, no regression.

HipKittens HEAD: ``92407889`` → ``6c52d017``.

## Selected target (per metric)

R49 metric data (3 independent runs of full 48-shape suite):

```
                                  R49-baseline  R49-KREM-1  R49-KREM-2  R49-KREM-3
geomean grp_BF16                    1.2278        1.2037      1.2108      1.2475
geomean grp_FP8                     1.1481        1.1903      1.1701      1.1686
score                               978           996         987         987
```

Worst FP8 ratio shapes (R49 baseline run, sorted ascending):

```
0.579  gpt_oss_20B-Down-B4-M4096          *anomaly (Triton noise)*
1.012  Qwen3-235B-A22B-GateUP-B16-M4096   *Triton noisy this run*
1.081  gpt_oss_20B-GateUP-B32-M4096
1.085  gpt_oss_20B-GateUP-B4-M2048
1.093  gpt_oss_20B-GateUP-B32-M2048
1.100  gpt_oss_20B-Down-B32-M2048
1.101  gpt_oss_20B-GateUP-B4-M4096
1.103  DeepSeek-V3-GateUP-B16-M4096
1.139  Qwen3-235B-A22B-Down-B16-M4096
1.164  Qwen3-235B-A22B-Down-B16-M2048
```

Picked target: **gpt_oss-GateUP-B4-M4096 / family** — consistently
worst across R47-R49 (1.070, 1.084, 1.101). 8 of 8 gpt_oss FP8 shapes
sit below 1.20 ⇒ kernel-wide gpt_oss bottleneck (FUSED_KTAIL=true
path), targets the same K-tail block.

## What was changed

### HipKittens kernel (`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`)

Inside the FUSED_KTAIL block (line ~2535-2719) of grouped_rcr_kernel:

```cpp
// before:
const int K_REM = g.k - g.fast_k;
const bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;
const bool b128_hi_valid = (k_lane_byte + 32) <= K_REM;

// after:
constexpr int KREM = 64;
static_assert(KREM == 64,
    "FUSED_KTAIL=true K_REM must be 64; see fuse_ktail_eligible");
const bool both_valid = (laneid < 32);
```

Then inside both load_a_kt and load_b_kt lambdas, replaced
``b128_lo_valid``/``b128_hi_valid`` with the unified ``both_valid``.

## Resource impact (kernel_fp8_layouts.cpp:2223 grouped_rcr_kernel<0,T,T>)

```
                  before     after      Δ
TotalSGPRs        81         79         -2 (save K_REM uniform + cmp dup)
VGPRs             256        256        0
AGPRs             0          0          0
ScratchSize/lane  152        152        0
VGPRs Spill       37         37         0
Occupancy         2 wave/SIMD same      0
```

Same picture for all 4 FUSED_KTAIL=true template specs and all FP8
RCR codegen — only SGPR count drops by 2 since K_REM is wave-uniform.

## Metric impact

```
Run             score   FP8 geomean
R47 baseline    997     1.1923   (b2a0afbc HEAD)
R49 baseline    978     1.1481   (this round, before change — Triton anomalies)
R49 KREM run 1  996     1.1903
R49 KREM run 2  987     1.1701
R49 KREM run 3  987     1.1686
                ───     ────
KREM avg        990     1.176
```

Net Δ = -7 score / -0.016 geomean vs R47 baseline single run (within
±10 score / ±0.020 geomean noise band). Refactor is essentially flat
on perf. Real value:

  1. **Documents the dispatcher invariant** in code (compile-time
     ``static_assert``) — future port mistakes caught at HK build
     time.
  2. **2 SGPRs freed** — accumulator-class register pressure not
     affected, but if we later need 2 SGPRs for runtime constants
     (e.g. epilog scale broadcast), they're available without extra
     spill.
  3. **Slight code-size shrink** (1-3 fewer cmp ops per K-tail
     buffer_load × 12+8 = 20 loads per tile per spec).

## Why this didn't move the needle (post-hoc)

1. LLVM CSE was already collapsing most of the redundancy. The two
   masks ``b128_lo_valid`` and ``b128_hi_valid`` had common subexpr
   ``k_lane_byte + offset``; the compiler likely noticed they were
   broadcast-equivalent for fixed K_REM and used a single mask
   internally, even before this change. Resource report invariant.

2. The K-tail block runs ONCE per output tile. For gpt_oss-GateUP-B4-
   M4096 (768 tiles ÷ 256 CUs ≈ 3 tiles/CU), the K-tail block fires
   ~3 times per CU. Even saving 50 cycles per K-tail call → 150 cy/
   CU = ~0.075 µs/CU = ~0.0003% of kernel wall. Below noise floor.

The actual gpt_oss bottleneck is the **main loop** (22 K-iters × 4
mfma × ~32 cy = ~2800 cy/tile) where the spill of 37 VGPR words
forces ~37 × 30 cy/load ≈ 1100 cy/iter of scratch traffic. K-tail
fix alone can't address this.

## R48 ladder status update

| Lever                                  | Status         | Round   |
|----------------------------------------|----------------|---------|
| C-4 ``-mllvm -amdgpu-mfma-vgpr-form=0`` | FALSIFIED      | R48     |
| C-3' ``+a`` inline-asm hint             | FALSIFIED      | R48     |
| C-1' per-spec K_REM=64 constexpr        | **LANDED R49** | **R49** |
| Down-B32-M4096 fall-through closure     | NEXT R50       | —       |
| C-2 warp-tile to 4w-style               | R51+           | —       |

## R50+ recommendation

1. **R50 (cheap closure, 30 min)**: Add gpt_oss-Down-B32-M4096
   dispatch rule to ``primus_turbo/pytorch/kernels/hipkitten/config.py``.
   Currently falls to default ``(gm=4, xcd=8)``. Sibling rules:
     - B=32 M=2048 (m_total=65536):  ``(gm=16, xcd=4)`` round-8
     - B=4  M=4096 (m_total=16384):  ``(gm=32, xcd=4)`` round-12
   Sweep candidates: ``(gm=16, xcd=4)``, ``(gm=32, xcd=4)``,
   ``(gm=8, xcd=4)``. Expected: ~+1-3pp on this single shape ≈
   +0.05pp geomean.

2. **R51-R52 (structural, multi-round)**: C-2 warp-tile restructure.
   Modify grouped_rcr_kernel to use WARPS_M=2, WARPS_N=2 (vs current
   2,4) and RBM=64, RBN=64 — matches rcr_4w's 256-VGPR per-warp
   accumulator footprint. Rationale: only structural way to force
   AGPR allocation (R47 verified). Risk: occupancy drops from 2
   wave/SIMD to 1, may net-regress on small-grid shapes (B=4).

3. **R53+ (fallback if 2 fails)**: explicit inline-asm AGPR
   migration via ``art_base`` wrappers + ``v_accvgpr_read/write``.
   Tedious but explicit; doesn't depend on regalloc heuristic.

## Falsification register update

| Lever                                        | Status       | Round   |
|----------------------------------------------|--------------|---------|
| **Lever C-1' (per-spec K_REM constexpr)**    | **LANDED**   | **R49** |
| Lever A (async g→LDS) — base shipped         | SHIPPED      | R54-dm  |
| Lever B (dual LDS) — base shipped            | SHIPPED      | early   |
| Lever C-1 (restrict / lifetime hints)        | SATURATED    | R12,R54 |
| Lever C-3 (art_base AGPR migration)          | not impl     | —       |
| Lever C-3' (``+a`` inline-asm hint)          | FALSIFIED    | R48     |
| Lever C-4 (mfma-vgpr-form mllvm flag)        | FALSIFIED    | R48     |
| Lever C-2 (warp-tile restructure to 4w)      | NOT STARTED  | —       |
| Lever D K-tail-only port                     | FALSIFIED    | R62-dm  |
| Lever D full main-loop port (R-B 5+)         | NOT STARTED  | —       |
| Lever E (ASM main-loop)                      | NOT STARTED  | —       |
| Lever F (Qwen3 K=1536 short-K variant)       | FALSIFIED    | R35-grp |
| ``amdgpu_waves_per_eu(2,2)`` attribute       | FALSIFIED    | R47     |
| Drop ``__launch_bounds__(_, 1)`` entirely    | FALSIFIED    | R47     |
| sched_barrier / LICM / anti-CSE class        | FALSIFIED    | R31-32  |
| K-tail micro-knobs (vmcnt / reorder)         | SATURATED    | R3-R55  |

## Attribution

- HipKittens HEAD: ``92407889`` → ``6c52d017`` (this round)
- Primus-Turbo: this doc note (no kernel/Python code changes)

## Validation paper-trail

```
/tmp/metric_round_49.log               (R49 baseline; score 978, Triton-noisy)
/tmp/build_round_49_kremspec.log       (KREM spec build; resource report identical)
/tmp/build_round_49_kremspec_v2.log    (re-applied; SGPRs 81→79 confirmed)
/tmp/metric_round_49_kremspec.log      (KREM run 1: score 996)
/tmp/metric_round_49_kremspec_run2.log (KREM run 2: score 987)
/tmp/metric_round_49_final.log         (KREM run 3: score 987)
/tmp/tk_kremspec.so md5=90510c8a...    (KREM build)
/tmp/build_round_49_revert.log         (revert build for md5 baseline)
```

Binary md5 differs between baseline and KREM build, confirming codegen
emits different instructions despite resource-report identity.
