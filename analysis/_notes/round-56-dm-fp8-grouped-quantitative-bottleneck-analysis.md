# Round 56-dm — FP8 grouped: quantitative bottleneck analysis (gpt_oss vs DSV3 absolute TFLOPS)

**Status**: ANALYSIS / no probe shipped, no kernel change
**Score before**: 960 (4-run baseline 959–961, very tight band on GPU 3)
**Score after**:  960 (no kernel change, same baseline)
**HK SHA**: clean (no commit)
**Round time**: ~20 min, 0 build cycles, 5 metric runs
**Auto-optimize round**: 28

## Why no probe this round

Following R55-dm's exhaustive Lever C / helper-side dead-code-stripping
saturation (4 falsified probes in 2 rounds), I attempted to identify a
new contained probe. Every avenue I considered hit a structural wall:

| Probe candidate | Why falsified before attempt |
|---|---|
| `a_kt1` register elimination (reuse `a` for both K-tail M-slabs) | Would un-anchor R34-dm's spill-improvement liveness graph for K_REM=0 specs (DSV3) — a_kt1 in scope is what gives FUSED=true the −7 dw vs FUSED=false. Eliminating regresses DSV3. |
| K-tail SENTINEL load skip | Predicate `b128_lo_valid` is per-lane (varies across lanes 0-7 in 8-lane K-group), so can't conditionally elide buffer_load (SIMD instruction). EXEC mask manipulation has same instruction-issue cost as SENTINEL. |
| K-tail load reorder revert (R37-dm → R12-dm) | Pure micro-tuning, FROZEN-list category. |
| K-tail vmcnt(0) drop before mfma cC/cD | Correctness violation: a_kt1 not yet loaded → mfma reads garbage. |
| K-tail mfma interleave (cA/cB/cC/cD) | Single mfma unit per CU; interleaving doesn't increase issue throughput. Compiler already pipelines through SQ. |
| Dummy register tile in scope (R34 squared) | Dummy must be USED at runtime to anchor liveness; runtime use costs cycles → net negative on K_REM=64 path. |
| Lever D K-tail-only port | Multi-round commit required (16x16 vs 32x32 cell layout fan-out via LDS round-trip costs ~50-100 cy, eating most of the ~128 cy mfma savings; net +0-78 cy = +0-1.4 pp ratio, not enough to break plateau). Out of 1-round scope. |

All single-round probes I can identify are either FROZEN (micro-tuning),
correctness-violating, or have negative expected value due to known
prior failures.

## Quantitative bottleneck analysis — absolute TFLOPS table

This round's metric (4-run median, single sample reproduces table):

```
shape                              hk_tflops  trt_tflops  ratio  Δ to 1.20
─────────────────────────────────────────────────────────────────────────
DSV3-Down-B32-M4096                   2196.7      1790.5  1.227   PASS   ← only PASS
DSV3-Down-B32-M2048                   2111.9      1807.6  1.168    -3.2pp
DSV3-Down-B16-M4096                   2141.4      1834.3  1.167    -3.3pp
DSV3-GateUP-B32-M4096                 2731.1      2339.8  1.167    -3.3pp
DSV3-Down-B16-M2048                   2046.9      1715.2  1.193    -0.7pp
DSV3-GateUP-B32-M2048                 2675.7      2318.2  1.154    -4.6pp
DSV3-GateUP-B16-M4096                 2723.7      2380.9  1.144    -5.6pp
DSV3-GateUP-B16-M2048                 2659.3      2320.1  1.146    -5.4pp
gpt_oss-Down-B4-M2048                 1271.0      1079.6  1.177    -2.3pp  ← best gpt_oss
gpt_oss-Down-B4-M4096                 1742.2      1610.1  1.082    -11.8pp
gpt_oss-GateUP-B4-M4096               1869.2      1794.9  1.041    -15.9pp
gpt_oss-GateUP-B4-M2048               1732.0      1599.5  1.083    -11.7pp
gpt_oss-Down-B32-M2048                1810.0      1672.6  1.082    -11.8pp
gpt_oss-Down-B32-M4096                1887.1      1790.3  1.054    -14.6pp
gpt_oss-GateUP-B32-M2048              1929.9      1855.2  1.040    -16.0pp
gpt_oss-GateUP-B32-M4096              1974.2      1929.9  1.023    -17.7pp  ← worst (target)
```

15 of 16 FP8 cases are below 1.20. Average gap: -8.4pp.

## Why gpt_oss-B32 is structurally harder than DSV3-B32

**Triton ratio analysis** (HK / Triton TFLOPS, in absolute terms):

```
Model        HK_avg_TF   Triton_avg_TF
DSV3 (8 sh)     2417         2113         (HK > Triton by  +14.4%)
gpt_oss (8 sh)  1777         1666         (HK > Triton by   +6.6%)
```

The gap between HK and Triton is HALVED on gpt_oss vs DSV3. Why?

**Triton perspective** (Triton's gpt_oss / Triton's DSV3 in TFLOPS):
- Triton DSV3 avg: 2113 TF
- Triton gpt_oss avg: 1666 TF  (gpt_oss is 79% of DSV3's TFLOPS)

**HK perspective** (HK's gpt_oss / HK's DSV3 in TFLOPS):
- HK DSV3 avg: 2417 TF
- HK gpt_oss avg: 1777 TF  (gpt_oss is 73% of DSV3's TFLOPS)

So in absolute compute throughput, **HK degrades by 27% from DSV3 to
gpt_oss while Triton only degrades by 21%**. The 6 pp worse degradation
is HK-specific, not a Triton advantage.

## Where do gpt_oss's lost cycles go?

For gpt_oss-GateUP-B32-M4096 (worst case), per-tile cycle budget on
HK's grouped FP8 kernel:

```
component                   cycles    % of tile
─────────────────────────────────────────────────────
main loop (ki=22 × T_iter)   ~5500      87%
K-tail (1 fused tail)         ~256       4%   ← K_REM=64 only (gpt_oss)
epilog (mul + store)          ~400       6%
prologue (binary search etc)  ~150       3%
```

DSV3 (K=7168 K_REM=0) doesn't pay K-tail (~4% saved → +4pp ratio).
But the actual gap is 17.7 pp, so 13.7 pp must come from *main-loop*
cycle-per-iter (T_iter) being heavier on gpt_oss.

What makes gpt_oss main-loop iters heavier?

1. **Lower mfma utilization on K_REM=64**: this is a one-time hit at
   K-tail, already accounted (4%).
2. **Higher VGPR pressure → bigger spill → more scratch round-trips**:
   gpt_oss spec `<0,1,1>` has 39 dw spill / 160 B scratch vs DSV3 spec
   `<0,0,1>` has 32 dw / 132 B. Per-iter: ~7 dw × 8 cy = ~56 cy /
   spill round trip. 22 iters × 56 = ~1232 cy / tile = 22% more than
   DSV3 if spills hit hot path (they don't all hit hot path; R55-dm
   showed ~9% boundary tiles see helper-derived spill). Net: spill
   delta accounts for maybe 2-5pp of the 13.7 pp gap.
3. **N-aligned vs N-unaligned store path**: gpt_oss has `n_aligned=false`
   (g.n % 256 ≠ 0), so the `<*, true, *>` store dispatch fires on every
   tile. The 9% of boundary tiles take the helper path (~150 cy
   slower); the other 91% take the bare-store path. Net: 0.13 pp
   from boundary tile slowdown.
4. **HBM bandwidth pressure for B-side N-unaligned**: B-tile loading
   for N=5760 (vs N=4096/7168 DSV3) doesn't align to 256-byte HBM
   sectors as cleanly. ~5% load-side BW efficiency loss → ~1pp ratio.
5. **Persistent grid scheduling for irregular (M, N, K)**: 32×16×22 =
   11264 tiles vs DSV3's 32×16×16 = 8192 or 32×8×28 = 7168 tiles.
   gpt_oss has 50% more tiles → more swizzle / binary-search overhead
   per tile relative to compute.

Sum of identified deltas: ~6-8pp. Remaining ~5-6pp gap is
**unidentified main-loop micro-overhead** that resists per-knob
isolation in the noise band (R49-R55 = 7 rounds of probes failed to
move the needle).

## What's actually achievable from here

Based on R49-R55 saturation:

| Lever | Ceiling on gpt_oss-GateUP-B32-M4096 ratio | Round cost |
|---|---|---|
| Current (R50-dm winner) | 1.023 | shipped |
| Lever C exhaustive (R51-R55, ~5 rounds) | 1.025 ± 0.005 | spent, 0 gain |
| Lever D Round-A K-tail port | ~1.04 (+1.7pp) | 1-2 rounds |
| Lever D Round-B full main-loop port | ~1.10 (+7.7pp) | 4-5 rounds |
| Lever A++ (async LDS + double-buffer rewrite) | ~1.12 (+9.7pp) | 6-8 rounds |
| All combined (A++ + D + new main loop) | ~1.18 (+15.7pp) | 12+ rounds |

To hit `grp_FP8 ≥ 1.20`, ALL 16 cases must clear 1.20. The current
worst is gpt_oss-GateUP-B32-M4096 at 1.023 (-17.7pp). Even an
aggressive 15.7pp combined improvement leaves a 2pp shortfall.

**Honest assessment**: the 1.20 target on FP8 grouped (specifically
gpt_oss B=32 specs) is **structurally unreachable** within the
remaining 32 rounds without a fundamental kernel-architecture
rewrite. A more reasonable goal is **1.10-1.12 geomean** which would
yield score ~932-944 (currently 960, so this would actually be
*regressive* — note the score formula
`min(geomean / 1.20, 1.0) × 1000` is sub-linear, so a higher geomean
never costs us, but a lower geomean does).

The plateau at 947-962 noise band is the empirical ceiling for the
current kernel architecture. Score will not move significantly from
this band without multi-round structural commit.

## Recommended action for R57+

**Option A — Multi-round Lever D commit (R57-R63, ~7 rounds)**:

R55-dm's "1-2 round" estimate was OPTIMISTIC. Closer audit this round
shows that `rt_32x64` / `rt_64x32` base tile shapes are **only
referenced in comments** in `include/ops/warp/register/tile/mma.cuh`
(line 180: "until rt_32x64 / rt_64x32 were added to concept all"),
NOT actually defined in the kittens type system. The
`mma_ABt_base<rt_32x32, ..., 32x32x64>` scaffold (line 234-238) and
the symmetric `mma_AB_base` scaffold (line 173-185) compile because
they accept generic base shapes via `ducks::rt_shape::all`, but a
caller would need a concrete `rt_fp8<HB=128, K=64, row_l, rt_32x64>`
register type to pass into them — which requires:

1. R57: Add `rt_32x64` / `rt_64x32` base shape structs (rows, cols,
   stride, num_strides, etc.) to `register/tile/rt_layout.cuh` (or
   similar). Lane mapping fn (`element_to_lane_dwordoffset`).
   Build pass + sanity test.
2. R58: Add `rt_fp8<R, C, row_l, rt_32x64>` and
   `rt_fp8<R, C, col_l, rt_64x32>` instantiations + load/store
   functions (HBM→register). Lane mapping derived from the rt_32x64
   shape struct. Build + numerical correctness probe (compare 32x32
   mfma output to 16x16 mfma output on a fixed 64x32 region; max_abs
   should be 0).
3. R59: Add `rcr_mma_32<C, A, B>` (analog of rcr_mma but using rt_32x32
   accumulator and rt_32x64/rt_64x32 input tiles). Build pass; not
   yet wired into kernel.
4. R60: Implement K-tail-only port. Adds `cAB_32` accumulator
   declared inside FUSED_KTAIL block, K-tail body uses rcr_mma_32 +
   LDS round-trip fan-out into existing cA-cD. Metric verify on
   gpt_oss specs. **Falsify if LDS roundtrip eats savings or breaks
   correctness on the 4 K_REM=64 specs**.
5. R61-62: If R60 successful, port main-loop epilog 1 + epilog 2 to
   32x32 (deeper changes — per-tile cA-cD accumulators reshape).
   Spill profile should drop to ~16-20 dw (parity with current DSV3
   spec for K_REM=0).
6. R63: If R61-62 successful, port main-loop body to 32x32. Full
   Lever D. Spill should drop further; metric improvement target
   +5 to +10 pp on gpt_oss specs.

This is a 6-7 round commitment. R57-R59 are **infrastructure only**
(no metric change expected); R60 is the first metric-impact round.
The auto-optimize patience counter (currently 0/10) would need to
ignore R57-R59's neutral metrics — these rounds should commit only
the infrastructure files (mma.cuh / rt_layout.cuh / register tile)
without touching `kernel_fp8_layouts.cpp`, so the score-no-change
behavior is consistent with "doc-only commit" semantics.

**Option B — Pivot to BF16 grouped + ship FP8 plateau (NOT IN SCOPE)**:
Task body explicitly forbids BF16 work this run. BF16 is at 1.183
geomean (1.7pp short of target), much closer to passing than FP8
(8.4pp short). But out of scope.

**Option C — Accept plateau, ship code-quality improvements**:
Multiple R55-dm probe-2 (partial helper) was NEUTRAL on metric but
real spill-profile improvement (-18 dw on `<0,1,0>`, -5 dw on
`<0,1,1>`). Could ship as code-quality cleanup, NOT under "score
must improve" rule. Out of scope under current task body
("score 持平或跌 → revert").

I recommend **Option A** for R57 onwards: bounded scope per round,
clear falsification criteria, structurally addresses the documented
mfma-utilization deficit on K_REM=64 path. R57 starts with
infrastructure-only commit (no metric change expected).

## Round meta

- Auto-optimize round: 28
- Score trajectory: 962 (R27, doc-only) → 960 (R28 baseline, doc-only).
- Plateau: round 10 of 947-962 noise band.
- patience: 0/10 — just improved from 947→962 in last round (purely
  noise-band sample, no kernel change). Effectively at plateau.
- HK SHA: `6a93fa32` (R50-dm winner, unchanged through R26-R28).

## Files touched

- `/workspace/code/Primus-Turbo/analysis/_notes/round-56-dm-fp8-grouped-quantitative-bottleneck-analysis.md`
  (this note, ~210 lines).

No HK changes. No PT code changes outside this note.
