# Round 93 (auto-loop round 16) — BF16 grouped GEMM: Lever B4 per-group block partition FALSIFIED via XCD-driven L2 thrash

## Context

R15 (auto-loop) finished R92 with Lever A1 (forward K-tail prefetch) classified
"partially exhausted" at +2 score / +6 VGPR cost — below the +5 commit gate.
The R92 falsification note recommended R16 try one of:

1. PMC-profile a DSV3/Qwen3 K%128==0 worst-shape (diagnostic only)
2. **Implement R91 Lever B4: per-group chunked partition for
   `grouped_var_k_kernel`** — distinct from the R75 atomic-claim variant.

This round picked option 2. Goal: lift the var-K backward path's L2 hit
rate by pinning each CU to a single group throughout the persistent loop.

## Hypothesis

Current schedule in `grouped_var_k_kernel` (HipKittens
`analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:4930`):

```cpp
int pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, 64);
for (int gt = pid; gt < total_tiles; gt += NUM_CUS) {
    const auto gl = compute_var_k_group_lookup(gt, tiles_per_group, s_offs);
    if (!gl.valid) continue;
    const int group_idx = gl.group_idx;
    ...
}
```

`chiplet_transform_chunked` is identity for grid 256 (the comment in the
kernel itself says so), so effectively `pid = blockIdx.x`. Each CU
iterates `gt = blockIdx.x, blockIdx.x + 256, ...`, advancing across
`group_idx = gt / tiles_per_group` per iteration.

Hypothesised problem: each CU touches `≈ ceil(total_tiles / NUM_CUS)`
distinct groups. For G=32 / `tiles_per_group=144`,
`total_tiles=4608`, that's 18 groups per CU, each with its own M_g × N
A-stripe (~11.8 MB for the gpt_oss-Down config). The naive picture says
this trashes L2.

Hypothesised fix: pin each CU to ONE group. When `cus_per_group =
NUM_CUS / G ≥ 2` and `cus_per_group * G == NUM_CUS` (metric env: G ∈
{4, 16, 32}, all divide 256), assign:

```
group_idx   = blockIdx.x / cus_per_group
cu_in_group = blockIdx.x % cus_per_group
```

Then the inner loop is `local_tile = cu_in_group; local_tile +=
cus_per_group` through `tiles_per_group`. Group state (`m_start_g`,
`ki_g`, `k_offset_tiles`) is loop-invariant — hoisted out of the inner
loop, freeing the per-iteration LDS reads + integer divides paid by
`compute_var_k_group_lookup`.

Predicted gain: 2-3% on the bwd path (~1-2% wall) from L2 reuse + a
sliver from hoisted group lookup. Naive arithmetic put the metric delta
at ≈ +5-7 score.

## Implementation

`HipKittens kernel_bf16_dynamic.cpp:4929-5011` rewritten to dispatch
between two partition modes at kernel entry based on `per_group_balanced
= (G > 0) && (cus_per_group * G == NUM_CUS) && (cus_per_group >= 2)`:

* True branch: per-group block partition with hoisted group state and
  loop-invariant `m_start_g_inv / ki_g_inv / k_offset_tiles_inv`.
* False branch: original stride-NUM_CUS partition (unchanged from R75
  baseline; fallback for non-metric workloads where G doesn't divide
  NUM_CUS evenly).

Both branches share the same per-tile body (zero accumulators →
`device_gemm_tile_body<Layout::CRR>` → 4× `store_c_tile_mn_masked_grouped`
→ `s_waitcnt vmcnt(0) lgkmcnt(0)` → barrier).

Build report:

```
TotalSGPRs: 98          (was 94 — +4 SGPRs for the partition predicate
                                  + group_idx + cu_in_group)
VGPRs:      256         (unchanged, ceiling)
SGPRs Spill: 0
VGPRs Spill: 0
Occupancy [waves/SIMD]: 2 (unchanged)
LDS Size [bytes/block]: 272 (unchanged)
```

No spills, no occupancy regression. SGPR delta is in noise.

## Correctness probe

Custom probe (`/tmp/probe_b4.py`) on 4 shapes covering G ∈ {4, 16, 32}
plus a Qwen3 K=1536 case (different K geometry):

| shape | dA SNR | dA allclose | dB SNR | dB allclose |
|---|---|---|---|---|
| gpt_oss-Down-B4-M2048   (G=4)  | 49.6 dB | ✓ | 49.6 dB | ✓ |
| DSV3-Down-B16-M2048     (G=16) | 49.6 dB | ✓ | 49.6 dB | ✓ |
| gpt_oss-Down-B32-M2048  (G=32) | 49.6 dB | ✓ | 49.6 dB | ✓ |
| Qwen3-Down-B16-M2048    (G=16) | 49.6 dB | ✓ | 49.6 dB | ✓ |

49.6 dB matches the OLD-stride baseline (it's the bf16 floor for this
class of GEMM). Per-group partition is bit-clean.

## Metric A/B

```
Baseline (R92 HEAD = 7207704e):  score 881-882 (noise ±1)
Lever B4 (per-group partition):  score 804     (one run, no need to repeat)
                                  delta = -77   (catastrophic)
```

Per-shape ratio delta:

| family / shape                       | OLD ratio | NEW ratio | Δ pp   |
|--------------------------------------|----------:|----------:|-------:|
| gpt_oss-GateUP-B4-M2048   (G=4)      | 1.126     | 1.111     |  -1.5  |
| gpt_oss-Down-B4-M2048     (G=4)      | 1.048     | 1.043     |  -0.5  |
| gpt_oss-GateUP-B4-M4096   (G=4)      | 1.120     | 1.111     |  -0.9  |
| gpt_oss-Down-B4-M4096     (G=4)      | 1.105     | 1.095     |  -1.0  |
| **gpt_oss-GateUP-B32-M2048 (G=32)**  | **1.102** | **0.970** | **-13.2** |
| **gpt_oss-Down-B32-M2048   (G=32)**  | **1.053** | **0.922** | **-13.1** |
| **gpt_oss-GateUP-B32-M4096 (G=32)**  | **1.100** | **0.958** | **-14.2** |
| **gpt_oss-Down-B32-M4096   (G=32)**  | **1.084** | **0.929** | **-15.5** |
| DSV3-GateUP-B16-M2048      (G=16)    | 1.120     | 1.043     |  -7.7  |
| DSV3-Down-B16-M2048        (G=16)    | 1.110     | 0.988     | -12.2  |
| DSV3-GateUP-B16-M4096      (G=16)    | 1.137     | 1.053     |  -8.4  |
| DSV3-Down-B16-M4096        (G=16)    | 1.106     | 0.978     | -12.8  |
| DSV3-GateUP-B32-M2048      (G=32)    | 1.125     | 0.980     | -14.5  |
| DSV3-Down-B32-M2048        (G=32)    | 1.102     | 0.968     | -13.4  |
| DSV3-GateUP-B32-M4096      (G=32)    | 1.150     | 0.984     | -16.6  |
| DSV3-Down-B32-M4096        (G=32)    | 1.105     | 0.960     | -14.5  |
| Qwen3-GateUP-B16-M2048     (G=16)    | 1.120     | 0.998     | -12.2  |
| Qwen3-Down-B16-M2048       (G=16)    | 1.106     | 0.994     | -11.2  |
| Qwen3-GateUP-B16-M4096     (G=16)    | 1.107     | 0.958     | -14.9  |
| Qwen3-Down-B16-M4096       (G=16)    | 1.103     | 0.986     | -11.7  |
| Qwen3-GateUP-B32-M2048     (G=32)    | 1.113     | 0.979     | -13.4  |
| Qwen3-Down-B32-M2048       (G=32)    | 1.100     | 0.980     | -12.0  |
| Qwen3-GateUP-B32-M4096     (G=32)    | 1.126     | 0.964     | -16.2  |
| Qwen3-Down-B32-M4096       (G=32)    | 1.115     | 0.973     | -14.2  |

**Strong G-dependence**: regression scales monotonically with `G`
(equivalently, scales inversely with `cus_per_group`):

| G  | cus_per_group | typical Δ pp |
|----|---------------|--------------|
| 4  | 64            | -0.5 to -1.5 |
| 16 | 16            | -7.7 to -14.9 |
| 32 |  8            | -12.0 to -16.6 |

## Why it falsified — XCD locality, not L2 capacity

The hypothesis was wrong about WHERE the L2 footprint lives in the OLD
stride. Walked through the OLD partition's actual access pattern per
XCD:

* HW round-robin assigns blockIdx.x to XCDs as `xcd = blockIdx.x %
  NUM_XCDS` (this is exactly what `chiplet_transform_chunked` is built
  around). So XCD 0 sees `blockIdx.x ∈ {0, 8, 16, ..., 248}` — 32 CUs.
* OLD stride: `gt = blockIdx.x` in iter 0 ⇒ XCD 0's gt sequence in
  iter 0 = {0, 8, ..., 248}. With `tiles_per_group=144` (gpt_oss-Down),
  that's `group_idx = floor(gt/144) ∈ {0(×18), 1(×14)}`. **XCD 0 in iter
  0 touches only 2 groups**, not 4-32.
* iter 1: `gt += 256` ⇒ XCD 0's gt ∈ {256, 264, ..., 504}. group_idx ∈
  {1(×6), 2(×14), 3(×12)}. 3 groups.
* iter 2: gt 512..760. group_idx covers 3-4 (depending on G).
* The pattern: each XCD touches a **rolling 2-3 group window** per
  iteration. L2 footprint per iteration phase ≈ 2-3 × 11.8 = 23-35 MB
  per XCD — fits L2 (or close).
* As the kernel progresses, the rolling window advances, evicting old
  groups for new ones. Aggregate footprint over the whole kernel is
  large, but the ACTIVE footprint at any instant is small.

The NEW per-group partition destroyed this rolling-window structure:

* Per-group partition: `group_idx = blockIdx.x / cus_per_group`. For
  G=32 / `cus_per_group=8`, blockIdx.x 0..7 → group 0. These 8
  blockIdx.x values, under HW round-robin XCD assignment, map to XCDs
  0, 1, 2, 3, 4, 5, 6, 7 — **one CU per XCD**.
* Each XCD has 32 CUs. With G=32 groups, EVERY XCD hosts 32 different
  groups simultaneously (1 CU per group per XCD).
* Per-XCD active footprint = 32 × 11.8 = **378 MB**. Vastly exceeds
  any plausible per-XCD L2 (~32-64 MB on MI355X). Total L2 thrash.
* G=16 case: 16 groups × 16 CUs = 32 CUs / XCD. Each XCD hosts 16
  groups → 16 × footprint. Still way over L2.
* G=4 case: 4 groups × 64 CUs = 32 CUs / XCD distributed over 4
  groups (8 CUs per XCD per group). Per-XCD footprint = 4 × 11.8 = 47
  MB. Closer to fitting; matches the small G=4 regression (-1pp).

The G-dependent regression scaling (4 → 16 → 32) **confirms the
mechanism**: smaller `cus_per_group` ⇒ HW round-robin spreads each
group across more XCDs ⇒ each XCD ends up servicing more groups
concurrently ⇒ proportionally larger L2 footprint per XCD.

## Lessons / what would actually work

The fundamental flaw is naively assigning `group_idx = blockIdx.x /
cus_per_group` without considering the HW round-robin XCD mapping. The
OLD stride works precisely because `gt = blockIdx.x` happens to be the
identity-cluster mapping under round-robin: consecutive blockIdx.x
values map to consecutive groups within a single XCD, so each XCD's CUs
naturally cluster on 2-3 groups at a time.

A working per-group partition needs to be **XCD-affine**:

```
xcd = blockIdx.x % NUM_XCDS              (0..7)
sub = blockIdx.x / NUM_XCDS              (0..31)  // CU index within XCD
groups_per_xcd = max(1, G / NUM_XCDS)
cus_per_group_per_xcd = max(1, 32 / groups_per_xcd)

local_group = sub / cus_per_group_per_xcd  (0..groups_per_xcd-1)
group_idx   = xcd * groups_per_xcd + local_group
cu_in_group = sub % cus_per_group_per_xcd
```

For G=32 / NUM_XCDS=8: `groups_per_xcd=4`, `cus_per_group_per_xcd=8`.
Each XCD hosts 4 groups (its assigned slice). Per-XCD footprint = 4 ×
11.8 = 47 MB. Closer but still overshoots L2.

For G=16: `groups_per_xcd=2`, per-XCD footprint = 23.6 MB. Fits.

For G=4: `groups_per_xcd=1` (G < NUM_XCDS), need to flip — multiple
XCDs per group. Adjusted formula required.

This is doable but **complex enough to be a new attack vector for R17+**,
not a one-line patch. Documented as Lever B4-XCD in the next section's
plan.

## Risk-assessment alignment with R75/R83-92 streak

This round adds to the running list of "naive var-K partition rewrites
that look obvious but fail":

* R75: atomic-claim work-stealing — falsified
* R83-90: var-K LDS swizzle / cross-permute / pad / rematerialize streak
  (8 rounds, all falsified, R91 declared "cumulative falsification
  pivot")
* R92: forward K-tail prefetch — partially falsified (+2 score below +5
  gate)
* **R93 (this round)**: per-group block partition (naive) — falsified
  via XCD-driven L2 thrash

The R91 cumulative-falsification thesis (var-K partition is at a stable
local optimum that resists naive rewrites) is reinforced. Real progress
needs PMC profiling to identify what the K%128==0 path is actually
bottlenecked on. R83-92 collectively show speculation-without-data
yields ~10% chance of clearing the +5 gate.

## Round result

* Lever: B4 per-group block partition for `grouped_var_k_kernel`
* Selected target: lowest-progress shape from R16 metric was
  `gpt_oss-Down-B4-M2048` (ratio 1.048, weight 3) — but the lever
  affects all 24 shapes since it's a backward-path change applied at the
  kernel level
* Files touched (then reverted):
  - `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:4929-5011`
* Build: VGPR=256 (unchanged), SGPR=98 (+4), no spills, occ=2 unchanged
* Correctness: 4/4 dA + dB allclose PASS (49.6 dB SNR matching baseline)
* Metric:
  - Before: score 881 (gpt_oss 1.092 / DSV3 1.119 / Qwen3 1.111)
  - After: score 804 (gpt_oss 1.014 / DSV3 0.994 / Qwen3 0.979)
  - Δ = **-77** (catastrophic, falsified)
* Action: REVERTED. Final baseline check after revert: score 882
  (within ±1 noise of pre-round 881).

## Recommended R17+ plan

Given R83-93 reinforce the "naive partition rewrites fail" pattern, the
next attack vector should be either (a) data-driven (PMC profile first,
then precision lever) or (b) a structurally different lever (forward
path, dispatcher, or compile-time specialization), not another naive
backward partition.

Options ranked:

1. **PMC profile a single representative shape first** (no-commit
   diagnostic round). Specifically `DSV3-Down-B16-M2048` (ratio 1.106,
   gpt-K%128==0 path) AND `gpt_oss-Down-B32-M2048` (ratio 1.053, K-tail
   path). Look for: MFMA util, LDS bank conflicts, HBM read BW vs peak,
   waves/SIMD waste. Document findings, don't commit code. R94 commits
   the diagnostic note only.

2. **Lever B4-XCD**: implement the XCD-affine per-group partition
   sketched in "Lessons" section above. Fixes the L2-thrash failure
   mode. May still under-deliver for G=32 (footprint 47 MB > L2). Best
   case is +2-5 score on G=16 shapes, with G=32 unchanged. Risk:
   another partial falsification.

3. **Lever C: uniform-M dB BMM dispatch**. R92 note flagged this
   needs new HK kernel binding (uniform-M dB doesn't currently route
   through `grouped_var_k_kernel`). Multi-round work; defer past R20.

4. **Lever B1: MFMA pipeline scheduling on K%128==0 path**. Untouched
   in 92 rounds. Highest theoretical ceiling on DSV3/Qwen3 (the
   K%128==0 path is the score plateau zone). Needs PMC data first to
   confirm the kernel is MFMA-undersaturated.

R17 picks option 1: PMC diagnostic round. R18+ uses the data to choose
between B4-XCD and B1.

## SHA pointers

* Primus-Turbo HEAD pre-round: `7207704ed63f78030552a2d3458cd6011db83a72`
* HipKittens HEAD: unchanged (revert restored to baseline)
* This note committed in Primus-Turbo only; HipKittens working tree
  clean post-revert.
