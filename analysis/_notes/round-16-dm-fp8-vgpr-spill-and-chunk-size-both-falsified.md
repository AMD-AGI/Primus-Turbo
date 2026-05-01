# Round-16-dm — VGPR/scratch reduction + chunk_size sweep BOTH FALSIFIED

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `cb93124e` (round-15-dm rocprof pivot)
**HipKittens HEAD**: unchanged (kernel reverted to round-15 state)
**Primus-Turbo HEAD after**: this commit (Primus-Turbo notes-only; kernel/config bytes unchanged)

**Metric**: 822 baseline → 822 (no kernel change shipped; all probes reverted)

---

## Round target (per round-15 priority list)

Lowest FP8 ratio shape this round = **`grpFP8_DeepSeek-V3-Down-B16-M4096` at
0.951** (HK 1359.7 TF / Triton 1430.0 TF). DSV3-Down dominates the FP8 gap
across all 4 metric variants (B∈{16,32}, M_per∈{2048,4096}); ratios
0.951–0.973. Round-15-dm rocprof established that DSV3 **MfmaUtil is at
parity with Triton** (40.5 vs 40.0%) — gap is non-PMC wall-time tax.

Round-15-dm's #1 candidate was VGPR/scratch reduction. Build with
`-Rpass-analysis=kernel-resource-usage` confirms `grouped_rcr_kernel`
hits the 256-VGPR ceiling and spills 67-76 VGPRs across all 4
template specialisations:

```
_Z18grouped_rcr_kernel<0,false,false>: VGPRs=256, AGPRs=0,
                                       Scratch=272 B/lane, Spill=67
_Z18grouped_rcr_kernel<0,true,false>:  VGPRs=256, AGPRs=0,
                                       Scratch=308 B/lane, Spill=76
_Z18grouped_rcr_kernel<0,false,true>:  VGPRs=256, AGPRs=0,
                                       Scratch=196 B/lane, Spill=48
_Z18grouped_rcr_kernel<0,true,true>:   VGPRs=256, AGPRs=0,
                                       Scratch=236 B/lane, Spill=58
_Z18grouped_rrr_kernel<0>:             VGPRs=256, AGPRs=0,
                                       Scratch=308 B/lane, Spill=76
_Z24grouped_var_k_kernel_fp8<0>:       VGPRs=256, AGPRs=0,
                                       Scratch=212 B/lane, Spill=52
```

For comparison the dense `kernel` (line 1034) already takes a **register
allocation path with no spills**: `VGPRs=198, AGPRs=256, Scratch=0, Spill=0`
(same Occupancy=2 waves/SIMD). The compiler has chosen to use AGPRs for
the dense kernel's accumulator but not for grouped's. The two kernels
differ in WG size (dense=4 warps, grouped=8 warps) and total live-range
count, which steer the allocator.

## Falsification 1: AMDGPU attribute hints have NO codegen effect

Tried 4 single-attribute interventions to coerce AGPR usage:

| attribute                                       | VGPRs | AGPRs | Scratch | Spill | Occupancy |
|------------------------------------------------|-------|-------|---------|-------|-----------|
| baseline `__launch_bounds__(_NUM_THREADS, 1)`  | 256   | 0     | 272     | 67    | 2 w/SIMD  |
| `+ amdgpu_num_vgpr(192)`                       | 256   | 0     | 272     | 67    | 2 w/SIMD  |
| `+ amdgpu_num_vgpr(128)`                       | 256   | 0     | 272     | 67    | 2 w/SIMD  |
| `__launch_bounds__(_NUM_THREADS, 2)`           | 256   | 0     | 272     | 67    | 2 w/SIMD  |
| `+ amdgpu_waves_per_eu(1, 1)`                  | 256   | 0     | 272     | 67    | 2 w/SIMD  |

**All four byte-identical.** The compiler ignores the hints because the
8-warp WG architecture + persistent loop's static live VGPR demand already
saturates the 256-VGPR/wave budget at the 2-waves-per-SIMD floor that an
8-warp WG forces on a 4-SIMD CU. There's no slack for the allocator to
trade against.

Confirmed via `clang -mllvm --help-list-hidden`: the only relevant LLVM
flag is `--amdgpu-mfma-vgpr-form` which **forces VGPR** for MFMA
operands, not the reverse. There is no "force AGPR" flag in this clang
build — AGPR allocation is purely heuristic-driven.

Implication: VGPR/scratch reduction requires **source-code refactoring**,
not attribute hints. Concretely, dropping below the 256-ceiling needs
either:

(a) A multi-section accumulator split — keep only 2 of {cA,cB,cC,cD}
    alive concurrently, swap via LDS roundtrip per section. Saves
    ~64 VGPR. Risk: LDS swap costs extra `ds_write_b128`/`ds_read_b128`
    pairs per section, may exceed spill cycles saved.

(b) Reload `b0`/`b1` per section instead of holding both live across all
    4 sections. Saves ~32 VGPR. Risk: the LDS slot is overwritten by the
    k+2 prefetch (line 2198/2211 in `kernel_fp8_layouts.cpp`) before
    section C runs, so the reload data isn't there — would require
    re-architecting the prefetch ordering.

Both are multi-round projects, not single-line changes. Defer to future.

## Falsification 2: chunk_size sweep regresses

`chiplet_transform_chunked(workgroup_id, NUM_CUS, num_xcds, chunk_size)`
hardcoded to `chunk_size=64`. With NUM_CUS=144 and num_xcds=8 (default
for gpt_oss + DSV3-GateUP — 8 of 16 metric shapes), `block = num_xcds *
chunk_size = 512 > 144 → limit = (144/512)*512 = 0 → return
workgroup_id` — the chiplet transform is a **NO-OP** for these 8 shapes.

Round-67's intent (per the comment around line 2002) was to swizzle the
WG → XCD assignment for L2 reuse. With chunk=64 the swizzle is silently
skipped on the default-xcds path. Hypothesised that reducing chunk_size
would activate the swizzle and improve L2 hit rate.

Sweep results (FP8 forward grouped only, single metric run each):

```
chunk_size  block(xcds=8)  active  score
   8           64           128/144  814   ← regression
  18          144           144/144  809   ← worse regression
  64          512             0/144  822   ← baseline (no swizzle, default)
```

Per-shape damage at chunk=18 vs baseline:

```
shape                                 baseline   chunk=18   delta
DSV3-V3-Down-B16-M4096                  0.951     0.927     −0.024
gpt_oss-Down-B4-M2048                   1.012     0.955     −0.057
gpt_oss-Down-B4-M4096                   0.964     0.949     −0.015
gpt_oss-GateUP-B4-M4096                 0.969     0.954     −0.015
DSV3-GateUP-B32-M4096                   1.063     1.022     −0.041
gpt_oss-Down-B32-M2048                  0.993     0.983     −0.010
... only DSV3-Down-B32-M4096 was unchanged ...
```

The active swizzle hurts L2 reuse for the metric mix. Reasoning: with
group_m=4 (default for gpt_oss / DSV3-GateUP), the natural
``blockIdx.x``-ordered scheduling already provides enough M-axis
locality (consecutive WGs share M-row). Cross-XCD swizzle scrambles
this M-locality without a corresponding gain in N-axis reuse.

The intended use of chunk_size is shape-specific (large chunks to keep
M-locality dominant; small chunks where N-locality dominates). Round-67
default of chunk=64 effectively *disables* the swizzle for default-xcds
shapes — that turns out to have been the right choice. The DSV3-Down
family explicitly opts in to num_xcds=2 (round-68 sweep), keeping
swizzle active there with block=128.

To make chunk_size shape-specific would require plumbing it through
`grouped_layout_globals` + the FP8 grouped binding + `HipKittenConfig`
(mirror of the round-67 `num_xcds` plumb). 4 file edits, additive
non-breaking. Not done this round; defer until rocprof shows L2 hit
rate is the bottleneck (current data does not).

## Updated falsification banks

Add to `_falsified_falsified_levers.md` (or equivalent):

- [r16-dm] **AMDGPU attribute coercion of VGPR/AGPR allocation** for
  `grouped_rcr_kernel`. Tried `amdgpu_num_vgpr(N∈{128,192})`,
  `amdgpu_waves_per_eu(1,1)`, `__launch_bounds__(_NUM_THREADS, 2)` —
  all silently ignored, byte-identical codegen. Compiler is locked into
  256 VGPR + 67 spills + 0 AGPR by the 8-warp WG arch. To break the
  lock, need source-code refactoring (multi-section accumulator split or
  prefetch-order reorganisation to allow b0/b1 reload). Multi-round
  project, deferred.

- [r16-dm] **Global `chunk_size` reduction in `chiplet_transform_chunked`**.
  chunk=8 → score 814 (−8 vs 822 baseline); chunk=18 → 809 (−13).
  Activating the chiplet swizzle on default-xcds=8 shapes scrambles
  M-axis locality without compensating N-reuse gain. chunk=64 (current,
  effectively NO-OP for xcds=8 and active for xcds=2 DSV3-Down) is the
  per-mix optimum. Per-shape `chunk_size` plumbing (mirror round-67
  num_xcds knob) is the only path forward here, and only worth doing
  if rocprof L2 hit rate becomes the bottleneck — currently isn't.

## What remains plausible (Round-17 candidates)

1. **Source-code VGPR refactoring** (option (b) above — `b0`/`b1` reload
   per section, requires prefetch re-architecture). Multi-round; high
   reward (estimate −67 spills × ~100 cyc/spill / wall_cycles ≈ 1-2 %
   per shape, applies to all 16 shapes including DSV3 + gpt_oss).
   **Risk: medium-high** — touches hot-loop + double-buffer scheme.

2. **Per-shape `chunk_size` plumb** (config knob mirror of `num_xcds`).
   Add `cfg.chunk_size`, `g.chunk_size`, dispatch `g.chunk_size > 0 ?
   g.chunk_size : 64`. Sweep per-shape. **Risk: low** (additive,
   default 64 preserves current behaviour). **Reward: speculative**;
   the round-16 sweep showed chunk!=64 hurts the *aggregate* mix,
   but a per-shape pick might still win small (<1 %) on individual
   shapes.

3. **End-to-end wall-time profiling** (round-15 candidate #3, untouched
   this round). HSA-trace inter-kernel gaps + wave drain analysis to
   localise the 5.77 % wall-time tax not visible in SQ_BUSY. Requires
   `rocprof --hsa-trace --kernel-trace`.

## Score impact

- Metric this round: **822 baseline** (8h ago, before any change).
- Tried chunk=8 → 814; tried chunk=18 → 809; reverted to chunk=64 → **822** (baseline).
- AGPR/VGPR attribute experiments byte-identical → 822 regardless.
- Final score: **822** (no kernel change shipped).
- Within noise band of rolling best 826 (820/825/826/825/821/822 = stdev 2.0).

## Probe artifacts

- `/tmp/hk_build.err` — full kernel-resource-usage report for all
  `grouped_*_kernel` template specialisations (post-revert, baseline
  state).
- `/tmp/metric_round_16.log`, `/tmp/metric_round_16_chunk8.log`,
  `/tmp/metric_round_16_chunk18.log`, `/tmp/metric_round_16_revert.log`
  — per-experiment metric outputs.

## Round-16-dm commits

- Primus-Turbo: this notes file. No kernel/config change.
- HipKittens: none (all probes reverted to round-15 state).
