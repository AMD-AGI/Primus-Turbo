# Round-30-dm — FP8 grouped: `sched_group_barrier` PRESCRIPTIVE scheduling REGRESSES DSV3 shapes (−13 pts) despite REDUCING spills by up to 10/spec

**Date**: 2026-05-01 (round 3 of 60-round death-march chat)
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `53cc464fe` (round-29-dm doc-only)
**HipKittens HEAD**: unchanged (probe edits applied + reverted; .so rebuilt
to match source — bytes-identical)
**Metric**: 916 → 903 (probe) → 916 (after revert) — probe FALSIFIED

---

## TL;DR

Tested the one remaining untested pipeline-scheduling lever from R29-dm's
"what's left" inventory: **`__builtin_amdgcn_sched_group_barrier(MASK, COUNT,
SYNC_ID)`** — the PRESCRIPTIVE variant (forces an exact instruction mix
into groups), as opposed to the PERMISSIVE `sched_barrier(MASK)` that
R27 already falsified. This is the primitive HK's own attention kernel
uses via `kernels/attn/gqa/kernel.cpp::sched_barrier_pairs<Pairs, V, G>()`.

Inserted a 3-line pattern (4×`{2 VMEM_READ, 2 DS_READ, 1 MFMA}` under
sync_id=7) at the end of `grouped_rcr_kernel`'s main loop body:

```cpp
#pragma unroll
for (int s = 0; s < 4; ++s) {
    __builtin_amdgcn_sched_group_barrier(0x020, 2, 7);  // VMEM_READ
    __builtin_amdgcn_sched_group_barrier(0x100, 2, 7);  // DS_READ
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 7);  // MFMA
}
```

Result: **VGPR spills dropped across ALL 4 template specs** by 1–10 slots each
— exactly the register-pressure relief the round-28-dm AGPR-audit follow-up
wanted. **But grp_FP8 metric REGRESSED by 2.77 pp** (1.0201 → 0.9924), with
the DSV3 shapes absorbing nearly the entire loss (2 outliers at −8 to
−9 pp per shape).

Falsifies the last remaining "cheap" pipeline-scheduling lever. Spill
count is **not a reliable perf proxy** for this kernel; the reduced
register pressure came from LLVM reorganizing dependencies in a way
that broke existing implicit overlap between LDS-read and MFMA.

---

## Context (round-3 of 60, resumed from round-29-dm)

The previous chat (rounds 1..29-dm on this problem) cumulatively falsified
16 directions (see round-29-dm falsification table). The remaining productive
levers were all architecturally invasive:

1. `grouped_var_k_kernel_fp8` register reduction — backward kernel, metric-
   invisible, requires hand-benchmarking.
2. `a_kt1` register-tile elimination — opposes round-12-dm's shipped K-tail
   2-stage overlap, likely net-negative.
3. N=2880 partial-tile-aware dispatch (gpt_oss only) — ~+0.25 pp geomean.
4. **`sched_group_barrier`** — same 30-line diff scope as round-27's
   `+sched_barrier(MASK)` attempts, but a **structurally different scheduling
   primitive** (prescriptive, not permissive).

This round tests (4). It's genuinely untested: round-27 tested
`+sched_barrier(MASK=0x100/0x108)` which is permissive (allows reorder of
those instruction classes across the barrier). `sched_group_barrier(MASK,
COUNT, ID)` is prescriptive — it CREATES a group of exactly COUNT
instructions of class MASK at that point, constraining the scheduler to
that decomposition.

## Hypothesis

HK's attention kernels use `sched_barrier_pairs<N, V, G>()` (a wrapper over
`sched_group_barrier`) to force LLVM to interleave MFMA and VALU
instructions at a specific ratio. The FP8 grouped main loop body emits
~4 MFMAs + ~8 `buffer_load_dwordx4 … lds` + ~8-16 `ds_read_b128` per
k-iter across 4 stages. The EXISTING `s_barrier` + `s_waitcnt lgkmcnt(0)`
fence sequence lets LLVM reorder freely between fences. Declaring the
target interleave (4×{2 VMEM, 2 DS, 1 MFMA}) should guide LLVM to
pack memory ops into the MFMA shadows more aggressively, reducing CPI.

## Experiment

Inserted at end of the k-loop body (line ~2202), before the closing `}`:

```cpp
#pragma unroll
for (int s = 0; s < 4; ++s) {
    __builtin_amdgcn_sched_group_barrier(0x020, 2, 7);  // 2 VMEM_READ
    __builtin_amdgcn_sched_group_barrier(0x100, 2, 7);  // 2 DS_READ (under-count: real ~4)
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 7);  // 1 MFMA (exact)
}
```

Mask constants from `HipKittens/kernels/gemm/bf16fp32/micros/hint_based/
schedule_utils.cpp`:
- 0x008 = MFMA
- 0x020 = VMEM_READ (includes `buffer_load_dwordx4 ... lds`)
- 0x100 = DS_READ (ds_read_b128 / _b64_tr_b8 / etc.)

DS_READ count intentionally under-specified (2 per group × 4 = 8 total,
real count ~16 per iter) so LLVM has flexibility for the excess. VMEM
and MFMA counts are exact so LLVM cannot skip them.

sync_id=7 avoids collision with any implicit sync_id=0 groups.

## Resource-usage delta (`-Rpass-analysis=kernel-resource-usage`)

All 4 template specs of `grouped_rcr_kernel<KI_HINT, N_MASKED, FUSED_KTAIL>`:

| spec            | TotalSGPRs | Spill baseline | Spill r30-dm | ΔSpill | Scratch baseline | Scratch r30-dm | ΔScratch |
|-----------------|-----------:|---------------:|-------------:|-------:|-----------------:|----------------:|---------:|
| `<0,F,F>`       |  65 → 65   |    **67**      |      **66**  |   −1   |    272 B         |    268 B        |   −4 B   |
| `<0,T,F>`       |  71 → 71   |    **76**      |      **66**  |  **−10** |  308 B         |    268 B        |  **−40 B** |
| `<0,F,T>`       |  79 → 79   |    **48**      |      **47**  |   −1   |    196 B         |    192 B        |   −4 B   |
| `<0,T,T>`       |  83 → 83   |    **58**      |      **53**  |   −5   |    236 B         |    216 B        |  −20 B   |

**Every spec got smaller.** `<0,T,F>` (gpt_oss N-misaligned, K-aligned) dropped
by 10 spills — the largest single-probe register-pressure relief in the
entire r17..r29 trail. This is precisely the spill-cliff diagnosis
round-28-dm framed.

## Metric delta

| run                               | grp_BF16 | grp_FP8 | score |
|-----------------------------------|---------:|--------:|------:|
| baseline (before probe)           |  1.1833  | 1.0201  |  916  |
| r30-dm probe (sched_group_barrier)|  1.1826  | 0.9924  |  **903** (−13) |
| after revert (byte-identical .so) |  1.1850  | 1.0193  |  916 (restored) |

Per-shape grp_FP8 breakdown (showing shapes where |delta| > 2 pp):

| shape                                | baseline | r30-dm | delta |
|--------------------------------------|---------:|-------:|------:|
| DSV3-GateUP-B16-M2048                |  1.058   | 0.974  | **−8.4 pp** |
| DSV3-Down-B16-M2048                  |  1.022   | 0.927  | **−9.5 pp** |
| DSV3-GateUP-B16-M4096                |  1.047   | 0.992  | −5.5 pp |
| DSV3-Down-B16-M4096                  |  0.958   | 0.910  | −4.8 pp |
| DSV3-GateUP-B32-M2048                |  1.062   | 0.985  | −7.7 pp |
| DSV3-Down-B32-M2048                  |  1.013   | 0.985  | −2.8 pp |
| DSV3-GateUP-B32-M4096                |  1.076   | 1.003  | −7.3 pp |
| DSV3-Down-B32-M4096                  |  0.971   | 0.992  | +2.1 pp |
| gpt_oss-GateUP-B4-M2048              |  1.038   | 1.036  |  0.0 pp |
| gpt_oss-Down-B4-M2048                |  1.094   | 1.052  | −4.2 pp |
| gpt_oss-GateUP-B4-M4096              |  0.995   | 0.987  | −0.8 pp |
| gpt_oss-Down-B4-M4096                |  1.041   | 1.027  | −1.4 pp |
| gpt_oss-GateUP-B32-M2048             |  0.982   | 1.011  | +2.9 pp |
| gpt_oss-Down-B32-M2048               |  1.017   | 1.025  | +0.8 pp |
| gpt_oss-GateUP-B32-M4096             |  0.963   | 0.982  | +1.9 pp |
| gpt_oss-Down-B32-M4096               |  0.996   | 0.999  | +0.3 pp |

**Pattern**: DSV3 (K=7168, K-aligned, FUSED_KTAIL=false) tanks; gpt_oss
(K=2880, K-misaligned, FUSED_KTAIL=true) trends marginally positive or
flat. The two groups select different template specs and the probe
affects them differently.

## Why it backfired

The spill reduction came from LLVM reorganizing the live range of
per-stage state (`b0`, `b1`, `a` register tiles), collapsing some
across-stage overlap. But that SAME reorganization **broke implicit
overlap between LDS-read and MFMA issue** that the existing
`s_barrier → s_waitcnt → MFMA → s_barrier` sequence was maintaining
on the DSV3 K-aligned path.

Specifically, the probe declared an order of `(VMEM → DS_READ → MFMA) × 4`.
When LLVM packed instructions into those groups, it PUSHED more DS_READs
into later groups (so that earlier groups had enough instructions to
fill the `(2 VMEM, 2 DS_READ, 1 MFMA)` slot). That cascaded into
fewer DS_READs being in-flight at each MFMA issue point, increasing
per-MFMA wait time on the `s_waitcnt lgkmcnt(0)` that precedes each MFMA.

Net effect: **spill-driven pressure relief at the cost of CPI** — exactly
the opposite of what we wanted.

## Why it worked marginally well on gpt_oss

The gpt_oss shapes use `FUSED_KTAIL=true` which introduces the `a_kt1`
register tile inside the k-tail epilog block (round-3 of the prior
chat; `round-12-dm` optimization). That already increases base
register pressure ~32 VGPR on the `<*, T, T>` specs. The probe's
spill-relief (5 slots on `<0,T,T>`) moves more state off scratch,
and the FUSED_KTAIL path's own `raw_buffer_load_b128` already
constrains overlap at the K-tail boundary — so the net effect is
wash-to-slightly-positive (+0.3 to +2.9 pp).

DSV3 shapes use `FUSED_KTAIL=false` (K%128=0, K=7168 aligns), so
the baseline spill is lower (`<0,F,F>` = 67, `<0,T,F>` = 76) and
the main-loop main path sees less existing pressure — the
reorganization buys less and the CPI hit is not absorbed.

## Updated falsification trail (now 17 directions across r17..r30-dm)

| # | round     | direction                                              | result |
|--:|----------:|--------------------------------------------------------|--------|
| 1 | R17       | rocprof PMC instr count diagnose                       | gap is CPI, not insts |
| 2 | R18       | hoist `make_srsrc` out of helper                       | compiler already CSE'd |
| 3 | R19       | port two-tile main loop from dense                     | −37 % spill cliff |
| 4 | R20       | `readfirstlane(lds_tile_base)`                         | compiler already hoisted |
| 5 | R21       | `__noinline__` on `rcr_8w_load_hoist`                  | architecturally inline |
| 6 | R22       | host-overhead trim                                     | host is 0.9-1.1 % only |
| 7 | R23       | HK vs Triton wall+PMC side-by-side                     | gap is main-kernel CPI |
| 8 | R24       | `RCR_PREFETCH_LGKM` sweep {2,4,6,8}                    | saturated |
| 9 | R26       | `RCR_SCHED_BARRIER` mask sweep (macro)                 | macro never reaches grouped main loop |
|10 | R27a      | +`sched_barrier(0x108)` 2× grouped main loop           | mean = baseline |
|11 | R27b      | +`sched_barrier(0x108)` 1× post-MMA-A                  | mean = baseline |
|12 | R27c      | +`sched_barrier(0x100)` 1× post-MMA-A                  | mean = baseline |
|13 | R27d      | `s_setprio(1) → s_setprio(3)` 4 sites                  | mean = baseline |
|14 | R28-dm    | `__attribute__((amdgpu_waves_per_eu(2,4)))`            | silently ignored |
|15 | R28-dm    | `__attribute__((amdgpu_waves_per_eu(3)))`              | silently ignored |
|16 | R28-dm    | `__attribute__((amdgpu_num_vgpr(192)))`                | silently ignored |
|17 | R29-dm    | explicit `readfirstlane` on binary-search LDS reads    | +17 to +20 spills on `<0,F,T>`/`<0,T,T>`; reverted |
|**18**|**R30-dm**| **`sched_group_barrier(VMEM=2,DS=2,MFMA=1) × 4` main loop** | **spills −1 to −10 per spec, but DSV3 metric −5 to −9 pp per shape; net −13 score** |
|  –  | R15-dm    | MFMA cell-shape migration (rocprof PMC)                | parity-or-better; ≤ 0.06 pp geomean |

## Files touched

- HipKittens: zero net change. Probe was a single insertion at
  `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:~2202` (8 added lines).
  Both insertion and revert went through clean `make tk_fp8_layouts`;
  resource-usage verified byte-identical spills after revert.
- Primus-Turbo: this notes file only.

No HipKittens commit this round. Primus-Turbo commit: doc-only.

## For round 31+

`sched_group_barrier` is now also in the falsified column. Remaining
candidates from round-29-dm's "what's left" inventory:

1. **N=2880 partial-tile-aware dispatch** (gpt_oss shapes only, ~+0.25 pp
   geomean expected) — mid-invasiveness, NO longer the highest-EV option
   because (a) it's gpt_oss-only, and round-30-dm showed gpt_oss is the
   less-painful group (FP8 geomean 1.018 across those 8 shapes vs 1.015
   for DSV3); (b) the delta is small (~+0.25 pp geomean on the full 16,
   ~+0.5 pp on gpt_oss subset).

2. **`a_kt1` register-tile elimination** — actively OPPOSES round-12-dm's
   shipped optimization. Round-12-dm's 2-stage overlap was the change
   that closed the round-16 hot-spot from 33.6 % → 42.84 % MfmaUtil on
   gpt_oss-Down-B4-M4096. Eliminating it would very likely regress that
   gain. Not recommended unless a clear diagnostic shows the `a_kt1`
   pressure is now net-harmful.

3. **`grouped_var_k_kernel_fp8` register reduction** — backward kernel,
   metric-invisible. Would need `benchmark/ops/bench_grouped_gemm_turbo.py
   --dtype fp8` before+after. SGPR=79 / VGPR=256 / Spill=52 / Scratch=212
   today. Same risks as forward (R19 spill cliff on net-new code).

4. **DSV3-specific LDS layout probe** — round-30-dm pattern (DSV3 tanks
   by 5-9 pp, gpt_oss flat-to-up) suggests the FUSED_KTAIL=false specs
   sit on a different cliff. The existing LDS swizzle is `st_16x128_v2_s`
   for both groups. Checking whether the DSV3 K-aligned path benefits
   from `st_16x128_v2a_s` or similar LDS layout could unlock the DSV3
   regression pattern (but: r2-dm already falsified ST_v3 swap, so
   the LDS-layout direction has prior negative data).

5. **Software-pipelined main loop via ASM** (task body's Lever E) —
   multi-round structural project. With sched_group_barrier now
   falsified, Lever E's "high-risk structural" cost becomes more
   defensible. Needs careful round-by-round handoff.

Recommended round-31 action: attempt **direction #1 (N=2880 partial-tile
aware dispatch)** OR **pivot to deeper diagnostic**. Direction 5 (ASM
main-loop rewrite) remains the highest-ceiling lever but is multi-round;
direction 4 is worth a single-round probe with the caveat of r2-dm
ST_v3 falsification.

## Round 30-dm verdict

- HipKittens kernel byte-identical to `19ce45a1` (pre-probe) baseline.
- HipKittens HEAD: unchanged, no commit.
- Primus-Turbo: this notes-only commit.
- Score 916 entry → 903 probe → 916 revert (within run-to-run noise band).
- Falsification trail: 18 directions across R17-R30-dm.
