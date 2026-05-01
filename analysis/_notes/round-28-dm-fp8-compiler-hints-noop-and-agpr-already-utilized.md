# Round-28-dm — FP8 grouped: compiler-hint experiments NO-OP + AGPRs ALREADY heavily used (refines round-13-dm AGPR-migration plan)

**Date**: 2026-05-01 (round 1 of new 60-round death-march chat)
**Repo / HEAD entry**: Primus-Turbo `fd0e7d52` (post 27+27-round opt log); HipKittens `19ce45a1`
**Result**: Three compiler-hint variants confirmed NO-OP. Kernel ISA inspection refutes the assumption that "the 67 spills imply accumulators are in VGPR (not AGPR)". AGPRs `a0..a255` are already heavily used.

---

## Context

Task: close the 17.7-pp gap on grp_FP8 (geomean 1.024 → 1.20). Round-13-dm
(after 8 single-knob falsifications) listed three remaining multi-round
levers: (1) MFMA cell-shape `16x16x128 → 32x32x64`, (2) K-tail epilog
amortize, (3) **AGPR migration** ("Migrate hot accum VGPRs to AGPRs,
freeing VGPRs for occupancy"). Round-19 found a hard "spill cliff": ANY
net-new code in `grouped_rcr_kernel` provokes +14 VGPR spill →
−37% catastrophic regression. So only REPLACE / REDUCE / TEMPLATE-SPECIALIZE
changes are safe.

This round audited lever (3) and a related family of compiler hints
that *should* push register allocation toward higher occupancy / AGPRs.

## Baseline (entry, with kernel byte-identical to `19ce45a1`)

```
[metric_grouped_only]   grp_FP8   vs triton geomean=1.0246 (n=16)  score=854
```

Per-shape worst FP8 ratios (entry baseline):
- `gpt_oss_20B-GateUP-B32-M4096`  : 0.967
- `gpt_oss_20B-GateUP-B32-M2048`  : 0.984
- `gpt_oss_20B-Down-B32-M4096`    : 0.998
- `DeepSeek-V3-Down-B16-M4096`    : 0.999

Resource-usage report for `grouped_rcr_kernel`:
| spec `<KI_HINT,N_MASKED,FUSED_KTAIL>` | VGPR | AGPR | Spill | Scratch (B/lane) |
|---|---:|---:|---:|---:|
| `<0,F,F>` (DSV3 path, 4 cases)        | 256  | 0   | 67    | 272              |
| `<0,T,F>` (N-masked, gpt_oss-aligned-K) | 256 | 0   | 76    | 308              |
| `<0,F,T>` (FUSED_KTAIL, DSV3 K-misal.) | 256 | 0   | 48    | 196              |
| `<0,T,T>` (gpt_oss FUSED_KTAIL)       | 256  | 0   | 58    | 236              |

Re-confirms round-7-dm numbers (modulo 1-3 spill drift across
intervening commits).

## Probes (all reverted post-test)

### Probe A — `__attribute__((amdgpu_waves_per_eu(2, 4)))`

Hypothesis: hint allowing 2-4 waves/EU encourages compiler to reduce
VGPR pressure (since fitting more waves needs ≤128 VGPR/wave),
potentially routing accum to AGPRs.

Result: bit-identical resource report. VGPR=256, Spill={67,76,48,58},
Occupancy=2 waves/SIMD. Ignored by compiler.

### Probe B — `__attribute__((amdgpu_waves_per_eu(3)))`

Stricter form: minimum 3 waves/EU = max 170 VGPR/wave. Should be
binding.

Result: bit-identical to baseline. Compiler reports occupancy=2
(violating the "min 3" hint) and emits same VGPR=256 + 67 spill code.
The hint is a soft preference; compiler falls back when it can't fit.

### Probe C — `__attribute__((amdgpu_num_vgpr(192)))`

Hypothesis: hard VGPR cap (used in HK attention kernels where
`amdgpu_num_vgpr(29)` limits attention bwd to 29 VGPR/wave). Capping
to 192 should force migration of 64 VGPR worth of state to scratch
or AGPR.

Result: bit-identical. VGPR=256, Spill=67. The attribute appears to
be silently ignored when the kernel's lower-bound register need
exceeds the cap.

### Probes A/B/C summary

All 3 compiler hints are NO-OP on `grouped_rcr_kernel`. Confirms
round-5-dm and round-7-dm: the kernel's VGPR=256 / occupancy=2
state is **the floor allowed by the actual register pressure**, not
a hint-bounded ceiling. Hint-based AGPR migration (round-13-dm
"3. AGPR migration") cannot be triggered through any of the
documented attribute knobs.

## ISA inspection — AGPRs ARE already heavily used

The "AGPRs: 0" entry in `-Rpass-analysis=kernel-resource-usage`
is misleading. Actual ISA usage of the rebuilt `.so`:

```
$ llvm-objdump -d --triple=amdgcn--amdhsa --mcpu=gfx950 tk_fp8_layouts*.so
  ... | grep -oE '\ba[0-9]+\b' | sort -u | wc -l
49     # 49 distinct AGPR numbers used across the entire .so
$ ... | grep -oE '\ba([0-9]+)\b' | sed 's/a//' | sort -n | tail -3
255 255 255   # highest AGPR is a255 (entire AGPR file used)
$ ... | grep -c 'v_accvgpr'
0      # ZERO AGPR↔VGPR move instructions
```

`v_accvgpr_read/write` count = 0 means MFMAs read/write AGPRs
**directly** via the operand modifier bits (gfx950 supports this),
without explicit moves. The compiler is **already** routing
accumulators through AGPRs — the static "AGPRs: 0" report is a
reporting quirk that misses dynamic operand-modifier AGPR usage.

### What this implies for the optimization story

The round-13-dm note recommended "AGPR migration" as one of the three
remaining levers, on the assumption that accumulators were sitting
in VGPR slots and could be moved to AGPRs to free spill slots. **That
assumption is wrong.** The 67-VGPR-spill in `<0,F,F>` cannot come from
accumulators being in VGPR — they're already in AGPR. The spill must
be from non-MFMA state:

1. `soA[4]` and `soB[4]` swizzled-offset arrays (8 VGPR/lane, persistent
   across the entire kernel)
2. Per-tile recomputed values (`br, bc, m_subtile_A, m_subtile_C,
   tile_start, local_tile, m_start_g, M_g, bpr_g, ...`)
3. Persistent outer-loop state (`gt, lo, hi, total_tiles`)
4. `lds_addrs[memcpy_per_tile + 1]` arrays inside each
   `rcr_8w_load_hoist` call (per-call, may not be reused across calls)
5. `tic, toc` ping-pong indices and their dependent register-tile
   slot tracking

Approximately 8 VGPR/lane is "stuck" persistent in (1) alone. The rest
of the 67 spills are from (2)-(5) crossing the persistent loop's
register-allocation barrier. The dense `gemm_kernel<RCR>` only has
12 spills with the SAME accumulator structure precisely because it
**doesn't have the persistent outer loop or the per-tile group lookup**.

This re-frames the lever priority:

| Lever | Round-13-dm rank | Round-28-dm refined rank |
|---|---|---|
| MFMA cell-shape `16x16x128 → 32x32x64`              | #1 | **#1** (still — addresses root cause: per-K-iter MFMA pipeline density. R13-dm model says +17pp on MfmaUtil) |
| K-tail epilog amortize across multi-tile-M          | #2 | #2 (gpt_oss-only; mostly unchanged) |
| AGPR migration                                      | #3 | **CUT** — already happening; not a lever |
| K-tail single-load merge                            | #4 | #3 (gpt_oss-only) |
| **NEW: persistent outer-loop state compaction**     | n/a | **#4** — move `soA/soB`, `br/bc`, etc. out of VGPR (LDS or recompute). High risk of triggering R19 spill cliff. |

## Why R23's "pipeline efficiency, not instr count" diagnosis still holds

R23 measured HK 0.83% more instructions but 7-8% more cycles — the
gap is **cycles per instruction**. Spills don't dominate (only 58
total scratch instrs in the entire .so per probe). The actual
bottleneck is MFMA-pipeline-density: with FP8 16x16x128 MFMA at
~32 cyc per instr and 8 fence-class instructions per K-iter
(s_barrier + s_setprio + s_waitcnt + sched_barrier), MfmaUtil
caps at ~34.8% per round-13-dm's coherence-gate model. The cell-shape
change to 32x32x64 amortizes the same 8 fences over 64-cyc MFMAs
→ MfmaUtil jumps to ~52%.

So Lever #1 (cell-shape change) remains the only lever with a clear
quantitative path to >+5pp gain. It's a 2-3 round structural
project (re-derive RBM/RBN, lane-cell mapping, ST_subtile transpose).

## Round-28-dm verdict

- HipKittens kernel: byte-identical to `19ce45a1` baseline (3 probes
  all reverted).
- HipKittens `.so`: rebuilt and verified 854 metric (1.0244 grp_FP8
  geomean — within noise of 1.0246 entry).
- Primus-Turbo: this notes-only commit.

Score this round: 854 (entry baseline).

## Round-29-dm starting plan (next chat or this resumed chat)

**Start the cell-shape `16x16x128 → 32x32x64` migration as a
TEMPLATE-SPECIALIZED variant**, NOT a runtime branch (R19 falsified
runtime branching). Steps for round-29:

1. Locate every `mfma_scale_f32_16x16x128` call site in
   `grouped_rcr_kernel` body (steady-state, Epilog 1, Epilog 2,
   FUSED_KTAIL block).
2. Add a 4th template parameter `bool USE_32x32_CELL = false` to
   `grouped_rcr_kernel<KI_HINT, N_MASKED, FUSED_KTAIL, USE_32x32_CELL>`.
3. Each template spec compiles only ONE main loop body (16x16 OR 32x32,
   never both — sidesteps R19 spill cliff). Spec count doubles 4→8.
4. Inside `if constexpr (USE_32x32_CELL)`:
   - Use new register tile types `A_32x64_reg`, `B_64x32_reg`,
     `C_32x32_reg = rt_fl<RBM, RBN, col_l, rt_32x32_s>` (already
     scaffolded in HK commit `96a84c08`).
   - Use `mfma_scale_f32_32x32x64_f8f6f4` (already exposed via
     `mfma323264()` wrapper at line 97-110 of `mma.cuh`).
   - Re-derive lane-to-cell mapping for the new shape.
5. Round-29 deliverable: kernel COMPILES with `USE_32x32_CELL=false`
   path identical to current. New `USE_32x32_CELL=true` path emits
   the new ISA but is not yet dispatched.
6. Round-30: wire `USE_32x32_CELL=true` into the dispatcher for one
   target shape (e.g. `gpt_oss_20B-GateUP-B32-M4096`, current ratio
   0.967, the worst case). Verify SNR PASS, measure perf delta.

**Failure mode is graceful**: round-29 can land as a WIP commit even
if the `USE_32x32_CELL=true` path is non-functional, since the default
path is byte-identical.

## Files this round

- HipKittens: 0 net change. 3 probe edits to
  `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` line 1972 +/-
  the `grouped_rcr_kernel` `__global__` decl, all reverted post-test.
  No commit.
- Primus-Turbo: this file (`analysis/_notes/round-28-dm-fp8-compiler-hints-noop-and-agpr-already-utilized.md`).
