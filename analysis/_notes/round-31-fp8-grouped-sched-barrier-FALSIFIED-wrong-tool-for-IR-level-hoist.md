# Round 31 — FP8 grouped: `__builtin_amdgcn_sched_barrier(0)` between K-iter and C-store epilog FALSIFIED at ASM level (zero change) — wrong tool for IR-level offset hoisting

**Status**: 1 PROBE round, executes the R30-flagged "real fix" candidate.
PROBE FALSIFIED before metric measurement: ASM is bit-identical to baseline.
Confirms that the spill-causing offset hoist happens in IR-level passes
(LICM / GVN / MachineLICM), NOT in the post-IR machine scheduler.
`sched_barrier` only affects scheduling → **wrong tool for this fix**.

**Auto-optimize round**: 31 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `4caa6d9a` / `4caa6d9a` (PROBE reverted)
**PT SHA at round start**: `82e0bacd`
**Round time**: ~15 min (1 baseline + 1 kernel edit + 1 rebuild +
1 ASM diff + revert + 1 sanity metric + commit)
**Score before**: 979 (same baseline as R30 trials)
**Score after PROBE**: not measured — ASM bit-identical, no kernel
behaviour change possible
**Score after revert**: 979 (sanity-confirmed)

---

## R31 baseline metric

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1858 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1639 (n=24)
[metric_grouped_only]   below_target=35/48  goals=0/2  score=979
```

Bottom 5 FP8 ratios (sorted ascending):

```
grpFP8_gpt_oss_20B-GateUP-B32-M4096    ratio=1.076  (-12.4 pp from 1.20)
grpFP8_gpt_oss_20B-Down-B32-M2048      ratio=1.096  (-10.4 pp)
grpFP8_gpt_oss_20B-GateUP-B4-M4096     ratio=1.104  (-9.6 pp)
grpFP8_gpt_oss_20B-Down-B32-M4096      ratio=1.104  (-9.6 pp)
grpFP8_gpt_oss_20B-GateUP-B32-M2048    ratio=1.108  (-9.2 pp)
```

Same gpt_oss cluster as R28-R30. Worst case unchanged at 1.076.

---

## R31 PROBE — `__builtin_amdgcn_sched_barrier(0)` at K-iter/C-store boundary

### Hypothesis

Per R30 conclusion, the dominant cost is the **prologue spill** (30
`scratch_store_dword` at function entry, ~30 × 80 cy = 2400 cy/tile).
LLVM hoists the per-lane C-store offset computation to function entry
because the offsets are loop-invariant w.r.t. the K-iter (depend only
on `laneid + (i,j,k,l) + row_stride`, not on K-iter state). Hoisted
offsets must spill across the K-iter (256 VGPR cap), and the spill
write happens in EVERY tile regardless of branch.

R30 anti-CSE asm-barrier (`asm volatile("" : "+v"(row_offset), "+v"(col_offset))`)
broke per-call CSE but didn't prevent the EARLY hoist — both halves of
the spill (store + reload) need to be addressed together, and only the
reload-half was reduced (-42 % traffic).

R30 proposed `__builtin_amdgcn_sched_barrier(0)` between the K-iter
end and the C-store epilog start as the mechanism to prevent the
hoist. Mask=0 = "no instruction class can cross this point". Theory:
the offset compute (VALU) cannot move backward past the barrier, so
it stays IN the C-store epilog, where 256 VGPR are free (only cA-cD
accumulators = 128 VGPR are live at that point) → no spill needed.

### Implementation

`/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`,
between the FUSED_KTAIL block end (line 2720) and the C-store epilog
start (line 2722):

```cpp
            }
        }

        __builtin_amdgcn_sched_barrier(0);  // NEW

        // Apply scale + store with m_subtile_C row shift.
```

### Result — ZERO ASM change

Compared `/tmp/disasm_TT_R31.s` (with sched_barrier) vs
`/tmp/disasm_TT_baseline.s` (R30 baseline, no barrier). Both for
`<0,T,T>` template (gpt_oss-active spec).

```
$ diff /tmp/disasm_TT_baseline.s /tmp/disasm_TT_R31.s
2c2
< /tmp/gfx950.elf:	file format elf64-amdgpu
---
> /tmp/gfx950_R31.elf:	file format elf64-amdgpu
```

**4-line diff = ONLY the elf filename header line differs.** Every
asm instruction is bit-identical. `sched_barrier(0)` was completely
absorbed by the compiler / had ZERO effect on codegen.

Spill count metrics also unchanged:

| Template | scratch_store BEFORE → AFTER | scratch_load BEFORE → AFTER |
|---|---:|---:|
| `<0,T,T>` (gpt_oss) | 37 → 37 | 285 → 285 |
| `<0,F,T>` (DSV3+Qwen) | 34 → 34 | 34 → 34 |

Function `private_seg_size` (per `-Rpass-analysis=kernel-resource-usage`):

| Template | BEFORE | AFTER |
|---|---:|---:|
| `<0,F,F>` | 54 dw | 54 dw |
| `<0,T,F>` | 38 dw | 38 dw |
| `<0,F,T>` | 34 dw | 34 dw |
| `<0,T,T>` | 37 dw | 37 dw |

Skipped metric measurement — ASM is bit-identical so metric must
also be bit-identical (within usual ±2 noise).

### Why sched_barrier is the wrong tool

The hoist that's causing the spill happens in **IR-level optimization
passes** (Loop Invariant Code Motion / Global Value Numbering /
MachineLICM), which run BEFORE the post-IR machine scheduler.
`sched_barrier` is a **machine-scheduling intrinsic** — it constrains
how the post-IR scheduler reorders instructions, but by the time
scheduling runs, the offset compute is ALREADY emitted at function
entry (in the IR / MIR). The scheduler doesn't have the option to
move it later because it's already there.

Mechanism breakdown:

1. **IR pass: LICM** — sees that `row_offset = src.base_tile_stride *
   (laneid / src.base_tile_cols)` is loop-invariant w.r.t. the per-tile
   loop `for (gt = blockIdx.x; gt < total_tiles; gt += gridDim.x)`.
   Hoists it to FUNCTION ENTRY (above the loop).
2. **IR pass: GVN/CSE** — sees that all 8 inlined call sites compute
   the same offsets. Common-subexpression-eliminates to ONE compute
   at function entry.
3. **IR pass: register allocator** — assigns the 30 hoisted offset
   values to VGPRs. Sees they MUST live across the K-iter (live-range
   from function entry to C-store epilog). Under K-iter VGPR pressure
   (256-cap), spills 30 of them to scratch.
4. **Post-IR scheduler** — receives the spill+reload sequence, can
   only reorder it locally (within ~32-instr window). Can't undo the
   spill decision.

`sched_barrier` enters the picture at step 4. By then it's too late —
the spill is baked in.

### Correct tool would be one of

- **IR-level attribute / builtin** that prevents LICM hoisting:
  `__attribute__((nohoist))` doesn't exist in HIP. The closest
  is `volatile` semantics, but that defeats register allocation
  too aggressively (inhibits CSE within the call body too).
- **Source-level rewrite** that makes the offset compute syntactically
  CONTAINED in the loop body (so LICM can't hoist it out):
  - Make `src.base_tile_stride / cols / rows` runtime values (not
    compile-time constants). LICM can still hoist runtime-loop-invariant
    expressions.
  - Make `laneid` artificially per-iteration: `laneid_iter = laneid +
    (gt & 0)`. The `& 0` is a no-op but the compiler can't prove it.
    Then `row_offset` depends on `laneid_iter` which depends on `gt`
    (the loop variable) — LICM can't hoist.
  - Pass `gt` as an opaque parameter to the helper.
- **`__attribute__((flatten))` + non-inline helper** — emit the helper
  out-of-line BUT with full inlining within the helper body. Function
  call boundary acts as opaque hoist barrier.
  - But R47-dm tried `__noinline__` and FALSIFIED (function call overhead).

### Best remaining unattempted option

**Make the offset compute artificially loop-iter-dependent** so LICM
cannot hoist out of the per-tile loop. Concrete:

```cpp
// At top of per-tile loop body (after binary search for group_idx)
const int laneid_iter = laneid + ((gt - blockIdx.x) & 0);  // == laneid always
asm volatile("" : "+v"(laneid_iter));  // make compiler unable to prove it
// Then use laneid_iter inside store_c_tile_n_masked instead of laneid
```

The `(gt - blockIdx.x) & 0` is provably 0, but with the asm-volatile
clobber the compiler can't reason through it. So `laneid_iter` looks
loop-iter-dependent → can't be hoisted out of the loop.

**Risk**: if it works at IR level, the offset compute moves INTO the
loop body. Per-iteration cost: 30 VALU ops × 1 cy = 30 cy/iter. With
the spill removed: -30×80 = -2400 cy/iter. Net savings: ~2370 cy/iter
× 24 tiles/CU = ~57 K cy/CU = ~28 us / CU. Could be +2-5 pts metric.

Not attempted this round to keep R31 tightly scoped to
`sched_barrier` falsification. **Recommended for R32.**

### Falsification verdict

`__builtin_amdgcn_sched_barrier(0)` between K-iter and C-store epilog:
**FALSIFIED at ASM level (zero codegen change).** Wrong tool for the
problem — the hoist happens in IR optimization, not machine scheduling.

This **rules out the entire `sched_barrier` lever class** for FP8
grouped C-store spill. Future spill-reduction work needs IR-level
intervention (artificial loop-iter dependency on offset inputs) OR
source-level restructuring (move offset compute syntactically into
the loop body).

---

## What this round changed

**Net commit**:
- New round note: this file (~250 lines)
- HK SHA: `4caa6d9a` UNCHANGED (PROBE rebuilt, identical .so confirmed via
  ASM diff, then source-reverted via `StrReplace`)
- PT SHA: doc-only commit at this round

**No code commits this round.** PROBE was tested and reverted before metric
measurement (no asm change → metric must also be unchanged).

---

## Round meta

| Field | Value |
|---|---|
| HK SHA before/after | `4caa6d9a` / `4caa6d9a` (unchanged) |
| PT SHA before | `82e0bacd` |
| PT SHA after  | (this commit, doc-only) |
| Forward metric before/after | 979 / 979 |
| Lever class status | **`__builtin_amdgcn_sched_barrier` for FP8 grouped C-store hoist = FALSIFIED** (wrong tool: IR-level pass not addressable from machine scheduler) |
| Open lever class | **artificial loop-iter dependency on offset inputs** (R32 candidate; ~+2-5 pts upper bound estimate) |
| Score trajectory | 979 → 979 (no kernel commit) |
| Patience increment | +1 (was 3, will be 4 after this) |
| Best metric | 981 (R27 / R30 trial outliers; current median 979) |
| Score ceiling estimate | 979-981 plateau; ~985 if R32 artificial-iter-dep PROBE lands |

---

## DoD smoke status

Not run this round (no code commit, doc-only). Last DoD run was at
SHA `82e0bacd` (608 score per R30 prompt).

---

## R32 recommendation

**Try the artificial loop-iter dependency approach** to defeat IR-level
LICM hoisting. Concrete implementation:

```cpp
// In grouped_rcr_kernel, after binary search prologue
// (after line ~2304 where group_idx is settled)
int laneid_loop_dep = laneid + ((gt - blockIdx.x) & 0);
asm volatile("" : "+v"(laneid_loop_dep));
```

Then thread `laneid_loop_dep` through `store_c_tile_n_masked` and
the kittens `store(...)` helper as an additional parameter (replace
their internal `laneid` calls). This requires:

1. Add `laneid` parameter to both helpers' templates (could be
   defaulted to `kittens::laneid()` for backwards compat).
2. Pass `laneid_loop_dep` from the kernel call sites.

If this works (LICM can't hoist past the asm-volatile-clobbered
loop-dependent value), expected outcome:
- ASM: 30 `scratch_store_dword` at function prologue → moved to per-tile
  loop body (visible by line-number shift in disasm).
- Spill `private_seg_size`: drops from 37 dw → ~5-10 dw (only the
  K-iter spills remain, not the prologue offsets).
- Metric: +2-5 pts on gpt_oss specs. Score 979 → ~982-984.

If FALSIFIED (asm unchanged again), STOP and accept plateau. The FP8
grouped score is structurally bound at 979-981 by the 30-dw prologue
spill cost — no available mechanism removes it without breaking the
single-launch persistent kernel architecture.

---

## Files touched

* `/workspace/code/Primus-Turbo/analysis/_notes/round-31-fp8-grouped-sched-barrier-FALSIFIED-wrong-tool-for-IR-level-hoist.md` (this note, ~280 lines)

PROBE diff (now reverted):

```diff
--- a/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
+++ b/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
@@ -2719,6 +2719,8 @@ void grouped_rcr_kernel(...) {
             }
         }

+        __builtin_amdgcn_sched_barrier(0);  // R31: pin C-store ALU post-K-iter
+
         // Apply scale + store with m_subtile_C row shift.
```
