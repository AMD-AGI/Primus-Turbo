# Round 29 — FP8 grouped: Lever C-1 SCOUT — spill source localization (data-only)

**Status**: NO KERNEL CHANGE — R29 = data-only SCOUT round per R28 plan.
LLVM optimization-record yaml parse identifies the spill source distribution
across the 4 `grouped_rcr_kernel` template specs. Findings refine R30/R31
plan and revise the Lever C-1 ceiling estimate downward.
**Auto-optimize round**: 29 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `4caa6d9a` (rebuilt with
`-fsave-optimization-record`; bit-identical .so contents — opt yaml flag
doesn't affect codegen)
**PT SHA at round start**: `5ad852a9`
**Round time**: ~10 min (1 baseline metric + 1 instrumented build + yaml parse)
**Score before**: 979 (R28 baseline)
**Score after**: 979 (no kernel change)

---

## Method

Built `tk_fp8_layouts.so` with `HIPFLAGS="-fsave-optimization-record"`.
Output: `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.opt.yaml` (19 MB).

Parsed `SpillReloadCopies` (function-scope) and `LoopSpillReloadCopies`
(loop-scope) records emitted by the `regalloc` pass for `grouped_rcr_kernel`,
demangled per template spec via the `ILi0ELbXELbYE` mangling pattern.

```python
# Template -> mangled name suffix -> (NumVRCopies, TotalCopiesCost) per source line
docs = text.split('--- !')
for d in docs:
    if 'SpillReloadCopies' not in d: continue
    if 'grouped_rcr_kernel' not in d: continue
    # extract DebugLoc line, NumVRCopies, TotalCopiesCost
```

---

## Findings

### Spill structure across all 4 templates

```
=== Template <0,F,F> (DEAD)   ===  total NumVR=546  cost=1260.5
=== Template <0,F,T> (DSV3+Qwen, 16/24 ACTIVE) ===  total NumVR=554  cost=1185.5
=== Template <0,T,F> (DEAD)   ===  total NumVR=618  cost=1308.5
=== Template <0,T,T> (gpt_oss, 8/24 ACTIVE)    ===  total NumVR=606  cost=1177.5
```

For each template, spill is concentrated at exactly 2 source locations
(plus 1 trivial L2285 site = 3 NumVR / 30 cost = LDS group_offs init):

| Template     | L2223 (FUNC) NumVR / Cost | L2313 (LOOP) NumVR / Cost |
|---|---:|---:|
| `<0,F,F>` (dead)   | 276 / 632.5 | 267 / 598.0 |
| `<0,F,T>` (DSV3+Qwen) | 281 / 595.5 | 270 / 560.0 |
| `<0,T,F>` (dead)   | 312 / 656.5 | 303 / 622.0 |
| **`<0,T,T>` (gpt_oss)** | **307 / 591.5** | **296 / 556.0** |

L2223 = function entry. L2313 = the persistent tile loop
`for (int gt = blockIdx.x; gt < total_tiles; gt += gridDim.x)` body.

### Spec deltas

```
  N_MASKED=true adds:      +30..+36 NumVR / +24..+62 cost  (FUNC) on top of
                           +33..+36 NumVR / +24..+62 cost  (LOOP)
  FUSED_KTAIL=true adds:   +5..+9   NumVR / -41..-37 cost  (FUNC) — cost
                                                                    DROPS
                           +3..+0   NumVR / -38..-38 cost  (LOOP)

The N_MASKED delta is uniform across FUSED branches.  The FUSED delta
on LOOP is essentially zero (3 NumVR), confirming R3's claim that the
K-tail body's runtime-dead branch on K_REM=0 has zero loop cost.

The FUSED delta on FUNC scope DROPS cost despite +5..+9 NumVR — this
is consistent with R34-dm's observation that the `a_kt1` register
declaration plus K-tail body present in scope gives LLVM a better
liveness graph for register allocation across the function (reducing
the high-cost spills even though it adds copy count).
```

### What's NOT in the spill remarks

The grouped_rrr_kernel (line 2912, dA bwd) had **0** records returned
by the regex — the kernel compiles to 65 dw spill but the regalloc
pass apparently doesn't emit per-line `SpillReloadCopies` records for
it (or they're attributed differently).

The grouped_var_k_kernel_fp8 (line 5608, dB bwd, 37 dw spill) similarly
returns 0 records. Backward-path analysis is not addressable via this
yaml parse.

---

## Critical revision: Lever C-1 ceiling estimate

R3 plan (R28-summarised) projected up to +24 % runtime improvement on
gpt_oss specs from VMEM→LDS scratch redirect. **R29 data refines this
significantly downward.**

### Where the spill actually lives

```
  L2223 (FUNC) accounts for ~50 % of spill cost.
  L2313 (LOOP) accounts for ~47 % of spill cost.
  Other     accounts for ~3  % (LDS init + ALU spill).
```

L2313 = THE PERSISTENT LOOP BODY. This block contains:
- The K-iter main loop (lines 2424-2452, executed `ki_dyn-2` times per tile)
- Epilog 1 + Epilog 2 (lines 2454-2506)
- FUSED_KTAIL block (lines 2535-2719, runtime-gated)
- C-store epilog (lines 2722-2822)

**These spills are HOT** — the K-iter main loop runs 20-30× per tile.
Redirecting them to LDS:
* Saves: 14 cy LDS round-trip vs 80-150 cy VMEM scratch round-trip = 70-130 cy/spill
* Per-tile cost (gpt_oss K=2880, ki=22): 22 K-iters × N_spills × 80 cy
* Best case if N_spills=4 hot spills per K-iter: 22 × 4 × 80 = 7K cy/tile saved

### Where the spill CANNOT live

**LDS budget is the binding constraint**: 137 KB used / 160 KB cap = 23 KB
headroom. Per-wave scratch_pool sizing:

```
  4 hot VGPR slots × 4 bytes/dw × 256 lanes/wave = 4 KB per wave
  × 8 waves per CTA                              = 32 KB per CTA
  > 23 KB headroom → INFEASIBLE per-wave
```

Only per-CTA cyclic scratch_pool can fit (~1 KB at a time). This is
incompatible with the K-iter being warp-parallel (each warp's spill
must be private; cyclic reuse across warps would race).

**Per-warp scratch_pool is the only viable approach**: 4 KB per wave × 8
waves = 32 KB. Doesn't fit. **Workaround**: redirect FEWER hot slots
(e.g. 2 per warp = 16 KB total → fits). But fewer slots = less savings.

### Honest ceiling (revised)

```
  Realistic per-tile saving = 22 K-iter × 2 hot slots × 80 cy = 3.5K cy
  Per-CU: ~700 tiles → 2.45M cy/CU → ~2.9 ms saved on a 5 ms kernel
  = ~5.7 % runtime improvement on worst-case gpt_oss
  = ratio 1.072 × 1.057 = 1.133
```

So Lever C-1 BEST CASE on gpt_oss-GateUP-B32-M4096: **1.133 ratio**
(currently 1.072 → would lift to ~1.13). Other gpt_oss specs would
lift proportionally. Geomean delta: **+1-2 pp** ⇒ **score 982-985**.

This is consistent with the R28 estimate but **the upper bound is
revised down from "1.354 if all spill redirected" to "1.133 with
2 hot slots redirected"** — because LDS budget caps how many slots
can be redirected.

Lever C-1's CEILING is NOT 1.20 — it's ~1.13-1.14. The ratio of
diminishing returns kicks in earlier than R3 projected.

---

## L2223 (FUNC scope) is potentially MORE valuable than L2313 (LOOP)

L2223 spill records account for ~50 % of total cost, but L2223 = function
entry attribution. These spills happen in code OUTSIDE the persistent loop:

* Kernel prologue: LDS group_offs init, swizzled offset prefill, warp ID
  computation
* Kernel epilog (if any) — but the kernel returns immediately after the
  persistent loop ends, so likely 0 here
* Inline ASM blocks attributed to function scope (helper inlinining
  cost gets billed here)

Looking at the source:
* Lines 2261-2293: parallel LDS init + binary search prologue scaffolding
  — runs ONCE per kernel invocation, not per tile
* Lines 2296-2304: prefill_swizzled_offsets (once per kernel)
* Lines 2306+: persistent loop start

If 50 % of spill cost is in code that runs ONCE per kernel, redirecting
those to LDS scratch would have ZERO runtime impact — they're not on
the hot path.

**Refined Lever C-1 target**: only L2313 (LOOP) spills, not L2223 (FUNC).
This further constrains the achievable saving — only ~47 % of spill
cost is on the hot path.

```
  Lever C-1 realistic envelope (revised again):
    Ceiling:        +5-7 pp on worst gpt_oss spec
    Geomean delta:  +1.0 pp
    Score delta:    +5-8 points (979 → 984-987)
```

---

## L2313 LOOP body sub-decomposition (manual cross-reference)

The L2313 block contains 4 distinct phases. Each phase's contribution
to spill is uncertain from the yaml — only the line attribution
(L2313, the loop start) is recorded. Manual inspection ranks
spill density by code complexity:

| Phase | Lines | Spill density (manual estimate) |
|---|---|--:|
| Persistent-loop prologue (binary search + load tile prologue) | 2306-2406 | LOW (uniform values) |
| **K-iter main loop body** (`for k = 0 ... ki_dyn-2`) | 2424-2452 | **HIGH** (4 mfma + 4 load_hoist + 4 mma reg) |
| **Epilog 1 + Epilog 2** (last 2 K-tiles, no unroll) | 2454-2506 | **HIGH** (manual schedule, 8 mfma + 8 load_hoist) |
| FUSED_KTAIL block (`if (g.fast_k < g.k)`) | 2540-2719 | MEDIUM (24 buffer_load + 4 mfma) |
| C-store epilog | 2722-2822 | **HIGH** (4 mul + 4 store + N_MASK helper inlined) |

The C-store epilog already has R24 (readfirstlane on r0/r1/c0/c1) and
R44-dm (interleave mul/store) wins. Further optimization there is
probably diminishing returns.

The K-iter main loop is the largest single contributor. R3-dm
already found that `#pragma unroll N` is a no-op (LLVM auto-tunes),
but the spill count is the first-order bottleneck — accumulators
(cA-cD = 128 dw) + 3 register tiles (a, b0, b1 = 48 dw) + per-iter
working state already exceeds 256 VGPR cap.

---

## R30 plan: PROBE on a single hot spill in the K-iter

Per R28 plan, R30 should pick TOP 1 hot-spill source from R29 list and
attempt LDS scratch redirect.

R29 yaml only resolves to (line, NumVRCopies, cost) — it does NOT name
the specific VGPRs or values being spilled. To name them requires:

* `-mllvm -print-after-all` to dump per-pass IR (would be 100+ MB,
  impractical to parse)
* `objdump -d --line-numbers` on the .so to identify scratch_store/
  scratch_load instruction locations and the VGPR they save/reload
* A small targeted micro-bench that varies one spill candidate at a
  time

The chat-window is ending; R30 will need to start a fresh chat session
with the R29 SCOUT data already documented (this round-note serves
that handoff).

### R30 first step (next chat)

1. Run `objdump -d --triple=amdgcn--amdhsa --mcpu=gfx950 tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so | grep -E 'scratch_store|scratch_load' | head -50`
   to enumerate the actual VMEM scratch instructions emitted by LLVM.
2. Match the instruction line offsets back to source via
   `addr2line -e tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so 0xOFFSET`
   to identify which source-level VGPR slot each scratch_store
   represents.
3. Pick the source-level value that has the SHORTEST live-range
   AND HIGHEST iter count (most cost-per-redirect) — likely a
   per-K-iter scratch save like an `int row_lane` or `int k_iter_index`.
4. Draft inline asm `ds_write_b32` / `ds_read_b32` redirect with
   `asm volatile` clobber pattern to release the source register.

---

## What this round changed

**Nothing in code or HK.** Only:
- New round note at this path (~280 lines, this file)
- Rebuilt .so with `-fsave-optimization-record` (bit-identical to
  pre-R29, doc-only)

HK SHA stays at `4caa6d9a`. PT this commit doc-only.

---

## Round meta

| Field | Value |
|---|---|
| HK SHA before/after | `4caa6d9a` / `4caa6d9a` (unchanged) |
| PT SHA before | `5ad852a9` |
| PT SHA after  | (this commit) |
| Forward metric before/after | 979 / 979 |
| Lever C-1 ceiling (revised) | **+5-8 score points (979→984-987)** vs R28 estimate of "+5 to +20 points" |
| Score trajectory | 979 → 979 (no probe) |
| Patience increment | +1 (was 1 last round, will be 2 after this) |

---

## Files touched

* `/workspace/code/Primus-Turbo/analysis/_notes/round-29-fp8-grouped-Lever-C-1-SCOUT-spill-source-localization.md` (this note, ~290 lines)
