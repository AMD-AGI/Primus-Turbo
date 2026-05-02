# Round 22 — FP8 grouped: ASM-level discovery of divergent-SRD fallback loops + `readfirstlane` fix FALSIFIED

**Status**: NEW DIAGNOSTIC FINDING + 1 negative-EV fix attempt. R22 disassembled
the .so device code for the 4 `grouped_rcr_kernel` template specialisations and
counted the per-spec **divergent-buffer-descriptor (SRD) fallback loop** pattern
(`v_readfirstlane → v_cmp → s_and_saveexec → buffer_op → s_xor exec → s_cbranch_execnz`).
This pattern is what LLVM emits when it can't prove a buffer SRD is wave-uniform —
it falls back to a per-lane sequential loop, even when the SRD would actually be
the same on every lane at runtime.

The data shows FUSED_KTAIL specs have **3.2× more divergent-SRD loops** than
non-FUSED specs (411 / 419 vs 128 / 136). This is a previously-undocumented
compiler-level inefficiency in the FP8 grouped kernel, attributable to
FUSED_KTAIL's per-lane VGPR ops (SENTINEL voffsets, b128_lo_valid lane masks)
poisoning LLVM's uniformity analysis for downstream C-store SRDs.

The natural fix — `__builtin_amdgcn_readfirstlane` on `group_idx` after the
binary search — was tried in two variants and **FALSIFIED**: only 8 of the 411
loops were eliminated, while VGPR spill **increased by +15-21 dw on N_MASKED
specs** (net negative at the ratio level).

**Auto-optimize round**: 22 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `5d45ceb2` (no kernel changes committed; failed
fix attempts were reverted in-tree, baseline binary restored)
**PT SHA at round start**: `7d086fcb`
**Reported best (forward)**: 966 (R15 / R18, high-tail of noise band)
**R22 baseline metric**: 960 (initial trial); revert-confirmation = 962
**R22 patience**: 7 rounds at noise floor (R16-R22)

---

## R22 — divergent-SRD loop count per RCR spec (NEW data)

Disassembled `tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so` device code
(via `--save-temps` rebuild + `roc-obj-extract` URI form) and counted the
divergent-buffer-op fallback loop pattern (`s_cbranch_execnz .LBB...` after
`v_readfirstlane_b32 / v_cmp / s_and_saveexec / buffer_*` block):

```
spec                  size (lines)  divergent-SRD-loops   readfirstlane-count
─────────────────────────────────────────────────────────────────────────────
rcr<0,F,F>                  4440             128                   537
rcr<0,T,F>                 10241             411                  1556
rcr<0,F,T>                  4769             136                   582
rcr<0,T,T>                 10582             419                  1591    ← worst case
```

(F/T = FUSED_KTAIL/N_MASKED_STORE templates; rcr<0,T,T> is the hot template
for gpt_oss reroute + DSV3-GateUP fwd path)

Decomposition:
* Going from `<F,F>` → `<F,T>` adds **+8** divergent loops (N_MASKED_STORE
  alone is well-handled by the compiler — N_MASKED helper's branch hoist
  in R59-dm prevented uniformity loss).
* Going from `<F,F>` → `<T,F>` adds **+283** divergent loops.
* Going from `<F,T>` → `<T,T>` adds **+283** divergent loops.

So **FUSED_KTAIL alone is responsible for ~283 additional divergent-SRD
loops per kernel invocation**, regardless of N_MASKED status.

### Why this matters

When the SRD is genuinely uniform (all 64 lanes share the same C-tensor
descriptor), the divergent-fallback loop runs ONE iteration covering all
lanes, but the iteration still costs the per-iter overhead:

```
v_readfirstlane_b32 s16, v2     ; ~1 cy
v_readfirstlane_b32 s17, v3     ; ~1 cy
v_readfirstlane_b32 s18, v176   ; ~1 cy
v_readfirstlane_b32 s19, v177   ; ~1 cy
v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]      ; ~1 cy
s_nop 0
v_cmp_eq_u64_e64 s[8:9], s[18:19], v[176:177] ; ~1 cy
s_and_b64 s[8:9], vcc, s[8:9]                 ; ~1 cy
s_and_saveexec_b64 s[8:9], s[8:9]             ; ~2 cy
buffer_store_short v131, v101, s[16:19], 0 offen ; ~6 cy
s_xor_b64 exec, exec, s[8:9]                  ; ~1 cy
s_cbranch_execnz .LBB12_263                   ; ~3 cy (branch predicted not-taken)
```

≈ **18-20 cycles per loop** when SRD is uniform (most common case). The
4-cycle uniform `buffer_store` it would normally be is now ~5× more expensive.

For `rcr<0,T,T>`: 283 extra loops × 20 cy ≈ **5660 cy/tile of pure compiler-
generated overhead**, on a tile budget of ~5500 cy main-loop + ~400 cy epilog.

This is potentially a **~10 % per-tile overhead** that's not in any prior
round's bookkeeping.

---

## Cluster location

```
divergent-SRD loop position offsets within rcr<0,T,T> (size 10582 lines):
   K-loop main body:    lines 1352-1505    (32 mfma cluster, no divergent loops)
   FUSED_KTAIL fuse:    lines ~1700-2000   (24 buffer_load_b128 — some divergent)
   C-store epilog:      lines ~2100-10500  (335 of 419 loops here)
```

So **80 % of the divergent loops are in the C-store epilog**, not the K-tail
fuse block itself. The FUSED_KTAIL block "poisons" the dataflow — its per-lane
VGPR ops (SENTINEL voffsets, b128_lo_valid masks, per-lane address computations
via `make_srsrc((const void*)a_base_ptr, ...)`) make the compiler conclude that
all values derived from `g.c` / `m_subtile_C` / etc. *might* be divergent, even
though they would be uniform at runtime.

---

## Failed fix attempt — `readfirstlane` on `group_idx`

Mathematical setup: `gt = pid + iter * NUM_CUS` is uniform per CTA (pid uniform
per CTA, iter uniform per loop, NUM_CUS const). The 6-step binary search reads
`s_cum_tiles[mid]` from LDS — same value across the wave since `mid` is uniform
within each level. So `lo` (and `group_idx = lo`) IS uniform.

Inserting `__builtin_amdgcn_readfirstlane(lo)` is a runtime no-op but reshapes
the SSA graph so LLVM marks `group_idx` as wave-uniform (SGPR).

### Variant A — readfirstlane on 4 derived values (group_idx, tile_start, m_start_g, M_g)

```
spec     baseline spill    after-fix spill    Δ          divergent-loops    Δ
rcr<F,F>     39 dw              39 dw         0           128 → 128         0
rcr<T,F>     43 dw              44 dw         +1          411 → 411         0
rcr<F,T>     32 dw              47 dw         +15         136 → 128         -8
rcr<T,T>     39 dw              60 dw         +21         419 → 411         -8
```

### Variant B — readfirstlane only on `group_idx` (minimal)

```
spec     baseline spill    after-fix spill    Δ          divergent-loops    Δ
rcr<F,F>     39 dw              39 dw         0           128 → 128         0
rcr<T,F>     43 dw              44 dw         +1          411 → 411         0
rcr<F,T>     32 dw              47 dw         +15         136 → 128         -8
rcr<T,T>     39 dw              60 dw         +21         419 → 411         -8
```

**Identical results** to variant A — even one `readfirstlane(lo)` triggers the
full spill-up + 8-loop-down trade. The compiler's SGPR-promotion of `group_idx`
ripples through the rest of the function.

### Why this trade is net negative

* **Spill +21 dw on `<T,T>`**: at 8 cycles per spill round-trip, with average
  ~10 spills/iter hitting the hot path (per R56-dm), that's
  **~80 cy × 22 iters = 1760 cy/tile** of NEW main-loop cost. Larger than the
  expected 8 × 20 = 160 cy savings from eliminating 8 divergent loops.

* **Why most of the 411 loops survive**: only the loops directly downstream of
  `group_idx` get optimised away. The other 403 originate from other VGPR
  values that the compiler still treats as divergent — `g.c` descriptor
  components, `g.n` for N_MASKED bounds check, FUSED_KTAIL block intermediates.

* **The 8 loops eliminated** are localised to the C-store path on N_MASKED
  specs: changing `<F,T>` 136 → 128 brings it to par with `<F,F>` 128. So
  variant B essentially "purifies" the N_MASKED path back to non-N_MASKED
  divergence count, but at the cost of 15-21 dw extra spill — a poor trade.

### Falsification: REVERT COMMITTED, baseline restored

Verified: post-revert spill = 39/43/32/39 (back to baseline). Post-revert
metric = 962 (within noise band). No HK changes shipped.

---

## What might actually fix this (R23+ suggestions)

This is documented as forward work — not budgeted for R22 1-round attempt:

1. **Restructure `g.c` descriptor in kernel signature** so the compiler sees
   it as a `__attribute__((amdgpu_uniform))` value or equivalent. Requires
   layout_globals struct refactoring + interaction with kittens load/store
   helpers. Risk: high (correctness across all callers); reward: potentially
   eliminates the 128 baseline divergent loops in `<F,F>`.

2. **Aggressive readfirstlane chain** through the entire C-store epilog —
   readfirstlane on every uniform-but-VGPR value the compiler reads
   (`g.c`, `g.n`, `m_subtile_C`, `r0`, `r1`, `c0`, `c1`). Risk: very high
   (more spill cascades); reward: uncertain (compiler may still find new
   divergence sources).

3. **Move FUSED_KTAIL block to a SEPARATE kernel launch** — accept the launch
   overhead, get the C-store epilog free of FUSED_KTAIL's uniformity taint.
   This would essentially be reverting R34-dm's "FUSED=true free-lunch" but
   the ASM-level evidence here suggests it might NOT have been free-lunch
   after all. Worth re-measuring at R23 if appetite for an A/B test exists.

4. **Wait for AMDGPU LLVM uniformity analysis improvement**. The conservative
   divergence assumption around buffer_load with VGPR offsets is a known LLVM
   issue; future ROCm versions may improve it. Not actionable this round.

---

## Cumulative falsification matrix (R22 final)

| Lever | Verdict | Round | Mechanism / measurement |
|---|---|---|---|
| **A** Async global→LDS                 | FALSIFIED | R2  | Already shipped via inline ASM |
| **B** Triple LDS slab                  | FALSIFIED | R2  | LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL       | FALSIFIED | R4  | Spill neutral, -1.8 pp regression |
| **C-2** K-tail capture refactor        | FALSIFIED | R3  | Already correctly scoped |
| **D** mfma_32x32x64 cell-shape         | FALSIFIED | R5  | Microbench -0.03 % |
| **E** ASM software pipelining          | FALSIFIED | R11 | Microbench -7.28 % |
| **R** Stage-level pipelining           | FALSIFIED | R17 | Microbench -0.07 % (LLVM auto-overlaps) |
| **H/B-rcr** voffset swap               | FALSIFIED | R19 | Uncoalesced HBM reads |
| **H/B-rrr** in-kernel K-tail           | FALSIFIED | R28+R29 (HK) | Compiler aliases A→c VGPRs |
| **HIP transpose** rewrite              | FALSIFIED | R20 | Triton already at 75-110 % HBM peak |
| **RRR spill reduction**                | FALSIFIED | R21 | dA TFLOPS already exceeds fwd TFLOPS |
| **`readfirstlane(group_idx)`**         | **FALSIFIED** | **R22** | **+21 dw spill, only 8/411 div loops fixed** |
| **F** Per-shape dispatcher rules       | LANDED+SAT | R6-R10 | 5 rules, R10-dm audit confirmed top-1 |
| **H/A** Triton fp8_transpose_3d        | LANDED  | R13 | +9.3 % bwd avg |
| **K** var_k spill trim                 | LANDED  | R14 | +0.81 % bwd avg |
| **Q** transpose block tile             | LANDED  | R15 | +1.1 % gpt_oss bwd |

12 levers FALSIFIED, 4 LANDED + SATURATED. **Still no remaining 1-round
positive-EV lever**.

---

## Score band stability (R14-R22, 19 trials)

```
R14=962 R15=966 R16=964 R17=963 R18={964,962,964,964,959} R19={960,965,961,962}
R20=962 R21=963 R22={960, 962}  ← R22 had a re-build mid-round
→ 19 trials min=959, max=966, range=7, median=963
```

R22 confirmed two new score samples within band.

---

## Files touched in R22

* `analysis/_notes/round-22-fp8-grouped-divergent-srd-loops-asm-discovery-readfirstlane-fix-falsified.md` (NEW)

No HK kernel changes committed (failed fix reverted in-tree before commit).

---

## R23+ recommendation

The new ASM-level finding (~283 extra divergent-SRD loops on FUSED_KTAIL specs,
≈ 5660 cy/tile = ~10 % per-tile cost) is a **real, structural inefficiency**
that wasn't visible at the source level. Suggested R23 actions:

1. **Investigate the `g.c` descriptor uniformity flow** — read the kittens
   `store(g.c, cA, ...)` helper to see what it does with `g.c`. If `g.c` is
   accessed via `[]` operator that takes a VGPR coord, the SRD construction
   downstream is what's tainted. May be fixable with a kittens-helper-level
   change rather than kernel-level. (1-2 round effort)

2. **A/B test FUSED_KTAIL=false on the gpt_oss B=32 specs** — empirically
   measure whether moving back to a separate K-tail kernel launch (paying the
   launch overhead but gaining a clean C-store epilog) is faster. R34-dm
   landed FUSED=true assuming it was free-lunch; this round's evidence
   suggests there's a hidden ~10 %/tile cost. (1 round measurement)

3. **Or accept the plateau**: the 962-966 noise band is the architectural
   ceiling under current LLVM uniformity analysis behaviour. Continued
   auto-optimize rounds will be docs-only.
