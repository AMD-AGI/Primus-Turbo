# Round 55-dm — FP8 grouped: helper-restructure probes (if-else swap + partial-helper) NEUTRAL on metric

**Status**: 2 PROBES NEUTRAL / no commit on HK kernel
**Score before**: 947–960 noise band (R26 was 947, R25 was 960; ceiling stuck at 960 since R22)
**Score after**:  948–951 (probe-2, 4 runs); 946–950 (probe-1, 6 runs); both within band → revert
**HK SHA**: clean (no commit; both probes reverted in this round)
**Round time**: ~30 min, 3 build cycles, 11 metric runs
**Auto-optimize round**: 27

## Goal

Continue Lever C / dead-code-stripping subspace exploration after R54-dm
falsified the `__noinline__` and `__builtin_expect` hints. Specific
hypothesis: the `<0,true,*>` (n_masked) specs run 7 dwords HEAVIER on
spill than the `<0,false,*>` specs (43 / 39 vs 39 / 32 dw). This 7-dw
gap is structurally unique to the n_masked store path; if we can shrink
the inlined `store_c_tile_n_masked` helper's liveness graph at the
grouped-FP8 call site, that gap should narrow.

The active runtime spec for FP8 metric shapes (per `dispatch_grouped_rcr`,
line 5096–5118):

| spec               | KI=0 / FUSED / N_MASK | shapes                              | spill (HEAD) |
|--------------------|-----------------------|-------------------------------------|--------------|
| `<0,false,false>`  | unused (no metric)    |                                     | 39 dw        |
| `<0,true ,false>`  | unused (no metric)    |                                     | 43 dw        |
| `<0,false,true>`   | DSV3 K=7168 N=4096/7168 (8 shapes) | aligned-N + fused-K-tail (K_REM=0)        | 32 dw        |
| `<0,true ,true>`   | gpt_oss K=2880 N=2880/5760 (8 shapes) | unaligned-N + fused-K-tail (K_REM=64)     | 39 dw        |

So the *runtime-active* gap is 7 dw between gpt_oss spec (`<0,1,1>` 39 dw)
and DSV3 spec (`<0,0,1>` 32 dw). Both probes targeted that 7-dw gap.

## Probe 1 — if-else swap (boundary path = THEN branch)

**Change**: at the n_masked C-store dispatch (`kernel_fp8_layouts.cpp`
line 2523–2542), invert the if-else so the boundary-tile (helper) path
becomes the THEN branch and the inner-tile (bare store) path becomes
the ELSE branch. R59 already hoisted the runtime branch out of
`store_c_tile_n_masked`; this is a pure label-ordering tweak.

**Theoretical mechanism**: LLVM register allocator might bias alloc
toward the THEN branch label, forcing the helper's local state into
scratch (more spill) while keeping the hot bare-store path in registers.
Mirrors the dual of R54-dm's `__builtin_expect(.., 1)` failure (which
pushed the boundary path into scratch → +2 dw spill on `<0,1,1>`); the
swap is structurally different (no PHI hint, just label ordering).

**Result — spill profile (rebuild w/ `-Rpass-analysis=kernel-resource-usage`)**:

| spec               | HEAD     | probe-1  |
|--------------------|----------|----------|
| `<0,false,false>`  | 39 dw    | 39 dw    |
| `<0,true ,false>`  | 43 dw    | 43 dw    |
| `<0,false,true>`   | 32 dw    | 32 dw    |
| `<0,true ,true>`   | 39 dw    | 39 dw    |

**Spill profile completely identical**. LLVM's SSA representation is
label-order-agnostic — branch ordering is a syntactic property, not a
semantic one. The register allocator operates on the SSA + PHI graph,
which is invariant under if-else inversion. A textbook-confirmed
NEUTRAL on the spill side.

**Result — runtime (6 metric runs)**: 946 / 948 / 947 / 950 / 950 / 948.
Median 948. Compared to baseline 950 single-run (within 947–960 noise
band). Marginally LOWER than baseline median but well within band.
NEUTRAL on metric.

**Conclusion**: FALSIFIED. Branch label ordering doesn't move the
register allocator. Reverted.

## Probe 2 — partial-helper variant (dead-code stripped)

**Change**: introduced a new `store_c_tile_n_masked_partial` template
that mirrors the partial-N tail of `store_c_tile_n_masked` but with
the leading two early-exit arms (`if (n0 >= n_limit) return;` and
`if (n1 <= n_limit) store(...)`) stripped. Since R59 already hoisted
the boundary check at the call site (line 2523), the helper's internal
fast-path `n1 <= n_limit` is dead at runtime — but LLVM's inliner
cannot algebraically prove this (call-site check uses
`(bc+1)*BLOCK_SIZE > g.n` with BLOCK_SIZE=256, helper uses
`c_tile*RT::cols + RT::cols <= n_limit` with RT::cols=32, and
`bc → c0` involves a multi-warp swizzle; LLVM doesn't see the dead
arm). Stripped helper used only at the grouped-FP8 boundary call
site; `store_c_tile_n_masked` (with fast paths) preserved for dense
FP8 RCR / FP8 partial-K dense / BF16 dense callers (line 1252–1255,
1777–1780).

**Result — spill profile**:

| spec               | HEAD    | probe-2     | Δ          |
|--------------------|---------|-------------|------------|
| `<0,false,false>`  | 39 dw   | 39 dw       | 0          |
| `<0,true ,false>`  | 43 dw   | **25 dw**   | **−18 dw** |
| `<0,false,true>`   | 32 dw   | 32 dw       | 0          |
| `<0,true ,true>`   | 39 dw   | **34 dw**   | **−5 dw**  |

The n_masked specs BOTH improved: `<0,1,0>` shed 18 dw (huge),
`<0,1,1>` shed 5 dw (modest). The non-n_masked specs are bit-identical
(no inlining of the helper, no change). This is a real, reproducible
structural simplification — the dead-code stripping does shrink the
helper's inlined liveness graph and LLVM converges on a smaller spill
footprint for the n_masked specs.

**Result — runtime (5 metric runs after probe-2)**: ~950 first run
(table-only printout), 948 / 950 / 949 / 950. Median ~949.5.
Compared to baseline 950 single-run / 947–960 noise band: NEUTRAL.
Geomean breakdown vs baseline single-run:

| segment          | baseline (R27 first run) | probe-2  | Δ        |
|------------------|--------------------------|----------|----------|
| grp_BF16 geomean | 1.1766                   | 1.1734   | −0.0032  |
| grp_FP8 geomean  | 1.1035                   | 1.1002   | −0.0033  |
| score            | 950                      | 948–950  | within noise |

Both segments shifted by −0.3pp. Within the documented per-run noise
spread (FP8 segment routinely ±0.5pp, BF16 ±0.3pp). The structural
change did not regress.

**Why no metric improvement despite −5 dw spill on the active spec?**

The 5 dw spill saved on `<0,1,1>` are spill ops that live in the
**cold (boundary-tile) helper path**, not the hot main K-loop. Per
gpt_oss FP8 metric shape (N=2880, BLOCK_SIZE=256): boundary tiles are
1 in (2880/256) = 1 / 11.25 ≈ 9% of tiles. The other 91% of tiles
hit the **bare-store** (inner-tile) path, which inlines no helper code
and pays no helper-derived spill cost.

Cost arithmetic:
- 5 dw × ~8 cy spill round-trip ≈ 40 cy saved per *boundary* tile
- × 9% boundary fraction = ~3.6 cy saved per *average* tile
- ÷ ~5500 cy/tile total = ~0.07% per-tile speedup
- → 0.7 pp ratio improvement *upper bound*, well below 5pp metric noise

This matches the observed NEUTRAL runtime. The spill metric improvement
is real but **on the wrong side of the hot-cold divide**.

**Conclusion**: NEUTRAL. The spill-counter view is *misleading* —
total VGPR spill across all template specs doesn't predict runtime.
What matters is **hot-path register pressure inside the main K-loop**,
which the helper changes do not touch. Reverted.

## Cross-probe insight — register spill is not fungible

R54-dm + R55-dm together saturate the "register-allocator hint /
helper-side dead code" subspace of Lever C. Three falsified probes
in 2 rounds:

1. R54-dm probe-1: `__noinline__` on helper → ABI conflict with
   `rcr_8w_load_hoist` SGPR constraints, build failure.
2. R54-dm probe-2: `__builtin_expect((cond), 1)` on bare-store path →
   pushed boundary path into scratch, +2 dw on `<0,1,1>`.
3. R55-dm probe-1: if-else label swap → SSA-invariant, 0 dw change.
4. R55-dm probe-2: dead-code-stripped partial helper → real −5 dw
   on `<0,1,1>` but on cold boundary path → 0 metric change.

The fundamental issue: **the 7-dw gap between gpt_oss and DSV3 specs
is dominated by helper-inlined registers that live ONLY on boundary
tiles**. Closing the gap doesn't help main-loop perf because those
registers aren't competing with the main-loop hot working set on the
~91% of tiles that take the bare-store path.

The real hot-path register pressure delta between gpt_oss and DSV3
is explained instead by:
- gpt_oss has `K_REM=64 ≠ 0` → FUSED K-tail body active at runtime,
  consuming `a_kt1` register tile (~16 dw) + K-tail control state.
- DSV3 has `K_REM=0` → FUSED K-tail body PRESENT in code (R34
  routing) but never executed at runtime, costs no scratch.

This means the **K-tail body itself** (a_kt1 + K-tail load + K-tail
mfma) is the real spill driver on gpt_oss — not the n_masked helper.
The 5 dw helper improvement is real but on the wrong axis.

## Recommendation for R56-dm (auto-optimize round 28)

**Pivot away from helper-side dead-code stripping; attack the K-tail
body itself.** Either:

**Option A (small, contained)**: route gpt_oss-style `K_REM=64`
shapes to the K-tail-only 32x32x64 mfma (Lever D Round-A — half mfma
issue cycles, full K-utilization). Requires solving the 16x16 vs
32x32 cell-layout fan-out problem (cAB_32 in 32x32 layout cannot
register-rebind into cA-cD in 16x16 layout; needs LDS round-trip
to redistribute, which costs ~50-100 cy and may negate the mfma
savings on 22-iter K-loop). Multi-round commit (2-3 rounds) due to
complexity.

**Option B (big, structural)**: full main-loop port to 32x32x64 mfma
(Lever D Round-B — change cA-cD shape from `rt_fl<64,32,col_l,
rt_16x16>` to `rt_fl<64,32,col_l,rt_32x32>`; rewrite ds_read /
buffer_load_lds for the new lane layout). Removes K-tail special
case entirely. 4-5 round commit. Was previously falsified at the
"whole-kernel migration" level (R15-dm) because LDS / load wasn't
co-migrated. Co-migrate this time, gated on LDS pressure budget.

**Option C (de-risk)**: accept score plateau at 947–960 band; ship
small code-quality cleanup (e.g., R55-dm probe-2's partial helper
as a permanent helper for future structural work) without expecting
metric gain. Out of scope for current task body which mandates
"score 持平或跌 → revert"; cannot ship probe-2 under that rule.

I recommend **Option A** for R56-dm: bounded scope (1-2 rounds),
attacks the documented hot-path bottleneck, doesn't require full
LDS / load migration. If Round-A proves the K-tail mfma migration
is not LDS-roundtrip-prohibitive, R57+ could extend to full main
loop; if Round-A's LDS round-trip eats all savings, falsify Lever D
entirely and pivot to Option C / accept plateau.

## Files touched / reverted

- `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - Probe-1: lines 2523–2542 if-else swap. **Reverted**.
  - Probe-2: added `store_c_tile_n_masked_partial` template (~60 lines)
    and routed grouped boundary path to it. **Reverted**.
  - Spill profile post-revert: 39/43/32/39 (bit-identical to HEAD
    `6a93fa32`).
- `/workspace/code/Primus-Turbo`: this note only.

No HK commit this round.

## Reproducer (for future audit)

```bash
# Probe-1: if-else swap
sed -i 's/if ((bc + 1) \* BLOCK_SIZE <= g.n) {/if ((bc + 1) * BLOCK_SIZE > g.n) {/' \
    /workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
# (and swap the bodies of the two arms)
cd /workspace/code/HipKittens/analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=/workspace/code/HipKittens make -j8 tk_fp8_layouts \
  2>&1 | grep -E '(grouped_rcr_kernel|VGPRs Spill)' | head -10
# Expect: 39 / 43 / 32 / 39 (unchanged from HEAD)

# Probe-2: see this note's Probe-2 section for the
# store_c_tile_n_masked_partial body. Compile and observe:
# Expect: 39 / 25 / 32 / 34
```

## Round meta

- Auto-optimize round: 27
- Score trajectory: 947 (R26) → 948 (R27 partial-helper) → revert →
  baseline 949–951 (3-run sample, post-revert).
- Plateau: round 9 of 947–960 noise band, no improvement since R22
  (score 960 from R50-dm winner).
- patience counter: 7/10 — 3 more rounds before patience exhausted.
- HK SHA: `6a93fa32` (R50-dm winner, unchanged through R26 + R27).
