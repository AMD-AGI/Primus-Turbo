# R49-dm — FP8 grouped: K-tail internal C-store interleave FALSIFIED by spill backlash

## TL;DR

- **Lever**: Move the C-store mul+store INTO the K-tail block (between cA/cB
  mfma and cC/cD mfma), free cA/cB VGPR slots BEFORE cC/cD mfma fires.
- **Theory**: Extends R44-dm's mul→store→next-mul interleave to fire one stage
  earlier (during K-tail). cA/cB are dead after their K-tail mfma → store them
  immediately → cC/cD mfma can reuse cA/cB's freed VGPR slots.
- **Falsified by**: spill INCREASED on FUSED specs (DSV3 32 → 37 dwords +5;
  gpt_oss 39 → **51 dwords +12**), runtime regressed -3 pts (952 → 949), and
  gpt_oss-GateUP-B32-M4096 dropped below 1.0 (1.014 → **0.992**).
- **Root cause**: R46-dm note item 4 was actually CORRECT despite my counter-
  argument. K-tail block has no clean store+free target because (a) hoisted
  store-state scalars extend live ranges across the 650-cy K-tail, and (b) the
  N_MASKED helper's internal live state cannot be absorbed by cA/cB's freed
  slots when K-tail's own state (a, b0, b1, a_kt1) is still alive.
- **Status**: REVERTED to R44 winner (HK SHA `37926c98`). Both repos clean.

## Numerical evidence

| Spec template `<grp_id, N_MASKED, FUSED_KTAIL>` | R44-baseline VGPR spill | R49-probe VGPR spill | Δ |
|--|--|--|--|
| `<0, false, false>` (FUSED=false n_aligned)         | 39 | 39 | 0   |
| `<0, true , false>` (FUSED=false n_masked)          | 43 | 43 | 0   |
| `<0, false, true >` (FUSED=true  n_aligned, DSV3)   | 32 | 37 | **+5**  |
| `<0, true , true >` (FUSED=true  n_masked, gpt_oss) | 39 | 51 | **+12** |

R49 only added code inside `if constexpr (FUSED_KTAIL)` AND
`if constexpr (N_MASKED_STORE)` branches, i.e., only the `<0,true,true>` spec
runs the new code at runtime. Yet `<0,false,true>` (DSV3) ALSO got +5 dwords
spill — purely from the hoisted `combined_scale` / `r0/r1/c0/c1` /
`n_aligned_runtime` scalars now spanning the K-tail (which DSV3 never enters).
This is the same "shared FUSED=true template body codegen perturbation"
mechanism that R47-dm hit.

| Metric                                       | R44 baseline | R49 probe | Δ             |
|---|---|---|---|
| Score                                        | 952          | 949       | -3 pts        |
| `grp_FP8` geomean                            | 1.1043       | 1.0985    | -0.6 pp       |
| gpt_oss-GateUP-B32-M4096 ratio (worst case)  | 1.014        | **0.992** | **-2.2 pp**   |
| gpt_oss-Down-B32-M4096                       | 1.048        | 1.042     | -0.6 pp       |
| gpt_oss-GateUP-B4-M4096                      | 1.057        | 1.022     | -3.5 pp       |
| DSV3-Down-B16-M2048 (DSV3, K-tail dormant)   | 1.146        | 1.134     | -1.2 pp       |
| DSV3-GateUP-B32-M2048                        | 1.148        | 1.144     | -0.4 pp       |
| Correctness                                  | 32/32 PASS   | 32/32 PASS | 0           |

DSV3-Down-B32-M2048 was the only winner (1.160 → 1.208, +4.8 pp) but the
geomean still dropped. The +5 dwords spill on the shared DSV3 template body
hurt 4-6 of the 8 DSV3 shapes by -0.5 to -1.5 pp each.

## Diff (reverted)

Reverted; baseline restored. The probe inserted (1) hoisted scalars before the
K-tail block, (2) interleaved mul+store of cA/cB after the cA/cB mfma but
before vmcnt(0), (3) interleaved mul+store of cC/cD after the cC/cD mfma, and
(4) gated the standard C-store epilog on
`!(FUSED_KTAIL && N_MASKED_STORE) || !(g.fast_k < g.k)`.

```cpp
// Inside K-tail block, after rcr_mma(cA, a, b0); rcr_mma(cB, a, b1):
if constexpr (N_MASKED_STORE) {
    if (wm == 0) __builtin_amdgcn_s_barrier();
    mul(cA, cA, combined_scale_hoist);
    store_or_masked(cA, r0_hoist, c0_hoist, ...);
    mul(cB, cB, combined_scale_hoist);
    store_or_masked(cB, r0_hoist, c1_hoist, ...);
}
asm volatile("s_waitcnt vmcnt(0)");
rcr_mma(cC, a_kt1, b0);
rcr_mma(cD, a_kt1, b1);
if constexpr (N_MASKED_STORE) {
    mul(cC, cC, combined_scale_hoist);
    store_or_masked(cC, r1_hoist, c0_hoist, ...);
    mul(cD, cD, combined_scale_hoist);
    store_or_masked(cD, r1_hoist, c1_hoist, ...);
}

// Standard C-store epilog gated:
constexpr bool ktail_has_internal_store = FUSED_KTAIL && N_MASKED_STORE;
const bool standard_store_active =
    !ktail_has_internal_store || !(g.fast_k < g.k);
if (standard_store_active) { /* unchanged R44 body */ }
```

## Root-cause analysis

I argued in the probe planning that R46-dm note item 4 ("no clear store+free
target inside K-tail") was wrong because cA/cB ARE dead after their K-tail
mfma (single-shot accumulation, no further reads). That argument is locally
correct: cA/cB have no future MFMA users after their K-tail mfma. But it
missed two systemic effects:

1. **Hoisted scalars extend live ranges over K-tail**. To do an interleaved
   store inside the K-tail block, the store inputs (`combined_scale`,
   `r0/r1/c0/c1`, `n_aligned_runtime`) must be computed BEFORE the K-tail
   block (so they're available at the interleave point). That places ~6
   scalar registers as alive across the entire ~650-cy K-tail body, vs the
   R44 baseline where they're computed AFTER the K-tail and used immediately.
   LLVM RA must spill 5+ extra dwords to VGPR scratch to free SGPR room (or
   spill VGPRs that lost their slot to the new live set).

2. **N_MASKED helper internal state cannot fit in cA/cB freed slots while
   K-tail state is alive**. The `store_c_tile_n_masked` helper has ~80 B/thread
   scratch (rocprof from R44 baseline); it needs a chunk of contiguous VGPR
   working set to cache row/col reconstruction state. Inside the K-tail block,
   `a` (height=4 reg, 8 dwords/lane), `b0/b1` (height=4 reg each, 8
   dwords/lane each), and `a_kt1` (8 dwords/lane) are ALL alive while the
   N_MASKED helper runs. cA/cB's freed slots (16 dwords/lane combined) are
   not enough to absorb the helper's working set in one contiguous block, so
   LLVM still spills to scratch. Net: helper still spills + extra
   bookkeeping for the now-live K-tail state.

3. **DSV3 codegen poisoning via shared template body**. Even though DSV3
   (`<0,false,true>`) never enters the new `if constexpr (N_MASKED_STORE)`
   branches at compile time, the hoisted scalars BEFORE the K-tail block live
   in the SAME `if constexpr (FUSED_KTAIL)` body that DSV3 also instantiates.
   The +5 dwords of hoisted scalar live range hits DSV3's spill count
   identically. R47-dm's "shared FUSED template body codegen perturbation"
   bites again.

The R44 winner mechanism (mul→store→next-mul interleave inside the standard
C-store epilog) works because at THAT point in the kernel, ALL of K-tail's
state is dead — `a`, `b0`, `b1`, `a_kt1` are all out of live range. The cA-cD
slots can be freed AND the N_MASKED helper has the entire post-K-tail
register file to work with. Moving the store earlier (into K-tail) shrinks
the helper's available scratch space below its working-set requirement.

R46-dm note item 4 was right about the conclusion ("no clear store+free
target") for the right systemic reason ("K-tail state is alive"), even if its
local reasoning ("cA-cD accumulate until C-store") was incomplete.

## Lever ranking after R49-dm falsification

R44-dm winner mechanism is essentially **already optimal** for the
register-tile-as-VGPR-array constraint:
- Cannot move store earlier (R49-dm: K-tail state competes with helper).
- Cannot eliminate K-tail body conditionally (R47-dm: a_kt1 reuse loses MFMA
  overlap; HAS_KTAIL_BODY=false would lose R34's codegen win).
- Cannot use runtime VGPR-array indexing in store helper (R48-dm: 11x
  slowdown from v_cmpx/v_cndmask chains).
- Cannot SRD-hoist K-tail setup (R46-dm: scalar register pressure neutral).

The remaining unfalsified architectural levers from R48-dm next-step plan:

### Strongly suggested next round
**Lever D — `rt_32x64` / `rt_64x32` cell shape switch**. HK has the scaffold
ready (commit 96a84c08); 32x32x64 MFMA on K=2880 → 45 K-iters with NO K-tail
(K%64=0 vs K%128=64). Eliminates the entire FUSED_KTAIL branch for gpt_oss,
which is where most of the spill is concentrated. R29-dm tried this without
the R44 baseline and falsified, but the landscape has changed:
- gpt_oss K-tail block now contributes ~10-15% of per-tile time (R36-dm est).
  Removing it entirely is 5-10 pt geomean upside if MFMA count parity holds.
- 32x32x64 MFMA halves register-tile count for the same K span → ~50% lower
  VGPR footprint per accumulator → ~halves the N_MASKED helper's spill room
  problem.
- Risk: B-load layout changes (32 col stride vs 16) require touching the
  load helpers; build may break.

### Last resort
**Lever B — Dual LDS buffer ping-pong**. Requires LDS budget recalculation
(BLOCK_SIZE * K_BLOCK = 256*128 = 32 KB FP8 per slab; 2 slabs = 64 KB hits
LDS limit on gfx950). Need to verify exact LDS allocation today before
committing to this path. If feasible, it lets the next K-iter's loads
overlap with current K-iter's mfma, hiding ~1 K-iter of HBM latency.

### Probably falsified by R49-dm
**R46(a) — Combine a + a_kt1 into single height=8 register tile**. Same
mechanism as R49-dm: extends K-tail register footprint, hoisted scalars
extend live ranges, codegen poisoning of DSV3. Skip without an explicit
reason to retest.

## Take-away for next agent

1. The "interleave to free dead state earlier" pattern from R44-dm DOES NOT
   generalize backward into the K-tail block. The store helper's working
   set requires the post-K-tail dead-zone to fit; moving stores into K-tail
   leaves them competing with K-tail's live state.
2. ANY change that adds new live state inside `if constexpr (FUSED_KTAIL)`
   (even if dead-coded out for DSV3 by `if constexpr (N_MASKED_STORE)`) WILL
   poison DSV3's spill count via the shared template body codegen path. This
   is the third confirmation of this mechanism (R42, R47, R49). Treat it as
   a hard invariant: any K-tail codegen change must INDIVIDUALLY check the
   `<0,*,true>` spill counts.
3. The remaining unfalsified architectural levers are all major rewrites
   (32x32x64 MFMA cell shape, dual LDS buffer). Micro-knob optimization is
   exhausted within the R44 winner's structure. Next round should commit to
   the 32x32x64 lever (5-10 pt upside, biggest unfalsified opportunity) or
   accept the current ~960 score as the local maximum.

## Repo state at end of round

- HipKittens: clean at SHA `37926c98` (R44 winner). No HK commit this round.
- Primus-Turbo: 1 doc-only commit (this note). No code change.
- Score: 952-959 (baseline noise band). No improvement this round.
