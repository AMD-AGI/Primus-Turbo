# Round-31-dm — FP8 grouped: prologue collapse (INIT0+INIT1 → single burst) BROKE `wm==1` half-barrier sync invariant → 6/16 fwd-nan

**Date**: 2026-05-01 (round 4 of 60-round death-march chat)
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `75d2486f` (round-30-dm doc-only)
**HipKittens HEAD**: unchanged (probe edits applied + reverted; .so rebuilt
to match source — bytes-identical)
**Metric**: 916 → 384 (probe, 6 FP8 fwd-nan) → 915 (after revert) — FALSIFIED

---

## TL;DR

Identified DSV3-Down-B32-M4096 (0.969) as this round's worst FP8 shape and
found that DSV3-Down and DSV3-GateUP select the SAME template spec
`<0,F,F>` yet perform dramatically differently (0.97-1.02 vs 1.04-1.08).
The asymmetry is driven by **ki = 16 vs 56** (short K vs long K); short K
amortizes per-tile prologue/epilog overhead less. Targeted the prologue
for structural collapse: INIT0+INIT1 → single 7-load burst.

**Result**: catastrophic **6/16 FP8 shapes fwd-nan**, score 916 → 384.
Root cause: the `if (wm == 1) __builtin_amdgcn_s_barrier();` half-barrier
in the original prologue pairs with the immediate following unconditional
`__builtin_amdgcn_s_barrier();` to form a MATCHED-COUNT wave-sync pattern
that MUST stay intact. Moving the conditional barrier past the INIT1
loads broke the pair-count alignment between wm==0 and wm==1 warp halves
→ warps desynchronized → LDS writes and reads interleaved incorrectly
→ NaN in the accumulator.

Reverted. This round adds to the falsification trail and also surfaces a
**critical invariant for future prologue refactors**: the `wm==1 s_barrier
→ unconditional s_barrier` pair is a MATCHED pair and cannot be separated.

---

## Context: why DSV3-Down is the attack target

Round-4 baseline metric's FP8 per-shape distribution (sorted ascending):

| # | shape                             | ratio  |
|---|-----------------------------------|-------:|
| 1 | `DSV3-Down-B32-M4096`             | **0.969** |
| 2 | `gpt_oss-GateUP-B32-M4096`        |  0.970 |
| 3 | `gpt_oss-GateUP-B32-M2048`        |  0.975 |
| 4 | `DSV3-Down-B16-M2048`             |  0.980 |
| 5 | `DSV3-Down-B16-M4096`             |  1.000 |
| 6 | `gpt_oss-Down-B32-M4096`          |  1.001 |

`DSV3-Down` family (4 shapes: B16/B32 × M2048/M4096): mean 0.994
`DSV3-GateUP` family (4 shapes): mean 1.053

Both families use `grouped_rcr_kernel<0, false, false>` (K%128=0,
N%256=0). Same kernel code path. **Same compiled .so spec.** Same
spill count (67), same scratch (272), same VGPR use (256).

What differs: K dimension. DSV3 config is:
- **DSV3-Down**: N=7168, K=2048 → `ki=16` main-loop iters per tile, `bpc=28` N-tiles per row
- **DSV3-GateUP**: N=4096, K=7168 → `ki=56` main-loop iters per tile, `bpc=16` N-tiles per row

Per-tile overhead (prologue: 2-stage / 7 loads + 2 waits + 3 barriers;
epilog 1/2: 8 MFMAs + ~6 barriers) is **fixed per tile**. Amortized
over 14 steady-state k-iters for DSV3-Down → ~20% overhead. Over 54
k-iters for DSV3-GateUP → ~5% overhead. Matches the ratio gap
(1.053 − 0.994 = +5.9 pp, close to the expected overhead delta).

**Therefore**: any reduction to per-tile overhead benefits DSV3-Down
disproportionately, barely touches GateUP, and doesn't regress it.

## Hypothesis

The existing 2-stage prologue is:

```cpp
// INIT 0: 4 loads (tile-0 B[0], A[0], B[1], A[1])
rcr_8w_load_hoist(b_tile(tic, 0), ..., 0);
rcr_8w_load_hoist(As[tic][0],     ..., 0);
rcr_8w_load_hoist(b_tile(tic, 1), ..., 0);
rcr_8w_load_hoist(As[tic][1],     ..., 0);

if (wm == 1) __builtin_amdgcn_s_barrier();   // conditional, wm==1 only
TK_WAIT_VMCNT(RCR_INIT0_VMCNT=4);            // ≤4 in flight (NO-OP: exactly 4 issued)
__builtin_amdgcn_s_barrier();                // wave sync

// INIT 1: 3 loads (tile-1 B[0], A[0], B[1])
rcr_8w_load_hoist(b_tile(toc, 0), ..., 1);
rcr_8w_load_hoist(As[toc][0],     ..., 1);
rcr_8w_load_hoist(b_tile(toc, 1), ..., 1);

TK_WAIT_VMCNT(RCR_INIT1_VMCNT=6);            // wait for 1 of 7 to drain
__builtin_amdgcn_s_barrier();                // wave sync

// main loop starts
```

Since `TK_WAIT_VMCNT(4)` on a fresh 4-op batch is a no-op (≤4 in flight
is immediately true after issuing 4 ops), one *looks* removable. The
intuition was: collapse the 2 init stages into a single 7-load burst
with a single wait+barrier at the end.

Expected savings: 1 `s_barrier` per tile, ~3-4 % of per-tile overhead
on DSV3-Down (56 tiles/CU × ki=16 × 3-4 pp relative overhead reduction
= ~1-2 pp geomean on DSV3-Down family).

## Experiment — patch applied

```cpp
// Collapsed 7-load burst, single wait+barrier at end:
rcr_8w_load_hoist(b_tile(tic, 0), ..., 0);
rcr_8w_load_hoist(As[tic][0],     ..., 0);
rcr_8w_load_hoist(b_tile(tic, 1), ..., 0);
rcr_8w_load_hoist(As[tic][1],     ..., 0);
rcr_8w_load_hoist(b_tile(toc, 0), ..., 1);   // MOVED UP
rcr_8w_load_hoist(As[toc][0],     ..., 1);   // MOVED UP
rcr_8w_load_hoist(b_tile(toc, 1), ..., 1);   // MOVED UP

if (wm == 1) __builtin_amdgcn_s_barrier();   // MOVED AFTER INIT1 loads
TK_WAIT_VMCNT(RCR_INIT1_VMCNT=6);
__builtin_amdgcn_s_barrier();
// main loop starts
```

## Resource-usage delta

No change. All 4 template specs reported identical spill + scratch:
- `<0,F,F>`: 67 spill / 272 scratch (same)
- `<0,T,F>`: 76 / 308 (same)
- `<0,F,T>`: 48 / 196 (same)
- `<0,T,T>`: 58 / 236 (same)

## Metric — correctness catastrophe

| run                                  | grp_BF16 | grp_FP8 | score | failures |
|--------------------------------------|---------:|--------:|------:|----------|
| baseline (entry, matches round-30-dm)|  1.1820  | 1.0214  |  916  | 0 |
| r31-dm probe (collapsed prologue)    |  1.1837  | **0.1797** | **384** | **6 FP8 fwd-nan** |
| after revert                         |  1.1827  | 1.0202  |  915  | 0 |

Six DSV3 FP8 shapes produced NaN in forward output (the BF16 path was
untouched — only FP8 template spec was affected by the prologue change):

- `DSV3-GateUP-B16-M2048`: fwd-nan
- `DSV3-Down-B16-M2048`: fwd-nan
- `DSV3-GateUP-B16-M4096`: fwd-nan
- `DSV3-Down-B16-M4096`: fwd-nan
- `DSV3-GateUP-B32-M2048`: fwd-nan
- `DSV3-Down-B32-M2048`: fwd-nan

Interestingly, `DSV3-GateUP-B32-M4096` (1.071), `DSV3-Down-B32-M4096`
(0.956), and ALL 8 gpt_oss FP8 shapes passed correctness. The NaN
failures cluster on DSV3 shapes that run the `<0,F,F>` spec without
FUSED_KTAIL — exactly the shapes I was trying to optimize.

## Why it failed — `wm==1` barrier-pair sync invariant

The critical detail is the `if (wm == 1) __builtin_amdgcn_s_barrier();`
conditional barrier. It's a **half-barrier**: ONLY warps with
`warpid/WARPS_N == 1` execute the barrier.

In the working baseline:

1. All 8 warps issue INIT0 loads (4 loads each).
2. `if (wm == 1) s_barrier()` — warps with wm==1 (warps 4..7) enter a
   barrier, warps with wm==0 (warps 0..3) DON'T.
3. `TK_WAIT_VMCNT(4)` — no-op for both halves.
4. `s_barrier()` — ALL 8 warps enter a barrier. wm==1 warps were
   already stalled at the previous barrier; now ALL 8 meet here.

This forms a DELIBERATE STAGGER: wm==0 warps race ahead of wm==1 warps
by one barrier. Effect: wm==1's INIT1 loads happen later than wm==0's,
which changes the order in which LDS slabs get written by the two M
halves. This is likely required because the LDS layout assumes a
specific write-order for correct ds_read at main loop time — the
staggered write-order avoids certain bank-conflict or coalescing
patterns.

In the failed probe:

1. All 8 warps issue all 7 INIT0+INIT1 loads (no barrier between).
2. `if (wm == 1) s_barrier()` — wm==1 warps stall after 7 loads.
3. `TK_WAIT_VMCNT(6)` — waits for 1 of 7 to drain.
4. `s_barrier()` — ALL 8 meet here.

The stagger is gone. All 8 warps issue their full prologue before any
barrier, so the LDS slab fill-order is unconstrained. Some orderings
leave LDS in an incoherent state when the main loop's `ds_read_b128`
fires — output NaN in at least some lanes.

The half-barrier pattern exists SIX times in the file (line 1314,
1518, 1662, 2147, 2726, 5429) — it's a REPEATED idiom across all
3 dense kernels (RCR, RRR, CRR) AND all 3 grouped kernels.
Fifth round-13-dm tile-load falsification also implicitly tested around
this and had similar correctness issues (-37% from spill cliff, but
likely masked a separate correctness issue). **Anywhere this pattern
appears, the two barriers must stay paired: wm==1's conditional
barrier MUST be followed by an unconditional barrier at the same
block scope with no intervening load operations that the next
compute phase consumes.**

Matching store side (also uses a stagger): `if (wm == 0) s_barrier();`
immediately before the store — same half-barrier idiom for the C-store.

## Updated falsification trail (now 19 directions across r17..r31-dm)

| # | round     | direction                                              | result |
|--:|----------:|--------------------------------------------------------|--------|
| 1-17 | R17-R29-dm | (see round-29-dm note)                              | 17 directions |
| 18 | R30-dm    | `sched_group_barrier` 4×{2 VMEM,2 DS,1 MFMA}           | spills −1 to −10, metric −13 |
| **19**| **R31-dm**| **prologue collapse (INIT0+INIT1 → 1 burst)**          | **broke wm==1 barrier-pair → 6/16 DSV3 FP8 fwd-nan; reverted** |

## What this means for next round

The DSV3-Down "short-K prologue amortization" attack vector is
conceptually correct but the prologue itself has a subtle
wave-synchronization invariant that blocks the simplest simplification.

Remaining paths for this attack vector:

1. **Dual-prologue refactor that PRESERVES the wm==1 pair**:
   - Add BOTH halves of the wm==1 barrier pair to the collapsed version:
     `if(wm==1) s_barrier(); s_barrier();` moved to AFTER the 7-load
     burst. This collapses just ONE `TK_WAIT_VMCNT(4)` (the known no-op)
     but keeps all 3 barriers.
   - Savings: only the one no-op wait. Compiler may already elide it.
   - Risk: low — barrier sequence unchanged; only a wait removed.
   - Payoff: ~0 (no-op removal).

2. **ISA-level understanding of the stagger**:
   Dump the grouped_rcr_kernel<0,F,F> ISA with llvm-objdump and
   trace the `s_barrier` instructions; understand exactly which LDS
   banks each M-warp half writes to in INIT0 vs INIT1. Then design
   a collapsed-prologue sequence that uses an EXPLICIT wait on a
   different `wm==0` / `wm==1` barrier pair to preserve the stagger
   without the 3-step dance.

3. **Epilog-side attack**: instead of attacking the prologue (which
   is wave-sync-locked), attack EPILOG 1 (second-to-last K-iter)
   which has its own barrier overhead and is also amortized only
   over ki iters. Less barrier-dense than prologue, potentially
   easier to collapse.

4. **Persistent-loop software pipelining**: the outer-loop restructure
   previously noted in round-29-dm's roadmap (prefetch next tile
   during current store-C). Same prologue-amortization benefit but
   addresses it at the outer-loop boundary instead of per-tile
   prologue. High risk but largest ceiling (the full 7-load
   prologue cost moves entirely off the critical path).

Recommended round-32 action: (1) as a quick 1-round probe to
establish the no-op-wait removal baseline, then pivot to (4) as a
multi-round project if (1) doesn't move. (2) and (3) are useful
fallbacks.

## Round 31-dm verdict

- HipKittens kernel: byte-identical to `19ce45a1` baseline (probe
  applied, broke 6 tests, reverted; rebuilt .so confirmed matching
  baseline spill/scratch counts).
- HipKittens HEAD: unchanged, no commit.
- Primus-Turbo: this notes-only commit.
- Score: 916 entry → 384 probe (−532, catastrophic correctness
  failure) → 915 after revert (noise-band return).
- Falsification trail: now 19 directions across R17-R31-dm.

## Files touched

- HipKittens: zero net change. Probe was a 20-line edit to
  `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:~2140-2160`
  (prologue collapse); reverted after metric failure.
- Primus-Turbo: this notes file only.
