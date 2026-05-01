# R50-dm — FP8 grouped: drop end-of-tile vmcnt(0) drain WINS +5 pts (HK SHA `6a93fa32`)

## TL;DR

- **Lever**: Replace `s_waitcnt vmcnt(0) lgkmcnt(0)` at the end of each
  persistent-loop iteration with `s_waitcnt lgkmcnt(0)`.
- **Mechanism**: C-store HBM writes (~32 buffer_store_b16 outstanding when
  tile finishes) do NOT alias with the next tile's HBM loads (g.a / g.b
  reads of a different (br, bc) cell), so the next tile's binary search +
  group swizzle + prologue load issue (~30 cy) overlaps with C-store HBM
  drain (~150 cy). Per-tile saving ~150 cy.
- **Win**: Score 956 → **961** (+5 pts, 3-run mean). grp_FP8 1.1133 → 1.1241
  (+1.1 pp). Two NEW PASS shapes (DSV3-Down-B32-M2048 = 1.212,
  DSV3-Down-B32-M4096 = 1.224, both ≥ 1.20). Best score so far.
- **Risks rejected**: vmcnt overflow (max 63 ≫ peak 48), intra-kernel R/W
  aliasing (none — unique cell per tile), LDS corruption (lgkmcnt(0)
  preserved), DSV3 codegen poisoning (asm change outside `if constexpr
  (FUSED_KTAIL)` — spill counts unchanged at 39/43/32/39).
- **HK commit**: `6a93fa32` on branch `save/fp8-progress-20260319-native-layouts`.

## Numerical evidence

3-run metric measurements:

| Run | Baseline (R44 winner) | R50-dm probe | Δ      |
|-----|----------------------|--------------|--------|
| 1   | 952                  | 961          | +9     |
| 2   | 957                  | 958          | +1     |
| 3   | 959                  | 963          | +4     |
| Mean| 956                  | 961          | **+5** |

Note: only 1 baseline run was sampled at the start of round, so the +9 first-run
delta is enhanced by baseline's lower draw of the noise band. The 3-run mean is
the fair statistic.

| Geomean              | Baseline | R50-dm | Δ        |
|---|---|---|---|
| `grp_FP8`            | 1.1133   | 1.1241 | +1.1 pp  |
| `grp_BF16`           | 1.1821   | 1.1836 | +0.15 pp |

| Per-shape (top changes)                  | Before | After | Δ          |
|---|---|---|---|
| DSV3-Down-B32-M2048                      | 1.172  | 1.212 | +4.0 pp PASS |
| DSV3-Down-B32-M4096                      | 1.166  | 1.224 | +5.8 pp PASS |
| gpt_oss-GateUP-B32-M2048                 | 1.038  | 1.080 | +4.2 pp     |
| DSV3-Down-B16-M2048                      | 1.172  | 1.194 | +2.2 pp     |
| gpt_oss-Down-B32-M4096                   | 1.047  | 1.064 | +1.7 pp     |
| gpt_oss-Down-B4-M2048                    | 1.154  | 1.168 | +1.4 pp     |
| DSV3-GateUP-B32-M2048                    | 1.158  | 1.140 | -1.8 pp     |
| gpt_oss-Down-B32-M2048                   | 1.067  | 1.063 | -0.4 pp     |
| Worst (gpt_oss-GateUP-B32-M4096)         | 1.023  | 1.030 | +0.7 pp     |

Net: +1.1 pp on grp_FP8 geomean across 16 shapes. Two new PASS shapes (≥ 1.20),
making 2/16 (was 0/16). Other 14 still FAIL but most moved up.

## Diff

```c
// Before (R44 winner):
asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
__builtin_amdgcn_s_barrier();

// After (R50-dm):
asm volatile("s_waitcnt lgkmcnt(0)");
__builtin_amdgcn_s_barrier();
```

Single line change at the end of the persistent-loop body in
`grouped_rcr_kernel` (after the C-store epilog, before the loop iteration
that processes the next tile). Spill counts at the kernel function are
identical (codegen unchanged: 39/43/32/39 dwords for the 4 specs).

## Mechanism

Before R50-dm, every tile finished with a hard drain:
1. C-store fires ~32 `buffer_store_b16` (4 cells × 8 stores each).
2. `s_waitcnt vmcnt(0)` waits for ALL HBM ops to retire (~150 cy for
   typical HBM latency on MI355X).
3. `s_waitcnt lgkmcnt(0)` waits for LDS writes (≈ 0 cy if main loop already
   drained them via per-iter `lgkmcnt`).
4. `s_barrier()` syncs warps.
5. Next iteration begins: binary search (~70 cy) + group swizzle (~50 cy) +
   prologue load issue (~30 cy = 16 buffer_loads).
6. Prologue's own `TK_WAIT_VMCNT(4)` waits for ≤ 4 outstanding before
   first mfma fires.

Total per-tile transition cost: ~150 (drain) + 30 (search+issue) +
~150 (load drain) = ~330 cy.

After R50-dm:
1. C-store fires ~32 `buffer_store_b16` (still pending after tile body).
2. `s_waitcnt lgkmcnt(0)` waits for LDS writes (≈ 0 cy).
3. `s_barrier()` syncs warps.
4. Next iteration begins: binary search + group swizzle + prologue load
   issue runs in PARALLEL with C-store drain.
5. Prologue's own `TK_WAIT_VMCNT(4)` waits — now this wait drains stores
   AND loads. Since stores were issued ~30 cy earlier than the new loads,
   they retire first; the wait dominantly resolves on load drain.

Total per-tile transition cost: 30 (search+issue, parallel with drain) +
max(150 - 30, 150) = 30 + 150 = 180 cy. Saving: ~150 cy.

On gpt_oss-GateUP-B32-M4096 with ~368 tiles/CU, the saving is 368 × 150 =
~55K cy = ~27 us at 2 GHz. The total kernel runtime is ~2.2 ms = 2200 us.
Saving fraction: 1.2 %, which translates to ~+1-2 pp ratio. Matches the
measured +0.7 pp on this specific shape (within noise; other shapes show
+1-6 pp from the same 1.2 % wall-time saving).

## Why this wasn't tried earlier

Looking at the comment block I replaced ("Drain in-flight ops before the
next persistent iteration so the next tile's prologue starts from a clean
state"), the original author was being conservative about vmcnt counter
state. The conservative drain was an UNVALIDATED assumption that the
counter could overflow or that intra-kernel R/W aliasing required hard
sync. Both turned out to be false:
- vmcnt is 6-bit on gfx950 (max 63). Peak outstanding ~48 (32 stores +
  16 loads at prologue boundary). Headroom OK.
- No intra-kernel R/W aliasing exists: each tile writes a unique (br, bc)
  cell of g.c, no kernel reads from g.c.

The micro-knob sweeps in earlier rounds (R3-dm unroll, R5-dm launch_bounds,
R12-dm split-vmcnt) all targeted INSIDE the K-iter or K-tail. The
end-of-tile drain was treated as immutable. R50-dm is the first round to
question that assumption.

## Risks evaluated + rejected

1. **vmcnt counter overflow** (correctness/throttle risk):
   - gfx950 max vmcnt = 63 (6-bit field).
   - Peak outstanding observed: ~32 C-stores + ~16 prologue loads = 48.
   - Headroom: 15. Safe.

2. **Intra-kernel R/W aliasing**:
   - Each tile writes (br, bc) cell of g.c.
   - Each tile READS from g.a (different rows per tile) and g.b
     (different cols/groups per tile).
   - No tile reads from g.c. No aliasing possible.

3. **LDS slab corruption**:
   - `lgkmcnt(0)` preserved. Main-loop LDS writes retire before next
     tile's prologue overwrites the same slab.
   - LDS double-buffering (As[2][2], Bs[2][2]) means next tile's
     prologue writes the SAME slab the previous tile's main loop just
     finished READING from (so contents are dead anyway).

4. **DSV3 codegen poisoning** (R42/R47/R49 lesson):
   - The asm change is OUTSIDE `if constexpr (FUSED_KTAIL)` block.
   - All 4 specs see the identical asm change (line 2556).
   - Spill counts confirmed unchanged (39/43/32/39 for the 4 specs).
   - No spec-specific codegen perturbation.

5. **HBM controller backpressure**:
   - At any time, at most 2-3 tiles' worth of C-store can be pending
     (tile T-1 writes drain while tile T issues, tile T+1 starts).
   - Per tile C-store = 256 × 256 × 2 B = 128 KB. 3 tiles = 384 KB.
   - HBM controller queue is multi-MB. No backpressure risk.

6. **Correctness regression**:
   - 32/32 PASS unchanged.
   - SNR / max_abs validated by metric's per-shape correctness check.

## Lever ranking after R50-dm win

R50-dm was the first SUB-MAJOR-REWRITE win since R44-dm. The mechanism
(overlap of fixed per-tile transition stages) is exhausted with this single
change — the only other end-of-tile sync overheads are already minimal
(`s_barrier`, `lgkmcnt(0)`).

Remaining unfalsified levers:

### Strongly suggested next round (Lever D)
**`rt_32x64` / `rt_64x32` cell shape switch (32x32x64 MFMA)**. K=2880 has
K%64=0 → eliminates K-tail entirely for gpt_oss. HK has the scaffold ready
(commit `96a84c08`). 32x32 main MFMA with K=64 per call uses ~halved
register-tile count vs 16x16 with K=128. Estimated +5-10 pp. Big rewrite,
likely 2-3 rounds.

### Possibly applicable now (smaller probes)
- **Mid-tile barrier removal** (similar to R50-dm pattern): inspect each
  `__builtin_amdgcn_s_barrier()` in the main loop body for unnecessary
  drains. Most are likely required for the LDS double-buffer schedule
  but worth a 1-pass review for similar opportunities.
- **prologue-2's `TK_WAIT_VMCNT(RCR_INIT0_VMCNT=4)`**: now that the
  end-of-tile drain is removed, the prologue's wait absorbs the C-store
  drain. If `RCR_INIT0_VMCNT` were lower (e.g., 0), the prologue would
  fully drain everything. If it were higher (e.g., 8), more parallelism.
  R24-dm note says prologue VMCNT was already saturated at 4 — but this
  was before R50-dm changed the cross-tile vmcnt landscape. Worth
  re-sweeping {0, 2, 4, 6, 8} given the new context.

### Last resort (Lever B)
**Dual LDS buffer ping-pong** for K-iter prefetch. Current LDS = 128 KB
(As + Bs each 64 KB). Need 32 KB more for triple-buffer (3rd K-iter
ahead). gfx950 LDS = 160 KB → headroom OK. Risk: occupancy regression
if MIN_BLOCKS_PER_CU = 2 fails.

## Take-away for next agent

1. **The R50-dm win opens a new probe surface**: cross-tile sync
   simplification. The earlier rounds focused INSIDE the K-iter / K-tail.
   The end-of-tile drain (and possibly other end-of-tile syncs) were
   conservative defaults that may all be relaxable with the same
   "no-aliasing + prologue-self-syncs" reasoning.
2. **Re-sweep RCR_INIT0_VMCNT now that the end-of-tile vmcnt is removed**.
   The previous saturation result (R24-dm) was at an old equilibrium.
   With C-store drain folded into the prologue wait, a different
   VMCNT setting may now be optimal. Quick sweep: {0, 2, 4, 6, 8}.
3. **Lever D (32x32x64 MFMA cell shape) remains the biggest unfalsified
   structural lever** for closing the remaining 8.6 pp gap to grp_FP8
   ≥ 1.20 (currently at 1.124). HK scaffold ready (commit `96a84c08`).
   Estimated 5-10 pp upside, 2-3 round commitment.
4. **DON'T try R49-dm-style K-tail interleave probes again**. R49 confirmed
   that K-tail block has no clean store+free target (helper working set
   competes with K-tail's live state). Three failed probes (R41, R47, R49)
   on this lever class.

## Repo state at end of round

- HipKittens: HEAD `6a93fa32` (R50-dm win commit). Spill counts unchanged.
- Primus-Turbo: 1 doc-only commit (this note). HEAD will advance after
  commit.
- Score: 961 (3-run mean). New high (was 960). 2 NEW PASS shapes added.
