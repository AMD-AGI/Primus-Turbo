# Round-37 — BF16 grouped GEMM: RRR ntail real slowness = 2-launch, not kernel body (R38 scout refinement)

## Status
**SCOUT-REFINEMENT, docs hand-off**. Chat at ~90 min window near-exhausted.
R37 reads the full RRR ntail kernel body + dispatcher to correct R36's
wrong hypothesis about the slow path.

## Metric (baseline only)
```
score = 886/1000  (HEAD 55fad39)
```

## Key correction of R36 hypothesis
R36 suggested (a) cross-boundary scalar path is the main slowness. **Wrong**:
for gpt_oss M=2048 (multiple of TBM=16), `s_offs[group_idx+1]` always
aligns to TBM=16 row boundaries, so `cross_boundary = (row_block_base +
TBM > s_offs[group_idx + 1])` **never fires**. R36 hypothesis (a) is
irrelevant to the metric shapes.

## Real slowness diagnosis (from kernel_bf16_dynamic.cpp line 4074-4103)
```cpp
if constexpr (L == Layout::RCR) {
    g.bpc = kittens::ceil_div(g.n, BLOCK_SIZE);   // RCR covers FULL n range
} else {
    g.bpc = g.fast_n / BLOCK_SIZE;                // RRR covers only [0, fast_n)
}
```

Comment at line 4086-4094:
> RRR/CRR: B is [G, K, N] — N lives on the COLUMN axis, so an OOB N
> column lands at byte-offset `row*N_stride + col_oob`, which is still
> inside the per-group SRD (just wraps to the next K row's valid columns)
> — no clamp triggers, garbage data feeds the MMA. Until we add a
> column-mask path on the B load itself (Phase 6+), RRR/CRR keep the
> legacy `bpc = fast_n / BLOCK_SIZE` and N-tail flows through
> `grouped_tail_kernel`.

**Real cause of 0.6× RCR throughput**: the RRR path needs 2 kernel
launches: the main `grouped_kernel<RRR>` for `[0, fast_n) × [0, g.k)`
PLUS a separate `grouped_ntail_kernel_lds_rrr<64>` for
`[fast_n, g.n) × [0, g.k)`. The second launch has its own prolog /
epilog overhead AND for small tail regions (gpt_oss GateUP
N=5760 → tail=128 cols = 2.2% of N) the ntail kernel occupancy is
poor (only 128/16 = 8 column blocks total).

**This is why H4 transpose → RCR fuse is faster**: H4 turns a 2-launch
RRR into a transpose + 1-launch RCR. The transpose cost (~7% wall)
is less than the second-launch cost + poor-occupancy tail kernel.

## R38 fix options (pick ONE, all multi-round)

### Option 1 (HIGHEST IMPACT) — Add B-load column mask to main RRR kernel
Change `g.bpc = g.fast_n / BLOCK_SIZE` → `g.bpc = ceil_div(g.n, BLOCK_SIZE)`
for RRR like the RCR path. Requires adding a column-mask on the B load
inside `grouped_kernel<RRR>` so OOB columns read 0 instead of wrapping.
This would:
- Eliminate the `grouped_ntail_kernel_lds_rrr` launch entirely on gpt_oss
- Let us drop the H4 transpose for RRR shapes (saves ~7% wall on
  gpt_oss fwd)
- Expected: gpt_oss geomean 1.086 → ~1.17 (+8%), score +30-45

**Effort**: 20-50 lines of C++ in `grouped_kernel` B-load path. Needs
correctness verification on all 24 shapes. Risk: B-load masking may
interact with the existing SRD-clamp mechanism in non-obvious ways
(same class of bug that caused R11 to disable cache-swizzle on the
unswizzled path).

### Option 2 (medium impact) — Reduce barrier count in ntail kernel
Current `grouped_ntail_kernel_lds_rrr<64>` body (line 3571-3639) has
**2 `__syncthreads()` per K_CHUNK iteration**:
- Line 3609: after A/B coop-loads, before FMA
- Line 3638: after FMA, before next-chunk LDS overwrite

Option: LDS double-buffer A_lds / B_lds (2× LDS footprint) so the
second barrier can be removed. Currently each kernel uses
`2 * TBM * K_CHUNK_LDS + 2 * TBN * K_CHUNK_LDS` bytes of LDS. For
TBM=TBN=16, K_CHUNK_LDS = K_CHUNK + 4. With K_CHUNK=64, that's
16 * 68 * 2 * 2 = 4352 bytes. Double-buffering → 8704 bytes, still
well under the 65536-byte LDS limit.

**Effort**: 10-20 lines. Risk: SIMD-level race on LDS write ordering if
double-buffer swap indexing is wrong.

### Option 3 (low impact, quick) — Grow TBM from 16 to 32
Current TBM=TBN=16 means 256 threads per block = 1 wavefront × 4 on
MI355X. Grow TBM to 32 (keep TBN=16) → 512 threads = 2 wavefronts × 4,
better occupancy for the tail region.

**Effort**: change `TAIL_BLOCK_M` constant + grid `blockIdx.y` divisor.
Risk: LDS usage doubles for A_lds.

## R38 cold-start checklist
1. Read rounds 35 / 36 / 37 consolidation + scout notes.
2. Pick Option 1 (highest upside, multi-round — may need R38 + R39).
3. Start by understanding the main RRR kernel's B-load path
   (`grouped_kernel<RRR>` — find via `grep -n "dispatch_grouped.*Layout::RRR" kernel_bf16_dynamic.cpp`).
4. Map how RCR implements `bpc = ceil_div` — the B OOB rows are
   clamped by the unswizzled per-group SRD limit. For RRR we need a
   column-mask approach since row-based clamp doesn't apply.
5. Small focused change → build → probe → metric. Commit if ≥ +5.

## Chat / commit discipline
R37 commit = docs-only scout refinement. R38+ = cold-start agent.
