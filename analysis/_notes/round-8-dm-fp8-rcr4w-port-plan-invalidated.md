# Round 8 — rcr_4w port plan invalidated by occupancy data (mirror)

See `/workspace/code/HipKittens/analysis/_notes/round-8-dm-fp8-rcr4w-port-plan-invalidated.md`
for resource remarks, hypothesis, and round-9 proposal.

## Baseline

`score=822, geomean=0.9858` (new high-side of noise band). 5-run mean
still ~816.

## Finding

Round-7 recommended porting `rcr_4w::kernel` on the claim that it
achieves 4 blocks/CU. Actual `-Rpass-analysis=kernel-resource-usage`
for `_ZN6rcr_4w6kernelE` shows:

```
VGPRs: 198, AGPRs: 256, Occupancy [waves/SIMD]: 1
```

**Lower** occupancy than grouped_rcr_kernel (occ=2). The 4-blocks/CU
README claim was aspirational; VGPR+AGPR footprint (454) blocks the
2nd block allocation.

## Why dense still dispatches to rcr_4w on large grids

Empirical. Likely from:
1. AGPR=256 accumulators → free VGPR for memory ILP
2. 256-thread block → finer work distribution
3. Non-persistent `grid=aligned_grid` → no outer-loop stall

All three are structurally incompatible with "port to persistent grouped
kernel" (RED LINE: single-launch INVARIANT).

## Round-8 action: plan correction only

No kernel change. 4-wave port is now REMOVED from the active roadmap as
a near-term lever (blast radius similar to round-6 WARPS flip + AGPR
migration on top).

## Round-9 candidate: AGPR accumulator migration on grouped_rcr_kernel

Narrower scope (4 accumulator decl sites + 4 `rcr_mma` calls + epilog):

- `rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD` → AGPR-backed variant
- `rcr_mma(acc, a, b)` → `mma_ABt_base(acc.tiles[..], a.tiles[..], b.tiles[..], acc.tiles[..])`
- Scale / store path: insert v_accvgpr_read before `mul(cA, cA, scale)`
  and `store(g.c, cA, ...)`.

If compiler tracks AGPR live ranges tightly, frees ~128 VGPR for spill
reduction / main-loop ILP. 1-2 rounds.

## Commits

- HipKittens: planning doc.
- Primus-Turbo: this mirror.
