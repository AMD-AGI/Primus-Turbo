# Round 9 (death-march) — FP8 grouped parallel LDS-init cleanup (perf-saturated)

**Scope:** `grouped_rcr_kernel` LDS group-metadata init (`s_offs[]` / `s_cum_tiles[]`)

## TL;DR

Replaced the single-threaded `if (threadIdx.x == 0)` init in
`grouped_rcr_kernel` with a 2-phase parallel pattern:

1. All threads with `threadIdx.x ∈ [0, g.G]` do a warp-coalesced HBM read
   of `g.group_offs[threadIdx.x]` into `s_offs[threadIdx.x]`.
2. Threads with `threadIdx.x ∈ (g.G, MAX_G_PLUS_1)` pad
   `s_cum_tiles[threadIdx.x] = 0x7FFFFFFF` (so the main-loop binary
   search always early-exits on padded entries).
3. `__syncthreads()` — HBM fetches retired, visible.
4. Thread 0 runs the O(G) prefix-scan from LDS (not HBM).

**Result (metric):**
- baseline (round-8 HEAD = 2edba7b7 / 9f47...):
  `score=815, geomean=0.9783, n=16`
- after change: 3 runs → score ∈ {813, 821, 812}, median 813.
- Net delta = **±0, inside ±10 noise band**.

**Resource usage (unchanged):** `grouped_rcr_kernel<0,false,false>`
```
VGPRs: 256   AGPRs: 0   SGPRs: 65   Spill: 67 (VGPR) / 0 (SGPR)
Scratch: 272 bytes/lane   Occupancy: 2 waves/SIMD   LDS: 139796 B
```

## Why no perf gain

The init block runs **once per kernel launch**, not per iteration.

Even in the cold-HBM case (worst for the serial version):
- Serial: `(g.G+1)` HBM reads × ~200 cyc = e.g. 33 × 200 = 6.6 Kcyc (≈ 3.4 µs).
- Parallel: single warp-coalesced wavefront (2-3 cachelines for `int64_t`
  `group_offs`) ≈ 200 cyc (≈ 0.1 µs).

But the metric driver warms up the GPU; `g.group_offs` is L2-hot across
all runs. So the serial version costs ~33 × 50 cyc = 1.65 Kcyc (≈ 0.85 µs).

For the shape bucket we're optimizing:
- gpt_oss-*-B4-M4096 (G=4): 5 L2 reads = 250 cyc = 0.13 µs,
  on an 800-µs kernel = 0.016 % of wall-clock.
- DSV3-*-B32-M4096 (G=32): 33 L2 reads = 1.65 Kcyc = 0.85 µs,
  on a 2-ms kernel = 0.04 %.

Both well below the metric's ±1 % noise floor. Falsification expected;
kept the parallel pattern as a micro-cleanup (strictly better idiom, no
downside, no codegen change in the main loop — register allocation
and spill identical to baseline).

## What this rules out

Host→device init overhead (HBM fetch of `group_offs` + serial cumsum)
is NOT a meaningful leverage point for the FP8 grouped kernel. The gap
is entirely inside the main loop + K-tail / N-tail epilogs (as the
starting hypothesis in the task body said — verified by this round's
absence-of-signal).

## Next-round suggestion

Remaining concrete levers (in order of likely leverage):

1. **AGPR accumulator migration** (round-8 plan, still untried).
   Explicitly steer `cA/cB/cC/cD` into AGPRs via inline-asm or
   `art_base_fl` to free ~128 VGPRs. High risk (needs matching
   `v_accvgpr_read` before `mul(cA, ...)` and `store(g.c, cA, ...)`)
   but would cut the 67-register VGPR spill substantially.
2. **N-tail column-masked C store amortize** across the warp-tile in
   `grouped_rcr_kernel` epilog (line 2473+). BF16-dense commit
   `5feba605` has this pattern; FP8 grouped path hasn't been audited
   against it yet.
3. **XCD swizzle chunk size** (currently hard-coded `64` at line 2004).
   Cross-reference BF16 grouped kernel where `chunk_size=64` was chosen
   via sweep.

## Attribution

- HipKittens kernel edit: lines 2011–2041 of
  `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
- No Python-side changes in this round.

## Metric log locations

- Pre-change baseline: `/tmp/metric_round_9.log`.
- Post-change runs: 813 / 821 / 812 (median 813 ≈ noise around 815).
