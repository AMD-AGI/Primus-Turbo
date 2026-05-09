---
name: round-8-direction-F-larger-tiles-PREFLIGHT-FALSIFIED
description: R8 preflight on task md "Direction F" (larger tiles, 256x384 / 512x256 / 256x320 / 256x288). Per-warp accumulator arithmetic shows EVERY larger-N candidate at WARPS_N=4 / 4-acc design overshoots R25's 256-fp32/lane AGPR-allocator threshold; larger-M (RBM>64) breaches it on the A side likewise. The only path that fits the threshold is "drop 4-acc → 2-acc", which structurally removes the FUSED_KTAIL double-buffer that all 8 metric shapes (K%128==64) use. Net F = multi-round restructure with negative single-round EV; defer. R9 forward-pointer = start variant-2 K-split (R7 / 81a05902 plan, Option α A1') rather than yet another preflight.
type: project
---

# Round-8 — Direction F (larger tiles) PREFLIGHT — A-PRIORI FALSIFIED on R25 AGPR-threshold × 4-acc/FUSED_KTAIL coupling

## TL;DR

R5 closed Directions B (cross-stream) and G (cross-shape) a-priori. R6
closed A3 (decoupled-warps) at preflight. R7 split A1 (Stream-K) into
variant-1 (no-op for uniform-K) and variant-2 (K-split, +25-55 envelope,
6-10 round budget); recommended variant-2 as the next implementation
direction. **Direction F (larger tiles) was the only "NEW DIRECTIONS"
entry in task md still without a preflight verdict.** R8 closes that gap.

**Verdict** (3-mechanism close):

1. Every larger-N candidate at the production 8w / 4-acc / `WARPS_N=4`
   design exceeds the R25 256-fp32/lane AGPR-allocator threshold:

   | Tile | RBN = N/WARPS_N | fp32/lane/acc | × 4 accs | vs threshold |
   |------|-----------------|---------------|----------|--------------|
   | 256×256 (current) | 32 | 32   | **128** | safe (1/2 budget) |
   | 256×288           | 72 | 72   | **288** | over by 32 |
   | 256×320           | 80 | 80   | **320** | over by 64 |
   | 256×384           | 96 | 96   | **384** | over by 128 |

   Per-warp acc footprint = `4 accs × (RBM × RBN / 64)` fp32/lane =
   `4 × RBN` fp32/lane (since RBM=64 fixed on the 8w design). The
   threshold (256) is hit at RBN=64 already, and crossed for any
   RBN > 64. R25 verified the AGPR allocator codegen bug at exactly
   this threshold (silent miscompute on `cAB[0][0].tiles[{0,1}][1]`
   under 4w/RBN=64); R59-R61 root-caused it to LLVM AGPR-allocator
   alias bug, not fixable from kernel source.

2. Larger-M (RBM > 64) hits the same threshold on the A side. RBM=128
   at WARPS_M=2 → 4 accs × 128*32/64 = **256 fp32/lane** (exactly at
   threshold; experimentally R25's 4w/64×64 design at 256 fp32/lane is
   the falsified case). RBM=256 (512×256 tile) → 4 × 128 = 512
   fp32/lane, well over.

3. The "drop 4 accs → 2 accs" path that *would* fit (e.g. 256×384 at
   2 accs = 192 fp32/lane, safely under 256) **structurally removes
   the FUSED_KTAIL design**:

   * `cA, cB, cC, cD` (line 3017) are not arbitrary register pressure
     — they are the **2 M-slab × 2 N-slab** acc set that the FUSED_KTAIL
     epilogue uses to handle K%128==64 without an LDS round-trip
     (see lines 3565 + 4386 `TK_WAIT_VMCNT(RCR_KTAIL_VMCNT)` →
     `rcr_mma(cA, a, b0); rcr_mma(cB, a, b1); rcr_mma(cC, a_kt1, b0);
     rcr_mma(cD, a_kt1, b1)`).
   * **All 8 metric shapes have K=2880 = 22*128 + 64**, so all 8
     pay the K-tail. Removing FUSED_KTAIL forces the un-fused fallback
     (Primus `_unfused_forward` + scalar `grouped_ktail_kernel_lds`),
     which the SKILL.md / task md identifies as 5-15% slower on cell
     PMC. Net F at 2-acc = -ve before any tile-size benefit lands.

**Direction F is therefore closed for any single-round payoff** without
first restructuring the K-tail handling (multi-round; lower-EV than
R7's variant-2 K-split which targets the same cells with documented
+25-55 score envelope).

R8 = **docs commit**, no HK changes, no PT code changes, bit-equivalent
to R7 (`81a05902`). The R8 daemon metric is sample #N+3 in the same
noise distribution as R1-R7 (R29 noise floor: ~±5 score around the
694-695 cluster median).

## Mechanism — per-warp accumulator footprint × R25 AGPR threshold

### The R25 AGPR-allocator codegen bug

R25, R59-R61 established that LLVM 17 / ROCm 6.x's AGPR allocator has
a register-aliasing bug at per-warp accumulator footprints ≥ 256
fp32/lane (per R47 hypothesis confirmed by R59-R61 root-cause). The
symptom is silent miscompute: `cAB[0][0].tiles[{0,1}][1]` gets
overwritten by `cAB[0][1]` writes, no SNR detection at small K, breaks
at M ≥ 2048. This is fixed by *staying below* 256 fp32/lane per warp;
no kernel-source workaround exists.

The threshold is **per warp**, not per CTA. The current 8w/256×256
production layout at 128 fp32/lane is **half** the threshold — there
is exactly one doubling of headroom available before the bug triggers.

### Larger-N footprint check

For `WARPS_N=4` (8w grid, can't drop to 2 without losing K-loop
parallelism — see R39b / R6 closure), per-warp `C_acc` cell:

```
RBN = N_TILE / WARPS_N
fp32_per_lane_per_acc = (RBM * RBN) / 64
total_fp32_per_lane    = 4 * fp32_per_lane_per_acc       (4 accs)
                       = 4 * RBM * RBN / 64
                       = (RBM=64) ⇒ 4 * RBN
```

Solve for the largest RBN that fits 256 / lane / warp:

```
4 * RBN ≤ 256
RBN     ≤ 64
N_TILE  ≤ 64 * 4 = 256
```

**Production 256×256 is at the binding RBN=32 (128 fp32/lane), but the
threshold-saturating RBN is 64 (= 256 fp32/lane = exactly the bug
trigger).** Any tile with N > 256 at the current 4-acc design is
**a-priori beyond the bug-safe envelope**.

### Larger-M footprint check (RBM > 64 path)

For `WARPS_M=2` (the production grid; halving WARPS_N to allow more
warps on M is closed by R6 cooperative-load-coupling and R39b 4w-port
falsification):

```
RBM_extended = M_TILE / WARPS_M
total_fp32_per_lane = 4 * RBM_extended * RBN / 64
                    = 4 * (M_TILE/2) * 32 / 64           (RBN=32 default)
                    = M_TILE
```

So `M_TILE ≤ 256` to stay safe. **RBM=64 (= M_TILE=128 — but production
already runs M_TILE=256 i.e. RBM=128? checking)** — wait, production
has `RBM = BLOCK_SIZE / WARPS_M / 2 = 256/2/2 = 64` at BLK=256. Correct,
RBM=64 at M_TILE=256.

Pushing to M_TILE=384: RBM=96, footprint = `4*96*32/64 = 192` fp32/lane
(safe). M_TILE=512: RBM=128, footprint = `4*128*32/64 = 256` fp32/lane
(at threshold). M_TILE=640: RBM=160, footprint = `4*160*32/64 = 320`
fp32/lane (over).

So **larger-M is more-feasible than larger-N** on the AGPR axis: M_TILE
up to 384 stays safely under threshold.

But coverage check kills larger-M for the metric suite:
- `m_per_group = 2048` ⇒ M_TILE=384: 2048/384 = 5.33 → not integer ⇒
  wastes per-group tail rows or requires partial-tile handling.
- `m_per_group = 4096` ⇒ M_TILE=384: 4096/384 = 10.67 → not integer.
- `m_per_group = 2048` ⇒ M_TILE=512: 2048/512 = 4 (clean), but only
  4 tiles_m × bpc tiles per group ⇒ Down-B4-M2048 grid drops to
  4*11*4 = 176 tiles, well below 304 CTAs (0.58 ws/CU), CU under-fill.

The only M_TILE that is **both AGPR-safe AND clean-divisor on every
metric m_per_group** is M_TILE=256 (current). 320 fails 2048 (320*6.4),
384 fails both, 512 wastes M=2048 grid. **Larger-M closed by coverage**.

### The 2-acc (drop FUSED_KTAIL) escape — structural cost

To go larger-N at 4 accs, RBN ≤ 64 binds. Drop to 2 accs:

```
fp32_per_lane = 2 * RBM * RBN / 64 = 2 * RBN  (RBM=64)
2 * RBN ≤ 256 ⇒ RBN ≤ 128 ⇒ N_TILE ≤ 512
```

256×512 fits the AGPR axis at 2 accs. But:

* **Coverage**: N=5760 / 512 = 11.25 → non-integer for GateUP. N=2880 /
  512 = 5.625 → non-integer for Down. **Neither metric N divides 512
  evenly**.

* **Coverage 2 (256×384)**: N=5760 / 384 = 15 (clean GateUP), N=2880 /
  384 = 7.5 (Down fails). At 2 accs, fits AGPR (192 fp32/lane), but
  GateUP-only.

* **K-tail cost** (the load-bearing constraint): the 4-acc design
  exists to enable FUSED_KTAIL — `cA + cB` cover M-slab 0, `cC + cD`
  cover M-slab 1 of the K-tail block. Dropping to 2 accs collapses the
  FUSED_KTAIL epilogue (lines ~3540-3570 + the 4386 N-masked sibling)
  to just `cA + cB`; M-slab 1 of the K-tail must then be handled in a
  separate pass. Two options:
  * Sequential 2× pass (M-slab 0 first, M-slab 1 second) — doubles
    the K-tail wall, kills the FUSED_KTAIL benefit.
  * Fall back to un-fused: caller-side `_unfused_forward` plus the
    scalar `grouped_ktail_kernel_lds` for the K-tail correction. SKILL
    md / R6 quotes 5-15 % slowdown vs FUSED_KTAIL.

* **All 8 metric shapes pay the K-tail** (K=2880 = 22*128 + 64 ⇒
  K%128 = 64 ≠ 0). So the un-fused fallback is the cost on EVERY shape
  if 2-acc is adopted. Hard to recoup with N-tile growth on GateUP-only
  cells (and Down cells get nothing from a non-integer-fit N).

### Numerical envelope (worst-case favorable assumption for F)

Assume best-case 2-acc 256×384 ships on 4 GateUP cells, with un-fused
K-tail penalty 8 % (mid-band SKILL.md range), and the larger N-tile
saves 1× tile of per-tile fixed overhead per CTA per group:

```
GateUP cells contribute 4/8 = 50 % of section sums.
Per-cell N-tile saving (1.5×N): tiles_n = 22 → 15, so per-CTA wave
  count drops by 7/22 = 31.8 %. Realised lift after per-tile prologue
  amortisation (R28 fence-cost model): ~5-10 %.
Un-fused K-tail penalty: -8 % per cell (all 8 cells).

Net section lift = 0.5 * (+8 %) + 0.5 * 0          - all 8 * (8 %)
                 = +4 %                            - 8 %
                 = -4 %.
```

Even in the favorable envelope, Direction F at 2-acc/256×384 is
**net-negative** by 4 % per section (≈ -25 score).

## Score-projection summary (vs R7's variant-2 envelope)

| Direction | Single-round? | Multi-round budget | Score envelope |
|-----------|---------------|--------------------|----------------|
| F (larger N, 4-acc)    | n/a — closed by AGPR threshold | 0          | 0           |
| F (larger M, 4-acc)    | n/a — closed by coverage / grid     | 0          | 0           |
| F (256×384, 2-acc, drop FUSED_KTAIL) | no | 4-6 rounds (K-tail restructure)  | **-25 to -40** |
| F (256×512, 2-acc)     | n/a — closed by coverage (non-integer N) | 0    | 0           |
| **A1' variant-2 K-split** (R7) | no  | 6-10 rounds                  | **+25 to +55** |
| E (incremental barrier removal) | yes/sub-noise | 10-40 rounds          | +20-100 (very wide) |
| γ (acknowledge gap)    | yes (docs)    | 0                                  | 0           |

A1' remains the highest-EV multi-round path. F is closed.

## Forward pointer — R9 starts A1' variant-2 K-split

R7 forward pointer (`81a05902`):

> R8 (if Option α picked): A1' variant-2 scaffolding — add `tile_counter`
> + `partial_buf` to `grouped_layout_globals`, host-side alloc/zero,
> control-flow atomic in `grouped_rcr_kernel` persistent loop (no K-split
> yet, must remain bit-eq). If bit-eq holds and metric is within ±5 of
> baseline, ship; advance to R9 K-split body.

R8 (this round) inverted that order: did the F preflight first because
F was the only "NEW DIRECTIONS" entry without a verdict and the
expected single-round cost was a small docs note. F now closed.

**R9 plan** (per R7 reaffirmed):

1. **R9 = scaffolding-only** (R7's R8 plan, deferred one round):
   * Add `int* tile_counter` to `grouped_layout_globals` (line 2821, as
     trailing field, ABI-extension precedent: `num_slots` R9, `chunk_size`
     R14, `fuse_ktail_off` R16 — all zero-init backward-compat).
   * Host-side: `dispatch_grouped_rcr` allocates `int*` (4 bytes per
     launch, ~negligible HBM), `hipMemsetAsync(0)`, passes pointer in
     globals. Free at end of launch (or stash-and-reuse via static
     thread-local for repeated calls).
   * Kernel: in `grouped_rcr_kernel` persistent loop (line 3121),
     replace `for (int gt = pid; gt < total_tiles; gt += slots_eff)`
     with `int gt = pid; while (gt < total_tiles) { ...; gt =
     atomicAdd(g.tile_counter, 1); }` — variant-1 control-flow atomic.
   * **Important**: variant-1 is a-priori NO-OP for uniform-K (R7 Gate
     3' derivation), so the R9 metric should land within ±5 of baseline
     (R29 noise band). This is the falsification gate: bit-eq + metric
     within ±5 ⇒ scaffolding succeeded, advance to R10 K-split body.
     If metric drops > 5, atomic overhead exceeds projection ⇒ rotate
     to E or γ.

2. **R10 = K-split body** (R7's R9 plan): K-axis split into N partials,
   `atomicAdd(&partial_buf[tile_id * out_size + lane], partial_fp32)`
   inside each CTA's K-loop body. `partial_buf` allocated by host
   (12-48 MB per call, see R7 cost table).

3. **R11 = reduce post-kernel + final fp8 cast**: separate kernel reads
   `partial_buf`, sums across SK-tile partials, writes fp8 output. SNR
   matrix gate.

4. **R12 = per-cell K-split tuning** (sweep `k_splits_per_tile ∈ {2, 4,
   8}` per cell, dispatcher rule).

5. **R13 = extend to RRR (dgrad)**.

6. **R14 (optional) = extend to var-K CRR (wgrad)** — higher SNR risk;
   may need per-group K-split scheduling.

**Cumulative envelope after R14**: +25 to +55 score (mid +40), per R7's
analysis. Round-budget consumption: 6-10 rounds (R9-R14 = 6, plus
optional R12 sweep extension = 1-3 more), well within the 92 remaining
auto-loop rounds.

## Code state this round

Single docs commit. No HK changes, no PT changes. Bit-equivalent to R7
(`81a05902`).

## Files touched this round

* **Primus-Turbo**:
  `analysis/_notes/round-8-direction-F-larger-tiles-PREFLIGHT-FALSIFIED-AGPR-threshold-couples-to-4-acc-FUSED-KTAIL.md`
  (this doc).
* **HipKittens**: none.

## Decision

**FALSIFIED** — Direction F closed at preflight on three independent
mechanisms (AGPR threshold × coverage × FUSED_KTAIL coupling). R9
forward-pointer = start variant-2 K-split scaffolding (R7's plan,
one-round delayed by this preflight).
