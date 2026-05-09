# Round 9 — Direction D first cut: var-K per-tile coord-decode closed-form port

**Date**: 2026-05-09
**Run**: gpt_oss_fp8_local_20260509_143917 (round_009/100, patience 6/40)
**Hypothesis class**: Direction D (SALU coord-decode optimization, var-K specific)
  per the task md "highest-EV first move" recommendation.
**Verdict**: Code SHIPPED (bit-equivalent, 8/8 SNR PASS); metric NEUTRAL
  (within R29 noise band, +0.4% wgrad section avg sub-threshold).

## What was changed

Single-spot edit in `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
inside `grouped_var_k_kernel_fp8` persistent loop (lines 8378-8395).

**Before** — 6-iter unrolled binary search over `s_cum_tiles[]` LDS reads:
```cpp
for (int gt = pid; gt < total_tiles; gt += slots_eff) {
    int lo = 0;
    int hi = MAX_G_PLUS_1 - 1;
    #pragma unroll
    for (int level = 0; level < 6; ++level) {
        const int mid = (lo + hi + 1) >> 1;
        if (gt >= s_cum_tiles[mid]) lo = mid;          // 1 LDS read
        else hi = mid - 1;
    }
    const int group_idx = lo;
    const int tile_start = s_cum_tiles[lo];            // 1 LDS read
    const int local_tile = gt - tile_start;
    ...
}
```

**After** — O(1) closed-form decode (math-equivalent given uniform
`tiles_per_group`):
```cpp
for (int gt = pid; gt < total_tiles; gt += slots_eff) {
    const int group_idx = gt / tiles_per_group;
    const int tile_start = group_idx * tiles_per_group;
    const int local_tile = gt - tile_start;
    ...
}
```

## Why bit-equivalent (a-priori, no probe needed)

The init code at line 8351-8356 (already shipped) populates:
```cpp
const int tiles_per_group = g.bpr * g.bpc;        // CTA-uniform constant
if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
    s_cum_tiles[threadIdx.x] = static_cast<int>(threadIdx.x) * tiles_per_group;
}
```

so `s_cum_tiles[k] == k * tiles_per_group` for all valid `k ∈ [0, g.G]`.
The binary search converges on `lo = max{k : k * tiles_per_group <= gt}`,
which is exactly `floor(gt / tiles_per_group)`. Math identical, no fp
reorder, no tile-coverage change. Confirmed empirically: metric script
runs 8/8 SNR PASS on every sample (1.9 s correctness gate, SNR > 25 dB
on out / dA / dB for all 8 metric shapes).

This is the same closed form already shipped in BF16 var-K
(`analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:4840`,
`compute_var_k_group_lookup` helper). FP8 var-K had inherited the
binary-search pattern from the RCR/forward kernel template, where
`tiles_per_group` varies per group (forward path). var-K is the special
case where it does not (uniform `[G, n, k]` output shape).

## Metric outcome (8 samples, GPU 3, R29 noise protocol)

```
sample  : 1   2   3   4   5   6   7   8
score   : 695 691 694 695 693 693 693 697
median  : 694
mean    : 693.875
range   : 691 .. 697  (Δ = 6)
```

R29 baseline cluster (rounds 4-8): median 692-695, range 691-696.
**This run is statistically indistinguishable from baseline** on the
score axis (Welch-equivalent ≈ 0.4 σ, ~30 % significance).

Per-section breakdown (mean of 8 samples):

| Section | Pre-R9 baseline | R9 mean | Δ      | % lift |
|---------|-----------------|---------|--------|--------|
| fwd     | 1898 T          | 1911 T  | +13 T  | +0.7 % |
| dgrad   | 2097 T          | 2102 T  | +5 T   | +0.2 % |
| wgrad   | 1807 T          | 1815 T  | +8 T   | +0.4 % |

All three deltas are within run-to-run noise (per-section σ ≈ 8-15 T
across an 8-sample run). No section shows >1 σ lift. **Metric verdict:
NEUTRAL.**

## Why the savings didn't register

R21 PMC measured `SALU/SQ_busy = 85 %` on var-K wgrad Down-B4. The
hypothesis was that per-tile coord-decode contributed materially to
this. Decode-side cycle accounting:

* **Decode SALU / outer-iter (before)**: 6 binary-search iters × ~3 SALU
  + 7 LDS reads ≈ 25-30 cycles per tile-decode.
* **Decode SALU / outer-iter (after)**: 1 scalar idiv + 1 imul + 2 sub
  ≈ 30-50 cycles per tile-decode (idiv on AMDGPU runtime non-pow-2 is
  via `S_DIV_F32` magic-number lowering — a few iters of Newton-Raphson;
  idiv on uniform scalars is fast but not free).

Net per-iter delta is **near-zero or slightly negative on cycle count**;
the win is structural (eliminates LDS load pressure on the persistent
loop's hot path, releasing 7 lds-issue slots / outer-iter).

**The K-loop body's SALU dominates.** Per-tile decode runs 1-4 times
per CTA per launch (tile-count / slots = 1-4 for B=4 cells, 9-37 for
B=32 cells). The K-loop body runs `ki_g` times per tile (= 16-32 for
M=2048-4096 cells). K-loop body SALU = address arithmetic on
`load_a` / `load_b` cooperative-load offsets, plus the per-iter `tic ^=
1, toc ^= 1` and barrier-pair scheduling. That's ~100-200 SALU per
K-iter, totally swamping the per-tile decode savings.

## Why ship anyway (perf-neutral but architecturally cleaner)

* **Bit-equivalent, mirrors a shipped baseline.** The closed-form port
  removes a structural redundancy where FP8 var-K was paying for a more
  general decode than its uniform tile-count requires.
* **Smallest-diff:** `s_cum_tiles[]` LDS storage + init writes left in
  place (could be removed in a follow-on but would need its own bit-eq
  audit on the `threadIdx == g.G + 1` sentinel write at line 8357). Net
  LDS budget unchanged → no occupancy risk.
* **Frees 7 LDS-issue slots / outer-iter** for the compiler to schedule
  HBM-load issue earlier in the persistent loop. Doesn't change wall-
  clock at this metric resolution but is correct-direction.
* **Cleans the surface for a per-K-iter SALU attack** (the actual
  bottleneck, see forward pointer below).

## R10 forward pointer: per-K-iter SALU attack (Direction D step 2)

The remaining Direction D headroom is in the K-loop body itself
(`grouped_var_k_kernel_fp8` lines ~8500-8800). Candidates:

1. **Hoist `tic ^= 1, toc ^= 1` out of the loop** — replace with
   compile-time `level % 2` indexing if the loop is unrolled, or with
   register-rotation idiom that doesn't issue an `s_xor` per iter.
   Expected: 2 SALU / iter × 32 iter × 2 tiles/CTA = 128 SALU saved /
   CTA. Sub-noise individually but stacks with other K-loop attacks.

2. **Cache `load_a` / `load_b` swizzled-offset constants** — `load_a` /
   `load_b` lambdas are inlined per call; the compiler should already
   do this, but if `wm * RBM` / `wn * RBN` are recomputed on every
   call there's hidden SALU. Audit via `-Rpass-analysis` on var-K.

3. **Move the `if (ki_g < 2) continue;` test** — currently inside the
   persistent loop after the decode. For the metric shapes (M_g ∈
   {2048, 4096}, HB=128), `ki_g` is always 16 or 32 ≥ 2. Branch is
   always taken (continue never). Compiler should DCE but worth
   verifying the cmp doesn't generate a SALU.

4. **PMC re-measure on this round's binary** to confirm whether decode
   SALU dropped from R21's measurement (validates the hypothesis even
   if metric-invisible). Use `_probe_round_21_vark_pmc_scaffold.py`
   reference scaffold from R21.

R10 should pick item 1 or 4 first (both are localized, low-risk, and
inform R11-R12 via direct PMC attribution).

## Falsified-but-shipped pattern

Direction D as a category remains open. R9 is the first non-trivial
Direction D code edit (R21 was PMC scoping; R22-R28 were all macro-
flag toggles on K-loop scheduling, not coord-decode). The verdict
"ship a bit-eq simplification, get sub-noise lift, document the next
attack vector" is the correct shape for this category given:

* Per-tile decode is small relative to K-loop body — confirmed by R9.
* The K-loop body has multiple SALU contributors (item list above) —
  next 2-3 rounds attack each individually.
* If R10-R12 also yield sub-noise lifts, Direction D as a whole closes
  with a `~+5 score cumulative` envelope, rotating to Direction A1'
  variant-2 K-split (R7-R8 forward pointer).

## Code state this round

* `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` —
  29 insertions / 10 deletions in `grouped_var_k_kernel_fp8` decode.
  Comment block in-place explains the closed form + BF16 precedent +
  R21 hypothesis link.
* `Primus-Turbo` — this docs file only.

## Why not the R8-forward-pointed A1' variant-2 scaffolding

R8 (`b4033c5f`) forward-pointed R9 to "execute R7's R8 plan (variant-2
scaffolding — add tile_counter to grouped_layout_globals, host
alloc/zero, control-flow atomic in grouped_rcr_kernel persistent loop,
no K-split yet, must remain bit-eq)". On reflection R7 itself proved
that variant-1 scaffolding has provably-zero envelope on uniform-K
(Gate 3' verdict, R7 doc lines ~265-282). One-round-of-no-op-plumbing
to verify a-priori-known bit-equivalence is poor round economics when
the actual lift mechanism (variant-2 K-split with fp32 partial buffer
+ atomicAdd + reduce post-kernel) is a 4-6 round restructure that
R9 cannot complete in one round.

Direction D first-cut (this round) is preferable as the round-9 move
because (a) it is a real code edit with a measurable bit-eq verdict,
(b) it directly targets the R21 PMC-identified bottleneck (vs A1's
indirect tail-wave-recovery attack on the same shapes), and (c) it
fits the "preflight first, code second" discipline by making the first
code edit on a localized, audit-cheap surface before committing to the
multi-round A1' restructure.

If Direction D yields cumulative <+5 score over R10-R12, rotate to
A1' variant-2 in R13. If Direction D yields ≥+5 score, continue
attacking K-loop SALU contributors through R10-R15.
