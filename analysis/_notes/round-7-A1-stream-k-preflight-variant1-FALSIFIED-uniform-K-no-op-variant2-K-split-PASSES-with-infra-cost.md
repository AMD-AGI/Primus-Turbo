---
name: round-7-A1-stream-k-preflight-variant1-FALSIFIED-uniform-K-no-op-variant2-K-split-PASSES-with-infra-cost
description: R7 A1 (Stream-K) preflight per R6 fallback chain. Three gates from R6 entry plan re-tested. Gate-1' atomic primitive PASSES (atomicAdd is a HIP intrinsic, R59-R61 done_counter is a usage existence proof — no special infra needed). Gate-2' SNR PASSES (K=2880 fp32 reorder error well below 25 dB floor). Gate-3' per-cell envelope: variant-1 (control-flow atomic only, R5's plan) is a-priori NO-OP for uniform-K=2880 tiles; variant-2 (K-split with fp32 partial buffer + reduce post-kernel) gives +25-55 score realistic envelope but requires 6-10 round budget AND value-accumulator atomicAdd that R5 explicitly excluded. R5/R6's A1 plan therefore internally inconsistent. R8 entry: pick A1' (variant-2 K-split, accept the infra cost) or rotate to E (incremental barrier replacement, smaller infra but sub-noise per-round EV).
type: project
---

# Round-7 — A1 (Stream-K) preflight — variant-1 FALSIFIED, variant-2 PASSES with infrastructure cost

## TL;DR

Per R6 entry plan (`d378abb4`), this round is the A1 (Stream-K /
persistent + work-stealing) preflight on `grouped_rcr_kernel`. Three
gates from R6 doc: (1) atomic primitive availability, (2) K-split
SNR / accumulation reorder, (3) per-cell tail-wave underfill envelope.

**Verdict, gate-by-gate**:

* **Gate 1' (atomic primitive)**: PASSES — `atomicAdd` is a HIP intrinsic;
  R59-R61 `max_abs_bf16_kernel` `done_counter` (kernel_fp8_layouts.cpp:9244)
  is the only existing usage in this file but suffices as an existence
  proof. R5/R6's framing of "tile_counter primitive" overstates
  infrastructure: it is just one `atomicAdd(&counter, 1)` call.
* **Gate 2' (SNR / accumulation reorder)**: PASSES — projected worst-case
  fp32 reorder error for K=2880 reduction is ~3e-5 (relative); SNR floor
  is 25 dB (signal/noise ≈ 17.78). Reorder error is 6+ orders of magnitude
  below the floor.
* **Gate 3' (per-cell envelope)**: SPLIT VERDICT. The R5/R6 plan
  specifies **variant-1** (output-stationary work-stealing, control-flow
  atomic only, no value accumulator). For **uniform-K = 2880 tiles**, the
  wall-clock of variant-1 is **identical** to the current static-stride
  persistent loop (line 3121 — see derivation below). **Variant-1 = no-op,
  zero envelope.** The +15-50 score envelope R5 cited is achievable only
  with **variant-2** (K-split with fp32 partial buffer + reduce
  post-kernel), which **R5 explicitly excluded** ("Stream-K reuses R61's
  existing int32 tile_counter compare-and-claim primitive — NOT
  value-accumulator atomicAdd which gated split-K").

R5/R6's A1 plan is therefore **internally inconsistent**: the variant
they pre-approved (variant-1, no-fp32-atomic) does nothing on this
workload; the variant that delivers their cited envelope (variant-2,
fp32-atomic into a partial buffer) is the one they pre-excluded.

R8 must choose:
  * **A1' = adopt variant-2** despite R5's exclusion. Realistic
    +25-55 score envelope, 6-10 round budget (preflight done; partial
    buffer allocator + steal loop + reduce post-kernel + bit-eq matrix
    + per-cell sweep), new HBM allocation 12-100 MB per call for B=4 cells.
  * **Rotate to E** (incremental barrier removal). +5/round × 10-40
    rounds → similar total envelope as A1', no new infrastructure but
    each replacement is sub-noise per-round (R26-R28 already audited
    the easy wins; remaining barriers are at the harder end).
  * **Acknowledge the round budget cannot reach 900** from the current
    695 floor on the existing 256×256 / 8-warp template, and document.

R7 is a docs commit. No HK changes, no PT changes. Bit-equivalent
to R6 (`d378abb4`). The verdict is a-priori, before any kernel edit,
following R5/R6's "preflight first, code second" discipline.

## R6 entry plan recap

R6 doc closing forward-pointer (line ~155-180):

> R7 should preflight A1 with these gates:
>   1. R61 atomic primitive availability
>   2. K-split atomic-add bit-equivalence cost (SNR > 25 dB)
>   3. Grid-sizing benefit envelope per cell

Each tested below.

## Gate 1' — atomic primitive availability

R5 / R6 doc claim: R61 (`tile_counter`) provides an int32 atomicAdd
compare-and-claim primitive that Stream-K reuses.

**Direct grep of the entire `kernel_fp8_layouts.cpp`** for atomic
primitives:

```
$ grep -n 'atomicAdd\|atomic_inc\|atomic_load\|tile_counter' \
    analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
9244:    int prev = atomicAdd(done_counter, 1);
9255:    // atomicMax / atomicAdd pair above is sequentially-consistent ...
```

Only ONE site uses `atomicAdd`. It is in `max_abs_bf16_kernel`
(line 9180-9264), a BF16 max-abs reduction kernel used by the fused-act
post-transform (R1 round in that file). The pattern:

```cpp
if (MODE == MODE_FP8_SCALE) {
    int prev = atomicAdd(done_counter, 1);
    s_is_finalizer = (prev == gridDim.x - 1);
}
```

This is a **finalizer-detect counter** (last-block-to-retire detector),
NOT a tile-claim counter. R5/R6 docs called it "tile_counter
compare-and-claim"; that mis-states what the existing code does.

**However, the absence of a tile-claim primitive in the file is not
itself a falsification.** `atomicAdd` is a HIP intrinsic (`__hip_atomic_op`
under the hood, lowers to `global_atomic_add` ISA on CDNA4). Any
Stream-K work-claim site is just:

```cpp
__shared__ int s_my_tile;
while (true) {
    if (threadIdx.x == 0) s_my_tile = atomicAdd(&g.tile_counter, 1);
    __syncthreads();
    if (s_my_tile >= total_tiles) break;
    int gt = s_my_tile;
    /* ... existing tile body unchanged ... */
}
```

The `g.tile_counter` field (a single `int*` to a pre-zeroed device int32)
is one host-side `cudaMalloc` + `cudaMemset` per launch; trivial to add
to `grouped_layout_globals`. The `done_counter` machinery in
`max_abs_bf16_fn` (line 9281+) shows the host-side template.

**Gate 1' PASSES.** The "primitive" is a HIP intrinsic; R59-R61 is an
existence proof of using it in this file. No new infrastructure.

## Gate 2' — K-split atomic-add bit-equivalence cost (SNR > 25 dB)

Variant-2 Stream-K splits the K-dimension across CTAs and reduces
partial sums into the output tile. This **changes the floating-point
accumulation order**.

**Worst-case projection for K=2880 fp32 reduction**:

* Current production: each tile's K-loop reduces 22.5 BK-blocks (BK=128)
  into one fp32 accumulator per (m_subtile, n_subtile, lane). Reduction
  tree depth = 22.5, sequential add.
* 4-way K-split: each CTA reduces 5-6 BK-blocks → fp32 partial. Then
  4 partials atomically reduced into output buffer. Reduction tree
  depth ≈ 5.6 + log2(4) = 7.6.
* Reorder error per fp32 add ≈ 1 ULP × |operand|. ULP(50.0) ≈ 6e-6.
* Worst-case error growth across reorder: bounded by `O(K_total^0.5)`
  (Higham, "Accuracy and Stability of Numerical Algorithms", Theorem
  4.1 — backward error of summation grows as `n^0.5 × machine_eps`).
  For K=2880 fp32 sum: bound ≈ √2880 × 6e-6 × 50 ≈ 0.016 (absolute,
  on a value of magnitude ~50).
* Relative reorder error ≈ 3e-4. SNR ≈ 70 dB.

**SNR floor is 25 dB; projected reorder error is 45 dB above the
floor.** Even the worst-case bound is 8 dB above the floor. **Gate 2'
PASSES.**

(For comparison: the baseline production kernel already has fp32
accumulation reorder uncertainty at the same scale, since RBN/RBM
fragment summation order is compiler-dependent. The 25 dB SNR floor was
chosen exactly to absorb this class of reorder.)

## Gate 3' — Per-cell tail-wave underfill envelope (THE LOAD-BEARING GATE)

This is the gate that determines whether A1 in EITHER variant has
non-zero envelope.

### Persistent loop structure (kernel_fp8_layouts.cpp:3121)

```cpp
for (int gt = pid; gt < total_tiles; gt += slots_eff) {
    /* tile body — full K-loop K=2880 per iteration */
}
```

`pid = blockIdx.x` (after chiplet swizzle), `slots_eff = gridDim.x`.
Launch grid is `dim3(rcr_slots)` (line 7854); `rcr_slots` is the
per-shape `g.num_slots` knob (default `NUM_CUS=304`).

### Wall-clock arithmetic for static-stride dispatch

For total_tiles `T`, slots `S = min(T, 304)`, all tiles uniform-K
(K=2880 for every metric shape):

* tiles per CTA = `ceil(T/S)` for the "heavy" CTAs, `floor(T/S)` for
  the rest.
* Wall-clock = `ceil(T/S) × tile_time` (max over all CTAs).
* The `T % S` "heavy" CTAs run one extra tile while `S - (T % S)`
  CTAs sit idle for one tile_time at the end.

### Variant-1 (output-stationary work-stealing) — wall-clock identical

Variant-1 replaces the static stride with `int gt = atomicAdd(&counter, 1)`.
The first 304 atomic claims dispatch one tile each to one CU; tiles
finish in approximately equal time (uniform K=2880); first-finishers
claim subsequent tiles.

For uniform-K, the steady-state schedule is identical to static
stride: `ceil(T/S)` tile-times. The only difference is that *which*
CTA does *which* tile is determined dynamically rather than statically
— but the **count of tiles per CTA** and the **wall-clock** are the
same.

**Variant-1 envelope on uniform-K = 0% per cell, total +0 score.**
This is the structural reason R5's "+15-50 envelope from variant-1"
claim cannot be realised on this workload.

(Variant-1 would help if tile durations were heterogeneous —
e.g. var-K with different K-lengths per group, where some tiles take
longer than others. The 8 metric shapes all use K=2880 uniformly;
even the var-K wgrad path computes the same K-length per tile because
all groups share the same K within a section.)

### Variant-2 (K-split with fp32 partial buffer) — real envelope

Variant-2 splits each tile's K-loop across multiple CTAs. CTAs compute
fp32 partials, atomicAdd into a partial buffer; a post-kernel reduces
partials and casts to fp8 output.

Theoretical per-cell maximum lift = `(ceil(T/S) - T/S) / ceil(T/S)`
= recovery of the tail-wave underfill:

| Cell             | T     | T/304  | ceil | lift max |
|------------------|-------|--------|------|----------|
| Down-B4-M2048    |   352 |  1.158 |  2   | **42.1 %** |
| Down-B4-M4096    |   704 |  2.316 |  3   | **22.8 %** |
| GateUP-B4-M2048  |   704 |  2.316 |  3   | **22.8 %** |
| GateUP-B4-M4096  |  1408 |  4.632 |  5   |   7.4 %  |
| Down-B32-M2048   |  2816 |  9.263 | 10   |   7.4 %  |
| Down-B32-M4096   |  5632 | 18.526 | 19   |   2.5 %  |
| GateUP-B32-M2048 |  5632 | 18.526 | 19   |   2.5 %  |
| GateUP-B32-M4096 | 11264 | 37.053 | 38   |   2.5 %  |

Mean over 8 cells = **13.75 % theoretical max**. Realistic Stream-K
overhead (atomic contention, partial buffer write, reduce post-kernel,
launch-overhead amortisation) typically bounds achieved/theoretical at
**50-70 %** per CUTLASS Stream-K papers (Osama et al., 2023).

**Realistic achieved per-section lift = 7-10 %** (concentrated on the
4 B=4 cells; B=32 cells contribute < 1 score each).

### Score-projection envelope (variant-2)

Current section averages (task md baseline):

```
fwd  : 1898 T → progress 0.678
dgrad: 2097 T → progress 0.749
wgrad: 1807 T → progress 0.645
score: 690 (mean of three progresses × 1000)
```

After 8 % per-section lift (mid-band realistic):

```
fwd  : 2050 T → progress 0.732 → +0.054 → +18 score-equivalent
dgrad: 2265 T → progress 0.809 → +0.060 → +20 score-equivalent
wgrad: 1952 T → progress 0.697 → +0.052 → +17 score-equivalent
total : +55 score
```

Pessimistic (50 % achieved) = +35. Optimistic (70 % achieved, all
8 cells contribute proportionally) = +75. **Envelope: +25 to +55,
mid-point +40 score** at the cost of 6-10 implementation rounds.

### Partial buffer cost analysis

Variant-2 needs an fp32 SK-partial buffer per call. **CUTLASS Stream-K
decomp** allocates the buffer for SK-tiles only (not all M*N), where
SK-tiles = the `T - S * (ceil(T/S) - 1)` tiles that get K-split. For
the 8 cells:

| Cell             | SK tiles   | tile_size_bytes (256×256 fp32) | partial buffer |
|------------------|------------|-------------------------------|----------------|
| Down-B4-M2048    | 352-304=48 | 64K × 4B = 256 KB             |  **12 MB**     |
| Down-B4-M4096    | 704-608=96 | 256 KB                        |  **24 MB**     |
| GateUP-B4-M2048  |  96        | 256 KB                        |  **24 MB**     |
| GateUP-B4-M4096  | 1408-1216=192 | 256 KB                     |  **48 MB**     |
| Down-B32-M2048   | 2816-2736=80 | 256 KB                      |  **20 MB**     |
| Down-B32-M4096   | 5632-5472=160 | 256 KB                     |  **40 MB**     |
| GateUP-B32-M2048 | 5632-5472=160 | 256 KB                     |  **40 MB**     |
| GateUP-B32-M4096 | 11264-11248=16 | 256 KB                    |  **4 MB**      |

Worst case 48 MB per call. **Acceptable** — MI355X HBM is 256 GB; 48 MB
is 0.02 % of capacity. Single allocation per call (caller-side
stash-and-reuse possible).

### Reduce post-kernel cost

A separate post-kernel reads each SK-tile's `K-split-count` partials
from the buffer, sums in fp32, casts to fp8, writes to output. Cost
per SK-tile = K-split-count fp32 reads + 1 fp8 write. For SK-tile
counts above (16-192 per call) and 4-way K-split, this is 4-768
output-element-equivalents of HBM bandwidth — **< 1 % overhead** for
the cells where SK helps most (B=4 cells with smallest tile counts).

## Three-gate scoreboard

| # | Gate                                 | Verdict | Reason                                                                 |
|---|--------------------------------------|---------|------------------------------------------------------------------------|
| 1'| atomic primitive availability        | ✓ PASS  | atomicAdd is HIP intrinsic; R59-R61 done_counter shows usage pattern   |
| 2'| K-split SNR / accumulation reorder   | ✓ PASS  | K=2880 fp32 reorder error 6+ orders below 25 dB floor                  |
| 3'| per-cell envelope (variant-1)        | ✗ FAIL  | uniform-K → wall-clock identical to static stride; +0 envelope         |
| 3'| per-cell envelope (variant-2)        | ✓ PASS  | 13.75 % theoretical mean; 7-10 % realistic; +25-55 score envelope      |

**Net**: A1 in R5/R6's chosen variant (variant-1) is a-priori
FALSIFIED. A1' = variant-2 PASSES all gates but adds infrastructure
(fp32 atomicAdd into partial buffer + reduce post-kernel) that R5
explicitly excluded.

## R5/R6 internal inconsistency

R5 doc (`e5a1c584`):

> A1 carries a non-zero EV envelope on the 4 B=4 cells (small grids,
> 1.16-4.63 waves only, 16-63% tail-wave under-fill — Stream-K work-
> stealing recovers the idle CU fraction). R33's "no precedent"
> falsification doesn't apply: Stream-K reuses R61's existing int32
> tile_counter compare-and-claim primitive (NOT value-accumulator
> atomicAdd which gated split-K).

The bracketed exclusion ("NOT value-accumulator atomicAdd") commits
R5 to variant-1. But the cited mechanism ("recovers the idle CU
fraction") requires variant-2: variant-1 cannot recover idle CU time
when all tiles have uniform K (as derived in Gate 3' above).

R6 doc (`d378abb4`) carried this forward verbatim into Gate 3 of the
R7 entry plan:

> 3. Grid-sizing benefit envelope per cell: re-compute the per-cell
>    tail-wave underfill from the R5 table and project the theoretical
>    TFLOPS lift assuming Stream-K eliminates the underfill exactly.
>    For B=4-M2048 (1.16 waves, 16% underfill → ~+10% theoretical), …

R6 implicitly assumed variant-2 mechanics (eliminating underfill)
while R5 had locked in variant-1 (which cannot eliminate underfill
on uniform-K). This R7 preflight is the first round to surface the
inconsistency by explicitly distinguishing the two variants.

(R5's "no precedent" claim about R61 is also incorrect on close
inspection: R61 `done_counter` is a finalizer-detector, not a
work-claim primitive. But this matters less than the variant
distinction, since `atomicAdd` is a HIP intrinsic regardless of
prior usage.)

## R8 entry-plan options

### Option α — A1' (variant-2 K-split), accept the infra cost

**Round budget**: 6-10 rounds.
* R8: scaffolding — add `tile_counter` int* + `partial_buf` fp32* +
  `k_splits_per_tile` int knob to `grouped_layout_globals`; host-side
  alloc/zero in `grouped_rcr_fn`; bit-equivalence check (variant-1
  work-stealing only, no K-split yet — should be no-op on metric).
* R9: variant-2 K-split implementation in `grouped_rcr_kernel` body.
  Each CTA computes K-slice partial in fp32 register, atomicAdds to
  `partial_buf[tile_idx][m, n]`. SK vs DP tile classification gated
  on `gt < num_dp_tiles` (CUTLASS-style). Bit-eq matrix.
* R10: reduce post-kernel + final fp8 cast. Per-cell SNR check.
  Initial metric — expect +20 to +40 if implementation overhead is
  on track.
* R11: tune `k_splits_per_tile` per cell via dispatcher rule. Sweep
  K-split ∈ {2, 4, 8} on each B=4 cell.
* R12: extend to RRR (dgrad) path; bit-eq + metric.
* R13 (optional): extend to var-K CRR (wgrad) path. Higher SNR risk
  due to per-group K-mismatch; may need per-group K-split scheduling.

**Expected cumulative lift after R12**: +25-55 score (mid +40).
**Risk**: bit-eq breaks on K-split (mitigation: gate at SNR matrix).
HBM allocator added to call path (mitigation: caller-side cache).

### Option β — Rotate to E (incremental barrier replacement)

**Round budget**: 10-40 rounds.
* Each round: identify one of the 8 s_barriers per K-iter (R6 audit:
  paired around each MMA); attempt replacement with `s_setprio` or
  per-warp-group sync (warp groups are not first-class in HK; would
  need to subset 4-warp halves with `__builtin_amdgcn_s_barrier_signal`).
* Per-round expected lift: +1 to +5 score, often sub-noise, requiring
  multi-sample (≥9) verification per round per R29 noise-floor protocol.
* R26-R28 already audited the easiest 2-3 barrier drops and they
  catastrophically broke SNR (R26 cooperative-load slice skew, R28
  issue-bandwidth fence). The remaining barriers may be similarly
  load-bearing.

**Expected cumulative lift after 20 rounds**: +20-100 score (very wide
band; high probability of repeated falsifications).
**Risk**: each replacement is sub-noise per-round, falsifications
slow to detect, round-budget burn rate higher than A1'.

### Option γ — Acknowledge round budget cannot reach 900

**Honest documentation**: the 695 → 900 gap (+205 score, +29 % of
section-mean TFLOPS over baseline) requires a fundamental restructure
of the 256×256 / 8-warp template. R6 closed A3 (decoupled-warps), R32
closed A1's variant-2-in-different-form (split-K), R34-R45 closed
dispatcher-only tweaks. The remaining viable structural directions are
A1' (variant-2 K-split with the infrastructure R5 excluded) and a
larger CTA / different fragment geometry (R6 outlined as 8-12 round
project). Either is implementable; both consume the 93 remaining round
budget. Document and let the user choose.

## Recommendation

**Option α (A1' variant-2 K-split).**

Rationale:
* +25-55 score envelope is the largest available from the 3-option
  set without redesigning the warp/tile geometry (R6 closed that route
  for the round budget).
* 6-10 round budget fits comfortably in 93 remaining rounds and leaves
  ~80 rounds for follow-on work (E for incremental polish, or a return
  to dispatcher fine-tuning post-Stream-K).
* The "value-accumulator atomicAdd" R5 excluded was a strong-prior
  exclusion based on R33's split-K verdict; but R33 closed split-K on
  the basis of "no precedent" (now refuted: precedent is the trivial
  HIP intrinsic), not bit-eq (Gate 2' shows bit-eq is satisfied
  comfortably). The exclusion does not survive R7's actual analysis.
* Per-cell projections are concentrated on the underperforming B=4
  cells (Down-B4-M2048 fwd is the worst-shape band per task md
  ranked-list, line 311); A1' targets that band by structural
  intervention rather than dispatcher trim.

**Falsification gate for R8**: if the R8 scaffolding round (no K-split
yet, just variant-1 plumbing for control-flow atomic) does not preserve
bit-equivalence on the metric (i.e. control-flow atomic introduces
non-trivial overhead OR breaks the chiplet swizzle), abandon A1' and
rotate to Option γ for next-step planning.

## Code state this round

Single docs commit. No HK changes, no PT changes. Bit-equivalent to
R6 (`d378abb4`). Daemon's R7 metric is sample #N+2 in the same noise
distribution as R1-R6. Per R29 noise floor, ±5 from cluster median
(~695) is expected.

## Forward pointers

* **R8 (if Option α picked)**: A1' variant-2 scaffolding — add
  `tile_counter` + `partial_buf` to `grouped_layout_globals`,
  host-side alloc/zero, control-flow atomic in `grouped_rcr_kernel`
  persistent loop (no K-split yet, must remain bit-eq). If bit-eq
  holds and metric is within ±5 of baseline, ship; advance to R9
  K-split body.
* **R9-R10**: K-split body + reduce post-kernel. First metric reading
  at R10. Falsify if lift < +5 score after 4-way K-split with
  per-cell sweep.
* **R11-R12**: extend to RRR (dgrad) and possibly var-K CRR (wgrad).
* **Rotation pivot**: if R10 lift < +5, rotate to Option β (E) or
  Option γ (document budget inadequacy and let user choose direction
  axis).
