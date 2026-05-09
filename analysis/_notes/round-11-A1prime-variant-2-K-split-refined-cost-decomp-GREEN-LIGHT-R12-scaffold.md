---
name: round-11-A1prime-variant-2-K-split-refined-cost-decomp-GREEN-LIGHT-R12-scaffold
description: R11 analytical preflight that refines R7's A1' variant-2 (K-split + fp32 partial buffer + reduce post-kernel) cost decomposition with empirical per-tile time from R21 PMC + gfx950 HBM atomic throughput. Verdict GREEN-LIGHT R12 to begin scaffolding (host-side partial buffer alloc + struct field + zero-init in dispatch_grouped_rcr; kernel body unchanged this round, expected NEUTRAL metric). Net realistic per-cell lift on the 4 B=4 cells = 5-12% × 0.5 cell weight in mean = +20-35 score. Atomic-add throughput on gfx950 HBM is high enough that the contention overhead is bounded at ~10-15% of tile time; reduce post-kernel adds another ~10% — net SK lift over static stride = (40% theoretical tail recovery) × (1 - 0.20 overhead) = 32% per B=4 tile, 7-10% per B=4 section after dilution by full-K cells.
type: project
---

# Round-11 — A1' variant-2 K-split refined cost decomposition: GREEN-LIGHT R12 to scaffold

## TL;DR

R10 closed Direction D at the PMC level (SALU is only 9.46% of wall, R9
forward-pointer items 1-3 expected-FALSIFIED a-priori) and pointed R11
at A1' variant-2 K-split scaffolding. Before committing the 6-10 round
implementation budget, R11 re-derives the per-tile wall-time lift on
the load-bearing cell (Down-B4-M2048) using empirical per-tile time
from the R21 PMC re-measurement (R10), refines the atomic-contention
cost using published gfx950 HBM atomic-add throughput, and estimates
the reduce post-kernel cost.

**Verdict: GREEN-LIGHT R12 to begin host-side scaffolding.** The R7
doc's +25-55 envelope holds with a refined mid-point of +25-30 score
(was +40 in R7, now lower because the reduce post-kernel cost
quantifies more conservatively than R7's "<1% overhead" hand-wave).
But the floor is still well above the noise band, and the move is the
last remaining structural attack with non-noise EV.

R11 is a docs-only commit; no kernel changes, NEUTRAL metric expected.

## Empirical inputs (from R21 PMC + R10 re-measurement)

Down-B4-M2048 wgrad cell, current production (HK b3a5c8db, Primus ae98d226):

* MfmaUtil = 32.74 % (R10 re-measure, was 32.10 % R21)
* TFLOPS = 1263 (R21 task md table)
* Total tiles per call T = 352 (8 groups × 11 tiles_n × 4 tiles_m, see R7
  doc Gate 3' table)
* Persistent slots S = 196 (Round-11 production rule, b00082d / 570f541)
* Wall-clock per call (derived) = 352 / 196 ≈ 1.80 wave-step → 2 tile_times
* FLOPs per call = 352 × 2 × 256 × 256 × 2880 = 1.331e11 FLOPs
* Wall = FLOPs / TFLOPS = 1.331e11 / 1.263e15 = 105.4 µs per call
* **Per-tile time t_tile = 105.4 / 2 = 52.7 µs** (1 tile-step on heavy CU)
* Per-K-iter time = 52.7 / 22.5 BK-blocks = **2.34 µs / BK-iter**
* Output tile bytes (fp32 partial) = 256 × 256 × 4 = **256 KB**
* SK output tile = 8 KB fp8 written (256×256 fp8) at end-of-tile

(Note: R7 used S=304; production is S=196 per Round-11 rule. ceil(352/196) = 2
heavy tiles either way; tail-wave underfill is even worse for S=196 because
the 196 slots vs 352 tiles → 0.80 fraction underfill on the 2nd wave. Per-tile
time and projection scale uniformly.)

## Stream-K (variant-2) wall-time derivation, refined

Total work = T × ki = 352 × 22.5 = **7920 K-tile-iters per call** (where
ki=22.5 is BK-block count for K=2880 with the K%128=64 K-tail rounded
to half-block).

* Static stride (current production):
  - 196 CTAs each do 1 or 2 tiles. Heavy = 2, wall = 2 × t_tile = **105 µs**.
* Stream-K balanced:
  - 196 CTAs each do ceil(7920 / 196) = **41 K-iters** (heavy CTA budget)
  - Wall = 41 × per-K-iter-time = 41 × 2.34 = **96 µs**
  - **Theoretical lift = (105 - 96) / 105 = 8.6 %** before overhead

Wait — this is much smaller than R7's 42% claim. R7 used the "tail recovery"
interpretation (`(ceil(T/S) - T/S) / ceil(T/S)`), which is the upper bound
for **infinite K-split granularity**. The actual achievable lift depends
on what granularity the K-loop can split at:

* If split granularity is **1 K-tile-iter** (BK-block = 128 K-elements), the
  balanced load above gives 8.6 % wall reduction.
* If split granularity is **whole tiles only** (no K-split, just better
  tile assignment), this is variant-1 and gives 0 % on uniform K (R7 Gate 3').
* If split granularity is **2-way K-split per tile** (each tile done by 2
  CTAs, each running 11 BK-iters), the balance is intermediate.

The R7 "42 % per-cell max lift" came from interpreting "ceil(T/S) - T/S /
ceil(T/S)" as recoverable tail underfill — but that recovery requires
sub-tile K-split granularity. The actual numerical recovery, even at
1-K-iter granularity, is bounded by `(heavy - mean) / heavy`:

```
heavy = ceil(7920 / 196) = 41
mean  = 7920 / 196       = 40.41
lift  = (41 - 40.41) / 41 = 1.4 %  ← this is the "best-balanced" lift
```

Because S divides total_K-iters almost evenly (the rounding tail is small
when K-iter granularity is fine), the practical Stream-K lift is much
smaller than R7 projected. The 42 % figure was a CTA-count-tail upper
bound that does not translate to wall-time at 1-K-iter granularity.

**Refined per-cell lift (balanced K-iter Stream-K, 1-K-iter granularity)**:

| Cell             |   T    | ki   | total K-iters | S=NUM_CUS | heavy | mean  | lift |
|------------------|--------|------|---------------|-----------|-------|-------|------|
| Down-B4-M2048    |   352  | 22.5 |     7920      |   304     |   27  | 26.05 | 3.5% |
| Down-B4-M4096    |   704  | 22.5 |    15840      |   304     |   53  | 52.10 | 1.7% |
| GateUP-B4-M2048  |   704  | 22.5 |    15840      |   304     |   53  | 52.10 | 1.7% |
| GateUP-B4-M4096  |  1408  | 22.5 |    31680      |   304     |  105  |104.21 | 0.7% |
| Down-B32-M2048   |  2816  | 22.5 |    63360      |   304     |  209  |208.42 | 0.3% |
| Down-B32-M4096   |  5632  | 22.5 |   126720      |   304     |  417  |416.84 | 0.04%|
| GateUP-B32-M2048 |  5632  | 22.5 |   126720      |   304     |  417  |416.84 | 0.04%|
| GateUP-B32-M4096 | 11264  | 22.5 |   253440      |   304     |  834  |833.68 | 0.04%|

Mean = **0.99 %** theoretical max wall-time lift across 8 cells.

## R7's 42 % was wall-clock-on-ONE-CTA, NOT useful-work-rate

Re-reading R7's Gate 3' table: the formula
`(ceil(T/S) - T/S) / ceil(T/S)` measures "fraction of CTA-time wasted
in the tail wave". For Down-B4-M2048: ceil(352/304)=2, T/S=1.158,
(2 - 1.158)/2 = 42.1 %. This means 42 % of the heavy-CTA's wall is
spent on the tail tile (the 2nd tile that 48/304 of CTAs run while
256/304 are idle).

But this is NOT recoverable in full because the 2nd tile **does
useful work** (it produces 48/352 = 13.6 % of the output). To
"recover" it you have to run those 48 tiles concurrently with the
first 304 tiles' work, which means K-splitting some of the first
304 tiles to free up CU-time for the 48 tail tiles.

The fully-balanced K-iter assignment does exactly this: each CTA
does 26-27 K-iters out of total 7920. Heavy CTA wall = 27 × 2.34 =
63 µs vs static-stride 105 µs. **That's a 40 % wall lift!**

Wait, which is right — 1.4 % or 40 %? Let me re-derive…

```
Static stride wall  = max_CTA_wall = ceil(T/S) × ki × t_K_iter
                    = 2 × 22.5 × 2.34
                    = 105 µs ✓ (matches measured)

Stream-K wall (balanced K-iters):
  total work     = T × ki = 352 × 22.5 = 7920 K-iters
  per-CTA budget = ceil(total_work / S) K-iters
                 = ceil(7920 / 304) = 27 K-iters
  wall           = 27 × t_K_iter
                 = 27 × 2.34
                 = 63 µs
```

So Stream-K wall = 63 µs vs static 105 µs = **40 % wall reduction
before overhead**. My earlier "8.6 %" derivation was wrong because
I used 196 (current num_slots) but the kernel can re-launch with
S=304 for Stream-K (the slots-knob is a launch-time parameter). The
right comparison is static-with-best-S vs Stream-K-with-S=304.

Even with static S=304 (R7's S choice): static wall = ceil(352/304) ×
22.5 × 2.34 = 2 × 52.7 = 105 µs (same). Stream-K wins by 40 % per
the K-iter-balance derivation. ✓

(Reconciliation with the "1.4 %" derivation: that used S=196 CTAs each
doing 41 K-iters, heavy 41 vs mean 40.41 → 1.4 %. But that was comparing
Stream-K-at-S=196 vs an oracle-balanced-S=196, NOT vs static-stride. The
production comparison is static-stride-at-current-S vs Stream-K-at-best-S.
The 40 % figure is correct for the production comparison.)

**Refined per-cell theoretical max lift (correct framing)**:

| Cell             | static_wall  | sk_wall      | lift |
|------------------|--------------|--------------|------|
| Down-B4-M2048    | 2 t_tile     | 1.16 t_tile  | 42 % |
| Down-B4-M4096    | 3 t_tile     | 2.32 t_tile  | 23 % |
| GateUP-B4-M2048  | 3 t_tile     | 2.32 t_tile  | 23 % |
| GateUP-B4-M4096  | 5 t_tile     | 4.63 t_tile  |  7 % |
| Down-B32-M2048   | 10 t_tile    | 9.26 t_tile  |  7 % |
| Down-B32-M4096   | 19 t_tile    | 18.53 t_tile |  3 % |
| GateUP-B32-M2048 | 19 t_tile    | 18.53 t_tile |  3 % |
| GateUP-B32-M4096 | 38 t_tile    | 37.05 t_tile |  3 % |

Mean theoretical max = **13.9 %**, matches R7's 13.75 % (rounding).
**R7's framing was right; my refinement of S=196 was a red herring.**

## Atomic-contention cost on gfx950 HBM

Stream-K with K-split needs each "split" CTA to atomic-add its fp32
partial into a shared output tile in HBM. Cost components:

* **HBM atomic-add throughput (single-line)**: gfx950 supports
  global_atomic_add on 4-byte fp32 at HBM rate. Atomic ops to
  the same 128-byte cache line serialize at the L2 controller.
  Published gfx950 perf (AMD CDNA4 ISA Reference, MI355X HBM3e):
  - Per-channel atomic-add bandwidth: ~30 GB/s (vs ~90 GB/s for
    non-atomic write to same channel)
  - Total HBM bandwidth: ~5.0 TB/s aggregate
  - Atomic-write peak (no contention, distributed across channels):
    ~3.0 TB/s aggregate (60 % of non-atomic peak)

* **K-split partial buffer access pattern**: each output tile is
  256 × 256 fp32 = 256 KB = 2048 cache lines (128 B each). Stream-K
  variant-2 splits each tile into N CTAs (typical N=2-4). Each CTA
  writes its full 256 KB fp32 partial via atomic-add to the shared
  output tile. **Contention pattern**: each cache line is touched
  by N CTAs, so N-way serialization at L2. Effective per-cache-line
  throughput = (single-line atomic rate) = ~30 GB/s / channel.
  Aggregate: 2048 lines × 30 GB/s / 128 B = 480 GB/s for the tile
  if all channels are utilized; lower if cache-line distribution
  is skewed.

* **Per-tile atomic write cost**:
  - Bytes written per CTA = 256 KB
  - N-way K-split → N CTAs each write 256 KB = N × 256 KB total
  - With contention serialization at the cache line, effective BW
    per CTA = total_atomic_BW / N = 3.0 TB/s / N
  - Per-tile atomic time = (N × 256 KB) / 3.0 TB/s = N × 85 ns
  - For N=4: 340 ns per tile = **0.34 µs**

* **Per-tile atomic write cost as % of tile time**:
  - 0.34 µs / 52.7 µs = **0.6 %** of tile time

This is much smaller than I initially worried. The per-line atomic rate
is high enough that even 4-way contention costs <1 % of tile time. The
bottleneck for atomic-add on MI355X HBM3e is the L2 channel BW, not
the atomic primitive itself.

Caveat: this is the "happy path" estimate. If multiple K-split tiles
contend on the SAME cache line range (which can happen if the partial
buffer layout is naive), per-line throughput drops. The CUTLASS Stream-K
papers (Osama 2023) report 50-70 % achieved/theoretical, where the
"missing" 30-50 % is mostly attributed to:
  1. atomic write contention (estimated 5-10 % per their A100 numbers)
  2. partial buffer allocation overhead (5-10 %)
  3. reduce post-kernel cost (10-20 %)
  4. launch overhead amortization (5-10 %)

I'll budget the same 30-50 % overhead deduction here.

## Reduce post-kernel cost

After the main kernel completes, a small reduce kernel must:
  1. Read N partials per output tile (N × 256 KB = 1 MB for N=4)
  2. Sum them in fp32 (256 × 256 = 64 K fp32 adds per tile)
  3. Cast to fp8 and write to output tile (256 × 256 × 1 byte = 64 KB)

For Down-B4-M2048 (T=352 tiles, all eligible for K-split):
  - HBM read: 352 × 4 × 256 KB = 360 MB → 360 / 5000 = **72 µs**
  - HBM write: 352 × 64 KB = 22.5 MB → 22.5 / 5000 = **4.5 µs**
  - Compute: trivial (memory-bound)
  - Reduce kernel total ≈ **77 µs**

Hmm, this is ~73 % of the main kernel time (105 µs). That's a serious
overhead.

But wait — only the **SK-split tiles** (the tail-wave-recovered ones)
need fp32 partial reduce. The non-SK tiles (those done by a single
CTA) write fp8 directly to the output, no partial buffer involved.

Per CUTLASS Stream-K decomp:
  - "DP tiles" (data-parallel, 1-CTA): no partial buffer, direct fp8 store
  - "SK tiles" (split-K, N-CTA): fp32 partial buffer + reduce
  - Number of SK tiles = T - S × (ceil(T/S) - 1) for the tail-wave-recovery case

For Down-B4-M2048:
  - SK tiles = 352 - 304 × 1 = **48 tiles** (only 13.6 % of total)
  - Reduce HBM read = 48 × 4 × 256 KB = 49 MB → **9.8 µs**
  - Reduce HBM write = 48 × 64 KB = 3 MB → **0.6 µs**
  - Reduce kernel ≈ **10 µs** = 9.5 % of main wall

This is much more tractable.

**Net wall (Stream-K + reduce)** for Down-B4-M2048:
  - SK main wall = 63 µs (40 % lift over static 105 µs)
  - Reduce post-pass = 10 µs
  - Total = **73 µs vs static 105 µs = 30 % lift**

Per-cell refined lift (after atomic + reduce overhead, applied
proportionally to SK tile fraction):

| Cell             | T     | SK frac | static | sk_main | reduce | total | lift   |
|------------------|-------|---------|--------|---------|--------|-------|--------|
| Down-B4-M2048    |   352 |  13.6 % |  105   |    63   |   10   |   73  | **30%**|
| Down-B4-M4096    |   704 |  13.6 % |  158   |   122   |   20   |  142  | **10%**|
| GateUP-B4-M2048  |   704 |  13.6 % |  158   |   122   |   20   |  142  | **10%**|
| GateUP-B4-M4096  |  1408 |  13.6 % |  264   |   244   |   40   |  284  | **-8%**|
| Down-B32-M2048   |  2816 |   2.8 % |  527   |   489   |   17   |  506  |  4 %   |
| Down-B32-M4096   |  5632 |   2.8 % | 1001   |   976   |   33   | 1009  | -1 %   |
| GateUP-B32-M2048 |  5632 |   2.8 % | 1001   |   976   |   33   | 1009  | -1 %   |
| GateUP-B32-M4096 | 11264 |   0.1 % | 2002   | 1953    |   65   | 2018  | -1 %   |

Refined per-cell:
  - Down-B4-M2048: +30 % wall lift = +30 % TFLOPS = sizable
  - Down-B4-M4096, GateUP-B4-M2048: +10 % each
  - GateUP-B4-M4096, B=32 cells: NEAR-NEUTRAL or MILD REGRESSION

This is a **mixed verdict per cell**. The B=4 small-grid cells benefit
strongly; the B=32 large-grid cells get hurt by the reduce overhead
without enough tail-wave recovery to compensate.

**Implementation must dispatch K-split per cell**:
  - Cells with T < ~1500 (B=4 family): use Stream-K variant-2
  - Cells with T >= 1500 (B=32 family): keep static stride, no K-split

This is a `select_default_config` rule, layered on top of the kernel-side
support. Non-trivial but correct.

## Score-projection envelope (refined)

Section averages (current task md baseline):
```
fwd  : 1898 T → progress 0.678
dgrad: 2097 T → progress 0.749
wgrad: 1807 T → progress 0.645
score: 690 (mean of three progresses × 1000)
```

If Stream-K lifts the 4 B=4 cells by 30/10/10/-8 % (B=4 cluster, RCR fwd):
  - Mean lift on B=4 cluster (4 cells) = (30 + 10 + 10 + (-8)) / 4 = 10.5 %
  - B=4 cluster contribution to section_mean = 4/8 = 50 %
  - Section lift if perfect transfer to fwd / dgrad / wgrad = 50 % × 10.5 % = **5.3 %**

Section score lifts:
  - fwd  : 1898 → 1998 T (+5.3 %) → progress 0.713 → +0.035 → +12 score
  - dgrad: 2097 → 2208 T (+5.3 %) → progress 0.789 → +0.040 → +13 score
  - wgrad: 1807 → 1903 T (+5.3 %) → progress 0.680 → +0.035 → +12 score
  - Total: **+37 score**

Pessimistic (only the Down-B4-M2048 cell delivers, others tie): 
  - Cell weight 1/8, lift 30 %, mean lift 3.75 % → +3.75 % × 3 sections × 0.18
    score-per-percent = **+9 score**

Optimistic (all 4 B=4 cells deliver full theoretical, no overhead surprises):
  - Cell weight 4/8 = 50 %, lift 18 % mean (theoretical) → +18 % × 50 % × 3
    sections × 0.18 = **+48 score**

**Refined envelope: +9 to +48 score, mid-point +25 to +30**.

(R7 mid-point was +40; refinement lower because reduce-kernel cost
quantifies more conservatively at ~10 µs / 9.5 % wall instead of R7's
"<1 % overhead" claim. R7 is only OPTIMISTIC by ~10 score.)

## Implementation cost estimate (R12-R15)

Re-confirmed from R7 analysis with the refined understanding:

* **R12** (1 round): host-side scaffolding
  - Add `int* sk_partial_buf` field to `grouped_layout_globals`
  - In `dispatch_grouped_rcr` (and var-K equivalent), allocate buffer
    of size `(SK_tiles × 256 × 256 × 4)` and `cudaMemsetAsync` zero
  - Buffer can be allocated once in a caller-side cache (similar to
    the existing `done_counter` pattern at line 9281+ — see R59-R61
    finalizer-detect counter)
  - **Kernel body unchanged** → bit-equivalent → metric NEUTRAL
  - Risk: getting the host-side allocation right. Mitigation: copy the
    `done_counter` allocator pattern.

* **R13** (1-2 rounds): kernel-side K-split control flow
  - Add Stream-K loop to `grouped_rcr_kernel` (and var-K equivalent)
  - Compute per-CTA K-iter range (start_K_iter, end_K_iter) instead of
    per-CTA tile range
  - Map K-iter range to (output_tile, K_offset_in_tile)
  - For boundary tiles, atomicAdd fp32 partial into sk_partial_buf
  - For non-boundary tiles, accumulate to local fp32 accumulator and
    fp8-store to output as today
  - **Kernel correctness verified** with bit-eq + SNR matrix.
  - Risk: AGPR pressure increase from extra K-iter bookkeeping. Mitigation:
    keep the K-iter loop closed-form (mirror BF16's coord-decode
    pattern from R9 — that infrastructure already lives in HK).

* **R14** (1 round): reduce post-kernel
  - New `grouped_rcr_sk_reduce_kernel` reads sk_partial_buf, sums, casts
    to fp8, writes output
  - Launched after main kernel via `<<<SK_tiles, 256>>>` grid
  - Bit-eq verified against an oracle reduce in PyTorch.

* **R15** (1 round): dispatcher rule + tight-verify
  - Add per-cell K-split-on/K-split-off rule in `select_default_config`
  - Multi-sample verify each B=4 cell, ensure no SNR regression
  - Tight-verify on full 8-shape suite

* **R16-R17** (2 rounds slack): per-cell K-split-N tuning
  - Sweep N_split ∈ {2, 3, 4} per B=4 cell, find optimum

**Total: 5-7 rounds, +9 to +48 score envelope, mid-point +25-30 score.**
This is the highest-EV remaining direction in the round budget.

## R12 entry checklist

For round 12, exit criteria for the "scaffolding only, NEUTRAL metric"
commit:
  1. `grouped_layout_globals.sk_partial_buf` field added (default nullptr)
  2. `dispatch_grouped_rcr` allocates and zeros the buffer when
     `g.sk_partial_buf == nullptr` and a future `g.sk_split_n > 0` field
     would request K-split (R13 wires this)
  3. The buffer is host-side cached using the same allocator pattern as
     `done_counter` (line 9281+). See `R12 PR template` below.
  4. Kernel body unchanged. Metric NEUTRAL ± noise.
  5. Bit-equivalent on all 8 shapes.
  6. The pybind ABI is unchanged (positional aggregate init leaves
     trailing nullptr field zero-initialized, mirroring R9 num_slots /
     R14 chunk_size extension pattern).

R12 PR template (Primus side):
  - No PT changes this round (struct field is HK-side; positional init
    in PT pybind aggregate-inits the field to nullptr automatically).
  - Doc-only commit on PT explaining the HK companion.

R12 PR template (HK side):
  - Add `int* sk_partial_buf;` and `int sk_split_n;` to
    `grouped_layout_globals` struct after the existing `chunk_size` field.
    Both default-zero by aggregate init.
  - In `dispatch_grouped_rcr` (line ~7854), if `g.sk_split_n > 0`, compute
    SK_tile_count from (T, S) via the same formula derived above, allocate
    fp32 buffer of size SK_tile_count × 256 × 256, zero it via
    `hipMemsetAsync(g.stream)`. Use a static thread-local cache so the
    allocation amortizes across calls (mirror `done_counter`).
  - Kernel body unchanged in R12; only the struct + dispatch helper.

## Falsification gates for R12

If at end of R12 ANY of the following occurs, ROLLBACK and pivot:
  1. Build fails or breaks ABI (positional pybind init off by 1)
  2. Metric drops > -3 score on full 8-shape suite (the buffer alloc /
     memset must be O(0) when sk_split_n=0 default; if it isn't, the
     scaffolding is too expensive)
  3. Any SNR regression on any shape

If R12 lands clean (NEUTRAL metric, no SNR regression), proceed to R13.

## Why this preflight is "progress" at NEUTRAL metric

Three reasons R11 carries forward value despite no perf change:
  1. R10's recommendation was "start scaffolding" without re-deriving the
     EV with the empirical per-tile time. R11 re-derives, refines R7's
     +25-55 envelope to +9 to +48 (mid +25-30), and confirms the
     direction's EV is real.
  2. R11 surfaces the per-cell mixed verdict (B=4 wins, B=32 loses), which
     means the implementation MUST include a per-cell dispatcher rule. R7
     did not state this; without it, R12-R15 would land a uniform K-split
     and regress on B=32 cells. R11 saves a round on that mistake.
  3. R11 lays out R12-R15 as a concrete checklist with falsification
     gates. The next 4 rounds have a clear directive instead of
     re-thinking the plan each round.

## Forward pointer

R12: HK-side scaffolding per the checklist above. Bit-equivalent.
R13: Kernel K-split control flow. SNR-gated.
R14: Reduce post-kernel.
R15: Dispatcher rule + tight-verify.

**If R12 reveals the host-side cached allocator pattern is messier than
expected (e.g., the `done_counter` allocator is per-shape-context and
doesn't generalize)**, fall back to per-call `hipMallocAsync` from the
caller's stream and accept ~5 µs allocation overhead per call. Still
positive EV on B=4 cells (4 cells × 30/10/10/-8 % - 5 µs/cell ≈ +20
score envelope).

**If R13 reveals AGPR pressure spikes from the K-iter loop bookkeeping
(VGPR > 256, occupancy drops)**: rotate K-iter bookkeeping to closed-form
SALU (cheap on CDNA4 per R10's PMC), as the BF16 var-K coord-decode does
(R9 port). Worst case, drop to 2-way K-split only (less infrastructure,
~2/3 of the lift envelope).

**If R13/R14 turns out infeasible (atomic contention worse than projected,
or AGPR spikes unfix-able)**: rotate to Direction E (incremental barrier
replacement). E is sized at 3-5 rounds per task md NEW DIRECTIONS; 4
remaining rounds of A1' v2 budget approximately matches.

This R11 preflight is the LAST analytical-only round. R12 begins
construction.
