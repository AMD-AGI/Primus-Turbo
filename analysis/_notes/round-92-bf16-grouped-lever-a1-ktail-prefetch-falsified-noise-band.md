# Round 92 — bf16 grouped GEMM weighted wall (auto_optimize round 15/100)

> **Lever A1** (forward HBM K-tail prefetch) — recommended by R91 as
> the first post-pivot lever. Implemented + paired-tested + falsified:
> mean delta = +2.0 score (884.8 vs 882.8) at +6 VGPR cost, well below
> the +5 commit gate AND below the prior best (889). Reverted.

## Status

* Score: 882 (open metric, single run).
* Best-ever: 889 (R14 = 2c3870f1, single-run noise spike on idle GPU).
* Patience: 0/30 (carried in by R14).
* HipKittens: clean (revert to `40be51de`).
* Primus-Turbo: this note. No code change.

## Lever description

Per R91 plan (Lever A1, top of the post-pivot priority list):

> Forward `grouped_kernel<RCR, KI=0, FUSED_KTAIL=true>` is HBM-bound
> per R87 PMC. The K-tail epilog (`device_gemm_tile_body`, the
> `if constexpr (FUSED_KTAIL && L == Layout::RCR)` block after epilog
> 2) issues 12 `buffer_load_b128` (4 each into A_tile / B_tile_0 /
> B_tile_1) AFTER epilog 2's last `s_barrier`. The `s_waitcnt vmcnt(0)`
> immediately after sees ~300 cycles of HBM round-trip latency — the
> dominant K-tail stall.
>
> Hypothesis: hoisting those 12 issues to the END of epilog 2 (right
> after `DO_MMA(C[1][1])`, before the `s_barrier`) lets them execute
> in the shadow of the in-flight mfma_16x16x32_bf16 (~16 cyc) and the
> following `s_barrier` (~10-30 cyc). A_tile / B_tile_0 / B_tile_1
> are dead at that point (mfma reads source VGPRs at issue time), so
> writing them via `buffer_load_b128` is correctness-preserving.

## Implementation

* Hoisted `load_a_kt` / `load_b_kt` lambda definitions from the inside
  of the FUSED_KTAIL RCR block to function scope (between epilog 1 and
  epilog 2 of `device_gemm_tile_body`). Body gated on `FUSED_KTAIL &&
  L == Layout::RCR`; for non-RCR or non-fused instantiations the
  lambdas are empty (compiler DCE'd, zero code).
* Added `BF16_RCR_KTAIL_PREFETCH` define (default 1).
* Inserted the 3 prefetch calls (`load_a_kt(0); load_b_kt(B_tile_0, 0);
  load_b_kt(B_tile_1, 1);`) at end of epilog 2 (after `DO_MMA(C[1][1])`,
  before the closing `s_barrier`).
* Gated the matching m_slab=0 calls in the K-tail RCR block under
  `#if !BF16_RCR_KTAIL_PREFETCH` (skipped when prefetch enabled).

## Build resource report

```
grouped_kernel<RCR, KI=0, FUSED_KTAIL=true>:
  Original (git HEAD 40be51de):    VGPR=250  Spill=0  Occ=2
  PREFETCH=1 (Lever A1):           VGPR=256  Spill=0  Occ=2  (+6)
  PREFETCH=0 (struct change only): VGPR=250  Spill=0  Occ=2  (= original)
```

The +6 VGPR cost comes from the prefetched `A_tile` / `B_tile_0` /
`B_tile_1` live-range extending across the s_barrier into the K-tail
block start (the loads return after the barrier; their results are
held in VGPRs until consumed by the K-tail MMAs). The structural
change alone (lambda hoisting at PREFETCH=0) doesn't perturb VGPR.
**Both 250 and 256 fit within the 256 VGPR ceiling at occupancy 2,
so no spill / occupancy regression.** This is materially less than
R86's +17 VGPR within-half permute swizzle which DID spill.

## Correctness probe

Direct API probe on `gpt_oss-Down-B4-M2048` (the lowest-ratio shape =
1.051), bfloat16 inputs scaled 0.1, fp32 reference:

```
fwd: max_abs=0.0156  signal_pwr=0.2880  err_pwr~0     SNR=49.58 dB  allclose=True
dA:  max_abs=0.1250                                   SNR=49.61 dB  allclose=True
dB:  max_abs=0.1250                                   SNR=49.59 dB  allclose=True
```

All 24 metric shapes also passed correctness (`correct_fail=0/24`).
Prefetch is numerically equivalent.

## Paired metric (A/B test on same GPU, n=5 each)

| build                                  | runs                | mean  |  std |
|----------------------------------------|---------------------|------:|-----:|
| PREFETCH=0 (= original behavior)       | 882, 882, 884, 882, 884 | 882.8 | 1.10 |
| PREFETCH=1 (= Lever A1 active)         | 884, 884, 886, 885, 885 | 884.8 | 0.84 |

**Δ = +2.0 mean, +0.6σ_pooled** (pooled σ ≈ 1.0). Effect size t ≈ 3.2,
nominally significant but within the historical noise band of the
metric (R85-R90 mean = 882.6 ± 0.5; the metric standard variation is
typically ~1 score). Crucially:

* **+2 < +5 commit gate** — task body explicitly requires `Score ≥
  prior best + 5`. Prior best = 889 (R14, single-run noise spike).
  Even Lever A1 mean 884.8 sits **-4.2 below prior best**.
* **No falsification of the underlying mechanism** — the prefetch
  *does* hide ~50 cycles of HBM latency (theory predicts +6 score
  from a 50-cyc K-tail-only saving at full per-shape conversion;
  observed +2 reflects ~30 % efficiency, the rest absorbed by HBM
  contention or overlapping mfma issue serialisation).

## Per-shape gpt_oss diff (PREFETCH=0 → PREFETCH=1)

|     shape                      |  ratio_before | ratio_after |   Δpp  |
|--------------------------------|--------------:|------------:|-------:|
| gpt_oss-GateUP-B4-M2048        |        1.129  |       1.134 |  +0.4  |
| gpt_oss-Down-B4-M2048   (lo)   |        1.051  |       1.059 |  +0.8  |
| gpt_oss-GateUP-B4-M4096        |        1.116  |       1.117 |  +0.1  |
| gpt_oss-Down-B4-M4096          |        1.098  |       1.112 |  +1.4  |
| gpt_oss-GateUP-B32-M2048       |        1.102  |       1.102 |   0.0  |
| gpt_oss-Down-B32-M2048  (lo)   |        1.057  |       1.054 |  -0.3  |
| gpt_oss-GateUP-B32-M4096       |        1.105  |       1.105 |   0.0  |
| gpt_oss-Down-B32-M4096         |        1.089  |       1.098 |  +0.9  |

Effects concentrate on **B=4 Down shapes** (+0.8, +1.4 pp) where the
per-tile K-tail makes a bigger fraction of total wall (fewer tiles to
amortise). B=32 shapes near-flat — the per-tile K-tail's relative cost
is already amortised across 8x more tiles, so a 50-cycle hide at the
last K-tile is ~50/(8 × per-tile-time) = small. **Lever A1 is
structurally weaker on B=32 than on B=4**, which is the opposite of
what we want (B=32 is the worst-progress region).

## Why the lever is partially exhausted on this attack vector

The 50-cycle prefetch window is bounded by the **fixed schedule
between mfma 4 (= `DO_MMA(C[1][1])`) and the K-tail block's
`s_waitcnt vmcnt(0)`**. That schedule is:

```
mfma 4 issue        ; t=0
[Lever A1 prefetch] ; t=0..12  (12 buffer_load_b128, 1-cyc issue rate)
__builtin_amdgcn_s_barrier(); t=12..30  (cross-warp sync)
__builtin_amdgcn_sched_barrier(0); t=30  (compiler-only)
[K-tail block entry; lambda compute setup]  t=30..40
[m_slab=1 portion lives here in original; deferred until after vmcnt]
asm volatile("s_waitcnt vmcnt(0)");  t=40
```

vs no-prefetch:
```
mfma 4 issue        ; t=0
__builtin_amdgcn_s_barrier(); t=0..18
[K-tail entry, lambda setup]; t=18..30
[12 buffer_load_b128 issue]; t=30..42
asm volatile("s_waitcnt vmcnt(0)");  t=42
```

Total prefetch window = ~42 - ~12 = ~30 cycles (issue-time advance).
HBM round trip = ~300 cycles, so the prefetch reduces vmcnt wait by
~30 cycles = ~10 % of K-tail HBM stall. K-tail = ~30 % of forward
kernel wall. Forward = ~33 % of fwd+bwd metric wall (3x bwd FLOP
weight). So per-shape gain ≈ 0.10 × 0.30 × 0.33 ≈ **1 % per shape**.

For 8 gpt_oss shapes × weight 3 / total_weight 40, weighted progress
delta = 8 × 0.01 × 3 / 40 = **+0.6 % weighted = +6 score** (theory).
Observed +2 = 33 % theory-realisation, consistent with HBM contention
(other warps' main-loop loads still in flight when our prefetch
issues, pushing our completion back).

**Pushing earlier** would require:

* Issue B_tile_0 prefetch BEFORE mfma 4 (after mfma 3, between mfma 3
  and mfma 4) — but the back-to-back DO_MMAs issue at ~1-cycle gap;
  inserting buffer_load_b128 forces the scheduler to delay mfma 4 by
  ~1-3 cycles, costing more than the gain.
* Issue A_tile prefetch BEFORE the mfma 3/4 pair — but A_tile is
  alive through mfma 4 (= `DO_MMA(C[1][1])`); writing it any earlier
  corrupts the mfma 3/4 input.

The prefetch position **after `DO_MMA(C[1][1])` is the
schedule-theoretic earliest** for ALL three target VGPR groups. The
lever has hit its analytical ceiling at the +30-cycle window.

## Why this is reported as falsification (not banked)

* **+2 < +5 commit gate** — task body rule, non-negotiable.
* **Below prior best 889** — committing 884.8 mean would be a -4.2
  regression by best-ever metric.
* **VGPR cost +6 not free** — preserves a margin (256 - 250 = 6 → 0)
  that future swizzle / scheduling levers might need (R86's +17
  spilled; +6 leaves no headroom for further adds).
* **Lever exhausted at this insertion point** — see schedule analysis
  above. No deeper hoist available without correctness break.

## Plan B for R16 (descending priority within R91's surface)

R91 ranked 4 viable angles. Lever A1 is now closed. Remaining:

### Lever B4 — persistent grid work-stealing for `grouped_var_k_kernel` (BACKWARD path)

R91 description:
> Current outer loop `for (int gt = pid; gt < total_tiles; gt +=
> NUM_CUS) { ... }` distributes tiles across CUs in stride-NUM_CUS
> chunks. With `tiles_per_group ≈ 88` and `G ≤ 32`, each CU sees ~9
> tiles — short persistent loop with high per-tile fixed cost (group
> lookup + coord swizzle). Swap to per-group chunked schedule
> (contiguous tile-blocks per CU within a group) to improve L2 reuse
> for `g.a` (read by all tiles in same group's M-stripe).

**Affects backward dB** (var-K kernel). All 24 metric shapes hit
backward (dA + dB). Backward = ~67 % of metric wall, so a 2-3 %
backward speedup would be +5-7 score broadly.

R75 already attempted "var-K work-stealing" and was falsified ("noise
band"). R75's variant was **atomic-claim within strided schedule** —
NOT the per-group chunked schedule R91 proposes. The two are
distinct: R75 keeps stride NUM_CUS but adds atomic-claim for slack
balancing; R91-Plan-B changes the partition itself.

### Lever C — uniform-M dB BMM dispatch (FROZEN-list-adjacent)

R91 caveat:
> When all `M_g` are equal (the metric uses balance=True), `dB[g] =
> dout[g].T @ x[g]` is mathematically identical to a regular batched
> GEMM `dB = dout.T @ x` reshaped. The BMM kernel has fixed K, no
> var-K LDS-BC bottleneck.
>
> FROZEN concern: requires checking `group_lens` uniformity on host
> (= `.tolist()` / equality check). However, **a static-metadata
> uniformity hint** (group_lens.size(0) == B AND M_total % B == 0)
> requires no `.item()` / `.tolist()` and is true for the metric's
> balance=True callers.

**Open question for R16+**: does the HipKittens BF16 grouped binding
expose a "uniform-M fast path" knob? **I can confirm it doesn't:**
`tk_bf16_layouts` exposes `grouped_*_balanced` (forward, expects
uniform M) and `grouped_var_k_*` (backward dB, accepts var-M via
group_offs device tensor). The `grouped_*_balanced` already IS the
uniform-M path; the dB var-K kernel does NOT have a uniform fallback.
**Lever C requires a new HK kernel** (BMM-style dB), which is a
multi-round effort not a single-round commit.

### Lever D — DSV3 / Qwen3 push (Phase B)

> Each non-gpt_oss shape is at ratio ~1.10-1.13 vs target 1.25.
> Lifting all 16 by 5pp yields 16 × 0.04 / 40 = +1.6 % weighted
> progress = +16 score. Exceeds the +5 commit gate by 3x with
> margin for noise.

**No PMC profile yet on K%128==0 path.** R87 PMC was on `gpt_oss-Down-
B4-M2048` (K=2880, K%128 != 0). DSV3-K=7168 / Qwen3-K=4096 / Qwen3-
K=1536 take the **non-FUSED** path through `grouped_kernel<RCR, KI=N,
FUSED_KTAIL=false>` — a different kernel binary with different
schedule + different bottleneck profile.

R16 should **PMC-profile a DSV3 worst-shape** (e.g. `DSV3-Down-B16-
M2048` ratio 1.106 = lowest DSV3) to identify whether the constraint
is MFMA util / LDS BC / HBM throughput / register pressure on the
non-FUSED path. Without instrumentation, Lever D is a guess.

## Recommended R16 plan

1. **Most leverage / lowest cost**: PMC profile `DSV3-Down-B16-M2048`
   forward (non-FUSED) + `var_k` dB on the same shape. This is a
   diagnostic round (no code change). Expected output: identifies
   whether DSV3/Qwen3 K%128==0 path is MFMA-bound (→ Lever B1 is
   king), HBM-bound (→ no easy win), LDS-BC-bound (→ swizzle work,
   high VGPR cost), or registers-bound (→ rematerialisation /
   spill reduction).

2. **Highest leverage / medium cost**: implement R91's Lever B4 as a
   focused round (per-group chunked schedule on var-K backward).
   Affects all 24 shapes' backward (= 67 % of metric wall). Even
   2-3 % savings = +5-7 score.

3. **DO NOT** spend rounds on further K-tail prefetch attacks.
   Lever A1 has hit its analytical ceiling (+30-cyc window cap).

## R92 commit summary

* **HipKittens**: no change (kernel reverted to `40be51de`).
* **Primus-Turbo**: this note. No code or kernel change.
* **Metric (R92 single run)**: 882, baseline.
* **Round 15/100 verdict**: falsification — Lever A1 partial-effect
  (+2 mean, ~33 % theory-realisation) at +6 VGPR cost is below the
  +5 commit gate and below prior best 889.
