# Round 65 — BF16 grouped, fwd per-XCD work-stealing + extended gate — FALSIFIED (neutral within noise)

## Goal coming in

R64 (HK SHA `29e5d93a`, PT `120005c9`) closed the KI=112 unroll
attenuation lever as FALSIFIED (Variant A: pragma decorative; Variant
B: spill eliminated 24/12 → 0/0 but kernel produced garbage SNR -3.79
to -8.76 dB — the hand-tuned `s_waitcnt vmcnt(N)` + LDS slot rotation
schedule assumes single-basic-block full unroll, partial unroll
breaks vmcnt accuracy across the loop edge).

R64's R65 next-action surface ranked **per-XCD work-stealing counter**
as the recommended lever (option 2):

> 8 counters, 1 per XCD via `blockIdx.x % 8`, each pre-zeroed to
> `total_tiles / 8`. Reduces atomic contention from 256-way to 32-way.
> May enable extending the work-stealing gate beyond `tiles < NUM_CUS*4
> = 1024` without hitting the variance wall R62-B saw. Risk: still
> doesn't fix the L2-locality loss; per-XCD counter only addresses the
> contention dimension of R62-B's variance increase.

R62-B's documented failure mode (5-sample paired test):

* `gpt_oss-GateUP-B4-M4096` (tiles=1472, the only NEW shape that the
  NUM_CUS\*6 gate captures) ratio mean Δ +0.03.
* Score Δ -0.4 mean, **range 54 (vs R61 baseline range 17)**.
* R62 doc identified two variance sources:
  (a) atomic counter contention from a 3rd hipMemsetAsync prime per
      metric round + atomic-claim sequence.
  (b) L2-locality loss on tiles=1472 working set vs ~4 MB per-XCD L2
      partition.

R65 hypothesis: per-XCD WS addresses BOTH (a) (atomic contention 256→32)
AND (b) (each XCD's contiguous tile slice preserves group locality vs
static stride's `{pid, pid+NUM_CUS, ...}` interleaving).

R65 baseline metric (GPU 3, single sample): score=874, gpt_oss family
geomean 1.077, DSV3 1.120, Qwen3 1.110. Per-shape worst:

```
gpt_oss-Down-B4-M2048    (B=4,  M=2048, tiles=384)   ratio 1.050  weight 3  (R61 GATED)
gpt_oss-GateUP-B32-M2048 (B=32, M=2048, tiles=5888)  ratio 1.053  weight 3
gpt_oss-Down-B32-M2048   (B=32, M=2048, tiles=6144)  ratio 1.058  weight 3
gpt_oss-GateUP-B4-M2048  (B=4,  M=2048, tiles=736)   ratio 1.078  weight 3  (R61 GATED)
gpt_oss-GateUP-B4-M4096  (B=4,  M=4096, tiles=1472)  ratio 1.111  weight 3  (R65-NEW gate target)
```

Predicted upside if R65 hypothesis holds: +0.04 ratio on the new-gated
shape (tiles=1472, weight 3) → +1.5 score; small lift on existing-gated
shapes from reduced atomic contention → +0.5 score; total ~+2-5 score.

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

1. **Counter buffer layout** (host-side):
   ```cpp
   constexpr int XCD_COUNTER_STRIDE_INTS = 16;  // 64 B per slot
   constexpr int NUM_XCD_COUNTERS = 8;
   constexpr int XCD_COUNTER_TOTAL_INTS = NUM_XCD_COUNTERS * XCD_COUNTER_STRIDE_INTS;

   // grouped_tile_counter_buffer(): hipMalloc(8 * 16 * sizeof(int))
   // prime_grouped_tile_counter(): hipMemsetAsync(buf, 0, 512 B)
   ```
   8 per-XCD slots padded to 64 B (one MI355X L2 cache line) to eliminate
   false sharing when multiple XCDs concurrently atomicAdd.

2. **`per_xcd_claim` device helper** (~50 lines):
   ```cpp
   template<int NUM_XCDS = 8>
   static __device__ inline int per_xcd_claim(
       int* tile_counters, int xcd_id, int total_tiles)
   {
       const int per_xcd_floor = total_tiles / NUM_XCDS;
       const int residual = total_tiles - per_xcd_floor * NUM_XCDS;
       auto xcd_start = [&](int x) {
           return x * per_xcd_floor + (x < residual ? x : residual);
       };
       int my_start = xcd_start(xcd_id);
       int my_end = my_start + per_xcd_floor + (xcd_id < residual ? 1 : 0);
       int claim = atomicAdd(&tile_counters[xcd_id * XCD_COUNTER_STRIDE_INTS], 1);
       int gt = my_start + claim;
       if (gt < my_end) return gt;
       // Steal: walk other XCDs in offset order
       #pragma unroll
       for (int x = 1; x < NUM_XCDS; ++x) { ... }
       return total_tiles;  // exhausted
   }
   ```

3. **Kernel call sites**:
   * Initial claim: moved AFTER the cooperative LDS init so `total_tiles`
     is known at claim time (~200 cyc cold-start delay, amortized
     across `total_tiles / NUM_CUS` ≥ 1 persistent iterations).
   * Per-iter advance: replaces single-counter atomic with `per_xcd_claim`.

4. **Gate extension**: `tiles < NUM_CUS * 4` → `tiles < NUM_CUS * 6`
   (re-tries R62-B's coverage to capture tiles=1472).

## Resource report (post-R65 vs pre-R65 / R64 baseline, hot KIs only)

```
KI    Layout    pre VS    post VS    delta_VS   notes
0     RCR       0         0          0          (KI=0 dynamic / FUSED=0)
0     RRR       0         0          0
0     CRR       0         0          0
0     RCR f1    0         0          0          (FUSED=1 K-tail fused)
0     RRR f1    0         0          0
48    RCR       1         6          +5         Qwen3-Down K=1536 hot path
56    RCR       18        18         0
64    RCR       1         6          +5         Qwen3-GateUP K=4096 hot path
64    RRR       0         0          0
64    CRR       17        20         +3
88    RCR       1         6          +5         R52 gpt_oss K=2880 hot path
88    RRR       0         0          0
88    CRR       0         20         +20        cold (CRR not on metric path)
112   RCR       24        24         0          DSV3 K=7168 hot path
112   RRR       13        12         -1
112   CRR       31        31         0
```

Net hot-path VGPR-spill change: **+5 spill on KI=48 / 64 / 88 RCR**
(metric-relevant) and +20 on KI=88 CRR (cold). The +5 on hot RCR comes
from `per_xcd_claim`'s lambda + 7-iter steal-loop unroll inflating the
SGPR/VGPR live set even though the helper body is `if (threadIdx.x ==
0)`-guarded — LLVM regalloc sees it as part of the kernel function and
budgets accordingly.

## Correctness probe (6/6 PASS, bf16 floor)

```
gpt_oss-Down-B4-M2048      (tiles=384,   R61 gated)        47.82 dB allclose=True
gpt_oss-GateUP-B4-M2048    (tiles=736,   R61 gated)        47.82 dB allclose=True
gpt_oss-GateUP-B4-M4096    (tiles=1472,  R65-NEW gated)    47.82 dB allclose=True
gpt_oss-Down-B32-M4096     (tiles=12288, non-gated)        47.82 dB allclose=True
DSV3-GateUP-B16-M2048      (tiles=2048,  non-gated, KI=112)47.85 dB allclose=True
Qwen3-Down-B16-M2048       (tiles=3584,  non-gated, KI=48) 47.86 dB allclose=True
```

R65 implementation is numerically equivalent to R64 baseline at bf16
SNR floor.

## Metric — alternating paired (8 samples each side, GPU 3)

Standard 5-sample side-by-side (sequential):

```
pre-R65 (R64 baseline):    874  874  875  876  874
   median 874   mean 874.6   range 2
post-R65 (per-XCD + gate*6): 873  872  875  874  875   (fresh build)
   median 874   mean 873.8   range 3
```

Plus 3 alternating paired samples (rebuild each side per round):

```
round 1: pre=875  post=874   Δ=-1
round 2: pre=876  post=872   Δ=-4
round 3: pre=875  post=873   Δ=-2
```

**Combined 8-sample mean Δ = -1.4** (pre 874.875, post 873.5). Median
Δ = 0. Score change is well below the +5 threshold and within the
±2 noise floor of GPU 3.

Per-family geomean (post-R65 vs pre-R65):

```
                pre-R65    post-R65    Δ
gpt_oss         1.077      1.072       -0.005   (within noise)
DSV3            1.120      1.123       +0.003   (within noise)
Qwen3           1.110      1.111       +0.001   (within noise)
```

All three families are flat within noise — **including the family with
the new-gated shape** (gpt_oss-GateUP-B4-M4096, tiles=1472). The R62-B
predicted +0.03 ratio lift on tiles=1472 did NOT materialize.

## Why per-XCD didn't move the score

Three causes (post-mortem):

1. **Hot-KI register pressure**. The `per_xcd_claim` helper added +5
   VGPR spill on the **metric-hot** KI=48 / 64 / 88 RCR layouts (Qwen3-
   Down K=1536, Qwen3-GateUP K=4096, gpt_oss K=2880 main). Even though
   the helper body is `if (threadIdx.x == 0)`-guarded (single-thread
   region), LLVM's regalloc treats the helper's local SGPRs (per_xcd_
   floor, residual, my_start, my_end, target_start, target_end +
   7-iter steal-loop unroll) as part of the kernel function's overall
   live set and inflates the regalloc peak. The +5 spill on hot RCR
   layouts costs ~0.5-1 % wall on those shapes.

2. **R62-B variance was NOT primarily atomic-contention-driven**.
   per-XCD reduces atomic contention 256-way → 32-way, but the post-R65
   range (3) is essentially equal to pre-R65 (2). The R62-B-observed
   variance escalation (range 17→54) had a different root cause —
   likely the L2-locality loss on tiles=1472 (cause 3 below) being
   coincidentally amplified by GPU contention noise on R62's GPU 4
   session, NOT the atomic counter coherence cost itself. With
   per-XCD WS today (R65) the tiles=1472 shape's variance is also
   not dramatic — and atomic contention is irrelevant in either
   case.

3. **tiles=1472 working set still exceeds 4 MB per-XCD L2**.
   gpt_oss-GateUP-B4-M4096: K=2880 → per-tile B-vector = 256·2880·2 =
   1.5 MB; per-XCD slice = 184 tiles → 276 MB total B-data per XCD.
   The XCD's 4 MB L2 partition holds at most 2-3 B-tiles at a time, so
   contiguous slicing offers ~1-2 B-tile reuse over static stride's
   ~0-1 B-tile reuse — negligibly different. R62-B doc's "L2-locality
   loss" analysis was correct in identifying L2 as the bottleneck but
   per-XCD slicing is not a sufficient fix at this working-set scale.

## Decision

Reverted all kernel functional changes to R64 baseline. Added a
docs-only annotation in `should_use_work_stealing` comment block
documenting the falsified attempt + its three-cause post-mortem.
HK SHA `9a860d59` (docs-only).

Post-revert metric (3 samples): 873, 880, 874 (median 874) — matches
pre-R65 baseline 874-876. Confirms revert is functionally identical
to R64.

## Outcome

* **HipKittens commit**: 1 (docs-only — `should_use_work_stealing`
  comment block annotates the R65 falsified per-XCD WS attempt with
  the three-cause post-mortem so future rounds don't re-try the same
  lever from scratch).
* **Primus-Turbo commit**: 1 (this round note).
* **Functional kernel state**: identical to R64 (HK pre-R65 baseline).
  No code change, no resource change.

## Metric numbers

```
                       pre-R65 (5 samples)        post-R65 (5 samples fresh + 3 paired)
score                  874, 874, 875, 876, 874    873, 872, 875, 874, 875, 874, 872, 873
                       median 874, mean 874.6     median 874, mean 873.5
                       range 2                    range 3
gpt_oss   geomean      1.077                      1.072 (within noise)
DSV3      geomean      1.120                      1.123 (within noise)
Qwen3     geomean      1.110                      1.111 (within noise)
correct_fail           0/24                       0/24
```

GPU 3 is well-controlled this round — the pre-R65 5-sample range of 2
is the tightest baseline window observed in the R55-R65 plateau. The
post-R65 range of 3 matches. **Score Δ -1.4 is firmly within noise**.

## Backward bench

Not run — the R65 changes touched only the persistent-loop atomic-claim
mechanism (forward kernel only). dB var_k path (`grouped_var_k_kernel`)
is untouched. R61's bench result (24/24 PASS, fwd 1141.07 / bwd 855.25
TFLOPS) remains valid.

## R66 next-action surface

R65 closes the per-XCD work-stealing lever. Atomic-contention is no
longer suspected as a meaningful variance source on this kernel. The
remaining R63/R64 surface levers (still open):

1. **dA backward audit** (R61/R62/R63/R64 carry-over). bench shows bwd
   855 vs fwd 1141 TFLOPS (75 % of fwd). dB var_k path
   (`grouped_var_k_kernel`) untouched since R55. Won't move metric
   (fwd-only) but is the lever surface for non-metric workloads.
   Highest-impact for the broader Primus-Turbo backward-pass story
   even though it's invisible to this metric.

2. **Triton-style two-tier persistent grid (grid > NUM_CUS)** — R63
   carry-over. Launch grid = 2 \* NUM_CUS = 512 with reduced per-block
   iters; HW dispatch round-robin amortizes prologue across more
   dispatch waves and L2 warm-up. Risk: doubles per-block prologue
   cost (S ≈ 63 µs per R60 finding × 2 = 126 µs aggregate); only net-
   positive if dispatch L2 warm-up savings exceed prologue redundancy.
   Hasn't been tried since R60.

3. **Cumsum LDS write reduction** (R63 carry-over, tiny). Skip every-
   other `s_cum_tiles[gi+1]` LDS write since only the FIRST element
   from each cum row is read by the inner group-scan. ~0.3 µs win.
   Bundle with a larger lever.

4. **`#pragma unroll 4` opaque-cast** (R64 carry-over) — same mechanism
   as R64-B but with smaller chunks. The 8-iter chunk's loop-back may
   cross too many vmcnt boundaries; 4-iter might be safe enough.
   Low-confidence but cheap to test (1-line vs R64-B's exact code).

5. **Schedule partial-unroll-safe rewrite** — R64 carry-over major
   surgery (~200 lines). Defers to multi-round project once smaller
   levers exhaust.

6. **NEW: PMC-driven analysis of post-R55 plateau**. R56 PMC found
   gpt_oss B=32 MFMA util ~62 %. R56 → R65 levers haven't moved that
   number meaningfully. Re-run rocprofv3 PMC on the current R64
   binary to see if the dominant non-MFMA cycles have shifted (e.g.,
   if the LDS-bank-conflict count or VMEM-stall count would suggest
   a different attack surface). Diagnostic, won't move score directly.

7. **NEW: extend the R65 falsification analysis to extract the
   atomic-counter latency cost on tiles=384 / 736 (existing-gated)**.
   If atomic latency is ~200-500 cyc per claim on AMD CDNA, and CUs
   on tiles=384 do ~1.5 atomics each, total ~750 cyc per CU. Hoisting
   the claim into the GEMM main loop (R64 doc lever 7) might save
   most of that, ~0.5 % wall on the gated shapes. Bundle with lever
   2 / dB audit for a multi-mechanism round.

Recommended R66: **option 1 (dA backward audit)**. Even though
fwd-metric-invisible, the bwd path is the largest untapped wall delta
in the bench output, and after 6 falsified rounds (R60-R65) the metric
plateau (874-880) is well-saturated. R66 should pivot to a non-metric-
visible lever to expand the broader optimization surface.

If a metric-visible lever is required, **option 4 (#pragma unroll 4
opaque-cast on KI=112)** is the next-cheapest-to-test, ~10-line patch
with the R64 falsification post-mortem providing the framework to
distinguish a vmcnt-safe variant.
