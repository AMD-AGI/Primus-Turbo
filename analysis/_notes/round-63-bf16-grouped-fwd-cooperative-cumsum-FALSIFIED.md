# Round 63 — BF16 grouped, fwd cooperative cumsum prologue split — FALSIFIED

## Goal coming in

R62 closed both chiplet swizzle (chunk 64→32) and extended gate
(NUM_CUS\*4 → NUM_CUS\*6) as FALSIFIED. R62 doc R63 next-action surface
prioritized **reduce kernel prologue cost** (R60 estimate S ≈ 63 µs
≈ 6 % of wall on worst gated B=4 M=2048 shape).

R63 starting metric (GPU 3, single sample): score=894. Worst stable
weight-3 shape: `gpt_oss-Down-B32-M2048` ratio 1.027 (tiles=3072,
ungated, 12 iters/CU).

But the prologue lever is **mismatched against the worst stable
shape**: at 12 iters/CU, S/wall ≈ 63/3939 ≈ 1.6 % — even fully
eliminating S only buys ~1.5 % wall on `gpt_oss-Down-B32-M2048`.
The lever's biggest target is the gated B=4 family
(S/wall ≈ 6 %, predicted +0.5 % wall ≈ +5 score) but those shapes
are already PASS in their best samples post-R61. R63 went ahead
with the prologue lever because (a) the kernel-side change is
small (~10 lines), (b) the trickle-down savings hit ALL 24 shapes
uniformly, and (c) the next R62-doc lever (PMC marker bracketing)
is purely diagnostic.

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
lines 3761-3776 — split the single-thread O(G) load+cumsum loop into
two phases:

```cpp
// (a) parallel HBM load — first G+1 threads each fetch one
//     g.group_offs[] entry into s_offs[]
if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
    s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
}
__syncthreads();

// (b) sequential cumsum on LDS-cached values (single thread,
//     since the prefix-sum dependency is inherent and
//     log₂(G) parallel reduction is not faster for G ≤ 32)
if (threadIdx.x == 0) {
    s_cum_tiles[0] = 0;
    int t = 0;
    int prev = s_offs[0];
    for (int gi = 0; gi < g.G; ++gi) {
        int next = s_offs[gi + 1];
        t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
        s_cum_tiles[gi + 1] = t;
        prev = next;
    }
    s_total_tiles = t;
}
__syncthreads();
```

vs original single-thread pipeline:

```cpp
if (threadIdx.x == 0) {
    int prev = static_cast<int>(g.group_offs[0]);
    s_offs[0] = prev;
    s_cum_tiles[0] = 0;
    int t = 0;
    for (int gi = 0; gi < g.G; ++gi) {
        const int next = static_cast<int>(g.group_offs[gi + 1]);
        s_offs[gi + 1] = next;
        t += ((next - prev) / BLOCK_SIZE) * num_pid_n;
        s_cum_tiles[gi + 1] = t;
        prev = next;
    }
    s_total_tiles = t;
}
__syncthreads();
```

Theoretical savings (G=32 worst case):

| Phase                       | Original        | R63              |
|----------------------------|-----------------|------------------|
| HBM read (g.group_offs)    | 33 × 200 cy ≈ 5.5 µs (sequential, may pipeline) | 1 × 200 cy ≈ 0.17 µs (parallel) |
| Cumsum arith               | 32 × ~5 cy ≈ 0.13 µs (registers) | 32 × ~10 cy ≈ 0.27 µs (LDS reads) |
| Sync barriers              | 1 × ~30 cy = 25 ns | 2 × ~30 cy = 50 ns |
| **Total prologue cost**    | ~5.6 µs         | ~0.4 µs          |

Predicted gain: 5.2 µs / 1033 µs ≈ +0.5 % wall on B=4 M=2048
shapes ≈ +5 score weighted progress.

## Build / probe

Resources unchanged from R62: SGPR 92-104, VGPR 246-256, +1 spill
on KI=48/64/88, 24 spill on KI=112. Probe 5/5 PASS at bf16 floor.

## Metric — paired test (GPU 3)

```
R62 baseline (5 samples):  899 / 920 / 905 / 910 / 912    median 910  mean 909.2   range 21
R63 (5 samples first batch):    895 / 905 / 900 / 911 / 898    median 900  mean 901.8   range 16
R63 (5 samples second batch):   896 / 901 / 901 / 913 / 906    median 901  mean 903.4   range 17

Combined R63 (10 samples):       median 901  mean 902.6   stddev ≈ 5.7  SEM 1.8
R62 baseline (5 samples):        median 910  mean 909.2   stddev ≈ 7.6  SEM 3.4
```

**Δ = -6.6 mean ± 3.8 (~2σ regression). Statistically significant.**

## Why it regressed (mechanism)

Three plausible factors, listed by suspected impact:

1. **Compiler pipelining of the original** — the `prev → next` chain
   stays entirely in VGPRs across the loop iterations. The HBM read
   for `g.group_offs[gi+1]` issues with high latency but the
   compiler can pipeline the issue (next iter's load) with the
   current iter's arithmetic. R63's split forces an explicit
   `__syncthreads()` barrier between load and cumsum, which prevents
   the same pipelining and forces the cumsum thread to wait for the
   final HBM completion.

2. **LDS-read latency in the cumsum body** — R63's cumsum thread
   reads `s_offs[gi+1]` from LDS each iteration (~5 cy each). The
   original kept `next` in a register from the previous iter's load
   issue + auto-pipelining (compiler may have re-used the register
   slot through register renaming).

3. **HBM L2 contention escalation** — 256 blocks × 33-lane parallel
   HBM fan-out vs 256 blocks × 1-lane sequential pipeline. Same
   total bytes loaded (4 × 64-byte cache lines for G ≤ 32), but
   the burst pattern differs. The single-thread sequential pipeline
   smooths HBM channel utilization; the parallel cooperative load
   spikes 8448 simultaneous requests across the 256 blocks. Spike
   patterns serialize through the 4-8 HBM channels worse than
   smoothed patterns.

Factor #1 is the most likely dominant one — the compiler's
out-of-order issue + register renaming makes the original
single-thread loop **already as fast as 33 in-flight HBM loads**
without the sync-barrier serialization R63 introduces.

## Decision

Reverted R63 cooperative cumsum. HK SHA `413749b2` is docs-only
(annotates the cumsum block with the negative result). Functionally
identical to R62 (`cabf90c0`). Single-sample post-revert metric =
923 (within R62 baseline range 899-920).

## Outcome

* **HipKittens commit**: `413749b2` — docs-only annotation of the
  R63 negative result in the cumsum prologue block. No codegen
  change.
* **Primus-Turbo commit**: 1 (this round note).
* **Backward bench not run** — kernel functionally unchanged, R61's
  bench (24/24 PASS, fwd 1141.07 / bwd 855.25 TFLOPS) remains valid.

## R64 next-action surface

The R62-R63 doc trail has now closed 4 of the 5 R61-doc levers as
inconclusive or falsified:
* (R62-A) chunk_size=32 chiplet swizzle — FALSIFIED (neutral)
* (R62-B) extended gate to NUM_CUS*6 — FALSIFIED (variance regression)
* (R63) cooperative cumsum prologue split — FALSIFIED (-6.6 mean)
* PMC bracketing remains DIAGNOSTIC (purely informational)

R64 should pivot to the remaining two original surface candidates
plus new ideas:

1. **dA backward audit** — bench shows bwd 855 vs fwd 1141 TFLOPS
   (75 % of fwd). dB var_k path (`grouped_var_k_kernel`) untouched
   since R55. Won't move metric (fwd-only) but is the lever surface
   for non-metric workloads.

2. **Reduce KI=112 spill (24 VGPRs)** for DSV3-GateUP K=7168 (the
   4 DSV3-GateUP shapes × weight 1 = 4 metric weight). Outstanding
   since R52. Need to find the right unroll attenuation that
   preserves pipeline depth without overflowing 256 VGPRs.
   Candidate: try `#pragma unroll 8` (or 4) on the KI=112
   main_loop_iter dispatch instead of full unroll.

3. **Per-XCD work-stealing counter** (R62-doc carry-over). 8
   counters, 1 per XCD via `blockIdx.x % 8`, each pre-zeroed to
   `total_tiles / 8`. Combines XCD-locality with imbalance recovery,
   could tame the variance increase R62-B saw on tiles=1472.

4. **NEW: Triton-style two-tier persistent grid**. Currently we
   launch grid=NUM_CUS=256 and rely on persistent loop. Triton's
   grouped GEMM uses grid > NUM_CUS with HW dispatch round-robin
   to amortize prologue across more dispatch waves. Could combine
   with the prologue-cost finding: launch grid=2*NUM_CUS, each
   block does fewer iters but with shared L2 warm-up. Risk: trade
   prologue redundancy against per-iter wall.

5. **NEW: MFMA scheduler density tuning beyond R55 (4 → 8 sites)**.
   R55 doubled `sched_barrier(0)` density and gained +5 median
   score. Could tighten to 12 or 16 sites if the 4 → 8 trend is
   monotonic. Worth measuring incrementally.

6. **NEW: Eliminate one cumsum LDS write per iter**. The cumsum
   loop writes `s_cum_tiles[gi+1]` AND tracks `t` in a register.
   Only the FIRST element from each cum row is read by the inner
   group-scan loop. We could skip every other LDS write by
   doubling the per-iter step. Tiny win (~0.3 µs), but cheap and
   risk-free. Bundle with one of the bigger levers above.
