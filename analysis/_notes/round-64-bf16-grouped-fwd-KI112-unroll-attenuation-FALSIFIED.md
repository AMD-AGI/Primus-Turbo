# Round 64 — BF16 grouped, fwd KI=112 unroll attenuation — FALSIFIED (correctness break)

## Goal coming in

R63 (HK SHA `413749b2`, PT `f027b5b5`) closed the cooperative-cumsum
prologue split as FALSIFIED (-6.6 mean, ~2σ regression). R63's
R64 next-action surface listed lever #2:

> **Reduce KI=112 spill (24 VGPRs)** for DSV3-GateUP K=7168 (the 4
> DSV3-GateUP shapes × weight 1 = 4 metric weight). Outstanding since
> R52. Need to find the right unroll attenuation that preserves
> pipeline depth without overflowing 256 VGPRs. Candidate: try
> `#pragma unroll 8` (or 4) on the KI=112 main_loop_iter dispatch
> instead of full unroll.

R64 starting metric (GPU 3, single sample): score=901, gpt_oss family
geomean 1.109, DSV3 1.123, Qwen3 1.186. Per-shape worst:

```
gpt_oss-Down-B32-M2048   (B=32, M=2048, N=2880, K=2880)  ratio 1.037  weight 3  progress 0.829
gpt_oss-GateUP-B32-M2048 (B=32, M=2048, N=5760, K=2880)  ratio 1.047  weight 3  progress 0.838
DSV3-GateUP-B16-M2048    (B=16, M=2048, N=4096, K=7168)  ratio 1.138  weight 1  progress 0.910 (KI=112)
DSV3-GateUP-B16-M4096    (B=16, M=4096, N=4096, K=7168)  ratio 1.143  weight 1  progress 0.914 (KI=112)
DSV3-GateUP-B32-M2048    (B=32, M=2048, N=4096, K=7168)  ratio 1.129  weight 1  progress 0.904 (KI=112)
DSV3-GateUP-B32-M4096    (B=32, M=4096, N=4096, K=7168)  ratio 1.141  weight 1  progress 0.913 (KI=112)
```

R15 (sub-experiment 2) closed the gpt_oss K=2880 kernel-side
optimization surface — the FUSED=true / KI=0 dynamic-K path is at a
local optimum. Further gpt_oss B=32 lift requires structural
architectural changes outside per-round scope. R64 pivots to Phase B
DSV3 lever target: KI=112 VGPR spill (24 VGPRs RCR / 12 RRR
post-R55) is the largest non-cold spill in the grouped_kernel
templates and affects 4 weight-1 metric shapes.

Predicted upside if KI=112 spills eliminated: ~+2-4% per-shape on the 4
DSV3-GateUP shapes; weight 1 each / total weight 40 → +3-5 score.

## Hypothesis

The 24 VGPR spill at `grouped_kernel<RCR, KI=112, FUSED=false>` comes
from LLVM's full-unroll of the main loop (55 inlined main_loop_iter
copies × ~85 lines = ~4760 lines per kernel). The compiler's auto-
heuristic register allocator runs out of budget at this scale and
spills 24 VGPRs to scratch.

R63 finding chain inferred: an attenuated `#pragma unroll 8` (8 inlined
copies per outer iter, 7 outer iters for KI=112) should fit comfortably
under the 256 VGPR ceiling while preserving cross-iter MFMA
scheduling depth (8 iters × 8 MMAs = 64 MMAs visible to the scheduler
window).

## Variant A — constexpr num_tiles + `#pragma unroll 8` (FALSIFIED, decorative)

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
(NOT committed):

```cpp
} else if constexpr (KI_HINT >= 112) {
    constexpr int num_tiles = KI_HINT;
    #pragma unroll 8
    for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
}
```

Resource report (KI=112):

```
                  pre-R64    R64-A
RCR  VGPRs:        256        256       (unchanged)
RCR  VGPRs Spill:   24         24       (unchanged)
RCR  Scratch:      136        100
RRR  VGPRs:        256        256       (unchanged)
RRR  VGPRs Spill:   12         13       (~ noise)
```

**`#pragma unroll N` is decorative** at KI_HINT >= 112 — confirming
R15's earlier finding (R15 sub-experiment 1 found pragma was
decorative at KI=44, identical 28-VGPR spill at unroll 2/4/full). The
constexpr `num_tiles = KI_HINT` triggers LLVM's auto-full-unroll
regardless of the `#pragma unroll N` hint. Variant A failed the
resource gate (no spill reduction).

## Variant B — opaque-cast num_tiles + `#pragma unroll 8` (FALSIFIED, breaks correctness)

`kernel_bf16_dynamic.cpp` (NOT committed):

```cpp
} else if constexpr (KI_HINT >= 112) {
    int num_tiles = KI_HINT;
    asm volatile("" : "+s"(num_tiles));   // force runtime SGPR
    #pragma unroll 8
    for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
}
```

The asm `"+s"` cast forces num_tiles into a scalar register at runtime,
opaque to LLVM constant propagation. This pattern is used elsewhere in
the kernel (lines 150, 158, 171) for SOFF / lds_byte addresses — known
to work.

### Resource report — exactly the predicted reduction

```
                  pre-R64    R64-B
RCR  VGPRs:        256        246       (-10)
RCR  VGPRs Spill:   24          0       (-24, ELIMINATED)
RCR  Scratch:      136          0
RRR  VGPRs:        256        246       (-10)
RRR  VGPRs Spill:   12          0       (-12, ELIMINATED)
RRR  Scratch:        ?          0
```

Build resource gate: **PASSED with margin**. KI=112 spill went from
24/12 (RCR/RRR) → 0/0; VGPR allocation also dropped 256→246. KI=64
unchanged at 256/1. KI=128, 172, 224, 256, 296, 448, 462, 832 (cold
metric specs) all see similar spill elimination — same partial-unroll
mechanism applies whenever KI_HINT >= 112.

### Correctness — FAILED catastrophically

`/tmp/probe_r55_correctness.py` (5 representative shapes covering all 3
families, identical script to R55-R63):

```
                                        SNR     allclose
Qwen3-Down-B16-M2048(KI48-RCR)         -5.37    False
Qwen3-Down-B32-M4096(KI48-RCR)         -3.79    False
gpt_oss-Down-B4-M2048(K2880)           -8.76    False
DSV3-GateUP-B32-M2048(KI112)           -3.79    False
Qwen3-GateUP-B16-M2048(KI64)           -6.50    False
```

**SNR -3.79 to -8.76 dB across ALL probes** — output is garbage, not
just numerically drifted. The variant breaks correctness of the entire
forward kernel (including KI specs that don't go through the new
branch — KI=64 / KI=48 / KI=0 dynamic). This was confirmed by
reverting the change and re-running R55's probe — same 5 shapes
produced 47.83-47.86 dB SNR (bf16-rounding floor) with the reverted
kernel.

### Why partial-unroll breaks the kernel (most likely explanation)

The hand-tuned schedule (lines 600-690) of `main_loop_iter` relies on a
**single-basic-block full-unroll** assumption. Specifically:

1. `s_waitcnt vmcnt(N)` instructions at lines 638, 682 with HARDCODED
   counter values (`vmcnt(6)` after a known-in-flight buffer_load
   pattern) only work when the compiler can statically count VMEM
   issues across all main_loop_iter copies. With partial-unroll's
   loop-back, instructions after the loop edge see a fresh "in-flight"
   count, and `vmcnt(6)` waits for the wrong number of pending loads.

2. LDS slot rotation pattern: each main_loop_iter writes the next-
   iter's data into Bs[0..1][0..1] / As[0..1][0..1] slots; the FOLLOWING
   iter consumes them. Within a single fully-unrolled basic block, LLVM
   can interleave / re-order across iter boundaries while preserving
   semantic correctness via the implicit data flow. With partial-unroll,
   the loop-back creates a scheduler wall — the data flow across the
   wall must obey strict program order, which breaks the interleaved
   prefetch/MMA pattern. The result is that some prefetches are issued
   too late and the consumer reads stale or uninitialized LDS slots.

3. R15's sub-experiment 2 (KI=0 + `#pragma unroll 4` for FUSED_KTAIL
   path) ran correctly but slower. That path uses **runtime
   num_tiles_dyn**, so partial-unroll's loop-back was already part of
   the assumption. R64-B forces partial-unroll on a path that was
   designed (R52-R55-R61 hand-tuning) for full-unroll — incompatible.

### Decision

Reverted variant B. The 4-line `if constexpr` branch is dropped from
HK; the path returns to its baseline `constexpr num_tiles + #pragma
unroll` form for KI_HINT > 0 / non-CRR. HK SHA `<TBD>` is **docs-only**
(annotates the falsified attempts inline at the dispatch site). 

R55 probe post-revert: 47.83-47.86 dB SNR on all 5 probe shapes
(matches pre-R64 baseline).

### Bound this places on KI=112 spill optimization

The KI=112 24-VGPR spill is **structural** to the full-unroll +
hand-tuned schedule combination. Eliminating it requires EITHER:

(a) Re-architecting the schedule to be partial-unroll-safe (rewrite all
    `s_waitcnt vmcnt(N)` to use loop-edge-aware counters; restructure
    LDS slot rotation to not rely on cross-iter interleave). Major
    surgery (~200+ lines of careful audit), risk of regressing the
    other 11 KI specs that share `device_gemm_tile_body`.

(b) Reducing the per-iter live set so full-unroll fits 256 VGPRs.
    Live state in main_loop_iter: A_tile (~16 VGPR), B_tile_0/1
    (~32 VGPR), C_accum[2][2] (~64 VGPR), prefetch buffers. Reducing
    register width on any of these would change the MMA tile shape
    and break the kernel layout invariants.

(c) Accepting the 24-VGPR spill: at KI=112 (DSV3-GateUP K=7168), 4
    weight-1 metric shapes already pass at ratio 1.13-1.14. Further
    perf gain on these shapes is bounded by what the spill costs vs
    the pipeline depth gain. R55 (which added the 5 spill VGPRs to
    KI=112) decided the +9.5 median lift from the EPILOG sched_barrier
    was worth the spill, implying spill cost is small relative to
    the scheduling structure win.

R64 lands evidence FOR option (c): the kernel architecture trades
24 VGPR spill for full-unroll scheduling correctness. A clean partial
unroll would require re-architecting the schedule.

## Outcome

* **HipKittens commit**: 1 (docs-only annotation of R64 falsification
  inline at the KI_HINT > 0 dispatch site).
* **Primus-Turbo commit**: 1 (this round note).
* **Functional kernel state**: identical to R63 / R55 (HK pre-R64
  baseline). No code change, no resource change.

## Metric numbers

```
                       R63 baseline (1 sample)   R64 post-revert (3 samples)
score                  901                       880, 874, 879  (median 879, mean 877.7)
gpt_oss   geomean      1.109                     ~1.07-1.10 (variance noise)
DSV3      geomean      1.123                     ~1.11-1.12
Qwen3     geomean      1.186                     ~1.11-1.18 (variance noise)
correct_fail           0/24                      0/24
```

GPU 3 contention noise was substantial during R64 (3 post-revert
samples ranged 874-880 vs pre-R64 single sample 901, a 22-30 point
drift in the same kernel binary). The historical R55-R63 plateau is
~880-920 with a noise floor of ±15. Three R64 post-revert samples land
at the low side of that band, consistent with a contended GPU 3.

The kernel is **functionally identical to R63 baseline** (R55 probe
gives 47.85 dB SNR pre-/post-revert; HK source diff = docs-only
comment block). The metric movement is pure variance.

## Backward bench

Not run — kernel functionally unchanged from R63. R61's bench result
(24/24 PASS, fwd 1141.07 / bwd 855.25 TFLOPS) remains valid.

## R65 next-action surface

The R63 next-action surface lever #2 (KI=112 spill reduction) is now
**closed as falsified**. The remaining R63 surface levers:

1. **dA backward audit** (R61/R62/R63 carry-over). bench shows bwd 855
   vs fwd 1141 TFLOPS (75% of fwd). dB var_k path
   (`grouped_var_k_kernel`) untouched since R55. Won't move metric
   (fwd-only) but is the lever surface for non-metric workloads.

2. **Per-XCD work-stealing counter** (R62-doc carry-over). 8 counters,
   1 per XCD via `blockIdx.x % 8`, each pre-zeroed to `total_tiles / 8`.
   Reduces atomic contention from 256-way to 32-way. May enable
   extending the work-stealing gate beyond `tiles < NUM_CUS*4 = 1024`
   without hitting the variance wall R62-B saw. Risk: still doesn't
   fix the L2-locality loss (which is independent of atomic
   contention); per-XCD counter only addresses the contention dimension
   of R62-B's variance increase.

3. **Triton-style two-tier persistent grid (grid > NUM_CUS)** — R63
   carry-over. Launch grid = 2 * NUM_CUS = 512 with reduced
   per-block iters; HW dispatch round-robin amortizes prologue across
   more dispatch waves and L2 warm-up. Risk: doubles per-block prologue
   cost (S ≈ 63 µs per R60 finding × 2 = 126 µs aggregate); only net-
   positive if dispatch L2 warm-up savings exceed prologue redundancy.

4. **Cumsum LDS write reduction** (R63 carry-over, tiny). Skip every-
   other `s_cum_tiles[gi+1]` LDS write since only the FIRST element
   from each cum row is read by the inner group-scan. ~0.3 µs win.
   Bundle with a larger lever.

5. **NEW: `#pragma unroll 4` (vs 8) on opaque-cast variant** — same
   mechanism as R64-B but with smaller chunks. The 8-iter chunk's
   loop-back may cross too many vmcnt boundaries; 4-iter might be
   safe enough. Low-confidence but cheap to test if R65 budget
   permits.

6. **NEW: schedule partial-unroll-safe rewrite** — major surgery,
   ~200+ lines, fixes vmcnt counters to be loop-edge-aware. Risk: high.
   Defer to a multi-round project once smaller levers exhaust.

Recommended R65: **option 2 (per-XCD work-stealing counter)** — focused
single-mechanism change, addresses an R62-B-identified variance
component (atomic contention) without re-trying the whole work-
stealing gate extension. Implementation cost ~30 lines (8 counters in
the global buffer, modify atomicAdd target via `blockIdx.x % 8`).
Expected upside: +2-5 score median, similar to R61's gated-WS
mechanism.
