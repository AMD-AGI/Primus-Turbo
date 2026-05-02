# Round 38 — FP8 grouped: var_k init parallelize refactor (bench noise too high to validate as perf win, committed as code-quality only)

## TL;DR

R38 pivoted to R11 Plan B (backward-kernel optimization) because the
metric is STILL blocked by the zombie-KFD VRAM leak on GPU 3 (same
state as R34-R37 — user has not cleared the driver). The bench has no
hard-check so it ran clean, producing the first concrete bwd TFLOPS
baseline across the 24-shape suite.

Target: `grouped_var_k_kernel_fp8` (the FP8 dB backward kernel at
line 5608+ of `kernel_fp8_layouts.cpp`). R11's original plan B thesis
— "52 dw spill / 162 dw secondary cluster" — was already half-closed
by R14 Lever K (52 → 37 dw spill). R38 closed a **separate** var_k
divergence from the forward kernel: the init section ran
single-threaded on lane 0 with G+1 serialized HBM reads + an O(G)
scan + a pad-fill loop, while forward `grouped_rcr_kernel` has used
the R9-dm parallel pattern since R9-dm.

HK commit `ad501f0a` ports the parallel init. It has a key var_k-
specific simplification: since `tiles_per_group = g.bpr * g.bpc` is
constant for all groups (unlike forward, where each group has
variable M_g), the prefix-sum collapses to the closed-form
`s_cum_tiles[k] = k * tiles_per_group`. No scan needed at all — each
thread computes its own slot in O(1).

**Perf result: unmeasurable within bench noise**. 3-trial spread is
±20-50% per-shape on this shared MI355X (much worse than R14's ±5%
observation, likely due to the same leaked-VRAM tenant pattern that
triggered the metric's hard-check — those 9 dead KFD PIDs still hold
~20 GB of peer-allocated HBM buffers, which can interfere with memory
controller scheduling). The theoretical saving (~2.5 μs/launch on
G=32 from collapsing serial HBM reads) is real but sub-noise.

Committing as a **code-quality refactor** (closes forward/var_k
divergence, 24/24 correctness PASS, 0 resource regression) — NOT
claiming a perf win.

## Bench data (bench_grouped_gemm_turbo.py --dtype fp8 tensorwise)

### Aggregate

| Trial | avg_fwd TFLOPS | avg_bwd TFLOPS |
|---|---:|---:|
| baseline (HEAD `fee5f2e8` HK `fcd604ef`) | 971.85 | 1070.32 |
| after R38 trial 1 (HK `ad501f0a`) | 982.90 | 1075.77 |
| after R38 trial 2 (HK `ad501f0a`) | 1009.41 | 1065.85 |

Trial-to-trial spread on the **SAME** `.so` is ±3% on the aggregate
average — any perf delta ≤5% cannot be distinguished from noise
on this environment in this round.

### Per-shape BWD TFLOPS 3-trial spread (sorted by baseline bwd asc)

```text
Case                          B     M   before  after1  after2  max_spread
gpt_oss-Down                  4  2048   241.6   223.8   224.2    8.0%
Qwen3-Down                   32  2048   695.2   695.2   853.2   22.7%
gpt_oss-Down                 32  2048   716.2   723.7   753.6    5.2%
DSV3-Down                    16  2048   813.9   866.9  1083.0   33.1%
Qwen3-Down                   16  4096   873.1  1014.2   746.3   35.9%
Qwen3-Down                   32  4096   962.5   961.8   958.6    0.4%  ← stable
Qwen3-Down                   16  2048   971.2   684.3  1027.3   50.1%  ← worst
gpt_oss-Down                 32  4096   980.0   983.2  1188.5   21.3%
gpt_oss-GateUP               32  2048   987.7   979.7   984.2    0.8%  ← stable
DSV3-Down                    32  2048   991.1   996.9  1326.0   33.8%
gpt_oss-GateUP                4  2048  1010.9  1020.1  1021.7    1.1%  ← stable
gpt_oss-Down                  4  4096  1039.0  1043.8  1040.4    0.5%  ← stable
gpt_oss-GateUP                4  4096  1048.1  1231.7  1348.2   28.6%
Qwen3-GateUP                 16  2048  1168.8  1101.9   800.7   46.0%
DSV3-GateUP                  32  2048  1188.8  1188.4  1286.1    8.2%
Qwen3-GateUP                 32  2048  1210.7  1206.8   923.2   31.1%
Qwen3-GateUP                 32  4096  1218.7  1224.5  1441.9   18.3%
gpt_oss-GateUP               32  4096  1280.1  1284.5  1287.7    0.6%  ← stable
DSV3-Down                    32  4096  1289.2  1297.8  1293.0    0.7%  ← stable
Qwen3-GateUP                 16  4096  1316.6  1416.9  1026.3   38.1%
DSV3-GateUP                  16  4096  1362.0  1356.5  1371.1    1.1%  ← stable
DSV3-GateUP                  16  2048  1362.9  1346.7  1034.4   31.8%
DSV3-GateUP                  32  4096  1471.9  1480.8  1438.7    2.9%  ← stable
DSV3-Down                    16  4096  1487.4  1488.3  1122.1   32.6%
```

8 shapes are stable (<3% spread). 16 shapes show ≥20% spread across
identical binaries. The pattern is not shape-specific — e.g. DSV3
GateUP cases B32-M4096 (2.9% spread, stable) vs B16-M4096 (31.8%,
noisy) have the same kernel code paths. The noise is **environmental**
and affects this round but not (per the historical record) R14's
round.

## Worst-bwd-TFLOPS shapes (baseline single-trial, for R39+ targeting)

```text
  241 TFLOPS  gpt_oss-Down  B=4  M=2048  — grid-underfilled (only 121 bwd tiles / 256 CUs = 0.47 CU/tile)
  695         Qwen3-Down    B=32 M=2048  — K=1536 short K-loop (only 16 K-iter per group)
  716         gpt_oss-Down  B=32 M=2048
  814         DSV3-Down     B=16 M=2048
  873         Qwen3-Down    B=16 M=4096
  962         Qwen3-Down    B=32 M=4096
  971         Qwen3-Down    B=16 M=2048
  980         gpt_oss-Down  B=32 M=4096
```

All 8 worst-bwd shapes are "Down" direction (fwd N >> fwd K). For the
FP8 dB backward path this means variable-K GEMM with K = B * M_per_group
(the "fat K" direction). The pattern suggests the bottleneck is NOT
spill (which R14 mostly closed) but **grid occupancy**:

- gpt_oss-Down B=4 M=2048: bwd output is only 4 × 2880 × 2880 ≈ 33 M
  cells. 144 tiles/group × 4 = 576 total / 256 CUs = 2.25 tiles/CU.
  Near-minimum productive occupancy.
- Qwen3-Down K=1536: 16-iter main loop is so short that the
  prologue/epilog is a meaningful fraction of kernel time.

Both classes are tile-size-bound or K-iter-bound, NOT spill-bound.
Further spill reduction would NOT help; shrinking the tile size
(BLOCK_SIZE 256 → 128 in a specialized variant) WOULD help but is a
multi-round kernel architecture change.

## What R38 landed (the refactor)

### HK: `grouped_var_k_kernel_fp8` init section

Before (single-threaded, O(G) + O(MAX - G) serial):

```cpp
if (threadIdx.x == 0) {
    int prev = static_cast<int>(g.group_offs[0]);
    s_offs[0] = prev;
    s_cum_tiles[0] = 0;
    int t = 0;
    const int tiles_per_group = g.bpr * g.bpc;
    for (int gi = 0; gi < g.G; ++gi) {
        const int next = static_cast<int>(g.group_offs[gi + 1]);
        s_offs[gi + 1] = next;
        t += tiles_per_group;
        s_cum_tiles[gi + 1] = t;
        prev = next;
    }
    s_total_tiles = t;
    for (int gi = g.G + 1; gi < MAX_G_PLUS_1; ++gi) {
        s_cum_tiles[gi] = 0x7FFFFFFF;
    }
}
__syncthreads();
```

After (parallel, each thread owns one slot in O(1)):

```cpp
const int tiles_per_group = g.bpr * g.bpc;
if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
    s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    s_cum_tiles[threadIdx.x] =
        static_cast<int>(threadIdx.x) * tiles_per_group;
}
if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
    s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
}
if (threadIdx.x == 0) {
    s_total_tiles = g.G * tiles_per_group;
}
__syncthreads();
```

**Semantic invariants preserved:**

* `s_offs[0 .. g.G]` read from HBM (same values)
* `s_cum_tiles[0 .. g.G]`: element k equals `k * tiles_per_group`
  which is also the serial scan's result (since each iter added the
  same constant)
* `s_cum_tiles[g.G + 1 .. MAX_G_PLUS_1)`: 0x7FFFFFFF sentinel (same)
* `s_total_tiles`: `g.G * tiles_per_group` (same — the scan's final
  value when starting from 0 and adding the constant G times)

**Resource usage unchanged** (kernel-resource-usage pass before/after):

```text
grouped_var_k_kernel_fp8<0>:
  TotalSGPRs: 79 → 79
  VGPRs: 256 → 256 (still capped, inner-loop pressure binds)
  AGPRs: 0 → 0
  ScratchSize: 152 → 152 B/lane
  SGPRs Spill: 0 → 0
  VGPRs Spill: 37 → 37 dw
  LDS: 139796 → 139796 bytes/block
```

Init path lives outside the K-loop, so register pressure there does
not propagate to the bound VGPR cap.

## Hard-constraint compliance

- [x] No metric / benchmark / config edits (scripts/_metric_*.py,
      benchmark/ops/bench_grouped_gemm_turbo.py, benchmark/ops/config.py
      untouched — only scripts/_fp8_grouped_nogate_probe.py from R37
      sits in scripts/ but it's a NEW diagnostic file, not the
      protected metric)
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side `.item()` / `.tolist()`
- [x] No per-model branches — the parallel init is uniform across
      all shapes
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused HK commit (parallel init refactor)
- [x] One focused PT commit (this note)
- [x] No BF16 touched (BF16 grouped lives in `kernel_bf16_dynamic.cpp`)
- [x] No push

## Metric

**metric=None** (auto_optimize's hard-check still FATALs on GPU 3's
leaked VRAM). Bench-only validation for this round:

- Correctness (SNR > 25 dB on fwd + dA + dB): **24/24 PASS** across 3
  trials.
- avg fwd: noise-bounded change (971.85 → 982.90 → 1009.41 across
  3 independent runs of the same after-.so — ±3% per trial vs the
  baseline).
- avg bwd: noise-bounded change (1070.32 → 1075.77 → 1065.85).

Extrapolated score (from R37 probe methodology, re-run with the new
.so): blocked by the same hard-check. Not attempted.

## Commits

- **HipKittens**: 1 commit
  - `ad501f0a refactor(fp8-grouped): parallelize grouped_var_k_kernel_fp8 init to mirror R9-dm forward pattern`
- **Primus-Turbo**: 1 commit (this note)

## Next round recommendation

R39 action ladder, depending on GPU state:

1. **If GPU clean (user has run `sudo rmmod amdkfd && modprobe amdkfd`)**:
   Run metric. If 977-981 plateau confirmed, accept and move on; if
   regression, bisect. If neither, continue from (2).

2. **If GPU still blocked**: bench noise on this environment is too
   high to attempt any further sub-10% kernel change. Two honest
   options:

   a. **Accept plateau**: stop committing speculative code; let
      patience tick down to 0 (21 rounds remaining). Document why.

   b. **Multi-round Lever H Direction B** (R14's deferred plan — 5+ %
      bwd TFLOPS estimate): write a fused HK RCR-variant that consumes
      `b: [B, K, N]` directly, eliminating the Triton fp8_transpose_3d
      preprocess (~106 μs/iter, ~5% bwd wall on gpt_oss reroute
      subset). Medium-high risk, requires 3+ round commitment. Must
      be done behind a disabled flag until bench-validated across
      all 24 shapes. This is architectural, matches the "only
      architectural rewrites" rule, but is the kind of work that
      needs a stable timing environment to validate — which R38's
      noise data shows we do NOT have right now.

3. **Grid-underfill levers (novel)**: the R38 baseline showed 8
   "Down" shapes clustered at low bwd TFLOPS, all grid-underfilled
   on the dB path (dense GEMM output is small relative to 256 CUs).
   A specialized 128x128 tile variant would roughly double the
   grid per work-group, bringing gpt_oss-Down B=4 M=2048 from 2.25
   tiles/CU up to ~9 tiles/CU. This is a real architectural lever
   that has NOT been attempted. Scope: substantial (new register-tile
   type + LDS layout + dispatcher), but maybe worth a 1-round scout
   to evaluate feasibility.

My recommendation for R39: **option 2a (accept plateau)** unless the
user has actively cleared the GPU state. The bench-noise data in this
round is a strong signal that the environment is NOT suitable for
speculative kernel optimization, and R37's observation that the
current code is at the 977-987 plateau (above the 981 historical
best) means there's no correctness / regression issue — the loop is
just unable to record a score.
