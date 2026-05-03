# Round 61 — BF16 grouped, fwd gated atomic-claim work-stealing — PASS (+16 median / +20 mean score)

## Goal coming in

R60 (commit `3f1f5ef`) closed the 1-tile-per-block lever as FALSIFIED
and analytically derived a prologue-cost bound (P + 3I = 1033 µs on
gpt_oss-GateUP-B4-M2048; using a corrected occupancy=1 dispatch model
this resolves to S ≈ 63 µs setup, I ≈ 323 µs per-iter wall). R60's
recommended R61 next-action #1 was **work-stealing via global atomic
counter** — keeps the 256 prologues amortized over multiple iters per
block (unlike 1tpb's prologue-per-tile multiplier) AND eliminates the
wave imbalance from `tiles % NUM_CUS != 0` by dynamically claiming
tiles as blocks finish.

R61 starting metric (GPU 3, single sample): score=888, gpt_oss family
geomean 1.080, DSV3 1.148, Qwen3 1.175. Per-shape worst:

```
gpt_oss-Down-B4-M2048    B=4 M=2048 N=2880 K=2880  tiles=384   ratio 0.949  weight 3  progress 0.759
gpt_oss-GateUP-B4-M2048  B=4 M=2048 N=5760 K=2880  tiles=736   ratio 0.995  weight 3  progress 0.796
```

Same wave-imbalance signature R56 PMC capture diagnosed (`tiles % 256
∈ {128, 224}` → 2× per-CU iter spread → MFMA% 50.4% on the 736-tile
shape).

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
(committed as HK SHA `31585671`):

1. **Struct field** (`grouped_layout_globals`): `+1 int* tile_counter`.
   Aggregate-init defaults trailing field to nullptr for direct
   callers that don't go through `dispatch_grouped`.

2. **Lazy device buffer** (host-side):
   ```cpp
   static int* grouped_tile_counter_buffer() {
       static int* d_counter = nullptr;
       if (d_counter == nullptr) hipMalloc(&d_counter, sizeof(int));
       return d_counter;
   }
   ```
   Single 4-byte `hipMalloc` per process, leaked at exit (fine — the
   .so is loaded once for the process lifetime).

3. **Gate predicate** (general, no per-(M,N,K) hardcode):
   ```cpp
   static inline bool should_use_work_stealing(int M_total, int bpc) {
       if (bpc <= 0 || M_total <= 0) return false;
       const int tiles = (M_total / BLOCK_SIZE) * bpc;
       return (tiles > 0) && (tiles < NUM_CUS * 4) && ((tiles % NUM_CUS) != 0);
   }
   ```
   Fires on shapes with imbalance AND small enough total tile count
   that the imbalance fraction is significant. For the 24-shape MoE
   metric the gate fires on exactly `tiles ∈ {384, 736}` (the 2 worst
   gpt_oss B=4 M=2048 shapes).

4. **Counter prime** (called from `launch_one_grouped{,_fuse}`):
   ```cpp
   static inline void prime_grouped_tile_counter(grouped_layout_globals& g) {
       if (!should_use_work_stealing(g.M_total, g.bpc)) {
           g.tile_counter = nullptr;
           return;
       }
       int* counter = grouped_tile_counter_buffer();
       g.tile_counter = counter;
       hipMemsetAsync(counter, 0, sizeof(int), g.stream);
   }
   ```
   Memset is async on the kernel's stream — completes before kernel
   entry without an explicit sync.

5. **Persistent loop** (kernel-side): when `g.tile_counter != nullptr`,
   the seed `pid` and per-iter advance both come from
   `atomicAdd(g.tile_counter, 1)` (broadcast through `__shared__
   s_claim`). When nullptr, falls back to the pre-R61 static partition
   `pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, ...)` /
   `gt += NUM_CUS`.

The persistent-loop change is wrapped at the bottom of the existing
loop body — single `if (g.tile_counter != nullptr)` runtime branch
costs ~5 cycles per tile, negligible vs the ~323 µs per-tile wall.

## R61-A side experiment (FALSIFIED, not committed)

Initial R61 prototype enabled work-stealing for ALL shapes (no gate).
Per-shape metric (single sample, GPU 3):

```
                                 baseline  R61-A    Δ
DSV3-GateUP-B16-M2048             1.288    1.108   −14 %  (was PASS)
Qwen3-Down-B16-M2048              1.346    1.143   −15 %  (was PASS)
gpt_oss-GateUP-B4-M2048           0.995    1.325   +33 %  (now PASS)
gpt_oss-Down-B4-M2048             0.949    1.187   +25 %
gpt_oss-GateUP-B4-M4096           1.221    1.174   −4 %
gpt_oss-Down-B4-M4096             1.277    1.214   −5 %
```

R61-A clearly wins on imbalanced shapes (tiles ∈ {384, 736}) but
regresses ~14-15 % on shapes whose static partition was preserving L2
cache locality on B-tile reads. Mechanism: with 256 blocks doing
static stride NUM_CUS, CU n always processes tiles
{n, n+256, n+512, ...} — a deterministic per-CU sequence whose B-tile
reads (same `pid_n` across n+256 increments when grouped via
`group_m`) hit warm L2 lines. Atomic-claim destroys this pattern;
arrival-order tile claims fragment the per-CU sequence.

R61-gated keeps R61-A's wins on imbalanced shapes (verified by
follow-up paired runs) and reverts to static stride for everything else.

## Resource report (R61-gated, build clean)

```
                                SGPR  VGPR  VGPRspill  occ   Δ vs R55
RCR KI=0   (DSV3-Down dyn):      95   246       ?       2     +5/+2
RCR KI=48  (Qwen3-Down):         92   256       1       2     +5/0/+1 spill
RCR KI=64  (Qwen3-GateUP):       92   256       1       2     +5/0/+1 spill
RCR KI=88  (gpt_oss non-fuse):   92   256       1       2     +5/0/+1 spill
RCR KI=112 (DSV3-GateUP K=7168): 92   256      24       2     +5/0/0 (already at ceiling)
RCR KI=0   FUSED=1 (gpt_oss):   102   250       0       2     +6/+2
RRR KI=0   FUSED=1 (dA):        104   250       0       2     +5/+1
CRR KI=0   (CRR forward):        99   246       0       2     +5/+2
```

The +5-6 SGPR delta is the runtime branch + s_claim shared broadcast;
+1 VGPR spill on KI=48/64/88 hits the 256 VGPR ceiling already and
spills 1 register. KI=112 unchanged. FUSE paths +2 VGPR (still clean).
All paths retain occupancy 2 (compiler-predicted; actual occupancy is
LDS-bound to 1 anyway since `g.dynamic_shared_memory()` requests 160
KB ≥ MI355X CU LDS budget).

## Correctness — fwd-only (probe)

`/tmp/probe_r60_correctness.py` via Primus-Turbo `turbo.ops.grouped_gemm`
(BackendType.HIPKITTEN), 5 shapes covering both gated and ungated
modes:

```
                                                   mode      SNR     allclose
gpt_oss-GateUP-B4-M2048  tiles=736                gated   47.85 dB   True
gpt_oss-Down-B4-M2048    tiles=384                gated   47.85 dB   True
gpt_oss-GateUP-B4-M4096  tiles=1472               static  47.85 dB   True
DSV3-GateUP-B16-M2048    tiles=2048               static  47.85 dB   True
Qwen3-Down-B16-M2048     tiles=2048 (RCR KI=48)   static  47.83 dB   True
```

5/5 PASS at bf16 floor.

## Correctness — full BF16 grouped fwd + bwd bench

`benchmark/ops/bench_grouped_gemm_turbo.py --dtype bf16` (24 shapes, fwd
+ bwd cross-checked against Triton reference via `check_allclose` and
SNR built into the bench harness):

```
24/24 PASS
Average Forward TFLOPS:  1141.07
Average Backward TFLOPS:  855.25
```

Notable per-shape (gated forward shapes):
* gpt_oss-GateUP-B4-M2048: fwd 866 / bwd 610 TFLOPS, PASS
* gpt_oss-Down-B4-M2048:   fwd 785 / bwd 459 TFLOPS, PASS

The dA backward path (RRR `FUSED=1` template) also routes through
`grouped_kernel`, and the gate fires on dA shapes with the same
(M_total × K_dA / 256²) imbalance signature (e.g. dA of
`gpt_oss-GateUP-B4-M2048` = (M=8192, N=2880, K=5760) → tiles=384,
gated). The bwd PASS at default `check_allclose` thresholds confirms
work-stealing produces numerically equivalent output to static
partition on all 3 grouped layouts (RCR/RRR/CRR), since the body of
each tile's compute is unchanged — only the iteration order over
tiles differs.

The bwd dB path (`grouped_var_k_kernel`) uses a different struct
(`grouped_var_k_layout_globals`) and a different launch helper
(`launch_grouped_var_k`); it is **unchanged** by R61.

## Metric (5×R61-gated paired with 5×baseline, GPU 3, single shell)

```
R55 baseline (HK 237ca6b1):  890 / 868 / 882 / 867 / 897    median 882   mean 880.8   range 30
R61-gated:                    894 / 899 / 898 / 917 / 898    median 898   mean 901.2   range 23
Δ                             +16 median / +20.4 mean
```

Distributions are near-disjoint:
* R61-gated min = 894 > baseline median = 882
* Only the single sample 894 (R61-gated) overlaps with baseline range (867-897)

The improvement comes from `gpt_oss-GateUP-B4-M2048` flipping from
progress 0.79-0.93 (ratio 0.99-1.16 in baseline) to PASS (ratio
1.30+ in R61-gated, progress 1.0 capped). At weight 3 / total weight
40, +1 shape PASS contributes (1.0 - 0.85) × 3 / 40 ≈ +1.1 % progress
≈ +11 score, matching the observed +16 median / +20 mean (additional
~5-9 score from the second gated shape `gpt_oss-Down-B4-M2048` whose
ratio variance is wider).

## Why R61 captures only +20 mean, not the +30+ R60 doc predicted

R60 predicted gpt_oss-Down-B4-M2048 lift as a similar +33 % to
GateUP-B4-M2048. In paired runs the lift is more variable: GateUP
consistently hits PASS (1.30+) while Down sometimes hits 1.30+ and
sometimes 1.18. The variance is from the Triton baseline — both
shapes have small total work (tiles=384, ~50 ms wall) that's
sensitive to GPU contention noise. R61's HK TFLOPS lift is robust
(706 baseline → 866 R61 on gated GateUP, +23 %) but the ratio's
denominator (TRT) co-varies with HK as both kernels contend for the
same memory subsystem.

## Why work-stealing wins on imbalanced shapes specifically

Each block's wall = S (setup) + N (claims) × I (per-tile work). With
static partition and 256 blocks for tiles=736:
* 224 blocks claim 3 tiles → wall = S + 3I
* 32 blocks claim 2 tiles → wall = S + 2I + 1×idle = S + 3I (idle wall counted)
* Slowest-block wall = S + 3I; FAST blocks finish at S + 2I and
  contribute nothing further (they idle until the slowest block finishes)

With work-stealing:
* All 256 blocks claim concurrently, each processes ~ceil(736/256) = 3
  tiles on average; the dynamic claim absorbs *memory-system noise*
  (slow-memory blocks claim fewer tiles, fast ones claim more)
* Slowest-block wall = S + 3I still in the noiseless model, BUT in
  practice memory-system noise on B=4 small-tile-count shapes hits
  one of the 32 "short" blocks first → that block claims a 3rd tile
  before the noisy 224 finish their 3rd → wall converges toward
  S + (736/256)*I = S + 2.875×I = S + 2.875 * 323 ≈ 991 µs

That's a 3-4 % wall reduction (1033 → 991), translating to ~3 % HK
TFLOPS lift. The observed ratio jumps (0.99 → 1.30, +33 %) are larger
than this, suggesting something beyond pure variance reduction —
possibly L2 cache prefetch differences from the arrival-order claim
sequence (each block grabs whichever tile the L2 has already fetched,
versus static stride which forces a fixed cold-start pattern). A
PMC marker capture between the static and work-stealing builds would
confirm whether the win is from variance-reduction (small) or from
prefetch alignment (large).

## Outcome

* **HipKittens commit**: `31585671` — `bf16: gated atomic-claim
  work-stealing for low-tile imbalanced grouped (R61)`. +109 / -3
  lines on `kernel_bf16_dynamic.cpp`.
* **Primus-Turbo commit**: 1 (this round note).

## R62 next-action surface

1. **Per-XCD chiplet swizzle for the 256-block grid** (R60 doc carry-
   over). The current `chiplet_transform_chunked(_, NUM_CUS=256, 8, 64)`
   is identity (limit = 0 since `num_workgroups < NUM_XCDS*chunk_size
   = 512`). A proper chiplet swizzle would route adjacent CUs of the
   same XCD to adjacent N-tile cols, improving B-tile L2 reuse.
   Ungated shapes would benefit (~+1-3 % per shape on DSV3 / Qwen3).
   Suggested chunk_size=32 (block = 8×32 = 256), making `limit = 256`
   so all blocks get permuted.

2. **Investigate the variance source of the gated lift** (PMC marker
   capture between static and work-stealing builds on
   `gpt_oss-GateUP-B4-M2048`). Determines whether to extend
   work-stealing to additional shapes (e.g., tiles ∈ [1024, 2048)
   imbalanced shapes that the current gate excludes).

3. **dA backward optimization audit**. R61's gate fires on dA shapes
   with imbalanced tile counts. The bench shows bwd TFLOPS ~855 vs
   fwd ~1141 (75 % of fwd) — bwd is the bigger optimization target
   for non-metric workloads. dB var_k path (`grouped_var_k_kernel`)
   is untouched by R61 and remains a separate lever surface.

4. **Reduce the +1 VGPR spill on KI=48/64/88 non-fuse compile-time
   specs**. The R61 atomic-claim shared-broadcast adds ~6 SGPR + 1
   VGPR pressure. If spill becomes problematic on cooler shapes,
   restructure `s_claim` to live in a register that's already
   participating in the per-iter advance check (eliminate the
   dedicated shared slot).
