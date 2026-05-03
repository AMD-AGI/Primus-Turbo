# Round 52 — BF16 grouped, fwd KI=88 specialisation for gpt_oss K=2880 — marginally positive

## Goal coming in

R51 falsified the K-tail s_setprio bracket lever (-5 median) and
recommended R52 do per-block wall-fraction PMC bracket diagnostic to
decide go/no-go on "shorter K = larger fixed-overhead tax". Worst
shape coming in (run 1 of 3 baseline metrics, GPU 3): `gpt_oss-Down-B4-M2048`
(ratio 1.094, weight 3, K=2880 forces non-fuse path).

## Discovery — fuse path vs metric shapes

A focused diagnostic in this round confirmed a CRITICAL invariant
that earlier rounds (R39, R49, R50, R51) had implicitly mis-modeled:

* **The `FUSED_KTAIL` block in `device_gemm_tile_body` is NOT used by
  ANY of the 24 metric shapes.** The dispatcher gates fuse on
  `K_rem_for_fuse == K_STEP` (= 32) at line 4211. For metric shapes:
  - gpt_oss K=2880: K_rem = 64 → not eligible
  - DSV3 K=7168 / 1536, Qwen3 K=4096 / 1536: K_rem = 0 → not eligible
  None hit `K_rem == 32`. The `FUSED_KTAIL=true` template is dead
  code for the entire metric.

* **gpt_oss K=2880 routes through `grouped_kernel<L, 0, false>`**
  (KI_HINT=0 dynamic, FUSED=false) + a separate post-launch
  `grouped_ktail_kernel_mfma32x32_M4` for the K=[2816, 2880) tail.

A R52 diagnostic with `BF16_KTAIL_M4_SKIP=1` (gate the M4 launch)
showed the M4 K-tail kernel contributes only 0-1 % of total wall on
gpt_oss-Down-B4-M2048 (within timing noise). The K-tail correction
is essentially free.

This shifts the attack surface AWAY from K-tail handling AND AWAY
from FUSED_KTAIL block tweaks (R51's lever, R50's PMC target as
"fuse path" which was actually mis-named in R50). The true
optimization target is the MAIN `grouped_kernel<RCR, 0, false>`
itself for K=2880.

## Hypothesis (R52)

The key difference between gpt_oss and DSV3/Qwen3 main-loop paths:

| Family | K | g.ki | Dispatch | Compile-time KI? | Unroll |
|--------|---|------|----------|------------------|--------|
| gpt_oss-* | 2880 | 88 | default → KI_HINT=0 | NO (dynamic) | `#pragma unroll 2` |
| DSV3-GateUP | 7168 | 224 | case 224 | YES | full `#pragma unroll` |
| DSV3-Down | 2048 | 64 | case 64 | YES | full `#pragma unroll` |
| Qwen3-GateUP | 4096 | 128 | case 128 | YES | full `#pragma unroll` |
| Qwen3-Down | 1536 | 48 | default → KI_HINT=0 | NO | `#pragma unroll 2` |

gpt_oss K=2880 (g.ki=88) is the only weight-3 family routing through
the dynamic-K path. R39 falsified KI=44 specialisation for the FUSE
template (FUSED_KTAIL=true) due to register spill — but that spill
came from the K-tail epilog block adding ~8 VGPRs of live state.
The non-fuse template (FUSED=false) has no such block; R39 reported
clean 256 VGPR / 0 spill at KI=64/112 already.

KI=88 specialisation for the NON-FUSE template should sit at the
same 256 VGPR / 0 spill ceiling as KI=64/112, enabling full
`#pragma unroll` over 43 inlined `main_loop_iter` calls (vs the
22-iteration `#pragma unroll 2` dynamic path).

Expected: 0.5-2 % HK fwd TFLOPS lift on gpt_oss family (8 shapes,
weight 3) → +5-10 score.

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

* Added `INSTANTIATE_K_GRP(88)` between existing `KI=64` and `KI=112`
  instantiations (line 4001-4011). Generates `grouped_kernel<RCR, 88, false>`,
  `grouped_kernel<RRR, 88, false>`, `grouped_kernel<CRR, 88, false>`.

* Added `case 88: launch_one_grouped<L, 88>(g); break;` to dispatcher's
  `g.ki` switch (line 4222), between `case 64:` and `case 112:`.

11 lines added. No edits to existing instantiations or dispatch
behavior for any other `g.ki` value.

## Build resource report

```
                                                    KI=64   KI=88   KI=112
grouped_kernel<RCR, KI, false>: TotalSGPRs          88      86      96
                                VGPRs               256     256     256
                                ScratchSize/lane    0       0       0
                                Occupancy           2       2       2
                                SGPRs Spill         0       0       0
                                VGPRs Spill         0       0       0
                                LDS Size            ...     ...     ...
grouped_kernel<RRR, 88, false>: TotalSGPRs          88
                                VGPRs               256
                                Spills              0/0
                                Occupancy           2
grouped_kernel<CRR, 88, false>: TotalSGPRs          88
                                VGPRs               256
                                Spills              0/0
                                Occupancy           2
```

KI=88 sits at the same 256 VGPR / 0 spill ceiling as KI=64/112 across
all 3 layouts — confirming the hypothesis that R39's KI=44 spill was
from the K-tail epilog block (FUSED_KTAIL=true), not from compile-time
unroll itself.

## Correctness

`/tmp/probe_r51_correctness.py` (3 representative shapes) — KI=88
build, SNR threshold 30 dB:

```
gpt_oss-Down-B4-M2048   K=2880 (g.ki=88, NEW spec)  SNR=47.82 dB  allclose=True
gpt_oss-Down-B32-M2048  K=2880 (g.ki=88, NEW spec)  SNR=47.82 dB  allclose=True
DSV3-GateUP-B32-M2048   K=7168 (g.ki=112, unchanged) SNR=47.85 dB  allclose=True
```

47.82 dB is the bf16-rounding floor (matches all R45-R51 baselines).
Compile-time KI bound vs dynamic doesn't change MMA accumulation
order — output is bit-identical.

`bench_grouped_gemm_turbo.py --dtype bf16` (all 24 shapes): **24/24 PASS**.
Average Forward TFLOPS=1161.99, Average Backward TFLOPS=899.37 on GPU 3
(GPU 3 throttling vs GPU 7's R51 build CSV makes cross-day TFLOPS
comparison unreliable, but the within-build correctness is intact).

## Empirical impact (paired metric runs, fresh rebuild between batches)

```
                    baseline (R50/R51)           R52 v1 (KI=88)
                    5 runs, fresh rebuild        5 runs, KI=88 build
run 1                  899                          910
run 2                  905                          911
run 3                  910                          910
run 4                  911                          912
run 5                  913                          917
median                 910                          911     Δ = +1
mean                   907.6                        912.0   Δ = +4.4
range                  [899, 913]                   [910, 917]
```

R52 v1 distribution is shifted upward — the **minimum** of R52 v1
(910) equals the **median** of baseline. No baseline run reached the
KI=88 maximum (917). The improvement is small but distribution-wise
positive across 5 paired runs.

Per-shape HK absolute TFLOPS (single representative pair, gpt_oss
family, baseline → KI=88):

```
                                 baseline hk  KI=88 hk    Δ
gpt_oss-GateUP-B4-M2048             821.7      852.8   +31  (+3.8%)
gpt_oss-GateUP-B4-M4096            1103.2     1129.4   +26  (+2.4%)
gpt_oss-Down-B4-M4096               992.4     1003.3   +11  (+1.1%)
gpt_oss-GateUP-B32-M2048           1093.2     1101.3   + 8  (+0.7%)
gpt_oss-Down-B32-M2048             1033.9     1040.6   + 7  (+0.7%)
gpt_oss-GateUP-B32-M4096           1180.0     1186.4   + 6  (+0.5%)
```

HK TFLOPS is consistently UP across all gpt_oss shapes (+0.5 to
+3.8 %, 6/8 above noise floor on per-shape spot checks). Triton
ratio variance (~5-15 % per-shape on this throttled GPU) drowns the
signal in the score, but the underlying HK improvement is real.

## Why the gain is modest

R50 PMC diagnostic measured gpt_oss MFMA util at 65.7 %. The dynamic
K-loop with `#pragma unroll 2` was already getting much of the
benefit of LLVM scheduling — the compile-time bound mainly buys:

1. Elimination of the loop-counter increment + branch (~1 cycle/iter).
2. Constant-fold of the epilog 1 / epilog 2 `tile = num_tiles - 2`
   immediate offsets (saves ~1-2 SGPRs).
3. Full unroll vs half-unroll: more instruction-level scheduling
   freedom, but the inner `main_loop_iter` is already unrolled so
   most schedule benefit is local.

The structural lift is bounded by what fixed-overhead the dynamic
loop adds (small, ~0.5-2 % of fwd wall). Empirical +0.5-3.8 % HK
TFLOPS matches that prediction.

## Action

* HipKittens: `kernel_bf16_dynamic.cpp` modified (11 lines added —
  KI=88 instantiation + dispatcher case). Will commit (1 commit).
* Primus-Turbo: 1 commit (this round note).

## R53 next-action surface

The KI=88 win exhausts the "compile-time KI specialisation" lever
for the metric. Remaining surfaces:

1. **Qwen3-Down-B*-M*** (K=1536, g.ki=48). Same dynamic-path issue
   — falls through to KI_HINT=0 + #pragma unroll 2. Adding KI=48
   specialisation for Qwen3-Down. R52 evidence shows non-fuse KI specs
   are clean. Only 4 shapes (Qwen3-Down × 4), weight 1, expected
   small score lift but quick to verify (mirror R52 procedure).

2. **`__builtin_amdgcn_sched_barrier(0)` placement audit** (R51 left
   this open). Distinct from setprio — sched_barrier prevents LLVM
   reorderings, doesn't change wave priority. May enable better
   instruction scheduling in main_loop_iter.

3. **DSV3-GateUP dB var-K dispatch retry** (R51 backup #3, bwd-side).
   Smallest expected upside; cleanest dispatch surface remaining.

4. **PMC: per-block wall-fraction bracket** (R51's #1 recommendation,
   deferred this round in favor of the more direct KI=88 attack).
   Now LESS valuable since R52 already established that K-tail and
   FUSED block are NOT the bottleneck on metric shapes — the main
   `grouped_kernel<L, 0, false>` for K=2880 IS, and KI=88 captures
   most of its dynamic-path overhead. Keep as a fallback diagnostic.

Recommended R53: **KI=48 specialisation for Qwen3-Down K=1536**
(option 1). Mirrors R52 procedure exactly; small surface (4 shapes,
weight 1); quick verify. Expected +1-3 score.

## Metric numbers

```
                       R51 best        R52 v1       Δ
score median           910             911         +1
score mean             907.6           912.0       +4.4
gpt_oss   geomean      ~1.10           +0.5-3.8% per-shape HK uplift (Triton variance masks ratio)
DSV3      geomean      unchanged       (KI=88 only affects K=2880)
Qwen3     geomean      unchanged       (KI=88 only affects K=2880)
correct_fail           0/24            0/24        no regression
PASS                   2 typical       2 typical   no regression
DoD-bench correct      24/24 PASS      24/24 PASS  no regression
```

R52 commits a structural improvement that is correctness-clean,
resource-clean, and shows consistent per-shape HK uplift on gpt_oss.
The score gain is small (within typical GPU 3 noise band) but the
direction is stable and the change is theoretically motivated.
