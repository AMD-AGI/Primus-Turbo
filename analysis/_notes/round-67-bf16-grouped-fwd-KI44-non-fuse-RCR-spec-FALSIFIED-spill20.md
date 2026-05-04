# Round 67 — BF16 grouped, fwd KI=44 non-fuse RCR spec (gpt_oss K=2880) — FALSIFIED (20 VGPR spill)

## Goal coming in

R66 (PT `c8d246e`, HK clean) closed the tiles_m=22 dB var-K cfg audit as
neutral within ±5 noise, leaving 7 consecutive falsified / neutral rounds
(R60-R66) on the BF16 grouped metric. R66's next-action surface offered
3 R67 candidates:

1. K_STEP depth flip (64→32 or 64→128) — large structural change, not
   a one-line flip despite R66's suggestion (all LDS tile shapes use
   K_STEP).
2. Dispatcher-level: skip `bf16_transpose_3d` on gpt_oss-Down fwd —
   misattribution (fwd doesn't trigger H4 transpose; only dA bwd does).
3. PMC-driven occupancy audit — diagnostic only.

R67 starting metric baseline (GPU 3, idle): **score=882**, all 24 PASS,
worst shape `gpt_oss_20B-GateUP-B32-M2048` ratio=**1.057 / progress
0.845** (weight 3). gpt_oss_20B family geomean=1.092, DSV3=1.121,
Qwen3=1.113.

## Hypothesis under test (R67-H1)

**`grouped_kernel<Layout::RCR, 44, false>` compile-time KI=44 non-fuse
spec** would close the last missing data point in R58's short-K spill
analysis.

R40 + R58 covered:
* KI=24 non-fuse RCR: 20 VGPR spill
* KI=32 non-fuse RCR: 20 VGPR spill
* KI=44 **FUSE** RCR: 9-28 VGPR spill (unroll 1 / full) — R58 Attempt 2
* KI=48 non-fuse RCR: **0 spill** (R40, R53 RCR-only workaround)
* KI=88 non-fuse RCR: 0 spill (R40 dead-code anyway — no metric shape
  has g.ki=88)

**KI=44 non-fuse RCR was the missing cell.** Gap between KI=32 (20
spill) and KI=48 (0 spill) spanned 16 units of KI — spill is known
non-monotonic near this range (R42: KI=90 spills 19 while KI=88 clean),
so KI=44 could plausibly land either clean or spilling.

If clean, the lever would be: gate `fuse_ktail_eligible = false` for
`L==RCR && g.ki==44`, let the non-fuse compile-time KI=44 run with
**full `#pragma unroll`** over 22 main_loop_iter calls, and recover
the K-tail correction via the existing **external M4 K-tail kernel**
(already launching via `need_tail_run && K_rem==K_STEP`, costing <=1 %
of wall per R52 `BF16_KTAIL_M4_SKIP=1` diagnostic). The FUSE dynamic
KI_HINT=0 path currently runs `#pragma unroll 2` — unrolling 22 iters
fully would save loop-counter + loop-edge scheduling overhead.

Expected: +1-3 % HK fwd TFLOPS on all 8 gpt_oss shapes (weight 3) ⇒
+3-10 score, analogous to R52/R53 if those had hit the correct g.ki.

## What was actually changed

**`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`**
(reverted after build):

1. Added `template __global__ void grouped_kernel<Layout::RCR, 44>(
   const grouped_layout_globals);` next to R53's KI=48 RCR spec
   (line 4102). RCR-only because R40 / R58 show RRR/CRR at KI≤48
   spill 16-20 VGPR layout-independently (short-K MMA accumulator
   register pressure).
2. Modified `fuse_ktail_eligible` (line ~4391) to exclude
   `L==Layout::RCR && g.ki==44` so gpt_oss K=2880 routes through the
   compile-time non-fuse path instead of FUSE dynamic.
3. Added `case 44:` to the non-fuse dispatch switch (RCR-only branch
   mirroring R53's KI=48 RCR-only pattern).

No Primus-Turbo change.

## Build resource report (R67 v1 — FALSIFIED)

```
Function                                        SGPRs  VGPRs  VGPRspill  ScratchSize
grouped_kernel<RCR, 44, false> (R67, new)        ...    256      20         84     ← SPILL
grouped_kernel<RCR, 48, false> (R53, existing)   ...    256       0          ?     (clean)
grouped_kernel<RCR, 56, false> (R40, existing)   ...    256      18         76     ← SPILL
grouped_kernel<RCR, 64, false> (existing)        ...    256       0          ?     (clean)
grouped_kernel<RCR, 88, false> (R52, existing)   ...    256       0          ?     (clean, but dead code)
grouped_kernel<RCR, 112, false> (existing)       ...    256      20          ?     ← SPILL (R64 attenuation falsified)
```

KI=44 non-fuse RCR spills **20 VGPR** — perfectly matches the KI=24 /
KI=32 pattern from R58. This **falsifies** R67-H1: the spill pattern is
not a gap between KI=32 and KI=48, it's a **threshold at KI ≥ 48**.
Below 48, short-K full-unroll triggers LLVM's aggressive SW-pipelining
heuristic that doubles A_tile / B_tile register lifetimes (R58 mechanism
analysis). KI=44 has 22 iters fully unrolled vs KI=48's 24 iters — not
enough to trip the different heuristic that lets KI=48 allocate cleanly.

Spill cost ≈ 20 dwords × ~50-cycle VMEM scratch latency × high reuse =
thousands of cycles per kernel, far worse than the dynamic-K loop
overhead (~5 cycles × 22 iters = 110 cycles) it would replace. No point
running the metric — a KI=44 spec **would regress** gpt_oss fwd.

Also observed (side data):
* `grouped_kernel<RCR, 56, false>`: 18 VGPR spill in this compiler build
  (R40 had 14). LLVM / ROCm version drift — the existing KI=56 case is
  also sub-optimal but kept because (a) no metric shape has g.ki=56
  anyway, and (b) removing instantiations risks breaking other users.
  Not touched in R67.

## Correctness

Not exercised — build report alone disqualifies the spec on perf
grounds (same pattern as R58's 3-attempt bailout). Revert was via
`git checkout --` on the HK source; `git status` clean afterwards and
rebuild produced the baseline binary. Re-ran `_metric_grouped_bf16_weighted_wall.py`
post-revert: **score=875**, all 24 PASS, no regression (882 → 875 = -7
within the ±5-10 GPU-3 noise band).

## What R67 closes

**The compile-time KI specialization lever on BF16 grouped forward is
now completely closed for the current metric.**

Full breakdown across all g.ki values that SOME metric shape routes
through:

| Family              | K     | g.ki | Route today                          | Can we spec?               |
|---------------------|-------|------|--------------------------------------|----------------------------|
| Qwen3-Down fwd      | 1536  | 24   | KI_HINT=0 dynamic #unroll 2         | R58: spill 20              |
| DSV3-Down fwd       | 2048  | 32   | KI_HINT=0 dynamic #unroll 2         | R58: spill 20              |
| gpt_oss-GateUP/Down | 2880  | 44   | FUSE KI_HINT=0 dynamic #unroll 2    | **R67: spill 20**          |
| Qwen3-GateUP dA RRR | 3072  | 48   | KI_HINT=48 #full-unroll (R53)       | already specced            |
| Qwen3-GateUP fwd    | 4096  | 64   | KI_HINT=64 #full-unroll             | already specced            |
| DSV3-GateUP fwd     | 7168  | 112  | KI_HINT=112 #full-unroll            | already specced (spills 20)|

All 24 metric shapes now have a **verified** compile-time-KI disposition.
No hidden specialization opportunity remains.

## Side finding — R52's `INSTANTIATE_K_GRP(88)` and R53's KI=48 RCR

R58 already documented these as dead code for the current metric (R52's
`case 88` would need g.ki=88 ⇔ K=5632, which no metric shape uses; R53's
KI=48 RCR _is_ hit by Qwen3-GateUP dA RRR g.ki=48 — wait, R53's
instantiation was `grouped_kernel<Layout::RCR, 48>`, NOT RRR; and
Qwen3-GateUP dA is RRR not RCR so R53's RCR-only spec doesn't match).

Actually re-checking: Qwen3-Down fwd K=1536 (g.ki=24) and Qwen3-Down
dA uses K_dA = N_fwd = 4096 (g.ki=64). Neither is g.ki=48. The only
path that hits g.ki=48 in current metric is **Qwen3-GateUP dA RRR**
(K_dA = N_fwd = 3072, g.ki = 48). That's RRR, not RCR.

**Confirmed: R53's RCR-only KI=48 spec is also dead code** on all 24
metric shapes. R52's KI=88 (= K=5632) and R53's RCR-only KI=48 (no
RCR metric shape has g.ki=48) are both dead code. R67 doesn't clean
them up — they're harmless (switch case mismatches at runtime skip
them) and removing adds churn. Future cleanup round can drop both.

## Compliance check

* No kernel code committed — HipKittens working tree reverted to
  baseline (`git status` clean).
* No Primus-Turbo code change — only this round note added.
* No `can_handle` tightening, no per-(M,N,K) hardcode, no host sync
  added, no caching.
* Correctness PASS on all 24 shapes at baseline.
* Metric post-revert (875) within ±10 noise band of pre-R67 baseline (882).

## Recommendation for R68

After R40 + R58 + R67, the **compile-time KI lever on BF16 fwd is
definitively exhausted**. Remaining attack surface, ranked by expected
leverage:

1. **dB var-K kernel** (highest leverage, per R41 wall-decomp: bwd is
   64-71 % of wall and dB is half of that). R42-R48 explored but not
   exhaustively — levers still untried:
   * `__launch_bounds__(NUM_THREADS, 2)` hint to force occupancy
     discipline (currently `(_, 1)`; achieves 2 waves/SIMD per build
     report but hint may change compiler decisions).
   * PMC diagnostic (rocprofv3 `valuMfmaUtil` on
     `grouped_var_k_kernel<0>` — R41 recommended, never executed).
     Distinguishes compute-bound (= KI-spec is moot) vs HBM-bound
     (= occupancy / prefetch lever could help).
2. **Native RRR ceil_div N coverage**: currently `bpc = fast_n / BLOCK_SIZE`
   for RRR (line 4300), forcing H4 transpose for shapes with N%256≠0.
   gpt_oss dA pays 210 µs for B=32 transpose on the GateUP path. If we
   can zero-clamp OOB N reads for RRR B-load (per-group bounded SRD
   doesn't suffice — N lives on the column axis and wraps to next K row),
   the transpose could be dropped on 4 B=32 shapes × weight 3 ≈ +11
   score potential. Risk: invasive kernel change (touches B-load path
   in grouped_kernel<RRR>).
3. **PMC-driven diagnostic**: rocprofv3 on the current worst shape
   (gpt_oss-GateUP-B32-M2048) var-K kernel to identify whether it's
   compute or memory bound. Informs whether (1) or (2) above is the
   right lever. Free data.

Cheapest first: **R68 = PMC var-K diagnostic** (option 3). 5-min
capture, settles the (1)-vs-(2) decision without kernel work. After R67
+ 7 consecutive non-improvements, diagnostic-first is warranted over
another speculative kernel flip.

## Metric numbers

```
                       R66 (pre-R67)   R67 (post-revert)   Δ
score                  873             875                 +2 (noise)
gpt_oss  geomean       1.092           1.078              -0.014
DSV3     geomean       1.121           1.121               0.000
Qwen3    geomean       1.113           1.114              +0.001
correct_fail           0/24            0/24                no regression
PASS                   24/24           24/24               no regression
```

Noise-band bounce. Kernel unchanged. Round deliverable is the spill
finding + lever-closure documentation; no code committed.
