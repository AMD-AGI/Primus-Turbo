# Round 53 (dm) — FP8 grouped: resource baseline captured, **FUSED_KTAIL is NOT spill-bound**, R54 C-2 acceptance criteria revised

## TL;DR
- Metric: **score=983**, FP8 geomean=1.1600 (BF16 1.2326 PASS, FP8 FAIL).
  Worst-shape `gpt_oss-Down-B4-M4096 = 0.543` is again a single-shot
  cold-start anomaly (R52 probe already showed median 1929 TF / ratio 1.20).
- Captured **concrete baseline `kernel-resource-usage`** for all four
  `grouped_rcr_kernel` template instances + the reference `rcr_4w::kernel`.
- **Key revision to R52's C-2 plan**: the `FUSED_KTAIL=true` instances
  (which serve gpt_oss K=2880 — the *exact* shapes with 1.07-1.11
  ratios) have **LOWER** VGPR spill (34-37) than `FUSED_KTAIL=false`
  (38-54). gpt_oss is thus **NOT spill-bound** at the LLVM level —
  the structural ceiling is **compute density**, not register file
  pressure. R54's C-2 acceptance criteria revised below.

## 1. Metric (R53)

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.2326 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1600 (n=24)
[metric_grouped_only] (1) grp_BF16  >= 1.20  : 1.2326  PASS
[metric_grouped_only] (2) grp_FP8   >= 1.20  : 1.1600  FAIL
[metric_grouped_only] score=983  weights=grpBF16:1 grpFP8:1
```

Worst FP8 (sub-1.10):

| shape                                  | HK   | Tri  | ratio | note               |
|----------------------------------------|------|------|-------|--------------------|
| gpt_oss-Down-B4-M4096                  |  878 | 1617 | 0.543 | cold-start anomaly |
| Qwen3-GateUP-B16-M4096                 | 2236 | 2150 | 1.040 | Triton-tail        |
| gpt_oss-GateUP-B4-M2048                | 1725 | 1614 | 1.069 | structural         |
| gpt_oss-GateUP-B32-M2048               | 2003 | 1829 | 1.095 | structural         |
| gpt_oss-Down-B32-M2048                 | 1848 | 1676 | 1.102 | structural         |
| gpt_oss-GateUP-B4-M4096                | 1927 | 1744 | 1.105 | structural         |
| gpt_oss-GateUP-B32-M4096               | 2111 | 1908 | 1.106 | structural         |

Same gpt_oss-GateUP K=2880 cluster as R52. Per R51 σ-stats (μ=990,
σ=7.3): score 983 sits 0.96σ below mean — well within noise floor.

## 2. Baseline kernel-resource-usage (HK SHA `6c52d017`)

Built clean with `THUNDERKITTENS_ROOT=/workspace/code/HipKittens make all`,
extracted from `-Rpass-analysis=kernel-resource-usage` remarks:

| kernel                                              | VGPR | AGPR | SGPR | Spill | Scratch | Occ |
|-----------------------------------------------------|------|------|------|-------|---------|-----|
| `rcr_4w::kernel` (line 1284) — **reference**        | 198  | **256** | 46   | **0** | 0       | 1   |
| `grouped_rcr_kernel<0, false, false>` (NM=F, FK=F) | 256  | 0    | 64   | **54** | 220     | 2   |
| `grouped_rcr_kernel<0, true , false>` (NM=T, FK=F) | 256  | 0    | 66   | 38    | 156     | 2   |
| `grouped_rcr_kernel<0, false, true >` (NM=F, FK=T) | 256  | 0    | 77   | **34** | 140     | 2   |
| `grouped_rcr_kernel<0, true , true >` (NM=T, FK=T) | 256  | 0    | 79   | 37    | 152     | 2   |

**FK=F serves DSV3/Qwen3 (K=4096/7168/1536, K-aligned).
FK=T serves gpt_oss (K=2880).**

## 3. Diagnosis revision: gpt_oss is NOT spill-bound

R47-R52 hypothesis: gpt_oss low-ratio caused by VGPR spill →
register-file pressure → MFMA pipeline stalls. **This hypothesis
needs revision.**

Evidence:

1. **FK=T instances have the LOWEST spill** (34, 37) of all four
   grouped variants. If spill were the dominant bottleneck, FK=T
   would be the FASTEST (relative-to-Triton). It is the SLOWEST.
2. FK=F (NM=F, NM=T) has spill 54 / 38 — those serve DSV3/Qwen3
   K-aligned shapes which sit at 1.18-1.31 ratio. So MORE spill
   (54) → BETTER ratio. Negative correlation.
3. `rcr_4w::kernel` runs occupancy=1, not 2. Spill=0 is from AGPR
   not from compute density. gpt_oss occupancy=2 already (matches
   most well-performing kernels).

**Real ceiling: compute density.** With WARPS_N=4, each warp
computes a 64×32 output tile (16×16×128 MFMA × 4×2 fragments =
8 MFMAs per tile). With WARPS_N=2 (rcr_4w), each warp computes
a 64×64 tile = 16 MFMAs/tile = **2× MFMA throughput per warp**.

For K=2880 (FUSED_KTAIL=true), the ki=22 (2880/128) main loop
issues 22 × 8 = 176 MFMAs/warp, vs hypothetical 4w-style 22 × 16
= 352 MFMAs/warp. The 4w variant amortizes the mfma-issue
latency over 2× more compute → closer to peak MFMA throughput.

## 4. R54 C-2 acceptance criteria (revised)

**Old criterion (R52 doc)**: AGPR ≥ 256, Spill = 0.

**New criterion (R53 evidence-based)**: 

a) **Mandatory**: AGPR ≥ 256, Spill ≤ 8 (acknowledging FK=T already
   has 34 spill, so anything ≤ FK=T baseline is acceptable; the
   point of C-2 is NOT to fix spill, it's to double MFMA density).
b) **Mandatory**: Occupancy stays ≥ 1 (waves/SIMD). 4w halves
   threads/block (256 vs 512) but the per-block VGPR/AGPR usage
   doubles per warp; net occupancy must remain feasible.
c) **Diagnostic**: per-warp MFMA count in the main loop must
   double (16x16x128 tile count for `rcr_mma` in
   `grouped_rcr_kernel` body must go from 8 to 16 per K-step).
d) **End-to-end**: gpt_oss-GateUP-B32-M2048 ratio ≥ 1.15 (current
   1.095 ± noise) on a microbench probe (5 trials × 50 iters, p20).
   This is the load-bearing shape.

If (a)+(b)+(c) hold but (d) fails (e.g. dispatcher mis-tuning for
256-thread blocks), R55 sweep micro-knobs.

If (a) fails — LLVM still picks AGPR=0 — fall through to Lever
C-3 (explicit `art_base` + ASM, see R47 notes).

## 5. R54 implementation entry point (preserved from R52)

`namespace grouped_4w { ... }` wraps lines 2221-2885 of
`/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`.
Local re-defs at namespace top:

```cpp
namespace grouped_4w {
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;            // was 4
    constexpr int _NUM_WARPS = WARPS_M * WARPS_N;
    constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;   // = 256
    constexpr int RBM = BLK / WARPS_M / 2;   // = 64
    constexpr int RBN = BLK / WARPS_N / 2;   // = 64, was 32
    using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
    using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
    using A_row_reg_32x64 = rt_fp8e4m3<RBM, 64, row_l, rt_32x64_s>;
    using B_row_reg_32x64 = rt_fp8e4m3<RBN, 64, row_l, rt_32x64_s>;
    // ... template <int KI_HINT, bool N_MASKED_STORE, bool FUSED_KTAIL>
    // grouped_rcr_kernel { /* verbatim from line 2223 */ }
    // ... extern instantiations at line 2882-2885 with new mangled names
}
```

`dispatch_grouped_rcr` (line 5321) needs no change in R54 step 1
(prototype only, not wired). R55 wires `grouped_4w::grouped_rcr_kernel`
into `dispatch_grouped_rcr` gated on `FUSED_KTAIL=true` (gpt_oss only).

## 6. Other potential structural levers (parked)

If C-2 step 1 falsifies the AGPR hypothesis, ranked alternatives:

- **Lever F** (Qwen3 K=1536 short-K variant): only 2 of 8 Qwen3
  shapes are sub-1.20 in R53; not the highest-ROI target.
- **Lever D** (16x16x128 → 32x32x64 cell-shape): half-ported in
  R63-R64 of prior session (see SKILL.md commit notes); unfinished.
- **Lever B** (triple-LDS ping-pong): ki for K=2880 is 22, may not
  benefit from deeper pipeline; lower priority.

## 7. R53 actions

- Metric run: ✅ score 983 (within R51 noise σ)
- Baseline resource report: ✅ captured for all 4 grouped variants
  + rcr_4w reference
- Diagnosis revised: ✅ FK=T is NOT spill-bound; gpt_oss bottleneck
  is **compute density (MFMAs/warp)**
- C-2 acceptance criteria updated for evidence
- Doc commit only this round

## 8. Next round (R54)

Start C-2 step 1: namespace-wrap, build, capture resource report,
verify revised acceptance criteria. Need full chat-window allocation
(~45 min). HK code change only, no Primus-Turbo touch in step 1.
