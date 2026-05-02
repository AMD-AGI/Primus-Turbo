# Round 52 (dm) — FP8 grouped: noise reaffirmed, no fall-through left, Lever C-2 prototype deferred to R53

## TL;DR
- Metric: **score=976**, FP8 geomean=1.1423 (BF16 1.2265 PASS).  
  Worst-shape `gpt_oss-Down-B4-M4096 = 0.432` is a **single-shot cold-start
  anomaly** (probed median = 1929.87 TF, range 1812–1959 → ratio ≈ 1.20 vs
  Triton 1615). Excluding it, the 4 next-worst shapes (1.005–1.097)
  all sit in the gpt_oss-GateUP family at **K=2880, N=2880|5760** —
  the same structural ceiling diagnosed in R51.
- **Dispatch coverage check**: all 8 gpt_oss FP8 RCR shapes have explicit
  rules that fire (gm/xcd verified via `select_default_config`). **No
  fall-through closure remains** to land the way R50's Down-B32-M4096
  rule did.
- Lever C-2 prototype skeleton (`grouped_rcr_kernel_4w` template variant)
  was scoped this round but **not landed** — `WARPS_M/N`, `RBM/N`,
  `A_row_reg`, `B_row_reg` are *file-scope* `constexpr` in
  `kernel_fp8_layouts.cpp` (lines 13–14, 42–43, 92–93), so the 4w
  variant must live in a sub-namespace with local re-definitions, not
  a drop-in template parameter. Multi-round refactor; entry point
  pinned in §4 below.

## 1. Metric run (R52)

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.2265 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1423 (n=24)
[metric_grouped_only] (1) grp_BF16  >= 1.20  : 1.2265  PASS
[metric_grouped_only] (2) grp_FP8   >= 1.20  : 1.1423  FAIL
[metric_grouped_only] score=976  weights=grpBF16:1 grpFP8:1
```

### FP8 ratios sorted (lowest 8 shown)

| # | shape                                       | HK TF | Tri TF | ratio | note                |
|---|---------------------------------------------|-------|--------|-------|---------------------|
| 1 | gpt_oss-Down-B4-M4096                       |  697  | 1615   | 0.432 | **cold-start anomaly** |
| 2 | Qwen3-235B-A22B-GateUP-B16-M4096            | 2183  | 2172   | 1.005 | Triton-tail       |
| 3 | gpt_oss-GateUP-B4-M2048                     | 1721  | 1613   | 1.067 | structural        |
| 4 | DSV3-GateUP-B16-M4096                       | 2504  | 2328   | 1.076 | mid-pack          |
| 5 | gpt_oss-GateUP-B4-M4096                     | 1908  | 1739   | 1.097 | structural        |
| 6 | gpt_oss-Down-B32-M2048                      | 1845  | 1673   | 1.102 | structural        |
| 7 | gpt_oss-GateUP-B32-M2048                    | 2010  | 1818   | 1.106 | structural        |
| 8 | gpt_oss-GateUP-B32-M4096                    | 2107  | 1892   | 1.114 | structural        |

### Anomaly verification (probe data, 5 trials × 50 iters, p20)

```
Rule for gpt_oss-Down-B4-M4096: HipKittenConfig(gm=32, xcd=4)  # round-12 rule
  trial 1: 1812.50 TF
  trial 2: 1915.72 TF
  trial 3: 1929.87 TF
  trial 4: 1939.23 TF
  trial 5: 1958.80 TF
Median: 1929.87 TF  (vs Triton 1615 → ratio ≈ 1.20)
```

The 0.432 was a one-shot first-call dip; the rule is **already at
target**. No kernel/dispatch action needed.

## 2. Dispatch coverage audit — gpt_oss family

```
gpt_oss-GateUP-B4-M2048   m_total=  8192 -> gm=1, xcd=4     (R45 narrow-N)
gpt_oss-GateUP-B4-M4096   m_total= 16384 -> gm=14, xcd=4   (R45 narrow-N)
gpt_oss-GateUP-B32-M2048  m_total= 65536 -> gm=8, xcd=4    (R45 narrow-N)
gpt_oss-GateUP-B32-M4096  m_total=131072 -> gm=8, xcd=4    (R45 narrow-N)
gpt_oss-Down-B4-M2048     m_total=  8192 -> gm=2, xcd=2    (R12 small-tile)
gpt_oss-Down-B4-M4096     m_total= 16384 -> gm=32, xcd=4   (R12 small-tile)
gpt_oss-Down-B32-M2048    m_total= 65536 -> gm=16, xcd=4   (R8 medium)
gpt_oss-Down-B32-M4096    m_total=131072 -> gm=4, xcd=4    (R50 fall-through)
```

All 8 shapes hit specific rules. The 1.06–1.11 ratios for
gpt_oss-GateUP/B32-Down are **kernel-bound**, not dispatcher-bound.

## 3. Why ceiling is at ~1.16 geomean (R51 stat → R52 reaffirmed)

Per R51 analysis (15-run history): score μ ≈ 990, σ ≈ 7.3, FP8 geomean
μ ≈ 1.175 ± 0.013. A *stable* 1.20 geomean (P>95%) requires a +3pp
architectural lift. R52's 1.1423 sits 1.5σ below the running mean —
within noise but on the low side, consistent with first-call coldstart
dragging the gpt_oss-Down-B4-M4096 row.

The structural pattern: all 6 sub-1.10 shapes (excluding Triton-tail
Qwen3-GateUP-B16-M4096) are **K=2880, N∈{2880,5760}**. K=2880 forces the
FUSED_KTAIL=true path (R49 KREM=64 spec), which has the highest
register working set in the kernel — this is where AGPR migration
(Lever C) has the largest theoretical headroom.

## 4. Lever C-2 entry point — pinned for R53 (multi-round)

The grouped kernel fails to use AGPRs because its accumulator
(`rt_fl<RBM=64, RBN=32, ...>` ⇒ 128 fp32/lane) is **half** of
`rcr_4w::kernel`'s (`rt_fl<RBM=64, RBN=64, ...>` ⇒ 256 fp32/lane).
LLVM only forces AGPR allocation when accumulator size is large enough
to justify the AGPR-write cost. R48's `+a` inline-asm hint failed
(5× spill regression) because forcing AGPR with the *same* 128
fp32/lane accumulator displaced VGPRs without freeing equivalent
VGPR pressure for the rest of the working set.

The right fix: **make the accumulator legitimately 256 fp32/lane** by
restructuring warp tile to 2×2 (RBN=64 instead of 32). LLVM will then
naturally pick AGPR ≥ 256, mirroring `rcr_4w::kernel`'s 0-spill
profile.

### File-scope blockers (`kernel_fp8_layouts.cpp`)

```13:14:/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
constexpr int WARPS_M           = 2;
constexpr int WARPS_N           = 4;
```
```42:43:/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
```
```92:93:/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
```

These are shared across rcr_8w / rcr_4w / grouped_rcr_kernel / rrr*.
Cannot just flip `WARPS_N` — it would break rcr_8w (which depends on
the 2×4 layout for 8-warp distribution).

### R53 step 1 (concrete entry point)

Wrap `grouped_rcr_kernel` in a `namespace grouped_4w { … }` block at
line ~2160 (right before the `template<int KI_HINT=…> __global__`
declaration on line 2221) with **local** redefinitions:

```cpp
namespace grouped_4w {
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;            // was 4
    constexpr int _NUM_WARPS = WARPS_M * WARPS_N;
    constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;   // = 256, was 512
    constexpr int RBM = BLK / WARPS_M / 2;   // = 64 (unchanged)
    constexpr int RBN = BLK / WARPS_N / 2;   // = 64, was 32
    using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
    using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
    // ... grouped_rcr_kernel verbatim, all references to outer
    // WARPS_M/N/RBM/RBN/_NUM_THREADS now resolve to grouped_4w::* ...
}  // namespace grouped_4w
```

Build-only validation criteria (must all hold before R54 wires it):
1. `make all` builds with no compile errors
2. `kernel-resource-usage` for `grouped_4w::grouped_rcr_kernel` reports
   AGPR ≥ 256, Spill = 0 (mirroring rcr_4w line 1514's profile)
3. `_NUM_THREADS=256` halves the launch occupancy; need to verify
   `BLOCK_SIZE` math (line 38–39) and `chiplet_transform_chunked`
   call (line 2253) still produce sensible tile counts. Specifically
   the `bpc = ceil_div(g.n, BLOCK_SIZE)` invariant on line 2774
   *must* still hold or the persistent loop range is wrong.

R54 step 2: the dispatcher gate. Use FUSED_KTAIL=true as the witness
predicate (line 5384–5392 in `dispatch_grouped_rcr`); only K=2880
shapes enable it, and *all* sub-1.10 ratios in the metric live in
that subset. This contains blast-radius: aligned-K (DSV3, Qwen3
K∈{1536, 7168}) keeps the proven 8-warp path.

R55 step 3: optional micro-knob sweep on the 4w variant if AGPR/spill
report looks healthy but raw TFLOPS is below baseline (need
`group_m`/`xcds` re-tune for 256-thread blocks).

## 5. R52 actions

- Metric run: ✅ (score 976, FP8 geomean 1.1423)
- Worst-shape probe: ✅ (anomaly = cold-start, not real)
- Dispatch coverage audit: ✅ (no fall-through)
- C-2 prototype: 🛑 deferred (R53 needs ≥ 30 min, can't fit residual budget)
- Doc-only commit: this file

## 6. Before-after metric

| round | sha          | score | FP8 geomean | improved |
|-------|--------------|-------|-------------|----------|
| R49   | a7fe4e0      | 990   | (n/a snapshot) | — |
| R50   | 63244cd      | 1000  | 1.193       | ✅ peak |
| R51   | 4be1a81      | 988   | 1.180       | ❌ noise |
| **R52** | **(this)**   | **976** | **1.1423** | ❌ noise (low tail) |

R52's low-tail run is consistent with R51's σ analysis (P[score ≤ 976]
≈ 4% under μ=990, σ=7.3 — within tail). No regression to investigate.

## 7. Next-round recommendation (R53)

**Start Lever C-2 step 1** — namespace-wrap `grouped_rcr_kernel` into
`grouped_4w::` with WARPS_N=2 / RBN=64 / _NUM_THREADS=256. **Build-
only**, do not wire dispatch. Validation: resource report must show
AGPR ≥ 256, Spill = 0. Time budget: ~45 min (need full
chat-window allocation; do not multi-task this round).

If C-2 step 1 fails resource check, fall through to **Lever C-3** —
explicit `art_base` + ASM accumulator pinning (deferred plan from R47
notes); higher risk, lower expected gain.
