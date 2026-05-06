# Round 85 — bf16 grouped GEMM weighted wall

> **Context:** auto_optimize round 8 / 100, MI355X. Continuation of
> R84's plan: PMC walk on var-K LDS access pattern, then a candidate
> single-round lever (KI specialization for the var-K dB kernel).

**Status:** PMC walk on var-K LDS **DONE** + **R85 KI=32 / KI=64
var-K specialization FALSIFIED** (-30 score, 14-18 VGPR spill,
MfmaU% regressed 42.4 → 37.4% on the KI=32 path). Diagnostic
discovery: var-K kernel's 16% LDSBC traces to `st_32x16_s` swizzle
having 2.0 bank conflicts per LDS instruction; refuting this is a
multi-round custom-swizzle problem (R74 already fell over).

| run | weighted score | gpt_oss geomean |
|-----|---------------|-----------------|
| baseline (R84 commit b057fae)             | 882-884 (±2 noise) | 1.093 |
| R85 KI=32/64 var-K spec (wired)           | 850 / 851 / 853    | (regressed) |
| post-revert (HK + Primus, file-clean)     | 883                | 1.093 |

KI=32 / KI=64 var-K spec: 14 / 18 VGPR spills, MfmaU% drops. **REVERT.**

## Part A: PMC walk on var-K LDS access pattern

Ran rocprofv3 with raw `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_IDX_ACTIVE`,
`SQ_INSTS_LDS` counters on gpt_oss-Down-B4-M2048 (13 iter, fwd+bwd):

| kernel                        |    BankConfl |   IdxActive |   LDS_Insts | Confl/Inst | LDSBC% |
|-------------------------------|--------------|-------------|-------------|------------|--------|
| `grouped_var_k_kernel<RCR>` (CRR layout) | 360,185,856 | 720,882,688 | 180,231,168 | **2.00**   | 16.0   |
| `grouped_kernel<RCR,0,FUSE>` (RCR fwd+dA) |       0     | 338,890,240 |  84,987,136 |  0.00      |  0.0   |
| `grouped_kernel<RRR,64>` (DSV3 RRR warmup)|  58,720,256 | 235,935,232 |  59,214,336 | **0.99**   |  9.3   |
| `grouped_kernel<RCR,112>` (DSV3 RCR warmup)|       0    | 176,772,608 |  44,325,376 |  0.00      |  0.0   |

**Pattern:** the `Confl/Inst` ratio scales **linearly** with the
number of `st_32x16_s` swizzle tiles in the layout:

* RCR (ST_A=`st_16x32_s`, ST_B=`st_16x32_s`)        → 0 st_32x16 → 0 conflicts/inst
* RRR (ST_A=`st_16x32_s`, ST_B=`st_32x16_s`)        → 1 st_32x16 → 1 conflict/inst (~0.99)
* CRR (ST_A=`st_32x16_s`, ST_B=`st_32x16_s`)        → 2 st_32x16 → 2 conflicts/inst (~2.00)

This DEFINITIVELY confirms that **`st_32x16_s` swizzle pattern emits
~1 LDS bank conflict per LDS read instruction** on CDNA4 / MI355X.
The current swizzle (`((offset % 1024) >> 9) << 4`) only XORs the
upper-half rows (offset bit 9 set) with a half-row offset (16
bytes). That breaks the upper-half ↔ lower-half alias but does NOT
break the within-half alias for `ds_read_b128` strided access — for
a 32-row tile of bf16 cols=16, the natural pattern (lanes 0,8,16,24
all hitting bank-cycle 0 at offsets 0, 128, 256, 384) is left with
a 4-way bank conflict that the current swizzle doesn't address.

**Cost:** 16% LDSBC% on var-K means ~16% of the kernel's GPU time
is wasted on bank conflict stalls. var-K kernel takes ~4.18 ms per
iteration on gpt_oss-Down-B4-M2048; eliminating conflicts could
recover ~660 µs (var-K dur 4180 → 3520) → ~+8% on total fwd+bwd
wall → ~+8% gpt_oss-Down ratio (1.05 → 1.13) → ~+5-7 weighted score.
Plus ~+3-4 score on RRR (gpt_oss-GateUP native R80 path), since
RRR uses st_32x16_s for B. Total potential: ~+8-12 score.

**The lever:** custom st_32x16_s swizzle that breaks the within-half
4-way alias (e.g., XOR bits 7-8 of offset into bits 4-5 to spread
lanes 0,8,16,24 across distinct bank cycles). R74 attempted a
related lever (`st_64x32_padded_b128`) and saw 66 VGPR spills due
to the LARGER tile shape; a more surgical change (modify swizzle
pattern within st_32x16, no shape change) might avoid the spill
while still reducing conflicts. **Multi-round work**: requires
correctness verification (the swizzle must be consistent between
HBM→LDS prefill and LDS→reg load, and `prefill_swizzled_offsets`
must match), per-shape PMC verification (RRR + CRR), and resource
report verification.

## Part B: R85 single-round lever — KI=32 / KI=64 var-K spec

Hypothesis: var-K's MfmaUtil=42.5% (vs DSV3 KI=64 RRR's 74%) is
partly driven by KI=0 dynamic K-loop with `#pragma unroll 2`. Adding
KI=32 / KI=64 specs covers all metric shapes (uniform M_g ∈ {2048,
4096} → ki_g = 32 or 64), enabling compile-time-bounded full unroll
that should let the compiler schedule the MFMA pipeline with fewer
gaps. var-K body has NO FUSE epilog (unlike R83/R84's RCR FUSE
attempts), so live-state should be smaller.

**Implementation:**
* HK kernel: `grouped_var_k_kernel<32>` and `<64>` instantiations,
  `launch_grouped_var_k` switch on `g.ki_max`, new `m_per_group`
  arg added to `grouped_var_k_crr_fn` binding.
* Primus: `GroupedGEMMVariableKHipKittenBackend.execute` passes
  `_avg_group_m(M_total, bs)` as the new 7th arg.

**Initial bug:** field-order error in the C++ `grouped_var_k_layout_globals`
struct initializer — `ki_max` was placed where `bpr` should be. PMC
showed `grouped_var_k_kernel<0>` running despite the new specs being
compiled. Fixed by re-ordering the C-style brace init list to match
the struct's field declaration order (`G, M_total, n, k, group_m,
num_xcds, bpr, bpc, ki_max, fast_n, fast_k`).

**After fix:** PMC confirmed `grouped_var_k_kernel<32>` actually
running on gpt_oss-Down-B4-M2048 (M_g=2048).

**Build report:**

| kernel                     | VGPRs | Spill | Scratch B/lane | Occ |
|----------------------------|-------|-------|----------------|-----|
| `grouped_var_k_kernel<0>`  | 256   | 0     | 0              | 2   |
| `grouped_var_k_kernel<32>` | 256   | **14**| 60             | 2   |
| `grouped_var_k_kernel<64>` | 256   | **18**| 76             | 2   |

**PMC after KI=32 active (gpt_oss-Down-B4-M2048):**

| kernel                  | dur_us | MfmaU% |
|-------------------------|--------|--------|
| KI=0 (baseline)         | 4166.5 | 42.40  |
| KI=32 (R85)             | **4629.5 (+11%)** | **37.36 (-5pp)** |

The unroll is happening (KI=32 spec runs) but it's WORSE than the
generic KI=0 + `#pragma unroll 2` path. The 14-VGPR spill creates
more scratch ld/st traffic than the partial-unroll-with-pragma path
emits. MfmaU drops because the spilled-reg refill stalls compete
with the MFMA pipe for issue cycles.

**Metric:** 850 / 851 / 853 → -30 score vs baseline 882-884. **REVERT.**

### Verdict

The KI specialization avenue for var-K is **FALSIFIED**, completing
a 3-round pattern of failed full-unroll levers:

| round | lever                             | spill | metric | direction |
|-------|-----------------------------------|-------|--------|-----------|
| R83   | RCR FUSE KI=88 (dead — never hit) |  9    | flat   | -         |
| R84   | RCR FUSE KI=44                    | 28    | -40    | regressed |
| R85   | var-K KI=32 / KI=64               | 14-18 | -30    | regressed |

The pattern: full-unroll over the BF16 grouped/var-K bodies hits the
256 VGPR ceiling with 9-28 VGPRs of scratch traffic. The MFMA
scheduling gain from full-unroll is consistently dominated by spill
overhead. The KI specs that DID work (KI=64, KI=112 in dense GEMM,
KI=64 in DSV3 RRR — the latter at 74% MfmaU%) are on cleaner code
paths (no FUSE epilog, no var-K group bookkeeping, no per-tile
chiplet swizzle). The grouped/var-K bodies' baseline VGPR pressure
is too close to the ceiling to absorb full-unroll's added live state.

KI specialization for these bodies would require **trimming the
body's baseline VGPR footprint** before adding new unroll —
multi-round work (live-range compression of the per-tile coord
helpers, group-bookkeeping, etc.). Not a single-round lever.

## Part C: revert + clean state

Both HK kernel and Primus changes fully reverted; `git diff` clean
on both files. Final 1-run metric post-revert: **883** (back at
baseline). HK kernel diff vs R84 commit b057fae: 0 lines (only the
NFS .nfs* lock files remain).

## Part D: direction for R86

The two remaining viable BF16 grouped levers, ordered by ROI:

1. **Custom `st_32x16_s` swizzle** to break the within-half 4-way
   bank-cycle alias. Multi-round (R86: prototype swizzle on a probe
   build, verify correctness on `prefill_swizzled_offsets` matching;
   R87: PMC-verify LDSBC% drop on var-K + RRR; R88: full metric +
   commit). Potential ~+8-12 score.

2. **VGPR live-range compression in the var-K body** to make room
   for full-unroll KI specs. Multi-round (R86: profile per-pass
   register usage; R87+: trim hot-path coords / group-bookkeeping;
   R88+: re-test KI specs at a lower baseline VGPR pressure).
   Potential ~+5-8 score (independent of the swizzle lever).

R86 candidate: **start the swizzle prototype**. Smaller surface than
R74's full padded-tile replacement (single-file change in
`include/types/shared/st_shape.cuh`); concrete falsifiable hypothesis
(LDSBC% drop measurable in 1 PMC walk); no shape changes to the
kernel body.

## Files touched

* `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  — KI=32/64 var-K instantiations + `m_per_group` binding arg + dispatch
  switch + struct-init field-order fix. **REVERTED** (file matches
  R84 b057fae baseline; HK working tree clean except NFS locks).
* `/workspace/code/Primus-Turbo/primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_impl.py`
  — passed `avg_m` as 7th arg to var-K binding. **REVERTED**.
* This round note (Primus-Turbo).
* `/tmp/r8_pmc_counters.txt`, `/tmp/r8_pmc_out/`, `/tmp/r8_analyze_lds.py`,
  `/tmp/r8b_pmc_out/`, `/tmp/r8c_pmc_out/` — PMC artifacts (offline,
  not committed).
