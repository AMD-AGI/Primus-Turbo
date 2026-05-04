# Round 68 — BF16 grouped, PMC-driven diagnostic on bwd path — var-K CRR LDS bank conflict identified as the actionable lever

## Why this round

After 10 consecutive non-improvements (R59-R67), every recent lever
was speculative kernel flip + metric check — falsified by
reproducible noise. R67 closed the compile-time KI specialization
lever (`KI<48` threshold confirmed). R67's next-action suggested a
**PMC diagnostic before another speculative attack**, specifically to
answer "is the var-K dB kernel compute-bound or memory-bound?" — this
unblocks the decision between:
* `__launch_bounds__` / occupancy flip (if memory-bound)
* LDS swizzle / MFMA scheduling (if compute-bound with stalls)
* native RRR ceil_div N to drop the dA H4 transpose (independent)

R68 executes that diagnostic. **No kernel code change.**

## Target shape

`gpt_oss_20B-GateUP-B32-M2048` — lowest-progress row of the R68
baseline metric (ratio 1.047, progress 0.837, weight 3×). 24/24 PASS.
`score=874`.

Shape: `B=32, M=2048, N=5760, K=2880` (grouped RRR fwd with
`trans_b=True`, i.e. b storage [G,N,K] = MoE standard).

## Methodology

`/tmp/r68_varK_probe.py` — a 2-shape probe that does 1 warmup on
DSV3-GateUP-B16-M4096 (SKILL.md mandates DSV3-first for the HK BF16
K-tail cold-start sync-fault workaround) + 3 iters on
gpt_oss-GateUP-B32-M2048 (to get stable per-kernel timing across
3 samples).

PMC captured with rocprofv3 7.2.0 in 3 single-counter passes on GPU 3
(identical code + kernel args in each pass, so the PMC samples line up
across passes):

1. `MfmaUtil` = `SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE * SIMD_NUM)
   * 100` — % of total SIMD time the MFMA unit was busy
2. `FETCH_SIZE` — total HBM read bytes (in KB)
3. `SQ_LDS_BANK_CONFLICT` + `SQ_WAVES` — bank conflict events + wave
   count

Kernel filter: `--kernel-include-regex "grouped_kernel|grouped_var_k_kernel"`.

## Wall breakdown (per-iter, stable = iter-3 / iter-4 of the 3-iter probe)

| Kernel                                   | Layout | FUSED | KI_HINT | dur_us  | % of wall |
|------------------------------------------|--------|-------|---------|---------|-----------|
| fwd `grouped_kernel<RCR, 0, true>`       | RCR    | true  | 0       |  1670   |  28 %     |
| dA H4 `bf16_transpose_3d_kernel`         | —      | —     | —       |   360   |   6 %     |
| dA `grouped_kernel<RCR, 0, false>`       | RCR    | false | 0       |  1620   |  27 %     |
| dB `grouped_var_k_kernel<0>`             | CRR    | —     | 0       |  2060   |  34 %     |
| **Total wall**                           |        |       |         | **5910**| **95 %** (rest = allclose setup / cudaMemcpy) |

* **Bwd** (dA + transpose + dB) = 4040 us = **68 % of wall**.
* **dB var-K alone** = 34 %, making it the single largest kernel.
* **dA H4 transpose** = 6 % — non-trivial. Kills on RRR if we can do
  native ceil_div N on `grouped_kernel<RRR>`.

This reproduces R41 wall-decomp (bwd ≥ 64 %) with per-kernel
granularity. Fwd path has 28 % of wall — any fwd lever caps at
+1-3 % HK/Triton ratio per 10 % fwd kernel speedup.

## Per-kernel PMC (stable iter)

| Kernel                                      | MfmaUtil % | FETCH_SIZE KB  | GB/s est.  | LDS bank conflicts | VGPR | SGPR |
|---------------------------------------------|------------|----------------|------------|--------------------|------|------|
| fwd `grouped_kernel<RCR, 0, true>`  (FUSE)  |    61.1    | 2 117 035      |  1250      |            0       |  128 |  112 |
| dA `grouped_kernel<RCR, 0, false>` (non-fuse)|    72.1    | 2 005 814      |  1240      |            0       |  124 |   96 |
| dB `grouped_var_k_kernel<0>`        (CRR)   |    61.3    | 2 561 616      |  1260      |  **217 055 232**   |  128 |   96 |

Fwd DSV3 warmup (for reference — K=7168 RCR KI=112 fast path):

| Kernel                                    | MfmaUtil % | FETCH_SIZE KB  | LDS bank conflicts |
|-------------------------------------------|------------|----------------|--------------------|
| DSV3 fwd `grouped_kernel<RCR, 112, false>`|    72.6    | 2 842 997      |           0        |
| DSV3 dA  `grouped_kernel<RRR, 64, false>` |    73.0    | 2 752 783      |  **117 440 512**   |
| DSV3 dB  `grouped_var_k_kernel<0>`        |    63.8    | 3 632 897      |  **352 321 536**   |

Clock assumed 2.1 GHz for bytes/cycle → GB/s conversion. MI355X HBM3E
peak ≈ 8 TB/s → all 3 kernels hit **~15 % of peak HBM**. Definitively
**not HBM-bound**. (Triton Phase A1 narrative assumed K-tail split was
HBM-bound; wrong assumption.)

## Conclusions

1. **Fwd FUSE path (gpt_oss K=2880) has 11 pp MFMA-util gap vs the
   non-fuse path** (61.1 % vs 72.1 %). Root cause: FUSE adds a
   conditional K-tail iter at the end of the main loop; +16 SGPRs
   usage (96 → 112) suggests extra SGPR-resident state for the
   masking. Same LDS (both 0 conflicts), same HBM (~1.25 TB/s). The
   ~11 pp gap is most likely **MFMA-schedule bubbles introduced by
   the tail branch**, not LDS nor HBM. **Lever candidate R69-C
   (fwd)**: fuse epilog restructuring — can the tail iter share the
   SW-pipeline with the last main-iter instead of being grafted as a
   post-loop branch?

2. **dB var-K CRR kernel has 217M LDS bank conflicts per gpt_oss
   launch**, compared to **0 on RCR** kernels. DSV3 var-K is worse
   (352M). **This is the single biggest actionable finding of R68.**

   Root cause identified: `/workspace/code/HipKittens/analysis/bf16_gemm/
   mi350x/kernel_bf16_dynamic.cpp:4723-4724` —

   ```cpp
   // grouped_var_k_kernel (CRR layout)
   using ST_A = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
   using ST_B = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
   ```

   `st_32x16_s` is an UN-PADDED shared-memory tile. In contrast, the
   RCR path (line 1187-1188) uses:

   ```cpp
   using ST_A_RCR = st_bf<HALF_BLOCK_SIZE, K_STEP, rcr_padded_st_shape>;
   using ST_B_RCR = st_bf<HALF_BLOCK_SIZE, K_STEP, rcr_padded_st_shape>;
   // rcr_padded_st_shape == kittens::st_64x32_padded_b128_s (line 118)
   ```

   The PADDED shape has XOR-based swizzle + boundary padding, which
   eliminates bank conflicts for its specific LDS access pattern (per
   P23 Session 2 Dev C comment at line 85-94).

3. **dB var-K MFMA util = 61 %, 39 % of MFMA is idle while HBM is at
   15 % of peak** → **compute underutilized, stalled on LDS**. A
   padded CRR swizzle could plausibly recover the ~10 pp util gap to
   RCR's 72 %.

4. **dA H4 `bf16_transpose_3d_kernel` = 360 us = 6 % of wall** on
   all 8 gpt_oss shapes (because K_fwd=2880 % BLOCK_SIZE=256 = 64,
   RRR rejects non-aligned N → dispatcher forces RCR via transpose).
   Killing this transpose via native RRR ceil_div N coverage would
   save ~6 % of gpt_oss wall — that's weight-3 shapes, so 6 % ratio
   pt × weight 3 × 8 shapes ≈ 11-12 score points. But R11's comment
   (line 4292-4296) says RCR already has ceil_div N; RRR was left
   intentionally gated on aligned N. The "Round 2" reason comment in
   the RRR branch (not shown here) would need to be re-audited.

## Secondary data points

* **DSV3 dA RRR has 117M bank conflicts** at 73.0 % MfmaUtil — RRR
  layout also has bank conflicts from `st_16x32_s` / `st_32x16_s`
  mix, yet its MFMA util is healthier than var-K's 61 %. So bank
  conflicts alone don't fully predict MFMA util; the absolute count
  scales with kernel duration too (DSV3 dA 3353us so 117M/3.35ms =
  35M/ms conflict rate; gpt_oss var-K 217M/2.06ms = 105M/ms — **3×
  worse per-ms conflict rate**). The var-K issue is qualitatively
  different, not just amortized RRR pattern.

* **VGPR / SGPR stats** captured from the kernel-trace CSV: all 3 HK
  kernels use 124-128 VGPR + 96-112 SGPR. Fits within MI350X's
  512 VGPR / 104 SGPR budget for 2 waves/SIMD occupancy. Not a
  register-pressure issue. Confirms R41's observation that occupancy
  isn't the lever.

## Recommendation for R69

**Priority 1 (highest leverage)**: Switch `grouped_var_k_kernel` ST_A
/ ST_B from `st_32x16_s` to a padded shape (either `rcr_padded_st_shape`
rotated to match the CRR K_STEP x HALF_BLOCK_SIZE dimensions, OR a
dedicated `st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_padded_s>` trait if
one exists in `thunderkittens/include/types/shared/`).

Expected: 200M → 0 LDS bank conflicts on var-K CRR. If this recovers
~10 pp MfmaUtil on var-K (matching RCR's 72 %), dB kernel speeds up
10-15 %, which is ~3-5 % of wall ⇒ **+3-5 score points on gpt_oss
(weight 3 × 4 B=32 shapes; ~+8-12 overall score**).

Risks / validation:
* MFMA layout must be preserved — the padded trait must yield the
  same per-lane bf16_2 pattern when loaded into `rt_bf<..., col_l,
  rt_32x16_s>` / `rt_bf<..., row_l, rt_16x32_s>` registers. This is
  what kittens' `subtile_inplace` / `load` helpers are supposed to
  abstract, but P23 Session 2 Dev C only wired RCR. CRR path B load
  (line 540-549, 4580) may need a parallel wiring.
* Correctness probe: run `bench_grouped_gemm_turbo.py --dtype bf16`
  to verify dB max_abs_err / SNR unchanged.

**Priority 2 (independent, complementary)**: Investigate FUSE MFMA
schedule — why +16 SGPR + 11 pp MFMA util drop for the K-tail branch?
Cheap first step: disable `fuse_ktail_eligible` for gpt_oss g.ki=44
(R67's config but without the failed KI=44 spec) to measure FUSE-vs-
non-fuse wall on the same input. If non-fuse + external M4 K-tail
kernel is faster end-to-end than FUSE, we have a simpler fix (gate
FUSE off for g.ki < N threshold).

**Priority 3**: Audit whether the RRR branch can safely relax to
ceil_div N (would drop dA transpose, -360 us on 4 gpt_oss B=32
shapes). Review round-11's rationale; compare against the round-5/6
RRR path B phantom-read history to be sure.

## Compliance check

* No kernel code change. No Primus source change. HK working tree
  clean post-diagnostic, PT has only this round note.
* No `can_handle` tightening, no per-(M,N,K) hardcode, no host sync,
  no caching.
* Correctness: all 24 metric shapes PASS at baseline (score=874).
* Metric baseline verified before diagnostic (not re-run after — PMC
  probe adds no state to the runtime, only reads counters).

## Metric snapshot

```
                       R67 post-commit   R68 baseline
score                  875               874
gpt_oss  geomean       1.078             1.076
DSV3     geomean       1.121             1.119
Qwen3    geomean       1.114             1.114
correct_fail           0/24              0/24
PASS                   24/24             24/24
```

No code change in R68 → metric identical within noise.

## R68 deliverable

Diagnostic data commit only. This unblocks R69 from speculative
kernel flipping — R69 has a concrete, PMC-grounded target (var-K CRR
LDS swizzle) with quantitative expected yield (+8-12 score).

## Artifact locations (kept for R69 cross-reference)

* `/tmp/r68_ktrace/chi2894/3087990_kernel_trace.csv` — full kernel
  timeline
* `/tmp/r68_pmc/chi2894/3088930_counter_collection.csv` — MfmaUtil
* `/tmp/r68_pmc2/.../counter_collection.csv` — FETCH_SIZE
* `/tmp/r68_pmc3/.../counter_collection.csv` — SQ_LDS_BANK_CONFLICT
  + SQ_WAVES
* `/tmp/r68_varK_probe.py` — the 2-shape 1+3-iter probe used for
  all 4 captures
