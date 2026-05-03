# Round 49 — BF16 grouped, fwd LDS bank-conflict PMC diagnostic — LEVER A3 FALSIFIED BY ZERO-CONFLICT EVIDENCE

## Goal coming in

R48 falsified the var-K KI specialisation lever (R47 backup #2):
compile-time KI on `grouped_var_k_kernel` spilled 14-17 VGPRs in
both unroll-2 and unroll-1 schedules, costing -27 score on the
weighted wall metric. R48 next-action surface ranked:

1. **Forward LDS swizzle audit for K=2880** (task body lever A3,
   un-tried). Use `rocprofv3 SQ_LDS_BANK_CONFLICT` PMC on the
   worst metric shape (`gpt_oss-GateUP-B32-M2048`, ratio 1.051,
   weight 3x) forward kernel. If high → swizzle change for K%128 != 0
   path could lift fwd ratio without touching K-loop control flow.
   If low → lever A3 is exhausted, fall through to (2)/(3).
2. DSV3-GateUP dB var-K dispatch retry (R47 backup, weight 1x).
3. Store batching 4 → 1 in `grouped_var_k_kernel` epilog (correctness
   risk medium).

R49 executes step 1: capture the PMC, decide.

## Hypothesis (R49)

The HK BF16 forward grouped kernel uses LDS-staged double-buffered
A/B tile streaming with a `st_32x16_s` swizzle tuned for the 128-wide
`K_TWO_TILE` boundary. For K=2880 (gpt_oss), `K_TWO_TILE=128` divides
2880 = 22.5 × 128 → `fast_k = 22 × 128 = 2816`, so the LDS-K-tail
correction kernel runs to cover `K=[2816, 2880)`. That K-tail kernel
re-uses the same swizzle on a partial K-stripe, which **may** generate
LDS bank conflicts because `K_REM = 64` doesn't align with the
swizzle's 128-bank pattern.

If true, `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE` would be
substantially higher on gpt_oss (K%128 != 0) than on DSV3 (K%128==0,
control). A swizzle tuned for K_REM=64 could close the gap without
touching K-loop control flow.

## Evidence

`/tmp/probe_r49_lds_workload.py` runs 50 fwd-only iterations of
`turbo.ops.grouped_gemm(..., trans_b=True)` (HIPKITTEN backend,
auto_tune off) with DSV3 cold-start warmup. Filter rocprofv3
counter collection to the active `grouped_kernel` instantiations
via `--kernel-include-regex "grouped_kernel"`.

PMC counters captured: `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_IDX_ACTIVE`,
`GRBM_GUI_ACTIVE`.

```
                                kernel signature              SQ_LDS_BANK_CONFLICT  SQ_LDS_IDX_ACTIVE   GRBM_GUI_ACTIVE
                                                              (per-call)            (per-call)          (per-call)
gpt_oss-GateUP-B32-M2048 fwd    grouped_kernel<L=RCR, KI=0,
                                    FUSED_KTAIL=true>         0.000e+00              2.022e+08          2.605e+07
DSV3-GateUP-B32-M2048    fwd    grouped_kernel<L=RCR, KI=112,
                                    FUSED_KTAIL=false>        0.000e+00              3.546e+10          3.624e+07
```

Both kernels have **exactly zero LDS bank conflicts** despite
> 200 M LDS-active cycles each. The HK BF16 grouped kernel's
`st_32x16_s` swizzle is conflict-free on both K%128==0 (DSV3) and
K%128 != 0 (gpt_oss + path B fused K-tail) paths.

The fused K-tail path (gpt_oss `KI=0, FUSED_KTAIL=true`) reads from
HBM directly into register tiles via per-lane `buffer_load_b128`
without touching LDS for the K-tail at all (round-5 path B design,
documented in `kernel_bf16_dynamic.cpp:801-844`), so even if the
LDS swizzle were imperfect for K_REM=64, the K-tail correction would
not generate LDS conflicts.

## Falsification consequence

R49 closes:

* **Forward LDS swizzle audit for K=2880** (task body lever A3 / R48
  backup #1). Both gpt_oss (K%128 != 0, fuse path) and DSV3
  (K%128==0, baseline) show 0 LDS bank conflicts under the existing
  `st_32x16_s` swizzle. The lever is FALSE — there are no LDS bank
  conflicts to optimize away on the fwd kernel.

The remaining fwd-side levers all require deeper structural changes
(MFMA scheduling, register pressure tradeoffs, HBM bandwidth
efficiency) and are higher-risk than the dispatch / kernel-body
levers in the bwd path.

## R50 next-action surface

Three candidates remain (R47 + R48 backup), shifted in priority by
the R49 LDS data:

1. **DSV3-GateUP dB var-K dispatch retry** (R47 backup #2 / R45
   backup). R24 dropped `xcds=0` due to allclose drift; re-sweep with
   `xcds ∈ {1, 2, 4, 8}` on (tiles_m=16, tiles_n=28) cells. Smaller
   upside (DSV3-GateUP weight 1x, current ratio 1.13-1.15) but
   cleanest dispatch surface remaining.
2. **Store batching 4 → 1** in `grouped_var_k_kernel` epilog (R47
   backup #3). Refactor the 4 `store_c_tile_mn_masked_grouped` calls
   into one larger MMA-tile-wide store. Reduces per-tile bounds-check
   overhead. Correctness validation cost is moderate (mask handling
   for the 4 sub-tiles must stay bit-identical), but the
   per-tile-cost win could be measurable on the gpt_oss B=32 shapes
   where tiles_per_group is large (242 for GateUP, 121 for Down).
3. **PMC-driven fwd analysis: MFMA utilization on gpt_oss vs DSV3**.
   Now that LDS bank conflicts are eliminated, the next free PMC
   datapoint is `SQ_INSTS_MFMA / SQ_THREAD_CYCLES_VALU` to check if
   the fuse path has lower MFMA throughput than the non-fuse path.
   If the fuse path is < 90 % MFMA-saturated, there's room for
   schedule tweaks (round-5 path B's per-lane buffer_load may be
   stalling MFMA on `vmcnt`).

Recommended for R50: **start with (3)** — free PMC data, single-shape
single-kernel capture (~10 sec), gives clear go/no-go signal for
deeper fwd-side work. If MFMA util ≥ 90 % → fuse path is saturated
→ fall through to (1) DSV3-GateUP dB var-K dispatch retry. If MFMA
util < 90 % → there's a real fwd lever.

## Action

* HipKittens: no change.
* Primus-Turbo: 1 commit (this diagnostic note).
* PMC capture artifacts: `/tmp/r49_pmc_gpt_oss/pmc_1/...` and
  `/tmp/r49_pmc_dsv3/pmc_1/...`. Probe driver:
  `/tmp/probe_r49_lds_workload.py`. PMC config: `/tmp/r49_pmc.txt`.
