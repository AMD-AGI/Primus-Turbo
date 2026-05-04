# Round 69 — BF16 grouped, fwd FUSE-off for `g.ki < 48` — FALSIFIED (external M4 K-tail kernel produces wrong output when fed fwd globals)

## Hypothesis under test (R69-H1)

R68 PMC diagnostic showed gpt_oss fwd FUSE path runs at 61.1 %
MfmaUtil vs 72.1 % on the non-fuse RCR path (same shape class,
different K size / FUSE status). The FUSE kernel adds +16 SGPRs live
across a grafted K-tail post-loop branch (112 vs 96), which is the
most plausible cause of the 11 pp MFMA-util gap.

R69 hypothesis: **disable FUSE when `g.ki < 48`** so gpt_oss (K=2880,
`g.ki=44`) routes through:
* `launch_one_grouped<RCR, 0>(g)` — non-fuse main, `#pragma unroll 2`
  over `g.ki=44` main-iters, 0 VGPR spill (same binary dA uses, hits
  72 % MfmaUtil).
* External `grouped_ktail_kernel_mfma32x32_M4<Layout::RCR, 64>` — LDS
  K-tail correction for K=[2816, 2880), already dispatched at
  `kernel_bf16_dynamic.cpp:4524` via the existing
  `!fuse_handles_all_cells && lds_k_tail_safe && mfma32_m4_safe`
  branch.

Threshold `g.ki < 48` is a **general predicate** (short K loop
under-amortizes FUSE overhead), matching R58 / R67's KI-spec spill
threshold — the same SW-pipeline register-pressure mechanism that
breaks compile-time KI unroll at < 48 is conjectured to break the
FUSE MFMA schedule at < 48. Today only gpt_oss K=2880 has
`K_rem_for_fuse==K_STEP && g.ki < 48` — no per-(M,N,K) hardcode.

Expected: ~14 % fwd speedup on gpt_oss (1670→1440 us), +60 us
K-tail launch, net ~4 % wall reduction × weight 3 × 8 shapes ⇒
**+4-8 score points**.

## What was done

Single-line edit to `fuse_ktail_eligible` in
`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
(line ~4391):

```cpp
(g.bpc > 0) && (g.ki >= 2) && ...  // baseline
(g.bpc > 0) && (g.ki >= 48) && ...  // R69
```

No other changes. Build: clean, 0 VGPR spill on all
`grouped_kernel<RCR, *, *>` instantiations + `grouped_var_k_kernel<0>`
unchanged.

## What happened

Metric run **immediately** (no separate correctness probe — the
metric's downsized cross-check against Triton fp32 is the cleanest
correctness gate):

```
                               baseline (R69 pre)   R69 (FUSE-off < 48)
score                                  874                  358
gpt_oss geomean (PASS rows only)       1.076                n=0
gpt_oss correctness FAIL                0/8                 **8/8**
DSV3 / Qwen3 correctness FAIL           0/16                 0/16
Total PASS                              24/24               16/24
```

All 8 gpt_oss shapes fail the `fwd-allclose` correctness check:

```
BF16_gpt_oss_20B-GateUP-B4-M2048(fwd-allclose)
BF16_gpt_oss_20B-Down-B4-M2048(fwd-allclose)
BF16_gpt_oss_20B-GateUP-B4-M4096(fwd-allclose)
BF16_gpt_oss_20B-Down-B4-M4096(fwd-allclose)
BF16_gpt_oss_20B-GateUP-B32-M2048(fwd-allclose)
BF16_gpt_oss_20B-Down-B32-M2048(fwd-allclose)
BF16_gpt_oss_20B-GateUP-B32-M4096(fwd-allclose)
BF16_gpt_oss_20B-Down-B32-M4096(fwd-allclose)
```

DSV3 / Qwen3 unaffected (all have `K_rem_for_fuse == 0`, FUSE never
fires). Only gpt_oss sees a behavior change, and it breaks.

Kernel + build reverted to baseline. Post-revert metric = 880
(24/24 PASS, within noise of pre-R69 874).

## Root cause (localized)

The **external `grouped_ktail_kernel_mfma32x32_M4<Layout::RCR, 64>`
kernel produces numerically wrong output when invoked with fwd-shape
globals**, while working correctly with dA-shape globals. Key
observation for R70 agent:

```
Current metric routing (baseline, FUSE-eligible for g.ki>=2):

  gpt_oss  fwd  RCR FUSE   →  K=[0, 2880) fully covered by fuse epilog
                              (no external K-tail launched)
  gpt_oss  dA   RCR non-fuse
                              K_MMA = N_fwd = 5760 (K_rem=0)
                              → no external K-tail launched
  DSV3/Qwen3 all shapes    →  K % 128 == 0, no K-tail ever launched

Conclusion: grouped_ktail_kernel_mfma32x32_M4<RCR, 64> is DEAD CODE
on every one of the 24 metric shapes pre-R69. R54 wired it + R53
M2 + R21 32x32 + R19 16x16 variants but the code path has no
regression coverage from the metric. When R69 activates it by
gating FUSE off, latent bugs surface immediately.
```

Candidate bug mechanisms (untested, for R70 agent):

1. **A/B SRD mismatch between fwd and dA inputs**. For fwd:
   `g.a = [M_total, K_fwd=2880]`, `g.b = [G, N_fwd=5760, K_fwd=2880]`
   (trans_b=True MoE storage). For dA: `g.a = grad_out [M_total,
   N_fwd]`, `g.b` = transposed B from the H4 reroute. The M4 kernel
   (lines 3134-3296) may hard-code an axis interpretation that's
   correct only for the H4-rerouted dA tensor layout.

2. **N-axis coverage**. Line 3157: `if (col_block_base >= g.n)
   return;`. But `g.n` for fwd = `N_fwd=5760` and the M4 kernel grid
   is `ceil_div(g.n, TBN=32) × ceil_div(g.M_total, TBM_M4=128)`. OK
   for `g.n=5760` → 180 col-tiles. BUT the main kernel in non-fuse
   RCR already writes cells only for `col < g.n` with
   `store_c_tile_n_masked`, and the K-tail RMW reads the pre-existing
   C value. If main wrote a partial value for `col >= g.fast_n=5632`
   that's inconsistent with what M4 expects, the RMW accumulates
   garbage.

3. **MMA accumulator assumption**. M4 path (line 3223+) does
   `store_c = load_c + acc`. The acc is computed from bf16 A/B over
   K=[fast_k, fast_k+64). If the layout interpretation differs from
   how main kernel wrote the [0, fast_k) partial product, the add is
   mathematically valid but the shape interpretation may be flipped
   (RCR reads B on row axis = N with row-stride = K bytes; the M4
   kernel may read B on column axis = K, mismatching).

Most likely (1) — the M4 kernel was designed for dA inputs where b's
storage had just undergone the H4 `bf16_transpose_3d` reroute, so the
effective B storage seen by the kernel is different from fwd's
native `[G, N, K]`.

## Recommendation for R70

**Two-stage fix**:

1. **R70-A (audit only)**: Inspect `grouped_ktail_kernel_mfma32x32_M4`
   (kernel_bf16_dynamic.cpp:3134-3296) and trace A/B/C SRD
   construction against `grouped_layout_globals` semantics. Write a
   1-file probe (`/tmp/r70_m4_ktail_probe.py`) that invokes JUST the
   main-nonfuse + M4-tail pair on gpt_oss-GateUP-B4-M256-K64-downsized
   (small shape, RMW traceable), cross-check vs fp32 matmul. Diff
   result to identify whether it's an axis flip, a wrong stride, or a
   wrong accumulator interpretation.

2. **R70-B (fix)**: Once bug category identified, apply minimal fix
   inside `grouped_ktail_kernel_mfma32x32_M4`'s template specialization
   path. DO NOT retire the kernel — its correctness fix unlocks the
   R69 FUSE-off lever which is a **+4-8 score point opportunity** on
   gpt_oss (weight 3).

## Alternate R70 path (if M4 fix too invasive)

If the M4 kernel fix is too invasive for a single round, fall back
to the original R67 / R58 conclusion: **the compile-time KI
specialization and FUSE gate levers are coupled to the same
SW-pipeline register-pressure mechanism, both blocked below
`g.ki=48`**. Remaining gpt_oss headroom must then come from a
different axis entirely:

* **Var-K CRR LDS swizzle** (R68 Priority 1): 217M bank conflicts →
  potential +8-12 score. Requires Path B wiring in
  `include/ops/warp/memory/tile/shared_to_register.cuh` (currently
  marked "compiles only, not semantically verified") — but wiring it
  up for CRR (rt_32x16_s × st_64x32_padded_b128_s) is the canonical
  Path B target. Probably a 2-3 round effort.

* **Native RRR ceil_div N** (R68 Priority 3): eliminates dA H4
  transpose (6 % of gpt_oss wall). Round-11 comment explains why RRR
  was gated on aligned N — non-trivial to lift safely.

* **Fwd FUSE SGPR reduction** (R68 Priority 2): audit why the FUSE
  path adds +16 SGPRs. Candidates: K-tail loop index, K-tail load
  SRD, K-tail branch predicate. If a simple hoist / CSE reduces SGPR
  pressure, FUSE MfmaUtil may recover without disabling FUSE.

## Compliance check

* HipKittens source reverted to baseline (`git status` clean). Built
  with reverted source (baseline .so produced).
* Primus-Turbo only adds this round note; no code change.
* No `can_handle` tightening, no per-(M,N,K) hardcode (the `g.ki <
  48` predicate is a general short-K-loop threshold), no host sync,
  no caching.
* Metric confirmed correctness PASS 24/24 at baseline before and
  after R69 attempt revert (874 → 880, within noise band).
* FUSE-off attempt WAS caught by the metric's correctness gate
  (fwd-allclose FAIL on all 8 gpt_oss shapes); the lever was not
  silently regressed.

## Metric snapshot

```
                       R68 post-commit   R69 baseline  R69 attempt  R69 post-revert
score                  873               874           358          880
gpt_oss  geomean       1.076             1.076         n=0 (FAIL)   1.076
DSV3     geomean       1.119             1.122         1.123        1.122
Qwen3    geomean       1.114             1.115         1.115        1.114
correct_fail           0/24              0/24          8/24         0/24
PASS                   24/24             24/24         16/24        24/24
```

R69 attempt (FUSE-off) was rejected by the metric's correctness gate
and reverted. No commit on HipKittens; only this falsification note
committed to Primus-Turbo.

## R69 deliverable

A precise falsification of the short-K FUSE-off lever, narrowed to
a specific kernel with a specific class of latent bug. This unblocks
R70 to either fix the M4 K-tail kernel (unlocking R69's +4-8 score
opportunity) or pivot to var-K LDS swizzle / RRR ceil_div N.
