# Round 70 — BF16 grouped, bwd var-K chiplet chunk_size sweep + partial M4 K-tail analysis — FALSIFIED (noise band)

## Round plan

R69 closed on the observation that the R68 Priority-2 lever (FUSE off
for `g.ki < 48`) is blocked by a latent bug in
`grouped_ktail_kernel_mfma32x32_M4<RCR, 64>` — the kernel was DEAD
CODE on every metric shape at baseline (fwd goes through FUSE, dA
K_rem=0) and produces 8/8 gpt_oss fwd-allclose FAIL when activated.

R70's two branches:
* **R70-A (diagnostic)** — try to isolate the M4 bug from source
  analysis, unlock R69's FUSE-off lever for R71.
* **R70-B (fallback lever)** — if M4 audit is inconclusive in one
  session, try an un-tested bwd-side cheap knob: **`chiplet_transform_chunked` chunk_size** on the var-K persistent kernel. R62 falsified this on the forward kernel (chunk 64→32), but var-K's access pattern is structurally different:
    - each pid is a `(group_idx, local_tile)` pair, persistent stride NUM_CUS across (G × bpr × bpc) total tiles;
    - A/B are full-tensor SRDs `[M_total, *]` — CUs that land on
      adjacent pids see adjacent M-row ranges clustered by group; L2 locality benefit from finer XCD striping is plausible.

R70-A timeline-boxed to the first ~15 min of the round: re-read M4
fast-MFMA path (kernel_bf16_dynamic.cpp:3223-3293). Confirmed that
the lane-to-(row, col) mapping for the 32x32x16 BF16 MFMA output
block matches the per-chunk / per-sub 128-M-row tiling structure
(lanes 0..31 cover chunk=0 rows, 32..63 cover chunk=1 rows, 4-sub
decomposition covers all 128 rows). A-load and B-load address
arithmetic are consistent with `grouped_layout_globals` fwd
semantics (g.a = [M_total, K], g.b = [G, N, K], g.c = [M_total, N]).

Two candidate failure hypotheses emerged, neither resolvable from
source alone in this round:
1. **Partial last col-tile N-masked write by main kernel**. The
   main kernel's `store_c_tile_n_masked` SKIPs C writes for
   `col >= g.n` on the partial last col-tile. For fwd, if the
   intermediate [fast_n, g.n) cells are uninitialized before the
   main kernel runs, M4's RMW (`existing = load_c(r, col); store_c(r, col, existing + acc)`) reads garbage. The R11 comment at line 4441-4448 claims "main always covers [0, g.n)", but 8/8 gpt_oss correctness FAIL suggests the claim doesn't hold for the
   specific partial last col-tile region when combined with non-fuse
   main + M4 K-tail.
2. **BF16 storage round-trip precision**. Main writes partial
   `K=[0, fast_k)` C in bf16. M4 RMW reads bf16, adds f32 acc,
   stores bf16. The fuse path keeps C in f32 accumulator throughout
   — maybe the bf16 round-trip induces large rel-diff on correlated
   N-sweep outputs specifically for K=2880 (gpt_oss) but not for
   K=5760 (dA — but dA doesn't trigger M4 anyway).

Both hypotheses require a runtime probe — not a source read — to
confirm. Deferred to R71. **R70 falls through to R70-B.**

## R70-B: var-K chiplet chunk_size sweep

Modified `grouped_var_k_kernel` at line 4745:

```cpp
int pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, 64);  // baseline
int pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, CS); // swept
```

Tested `CS` ∈ {16, 32, 64, 128}:

| chunk_size | Score | gpt_oss | DSV3  | Qwen3 | Δ score vs 874 baseline |
|------------|-------|---------|-------|-------|-------------------------|
| 16         | 871   | 1.072   | 1.121 | 1.107 | −3                      |
| 32         | **877** | 1.084 | 1.117 | 1.110 | **+3**                  |
| 64 (baseline) | 874 | 1.076 | 1.120 | 1.113 | 0 (reference)           |
| 128        | 867   | 1.066   | 1.120 | 1.096 | −7                      |

Peak at `CS=32`: +3 score, gpt_oss +0.8 pp geomean. But:
* DSV3 / Qwen3 both down ~0.3-0.5 pp (coincidentally offset by
  gpt_oss gain — not a clean win).
* Baseline re-verified post-revert: **879** (within the ±5-10 GPU 3
  noise band centered on 874-880 observed all round).
* +3 does not meet the +5 commit threshold mandated by the workflow.
* Trend non-monotonic (16 low, 32 peak, 64 baseline, 128 lower) →
  no clear scaling law, most likely a minor alignment artifact
  between XCD tile-count rounding and `NUM_CUS` for this specific
  workload shape.

**Falsification**: `CS=32` on var-K does not sustain a commit-worthy
improvement. Kernel reverted to baseline `CS=64`; post-revert
baseline 879 confirms no state damage.

## What closes / opens for R71

**Closed**: var-K chiplet chunk_size lever. R70 exhausts {16, 32, 64,
128} sweep; all within ±10 noise. Combined with R62's fwd-side
falsification, chunk_size is no longer an open lever on either
kernel.

**Still open** (ranked by tractability × expected yield):

1. **M4 K-tail kernel fix (R70-A deferred)**. Unlocks the R69
   FUSE-off lever worth +4-8 score. R71 should write a probe that:
    a. Invokes ONLY the main non-fuse kernel on gpt_oss-Down
       downsized (B=4, M=256, N=2880, K=2880) with FUSE off via an
       env flag (to be added as a 1-line build-time dispatch toggle);
    b. Captures the intermediate C buffer and compares to the
       expected `sum_{k=0}^{2816} A[m,k] * B[n,k]` partial product
       via fp32 reference;
    c. If main kernel's partial C differs from fp32 ref on cells
       `col >= fast_n`, the bug is (1) uninitialized C; fix by
       pre-zeroing C in dispatcher. If main matches ref but M4's
       RMW output is wrong, bug is (2) BF16 round-trip OR M4's C
       axis interpretation.
    d. Fix the confirmed bug category inside
       `grouped_ktail_kernel_mfma32x32_M4` (or the M2/32x32/16x16
       variants — the same bug likely lives in all 4 RCR K-tail
       specializations).
    e. Re-apply R69 gate, re-run metric — expected +4-8.

2. **Var-K CRR LDS swizzle** (R68 Priority 1). +8-12 score
   potential; requires wiring `st_64x32_padded_b128_s` +
   `rt_32x16_s` Path B in `include/ops/warp/memory/tile/shared_to_register.cuh` for the CRR
   `grouped_var_k_kernel`. Path B currently flagged "compiles
   only, not semantically verified" (line 530-532 of
   shared_to_register.cuh). Probably 2-3 rounds: (a) validate
   Path B's lane→VGPR mapping on a synthetic load-store round-trip
   bench outside the GEMM loop; (b) verify MMA semantic on dense
   BF16 CRR bench; (c) wire into `grouped_var_k_kernel`'s ST_A /
   ST_B types + rebuild.

3. **Native RRR ceil_div N** (R68 Priority 3). Drops dA H4
   transpose (360 us / iter = 6 % of gpt_oss wall). Round-11
   comment restricts RRR to `fast_n % BLOCK_SIZE == 0` because
   OOB N columns wrap to next K row and feed garbage to MMA.
   Solution: column-mask on the B load itself (K-major stride on
   the OOB col offsets → mask before packing into LDS). Similar
   complexity to (2); ~2 rounds.

4. **Fwd FUSE SGPR reduction** (R68 Priority 2). Audit which 16
   SGPRs are added by the FUSE path vs non-fuse (build-report
   diff). If a CSE / hoist opportunity exists (e.g., `K_tail_base_bytes` recomputed per issue), shave SGPR and measure
   MfmaUtil recovery. Cheapest of the 4; could land in 1 round if
   lucky.

## Metric snapshot

```
                       R69 post-revert   R70 baseline  chunk=32 attempt  R70 post-revert
score                  880               874           877                879
gpt_oss  geomean       1.076             1.076         1.084              1.076
DSV3     geomean       1.122             1.120         1.117              1.121
Qwen3    geomean       1.114             1.113         1.110              1.114
correct_fail           0/24              0/24          0/24                0/24
PASS                   24/24             24/24         24/24              24/24
```

All measurements on GPU 3 with 15-20 s metric wall. Noise σ ≈ 3-4
per run; 95-% band is ±6-8 around the local mean. +3 with mixed
per-family Δ fits cleanly inside that band.

## Compliance check

* HipKittens source reverted to baseline (`git status` clean under
  `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`).
* Primus-Turbo only adds this round note.
* No `can_handle` tightening, no per-(M,N,K) hardcode, no host sync,
  no caching.
* Correctness PASS 24/24 across all 4 chunk_size variants AND
  baseline re-verify.
* chunk_size is a pure runtime dispatch parameter (no tensor-layout
  or numerics change), so correctness invariance was expected and
  confirmed.

## R70 deliverable

Falsifies the final open "cheap knob" class (chiplet chunk_size) on
the bwd path, narrowing the remaining BF16 grouped headroom to
structural kernel-body changes ranked in the "Still open" list
above. R71 should pick up the M4 K-tail audit probe — highest
tractability × concrete yield (+4-8 score).
