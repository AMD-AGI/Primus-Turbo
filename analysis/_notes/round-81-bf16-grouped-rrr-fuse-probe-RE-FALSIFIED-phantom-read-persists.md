# Round-81 — BF16 grouped wall — RRR FUSE_PROBE re-attempt RE-FALSIFIED (R29-era phantom-read persists, max_abs=58)

**Date**: 2026-05-05  **HEAD before**: `85dd7244ea7dfc80a52179ae`  **score before**: 883 / 1000 (best 883)
**HEAD after** : same (no commit; revert)  **score after** : same (no metric run after revert)

## Lever (planned per R80 note's R81 direction)

R80 landed Path A Step 1: gpt_oss-GateUP shapes take native RRR + new
ceil_div N coverage (+9 score). Down shapes still take H4-RCR because
their K-tail RMW kernel introduces a bf16 round-trip (load existing C
bf16, fp32 add K-tail acc, store bf16) that exceeds `check_allclose`
tolerance vs Triton's fp32-fused reduction (R80 measured mean_abs 0.12 →
0.238 ≈ 2x noise; max_abs=2 vs Triton's bf16-final-only).

R80's R81 plan: "re-attempt RRR FUSE_KTAIL via the BF16_RRR_FUSE_PROBE
flag" — the kernel may have evolved enough post-R55 LDS-staged ntail /
post-R56 K-tail RMW restructuring / post-R80 ceil_div coverage that
the R7-R29 phantom-read no longer reproduces.

## Experiment

Set `BF16_RRR_FUSE_PROBE=1` in `kernel_bf16_dynamic.cpp:4398-4400`.
This makes RRR eligible for `fuse_ktail_eligible` → `launch_one_grouped_fuse<RRR>`
→ instantiates `grouped_kernel<RRR, 0, true>`. Also temporarily disabled the
Primus-side H4 K_TWO_TILE gate so Down shapes route to native RRR + fuse
instead of H4-RCR.

Build: clean, no spills, no SGPR/VGPR regression. The fuse-RRR template
already had instantiations from R29-era work (line 4087:
`template __global__ void grouped_kernel<Layout::RRR, 0, true>`).

## Result — phantom-read persists, identical signature to R29

`/tmp/r80_native_rrr_correctness.py` (8 gpt_oss shapes, post-DSV3 warmup):

```
PASS gpt_oss-GateUP B=4  M=2048  reason=''
FAIL gpt_oss-Down   B=4  M=2048  reason='dA-allclose'
PASS gpt_oss-GateUP B=32 M=2048  reason=''
FAIL gpt_oss-Down   B=32 M=2048  reason='dA-allclose'
PASS gpt_oss-GateUP B=4  M=4096  reason=''
FAIL gpt_oss-Down   B=4  M=4096  reason='dA-allclose'
PASS gpt_oss-GateUP B=32 M=4096  reason=''
FAIL gpt_oss-Down   B=32 M=4096  reason='dA-allclose'
```

GateUP all PASS (same as R80 — no K-tail, no fuse triggers). Down all
FAIL on dA-allclose, with diff stats much worse than R80's K-tail RMW
path:

```
                          R80 K-tail RMW    R81 FUSE_PROBE
dA mean_abs               0.238             2.28      (10x worse)
dA max_abs                2.0               58.2      (29x worse)
dA partial cols max_abs   1.0               1.0       (unchanged)
dA interior cols max_abs  2.0               58.2      (uniform, all cols affected)
```

The interior-col max_abs of 58 (vs ~50 magnitude expected output)
matches **exactly** the R29-era SNR ~22 dB profile documented at
`kernel_bf16_dynamic.cpp:4366-4382`:

> Round-7: extended RRR fuse via path A hybrid (A direct HBM→reg + B
> LDS-staged + manual ds_read_b64_tr_b16). SNR 18.68 dB (phantom-read
> still observed for warp_row=0 wc∈{1,3}).
> Round-8: switched manual mode default ON (BF16_RRR_FUSE_USE_KITTENS=0)
> + added missing s_waitcnt lgkmcnt(0) + __syncthreads. SNR 18.68 →
> 25.45 dB but allclose still FAIL — bypass of subtile_inplace ONLY
> partially fixes phantom-read, ~25 % cells (matching round-3 phantom
> pattern: warp_row=0 wc∈{1,3}) still receive stale K-tile data.

## Why the bug persists despite R55/R56/R80 evolution

The phantom-read is in the K-tail epilog block (`device_gemm_tile_body.cpp:828-1167`)
which runs AFTER main_loop_iter completes. The bug is that warp_col∈{1,3}
threads read stale K-tile data from `Bs[1][n_strip]` LDS slots when
loading B_tile_0 / B_tile_1 for the K-tail MFMA.

Recent kernel changes don't touch this epilog block:
* R55 (LDS-staged ntail kernel) is a SEPARATE kernel (`grouped_ntail_kernel_lds_rrr<64>`),
  not the FUSE epilog.
* R56 (K-tail RMW) is in `grouped_ktail_kernel_lds_rrr<64>`, also separate.
* R80 (ceil_div bpc) only changed `dispatch_grouped<L>` and the K-tail
  RMW kernel grid; the FUSE epilog is untouched.

So the phantom-read is byte-identical to R29's reproducer, with the same
diagnostic profile.

## Root cause hypotheses (R29 backlog, unchanged)

R29 narrowed the bug to one of two:

1. **Cross-warp G::load LDS visibility**: G::load is cooperative across
   the 4 warp_cols of the MMA, but the K-tail epilog runs ONLY in
   each warp's local view. If G::load's writes to `Bs[1][n_strip]`
   from one warp aren't visible to another warp's reads (despite
   `s_waitcnt vmcnt(0) lgkmcnt(0)` and `__syncthreads`), the phantom
   read is unavoidable without restructuring.

2. **`col_l rt_32x16_s` lane → cell mapping after `ds_read_b64_tr_b16`**:
   the swizzle bypasses the kittens helper indirection but the manual
   address derivation may still miss a cross-lane data shuffle that
   the helper handles. R29's USE_KITTENS=1 build gave SAME 19.59 dB
   SNR as USE_KITTENS=0, suggesting the helper isn't the issue
   either — but the symmetry of failure between manual + helper
   doesn't disprove (2); both could be making the same lane-mapping
   error.

## Why a one-round fix is implausible

The fuse epilog has ~340 lines of K-tail load + MMA logic. Either
hypothesis requires 50-100 lines of restructuring + 4-6 SNR probe
iterations to identify the exact failing lane pattern. Single-round
budget (4-1.5h chat window) doesn't fit.

## Action — REVERT, no commit, write this falsification note only

* `kernel_bf16_dynamic.cpp`: BF16_RRR_FUSE_PROBE flipped back to 0.
* `grouped_gemm_impl.py`: K_TWO_TILE gate restored.
* No metric run after revert (kernel state byte-identical to R80
  HEAD `85dd7244` post-rebuild; metric expected to give 883 ± 5
  noise band).
* This note documents the negative result so R82+ doesn't waste a
  round on the same flag-flip.

## Direction for R82

Two remaining structural levers (both multi-round):

1. **RRR fuse phantom-read root cause**: instrument the K-tail epilog
   with per-warp_col / per-h_b SNR diagnostic to identify the EXACT
   lane that's stale, then derive a fix targeted at that lane.
   Multi-round (R82 = instrument + characterise; R83 = fix attempt 1;
   R84+ = iterate). Estimated +5-10 score IF eventually fixed
   (eliminates K-tail RMW round-trip noise on gpt_oss-Down dA →
   drops H4 transpose).

2. **dB var-K LDS swizzle without sub-tile padding** (per R68 PMC's
   217M LDS bank conflicts): R74's `st_64x32_padded_b128_s` swap
   spilled 66 VGPRs / 24/24 dB FAIL due to incompatible padding.
   A swizzle-only change (keep `st_32x16_s` shape, override the
   per-row XOR mask via a custom shape struct that inherits from
   st_32x16 but adds a different swizzle term) avoids the padding
   trap entirely. Risk: low VGPR (no shape change), risk on the
   correctness side (bank-conflict-free swizzle still needs to
   match the rt_32x16_s lane mapping). Estimated +5-10 score IF
   it works (gpt_oss-Down dB is the family's 35% wall fraction).

R82 picks (2) — dB var-K swizzle — as the cleaner test (single struct
definition + dispatch flip, ~30-50 lines of C++).

## Score variance note (informational)

R3 prompt reported score=883 as "this round's metric" (post-R80 commit).
R4 first metric run gave score=888 (same kernel). The +5 inter-run
variance is consistent with prior rounds' noise band on a busy MI355X
node. None of R81's experiments are reflected in either number — the
FUSE_PROBE was tested via standalone correctness probe only; no metric
run after enabling, immediate revert + this note.
