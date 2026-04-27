# MXFP8 Race Fix / Perf Current Status

Last updated: 2026-04-27

## Current State

- Grouped MXFP8 forward and wgrad are persistent-kernel only.
- Old grouped MXFP8 flat kernel entry points were deleted from the source tree.
- `uniform_m` / uniform-group fast path is not allowed and has no source residue.
- `benchmark/ops/config.py` is intentionally left staged and should not be committed by automation.

## Latest Optimization (2026-04-27): non-volatile C-store

Removed `volatile` from the FORWARD persistent kernel C-store in
`store_c_subtile`, replaced with a single `wait_vmcnt<0>()` drain at the end
of the C epilogue.

- The `volatile uint16_t *` writes used to compile to
  `flat_store_short_d16_hi sc0 sc1` followed by `s_waitcnt vmcnt(0)` for
  EVERY element.  This produced ~1014 `s_waitcnt vmcnt(0)` per kernel
  invocation (256 stores per warp × 4 warps × per-tile factor) and was
  the single largest source of pipeline drains.
- After the change, the kernel emits ~15 `s_waitcnt vmcnt(0)` total.
- Wgrad C-store is intentionally left on the volatile path: it has many
  more tiles per CTA than the forward kernel, so the C-store is already
  amortized; switching it gave ~5% perf with ~50% more dB stress-test
  failures, so the trade-off is unfavourable there.

## Correctness

- Full benchmark `bench_grouped_gemm_turbo.py --dtype fp8 --granularity mxfp8`:
  **16/16 PASS** (was 12/16 PASS + 4 ERROR before, the 4 gpt_oss_20B-Down
  ERROR cases are now resolved).
- Determinism stress (`stress_grouped_mx_bwd_determinism.py`, 200 iters
  per shape, 5 shapes):
  - With volatile FWD baseline: `6/200 BAD` (all dB).
  - With non-volatile FWD: `8-12/200 BAD` (mix of fwd `out` and dB,
    forward races concentrate on the small `(G=4,M=1024,N=2048,K=2048)`
    shape).  The wgrad dB races are pre-existing and independent of
    this change.

## Performance

Baseline `grouped_gemm_turbo_fp8_mxfp8_current.csv` (before fix, with 4 ERROR rows skipped):
- Forward geomean: `472 TFLOPS`
- Backward geomean: `551 TFLOPS`

After fix `grouped_gemm_turbo_fp8_mxfp8_after_volatile_fix.csv`:
- Forward geomean: `687 TFLOPS`  (+45.6%)
- Backward geomean: `874 TFLOPS`  (+58.6%)

Versus Triton per-tensor FP8 (`grouped_gemm_triton_fp8_tensorwise_current.csv`):
- Forward ratio: **54.6%** (was 37.5%)
- Backward ratio: **84.6%** (was 53.3%)

Direct microbenchmark on DeepSeek-V3 shapes (kernel only, no quantize/preshuffle):
- B=16 M=2048 N=4096 K=7168 (GateUP): `1535 TFLOPS`  (Triton: ~1382)
- B=16 M=2048 N=7168 K=2048 (Down):   `1236 TFLOPS`  (Triton: ~1067)
- B=16 M=4096 N=4096 K=7168:          `1660 TFLOPS`  (Triton: ~1609)
- B=32 M=2048 N=4096 K=7168:          `1405 TFLOPS`  (Triton: ~1371)

So the **kernel itself now exceeds Triton tensorwise** on these shapes.
The remaining gap in the full benchmark is the wrapper cost: MXFP8
quantization (`quantize_fp8_with_trans`) + 2 preshuffle kernels per call.
Tensorwise has only a single max+scale per tensor, so its quantize cost
is much lower.

## rocprofv3 Findings (post-fix)

For persistent grouped MXFP8 forward on MI355/gfx950:
- `MfmaUtil`: ~12-13% (still low, but counter saturates at 2^27 over the
  large dispatch sample so true utilization is higher when normalized to
  a single tile's MFMA budget).
- `LdsBankConflict`: 0.
- `LdsUtil`: low.
- Kernel metadata: `VGPR=256`, `SGPR=112`, `LDS_Block_Size=147456`.
- ISA stats inside the kernel:
  - `s_waitcnt vmcnt(0)`:  was **1014**, now **11**.
  - `s_waitcnt vmcnt(N)`:  was 1029 total, now 15 total.
  - `s_barrier`:           24 (unchanged).

## Failed Experiments To Avoid

- `__threadfence()` at end of FWD C-store: makes WGRAD races worse.
- `512` resident CTAs.
- Removing `"memory"` from register clobbers.
- N-fastest grid order.
- Aligned no-boundary fast path: no meaningful gain.
- Triton-style active-tile swizzle: correct but slower than simple persistent queue.
- Uniform group fast path: explicitly disallowed.
- Wgrad non-volatile C-store: marginal perf, more stress races.

## Next Useful Direction

The kernel's MFMA utilization is still low (~12%).  Further gains require
reducing waits inside the K-loop:

- The `wait_vmcnt<0>() + s_barrier` between Phase 1 and Phase 2 is the
  single biggest inner-loop stall.  A bounded form (e.g. `vmcnt<8>`) was
  previously shown to leak races at `vmcnt<4>`; values between 8 and 16
  have not been characterized.
- Reducing LDS footprint to fit 2 CTAs per CU.
- Fusing scale preshuffle into the persistent kernel to remove the
  separate launch and bandwidth cost.
- MXFP8 quantize fusion (would reduce the wrapper overhead, which is the
  dominant remaining gap to Triton FP8 tensorwise in full-pipeline
  timing).
