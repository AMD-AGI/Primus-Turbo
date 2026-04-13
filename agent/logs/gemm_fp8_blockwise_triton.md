# gemm_fp8_blockwise Optimization Log (TRITON)

## Config
- Target: gemm_fp8_blockwise (fp8, blockwise)
- Language: TRITON
- GPU: gfx942 (MI300X, 304 CUs, FP8 peak 1309.5 TFLOPS)
- Kernel: `primus_turbo/triton/gemm/gemm_fp8_kernel.py`
- Baseline Forward: avg 567.7 TFLOPS, max 639.5 TFLOPS (48.8% efficiency)
- Baseline Backward: avg 410.8 TFLOPS, max 497.2 TFLOPS (38.0% efficiency)
- PASS: 62/84 shapes, ERROR: 22 (large shapes, likely benchmark OOM)

## Hypothesis List (ranked by expected impact)
1. **Remove per-iteration K masking**: Add EVEN_K constexpr to skip mask loads when K % BLOCK_K == 0 (high impact, low risk)
2. **Add BLOCK_N=256 to autotune space**: Currently fixed at 128, larger tiles improve compute density (high impact, medium risk)
3. **Use int64 pointer arithmetic**: Prevent potential int32 overflow for large matrices (correctness fix, low risk)
4. **Add persistent kernel / Stream-K scheduling**: Currently data-parallel only, persistent improves CU utilization for small tile counts (high impact, medium risk)
5. **Optimize scale application**: Hoist scale multiply when possible, reduce redundant loads (medium impact, low risk)
6. **Add cache modifiers (.ca/.cg)**: Profile-guided cache hint selection (medium impact, low risk)
7. **Add `tl.multiple_of` hints**: Help compiler optimize memory access patterns (medium impact, low risk)
8. **Expand num_stages to {1,2,3}**: More pipeline stages for memory latency hiding (medium impact, medium risk)
9. **Add kpack=2 to compiler options**: Pack 2 FP8 values per VGPR load (medium impact, low risk)
10. **XCD-aware tile scheduling**: Better cross-chiplet load balancing (medium impact, medium risk)

