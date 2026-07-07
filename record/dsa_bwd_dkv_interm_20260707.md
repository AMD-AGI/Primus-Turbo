# FlyDSL dKV-intermediate MFMA port — CORRECT but slower, gated OFF (2026-07-07)

> After the dQ M=16 win (25.5->8.8ms), the next bwd lever was the dKV-intermediate
> (Triton, ~3454 us/call, ~25% of bwd). Ported it to a native FlyDSL M=16 head-
> contraction MFMA. Result: CORRECT (dkv 69 dB) but SLOWER than the Triton version.
> Kept env-gated OFF as a correct reference.

## What it computes
interm[key, d] = sum_h ( Q[h,d]*dS[h,key] + dO[h,d]*P[h,key] )  (contract head H).
Output interm[T, TOPK, D_V], feeds the (unchanged) CSR dKV gather-reduce.
Reuses dS/P from the dQ kernel's HBM buffers (no recompute).

## Implementation (dsa_bwd_dkv_interm_kernel.py)
Modeled on the proven head-contraction output GEMM in
build_csa_pool_bwd_dpool_mfma_module: qT/doT staged transposed to LDS [D,H_PAD],
dS/P to LDS [BLOCK_K,H_PAD], output GEMM part[d,key]=qT@dS + doT@P (16x16x32, K=32
head-contraction). Grid (T,), single wave/token.
- D-blocked (D_BLOCK=128, DB_ITERS=4) to fit the 160KB LDS (full [D_V=512,H_PAD=128]
  qT+doT would be 256KB). dS/P re-staged per D-block.
- i64 offsets for q/do/dS/interm (T*H*TOPK and T*TOPK*D_QK exceed i32 element range).
- LAUNCH CHUNKED over T: flydsl packs a tensor's BYTE size as i32, so interm
  (T*TOPK*d_qk*2 = 2.4 GB at H128 K512) exceeds 2^31 bytes. The launcher loops
  T-chunks each < 2^31 bytes (kernel is grid=(T,) per-token independent).

## Result: correct but slower
| bwd total (H128 K512) | dq-M16 only | + interm-MFMA | triton_v2 |
|---|---|---|---|
| ms | 14.6 | 22.6 | 9.25 |
dkv SNR 69 dB (unchanged -> interm is numerically correct). 138 tests pass.
The interm-MFMA path is 8 ms SLOWER, so it's gated OFF (PRIMUS_DSA_FLYDSL_BWD_INTERM_MFMA).

## Why it's slower (root cause)
Single wave (64 threads) per token must stage the FULL qT/doT transpose (128 heads
x 512 d) into LDS via per-element SCALAR stores, plus re-stage dS/P per D-block
(DB_ITERS=4). The Triton _bwd_compute_dkv_intermediate uses 4 warps + vectorized
access and better hides that staging. D_BLOCK sweep (128->256) made it WORSE (more
per-tile MFMA vs staging imbalance), confirming staging (not compute) is the bound.
Making the FlyDSL version competitive needs a MULTI-WAVE redesign (split heads or
D across waves, vectorized transpose) — a substantial further effort.

## Verdict
- dQ M=16 (25.5->8.8ms, committed) is the real bwd win; keep it default-candidate.
- dkv-interm FlyDSL port: correct reference, gated OFF. Triton interm stays the
  default (faster). The next real bwd lever would be a multi-wave interm rewrite.

## Files
- New: primus_turbo/flydsl/attention/kernels/sparse_mla_v2/dsa_bwd_dkv_interm_kernel.py
- Wired (gated OFF): dsa_bwd.py (_get_interm_kernel, PRIMUS_DSA_FLYDSL_BWD_INTERM_MFMA)
  + T-chunked launch (2^31-byte tensor limit workaround).
- Harness: output/harness_bwd.py
