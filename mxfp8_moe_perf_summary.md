# MXFP8 Mega-MoE — Performance Summary

DeepSeek-V3 MoE · MI355X (gfx950) · EP8 · T=8192 H=7168 I=2048 E=256 top-8 · per-1×32 E8M0 mxfp8 vs bf16

**routing: load_balanced** — balanced routing (~T·K/E tokens per expert); Backward S1 also measured under round_robin; whole-forward uses a real random gate.

## Headline (fp8 speedup per pipeline stage)

| Stage | Speedup |
|---|---|
| Fwd L1 — dispatch + fc1 (preshuffle opt) | **1.59×** |
| Fwd L2 — fc2 + combine | **1.38×** |
| Bwd S1 — fc2-dgrad + comm | **1.41–1.56×** |
| Bwd S3 — fc1-dgrad + combine (WIP) | **1.56×** |

## Fused wall-clock (per-step training metric, lower is better · ms)

| Stage | Structure | bf16 | fp8 | Speedup | Accuracy | Machine |
|---|---|---:|---:|---:|---:|---|
| Forward L1 | dispatch + fc1 (comm→compute) | 3.62 | 2.285 | **1.59×** | cos 0.9991 | n04-25 |
| Forward L2 | fc2 + combine (compute→comm) | 2.98 | 2.16 | **1.38×** | cos 1.000 | chi2761 |
| Backward S1 | fc2-dgrad + comm (comm→compute) | 2.415 | 1.715 | **1.41×** | cos 0.9992 | n04-25 |
| Backward S3 (WIP) | fc1-dgrad + combine (compute→comm) | 4.10 | 2.63 | **1.56×** | 26 dB\* | n05-33 |
| **Whole forward (L1+L2)** | real random gate | 6.79 | 5.47 | **1.24×** | 18.3 / 31.6 dB | chi2761 |

Fused wall-clock = single-kernel wall after comm∥compute overlap. Whole-forward 1.24× uses the old L1 (2.56 ms); after the L1 preshuffle optimization (2.285 ms) the whole forward is actually better.

\*Backward S3: bf16 = canonical bench (best combine_cu); fp8 = `grouped_gemm_combine_fp8_bwd` @ ncu=32. 26 dB is the finite-part SNR — S3 fp8 is dev-gated (WIP) pending a cross-rank write-order fix (see below); production backward S3 still uses bf16.

## GEMM compute ceiling (kernel-only, comm stripped)

| GEMM | shape | bf16 TF | mxfp8 TF | × bf16 | vs fp8-tw |
|---|---|---:|---:|---:|---:|
| fc1 (fwd L1) | N=4096 K=7168 | 1138 | **2400** | 2.11× | 0.93× |
| fc2 (fwd L2) | N=7168 K=2048 | 975 | **1946** | 2.00× | 0.93× |
| fc2-dgrad (bwd) | N=2048 K=7168 | 1116 | **2377** | 2.13× | 0.93× |

Single GPU (n04-25), uniform M_per=2048, Triton bf16 reference, SNR ~28 dB. mxfp8 grouped NT GEMM peaks at ~2.4k TFLOPS ≈ 2× bf16, ~0.93× of the per-tensor-fp8 ceiling — per-1×32 block scaling is essentially free. dgrad runs NT (static w2 transpose) reusing the same tile.

## Backward STEP3 (fc1-dgrad + combine, fp8-PUSH)

n05-33 · real DSv3 EP8 pool (M_pad=66048, out=H=7168, K=2I=4096) · same pool/routing · reduce overlaps on empty GEMM blocks. Mirrors the production forward `grouped_gemm_combine_fp8` (mxfp8 GEMM + fp8 PUSH).

### Isolation ladder (ms, full_call − grad_l1 quant)

| Isolation | ncu=16 | ncu=32 (sweet spot) |
|---|---:|---:|
| GEMM only | ~1.85 (all CUs) | — |
| push only (fp8) | 2.976 | 1.813 |
| push only (bf16 ref) | 4.043 | 2.631 |
| GEMM + push (reduce off) | 3.884 | 2.525 |
| **full (GEMM+push+reduce)** | 3.966 | **2.499** |

### combine PUSH — ncu sweep (fp8 vs bf16, same pool/routing)

| ncu | fp8 push (ms) | bf16 push (ms) | fp8/bf16 |
|---:|---:|---:|---:|
| 16 | 2.976 | 4.043 | 0.74× |
| 32 | 1.813 | 2.631 | 0.69× |
| 48 | 1.779 | 3.048 | 0.58× |
| 64 | **1.245** | 1.938 | 0.64× |

Byte ratio fp8/bf16 ≈ 0.52× (fp8 payload H×1 + E8M0 H/32 vs bf16 H×2). Higher ncu approaches the byte lever; ncu=16 is combine-CU-starved and not representative of the true push cost.

### full STEP3 — fp8 vs bf16 (fused = GEMM + push + reduce)

| | fused (ms) | gemm-only (ms) | push-only (ms) | accuracy |
|---|---:|---:|---:|---:|
| bf16 STEP3 (canonical bench) | 4.10 | 3.41 | 2.89 | cos 1.000 |
| **fp8-PUSH STEP3** | **~2.63** | 1.85 | 1.81 | 26 dB |
| speedup (fp8 vs bf16) | **1.56×** | 1.84× | 1.60× | — |

bf16 = `grouped_gemm_combine_impl`(nn), canonical bench (best combine_cu=16–18; reduce flags pre-set ready = perf-only). fp8 = `grouped_gemm_combine_fp8_bwd` @ ncu=32 (reduce actually waits → slightly pessimistic, so the real gap is ≥1.56×). Both bf16-STEP3 halves (GEMM K=2I=4096, push) are ~1.6–1.8× slower than fp8. bf16 STEP3 is GEMM-bound (gemm 3.41 > push 2.89).

**Performance conclusion.** full fp8 STEP3 ≈ **2.63 ms** vs bf16 **4.10 ms** → **~1.56× faster**, in line with the forward L2 / STEP1 wins. reduce is basically free (hidden on empty GEMM blocks). comm-bound: push dominates and scales with ncu (fp8 push down to 1.245 ms @ ncu=64).

**Production blocker.** Cross-rank write-ordering race → intermittent non-finite (math is correct, finite part 26 dB); ncu≥48 full triggers a liveness stall (deadlock). Three device-scope fence pairings (l2_writeback/l2_invalidate) did not fix it → needs rocprofv3 single-rank localization. Currently dev-gated; production backward still uses bf16-PUSH.

## Status & provenance

**Landed / validated.** Forward L1+L2 all-mxfp8 (production default, SNR 18.3 dB > 15 gate); backward STEP1 wins 1.41–1.56× (cos 0.999). Weight prep w1/w2/w2ᵀ/w1ᵀ is all module-owned, version-keyed, computed once per optim.step (reused across a grad-accum window).

**Weight-prep cost (must cache).** w1ᵀ prepare (grouped quant [32,7168,4096] + preshuffle) = 2.9 ms, larger than the GEMM itself, but static → amortized once per optim.step. grad_l1 quant = 0.39 ms, token-dependent, unavoidable, counted per step.

**Provenance.** Fused wall-clock spans machines (Fwd chi2761 / Bwd S1 n04-25 / Bwd S3 + bf16 n05-33, all MI355X EP8 same shape); GEMM ceiling is single-GPU. Bwd STEP1 fused ~1192–1239 TF (incl. comm) is below the comm-stripped ~2377 TF; the gap is the exposed XGMI dispatch.
