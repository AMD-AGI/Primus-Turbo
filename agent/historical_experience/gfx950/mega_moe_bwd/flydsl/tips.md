# gfx950 mega-MoE backward (FlyDSL) — reusable tips

## MoE dgrad: do it as NT via a static weight transpose, NOT as an in-kernel NN transpose-read
On gfx950, the mxfp8 scaled MMA (`v_mfma_scale_f32_16x16x128_f8f6f4`, `MfmaScale16x16x128`)
accumulates in **VGPR** and has **no AGPR variant**. The per-tensor fp8 NN kernel's fast B
path (`S2RLoaderTr(inline_asm=True)` + `ds_read_b64_tr_b8`) requires **AGPR** accumulation
(pinned `agpr_alloc>0`), so it CANNOT be combined with the scaled MMA — the fused kernel
fails to compile (`asm_mma=True` + inline tr8). The only NN option for mxfp8 is the intrinsic
`ds_read_tr8_b64` transpose-read, which is no faster than NT (the NN dgrad GEMM is already
near fp8 roofline, ~2109 TFLOPS at M=8192).
=> For MoE dgrad `grad_act = dispatch(dy) @ w2`, transpose the (static) weight
`w2 [G,H,I] -> [G,I,H]` (cached, free) and reuse the near-roofline **NT** tile / the forward
L1 fused NT kernel `dispatch_grouped_gemm_mxfp8` with `dy` as tokens. Validated: identical
grad (cos 0.99921 vs bf16), 1.06x vs bf16 fused, zero new kernel. A bespoke NN mxfp8 tile
(built + validated) was strictly dominated and removed. Matches `mxfp8_grouped_kernel.py`.

## Fused backward STEP1 is comm/overlap-bound, not GEMM-bound
The fused dispatch(dy)+dgrad STEP1 cost (~2.2-2.3 ms) is dominated by the cross-rank fp8
dispatch push + padded-pool GEMM overlap, not the per-tile GEMM (near roofline). Tuning the
tile (tr8 co-schedule etc.) gives ~0. Levers are the CU split (ndcu=16/pscu=16 best, same as
fwd L1; pscu=8 exposes the preshuffle) and reducing padded work.

## Fence granularity in the fused STEP1: hard to reclaim (parallelism trade + masking)
The L2 fences cost ~0.37 ms (17%) at STEP1: gemm-role full-L2 `buffer_inv` = 0.143 ms (2048
per-tile invalidates), preshuffle-role inv+wbl2 = 0.242 ms (~256 each). Isolate with a `diag`
compile toggle (`NO_GEMM=1`, `GEMM_ONLY=2`, `NO_FENCE=4`, `NO_GEMM_FENCE=5`, `COMM_ONLY=8`).
- The acquire (invalidate) only needs to drop stale lines ONCE per workgroup for a single-owner,
  gated reader (preshuffle owns each pool-block, gemm gated per block_m). Hoisting the preshuffle
  invalidate to once/workgroup is correct but NEUTRAL — the gemm's 2048 full-L2 nukes dominate
  device L2 and mask it.
- Cutting gemm invalidate COUNT by putting >1 tile per workgroup (loop block_n) is a DEAD END:
  it collapses the 2048-way parallelism that hides memory latency (256 wg × 8 serial GEMMs →
  3.5 ms, −63%), and sequential `gemm_mxfp8_nt_tile` calls need a barrier between them (shared
  LDS race). Invalidate saving (≤0.143 ms) ≪ serialization cost.
- Parallelism-preserving option = invalidate ONCE PER XCD (first wg on each XCD invalidates, rest
  skip). Needs a hardware XCD/SE-id read (no FlyDSL intrinsic; only `workgroup_id`/`wave_id`
  exist → raw `s_getreg HW_ID` inline-asm + arch bit-decode) + per-XCD flag + atomic CAS/spin.
  High correctness risk on a distributed kernel for ~7% on a comm-bound shape. Not worth it
  unless STEP1 stops being comm-bound.

## STEP1: don't add comm CUs or batch the push to speed the full kernel
Measured on the fused dispatch(dy)+dgrad STEP1 (EP8 MI355X, T=8192 H=7168 I=2048 E=256 K=8):
- comm-only (diag=8) DOES scale with comm CU: ndcu 8/16/24/32 -> 2.367/1.644/1.499/1.404 ms
  (CU-limited in isolation, diminishing returns).
- But the FULL kernel gets WORSE with more comm CU: ndcu 16/24/32 -> 2.170/2.345/2.521 ms. The
  fused kernel shares a fixed ~256-CU budget between comm and gemm; comm CUs are stolen from gemm,
  starving it of CU residency + HBM bandwidth, and that loss exceeds the comm-stage saving.
  ndcu=16/pscu=16 is the sweet spot (same as forward L1).
- "Send 2 tokens per warp-iter" (tok_unroll=2: issue U rows' b128 loads before storing, deeper
  MLP) is NEUTRAL (full 2.170->2.161, comm-only 1.644->1.636): the push is already XGMI-bandwidth-
  bound, so more in-flight loads per warp don't help. (Implemented as a forward-safe `tok_unroll=1`
  default kwarg on `dispatch_fp8_copy_tile`; the tok_unroll>1 path uses a clamped-index load + a
  guarded store so it is numerically identical to the tok_unroll=1 path.)
=> STEP1 is comm-bandwidth/contention-bound. CU split and per-send granularity are dead ends; the
   real lever is reducing the cross-rank push VOLUME / padded-pool work, not moving CUs around.

## FlyDSL gotcha: branch fns passed to _emit_if_then / scf_if_dispatch MUST take 0 args
`ReplaceIfWithDispatch._call_branch` injects `result_names` (an empty tuple `()` when there are no
carried results) into any branch fn that ACCEPTS an argument. A branch fn written with default
params (`def _store(dest_row=dest_row): ...`) receives `dest_row=()` -> `() * Int32 + col` raises
`TypeError: can only concatenate tuple (not "Int32") to tuple`. Write branch fns with ZERO params
and capture values via closure (they run immediately, so loop late-binding is not a problem), or
wrap in a 0-arg helper that takes the values as its own args. This matches the `_one_scale`/`_signal`
idiom already in `ep_fp8.py`.

## Benchmark idiom hides missing-fence bugs — validate fence changes by REASONING
The STEP1 bench reuses the same `dy`/routing every iter, so stale-L2 == fresh-L2 and a kernel
with NO fences (`diag=4`) still passes cos≈0.999. Comparing accuracy on a FRESH `dy_acc` the
timing loop never pushed is a stricter gate, but pool_fp8 (~470 MB) ≫ L2 and the inter-run
`torch.cuda.synchronize` evicts warm lines, so even that can't reliably force the stale
condition. => Do NOT accept a fence-removal/coarsening based on the bench passing; prove it from
the coherence argument (single-owner + gated read + one invalidate drops stale, writeback stays
per-block before each sentinel). A coarsened fence that still invalidates once is safe; skipping
invalidation entirely is not (fails in real training with fresh per-step data).

## Do NOT fold the mxfp8 A-scale preshuffle into the gemm workgroup (2-stage) — dead end
Tried removing the separate preshuffle stage + its 2 L2 fences (~0.24 ms) by having each gemm WG
preshuffle its own tile's A-scale into a write-through global scratch, then read it back via the
fast `ScaleS2R` (`PT_MXFP8_BWD_2STAGE=2`, FUSED_PS). Correct (cos 0.999 on fresh dy) but REGRESSED
to 3.66 ms vs 2.15 ms 3-stage (−69%). Three compounding costs the dedicated preshuffle role avoids:
1. **8× redundant** — `n_blocks = N/BLOCK_N` gemm tiles share one `block_m`'s A-scale, but each
   tile re-preshuffles all 256 rows (2048 tiles vs the role's 256 block_m).
2. **write-through is uncached / HBM-latency** — to survive concurrent device-wide `buffer_inv sc1`
   from other WGs' acquires, the 224 KB (4×-inflated broadcast) scratch MUST be sc1 write-through
   (~458 MB HBM-latency stores, no L2 coalescing). A cached store + `l2_writeback` is the very fence
   being removed; a plain store is unsafe (another WG's device-wide invalidate drops the dirty line).
3. **not overlapped** — serial prologue before every tile's K-loop; the role overlaps with comm+gemm.
The raw on-the-fly variant (`=1`, `ScaleS2RRaw`) is also net-negative (~2.8–3.1 ms, scattered loads).
=> The mxfp8 A-scale transform is intrinsically **per-block_m**; a dedicated preshuffle ROLE that
dedups to once/block + uses cached stores + overlaps is MORE efficient than the 2 fences it costs.
The 3-stage pipeline (comm → preshuffle-role → gemm) is the efficient structure; stop trying to
collapse it to 2 stages. LDS-resident preshuffle is also infeasible: one tile's broadcast A-scale
is 224 KB (raw 56 KB) vs only ~32 KB LDS headroom (gemm already uses 128 KB of the 160 KB/CU).

## fp8 STEP1 dgrad does not transfer alone — needs fp8 dW2 wgrad
The bf16 STEP1 produces both `grad_swiglu` and `dispatch_l2_grad` (bf16 scattered dy);
`dispatch_l2_grad` feeds `dW2`. fp8 STEP1 pushes dy fp8-along-H, so keeping dW2 bf16 needs a
dequant (~0.2 ms) that erases the STEP1 win. The coherent unit is STEP1 dgrad + dW2 wgrad;
dgrad quantizes along H, wgrad contracts over the pool/token axis (quantize along pool), so
they need different quantizations of dy (no free pool reuse). Use `grouped_quantize_fp8_with_trans`
(dual-axis) + `grouped_gemm_fp8_variable_k_impl(MX_BLOCKWISE)` for dW2, mirroring
`MXFP8GroupedGEMMFunction.backward` (validated). fp8 weight-grad prefers E5M2 + SNR gate.
