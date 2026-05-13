## Forward row+col fused quant of activation is a high-leverage wrapper-level win on long-K shapes

- Observation: `quant_fp8_blockwise_dual_impl(a, ...)` is 27-35% faster than separate row+col `quant_fp8_blockwise_impl(a, ...)` calls on the same tensor (measured across 8192x3584, 4096x11008, 8192x18944, 16384x3584, 4096x14336, 32768x16384 → 16 to 293us saved per step).
- Key insight: the forward blockwise FP8 GEMM path only needs the row-axis quant of `a`, while the backward path needs the col-axis quant of the same tensor. When forward and backward share the same FP8 dtype (i.e. non-HYBRID formats), the col result can be precomputed in the forward and saved into the autograd ctx, eliminating the backward col-quant launch entirely.
- This is a kernel fusion (single read of `a` produces two quant outputs) and a same-`ctx` data flow — it is independent of any cross-iteration tensor identity, so the gain transfers cleanly to real training and is not a benchmark-only artifact.
- Gate it: only fuse when fwd and bwd dtypes agree (else the saved col is unusable in bwd), and when `a` is contiguous (the dual kernel requires contiguous input).


## Autotune `warmup` / `rep` tuning is just as unstable as autotune-space changes

- Round-48 added `warmup=25, rep=100` to the `@triton.autotune` decorator on the blockwise FP8 kernel. A single shape (`16384x6144x4096 mbs2`) collapsed -31.25% on step TFLOPS in this fresh process.
- The root cause is identical to the R23-R27 autotune-space instability: a crowded autotune candidate space on this kernel family has many near-tied configs, and any perturbation to the timing loop (more rep, different warmup, different config set) can re-roll the winner and land on a catastrophic one for specific shapes.
- Rule of thumb: avoid `warmup=/rep=/cache_results=` parameter tuning on this blockwise FP8 kernel unless combined with a stabilisation mechanism (e.g. explicit config seeding that bypasses the autotune timing loop for a specific key).


## Fresh-process autotune variance can flip whole shapes between ±5% unrelated to the change

- On this benchmark, certain small-N / moderate-N shapes (notably `Mistral-7B mbs1 4096x6144x4096` and `Qwen2.5-72B mbs2 16384x10240x8192`) swing by up to ±10-12% step TFLOPS between fresh processes, even when the binary is identical.
- This means the "no shape regressed >= 5%" acceptance rule can trip incorrectly on a change that actually improves the suite. To guard against this, when a candidate shows a strong suite-wide improvement but exactly one shape trips the 5% rule, run the benchmark a second time to verify whether the regression reproduces; if the flaky shape flips positive in run 2, accept with a note documenting the variance signature.


## Shared-kernel guard-flag changes regress strided backward paths even when helping contiguous forward

- Round-53 added an `EVEN_K: tl.constexpr` to `_blockwise_fp8_autotune_kernel` to skip mask arithmetic when `K % BLOCK_K == 0` (true for all benchmark shapes). Forward gained +0.92% geomean but backward regressed -1.49% across 27 of 84 shapes, yielding -0.92% combined-step.
- The shared kernel services 3 different layouts (NT forward, NN dgrad, TN wgrad) with different `A_K_CONTIGUOUS` / `B_K_CONTIGUOUS` / `SCALE_2D_B` combinations. Forward (all-contiguous) benefits from unmasked loads, but backward (one or both strided) regresses when the mask branch is removed — the compiler seems to use the mask predicate to inform memory-access scheduling on strided paths.
- This matches R26's documented fragility: any shared-kernel code-gen-guard change (modulo simplification, mask skip, etc.) interacts non-trivially with the compiler's layout heuristics and will not cleanly improve all 3 layouts at once.
- Rule of thumb: shared-kernel guard flags are a closed direction on this kernel family. If forward-only gains are needed, duplicate the kernel source into layout-specific copies first.


## Layout-specific kernel duplication unlocks per-layout guard flags safely

- The shared `_blockwise_fp8_autotune_kernel` serves all 3 layouts (NT forward, NN dgrad, TN wgrad). Adding guard flags (e.g. `EVEN_K`, modulo simplification) to the shared kernel regresses the strided backward paths (R26, R53 both rolled back).
- Round-55 duplicates the kernel into a forward-only `_blockwise_fp8_nt_kernel` with `A_K_CONTIGUOUS=True, B_K_CONTIGUOUS=True, SCALE_2D_B=True` hardcoded and an `EVEN_K` fast path. Forward gained +1.04% geomean across two fresh-process runs with backward completely unaffected.
- General lesson: when a shared kernel's guard-flag change helps one layout but hurts others, duplicating the kernel source and specializing the duplicate is the safe path. The Triton compiler lowers both copies independently, so there is no code-gen interaction.
- The same pattern should work for dgrad (`A_K_CONTIGUOUS=True, B_K_CONTIGUOUS=False, SCALE_2D_B=True`) and potentially for wgrad (`A_K_CONTIGUOUS=False, B_K_CONTIGUOUS=False, SCALE_2D_B=False`) — they can be individually specialized without the R26/R53 fragility.


## Layout-specific kernel duplication pattern scales across all 3 layouts

- R55 and R56 each duplicated the shared `_blockwise_fp8_autotune_kernel` into a layout-specialised copy (forward NT and backward NN respectively) with `EVEN_K` mask-skip.
- Both accepted cleanly: R55 gained fwd +1.04% geomean, R56 gained bwd +0.636% geomean, and neither regressed shapes the other layout is responsible for.
- The pattern generalises: when a layout has `X_K_CONTIGUOUS` / `SCALE_2D_B` flags hardcodable and the K dimension is commonly a multiple of `BLOCK_K`, duplicating the kernel and adding an `EVEN_K` fast path is a robust way to reclaim the per-K overhead that the shared kernel cannot drop without regressing the other layouts.
- Combined impact of the pattern across two layouts: step TFLOPS geomean +0.78% on an 84-shape suite with zero per-shape regressions `>= 2%`.


## `EVEN_K` mask-skip is safe for all-contiguous and one-strided layouts but NOT dual-strided

- Round 57 tried applying the R55/R56 kernel-duplication pattern with `EVEN_K` mask-skip to TN wgrad (`A_K_CONTIGUOUS=False, B_K_CONTIGUOUS=False`). Result: bwd -2.10% geomean, 40 shapes regressed >= 2%.
- The TN wgrad layout has **both** A and B strided in K, unlike NT (both contiguous, R55 accepted) and NN (only B strided, R56 accepted). The dual-strided load pattern is where Triton's code generator needs the mask predicate to produce efficient scheduling; removing it via the EVEN_K fast path produces systematically worse code.
- General rule for this kernel family: the `EVEN_K` duplication pattern works for layouts with 0 or 1 strided-K tensors, and is harmful for layouts with 2 strided-K tensors. Do not attempt the duplication on TN (wgrad).


## Adding `matrix_instr_nonkdim=16` autotune candidates only on AGPR-overflow paths is safe and net-positive

- Round 4 (campaign `gemm_fp8_blockwise_triton_gfx950_20260512`) added `matrix_instr_nonkdim=16` as an extra autotune candidate for NN (dgrad), but **only stacked it on `BLOCK_M=256` configs**. `BLOCK_M=128 num_warps=4` already needs only 16 AGPR/wave and does not overflow, so duplicating those configs is pure autotune cost with no upside.
- Result on 84-shape suite: combined +0.16% geomean, bwd +0.45%, 24/9 shapes >=+1% / <=-1% on bwd. Best movers: TestID 64 (Qwen2.5-72B, K=29568) bwd +5.96% and TestID 24 (Llama-2-70B, K=28672) bwd +5.86%, where 32x32x64 MFMA overflows from 16 AGPR/warp and 16x16x128 (4 AGPR/warp) wins.
- Autotuner picked `nonkdim=16` for 49 of 69 deduped NN keys (71%), all on the `BM=256 nw=8 ns=3` path — confirming the AGPR-overflow targeting hypothesis.
- The trade-off cost is autotune time: NN candidate count 96 → 144, cold-autotune wall-time +50%. For production deployments, the right follow-up is to capture the `nonkdim=16`-selecting shapes and serve them via a pinned-config table, then drop the candidate to reduce ongoing autotune cost.
- Pattern for other AGPR-bound kernels: add the smaller-instruction MFMA variant as a candidate **only on the path that actually overflows** (read the dev-branch `.amdgcn` and count `v_accvgpr_*` per K-loop iteration; if >50 instructions, that path is the candidate). Do not enable globally.


## Enabling `use_in_thread_transpose` knob globally on gfx950 regresses paths shared with `matrix_instr_nonkdim=16`

- Round 5 (campaign `gemm_fp8_blockwise_triton_gfx950_20260512`) tried `triton.knobs.amd.use_in_thread_transpose = True` to push wgrad's remaining 50% VGPR-staged `buffer_load_dwordx4` through `amdg.in_thread_transpose` to direct-LDS.
- Result on 84-shape suite: combined -0.51% vs round-4 (last accepted), bwd -0.65%, 4 shapes combined <=-5% and 6 shapes bwd <=-5%, 0 shapes >=+5% to offset.
- The worst regressors (TestID 37/39/43 Llama-3.1-405B K=16384, TestID 20/24 Llama-2-70B K=28672) are **exactly the shapes that round-4 `nonkdim=16` helped most**. The autotuner kept selecting `nonkdim=16` on those shapes (round-5 picked `nonkdim=16` for 51 NN keys vs 49 in round-4), but the same selected config compiled into a slower binary in round-5. Mechanism: the `tritonamdgpu-in-thread-transpose` pass picks a transposable_layout / shared layout to match the dot operand's `amd_mfma` layout, and that layout differs between `16x16x128` (round-4 best on these shapes) and the default `32x32x64`. When both passes are on, the cross-product likely emits extra register moves or LDS bank conflicts in the 16x16x128 path.
- gfx950 ships with `is_in_thread_transpose_enabled` returning `False` by default (only gfx942 / gfx120 default-on). This default decision has cross-layout regression testing behind it; flipping it globally without per-config gating is unsafe.
- Rule of thumb: pass-level backend knobs that touch dot-operand shared layouts (`use_in_thread_transpose`, `scalarize_packed_fops` partial knobs, etc.) must be evaluated **per autotune key**, not as a global switch, because they interact with whatever MFMA shape the autotuner selects per shape. If a future round wants to recover the wgrad VGPR-staged path, the right route is either (a) wgrad-specific kernel duplication + force the knob on only that compile path, or (b) extend the autotune config to carry an `in_thread_transpose` hint and let the autotuner pick.


## Transposed-store epilogue eliminates wgrad `buffer_store_short` x64 → `buffer_store_dwordx4` x8

- Round 6 (campaign `gemm_fp8_blockwise_triton_gfx950_20260512`) added a wgrad-specialised kernel `_blockwise_fp8_tn_kernel` whose only change vs the shared kernel is the store epilogue: `acc_t = tl.trans(acc.to(C_ptr.type.element_ty))` followed by pointer addressing that puts the BM axis innermost.
- Root cause: wgrad runs under `trans_c=True` (output buffer `(N, M)` with `stride_cm=1, stride_cn=N`). The original `tl.store(c_ptrs, acc[BM, BN], mask)` had BN strided in memory and BM contiguous, but acc's leading dim was BM. Without the transpose, Triton serialised the BF16 write into 64 element-wise `global_store_short[_d16_hi]` per tile (32 low + 32 high).
- Result on 84-shape suite: combined +1.37% vs round-4 (last accepted), bwd +1.98%, **4 shapes combined ≥+5%** (max +7.22%) and **9 shapes bwd ≥+5%** (max +8.35%). Per-shape distribution heavily right-skewed on bwd (59 shapes ≥+1% vs 7 shapes ≤−1%).
- ASM-level verification: 96/96 cached TN binaries write via `(buffer|global)_store_dwordx4 × 4-8` instead of `buffer_store_short × 64`. The root cause is fully eliminated, not just suppressed.
- The trade-off: 3 shapes regressed ≥5% combined, all with **N ≥ 57344** (e.g. TID 43 N=106496 K=16384, TID 39 N=106496 K=16384, TID 15 N=57344 K=8192). Hypothesis: at very large BN tile counts the `tl.trans(acc)` is spilled to LDS exchange because register pressure is tight, and the per-tile cost exceeds the saved store overhead.
- Pattern for other strided-output kernels on gfx950: if a kernel's output buffer has `stride[contiguous_axis] != acc_axis_layout_contiguous`, the BF16 epilogue will serialise into `short` stores. The fix is a single `tl.trans(acc.to(bf16))` plus swapped pointer indexing — it does not require changing the K-loop, the scale path, or the MFMA layout, and is therefore a low-risk, high-ROI change. Be aware of the large-N trade-off and consider gating the transpose on `BN <= 128` if the large-N subspace is critical.


## P1-#7.4 (`BK=64` for small-K shapes) is blocked by the blockwise scale 128-wise contract

- The blockwise FP8 scale tensors carry one scale per 128-element K stripe. Inside every blockwise GEMM kernel the K loop loads scales as `tl.load(as_ptrs + ki * stride_as_k)`, advancing one scale per K-tile and implicitly assuming `BLOCK_K == 128`.
- Allowing `BLOCK_K=64` requires reindexing the scale loads (`ki // (128 // BLOCK_K)`) plus updating mask logic in three kernels (NT, NN dgrad, TN wgrad). The change is structurally non-trivial and breaks the "one change per round" budget on its own.
- Expected payoff is bounded: only the K ≤ 4096 subspace (≈12% of typical 84-shape suite) is K-loop pipeline starved at BLOCK_K=128, and even there the autotuner picks `BM=128 nw=4` configs which already amortise the prologue.
- Recommendation: when revisiting small-K performance, prefer alternative directions (XCD/grid scheduling, persistent kernel reuse) before paying the scale-reindex refactor cost. `GROUP_M ∈ {4, 8}` is already the autotune sweet-spot edge (122/207 keys land on 8, 85/207 on 4 in this campaign), so widening `GROUP_M` to 2 or 16 has been observed to have <1% expected gain and high noise risk.
