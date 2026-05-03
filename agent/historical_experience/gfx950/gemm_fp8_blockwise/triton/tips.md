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
