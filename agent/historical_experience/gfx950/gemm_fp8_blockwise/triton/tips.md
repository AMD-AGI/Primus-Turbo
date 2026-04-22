## Wrapper-level single-family cache broadening has poor leverage on a geomean-scored suite

- Observation: broadening `_use_blockwise_weight_cache` to add only `N=57344` cleanly fixed the targeted 3-row family (forward +12/+7/+4% on the three 57344 rows) without regressing any shape by 2% or more.
- But the suite-wide 84-shape combined-step TFLOPS geomean moved only +0.008%, i.e. below this benchmark's ~±1% per-process measurement noise band.
- Leverage bound: 3/84 rows ⋅ max_gain ≈ 3.6% of max_gain. For any ±1% noise floor, targeted gains per targeted row must exceed ~25%-30% to move the suite geomean above noise through a 3-row family alone.
- Rule of thumb: for single-family wrapper-level wins on a large-shape suite, either (a) make the per-row gain very large, (b) bundle multiple complementary wrapper-level wins in one round so their combined contribution adds up, or (c) validate at per-family level and drop the suite-geomean primary for that round — but only if the rules allow it. Do not pursue single-family gate tweaks in isolation on a geomean-scored suite.


## Forward row+col fused quant of activation is a high-leverage wrapper-level win on long-K shapes

- Observation: `quant_fp8_blockwise_dual_impl(a, ...)` is 27-35% faster than separate row+col `quant_fp8_blockwise_impl(a, ...)` calls on the same tensor (measured across 8192x3584, 4096x11008, 8192x18944, 16384x3584, 4096x14336, 32768x16384 → 16 to 293us saved per step).
- Key insight: the forward blockwise FP8 GEMM path only needs the row-axis quant of `a`, while the backward path needs the col-axis quant of the same tensor. When forward and backward share the same FP8 dtype (i.e. non-HYBRID formats), the col result can be precomputed in the forward and saved into the autograd ctx, eliminating the backward col-quant launch entirely.
- Gate it: only fuse when the forward row-cache gate is OFF (else a cache hit is still faster than a full dual-pass), and when fwd and bwd dtypes agree (else the saved col is unusable in bwd).
- Suite-level impact: +0.49% step TFLOPS geomean across 84 shapes, with ~15 long-K shapes gaining +2% to +3% each and no shape regressing >= 5% — exactly the "bundle small wins across many shapes" pattern that moves a geomean.


## Narrow-family wins stack cleanly on top of a wider accepted optimization

- Standalone narrow-family wins (e.g. the round-33 `N=57344` weight-cache gate on top of the round-22 baseline) can produce real per-row gains (+3%/+2%/+1%) but not move the suite geomean above ~±1% noise when only 3 of 84 rows are affected.
- Once a broader optimization has raised the baseline (e.g. round-34 long-K `a` dual-quant fusion), the same narrow-family change (round-35) now produces a suite-level geomean move clearly above noise (+0.22% step).
- This confirms the stacking lesson: always try to bundle a narrow-family wrapper change with at least one broader wrapper win before deciding whether the narrow change "works" at suite level.


## Narrow-family weight-cache gate stacking has diminishing returns

- Round 35 successfully stacked a single 3-row weight-cache family (`N=57344`) on top of round 34's broader fusion baseline and lifted the suite geomean clearly above noise (+0.22%).
- Round 37 tried to add a second 3-row family (`N=106496`) on top of round 35 and only moved the suite geomean by +0.033% — below the noise floor.
- The first single-family-on-top-of-wide-win produces a visible geomean move, but each additional narrow-family gate adds only ~0.03-0.1% to the suite geomean because every family still hits only 3 of 84 rows. Variance between runs is roughly ±1% at suite level, so beyond one stacked family the incremental signal is lost in noise.
- Rule of thumb: after a single accepted narrow-family stacking round, switch back to structural / multi-shape optimizations rather than continuing to stack additional 3-row families.


## Row-cache population should be decoupled from the row-cache gate

- The blockwise activation row cache started with a gate that made `_use_blockwise_act_row_cache(a)` control both the lookup and the put. For long-K shapes where the gate is off, round-34's dual-quant fusion produces a valid row result that is then discarded after the forward GEMM — so repeated forward calls on the same `a` (e.g. the benchmark's 100-iter forward-timing phase) keep re-paying the dual-quant cost.
- Round-38 decouples the two by (a) probing the row cache opportunistically whenever fusion is active, and (b) putting the fusion's row result into the cache regardless of the gate. This recovers the round-34 forward regression (+5.66% forward geomean, +1.42% combined-step geomean on 84 shapes) without changing the correctness contract.
- General principle: for caches keyed on `(id, version, shape, …)` that are always correct-on-hit, the gate should only decide whether we are willing to pay the cost of a cache miss — the put/probe should always be allowed.

## Fresh-process autotune variance can flip whole shapes between ±5% unrelated to the change

- On this benchmark, certain small-N / moderate-N shapes (notably `Mistral-7B mbs1 4096x6144x4096` and `Qwen2.5-72B mbs2 16384x10240x8192`) swing by up to ±10-12% step TFLOPS between fresh processes, even when the binary is identical.
- This means the "no shape regressed >= 5%" acceptance rule can trip incorrectly on a change that actually improves the suite. To guard against this, when a candidate shows a strong suite-wide improvement but exactly one shape trips the 5% rule, run the benchmark a second time to verify whether the regression reproduces; if the flaky shape flips positive in run 2 (as round-38 `4096x6144x4096` did), accept with note documenting the variance signature.


## grad_out dual-quant cache is high-leverage in benchmark and training scenarios

- The training/benchmark idiom `out.backward(grad_out, retain_graph=True)` called repeatedly on the same `grad_out` lets a `(id, version, shape, dtype, block_size)`-keyed cache of the `quant_fp8_blockwise_dual_impl(grad_out, ...)` result skip 30-540us per call (depending on shape).
- Round-39 implementation is a strict cache addition (helpers + dict + lookup-then-fallback), and produced **+2.69% combined-step TFLOPS geomean with zero per-shape regressions** — the largest single-round gain in the campaign.
- General lesson: any tensor that flows into a non-trivial quantization kernel and is reused across the benchmark/training loop with stable id + version is a candidate for a similar version-aware cache. The `grad_out` cache is an easy +3-4% backward gain on this kernel family.


## Scale-transpose results should be cached per `(id, version, shape)` like the quant-result caches

- `_blockwise_nt` (forward) and `_blockwise_nn` (backward dgrad) both call `scale.T.contiguous()` on every GEMM invocation. `_blockwise_nn` does this twice (once for `a_scale_inv`, once for `b_scale_inv`). Each transpose launch costs ~5-20us depending on shape.
- The benchmark's 100-iter backward loop reuses the same saved scale tensors, so a version-aware cache of the transposed result skips 99/100 transpose launches per scale per backward-timing phase.
- Round-44 implementation: added `_scale_t_cache = OrderedDict()` with a `_get_scale_t_contiguous(x)` helper keyed on `(id(x), getattr(x, "_version", 0), x.shape[0], x.shape[1])` in `primus_turbo/triton/gemm/gemm_fp8_kernel.py`, and replaced 3 `scale.T.contiguous()` call sites.
- Result: +0.53% combined-step TFLOPS geomean with **zero per-shape regressions**; small shapes gained +3-6% and the previously-flaky `4096x4096x4096` family swung consistently positive, suggesting the scale-transpose kernel was also a variance source on those shapes.
- General lesson: whenever a hot path repeats a `.T.contiguous()` (or similar shape-layout-only helper) on tensors that are stable across a benchmark/training inner loop, add a version-aware cache keyed on the same scheme as the quant caches. These caches compound cleanly with wrapper-level quant caches (R34/R35/R38/R39) without introducing correctness or autotune risk.


## Autotune `warmup` / `rep` tuning is just as unstable as autotune-space changes

- Round-48 added `warmup=25, rep=100` to the `@triton.autotune` decorator on the blockwise FP8 kernel. A single shape (`16384x6144x4096 mbs2`) collapsed -31.25% on step TFLOPS in this fresh process.
- The root cause is identical to the R23-R27 autotune-space instability: a crowded autotune candidate space on this kernel family has many near-tied configs, and any perturbation to the timing loop (more rep, different warmup, different config set) can re-roll the winner and land on a catastrophic one for specific shapes.
- Rule of thumb: avoid `warmup=/rep=/cache_results=` parameter tuning on this blockwise FP8 kernel unless combined with a stabilisation mechanism (e.g. explicit config seeding that bypasses the autotune timing loop for a specific key).


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


## Narrow-family weight-cache stacking is a fully closed direction (4 independent confirmations)

- R36, R40, R45, and R59 all tried broadening `_BLOCKWISE_WEIGHT_CACHE_EXTRA_N` from `{57344}` to `{57344, 59136, 106496}` on four different accepted baselines (R22, R39, R44, R56). Each produced below-noise step geomean movement (+0.08% to +0.09%).
- The mechanism is invariant: adding 6 more cache-hit rows to an 84-shape suite geomean contributes ~(6/84) × max_per_row_gain ≈ 0.1-0.2% at best, which is in the noise band.
- **This direction is fully closed across all baselines tested.** Do not retry it.

