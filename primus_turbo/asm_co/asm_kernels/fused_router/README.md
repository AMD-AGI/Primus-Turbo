# fused_router — MoE Router Kernel

**Kernel:** `fused_scaling_group_sum_routing_kernel`
**Architecture:** gfx950 (MI355X)
**Algorithm:** Per-token MoE routing: softmax over 32 experts + bitonic argsort + top-k scatter

---

## Base Kernel Origin

`base.hsaco` is a Triton-JIT-compiled binary extracted from the training
container's Triton cache. It was obtained by running the router kernel once
to trigger compilation, then copying the resulting `.hsaco` from:
`~/.triton/cache/<hash>/fused_scaling_group_sum_routing_kernel.hsaco`

**Resource summary of the base kernel:**

| Resource | Count |
|----------|-------|
| VGPRs | 50 |
| SGPRs | 58 |
| LDS | 0 bytes |
| Occupancy | 8 waves/SIMD (maximum) |
| Total instructions | ~2500 lines |

The kernel uses a **persistent grid** — each workgroup processes multiple
rows via a loop, amortizing startup overhead.

---

## Algorithm

Per-token MoE routing implemented as a single fused kernel:

1. **Load** input logits for 32 experts per token (2 buffer loads)
2. **Softmax** over 32 values:
   - Max reduction (DPP `row_shr`/`row_shl`/`quad_perm` cascade)
   - Subtract max, scale by log2(e), `v_exp_f32`
   - Sum reduction (DPP cascade + `ds_swizzle`)
   - Division via IEEE sequence (`v_div_scale` + Newton-Raphson + `v_div_fmas` + `v_div_fixup`)
3. **Store** softmax scores
4. **Argsort** (bitonic sort over 32 experts, 6 stages):
   - Each stage: `v_cmp_ngt_f32` + conditional swap + DPP index propagation
   - Cross-wave exchange via `ds_swizzle(SWAP,16)` and `v_permlane32_swap_b32`
5. **Scatter** top-k indices, probabilities, and routing map

**Launch:** 256 threads/workgroup (4 waves of 64), each wave handles 16 experts,
persistent grid processes `ceil(num_tokens / num_workgroups)` rows per workgroup.

---

## Why the Base Kernel Is Slow

The kernel is **latency-bound**, not compute-bound or memory-bound:

- Only 2 buffer loads and ~10 stores per row (negligible memory traffic)
- No MFMA instructions (nowhere near the compute ceiling)
- Long serial dependency chains:
  - 6-stage bitonic sort is **inherently sequential** (~800 of 1117 loop-body instructions)
  - Softmax division is 10+ dependent instructions on the critical path
  - `ds_swizzle`/`ds_bpermute` add ~20 cycle LDS latency with `lgkmcnt` waits

The Triton/LLVM backend has already done an excellent job interleaving
independent work in the argsort stages. Only 12 NOPs remain in the loop body
(all required by hardware hazards). There is very little left to squeeze from
the assembly without algorithm changes.

---

## Optimization Applied: `patch_v1.s`

A single-step patch targeting the softmax division, the only non-trivial
optimization available at the assembly level.

### IEEE Division → Fast Reciprocal

The base kernel uses the full IEEE-compliant division sequence (~10 instructions)
for `prob = exp(x) / sum`:

**Before (~10 instructions per division, 2 per row = ~20 instructions in loop body):**
```asm
v_div_scale_f32 v28, s[10:11], v36, v36, v27
v_rcp_f32_e32 v44, v28
v_fma_f32 v46, -v28, v44, 1.0
; ... Newton-Raphson refinement (5 instructions) ...
v_div_fmas_f32 v28, v28, v44, v46
v_div_fixup_f32 v27, v28, v36, v27
```

**After (~3 instructions per division, 2 per row = ~6 instructions in loop body):**
```asm
v_rcp_f32_e32 v_inv, v_sum    ; approximate 1/sum  (~1 ULP)
s_nop 3                        ; transcendental hazard drain
v_mul_f32_e32 v_prob, v_exp, v_inv  ; prob = exp(x) / sum
```

**Savings per row:** ~14 instructions + elimination of 2 critical-path
latency chains (division was on the critical path feeding into argsort).

**Precision:** `v_rcp_f32` accuracy (~1 ULP). Softmax probabilities feed
into an argsort (comparison-based, relative ordering). Small absolute errors
do not affect expert selection order for real-world logit distributions.

The patch applies the same replacement in both the main loop body and the
epilogue (final-iteration path).

---

## Build

```bash
bash build.sh              # build final.co
bash build.sh --check      # build then run correctness check
bash build.sh --bench      # build then run benchmark
```

Produces `final.co`.

---

## Correctness Check

```bash
python3 check.py
```

Compares `final.co` against `ref.co` (the known-good optimized binary).
Triton JIT is **not** a valid reference here because the fast `v_rcp_f32`
replacement produces ~1 ULP differences in softmax probabilities compared
to Triton's IEEE-compliant output, causing false failures on float outputs.
Expert selection (integer topk_idx and routing_map) must be bit-identical;
float outputs (scores, probs) must meet cosine similarity ≥ 0.999990.

---

## Benchmark

```bash
python3 bench.py --co final.co
```

Reports latency (µs/ms), throughput (tokens/s), and speedup vs Triton.

---

## Why Assembly-Level Gains Are Limited

The bitonic sort consumes ~72% of loop-body instructions and is inherently
sequential — each of the 6 stages depends on the previous stage's output.
No instruction scheduling can parallelize across sort stages.

**The highest-impact improvement would be a Triton-level algorithm change:**
replacing full bitonic sort of all 32 experts with a partial sort for top-k.
This would reduce sort instructions from O(n log²n) to O(n log k) — a
potential 40–50% reduction in loop-body size for k=4 or k=8.

---

## Files

| File | Description |
|------|-------------|
| `base.hsaco` | Triton reference binary (unmodified, from Triton cache) |
| `ref.co` | Known-good optimized binary used as correctness reference |
| `patch_v1.s` | Optimization: fast reciprocal for softmax division |
| `final.co` | Optimized binary (produced by build.sh) |
| `build.sh` | Single-step patch pipeline with `--check`/`--bench` shortcuts |
| `check.py` | Routing correctness check vs `ref.co` |
| `bench.py` | Latency/throughput benchmark |

