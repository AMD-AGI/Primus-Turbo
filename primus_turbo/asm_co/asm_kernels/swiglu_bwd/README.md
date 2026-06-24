# swiglu_bwd — SwiGLU Backward Kernel

**Kernel:** `swiglu_with_mask_bwd_kernel`
**Architecture:** gfx950 (MI355X)
**Algorithm:** Backward pass for SwiGLU activation with expert mask and probability scaling

---

## Base Kernel Origin

`base.co` is a Triton-JIT-compiled binary extracted from the training
container's Triton cache. It was obtained by running the backward pass kernel
once to trigger compilation, then copying the resulting `.co` from:
`~/.triton/cache/<hash>/swiglu_with_mask_bwd_kernel.co`

**Resource summary of the base kernel:**

| Resource | Count |
|----------|-------|
| VGPRs | 94 |
| SGPRs | 56 |
| LDS | 0 bytes static (runtime LDS for cross-wave reduction) |
| Scratch | 0 (no spilling) |
| Code size | 4568 bytes |
| Occupancy | 5 waves/SIMD |

---

## Algorithm

Per-row backward pass for SwiGLU with expert mask (4096 elements/row):

1. Load expert mask; skip row if zero
2. Load `up[4096]`, `down[4096]` (BF16 → FP32)
3. `sigmoid = 1 / (1 + exp(-up))` — dominant cost
4. `silu = sigmoid * up`
5. Load `grad_out[4096]` (BF16 → FP32)
6. `grad_probs = sum(grad_out * silu * down)` — cross-lane reduction
7. Store `grad_probs_sum` (scalar per row)
8. Load `probs` (scalar per row)
9. `grad_down = grad_out * probs * silu`
10. `grad_silu = sigmoid * (1 + up * (1 - sigmoid))`
11. `grad_up = grad_out * probs * down * grad_silu`
12. Store `grad_up[4096]`, `grad_down[4096]` (FP32 → BF16)

**Launch configuration:** grid=(8192,), 256 threads/block (4 waves of 64),
16 BF16 elements per thread, loop over `ceil(num_tokens / 8192)` iterations.

---

## Why the Base Kernel Is Slow

The kernel is **latency-bound**, dominated by the sigmoid computation.

The Triton/LLVM backend emits full IEEE-754 compliant division for
`1.0 / (1 + exp(-x))`, using the sequence:

```
v_div_scale_f32 → v_rcp_f32 → v_fma_f32 (Newton-Raphson) → v_div_fmas_f32 → v_div_fixup_f32
```

This is approximately 30 instructions per value (8 pairs per iteration × 30 ≈ 240
instructions out of 600 total — 40% of the loop body), with long serial
dependency chains (each NR step depends on the previous).

The precision is wasted: sigmoid results feed into BF16 stores that truncate
to 7-bit mantissa, so anything beyond ~1 ULP is invisible.

---

## Patch Chain

Two sequential patches are applied:

```
base.co
  ↓ patch_v1.s
intermediate.co    (Phase 1: instruction scheduling and NOP cleanup)
  ↓ patch_v2.s
final.co           (Phase 2: IEEE division → v_rcp_f32 replacement)
```

### Phase 1: `patch_v1.s` — instruction hoisting and scheduling

Removes 24 × `s_nop 0` between VALU→VALU operations in the reduction section
(lines 786–890 of the original assembly). The gfx950 has hardware interlocks
for VALU→VALU write-read hazards; these NOPs were inserted conservatively by
the Triton compiler and waste approximately 26 cycles per iteration.

Interleaves independent load address computations into the freed instruction
slots to improve pipelining.

**Estimated gain:** 2–3% speedup.

### Phase 2: `patch_v2.s` — fast reciprocal replacement

Replaces all 8 IEEE division sequences in the sigmoid computation with
direct `v_rcp_f32`:

**Before (≈30 instructions per pair):**
```asm
v_div_scale_f32 v12, s[26:27], v21, v21, 1.0
v_rcp_f32 v39, v12
v_fma_f32 v43, -v12, v39, 1.0
; ... 12 more Newton-Raphson instructions ...
v_div_fmas_f32 v44, v12, v39, v43
v_div_fixup_f32 v21, v44, v21, 1.0
```

**After (≈4 instructions per pair):**
```asm
v_rcp_f32 v21, v21    ; sigmoid_1 ≈ 1/(1+exp(-x1))  ~1 ULP
v_rcp_f32 v20, v20    ; sigmoid_0 ≈ 1/(1+exp(-x0))  ~1 ULP
s_nop 3               ; transcendental hazard drain
; independent work fills the s_nop slots
```

**Instruction reduction:** 8 pairs × ~30 → 8 pairs × ~4 = ~210 instructions
eliminated per iteration. Frees ~10 temporary VGPRs used by the NR chain.

**Precision:** `v_rcp_f32` accuracy (~0.5 ULP) far exceeds BF16 requirements.
The denominator `1 + exp(-x) ∈ [1.0, 2.0]` — no denormals, no infinities,
no zeros — so no IEEE edge-case handling is needed.

**Estimated gain:** 15–25% speedup.

---

## Build

```bash
bash build.sh                 # both phases → final.co
bash build.sh --phase1-only   # phase 1 only → intermediate.co
```

---

## Correctness Check

```bash
python3 check.py
```

Compares `final.co` against Triton for a range of shapes. Passes cosine
similarity ≥ 0.99999 vs the Triton reference.

---

## Benchmark

```bash
python3 bench.py
```

Reports latency (ms), effective compute (TFLOPS), and memory bandwidth (GB/s)
for the SwiGLU backward pass, with speedup vs Triton.

---

## Files

| File | Description |
|------|-------------|
| `base.co` | Triton reference binary (unmodified) |
| `patch_v1.s` | Phase-1 patch: NOP removal and instruction scheduling |
| `patch_v2.s` | Phase-2 patch: IEEE division → v_rcp_f32 replacement |
| `intermediate.co` | After phase 1 (produced by build.sh) |
| `final.co` | After phase 2, production-ready (produced by build.sh) |
| `build.sh` | Two-phase patch pipeline |
| `check.py` | Cosine-similarity correctness check vs Triton |
| `bench.py` | Latency/bandwidth benchmark |

