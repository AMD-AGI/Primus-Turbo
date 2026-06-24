# grouped_gemm_wgrad — FP8 WGRAD Grouped GEMM

**Kernel:** `grouped_variable_k_dot_scaled_kernel`
**Architecture:** gfx950 (MI355X)
**Operation:** `C = beta * C + A^T @ B * scale` (WGRAD: trans_a=True, trans_b=False)

---

## Base Kernel Origin

`base.co` is the **load-hoisted `dot_scaled_v2`** binary — the ggemm-asm
RHS-buffer-load-hoisting optimization applied on top of the Triton-JIT
`grouped_variable_k_dot_scaled_kernel`. It is produced by:

```
patch_co.py  ggemm-asm/kernels/dot_scaled_compiled.co \
             ggemm-asm/kernels/dot_scaled_v2.s \
          -> base.co
```

`dot_scaled_compiled.co` is the raw Triton-JIT binary (produced by
`grouped_vark_dot_scaled.py`). `dot_scaled_v2.s` adds RHS buffer_load
hoisting (+0.8%) and removes a redundant `s_waitcnt lgkmcnt(0)`.

The primus-specific patches (`patch_cbsz_fix.s`, `patch_beta1.s`) are then
applied on top of this already-optimized base.

### To reproduce `base.co` from scratch

`llvm-mc` must be in `PATH` (or at `/opt/rocm/llvm/bin/llvm-mc`).

```bash
# From the primus root directory:
python3 asm_kernels/tools/patch_co.py \
  ggemm-asm/kernels/dot_scaled_compiled.co \
  ggemm-asm/kernels/dot_scaled_v2.s \
  asm_kernels/grouped_gemm_wgrad/base.co \
  --llvm-mc /opt/rocm/llvm/bin/llvm-mc
```

If `dot_scaled_compiled.co` is ever missing, regenerate it by running
`grouped_vark_dot_scaled.py` inside the primus ROCm container to trigger
Triton JIT, then copy the cached `.co` from `~/.triton/cache/`.

The pre-built `base.co` is committed to the repo for convenience.

The original kernel contained two compounding bugs (see Bug section below)
that caused gradient norm instability during training. The base binary is
preserved here unmodified as the patch chain starting point.

---

## Bugs in the Base Kernel

### Bug 1: CBSZ encoding error (2x scale error)

All 16 `v_mfma_f32_32x32x64_f8f6f4` instructions were encoded with
`cbsz:1` (BF8/E5M2 format for operand A), but the input data is FP8 E4M3.
The `cbsz` field selects the floating-point format of source operand A; the
wrong encoding causes the hardware to reinterpret E4M3 bit patterns as E5M2,
introducing a systematic 2x scale error in every MFMA output.

### Bug 2: Missing OCP implicit scale correction (4x scale error)

`v_mfma_f32_32x32x64_f8f6f4` on gfx950 applies an implicit 0.25x downscale
to OCP E4M3 data as part of the hardware specification. The Triton kernel
compensates by multiplying the explicit scale factors by 4.0 before launch;
the ASM kernel omitted this correction.

**Combined effect:** 1/2 × 1/4 = 1/8 of correct output value → BF16
overflow on real training data → NaN in `grad_norm`.

---

## Patch Chain

Two sequential patches are applied to produce `final.co`:

```
base.co
  ↓ patch_cbsz_fix.s
intermediate.co        (cbsz:1 → cbsz:0 in all 16 MFMA instructions)
  ↓ patch_beta1.s
final.co               (adds C = C + A^T@B accumulation trampoline)
```

### Patch 1: `patch_cbsz_fix.s` — cbsz correction

Modifies all 16 `v_mfma_f32_32x32x64_f8f6f4` instructions to use `cbsz:0`
(FP8 E4M3 format). In the 8-byte instruction encoding:

```
[VDST_lo] [flags_with_cbsz] [0xAE] [0xD3] [src0] [src1] [src2] [modifiers]
```

Bit 0 of byte 1 (the `cbsz` field) is cleared from `0x01` to `0x00`.
This is the minimal fix for the 2x scale error.

### Patch 2: `patch_beta1.s` — fused accumulation (beta=1)

Adds an in-kernel accumulation trampoline that implements
`C = C + A^T @ B * scale` instead of `C = A^T @ B * scale`.

**The trampoline implements (190 instructions injected into NOP padding):**

1. Set up C buffer descriptor bounds/format fields (2 instructions)
2. Issue 8 × `buffer_load_dwordx4` to load the previous C tile (16 instructions)
3. Apply `v_pk_mul_f32` scale to the remaining 25 accumulator pairs during load latency
4. `s_waitcnt vmcnt(0)`
5. Reverse `v_permlane32_swap_b32` to undo memory→MFMA layout transform (16 instructions)
6. Unpack BF16 → FP32 and add to FP32 accumulator (128 instructions):
   - Low BF16: `v_lshlrev_b32 tmp, 16, src` then `v_add_f32 acc, tmp, acc`
   - High BF16: `v_and_b32 tmp, 0xffff0000, src` then `v_add_f32 acc, tmp, acc`
7. `s_branch .L_BETA1_RET` back to the main store epilogue

The trampoline reuses dead VGPRs from the MFMA K-loop (v70–v107) for load
destinations and v94 as a scratch temporary. No register pressure increase;
kernel metadata is unchanged.

The beta=1 kernel **always accumulates** — pass a zero-initialized C to get
beta=0 semantics.

---

## Build

```bash
bash build.sh
# Produces: intermediate.co and final.co
```

---

## Correctness Check

```bash
python3 check.py
```

Compares `final.co` against the Triton `dot_scaled` kernel for all
production shapes. Expected results:

```
Shape (2880, 2880), G=32, tokens=131072: cos_sim=1.000000  PASS
Shape (2880, 5760), G=32, tokens=131072: cos_sim=1.000000  PASS
```

---

## Benchmark

```bash
python3 bench.py
```

Reports TFLOPS, memory bandwidth (GB/s), latency (ms), and speedup vs
Triton for both beta=0 and beta=1 modes.

| Kernel | TFLOPS | Beta=1 overhead |
|--------|--------|-----------------|
| ASM fixed (beta=0) | ~2018 | — |
| ASM beta=1 | ~1920 | 5.1% |
| Triton (beta=0) | ~2007 | — |
| Triton (beta=1) | ~1757 | 12.5% |

---

## Files

| File | Description |
|------|-------------|
| `base.co` | Original broken kernel (cbsz:1, for reference only) |
| `patch_cbsz_fix.s` | Patch 1: fix cbsz:1 → cbsz:0 in all MFMA instructions |
| `patch_beta1.s` | Patch 2: inject beta=1 accumulation trampoline |
| `intermediate.co` | After patch 1 (produced by build.sh) |
| `final.co` | After patch 2, production-ready (produced by build.sh) |
| `build.sh` | Two-step patch pipeline |
| `check.py` | Cosine-similarity correctness check vs Triton |
| `bench.py` | TFLOPS/bandwidth/latency benchmark |

