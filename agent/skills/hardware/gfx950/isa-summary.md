# MI350X / MI355X (`gfx950`, CDNA4) ISA Summary

This file is a distilled markdown reference derived from the CDNA4 ISA manual, CDNA4 whitepaper, and MI355X product material so the repo remains usable without external PDF paths.

## Source Materials

- `amd-instinct-cdna4-instruction-set-architecture.pdf`
- `amd-cdna-4-architecture-whitepaper.pdf`
- `amd-instinct-mi355x-gpu-brochure.pdf`

## When to Read This

Read this when you need exact MI355X / `gfx950` capability details beyond the short hardware guide:

- deciding whether to use CDNA4-only matrix paths
- planning `FP8` vs `FP6` / `FP4` / block-scaled kernels
- reasoning about `64`-bank LDS behavior
- checking what changed versus MI300X

## Execution Model

- Wavefront width is `64`.
- The ISA keeps the familiar CDNA shader control model with `PC`, `EXEC`, `VCC`, `SCC`, `MODE`, and `M0`.
- `MODE` still controls rounding, denorm behavior, and VGPR indexing state.
- `VGPR`, `SGPR`, and matrix-accumulation register management all still affect legal encodings and occupancy.

## Register Model

### Architectural registers

- `SGPRs` hold scalar control and descriptor state.
- `VGPRs` hold per-lane vector data.
- Packed and indexed access rules still matter, especially for low-precision conversion paths.

### Matrix accumulation registers

- CDNA4 matrix instructions use accumulation registers (`AccVGPRs`) in addition to normal `VGPRs`.
- `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` move values between the matrix accumulation file and normal vector registers.
- MFMA register groups still need contiguous and alignment-compatible layouts.

## Major Instruction Families

### Compute and control

- Scalar ALU: `SOP*`
- Vector ALU: `VOP*`
- Packed math: `VOP3P`
- VGPR indexing via `S_SET_GPR_IDX_*`
- Compare and predicate instructions updating `VCC` / `EXEC`

### Memory

- Scalar memory instructions
- Vector buffer instructions (`MUBUF`)
- Typed buffer instructions (`MTBUF`)
- Flat / global / scratch instructions
- Float atomics
- LDS instructions

## Matrix Core / MFMA Capability

CDNA4 keeps dense and sparse `MFMA`, but the important change is the expansion of the low-precision matrix family.

### Dense / sparse matrix support

- Dense `MFMA`
- Structured sparse `2:4` matrix support
- `AccVGPR`-backed accumulation model

### Matrix datatypes

- `FP64`
- `FP32`
- `FP16`
- `BF16`
- `INT8`
- `BF8`
- `FP8`
- new `FP6`
- new `BF6`
- new `FP4`

### CDNA4-only matrix paths

- `V_MFMA_F32_*_F8F6F4`
- `V_MFMA_SCALE_F32_*_F8F6F4`
- block-scaled low-precision MFMA with shared exponent bias

### Practical MFMA rules

- The matrix core still uses an outer-product-oriented internal model, so register layouts are instruction-specific.
- Independent instructions are required between MFMA issue and later use of results or source overwrite.
- Register groups for inputs and outputs must remain contiguous and aligned.

## Block-Scaled Low Precision

This is the largest portability difference versus MI300X.

- `CVT_SCALE_*` instructions convert between higher precision and `FP8` / `BF8` / `FP6` / `BF6` / `FP4`.
- These scale-aware conversions use an `E8M0` exponent bias shared across blocks of `32` K values.
- On CDNA4, conversion and scaling are part of the kernel design space, not just preprocessing.

## Memory and LDS Capability

### Memory-path capability

- `Memory Buffer Load to LDS` exists.
- CDNA4 also adds `MFMA Transpose Load from LDS`, which can remove some explicit transpose / shuffle work before matrix execution.

### LDS model

- LDS capacity is `160 KiB` per CU.
- LDS has `64` banks on this generation.
- Larger staged tiles are more realistic, but every swizzle and bank-conflict rule from MI300X must be reconsidered.

## Important CDNA4 Caveats

- There is no hardware-native `TF32` matrix path on CDNA4.
- Hardware support for new low-precision datatypes does not remove the need for strict validation of software format, scaling, and accumulator choices.
- More LDS does not mean free occupancy; register pressure and staging depth can still collapse concurrency.

## Optimization Consequences

- If a low-precision kernel is K-heavy, evaluate block-scaled `F8/F6/F4` MFMA early instead of treating it as a late-stage novelty.
- If a matrix kernel spends too much time permuting data into MFMA-friendly layout, check whether `MFMA Transpose Load from LDS` can replace part of that work.
- Use CDNA4's larger LDS budget to explore deeper staging, but always re-check occupancy and bank conflicts under the `64`-bank model.
- Because TF32 is not hardware-native here, frame FP32-like optimization choices around `BF16`, `FP16`, or explicitly validated mixed-precision alternatives.

## Do Not Assume

- Do not assume MI300X TF32 guidance applies.
- Do not assume MI300X `32`-bank LDS padding rules still work.
- Do not assume plain FP8 is always the best low-precision choice when CDNA4 scale-aware paths are available.
- Do not assume a CDNA4-specific kernel structure will backport well to `gfx942`.
