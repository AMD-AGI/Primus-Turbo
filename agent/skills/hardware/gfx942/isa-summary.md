# MI300X / MI325X (`gfx942`, CDNA3) ISA Summary

This file is a distilled markdown reference derived from the MI300 CDNA3 ISA manual and CDNA3 whitepaper so the repo remains usable without external PDF paths.

## Source Materials

- `amd-instinct-mi300-cdna3-instruction-set-architecture.pdf`
- `amd-cdna-3-white-paper.pdf`

## When to Read This

Read this when you need exact MI300X / `gfx942` instruction-model guidance beyond the short hardware skill pages:

- MFMA capability checks
- low-precision format planning
- LDS / memory-path reasoning
- register and wavefront model details
- CDNA3 vs CDNA4 porting questions

## Execution Model

- Wavefront width is `64`.
- The ISA exposes standard shader control state such as `PC`, `EXEC`, `VCC`, `SCC`, `MODE`, and `M0`.
- `SGPR` and `VGPR` allocation and alignment rules matter for occupancy and legal encodings.
- `M0` participates in VGPR indexing and memory-descriptor style control.
- The `MODE` register controls rounding, denorm handling, and VGPR indexing enablement.

## Register Model

### Architectural registers

- `SGPRs` carry scalar state, descriptors, loop invariants, and control data.
- `VGPRs` carry per-lane vector operands and results.
- Out-of-range VGPR reads fall back to `VGPR0`; out-of-range destinations do not write results.

### Matrix accumulation registers

- CDNA3 matrix instructions use separate accumulation registers (`AccVGPRs` / AGPR-style storage).
- `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` move data between normal vector registers and accumulation registers.
- MFMA operands and outputs must use contiguous, alignment-compatible register groups.

## Major Instruction Families

### Compute and control

- Scalar ALU: `SOP*`
- Vector ALU: `VOP*`
- Packed math: `VOP3P`
- Compare and predicate instructions writing `VCC` / `EXEC`
- VGPR indexing via `S_SET_GPR_IDX_*`
- DPP / SDWA lane-control forms in the vector ISA

### Memory

- Scalar memory instructions
- Vector buffer instructions (`MUBUF`)
- Typed buffer instructions (`MTBUF`)
- Flat / global / scratch instructions
- Float atomics
- LDS instructions and `GWS`

## Matrix Core / MFMA Capability

CDNA3 matrix cores support both dense and sparse matrix-fused-multiply-add (`MFMA`) execution.

### Dense MFMA datatypes

- `FP64`
- `FP32`
- `FP16`
- `BF16`
- `INT8`
- `BF8`
- `FP8`

### Additional CDNA3 matrix paths

- CDNA3 also documents reduced-precision `TF32` / `XF32` style matrix support.
- Sparse `2:4` matrix forms are supported when the workload can satisfy the required sparsity structure.

### Practical MFMA rules

- The matrix core primitive is outer-product based; output layouts are not naive row-major register layouts.
- Inputs and outputs are packed into wave lanes and contiguous register groups using fixed per-instruction layouts.
- Independent instructions are often required between an `MFMA` issue and any use of its results or overwrite of its sources.
- When exact layouts matter, use the AMD matrix instruction calculator rather than guessing from intuition.

## Low-Precision and Conversion Capability

- The vector ISA includes explicit `BF8` / `FP8` convert and packed-convert instructions.
- CDNA3 supports matrix paths for `FP8` and `BF8`.
- The ROCm software ecosystem around `gfx942` is often handled as `FNUZ` at the API / type level, so software-visible format validation is still required even when the ISA exposes FP8/BF8 capability.

### Important absence

CDNA3 does **not** provide the CDNA4 block-scaled low-precision ISA family:

- no native `FP6`
- no native `BF6`
- no native `FP4`
- no `V_MFMA_SCALE_*`
- no `CVT_SCALE_*`

## Memory and Synchronization Model

### Memory-path capability

- `Memory Buffer Load to LDS` exists, so buffer-backed global-to-LDS staging is part of the ISA toolbox.
- Vector buffer, flat, global, and scratch memory instructions all exist and must be selected with intent.

### LDS model

- LDS capacity is `64 KiB` per CU.
- LDS has `32` banks on this generation.
- LDS instructions support indexed access and atomics.

### Ordering and coherence

- The ISA and whitepaper both imply a relaxed vector-cache coherence model; strong ordering still requires explicit synchronization.
- `GWS`, wait-state handling, and memory-ordering instructions are all part of legal synchronization strategy.

## Optimization Consequences

- Default GEMM mental model on MI300X: classic `MFMA` plus careful buffer-to-LDS staging.
- Because LDS is only `64 KiB`, aggressive staging competes quickly with occupancy.
- Use sparse `MFMA` only when the algorithm can truly sustain the required structured sparsity.
- Keep TF32 / XF32 in mind for FP32-adjacent workloads on CDNA3; this is a real hardware difference vs CDNA4.
- Validate the exact FP8 software format used by the stack before trusting numerical results.

## Do Not Assume

- Do not assume CDNA4 `64`-bank LDS formulas apply here.
- Do not assume `MFMA Transpose Load from LDS` exists on MI300X.
- Do not assume block-scaled `MXFP` paths exist.
- Do not assume a kernel structure tuned around `160 KiB` LDS will port cleanly to `gfx942`.
