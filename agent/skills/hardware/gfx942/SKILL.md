---
name: gfx942
description: MI300X and MI325X hardware guidance for AMD kernel optimization. Use when the target GPU is gfx942, MI300X, or MI325X, or when tuning kernels against CDNA3 limits such as 304 CUs, 64 KiB LDS, 32-bank LDS, and FP8 FNUZ behavior.
---

# gfx942 Hardware Guide

Use this guide when the target GPU is `gfx942` (`MI300X` or `MI325X`).

## Quick Facts

| Item | Value |
|------|-------|
| Architecture | CDNA3 |
| Active CUs | 304 |
| Wavefront size | 64 |
| LDS per CU | 64 KiB |
| LDS banks | 32 |
| FP8 ecosystem | FNUZ |
| Common build target | `--offload-arch=gfx942` |

## First Reads

1. Read `optimization-guide.md` before choosing tiles, launch geometry, or occupancy targets.
2. Read `isa-summary.md` for the portable in-repo ISA digest.
3. If you are porting across generations, read `../hardware-comparison.md`.
4. When profiling, pair this guide with `../../tool-rocprof/SKILL.md`.

## MI300X ISA Capability Summary

The MI300X `gfx942` ISA manuals describe a full CDNA compute ISA with these optimization-relevant capabilities:

- Scalar and vector instruction families: SALU, VALU, packed math (`VOP3P`), VGPR indexing, DPP / SDWA lane-control forms, and explicit rounding / denormal controls through `MODE`.
- Matrix-core ISA: dense `MFMA` plus sparse `2:4` matrix paths with separate accumulation registers (`AccVGPRs`) and `V_ACCVGPR_READ` / `V_ACCVGPR_WRITE` transfers.
- Matrix datatypes: `FP64`, `FP32`, `FP16`, `BF16`, `INT8`, `BF8`, `FP8`, plus reduced-precision `TF32` / `XF32` style matrix support on CDNA3.
- Low-precision conversion support: explicit `BF8` / `FP8` conversion and packed-convert instructions in the VALU path.
- Memory ISA: scalar memory ops, vector buffer ops, flat / global / scratch instructions, float atomics, LDS instructions, and `Memory Buffer Load to LDS`.
- Synchronization primitives: LDS atomics, `GWS`, explicit wait-state handling, and relaxed vector-cache behavior that still needs explicit synchronization for strong ordering.

What `gfx942` does **not** provide relative to CDNA4 is just as important:

- No native `FP6` / `FP4` / block-scaled `MXFP` matrix path.
- No CDNA4 `MFMA Transpose Load from LDS`.
- Smaller LDS budget (`64 KiB`, `32` banks), so staging-heavy kernels hit occupancy limits much sooner.

## Main Tuning Implications

- `gfx942` has `304` active CUs. Small grids often underfill the device; prefer enough blocks to cover the machine or use persistent scheduling.
- Wavefront size is `64`, so reductions, bank-conflict reasoning, and warp-synchronous CUDA ports must all be adjusted.
- LDS is only `64 KiB` per CU. Double-buffering and large tiles are useful, but they can quickly become occupancy-limited.
- CDNA3 FP8 work commonly assumes `FNUZ`; do not reuse CDNA4 `OCP` assumptions without conversion checks.
- For GEMM-like kernels, treat low MFMA utilization as a first-class bottleneck signal. For memory-bound kernels, validate coalescing and LDS pressure before chasing instruction-level tweaks.
- If your algorithm can use TF32-style reduced-precision accumulation, `gfx942` still has a hardware matrix path for it; do not import CDNA4's "no hardware TF32" assumption.
- Use standard `MFMA` + buffer-to-LDS staging patterns as the default mental model on MI300X; block-scaled and transpose-load optimizations belong to CDNA4, not CDNA3.

## Healthy Ranges

| Metric | Good signal | Needs attention |
|--------|-------------|-----------------|
| Occupancy | > 50% | < 25% |
| HBM BW utilization | > 60% on memory-bound kernels | < 30% |
| MFMA utilization | > 70% on compute-bound kernels | < 40% |
| LDS bank conflicts | < 5% | > 20% |
| Scratch / spills | 0 | Any spill |

## Starting Configs

- Triton: start with `BLOCK_SIZE` `128` or `256`, `num_warps` `4-8`.
- HIP: start with `256` threads per block, then re-check register and LDS pressure.
- CK GEMM: reasonable first pass is `block_m=256`, `block_n=128`, `block_k=64`, then re-profile.

## Common Mistakes

- Assuming MI300X has fewer CUs than it really does and launching too little work.
- Reusing NVIDIA warp-size assumptions.
- Treating CDNA3 FP8 as interchangeable with CDNA4 FP8.
- Filling LDS aggressively without checking the occupancy tradeoff.
- Chasing instruction-level tweaks before confirming whether the kernel is memory-, compute-, or stall-bound.

## Local Reference

- `isa-summary.md`

## Original Source PDFs

- `amd-instinct-mi300-cdna3-instruction-set-architecture.pdf`
- `amd-cdna-3-white-paper.pdf`
