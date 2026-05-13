---
name: gfx950
description: MI350X and MI355X hardware guidance for AMD kernel optimization. Use when the target GPU is gfx950, MI350X, or MI355X, or when tuning kernels against CDNA4 limits such as 256 CUs, 160 KiB LDS, 64-bank LDS, FP8 OCP, and MXFP support.
---

# gfx950 Hardware Guide

Use this guide when the target GPU is `gfx950` (`MI350X` or `MI355X`).

## Quick Facts

| Item | Value |
|------|-------|
| Architecture | CDNA4 |
| Active CUs | 256 |
| Wavefront size | 64 |
| LDS per CU | 160 KiB |
| LDS banks | 64 |
| FP8 ecosystem | OCP |
| New low precision paths | FP6, FP4, MXFP |
| Common build target | `--offload-arch=gfx950` |

## First Reads

1. Read `optimization-guide.md` before reusing any `gfx942` tile, swizzle, or occupancy assumptions.
2. Read `isa-summary.md` for the portable in-repo ISA digest.
3. If you are porting from MI300X or targeting both generations, read `../hardware-comparison.md`.
4. When profiling, pair this guide with `../../tool-rocprof/SKILL.md`.

## MI355X ISA Capability Summary

The MI355X `gfx950` ISA adds new matrix and low-precision capabilities on top of the baseline CDNA compute ISA:

- Baseline compute ISA remains broad: SALU, VALU, packed math, VGPR indexing, explicit rounding / denorm controls, scalar and vector memory ops, flat / global / scratch instructions, float atomics, and LDS instructions.
- Matrix-core ISA keeps dense and sparse `MFMA`, `AccVGPR`-backed accumulation, and `V_ACCVGPR_READ` / `V_ACCVGPR_WRITE`, but adds new low-precision matrix families for `F8/F6/F4`.
- New block-scaled matrix support: `V_MFMA_SCALE_*` and `CVT_SCALE_*` instructions support block exponent scaling with an `E8M0` bias shared across blocks of `32` K values.
- New datatypes: `FP6`, `BF6`, `FP4`, plus `FP8` / `BF8` paths that integrate with the CDNA4 matrix core.
- New LDS-assisted matrix path: `MFMA Transpose Load from LDS`, which can remove extra transpose / shuffle work in some kernels.
- Memory path enhancements matter more on CDNA4: `Memory Buffer Load to LDS`, larger LDS (`160 KiB`), `64` banks, and a stronger staging story overall.

Key negative difference versus MI300X:

- There is no hardware-native TF32 matrix path on CDNA4, so do not port TF32 assumptions from MI300X unchanged.

## Main Tuning Implications

- `gfx950` has fewer CUs than `gfx942` (`256` vs `304`), so work partitioning and persistent-kernel thresholds need to be re-tuned.
- LDS jumps to `160 KiB` per CU and `64` banks. This creates room for larger tiles and deeper staging, but bank-conflict formulas and occupancy math must be recalculated.
- CDNA4 FP8 uses `OCP`, not `FNUZ`. Treat FP8 format choice as a correctness gate, not a cosmetic detail.
- CDNA4 adds `FP6`, `FP4`, and `MXFP`; use them only when toolchain support, validation, and the workload all line up.
- Larger LDS and new matrix paths can shift the bottleneck from memory to register pressure or pipeline scheduling. Re-profile after each accepted change.
- If the kernel is low-precision and K-dominant, consider whether block-scaled `F8/F6/F4` MFMA is a better fit than a plain FP8 path.
- CDNA4-specific transpose-load and larger LDS can justify memory-layout changes that would be too expensive or impossible on MI300X.

## Healthy Ranges

| Metric | Good signal | Needs attention |
|--------|-------------|-----------------|
| Occupancy | > 50% | < 25% |
| HBM BW utilization | > 60% on memory-bound kernels | < 30% |
| MFMA / MXFP utilization | > 70% on compute-bound kernels | < 40% |
| LDS bank conflicts | < 5% | > 20% |
| Scratch / spills | 0 | Any spill |

## Starting Configs

- Triton: start with the same `num_warps` range (`4-8`), but consider larger tiles only after checking register and LDS pressure.
- HIP: start around `256` threads per block, then re-measure occupancy after enabling larger shared structures.
- Low-precision paths: explicitly log whether the round is `FP8 OCP`, `FP6`, `FP4`, or `MXFP`.

## Common Mistakes

- Reusing CDNA3 LDS-bank assumptions on CDNA4.
- Forgetting that CDNA4 FP8 is `OCP`.
- Treating `160 KiB` LDS as “free” without checking occupancy.
- Porting old TF32 assumptions even though TF32 is not a hardware-native CDNA4 path.
- Comparing utilization against the wrong architecture peak tables.

## Local Reference

- `isa-summary.md`

## Original Source PDFs

- `amd-instinct-cdna4-instruction-set-architecture.pdf`
- `amd-cdna-4-architecture-whitepaper.pdf`
- `amd-instinct-mi355x-gpu-brochure.pdf`
