# MI350X / MI355X (`gfx950`, CDNA4) Optimization Guide

`MI350X` and `MI355X` both target `gfx950`. They share the same ISA, CU layout, cache hierarchy, and low-precision feature set. SKU differences are mainly clock, power, cooling, and rated peak throughput.

## Core Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | CDNA4 |
| LLVM / offload-arch | `gfx950` |
| Compute Units | `256` active (`8 XCD x 32`) |
| Wavefront size | `64` |
| LDS / CU | `160 KiB` |
| LDS banks | `64` |
| HBM | `288 GiB HBM3E` |
| Peak HBM BW | `8 TB/s` |
| L2 cache | `32 MiB` total (`4 MiB / XCD`) |
| Infinity Cache | `256 MiB` |
| FP8 format | OCP |
| New low precision paths | FP6, FP4, MXFP |

### SKU Notes

| Item | MI350X | MI355X |
|------|--------|--------|
| Clock | `2200 MHz` | `2400 MHz` |
| TBP | `1000 W` | `1400 W` |
| Cooling | Air | Liquid |
| FP16 matrix peak | `2.3 PFLOPS` | `2.5 PFLOPS` |
| FP8 matrix peak | `4.6 PFLOPS` | `5.0 PFLOPS` |

## Per-Clock, Per-CU Throughput

| Computation | MI300X | MI355X |
|-------------|--------|--------|
| Vector FP64 / FP32 / FP16 | `128 / 256 / 256` | `128 / 256 / 256` |
| Matrix FP64 | `256` | `128` |
| Matrix FP32 | `256` | `256` |
| Matrix FP16 / sparsity | `2048` | `4096` |
| Matrix BF16 / sparsity | `2048` | `4096` |
| Matrix FP8 / sparsity | `4096` | `8192` |
| Matrix INT8 / sparsity | `4096` | `8192` |
| Matrix MXFP6 / MXFP4 | N/A | `16384` |

## ISA Capability Summary

MI355X's CDNA4 ISA expands the practical kernel design space in ways that should change optimization strategy:

| Area | MI355X ISA capability |
|------|------------------------|
| Core ISA families | SALU, VALU, packed math, VGPR indexing, scalar + vector memory ops, flat / global / scratch, LDS instructions, float atomics |
| Matrix core | Dense and `2:4` sparse `MFMA` with `AccVGPR` accumulation and `V_ACCVGPR_*` register moves |
| Matrix datatypes | `FP64`, `FP32`, `FP16`, `BF16`, `INT8`, `BF8`, `FP8`, plus new `F8/F6/F4` matrix families |
| Block-scaled low precision | `V_MFMA_SCALE_*` plus `CVT_SCALE_*` conversion instructions with shared `E8M0` exponent bias across blocks of `32` K values |
| Memory staging | `Memory Buffer Load to LDS` and `MFMA Transpose Load from LDS` |
| LDS model | `160 KiB` LDS, `64` banks, better support for larger staged tiles |
| Missing vs CDNA3 | No hardware-native `TF32` matrix path |

## Optimization Implications from the ISA

- If a low-precision kernel is still using a plain FP8 path, check whether CDNA4 block-scaled `F8/F6/F4` MFMA is a better fit.
- `MFMA Transpose Load from LDS` can remove extra permute / transpose overhead in matrix-heavy kernels; it is worth considering explicitly on MI355X.
- `CVT_SCALE_*` instructions mean conversion and scaling are part of the performance design space, not just correctness scaffolding.
- CDNA4's larger LDS makes deeper staging and more aggressive swizzles more realistic, but the `64`-bank layout means every MI300X padding rule must be revalidated.
- Because hardware-native TF32 is gone, FP32-adjacent training or accumulation paths need to be reasoned about via BF16 / FP16 / explicit mixed-precision choices instead.

## Partition and NPS

| Mode | Configuration | VRAM per partition |
|------|---------------|--------------------|
| `SPX` | Full card | ~`288 GB` |
| `DPX` | `4` XCD / partition | ~`144 GB` |
| `QPX` | `2` XCD / partition | ~`72 GB` |
| `CPX` | `1` XCD / partition | ~`36 GB` |

CDNA4 commonly uses `NPS1` or `NPS2`. AMD documentation often highlights `DPX + NPS2` as a good locality / efficiency point, but you still need to benchmark the exact deployment mode you care about.

## Compilation

```bash
hipcc -O3 --offload-arch=gfx950 kernel.cpp

# Multi-target
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 kernel.cpp
```

## Good Starting Points

| Path | Parameter | Starting point |
|------|-----------|----------------|
| Triton | Tile size | Start from `gfx942` defaults, then test larger tiles if registers and LDS allow |
| Triton | `num_warps` | `4-8` |
| HIP | Thread block | `256` threads |
| MFMA / MXFP | Datatype | Use only with explicit toolchain and validation support |

## Healthy Ranges

| Metric | Good | Needs attention |
|--------|------|-----------------|
| Occupancy | `> 50%` | `< 25%` |
| HBM BW utilization | `> 60%` on memory-bound kernels | `< 30%` |
| MFMA / MXFP utilization | `> 70%` on compute-bound kernels | `< 40%` |
| LDS bank conflicts | `< 5%` | `> 20%` |
| Register spill | `0` scratch waves | Any scratch |

## Hardware-Specific Implications

- `gfx950` has `256` CUs instead of `304`, so grid mapping and persistent scheduling thresholds must be re-tuned instead of copied from MI300X.
- LDS expands to `160 KiB` and `64` banks. This is a major opportunity, but all swizzle, padding, and occupancy assumptions from CDNA3 need to be revalidated.
- FP8 uses `OCP`; never reuse CDNA3 `FNUZ` conversions or validation baselines without checking type semantics.
- CDNA4 introduces `FP6`, `FP4`, and `MXFP`. These can be very attractive for custom kernels, but only when the compiler, libraries, and correctness harness all support them.
- CDNA4 does not have a hardware-native TF32 path. If you port CUDA or older ROCm tuning logic, keep that in mind when interpreting matrix-path utilization.
- When a kernel is bottlenecked by data movement into the matrix core, prefer evaluating transpose-load and scale-aware matrix paths before over-investing in manual shuffle logic.

## Migration Checklist from `gfx942`

1. Change the build target to `--offload-arch=gfx950`.
2. Re-tune block sizes for `256` CUs and `160 KiB` LDS.
3. Recompute bank-conflict-safe layouts for `64` LDS banks.
4. Revalidate FP8 as `OCP`, not `FNUZ`.
5. Consider whether `FP6`, `FP4`, or `MXFP` can replace an older `FP8` or `BF16` path.

## Common Mistakes

- Building without `-O3`.
- Reusing CDNA3 tile shapes without checking register and LDS pressure.
- Assuming FP8 bit patterns are compatible across generations.
- Ignoring the new `64`-bank LDS organization.
- Comparing utilization to the wrong architecture peak.

## Cross References

- Portable ISA digest: `isa-summary.md`
- Cross-generation comparison: `../hardware-comparison.md`
- CDNA3 counterpart: `../gfx942/optimization-guide.md`

## Original Source PDFs

- `amd-instinct-cdna4-instruction-set-architecture.pdf`
- `amd-cdna-4-architecture-whitepaper.pdf`
- `amd-instinct-mi355x-gpu-brochure.pdf`
