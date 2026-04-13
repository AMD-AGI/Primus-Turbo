# AMD CDNA3 vs CDNA4 Hardware Comparison

This file is the cross-generation reference for `gfx942` (`MI300X` / `MI325X`) and `gfx950` (`MI350X` / `MI355X`).

For self-contained local ISA references, start with `gfx942/isa-summary.md` and `gfx950/isa-summary.md`.

## Quick Reference

| Parameter | MI300X / MI325X (`gfx942`) | MI350X / MI355X (`gfx950`) |
|-----------|----------------------------|----------------------------|
| Architecture | CDNA3 | CDNA4 |
| Active CUs | `304` | `256` |
| Wavefront size | `64` | `64` |
| LDS / CU | `64 KiB` | `160 KiB` |
| LDS banks | `32` | `64` |
| L2 cache | `32 MiB` | `32 MiB` |
| Infinity Cache | `256 MiB` | `256 MiB` |
| Peak HBM BW | `5.3-6.0 TB/s` class | `8 TB/s` |
| FP8 ecosystem | FNUZ | OCP |
| New low precision types | No MXFP path | FP6, FP4, MXFP |
| offload-arch | `gfx942` | `gfx950` |

## What Changes Matter Most

### 1. CU count and grid shape

- `gfx942` has more CUs (`304` vs `256`), so launch geometry and persistent scheduling thresholds must be re-tuned.
- A launch plan that fills MI300X well can still underfill or over-fragment MI355X, and vice versa.

### 2. LDS capacity and bank model

| Item | `gfx942` | `gfx950` |
|------|----------|----------|
| LDS size | `64 KiB` | `160 KiB` |
| LDS banks | `32` | `64` |
| Read bandwidth | Lower | Higher |
| Direct L1 -> LDS path | No | Yes |

Implication: CDNA4 makes larger staging buffers and deeper pipelines more realistic, but every swizzle, padding rule, and occupancy assumption has to be recalculated.

### 3. Precision and MFMA assumptions

| Topic | `gfx942` | `gfx950` |
|-------|----------|----------|
| FP8 | FNUZ | OCP |
| New paths | None | FP6, FP4, MXFP |
| Matrix FP64 per CU | `256` | `128` |
| Matrix FP16/BF16/FP8 per CU | Lower | Higher |

Implication: porting low-precision kernels across generations is not a search-and-replace job. Treat datatype choice and validation as correctness-critical.

### 4. ISA capability delta

| Capability | `gfx942` / MI300X | `gfx950` / MI355X |
|------------|-------------------|-------------------|
| Baseline compute ISA | SALU, VALU, packed math, VGPR indexing, scalar/vector memory, LDS, flat/global/scratch | Same baseline families |
| Matrix engine | Dense + sparse `MFMA` with `AccVGPR` accumulation | Same, but with larger low-precision matrix family |
| Matrix datatypes | `FP64`, `FP32`, `FP16`, `BF16`, `INT8`, `BF8`, `FP8`, `TF32` / `XF32` style reduced-precision matrix support | `FP64`, `FP32`, `FP16`, `BF16`, `INT8`, `BF8`, `FP8`, plus `FP6`, `BF6`, `FP4`, block-scaled `F8/F6/F4` |
| Scale-aware low precision | No block-scaled matrix ISA | `V_MFMA_SCALE_*` and `CVT_SCALE_*` |
| Buffer -> LDS | Yes | Yes |
| LDS transpose assist | No `MFMA Transpose Load from LDS` | Yes |
| TF32 hardware path | Present on CDNA3 | Not hardware-native on CDNA4 |

Implication: MI355X is not just "more LDS and different FP8"; it introduces new scale-aware and transpose-aware matrix paths that can justify different kernel structures.

### 5. Partitioning

| Mode | CDNA3 | CDNA4 |
|------|-------|-------|
| `SPX` | Yes | Yes |
| `DPX` | Yes | Yes |
| `CPX` | Yes | Yes |
| `QPX` | No | Yes |

Implication: if the target deployment uses partitioned GPUs, benchmark on the exact partition / NPS mode you will ship.

## Porting Checklist

1. Change the compilation target: `gfx942` <-> `gfx950`.
2. Re-tune tile sizes for the destination CU count and LDS size.
3. Recalculate LDS padding / swizzle formulas for `32` vs `64` banks.
4. Revalidate low-precision numerics, especially FP8 (`FNUZ` vs `OCP`).
5. Recompute roofline and utilization denominators against the correct hardware peak.
6. Re-profile after the port before trusting any old bottleneck classification.

## Practical Guidance

- Moving from `gfx942` to `gfx950`: try larger staged tiles, but watch register pressure and occupancy.
- Moving from `gfx950` to `gfx942`: budget LDS more aggressively and avoid importing `64`-bank assumptions.
- On either generation, keep wavefront size `64` in mind when porting from CUDA or tuning subgroup logic.
- If you are unsure whether a regression is due to architecture or code shape, re-run the same focused benchmark under both hardware guides and compare the roofline context first.

## Related Files

- `gfx942/isa-summary.md`
- `gfx950/isa-summary.md`
- `gfx942/SKILL.md`
- `gfx942/optimization-guide.md`
- `gfx950/SKILL.md`
- `gfx950/optimization-guide.md`

## Original Source PDFs

- `amd-instinct-mi300-cdna3-instruction-set-architecture.pdf`
- `amd-instinct-cdna4-instruction-set-architecture.pdf`
- `amd-cdna-3-white-paper.pdf`
- `amd-cdna-4-architecture-whitepaper.pdf`
- `amd-instinct-mi355x-gpu-brochure.pdf`
