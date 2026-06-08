# gfx950 (CDNA4) — Hardware Orientation for Kernel Authors

Target this guide when `target_gpu = gfx950`, i.e. **AMD Instinct MI350X / MI355X**. Architecture codename is **CDNA4**, build target `--offload-arch=gfx950`.

This file gives the orientation an agent needs before writing or tuning a kernel: the hardware budget, the occupancy model, and the new CDNA4 capabilities that justify different kernel structures than CDNA3. Detailed per-subsystem guidance lives in [`kernel-implementation-notes.md`](kernel-implementation-notes.md).

## Quick Facts

| Item | Value |
|------|-------|
| Architecture | CDNA4 (4th-gen Matrix Core) |
| Products | MI350X (air, 1000 W, ~2.2 GHz), MI355X (liquid, 1400 W, ~2.4 GHz) |
| Memory | 288 GB HBM3E, 8 TB/s (both) |
| Active CUs | 256 (8 XCD chiplets × 32 CU) |
| Stream processors | 16384 |
| Wavefront size | 64 lanes |
| LDS per CU | 160 KiB = 64 banks × 640 Dwords (32-bit/bank) |
| LDS allocation quantum | 1280 B (per work-group, no wrap) |
| L1 vector cache | 32 KB per CU |
| L2 cache | 4 MB **per XCD**, partitioned — not one shared L2 |
| Last-level (Infinity) cache | 256 MB, shared across the 8 XCDs |
| VGPR file | 512 Dwords/lane pool, split as ≤256 arch-VGPR + ≤256 AccVGPR; 8-Dword allocation quantum |
| SGPR file | 16…102 per wave, 16-Dword allocation quantum (VCC aliases the top pair) |
| FP8 ecosystem | **OCP** (E4M3 bias 7, max 448; BF8 E5M2 bias 15, max 57344, with Inf) |
| New low-precision dtypes | FP6 (E2M3), BF6 (E3M2), FP4 (E2M1) |
| Block-scaled (MX) matrix | MXFP8 / MXFP6 / MXFP4 via `V_MFMA_SCALE_F32_*_F8F6F4` (E8M0 per-32-K-block) |
| New matrix path | `F8F6F4` mixed-precision MFMA; `DS_READ_*_TR` transpose-load from LDS |

### Peak matrix throughput (per GPU; MI350X / MI355X; dense, ×2 with 2:4 sparsity)

| dtype | MI350X | MI355X | per-CU-per-cycle (derived, MI355X) |
|-------|--------|--------|-------------------------------------|
| MXFP4 / FP4 | 9.2 PFLOPS | 10.1 PFLOPS | ≈16384 FLOP |
| MXFP6 / FP6 | 9.2 PFLOPS | 10.1 PFLOPS | ≈16384 FLOP |
| FP8 / MXFP8 (OCP) | 4.6 PFLOPS | 5.0 PFLOPS | ≈8192 FLOP |
| FP16 / BF16 matrix | 2.3 PFLOPS | 2.5 PFLOPS | ≈4096 FLOP |
| FP64 matrix | 72.1 TFLOPS | 78.6 TFLOPS | ≈128 FLOP |

> Two structural shifts vs CDNA3: low-precision matrix throughput **roughly doubled per CU per cycle** (FP16 2048→4096, FP8 4096→8192 FLOP/cycle/CU), while **FP64 matrix was halved per CU** (256→128). For FP64 HPC matmul, the older gfx942 is actually the stronger generation; for FP16/FP8/FP6/FP4 AI workloads, gfx950 is decisively faster.

## Resource Allocation & Occupancy Model

Occupancy (waves/CU) is `min` over four pools of `floor(pool / per-wave allocation)`, each axis rounded **up** to its quantum first:

- **VGPR**: round up to 8 Dwords. Arch-VGPR and AccVGPR are two pools, each capped at 256, summing to ≤512 total; the split is flexible only below 512. MFMA accumulators consume the AccVGPR side — budget the two pools separately (treating them as one 512 pool over-states occupancy when AccVGPR usage is heavy).
- **SGPR**: round up to 16 Dwords (legal 16…102). Budget `used + 2 (VCC) + 16 (if a trap handler is enabled)`, plus the launch payload (user SGPRs + `tgid_{x,y,z}` + `tg_size`).
- **LDS**: round up to **1280 B** blocks, on 1280 B alignment, no wrap. A 1281-byte request costs 2560 B and one extra block can kill occupancy. (This block size differs from CDNA3's 512 B — re-budget when porting.)

Trimming a resource only helps if it crosses a quantum boundary (8 VGPR, 16 SGPR, 1280 B LDS). 64-bit operands force even-pair alignment and a 4-Dword SMEM load forces quad-aligned destination SGPRs — either can silently bump you into the next quantum.

Out-of-range GPR/LDS access **does not fault** — over-indexed source reads register 0, over-indexed destination drops the write (multi-dest VMEM/atomic issues with EXEC=0), LDS over-range reads return 0 and writes drop. A sizing bug looks like a numerical bug; cross-check indices against the rounded budget.

## Default Starting Configs

- **Triton**: `num_warps` 4–8 as on CDNA3, but the 160 KiB LDS allows larger tiles / deeper staging — only after re-checking register and LDS pressure.
- **HIP / CK**: 256 threads/block first; the larger LDS makes deeper pipelines and bigger staged tiles feasible that would not fit on gfx942.
- For any low-precision GEMM, explicitly record whether the round is FP8 (OCP), MXFP8, MXFP6, MXFP4, or FP6/FP4 — the matrix path and packing differ.
- Aim to cover all 256 CUs (fewer than gfx942), so persistent-kernel and work-partition thresholds need re-tuning vs CDNA3.

## Relationship to gfx942 (CDNA3)

gfx942 is not simply "older and slower": it keeps a stronger FP64 matrix path (~2× the per-CU rate) and has more CUs (304 vs 256), while lacking every CDNA4-only path (block-scaled MX, `F8F6F4` mixed MFMA, `DS_READ_*_TR` transpose-load, FP6/FP4, and the OCP FP8 encoding). Porting down means re-architecting around those gaps, not just recompiling — and re-budgeting for the smaller LDS (64 KiB, 32 banks, 512 B allocation block).

The cross-generation outline — headline differences, the correctness/occupancy gates, the bidirectional porting checklists, the "which generation for which workload" decision guide, and the regression-diagnosis steps — lives in **[`../overview.md`](../overview.md)**; consult it before porting a kernel in either direction. The peak-throughput numbers themselves live in each generation's `overview.md`.
