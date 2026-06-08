# gfx942 (CDNA3) — Hardware Orientation for Kernel Authors

Target this guide when `target_gpu = gfx942`, i.e. **AMD Instinct MI300X / MI325X** (and the MI300A APU variant). Architecture codename is **CDNA3**, build target `--offload-arch=gfx942`.

This file gives the orientation an agent needs before writing or tuning a kernel: the hardware budget, the occupancy model, and how CDNA3 differs from CDNA4. Detailed per-subsystem implementation guidance lives in [`kernel-implementation-notes.md`](kernel-implementation-notes.md).

## Quick Facts

| Item | Value |
|------|-------|
| Architecture | CDNA3 (3rd-gen Matrix Core) |
| Products | MI300X (192 GB HBM3, 5.3 TB/s), MI325X (256 GB HBM3E, 6.0 TB/s) |
| Active CUs | 304 (8 XCD chiplets × 38 CU) |
| Matrix cores | 1216 (4 per CU) |
| Wavefront size | 64 lanes |
| Peak clock | ~2.1 GHz |
| LDS per CU | 64 KiB = 32 banks × 512 Dwords (32-bit/bank) |
| LDS allocation quantum | 512 B |
| L1 vector cache | 32 KB per CU |
| L2 cache | 4 MB **per XCD** (≈32 MB aggregate), partitioned — not one shared L2 |
| Last-level (Infinity) cache | 256 MB, shared across the 8 XCDs |
| VGPR file | 512 Dwords/lane pool, split as ≤256 arch-VGPR + ≤256 AccVGPR; 8-Dword allocation quantum |
| SGPR file | 16…102 per wave, 16-Dword allocation quantum (VCC aliases the top pair) |
| FP8 ecosystem | **FNUZ** (E4M3 bias 8, max 240, no Inf; BF8 E5M2 bias 16) |
| Low-precision matrix dtypes | FP64, FP32, TF32/XF32, FP16, BF16, INT8, FP8, BF8 |
| New CDNA4 paths NOT present here | block-scaled MX (FP6/FP4/MXFP), `DS_READ_*_TR` transpose-load, `F8F6F4` mixed MFMA |

### Peak matrix throughput (per GPU, dense / with 2:4 structured sparsity)

| dtype | TFLOPS | per-CU-per-cycle (derived) |
|-------|--------|----------------------------|
| FP16 / BF16 | 1307.4 / 2614.9 | ≈2048 FLOP |
| FP8 / INT8 | 2614.9 / 5229.8 | ≈4096 FLOP |
| TF32 (XF32) | 653.7 / 1307.4 | ≈1024 FLOP |
| FP64 matrix | 163.4 (no sparsity path) | ≈256 FLOP |

> Note the FP64 row: CDNA3 has a **strong FP64 matrix path (~256 FLOP/cycle/CU)**. CDNA4 trades this away (halved per CU). For double-precision HPC matmul, gfx942 is the stronger generation — do not assume newer is faster here.

## Resource Allocation & Occupancy Model

Occupancy (waves/CU) is `min` over four pools of `floor(pool / per-wave allocation)`, where each axis is rounded **up** to its quantum before the division:

- **VGPR**: round up to 8 Dwords. Arch-VGPR and AccVGPR draw from one 512-Dword pool, each capped at 256; the split is flexible only while the total stays < 512. MFMA accumulators consume the AccVGPR side, so freeing arch-VGPR can be re-spent on bigger accumulators within the same occupancy tier.
- **SGPR**: round up to 16 Dwords (legal 16…102). Budget `used + 2 (VCC) + 16 (if a trap handler is enabled)`, and remember the launch payload (user SGPRs + `tgid_{x,y,z}` + `tg_size`) is charged before the kernel runs.
- **LDS**: round up to **512 B** blocks (per work-group). A 513-byte request costs 1024 B and can demote a tier.

Practical rule: trimming a register/byte only helps if it crosses a quantum boundary. Trimming one VGPR that drops you from 85→84 (a multiple-of-8 boundary) can buy a whole extra wave; trimming one that does not buys nothing. 64-bit operands force even-pair alignment and can silently push you past a quantum.

Out-of-range GPR/LDS access on CDNA3 **does not fault** — an over-indexed source reads register 0, an over-indexed destination drops the write, and LDS over-range reads return 0. A budgeting bug therefore looks like a numerical bug; cross-check indices against the rounded budget.

## Default Starting Configs

- **Triton**: `BLOCK` 128 or 256, `num_warps` 4–8, then re-check register/LDS pressure.
- **HIP / CK**: 256 threads/block first; a reasonable GEMM first pass is `block_m=256, block_n=128, block_k=64`, then re-profile.
- Prefer enough work-groups to cover all 304 CUs (or persistent scheduling) — small grids underfill the device.

## Relationship to gfx950 (CDNA4)

gfx950 is a different optimization point, not just a faster gfx942: it roughly doubles per-CU low-precision matrix throughput and adds block-scaled MX (FP6/FP4/MXFP), `F8F6F4` mixed MFMA, and `DS_READ_*_TR` LDS transpose-load — while switching FP8 to the OCP encoding and **halving** the per-CU FP64 matrix rate. Several of these (FP8 FNUZ→OCP, LDS 32→64 banks, 512 B→1280 B allocation block) are correctness/occupancy gates rather than recompiles.

The cross-generation outline — headline differences, the correctness/occupancy gates, the bidirectional porting checklists, the "which generation for which workload" decision guide, and the regression-diagnosis steps — lives in **[`../overview.md`](../overview.md)**; consult it before porting a kernel in either direction. The peak-throughput numbers themselves live in each generation's `overview.md`.
