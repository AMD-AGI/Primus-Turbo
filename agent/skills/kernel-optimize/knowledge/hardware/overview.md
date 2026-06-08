# AMD GPU Hardware — Orientation Outline (CDNA3 gfx942 / CDNA4 gfx950)

Primus-Turbo targets two AMD Instinct generations. This file is the **outline and router**: it states what each generation is, the handful of cross-generation differences that change kernel structure or correctness, and where to read the numbers. **Detailed specs, peak-throughput tables, the occupancy model, default configs, and per-subsystem ISA notes live in the per-generation files** — read those rather than re-deriving anything here:

- gfx942 (CDNA3): [`gfx942/overview.md`](gfx942/overview.md) → [`gfx942/kernel-implementation-notes.md`](gfx942/kernel-implementation-notes.md)
- gfx950 (CDNA4): [`gfx950/overview.md`](gfx950/overview.md) → [`gfx950/kernel-implementation-notes.md`](gfx950/kernel-implementation-notes.md)

## The Two Generations

- **gfx942 / CDNA3** — MI300X (192 GB HBM3, 5.3 TB/s), MI325X (256 GB HBM3E, 6.0 TB/s). 304 active CUs, 64 KiB LDS/CU, FP8 in the **FNUZ** encoding, a strong FP64 matrix path. Build target `--offload-arch=gfx942`.
- **gfx950 / CDNA4** — MI350X, MI355X (288 GB HBM3E, 8 TB/s). 256 active CUs, 160 KiB LDS/CU, FP8 in the **OCP** encoding, block-scaled MX (FP6/FP4/MXFP) and LDS transpose-load as new hardware paths, roughly double the per-CU low-precision matrix throughput, and a halved per-CU FP64 rate. Build target `--offload-arch=gfx950`.

gfx950 is a different optimization point, not "gfx942 with more LDS and a different FP8". Several of the differences below are correctness or occupancy gates, not recompiles.

## Headline Cross-Generation Differences (outline)

| Dimension | gfx942 (CDNA3) | gfx950 (CDNA4) | Why it matters for a kernel |
|---|---|---|---|
| Active CUs | 304 | 256 | re-tune launch grid / persistent work-partition |
| LDS per CU | 64 KiB, 32 banks, 512 B block | 160 KiB, 64 banks, 1280 B block | re-derive bank swizzle/padding; re-budget tile footprint |
| Low-precision matrix rate | baseline | ≈2× per CU per cycle; wider-K MFMA | re-select the MFMA opcode |
| Sub-8-bit / MX path | none — dequantize, then MFMA | block-scaled MX, `F8F6F4` mixed MFMA, FP6/FP4 | new hardware path with no CDNA3 fallback |
| LDS transpose-load | none | `DS_READ_*_TR` | removes the manual VGPR transpose pass |
| FP8 encoding | FNUZ (E4M3 max 240, no Inf) | OCP (E4M3 max 448; BF8 with Inf) | **correctness gate** — re-encode scales |
| FP64 matrix | strong (≈256 FLOP/cycle/CU) | halved (≈128) | gfx942 is the faster double-precision part |

For the full spec table and the peak-throughput numbers (TFLOPS/PFLOPS per dtype, per-CU-per-cycle), read each generation's `overview.md` — they are the source of truth for the detailed metrics.

## Correctness / Occupancy Gates When Porting

These do not survive a recompile; they must be re-derived for the destination generation:

1. **FP8 FNUZ ↔ OCP** — the encodings differ in bias, max magnitude, and Inf handling. Re-encode per-tensor/per-block scales and re-check saturation, or the numerics are silently wrong.
2. **LDS bank count and allocation block** — a 32-bank XOR swizzle leaves conflicts on 64 banks; a footprint that rounded cleanly on a 512 B block can waste a whole block on a 1280 B block (or the reverse) and cost an occupancy tier.
3. **CDNA4-only paths** — block-scaled MX MFMA, `F8F6F4`, `DS_READ_*_TR`, `V_PRNG_B32`, `V_BITOP3`, scalar atomics, `v_permlane16_swap` have no gfx942 fallback. Porting down means re-architecting (dequantize-then-MFMA, explicit VGPR transpose, multi-instruction bitops), not recompiling.
4. **MFMA opcode width** — the wider-K CDNA4 forms change the register layout, so LDS tiles and the epilogue swizzle move with the opcode.

What stays identical across generations (do not over-rotate when porting): wavefront size 64, the VGPR/SGPR register model, 8 XCD chiplets with a partitioned 4 MB-per-XCD L2 plus a 256 MB last-level cache, the AccVGPR/MFMA accumulator model, and the core cross-lane primitives.

## Porting, In Brief

**gfx942 → gfx950**: change the build target → re-tune the grid for 256 CUs → recompute LDS swizzle/padding for 64 banks and the 1280 B block, and consider larger tiles / deeper pipelines now that 160 KiB is available → **revalidate FP8 (FNUZ → OCP)** → adopt wider-K MFMA and `DS_READ_*_TR`, and evaluate block-scaled MX / `F8F6F4` for sub-8-bit dtypes → recompute the roofline against gfx950 peak tables → re-profile (the bottleneck commonly moves from memory toward register/scheduling).

**gfx950 → gfx942**: change the build target → **remove every CDNA4-only path** (there is no hardware fallback; re-architect around the gaps) → re-budget LDS for 64 KiB / 32 banks / 512 B → **revalidate FP8 (OCP → FNUZ)** → re-select the narrower CDNA3 MFMA K widths → re-tune the grid for 304 CUs → recompute the roofline and re-profile.

## Which Generation for Which Workload

| Workload characteristic | Stronger generation | Why |
|---|---|---|
| FP64 / HPC matmul | gfx942 | ≈2× the per-CU FP64 matrix rate |
| FP16 / BF16 / FP8 AI matmul | gfx950 | ≈2× the per-CU low-precision rate + higher HBM bandwidth |
| MXFP8 / MXFP6 / MXFP4, sub-8-bit | gfx950 | the only generation with block-scaled MX and FP6/FP4 hardware |
| Memory-capacity-bound (large weights / KV cache) | gfx950 (288 GB) or MI325X (256 GB) | capacity plus 8 TB/s on gfx950 |
| LDS-heavy staging / deep software pipelines | gfx950 | 160 KiB vs 64 KiB LDS |

## Diagnosing a Cross-Generation Regression

When a kernel that ran well on one generation regresses on the other, separate the architectural cause from the code-shape cause before editing: benchmark both generations against **each one's own roofline** (correct peak and bandwidth denominators), check the gates above first (FP8 format, LDS bank count / footprint, MFMA opcode width, removed or added hardware paths), then re-profile to re-classify the bottleneck. The larger LDS and faster matrix core on gfx950 frequently shift a memory-bound kernel into register- or scheduling-bound territory that the old CDNA3 tuning will not address.
