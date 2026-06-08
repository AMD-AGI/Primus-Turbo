# GEMM — Optimization Orientation

Read this first when `target_op` is a GEMM-class kernel: dense matmul (BF16/FP16/FP8/BF8/INT8) on gfx942/gfx950, low-bit weight GEMM (FP4/MX, W4A16/W4A8), and the grouped / MoE GEMM family. Training runs both a forward GEMM and one or two backward GEMMs, so treat the **combined fwd+bwd step** as the thing being optimized, not the forward kernel in isolation.

This file gives the anatomy, the bottleneck model, and the lever hierarchy. The detailed catalog of techniques lives in [`optimization-directions.md`](optimization-directions.md). Hardware budgets, the MFMA/LDS/occupancy rules, and the cross-generation differences live in [`../../hardware/gfx942/`](../../hardware/gfx942/overview.md), [`../../hardware/gfx950/`](../../hardware/gfx950/overview.md), and [`../../hardware/overview.md`](../../hardware/overview.md) — consult those for any instruction-level question rather than re-deriving it here.

## Kernel Anatomy

A high-throughput MFMA GEMM on CDNA is built from four nested layers plus a pipelined K-loop:

| Layer | What it maps | Typical choice |
|-------|--------------|----------------|
| Grid | One work-group owns one `(tile_m, tile_n)` C-tile | `block_x → M`, `block_y → N` (after optional swizzle) |
| Wave | Each wave owns an N-slab of the C-tile | 4 waves / 256 threads is the common starting point |
| Lane | Each lane owns rows/cols inside one MFMA atom | `lane_div_16 → M`, `lane_mod_16 → N` for a 16×16 atom |
| MFMA | One opcode does an `M×N×K` outer product `D = C + A·B` | pick the largest-K atom that divides the K-loop |

The K-loop is a software pipeline with three phases:

- **Prologue** — prime the first K-tile(s) into LDS / registers.
- **Steady state** — while MFMAs consume the current K-tile, prefetch the next K-tile (global→LDS, often async), so compute and memory overlap.
- **Epilogue** — drain the last in-flight tiles, then write `C` (optionally through an LDS re-tile for coalesced stores).

Per-tile MFMA count is fully determined by the tile/atom choice (`MFMA_per_tile = k_unroll · m_repeat · num_acc_n · …`); deciding the MFMA atom **fixes** the A/B/C/D register layout, so it must be chosen before LDS tiling, swizzle, and the epilogue are designed.

## Bottleneck Classification

Always classify before optimizing — the profiler tells you which third of the catalog applies. Compute the arithmetic intensity and compare against the roofline for the chosen dtype.

| Class | Signature | What helps (see directions catalog) |
|-------|-----------|-------------------------------------|
| Memory-bound | HBM BW near peak, MFMA-issue ratio low, large K small M·N | bigger tiles, async G2S prefetch, L2/XCD locality, split-K, preshuffle to cut redundant loads |
| Compute-bound | MFMA-issue ratio high, HBM BW slack | wider-K MFMA atom, fewer dynamic MFMAs, keep accumulators in AccVGPR, raise occupancy |
| Stall-bound | both BW and MFMA ratio low, high `s_barrier` / `s_waitcnt` time | LDS ping-pong overlap, hot-loop scheduling hints, bank-conflict-free swizzle, fix prefetch depth |

A high MFMA-issue ratio with low TFLOPS is the classic trap: it usually means barrier or `s_waitcnt` stalls, **not** that the scheduler is fine. Cross-check the trace categories before turning a knob.

## Optimization Lever Hierarchy

Work top-down; the early levers move TFLOPS the most and constrain everything below them.

1. **Tile shape + MFMA atom** — the single largest lever. Sets dynamic MFMA count, register layout, LDS footprint, and occupancy ceiling at once.
2. **LDS pipelining** — 2-stage (or deeper) ping-pong so global→LDS latency hides under MFMA; this is what turns a stall-bound kernel compute-bound.
3. **Operand layout / preshuffle** — preshuffle B offline so loads map directly onto MFMA register layout; eliminates an in-loop shuffle pass.
4. **Bank-conflict-free LDS** — XOR swizzle (sized to the bank count of the target arch) on both write and read paths; near-zero cost, removes a hidden serializer.
5. **Hot-loop instruction scheduling** — `sched_*` hints / token-budget scheduler to interleave MFMA, LDS-read, VMEM, LDS-write so no pipe starves.
6. **Epilogue** — CShuffle LDS re-tile for coalesced vectorized stores when `tile_n` is large; fuse dequant/bias/activation.
7. **Occupancy / register budget** — accumulators in AGPR; keep arch-VGPR under the occupancy boundary; async-copy to reclaim VGPRs.
8. **Locality / scheduling** — XCD-aware work-group remap and persistent scheduling for spatially-local tiles.

## Healthy Metric Ranges (starting heuristics)

| Metric | Good | Needs attention |
|--------|------|-----------------|
| MFMA-issue ratio in hot loop (compute-bound) | > 40% | < 25% |
| HBM BW utilization (memory-bound) | > 60% | < 30% |
| Occupancy (waves/SIMD) | ≥ 2 | 1 |
| LDS bank conflicts | < 5% | > 20% |
| arch-VGPR | ≤ 128 (2 waves) | > 256 (spills) |
| Register / scratch spills | 0 | any |

## Default Starting Configs

- **Triton**: `BLOCK_M/N` 128–256, `BLOCK_K` 32–64, `num_warps` 4–8, then re-check register/LDS pressure and let autotune sweep around it.
- **HIP / CK / hand-written**: 256 threads / 4 waves owning one `(tile_m, tile_n)`; `tile_m % 16 == 0`, `tile_n % 64 == 0`, `tile_k·elem_bytes % 64 == 0`; a reasonable first pass is `tile_m=256, tile_n=128, tile_k=64`, then re-profile.
- Launch enough work-groups to cover all CUs (304 on gfx942, 256 on gfx950) or use a persistent kernel; small grids underfill the device.

## Training-Transfer Caveats

These knowledge directions must improve a real fwd+bwd training step, not just the benchmark loop (see the campaign iteration rules):

- Optimize the **combined-step** metric. A backward-focused change may dip forward slightly; that is acceptable when the combined-step metric improves.
- A **weight-quantization cache** keyed on `(id(weight), weight._version)` is bounded — its real-step gain is `quant_time(weight) / step_time`, typically a low single-digit percent. Never cache on activation / grad_out identity.
- For **grouped / MoE GEMM**, every gain must hold under a *skewed* `tokens_per_expert` distribution, not just the uniform, tile-aligned benchmark layout. Treat per-expert token counts as runtime values and handle empty groups; do not bake `M_per_group` into a compile-time constant or assume tile-divisibility.

## Directions Map

| If the kernel is… | Start with these themes in [`optimization-directions.md`](optimization-directions.md) |
|-------------------|------------------------------------------------------|
| A generic dense GEMM | Tiling & MFMA atom; LDS ping-pong & prefetch; XOR swizzle; hot-loop scheduling |
| Memory- or BW-bound | Async G2S staging; preshuffle B; split-K; L2/XCD locality |
| Skinny / decode (small M) | Small-M / skinny GEMM; split-K; N-tile A-reuse |
| FP8 / FP4 / MX / low-bit | Low-precision & block-scale; preshuffle + packed conversion |
| MoE / grouped | Grouped & MoE GEMM (with skew robustness); fused-activation epilogue |
| Store-bound epilogue | CShuffle LDS epilogue; packed VOP3P epilogue |
| Occupancy-limited | Occupancy & register budget; async-copy VGPR reclaim |
