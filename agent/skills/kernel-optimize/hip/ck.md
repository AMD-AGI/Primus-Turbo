# Composable Kernel Tuning Notes

Use this file when the hot path is implemented with `ck::`, CK-Tile abstractions, or generated CK kernels.

## Read Before Tuning

1. `../../../rules/iteration_rules.mdc`
2. `../../tool-rocprof/SKILL.md`
3. The matching hardware guide under `../../hardware/`

## CK Mental Model

Think in this order:

```text
Tile shape -> partitioner -> scheduler -> pipeline -> kernel
```

Changing multiple layers at once makes attribution hard, so keep each round focused.

## High-Value CK Knobs

| Area | What to vary |
|------|---------------|
| Tile shape | `M_Tile`, `N_Tile`, `K_Tile` |
| Work mapping | 2D / 1D / spatially-local partitioning |
| Scheduler | Default / intrawave / interwave |
| Pipeline | memory-oriented vs compute-oriented vs async / ping-pong style |
| Layout | warp layout, LDS staging shape, XCD remap when applicable |

## Good First Choices

| Workload | Good first try | Why |
|----------|----------------|-----|
| Memory-bound GEMM | Memory-heavy pipeline + interwave scheduler | Gives more room for prefetch overlap |
| Compute-bound GEMM | Compute-oriented pipeline + intrawave scheduler | Keeps the loop tight around MFMA work |
| Larger GEMM with clear staging benefit | Double-buffer / ping-pong style pipeline | Hides load latency when LDS budget allows |
| Small repeated kernels | Revisit launch overhead before large tile sweeps | Tile tuning alone may not move the bottleneck |

## Practical Search Strategy

1. Start from a known-good config, not a blank slate.
2. Change one family of knobs at a time:
   - tile shape
   - pipeline
   - scheduler
   - partitioner
3. Re-profile after every accepted round.
4. If a larger tile helps throughput but introduces spills or low occupancy, roll it back and explore a different direction.

## CK-Specific Bottleneck Mapping

| Symptom | Likely direction |
|---------|------------------|
| High HBM pressure, low reuse | Try a more memory-friendly pipeline or stronger staging |
| Low MFMA efficiency, good occupancy | Revisit tile shape or scheduler |
| Occupancy collapse after a tile increase | Back off tile size or reduce staging |
| Good per-wave behavior, poor multi-die scaling | Check partitioning / XCD remap strategy |

## Reminders

- CK tuning is still subject to the same linear iteration contract: one hypothesis, one accepted baseline.
- Do not hide a regression by changing multiple tile dimensions and the scheduler at once.
- If the benchmark target is narrow, document that explicitly before accepting a tradeoff that hurts other shapes.
- When the tuned CK path stops improving, compare it to a Triton or library baseline before investing in deeper CK-only work.
