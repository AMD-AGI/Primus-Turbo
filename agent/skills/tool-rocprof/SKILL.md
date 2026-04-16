---
name: tool-rocprof
description: ROCm profiling workflow for AMD GPU kernels using rocprofv3 and rocprof-compute. Use when profiling hot kernels, collecting counters, diagnosing memory-vs-compute-vs-stall bottlenecks, reading Perfetto traces, or validating low-precision AMD kernels.
---

# ROCm Profiling Workflow

Use this skill when you need runtime evidence before changing an AMD kernel.

## Pick the Right Tool

| Tool | Best for | Output |
|------|----------|--------|
| `rocprofv3` | Timeline traces, raw counters, API/kernel overlap, Perfetto | CSV / JSON / PFTrace / rocpd |
| `rocprof-compute` | Speed-of-Light, roofline, occupancy, MFMA / memory panels | Workload directory + CLI / GUI analysis |
| `rocm-smi`, `rocminfo`, `hipcc --resource-usage` | Environment, clocks, device properties, register / spill clues | Text diagnostics |

Rule of thumb:

- Start with `rocprof-compute` if you want a fast high-level bottleneck classification.
- Use `rocprofv3` when you need exact trace timing, kernel overlap, or raw counters.
- Use both when the bottleneck is ambiguous.

## Minimal Workflow

1. Identify the hot kernel or focused benchmark from the project skill.
2. Run `rocprof-compute` on that focused benchmark to get Speed-of-Light and occupancy context.
3. If the result is unclear, run `rocprofv3` for timeline gaps, counter detail, or multi-kernel overlap.
4. Classify the bottleneck as memory-bound, compute-bound, or stall / concurrency-bound.
5. Make one change, then re-profile the same workload.

## Useful Commands

```bash
# High-level metric collection
rocprof-compute profile --name workload_name -- python run_kernel.py
rocprof-compute analyze -p workload_name/MI300X --cli

# Discover available metric blocks / help
rocprof-compute profile -h
rocprof-compute profile --list-metrics
rocprof-compute analyze -h

# Trace or raw-counter collection
rocprofv3 --help
rocprofv3 --list-counters
rocprofv3 <trace-or-counter-options> -- python run_kernel.py

# Supporting commands
rocm-smi
rocminfo
hipcc --resource-usage kernel.hip
```

The exact `rocprofv3` flags vary by ROCm release, so verify them with local `--help` before scripting.

## First Classification

Use Speed-of-Light or equivalent GPU activity to decide whether microarchitecture tuning is even the right problem:

| Signal | Interpretation | Next step |
|--------|----------------|-----------|
| GPU utilization `> 60%` | GPU-bound enough to optimize at kernel level | Continue to memory vs compute split |
| GPU utilization `< 30%` | Likely launch, CPU, synchronization, or scheduling issue | Inspect timeline / host-side gaps first |
| `30-60%` | Mixed or unclear | Cross-check trace overlap and SoL panels |

Do not start with micro-tuning if the GPU is mostly idle.

## Memory vs Compute vs Stall

### Memory-bound signals

- HBM bandwidth utilization is high and near the device roofline.
- L2 hit rate is poor or coalescing looks worse than expected.
- LDS bank conflicts or VMEM latency counters are elevated.

Typical actions:

- Improve coalescing and vectorization.
- Add or improve LDS tiling.
- Reduce redundant global traffic.
- Revisit traversal order and cache reuse.

### Compute-bound signals

- MFMA utilization is high or should be high, but achieved throughput is still below expectation.
- Instruction mix is dominated by MFMA or should be, yet issue efficiency is poor.
- VALU work appears inflated relative to the intended GEMM / tensor-core style path.

Typical actions:

- Make sure the kernel is actually using MFMA.
- Revisit tile sizes, K unrolling, or matrix-instruction selection.
- Reduce dependency chains and improve overlap between loads and MFMA.
- For Triton, inspect AMD-specific compiler behavior before blaming the algorithm.

### Stall / concurrency-bound signals

- HBM and MFMA are both below roofline even though the GPU is busy.
- Occupancy is low because of VGPR, LDS, barrier, or wave-limit pressure.
- Scratch usage or spill indicators are non-zero.

Typical actions:

- Reduce register pressure or live ranges.
- Reduce LDS footprint or change block shape.
- Re-check launch bounds and occupancy.
- Inspect synchronization patterns and pipeline bubbles.

## Practical Thresholds

These are starting heuristics, not laws:

| Metric | Good | Needs attention |
|--------|------|-----------------|
| Occupancy | `> 50%` | `< 25%` |
| HBM utilization | `> 60%` when memory-bound | `< 30%` |
| MFMA utilization | `> 50-70%` when compute-bound | `< 40%` |
| LDS bank conflict ratio | `< 5%` | `>= 5%` |
| Scratch / spills | `0` | Any non-zero |

Always compare against the correct hardware guide:

- `../hardware/gfx942/SKILL.md`
- `../hardware/gfx950/SKILL.md`

## Counter Families Worth Watching

| Area | Typical counters / panels | Why |
|------|---------------------------|-----|
| MFMA | `SQ_INSTS_MFMA`, `SQ_INSTS_VALU_MFMA_MOPS_*`, MFMA busy panels | Detect matrix-core usage and efficiency |
| VMEM / LDS | `SQ_INSTS_VMEM_*`, `SQ_INSTS_LDS`, `SQ_LDS_BANK_CONFLICT` | Detect bandwidth waste and LDS issues |
| Cache | `TCC_HIT`, `TCC_MISS`, TCP / TCC panels | Estimate reuse and coalescing quality |
| Occupancy / resources | Occupancy report, scratch metrics, `hipcc --resource-usage` | Find register or LDS pressure |
| Timeline | PFTrace / Perfetto | Detect launch gaps, overlap issues, sequencing problems |

Counter names vary across ROCm releases. Use local tool output to confirm the exact spelling.

## AMD Low-Precision Validation

Profiling is not enough for low-precision work. Every FP8 / FP6 / FP4 / MXFP round should explicitly record:

- target architecture
- exact input and output format
- accumulator format
- scale format and scale granularity
- tolerance used
- whether the failure is conversion-, scale-, or math-related

Critical rule:

- `gfx942` FP8 is typically `FNUZ`
- `gfx950` FP8 is `OCP`

Never label a result only as "`fp8`" when multiple encodings are possible.

Validation strategy:

1. Validate conversion and packing separately.
2. Validate scale handling separately for block-scaled paths.
3. Compare kernel math against a higher-precision or dequantized reference.
4. Only then trust performance numbers.

## Recommended Decision Loop

1. Capture baseline profile on the focused benchmark.
2. Write down the bottleneck classification and one hypothesis.
3. Change exactly one thing in the kernel.
4. Re-run correctness first.
5. Re-run the same profile / benchmark path.
6. Accept only if the data supports the hypothesis.

## Related Files

- `../../rules/iteration_rules.mdc`
- `../kernel-optimize/workflow/optimize-loop.md`
- `../hardware/gfx942/optimization-guide.md`
- `../hardware/gfx950/optimization-guide.md`
