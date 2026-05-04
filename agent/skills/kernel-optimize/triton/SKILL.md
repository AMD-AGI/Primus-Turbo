---
name: kernel-optimize-triton
description: Triton kernel optimization guidance for AMD ROCm GPUs. Use when working on Triton kernels, autotune spaces, matrix-instruction selection, AMD Triton compiler behavior, or Python kernel files that use `triton` or `tl.`.
---

# Triton Kernel Optimization

Use this skill for AMD ROCm Triton kernels.

## Start Here

1. Read `../../../rules/iteration_rules.mdc`.
2. Read `../../tool-rocprof/SKILL.md`.
3. Read the matching hardware guide:
   - `../../hardware/gfx942/SKILL.md`
   - `../../hardware/gfx950/SKILL.md`

## What This Skill Covers

- Triton autotune space design on AMD
- MFMA- and tile-oriented reasoning from the Triton side
- AMD-specific compiler behavior and debugging hooks
- launch-overhead, memory, and occupancy tradeoffs in Triton kernels
- low-precision format discipline for `gfx942` vs `gfx950`

## Baseline Workflow

Use the project skill's focused test and benchmark commands as the source of truth. Around that workflow:

```bash
# High-level classification
rocprof-compute profile --name triton_baseline -- python run_kernel.py
rocprof-compute analyze -p triton_baseline/MI300X --cli

# Optional compiler / generated-code debugging
MLIR_ENABLE_DUMP=1 python run_kernel.py
TORCH_COMPILE_DEBUG=1 python run_kernel.py
```

If the kernel is dominated by launch gaps or host-side overhead, address that before fine-grained autotune work.

## Good First Knobs

| Knob | Typical starting range |
|------|------------------------|
| `num_warps` | `4`, `8`, `16` |
| `num_stages` | `1`, `2`, `3` |
| Tile / block sizes | Multiples aligned with wave64 and the hardware guide |
| Matrix-instruction choice | Re-check when MFMA efficiency is low |

Keep the autotune space purposeful. A huge search space is not a substitute for profiling.

## AMD-Specific Reminders

- Wavefront size is `64`, so tile and reduction logic must reflect that.
- `gfx942` FP8 work usually assumes `FNUZ`; `gfx950` uses `OCP`.
- Large tiles that look attractive on `gfx950` may not fit cleanly on `gfx942`.
- AMD Triton passes can materially affect the generated code; use compiler dumps when performance is surprising.

## Bottleneck-to-Action Mapping

| Bottleneck | First directions to try |
|------------|-------------------------|
| Memory-bound | Improve coalescing, reduce redundant loads, revisit staging, revisit tile traversal |
| Compute-bound | Revisit tile shape, matrix instruction choice, `num_stages`, inner-loop structure |
| Stall / occupancy-bound | Reduce register pressure, shrink staging, simplify index math, re-check launch geometry |
| Launch-overhead bound | Reduce dispatch count, consider graph capture or batching before retuning math |

## When to Inspect Generated Artifacts

Inspect generated MLIR / ISA when:

- MFMA utilization is lower than expected.
- A parameter sweep changes performance in a way that does not match the profiler story.
- A supposedly local change caused a large register or occupancy shift.
- The kernel behaves differently on `gfx942` and `gfx950` and you need to confirm the compiler path.

## Low-Precision Rules

- Always record whether the round is `FP8 FNUZ`, `FP8 OCP`, `FP6`, `FP4`, or `MXFP`.
- Validate conversions and scaling separately before trusting benchmark gains.
- Do not accept a performance win that rides on a silent format mismatch.

## When to Escalate

- Two failed rounds: refresh the profile and verify the bottleneck classification.
- Three failed rounds: inspect generated code and compare against a known-good baseline.
- Borderline improvements: re-measure enough times to separate signal from noise.

## Related Files

- `../../../rules/iteration_rules.mdc`
- `../../tool-rocprof/SKILL.md`
- `../../hardware/gfx942/optimization-guide.md`
- `../../hardware/gfx950/optimization-guide.md`
