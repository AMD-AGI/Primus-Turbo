---
name: kernel-optimize-hip
description: HIP C++ kernel optimization guidance for AMD GPUs. Use when working on HIP or C++ kernels, MFMA intrinsics, rocWMMA-style code, hipcc builds, or any kernel source containing `__global__`, `hip`, or `__builtin_amdgcn_*`.
---

# HIP Kernel Optimization

Use this skill for handwritten HIP C++ kernels and HIP-backed operator codepaths.

## Start Here

1. Read `../../../rules/iteration_rules.mdc`.
2. Read `../../tool-rocprof/SKILL.md`.
3. Read the matching hardware guide:
   - `../../hardware/gfx942/SKILL.md`
   - `../../hardware/gfx950/SKILL.md`
4. If the kernel is built on `ck::`, `TilePartitioner`, or CK pipelines, read `ck.md`.

## What This Skill Covers

- MFMA-oriented HIP kernels
- LDS and register tradeoffs
- `hipcc` compilation and resource usage
- wave64 behavior and CUDA-to-HIP mental-model corrections
- hardware-specific divergence between `gfx942` and `gfx950`

## Baseline Workflow

Use the project skill's focused benchmark and test commands as the source of truth. Around that workflow:

```bash
# Profile the focused benchmark or a reduced reproducer
rocprof-compute profile --name hip_baseline -- python run_kernel.py
rocprof-compute analyze -p hip_baseline/MI300X --cli

# Inspect resource pressure for a standalone HIP source
hipcc -O3 --offload-arch=gfx942 --resource-usage kernel.cpp
```

Interpret the first profile before making changes. If the GPU is mostly idle, fix launch density or host-side gaps first.

## Core HIP Rules

- Compile with `-O3`.
- Treat wavefront size as `64`, never `32`.
- Re-check occupancy after every change that increases LDS or live register count.
- Keep low-precision format explicit in every round (`FNUZ` on `gfx942`, `OCP` on `gfx950`).
- Use one primary hypothesis per round and benchmark the full active validation set (`representative_shapes` for quick validation, `target_shapes` for full validation).

## What to Inspect in the Code

### Compute path

- Is the hot path actually using MFMA, or did the compiler fall back to scalar / vector math?
- Are matrix instructions sized sensibly for the chosen tile shape?
- Is the inner loop spending too much time on conversion, epilogue work, or address generation?

### Memory path

- Are global loads coalesced and vectorized?
- Is LDS used for real reuse, or just adding pressure?
- Are LDS layouts creating avoidable bank conflicts?

### Resource path

- Did a tile change increase VGPR pressure enough to lower occupancy?
- Is scratch usage non-zero?
- Did extra staging or double-buffering consume too much LDS?

## High-Value Optimization Directions

| Bottleneck | First directions to try |
|------------|-------------------------|
| Memory-bound | Better coalescing, vector loads, LDS tiling, less redundant traffic |
| Compute-bound | Better MFMA mapping, K unrolling, tile selection, reduced dependency chains |
| Stall / occupancy-bound | Reduce VGPR pressure, reduce LDS footprint, simplify synchronization |

## HIP-Specific Reminders

- Use `#if defined(__gfx942__)` / `#if defined(__gfx950__)` only when the optimization really is architecture-specific.
- Treat `hipcc --resource-usage` and profiler occupancy reports as first-class evidence.
- When a round plateaus, inspect generated ISA or compare against a known-good library implementation before trying random parameter sweeps.
- For wave-level collectives or CUDA ports, keep wave64 semantics visible in the design notes for the round.

## When to Escalate

- Two failed rounds in a row: re-profile with fresh counters.
- Three failed rounds: inspect generated ISA and compare against a reference implementation.
- Low-precision failures: stop tuning and fix conversion / scaling validation first.

## Related Files

- `ck.md`
- `../../../rules/iteration_rules.mdc`
- `../../tool-rocprof/SKILL.md`
- `../../hardware/gfx942/optimization-guide.md`
- `../../hardware/gfx950/optimization-guide.md`
