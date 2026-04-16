# Kernel Optimization Examples

Use these examples as routing hints, not as literal commands.

## Example 1: Triton FP8 GEMM on MI300X

1. Read `SKILL.md` and gather project-specific test / benchmark commands.
2. Read `../../rules/iteration_rules.mdc`.
3. Read `triton/SKILL.md`.
4. Read `../hardware/gfx942/SKILL.md`.
5. Read `../tool-rocprof/SKILL.md`.
6. Create `<campaign_dir>/related_work.md` using `related-work-template.md`, cloning any temporary comparison repos into `agent/tmp/<campaign_name>/related-work/repos/`.
7. Run baseline correctness + benchmark.
8. Make one change, validate, and accept or roll back.

## Example 2: HIP MFMA Kernel on MI355X

1. Read `SKILL.md`.
2. Read `../../rules/iteration_rules.mdc`.
3. Read `hip/SKILL.md`.
4. Read `../hardware/gfx950/SKILL.md`.
5. Create `<campaign_dir>/related_work.md` before BASELINE, including any relevant ROCm docs, GitHub repos, or competitor baselines.
6. Use `rocprof-compute` and `hipcc --resource-usage` to classify the bottleneck.
7. Change one thing in the hot path and re-measure.

## Example 3: CK Tile Tuning

1. Read `SKILL.md`.
2. Read `../../rules/iteration_rules.mdc`.
3. Read `hip/SKILL.md` and then `hip/ck.md`.
4. Produce `<campaign_dir>/related_work.md` to capture known CK configs, alternate repos, and reported baselines.
5. Start from a known-good tile shape.
6. Change only one knob family per round: tile, scheduler, pipeline, or partitioner.
7. Re-profile after each accepted version.
