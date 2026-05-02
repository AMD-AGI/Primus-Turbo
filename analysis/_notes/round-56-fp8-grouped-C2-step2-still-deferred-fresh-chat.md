# Round 56 (dm) — FP8 grouped: C-2 step-2 still deferred (chat at 89/90 min limit on entry)

## TL;DR
- Metric: **score=977**, BF16=1.1864, FP8=1.1577. Same noise band
  as R51-R55 (mean ~985, σ ~5). No regression.
- Chat session arrived at 89 min / 90 min cap — physically no time
  for C-2 step 2 (~30-45 min). Same situation as R55.
- All R56 actionables are **identical to R55's R56-handoff plan**.
  Forward to R57 fresh chat.

## 1. Metric

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1864 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1577 (n=24)
score=977
```

5-round running history (R52-R56): {989, 987, 984, 983, 977},
μ=984, σ=4.5. R56 is 1.5σ below mean — within R51's noise model.

## 2. C-2 step-2 mechanical recipe (preserved from R55 doc)

Source file:
`/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`

Steps:
1. Copy the `grouped_rcr_kernel` template body (line ~2319) into
   `namespace lever_c2_round_54_step1_scaffold` (line 184, ends
   line ~280).
2. Lift the four explicit instantiations (~line 2978-2981) into
   the namespace.
3. `dispatch_grouped_rcr` (line ~5480) does NOT change in step 2.
4. Build with `THUNDERKITTENS_ROOT=/workspace/code/HipKittens
   make all` and grep build log for the namespace-mangled
   `_ZN36lever_c2_round_54_step1_scaffold18grouped_rcr_kernel*`
   resource report.
5. Acceptance:
   - (a) AGPR ≥ 256 → C-2 hypothesis CONFIRMED, R57 wires
     dispatcher gated on FUSED_KTAIL=true.
   - (a) AGPR=0 → FALSIFIED, fall back to Lever C-3 (`art_base`
     + ASM, see R47 doc).
   - (b) Spill ≤ 8 (FK=T baseline 34, any reduction OK).
   - (c) Occupancy ≥ 1 waves/SIMD.

## 3. R56 actions

- Metric: ✅ score 977 (within σ-band)
- HK code: ❌ NO change (window at 89/90 cap on entry)
- Doc commit: this file (R57 entry-point, == R55 entry-point)

## 4. Before-after metric

| Round | sha (PT) | sha (HK) | score |
|-------|----------|----------|-------|
| R55   | f1cb14f  | 73da21c6 | 983   |
| **R56** | **(this)** | **73da21c6** | **977** |

## 5. Next round (R57)

R57 must arrive in a fresh 90-min chat window. Read R54+R55 docs,
execute step 2 per the mechanical recipe above. Decision gate at
end determines whether R58 enters R57's dispatcher-wiring step or
falls back to Lever C-3.
