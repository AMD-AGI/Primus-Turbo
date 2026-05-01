# Round 27 (DM): grouped main loop sched_barrier + s_setprio(3) — falsified, plus R26 correction

## Major correction to round-26-dm

Round-26-dm reported "per-shape signal IS real" for the
`RCR_SCHED_BARRIER` mask sweep — claiming consistent +2-3pp wins on
DSV3-Down/GateUP-B32-* shapes and a -4.3pp loss on DSV3-Down-B16-M2048.
Round 27 audit of the kernel call graph **falsifies that interpretation**.

### What R26 actually changed

The macro `RCR_SCHED_BARRIER()` is called from 9 sites in
`kernel_fp8_layouts.cpp`:

| lines           | function                               | reachable from FP8 grouped metric? |
|-----------------|----------------------------------------|-----------------------------------:|
| 1334-1466       | `gemm_rcr_kernel` (DENSE, single-shape)| **NO** — metric is grouped-only    |
| 2212, 2230, 2243| `grouped_rcr_kernel` K-tail epilog     | only for K%128≠0 (gpt_oss K=2880)  |

The grouped main loop body (`grouped_rcr_kernel` lines 2175-2202) has
**zero** `RCR_SCHED_BARRIER()` call sites. Round-4 (commit `333074d6`)
removed the original 2x per K-iter (gain +0.45pp at the time).

So R26's mask sweep on the macro:
- Did NOT touch the grouped main loop (8/16 of the metric is DSV3 with
  K%128=0, so ALL DSV3 shape variations were noise — the macro change
  literally couldn't affect them).
- DID touch the K-tail epilog used by gpt_oss shapes (K=2880, K_REM=64).
- DID touch the dense `gemm_rcr_kernel` (irrelevant — not grouped).

### Therefore R26's per-shape "signal" was noise

The reported "consistent +2-3pp" on DSV3-Down/GateUP-B32-* and "-4.3pp"
on DSV3-Down-B16-M2048 cannot have been caused by the mask change,
because the kernel reached by these shapes has no instance of the
modified macro. They were noise that happened to alias-cluster across
3 quick consecutive runs in the R26 chat session. Round-27 5-sample
re-tests of those exact shapes show variance of ±3-4pp across runs,
inside which any 3-sample subset can look "directionally consistent".

This rules out the round-26 follow-up plan A (per-template-parameter
mask gating on RCR_SCHED_BARRIER) — there's nothing to gate.

## Round 27 experiments (all in noise band)

Now armed with the correct call graph, R27 ran 4 NEW experiments that
DO target the grouped main loop body:

| # | experiment                                   | runs                  | mean  | median |
|--:|----------------------------------------------|-----------------------|-------|--------|
| 0 | baseline (R27 fresh)                         | 826                   | —     | —      |
|   | baseline (R26 5-sample for fair compare)     | 829,823,825,823,822   | 824.4 | 823    |
| 1 | +2x `sched_barrier(0x108)` post-MMA-A,C      | 828,821,830,821,828   | 825.6 | 828    |
| 2 | +1x `sched_barrier(0x108)` post-MMA-A only   | 822,829,821,827,821   | 824.0 | 822    |
| 3 | +1x `sched_barrier(0x100)` (DS_READ only)    | 827,822,828,826,822   | 825.0 | 826    |
| 4 | bump `s_setprio(1)` → `s_setprio(3)` (4 sites) | 829,820,823,828,822 | 824.4 | 823    |

All 4 means are within ±1.2 of baseline mean 824.4. Two-sample t-test
on Exp1 vs baseline: p > 0.4 (not significant). The single 830 run in
Exp1 (a new 1-sample high) is within the expected max of an n=5 sample
of Normal(824, σ=4) → p(max ≥ 830) ≈ 25%. **Not a real win.**

This means:
- The grouped main loop's existing `s_barrier()` + `s_setprio(1/0)` +
  `s_waitcnt {vmcnt,lgkmcnt}` fence sequence already constrains the
  LLVM machine scheduler tightly enough that adding `sched_barrier(MASK)`
  hints (any of mask=0x100, 0x108, or 2x of either) makes no
  reorder freedom available beyond the noise floor.
- Pumping `s_setprio` from 1 to 3 changes nothing — under no-contention
  benchmark conditions, the wave never had peer-wave priority pressure
  to lose.

## Updated falsification trail (now 13 directions)

| # | round | direction                              | result                                  |
|--:|------:|----------------------------------------|-----------------------------------------|
| 1 |   R17 | rocprof PMC instr count diagnose       | locates gap in CPI, not insts           |
| 2 |   R18 | hoist `make_srsrc` out of helper       | compiler already CSE'd it               |
| 3 |   R19 | port two-tile main loop from dense     | -37% spill cliff (catastrophic)         |
| 4 |   R20 | `readfirstlane(lds_tile_base)`         | compiler already hoisted                |
| 5 |   R21 | `__noinline__` on `rcr_8w_load_hoist`  | build-fail (architecturally inline)     |
| 6 |   R22 | host-overhead trim                     | host is 0.9-1.1% only                   |
| 7 |   R23 | (no change) HK vs Triton breakdown     | confirmed gap is 100% in main kernel    |
| 8 |   R24 | RCR_PREFETCH_LGKM sweep {2,4,6,8}      | saturated — all in noise band           |
| 9 |   R26 | RCR_SCHED_BARRIER mask sweep (macro)   | **macro doesn't reach grouped main loop — affects only K-tail epilog & dense kernel; "per-shape signal" was noise from sample aliasing** |
|10 |   **R27**a | +sched_barrier(0x108) 2x grouped main loop | mean +1.2 of baseline, ns           |
|11 |   **R27**b | +sched_barrier(0x108) 1x post-MMA-A    | mean = baseline                         |
|12 |   **R27**c | +sched_barrier(0x100) 1x post-MMA-A    | mean +0.6 of baseline, ns               |
|13 |   **R27**d | s_setprio(1) → s_setprio(3) 4 sites    | mean = baseline                         |

## What's left to try (round 28+)

The 4-experiment sweep at single-knob granularity is exhausted for the
"main loop scheduler hint" lever. Productive directions remaining:

### A. Combined sched_barrier + setprio + wider unroll

Mean stays at 825 across all 4 R27 sub-experiments — the schedule is
saturated at single-knob granularity. A multi-knob combination may be
different. Combined plan:
- `s_setprio(3)` + `sched_barrier(0x108)` 2x + `RCR_MAIN_UNROLL=4`.
- Risk: 4x unroll may regress register pressure.

### B. PMC `SQ_LDS_BANK_CONFLICT` HK vs Triton (Option B from R25 doc)

Still untested across R26+R27. Concrete command:

```bash
cd /workspace/code/Primus-Turbo
# Capture PMC for HK
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
rocprofv3 -i /tmp/lds_pmc.txt --kernel-include grouped_rcr_kernel \
  python3 -c '...probe DSV3-Down-B16-M4096...'

# Compare with Triton
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON \
rocprofv3 -i /tmp/lds_pmc.txt --kernel-include _grouped_fp8_persistent \
  python3 -c '...same probe...'
```

`/tmp/lds_pmc.txt` contents:
```
pmc: SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_WAIT_INST_LDS
pmc: SQ_INSTS_MFMA SQ_BUSY_CU_CYCLES SQ_WAIT_INST_VMEM
```

If `SQ_LDS_BANK_CONFLICT_HK / SQ_INSTS_LDS_HK >> Triton's same ratio`,
the answer is LDS layout (re-pack `ST_v2` swizzle around the conflict
pattern). Otherwise, the gap is something else (instruction cache,
issue-port pressure, AGPR migration, etc.).

### C. RCR_STEADY_VMCNT sweep (untested counter)

Round 24 swept RCR_PREFETCH_LGKM (line 55 = 4) and saturated. The other
related counter is RCR_STEADY_VMCNT (line 58 = 8) — controls how many
outstanding HBM loads are allowed during steady-state main loop. Has
NOT been swept. Quick experiment for round 28:
- Try {4, 6, 10, 12} and 5-sample mean each.
- Hypothesis: too-large slack increases register pressure (each
  outstanding VMEM holds destination registers); too-small slack
  serializes the pipeline.

## Round 27 verdict

- Final HipKittens kernel diff: zero (4 sub-experiments reverted after sweep).
- Final HipKittens .so: rebuilt to match source (verified clean 826 baseline).
- No commit on HipKittens repo.
- Doc-only commit on Primus-Turbo (this file).

Score: 826 final clean baseline run (within 818-829 noise band).
Patience now 8/30. Score band: 818-830 (one R27 sub-experiment hit
830, R18 best=829 — both noise-bound, can't claim 830 as new high).
Geomean: ~0.99 (target 1.20, gap 21pp).

## Files touched this round

- `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  — lines 2175-2202 (grouped main loop body). 4 successive edits:
  add 2x `sched_barrier(0x108)`, reduce to 1x, swap to mask=0x100,
  bump s_setprio(1)→s_setprio(3). All reverted.
- `/workspace/code/Primus-Turbo/analysis/_notes/round-27-dm-fp8-grouped-main-loop-sched-barrier-audit-and-falsification.md`
  — this doc.

## For round 28+ (if same chat continues)

Read this doc + R26 (with the R27 correction). Skip the macro
RCR_SCHED_BARRIER thread entirely (R26 was misdirected, R27 audit
confirms). Productive direction in priority order:

1. **C. RCR_STEADY_VMCNT sweep** — quickest test (single line,
   `#define RCR_STEADY_VMCNT 8` → try {4, 6, 10, 12} × 5 runs).
   Compares apples-to-apples with R24 LGKM sweep. May find a knob
   R24 didn't.
2. **B. PMC LDS bank conflict** — needs `rocprofv3` setup. Does NOT
   move score by itself but localizes the actual bottleneck for the
   next 5 rounds. Worth doing if RCR_STEADY_VMCNT is also flat.
3. **A. Combined multi-knob** — too many degrees of freedom, low ROI
   without the diagnostic from B.
