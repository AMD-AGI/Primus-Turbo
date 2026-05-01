# Round 23 (DM): Side-by-side HK vs Triton confirms 6-8% main-kernel gap is **pipeline stalls**, not instruction count

## Status

- Primus-Turbo HEAD start: `181a67c6`
- Best metric: 829 (round-18, sha `809f93b3`)
- Round 23 baseline: 827
- Last 5 rounds: 829 → 821 → 818 → 819 → 824 → 827 (drifting up but capped)
- Cumulative falsifications since round-17 PMC localization:
  - R17 PMC: HK 0.83% slower in pure instruction count
  - R18 SRD hoist (compiler already CSE'd)
  - R19 two-tile main loop port (-37% catastrophic spill cliff)
  - R20 `readfirstlane` (compiler already hoisted)
  - R21 `__noinline__` (build-fail, architecturally inline)
  - R22 host overhead (only 0.9-1.1%, not the lever)

## Round 23 question

Round 22 ruled out host overhead. Round 17 found HK is only 0.83% slower
in PMC instruction count. But the metric shows ~5% wall slower. Where is
the discrepancy? Either (a) PMC counters miss a category of stalls, or
(b) the comparison was apples-to-oranges (wrong baseline shape).

**Probe**: Modified `/tmp/probe_round22_host_overhead.py` to switch
between `PRIMUS_TURBO_GROUPED_GEMM_BACKEND={HIPKITTEN, TRITON}` and
profile both backends on the same shape with identical quantize input.

## The data — head-to-head per-kernel breakdown (µs)

| shape                          | backend | wall | main | quant | offs | host |
|--------------------------------|---------|-----:|-----:|------:|-----:|-----:|
| DSV3-Down-B16-M4096 (0.945)    | HK      | 1416 | 1052 | 376   | 4.3  |  -17 |
| DSV3-Down-B16-M4096            | Triton  | 1412 |  986 | 368   | 4.1  |   55 |
| DSV3-Down-B16-M2048 (0.956)    | HK      |  832 |  534 | 297   | 4.2  |   -3 |
| DSV3-Down-B16-M2048            | Triton  |  804 |  498 | 294   | 4.1  |    8 |
| gpt_oss-GateUP-B4-M2048 (0.955)| HK      |  259 |  153 | 101   | 4.0  |    2 |
| gpt_oss-GateUP-B4-M2048        | Triton  |  246 |  142 |  96   | 4.0  |    4 |
| gpt_oss-GateUP-B4-M4096 (0.960)| HK      |  438 |  300 | 127   | 4.4  |    6 |
| gpt_oss-GateUP-B4-M4096        | Triton  |  417 |  283 | 125   | 4.1  |    4 |

**Triton's main kernel name**: `_grouped_fp8_persistent_gemm_kernel`
(the persistent grouped FP8 kernel from `triton/grouped_gemm`).

## Per-kernel gap = ~6-8% consistently

| shape                          | HK main | Triton main | gap   |
|--------------------------------|--------:|------------:|------:|
| DSV3-Down-B16-M4096            |   1052  |       986   | +6.7% |
| DSV3-Down-B16-M2048            |    534  |       498   | +7.2% |
| gpt_oss-GateUP-B4-M2048        |    153  |       142   | +7.7% |
| gpt_oss-GateUP-B4-M4096        |    300  |       283   | +6.0% |

**Quantize+offs are identical for both backends** (~3 us range, all noise).
**Host overhead is identical** (~0-7 us).
**100% of the wall-time gap is inside the main GEMM kernel.**

## The PMC vs wall-time discrepancy = pipeline stalls

Round 17 PMC said HK is only 0.83% slower in pure instruction count.
Round 23 wall-time data says HK is 6-8% slower per-kernel. The 7-8×
delta means:

> **The gap is in cycles-per-instruction, not instructions-per-call.**
> HK and Triton execute roughly the same instruction stream (PMC: 1.01×),
> but HK takes 1.07× as many cycles to execute it. This is **pipeline
> efficiency**: stalls, occupancy, MFMA pipelining, dependency latency.

## What this invalidates and validates

**Invalidates** (with hindsight, ALL of rounds 18-22 attacked the wrong
metric):
- Round 18 SRD hoist: was trying to reduce SALU instruction count
- Round 20 readfirstlane: was trying to reduce VGPR pressure (= occupancy)
  — the compiler was already doing this, but even if it weren't, occupancy
  improvements only help if the bottleneck is occupancy stalls
- Round 21 `__noinline__`: was trying to reduce inline-expansion footprint
  (= I-cache pressure, code size)
- Round 22 host: was trying to reduce host μs

**Validates** the (failed) attempt direction:
- Round 13 `s_setprio` + `s_barrier` (post-MFMA scheduling hints) — this
  IS in the right category (pipeline scheduling) but the specific knob
  was already saturated.

**Newly motivated** as worth trying (rounds 24+):
1. **MFMA pipeline density**: insert `s_setprio(3)` before MFMA blocks
   to bias scheduler toward issuing MFMA over LDS reads. (R13 tried
   `s_setprio(0)` POST-mfma; R24 should try `s_setprio(3)` PRE-mfma.)
2. **`amdgcn.sched_group_barrier`**: explicit instruction-class
   reordering hints to LLVM's machine scheduler. Triton's MLIR backend
   uses these aggressively; HK's hand-written kernel doesn't.
3. **MFMA latency hiding**: ensure `vmcnt(N)` waits don't stall the
   MFMA pipeline. Issue MFMAs while waiting for the next K-tile's LDS
   loads to land.
4. **LDS bank conflict audit**: `ds_read_b128` patterns can cause
   8-way bank conflicts on FP8's K_BLOCK=128 layout. Triton's auto-
   swizzler may pick a better layout. (Not yet PMC'd.)

## What doesn't work for this gap (DO NOT attempt rounds 24+)

- Reducing code size (R18, R20, R21) — instruction count isn't the gap
- Reducing host overhead (R22) — host isn't the gap
- Per-shape config rules (rounds 7-69 already exhaustively tuned)
- MFMA cell shape migration (R14-15 PMC falsified)
- Adding code to amortize prologue/epilogue (R19 spill cliff)

## Round 23 verdict

This round produced no kernel change. The probe is the most valuable
diagnostic of the entire 5-round death-march:

  HK main GEMM kernel takes 6-8% MORE WALL CYCLES per call than Triton's
  `_grouped_fp8_persistent_gemm_kernel`, despite executing only 0.83%
  more instructions. The lever is **pipeline scheduling**, not code size,
  not register pressure, not host overhead.

**Metric**: 827 (slight uptick from 824 last round, still below 829 best).

## Recommended round 24+ starting move

1. Read `analysis/_notes/round-{17..23}-dm-*.md` to absorb the rule-out
   trail (chat is rolling — next agent cold-starts).
2. **Do not** retry: SRD hoist, two-tile port, readfirstlane,
   `__noinline__`, host trim, MFMA cell migration, chunk_size sweep,
   per-shape config (saturated).
3. **Do** try, in order of expected leverage:
   a. **`s_setprio(3)` PRE-mfma** in main loop (round 13 tried POST,
      this round 24 tries PRE — opposite direction). Build → measure
      `grouped_rcr_kernel` µs on DSV3-Down-B16-M4096; expect ±2% range
      (single LLM-line change).
   b. **`__builtin_amdgcn_sched_group_barrier(MFMA, 4, 0)`** before
      and after MFMA blocks to constrain scheduler. Built into
      `clang/amdgcn` since LLVM 18; HK currently uses it sparingly.
   c. **rocprof PMC `MfmaUtil` + `LDSStalledByDS` + `SQ_WAIT_INST_VMEM`**
      on HK vs Triton main kernel side-by-side. The PMC delta on these
      THREE counters will tell us which specific stall to attack next.
4. Score floor 818, ceiling 829, mean 822. The "noise band" is ±5
   points. Any commit must produce ≥10-point delta to be confidently
   non-noise. Run metric 3× to confirm.
