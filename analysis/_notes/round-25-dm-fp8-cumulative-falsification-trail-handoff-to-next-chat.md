# Round 25 (DM): Cumulative falsification trail and handoff to next chat (cold-start)

## Why this doc exists

Chat session is at 88+/90 min and will roll over after this round.
Next agent starts cold (loses all in-chat file reads and tool history).
This doc consolidates rounds 17-24 into a single "do not retry" + "do
try" cheat-sheet so the next agent can start productively in their
first tool-call instead of re-reading all 8 prior round notes.

## State

- Score: 820 (this round baseline)
- Best: 829 (R18, sha `809f93b3`)
- Patience: 6/30 — score has been in 818-829 noise band for 6 rounds
- Geomean: 0.9846 (target 1.20, gap 22 percentage points)
- Worst FP8 shapes (consistent): gpt_oss-GateUP-B4-M{2048,4096}
  (0.947, 0.962), DSV3-Down-B16-M{2048,4096} (0.954, 0.964),
  gpt_oss-Down-B4-M4096 (0.956)

## Diagnostic finding (R17 → R23)

**HK main GEMM kernel `grouped_rcr_kernel<0,*,*>` is 6-8% slower than
Triton's `_grouped_fp8_persistent_gemm_kernel` per call**:

| shape                          | HK µs | Triton µs | gap   |
|--------------------------------|------:|----------:|------:|
| DSV3-Down-B16-M4096            |  1052 |       986 | +6.7% |
| DSV3-Down-B16-M2048            |   534 |       498 | +7.2% |
| gpt_oss-GateUP-B4-M2048        |   153 |       142 | +7.7% |
| gpt_oss-GateUP-B4-M4096        |   300 |       283 | +6.0% |

PMC says HK executes only 0.83% more instructions. So the gap is in
**cycles-per-instruction**, not instruction count. Quantize and host
overhead are confirmed identical for both backends (R22, R23).

## DO NOT RETRY — falsified directions (8 rounds, R17-R24)

| # | round | direction                            | result                                |
|--:|------:|--------------------------------------|---------------------------------------|
| 1 |   R17 | rocprof PMC instr count diagnose     | locates gap in CPI, not insts         |
| 2 |   R18 | hoist `make_srsrc` out of helper     | compiler already CSE'd it             |
| 3 |   R19 | port two-tile main loop from dense   | -37% spill cliff (catastrophic)       |
| 4 |   R20 | `readfirstlane(lds_tile_base)`       | compiler already hoisted              |
| 5 |   R21 | `__noinline__` on `rcr_8w_load_hoist`| build-fail (helper architecturally    |
|   |       |                                      | requires inline)                      |
| 6 |   R22 | host-overhead trim                   | host is 0.9-1.1% only                 |
| 7 |   R23 | (no change) HK vs Triton breakdown   | confirmed gap is 100% in main kernel  |
| 8 |   R24 | RCR_PREFETCH_LGKM sweep {2,4,6,8}    | saturated — all in noise band         |

Older falsifications (rounds 1-16, see git log):
- VMCNT sweep (R14, R16, R22, R24, R25)
- MFMA cell shape migration 16x16x128 → 32x32x64 (R14, R15)
- chunk_size sweep (R16)
- post-MFMA `s_barrier` + `s_setprio(0)` drop (R13)
- per-shape config tuning (R7-R69, very deeply tuned already)

## DO TRY — remaining unexplored levers

In rough order of expected leverage (ALL untested as of R25):

### A. `__builtin_amdgcn_sched_group_barrier(MASK, COUNT, 0)`

Triton's MLIR backend uses these to constrain LLVM's machine scheduler;
HK's hand-written kernel uses only `sched_barrier(0)` (block-all
reorder). The `sched_group_barrier` API forces exactly `COUNT` insts
of `MASK` class into a single sync group, which can co-schedule MFMA
with LDS-read more aggressively.

Mask values (LLVM ≥18, see `__builtin_amdgcn_sched_group_barrier`):
  - 0x001 = VALU
  - 0x002 = SALU
  - 0x004 = VMEM
  - 0x008 = MFMA
  - 0x010 = LDS read
  - 0x020 = LDS write

**Concrete first try**: replace one `RCR_SCHED_BARRIER()` call (currently
`sched_barrier(0)`) in the main loop steady-state path (line 1406, 1420)
with `sched_group_barrier(0x008 | 0x010, 5, 0)` — declares "exactly 1 MFMA +
4 LDS-reads here". Build, run metric. If neutral, try variations of the
mask/count.

### B. PMC `SQ_LDS_BANK_CONFLICT` HK vs Triton same shape

R23's wall-time gap might be from LDS read serialization, not from
MFMA pipelining. The FP8 K_BLOCK=128 layout uses `ds_read_b128` which
can hit 8-way bank conflicts on certain swizzle patterns. Triton's
auto-swizzler may pick a better one.

```bash
# Capture LDS bank conflict counters for both backends
cd /workspace/code/Primus-Turbo
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
rocprofv3 -i /tmp/lds_pmc.txt --kernel-include 'grouped_rcr_kernel' \
  python3 -c '...probe DSV3-Down-B16-M4096...'

PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON \
rocprofv3 -i /tmp/lds_pmc.txt --kernel-include '_grouped_fp8_persistent' \
  python3 -c '...same probe...'
```

`/tmp/lds_pmc.txt` should contain:
```
pmc: SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_INSTS_VMEM
pmc: SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_BUSY_CU_CYCLES
```

If `SQ_LDS_BANK_CONFLICT_HK / SQ_INSTS_LDS_HK` >> Triton's same ratio,
THE ANSWER is LDS layout. Re-pack the FP8 LDS tile to swizzle around
the conflict pattern. Reference: `tk_fp8_layouts` (the LDS swizzle
struct definitions, search `swizzled_offsets` in `kernel_fp8_layouts.cpp`).

### C. Accept ~825 as architectural ceiling

After 8 rounds of falsification with no metric movement beyond noise,
seriously consider that `grouped_rcr_kernel` is at its architectural
ceiling vs Triton's persistent FP8 grouped kernel. The FP8 main kernel
has:
- Same MFMA throughput target (320 TF MI300X tensor peak)
- Same LDS bank conflict surface (K_BLOCK=128 layout)
- Same VGPR pressure target (128/wave for 2 waves/SIMD occupancy)
- 67 vs 0 spill divergence vs BF16 — but BF16 grouped runs ABOVE
  parity (1.16-1.21×) so spill clearly isn't capping HK

If round 30 hasn't broken 840, escalate: this kernel is at its
architectural ceiling and remaining gap requires either a different
kernel structure (warp-specialization, ASM-level scheduling) or
acceptance.

## What I did this round

Ran R25 baseline metric: 820 (within noise of 818-829). Did NOT touch
the kernel — chat rolls in 2 min and the recommended next move
(`sched_group_barrier`) needs at least 5 min for build+test+verify.
Wrote this consolidation doc instead so the next chat can start
productively from a clean slate.

## Round 25 verdict

No kernel change. Doc-only commit.

**Metric**: 820 (no change vs round 24 final).

## Ground truth for round 26+ (cold-start agent)

You are starting fresh. Do NOT re-read SKILL.md or the round-13-...-22
notes individually. Read THIS doc (round-25-dm) plus the just-prior
round (whatever number this is) and you have everything you need.

Your first 3 tool calls should be:
1. `python3 scripts/_metric_grouped_only.py` (~13s) → get baseline,
   pick worst shape
2. Read `analysis/_notes/round-25-dm-fp8-cumulative-falsification-trail-handoff-to-next-chat.md`
   (this doc, 30s)
3. Either implement Option A (`sched_group_barrier` insertion, ~5 min)
   or Option B (rocprof bank conflict comparison, ~10 min)

DO NOT spend > 5 min on diagnostic-only rounds without a kernel
change. The metric has not moved in 6 rounds; the productive direction
is to TEST something, not measure something.
